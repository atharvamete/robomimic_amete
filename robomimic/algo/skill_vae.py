"""
Implementation of Behavioral Cloning (BC).
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim

import robomimic.models.skill_nets as SkillNets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

@register_algo_factory_func("skill_vae")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the DQL algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    return Skill_VAE, {}

class Skill_VAE(PolicyAlgo):
    def __init__(self, **kwargs):
        PolicyAlgo.__init__(self, **kwargs)
    
    def _create_optimizers(self):
        """
        Creates optimizers using @self.optim_params and places them into @self.optimizers.
        """
        def get_optimizer(net, net_optim_params):
            return optim.AdamW(
                params=net.parameters(),
                lr=net_optim_params["learning_rate"],
                betas=net_optim_params["betas"],
                weight_decay=net_optim_params["weight_decay"],
            )
        def get_scheduler(net_optim_params, optimizer):
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=net_optim_params["num_epochs"],
                eta_min=net_optim_params["min_lr"],
                last_epoch=net_optim_params["last_epoch"],
            )
        self.optimizers = dict()
        self.lr_schedulers = dict()

        for k in self.optim_params:
            # only make optimizers for networks that have been created - @optim_params may have more
            # settings for unused networks
            if k in self.nets:
                if isinstance(self.nets[k], nn.ModuleList):
                    self.optimizers[k] = [
                        get_optimizer(net_optim_params=self.optim_params[k], net=self.nets[k][i])
                        for i in range(len(self.nets[k]))
                    ]
                    self.lr_schedulers[k] = [
                        get_scheduler(net_optim_params=self.optim_params[k], optimizer=self.optimizers[k][i])
                        for i in range(len(self.nets[k]))
                    ]
                else:
                    self.optimizers[k] = get_optimizer(
                        net_optim_params=self.optim_params[k], net=self.nets[k])
                    self.lr_schedulers[k] = get_scheduler(
                        net_optim_params=self.optim_params[k], optimizer=self.optimizers[k])
    
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.

        Networks for this algo: critic (potentially ensemble), actor, value function
        """
        self.nets = nn.ModuleDict()
        self.nets["vae"] = SkillNets.SkillVAE(self.algo_config.skill_vae)

        self.nets = self.nets.float().to(self.device)

        self.loss = torch.nn.L1Loss()

        self.action_check_done = False
    
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        input_batch = dict()

        # n-step returns (default is 1)
        n_step = self.algo_config.skill_vae.skill_block_size
        assert batch["actions"].shape[1] >= n_step
        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError('"actions" must be in range [-1,1] for Skill-VAE! Check if hdf5_normalize_action is enabled.')
            self.action_check_done = True
        input_batch["actions"] = batch["actions"][:, :n_step, :]

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
    
    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = PolicyAlgo.train_on_batch(self, batch, epoch, validate=validate)
            pred, pp, pp_sample, aux_loss = self.nets["vae"](batch["actions"])
            recon_loss = self.loss(pred,batch["actions"])
            loss = recon_loss + aux_loss
            if not validate:
                grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets["vae"],
                    optim=self.optimizers["vae"],
                    loss=loss,
                )
                info["grad_norms"] = grad_norms
            info["recon_loss"] = recon_loss.detach().cpu().item()
            info["pp"] = pp.cpu()
            info["pp_sample"] = pp_sample.cpu()
            info["aux_loss"] = aux_loss.detach().cpu().sum().item()
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        loss_log = OrderedDict()
        loss_log["recon_loss"] = info["recon_loss"]
        loss_log["aux_loss"] = info["aux_loss"]
        loss_log["grad_norms"] = info["grad_norms"]
        loss_log["pp"] = info["pp"]
        loss_log["pp_sample"] = info["pp_sample"]
        return loss_log

