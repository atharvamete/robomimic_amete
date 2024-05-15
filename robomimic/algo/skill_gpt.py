"""
Implementation of Behavioral Cloning (BC).
"""
from collections import OrderedDict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
import numpy as np

import robomimic.models.skill_nets as SkillNets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

@register_algo_factory_func("skill_gpt")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the DQL algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    return Skill_GPT, {}

class Skill_GPT(PolicyAlgo):
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
        self.nets["gpt"] = SkillNets.SkillGPT(self.algo_config.skill_gpt)
        if self.algo_config.skill_vae.path is not None:
            checkpoint = torch.load(self.algo_config.skill_vae.path)['model']
            # remove prefix "vae." from keys
            checkpoint = {k[4:]: v for k, v in checkpoint.items()}
            self.nets["vae"].load_state_dict(checkpoint, strict=True)
        # if not self.algo_config.tune_decoder:
        #     self.nets["vae"].eval()
        #     for param in self.nets["vae"].parameters():
        #         param.requires_grad = False
        # else:
        #     self.nets["vae"].train()

        self.nets = self.nets.float().to(self.device)

        self.loss = torch.nn.L1Loss()
        self.return_offset = True if self.algo_config.skill_gpt.offset_layers > 0 else False
        self.codebook_size = np.array(self.algo_config.skill_vae.fsq_level).prod()
        self.start_token = self.algo_config.skill_gpt.start_token
        self.mpc_horizon = self.algo_config.mpc_horizon
        self.action_queue = deque(maxlen=self.mpc_horizon)

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

        # remove temporal batches for all
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}

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
            with torch.no_grad():
                indices = self.nets["vae"].get_indices(batch["actions"]).long()
            obs = [batch["obs"]['object'], batch["obs"]['robot0_eef_pos'], batch["obs"]['robot0_eef_quat'], batch["obs"]['robot0_gripper_qpos']]
            context = self.nets["gpt"].obs_encoder(obs)
            start_tokens = (torch.ones((context.shape[0], 1))*self.start_token).long().to(self.device)
            x = torch.cat([start_tokens, indices[:,:-1]], dim=1)
            targets = indices.clone()
            logits, prior_loss, offset = self.nets["gpt"](x, context, targets, return_offset=self.return_offset)
            with torch.no_grad():
                logits = logits[:,:,:self.codebook_size]
                probs = torch.softmax(logits, dim=-1)
                sampled_indices = torch.multinomial(probs.view(-1,logits.shape[-1]),1)
                sampled_indices = sampled_indices.view(-1,logits.shape[1])
                pred_actions = self.nets["vae"].decode_actions(sampled_indices)
            if self.return_offset:
                offset = offset.view(-1, self.algo_config.skill_vae.skill_block_size, self.algo_config.skill_vae.action_dim)
                pred_actions = pred_actions + offset
            offset_loss = self.loss(pred_actions, batch["actions"])
            loss = prior_loss + self.algo_config.offset_loss_scale*offset_loss

            if not validate:
                grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets["gpt"],
                    optim=self.optimizers["gpt"],
                    loss=loss,
                )
                info["grad_norms"] = grad_norms
            info["prior_loss"] = prior_loss.detach().cpu().item()
            info["offset_loss"] = offset_loss.detach().cpu().item()
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
        loss_log["prior_loss"] = info["prior_loss"]
        loss_log["offset_loss"] = info["offset_loss"]
        loss_log["grad_norms"] = info["grad_norms"]
        return loss_log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training
        if len(self.action_queue) == 0:
            with torch.no_grad():
                obs = [obs_dict['object'], obs_dict['robot0_eef_pos'], obs_dict['robot0_eef_quat'], obs_dict['robot0_gripper_qpos']]
                actions = self.sample_actions(obs)
                self.action_queue.extend(actions[:self.mpc_horizon])
        action = self.action_queue.popleft()
        return action

    def sample_actions(self, obs):
        context = self.nets["gpt"].obs_encoder(obs)
        x = torch.ones((context.shape[0], 1)).long().to(self.device)*self.start_token
        for i in range(self.algo_config.skill_gpt.block_size):
            if i == self.algo_config.skill_gpt.block_size-1:
                logits,offset = self.nets["gpt"](x, context, return_offset=self.return_offset)
                logits = logits[:,:,:self.codebook_size]
                offset = offset.view(-1, self.algo_config.skill_vae.skill_block_size, self.algo_config.skill_vae.action_dim) if self.return_offset else None
            else:
                logits,_ = self.nets["gpt"](x, context)
                logits = logits[:,:,:self.codebook_size]
            next_indices = self.top_k_sampling(logits[:,-1,:], self.algo_config.skill_gpt.beam_size, self.algo_config.skill_gpt.temperature)
            x = torch.cat([x, next_indices], dim=1)
        sampled_indices = x[:,1:]
        pred_actions = self.nets["vae"].decode_actions(sampled_indices)
        pred_actions_with_offset = pred_actions + offset if offset is not None else pred_actions
        pred_actions_with_offset = pred_actions_with_offset.permute(1,0,2)
        return pred_actions_with_offset.detach().cpu().numpy()

    def top_k_sampling(self, logits, k, temperature=1.0):
        # Apply temperature scaling
        scaled_logits = logits / temperature
        # Find the top k values and indices
        top_values, top_indices = torch.topk(scaled_logits, k, dim=-1)
        # Compute probabilities from top values
        top_probs = torch.softmax(top_values, dim=-1)
        # Sample token index from the filtered probabilities
        sampled_indices = torch.multinomial(top_probs, num_samples=1, replacement=True)
        # Map the sampled index back to the original logits tensor
        original_indices = top_indices.gather(-1, sampled_indices)
        return original_indices