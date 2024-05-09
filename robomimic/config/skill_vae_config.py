"""
Config for Skill_VAE.
"""

from robomimic.config.base_config import BaseConfig


class Skill_VAEConfig(BaseConfig):
    ALGO_NAME = "skill_vae"

    def experiment_config(self):
        """
        Update from subclass to set paper defaults for gym envs.
        """
        super(Skill_VAEConfig, self).experiment_config()
        self.experiment.name = "skill_vae_32_f4_k3s4"
        self.experiment.validate = False
        self.experiment.render_video = False
        self.experiment.logging.log_tb = False
        self.experiment.save.every_n_epochs = 20 
        self.experiment.save.on_best_rollout_success_rate = False
        self.experiment.epoch_every_n_steps = None
        self.experiment.rollout.enabled = False
        self.experiment.logging.wandb_proj_name = "lvm_skill"
    
    def train_config(self):
        """
        Update from subclass to set paper defaults for gym envs.
        """
        super(Skill_VAEConfig, self).train_config()
        self.train.dataset_keys = ("actions",)
        self.train.hdf5_load_next_obs = False
        self.train.seq_length = 32
        self.train.batch_size = 256
        self.train.num_epochs = 100
    
    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """
        # optimization parameters
        self.algo.optim_params.vae.learning_rate = 0.0001       # vae learning rate
        self.algo.optim_params.vae.weight_decay = 0.0001        # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.vae.betas = [0.9, 0.999]         # 
        self.algo.optim_params.vae.num_epochs = 100
        self.algo.optim_params.vae.min_lr = 1e-5                # 
        self.algo.optim_params.vae.last_epoch = -1              # 

        self.algo.skill_vae.action_dim = 7
        self.algo.skill_vae.encoder_dim = 256
        self.algo.skill_vae.decoder_dim = 256
        self.algo.skill_vae.skill_block_size = 32 # this is input sequence length to encoder

        self.algo.skill_vae.encoder_heads = 4
        self.algo.skill_vae.encoder_layers = 2
        self.algo.skill_vae.decoder_heads = 4
        self.algo.skill_vae.decoder_layers = 4

        self.algo.skill_vae.attn_pdrop = 0.1
        self.algo.skill_vae.use_causal_encoder = True
        self.algo.skill_vae.use_causal_decoder = True

        self.algo.skill_vae.vq_type = "fsq" # "vq" or "fsq"
        self.algo.skill_vae.fsq_level = [8,5,5,5]
        self.algo.skill_vae.codebook_dim = 512 # only used for vq
        self.algo.skill_vae.codebook_size = 1024 # only used for vq

        self.algo.skill_vae.kernel_sizes = [5,3,3] # conv module will have 3 layers with kernel sizes 5,3,3
        self.algo.skill_vae.strides = [2,2,1] # conv module will have 3 layers with strides 2,2,1
