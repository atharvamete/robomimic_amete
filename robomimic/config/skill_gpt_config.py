"""
Config for Skill_GPT.
"""

from robomimic.config.base_config import BaseConfig


class Skill_GPTConfig(BaseConfig):
    ALGO_NAME = "skill_gpt"

    def experiment_config(self):
        """
        Update from subclass to set paper defaults for gym envs.
        """
        super(Skill_GPTConfig, self).experiment_config()
        self.experiment.name = "skill_gpt_32_f4_n6d384_lift_mh"
        self.experiment.validate = False
        self.experiment.render_video = False
        self.experiment.logging.log_tb = False
        self.experiment.save.every_n_epochs = 5 
        self.experiment.save.on_best_rollout_success_rate = True
        self.experiment.epoch_every_n_steps = None
        self.experiment.rollout.enabled = True                      # enable evaluation rollouts
        self.experiment.rollout.n = 50                              # number of rollouts per evaluation
        self.experiment.rollout.horizon = 400                       # maximum number of env steps per rollout
        self.experiment.rollout.rate = 5                           # do rollouts every @rate epochs
        self.experiment.rollout.warmstart = 10                       # number of epochs to wait before starting rollouts
        self.experiment.rollout.terminate_on_success = True         # end rollout early after task success

        self.experiment.logging.wandb_proj_name = "lvm_skill"
    
    def train_config(self):
        """
        Update from subclass to set paper defaults for gym envs.
        """
        super(Skill_GPTConfig, self).train_config()
        self.train.dataset_keys = ("actions",)
        self.train.hdf5_load_next_obs = False
        self.train.seq_length = 32
        self.train.batch_size = 256
        self.train.num_epochs = 50
    
    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """
        # optimization parameters
        self.algo.optim_params.gpt.learning_rate = 0.0001       # gpt learning rate
        self.algo.optim_params.gpt.weight_decay = 0.0001        # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.gpt.betas = [0.9, 0.999]         # 
        self.algo.optim_params.gpt.num_epochs = 50
        self.algo.optim_params.gpt.min_lr = 1e-5                # 
        self.algo.optim_params.gpt.last_epoch = -1              # 

        self.algo.offset_loss_scale = 0.0
        self.algo.mpc_horizon = 16

        self.algo.skill_gpt.start_token = 1000
        self.algo.skill_gpt.vocab_size = 1000
        self.algo.skill_gpt.block_size = 8
        self.algo.skill_gpt.n_layer = 6
        self.algo.skill_gpt.n_head = 6
        self.algo.skill_gpt.n_embd = 384
        self.algo.skill_gpt.attn_pdrop = 0.1
        self.algo.skill_gpt.embd_pdrop = 0.1
        self.algo.skill_gpt.beam_size = 5
        self.algo.skill_gpt.temperature = 1.0
        self.algo.skill_gpt.offset_layers = 0
        self.algo.skill_gpt.offset_hidden_dim = 512

        self.algo.skill_gpt.encoder.input_dim = [10,3,4,2]
        self.algo.skill_gpt.encoder.output_dim = [128,32,32,32]
        self.algo.skill_gpt.encoder.num_layers = [2,1,1,1]
        self.algo.skill_gpt.encoder.dropout = [0.1,0,0,0]
        self.algo.skill_gpt.encoder.proj_dim = 384

        self.algo.skill_vae.path = "/satassdscratch/amete7/robomimic_amete/skill_vae_trained_models/skill_vae_32_f4_k3s4_lift_mh/20240509170634/models/model_epoch_100.pth"
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
