# Trainer Configuration and Factory
# Provides trainers for different model types with unified configuration management

from dataclasses import dataclass
from pathlib import Path
import gin

# Import trainer implementations
from .unet_diffusion import UNetFullDeformDiffuseTrainer
from .unet_determ import UNetFullDeformDetermTrainer
from .wmap_vaekl import WMapVAEKLTrainer


@gin.configurable()
@dataclass
class TrainerConfig:
    # Configuration dataclass for all trainer implementations
    # Manages model paths, training hyperparameters, and output directories
    __trainer_name__: str = "WMapVAEKLTrainer"

    # Model paths and revisions
    non_ema_revision: str = None
    output_dir: str = None
    exp_name: str = "debug"
    logging_name: str = "logs"
    
    # Training hyperparameters
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "no"
    report_to: str = "tensorboard"
    seed: int = 42
    
    # Pretrained model configuration
    pretrained_model_name_or_path: str = (
        "stabilityai/stable-video-diffusion-img2vid-xt"
    )
    cache_dir: str = 'cache/'
    revision: str = None  # Revision of pretrained model from HuggingFace
    
    # VAE and UNet configuration
    vae_type: str = None
    pretrained_vae: str = None
    unet_type: str = None
    pretrained_unet: str = None
    
    # Model architecture options
    conv_padding_mode: str = 'zeros'  # 'zeros', 'replicate', 'reflect', etc.
    use_ema: bool = False
    enable_xformers_memory_efficient_attention: bool = True
    gradient_checkpointing: bool = True
    # Whether to allow TF32 on Ampere GPUs for faster training
    allow_tf32: bool = False
    
    # Learning rate and optimization
    scale_lr: bool = False
    per_gpu_batch_size: int = 1
    use_8bit_adam: bool = False
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    
    # Data and training schedule
    num_workers: int = 32
    num_train_epochs: int = None
    max_train_steps: int = 20000
    learning_rate: float = 1e-5
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    resume_from_checkpoint: str = None
    
    # Checkpointing and validation
    checkpointing_steps: int = 2500
    checkpoints_total_limit: int = 20
    validation_steps: int = 2500
    save_sequence: bool = False
    
    # Model conditioning and data processing
    conditioning_dropout_prob: float = 0.1
    train_params: str = "all"
    empty_cache_per_iter: bool = False
    
    # Video dimensions
    height: int = 320
    width: int = 640
    eval_dataset: str = None

    def __post_init__(self):
        # Create output directories based on configuration
        self.output_path = Path(self.output_dir)
        self.exp_path = self.output_path / self.exp_name
        self.logging_path = self.exp_path / self.logging_name

        # Create directories if they don't exist
        self.exp_path.mkdir(parents=True, exist_ok=True)
        self.logging_path.mkdir(parents=True, exist_ok=True)

    def build(self):
        # Factory method: instantiate trainer based on __trainer_name__
        # Returns the appropriate trainer instance configured with this config
        return globals()[self.__trainer_name__](self)
