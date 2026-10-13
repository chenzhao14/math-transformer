from datetime import datetime

class Config:
    def __init__(self):

        # Directory for pretokenized data.
        self.pretok_bin_dir = "demo_data/TinyStories_all_data_pretok"

        # Directory for output data.
        self.model_out_dir = "models/temp"

        # Interval for performing evaluation.
        self.eval_interval = 2000

        # Interval for logging outputs.
        self.log_interval = 1

        # Number of iterations to use for evaluation.
        self.eval_iters = 100

        # If True, script exits right after the first eval.
        self.eval_only = False
 
        # If True, always save a checkpoint after each eval.
        self.always_save_checkpoint = False

        # Initialize from scratch or resume from checkpoint.
        self.init_from = "scratch"

        # Wandb logging settings.
        self.wandb_log = False
        self.wandb_project = "Stat8931"
        self.wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # Data handling configurations.
        self.batch_size = 256  # Micro-batch size if using gradient accumulation.
        self.max_seq_len = 256
        self.vocab_size = 2048

        # Model configurations.
        self.dim = 288
        self.n_layers = 6
        self.n_heads = 6
        self.n_kv_heads = 6
        self.multiple_of = 32
        self.dropout = 0.1

        # Optimizer settings.
        self.gradient_accumulation_steps = 4 # must be a multiple of number of gpus
        self.learning_rate = 5e-4
        self.max_iters = 10000
        self.weight_decay = 1e-1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0

        # Learning rate decay.
        self.decay_lr = True
        self.warmup_iters = 1000

        # System-specific configurations.
        self.device = "cpu"
        self.dtype = "float32"
        # self.compile = True

    def to_dict(self):
        """
        Convert the configuration object to a dictionary.
        This is useful for logging, serializations, or passing around configurations.
        """
        return {k: getattr(self, k) for k in dir(self) if not k.startswith('_') and not callable(getattr(self, k))}

if __name__ == "__main__":
    # Usage example
    config = Config()
    config_dict = config.to_dict()
    print(f"config: {config_dict}")
