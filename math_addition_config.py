"""
Math Addition Training Configuration

"""

from datetime import datetime
from demo_training_config import Config as BaseConfig


class MathAdditionConfig(BaseConfig):
    def __init__(self, pretok_bin_dir, model_out_dir, vocab_size, max_iters=1000):
        """
        Initialize configuration for math addition training
        
        Args:
            pretok_bin_dir: Directory containing pretokenized data
            model_out_dir: Directory for model outputs
            vocab_size: Vocabulary size (should be 18 for our math tokenizer)
            max_iters: Maximum training iterations
        """
        # Initialize base config
        super().__init__()
        
        self.pretok_bin_dir = pretok_bin_dir
        self.model_out_dir = model_out_dir

        # Evaluation and logging
        self.eval_interval = 100
        self.log_interval = 10
        self.eval_iters = 50
        self.eval_only = False
        self.always_save_checkpoint = True
        self.init_from = "scratch"
        
        # Data handling 
        self.batch_size = 64       
        self.max_seq_len = 32       
        self.vocab_size = vocab_size  
        
        # Model configurations 
        self.dim = 128            
        self.n_layers = 4        
        self.n_heads = 4           
        self.n_kv_heads = 4         
        self.multiple_of = 32       
        self.dropout = 0.1          
        
        # Optimizer settings 
        self.gradient_accumulation_steps = 1  
        self.learning_rate = 1e-3   # Higher than demo's 5e-4 for simpler task
        self.max_iters = max_iters
        self.weight_decay = 1e-2    
        self.beta1 = 0.9          
        self.beta2 = 0.95          
        self.grad_clip = 1.0      
        
        # Learning rate decay (following demo)
        self.decay_lr = True
        self.warmup_iters = 100    
        
        # System configurations (force CPU like demo)
        self.device = "cpu"
        self.dtype = "float32"


def create_math_addition_config(pretok_bin_dir, model_out_dir, vocab_size, max_iters=1000):
    """
    Factory function to create a math addition training configuration
    
    Args:
        pretok_bin_dir: Directory containing pretokenized data
        model_out_dir: Directory for model outputs  
        vocab_size: Vocabulary size (should be 18 for our math tokenizer)
        max_iters: Maximum training iterations
        
    Returns:
        MathAdditionConfig instance
    """
    return MathAdditionConfig(pretok_bin_dir, model_out_dir, vocab_size, max_iters)


if __name__ == "__main__":
    # Usage example
    config = create_math_addition_config(
        pretok_bin_dir="pretok_data/simple_math_range_10",
        model_out_dir="models/math_addition_test",
        vocab_size=18,
        max_iters=500
    )
    
    print("Math Addition Training Configuration:")
    print("=" * 50)
    config_dict = config.to_dict()
    for key, value in config_dict.items():
        if not key.startswith('_') and not callable(value):
            print(f"{key}: {value}")
