"""
Dataset Generator for Addition Problems
Utility functions for generating and managing addition datasets for LLM training
"""

import random
import os


def generate_addition_dataset(max_val, n_samples=None, seed=42):
    """
    Generate addition problems for a, b in [0, max_val]
    If n_samples is None, generate all possible combinations
    
    Args:
        max_val: Maximum value for operands a and b
        n_samples: Number of samples to generate (None for all combinations)
        seed: Random seed for reproducibility
        
    Returns:
        List of (problem, answer) tuples
    """
    random.seed(seed)
    
    if n_samples is None:
        # Generate all possible combinations
        data = []
        for a in range(max_val + 1):
            for b in range(max_val + 1):
                problem = f"{a} + {b} ="
                answer = str(a + b)
                data.append((problem, answer))
    else:
        # Generate random samples
        data = []
        for _ in range(n_samples):
            a = random.randint(0, max_val)
            b = random.randint(0, max_val)
            problem = f"{a} + {b} ="
            answer = str(a + b)
            data.append((problem, answer))
    
    return data


def save_dataset_for_training(data, filename, stop_seq=""):
    """
    Save dataset in format expected by the training pipeline
    
    Args:
        data: List of (problem, answer) tuples
        filename: Output filename
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        for problem, answer in data:
            # Format as "problem answer" for language modeling
            f.write(f"{problem} {answer}{stop_seq}\n")


def split_train_val(data, val_ratio=0.1, seed=42):
    """
    Split data into train and validation sets.
    NOTE: Validation will overlap with training (train contains all samples).
    """
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * (1 - val_ratio))
    train_data = shuffled  # allow overlap: keep all data for training
    val_data = shuffled[split_idx:]
    
    return train_data, val_data


def generate_datasets_for_ranges(ranges, n_samples=50000, data_dir="data", val_ratio=0.1, seed=42, stop_seq=""):
    """
    Generate addition datasets for multiple ranges
    """
    datasets = {}
    
    for max_val in ranges:
        print(f"Generating data for range 0-{max_val}...")
        
        if max_val <= 30:
            data = generate_addition_dataset(max_val, seed=seed)
        else:
            n_samples = min(50000, (max_val + 1) ** 2)
            data = generate_addition_dataset(max_val, n_samples, seed=seed)
        
        train_data, val_data = split_train_val(data, val_ratio, seed=seed)
        datasets[max_val] = {
            'train': train_data,
            'val': val_data,
            'full': data
        }
        
        print(f"Range {max_val}: {len(train_data)} train, {len(val_data)} val samples")
        
        # Save datasets to files, appending stop sequence
        save_dataset_for_training(train_data, f'{data_dir}/train_range_{max_val}.txt', stop_seq=stop_seq)
        save_dataset_for_training(val_data, f'{data_dir}/val_range_{max_val}.txt', stop_seq=stop_seq)
    
    print(f"\nDatasets saved to '{data_dir}/' directory")
    return datasets


if __name__ == "__main__":
    # Example usage
    ranges = [10, 30, 100, 300]
    datasets = generate_datasets_for_ranges(ranges)
    
    print("\nDataset generation complete!")
    for max_val in ranges:
        total = len(datasets[max_val]['full'])
        train = len(datasets[max_val]['train'])
        val = len(datasets[max_val]['val'])
        print(f"Range {max_val}: {total} total ({train} train, {val} val)")
