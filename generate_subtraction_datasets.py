#!/usr/bin/env python3
"""
Generate dedicated subtraction datasets for training and testing
"""

from simple_math_tokenizer import SimpleMathTokenizer

def main():
    # Create tokenizer
    tokenizer = SimpleMathTokenizer()
    
    # Print updated vocabulary
    print("Updated tokenizer vocabulary:")
    tokenizer.print_vocab()
    print()
    
    # Define ranges to generate subtraction datasets for
    ranges = [10, 30, 100, 300]
    
    print("Generating subtraction datasets...")
    print("=" * 50)
    
    for max_val in ranges:
        print(f"\nGenerating subtraction dataset for range 0-{max_val}...")
        
        # Generate and save subtraction datasets
        train_file, test_file = tokenizer.save_subtraction_datasets(
            max_val=max_val, 
            data_dir="data",
            train_ratio=0.8
        )
        
        # Show some examples
        train_data, test_data = tokenizer.split_subtraction_datasets(max_val)
        print(f"  Example training problems:")
        for i, (problem, answer) in enumerate(train_data[:3]):
            print(f"    {problem} {answer}")
        
        print(f"  Example testing problems:")
        for i, (problem, answer) in enumerate(test_data[:3]):
            print(f"    {problem} {answer}")
    
    print(f"\n" + "=" * 50)
    print("All subtraction datasets generated!")
    print("Files created:")
    for max_val in ranges:
        print(f"  - data/subtraction_train_range_{max_val}.txt")
        print(f"  - data/subtraction_test_range_{max_val}.txt")
    
    print(f"\nNote: These are separate from addition datasets and will be used later.")
    print(f"Current training will continue to focus on addition only.")

if __name__ == "__main__":
    main()
