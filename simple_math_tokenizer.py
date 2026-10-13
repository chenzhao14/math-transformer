"""
Simple Math Tokenizer for Addition and Subtraction
Designed specifically for arithmetic expressions like "123 + 456 = 579"
"""

import re
from typing import List, Dict

class SimpleMathTokenizer:
    def __init__(self):
        """
        Initialize a simple tokenizer for math expressions
        Vocabulary includes:
        - Individual digits: 0-9
        - Math operators: +, -, =
        - Spaces (for formatting)
        - Special tokens: <pad>, <bos>, <eos>, <unk>
        """
        # Define the vocabulary (addition and subtraction)
        self.vocab = {
            '<pad>': 0,   # Padding token
            '<bos>': 1,   # Beginning of sequence
            '<eos>': 2,   # End of sequence
            '<unk>': 3,   # Unknown token
            ' ': 4,       # Space
            '+': 5,       # Addition
            '-': 6,       # Subtraction
            '=': 7,       # Equals
            '0': 8,       # Digits start at 8
            '1': 9,
            '2': 10,
            '3': 11,
            '4': 12,
            '5': 13,
            '6': 14,
            '7': 15,
            '8': 16,
            '9': 17,
        }
        
        # Create reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Special token IDs
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        
        # Vocabulary size
        self.vocab_size = len(self.vocab)
        
        print(f"Simple Math Tokenizer (Addition & Subtraction) initialized with vocab size: {self.vocab_size}")
    
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Encode text into token IDs
        
        Args:
            text: Input text (e.g., "123 + 456 = 579")
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token
            
        Returns:
            List of token IDs
        """
        # Convert text to list of characters
        chars = list(text)
        
        # Map each character to token ID
        token_ids = []
        for char in chars:
            if char in self.vocab:
                token_ids.append(self.vocab[char])
            else:
                token_ids.append(self.unk_id)  # Unknown character
        
        # Add special tokens if requested
        if add_bos:
            token_ids = [self.bos_id] + token_ids
        if add_eos:
            token_ids = token_ids + [self.eos_id]
            
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        chars = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                # Skip special tokens in output (except space)
                if token not in ['<pad>', '<bos>', '<eos>', '<unk>']:
                    chars.append(token)
                elif token == '<unk>':
                    chars.append('?')  # Show unknown tokens as ?
            else:
                chars.append('?')  # Unknown token ID
        
        return ''.join(chars)
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into individual tokens (for analysis)
        
        Args:
            text: Input text
            
        Returns:
            List of token strings
        """
        return list(text)
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size"""
        return self.vocab_size
    
    def print_vocab(self):
        """Print the vocabulary for inspection"""
        print("Simple Math Tokenizer Vocabulary:")
        print("-" * 40)
        for token, token_id in sorted(self.vocab.items(), key=lambda x: x[1]):
            if token == ' ':
                print(f"ID {token_id:2d}: '<space>'")
            else:
                print(f"ID {token_id:2d}: '{token}'")
    
    def analyze_expression(self, expression: str):
        """
        Analyze how an expression is tokenized
        
        Args:
            expression: Math expression to analyze
        """
        tokens = self.tokenize_text(expression)
        token_ids = self.encode(expression)
        decoded = self.decode(token_ids)
        
        print(f"Expression: '{expression}'")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Decoded: '{decoded}'")
        print(f"Length: {len(token_ids)} tokens")
        print()

    def generate_subtraction_dataset(self, max_val: int, n_samples: int = None, seed: int = 42):
        """
        Generate subtraction problems for a, b in [0, max_val] where a >= b (no negative results)
        
        Args:
            max_val: Maximum value for operands
            n_samples: Number of samples to generate (None for all valid combinations)
            seed: Random seed for reproducibility
            
        Returns:
            List of (problem, answer) tuples
        """
        import random
        random.seed(seed)
        
        if n_samples is None:
            # Generate all valid combinations (a >= b to avoid negative results)
            data = []
            for a in range(max_val + 1):
                for b in range(a + 1):  # b <= a to ensure non-negative results
                    problem = f"{a} - {b} ="
                    answer = str(a - b)
                    data.append((problem, answer))
        else:
            # Generate random samples
            data = []
            for _ in range(n_samples):
                a = random.randint(0, max_val)
                b = random.randint(0, a)  # Ensure b <= a for non-negative result
                problem = f"{a} - {b} ="
                answer = str(a - b)
                data.append((problem, answer))
        
        return data
    
    def split_subtraction_datasets(self, max_val: int, train_ratio: float = 0.8, seed: int = 42):
        """
        Generate and split subtraction datasets into training and testing sets
        
        Args:
            max_val: Maximum value for operands
            train_ratio: Ratio of data for training (rest goes to testing)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, test_data)
        """
        import random
        random.seed(seed)
        
        # Generate all valid subtraction combinations
        full_data = self.generate_subtraction_dataset(max_val)
        
        # Shuffle the data
        shuffled_data = full_data.copy()
        random.shuffle(shuffled_data)
        
        # Split into train and test
        split_idx = int(len(shuffled_data) * train_ratio)
        train_data = shuffled_data[:split_idx]
        test_data = shuffled_data[split_idx:]
        
        return train_data, test_data
    
    def token_to_id(self, t):          # string -> int id
        return self.vocab.get(t, self.vocab['<unk>'])

    def id_to_token(self, i):          # int id -> string
        return self.inv_vocab.get(i, '<unk>')

    
    def save_subtraction_datasets(self, max_val: int, data_dir: str = "data", train_ratio: float = 0.8):
        """
        Generate and save subtraction datasets to files
        
        Args:
            max_val: Maximum value for operands
            data_dir: Directory to save datasets
            train_ratio: Ratio of data for training
        """
        import os
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate datasets
        train_data, test_data = self.split_subtraction_datasets(max_val, train_ratio)
        
        # Save training dataset
        train_file = os.path.join(data_dir, f"subtraction_train_range_{max_val}.txt")
        with open(train_file, 'w') as f:
            for problem, answer in train_data:
                f.write(f"{problem} {answer}\n")
        
        # Save testing dataset
        test_file = os.path.join(data_dir, f"subtraction_test_range_{max_val}.txt")
        with open(test_file, 'w') as f:
            for problem, answer in test_data:
                f.write(f"{problem} {answer}\n")
        
        print(f"Subtraction datasets saved for range {max_val}:")
        print(f"  Training: {len(train_data)} samples -> {train_file}")
        print(f"  Testing: {len(test_data)} samples -> {test_file}")
        
        return train_file, test_file

# Test the tokenizer
if __name__ == "__main__":
    # Create tokenizer
    tokenizer = SimpleMathTokenizer()
    
    # Print vocabulary
    tokenizer.print_vocab()
    print()
    
    # Test expressions (addition and subtraction)
    test_expressions = [
        "1 + 2 = 3",
        "12 + 34 = 46",
        "123 + 456 = 579",
        "5 - 2 = 3",
        "100 - 25 = 75",
        "456 - 123 = 333"
    ]
    
    print("Tokenization Examples:")
    print("=" * 50)
    for expr in test_expressions:
        tokenizer.analyze_expression(expr)
