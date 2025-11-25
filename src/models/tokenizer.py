"""
Character-level and Word-level Tokenizers for URL encoding
"""

import numpy as np
import pickle
import string
from pathlib import Path


class CharacterTokenizer:
    """
    Character-level tokenizer for URLs
    Maps each character to an integer ID
    """
    
    def __init__(self, max_length=200):
        """
        Args:
            max_length: Maximum sequence length for padding/truncation
        """
        self.max_length = max_length
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
        
    def fit(self, urls):
        """
        Build vocabulary from URLs
        
        Args:
            urls: List of URL strings
        """
        # Collect all unique characters
        chars = set()
        for url in urls:
            chars.update(url.lower())
        
        # Sort for consistent ordering
        chars = sorted(list(chars))
        
        # Reserve 0 for padding, start IDs from 1
        self.char_to_id = {char: idx + 1 for idx, char in enumerate(chars)}
        self.id_to_char = {idx + 1: char for idx, char in enumerate(chars)}
        self.vocab_size = len(chars) + 1  # +1 for padding
        
        print(f"✓ Character vocabulary built: {self.vocab_size} characters")
        print(f"  Sample chars: {list(self.char_to_id.keys())[:20]}")
        
        return self
    
    def encode_sequence(self, url):
        """
        Encode a single URL to sequence of character IDs
        
        Args:
            url: URL string
            
        Returns:
            numpy array of shape (max_length,)
        """
        url_lower = url.lower()
        
        # Convert to IDs
        ids = [self.char_to_id.get(char, 0) for char in url_lower]
        
        # Truncate or pad
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        else:
            ids = ids + [0] * (self.max_length - len(ids))
        
        return np.array(ids, dtype=np.int32)
    
    def encode_batch(self, urls):
        """
        Encode multiple URLs
        
        Args:
            urls: List of URL strings
            
        Returns:
            numpy array of shape (len(urls), max_length)
        """
        return np.array([self.encode_sequence(url) for url in urls])
    
    def decode_sequence(self, ids):
        """
        Decode sequence of IDs back to URL string
        
        Args:
            ids: numpy array or list of character IDs
            
        Returns:
            URL string
        """
        chars = [self.id_to_char.get(idx, '') for idx in ids if idx != 0]
        return ''.join(chars)
    
    def save(self, filepath):
        """Save tokenizer to file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'max_length': self.max_length,
                'char_to_id': self.char_to_id,
                'id_to_char': self.id_to_char,
                'vocab_size': self.vocab_size
            }, f)
        
        print(f"✓ Tokenizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load tokenizer from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls(max_length=data['max_length'])
        tokenizer.char_to_id = data['char_to_id']
        tokenizer.id_to_char = data['id_to_char']
        tokenizer.vocab_size = data['vocab_size']
        
        print(f"✓ Tokenizer loaded from {filepath}")
        print(f"  Vocab size: {tokenizer.vocab_size}, Max length: {tokenizer.max_length}")
        
        return tokenizer


# class WordTokenizer:
#     """
#     Word-level tokenizer for URLs
#     Splits URL into tokens (domain parts, path segments, parameters)
#     """
    
#     def __init__(self, max_length=50, vocab_size=10000):
#         """
#         Args:
#             max_length: Maximum sequence length
#             vocab_size: Maximum vocabulary size
#         """
#         self.max_length = max_length
#         self.max_vocab_size = vocab_size
#         self.word_to_id = {}
#         self.id_to_word = {}
#         self.vocab_size = 0
        
#     def tokenize_url(self, url):
#         """
#         Split URL into tokens
        
#         Args:
#             url: URL string
            
#         Returns:
#             List of tokens
#         """
#         import re
        
#         # Remove protocol
#         url = re.sub(r'https?://', '', url.lower())
        
#         # Split by common delimiters
#         tokens = re.split(r'[/\.\-_=&?;:]', url)
        
#         # Remove empty tokens
#         tokens = [t for t in tokens if t]
        
#         return tokens
    
#     def fit(self, urls):
#         """
#         Build vocabulary from URLs
        
#         Args:
#             urls: List of URL strings
#         """
#         from collections import Counter
        
#         # Collect all tokens
#         all_tokens = []
#         for url in urls:
#             all_tokens.extend(self.tokenize_url(url))
        
#         # Count frequencies
#         token_counts = Counter(all_tokens)
        
#         # Keep top tokens
#         top_tokens = token_counts.most_common(self.max_vocab_size - 1)
        
#         # Build vocabulary (reserve 0 for padding)
#         self.word_to_id = {word: idx + 1 for idx, (word, _) in enumerate(top_tokens)}
#         self.id_to_word = {idx + 1: word for idx, (word, _) in enumerate(top_tokens)}
#         self.vocab_size = len(top_tokens) + 1
        
#         print(f"✓ Word vocabulary built: {self.vocab_size} tokens")
#         print(f"  Sample tokens: {list(self.word_to_id.keys())[:20]}")
        
#         return self
    
#     def encode_sequence(self, url):
#         """
#         Encode URL to sequence of token IDs
        
#         Args:
#             url: URL string
            
#         Returns:
#             numpy array of shape (max_length,)
#         """
#         tokens = self.tokenize_url(url)
        
#         # Convert to IDs
#         ids = [self.word_to_id.get(token, 0) for token in tokens]
        
#         # Truncate or pad
#         if len(ids) > self.max_length:
#             ids = ids[:self.max_length]
#         else:
#             ids = ids + [0] * (self.max_length - len(ids))
        
#         return np.array(ids, dtype=np.int32)
    
#     def encode_batch(self, urls):
#         """
#         Encode multiple URLs
        
#         Args:
#             urls: List of URL strings
            
#         Returns:
#             numpy array of shape (len(urls), max_length)
#         """
#         return np.array([self.encode_sequence(url) for url in urls])
    
#     def save(self, filepath):
#         """Save tokenizer to file"""
#         filepath = Path(filepath)
#         filepath.parent.mkdir(parents=True, exist_ok=True)
        
#         with open(filepath, 'wb') as f:
#             pickle.dump({
#                 'max_length': self.max_length,
#                 'max_vocab_size': self.max_vocab_size,
#                 'word_to_id': self.word_to_id,
#                 'id_to_word': self.id_to_word,
#                 'vocab_size': self.vocab_size
#             }, f)
        
#         print(f"✓ Tokenizer saved to {filepath}")
    
#     @classmethod
#     def load(cls, filepath):
#         """Load tokenizer from file"""
#         with open(filepath, 'rb') as f:
#             data = pickle.load(f)
        
#         tokenizer = cls(
#             max_length=data['max_length'],
#             vocab_size=data['max_vocab_size']
#         )
#         tokenizer.word_to_id = data['word_to_id']
#         tokenizer.id_to_word = data['id_to_word']
#         tokenizer.vocab_size = data['vocab_size']
        
#         print(f"✓ Tokenizer loaded from {filepath}")
#         print(f"  Vocab size: {tokenizer.vocab_size}, Max length: {tokenizer.max_length}")
        
#         return tokenizer


if __name__ == "__main__":
    # Test tokenizers
    print("="*80)
    print("Testing Tokenizers")
    print("="*80)
    
    # Sample URLs
    urls = [
        "https://www.google.com/search?q=test",
        "http://example.com/path/to/page",
        "https://phishing-site.com/fake-login.php",
        "https://secure-bank.com/account/login"
    ]
    
    # Test Character Tokenizer
    print("\n1. Character Tokenizer")
    print("-" * 40)
    char_tokenizer = CharacterTokenizer(max_length=50)
    char_tokenizer.fit(urls)
    
    # encoded = char_tokenizer.encode_sequence(urls[0])
    # print(f"\nOriginal URL: {urls[0]}")
    # print(f"Encoded: {encoded}")
    # print(f"Shape: {encoded.shape}")
    
    # decoded = char_tokenizer.decode_sequence(encoded)
    # print(f"Decoded: {decoded}")
    
    # # Test Word Tokenizer
    # print("\n2. Word Tokenizer")
    # print("-" * 40)
    # word_tokenizer = WordTokenizer(max_length=20)
    # word_tokenizer.fit(urls)
    
    # encoded = word_tokenizer.encode_sequence(urls[0])
    # print(f"\nOriginal URL: {urls[0]}")
    # print(f"Tokens: {word_tokenizer.tokenize_url(urls[0])}")
    # print(f"Encoded: {encoded}")
    # print(f"Shape: {encoded.shape}")
    
    # print("\n" + "="*80)
    # print("✓ All tokenizers tested successfully!")
    # print("="*80)
