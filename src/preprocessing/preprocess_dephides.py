"""
Preprocessing pipeline for DEPHIDES models
- Load collected data
- Tokenize URLs (character-level)
- Split train/val/test
- Save processed data
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import sys
from sklearn.model_selection import train_test_split
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.tokenizer import CharacterTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess data for deep learning models"""
    
    def __init__(self, max_length=200):
        """
        Args:
            max_length: Maximum URL length for padding/truncation
        """
        self.max_length = max_length
        self.tokenizer = CharacterTokenizer(max_length=max_length)
        
    def load_collected_data(self, data_dir='data/processed'):
        """
        Load all collected data from CSV files
        
        Args:
            data_dir: Directory containing collected data
            
        Returns:
            DataFrame with columns: url, label
        """
        logger.info("Loading collected data...")
        
        data_dir = Path(data_dir)
        all_data = []
        
        # Find all CSV files in data/processed
        csv_files = list(data_dir.glob('*.csv'))
        
        if not csv_files:
            logger.error(f"No CSV files found in {data_dir}")
            logger.info("Please run data collection first:")
            logger.info("  python src/data_collection/collect_all_data.py")
            return None
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'url' in df.columns and 'label' in df.columns:
                    all_data.append(df[['url', 'label']])
                    logger.info(f"  ✓ Loaded {len(df)} URLs from {csv_file.name}")
            except Exception as e:
                logger.warning(f"  ✗ Failed to load {csv_file.name}: {e}")
        
        if not all_data:
            logger.error("No valid data found!")
            return None
        
        # Combine all data
        df_combined = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates
        before = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=['url'])
        after = len(df_combined)
        
        logger.info(f"\n✓ Total URLs: {after} (removed {before - after} duplicates)")
        logger.info(f"  - Phishing: {(df_combined['label'] == 1).sum()}")
        logger.info(f"  - Legitimate: {(df_combined['label'] == 0).sum()}")
        
        return df_combined
    
    def tokenize_urls(self, df):
        """
        Tokenize URLs using character-level encoding
        
        Args:
            df: DataFrame with 'url' and 'label' columns
            
        Returns:
            X: numpy array of shape (n_samples, max_length)
            y: numpy array of shape (n_samples,)
        """
        logger.info("\nTokenizing URLs (character-level)...")
        
        # Fit tokenizer on all URLs
        self.tokenizer.fit(df['url'].tolist())
        
        # Encode URLs
        X = self.tokenizer.encode_batch(df['url'].tolist())
        y = df['label'].values
        
        logger.info(f"✓ Tokenization complete")
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  y shape: {y.shape}")
        logger.info(f"  Vocab size: {self.tokenizer.vocab_size}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into train/val/test sets
        
        Args:
            X: Features
            y: Labels
            test_size: Proportion for test set
            val_size: Proportion for validation set (from train)
            random_state: Random seed
            
        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        logger.info("\nSplitting data...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        
        logger.info(f"✓ Data split complete")
        logger.info(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        logger.info(f"  Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        logger.info(f"  Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Check class distribution
        logger.info(f"\n  Train - Phishing: {(y_train==1).sum()}, Legitimate: {(y_train==0).sum()}")
        logger.info(f"  Val   - Phishing: {(y_val==1).sum()}, Legitimate: {(y_val==0).sum()}")
        logger.info(f"  Test  - Phishing: {(y_test==1).sum()}, Legitimate: {(y_test==0).sum()}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def save_processed_data(self, train_data, val_data, test_data, output_dir='data/datasets'):
        """
        Save processed data to disk
        
        Args:
            train_data: (X_train, y_train)
            val_data: (X_val, y_val)
            test_data: (X_test, y_test)
            output_dir: Output directory
        """
        logger.info(f"\nSaving processed data to {output_dir}...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        # Save as numpy arrays
        np.save(output_dir / 'char_X_train.npy', X_train)
        np.save(output_dir / 'char_y_train.npy', y_train)
        np.save(output_dir / 'char_X_val.npy', X_val)
        np.save(output_dir / 'char_y_val.npy', y_val)
        np.save(output_dir / 'char_X_test.npy', X_test)
        np.save(output_dir / 'char_y_test.npy', y_test)
        
        # Save tokenizer
        tokenizer_path = output_dir / 'char_tokenizer.pkl'
        self.tokenizer.save(tokenizer_path)
        
        logger.info(f"✓ All data saved successfully!")
        logger.info(f"  - char_X_train.npy: {X_train.shape}")
        logger.info(f"  - char_y_train.npy: {y_train.shape}")
        logger.info(f"  - char_X_val.npy: {X_val.shape}")
        logger.info(f"  - char_y_val.npy: {y_val.shape}")
        logger.info(f"  - char_X_test.npy: {X_test.shape}")
        logger.info(f"  - char_y_test.npy: {y_test.shape}")
        logger.info(f"  - char_tokenizer.pkl")


def main():
    """Main preprocessing pipeline"""
    print("=" * 80)
    print("DATA PREPROCESSING FOR DEPHIDES MODELS")
    print("=" * 80)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(max_length=200)
    
    # Step 1: Load collected data
    df = preprocessor.load_collected_data('data/processed')
    if df is None or len(df) == 0:
        logger.error("\n❌ No data to process!")
        logger.info("Please run data collection first:")
        logger.info("  python src/data_collection/collect_all_data.py")
        return
    
    # Step 2: Tokenize URLs
    X, y = preprocessor.tokenize_urls(df)
    
    # Step 3: Split data
    train_data, val_data, test_data = preprocessor.split_data(X, y)
    
    # Step 4: Save processed data
    preprocessor.save_processed_data(train_data, val_data, test_data)
    
    print("\n" + "=" * 80)
    print("✓ PREPROCESSING COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Train models: python src/train_all_models.py")
    print("  2. View results in: results/model_comparison.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
