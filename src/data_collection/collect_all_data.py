"""
Complete data collection pipeline for phishing detection
Crawl from multiple sources automatically
"""

import requests
import json
import pandas as pd
from datetime import datetime
import time
from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm
import csv
import zipfile
import io

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhishingDataCollector:
    """Complete data collection from all sources"""
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_phishtank(self) -> pd.DataFrame:
        """Collect from PhishTank"""
        logger.info("Collecting from PhishTank...")
        try:
            url = "http://data.phishtank.com/data/online-valid.csv"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            
            # Read CSV
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            # Ensure 'url' column exists
            if 'url' not in df.columns:
                logger.warning("PhishTank: 'url' column not found, skipping...")
                return pd.DataFrame()
            
            df['source'] = 'phishtank'
            df['label'] = 1
            
            logger.info(f"✓ PhishTank: {len(df)} URLs")
            return df[['url', 'source', 'label']]
        except Exception as e:
            logger.error(f"✗ PhishTank error: {e}")
            logger.info("Skipping PhishTank, continuing with other sources...")
            return pd.DataFrame()
    
    def collect_openphish(self) -> pd.DataFrame:
        """Collect from OpenPhish"""
        logger.info("Collecting from OpenPhish...")
        try:
            url = "https://openphish.com/feed.txt"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            urls = response.text.strip().split('\n')
            df = pd.DataFrame({
                'url': urls,
                'source': 'openphish',
                'label': 1
            })
            
            logger.info(f"✓ OpenPhish: {len(df)} URLs")
            return df
        except Exception as e:
            logger.error(f"✗ OpenPhish error: {e}")
            return pd.DataFrame()
    
    def collect_urlhaus(self) -> pd.DataFrame:
        """Collect from URLhaus"""
        logger.info("Collecting from URLhaus...")
        try:
            url = "https://urlhaus.abuse.ch/downloads/csv_recent/"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            lines = response.text.strip().split('\n')
            
            # Find header line (starts with #)
            header_line = None
            data_lines = []
            for line in lines:
                if line.startswith('#') and 'id' in line and 'url' in line:
                    # Remove # and strip whitespace from header
                    header_line = line.lstrip('#').strip()
                elif not line.startswith('#') and line.strip():
                    data_lines.append(line)
            
            if not header_line or not data_lines:
                logger.warning("URLhaus: No valid data found, skipping...")
                return pd.DataFrame()
            
            # Combine header and data
            csv_content = header_line + '\n' + '\n'.join(data_lines)
            
            from io import StringIO
            df = pd.read_csv(StringIO(csv_content), quotechar='"')
            
            # Check if 'url' column exists
            if 'url' not in df.columns:
                logger.warning(f"URLhaus: 'url' column not found. Available columns: {df.columns.tolist()}")
                return pd.DataFrame()
            
            df_result = pd.DataFrame()
            df_result['url'] = df['url']
            df_result['source'] = 'urlhaus'
            df_result['label'] = 1
            
            logger.info(f"✓ URLhaus: {len(df_result)} URLs")
            return df_result
        except Exception as e:
            logger.error(f"✗ URLhaus error: {e}")
            logger.info("Skipping URLhaus, continuing with other sources...")
            return pd.DataFrame()
    
    def collect_tranco(self, top_n=100000) -> pd.DataFrame:
        """Collect from Tranco (legitimate sites)"""
        logger.info(f"Collecting top {top_n} from Tranco...")
        try:
            url = "https://tranco-list.eu/top-1m.csv.zip"
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    df = pd.read_csv(f, names=['rank', 'domain'], nrows=top_n)
            
            df['url'] = 'https://' + df['domain']
            df['source'] = 'tranco'
            df['label'] = 0
            
            logger.info(f"✓ Tranco: {len(df)} URLs")
            return df[['url', 'source', 'label']]
        except Exception as e:
            logger.error(f"✗ Tranco error: {e}")
            return pd.DataFrame()
    
    def collect_majestic(self, top_n=50000) -> pd.DataFrame:
        """Collect from Majestic Million (legitimate sites)"""
        logger.info(f"Collecting top {top_n} from Majestic Million...")
        try:
            url = "https://downloads.majestic.com/majestic_million.csv"
            df = pd.read_csv(url, nrows=top_n)
            
            df['url'] = 'https://' + df['Domain']
            df['source'] = 'majestic'
            df['label'] = 0
            
            logger.info(f"✓ Majestic: {len(df)} URLs")
            return df[['url', 'source', 'label']]
        except Exception as e:
            logger.error(f"✗ Majestic error: {e}")
            return pd.DataFrame()
    
    def collect_all_phishing(self) -> pd.DataFrame:
        """Collect all phishing URLs"""
        logger.info("\n" + "="*80)
        logger.info("COLLECTING PHISHING URLs")
        logger.info("="*80)
        
        dfs = []
        
        # PhishTank
        df = self.collect_phishtank()
        if not df.empty:
            dfs.append(df)
        time.sleep(2)
        
        # OpenPhish
        df = self.collect_openphish()
        if not df.empty:
            dfs.append(df)
        time.sleep(2)
        
        # URLhaus
        df = self.collect_urlhaus()
        if not df.empty:
            dfs.append(df)
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            combined = combined.drop_duplicates(subset=['url'], keep='first')
            logger.info(f"\n✓ Total phishing URLs: {len(combined)}")
            return combined
        else:
            logger.error("No phishing data collected!")
            return pd.DataFrame()
    
    def collect_all_legitimate(self, max_per_source=100000) -> pd.DataFrame:
        """Collect all legitimate URLs"""
        logger.info("\n" + "="*80)
        logger.info("COLLECTING LEGITIMATE URLs")
        logger.info("="*80)
        
        dfs = []
        
        # Tranco
        df = self.collect_tranco(max_per_source)
        if not df.empty:
            dfs.append(df)
        time.sleep(2)
        
        # Majestic
        # df = self.collect_majestic(max_per_source)
        # if not df.empty:
        #     dfs.append(df)
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            combined = combined.drop_duplicates(subset=['url'], keep='first')
            logger.info(f"\n✓ Total legitimate URLs: {len(combined)}")
            return combined
        else:
            logger.error("No legitimate data collected!")
            return pd.DataFrame()
    
    def balance_dataset(self, df_phishing: pd.DataFrame, 
                       df_legitimate: pd.DataFrame) -> pd.DataFrame:
        """Balance phishing and legitimate datasets"""
        logger.info("\n" + "="*80)
        logger.info("BALANCING DATASET")
        logger.info("="*80)
        
        min_size = min(len(df_phishing), len(df_legitimate))
        logger.info(f"Using {min_size} samples from each class")
        
        df_phishing_sampled = df_phishing.sample(n=min_size, random_state=42)
        df_legitimate_sampled = df_legitimate.sample(n=min_size, random_state=42)
        
        df_combined = pd.concat([df_phishing_sampled, df_legitimate_sampled], 
                               ignore_index=True)
        df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"✓ Balanced dataset: {len(df_combined)} URLs")
        logger.info(f"  - Phishing: {(df_combined['label'] == 1).sum()}")
        logger.info(f"  - Legitimate: {(df_combined['label'] == 0).sum()}")
        
        return df_combined
    
    def split_dataset(self, df: pd.DataFrame, 
                     train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split into train/val/test"""
        logger.info("\n" + "="*80)
        logger.info("SPLITTING DATASET")
        logger.info("="*80)
        
        from sklearn.model_selection import train_test_split
        
        # First split: train and temp
        train_df, temp_df = train_test_split(
            df, test_size=(1 - train_ratio), random_state=42, stratify=df['label']
        )
        
        # Second split: val and test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df, test_size=(1 - val_size), random_state=42, stratify=temp_df['label']
        )
        
        logger.info(f"✓ Train: {len(train_df)} samples")
        logger.info(f"✓ Val:   {len(val_df)} samples")
        logger.info(f"✓ Test:  {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def save_datasets(self, train_df, val_df, test_df):
        """Save datasets"""
        logger.info("\n" + "="*80)
        logger.info("SAVING DATASETS")
        logger.info("="*80)
        
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        train_file = processed_dir / f"train_{timestamp}.csv"
        val_file = processed_dir / f"val_{timestamp}.csv"
        test_file = processed_dir / f"test_{timestamp}.csv"
        
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        logger.info(f"✓ Saved to:")
        logger.info(f"  - {train_file}")
        logger.info(f"  - {val_file}")
        logger.info(f"  - {test_file}")
        
        return train_file, val_file, test_file


def main():
    """Main data collection pipeline"""
    print("\n" + "="*80)
    print("PHISHING DETECTION - DATA COLLECTION PIPELINE")
    print("="*80 + "\n")
    
    collector = PhishingDataCollector()
    
    # Step 1: Collect phishing URLs
    df_phishing = collector.collect_all_phishing()
    
    # Step 2: Collect legitimate URLs
    df_legitimate = collector.collect_all_legitimate(max_per_source=100000)
    
    if df_phishing.empty or df_legitimate.empty:
        logger.error("\n✗ Data collection failed!")
        return
    
    # Step 3: Balance dataset
    df_balanced = collector.balance_dataset(df_phishing, df_legitimate)
    
    # Step 4: Split dataset
    train_df, val_df, test_df = collector.split_dataset(df_balanced)
    
    # Step 5: Save datasets
    train_file, val_file, test_file = collector.save_datasets(train_df, val_df, test_df)
    
    # Summary
    print("\n" + "="*80)
    print("DATA COLLECTION COMPLETE!")
    print("="*80)
    print(f"\n📊 Dataset Summary:")
    print(f"  Total URLs collected: {len(df_balanced)}")
    print(f"  - Phishing:   {(df_balanced['label'] == 1).sum():,}")
    print(f"  - Legitimate: {(df_balanced['label'] == 0).sum():,}")
    print(f"\n📁 Files saved:")
    print(f"  - Training:   {train_file}")
    print(f"  - Validation: {val_file}")
    print(f"  - Test:       {test_file}")
    print("\n✓ Ready for preprocessing!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
