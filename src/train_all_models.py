"""
Train and compare all 5 models: ANN, ATT, RNN, BRNN, CNN
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import json
import sys
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.all_models import create_model
from models.tokenizer import CharacterTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiModelTrainer:
    """Train multiple models and compare performance"""
    
    def __init__(self):
        self.models = {}
        self.histories = {}
        self.results = {}
        self.tokenizer = None
    
    def load_data(self):
        """Load preprocessed data"""
        logger.info("Loading preprocessed data...")
        
        data_dir = Path("data/datasets")
        
        X_train = np.load(data_dir / 'char_X_train.npy')
        y_train = np.load(data_dir / 'char_y_train.npy')
        X_val = np.load(data_dir / 'char_X_val.npy')
        y_val = np.load(data_dir / 'char_y_val.npy')
        X_test = np.load(data_dir / 'char_X_test.npy')
        y_test = np.load(data_dir / 'char_y_test.npy')
        
        logger.info(f"Train: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Val:   X={X_val.shape}, y={y_val.shape}")
        logger.info(f"Test:  X={X_test.shape}, y={y_test.shape}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def load_tokenizer(self):
        """Load tokenizer"""
        datasets_dir = Path("data/datasets")
        tokenizer_path = datasets_dir / 'char_tokenizer.pkl'
        self.tokenizer = CharacterTokenizer.load(str(tokenizer_path))
        logger.info(f"Tokenizer loaded: vocab_size={self.tokenizer.vocab_size}")
    
    def build_all_models(self, vocab_size):
        """Build all 5 models"""
        logger.info("\n" + "="*80)
        logger.info("BUILDING ALL MODELS")
        logger.info("="*80)
        
        # Model configurations
        configs = {
            'ann': {
                'vocab_size': vocab_size,
                'embedding_dim': 128,
                'hidden_dim': 256,
                'dropout': 0.3
            },
            'att': {
                'vocab_size': vocab_size,
                'embedding_dim': 128,
                'attention_dim': 128,
                'dropout': 0.3
            },
            'rnn': {
                'vocab_size': vocab_size,
                'embedding_dim': 128,
                'lstm_units': 256,
                'dropout': 0.3
            },
            'brnn': {
                'vocab_size': vocab_size,
                'embedding_dim': 128,
                'lstm_units': 256,
                'dropout': 0.3
            },
            'cnn': {
                'vocab_size': vocab_size,
                'embedding_dim': 128,
                'num_filters': 256,
                'kernel_size': 3,
                'dropout': 0.3
            }
        }
        
        for model_name, config in configs.items():
            logger.info(f"\n✓ Building {model_name.upper()} model...")
            self.models[model_name] = create_model(model_name, config)
    
    def compile_models(self):
        """Compile all models"""
        logger.info("\nCompiling models...")
        
        for model_name, model in self.models.items():
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')
                ]
            )
            logger.info(f"✓ {model_name.upper()} compiled")
    
    def get_callbacks(self, model_name):
        """Get callbacks for training"""
        checkpoint_dir = Path(f"trained_models/multi_models/{model_name}/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        log_dir = Path(f"logs/multi_models/{model_name}/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_dir / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            ),
            keras.callbacks.TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def train_model(self, model_name, model, X_train, y_train, X_val, y_val, 
                   epochs=50, batch_size=128):
        """Train a single model"""
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING {model_name.upper()} MODEL")
        logger.info(f"{'='*80}")
        
        # Special handling for ANN (needs flattened input)
        if model_name == 'ann':
            # Average pool the character embeddings
            X_train_ann = np.mean(X_train.reshape(X_train.shape[0], -1, 1), axis=1)
            X_val_ann = np.mean(X_val.reshape(X_val.shape[0], -1, 1), axis=1)
            
            # Use first 100 features
            X_train_use = X_train_ann[:, :100]
            X_val_use = X_val_ann[:, :100]
        else:
            X_train_use = X_train
            X_val_use = X_val
        
        # Calculate class weights
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight = dict(zip(classes, weights))
        
        # Train
        history = model.fit(
            X_train_use, y_train,
            validation_data=(X_val_use, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(model_name),
            class_weight=class_weight,
            verbose=1
        )
        
        self.histories[model_name] = history
        return history
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        """Evaluate a single model"""
        logger.info(f"\nEvaluating {model_name.upper()} model...")
        
        # Special handling for ANN
        if model_name == 'ann':
            X_test_ann = np.mean(X_test.reshape(X_test.shape[0], -1, 1), axis=1)
            X_test_use = X_test_ann[:, :100]
        else:
            X_test_use = X_test
        
        # Evaluate
        results = model.evaluate(X_test_use, y_test, verbose=0)
        metrics = dict(zip(model.metrics_names, results))
        
        # Predictions
        y_pred_proba = model.predict(X_test_use, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Classification report
        report = classification_report(y_test, y_pred, 
                                      target_names=['Legitimate', 'Phishing'],
                                      output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        self.results[model_name] = {
            'metrics': metrics,
            'report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred
        }
        
        logger.info(f"✓ {model_name.upper()} Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  AUC:       {metrics['auc']:.4f}")
        
        return metrics
    
    def train_all(self, X_train, y_train, X_val, y_val, epochs=50):
        """Train all models"""
        logger.info("\n" + "="*80)
        logger.info("TRAINING ALL MODELS")
        logger.info("="*80)
        
        for model_name, model in self.models.items():
            self.train_model(model_name, model, X_train, y_train, X_val, y_val, epochs)
    
    def evaluate_all(self, X_test, y_test):
        """Evaluate all models"""
        logger.info("\n" + "="*80)
        logger.info("EVALUATING ALL MODELS")
        logger.info("="*80)
        
        for model_name, model in self.models.items():
            self.evaluate_model(model_name, model, X_test, y_test)
    
    def compare_results(self):
        """Compare all model results"""
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON")
        logger.info("="*80)
        
        comparison = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison.append({
                'Model': model_name.upper(),
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'AUC': f"{metrics['auc']:.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison)
        print("\n" + df_comparison.to_string(index=False))
        
        # Save comparison
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        df_comparison.to_csv(results_dir / 'model_comparison.csv', index=False)
        
        # Save detailed results
        with open(results_dir / 'detailed_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\n✓ Results saved to {results_dir}/")
    
    def save_models(self):
        """Save all trained models"""
        logger.info("\nSaving models...")
        
        models_dir = Path("trained_models/multi_models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for model_name, model in self.models.items():
            model_path = models_dir / f"{model_name}_model_{timestamp}.h5"
            model.save(model_path)
            logger.info(f"✓ Saved {model_name.upper()} to {model_path}")


def main():
    """Main training function"""
    print("\n" + "="*80)
    print("MULTI-MODEL TRAINING: ANN, ATT, RNN, BRNN, CNN")
    print("="*80 + "\n")
    
    trainer = MultiModelTrainer()
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = trainer.load_data()
    
    # Load tokenizer
    trainer.load_tokenizer()
    
    # Build models
    trainer.build_all_models(vocab_size=trainer.tokenizer.vocab_size)
    
    # Compile models
    trainer.compile_models()
    
    # Train all models
    trainer.train_all(X_train, y_train, X_val, y_val, epochs=30)
    
    # Evaluate all models
    trainer.evaluate_all(X_test, y_test)
    
    # Compare results
    trainer.compare_results()
    
    # Save models
    trainer.save_models()
    
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
