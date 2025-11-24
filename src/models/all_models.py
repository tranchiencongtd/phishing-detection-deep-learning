"""
Multiple Deep Learning Models for Phishing Detection
Following DEPHIDES paper architecture:
- ANN: Embedding → Dense → Flatten → Dense
- ATT: Embedding → SeqSelfAttention → Flatten → Dense
- RNN: Embedding → LSTM → Dense
- BRNN: Embedding → BidirectionalRNN → Dense
- CNN: Embedding → Conv1D → Flatten → Dense
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np


class SeqSelfAttention(layers.Layer):
    """Sequence Self-Attention Layer"""
    
    def __init__(self, attention_dim=128, **kwargs):
        super(SeqSelfAttention, self).__init__(**kwargs)
        self.attention_dim = attention_dim
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.attention_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.attention_dim,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.attention_dim,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(SeqSelfAttention, self).build(input_shape)
    
    def call(self, inputs):
        # inputs: (batch, seq_len, hidden_dim)
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        attention_weights = tf.nn.softmax(ait, axis=1)
        attention_weights = tf.expand_dims(attention_weights, axis=-1)
        weighted_input = inputs * attention_weights
        return weighted_input
    
    def get_config(self):
        config = super().get_config()
        config.update({'attention_dim': self.attention_dim})
        return config


# ============================================================================
# 1. ANN Model: Embedding → Dense → Flatten → Dense
# ============================================================================
class ANNModel(Model):
    """
    ANN Architecture:
    Input (batch, seq_len) 
    → Embedding (batch, seq_len, embedding_dim)
    → Dense (batch, seq_len, hidden_dim)
    → Flatten (batch, seq_len * hidden_dim)
    → Dense (batch, output_dim)
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, dropout=0.3, **kwargs):
        super(ANNModel, self).__init__(**kwargs)
        
        # Layer 1: Embedding
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=False)
        
        # Layer 2: Dense
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dropout1 = layers.Dropout(dropout)
        
        # Layer 3: Flatten
        self.flatten = layers.Flatten()
        
        # Layer 4: Dense (output)
        self.dense2 = layers.Dense(128, activation='relu')
        self.dropout2 = layers.Dropout(dropout)
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        # Embedding
        x = self.embedding(inputs)  # (batch, seq_len, embedding_dim)
        
        # Dense
        x = self.dense1(x)  # (batch, seq_len, hidden_dim)
        x = self.dropout1(x, training=training)
        
        # Flatten
        x = self.flatten(x)  # (batch, seq_len * hidden_dim)
        
        # Dense (output)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)


# ============================================================================
# 2. ATT Model: Embedding → SeqSelfAttention → Flatten → Dense
# ============================================================================
class ATTModel(Model):
    """
    ATT Architecture:
    Input (batch, seq_len)
    → Embedding (batch, seq_len, embedding_dim)
    → SeqSelfAttention (batch, seq_len, embedding_dim)
    → Flatten (batch, seq_len * embedding_dim)
    → Dense (batch, output_dim)
    """
    
    def __init__(self, vocab_size, embedding_dim=128, attention_dim=128, dropout=0.3, **kwargs):
        super(ATTModel, self).__init__(**kwargs)
        
        # Layer 1: Embedding
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=False)
        
        # Layer 2: SeqSelfAttention
        self.attention = SeqSelfAttention(attention_dim)
        self.dropout1 = layers.Dropout(dropout)
        
        # Layer 3: Flatten
        self.flatten = layers.Flatten()
        
        # Layer 4: Dense (output)
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout2 = layers.Dropout(dropout)
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        # Embedding
        x = self.embedding(inputs)  # (batch, seq_len, embedding_dim)
        
        # SeqSelfAttention
        x = self.attention(x)  # (batch, seq_len, embedding_dim)
        x = self.dropout1(x, training=training)
        
        # Flatten
        x = self.flatten(x)  # (batch, seq_len * embedding_dim)
        
        # Dense (output)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)


# ============================================================================
# 3. RNN Model: Embedding → LSTM → Dense
# ============================================================================
class RNNModel(Model):
    """
    RNN Architecture:
    Input (batch, seq_len)
    → Embedding (batch, seq_len, embedding_dim)
    → LSTM (batch, lstm_units)
    → Dense (batch, output_dim)
    """
    
    def __init__(self, vocab_size, embedding_dim=128, lstm_units=256, dropout=0.3, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        
        # Layer 1: Embedding
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        
        # Layer 2: LSTM (return last output)
        self.lstm = layers.LSTM(
            lstm_units,
            return_sequences=False,  # Only return last output
            dropout=dropout,
            recurrent_dropout=dropout * 0.5
        )
        
        # Layer 3: Dense (output)
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(dropout)
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        # Embedding
        x = self.embedding(inputs)  # (batch, seq_len, embedding_dim)
        
        # LSTM
        x = self.lstm(x, training=training)  # (batch, lstm_units)
        
        # Dense (output)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        return self.output_layer(x)


# ============================================================================
# 4. BRNN Model: Embedding → BidirectionalRNN → Dense
# ============================================================================
class BRNNModel(Model):
    """
    BRNN Architecture:
    Input (batch, seq_len)
    → Embedding (batch, seq_len, embedding_dim)
    → Bidirectional LSTM (batch, lstm_units * 2)
    → Dense (batch, output_dim)
    """
    
    def __init__(self, vocab_size, embedding_dim=128, lstm_units=256, dropout=0.3, **kwargs):
        super(BRNNModel, self).__init__(**kwargs)
        
        # Layer 1: Embedding
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        
        # Layer 2: Bidirectional LSTM
        self.bi_lstm = layers.Bidirectional(
            layers.LSTM(
                lstm_units,
                return_sequences=False,  # Only return last output
                dropout=dropout,
                recurrent_dropout=dropout * 0.5
            )
        )
        
        # Layer 3: Dense (output)
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(dropout)
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        # Embedding
        x = self.embedding(inputs)  # (batch, seq_len, embedding_dim)
        
        # Bidirectional LSTM
        x = self.bi_lstm(x, training=training)  # (batch, lstm_units * 2)
        
        # Dense (output)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        return self.output_layer(x)


# ============================================================================
# 5. CNN Model: Embedding → Conv1D → Flatten → Dense
# ============================================================================
class CNNModel(Model):
    """
    CNN Architecture:
    Input (batch, seq_len)
    → Embedding (batch, seq_len, embedding_dim)
    → Conv1D (batch, seq_len, num_filters)
    → MaxPooling1D (batch, seq_len/2, num_filters)
    → Flatten (batch, seq_len/2 * num_filters)
    → Dense (batch, output_dim)
    """
    
    def __init__(self, vocab_size, embedding_dim=128, num_filters=256, 
                 kernel_size=3, dropout=0.3, **kwargs):
        super(CNNModel, self).__init__(**kwargs)
        
        # Layer 1: Embedding
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=False)
        
        # Layer 2: Conv1D
        self.conv1 = layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu'
        )
        self.pool1 = layers.MaxPooling1D(pool_size=2)
        self.dropout1 = layers.Dropout(dropout)
        
        # Additional Conv layers
        self.conv2 = layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu'
        )
        self.pool2 = layers.MaxPooling1D(pool_size=2)
        self.dropout2 = layers.Dropout(dropout)
        
        # Layer 3: Flatten
        self.flatten = layers.Flatten()
        
        # Layer 4: Dense (output)
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout3 = layers.Dropout(dropout)
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        # Embedding
        x = self.embedding(inputs)  # (batch, seq_len, embedding_dim)
        
        # Conv1D + Pooling
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout2(x, training=training)
        
        # Flatten
        x = self.flatten(x)
        
        # Dense (output)
        x = self.dense1(x)
        x = self.dropout3(x, training=training)
        return self.output_layer(x)


# ============================================================================
# Model Factory
# ============================================================================
def create_model(model_type, config):
    """
    Create model based on type
    
    Args:
        model_type: 'ann', 'att', 'rnn', 'brnn', 'cnn'
        config: Model configuration dict
        
    Returns:
        Model instance
    """
    model_type = model_type.lower()
    
    if model_type == 'ann':
        return ANNModel(**config)
    elif model_type == 'att':
        return ATTModel(**config)
    elif model_type == 'rnn':
        return RNNModel(**config)
    elif model_type == 'brnn':
        return BRNNModel(**config)
    elif model_type == 'cnn':
        return CNNModel(**config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    print("="*80)
    print("Testing All Models - DEPHIDES Architecture")
    print("="*80)
    
    vocab_size = 128  # Character vocabulary
    batch_size = 32
    seq_length = 200
    
    # Test input: sequences of character indices
    dummy_input = tf.random.uniform((batch_size, seq_length), 
                                    minval=0, maxval=vocab_size, dtype=tf.int32)
    
    # Test ANN: Embedding → Dense → Flatten → Dense
    print("\n1. ANN Model: Embedding → Dense → Flatten → Dense")
    ann_model = ANNModel(vocab_size=vocab_size)
    ann_output = ann_model(dummy_input)
    print(f"   Input: {dummy_input.shape} → Output: {ann_output.shape}")
    ann_model.summary()
    
    # Test ATT: Embedding → SeqSelfAttention → Flatten → Dense
    print("\n2. ATT Model: Embedding → SeqSelfAttention → Flatten → Dense")
    att_model = ATTModel(vocab_size=vocab_size)
    att_output = att_model(dummy_input)
    print(f"   Input: {dummy_input.shape} → Output: {att_output.shape}")
    
    # Test RNN: Embedding → LSTM → Dense
    print("\n3. RNN Model: Embedding → LSTM → Dense")
    rnn_model = RNNModel(vocab_size=vocab_size)
    rnn_output = rnn_model(dummy_input)
    print(f"   Input: {dummy_input.shape} → Output: {rnn_output.shape}")
    
    # Test BRNN: Embedding → BidirectionalRNN → Dense
    print("\n4. BRNN Model: Embedding → Bidirectional LSTM → Dense")
    brnn_model = BRNNModel(vocab_size=vocab_size)
    brnn_output = brnn_model(dummy_input)
    print(f"   Input: {dummy_input.shape} → Output: {brnn_output.shape}")
    
    # Test CNN: Embedding → Conv1D → Flatten → Dense
    print("\n5. CNN Model: Embedding → Conv1D → Flatten → Dense")
    cnn_model = CNNModel(vocab_size=vocab_size)
    cnn_output = cnn_model(dummy_input)
    print(f"   Input: {dummy_input.shape} → Output: {cnn_output.shape}")
    
    print("\n" + "="*80)
    print("✓ All models follow DEPHIDES architecture!")
    print("="*80)
