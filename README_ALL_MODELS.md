# Phishing Detection with 5 Deep Learning Models

Dự án phát hiện website lừa đảo sử dụng 5 models Deep Learning khác nhau.

## 🎯 Models

1. **ANN** - Artificial Neural Network (Feed-Forward)
2. **ATT** - Attention-based Network
3. **RNN** - Recurrent Neural Network (LSTM)
4. **BRNN** - Bidirectional RNN
5. **CNN** - Convolutional Neural Network

## 📊 Architecture

```
Input URL → Character Tokenization → Deep Learning Model → Phishing/Legitimate
```

### Model Details:

**ANN**: Dense layers with dropout
- Simple but effective baseline
- Fast training and inference

**ATT**: Embedding + Attention mechanism
- Focus on important parts of URL
- Good for variable-length inputs

**RNN**: LSTM layers
- Captures sequential patterns
- Learns temporal dependencies

**BRNN**: Bidirectional LSTM
- Processes URL forward and backward
- Better context understanding

**CNN**: Conv1D layers
- Extracts local patterns
- Efficient parallel processing

## 🚀 Quick Start

### Option 1: Run complete pipeline
```bash
# Windows
run_all_models.bat

# Linux/Mac
chmod +x run_all_models.sh
./run_all_models.sh
```

### Option 2: Step by step

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect data (auto-crawl from multiple sources)
python src/data_collection/collect_all_data.py

# 3. Preprocess
python src/preprocessing/preprocess_dephides.py --tokenizer char

# 4. Train all models
python src/train_all_models.py

# 5. Predict
python src/predict_dephides.py --url "http://example.com" --model brnn
```

## 📈 Dataset

**Phishing Sources** (auto-crawled):
- PhishTank
- OpenPhish  
- URLhaus

**Legitimate Sources** (auto-crawled):
- Tranco (top 1M websites)
- Majestic Million

**Total**: ~100K URLs (balanced)

## 🎓 Training

```bash
# Train all 5 models at once
python src/train_all_models.py

# Or train individual model
python src/train_dephides.py --model brnn
```

Models supported: `ann`, `att`, `rnn`, `brnn`, `cnn`

## 📊 Results

After training, check:
- `results/model_comparison.csv` - Performance comparison
- `results/detailed_results.json` - Full metrics
- `trained_models/multi_models/` - Saved models

## 🔮 Prediction

```bash
# Single URL
python src/predict_dephides.py --url "http://phishing-site.com" --model brnn

# Batch prediction from file
python src/predict_dephides.py --file urls.txt --model cnn
```

## 📁 Project Structure

```
phishing-detection/
├── src/
│   ├── data_collection/
│   │   └── collect_all_data.py      # Auto-crawl data
│   ├── preprocessing/
│   │   └── preprocess_dephides.py   # Tokenization
│   ├── models/
│   │   ├── all_models.py            # 5 model definitions
│   │   └── tokenizer.py             # Character tokenizer
│   ├── train_all_models.py          # Train all at once
│   └── predict_dephides.py          # Inference
├── data/                             # Datasets
├── trained_models/                   # Saved models
├── results/                          # Performance metrics
├── config/                           # Configuration files
└── requirements.txt                  # Dependencies
```

## 🎯 Performance (Expected)

| Model | Accuracy | Precision | Recall | AUC |
|-------|----------|-----------|--------|-----|
| ANN   | ~95%     | ~94%      | ~96%   | ~98%|
| ATT   | ~96%     | ~95%      | ~97%   | ~98%|
| RNN   | ~96%     | ~95%      | ~97%   | ~99%|
| BRNN  | ~97%     | ~96%      | ~98%   | ~99%|
| CNN   | ~96%     | ~95%      | ~97%   | ~99%|

**Best Model**: BRNN (Bidirectional RNN)

## ⚙️ Configuration

Edit `config/config_dephides.yaml` to customize:
- Model hyperparameters
- Training settings
- Data collection sources
- Batch size, epochs, etc.

## 🔧 Requirements

- Python 3.8+
- TensorFlow 2.13+
- pandas, numpy, scikit-learn
- requests, beautifulsoup4
- See `requirements.txt` for full list

## 💡 Tips

1. **GPU recommended** for faster training (but not required)
2. **Start with BRNN** - usually gives best results
3. **Increase epochs** if needed (default: 30)
4. **Use ensemble** - combine predictions from multiple models
5. **Monitor with TensorBoard**: `tensorboard --logdir logs/`

## 🐛 Troubleshooting

**Out of memory**: Reduce `batch_size` in config

**Data collection fails**: Check internet connection, some sources may be temporarily down

**Model not converging**: Try different learning rate or increase epochs

## 📚 References

- DEPHIDES: Deep Learning Based Phishing Detection System
- Character-level CNN for text classification
- Attention mechanisms in neural networks

## 📧 Support

Open an issue if you encounter problems or have questions!

---

**✨ Features:**
- ✅ Auto-crawl data from 5+ sources
- ✅ 5 different deep learning architectures
- ✅ Character-level tokenization (end-to-end)
- ✅ Balanced dataset with train/val/test splits
- ✅ Model comparison and evaluation
- ✅ Easy prediction interface
- ✅ Complete pipeline automation
