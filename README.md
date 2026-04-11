# LSTM Next Word Predictor (Alice in Wonderland)

A Deep Learning model built with **TensorFlow/Keras** that predicts the next words in a sentence. Trained on *Alice's Adventures in Wonderland* to learn language patterns and word sequences using an LSTM neural network.

## Features

- **Text Tokenization & Padding** — Converts raw text into numerical n-gram sequences
- **LSTM Neural Network** — Captures long-term dependencies in sequential text data
- **Word Embedding** — 100-dimensional dense vectors for contextual word representation
- **Model Persistence** — Saves trained model as `.h5` for instant predictions without retraining

## Architecture

```
Input (seed text)
      │
      ▼
┌─────────────────────┐
│  Tokenizer          │  Convert words → integer sequences
├─────────────────────┤
│  Padding            │  Pad to fixed length (pre-padding)
├─────────────────────┤
│  Embedding (100d)   │  Word index → dense vector
├─────────────────────┤
│  LSTM (150 units)   │  Learn sequential patterns
├─────────────────────┤
│  Dense (softmax)    │  Predict probability over vocabulary
└─────────────────────┘
      │
      ▼
  Predicted next word
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.x |
| Deep Learning | TensorFlow / Keras |
| Data Processing | NumPy |
| Dataset | Project Gutenberg's *Alice's Adventures in Wonderland* |

## Project Structure

```
├── word_predictor.py         # Model architecture, training & prediction logic
├── Alice.txt                 # Source text dataset
├── word_predictor_model.h5   # Pre-trained model weights
├── requirements.txt          # Python dependencies
└── .gitignore
```

## Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/philicamit/LSTM-Next-Word-Predictor.git
cd LSTM-Next-Word-Predictor
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model (optional)

Uncomment the training lines in `word_predictor.py` (lines for `model.fit` and `model.save`), then run:

```bash
python word_predictor.py
```

Training runs for 100 epochs on n-gram sequences generated from the text.

### 5. Run predictions

With the pre-trained model (`word_predictor_model.h5`) in place:

```bash
python word_predictor.py
```

Change the seed word in the script (`t = "author"`) to any word from the vocabulary to predict the next 2 words.

**Example output:**

```
Starting seed: author
Next 2 words after 'author': author of the
```

## How It Works

1. **Load & preprocess** — Reads `Alice.txt`, converts to lowercase
2. **Tokenize** — Builds a word-to-index vocabulary from the entire text
3. **Generate n-grams** — Creates input sequences of increasing length from each line
4. **Pad sequences** — Pre-pads all sequences to uniform length
5. **Train LSTM** — Model learns to predict the next word given preceding words
6. **Predict** — Given a seed word, iteratively predicts the next words by feeding predictions back as input

## License

MIT
