# LSTM Next Word Predictor (Alice in Wonderland)

This project is a Deep Learning model built using **TensorFlow/Keras** that predicts the next two words of a sentence. It has been trained on the classic text *'Alice's Adventures in Wonderland'* to understand language patterns and word sequences.

## 🚀 Features
- **Natural Language Processing (NLP)**: Uses Tokenization and Padding to process raw text.
- **LSTM Neural Network**: A Long Short-Term Memory (LSTM) architecture to capture long-term dependencies in text.
- **Word Embedding**: Converts word indices into dense vectors of fixed size for better context understanding.
- **Model Persistence**: Saves the trained model as an `.h5` file for fast future predictions without retraining.



## 🛠️ Tech Stack
- **Language**: Python 3.x
- **Libraries**: TensorFlow, Keras, NumPy, OS
- **Dataset**: Project Gutenberg's Alice's Adventures in Wonderland

## 📂 Project Structure
- `word_predictor.py`: The main Python script containing the model architecture, training, and prediction logic.
- `Alice.txt`: The source text dataset used for training.
- `word_predictor_model.h5`: The pre-trained model weights.
- `README.md`: Project documentation.

## ⚙️ How to Run
1. **Clone the repository**:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/LSTM-Next-Word-Predictor.git](https://github.com/YOUR_USERNAME/LSTM-Next-Word-Predictor.git)
   cd LSTM-Next-Word-Predictor
