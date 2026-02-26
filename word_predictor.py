import os
import numpy as np

# 1. Load the Text File
filename = 'Alice.txt'
if not os.path.exists(filename):
    print(f"Error: {filename} not found!")
else:
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read().lower()

#print(text[:500])  # Print the first 500 characters of the text

#import tensorflow and keras
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 2. Tokenize the Text
tokenizer = Tokenizer()

tokenizer.fit_on_texts([text])
#print("First 50 words in the word index:")
#for i, (word, index) in enumerate(list(tokenizer.word_index.items())[:50]):
#    print(f"Word: '{word}', Index: {index}")

# 3. Create Sequences
input_sequences = []
for sentence in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

#print(input_sequences[:5])  # Print the first 5 input sequences

# 4. Pad Sequences
max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

X = input_sequences[:,:-1]
y = input_sequences[:,-1]

#print(X.shape)
#print(y.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, LSTM, Dense, Input

# 5. Build the Model
model = Sequential()

# Explicitly defining the Input shape makes the summary appear correctly
model.add(Input(shape=(max_sequence_len - 1,))) 

model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100))
model.add(LSTM(150))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Now this will show all your parameters and shapes!
#print(model.summary())

# 6. Train the Model
#model.fit(X, y, epochs=100, verbose=1)

#model.save('word_predictor_model.h5')

# 7. Predict the Next Word
from tensorflow.keras.models import load_model
loaded_model = load_model('word_predictor_model.h5')

t = "author" 
print(f"Starting seed: {t}")

# Use the 'else' block correctly with matching indentation
if not tokenizer.texts_to_sequences([t])[0]:
    print(f"Word '{t}' not found in vocabulary.")
else:
    # Predict next 2 words
    for i in range(2):
        # 1. Tokenize the current string 't'
        token_list = tokenizer.texts_to_sequences([t])[0]
        
        # 2. Pad to match the model's expected input length
        padded_token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        # 3. Get the index of the predicted word
        prediction_index = np.argmax(loaded_model.predict(padded_token_list, verbose=0), axis=-1)[0]

        # 4. Find the word in our dictionary
        predicted_word = ""
        for word, index in tokenizer.word_index.items():
            if index == prediction_index:
                predicted_word = word
                break
        
        # 5. Print and Update 't' for the next iteration
        if predicted_word:
            t += " " + predicted_word  # Add the new word to our sentence
        else:
            break

print(f"Next 2 words after '{t.split()[0]}': {t}")