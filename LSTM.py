import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Sample text with 10 sentences
text = [
    "Ecole hexagone is providing NLP techniques within the AI master courses.",
    "The AI master program covers topics like machine learning and deep learning.",
    "Students learn to build models for natural language processing.",
    "The courses are designed to provide hands-on experience with real-world projects.",
    "NLP techniques include tokenization, sentiment analysis, and named entity recognition.",
    "The AI program covers advanced topics like transformers and attention mechanisms.",
    "Students work on projects that involve text classification and machine translation.",
    "The faculty includes experts in AI, machine learning, and NLP.",
    "The program prepares students for careers in AI and data sciences.",
    "Graduates of the program have gone on to work at top tech companies."
]

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)

# Pad sequences to ensure uniform length
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Convert text to numerical labels (dummy labels for demonstration)
labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Binary classification

# Build the LSTM model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_length),
    LSTM(128, return_sequences=False),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=10, batch_size=2)

# Predict on new text
new_text = ["NLP is a key component of the AI master program."]
new_sequence = tokenizer.texts_to_sequences(new_text)
new_padded_sequence = pad_sequences(new_sequence, maxlen=max_length, padding='post')
prediction = model.predict(new_padded_sequence)

print(f"Prediction:{prediction}")