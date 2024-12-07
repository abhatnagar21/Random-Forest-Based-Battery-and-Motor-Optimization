import numpy as np
import random
import string
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# 1. Generate Random Substitution Cipher
def generate_cipher_map():
    """Generates a random substitution cipher map."""
    letters = list(string.ascii_lowercase)  # English alphabet a-z
    shuffled = letters.copy()
    random.shuffle(shuffled)
    cipher_map = {plain: cipher for plain, cipher in zip(letters, shuffled)}
    reverse_map = {cipher: plain for plain, cipher in cipher_map.items()}
    return cipher_map, reverse_map

# 2. Encrypt a Text using the Cipher Map
def encrypt_text(text, cipher_map):
    """Encrypts the input text using a substitution cipher."""
    return ''.join(cipher_map.get(char, char) for char in text)

# 3. Prepare Dataset
def generate_dataset(num_samples=10000, max_len=20):
    """
    Generates a dataset of plaintext-ciphertext pairs using a random substitution cipher.
    """
    cipher_map, reverse_map = generate_cipher_map()
    plaintexts = []
    ciphertexts = []

    for _ in range(num_samples):
        plain_text = ''.join(random.choices(string.ascii_lowercase + ' ', k=random.randint(5, max_len)))
        cipher_text = encrypt_text(plain_text, cipher_map)
        plaintexts.append(plain_text)
        ciphertexts.append(cipher_text)

    return plaintexts, ciphertexts, reverse_map

# 4. Tokenization for Neural Network
def tokenize_texts(texts, vocab):
    """
    Converts texts to sequences of integers based on a vocabulary.
    """
    tokenizer = {char: idx+1 for idx, char in enumerate(vocab)}  # Reserve 0 for padding
    sequences = [[tokenizer[char] for char in text] for text in texts]
    return sequences, tokenizer

# 5. Padding Sequences
def pad_sequences(sequences, max_len):
    """
    Pads sequences to the same length.
    """
    return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, padding='post')

# 6. Build LSTM Sequence-to-Sequence Model
def build_model(input_dim, output_dim, max_len):
    """
    Builds a sequence-to-sequence model using LSTM layers.
    """
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=64, input_length=max_len),
        LSTM(128, return_sequences=True),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 7. Main Function
def main():
    # Generate dataset
    print("Generating dataset...")
    vocab = list(string.ascii_lowercase + ' ')  # a-z and space
    num_classes = len(vocab) + 1  # Include 0 for padding
    num_samples = 10000
    max_len = 20

    plaintexts, ciphertexts, reverse_map = generate_dataset(num_samples, max_len)
    print("Sample Plaintext:", plaintexts[0])
    print("Sample Ciphertext:", ciphertexts[0])

    # Tokenize and pad texts
    plaintext_seq, plain_tokenizer = tokenize_texts(plaintexts, vocab)
    ciphertext_seq, cipher_tokenizer = tokenize_texts(ciphertexts, vocab)

    X = pad_sequences(ciphertext_seq, max_len)
    y = pad_sequences(plaintext_seq, max_len)
    y = np.expand_dims(y, -1)  # Add an extra dimension for sparse categorical loss

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    print("Building model...")
    model = build_model(input_dim=num_classes, output_dim=num_classes, max_len=max_len)
    model.summary()

    print("Training model...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

    # Testing the model
    idx_to_char = {idx: char for char, idx in plain_tokenizer.items()}
    print("\nTesting model on encrypted text:")
    test_sample = X_test[0]
    predicted = model.predict(np.array([test_sample]))
    predicted_text = ''.join([idx_to_char.get(np.argmax(p), '') for p in predicted[0]])

    original_text = ''.join([idx_to_char.get(idx, '') for idx in y_test[0].flatten()])
    encrypted_text = ''.join([idx_to_char.get(idx, '') for idx in test_sample])

    print(f"Encrypted: {encrypted_text}")
    print(f"Predicted Plaintext: {predicted_text}")
    print(f"Original Plaintext: {original_text}")

if __name__ == "__main__":
    main()
