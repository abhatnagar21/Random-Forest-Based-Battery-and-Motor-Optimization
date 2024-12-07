# Breaking-Substitution-Ciphers-Using-LSTM-Neural-Networks
Project Description:
This project involves building a sequence-to-sequence neural network using LSTM layers to decrypt text encrypted with a random substitution cipher. By treating the cipher as a translation problem, the model learns to map ciphertext sequences back to their original plaintext sequences.
Key Features:
Random Substitution Cipher Generation:

A cipher map randomly substitutes each letter in the English alphabet (a-z) and space with another character.
Text is encrypted using this map, making it a classic cryptography problem.
Dataset Generation:

A synthetic dataset of 10,000 plaintext-ciphertext pairs is generated.
Plaintexts are random sequences of lowercase letters and spaces, with varying lengths (5–20 characters).
Tokenization and Preprocessing:

Text sequences (both plaintext and ciphertext) are tokenized into integers.
Sequences are padded to ensure uniform length for LSTM input.
Neural Network Architecture:

LSTM-based Sequence-to-Sequence Model:
Input Layer: Embedding layer for tokenized ciphertext.
Hidden Layer: LSTM layer processes sequential data.
Output Layer: Dense layer with softmax activation to predict the plaintext character probabilities.
Loss Function: Sparse categorical cross-entropy.
Optimizer: Adam.
Training and Testing:

The model is trained to decode ciphertext back to its original plaintext.
Predictions are evaluated on unseen encrypted text.
Results and Insights:

The model successfully predicts plaintext sequences from encrypted text.
Key performance metrics include accuracy on test data.
