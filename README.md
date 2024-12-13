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

Generate Data (Substitution Cipher)

Generate a random substitution cipher map.
Create random plaintext messages.
Encrypt plaintext to generate ciphertext using the cipher map.
Tokenization and Padding

Tokenize both plaintext and ciphertext into integer sequences.
Pad the sequences to a fixed length to ensure uniformity in input and output.
Model Building

Create a sequence-to-sequence model:
Embedding layer: Converts integer sequences to dense vectors.
LSTM layer: Learns to map ciphertext to plaintext.
Dense layer: Outputs the predicted plaintext character for each input ciphertext character.
Train Model

Use the ciphertext as input and plaintext as the target output for training.
Train the model using sparse categorical cross-entropy loss and the Adam optimizer.
Test the Model

Evaluate the model on unseen test data (ciphertext).
The model predicts the corresponding plaintext for the ciphertext.
Compare Results

Print the original ciphertext, the model’s predicted plaintext, and the actual plaintext.
Evaluate model accuracy by comparing predicted plaintext to original plaintext.
+-------------------+
| Generate Data     |
| (Cipher Map,      |
| Plaintext,        |
| Ciphertext)       |
+-------------------+
        |
        v
+-------------------+
| Tokenize and Pad  |
| Text Sequences    |
+-------------------+
        |
        v
+-------------------+
| Build Model       |
| (Embedding, LSTM, |
| Dense layers)     |
+-------------------+
        |
        v
+-------------------+
| Train Model       |
| (Ciphertext ->    |
| Plaintext)        |
+-------------------+
        |
        v
+-------------------+
| Test Model        |
| (Ciphertext ->    |
| Predicted Plaintext)|
+-------------------+
        |
        v
+-------------------+
| Compare Results   |
| (Encrypted,       |
| Predicted,        |
| Original)         |
+-------------------+

