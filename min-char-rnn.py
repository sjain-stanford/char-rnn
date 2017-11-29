# Minimal character level language model using vanilla RNN in python and numpy
# Based on min-char-rnn.py by Andrej Karpathy 
# https://gist.github.com/karpathy/d4dee566867f8291f086

"""
Vanilla RNN equations
Hidden vector: h_t = tanh( Wxh . x_t  +  Whh . h_t-1  +  bh )
Output vector: y_t = ( Why . h_t  +  by )
Probabilities: p_t = exp( y_t )
"""

import numpy as np

# Data IO
file = open("input.txt", "r")
data = file.read()
chars = list(set(data))  # Extract all unique characters that form the vocabulary
data_size = len(data)
vocab_size = len(chars)
print("Input data has %d characters, %d of which are unique." % (data_size, vocab_size))

# Dictionaries with indices assigned to each unique character in vocabulary
char_to_idx = {ch:i for i, ch in enumerate(chars)}
idx_to_char = {i:ch for i, ch in enumerate(chars)}

# Hyperparameters
learning_rate = 1e-1
hidden_size = 100  # Number of hidden layer neurons
seq_length = 25  # Determined by memory required for truncated back prop through seq_length time steps

# RNN Model parameters
Wxh = np.random.randn(vocab_size, hidden_size) * 0.01  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
Why = np.zeros(hidden_size, vocab_size) * 0.01  # hidden to output
bh = np.zeros((hidden_size, 1))  # bias for hidden layer
by = np.zeros((vocab_size, 1))  # bias for output layer

# Forward and Backward Prop
def loss_function(inputs, labels, hprev):
"""
 Params:
 - inputs: list of indices (integers) corresponding to seq_length set of input characters
 - labels: list of indices (integers) corresponding to seq_length set of target characters
 - hprev: initial hidden state vector of shape (hidden_size, 1)

 Returns:
 - loss:
 - dWxh, dWhh, dWhy: gradients wrt weights
 - dbh, dby: gradients wrt biases
 - hnext: last hidden state vector of shape (hidden_size, 1)
"""


# MAIN 
n = 0  # iteration counter
p = 0  # data position counter (increments seq_length characters at a time)

while True:
  # Reset RNN hidden state for first iteration or if fewer than seq_length characters remain in the input data
  if (n == 0) or (p + seq_length + 1 >= data_size):
    hprev = np.zeros((hidden_size, 1))
    p = 0

  inputs = [char_to_idx[ch] for ch in data[p:p+seq_length]]
  labels = [char_to_idx[ch] for ch in data[p+1:p+seq_length+1]]

  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = loss_function(inputs, labels, hprev)
  
