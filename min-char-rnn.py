# Minimal character level language model using vanilla RNN in python and numpy
# Based on min-char-rnn.py by Andrej Karpathy 
# https://gist.github.com/karpathy/d4dee566867f8291f086

"""
Vanilla RNN equations
Hidden vector: h_t = tanh( Wxh . x_t  +  Whh . h_t-1  +  bh )
Output vector: y_t = ( Why . h_t  +  by )
Probabilities: p_t = exp( y_t ) / sum(exp( y_t ))
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
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
bh = np.zeros((hidden_size, 1))  # bias for hidden layer
by = np.zeros((vocab_size, 1))  # bias for output layer


def loss_function(inputs, labels, hprev):
"""
 Params:
 - inputs: list of indices (integers) corresponding to seq_length set of input characters
 - labels: list of indices (integers) corresponding to seq_length set of target characters
 - hprev: initial hidden state vector  # (hidden_size, 1)

 Returns:
 - loss:
 - dWxh, dWhh, dWhy: gradients wrt weights
 - dbh, dby: gradients wrt biases
 - hnext: last hidden state vector  # (hidden_size, 1)
"""
  x, h, y, p = {}, {}, {}, {}
  h[-1] = np.copy(hprev)
  loss = 0
  
  # Forward pass
  for t in range(len(inputs)):
    x[t] = np.zeros((vocab_size, 1))  # input vector at t  # (vocab_size, 1)
    x[t][inputs[t]] = 1  # one-hot encoding
    h[t] = np.tanh(np.dot(Wxh, x[t]) + np.dot(Whh, h[t-1]) + bh)  # hidden state vector at t  # (hidden_size, 1)
    y[t] = np.dot(Why, h[t]) + by  # output vector at t (unnormalized log probabilities for next char)  # (vocab_size, 1)
    p[t] = np.exp(y[t]) / np.sum(np.exp(y[t]))  # probabilities for next char (softmax)  # (vocab_size, 1)
    loss += -np.log(p[t][labels[t], 0])  # cross-entropy loss (negative log likelihood)
  
  hnext = h[len(inputs) - 1]

  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hnext)
  
  # Backward pass
  # https://github.com/sjain-stanford/char-rnn/blob/master/docs/VanillaRNN_CompGraph.pdf
  # http://cs231n.github.io/neural-networks-case-study/#grad
  for t in reversed(range(len(inputs))):
    dy = np.copy(p[t])  # (vocab_size, 1)
    dy[labels[t]] -= 1  # see reference above
    dby += dy  # (vocab_size, 1)
    dWhy += np.dot(dy, h[t].T)  # (vocab_size, hidden_size)
    dh = dhnext + np.dot(Why.T, dy)  # (hidden_size, 1)
    dhraw = (1 - h[t] ** 2) * dh  # dtanh(x) = 1 - tanh(x) ** 2   # (hidden_size, 1)
    dbh += dhraw  # (hidden_size, 1)
    dWxh += np.dot(dhraw, x[t].T)  # (hidden_size, vocab_size)
    dWhh += np.dot(dhraw, h[t-1].T)  # (hidden_size, hidden_size)
    dhnext = np.dot(Whh.T, dhraw)  # (hidden_size, 1)
  
  # Gradient clipping (to avoid exploding gradients)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out = dparam)
     
  return loss, dWxh, dWhh, dWhy, dbh, dby, hnext


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

  smooth_loss
  
