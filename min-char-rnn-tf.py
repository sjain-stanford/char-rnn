# Minimal character level language model using vanilla RNN in tensorflow
# Based on min-char-rnn.py by Andrej Karpathy and min-char-rnn-tensorflow.py by Vinh Khuc
# https://gist.github.com/karpathy/d4dee566867f8291f086
# https://gist.github.com/vinhkhuc/7ec5bf797308279dc587

"""
Vanilla RNN equations
Hidden vector: h_t = tanh( Wxh . x_t  +  Whh . h_t-1  +  bh )
Output vector: y_t = ( Why . h_t  +  by )
Probabilities: p_t = exp( y_t ) / sum(exp( y_t ))
Loss: l_t = -log(p_yi)
"""

from random import uniform
import numpy as np
import tensorflow as tf

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
Wxh = tf.Variable(tf.random_normal(shape=[hidden_size, vocab_size], mean=0, stddev=0.01, name='Wxh'))  # input to hidden
Whh = tf.Variable(tf.random_normal(shape=[hidden_size, hidden_size], mean=0, stddev=0.01, name='Whh'))  # hidden to hidden
Why = tf.Variable(tf.random_normal(shape=[vocab_size, hidden_size], mean=0, stddev=0.01, name='Why'))  # hidden to output
bh = np.zeros((hidden_size, 1))  # bias for hidden layer
by = np.zeros((vocab_size, 1))  # bias for output layer


def loss_function(inputs, targets, hprev):
  """
  Perform forward and backward pass through one TBTT (truncated backprop through time) of seq_length

  Params:
  - inputs: list of indices (integers) corresponding to seq_length set of input characters
  - targets: list of indices (integers) corresponding to seq_length set of target characters
  - hprev: initial hidden state vector  # (hidden_size, 1)
  
  Returns:
  - loss
  - dWxh, dWhh, dWhy: gradients wrt weights
  - dbh, dby: gradients wrt biases
  - hnext: last hidden state vector  # (hidden_size, 1)
  """

  x, h, y, p = {}, {}, {}, {}
  h[-1] = np.copy(hprev)
  loss = 0
  
  # Forward pass
  for t in range(seq_length):
    x[t] = np.zeros((vocab_size, 1))  # input vector at t  # (vocab_size, 1)
    x[t][inputs[t]] = 1  # one-hot encoding
    h[t] = np.tanh(np.dot(Wxh, x[t]) + np.dot(Whh, h[t-1]) + bh)  # hidden state vector at t  # (hidden_size, 1)
    y[t] = np.dot(Why, h[t]) + by  # output vector at t (unnormalized log probabilities for next char)  # (vocab_size, 1)
    p[t] = np.exp(y[t]) / np.sum(np.exp(y[t]))  # probabilities for next char (softmax)  # (vocab_size, 1)
    loss += -np.log(p[t][targets[t], 0])  # cross-entropy loss (negative log likelihood)
  
  hnext = h[seq_length - 1]

  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hnext)
  
  # Backward pass
  # https://github.com/sjain-stanford/char-rnn/blob/master/docs/VanillaRNN_CompGraph.pdf
  # http://cs231n.github.io/neural-networks-case-study/#grad
  for t in reversed(range(seq_length)):
    dy = np.copy(p[t])  # (vocab_size, 1)
    dy[targets[t]] -= 1  # see reference above
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


def sample(h, seed_idx, n):
  """
  Sample a sequence of integers of length n from the model
  
  Params:
  - h: memory / hidden state at some point during training
  - seed_idx: starting index for first character
  - n: length of sequence to sample
  
  Returns:
  - y_idx: list of indices of sampled characters of length n

  """
  x = np.zeros((vocab_size, 1))
  x[seed_idx] = 1
  y_idx = []

  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    idx = np.random.choice(range(vocab_size), p = np.ravel(p))
    y_idx.append(idx)
    x = np.zeros((vocab_size, 1))
    x[idx] = 1

  return y_idx


def grad_check(inputs, targets, hprev):
  """
  Gradient checking
  """

  #global Wxh, Whh, Why, bh, by
  num_checks, delta = 10, 1e-5
  _, dWxh, dWhh, dWhy, dbh, dby, _ = loss_function(inputs, targets, hprev)

  for param,dparam,name in zip([Wxh, Whh, Why, bh, by],
                               [dWxh, dWhh, dWhy, dbh, dby],
			       ['Wxh', 'Whh', 'Why', 'bh', 'by']):
    s0 = dparam.shape
    s1 = param.shape
    assert s0 == s1, 'Error dims dont match: %s and %s.' % (s0, s1)
    print(name)

    for i in range(num_checks):
      ri = int(uniform(0, param.size))
      # evaluate cost at [x + delta] and [x - delta]
      old_val = param.flat[ri]
      param.flat[ri] = old_val + delta
      cg0, _, _, _, _, _, _ = loss_function(inputs, targets, hprev)
      param.flat[ri] = old_val - delta
      cg1, _, _, _, _, _, _ = loss_function(inputs, targets, hprev)
      param.flat[ri] = old_val # reset old value for this parameter
      # fetch both numerical and analytic gradient
      grad_analytic = dparam.flat[ri]
      grad_numerical = (cg0 - cg1) / ( 2 * delta )
      rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
      print('%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
      # rel_error should be on order of 1e-7 or less


# MAIN 
n = 0  # iteration counter
p = 0  # data position counter (increments seq_length characters at a time)

smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # Loss at iteration 0

# # Adam parameters
# beta1 = 0.9
# beta2 = 0.999
# fmWxh, fmWhh, fmWhy, fmbh, fmby = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why), np.zeros_like(bh), np.zeros_like(by)  # First moments (unbiased)
# smWxh, smWhh, smWhy, smbh, smby = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why), np.zeros_like(bh), np.zeros_like(by)  # Second moments (unbiased)

# Adagrad parameters / memory variables
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)

while True:
  # Reset RNN hidden state for first iteration or if fewer than seq_length characters remain in the input data
  if (n == 0) or (p + seq_length + 1 >= data_size):
    hprev = np.zeros((hidden_size, 1))
    p = 0

  inputs = [char_to_idx[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_idx[ch] for ch in data[p+1:p+seq_length+1]]

  # grad_check(inputs, targets, hprev) 

  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = loss_function(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if (n % 100 == 0): print("iter %d, loss: %f" % (n, smooth_loss))

  # # Parameter upate with Adam
  # for param, dparam, first_moment, second_moment in zip([Wxh, Whh, Why, bh, by],
  #                                                       [dWxh, dWhh, dWhy, dbh, dby],
  #       						[fmWxh, fmWhh, fmWhy, fmbh, fmby],
  #       						[smWxh, smWhh, smWhy, smbh, smby]):

  #   first_moment = (beta1 * first_moment + (1 - beta1) * dparam) / (1 - beta1 ** (n+1))
  #   second_moment = (beta2 * second_moment + (1 - beta2) * dparam * dparam) / (1 - beta2 ** (n+1))
  #   param -= learning_rate * first_moment / (np.sqrt(second_moment) + 1e-7)

  # Parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  # Sample ~200 characters from the model occasionally
  if (n % 100 == 0):
    sample_idx = sample(hprev, inputs[0], 200)
    text = "".join(idx_to_char[i] for i in sample_idx)
    print("-----\n %s \n -----" % (text))

  n += 1
  p += seq_length

