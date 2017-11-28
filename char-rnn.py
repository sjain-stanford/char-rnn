import numpy as np

# Data IO
file = open("input.txt","r")
data = file.read()
chars = list(set(data))
print("Input data has %d characters, %d of which are unique." % (len(data), len(chars)))

# Dictionaries with index assigned for each unique character in vocabulary
char_to_idx = {ch:i for i, ch in enumerate(chars)}
idx_to_char = {i:ch for i, ch in enumerate(chars)}



