# This script creates random data by adding noise to a linear relation (so far).

# TensorFlow outputs a bunch of warnings on my Linux WSL2 environment so I disable 
# debugging logs by first importing os and changing the message logger. This may 
# not be necessary depending on your operating system and installation configurations.

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import all required modules:

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random

# Data size:

n = 100

# Create linear relation:

x = np.linspace(0, 10, n)
y = np.linspace(0, 10, n)

# Create noise:

noise_x = random.choices(np.linspace(-1,1,1000), k = n)
noise_y = random.choices(np.linspace(-1,1,1000), k = n)

# Add noise to data:

x += noise_x
y += noise_y

# Plot result:

plt.scatter(x,y)
plt.show()
