# This script performs linear regression based on tensorflow methods.

# TensorFlow outputs a bunch of warnings on my Linux WSL2 environment so I disable 
# debugging logs by first importing os and changing the message logger. This may 
# not be necessary depending on your operating system and installation.

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import all required modules:

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import tensorflow.compat.v1 as tf
import numpy as np
import random

def gen_lin_data(n=100, noise_mag=0.05, startx=0, endx=1, starty=0, endy=1):
	"""
	Takes in user-defined paramaters and returns noisy data for regression analysis.

		Parameters:
			n (int): Number of datapoints.
			noise_mag (float): Magnitude of noise.
			startx (float): Starting point for x.
			endx (float): Ending point for x.
			starty (float): Starting point for y.
			endy (float): Ending point for y.

		Returns: 
			xdata (numpy n x 1 float array): noisy x data.
			ydata (numpy n x 1 float array): noisy y data.

	"""

	# Create linear data:

	xdata = np.linspace(startx, endx, n)
	ydata = np.linspace(starty, endy, n)

	# Create and add noise:

	xdata += random.choices(np.linspace(-noise_mag,noise_mag,1000), k = n)
	ydata += random.choices(np.linspace(-noise_mag,noise_mag,1000), k = n)

	return xdata, ydata

def lin_regress(xdata, ydata, c_init = 5.0, m_init = -1.0, n_steps = 100):
	"""
	Takes in linear regression parameters and returns a matrix of predicted values.

		Parameters:
			xdata (numpy n x 1 float array): noisy x data.
			ydata (numpy n x 1 float array): noisy y data.
			c_init (float): initial guess for intercept.
			m_init (float): initial guess for slope.
			n_steps (int): number of time steps.

		Returns: 
			ypred (numpy n x n_steps float array): predicted y-values after each time step.

	"""

	# Define matrix to hold predicted values:

	y_pred = np.zeros([n_steps, len(ydata)])

	# Allow TensorFlow 1 behavior:

	tf.compat.v1.disable_eager_execution()

	# Define placeholders to be fed with xdata and ydata:

	X1 = tf.placeholder("float")
	Y1 = tf.placeholder("float")

	# Define variables to be optimized:

	c_init = tf.Variable(c_init, name = "c_init")
	m_init = tf.Variable(m_init, name = "m_init")

	# Define hypothesis:

	y = tf.add(tf.multiply(m_init, X1), c_init)

	# Define loss function:

	cost_function = tf.reduce_mean(tf.square(Y1 - y))

	# Define numerical optimizer:

	optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(cost_function)

	# Run TensorFlow optimization for n_steps:

	with tf.Session() as session:

		session.run(tf.global_variables_initializer())

		for i in range(n_steps):

			session.run(optimizer, feed_dict={X1: xdata, Y1: ydata})

			y_pred[i, :] = session.run(y, {X1: xdata})

		return y_pred

def animate_data(xdata, ydata, y_pred):
	"""
	Takes in linear regression results and shows optimization process.

		Parameters:
			xdata (numpy n x 1 float array): noisy x data.
			ydata (numpy n x 1 float array): noisy y data.
			ypred (numpy n x n_steps float array): predicted y-values after each time step.

		Returns: 
			None. 

	"""

	# Create figure and define axes limits:

	fig, ax = plt.subplots()
	ax.set_xlim(xdata[0], xdata[-1])
	ax.set_ylim(ydata[0], ydata[-1])

	# Scatter plot of original noisy data:

	plt.scatter(xdata, ydata, color="blue")

	# Define element to be animated:

	line, = ax.plot(xdata, y_pred[0, :], color = "red")

	def update_plot(i):

		# Update both plot x-label and best fit line: 

		label = 'Step no. {0}'.format(i)
		ax.set_xlabel(label)
		line.set_ydata(y_pred[i, :])

		return line, ax

	# Save animation as a GIF and show result:

	anim = FuncAnimation(fig, update_plot, repeat=True, frames=np.arange(1,len(y_pred)), interval=50)

	anim.save("regression.gif", dpi=300, writer=PillowWriter(fps=25))

	plt.show()

if __name__ == "__main__":

	xdata, ydata = gen_lin_data(endx=2)

	y_pred = lin_regress(xdata, ydata)

	animate_data(xdata, ydata, y_pred)
