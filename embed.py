import tensorflow as tf


def lsq_error(B,A):
	''' Least Square Error
	Least-square error between tensors A and B
	'''
	return tf.math.reduce_sum((B - A)**2)


class embedding():
	''' 
	An object containing the latent coordinates, and function with which to reconstruct the network.

	Args:
		N: number of nodes in the network to be embedded
		d: number of latent dimensions 
		call_func: a function that acts on the latent coordinates to reconstruct the network. Must take Z as an argument, and return
					an adjacency matrix.  

	Kwargs:
		loss: the objective function for the embedding. Must take two tensors of the same shape; loss(reconstruction, target)
		lr: the learning rate for the optimizer
		spread: The iterval over which the coordinates are initially uniformly generated. I.e. every coordinate will initially be
				on the interval (0,spread) along each dimension. 
		opt: The optimizer to use. Either adam or SGD.

	Attributes: 
		loss, call, opt
		Z: A tensor of latent coordinates, has shape (N,d)

	'''


	def __init__(self, N, d, call_func, loss=lsq_error, lr=0.1, spread=1.0, opt='adam'):

		self.loss = loss
		self.call = call_func 
		self.Z = tf.Variable(spread*tf.random.uniform([N,d]))
		self.lr = lr

		if opt == 'adam':
			self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=15.0)
		else:
			self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr, clipvalue=15.0)


	def __call__(self):
		# reconstruct the adjacency matrix with call_func
		B = self.call(self.Z)
		return B
		

	def fit(self, A, epochs=150, verbose=True, zero_diag=True):
		'''
		Optimize latent coordinates to minimize reconstruction error according to self.loss. 

		Args: 
			A: The target adjacency matrix
			
		Kwargs:
			epochs: the number of iterations to run for
			verbose: if true, will print out the reconstruction error every 50 iterations
			zero_diag: Ignore diagonal elements in the target adjacency matrix, i.e. don't account for self-links

		'''
		A = A.astype('float32')

		if zero_diag:
			for i in range(epochs):
				with tf.GradientTape() as tape:
					B = self.__call__()
					loss_value = self.loss(tf.linalg.set_diag(B, tf.zeros(B.shape[0])), A)
				 
				grads = tape.gradient(loss_value, [self.Z])
				grads = tf.where(tf.math.is_nan(grads), tf.zeros_like(grads), grads)
				self.optimizer.apply_gradients(zip(grads, [self.Z]))

				if verbose and i % 100 == 0:
					B = self.__call__()
					print(f'iteration {i}: {self.loss(tf.linalg.set_diag(B, tf.zeros(B.shape[0])), A)}')

		else:
			for i in range(epochs):
				with tf.GradientTape() as tape:
					loss_value = self.loss(self.__call__(), A)
				 
				grads = tape.gradient(loss_value, [self.Z])
				grads = tf.where(tf.math.is_nan(grads), tf.zeros_like(grads), grads)
				self.optimizer.apply_gradients(zip(grads, [self.Z]))

				if verbose and i %100 == 0:
				 	print(f'iteration {i}: {self.loss(self.__call__(), A)}')



