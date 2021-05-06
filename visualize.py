import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import amax

def plot_layout(A,Z,title=''):
	'''
	Plots the latent coordinates as a network layout. Every latent coordinate is plotted, and a link
	is drawn between coordinates that are connected in the network. Only works in 2 dimensions. 

	Args:
		A: Adjacency matrix, an (N,N) numpy array. 
		Z: The latent coordinates from the embedding, a shape (N,2) tensor. 

	Kwargs:
		title: the title to draw on the plot

	'''

	plt.figure(figsize=(5,5))
	plt.scatter(*Z.numpy().T, alpha=0.5)

	N = A.shape[0]
	for i in range(N):
		for j in range(i):

			if A[i,j] == 1:
				plt.plot([Z[i,0],Z[j,0]],[Z[i,1],Z[j,1]], c='grey',alpha=0.3)

	plt.title(title)
	plt.show()


def plot_layout3d(A,Z,title=''):
	'''
	Does the same thing as plot_layout, but in 3 dimensions. 

	If running in a jupyter notebook, it is nice to run %matplotlib qt in a cell first, then the 3d plot is interactive. 

	Args:
		A: Adjacency matrix, an (N,N) numpy array. 
		Z: The latent coordinates from the embedding, a shape (N,3) tensor. 

	Kwargs:
		title: the title to draw on the plot
	'''

	fig = plt.figure(figsize=(7,7))
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(*Z.numpy().T, s=10)

	for i in range(A.shape[0]):
		for j in range(i):
			if A[i,j] == 1:
				plt.plot([Z[i,0],Z[j,0]],[Z[i,1],Z[j,1]],[Z[i,2],Z[j,2]], c='grey',alpha=0.3)

	plt.title(title)


def plot_comparison(A,B,Z):
	'''
	Plots the 2D layout twice; once drawing links from the reconstruction, and once drawing links from the original network.
	Helpful for getting intuitions about what an embedding is doing. 

	Args:
		A: adjacency matrix
		B: reconstructed adjacency matrix
		Z: latent coordinates

	'''
	B_max = amax(B)

	plt.subplots(figsize=(10,5))

	plt.subplot(121)
	plt.title('Reconstruction')
	plt.scatter(*Z.numpy().T, alpha=0.5)
  
	N = A.shape[0]
	for i in range(N):
		for j in range(i):
			plt.plot([Z[i,0],Z[j,0]],[Z[i,1],Z[j,1]], c='black',alpha=B[i,j]/B_max)

	plt.subplot(122)
	plt.title('Original')
	plt.scatter(*Z.numpy().T, alpha=0.5)

	N = A.shape[0]
	for i in range(N):
		for j in range(i):
			if A[i,j] == 1:
				plt.plot([Z[i,0],Z[j,0]],[Z[i,1],Z[j,1]], c='black',alpha=1)

