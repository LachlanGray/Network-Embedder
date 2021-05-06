from celluloid import Camera
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf


def lsq_error(B,A):
    return tf.math.reduce_sum((B - A)**2)

def animate(embedding, A, n_iterations, links=True):
    '''
    Returns an animation of the embedding optimization using a least-square error. Mostly for fun. A new frame is drawn
    every 15 iterations.

    Args:
        embedding: an embedding object
        A: adjacency matrix
        n_iterations: the number of iterations to run for

    Kwargs:
        links: boolean. If true, links are drawn in the animation, which makes it significantly slower. 
    '''
    
    fig = plt.figure(figsize=(5,5))
    cam = Camera(fig) 
    plt.scatter(*embedding.Z.numpy().T, c='blue',alpha=0.5)
    
    B = embedding()
    plt.legend([f'error = {round(float(lsq_error(tf.linalg.set_diag(B, tf.zeros(B.shape[0])), A)),1)}'], loc='lower right')
        
    cam.snap()
    
    for _ in tqdm(range(n_iterations)):  
        embedding.fit(A, 15,verbose=False)   
        plt.scatter(*embedding.Z.numpy().T, c='blue',alpha=0.5)   
        B = embedding()
        plt.legend([f'error = {round(float(lsq_error(tf.linalg.set_diag(B, tf.zeros(B.shape[0])), A)),1)}'], loc='lower right')
        
        if links:
            Z = embedding.Z
            for i in range(A.shape[0]):
                for j in range(i):

                    if A[i,j] == 1:
                        plt.plot([Z[i,0],Z[j,0]],[Z[i,1],Z[j,1]], c='grey',alpha=0.3)
            
        cam.snap()
    animation = cam.animate()
    
    return animation