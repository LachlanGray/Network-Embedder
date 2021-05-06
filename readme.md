

## What is Network Embedding?

The aim of network embedding is to find a low-dimensional latent representation for each node that represents its structural or functional role in the network. The intention is that nodes that are somehow similar to one another should have similar latent dimensions. 

## What is this for?

Often a network embedding is realized as a reconstruction function that acts on two latent coordinates, and returns the expected value of a link between nodes with those coordinates, which is the focus of everything here. To find an embedding, you choose the reconstruction function, and then you reconstruct the links (and lack thereof) between all pairs of nodes from the latent coordinates. Then, you optimize all of the latent coordinates to minimize an error between the original network and the reconstructed network. 

Sometimes it's not obvious what an embedding is doing. It can be that an embedding is super effective at capturing the excact links in a network but doesn't say anything about general structural features. On the other hand, some embeddings might not allow an accurate reconstruction, but reveal meaningful features of the network. 

The little tools here are to help explore different embeddings really quickly and get an intuition for what they are good at. 


## What can this do?

The embedding object in embed.py stores latent coordinates for a set of nodes, a function for the network reconstruction, and a fit method that optimizes the latent coordinates for the reconstruction function. 

There are also:
- visualizations that plot the nodes in 2 and 3 dimensions and the links between them
- a visualization that compares the reconstructed network side-by-side with the original network
- a tool to save an mp4 file of the optimization process in two dimensions

There are examples of how to use everything in example.ipynb :)
