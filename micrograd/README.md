# micrograd

Train a neural net following [building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1)

Micrograd is an autograd engine. This implements backpropagation which lets you efficiently evaluate the gradient of some kind of a loss function with respect to the weights of a neural network. We can iteratively tune the weights of the neural network to minimize the loss function and improve the efficiency of the network.

Claim: Micrograd is what you need to train a neural network. Everything else is just efficiency.