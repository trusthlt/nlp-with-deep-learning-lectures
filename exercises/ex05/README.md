# NLPwDL 2024/25 Exercise 05

We are going to gradually extend our framework.

* `utils.py` contains a method `graph_to_graphviz` which prints the computational graph in Graphviz format, it can be rendered as an image 

## Tasks

### Task 1

Implement a list of `ScalarNodes` that we will call a `LinearLayer` that takes the previous layer's nodes and create a pre-defined number of linear nodes in this layer (so-called "fully connected layer"). The parameters (weights and biases) should all be set to 0.01.

### Task 2

Implement online gradient descent training of a log-linear model. We have synthetic, linearly separable data for a binary classification task (labeled as 0 and 1), see the Jupyter notebook in bin/notebooks/ex5_dataset_visualisations.ipynb

## Tasks 3

Implement a ReLU (rectified linear unit) node

## Task 4

Implement a ReLU layer. Similar to the `LinearLayer` from the previous exercise, it takes a previous layer (which contains `nodes`, a list of `ScalarNode` instances or any of its subclasses) and for each previous layer's node, it should simply create a new ReLU node, such that the dimension of the ReLU layer is the same as its parent layer. Typically, the parent layer will be a LinearLayer, but it's not important for the actual implementation of the ReLU layer.

## Task 5

Implement the softmax layer.

Derive the partial derivatives of softmax.

## Homeworks

You can continue to play around and implement:

- Classification of movie reviews using average bag-of-words features, MLP with one or more hidden layers and ReLU, and (a) sigmoid or (b) categorical cross-entropy (with two classes)


## Setup

Create a virtual environment

```bash
$ virtualenv venv
$ source venv/bin/activate
```

Run unittests from the command line

```bash
$ python -m unittest
```