# NLPwDL 2023/24 Exercise 05

## Tasks 1

Implement a ReLU (rectified linear unit) node

## Task 2

Implement a ReLU layer. Similar to the `LinearLayer` from the previous exercise, it takes a previous layer (which contains `nodes`, a list of `ScalarNode` instances or any of its subclasses) and for each previous layer's node, it should simply create a new ReLU node, such that the dimension of the ReLU layer is the same as its parent layer. Typically, the parent layer will be a LinearLayer, but it's not important for the actual implementation of the ReLU layer.

## Task 3

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