# NLPwDL 2024/26 Exercise 04

Gradient of a log-linear function with input (features), binary output value (gold label), and logistic loss

Note: This repository re-uses our code from Exercise 3, such as the ScalarNode and efficient implementation of backpropagation with caching.

We renamed the unit test file `test_nodes.py` from Exercise 2 to `test_tasks_ex2.py`. These unit tests must keep working!

Changes from ex03: I renamed `arguments` in `ScalarNode` to `children`. We will use not only the arguments of a function (e.g., $x$ is an argument of $y = a x + b$) but also the parameters (e.g., $a$ is a parameter in the previous example). Both arguments and parameters are children of a node and thus must smoothly propagate gradients during backpropagation.

## Tasks

### Task 0 (warm-up, do at home before the exercise class)

Implement a new scalar node (named ParameterNode) which is almost similar to a ConstantNode but its value can be changed.

### Task 1

Implement a linear function node (compute the output value)

Recall: Linear function $y = f(x_1, x_2, ..., x_n)$ has $n$ weight parameters $w_1, ... w_n$ and a single bias parameter $b$.

$y = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$

I extended `ScalarNode` and created a `LinearNode`. The arguments `x_1, ... x_n` of `LinearNode` will again be just a list of other nodes. However, we will also pass a list of parameters `w_1, ... w_n` and `b` which should be created using the `ParameterNode`.

### Task 2

Implement the rest of the linear function node, namely the partial derivatives.

### Task 3

Implement a sigmoid node

### Task 4

Implement a per-example binary logistic loss (cross-entropy loss)

### Task 5

Implement updating the parameters by taking the step determined by the gradient

## Solutions

Will be included in the codebase in the next exercise.

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