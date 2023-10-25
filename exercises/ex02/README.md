# NLPwDL 2023/24 Exercise 02

Computational graph and backpropagation from scratch

## Tasks

Task 1: Download empty project, make sure you can run it and run unit tests, fix unit tests

Task 2: Implement a sum function with derivatives

Task 3: Implement a product function (of two or more parameters)

Task 4: Implement the function from lecture 2: (a + b)(b + 1) and compute output for a = 2 and b = 3

Task 5: Add function to compute "global" gradient wrt. every node. Verify with manual computation for a = 2, b = 3.

Task 6: Current implementation is extremely inefficient, takes 5 seconds to compute the gradient. Implement caching to store intermediate results to speed up. Compare the running times.

## Setup

Create a virtual environment

```bash
$ virtualenv venv
$ source venv/bin/activate
```

Install project and its dependencies (there are none so far)

```bash
$ python setup.py install
```

Run unittests from the command line

```bash
$ python -m unittest
```