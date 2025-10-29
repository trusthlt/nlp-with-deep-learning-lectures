# NLPwDL 2025/26 Exercise 03

Computational graph (and backpropagation) from scratch

## Tasks

### Setup

#### Using a command line

Create a virtual environment

```bash
$ virtualenv venv
$ source venv/bin/activate
```

Run unittests from the command line

```bash
$ python -m unittest
```

#### Using PyCharm

Simply open the folder `blank` as new project, PyCharm asks you automatically to create venv. Run unit test as usual in PyCharm.


Task 1: Download this empty Python project, make sure you can run it in an IDE of your choice (e.g., PyCharm) and run unit tests. Fix the first unit test entitled task 1.


### Computational graph

Task 2: Implement a sum function with derivatives

Task 3: Implement a product function (of two or more parameters)

Task 4: Implement the function from lecture 2: (a + b)(b + 1) and compute output for a = 2 and b = 3

Task 5: Implement the Rosenbrock function (lecture 2). To do so, you will extend the framework by adding a SquareNode (which accepts only a single argument, so the length of arguments will be one). Compute all values in a mesh (-1.5:1.5, -1.5:2.0) as on slide 2 and plot the result using the existing code I used for slide 19 ( https://colab.research.google.com/drive/1mlZtxPXuk3mls56CQArmDzjdp5bLbrJC?usp=sharing ).

### Gradients


Task 5: Add function to compute "global" gradient wrt. every node. Verify with manual computation for a = 2, b = 3.

Task 6: Current implementation is extremely inefficient, takes 5 seconds to compute the gradient. Implement caching to store intermediate results to speed up. Compare the running times.

Task 7: Rosebrock function cont. Pick a few points and compute the gradient at the point, consult Slide 25. Make sure your intuition matches the results you see.


