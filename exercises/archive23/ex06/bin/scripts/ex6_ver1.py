# Based on https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
import math

import torch

if __name__ == "__main__":

    # Create random input and output data
    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    # Randomly initialize weights
    a = torch.randn((), )
    b = torch.randn((), )
    c = torch.randn((), )
    d = torch.randn((), )

    learning_rate = 1e-6
    for t in range(2000):
        # Forward pass: compute predicted y
        y_prediction = a + b * x + c * x ** 2 + d * x ** 3

        # Compute and print loss
        loss = (y_prediction - y).pow(2).sum().item()
        if t % 100 == 99:
            print(t, loss)

        # Backprop to compute gradients of loss with respect to a, b, c, d
        grad_y_prediction = 2.0 * (y_prediction - y)
        grad_a = grad_y_prediction.sum()
        grad_b = (grad_y_prediction * x).sum()
        grad_c = (grad_y_prediction * x ** 2).sum()
        grad_d = (grad_y_prediction * x ** 3).sum()

        # Update weights using gradient descent
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d

    print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
