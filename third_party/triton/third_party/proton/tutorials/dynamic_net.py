import random
import torch
import math

import triton.profiler as proton
import argparse

mode = "torch"


class DynamicNet(torch.nn.Module):
    # https://pytorch.org/tutorials/beginner/examples_nn/dynamic_net.html
    def __init__(self):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 4, 5
        and reuse the e parameter to compute the contribution of these orders.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same parameter many
        times when defining a computational graph.
        """
        y = self.a + self.b * x + self.c * x**2 + self.d * x**3
        for exp in range(4, random.randint(4, 6)):
            y = y + self.e * x**exp
        return y

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f"y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3 + {self.e.item()} x^4 ? + {self.e.item()} x^5 ?"


def run():
    # Create Tensors to hold input and outputs.
    with proton.scope("init"):
        x = torch.linspace(-math.pi, math.pi, 2000, device="cuda")
        y = torch.sin(x)

        # Construct our model by instantiating the class defined above
        model = DynamicNet().to("cuda")
        if mode == "torchinductor":
            model = torch.compile(model)

        # Construct our loss function and an Optimizer. Training this strange model with
        # vanilla stochastic gradient descent is tough, so we use momentum
        criterion = torch.nn.MSELoss(reduction="sum")
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)
    for t in range(1000):
        # Forward pass: Compute predicted y by passing x to the model
        with proton.scope("forward"):
            y_pred = model(x)

        # Compute and print loss
        with proton.scope("loss"):
            loss = criterion(y_pred, y)
            if t % 200 == 199:
                print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        with proton.scope("backward"):
            optimizer.zero_grad()
            loss.backward()
        with proton.scope("optimizer"):
            optimizer.step()

    print(f"Result: {model.string()}")


argparser = argparse.ArgumentParser()
argparser.add_argument("--profile", action="store_true")
argparser.add_argument("--mode", default="torch", choices=["torch", "torchinductor"])
argparser.add_argument("--context", default="shadow", choices=["shadow", "python"])
argparser.add_argument("--backend", default=None, choices=["cupti", "roctracer", "cupti_pcsampling"])

args = argparser.parse_args()

mode = args.mode

if args.profile:
    func = proton.profile(run, name="dynamic_net", context=args.context, backend=args.backend)
else:
    func = run

func()
# Write out the profile
# Visualize using `proton-viewer -m time/s ./dynamic_net.hatchet`
proton.finalize()
