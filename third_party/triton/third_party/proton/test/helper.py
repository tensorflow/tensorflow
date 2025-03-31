import triton.profiler as proton

import torch
import sys

from helper_kernels import custom_add


def main():
    a = torch.zeros(1, device="cuda")
    with proton.scope("test"):
        custom_add[(1, )](a)


def test_main():
    main()


if __name__ == "__main__":
    if sys.argv[1] == "test":
        main()
