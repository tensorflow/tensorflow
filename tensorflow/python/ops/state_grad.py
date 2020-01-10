"""Gradients for operators defined in state_ops.py."""

from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops

ops.NoGradient("Assign")


ops.NoGradient("AssignAdd")


ops.NoGradient("AssignSub")


ops.NoGradient("ScatterAdd")


ops.NoGradient("ScatterSub")
