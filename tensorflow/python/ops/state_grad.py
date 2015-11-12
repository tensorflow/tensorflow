"""Gradients for operators defined in state_ops.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops

ops.NoGradient("Assign")


ops.NoGradient("AssignAdd")


ops.NoGradient("AssignSub")


ops.NoGradient("ScatterAdd")


ops.NoGradient("ScatterSub")
