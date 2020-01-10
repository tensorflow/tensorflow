"""Gradients for operators defined in sparse_ops.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import sparse_ops


ops.NoGradient("SparseToDense")


ops.NoGradient("SparseConcat")


ops.NoGradient("SparseReorder")
