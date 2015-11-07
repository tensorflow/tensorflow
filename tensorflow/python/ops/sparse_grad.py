"""Gradients for operators defined in sparse_ops.py."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import sparse_ops


ops.NoGradient("SparseToDense")


ops.NoGradient("SparseConcat")


ops.NoGradient("SparseReorder")
