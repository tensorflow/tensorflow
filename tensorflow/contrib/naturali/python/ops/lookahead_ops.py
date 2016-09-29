
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.framework import common_shapes
from tensorflow.python.platform import resource_loader
# from tensorflow.contrib.naturali.python.ops import lookahead_grad_ops

_lookahead_ops_so = load_library.load_op_library(
    resource_loader.get_path_to_datafile("_lookahead_ops.so"))
assert _lookahead_ops_so, "Could not load _lookahead_ops.so."

def lookahead(x1, x2):
    return _lookahead_ops_so.lookahead(x1, x2)

def lookaheadgrad(x1, x2, x3):
    return _lookahead_ops_so.lookaheadgrad(x1, x2, x3)

@ops.RegisterShape("Lookahead")
def _lookahead(op):
    inputs_shape = op.inputs[0].get_shape().with_rank(3)
    return [inputs_shape]


@ops.RegisterShape("Lookaheadgrad")
def _lookaheadgrad(op):
    inputs_shape1 = op.inputs[0].get_shape().with_rank(3)
    inputs_shape2 = op.inputs[1].get_shape().with_rank(2)
    return [inputs_shape1, inputs_shape2]


@ops.RegisterGradient("Lookahead")
def _lookahead_grad(op, grad):
    """
    Args:
        op: the lookahead op.
        grad: the output grad
    Returns:
        the input grad and the filter grad
    """
    return lookaheadgrad(op.inputs[0], op.inputs[1], grad)
