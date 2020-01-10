"""Logging Operations."""

from tensorflow.python.framework import ops
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import gen_logging_ops
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_logging_ops import *
# pylint: enable=wildcard-import


# Assert and Print are special symbols in python, so we must
# use an upper-case version of them.
def Assert(condition, data, summarize=None, name=None):
  """Asserts that the given condition is true.

  If `condition` evaluates to false, print the list of tensors in `data`.
  `summarize` determines how many entries of the tensors to print.

  Args:
    condition: The condition to evaluate.
    data: The tensors to print out when condition is false.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).
  """
  return gen_logging_ops._assert(condition, data, summarize, name)


def Print(input_, data, message=None, first_n=None, summarize=None,
          name=None):
  """Prints a list of tensors.

  This is an identity op with the side effect of printing `data` when
  evaluating.

  Args:
    input_: A tensor passed through this op.
    data: A list of tensors to print out when op is evaluated.
    message: A string, prefix of the error message.
    first_n: Only log `first_n` number of times. Negative numbers log always;
             this is the default.
    summarize: Only print this many entries of each tensor.
    name: A name for the operation (optional).

  Returns:
    Same tensor as `input_`.
  """
  return gen_logging_ops._print(input_, data, message, first_n, summarize, name)


@ops.RegisterGradient("Print")
def _PrintGrad(op, *grad):
  return list(grad) + [None] * (len(op.inputs) - 1)


ops.RegisterShape("Assert")(common_shapes.no_outputs)
ops.RegisterShape("Print")(common_shapes.unchanged_shape)
