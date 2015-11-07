"""Connects all float and double tensors to CheckNumericsOp."""

from tensorflow.python.framework import ops
from tensorflow.python.framework import types
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops


def verify_tensor_all_finite(t, msg, name=None):
  """Assert that the tensor does not contain any NaN's or Inf's.

  Args:
    t: Tensor to check.
    msg: Message to log on failure.
    name: A name for this operation (optional).

  Returns:
    Same tensor as `t`.
  """
  with ops.op_scope([t], name, "VerifyFinite") as name:
    t = ops.convert_to_tensor(t, name="t")
    with ops.device(t.device or t.graph.get_default_device()):
      verify_input = array_ops.check_numerics(t, message=msg)
      out = control_flow_ops.with_dependencies([verify_input], t)
  return out


def add_check_numerics_ops():
  """Connect a check_numerics to every floating point tensor.

  `check_numerics` operations themselves are added for each `float` or `double`
  tensor in the graph. For all ops in the graph, the `check_numerics` op for
  all of its (`float` or `double`) inputs is guaranteed to run before the
  `check_numerics` op on any of its outputs.

  Returns:
    A `group` op depending on all `check_numerics` ops added.
  """
  check_op = []
  # This code relies on the ordering of ops in get_operations().
  # The consumer of a tensor always comes before that tensor's producer in
  # this list. This is true because get_operations() returns ops in the order
  # added, and ops can only be added once its inputs are added.
  for op in ops.get_default_graph().get_operations():
    for output in op.outputs:
      if output.dtype in [types.float32, types.float64]:
        message = op.name + ":" + str(output.value_index)
        with ops.control_dependencies(check_op):
          check_op = [array_ops.check_numerics(output, message=message)]
  return control_flow_ops.group(*check_op)
