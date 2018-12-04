# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Elementwise operations for RaggedTensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_shape
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export
from tensorflow.python.util import tf_inspect

# Information about an argument to an operation: The name of the argument, its
# position in the argument list, and a boolean flag indicating whether it
# expects a list of tensors.
_ArgInfo = collections.namedtuple('ArgInfo', ['name', 'position', 'is_list'])


def make_elementwise_op(op, *elementwise_args):
  """Returns a ragged-tensor version of the elementwise operation `op`.

  The returned operation will:

  1. Broadcast the elementwise arguments to have a compatible shape.
     An exception is raised if the tensors not broadcast-compatible.
  2. Call `op`, substituting the dense values of the broadcasted tensor for
     each elementwise argument.
  3. Return a potentially ragged tensor constructed from the output of `op`
     and the broadcasted tensors' nested row splits.

  For example, you can construct a ragged-tensor version of the standard
  operation `tf.add` by calling `make_elementwise_op(tf.add, 'x', 'y')`.

  Args:
    op: The operation to wrap.
    *elementwise_args: The names of arguments to `op` that are treated as
      elementwise.  Arguments that take a list of tensors should have their
      names wrapped in square brackets (e.g. "[inputs]").

  Raises:
    ValueError: If any name specified in `elementwise_args` is not the name
      of an argument to `op`.
  """
  elementwise_arg_infos = _get_arg_infos(op, elementwise_args)

  def ragged_op(*args, **kwargs):
    """Ragged version of `op`."""
    args = list(args)

    # Collect all of the elementwise arguments, and put them in a single
    # dict whose values are the (potentially ragged) tensors that need to
    # be broadcast to a common shape.  The keys of this dict are tuples
    # (argkey, index), where argkey is an int for poitional args or a string
    # for keyword args; and index is None for non-list args and the index of the
    # tensor for list args.
    elementwise_args = {}
    for (name, position, is_list) in elementwise_arg_infos.values():
      if position < len(args):
        if is_list:
          args[position] = list(args[position])
          for (index, arg) in enumerate(args[position]):
            elementwise_args[position, index] = arg
        else:
          elementwise_args[position, None] = args[position]
      elif name in kwargs:
        if is_list:
          kwargs[name] = list(kwargs[name])
          for (i, arg) in enumerate(kwargs[name]):
            elementwise_args[name, i] = arg
        else:
          elementwise_args[name, None] = kwargs[name]

    with ops.name_scope(None, op.__name__, elementwise_args.values()):
      # Convert all inputs to tensors or ragged tensors.
      for ((key, index), tensor) in elementwise_args.items():
        argname = elementwise_arg_infos[key].name
        converted = ragged_factory_ops.convert_to_tensor_or_ragged_tensor(
            tensor, name=argname)
        elementwise_args[key, index] = converted

      # Broadcast tensors to have compatible shapes.
      broadcast_args, result_splits, broadcast_check_ops = \
          _broadcast_elementwise_args(elementwise_args)

      # Replace tensor arguments with their dense values.
      for ((key, index), tensor) in broadcast_args.items():
        if ragged_tensor.is_ragged(tensor):
          if isinstance(key, int) and index is None:
            args[key] = tensor.inner_values
          elif isinstance(key, int) and index is not None:
            args[key][index] = tensor.inner_values
          elif isinstance(key, str) and index is None:
            kwargs[key] = tensor.inner_values
          else:
            assert isinstance(key, str) and index is not None
            kwargs[key][index] = tensor.inner_values

      # Call the elementwise op on the broadcasted dense values.
      with ops.control_dependencies(broadcast_check_ops):
        result_values = op(*args, **kwargs)

      # Restore any ragged dimensions that we stripped off, and return the
      # result.
      return ragged_factory_ops.from_nested_row_splits(result_values,
                                                       result_splits)

  # Construct the docstring.
  op_name = tf_export.get_canonical_name_for_symbol(op)
  assert op_name is not None, op
  argnames = ', '.join('`%s`' % s.strip('[]') for s in elementwise_args)
  docstring = _ELEMENTWISE_DOCSTRING % dict(op_name=op_name, argnames=argnames)

  # Update name, docstring, signature, etc., for the wrapper, and return it.
  return tf_decorator.make_decorator(op, ragged_op, decorator_doc=docstring)


_ELEMENTWISE_DOCSTRING = """\
Ragged version of the elementwise operation `tf.%(op_name)s`.

  The following elementwise arguments may be ragged or dense:
  %(argnames)s.
  These arguments will be broadcast to a compatible shape if necessary.
  """


def _get_arg_infos(func, elementwise_args):
  """Returns `_ArgInfo`s for each `func` arg specified by `elementwise_args`.

  Args:
    func: The function whose arguments should be described.
    elementwise_args: The names of the arguments to get info for.

  Returns:
    A dictionary that maps both names and positions of arguments to
    `_ArgInfo` tuples.
  """
  arg_infos = {}

  # Inspect the func's argspec to find the position of each arg.
  arg_spec = tf_inspect.getargspec(func)
  for argname in elementwise_args:
    assert isinstance(argname, str)
    is_list = argname.startswith('[') and argname.endswith(']')
    if is_list:
      argname = argname[1:-1]
    assert argname in arg_spec.args, (func, argname, arg_spec.args)
    arg_info = _ArgInfo(argname, arg_spec.args.index(argname), is_list)
    arg_infos[arg_info.name] = arg_info
    arg_infos[arg_info.position] = arg_info
  return arg_infos


def _broadcast_elementwise_args(elementwise_args):
  """Broadcasts the values of `elementwise_args` to have compatible shapes.

  Args:
    elementwise_args: A dictionary whose keys are potentially ragged tensors.

  Returns:
    A tuple `(broadcast_args, broadcast_splits, checks)` where:

    * `broadcast_args` is a dictionary with the same keys as
      `elementwise_args`, mapping to broadcasted tensors.
    * `broadcast_splits` is the broadcasted nested row splits.
    * `checks` is a possibly empty tuple of assertion operations that should
      be added as control dependencies.

  Raises:
    ValueError: If broadcasting fails.
  """
  # No elementwise arguments were used: nothing to do!
  if not elementwise_args:
    return elementwise_args, (), ()

  # A single elementwise argument was used: no broadcasting necessary.
  if len(elementwise_args) == 1:
    arg = list(elementwise_args.values())[0]
    if ragged_tensor.is_ragged(arg):
      return elementwise_args, arg.nested_row_splits, ()
    else:
      return elementwise_args, (), ()

  # Multiple elementwise arguments.
  else:
    is_ragged = [ragged_tensor.is_ragged(t) for t in elementwise_args.values()]
    if not any(is_ragged):
      return elementwise_args, (), ()

    # If we have a single ragged tensor plus a set of scalars, then we can
    # rely on the underlying elementwise op to do broadcasting.
    if (sum(is_ragged) == 1 and
        all((ragged_tensor.is_ragged(t) or t.shape.ndims == 0)
            for t in elementwise_args.values())):
      nested_splits_lists = [
          t.nested_row_splits
          for t in elementwise_args.values()
          if ragged_tensor.is_ragged(t)][0]
      return elementwise_args, nested_splits_lists, ()

    else:
      # Get the shapes of all the elementwise arguments.
      shapes = [ragged_tensor_shape.RaggedTensorDynamicShape.from_tensor(t)
                for t in elementwise_args.values()]

      # Broadcast the shapes to all have the same rank (the max rank).
      ranks = [t.shape.ndims for t in elementwise_args.values()]
      if any(rank is None for rank in ranks):
        raise ValueError('Unable to broadcast: unknown rank')
      broadcast_rank = max(ranks)
      shapes = [shape.broadcast_to_rank(broadcast_rank) for shape in shapes]

      # For each dimension, broadcast the shapes to be compatible.
      for axis in range(broadcast_rank):
        # For each i, broadcast shape[i+1] to be compatible with shape[i]; and
        # then finally broadcast shape[0] to be compatible with shape[-1].
        for i in range(len(shapes)):
          j = (i + 1) % len(shapes)
          dim_size = shapes[i].dimension_size(axis)
          shapes[j] = shapes[j].broadcast_dimension(axis, dim_size)
      broadcast_shape = shapes[0]

      # Broadcast every elementwise arg to the shape that we calculated.
      elementwise_args = dict([
          (key, ragged_tensor_shape.broadcast_to(t, broadcast_shape, False))
          for (key, t) in elementwise_args.items()])
      nested_splits_lists = list(elementwise_args.values())[0].nested_row_splits
      return elementwise_args, nested_splits_lists, ()


# A list of symbols that should be exported in the "ragged" package.
_symbols_to_export = []


def _add_elementwise_ops_to_this_module(specs, verbose=False):
  """Adds ragged versions of the given ops to this module.

  Args:
    specs: A list of tuples containing the arguments for `make_elementwise_op`.
    verbose: If true, then display each op that gets added.
  """
  for spec in specs:
    original_op = spec[0]
    ragged_op = make_elementwise_op(*spec)
    canonical_name = tf_export.get_canonical_name_for_symbol(original_op)
    if '.' not in canonical_name:
      op_name = canonical_name
    else:
      op_name = original_op.__name__

    # Temporary hack (will be removed once dispatch is added for RaggedTensors):
    if op_name == 'neg': op_name = 'negative'

    if verbose:
      print('Adding ragged_elementwise_op: tf.ragged.%s (based on tf.%s)' %
            (op_name, canonical_name))
    globals()[op_name] = ragged_op
    _symbols_to_export.append(op_name)


# A list of tuples containing arguments for `make_elementwise_op`, for each
# elementwise operation that should have a ragged version built.  Each tuple
# contains a standard `Tensor` operation, and the names of any arguments
# that are processed in elementwise fashion.
_TF_ELEMENTWISE_OPS = [
    # Unary math operations.
    (clip_ops.clip_by_value, 't'),
    (math_ops.abs, 'x'),
    (math_ops.acos, 'x'),
    (math_ops.acosh, 'x'),
    (math_ops.angle, 'input'),
    (math_ops.asin, 'x'),
    (math_ops.asinh, 'x'),
    (math_ops.atan, 'x'),
    (math_ops.atanh, 'x'),
    (math_ops.cast, 'x'),
    (math_ops.ceil, 'x'),
    (math_ops.conj, 'x'),
    (math_ops.cos, 'x'),
    (math_ops.cosh, 'x'),
    (math_ops.digamma, 'x'),
    (math_ops.erf, 'x'),
    (math_ops.erfc, 'x'),
    (math_ops.exp, 'x'),
    (math_ops.expm1, 'x'),
    (math_ops.floor, 'x'),
    (math_ops.imag, 'input'),
    (math_ops.is_finite, 'x'),
    (math_ops.is_inf, 'x'),
    (math_ops.is_nan, 'x'),
    (math_ops.lgamma, 'x'),
    (math_ops.log, 'x'),
    (math_ops.log1p, 'x'),
    (math_ops.log_sigmoid, 'x'),
    (math_ops.logical_not, 'x'),
    (math_ops.negative, 'x'),
    (math_ops.real, 'input'),
    (math_ops.reciprocal, 'x'),
    (math_ops.rint, 'x'),
    (math_ops.round, 'x'),
    (math_ops.rsqrt, 'x'),
    (math_ops.saturate_cast, 'value'),
    (math_ops.sign, 'x'),
    (math_ops.sin, 'x'),
    (math_ops.sinh, 'x'),
    (math_ops.sqrt, 'x'),
    (math_ops.square, 'x'),
    (math_ops.tan, 'x'),

    # Binary math operations
    (math_ops.add, 'x', 'y'),
    (math_ops.atan2, 'y', 'x'),
    (math_ops.complex, 'real', 'imag'),
    (math_ops.div, 'x', 'y'),
    (math_ops.div_no_nan, 'x', 'y'),
    (math_ops.divide, 'x', 'y'),
    (math_ops.equal, 'x', 'y'),
    (math_ops.floordiv, 'x', 'y'),
    (math_ops.floormod, 'x', 'y'),
    (math_ops.greater, 'x', 'y'),
    (math_ops.greater_equal, 'x', 'y'),
    (math_ops.less, 'x', 'y'),
    (math_ops.less_equal, 'x', 'y'),
    (math_ops.logical_and, 'x', 'y'),
    (math_ops.logical_or, 'x', 'y'),
    (math_ops.logical_xor, 'x', 'y'),
    (math_ops.maximum, 'x', 'y'),
    (math_ops.minimum, 'x', 'y'),
    (math_ops.multiply, 'x', 'y'),
    (math_ops.not_equal, 'x', 'y'),
    (math_ops.pow, 'x', 'y'),
    (math_ops.realdiv, 'x', 'y'),
    (math_ops.squared_difference, 'x', 'y'),
    (math_ops.subtract, 'x', 'y'),
    (math_ops.truediv, 'x', 'y'),
    (math_ops.truncatediv, 'x', 'y'),
    (math_ops.truncatemod, 'x', 'y'),

    # N-ary math operations
    (math_ops.add_n, '[inputs]'),

    # String operations
    (string_ops.as_string, 'input'),
    (string_ops.decode_base64, 'input'),
    (string_ops.encode_base64, 'input'),
    (string_ops.regex_full_match, 'input'),
    (string_ops.regex_replace, 'input'),
    (string_ops.string_join, '[inputs]'),
    (string_ops.string_strip, 'input'),
    (string_ops.string_to_hash_bucket, 'input'),
    (string_ops.string_to_hash_bucket_fast, 'input'),
    (string_ops.string_to_hash_bucket_strong, 'input'),
    (string_ops.substr, 'input'),
    (string_ops.unicode_script, 'input'),

    # Array ops
    (array_ops.check_numerics, 'tensor'),
    (array_ops.identity, 'input'),
    (array_ops.ones_like, 'tensor'),
    (array_ops.zeros_like, 'tensor'),

    # Parsing ops
    (parsing_ops.decode_compressed, 'bytes'),
    (parsing_ops.string_to_number, 'string_tensor'),
]
_add_elementwise_ops_to_this_module(_TF_ELEMENTWISE_OPS)

