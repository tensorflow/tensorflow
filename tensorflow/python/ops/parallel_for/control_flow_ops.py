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
"""for_loop and pfor ops."""
# pylint: disable=g-direct-tensorflow-import

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.parallel_for.pfor import PFor
from tensorflow.python.ops.parallel_for.pfor import PForConfig
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect


def for_loop(loop_fn, loop_fn_dtypes, iters, parallel_iterations=None):
  """Runs `loop_fn` `iters` times and stacks the outputs.


  Runs `loop_fn` `iters` times, with input values from 0 to `iters - 1`, and
  stacks corresponding outputs of the different runs.

  Args:
    loop_fn: A function that takes an int32 scalar tf.Tensor object representing
      the iteration number, and returns a possibly nested structure of tensor
      objects. The shape of these outputs should not depend on the input.
    loop_fn_dtypes: dtypes for the outputs of loop_fn.
    iters: Number of iterations for which to run loop_fn.
    parallel_iterations: The number of iterations that can be dispatched in
      parallel. This knob can be used to control the total memory usage.

  Returns:
    Returns a nested structure of stacked output tensor objects with the same
    nested structure as the output of `loop_fn`.
  """

  flat_loop_fn_dtypes = nest.flatten(loop_fn_dtypes)
  is_none_list = []

  def while_body(i, *ta_list):
    """Body of while loop."""
    fn_output = nest.flatten(loop_fn(i))
    if len(fn_output) != len(flat_loop_fn_dtypes):
      raise ValueError(
          "Number of expected outputs, %d, does not match the number of "
          "actual outputs, %d, from loop_fn" % (len(flat_loop_fn_dtypes),
                                                len(fn_output)))
    outputs = []
    del is_none_list[:]
    is_none_list.extend([x is None for x in fn_output])
    for out, ta in zip(fn_output, ta_list):
      # TODO(agarwal): support returning Operation objects from loop_fn.
      if out is not None:
        # out may be a ref tensor, wrap it in identity to get a non-ref tensor.
        ta = ta.write(i, array_ops.expand_dims(out, 0))
      outputs.append(ta)
    return tuple([i + 1] + outputs)

  if parallel_iterations is not None:
    extra_args = {"parallel_iterations": parallel_iterations}
  else:
    extra_args = {}
  ta_list = control_flow_ops.while_loop(
      lambda i, *ta: i < iters,
      while_body,
      [0] + [tensor_array_ops.TensorArray(dtype.base_dtype, iters)
             for dtype in flat_loop_fn_dtypes],
      **extra_args)[1:]

  # TODO(rachelim): enable this for sparse tensors

  output = [None if is_none else ta.concat()
            for ta, is_none in zip(ta_list, is_none_list)]
  return nest.pack_sequence_as(loop_fn_dtypes, output)


def _flatten_first_two_dims(x):
  """Flattens the first two dimensions of x into a single dimension."""
  old_shape = array_ops.shape(x)
  new_shape = array_ops.concat([[old_shape[0] * old_shape[1]], old_shape[2:]],
                               axis=0)
  return array_ops.reshape(x, new_shape)


PFOR_CONFIG_ARG = "pfor_config"


def pfor(loop_fn, iters, parallel_iterations=None):
  """Equivalent to running `loop_fn` `iters` times and stacking the outputs.

  `pfor` has functionality similar to `for_loop`, i.e. running `loop_fn` `iters`
  times, with input from 0 to `iters - 1`, and stacking corresponding output of
  each iteration. However the implementation does not use a tf.while_loop.
  Instead it adds new operations to the graph that collectively compute the same
  value as what running `loop_fn` in a loop would compute.


  This is an experimental feature and currently has a lot of limitations:
    - There should be no data depenendency between the different iterations. For
      example, a future iteration should not depend on a value or side-effect of
      a previous iteration.
    - Stateful kernels may mostly not be supported since these often imply a
      data dependency or ordering of the iterations. We do support a limited set
      of such stateful kernels though (like RandomFoo, Variable operations like
      reads, etc).
    - Conversion works only on a limited set of kernels for which a converter
      has been registered.
    - loop_fn has limited support for control flow operations. tf.cond in
      particular is not supported.
    - `loop_fn` should return nested structure of Tensors or Operations. However
      if an Operation is returned, it should have zero outputs.
    - The shape and dtype of `loop_fn` outputs should not depend on the input
      to loop_fn.

  Args:
    loop_fn: A function that takes an int32 scalar tf.Tensor object representing
      the iteration number, and optionally a keyword argument `pfor_config` set
      to a PForConfig object. It returns a possibly nested structure of Tensor
      or Operation objects. Note that if setting `parallel_iterations` argument
      to something other than None, `loop_fn` may be called more than once
      during graph construction. So it may need to avoid mutating global state.
    iters: Number of iterations for which to run loop_fn.
    parallel_iterations: A knob to control how many iterations are vectorized
      and dispatched in parallel. The default value of None corresponds to
      vectorizing all the iterations.  If `parallel_iterations` is smaller than
      `iters`, then chunks of at most that many iterations are dispatched in
      sequence. This knob can be used to control the total memory usage.

  Returns:
    Returns a nested structure of stacked tensor objects with the same nested
    structure as the output of `loop_fn`.
  Raises:
    ValueError: If parallel_iterations is not None and not an integer > 1.
  """
  def f():
    return _pfor_impl(loop_fn, iters, parallel_iterations=parallel_iterations)
  if context.executing_eagerly():
    f = function.defun(f)
  return f()


def _loop_fn_has_config(loop_fn):
  """Test if `loop_fn` has a `pfor_config` argument."""
  if tf_inspect.isfunction(loop_fn):
    argspec = tf_inspect.getargspec(loop_fn)
    return PFOR_CONFIG_ARG in argspec.args
  elif isinstance(loop_fn, functools.partial):
    fn = loop_fn.func
    argspec = tf_inspect.getargspec(fn)
    return (PFOR_CONFIG_ARG in argspec.args and
            PFOR_CONFIG_ARG not in loop_fn.keywords)
  else:
    loop_class = tf_decorator.unwrap(loop_fn)[1]
    if not hasattr(loop_class, "__call__"):
      raise ValueError("loop_fn object did not have a __call__ method")
    argspec = tf_inspect.getargspec(loop_class.__call__)
    return PFOR_CONFIG_ARG in argspec.args


def _pfor_impl(loop_fn, iters, parallel_iterations=None, pfor_config=None):
  """Implementation of pfor."""
  loop_fn_has_config = _loop_fn_has_config(loop_fn)
  existing_ops = set(ops.get_default_graph().get_operations())
  with ops.name_scope("loop_body"):
    loop_var = array_ops.placeholder(dtypes.int32, shape=[])
    if loop_fn_has_config:
      if pfor_config is None:
        pfor_config = PForConfig()
        pfor_config._set_iters(iters)  # pylint: disable=protected-access
      loop_fn_outputs = loop_fn(loop_var, **{PFOR_CONFIG_ARG: pfor_config})
    else:
      assert pfor_config is None
      loop_fn_outputs = loop_fn(loop_var)
  new_ops = set(ops.get_default_graph().get_operations()) - existing_ops
  iters = ops.convert_to_tensor(iters)
  if parallel_iterations is not None:
    if parallel_iterations < 1:
      raise ValueError("parallel_iterations must be None or a positive integer")
    if parallel_iterations == 1:
      raise ValueError("Found parallel_iterations == 1. Use for_loop instead.")
    iters_value = tensor_util.constant_value(iters)
    if iters_value is not None and iters_value < parallel_iterations:
      parallel_iterations = None
  if parallel_iterations is None:
    with ops.name_scope("pfor"):
      converter = PFor(loop_var, iters, new_ops, pfor_config=pfor_config)
      outputs = []
      for loop_fn_output in nest.flatten(loop_fn_outputs):
        outputs.append(converter.convert(loop_fn_output))
      return nest.pack_sequence_as(loop_fn_outputs, outputs)
  else:
    if pfor_config is not None and pfor_config._has_reductions():  # pylint: disable=protected-access
      raise ValueError("Setting parallel_iterations currently unsupported if"
                       " reductions across iterations are performed.")
    num_tiled_iterations = iters // parallel_iterations
    num_remaining_iterations = iters % parallel_iterations
    # TODO(agarwal): Avoid calling loop_fn twice. Generate the loop body inside
    # a tf.function and extract the graph from there to vectorize it.
    with ops.name_scope("pfor_untiled"):
      converter = PFor(loop_var, num_remaining_iterations, new_ops,
                       pfor_config=pfor_config)
      remaining_outputs = []
      flattened_loop_fn_outputs = nest.flatten(loop_fn_outputs)
      for loop_fn_output in flattened_loop_fn_outputs:
        remaining_outputs.append(converter.convert(loop_fn_output))

    with ops.name_scope("pfor_tiled"):
      loop_fn_dtypes = [ops.convert_to_tensor(x).dtype
                        for x in flattened_loop_fn_outputs]

      def tiled_loop_body(j):
        offset = j * parallel_iterations + num_remaining_iterations

        def tiled_loop_fn(i, pfor_config=None):
          if loop_fn_has_config:
            return nest.flatten(loop_fn(i + offset, pfor_config=pfor_config))
          else:
            return nest.flatten(loop_fn(i + offset))

        return _pfor_impl(
            tiled_loop_fn, parallel_iterations, pfor_config=pfor_config)

      tiled_outputs = for_loop(tiled_loop_body, loop_fn_dtypes,
                               num_tiled_iterations, parallel_iterations=1)
      tiled_outputs = [_flatten_first_two_dims(y) for y in tiled_outputs]

    with ops.name_scope("pfor"):
      iters_value = tensor_util.constant_value(iters)
      if iters_value is None or iters_value % parallel_iterations:
        outputs = control_flow_ops.cond(
            math_ops.equal(num_remaining_iterations, 0),
            lambda: tiled_outputs,
            lambda: [array_ops.concat([x, y], axis=0)
                     for x, y in zip(remaining_outputs, tiled_outputs)])
      else:
        outputs = tiled_outputs
      return nest.pack_sequence_as(loop_fn_outputs, nest.flatten(outputs))
