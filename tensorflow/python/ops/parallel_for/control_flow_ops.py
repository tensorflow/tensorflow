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
from tensorflow.python.eager import def_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.parallel_for.pfor import PFor
from tensorflow.python.ops.parallel_for.pfor import PForConfig
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


def for_loop(loop_fn, loop_fn_dtypes, iters, parallel_iterations=None):
  """Runs `loop_fn` `iters` times and stacks the outputs.


  Runs `loop_fn` `iters` times, with input values from 0 to `iters - 1`, and
  stacks corresponding outputs of the different runs.

  Args:
    loop_fn: A function that takes an int32 scalar tf.Tensor object representing
      the iteration number, and returns a possibly nested structure of tensor
      objects. The shape of these outputs should not depend on the input.
    loop_fn_dtypes: dtypes for the outputs of `loop_fn`.
    iters: Number of iterations for which to run `loop_fn`.
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
          f"Number of expected outputs {len(flat_loop_fn_dtypes)}, does not "
          f"match the number of actual outputs {len(fn_output)} from loop_fn: "
          f"{loop_fn} with output {fn_output}.")
    outputs = []
    del is_none_list[:]
    is_none_list.extend(x is None for x in fn_output)
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
  assert len(output) in (0, len(flat_loop_fn_dtypes))
  if not output:
    # This may happen for the case where iters == 0.
    return None
  else:
    return nest.pack_sequence_as(loop_fn_dtypes, output)


def _flatten_first_two_dims(x):
  """Flattens the first two dimensions of x into a single dimension."""
  old_shape = array_ops.shape(x)
  new_shape = array_ops.concat([[old_shape[0] * old_shape[1]], old_shape[2:]],
                               axis=0)
  return array_ops.reshape(x, new_shape)


PFOR_CONFIG_ARG = "pfor_config"


def _is_under_xla_context():
  """Check if we are currently inside an XLA compile context."""
  g = ops.get_default_graph()
  while g is not None:
    control_flow_context = g._get_control_flow_context()  # pylint: disable=protected-access
    while control_flow_context is not None:
      if control_flow_context.IsXLAContext():
        return True
      else:
        control_flow_context = control_flow_context.outer_context
    # If g is a FuncGraph, get its outer_graph.
    g = getattr(g, "outer_graph", None)
  return False


def pfor(loop_fn, iters, fallback_to_while_loop=True, parallel_iterations=None):
  """Equivalent to running `loop_fn` `iters` times and stacking the outputs.

  `pfor` has functionality similar to `for_loop`, i.e. running `loop_fn` `iters`
  times, with input from 0 to `iters - 1`, and stacking corresponding output of
  each iteration. However the implementation does not use a `tf.while_loop`.
  Instead it adds new operations to the graph that collectively compute the same
  value as what running `loop_fn` in a loop would compute.


  This is an experimental feature and currently has a lot of limitations:
    - There should be no data dependency between the different iterations. For
      example, a future iteration should not depend on a value or side-effect of
      a previous iteration.
    - Stateful kernels may mostly not be supported since these often imply a
      data dependency or ordering of the iterations. We do support a limited set
      of such stateful kernels though (like RandomFoo, Variable operations like
      reads, etc).
    - Conversion works only on a limited set of kernels for which a converter
      has been registered.
    - `loop_fn` has limited support for control flow operations. `tf.cond` in
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
    iters: Number of iterations for which to run `loop_fn`.
    fallback_to_while_loop: If true, on failing to vectorize an operation, pfor
      fallbacks to using a `tf.while_loop` to dispatch the iterations.
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
    return _pfor_impl(loop_fn,
                      iters,
                      fallback_to_while_loop=fallback_to_while_loop,
                      parallel_iterations=parallel_iterations)
  # Note that we wrap into a tf.function if in eager execution mode or under
  # XLA compilation. The latter is so that we don't compile operations like
  # tf.placeholder that are created by the loop body.
  functions_run_eagerly = None
  if context.executing_eagerly() or _is_under_xla_context():
    functions_run_eagerly = def_function.functions_run_eagerly()
    if functions_run_eagerly:
      logging.warning(
          "It looks like tf.function behavior was disabled, perhaps using "
          "tf.config.run_functions_eagerly. Vectorization "
          "primitives (e.g. tf.vectorized_map) require tf.function to work. "
          "These primitives will override the disable.")
      def_function.run_functions_eagerly(False)
    f = def_function.function(f)
  outputs = f()
  if functions_run_eagerly is not None:
    def_function.run_functions_eagerly(functions_run_eagerly)
  return outputs


def _should_expand_composite(value):
  return (isinstance(value, composite_tensor.CompositeTensor)
          # Leave sparse tensors to be converted by `PFor._convert_sparse`.
          and not isinstance(value, sparse_tensor.SparseTensor)
          and not isinstance(value, indexed_slices.IndexedSlices))


# pylint: disable=protected-access
def _composite_to_tensors(value, is_batched=False):
  """Converts a CompositeTensor into a list of stackable tensors."""
  if _should_expand_composite(value):
    spec = value._type_spec
    if not isinstance(spec, type_spec.BatchableTypeSpec):
      raise ValueError(f"CompositeTensor instance {value} returned from "
                       "parallel_for or vectorized_map loop body must provide "
                       f"a `BatchableTypeSpec` (saw: {spec}).")
    if is_batched:
      return spec._to_batched_tensor_list(value)
    return spec._to_tensor_list(value)
  return value
# pylint: enable=protected-access


# pylint: disable=protected-access
def _composite_from_tensors(stacked_tensors,
                            preconverted_value,
                            batch_size):
  """Converts a list of stacked tensors to a batch CompositeTensor."""
  if _should_expand_composite(preconverted_value):
    batch_type_spec = preconverted_value._type_spec._batch(batch_size)
    return batch_type_spec._from_compatible_tensor_list(stacked_tensors)
  return stacked_tensors
# pylint: enable=protected-access


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
      raise ValueError("`loop_fn` object did not have a __call__ method")
    argspec = tf_inspect.getargspec(loop_class.__call__)
    return PFOR_CONFIG_ARG in argspec.args


def _pfor_impl(loop_fn,
               iters,
               fallback_to_while_loop,
               parallel_iterations=None,
               pfor_config=None):
  """Implementation of pfor."""
  assert not context.executing_eagerly()
  loop_fn_has_config = _loop_fn_has_config(loop_fn)
  existing_ops = set(ops.get_default_graph().get_operations())
  iters_value = tensor_util.constant_value(iters)
  # Run the loop body
  with ops.name_scope("loop_body"):
    loop_var = array_ops.placeholder_with_default(0, shape=[])
    if loop_fn_has_config:
      if pfor_config is None:
        pfor_config = PForConfig()
        pfor_config._set_iters(iters)  # pylint: disable=protected-access
      loop_fn_outputs = loop_fn(loop_var, **{PFOR_CONFIG_ARG: pfor_config})
    else:
      assert pfor_config is None
      loop_fn_outputs = loop_fn(loop_var)
    loop_fn_output_tensors = nest.map_structure(_composite_to_tensors,
                                                loop_fn_outputs)

  # Convert outputs to Tensor if needed.
  tmp_loop_fn_outputs = []
  for loop_fn_output in nest.flatten(loop_fn_output_tensors):
    if (loop_fn_output is not None and not isinstance(
        loop_fn_output,
        (ops.Operation, ops.Tensor, sparse_tensor.SparseTensor))):
      if isinstance(loop_fn_output, indexed_slices.IndexedSlices):
        logging.warn("Converting %s to a dense representation may make it slow."
                     " Alternatively, output the indices and values of the"
                     " IndexedSlices separately, and handle the vectorized"
                     " outputs directly." % loop_fn_output)
        loop_fn_output = ops.convert_to_tensor(loop_fn_output)
      else:
        loop_fn_output = ops.convert_to_tensor(loop_fn_output)
    tmp_loop_fn_outputs.append(loop_fn_output)
  loop_fn_output_tensors = nest.pack_sequence_as(loop_fn_output_tensors,
                                                 tmp_loop_fn_outputs)

  new_ops = set(ops.get_default_graph().get_operations()) - existing_ops
  iters = ops.convert_to_tensor(iters)
  if parallel_iterations is not None:
    if parallel_iterations < 1:
      raise ValueError(
          "Argument `parallel_iterations` must be None or a positive integer. "
          f"Received: {parallel_iterations}.")
    if parallel_iterations == 1:
      raise ValueError(
          "Found `parallel_iterations == 1`. Use `for_loop` instead.")
    if iters_value is not None and iters_value < parallel_iterations:
      parallel_iterations = None
  if parallel_iterations is None:
    with ops.name_scope("pfor"):
      converter = PFor(loop_var, iters, new_ops,
                       fallback_to_while_loop=fallback_to_while_loop,
                       pfor_config=pfor_config)
      flattened_output_tensors = []
      for loop_fn_output in nest.flatten(loop_fn_output_tensors):
        output = converter.convert(loop_fn_output)
        flattened_output_tensors.append(output)
  else:
    if pfor_config is not None and pfor_config._has_reductions():  # pylint: disable=protected-access
      raise ValueError("Setting `parallel_iterations` currently unsupported if "
                       "reductions across iterations are performed.")
    num_tiled_iterations = iters // parallel_iterations
    num_remaining_iterations = iters % parallel_iterations
    # TODO(agarwal): Avoid calling loop_fn twice. Generate the loop body inside
    # a tf.function and extract the graph from there to vectorize it.
    with ops.name_scope("pfor_untiled"):
      converter = PFor(loop_var, num_remaining_iterations, new_ops,
                       fallback_to_while_loop=fallback_to_while_loop,
                       pfor_config=pfor_config)
      remaining_output_tensors = []
      flattened_output_tensors = nest.flatten(loop_fn_output_tensors)
      for loop_fn_output in flattened_output_tensors:
        output = converter.convert(loop_fn_output)
        remaining_output_tensors.append(output)

    with ops.name_scope("pfor_tiled"):
      loop_fn_dtypes = [ops.convert_to_tensor(x).dtype
                        for x in flattened_output_tensors]

      def tiled_loop_body(j):
        offset = j * parallel_iterations + num_remaining_iterations

        def tiled_loop_fn(i, pfor_config=None):
          if loop_fn_has_config:
            loop_fn_outputs = loop_fn(i + offset, pfor_config=pfor_config)
          else:
            loop_fn_outputs = loop_fn(i + offset)
          return nest.flatten(
              # Stacking across iterations requires explicit Tensors.
              nest.map_structure(_composite_to_tensors, loop_fn_outputs))

        return _pfor_impl(
            tiled_loop_fn,
            parallel_iterations,
            fallback_to_while_loop=fallback_to_while_loop,
            pfor_config=pfor_config)

      tiled_output_tensors = for_loop(
          tiled_loop_body, loop_fn_dtypes,
          num_tiled_iterations, parallel_iterations=1)
      tiled_output_tensors = [
          _flatten_first_two_dims(y) for y in tiled_output_tensors]

    with ops.name_scope("pfor"):
      if iters_value is None or iters_value % parallel_iterations:
        output_tensors = control_flow_ops.cond(
            math_ops.equal(num_remaining_iterations, 0),
            lambda: tiled_output_tensors,
            lambda: [array_ops.concat([x, y], axis=0)  # pylint: disable=g-long-lambda
                     for x, y in zip(remaining_output_tensors,
                                     tiled_output_tensors)])
      else:
        output_tensors = tiled_output_tensors
      flattened_output_tensors = nest.flatten(output_tensors)

      for output, original_output in zip(flattened_output_tensors,
                                         nest.flatten(loop_fn_output_tensors)):
        # Restore any shape information lost from tiling.
        # TODO(b/174254748): this may not be correct for stacked `variant`s.
        output.set_shape(
            tensor_shape.TensorShape([iters_value]).concatenate(
                original_output.shape))

  return nest.map_structure_up_to(
      loop_fn_outputs,
      functools.partial(_composite_from_tensors, batch_size=iters_value),
      nest.pack_sequence_as(loop_fn_output_tensors,
                            flattened_output_tensors),
      loop_fn_outputs)


def _broadcasting_gather(x, i):
  """Wrapper for gather that implicitly broadcasts unit dimensions."""
  static_first_dim = tensor_shape.dimension_value(x.shape[0])
  if static_first_dim == 1:
    i = 0
  elif static_first_dim is None:
    i = array_ops.where_v2(array_ops.shape(x)[0] > 1, i, 0)
  result = array_ops.gather(x, i)
  return result


# pylint: disable=protected-access
def _gather_from_tensor_or_composite(x, i):
  """Wrapper for gather that handles CompositeTensors."""
  if _should_expand_composite(x):
    spec = x._type_spec
    gathered_tensors = [_broadcasting_gather(t, i)
                        for t in spec._to_batched_tensor_list(x)]
    return spec._unbatch()._from_compatible_tensor_list(gathered_tensors)
  return _broadcasting_gather(x, i)
# pylint: enable=protected-access


@tf_export("vectorized_map")
def vectorized_map(fn, elems, fallback_to_while_loop=True):
  """Parallel map on the list of tensors unpacked from `elems` on dimension 0.

  This method works similar to `tf.map_fn` but is optimized to run much faster,
  possibly with a much larger memory footprint. The speedups are obtained by
  vectorization (see [Auto-Vectorizing TensorFlow Graphs: Jacobians,
  Auto-Batching and Beyond](https://arxiv.org/pdf/1903.04243.pdf)). The idea
  behind vectorization is to semantically launch all the invocations of `fn` in
  parallel and fuse corresponding operations across all these invocations. This
  fusion is done statically at graph generation time and the generated code is
  often similar in performance to a manually fused version.

  Because `tf.vectorized_map` fully parallelizes the batch, this method will
  generally be significantly faster than using `tf.map_fn`, especially in eager
  mode. However this is an experimental feature and currently has a lot of
  limitations:
    - There should be no data dependency between the different semantic
      invocations of `fn`, i.e. it should be safe to map the elements of the
      inputs in any order.
    - Stateful kernels may mostly not be supported since these often imply a
      data dependency. We do support a limited set of such stateful kernels
      though (like RandomFoo, Variable operations like reads, etc).
    - `fn` has limited support for control flow operations.
    - `fn` should return nested structure of Tensors or Operations. However
      if an Operation is returned, it should have zero outputs.
    - The shape and dtype of any intermediate or output tensors in the
      computation of `fn` should not depend on the input to `fn`.

  Examples:
  ```python
  def outer_product(a):
    return tf.tensordot(a, a, 0)

  batch_size = 100
  a = tf.ones((batch_size, 32, 32))
  c = tf.vectorized_map(outer_product, a)
  assert c.shape == (batch_size, 32, 32, 32, 32)
  ```

  ```python
  # Computing per-example gradients

  batch_size = 10
  num_features = 32
  layer = tf.keras.layers.Dense(1)

  def model_fn(arg):
    with tf.GradientTape() as g:
      inp, label = arg
      inp = tf.expand_dims(inp, 0)
      label = tf.expand_dims(label, 0)
      prediction = layer(inp)
      loss = tf.nn.l2_loss(label - prediction)
    return g.gradient(loss, (layer.kernel, layer.bias))

  inputs = tf.random.uniform([batch_size, num_features])
  labels = tf.random.uniform([batch_size, 1])
  per_example_gradients = tf.vectorized_map(model_fn, (inputs, labels))
  assert per_example_gradients[0].shape == (batch_size, num_features, 1)
  assert per_example_gradients[1].shape == (batch_size, 1)
  ```

  Args:
    fn: The callable to be performed. It accepts one argument, which will have
      the same (possibly nested) structure as `elems`, and returns a possibly
      nested structure of Tensors and Operations, which may be different than
      the structure of `elems`.
    elems: A tensor or (possibly nested) sequence of tensors, each of which will
      be unpacked along their first dimension. The nested sequence of the
      resulting slices will be mapped over by `fn`. The first dimensions of all
      elements must broadcast to a consistent value; equivalently, each
      element tensor must have first dimension of either `B` or `1`, for some
      common batch size `B >= 1`.
    fallback_to_while_loop: If true, on failing to vectorize an operation,
      the unsupported op is wrapped in a tf.while_loop to execute the map
      iterations. Note that this fallback only happens for unsupported ops and
      other parts of `fn` are still vectorized. If false, on encountering an
      unsupported op, a ValueError is thrown. Note that the fallbacks can result
      in slowdowns since vectorization often yields speedup of one to two orders
      of magnitude.

  Returns:
    A tensor or (possibly nested) sequence of tensors. Each tensor packs the
    results of applying fn to tensors unpacked from elems along the first
    dimension, from first to last.

    Although they are less common as user-visible inputs and outputs, note that
    tensors of type `tf.variant` which represent tensor lists (for example from
    `tf.raw_ops.TensorListFromTensor`) are vectorized by stacking the list
    contents rather than the variant itself, and so the container tensor will
    have a scalar shape when returned rather than the usual stacked shape. This
    improves the performance of control flow gradient vectorization.

  Raises:
    ValueError: If vectorization fails and fallback_to_while_loop is False.
  """
  elems = nest.map_structure(ops.convert_to_tensor,
                             elems,
                             expand_composites=True)

  def loop_fn(i):
    gathered_elems = nest.map_structure(
        lambda x: _gather_from_tensor_or_composite(x, i), elems)
    return fn(gathered_elems)

  # Extract batch size from the maximum first dimension of any element.
  flat_elems = nest.flatten(
      nest.map_structure(
          functools.partial(_composite_to_tensors,
                            is_batched=True),
          elems))
  def _get_shape(x):
    if x.shape.rank is None:
      return None
    return x.shape.as_list()[0]
  static_first_dims = [_get_shape(elem) for elem in flat_elems]
  if any(s is None for s in static_first_dims):
    batch_size = math_ops.reduce_max(
        [array_ops.shape(elem)[0] for elem in flat_elems])
  else:
    batch_size = max(static_first_dims)

  return pfor(loop_fn, batch_size,
              fallback_to_while_loop=fallback_to_while_loop)
