# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Autograph specifc overrides for dataset_ops."""
from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.experimental.ops import take_while_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest


def _general_purpose_scan(ds, init_state, body):
  """Variant of Dataset.scan with semantics of general-purpose computation."""
  # Datasets are typically intended for data preprocessing. However, in
  # autograph loops they usually appear as general-purpose computations (for
  # example, a custom training loop). These two use cases require significantly
  # different optimization policies, the most important of which is the device
  # placement. The flag override for use_default_device below instructs the
  # runtime to treat the computation as general-purpose, rather than data
  # preprocessing.

  # Loaded lazily due to a circular dependency (dataset_ops ->
  # scan_op -> dataset_ops).
  # pylint: disable=g-import-not-at-top,protected-access
  from tensorflow.python.data.ops import scan_op
  return scan_op._ScanDataset(ds, init_state, body, use_default_device=False)
  # pylint: enable=g-import-not-at-top,protected-access


def _tf_ag_dataset_for_stmt(
    ds, extra_test, body, get_state, set_state, symbol_names, opts
):
  """Overload of _dataset_for_stmt with early stopping. See for_stmt."""
  # Note: This is easier to follow with the insight that the computations in
  # a dataset pipeline are transposed (aka fused).
  # For example, given a pipeline input -> scan -> take_while -> reduce,
  # and a dataset with input [1, 2, 3], the computations occur in the following
  # order:
  #  reduce(take_while(scan(1)))
  #  reduce(take_while(scan(2)))
  #  reduce(take_while(scan(3)))

  init_vars = get_state()
  control_flow.verify_loop_init_vars(init_vars, symbol_names)

  # Workaround for Dataset.reduce not allowing empty state tensors - create
  # a dummy state variable that remains unused.
  # TODO(mdan): reduce should allow and match empty structures.
  if not init_vars:
    init_vars = (constant_op.constant(0),)
    symbol_names = ("<internal dummy>",)

    def dummy_set_state(unused_dummy):
      pass

    def dummy_get_state():
      return (constant_op.constant(0),)

    get_state, set_state = dummy_get_state, dummy_set_state

  def scan_body(scan_state, scan_inputs):
    """Main body of the Dataset.scan."""
    loop_vars, iterate = scan_state, scan_inputs
    set_state(loop_vars)

    def main_path():
      body(iterate)
      new_loop_vars = get_state()
      control_flow.verify_tf_loop_vars(
          init_vars,
          loop_vars,
          new_loop_vars,
          symbol_names,
          opts,
          check_shapes=False)
      return new_loop_vars

    if extra_test is not None:
      extra_cond = extra_test()
      new_loop_vars = control_flow_ops.cond(extra_cond, main_path,
                                            lambda: loop_vars)
    else:
      # TODO(mdan): the optimizer should be able to remove an invariant cond?
      extra_cond = (constant_op.constant(True),)  # dummy value, unused
      new_loop_vars = main_path()

    scan_outputs = new_loop_vars, extra_cond
    new_scan_state = new_loop_vars
    return new_scan_state, scan_outputs

  def take_while_predicate(unused_loop_vars, extra_cond):
    return extra_cond

  def reduce_body(unused_reduce_state, scan_outputs):
    output_loop_vars, unused_extra_cond = scan_outputs
    new_reduce_state = output_loop_vars
    return new_reduce_state

  ds = _general_purpose_scan(ds, init_vars, scan_body)
  if extra_test is not None:
    ds = ds.apply(take_while_ops.take_while(take_while_predicate))
  final_loop_vars = ds.reduce(init_vars, reduce_body)
  set_state(final_loop_vars)


def _tf_ag_dataset_abs(ds):
  specs = nest.flatten(ds.element_spec)
  if len(specs) == 1:
    return ds.map(math_ops.abs, num_parallel_calls=dataset_ops.AUTOTUNE)
  return ds.map(
      lambda *e: nest.map_structure(math_ops.abs, e),
      num_parallel_calls=dataset_ops.AUTOTUNE)


def _tf_ag_dataset_len(s):
  """Autograph override of the builtin len for dataset_ops.DataSetV2."""
  l = s.cardinality()
  msg = gen_string_ops.string_join([
      "len requires dataset with definitive cardinality, got ",
      gen_string_ops.as_string(l),
  ])
  # TODO(yongtang): UNKNOWN is treated as an error.
  # In case there are more UNKNOWN cases for dataset, we could
  # use dataset.reduce() to find out the length (in an expensive way).
  with ops.control_dependencies([
      control_flow_assert.Assert(
          math_ops.logical_and(
              math_ops.not_equal(l, dataset_ops.INFINITE),
              math_ops.not_equal(l, dataset_ops.UNKNOWN)), [msg])
  ]):
    l = array_ops.identity(l)

  return l


def _tf_ag_dataset_enumerate(ds, start=0):
  return ds.enumerate(start)


def _tf_ag_dataset_zip(*iterables):
  return dataset_ops.DatasetV2.zip(iterables)


def _tf_ag_dataset_map(fn, *iterables):
  return dataset_ops.DatasetV2.zip(iterables).map(fn)


def _tf_ag_dataset_filter(fn, iterable):
  return iterable.filter(fn)


# any() operation is essentially a "if first True element exist".
# For that it could be translated to `filter(True)` to filter out
# only `True` element, and then `take(1)`. This works in tf.data
# as tf.data's filter+take is done in pipeline so it will stop
# as soon as `take(1)` returns.
def _tf_ag_dataset_any(iterable):
  # check and make sure iterable.element_spec only consists of one
  # element of tf.bool.
  specs = nest.flatten(iterable.element_spec)
  if len(specs) != 1 or specs[0].dtype != dtypes.bool:
    raise ValueError('in graph mode, the "any" builtin only supports datasets '
                     'that return bool scalars; got: {}'.format(
                         iterable.element_spec))
  ds = iterable.filter(lambda x: x)
  ds = ds.take(1)
  ds = ds.reduce(constant_op.constant(False, dtype=dtypes.bool), lambda _, y: y)
  return ds


# all() operation is similar to any() and could be translated
# to `filter(False)` then `take(1)`, and check if `False` exists.
def _tf_ag_dataset_all(iterable):
  # check and make sure iterable.element_spec only consists of one
  # element of tf.bool.
  specs = nest.flatten(iterable.element_spec)
  if len(specs) != 1 or specs[0].dtype != dtypes.bool:
    raise ValueError('in graph mode, the "all" builtin only supports datasets '
                     'that return bool scalars; got: {}'.format(
                         iterable.element_spec))
  ds = iterable.filter(math_ops.logical_not)
  ds = ds.take(1)
  ds = ds.reduce(constant_op.constant(True, dtype=dtypes.bool), lambda _, y: y)
  return ds


def register_overrides():
  """Registers the autograph specific overrides for dataset_ops."""
  control_flow.for_loop_registry.register(
      dataset_ops.DatasetV2, _tf_ag_dataset_for_stmt
  )
  py_builtins.abs_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_abs)
  py_builtins.len_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_len)
  py_builtins.enumerate_registry.register(
      dataset_ops.DatasetV2, _tf_ag_dataset_enumerate
  )
  py_builtins.zip_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_zip)
  py_builtins.map_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_map)
  py_builtins.filter_registry.register(
      dataset_ops.DatasetV2, _tf_ag_dataset_filter
  )
  py_builtins.any_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_any)
  py_builtins.all_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_all)
