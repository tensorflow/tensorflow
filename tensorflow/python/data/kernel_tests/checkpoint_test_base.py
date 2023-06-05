# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Base test class for checkpointing datasets."""

import os

import numpy as np
from tensorflow.python.checkpoint import checkpoint as tracking_util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.data.experimental.ops import iterator_ops as contrib_iterator_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.util import nest


def remove_variants(get_next_op):
  # TODO(b/72408568): Remove this once session.run can get variant tensors.
  """Remove variants from a nest structure, so sess.run will execute."""

  def _remove_variant(x):
    if isinstance(x, ops.Tensor) and x.dtype == dtypes.variant:
      return ()
    else:
      return x

  return nest.map_structure(_remove_variant, get_next_op)


def default_test_combinations():
  """Returns the default test combinations for testing checkpointing."""

  def disable_optimizations(ds_fn):
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False

    def ds_fn_no_opt():
      return ds_fn().with_options(options)

    return ds_fn_no_opt

  def verify_unused_iterator(obj, ds_fn, num_outputs, sparse_tensors=False):
    obj.verify_unused_iterator(
        ds_fn=disable_optimizations(ds_fn=ds_fn),
        num_outputs=num_outputs,
        sparse_tensors=sparse_tensors)

  verify_unused_iterator_combination = combinations.combine(
      verify_fn=combinations.NamedObject("verify_unused_iterator",
                                         verify_unused_iterator))

  def verify_fully_used_iterator(obj, ds_fn, num_outputs, sparse_tensors=False):
    obj.verify_fully_used_iterator(
        ds_fn=disable_optimizations(ds_fn=ds_fn),
        num_outputs=num_outputs,
        sparse_tensors=sparse_tensors)

  verify_fully_used_iterator_combination = combinations.combine(
      verify_fn=combinations.NamedObject("verify_fully_used_iterator",
                                         verify_fully_used_iterator))

  def verify_exhausted_iterator(obj, ds_fn, num_outputs, sparse_tensors=False):
    obj.verify_exhausted_iterator(
        ds_fn=disable_optimizations(ds_fn=ds_fn),
        num_outputs=num_outputs,
        sparse_tensors=sparse_tensors)

  verify_exhausted_iterator_combination = combinations.combine(
      verify_fn=combinations.NamedObject("verify_exhausted_iterator",
                                         verify_exhausted_iterator))

  def verify_multiple_breaks(obj, ds_fn, num_outputs, sparse_tensors=False):
    obj.verify_multiple_breaks(
        ds_fn=disable_optimizations(ds_fn=ds_fn),
        num_outputs=num_outputs,
        sparse_tensors=sparse_tensors)

  verify_multiple_breaks_combination = combinations.combine(
      verify_fn=combinations.NamedObject("verify_multiple_breaks",
                                         verify_multiple_breaks))

  def verify_reset_restored_iterator(obj,
                                     ds_fn,
                                     num_outputs,
                                     sparse_tensors=False):
    obj.verify_reset_restored_iterator(
        ds_fn=disable_optimizations(ds_fn=ds_fn),
        num_outputs=num_outputs,
        sparse_tensors=sparse_tensors)

  verify_reset_restored_iterator_combination = combinations.combine(
      verify_fn=combinations.NamedObject("verify_reset_restored_iterator",
                                         verify_reset_restored_iterator))

  return (verify_unused_iterator_combination +
          verify_fully_used_iterator_combination +
          verify_exhausted_iterator_combination +
          verify_multiple_breaks_combination +
          verify_reset_restored_iterator_combination)


# TODO(b/72657739): Remove sparse_tensor argument, which is to test the
# (deprecated) saveable `SparseTensorSliceDataset`, once the API
# `from_sparse_tensor_slices()` and related tests are deleted.
class CheckpointTestBase(test.TestCase):
  """Base test class for checkpointing datasets."""

  def tearDown(self):
    self._delete_ckpt()
    super(CheckpointTestBase, self).tearDown()

  def verify_unused_iterator(self,
                             ds_fn,
                             num_outputs,
                             sparse_tensors=False,
                             verify_exhausted=True):
    """Verifies that saving and restoring an unused iterator works.

    Args:
      ds_fn: 0-argument function that returns a Dataset.
      num_outputs: Total number of outputs expected from this Dataset.
      sparse_tensors: Whether dataset is built from SparseTensor(s).
      verify_exhausted: Whether to verify that the iterator has been exhausted
        after producing `num_outputs` elements.

    Raises:
      AssertionError if any test fails.
    """
    self.verify_run_with_breaks(
        ds_fn, [0],
        num_outputs,
        sparse_tensors=sparse_tensors,
        verify_exhausted=verify_exhausted)

  def verify_fully_used_iterator(self,
                                 ds_fn,
                                 num_outputs,
                                 sparse_tensors=False):
    """Verifies that saving and restoring a fully used iterator works.

    Note that this only checks saving and restoring an iterator from which
    `num_outputs` items have been produced but does not check for an
    exhausted iterator, i.e., one from which an OutOfRange error has been
    returned.

    Args:
      ds_fn: 0-argument function that returns a Dataset.
      num_outputs: Total number of outputs expected from this Dataset.
      sparse_tensors: Whether dataset is built from SparseTensor(s).

    Raises:
      AssertionError if test fails.
    """
    self.verify_run_with_breaks(
        ds_fn, [num_outputs], num_outputs, sparse_tensors=sparse_tensors)

  def verify_exhausted_iterator(self, ds_fn, num_outputs, sparse_tensors=False):
    """Verifies that saving and restoring an exhausted iterator works.

    An exhausted iterator is one which has returned an OutOfRange error.

    Args:
      ds_fn: 0-argument function that returns a Dataset.
      num_outputs: Total number of outputs expected from this Dataset.
      sparse_tensors: Whether dataset is built from SparseTensor(s).

    Raises:
      AssertionError if any test fails.
    """
    self.gen_outputs(
        ds_fn, [],
        num_outputs,
        verify_exhausted=True,
        sparse_tensors=sparse_tensors)
    actual = self.gen_outputs(
        ds_fn, [],
        0,
        ckpt_saved=True,
        verify_exhausted=True,
        sparse_tensors=sparse_tensors)
    self.assertLen(actual, 0)

  def verify_multiple_breaks(self,
                             ds_fn,
                             num_outputs,
                             num_breaks=10,
                             sparse_tensors=False,
                             verify_exhausted=True):
    """Attempts to save/restore at multiple break points.

    Args:
      ds_fn: 0-argument function that returns a Dataset.
      num_outputs: Total number of outputs expected from this Dataset.
      num_breaks: The number of break points. These are uniformly spread in [0,
        num_outputs] both inclusive.
      sparse_tensors: Whether dataset is built from SparseTensor(s).
      verify_exhausted: Whether to verify that the iterator has been exhausted
        after producing `num_outputs` elements.

    Raises:
      AssertionError if any test fails.
    """
    self.verify_run_with_breaks(
        ds_fn,
        self.gen_break_points(num_outputs, num_breaks),
        num_outputs,
        sparse_tensors=sparse_tensors,
        verify_exhausted=verify_exhausted)

  def verify_reset_restored_iterator(self,
                                     ds_fn,
                                     num_outputs,
                                     break_point=None,
                                     sparse_tensors=False,
                                     verify_exhausted=True):
    """Attempts to re-initialize a restored iterator.

    This is useful when restoring a training checkpoint during validation.

    Args:
      ds_fn: 0-argument function that returns a Dataset.
      num_outputs: Total number of outputs expected from this Dataset.
      break_point: Break point. Optional. Defaults to num_outputs/2.
      sparse_tensors: Whether dataset is built from SparseTensor(s).
      verify_exhausted: Whether to verify that the iterator has been exhausted
        after producing `num_outputs` elements.

    Raises:
      AssertionError if any test fails.
    """
    if context.executing_eagerly():
      self.skipTest("Eager mode iteration do not support re-initialization.")

    break_point = num_outputs // 2 if not break_point else break_point

    # Collect ground truth containing all outputs.
    expected = self.gen_outputs(
        ds_fn, [],
        num_outputs,
        sparse_tensors=sparse_tensors,
        verify_exhausted=verify_exhausted)

    # Skip some items and save checkpoint.
    self.gen_outputs(
        ds_fn, [],
        break_point,
        sparse_tensors=sparse_tensors,
        verify_exhausted=False)

    actual = []
    # Restore from checkpoint and then run init_op.
    with ops.Graph().as_default() as g:
      saver = self._import_meta_graph()
      init_op, get_next_op = self._get_iterator_ops_from_collection(
          ds_fn, sparse_tensors=sparse_tensors)
      get_next_op = remove_variants(get_next_op)
      with self.session(graph=g) as sess:
        self._initialize(init_op, sess)
        self._restore(saver, sess)
        self._initialize(init_op, sess)
        for _ in range(num_outputs):
          actual.append(sess.run(get_next_op))
        if verify_exhausted:
          with self.assertRaises(errors.OutOfRangeError):
            sess.run(get_next_op)
    self.match(expected, actual)

  def verify_error_on_save(self,
                           ds_fn,
                           num_outputs,
                           error,
                           break_point=None,
                           sparse_tensors=False):
    """Attempts to save a non-saveable iterator.

    Args:
      ds_fn: 0-argument function that returns a Dataset.
      num_outputs: Total number of outputs expected from this Dataset.
      error: Declared error when trying to save iterator.
      break_point: Break point. Optional. Defaults to num_outputs/2.
      sparse_tensors: Whether dataset is built from SparseTensor(s).

    Raises:
      AssertionError if any test fails.
    """
    break_point = num_outputs // 2 if not break_point else break_point
    if context.executing_eagerly():
      iterator = iter(ds_fn())
      ckpt = tracking_util.Checkpoint(iterator=iterator)
      for _ in range(break_point):
        next(iterator)
      with self.assertRaises(error):
        ckpt.save(self._ckpt_path())
    else:
      with ops.Graph().as_default() as g:
        init_op, get_next_op, saver = self._build_graph(
            ds_fn, sparse_tensors=sparse_tensors)
        get_next_op = remove_variants(get_next_op)
        with self.session(graph=g) as sess:
          self._initialize(init_op, sess)
          for _ in range(break_point):
            sess.run(get_next_op)
          with self.assertRaises(error):
            self._save(sess, saver)

  def verify_run_with_breaks(self,
                             ds_fn,
                             break_points,
                             num_outputs,
                             sparse_tensors=False,
                             verify_exhausted=True):
    """Verifies that ds_fn() produces the same outputs with and without breaks.

    1. Builds a Dataset using `ds_fn` and produces `num_outputs` items from it
       *without* stopping at break points.
    2. Builds a Dataset using `ds_fn` and produces `num_outputs` items from it
       with stopping at break points.

    Deep matches outputs from 1 and 2.

    Args:
      ds_fn: 0-argument function that returns a Dataset.
      break_points: A list of integers. For each `break_point` in
        `break_points`, we produce outputs till `break_point` number of items
        have been produced and then checkpoint the state. The current graph and
        session are destroyed and a new graph and session are used to produce
        outputs till next checkpoint or till `num_outputs` elements have been
        produced. `break_point` must be <= `num_outputs`.
      num_outputs: Total number of outputs expected from this Dataset.
      sparse_tensors: Whether dataset is built from SparseTensor(s).
      verify_exhausted: Whether to verify that the iterator has been exhausted
        after producing `num_outputs` elements.

    Raises:
      AssertionError if any test fails.
    """
    expected = self.gen_outputs(
        ds_fn, [],
        num_outputs,
        sparse_tensors=sparse_tensors,
        verify_exhausted=verify_exhausted)

    actual = self.gen_outputs(
        ds_fn,
        break_points,
        num_outputs,
        sparse_tensors=sparse_tensors,
        verify_exhausted=verify_exhausted)

    self.match(expected, actual)

  def gen_outputs(self,
                  ds_fn,
                  break_points,
                  num_outputs,
                  ckpt_saved=False,
                  sparse_tensors=False,
                  verify_exhausted=True,
                  save_checkpoint_at_end=True):
    """Generates elements from input dataset while stopping at break points.

    Produces `num_outputs` outputs and saves the state of the iterator in the
    Saver checkpoint.

    Args:
      ds_fn: 0-argument function that returns the dataset.
      break_points: A list of integers. For each `break_point` in
        `break_points`, we produce outputs till `break_point` number of items
        have been produced and then checkpoint the state. The current graph and
        session are destroyed and a new graph and session are used to produce
        outputs till next checkpoint or till `num_outputs` elements have been
        produced. `break_point` must be <= `num_outputs`.
      num_outputs: The total number of outputs to produce from the iterator.
      ckpt_saved: Whether a checkpoint already exists.
      sparse_tensors:  Whether dataset is built from SparseTensor(s).
      verify_exhausted: Whether to verify that the iterator has been exhausted
        after producing `num_outputs` elements.
      save_checkpoint_at_end: Whether to save a checkpoint after producing all
        outputs. If False, checkpoints are saved each break point but not at the
        end. Note that checkpoints overwrite each other so there is always only
        a single checkpoint available. Defaults to True.

    Returns:
      A list of `num_outputs` items.
    """
    outputs = []

    if context.executing_eagerly():
      for i in range(len(break_points) + 1):
        iterator = iter(ds_fn())
        ckpt = tracking_util.Checkpoint(iterator=iterator)
        if ckpt_saved:
          ckpt_path = self._latest_ckpt()
          ckpt.restore(ckpt_path)
        start = break_points[i - 1] if i > 0 else 0
        end = break_points[i] if i < len(break_points) else num_outputs
        num_iters = end - start
        for _ in range(num_iters):
          outputs.append(self.evaluate(next(iterator)))
        if i == len(break_points) and verify_exhausted:
          with self.assertRaises(StopIteration):
            next(iterator)
        if save_checkpoint_at_end or i < len(break_points):
          # TODO(b/275117275): Verify if TF2 async checkpoint works.
          ckpt_options = checkpoint_options.CheckpointOptions()
          ckpt_options.experimental_enable_async_checkpoint = False
          ckpt_options.enable_async = False
          ckpt_path = ckpt.save(self._ckpt_path(), options=ckpt_options)
          ckpt_saved = True
    else:
      def get_ops():
        if ckpt_saved:
          saver = self._import_meta_graph()
          init_op, get_next_op = self._get_iterator_ops_from_collection(
              ds_fn, sparse_tensors=sparse_tensors)
        else:
          init_op, get_next_op, saver = self._build_graph(
              ds_fn, sparse_tensors=sparse_tensors)
        return init_op, get_next_op, saver

      for i in range(len(break_points) + 1):
        with ops.Graph().as_default() as g:
          init_op, get_next_op, saver = get_ops()
          get_next_op = remove_variants(get_next_op)
          with self.session(graph=g) as sess:
            if ckpt_saved:
              self._initialize(init_op, sess)
              self._restore(saver, sess)
            else:
              self._initialize(init_op, sess)
            start = break_points[i - 1] if i > 0 else 0
            end = break_points[i] if i < len(break_points) else num_outputs
            num_iters = end - start
            for _ in range(num_iters):
              outputs.append(sess.run(get_next_op))
            if i == len(break_points) and verify_exhausted:
              with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next_op)
            if save_checkpoint_at_end or i < len(break_points):
              self._save(sess, saver)
              ckpt_saved = True

    return outputs

  def match(self, expected, actual):
    """Matches nested structures.

    Recursively matches shape and values of `expected` and `actual`.
    Handles scalars, numpy arrays and other python sequence containers
    e.g. list, dict, as well as SparseTensorValue and RaggedTensorValue.

    Args:
      expected: Nested structure 1.
      actual: Nested structure 2.

    Raises:
      AssertionError if matching fails.
    """
    if isinstance(expected, np.ndarray):
      expected = expected.tolist()
    if isinstance(actual, np.ndarray):
      actual = actual.tolist()
    self.assertEqual(type(expected), type(actual))

    if nest.is_nested(expected):
      self.assertEqual(len(expected), len(actual))
      if isinstance(expected, dict):
        for key1, key2 in zip(sorted(expected), sorted(actual)):
          self.assertEqual(key1, key2)
          self.match(expected[key1], actual[key2])
      else:
        for item1, item2 in zip(expected, actual):
          self.match(item1, item2)
    elif isinstance(expected, sparse_tensor.SparseTensorValue):
      self.match((expected.indices, expected.values, expected.dense_shape),
                 (actual.indices, actual.values, actual.dense_shape))
    elif isinstance(expected, ragged_tensor_value.RaggedTensorValue):
      self.match((expected.values, expected.row_splits),
                 (actual.values, actual.row_splits))
    else:
      self.assertEqual(expected, actual)

  def does_not_match(self, expected, actual):
    with self.assertRaises(AssertionError):
      self.match(expected, actual)

  def gen_break_points(self, num_outputs, num_samples=10):
    """Generates `num_samples` unique break points in [0, num_outputs]."""
    return np.unique(np.linspace(0, num_outputs, num_samples, dtype=int))

  def _build_graph(self, ds_fn, sparse_tensors=False):
    dataset = ds_fn()
    iterator = dataset_ops.make_initializable_iterator(dataset)
    external_state_policy = dataset.options().experimental_external_state_policy
    saveable = contrib_iterator_ops.make_saveable_from_iterator(
        iterator, external_state_policy=external_state_policy)
    ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)
    init_op = iterator.initializer
    if sparse_tensors:
      get_next = sparse_tensor.SparseTensor(*iterator.get_next())
    else:
      get_next = iterator.get_next()
    self._add_iterator_ops_to_collection(init_op, get_next, ds_fn,
                                         sparse_tensors)
    saver = saver_lib.Saver(allow_empty=True)
    return init_op, get_next, saver

  def _add_iterator_ops_to_collection(self,
                                      init_op,
                                      get_next,
                                      ds_fn,
                                      sparse_tensors=False):
    ops.add_to_collection("iterator_ops", init_op)
    # `get_next` may be a tuple e.g. in TensorSliceDataset. Since Collections
    # do not support tuples we flatten the tensors and restore the shape in
    # `_get_iterator_ops_from_collection`.
    if sparse_tensors:  # specific for deprecated `from_sparse_tensor_slices`.
      ops.add_to_collection("iterator_ops", get_next.indices)
      ops.add_to_collection("iterator_ops", get_next.values)
      ops.add_to_collection("iterator_ops", get_next.dense_shape)
      return

    get_next_list = nest.flatten(get_next)
    for i, output_class in enumerate(
        nest.flatten(self._get_output_classes(ds_fn))):
      if output_class is sparse_tensor.SparseTensor:
        ops.add_to_collection("iterator_ops", get_next_list[i].indices)
        ops.add_to_collection("iterator_ops", get_next_list[i].values)
        ops.add_to_collection("iterator_ops", get_next_list[i].dense_shape)
      else:
        ops.add_to_collection("iterator_ops", get_next_list[i])

  def _get_iterator_ops_from_collection(self, ds_fn, sparse_tensors=False):
    all_ops = ops.get_collection("iterator_ops")
    if sparse_tensors:  # specific for deprecated `from_sparse_tensor_slices`.
      init_op, indices, values, dense_shape = all_ops
      return init_op, sparse_tensor.SparseTensor(indices, values, dense_shape)
    get_next_list = []
    i = 1
    for output_class in nest.flatten(self._get_output_classes(ds_fn)):
      if output_class is sparse_tensor.SparseTensor:
        indices, values, dense_shape = all_ops[i:i + 3]
        i += 3
        get_next_list.append(
            sparse_tensor.SparseTensor(indices, values, dense_shape))
      else:
        get_next_list.append(all_ops[i])
        i += 1
    return all_ops[0], nest.pack_sequence_as(
        self._get_output_types(ds_fn), get_next_list)

  def _get_output_types(self, ds_fn):
    assert not context.executing_eagerly()
    with ops.Graph().as_default():
      return dataset_ops.get_legacy_output_types(ds_fn())

  def _get_output_shapes(self, ds_fn):
    assert not context.executing_eagerly()
    with ops.Graph().as_default():
      return dataset_ops.get_legacy_output_shapes(ds_fn())

  def _get_output_classes(self, ds_fn):
    assert not context.executing_eagerly()
    with ops.Graph().as_default():
      return dataset_ops.get_legacy_output_classes(ds_fn())

  def _ckpt_path(self):
    return os.path.join(self.get_temp_dir(), "iterator")

  def _latest_ckpt(self):
    return checkpoint_management.latest_checkpoint(self.get_temp_dir())

  def _save(self, sess, saver):
    saver.save(sess, self._ckpt_path())

  def _restore(self, saver, sess):
    sess.run(lookup_ops.tables_initializer())
    saver.restore(sess, self._latest_ckpt())

  def _initialize(self, init_op, sess):
    sess.run(variables.global_variables_initializer())
    sess.run(lookup_ops.tables_initializer())
    sess.run(init_op)

  def _import_meta_graph(self):
    meta_file_path = self._ckpt_path() + ".meta"
    return saver_lib.import_meta_graph(meta_file_path)

  def _delete_ckpt(self):
    # Remove all checkpoint files.
    prefix = self._ckpt_path()
    pattern = prefix + "*"
    files = gfile.Glob(pattern)
    map(gfile.Remove, files)
