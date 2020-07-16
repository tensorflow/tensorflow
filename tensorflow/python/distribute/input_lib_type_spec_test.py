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
"""Tests for the input_lib library which tests iterator type specs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl.testing import parameterized
import numpy as np

from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import values
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_tensor as ragged_tensor_lib
from tensorflow.python.util import nest


class DistributedIteratorTest(test.TestCase,
                              parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          input_type=["dataset"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
          ],
          enable_get_next_as_optional=[True, False]))
  def testTypeSpec(self, input_type, distribution,
                   enable_get_next_as_optional):
    if not tf2.enabled():
      self.skipTest("DistributedIterator has CompositeTensor support in "
                    "TF 2 only.")
    dataset = dataset_ops.DatasetV2.range(10).batch(2)

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)

    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    with distribution.scope():
      iterator = iter(dist_dataset)
      _check_type_spec_structure(iterator)

    spec = iterator._type_spec
    self.assertEqual(spec._input_workers, iterator._input_workers)
    self.assertEqual(spec._element_spec._value_specs,
                     (tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.int64,
                                             name=None),
                      tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.int64,
                                             name=None)))

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          input_type=["dataset"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
          ],
          enable_get_next_as_optional=[True, False]))
  def testTypeSpecRoundTrip(self, input_type,
                            distribution, enable_get_next_as_optional):
    if not tf2.enabled():
      self.skipTest("DistributedIterator CompositeTensor support is only "
                    "present in TF 2.0 only.")

    dataset = dataset_ops.DatasetV2.range(10).batch(2)

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)

    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    with distribution.scope():
      iterator = iter(dist_dataset)
      _check_type_spec_structure(iterator)

    spec = iterator._type_spec

    tensor_list = spec._to_components(iterator)
    re_iterator = spec._from_components(tensor_list)

    self.assertEqual(iterator._input_workers, re_iterator._input_workers)
    self.assertAllEqual(iterator._iterators, re_iterator._iterators)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          input_type=["dataset"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
          ],
          enable_get_next_as_optional=[True, False]))
  def testDoesNotTriggerFunctionTracing(self, input_type, distribution,
                                        enable_get_next_as_optional):
    if not tf2.enabled():
      self.skipTest("DistributedIterator CompositeTensor support is only "
                    "present in TF 2.0 only.")

    trace_count = [0]

    @def_function.function
    def f(iterator):
      trace_count[0] += 1
      counter = np.int64(0)
      for _ in range(5):
        next(iterator)
        counter += 1
      return counter

    dataset = dataset_ops.DatasetV2.range(10).batch(2)

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)

    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    with distribution.scope():
      for _ in range(3):
        iterator = iter(dist_dataset)
        _check_type_spec_structure(iterator)
        counter = f(iterator)

        self.assertEqual(trace_count[0], 1)
        self.assertEqual(counter, 5)


class InputTypeSpecTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.central_storage_strategy_with_two_gpus,
          ],
          input_type=["dataset", "dataset_fn"],
      ))
  def testInputSignatureForPerReplicaValues(self, distribution, input_type):
    def dataset_fn(ctx):
      del ctx  # unused
      return dataset_ops.DatasetV2.from_tensor_slices(
          np.ones([10, 12]).astype(np.float32)).batch(4)

    if input_type == "dataset":
      ds = distribution.experimental_distribute_dataset(
          dataset_fn(distribute_lib.InputContext()))
      type_spec = ds.element_spec
    else:
      ds = distribution.experimental_distribute_datasets_from_function(
          dataset_fn)
      iterator = iter(ds)
      _check_type_spec_structure(iterator)
      type_spec = iterator.element_spec

    @def_function.function(input_signature=[type_spec])
    def process_inputs(inputs):
      distribution.run(lambda inputs: inputs, args=(inputs,))

    for x in ds:
      process_inputs(x)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.central_storage_strategy_with_two_gpus,
          ],
      ))
  def testInputSignatureForNestedPerReplicaValues(self, distribution):
    a = np.ones((10, 2)) * 5
    b = np.ones((10, 3)) * 6
    dataset = dataset_ops.DatasetV2.from_tensor_slices((a, b)).batch(2)

    dist_dataset = distribution.experimental_distribute_dataset(dataset)

    @def_function.function(input_signature=[dist_dataset.element_spec])
    def process_inputs(inputs):
      distribution.run(lambda inputs: inputs, args=(inputs,))

    for x in dist_dataset:
      process_inputs(x)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          input_type=["dataset"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
          ],
          enable_get_next_as_optional=[True, False]))
  def testMostSpecificCompatibleType(self, input_type, distribution,
                                     enable_get_next_as_optional):
    if not tf2.enabled():
      self.skipTest("DistributedIterator has CompositeTensor support in "
                    "TF 2 only.")
    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)

    ds1 = dataset_ops.DatasetV2.range(10).batch(2).batch(5)
    ds2 = dataset_ops.DatasetV2.from_tensors(
        array_ops.zeros([5, 2], dtypes.int64))
    dist_ds1 = distribution.experimental_distribute_dataset(ds1)
    dist_ds2 = distribution.experimental_distribute_dataset(ds2)

    with distribution.scope():
      iter1 = iter(dist_ds1)
      iter2 = iter(dist_ds2)

    spec1 = iter1._type_spec  # Wrapped TensorSpec has shape [None, None]
    spec2 = iter2._type_spec  # Wrapped TensorSpec has shape [None, 2]

    self.assertNotEqual(spec1, spec2)
    self.assertEqual(spec1, spec1.most_specific_compatible_type(spec2))
    self.assertEqual(spec1, spec2.most_specific_compatible_type(spec1))


class RaggedTensorDistributedIteratorTest(test.TestCase,
                                          parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          enable_get_next_as_optional=[True, False]))
  def testTypeSpec(self, distribution, enable_get_next_as_optional):
    if not tf2.enabled():
      self.skipTest("DistributedIterator has CompositeTensor support in "
                    "TF 2.0 only.")
    ctx = distribute_lib.InputContext()
    batch_size = ctx.get_per_replica_batch_size(8)
    # Use 20 which isn't divisible by 8 to test partial batch behavior.
    row_lengths = np.mod(np.arange(20), 4).astype(np.int64)
    ragged_tensor = ragged_tensor_lib.RaggedTensor.from_row_lengths(
        np.repeat(np.arange(20, dtype=np.float32), row_lengths), row_lengths)
    dataset = dataset_ops.DatasetV2.from_tensor_slices({
        "dense": ragged_tensor.to_tensor(),
        "ragged": ragged_tensor,
        "sparse": ragged_tensor.to_sparse(),
    })
    dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
    dataset = dataset.batch(batch_size)

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)

    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    with distribution.scope():
      iterator = iter(dist_dataset)
      _check_type_spec_structure(iterator)

    spec = iterator._type_spec
    self.assertEqual(spec._input_workers, iterator._input_workers)
    self.assertEqual(
        spec._element_spec, {
            "sparse":
                values.PerReplicaSpec(
                    sparse_tensor.SparseTensorSpec(
                        tensor_shape.TensorShape([None, 3]), dtypes.float32),
                    sparse_tensor.SparseTensorSpec(
                        tensor_shape.TensorShape([None, 3]), dtypes.float32)),
            "dense":
                values.PerReplicaSpec(
                    tensor_spec.TensorSpec(
                        shape=(None, 3), dtype=dtypes.float32, name=None),
                    tensor_spec.TensorSpec(
                        shape=(None, 3), dtype=dtypes.float32, name=None)),
            "ragged":
                values.PerReplicaSpec(
                    ragged_tensor_lib.RaggedTensorSpec(
                        tensor_shape.TensorShape([None, None]), dtypes.float32,
                        1, dtypes.int64),
                    ragged_tensor_lib.RaggedTensorSpec(
                        tensor_shape.TensorShape([None, None]), dtypes.float32,
                        1, dtypes.int64))
        })

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
          ],
          enable_get_next_as_optional=[True, False]))
  def testTypeSpecRoundTrip(self, distribution, enable_get_next_as_optional):
    if not tf2.enabled():
      self.skipTest("DistributedIterator CompositeTensor support is only "
                    "present in TF 2.0 only.")

    ctx = distribute_lib.InputContext()
    batch_size = ctx.get_per_replica_batch_size(8)
    # Use 20 which isn't divisible by 8 to test partial batch behavior.
    row_lengths = np.mod(np.arange(20), 4).astype(np.int64)
    ragged_tensor = ragged_tensor_lib.RaggedTensor.from_row_lengths(
        np.repeat(np.arange(20, dtype=np.float32), row_lengths), row_lengths)
    dataset = dataset_ops.DatasetV2.from_tensor_slices({
        "dense": ragged_tensor.to_tensor(),
        "ragged": ragged_tensor,
        "sparse": ragged_tensor.to_sparse(),
    })
    dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
    dataset = dataset.batch(batch_size)

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)

    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    with distribution.scope():
      iterator = iter(dist_dataset)
      _check_type_spec_structure(iterator)

    spec = iterator._type_spec

    tensor_list = spec._to_components(iterator)
    re_iterator = spec._from_components(tensor_list)

    self.assertEqual(iterator._input_workers, re_iterator._input_workers)
    self.assertAllEqual(iterator._iterators, re_iterator._iterators)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
          ],
          enable_get_next_as_optional=[True, False]))
  def testDoesNotTriggerFunctionTracing(self, distribution,
                                        enable_get_next_as_optional):
    if not tf2.enabled():
      self.skipTest("DistributedIterator CompositeTensor support is only "
                    "present in TF 2.0 only.")

    trace_count = [0]

    @def_function.function
    def f(iterator):
      trace_count[0] += 1
      counter = np.int64(0)
      for _ in range(5):
        next(iterator)
        counter += 1
      return counter

    ctx = distribute_lib.InputContext()
    batch_size = ctx.get_per_replica_batch_size(8)
    # Use 20 which isn't divisible by 8 to test partial batch behavior.
    row_lengths = np.mod(np.arange(50), 4).astype(np.int64)
    ragged_tensor = ragged_tensor_lib.RaggedTensor.from_row_lengths(
        np.repeat(np.arange(50, dtype=np.float32), row_lengths), row_lengths)
    dataset = dataset_ops.DatasetV2.from_tensor_slices({
        "dense": ragged_tensor.to_tensor(),
        "ragged": ragged_tensor,
        "sparse": ragged_tensor.to_sparse(),
    })
    dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
    dataset = dataset.batch(batch_size)

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)

    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    with distribution.scope():
      for _ in range(3):
        iterator = iter(dist_dataset)
        _check_type_spec_structure(iterator)
        counter = f(iterator)

        self.assertEqual(trace_count[0], 1)
        self.assertEqual(counter, 5)


def _check_type_spec_structure(x):
  """Verifies that `x` has the same structure as its `TypeSpec`."""
  if isinstance(x, composite_tensor.CompositeTensor):
    nest.assert_same_structure(x, x._type_spec, expand_composites=True)


if __name__ == "__main__":
  test.main()
