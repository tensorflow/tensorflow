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

import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import values
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
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
          enable_get_next_as_optional=[True, False],
          drop_remainder=[True, False],
          tf_api_version=2,
      ))
  def testDoesNotTriggerFunctionTracing(self, input_type, distribution,
                                        enable_get_next_as_optional,
                                        drop_remainder):
    trace_count = [0]

    @def_function.function
    def f(iterator):
      trace_count[0] += 1
      counter = np.int64(0)
      for _ in range(5):
        next(iterator)
        counter += 1
      return counter

    dataset = dataset_ops.DatasetV2.range(10).batch(
        2, drop_remainder=drop_remainder)

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
              strategy_combinations.central_storage_strategy_with_gpu_and_cpu,
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
              strategy_combinations.multi_worker_mirrored_2x1_gpu,
              strategy_combinations.multi_worker_mirrored_2x2_gpu,
          ],
          tf_api_version=2,
          enable_get_next_as_optional=[True, False],
          drop_remainder=[True, False],
      ))
  def testInputSignatureForPerReplicaValues(self, distribution,
                                            enable_get_next_as_optional,
                                            drop_remainder):
    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)
    ds = dataset_ops.DatasetV2.from_tensor_slices(
        np.ones([9, 12]).astype(np.float32)).batch(
            4, drop_remainder=drop_remainder)
    ds = distribution.experimental_distribute_dataset(ds)
    _check_type_spec_structure(iter(ds))
    element_spec = ds.element_spec
    iter_element_spec = iter(ds).element_spec
    nest.assert_same_structure(element_spec, iter_element_spec)
    self.assertAllEqual(
        nest.flatten(element_spec), nest.flatten(iter_element_spec))

    @def_function.function(input_signature=[element_spec])
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

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          tf_api_version=2,
          distribution=[
              strategy_combinations.mirrored_strategy_with_two_gpus,
              strategy_combinations.mirrored_strategy_with_cpu_1_and_2,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          enable_get_next_as_optional=[True, False],
          experimental_place_dataset_on_device=[True, False],
          experimental_fetch_to_device=[True, False],
      ))
  def testFromFunctionInputSignatureForPerReplicaValuesWithOptions(
      self, distribution, enable_get_next_as_optional,
      experimental_place_dataset_on_device, experimental_fetch_to_device):

    if experimental_place_dataset_on_device and experimental_fetch_to_device:
      self.skipTest("Setting experimental_place_dataset_on_device and "
                    "experimental_fetch_to_device to `True` is not "
                    "allowed when using "
                    "distribute_lib.InputReplicationMode.PER_REPLICA.")

    fname1 = os.path.join(self.get_temp_dir(), "1.txt")
    _create_text_file(fname1, 5)
    fname2 = os.path.join(self.get_temp_dir(), "2.txt")
    _create_text_file(fname2, 9)

    def dataset_fn(input_context):
      dataset = dataset_ops.DatasetV2.from_tensor_slices([fname1, fname2])
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)
      return readers.TextLineDatasetV2(dataset).map(
          string_ops.string_to_number).batch(
              input_context.get_per_replica_batch_size(4))

    options = distribute_lib.InputOptions(
        experimental_place_dataset_on_device=(
            experimental_place_dataset_on_device),
        experimental_fetch_to_device=experimental_fetch_to_device,
        experimental_replication_mode=(
            distribute_lib.InputReplicationMode.PER_REPLICA))

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)
    ds = distribution.experimental_distribute_datasets_from_function(
        dataset_fn, options)

    iterator = iter(ds)
    _check_type_spec_structure(iterator)
    spec = iterator._type_spec
    tensor_list = spec._to_components(iterator)
    re_iterator = spec._from_components(tensor_list)

    _check_type_spec_structure(iter(ds))
    element_spec = ds.element_spec
    iter_element_spec = iter(ds).element_spec
    nest.assert_same_structure(element_spec, iter_element_spec)
    self.assertAllEqual(
        nest.flatten(element_spec), nest.flatten(iter_element_spec))
    self.assertEqual(iterator._input_workers, re_iterator._input_workers)
    self.assertAllEqual(iterator._iterators, re_iterator._iterators)

    @def_function.function(input_signature=[element_spec])
    def process_inputs(inputs):
      distribution.run(lambda inputs: inputs, args=(inputs,))

    for x in ds:
      process_inputs(x)


class DistributedDatasetTypeSpecTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          tf_api_version=2,
          distribution=[
              strategy_combinations.mirrored_strategy_with_two_gpus,
              strategy_combinations.mirrored_strategy_with_cpu_1_and_2,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          enable_get_next_as_optional=[True, False]))
  def testTypeSpecBase(self, distribution, enable_get_next_as_optional):

    def create_dataset():
      dataset = dataset_ops.DatasetV2.range(10).batch(2)
      return dataset

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)

    dist_dataset = distribution.experimental_distribute_dataset(
        create_dataset())

    spec = dist_dataset._type_spec
    self.assertEqual(spec._input_workers, dist_dataset._input_workers)
    self.assertEqual(
        spec._element_spec._value_specs,
        (tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.int64, name=None),
         tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.int64, name=None)))

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          tf_api_version=2,
          distribution=[
              strategy_combinations.mirrored_strategy_with_cpu_1_and_2,
          ],
          enable_get_next_as_optional=[True, False]))
  def testTypeSpecReturnedFromTFFunction(self, distribution,
                                         enable_get_next_as_optional):
    # TODO(ishark): This is observed when tensor is copied from one device to
    # other and since DatasetVariantWrapper does not have a copy
    # function. Some Context: b/146981184
    # Try to renable with non-canonicalized input workers, which
    # helped in PS Strategy for similar error.
    self.skipTest("Failures observed in Ubuntu presubmit: No unary variant  "
                  "device copy function found for direction: 1 and Variant "
                  "type_index:tensorflow::data::(anonymous namespace)::"
                  "DatasetVariantWrapper")

    @def_function.function
    def create_dist_dataset():
      dataset = dataset_ops.DatasetV2.range(10).batch(2)
      return distribution.experimental_distribute_dataset(dataset)

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)

    dist_dataset = create_dist_dataset()

    spec = dist_dataset._type_spec
    self.assertEqual(spec._input_workers, dist_dataset._input_workers)
    self.assertEqual(
        spec._element_spec._value_specs,
        (tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.int64, name=None),
         tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.int64, name=None)))

    # Read distributed data to confirm values are correct.
    iterator = iter(dist_dataset)
    data = []
    for it in iterator:
      data.append(distribution.experimental_local_results(it))
    self.assertAllEqual(
        nest.flatten(data),
        list(dataset_ops.DatasetV2.range(10).batch(1).as_numpy_iterator()))

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          tf_api_version=2,
          distribution=[
              strategy_combinations.mirrored_strategy_with_two_gpus,
              strategy_combinations.mirrored_strategy_with_cpu_1_and_2,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          enable_get_next_as_optional=[True, False]))
  def testTypeSpecRaggedTensor(self, distribution, enable_get_next_as_optional):
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
    spec = dist_dataset._type_spec
    self.assertEqual(spec._input_workers, dist_dataset._input_workers)
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
          tf_api_version=2,
          distribution=[
              strategy_combinations.mirrored_strategy_with_two_gpus,
              strategy_combinations.mirrored_strategy_with_cpu_1_and_2,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          enable_get_next_as_optional=[True, False],
          experimental_place_dataset_on_device=[True, False],
          experimental_fetch_to_device=[True, False]))
  def testTypeSpecComponents(self, distribution, enable_get_next_as_optional,
                             experimental_place_dataset_on_device,
                             experimental_fetch_to_device):
    dataset = dataset_ops.DatasetV2.range(10).batch(2)
    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)

    options = distribute_lib.InputOptions(
        experimental_place_dataset_on_device=
        experimental_place_dataset_on_device,
        experimental_fetch_to_device=experimental_fetch_to_device)

    dist_dataset = distribution.experimental_distribute_dataset(
        dataset, options)

    spec = dist_dataset._type_spec
    self.assertEqual(spec._input_workers, dist_dataset._input_workers)
    self.assertEqual(
        spec._element_spec._value_specs,
        (tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.int64, name=None),
         tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.int64, name=None)))
    components = spec._to_components(dist_dataset)
    re_dist_dataset = spec._from_components(components)

    self.assertEqual(dist_dataset._input_workers,
                     re_dist_dataset._input_workers)
    self.assertAllEqual(dist_dataset._cloned_datasets,
                        re_dist_dataset._cloned_datasets)
    self.assertEqual(dist_dataset._element_spec, re_dist_dataset._element_spec)
    self.assertEqual(dist_dataset._enable_get_next_as_optional,
                     re_dist_dataset._enable_get_next_as_optional)
    self.assertEqual(dist_dataset._options, re_dist_dataset._options)


class DistributedDatasetsFromFunctionSpecTest(test.TestCase,
                                              parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          tf_api_version=2,
          distribution=[
              strategy_combinations.mirrored_strategy_with_two_gpus,
              strategy_combinations.mirrored_strategy_with_cpu_1_and_2,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          enable_get_next_as_optional=[True, False],
          experimental_place_dataset_on_device=[True, False],
          experimental_fetch_to_device=[True, False],
      ))
  def testDistributedDatasetsFromFunctionSpec(
      self, distribution, enable_get_next_as_optional,
      experimental_place_dataset_on_device, experimental_fetch_to_device):

    if experimental_place_dataset_on_device and experimental_fetch_to_device:
      self.skipTest("Setting experimental_place_dataset_on_device and "
                    "experimental_fetch_to_device to `True` is not "
                    "allowed when using "
                    "distribute_lib.InputReplicationMode.PER_REPLICA.")

    fname1 = os.path.join(self.get_temp_dir(), "1.txt")
    _create_text_file(fname1, 5)
    fname2 = os.path.join(self.get_temp_dir(), "2.txt")
    _create_text_file(fname2, 9)

    def dataset_fn(input_context):
      dataset = dataset_ops.DatasetV2.from_tensor_slices([fname1, fname2])
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)
      return readers.TextLineDatasetV2(dataset).map(
          string_ops.string_to_number).batch(
              input_context.get_per_replica_batch_size(4))

    options = distribute_lib.InputOptions(
        experimental_place_dataset_on_device=
        experimental_place_dataset_on_device,
        experimental_fetch_to_device=experimental_fetch_to_device,
        experimental_replication_mode=(
            distribute_lib.InputReplicationMode.PER_REPLICA))

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)
    ds = distribution.experimental_distribute_datasets_from_function(
        dataset_fn, options)

    spec = ds._type_spec
    components = spec._to_components(ds)
    re_ds = spec._from_components(components)

    element_spec = re_ds.element_spec
    iter_element_spec = iter(ds).element_spec
    nest.assert_same_structure(element_spec, iter_element_spec)
    self.assertAllEqual(
        nest.flatten(element_spec), nest.flatten(iter_element_spec))
    self.assertEqual(ds._input_workers, re_ds._input_workers)
    self.assertEqual(ds._element_spec, re_ds._element_spec)

    @def_function.function(input_signature=[element_spec])
    def process_inputs(inputs):
      distribution.run(lambda inputs: inputs, args=(inputs,))

    for x in ds:
      process_inputs(x)


class RaggedTensorDistributedIteratorTest(test.TestCase,
                                          parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          tf_api_version=2,
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.central_storage_strategy_with_gpu_and_cpu,
              strategy_combinations.multi_worker_mirrored_2x2_gpu,
          ],
          enable_get_next_as_optional=[True, False]))
  def testTypeSpec(self, distribution, enable_get_next_as_optional):
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
          tf_api_version=2,
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.central_storage_strategy_with_gpu_and_cpu,
              strategy_combinations.multi_worker_mirrored_2x2_gpu,
          ],

          enable_get_next_as_optional=[True, False]))
  def testTypeSpecRoundTrip(self, distribution, enable_get_next_as_optional):
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

    if isinstance(distribution,
                  (tpu_strategy.TPUStrategyV2, tpu_strategy.TPUStrategy)):
      # TPUStrategy does not support distributed datasets with device prefetch
      # when using sparse or ragged tensors.
      options = distribute_lib.InputOptions(experimental_fetch_to_device=False)
    else:
      options = None

    dist_dataset = distribution.experimental_distribute_dataset(
        dataset, options)
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
          tf_api_version=2,
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
          ],
          enable_get_next_as_optional=[True, False]))
  def testDoesNotTriggerFunctionTracing(self, distribution,
                                        enable_get_next_as_optional):
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

    if isinstance(distribution,
                  (tpu_strategy.TPUStrategyV2, tpu_strategy.TPUStrategy)):
      # TPUStrategy does not support distributed datasets with device prefetch
      # when using sparse or ragged tensors.
      options = distribute_lib.InputOptions(experimental_fetch_to_device=False)
    else:
      options = None

    dist_dataset = distribution.experimental_distribute_dataset(
        dataset, options)
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


def _create_text_file(fname, num_lines):
  with open(fname, "w") as f:
    for i in range(num_lines):
      f.write("%d\n" % i)


if __name__ == "__main__":
  test_util.main()
