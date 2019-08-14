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
"""Tests for `tf.data.Dataset`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


@test_util.run_all_in_graph_and_eager_modes
class DatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

  def testAsSerializedGraph(self):
    dataset = dataset_ops.Dataset.range(10)
    graph = graph_pb2.GraphDef().FromString(
        self.evaluate(dataset._as_serialized_graph()))
    self.assertTrue(any([node.op == "RangeDataset" for node in graph.node]))

  def testAsSerializedGraphStateful(self):
    dataset = dataset_ops.Dataset.range(10).map(
        lambda _: random_ops.random_uniform(()))
    with self.assertRaises(errors.FailedPreconditionError):
      self.evaluate(dataset._as_serialized_graph())

  def testAsFunctionWithMap(self):
    if not context.executing_eagerly():
      self.skipTest("Only works executing eagerly")
    with ops.device("CPU"):
      original_dataset = dataset_ops.Dataset.range(5).map(lambda x: x * 2)
      fn = original_dataset._trace_variant_creation()
      variant = fn()

      revived_dataset = dataset_ops._VariantDataset(
          variant, original_dataset.element_spec)
      self.assertDatasetProduces(revived_dataset, range(0, 10, 2))

  def testAsFunctionWithMapInFlatMap(self):
    if not context.executing_eagerly():
      self.skipTest("Only works executing eagerly")
    with ops.device("CPU"):
      original_dataset = dataset_ops.Dataset.range(5).flat_map(
          lambda x: dataset_ops.Dataset.range(5).map(lambda x: x * 2))
      fn = original_dataset._trace_variant_creation()
      variant = fn()

      revived_dataset = dataset_ops._VariantDataset(
          variant, original_dataset.element_spec)
      self.assertDatasetProduces(revived_dataset, list(original_dataset))

  @staticmethod
  def make_apply_fn(dataset):

    def apply_fn(dataset):

      def _apply_fn(dataset):
        return dataset.cache()

      return dataset.apply(_apply_fn)

    return apply_fn

  @staticmethod
  def make_gen():

    def gen():
      yield 42

    return gen

  @staticmethod
  def make_interleave_fn(dataset, num_parallel_calls=None):

    def interleave_fn(dataset):
      return dataset.interleave(
          lambda x: dataset_ops.Dataset.range(0),
          cycle_length=2,
          num_parallel_calls=num_parallel_calls)

    return interleave_fn

  @parameterized.named_parameters(
      ("FixedLengthRecord",
       lambda: readers.FixedLengthRecordDataset("", 42)),
      ("FromGenerator",
       lambda: dataset_ops.Dataset.from_generator(
           DatasetTest.make_gen(), dtypes.int32),
       1),
      ("FromTensors", lambda: dataset_ops.Dataset.from_tensors([42])),
      ("FromTensorSlices", lambda: dataset_ops.Dataset.from_tensors([42])),
      ("Range", lambda: dataset_ops.Dataset.range(10)),
      ("TextLine", lambda: readers.TextLineDataset("")),
      ("TFRecord", lambda: readers.TFRecordDataset(""), 1),
  )
  def testDatasetSimpleSourceInputs(self, dataset_fn, num_inputs=0):
    self.assertLen(dataset_fn()._inputs(), num_inputs)

  @test_util.run_v1_only("deprecated API, no eager or V2 test coverage")
  def testDatasetComplexSourceInputs(self):
    dataset_fn = dataset_ops.Dataset.from_sparse_tensor_slices(
        sparse_tensor.SparseTensor(
            indices=np.array([[0, 0], [1, 0], [2, 0]]),
            values=np.array([0, 0, 0]),
            dense_shape=np.array([3, 1])))
    self.assertEmpty(dataset_fn._inputs())

  @parameterized.named_parameters(
      ("Batch",
       lambda x: x.batch(10),
       lambda: dataset_ops.Dataset.range(0)),
      ("Cache",
       lambda x: x.cache(),
       lambda: dataset_ops.Dataset.range(0)),
      ("Filter",
       lambda x: x.filter(lambda x: True),
       lambda: dataset_ops.Dataset.range(0)),
      ("FlatMap",
       lambda x: x.flat_map(lambda x: dataset_ops.Dataset.range(0)),
       lambda: dataset_ops.Dataset.range(0)),
      ("Map",
       lambda x: x.map(lambda x: x),
       lambda: dataset_ops.Dataset.range(0)),
      ("PaddedBatch",
       lambda x: x.padded_batch(10, []),
       lambda: dataset_ops.Dataset.range(0)),
      ("ParallelMap",
       lambda x: x.map(lambda x: x, num_parallel_calls=2),
       lambda: dataset_ops.Dataset.range(0)),
      ("Repeat",
       lambda x: x.repeat(),
       lambda: dataset_ops.Dataset.range(0)),
      ("Shuffle",
       lambda x: x.shuffle(10),
       lambda: dataset_ops.Dataset.range(0)),
      ("Skip",
       lambda x: x.skip(1),
       lambda: dataset_ops.Dataset.range(0)),
      ("Take",
       lambda x: x.take(1),
       lambda: dataset_ops.Dataset.range(0)),
      ("Window",
       lambda x: x.window(10),
       lambda: dataset_ops.Dataset.range(0)),
  )
  def testUnaryTransformationInputs(self, dataset_fn, input_dataset_fn):
    input_dataset = input_dataset_fn()
    self.assertEqual([input_dataset], dataset_fn(input_dataset)._inputs())

  def testUnaryTransformationInputsApply(self):
    input_dataset = dataset_ops.Dataset.range(0)
    dataset_fn = self.make_apply_fn(dataset_ops.Dataset.range(0))
    self.assertEqual([input_dataset], dataset_fn(input_dataset)._inputs())

  @parameterized.named_parameters(
      ("ParallelInterleave",
       [lambda: dataset_ops.Dataset.range(0), 2],
       lambda: dataset_ops.Dataset.range(0)),
      ("Interleave",
       [lambda: dataset_ops.Dataset.range(0), None],
       lambda: dataset_ops.Dataset.range(0)),
  )
  def testUnaryTransformationInputsWithInterleaveFn(
      self, interleave_fn_args, input_dataset_fn):
    input_dataset = input_dataset_fn()
    dataset_fn = self.make_interleave_fn(*interleave_fn_args)
    self.assertEqual([input_dataset], dataset_fn(input_dataset)._inputs())

  def testNoWarnings(self):
    with test.mock.patch.object(warnings, "warn") as mock_log:
      dataset_fn = self.make_interleave_fn(dataset_ops.Dataset.range(10))
      dataset_fn(dataset_ops.Dataset.range(10))
      self.assertEmpty(mock_log.call_args_list)

  @parameterized.named_parameters(
      ("Concatenate", lambda x, y: x.concatenate(y),
       lambda: dataset_ops.Dataset.range(0),
       lambda: dataset_ops.Dataset.range(1)))
  def testBinaryTransformationInputs(self, dataset_fn, input1_fn, input2_fn):
    input1 = input1_fn()
    input2 = input2_fn()
    self.assertEqual([input1, input2], dataset_fn(input1, input2)._inputs())

  @parameterized.named_parameters(
      ("ZipOne",
       dataset_ops.Dataset.zip,
       lambda: (dataset_ops.Dataset.range(0))),
      ("ZipNest",
       dataset_ops.Dataset.zip,
       lambda: (dataset_ops.Dataset.range(0),
                (dataset_ops.Dataset.range(1),
                 dataset_ops.Dataset.range(2)))),
      ("ZipTuple",
       dataset_ops.Dataset.zip,
       lambda: (dataset_ops.Dataset.range(0),
                dataset_ops.Dataset.range(1))),
  )
  def testVariadicTransformationInputs(self, dataset_fn, input_datasets_fn):
    input_datasets = input_datasets_fn()
    self.assertEqual(
        nest.flatten(input_datasets),
        dataset_fn(input_datasets)._inputs())

  def testFunctions(self):
    dataset = dataset_ops.Dataset.range(5).map(lambda x: x * 2)
    self.assertLen(dataset._functions(), 1)

  def testCollectInputs(self):
    ds1 = dataset_ops.Dataset.range(0)
    ds2 = ds1.concatenate(ds1)
    ds3 = dataset_ops.Dataset.zip((ds2, ds1, ds2))

    inputs = []
    queue = [ds3]
    while queue:
      ds = queue[0]
      queue = queue[1:]
      queue.extend(ds._inputs())
      inputs.append(ds)

    self.assertEqual(5, inputs.count(ds1))
    self.assertEqual(2, inputs.count(ds2))
    self.assertEqual(1, inputs.count(ds3))

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters(
      ("Tensor", lambda: constant_op.constant(37.0),
       tensor_spec.TensorSpec([], dtypes.float32)),
      ("SparseTensor", lambda: sparse_tensor.SparseTensor(
          indices=[[0]],
          values=constant_op.constant([0], dtype=dtypes.int32),
          dense_shape=[1]), sparse_tensor.SparseTensorSpec([1], dtypes.int32)),
      ("Nest", lambda: {
          "a": constant_op.constant(37.0),
          "b": (constant_op.constant(["Foo"]), constant_op.constant("Bar"))
      }, {
          "a":
              tensor_spec.TensorSpec([], dtypes.float32),
          "b": (
              tensor_spec.TensorSpec([1], dtypes.string),
              tensor_spec.TensorSpec([], dtypes.string),
          )
      }),
      ("Dataset", lambda: dataset_ops.Dataset.from_tensor_slices(
          constant_op.constant([1, 2, 3])),
       dataset_ops.DatasetSpec(tensor_spec.TensorSpec([], dtypes.int32))),
      ("Optional", lambda: optional_ops.Optional.from_value(37.0),
       optional_ops.OptionalSpec(
           tensor_spec.TensorSpec([], dtypes.float32))),
  )
  def testDatasetSpec(self, tf_value_fn, expected_element_structure):
    dataset = dataset_ops.Dataset.from_tensors(0).map(lambda _: tf_value_fn())
    dataset_structure = structure.type_spec_from_value(dataset)
    self.assertIsInstance(dataset_structure, dataset_ops.DatasetSpec)

    self.assertTrue(
        structure.are_compatible(
            dataset_ops.get_structure(dataset), expected_element_structure))
    self.assertEqual([dtypes.variant],
                     structure.get_flat_tensor_types(dataset_structure))
    self.assertEqual([tensor_shape.TensorShape([])],
                     structure.get_flat_tensor_shapes(dataset_structure))

    # Assert that the `Dataset` survives a round-trip via _from_tensor_list()
    # and _to_tensor_list().
    round_trip_dataset = dataset_structure._from_tensor_list(
        dataset_structure._to_tensor_list(dataset))

    value = tf_value_fn()

    if isinstance(value, dataset_ops.Dataset):
      self.assertDatasetsEqual(value, dataset.flat_map(lambda x: x))
    elif isinstance(value, optional_ops.Optional):
      self.assertDatasetProduces(
          round_trip_dataset.map(lambda opt: opt.get_value()),
          [self.evaluate(value.get_value())],
          requires_initialization=True)
    else:
      self.assertDatasetProduces(
          round_trip_dataset, [self.evaluate(tf_value_fn())],
          requires_initialization=True)

  @test_util.run_v1_only("graph mode specific, no eager or V2 test coverage")
  def testSkipEagerSameGraphErrorOneShot(self):
    dataset = dataset_ops.Dataset.range(10)
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(ValueError, "must be from the same graph"):
        dataset = dataset.batch(2)

  @test_util.run_v1_only("graph mode specific, no eager or V2 test coverage")
  def testSkipEagerSameGraphErrorOneShotSimple(self):
    dataset = dataset_ops.Dataset.range(10)
    with ops.Graph().as_default():
      with test.mock.patch.object(tf_logging, "warning") as mock_log:
        _ = dataset_ops.make_one_shot_iterator(dataset)
        self.assertRegexpMatches(
            str(mock_log.call_args), "Please ensure that all datasets in the "
            "pipeline are created in the same graph as the iterator.")

  @test_util.run_v1_only("graph mode specific, no eager or V2 test coverage")
  def testSkipEagerSameGraphErrorInitializable(self):
    dataset = dataset_ops.Dataset.range(10)
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(ValueError, "must be from the same graph"):
        dataset = dataset.batch(2)

  @parameterized.named_parameters(
      ("Async", context.ASYNC),
      ("Sync", context.SYNC),
  )
  def testDatasetEagerIteration(self, execution_mode):
    with context.eager_mode(), context.execution_mode(execution_mode):
      val = 0
      dataset = dataset_ops.Dataset.range(10)
      for foo in dataset:
        self.assertEqual(val, foo.numpy())
        val += 1

  def testDatasetAsFunctionArgument(self):

    @def_function.function
    def _uses_dataset(d):
      accumulator = array_ops.zeros([], dtype=dtypes.int64)
      for value in d:
        accumulator += value
      return accumulator

    with ops.device("CPU"):
      first_dataset = dataset_ops.Dataset.range(10)
      self.assertEqual(45, self.evaluate(_uses_dataset(first_dataset)))
      second_dataset = dataset_ops.Dataset.range(11)
      self.assertEqual(55, self.evaluate(_uses_dataset(second_dataset)))
      first_concrete = _uses_dataset.get_concrete_function(first_dataset)
      # The dataset should not be a captured input
      self.assertEmpty(first_concrete.graph.captures)
      # The two datasets have the same structure and so should re-use a trace.
      self.assertIs(first_concrete,
                    _uses_dataset.get_concrete_function(second_dataset))
      # With a different structure we should use a different trace.
      self.assertIsNot(
          first_concrete,
          _uses_dataset.get_concrete_function(
              dataset_ops.Dataset.zip((first_dataset, second_dataset))))

  def testLimitedRetracing(self):
    trace_count = [0]

    @def_function.function
    def f(ds):
      trace_count[0] += 1
      counter = np.int64(0)
      for elem in ds:
        counter += elem
      return counter

    dataset = dataset_ops.Dataset.range(5)
    dataset2 = dataset_ops.Dataset.range(10)

    for _ in range(10):
      self.assertEqual(self.evaluate(f(dataset)), 10)
      self.assertEqual(self.evaluate(f(dataset2)), 45)
      self.assertEqual(trace_count[0], 1)


if __name__ == "__main__":
  test.main()
