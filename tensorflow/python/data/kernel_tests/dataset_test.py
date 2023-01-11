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

import collections
import os
import warnings

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import readers
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class DatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testAsSerializedGraph(self):
    dataset = dataset_ops.Dataset.range(10)
    graph = graph_pb2.GraphDef().FromString(
        self.evaluate(dataset._as_serialized_graph()))
    self.assertTrue(any(node.op == "RangeDataset" for node in graph.node))

  def testAsSerializedGraphStateful(self):
    dataset = dataset_ops.Dataset.range(10).map(
        lambda _: random_ops.random_uniform(()))
    with self.assertRaises(errors.FailedPreconditionError):
      self.evaluate(
          dataset._as_serialized_graph(external_state_policy=options_lib
                                       .ExternalStatePolicy.FAIL))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              init_source=["textfile", "keyvaluetensor", "dataset"])))
  def testLookupTableGraphSerialization(self, init_source):
    vals = [10, 11]
    initializer = self.lookupTableInitializer(init_source, vals)
    table = lookup_ops.StaticHashTable(initializer, -1)
    dataset = dataset_ops.Dataset.range(3)
    dataset = dataset.map(table.lookup)
    self.evaluate(lookup_ops.tables_initializer())
    round_tripped = self.graphRoundTrip(dataset)
    del table
    del dataset
    self.assertDatasetProduces(
        round_tripped, [10, 11, -1], requires_initialization=True)

  @combinations.generate(test_base.eager_only_combinations())
  def testAsFunctionWithMap(self):
    with ops.device("CPU"):
      original_dataset = dataset_ops.Dataset.range(5).map(lambda x: x * 2)
      fn = original_dataset._trace_variant_creation()
      variant = fn()

      revived_dataset = dataset_ops._VariantDataset(
          variant, original_dataset.element_spec)
      self.assertDatasetProduces(revived_dataset, range(0, 10, 2))

  @combinations.generate(test_base.eager_only_combinations())
  def testAsFunctionWithMapInFlatMap(self):
    with ops.device("CPU"):
      original_dataset = dataset_ops.Dataset.range(5).flat_map(
          lambda x: dataset_ops.Dataset.range(5).map(lambda x: x * 2))
      fn = original_dataset._trace_variant_creation()
      variant = fn()

      revived_dataset = dataset_ops._VariantDataset(
          variant, original_dataset.element_spec)
      self.assertDatasetProduces(revived_dataset, list(original_dataset))

  @combinations.generate(test_base.eager_only_combinations())
  def testAsFunctionFromReader(self):
    with ops.device("CPU"):
      file_path = os.path.join(self.get_temp_dir(),
                               "{}.tfrecord.gz".format("tf_record_asset"))
      with tf_record.TFRecordWriter(file_path, "GZIP") as f:
        for v in ["a", "aa", "aaa"]:
          f.write(str(v))
      original_dataset = readers.TFRecordDataset([file_path],
                                                 compression_type="GZIP")
      fn = original_dataset._trace_variant_creation()
      variant = fn()

      revived_dataset = dataset_ops._VariantDataset(
          variant, original_dataset.element_spec)
      self.assertDatasetProduces(revived_dataset, ["a", "aa", "aaa"])

  def _testNumInputs(self, dataset, num_inputs):
    self.assertLen(dataset._inputs(), num_inputs)

  @combinations.generate(test_base.default_test_combinations())
  def testFixedLengthRecordInputs(self):
    dataset = readers.FixedLengthRecordDataset("", 42)
    self._testNumInputs(dataset, 0)

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorInputs(self):
    def gen():
      yield 42

    dataset = dataset_ops.Dataset.from_generator(gen, dtypes.int32)
    self._testNumInputs(dataset, 1)

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorsInputs(self):
    dataset = dataset_ops.Dataset.from_tensors([42])
    self._testNumInputs(dataset, 0)

  @combinations.generate(test_base.default_test_combinations())
  def testRangeInputs(self):
    dataset = dataset_ops.Dataset.range(10)
    self._testNumInputs(dataset, 0)

  @combinations.generate(test_base.default_test_combinations())
  def testTextLineInputs(self):
    dataset = readers.TextLineDataset("")
    self._testNumInputs(dataset, 0)

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordInputs(self):
    dataset = readers.TFRecordDataset("")
    self._testNumInputs(dataset, 1)

  @combinations.generate(
      combinations.combine(tf_api_version=1, mode=["eager", "graph"]))
  def testDatasetComplexSourceInputs(self):
    dataset_fn = dataset_ops.Dataset.from_sparse_tensor_slices(
        sparse_tensor.SparseTensor(
            indices=np.array([[0, 0], [1, 0], [2, 0]]),
            values=np.array([0, 0, 0]),
            dense_shape=np.array([3, 1])))
    self.assertEmpty(dataset_fn._inputs())

  def _testUnaryInputs(self, dataset_fn):
    input_dataset = dataset_ops.Dataset.range(0)
    self.assertEqual([input_dataset], dataset_fn(input_dataset)._inputs())

  @combinations.generate(test_base.default_test_combinations())
  def testBatchInputs(self):
    self._testUnaryInputs(lambda x: x.batch(10))

  @combinations.generate(test_base.default_test_combinations())
  def testCacheInputs(self):
    self._testUnaryInputs(lambda x: x.cache())

  @combinations.generate(test_base.default_test_combinations())
  def testFilterInputs(self):
    self._testUnaryInputs(lambda x: x.filter(lambda x: True))

  @combinations.generate(test_base.default_test_combinations())
  def testFlatMapInputs(self):
    self._testUnaryInputs(
        lambda x: x.flat_map(lambda x: dataset_ops.Dataset.range(0)))

  @combinations.generate(test_base.default_test_combinations())
  def testMapInputs(self):
    self._testUnaryInputs(lambda x: x.map(lambda x: x))

  @combinations.generate(test_base.default_test_combinations())
  def testPaddedBatchInputs(self):
    self._testUnaryInputs(lambda x: x.padded_batch(10, []))

  @combinations.generate(test_base.default_test_combinations())
  def testParallelMapInputs(self):
    self._testUnaryInputs(lambda x: x.map(lambda x: x, num_parallel_calls=2))

  @combinations.generate(test_base.default_test_combinations())
  def testRepeatInputs(self):
    self._testUnaryInputs(lambda x: x.repeat())

  @combinations.generate(test_base.default_test_combinations())
  def testShuffleInputs(self):
    self._testUnaryInputs(lambda x: x.shuffle(10))

  @combinations.generate(test_base.default_test_combinations())
  def testSkipInputs(self):
    self._testUnaryInputs(lambda x: x.skip(1))

  @combinations.generate(test_base.default_test_combinations())
  def testTakeInputs(self):
    self._testUnaryInputs(lambda x: x.take(1))

  @combinations.generate(test_base.default_test_combinations())
  def testWindowInputs(self):
    self._testUnaryInputs(lambda x: x.window(10))

  @combinations.generate(test_base.default_test_combinations())
  def testUnaryTransformationInputsApply(self):
    input_dataset = dataset_ops.Dataset.range(0)
    dataset = input_dataset.apply(lambda dataset: dataset.cache())

    self.assertEqual([input_dataset], dataset._inputs())

  def _testInputsWithInterleaveFn(self, dataset_fn, interleave_parallelism):
    input_dataset = dataset_ops.Dataset.range(0)
    dataset = input_dataset.interleave(
        lambda x: dataset_ops.Dataset.range(0),
        cycle_length=2,
        num_parallel_calls=interleave_parallelism)
    self.assertEqual([input_dataset], dataset._inputs())

  @combinations.generate(test_base.default_test_combinations())
  def testParallelInterleaveInputs(self):
    self._testInputsWithInterleaveFn(lambda: dataset_ops.range(0), 2)

  @combinations.generate(test_base.default_test_combinations())
  def testInterleaveInputs(self):
    self._testInputsWithInterleaveFn(lambda: dataset_ops.range(0), None)

  @combinations.generate(test_base.default_test_combinations())
  def testDebugString(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.map(lambda x: x**2)
    dataset = dataset.filter(lambda x: x > 10)
    debug_string = dataset.__debug_string__()
    for transformation in ["Range", "Map", "Filter"]:
      self.assertContainsSubsequence(debug_string, transformation)

  @combinations.generate(test_base.default_test_combinations())
  def testNoWarnings(self):
    with test.mock.patch.object(warnings, "warn") as mock_log:
      dataset_ops.Dataset.range(0).interleave(
          lambda x: dataset_ops.Dataset.range(0), cycle_length=2)
      self.assertEmpty(mock_log.call_args_list)

  def _testBinaryInputs(self, dataset_fn):
    input1 = dataset_ops.Dataset.range(0)
    input2 = dataset_ops.Dataset.range(1)
    self.assertEqual([input1, input2], dataset_fn(input1, input2)._inputs())

  @combinations.generate(test_base.default_test_combinations())
  def testConcatenateInputs(self):
    self._testBinaryInputs(lambda x, y: x.concatenate(y))

  def _testVariadicInputs(self, dataset_fn, input_datasets):
    self.assertEqual(
        nest.flatten(input_datasets),
        dataset_fn(input_datasets)._inputs())

  @combinations.generate(test_base.default_test_combinations())
  def testZipOneInputs(self):
    input_datasets = dataset_ops.Dataset.range(0)
    self._testVariadicInputs(dataset_ops.Dataset.zip, input_datasets)

  @combinations.generate(test_base.default_test_combinations())
  def testZipNestInputs(self):
    input_datasets = (dataset_ops.Dataset.range(0),
                      (dataset_ops.Dataset.range(1),
                       dataset_ops.Dataset.range(2)))
    self._testVariadicInputs(dataset_ops.Dataset.zip, input_datasets)

  @combinations.generate(test_base.default_test_combinations())
  def testZipTupleInputs(self):
    input_datasets = (dataset_ops.Dataset.range(0),
                      dataset_ops.Dataset.range(1))
    self._testVariadicInputs(dataset_ops.Dataset.zip, input_datasets)

  @combinations.generate(test_base.default_test_combinations())
  def testFunctions(self):
    dataset = dataset_ops.Dataset.range(5).map(lambda x: x * 2)
    self.assertLen(dataset._functions(), 1)

  @combinations.generate(test_base.default_test_combinations())
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

  def _testDatasetSpec(self, tf_value, expected_element_structure):
    dataset = dataset_ops.Dataset.from_tensors(0).map(lambda _: tf_value)
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

    value = tf_value

    if isinstance(value, dataset_ops.Dataset):
      self.assertDatasetsEqual(value, dataset.flat_map(lambda x: x))
    elif isinstance(value, optional_ops.Optional):
      self.assertDatasetProduces(
          round_trip_dataset.map(lambda opt: opt.get_value()),
          [self.evaluate(value.get_value())],
          requires_initialization=True)
    else:
      self.assertDatasetProduces(
          round_trip_dataset, [self.evaluate(tf_value)],
          requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def testTensorDatasetSpec(self):
    self._testDatasetSpec(
        constant_op.constant(37.0), tensor_spec.TensorSpec([], dtypes.float32))

  @combinations.generate(test_base.default_test_combinations())
  def testSparseTensorDatasetSpec(self):
    self._testDatasetSpec(
        sparse_tensor.SparseTensor(
            indices=[[0]],
            values=constant_op.constant([0], dtype=dtypes.int32),
            dense_shape=[1]), sparse_tensor.SparseTensorSpec([1], dtypes.int32))

  @combinations.generate(test_base.default_test_combinations())
  def testNestDatasetSpec(self):
    self._testDatasetSpec(
        {
            "a": constant_op.constant(37.0),
            "b": (constant_op.constant(["Foo"]), constant_op.constant("Bar"))
        }, {
            "a":
                tensor_spec.TensorSpec([], dtypes.float32),
            "b": (
                tensor_spec.TensorSpec([1], dtypes.string),
                tensor_spec.TensorSpec([], dtypes.string),
            )
        })

  @combinations.generate(test_base.default_test_combinations())
  def testDatasetDatasetSpec(self):
    self._testDatasetSpec(
        dataset_ops.Dataset.from_tensor_slices(
            constant_op.constant([1, 2, 3])),
        dataset_ops.DatasetSpec(tensor_spec.TensorSpec([], dtypes.int32)))

  @combinations.generate(test_base.default_test_combinations())
  def testOptionalDatasetSpec(self):
    self._testDatasetSpec(
        optional_ops.Optional.from_value(37.0),
        optional_ops.OptionalSpec(tensor_spec.TensorSpec([], dtypes.float32)))

  @combinations.generate(test_base.graph_only_combinations())
  def testSameGraphError(self):
    dataset = dataset_ops.Dataset.range(10)
    with ops.Graph().as_default():
      with self.assertRaisesRegex(ValueError, "must be from the same graph"):
        dataset = dataset.batch(2)

  @combinations.generate(
      combinations.combine(tf_api_version=[1], mode=["graph"]))
  def testSameGraphErrorOneShot(self):
    dataset = dataset_ops.Dataset.range(10)
    with ops.Graph().as_default():
      with self.assertRaisesRegex(ValueError,
                                  "make sure that the dataset is created in "
                                  "the same graph as the iterator"):
        _ = dataset_ops.make_one_shot_iterator(dataset)

  @combinations.generate(
      combinations.combine(tf_api_version=[1], mode=["graph"]))
  def testSameGraphErrorInitializable(self):
    dataset = dataset_ops.Dataset.range(10)
    with ops.Graph().as_default():
      with self.assertRaisesRegex(ValueError,
                                  "make sure that the dataset is created in "
                                  "the same graph as the iterator"):
        _ = dataset_ops.make_initializable_iterator(dataset)

  @combinations.generate(
      combinations.times(
          test_base.eager_only_combinations(),
          combinations.combine(execution_mode=[context.ASYNC, context.SYNC])))
  def testEagerIteration(self, execution_mode):
    with context.execution_mode(execution_mode):
      val = 0
      dataset = dataset_ops.Dataset.range(10)
      for foo in dataset:
        self.assertEqual(val, foo.numpy())
        val += 1

  @combinations.generate(test_base.default_test_combinations())
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

  @combinations.generate(test_base.default_test_combinations())
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

  # pylint: disable=g-long-lambda,unnecessary-lambda
  @combinations.generate(test_base.default_test_combinations())
  def testLegacyStructureAPI(self):
    components = (np.array([1, 2, 3], dtype=np.int64), (np.array([4., 5.]),
                                                        np.array([6., 7.])),
                  np.array([8, 9, 10], dtype=np.int64))

    dataset = dataset_ops.Dataset.from_tensors(components)
    self.assertEqual(
        (dtypes.int64, (dtypes.float64, dtypes.float64), dtypes.int64),
        dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual(([3], ([2], [2]), [3]),
                     dataset_ops.get_legacy_output_shapes(dataset))

    dataset = dataset.shuffle(10, 10)
    self.assertEqual(
        (dtypes.int64, (dtypes.float64, dtypes.float64), dtypes.int64),
        dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual(([3], ([2], [2]), [3]),
                     dataset_ops.get_legacy_output_shapes(dataset))

    dataset = dataset.repeat(-1)
    self.assertEqual(
        (dtypes.int64, (dtypes.float64, dtypes.float64), dtypes.int64),
        dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual(([3], ([2], [2]), [3]),
                     dataset_ops.get_legacy_output_shapes(dataset))

    dataset = dataset.filter(lambda x, y, z: True)
    self.assertEqual(
        (dtypes.int64, (dtypes.float64, dtypes.float64), dtypes.int64),
        dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual(([3], ([2], [2]), [3]),
                     dataset_ops.get_legacy_output_shapes(dataset))

    dataset = dataset.take(5)
    self.assertEqual(
        (dtypes.int64, (dtypes.float64, dtypes.float64), dtypes.int64),
        dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual(([3], ([2], [2]), [3]),
                     dataset_ops.get_legacy_output_shapes(dataset))

    dataset = dataset.map(lambda x, y, z: ((x, z), (y[0], y[1])))
    self.assertEqual(
        ((dtypes.int64, dtypes.int64), (dtypes.float64, dtypes.float64)),
        dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual((([3], [3]), ([2], [2])),
                     dataset_ops.get_legacy_output_shapes(dataset))

    dataset = dataset.flat_map(lambda x, y: dataset_ops.Dataset.from_tensors(
        ((x[0], x[1]), (y[0], y[1]))))
    self.assertEqual(
        ((dtypes.int64, dtypes.int64), (dtypes.float64, dtypes.float64)),
        dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual((([3], [3]), ([2], [2])),
                     dataset_ops.get_legacy_output_shapes(dataset))

    dataset = dataset.batch(32)
    self.assertEqual(
        ((dtypes.int64, dtypes.int64), (dtypes.float64, dtypes.float64)),
        dataset_ops.get_legacy_output_types(dataset))
    dataset_output_shapes = dataset_ops.get_legacy_output_shapes(dataset)
    self.assertEqual(
        (([None, 3], [None, 3]), ([None, 2], [None, 2])),
        nest.pack_sequence_as(
            dataset_output_shapes,
            [s.as_list() for s in nest.flatten(dataset_output_shapes)]))

    # Define a separate set of components with matching leading
    # dimension for the from-slices constructor.
    components_for_slices = (np.array([1, 2, 3],
                                      dtype=np.int64), (np.array([4., 5., 6.]),
                                                        np.array([7., 8., 9.])),
                             np.array([10, 11, 12], dtype=np.int64))

    dataset = dataset_ops.Dataset.from_tensor_slices(components_for_slices)
    self.assertEqual(
        (dtypes.int64, (dtypes.float64, dtypes.float64), dtypes.int64),
        dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual(([], ([], []), []),
                     dataset_ops.get_legacy_output_shapes(dataset))

  @combinations.generate(test_base.default_test_combinations())
  def testNoneComponent(self):
    dataset = dataset_ops.Dataset.from_tensors((42, None))
    if context.executing_eagerly():
      self.assertDatasetProduces(dataset, expected_output=[(42, None)])
    else:
      iterator = dataset_ops.make_one_shot_iterator(dataset)
      next_first, next_second = iterator.get_next()
      self.assertIsNone(next_second)
      with self.cached_session() as sess:
        self.assertEqual(sess.run(next_first), 42)

  @combinations.generate(test_base.default_test_combinations())
  def testNoneComponentInFunction(self):

    @def_function.function
    def fn(ds):
      total = 0
      it = iter(ds)
      for elem in it:
        x, _ = elem
        total += x
      return total

    dataset = dataset_ops.Dataset.range(
        10, output_type=dtypes.int32).map(lambda x: (x, None))
    self.assertEqual(self.evaluate(fn(dataset)), 45)

  @combinations.generate(test_base.default_test_combinations())
  def testIncorrectPythonStructure(self):
    # Tests that an exception is raised (as opposed to a segfault) when the
    # Python structure assigned to a dataset is incorrect.
    dataset = dataset_ops.Dataset.range(10)
    spec = tensor_spec.TensorSpec([], dtypes.int64)
    new_structure = (spec, spec)
    dataset = dataset_ops._RestructuredDataset(dataset, new_structure)
    dataset = dataset.map(lambda x, y: y)

    with self.assertRaisesOpError(""):
      self.getDatasetOutput(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testNamedTupleStructure(self):
    Foo = collections.namedtuple("Foo", ["a", "b"])
    x = Foo(a=3, b="test")
    dataset = dataset_ops.Dataset.from_tensors(x)
    dataset = dataset_ops.Dataset.from_tensor_slices([dataset, dataset])
    self.assertEqual(
        str(dataset.element_spec),
        "DatasetSpec(Foo(a=TensorSpec(shape=(), dtype=tf.int32, name=None), "
        "b=TensorSpec(shape=(), dtype=tf.string, name=None)), TensorShape([]))")

  @combinations.generate(test_base.eager_only_combinations())
  def testIterationError(self):

    @def_function.function(autograph=False)
    def fn(ds):
      for _ in ds:
        pass

    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaises(ValueError):
      self.evaluate(fn(dataset))


class DebugDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

  def setUp(self):
    super(DebugDatasetTest, self).setUp()
    debug_mode.toggle_debug_mode(True)

  def tearDown(self):
    debug_mode.toggle_debug_mode(False)
    super(DebugDatasetTest, self).tearDown()

  @combinations.generate(test_base.eager_only_combinations())
  def testDebugModeEagerExecution(self):
    counter = []
    ds = dataset_ops.Dataset.range(10)

    def map_fn(x):
      counter.append(1)
      return x

    ds = ds.map(map_fn)
    self.assertDatasetProduces(ds, list(range(10)))

    # The body of `map_fn` will be executed 11 times since the implementation
    # traces the function to figure out what the types and shapes of its
    # outputs are.
    self.assertLen(counter, 11)

  @combinations.generate(test_base.eager_only_combinations())
  def testDebugModeGenerator(self):
    def gen():
      yield from range(10)

    ds = dataset_ops.Dataset.from_generator(
        gen,
        output_signature=tensor_spec.TensorSpec(shape=(), dtype=dtypes.int64))
    self.assertDatasetProduces(ds, list(range(10)))

  @combinations.generate(test_base.eager_only_combinations())
  def testDebugModeGeneratorTwoComponents(self):
    data = [(n, n+1) for n in range(10)]

    def gen():
      yield from data

    ds = dataset_ops.Dataset.from_generator(
        gen,
        output_signature=(tensor_spec.TensorSpec(shape=(), dtype=dtypes.int64),
                          tensor_spec.TensorSpec(shape=(), dtype=dtypes.int64)))
    self.assertDatasetProduces(ds, data)

  @combinations.generate(test_base.eager_only_combinations())
  def testDebugModeSequentialExecution(self):
    ds = dataset_ops.Dataset.range(10)
    ds = ds.apply(
        testing.assert_next(["Interleave", "Map", "Batch", "FiniteTake"]))
    ds = ds.interleave(
        dataset_ops.Dataset.from_tensors,
        cycle_length=10,
        num_parallel_calls=10)
    ds = ds.map(lambda x: x * x, num_parallel_calls=10)
    ds = ds.batch(batch_size=5, num_parallel_calls=2)
    ds = ds.prefetch(buffer_size=2)
    ds = ds.take(2)
    self.assertDatasetProduces(ds, [[0, 1, 4, 9, 16], [25, 36, 49, 64, 81]])


if __name__ == "__main__":
  test.main()
