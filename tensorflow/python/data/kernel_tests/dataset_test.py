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

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


@test_util.run_all_in_graph_and_eager_modes
class DatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

  def testAsSerializedGraph(self):
    dataset = dataset_ops.Dataset.range(10)
    graph = graph_pb2.GraphDef().FromString(
        self.evaluate(dataset._as_serialized_graph()))
    self.assertTrue(any([node.op != "RangeDataset" for node in graph.node]))

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
    self.assertEqual(num_inputs, len(dataset_fn()._inputs()))

  def testDatasetComplexSourceInputs(self):
    dataset_fn = dataset_ops.Dataset.from_sparse_tensor_slices(
        sparse_tensor.SparseTensor(
            indices=np.array([[0, 0], [1, 0], [2, 0]]),
            values=np.array([0, 0, 0]),
            dense_shape=np.array([3, 1])))
    self.assertEqual(0, len(dataset_fn._inputs()))

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
       structure.TensorStructure(dtypes.float32, [])),
      ("SparseTensor", lambda: sparse_tensor.SparseTensor(
          indices=[[0]], values=constant_op.constant([0], dtype=dtypes.int32),
          dense_shape=[1]),
       structure.SparseTensorStructure(dtypes.int32, [1])),
      ("Nest", lambda: {
          "a": constant_op.constant(37.0),
          "b": (constant_op.constant(["Foo"]), constant_op.constant("Bar"))},
       structure.NestedStructure({
           "a": structure.TensorStructure(dtypes.float32, []),
           "b": (structure.TensorStructure(dtypes.string, [1]),
                 structure.TensorStructure(dtypes.string, []))})),
      ("Dataset", lambda: dataset_ops.Dataset.from_tensor_slices(
          constant_op.constant([1, 2, 3])),
       dataset_ops.DatasetStructure(
           structure.TensorStructure(dtypes.int32, []))),
      ("Optional", lambda: optional_ops.Optional.from_value(37.0),
       optional_ops.OptionalStructure(
           structure.TensorStructure(dtypes.float32, []))),
  )
  def testDatasetStructure(self, tf_value_fn, expected_element_structure):
    dataset = dataset_ops.Dataset.from_tensors(0).map(lambda _: tf_value_fn())
    dataset_structure = structure.Structure.from_value(dataset)
    self.assertIsInstance(dataset_structure, dataset_ops.DatasetStructure)

    # TODO(b/110122868): Add a public API to `tf.data.Dataset` for accessing
    # the element structure.
    self.assertTrue(expected_element_structure.is_compatible_with(
        dataset_structure._element_structure))
    self.assertTrue(dataset_structure._element_structure.is_compatible_with(
        expected_element_structure))

    self.assertEqual([dtypes.variant], dataset_structure._flat_types)
    self.assertEqual([tensor_shape.scalar()], dataset_structure._flat_shapes)

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

  # NOTE: This test is specific to graph mode and is skipped in eager mode.
  @test_util.run_deprecated_v1
  def testSkipEagerSameGraphErrorOneShot(self):
    dataset = dataset_ops.Dataset.range(10)
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(ValueError, "must be from the same graph"):
        dataset = dataset.batch(2)

  # NOTE: This test is specific to graph mode and is skipped in eager mode.
  @test_util.run_deprecated_v1
  def testSkipEagerSameGraphErrorOneShotSimple(self):
    dataset = dataset_ops.Dataset.range(10)
    with ops.Graph().as_default():
      with test.mock.patch.object(logging, "warning") as mock_log:
        _ = dataset.make_one_shot_iterator()
        self.assertRegexpMatches(
            str(mock_log.call_args), "Please ensure that all datasets in the "
            "pipeline are created in the same graph as the iterator.")

  # NOTE: This test is specific to graph mode and is skipped in eager mode.
  @test_util.run_deprecated_v1
  def testSkipEagerSameGraphErrorInitializable(self):
    dataset = dataset_ops.Dataset.range(10)
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(ValueError, "must be from the same graph"):
        dataset = dataset.batch(2)


if __name__ == "__main__":
  test.main()
