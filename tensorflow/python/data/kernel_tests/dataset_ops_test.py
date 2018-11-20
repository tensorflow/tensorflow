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
"""Tests for the input pipeline ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.platform import test


class DatasetOpsTest(test_base.DatasetTestBase, parameterized.TestCase):

  def testAsSerializedGraph(self):
    dataset = dataset_ops.Dataset.range(10)
    with self.cached_session() as sess:
      graph = graph_pb2.GraphDef().FromString(
          sess.run(dataset._as_serialized_graph()))
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
           DatasetOpsTest.make_gen(), dtypes.int32),
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

  def testOptionsDefault(self):
    ds = dataset_ops.Dataset.range(0)
    self.assertEqual(dataset_ops.Options(), ds.options())

  def testOptionsOnce(self):
    options = dataset_ops.Options()
    ds = dataset_ops.Dataset.range(0).with_options(options).cache()
    self.assertEqual(options, ds.options())

  def testOptionsTwiceSame(self):
    options = dataset_ops.Options()
    options.experimental_autotune = True
    ds = dataset_ops.Dataset.range(0).with_options(options).with_options(
        options)
    self.assertEqual(options, ds.options())

  def testOptionsTwiceDifferent(self):
    options1 = dataset_ops.Options()
    options1.experimental_autotune = True
    options2 = dataset_ops.Options()
    options2.experimental_filter_fusion = False
    ds = dataset_ops.Dataset.range(0).with_options(options1).with_options(
        options2)
    self.assertTrue(ds.options().experimental_autotune)
    # Explicitly check that flag is False since assertFalse allows None
    self.assertIs(ds.options().experimental_filter_fusion, False)

  def testOptionsTwiceDifferentError(self):
    options1 = dataset_ops.Options()
    options1.experimental_autotune = True
    options2 = dataset_ops.Options()
    options2.experimental_autotune = False
    with self.assertRaisesRegexp(ValueError,
                                 "Cannot merge incompatible values of option"):
      dataset_ops.Dataset.range(0).with_options(options1).with_options(options2)

  def testOptionsMergeOptionsFromMultipleInputs(self):
    options1 = dataset_ops.Options()
    options1.experimental_autotune = True
    options2 = dataset_ops.Options()
    options2.experimental_filter_fusion = True
    ds = dataset_ops.Dataset.zip(
        (dataset_ops.Dataset.range(0).with_options(options1),
         dataset_ops.Dataset.range(0).with_options(options2)))
    self.assertTrue(ds.options().experimental_autotune)
    self.assertTrue(ds.options().experimental_filter_fusion)


if __name__ == "__main__":
  test.main()
