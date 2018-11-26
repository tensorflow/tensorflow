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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class InputsTest(test_base.DatasetTestBase, parameterized.TestCase):

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
      ("FixedLengthRecord", readers.FixedLengthRecordDataset("", 42)),
      ("FromGenerator",
       dataset_ops.Dataset.from_generator(make_gen.__func__(), dtypes.int32),
       1),
      ("FromSparseTensorSlices",
       dataset_ops.Dataset.from_sparse_tensor_slices(
           sparse_tensor.SparseTensor(
               indices=np.array([[0, 0], [1, 0], [2, 0]]),
               values=np.array([0, 0, 0]),
               dense_shape=np.array([3, 1])))),
      ("FromTensors", dataset_ops.Dataset.from_tensors([42])),
      ("FromTensorSlices", dataset_ops.Dataset.from_tensors([42])),
      ("Range", dataset_ops.Dataset.range(10)),
      ("TextLine", readers.TextLineDataset("")),
      ("TFRecord", readers.TFRecordDataset(""), 1),
  )
  def testDatasetSourceInputs(self, dataset, num_inputs=0):
    self.assertEqual(num_inputs, len(dataset._inputs()))

  @parameterized.named_parameters(
      ("Apply", make_apply_fn.__func__(dataset_ops.Dataset.range(0)),
       dataset_ops.Dataset.range(0)),
      ("Batch", lambda x: x.batch(10), dataset_ops.Dataset.range(0)),
      ("Cache", lambda x: x.cache(), dataset_ops.Dataset.range(0)),
      ("Filter", lambda x: x.filter(lambda x: True),
       dataset_ops.Dataset.range(0)),
      ("FlatMap", lambda x: x.flat_map(lambda x: dataset_ops.Dataset.range(0)),
       dataset_ops.Dataset.range(0)),
      ("Interleave", make_interleave_fn.__func__(dataset_ops.Dataset.range(0)),
       dataset_ops.Dataset.range(0)),
      ("Map", lambda x: x.map(lambda x: x), dataset_ops.Dataset.range(0)),
      ("PaddedBatch", lambda x: x.padded_batch(10, []),
       dataset_ops.Dataset.range(0)),
      ("ParallelInterleave",
       make_interleave_fn.__func__(dataset_ops.Dataset.range(0), 2),
       dataset_ops.Dataset.range(0)),
      ("ParallelMap", lambda x: x.map(lambda x: x, num_parallel_calls=2),
       dataset_ops.Dataset.range(0)),
      ("Repeat", lambda x: x.repeat(), dataset_ops.Dataset.range(0)),
      ("Shuffle", lambda x: x.shuffle(10), dataset_ops.Dataset.range(0)),
      ("Skip", lambda x: x.skip(1), dataset_ops.Dataset.range(0)),
      ("Take", lambda x: x.take(1), dataset_ops.Dataset.range(0)),
      ("Window", lambda x: x.window(10), dataset_ops.Dataset.range(0)),
  )
  def testUnaryTransformationInputs(self, dataset_fn, input_dataset):
    self.assertEqual([input_dataset], dataset_fn(input_dataset)._inputs())

  @parameterized.named_parameters(
      ("Concatenate", lambda x, y: x.concatenate(y),
       dataset_ops.Dataset.range(0), dataset_ops.Dataset.range(1)))
  def testBinaryTransformationInputs(self, dataset_fn, input1, input2):
    self.assertEqual([input1, input2], dataset_fn(input1, input2)._inputs())

  @parameterized.named_parameters(
      ("ZipOne", dataset_ops.Dataset.zip, (dataset_ops.Dataset.range(0))),
      ("ZipNest", dataset_ops.Dataset.zip,
       (dataset_ops.Dataset.range(0),
        (dataset_ops.Dataset.range(1), dataset_ops.Dataset.range(2)))),
      ("ZipTuple", dataset_ops.Dataset.zip,
       (dataset_ops.Dataset.range(0), dataset_ops.Dataset.range(1))))
  def testVariadicTransformationInputs(self, dataset_fn, input_datasets):
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


if __name__ == "__main__":
  test.main()
