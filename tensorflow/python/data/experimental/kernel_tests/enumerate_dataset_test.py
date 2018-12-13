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
"""Tests for `tf.data.experimental.enumerate_dataset()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.ops import enumerate_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class EnumerateDatasetTest(test_base.DatasetTestBase):

  def testEnumerateDataset(self):
    components = (["a", "b"], [1, 2], [37.0, 38])
    start = constant_op.constant(20, dtype=dtypes.int64)

    dataset = dataset_ops.Dataset.from_tensor_slices(components).apply(
        enumerate_ops.enumerate_dataset(start))

    self.assertEqual(dtypes.int64, dataset.output_types[0])
    self.assertEqual((), dataset.output_shapes[0])
    self.assertEqual([tensor_shape.TensorShape([])] * 3,
                     [shape for shape in dataset.output_shapes[1]])

    self.assertDatasetProduces(dataset, [(20, (b"a", 1, 37.0)),
                                         (21, (b"b", 2, 38.0))])


if __name__ == "__main__":
  test.main()
