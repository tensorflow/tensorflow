# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for tensorflow.python.framework.ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class SparseTensorTest(test_util.TensorFlowTestCase):

  def testPythonConstruction(self):
    indices = [[1, 2], [2, 0], [3, 4]]
    values = [b"a", b"b", b"c"]
    shape = [4, 5]
    sp_value = sparse_tensor.SparseTensorValue(indices, values, shape)
    for sp in [
        sparse_tensor.SparseTensor(indices, values, shape),
        sparse_tensor.SparseTensor.from_value(sp_value),
        sparse_tensor.SparseTensor.from_value(
            sparse_tensor.SparseTensor(indices, values, shape))]:
      self.assertEqual(sp.indices.dtype, dtypes.int64)
      self.assertEqual(sp.values.dtype, dtypes.string)
      self.assertEqual(sp.shape.dtype, dtypes.int64)
      self.assertEqual(sp.get_shape(), (4, 5))

      with self.test_session() as sess:
        value = sp.eval()
        self.assertAllEqual(indices, value.indices)
        self.assertAllEqual(values, value.values)
        self.assertAllEqual(shape, value.shape)
        sess_run_value = sess.run(sp)
        self.assertAllEqual(sess_run_value.indices, value.indices)
        self.assertAllEqual(sess_run_value.values, value.values)
        self.assertAllEqual(sess_run_value.shape, value.shape)


if __name__ == "__main__":
  googletest.main()
