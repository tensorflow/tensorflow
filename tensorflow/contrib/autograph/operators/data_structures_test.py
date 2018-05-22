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
"""Tests for data_structures module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.autograph.operators import data_structures
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test


class AppendTest(test.TestCase):

  def test_tf_tensorarray(self):
    l = tensor_array_ops.TensorArray(dtypes.int32, size=0, dynamic_size=True)
    l1 = data_structures.append(l, 1)
    l2 = data_structures.append(l1, 2)
    with self.test_session() as sess:
      self.assertAllEqual(sess.run(l1.stack()), [1])
      self.assertAllEqual(sess.run(l2.stack()), [1, 2])

  def test_python(self):
    l = []
    self.assertAllEqual(data_structures.append(l, 1), [1])
    self.assertAllEqual(data_structures.append(l, 2), [1, 2])


if __name__ == '__main__':
  test.main()
