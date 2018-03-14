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
"""Tests for the experimental rolling_window ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.contrib.data.python.ops import rolling_ops

class RollingWindowTest(test.TestCase):

  def test_rolling_window(self):
    input_data = dataset_ops.Dataset.from_tensor_slices([
        [[0], [1], [2], [3], [4], [5], [6], [7], [8]]])
    rolled_data = input_data.apply(rolling_ops.rolling_window(window_size=3,
                                                              stride=2))

    self.assertEqual(rolled_data.output_shapes.as_list(), [None, 1])
    iterator = rolled_data.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for i in range(4):
        result = sess.run(get_next)
        self.assertAllEqual(np.array([[x] for x in range(i*2, i*2+3)]), result)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def test_rolling_window_empty(self):
    input_data = dataset_ops.Dataset.from_tensor_slices([[]])
    rolled_data = input_data.apply(rolling_ops.rolling_window(window_size=7,
                                                              stride=1))
    self.assertEqual(rolled_data.output_shapes.as_list(), [None,])

if __name__ == "__main__":
  test.main()
