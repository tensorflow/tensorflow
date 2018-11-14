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

"""Test for version 1 of the zero_out op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf
from tensorflow.examples.adding_an_op import zero_out_op_1


class ZeroOut1Test(tf.test.TestCase):

  def test(self):
    with self.cached_session():
      result = zero_out_op_1.zero_out([5, 4, 3, 2, 1])
      self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])

  def testLoadTwice(self):
    zero_out_loaded_again = tf.load_op_library(os.path.join(
        tf.resource_loader.get_data_files_path(), 'zero_out_op_kernel_1.so'))
    self.assertEqual(zero_out_loaded_again, zero_out_op_1._zero_out_module)


if __name__ == '__main__':
  tf.test.main()
