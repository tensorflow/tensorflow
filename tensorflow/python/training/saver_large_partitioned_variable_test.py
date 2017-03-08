# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Tests for tensorflow.python.training.saver.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf


class SaverLargePartitionedVariableTest(tf.test.TestCase):

  # Need to do this in a separate test because of the amount of memory needed
  # to run this test.
  def testLargePartitionedVariables(self):
    save_path = os.path.join(self.get_temp_dir(), "large_variable")
    var_name = "my_var"
    # Saving large partition variable.
    with tf.Session("", graph=tf.Graph()) as sess:
      with tf.device("/cpu:0"):
        # Create a partitioned variable which is larger than int32 size but
        # split into smaller sized variables.
        init = lambda shape, dtype, partition_info: tf.constant(
            True, dtype, shape)
        partitioned_var = tf.create_partitioned_variables(
            [1 << 31], [4], init, dtype=tf.bool, name=var_name)
        tf.global_variables_initializer().run()
        save = tf.train.Saver(partitioned_var)
        val = save.save(sess, save_path)
        self.assertEqual(save_path, val)


if __name__ == "__main__":
  tf.test.main()
