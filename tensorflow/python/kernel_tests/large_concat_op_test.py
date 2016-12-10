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
# ==============================================================================
"""Functional tests for Concat Op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class LargeConcatOpTest(tf.test.TestCase):
  """Tests that belong in concat_op_test.py, but run over large tensors."""

  def testConcatLargeTensors(self):
    # CPU-only test, because it fails on GPUs with <= 4GB memory.
    with tf.device("/cpu:0"):
      a = tf.ones([2**31 + 6], dtype=tf.int8)
      b = tf.zeros([1024], dtype=tf.int8)
      onezeros = tf.concat_v2([a, b], 0)
    with self.test_session(use_gpu=False):
      # TODO(dga):  Add more depth to this test to validate correctness,
      # not just non-crashingness, once other large tensor fixes have gone in.
      _ = onezeros.eval()


if __name__ == "__main__":
  tf.test.main()
