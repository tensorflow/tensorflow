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
"""Tests for tf.contrib.kfac.op_queue."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.kfac.python.ops import op_queue
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class OpQueueTest(test.TestCase):

  def testNextOp(self):
    """Ensures all ops get selected eventually."""
    with tf_ops.Graph().as_default():
      ops = [
          math_ops.add(1, 2),
          math_ops.subtract(1, 2),
          math_ops.reduce_mean([1, 2]),
      ]
      queue = op_queue.OpQueue(ops, seed=0)

      with self.test_session() as sess:
        # Ensure every inv update op gets selected.
        selected_ops = set([queue.next_op(sess) for _ in ops])
        self.assertEqual(set(ops), set(selected_ops))

        # Ensure additional calls don't create any new ops.
        selected_ops.add(queue.next_op(sess))
        self.assertEqual(set(ops), set(selected_ops))


if __name__ == "__main__":
  test.main()
