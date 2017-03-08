# pylint: disable=g-bad-file-header
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

"""Tests for tensorflow.contrib.graph_editor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import graph_editor as ge


class SubgraphviewTest(tf.test.TestCase):

  def test_simple_swap(self):
    g = tf.Graph()
    with g.as_default():
      a0 = tf.constant(1.0, shape=[2], name="a0")
      b0 = tf.constant(2.0, shape=[2], name="b0")
      c0 = tf.add(a0, b0, name="c0")
      a1 = tf.constant(3.0, shape=[2], name="a1")
      b1 = tf.constant(4.0, shape=[2], name="b1")
      c1 = tf.add(a1, b1, name="b1")

    ge.util.swap_ts([a0, b0], [a1, b1])
    assert c0.op.inputs[0] == a1 and c0.op.inputs[1] == b1
    assert c1.op.inputs[0] == a0 and c1.op.inputs[1] == b0


if __name__ == "__main__":
  tf.test.main()
