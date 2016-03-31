# Copyright 2016 Google Inc. All Rights Reserved.
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
"""tensor_util tests."""

# pylint: disable=unused-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class OpsTest(tf.test.TestCase):

  def testGetGraphFromEmptyInputs(self):
    with tf.Graph().as_default() as g0:
      self.assertIs(g0, tf.contrib.framework.get_graph_from_inputs([]))

  def testGetGraphFromValidInputs(self):
    g0 = tf.Graph()
    with g0.as_default():
      values = [tf.constant(0.0), tf.constant(1.0)]
    self.assertIs(g0, tf.contrib.framework.get_graph_from_inputs(values))
    self.assertIs(g0, tf.contrib.framework.get_graph_from_inputs(values, g0))
    with tf.Graph().as_default():
      self.assertIs(g0, tf.contrib.framework.get_graph_from_inputs(values))
      self.assertIs(g0, tf.contrib.framework.get_graph_from_inputs(values, g0))

  def testGetGraphFromInvalidInputs(self):
    g0 = tf.Graph()
    with g0.as_default():
      values = [tf.constant(0.0), tf.constant(1.0)]
    g1 = tf.Graph()
    with self.assertRaisesRegexp(ValueError, "not from the passed-in graph"):
      tf.contrib.framework.get_graph_from_inputs(values, g1)
    with g1.as_default():
      values.append(tf.constant(2.0))
    with self.assertRaisesRegexp(ValueError, "must be from the same graph"):
      tf.contrib.framework.get_graph_from_inputs(values)
    with self.assertRaisesRegexp(ValueError, "not from the passed-in graph"):
      tf.contrib.framework.get_graph_from_inputs(values, g0)
    with self.assertRaisesRegexp(ValueError, "not from the passed-in graph"):
      tf.contrib.framework.get_graph_from_inputs(values, g1)

if __name__ == "__main__":
  tf.test.main()
