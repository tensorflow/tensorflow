# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf upgrader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib

_TEST_VERSION = 1


class TestUpgrade(test_util.TensorFlowTestCase):
  """Test various APIs that have been changed in 2.0."""

  @classmethod
  def setUpClass(cls):
    cls._tf_api_version = 1 if hasattr(tf, 'contrib') else 2

  def setUp(self):
    tf.compat.v1.enable_v2_behavior()

  def testRenames(self):
    self.assertAllClose(1.04719755, tf.acos(0.5))
    self.assertAllClose(0.5, tf.rsqrt(4.0))

  def testSerializeSparseTensor(self):
    sp_input = tf.SparseTensor(
        indices=tf.constant([[1]], dtype=tf.int64),
        values=tf.constant([2], dtype=tf.int64),
        dense_shape=[2])

    with self.cached_session():
      serialized_sp = tf.serialize_sparse(sp_input, 'serialize_name', tf.string)
      self.assertEqual((3,), serialized_sp.shape)
      self.assertTrue(serialized_sp[0].numpy())  # check non-empty

  def testSerializeManySparse(self):
    sp_input = tf.SparseTensor(
        indices=tf.constant([[0, 1]], dtype=tf.int64),
        values=tf.constant([2], dtype=tf.int64),
        dense_shape=[1, 2])

    with self.cached_session():
      serialized_sp = tf.serialize_many_sparse(
          sp_input, 'serialize_name', tf.string)
      self.assertEqual((1, 3), serialized_sp.shape)

  def testArgMaxMin(self):
    self.assertAllClose(
        [1],
        tf.argmax([[1, 3, 2]], name='abc', dimension=1))
    self.assertAllClose(
        [0, 0, 0],
        tf.argmax([[1, 3, 2]], dimension=0))
    self.assertAllClose(
        [0],
        tf.argmin([[1, 3, 2]], name='abc', dimension=1))

  def testSoftmaxCrossEntropyWithLogits(self):
    out = tf.nn.softmax_cross_entropy_with_logits(
        logits=[0.1, 0.8], labels=[0, 1])
    self.assertAllClose(out, 0.40318608)
    out = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=[0.1, 0.8], labels=[0, 1])
    self.assertAllClose(out, 0.40318608)

  def testUniformUnitScalingInitializer(self):
    init = tf.initializers.uniform_unit_scaling(0.5, seed=1)
    self.assertArrayNear(
        [-0.45200047, 0.72815341],
        init((2,)).numpy(),
        err=1e-6)


if __name__ == "__main__":
  test_lib.main()
