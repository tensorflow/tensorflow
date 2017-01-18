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
"""Ops tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np

from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.learn.python.learn import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import variables
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class OpsTest(test.TestCase):
  """Ops tests."""

  def test_softmax_classifier(self):
    with self.test_session() as session:
      features = array_ops.placeholder(dtypes.float32, [None, 3])
      labels = array_ops.placeholder(dtypes.float32, [None, 2])
      weights = constant_op.constant([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])
      biases = constant_op.constant([0.2, 0.3])
      class_weight = constant_op.constant([0.1, 0.9])
      prediction, loss = ops.softmax_classifier(features, labels, weights,
                                                biases, class_weight)
      self.assertEqual(prediction.get_shape()[1], 2)
      self.assertEqual(loss.get_shape(), [])
      value = session.run(loss, {features: [[0.2, 0.3, 0.2]], labels: [[0, 1]]})
      self.assertAllClose(value, 0.55180627)

  def test_embedding_lookup(self):
    d_embed = 5
    n_embed = 10
    ids_shape = (2, 3, 4)
    embeds = np.random.randn(n_embed, d_embed)
    ids = np.random.randint(0, n_embed, ids_shape)
    with self.test_session():
      embed_np = embeds[ids]
      embed_tf = ops.embedding_lookup(embeds, ids).eval()
    self.assertEqual(embed_np.shape, embed_tf.shape)
    self.assertAllClose(embed_np, embed_tf)

  def test_categorical_variable(self):
    random_seed.set_random_seed(42)
    with self.test_session() as sess:
      cat_var_idx = array_ops.placeholder(dtypes.int64, [2, 2])
      embeddings = ops.categorical_variable(
          cat_var_idx, n_classes=5, embedding_size=10, name="my_cat_var")
      sess.run(variables.global_variables_initializer())
      emb1 = sess.run(embeddings,
                      feed_dict={cat_var_idx.name: [[0, 1], [2, 3]]})
      emb2 = sess.run(embeddings,
                      feed_dict={cat_var_idx.name: [[0, 2], [1, 3]]})
    self.assertEqual(emb1.shape, emb2.shape)
    self.assertAllEqual(np.transpose(emb2, axes=[1, 0, 2]), emb1)


if __name__ == "__main__":
  test.main()
