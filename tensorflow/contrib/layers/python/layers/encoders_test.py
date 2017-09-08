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
"""Tests for tensorflow.contrib.layers.python.layers.encoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.layers.python.layers import encoders
from tensorflow.contrib.layers.python.ops import sparse_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def _get_const_var(name, shape, value):
  return variable_scope.get_variable(
      name, shape, initializer=init_ops.constant_initializer(value))


class EncodersTest(test.TestCase):

  def testBowEncoderSparse(self):
    with self.test_session() as sess:
      docs = [[0, 1], [2, 3]]
      enc = encoders.bow_encoder(docs, 4, 3)
      sess.run(variables.global_variables_initializer())
      self.assertAllEqual([2, 3], enc.eval().shape)

  def testBowEncoderSparseTensor(self):
    with self.test_session() as sess:
      docs = [[0, 1], [2, 3]]
      sparse_docs = sparse_ops.dense_to_sparse_tensor(docs)
      enc = encoders.bow_encoder(sparse_docs, 4, 3)
      sess.run(variables.global_variables_initializer())
      self.assertAllEqual([2, 3], enc.eval().shape)

  def testBowEncoderSparseEmptyRow(self):
    with self.test_session() as sess:
      docs = [[0, 1], [2, 3], [0, 0]]
      enc = encoders.bow_encoder(docs, 4, 5)
      sess.run(variables.global_variables_initializer())
      self.assertAllEqual([3, 5], enc.eval().shape)

  def testBowEncoderDense(self):
    with self.test_session() as sess:
      docs = [[0, 1], [2, 3], [0, 0], [0, 0]]
      enc = encoders.bow_encoder(docs, 4, 3, sparse_lookup=False)
      sess.run(variables.global_variables_initializer())
      self.assertAllEqual([4, 3], enc.eval().shape)

  def testBowEncoderSparseTensorDenseLookup(self):
    with self.test_session():
      docs = [[0, 1]]
      sparse_docs = sparse_ops.dense_to_sparse_tensor(docs)
      with self.assertRaises(TypeError):
        encoders.bow_encoder(sparse_docs, 4, 3, sparse_lookup=False)

  def testBowEncodersSharingEmbeddings(self):
    with self.test_session() as sess:
      docs = [[0, 1], [2, 3]]
      enc_1 = encoders.bow_encoder(docs, 4, 3, scope='test')
      enc_2 = encoders.bow_encoder(docs, 4, 3, scope='test', reuse=True)
      sess.run(variables.global_variables_initializer())
      avg_1, avg_2 = sess.run([enc_1, enc_2])
      self.assertAllEqual(avg_1, avg_2)

  def testBowEncodersSharingEmbeddingsInheritedScopes(self):
    with self.test_session() as sess:
      docs = [[0, 1], [2, 3]]
      with variable_scope.variable_scope('test'):
        enc_1 = encoders.bow_encoder(docs, 4, 3)
      with variable_scope.variable_scope('test', reuse=True):
        enc_2 = encoders.bow_encoder(docs, 4, 3)
      sess.run(variables.global_variables_initializer())
      avg_1, avg_2 = sess.run([enc_1, enc_2])
      self.assertAllEqual(avg_1, avg_2)

  def testBowEncodersSharingEmbeddingsSharedScope(self):
    with self.test_session() as sess:
      docs = [[0, 1], [2, 3]]
      enc_1 = encoders.bow_encoder(docs, 4, 3, scope='bow')
      variable_scope.get_variable_scope().reuse_variables()
      enc_2 = encoders.bow_encoder(docs, 4, 3, scope='bow')
      sess.run(variables.global_variables_initializer())
      avg_1, avg_2 = sess.run([enc_1, enc_2])
      self.assertAllEqual(avg_1, avg_2)

  def testBowEncoderReuseEmbeddingsVariable(self):
    with self.test_session() as sess:
      docs = [[1, 1], [2, 3]]
      with variable_scope.variable_scope('test'):
        v = _get_const_var('embeddings', (4, 3),
                           [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
        self.assertEqual(v.name, 'test/embeddings:0')
      enc = encoders.bow_encoder(docs, 4, 3, scope='test', reuse=True)
      sess.run(variables.global_variables_initializer())
      self.assertAllClose([[3., 4., 5.], [7.5, 8.5, 9.5]], enc.eval())

  def testEmbedSequence(self):
    with self.test_session() as sess:
      docs = [[1, 1], [2, 3]]
      with variable_scope.variable_scope('test'):
        v = _get_const_var('embeddings', (4, 3),
                           [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
        self.assertEqual(v.name, 'test/embeddings:0')
      emb = encoders.embed_sequence(docs, 4, 3, scope='test', reuse=True)
      sess.run(variables.global_variables_initializer())
      self.assertAllClose(
          [[[3., 4., 5.], [3., 4., 5.]], [[6., 7., 8.], [9., 10., 11.]]],
          emb.eval())


if __name__ == '__main__':
  test.main()
