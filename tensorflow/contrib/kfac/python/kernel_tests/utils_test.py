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
"""Tests for tf.contrib.kfac.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr

from tensorflow.contrib.kfac.python.ops import utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.platform import test


class SequenceDictTest(test.TestCase):

  def testSequenceDictInit(self):
    seq_dict = utils.SequenceDict()
    self.assertFalse(seq_dict._dict)

  def testSequenceDictInitWithIterable(self):
    reg_dict = {'a': 'foo', 'b': 'bar'}
    itr = zip(reg_dict.keys(), reg_dict.values())
    seq_dict = utils.SequenceDict(itr)
    self.assertEqual(reg_dict, seq_dict._dict)

  def testGetItemSingleKey(self):
    seq_dict = utils.SequenceDict({'a': 'foo', 'b': 'bar'})
    self.assertEqual('foo', seq_dict['a'])

  def testGetItemMultipleKeys(self):
    seq_dict = utils.SequenceDict({'a': 'foo', 'b': 'bar'})
    self.assertEqual(['foo', 'bar'], seq_dict[('a', 'b')])

  def testSetItemSingleKey(self):
    seq_dict = utils.SequenceDict()
    seq_dict['a'] = 'foo'
    self.assertEqual([('a', 'foo')], seq_dict.items())

  def testSetItemMultipleKeys(self):
    seq_dict = utils.SequenceDict()
    keys = ('a', 'b', 'c')
    values = ('foo', 'bar', 'baz')
    seq_dict[keys] = values
    self.assertItemsEqual(list(zip(keys, values)), seq_dict.items())


class SubGraphTest(test.TestCase):

  def testBasicGraph(self):
    a = array_ops.constant([[1., 2.], [3., 4.]])
    b = array_ops.constant([[5., 6.], [7., 8.]])
    c = a + b
    d = a * b
    sub_graph = utils.SubGraph((c,))
    self.assertTrue(sub_graph.is_member(a))
    self.assertTrue(sub_graph.is_member(b))
    self.assertTrue(sub_graph.is_member(c))
    self.assertFalse(sub_graph.is_member(d))

  def testRepeatedAdds(self):
    a = array_ops.constant([[1., 2.], [3., 4.]])
    b = array_ops.constant([[5., 6.], [7., 8.]])
    c = a + b + a  # note that a appears twice in this graph
    sub_graph = utils.SubGraph((c,))
    self.assertTrue(sub_graph.is_member(a))
    self.assertTrue(sub_graph.is_member(b))
    self.assertTrue(sub_graph.is_member(c))

  def testFilterList(self):
    a = array_ops.constant([[1., 2.], [3., 4.]])
    b = array_ops.constant([[5., 6.], [7., 8.]])
    c = a + b
    d = a * b
    sub_graph = utils.SubGraph((c,))
    input_list = [b, d]
    filtered_list = sub_graph.filter_list(input_list)
    self.assertEqual(filtered_list, [b])


class UtilsTest(test.TestCase):

  def _fully_connected_layer_params(self):
    weights_part = array_ops.constant([[1., 2.], [4., 3.]])
    bias_part = array_ops.constant([1., 2.])
    return (weights_part, bias_part)

  def _conv_layer_params(self):
    weights_shape = 2, 2, 3, 4
    biases_shape = weights_shape[-1:]
    weights = array_ops.constant(npr.RandomState(0).randn(*weights_shape))
    biases = array_ops.constant(npr.RandomState(1).randn(*biases_shape))
    return (weights, biases)

  def testFullyConnectedLayerParamsTupleToMat2d(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      layer_params = self._fully_connected_layer_params()
      output = utils.layer_params_to_mat2d(layer_params)
      self.assertListEqual([3, 2], output.get_shape().as_list())
      self.assertAllClose(
          sess.run(output), np.array([[1., 2.], [4., 3.], [1., 2.]]))

  def testFullyConnectedLayerParamsTensorToMat2d(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      layer_params = self._fully_connected_layer_params()
      output = utils.layer_params_to_mat2d(layer_params[0])
      self.assertListEqual([2, 2], output.get_shape().as_list())
      self.assertAllClose(sess.run(output), np.array([[1., 2.], [4., 3.]]))

  def testConvLayerParamsTupleToMat2d(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(200)
      layer_params = self._conv_layer_params()
      output = utils.layer_params_to_mat2d(layer_params)
      self.assertListEqual([2 * 2 * 3 + 1, 4], output.get_shape().as_list())

  def testKron(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      mat1 = np.array([[1., 2.], [3., 4.]])
      mat2 = np.array([[5., 6.], [7., 8.]])
      mat1_tf = array_ops.constant(mat1)
      mat2_tf = array_ops.constant(mat2)
      ans_tf = sess.run(utils.kronecker_product(mat1_tf, mat2_tf))
      ans_np = np.kron(mat1, mat2)
      self.assertAllClose(ans_tf, ans_np)

  def testMat2dToFullyConnectedLayerParamsTuple(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      vector_template = self._fully_connected_layer_params()
      mat2d = array_ops.constant([[5., 4.], [3., 2.], [1., 0.]])

      output = sess.run(utils.mat2d_to_layer_params(vector_template, mat2d))

      self.assertIsInstance(output, tuple)
      self.assertEqual(len(output), 2)
      a, b = output
      self.assertAllClose(a, np.array([[5., 4.], [3., 2.]]))
      self.assertAllClose(b, np.array([1., 0.]))

  def testMat2dToFullyConnectedLayerParamsTensor(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      vector_template = self._fully_connected_layer_params()[0]
      mat2d = array_ops.constant([[5., 4.], [3., 2.]])

      output = sess.run(utils.mat2d_to_layer_params(vector_template, mat2d))

      self.assertAllClose(output, np.array([[5., 4.], [3., 2.]]))

  def testTensorsToColumn(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)

      vector = array_ops.constant(np.array([[0., 1.], [2., 3.]]))
      output = utils.tensors_to_column(vector)
      self.assertListEqual([4, 1], output.get_shape().as_list())
      self.assertAllClose(sess.run(output), np.array([0., 1., 2., 3.])[:, None])

      vector = self._fully_connected_layer_params()
      output = utils.tensors_to_column(vector)
      self.assertListEqual([6, 1], output.get_shape().as_list())
      self.assertAllClose(
          sess.run(output), np.array([1., 2., 4., 3., 1., 2.])[:, None])

      vector = list(vector)
      vector.append(array_ops.constant([[6.], [7.], [8.], [9.]]))

      output = utils.tensors_to_column(vector)
      self.assertListEqual([10, 1], output.get_shape().as_list())
      self.assertAllClose(
          sess.run(output),
          np.array([1., 2., 4., 3., 1., 2., 6., 7., 8., 9.])[:, None])

  def testColumnToTensors(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)

      vector_template = array_ops.constant(np.array([[0., 1.], [2., 3.]]))
      colvec = array_ops.constant(np.arange(4.)[:, None])
      output = sess.run(utils.column_to_tensors(vector_template, colvec))
      self.assertAllClose(output, np.array([[0., 1.], [2., 3.]]))

      vector_template = self._fully_connected_layer_params()
      colvec = array_ops.constant(np.arange(6.)[:, None])
      output = sess.run(utils.column_to_tensors(vector_template, colvec))

      self.assertIsInstance(output, tuple)
      self.assertEqual(len(output), 2)
      a, b = output
      self.assertAllClose(a, np.array([[0., 1.], [2., 3.]]))
      self.assertAllClose(b, np.array([4., 5.]))

      vector_template = list(vector_template)
      vector_template.append(array_ops.constant([[6.], [7.], [8.], [9.]]))
      colvec = array_ops.constant(np.arange(10.)[:, None])
      output = sess.run(utils.column_to_tensors(vector_template, colvec))
      self.assertIsInstance(output, tuple)
      self.assertEqual(len(output), 3)
      a, b, c = output
      self.assertAllClose(a, np.array([[0., 1.], [2., 3.]]))
      self.assertAllClose(b, np.array([4., 5.]))
      self.assertAllClose(c, np.array([[6.], [7.], [8.], [9.]]))

  def testPosDefInvCholesky(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      npr.seed(0)
      square = lambda x: np.dot(x, x.T)

      size = 3
      x = square(npr.randn(size, size))
      damp = 0.1
      identity = linalg_ops.eye(size, dtype=dtypes.float64)

      tf_inv = utils.posdef_inv_cholesky(array_ops.constant(x), identity, damp)
      np_inv = np.linalg.inv(x + damp * np.eye(size))
      self.assertAllClose(sess.run(tf_inv), np_inv)

  def testPosDefInvMatrixInverse(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      npr.seed(0)
      square = lambda x: np.dot(x, x.T)

      size = 3
      x = square(npr.randn(size, size))
      damp = 0.1
      identity = linalg_ops.eye(size, dtype=dtypes.float64)

      tf_inv = utils.posdef_inv_matrix_inverse(
          array_ops.constant(x), identity, damp)
      np_inv = np.linalg.inv(x + damp * np.eye(size))
      self.assertAllClose(sess.run(tf_inv), np_inv)


if __name__ == '__main__':
  test.main()
