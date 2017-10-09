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
"""Tests for tf.contrib.kfac.fisher_blocks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.kfac.python.ops import fisher_blocks as fb
from tensorflow.contrib.kfac.python.ops import layer_collection as lc
from tensorflow.contrib.kfac.python.ops import utils
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import test


def _make_psd(dim):
  """Constructs a PSD matrix of the given dimension."""
  mat = np.ones((dim, dim), dtype=np.float32)
  mat[np.arange(dim), np.arange(dim)] = 2. + np.arange(dim)
  return array_ops.constant(mat)


class FullFBTest(test.TestCase):

  def testFullFBInitSingleTensor(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(200)
      params = (array_ops.constant([1., 2.]), array_ops.constant(3.))
      block = fb.FullFB(lc.LayerCollection(), params, 32)

      self.assertAllEqual(params, block.tensors_to_compute_grads())

  def testFullFBInitTensorTuple(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(200)
      params = (array_ops.constant([1., 2.]), array_ops.constant(3.))
      block = fb.FullFB(lc.LayerCollection(), params, 32)

      self.assertAllEqual(params, block.tensors_to_compute_grads())

  def testInstantiateFactors(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(200)
      params = (array_ops.constant([1., 2.]), array_ops.constant(3.))
      block = fb.FullFB(lc.LayerCollection(), params, 32)

      grads = (params[0]**2, math_ops.sqrt(params[1]))
      block.instantiate_factors(grads, 0.5)

  def testMultiplyInverseTuple(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      params = (array_ops.constant([1., 2.]), array_ops.constant(3.))
      block = fb.FullFB(lc.LayerCollection(), params, 32)
      grads = (params[0]**2, math_ops.sqrt(params[1]))
      block.instantiate_factors((grads,), 0.5)

      # Make sure our inverse is something other than the identity.
      sess.run(tf_variables.global_variables_initializer())
      sess.run(block._factor.make_inverse_update_ops())

      vector = array_ops.ones(3,) * 2
      output = block.multiply_inverse(vector)

      self.assertAllClose(sess.run(vector * 2 / 3.), sess.run(output))

  def testMultiplyInverseNotTuple(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      params = array_ops.constant([[1.], [2.]])
      block = fb.FullFB(lc.LayerCollection(), params, 32)
      grads = params**2
      block.instantiate_factors((grads,), 0.5)

      # Make sure our inverse is something other than the identity.
      sess.run(tf_variables.global_variables_initializer())
      sess.run(block._factor.make_inverse_update_ops())

      vector = array_ops.ones(2,) * 2
      output = block.multiply_inverse(vector)

      self.assertAllClose(sess.run(vector * 2 / 3.), sess.run(output))

  def testMultiplyInverseAgainstExplicit(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      params = (array_ops.constant([1., 2.]), array_ops.constant(3.))
      block = fb.FullFB(lc.LayerCollection(), params, 32)
      grads = (array_ops.constant([2., 3.]), array_ops.constant(4.))
      damping = 0.5
      block.instantiate_factors((grads,), damping)

      # Make sure our inverse is something other than the identity.
      sess.run(state_ops.assign(block._factor._cov, _make_psd(3)))
      sess.run(block._factor.make_inverse_update_ops())

      v_flat = np.array([4., 5., 6.], dtype=np.float32)
      vector = utils.column_to_tensors(params, array_ops.constant(v_flat))
      output = block.multiply_inverse(vector)
      output_flat = sess.run(utils.tensors_to_column(output)).ravel()

      full = sess.run(block.full_fisher_block())
      explicit = np.dot(np.linalg.inv(full + damping * np.eye(3)), v_flat)

      self.assertAllClose(output_flat, explicit)


class NaiveDiagonalFBTest(test.TestCase):

  def testNaiveDiagonalFBInitSingleTensor(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(200)
      params = (array_ops.constant([1., 2.]), array_ops.constant(3.))
      block = fb.NaiveDiagonalFB(lc.LayerCollection(), params, 32)

      self.assertAllEqual(params, block.tensors_to_compute_grads())

  def testNaiveDiagonalFBInitTensorTuple(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(200)
      params = (array_ops.constant([1., 2.]), array_ops.constant(3.))
      block = fb.NaiveDiagonalFB(lc.LayerCollection(), params, 32)

      self.assertAllEqual(params, block.tensors_to_compute_grads())

  def testInstantiateFactors(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(200)
      params = (array_ops.constant([1., 2.]), array_ops.constant(3.))
      block = fb.NaiveDiagonalFB(lc.LayerCollection(), params, 32)

      grads = (params[0]**2, math_ops.sqrt(params[1]))
      block.instantiate_factors(grads, 0.5)

  def testMultiplyInverseTuple(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      params = (array_ops.constant([1., 2.]), array_ops.constant(3.))
      block = fb.NaiveDiagonalFB(lc.LayerCollection(), params, 32)
      grads = (params[0]**2, math_ops.sqrt(params[1]))
      block.instantiate_factors((grads,), 0.5)

      # Make sure our inverse is something other than the identity.
      sess.run(tf_variables.global_variables_initializer())
      sess.run(block._factor.make_inverse_update_ops())

      vector = array_ops.ones(3,) * 2
      output = block.multiply_inverse(vector)

      self.assertAllClose(sess.run(vector * 2 / 3.), sess.run(output))

  def testMultiplyInverseNotTuple(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      params = array_ops.constant([[1.], [2.]])
      block = fb.NaiveDiagonalFB(lc.LayerCollection(), params, 32)
      grads = params**2
      block.instantiate_factors((grads,), 0.5)

      # Make sure our inverse is something other than the identity.
      sess.run(tf_variables.global_variables_initializer())
      sess.run(block._factor.make_inverse_update_ops())
      vector = array_ops.ones(2,) * 2
      output = block.multiply_inverse(vector)

      self.assertAllClose(sess.run(vector * 2 / 3.), sess.run(output))

  def testMultiplyInverseAgainstExplicit(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      params = (array_ops.constant([1., 2.]), array_ops.constant(3.))
      block = fb.NaiveDiagonalFB(lc.LayerCollection(), params, 32)
      grads = (params[0]**2, math_ops.sqrt(params[1]))
      damping = 0.5
      block.instantiate_factors((grads,), damping)

      cov = array_ops.reshape(array_ops.constant([2., 3., 4.]), [-1, 1])
      sess.run(state_ops.assign(block._factor._cov, cov))
      sess.run(block._factor.make_inverse_update_ops())

      v_flat = np.array([4., 5., 6.], dtype=np.float32)
      vector = utils.column_to_tensors(params, array_ops.constant(v_flat))
      output = block.multiply_inverse(vector)
      output_flat = sess.run(utils.tensors_to_column(output)).ravel()

      full = sess.run(block.full_fisher_block())
      explicit = np.dot(np.linalg.inv(full + damping * np.eye(3)), v_flat)

      self.assertAllClose(output_flat, explicit)


class FullyConnectedKFACBasicFBTest(test.TestCase):

  def testFullyConnectedKFACBasicFBInit(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(200)
      inputs = array_ops.constant([1., 2.])
      outputs = array_ops.constant([3., 4.])
      block = fb.FullyConnectedKFACBasicFB(lc.LayerCollection(), inputs,
                                           outputs)

      self.assertAllEqual(outputs, block.tensors_to_compute_grads())

  def testInstantiateFactorsHasBias(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(200)
      inputs = array_ops.constant([[1., 2.], [3., 4.]])
      outputs = array_ops.constant([[3., 4.], [5., 6.]])
      block = fb.FullyConnectedKFACBasicFB(
          lc.LayerCollection(), inputs, outputs, has_bias=True)

      grads = outputs**2
      block.instantiate_factors((grads,), 0.5)

  def testInstantiateFactorsNoBias(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(200)
      inputs = array_ops.constant([[1., 2.], [3., 4.]])
      outputs = array_ops.constant([[3., 4.], [5., 6.]])
      block = fb.FullyConnectedKFACBasicFB(
          lc.LayerCollection(), inputs, outputs, has_bias=False)

      grads = outputs**2
      block.instantiate_factors((grads,), 0.5)

  def testMultiplyInverseTuple(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      inputs = array_ops.constant([[1., 2., 3.], [3., 4., 5.], [5., 6., 7.]])
      outputs = array_ops.constant([[3., 4.], [5., 6.]])
      block = fb.FullyConnectedKFACBasicFB(
          lc.LayerCollection(), inputs, outputs, has_bias=False)
      grads = outputs**2
      block.instantiate_factors((grads,), 0.5)

      # Make sure our inverse is something other than the identity.
      sess.run(tf_variables.global_variables_initializer())
      sess.run(block._input_factor.make_inverse_update_ops())
      sess.run(block._output_factor.make_inverse_update_ops())

      vector = (np.arange(2, 6).reshape(2, 2).astype(np.float32), np.arange(
          1, 3).reshape(2, 1).astype(np.float32))
      output = block.multiply_inverse((array_ops.constant(vector[0]),
                                       array_ops.constant(vector[1])))

      output = sess.run(output)
      self.assertAllClose([[0.686291, 1.029437], [1.372583, 1.715729]],
                          output[0])
      self.assertAllClose([0.343146, 0.686291], output[1])

  def testMultiplyInverseNotTuple(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      inputs = array_ops.constant([[1., 2.], [3., 4.]])
      outputs = array_ops.constant([[3., 4.], [5., 6.]])
      block = fb.FullyConnectedKFACBasicFB(
          lc.LayerCollection(), inputs, outputs, has_bias=False)
      grads = outputs**2
      block.instantiate_factors((grads,), 0.5)

      # Make sure our inverse is something other than the identity.
      sess.run(tf_variables.global_variables_initializer())
      sess.run(block._input_factor.make_inverse_update_ops())
      sess.run(block._output_factor.make_inverse_update_ops())

      vector = np.arange(2, 6).reshape(2, 2).astype(np.float32)
      output = block.multiply_inverse(array_ops.constant(vector))

      self.assertAllClose([[0.686291, 1.029437], [1.372583, 1.715729]],
                          sess.run(output))

  def testMultiplyInverseAgainstExplicit(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      input_dim, output_dim = 3, 2
      inputs = array_ops.zeros([32, input_dim])
      outputs = array_ops.zeros([32, output_dim])
      params = array_ops.zeros([input_dim, output_dim])
      block = fb.FullyConnectedKFACBasicFB(
          lc.LayerCollection(), inputs, outputs, has_bias=False)
      grads = outputs**2
      damping = 0.  # This test is only valid without damping.
      block.instantiate_factors((grads,), damping)

      sess.run(state_ops.assign(block._input_factor._cov, _make_psd(3)))
      sess.run(state_ops.assign(block._output_factor._cov, _make_psd(2)))
      sess.run(block._input_factor.make_inverse_update_ops())
      sess.run(block._output_factor.make_inverse_update_ops())

      v_flat = np.arange(6, dtype=np.float32)
      vector = utils.column_to_tensors(params, array_ops.constant(v_flat))
      output = block.multiply_inverse(vector)
      output_flat = sess.run(utils.tensors_to_column(output)).ravel()

      full = sess.run(block.full_fisher_block())
      explicit = np.dot(np.linalg.inv(full + damping * np.eye(6)), v_flat)

      self.assertAllClose(output_flat, explicit)


class ConvKFCBasicFBTest(test.TestCase):

  def _testConvKFCBasicFBInitParams(self, params):
    with ops.Graph().as_default():
      random_seed.set_random_seed(200)
      if isinstance(params, (list, tuple)):
        params = [array_ops.constant(param) for param in params]
      else:
        params = array_ops.constant(params)
      inputs = random_ops.random_normal((2, 2, 2))
      outputs = random_ops.random_normal((2, 2, 2))
      block = fb.ConvKFCBasicFB(lc.LayerCollection(), params, inputs, outputs,
                                [1, 1, 1], 'SAME')

      self.assertAllEqual(outputs, block.tensors_to_compute_grads())

  def testConvKFCBasicFBInitParamsParamsTuple(self):
    self._testConvKFCBasicFBInitParams([np.array([1., 2.]), np.array(3.)])

  def testConvKFCBasicFBInitParamsParamsSingle(self):
    self._testConvKFCBasicFBInitParams([np.array([1., 2.])])

  def testMultiplyInverseTuple(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      params = random_ops.random_normal((2, 2, 2, 2))
      inputs = random_ops.random_normal((2, 2, 2, 2))
      outputs = random_ops.random_normal((2, 2, 2, 2))
      block = fb.ConvKFCBasicFB(lc.LayerCollection(), params, inputs, outputs,
                                (1, 1, 1, 1), 'SAME')
      grads = outputs**2
      block.instantiate_factors((grads,), 0.5)

      # Make sure our inverse is something other than the identity.
      sess.run(tf_variables.global_variables_initializer())
      sess.run(block._input_factor.make_inverse_update_ops())
      sess.run(block._output_factor.make_inverse_update_ops())

      vector = (np.arange(1, 15).reshape(7, 2).astype(np.float32), np.arange(
          2, 4).reshape(2, 1).astype(np.float32))
      output = block.multiply_inverse((array_ops.constant(vector[0]),
                                       array_ops.constant(vector[1])))

      output = sess.run(output)
      self.assertAllClose([0.136455, 0.27291], output[0][0])
      self.assertAllClose([0.27291, 0.409365], output[1])

  def testMultiplyInverseNotTuple(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      params = random_ops.random_normal((2, 2, 2, 2))
      inputs = random_ops.random_normal((2, 2, 2, 2))
      outputs = random_ops.random_normal((2, 2, 2, 2))
      block = fb.ConvKFCBasicFB(lc.LayerCollection(), params, inputs, outputs,
                                (1, 1, 1, 1), 'SAME')
      self.assertFalse(block._has_bias)
      grads = outputs**2
      block.instantiate_factors((grads,), 0.5)

      # Make sure our inverse is something other than the identity.
      sess.run(tf_variables.global_variables_initializer())
      sess.run(block._input_factor.make_inverse_update_ops())
      sess.run(block._output_factor.make_inverse_update_ops())

      vector = np.arange(1, 17).reshape(8, 2).astype(np.float32)
      output = block.multiply_inverse(array_ops.constant(vector))

      self.assertAllClose([0.136455, 0.27291], sess.run(output)[0])

  def testMultiplyInverseNotTupleWithBias(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      params = [random_ops.random_normal((2, 2, 2, 2))]
      inputs = random_ops.random_normal((2, 2, 2, 2))
      outputs = random_ops.random_normal((2, 2, 2, 2))
      block = fb.ConvKFCBasicFB(lc.LayerCollection(), params, inputs, outputs,
                                (1, 1, 1, 1), 'SAME')
      self.assertTrue(block._has_bias)
      grads = outputs**2
      block.instantiate_factors((grads,), 0.5)

      # Make sure our inverse is something other than the identity.
      sess.run(tf_variables.global_variables_initializer())
      sess.run(block._input_factor.make_inverse_update_ops())
      sess.run(block._output_factor.make_inverse_update_ops())

      vector = np.arange(1, 19).reshape(9, 2).astype(np.float32)
      output = block.multiply_inverse(array_ops.constant(vector))

      self.assertAllClose([0.136455, 0.27291], sess.run(output)[0])

  def testMultiplyInverseAgainstExplicit(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      params = array_ops.zeros((2, 2, 2, 2))
      inputs = array_ops.zeros((2, 2, 2, 2))
      outputs = array_ops.zeros((2, 2, 2, 2))
      block = fb.ConvKFCBasicFB(lc.LayerCollection(), params, inputs, outputs,
                                (1, 1, 1, 1), 'SAME')
      grads = outputs**2
      damping = 0.  # This test is only valid without damping.
      block.instantiate_factors((grads,), damping)

      sess.run(state_ops.assign(block._input_factor._cov, _make_psd(8)))
      sess.run(state_ops.assign(block._output_factor._cov, _make_psd(2)))
      sess.run(block._input_factor.make_inverse_update_ops())
      sess.run(block._output_factor.make_inverse_update_ops())

      v_flat = np.arange(16, dtype=np.float32)
      vector = utils.column_to_tensors(params, array_ops.constant(v_flat))
      output = block.multiply_inverse(vector)
      output_flat = sess.run(utils.tensors_to_column(output)).ravel()

      full = sess.run(block.full_fisher_block())
      explicit = np.dot(np.linalg.inv(full + damping * np.eye(16)), v_flat)

      self.assertAllClose(output_flat, explicit)


if __name__ == '__main__':
  test.main()
