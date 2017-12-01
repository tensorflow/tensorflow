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


class FullyConnectedDiagonalFB(test.TestCase):

  def setUp(self):
    super(FullyConnectedDiagonalFB, self).setUp()

    self.batch_size = 4
    self.input_size = 6
    self.output_size = 3

    self.inputs = np.random.randn(self.batch_size, self.input_size).astype(
        np.float32)
    self.outputs = np.zeros([self.batch_size, self.output_size]).astype(
        np.float32)
    self.output_grads = np.random.randn(self.batch_size,
                                        self.output_size).astype(np.float32)
    self.w = np.random.randn(self.input_size, self.output_size).astype(
        np.float32)
    self.b = np.random.randn(self.output_size).astype(np.float32)

  def fisherApprox(self, has_bias=False):
    """Fisher approximation using default inputs."""
    if has_bias:
      inputs = np.concatenate(
          [self.inputs, np.ones([self.batch_size, 1])], axis=1)
    else:
      inputs = self.inputs
    return self.buildDiagonalFisherApproximation(inputs, self.output_grads)

  def buildDiagonalFisherApproximation(self, inputs, output_grads):
    """Builds explicit diagonal Fisher approximation.

    Fisher's diagonal is (d loss / d w)'s elements squared for
      d/dw = E[outer(input, output_grad)]

    where the expectation is taken over examples.

    Args:
      inputs: np.array of shape [batch_size, input_size].
      output_grads: np.array of shape [batch_size, output_size].

    Returns:
      Diagonal np.array of shape [num_params, num_params] for num_params =
      input_size * output_size.
    """
    batch_size = inputs.shape[0]
    assert output_grads.shape[0] == batch_size
    input_size = inputs.shape[1]
    output_size = output_grads.shape[1]
    fisher_diag = np.zeros((input_size, output_size))
    for i in range(batch_size):
      fisher_diag += np.square(np.outer(inputs[i], output_grads[i]))
    return np.diag(fisher_diag.flatten()) / batch_size

  def testMultiply(self):
    result, _ = self.runFisherBlockOps(self.w, [self.inputs], [self.outputs],
                                       [self.output_grads])

    # Construct Fisher-vector product.
    expected_result = self.fisherApprox().dot(self.w.flatten())
    expected_result = expected_result.reshape(
        [self.input_size, self.output_size])

    self.assertAllClose(expected_result, result)

  def testMultiplyInverse(self):
    _, result = self.runFisherBlockOps(self.w, [self.inputs], [self.outputs],
                                       [self.output_grads])

    # Construct inverse Fisher-vector product.
    expected_result = np.linalg.inv(self.fisherApprox()).dot(self.w.flatten())
    expected_result = expected_result.reshape(
        [self.input_size, self.output_size])

    self.assertAllClose(expected_result, result)

  def testRegisterAdditionalMinibatch(self):
    """Ensure 1 big minibatch and 2 small minibatches are equivalent."""
    multiply_result_big, multiply_inverse_result_big = self.runFisherBlockOps(
        self.w, [self.inputs], [self.outputs], [self.output_grads])
    multiply_result_small, multiply_inverse_result_small = (
        self.runFisherBlockOps(self.w,
                               np.split(self.inputs, 2),
                               np.split(self.outputs, 2),
                               np.split(self.output_grads, 2)))

    self.assertAllClose(multiply_result_big, multiply_result_small)
    self.assertAllClose(multiply_inverse_result_big,
                        multiply_inverse_result_small)

  def testMultiplyHasBias(self):
    result, _ = self.runFisherBlockOps((self.w, self.b), [self.inputs],
                                       [self.outputs], [self.output_grads])
    expected_result = self.fisherApprox(True).dot(
        np.concatenate([self.w.flatten(), self.b.flatten()]))
    expected_result = expected_result.reshape(
        [self.input_size + 1, self.output_size])
    expected_result = (expected_result[:-1], expected_result[-1])

    self.assertEqual(len(result), 2)
    self.assertAllClose(expected_result[0], result[0])
    self.assertAllClose(expected_result[1], result[1])

  def runFisherBlockOps(self, params, inputs, outputs, output_grads):
    """Run Ops guaranteed by FisherBlock interface.

    Args:
      params: Tensor or 2-tuple of Tensors. Represents weights or weights and
        bias of this layer.
      inputs: list of Tensors of shape [batch_size, input_size]. Inputs to
        layer.
      outputs: list of Tensors of shape [batch_size, output_size].
        Preactivations produced by layer.
      output_grads: list of Tensors of shape [batch_size, output_size].
        Gradient of loss with respect to 'outputs'.

    Returns:
      multiply_result: Result of FisherBlock.multiply(params)
      multiply_inverse_result: Result of FisherBlock.multiply_inverse(params)
    """
    with ops.Graph().as_default(), self.test_session() as sess:
      inputs = as_tensors(inputs)
      outputs = as_tensors(outputs)
      output_grads = as_tensors(output_grads)
      params = as_tensors(params)

      block = fb.FullyConnectedDiagonalFB(
          lc.LayerCollection(), has_bias=isinstance(params, (tuple, list)))
      for (i, o) in zip(inputs, outputs):
        block.register_additional_minibatch(i, o)

      block.instantiate_factors((output_grads,), damping=0.0)

      sess.run(tf_variables.global_variables_initializer())
      sess.run(block._factor.make_covariance_update_op(0.0))
      multiply_result = sess.run(block.multiply(params))
      multiply_inverse_result = sess.run(block.multiply_inverse(params))

    return multiply_result, multiply_inverse_result


class FullyConnectedKFACBasicFBTest(test.TestCase):

  def testFullyConnectedKFACBasicFBInit(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(200)
      inputs = array_ops.constant([1., 2.])
      outputs = array_ops.constant([3., 4.])
      block = fb.FullyConnectedKFACBasicFB(lc.LayerCollection())
      block.register_additional_minibatch(inputs, outputs)

      self.assertAllEqual([outputs], block.tensors_to_compute_grads())

  def testInstantiateFactorsHasBias(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(200)
      inputs = array_ops.constant([[1., 2.], [3., 4.]])
      outputs = array_ops.constant([[3., 4.], [5., 6.]])
      block = fb.FullyConnectedKFACBasicFB(lc.LayerCollection(), has_bias=True)
      block.register_additional_minibatch(inputs, outputs)

      grads = outputs**2
      block.instantiate_factors(([grads],), 0.5)

  def testInstantiateFactorsNoBias(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(200)
      inputs = array_ops.constant([[1., 2.], [3., 4.]])
      outputs = array_ops.constant([[3., 4.], [5., 6.]])
      block = fb.FullyConnectedKFACBasicFB(lc.LayerCollection(), has_bias=False)
      block.register_additional_minibatch(inputs, outputs)

      grads = outputs**2
      block.instantiate_factors(([grads],), 0.5)

  def testMultiplyInverseTuple(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      inputs = array_ops.constant([[1., 2., 3.], [3., 4., 5.], [5., 6., 7.]])
      outputs = array_ops.constant([[3., 4.], [5., 6.]])
      block = fb.FullyConnectedKFACBasicFB(lc.LayerCollection(), has_bias=False)
      block.register_additional_minibatch(inputs, outputs)
      grads = outputs**2
      block.instantiate_factors(([grads],), 0.5)

      # Make sure our inverse is something other than the identity.
      sess.run(tf_variables.global_variables_initializer())
      sess.run(block._input_factor.make_inverse_update_ops())
      sess.run(block._output_factor.make_inverse_update_ops())

      vector = (
          np.arange(2, 6).reshape(2, 2).astype(np.float32),  #
          np.arange(1, 3).reshape(2, 1).astype(np.float32))
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
      block = fb.FullyConnectedKFACBasicFB(lc.LayerCollection(), has_bias=False)
      block.register_additional_minibatch(inputs, outputs)
      grads = outputs**2
      block.instantiate_factors(([grads],), 0.5)

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
      block = fb.FullyConnectedKFACBasicFB(lc.LayerCollection(), has_bias=False)
      block.register_additional_minibatch(inputs, outputs)
      grads = outputs**2
      damping = 0.  # This test is only valid without damping.
      block.instantiate_factors(([grads],), damping)

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


class ConvDiagonalFBTest(test.TestCase):

  def setUp(self):
    super(ConvDiagonalFBTest, self).setUp()

    self.batch_size = 2
    self.height = 8
    self.width = 4
    self.input_channels = 6
    self.output_channels = 3
    self.kernel_size = 1

    self.inputs = np.random.randn(self.batch_size, self.height, self.width,
                                  self.input_channels).astype(np.float32)
    self.outputs = np.zeros(
        [self.batch_size, self.height, self.width,
         self.output_channels]).astype(np.float32)
    self.output_grads = np.random.randn(
        self.batch_size, self.height, self.width, self.output_channels).astype(
            np.float32)
    self.w = np.random.randn(self.kernel_size, self.kernel_size,
                             self.input_channels, self.output_channels).astype(
                                 np.float32)
    self.b = np.random.randn(self.output_channels).astype(np.float32)

  def fisherApprox(self, has_bias=False):
    """Fisher approximation using default inputs."""
    if has_bias:
      inputs = np.concatenate(
          [self.inputs,
           np.ones([self.batch_size, self.height, self.width, 1])],
          axis=-1)
    else:
      inputs = self.inputs
    return self.buildDiagonalFisherApproximation(inputs, self.output_grads,
                                                 self.kernel_size)

  def buildDiagonalFisherApproximation(self, inputs, output_grads, kernel_size):
    r"""Builds explicit diagonal Fisher approximation.

    Fisher's diagonal is (d loss / d w)'s elements squared for
      d/dw = E[\sum_{loc} outer(input_{loc}, output_grad_{loc})]

    where the expectation is taken over examples and the sum over (x, y)
    locations upon which the convolution is applied.

    Args:
      inputs: np.array of shape [batch_size, height, width, input_channels].
      output_grads: np.array of shape [batch_size, height, width,
        output_channels].
      kernel_size: int. height and width of kernel.

    Returns:
      Diagonal np.array of shape [num_params, num_params] for num_params =
      kernel_size^2 * input_channels * output_channels.
    """
    batch_size, height, width, input_channels = inputs.shape
    assert output_grads.shape[0] == batch_size
    assert output_grads.shape[1] == height
    assert output_grads.shape[2] == width
    output_channels = output_grads.shape[3]

    # If kernel_size == 1, then we don't need to worry about capturing context
    # around the pixel upon which a convolution is applied. This makes testing
    # easier.
    assert kernel_size == 1, "kernel_size != 1 isn't supported."
    num_locations = height * width
    inputs = np.reshape(inputs, [batch_size, num_locations, input_channels])
    output_grads = np.reshape(output_grads,
                              [batch_size, num_locations, output_channels])

    fisher_diag = np.zeros((input_channels, output_channels))
    for i in range(batch_size):
      # Each example's approximation is a square(sum-of-outer-products).
      example_fisher_diag = np.zeros((input_channels, output_channels))
      for j in range(num_locations):
        example_fisher_diag += np.outer(inputs[i, j], output_grads[i, j])
      fisher_diag += np.square(example_fisher_diag)

    # Normalize by batch_size (not num_locations).
    return np.diag(fisher_diag.flatten()) / batch_size

  def testMultiply(self):
    result, _ = self.runFisherBlockOps(self.w, [self.inputs], [self.outputs],
                                       [self.output_grads])

    # Construct Fisher-vector product.
    expected_result = self.fisherApprox().dot(self.w.flatten())
    expected_result = expected_result.reshape([
        self.kernel_size, self.kernel_size, self.input_channels,
        self.output_channels
    ])

    self.assertAllClose(expected_result, result)

  def testMultiplyInverse(self):
    _, result = self.runFisherBlockOps(self.w, [self.inputs], [self.outputs],
                                       [self.output_grads])

    # Construct inverse Fisher-vector product.
    expected_result = np.linalg.inv(self.fisherApprox()).dot(self.w.flatten())
    expected_result = expected_result.reshape([
        self.kernel_size, self.kernel_size, self.input_channels,
        self.output_channels
    ])

    self.assertAllClose(expected_result, result, atol=1e-3)

  def testRegisterAdditionalMinibatch(self):
    """Ensure 1 big minibatch and 2 small minibatches are equivalent."""
    multiply_result_big, multiply_inverse_result_big = self.runFisherBlockOps(
        self.w, [self.inputs], [self.outputs], [self.output_grads])
    multiply_result_small, multiply_inverse_result_small = (
        self.runFisherBlockOps(self.w,
                               np.split(self.inputs, 2),
                               np.split(self.outputs, 2),
                               np.split(self.output_grads, 2)))

    self.assertAllClose(multiply_result_big, multiply_result_small)
    self.assertAllClose(multiply_inverse_result_big,
                        multiply_inverse_result_small)

  def testMultiplyHasBias(self):
    result, _ = self.runFisherBlockOps((self.w, self.b), [self.inputs],
                                       [self.outputs], [self.output_grads])
    # Clone 'b' along 'input_channels' dimension.
    b_filter = np.tile(
        np.reshape(self.b, [1, 1, 1, self.output_channels]),
        [self.kernel_size, self.kernel_size, 1, 1])
    params = np.concatenate([self.w, b_filter], axis=2)
    expected_result = self.fisherApprox(True).dot(params.flatten())

    # Extract 'b' from concatenated parameters.
    expected_result = expected_result.reshape([
        self.kernel_size, self.kernel_size, self.input_channels + 1,
        self.output_channels
    ])
    expected_result = (expected_result[:, :, 0:-1, :], np.reshape(
        expected_result[:, :, -1, :], [self.output_channels]))

    self.assertEqual(len(result), 2)
    self.assertAllClose(expected_result[0], result[0])
    self.assertAllClose(expected_result[1], result[1])

  def runFisherBlockOps(self, params, inputs, outputs, output_grads):
    """Run Ops guaranteed by FisherBlock interface.

    Args:
      params: Tensor or 2-tuple of Tensors. Represents weights or weights and
        bias of this layer.
      inputs: list of Tensors of shape [batch_size, input_size]. Inputs to
        layer.
      outputs: list of Tensors of shape [batch_size, output_size].
        Preactivations produced by layer.
      output_grads: list of Tensors of shape [batch_size, output_size].
        Gradient of loss with respect to 'outputs'.

    Returns:
      multiply_result: Result of FisherBlock.multiply(params)
      multiply_inverse_result: Result of FisherBlock.multiply_inverse(params)
    """
    with ops.Graph().as_default(), self.test_session() as sess:
      inputs = as_tensors(inputs)
      outputs = as_tensors(outputs)
      output_grads = as_tensors(output_grads)
      params = as_tensors(params)

      block = fb.ConvDiagonalFB(
          lc.LayerCollection(), params, strides=[1, 1, 1, 1], padding='SAME')
      for (i, o) in zip(inputs, outputs):
        block.register_additional_minibatch(i, o)

      block.instantiate_factors((output_grads,), damping=0.0)

      sess.run(tf_variables.global_variables_initializer())
      sess.run(block._factor.make_covariance_update_op(0.0))
      multiply_result = sess.run(block.multiply(params))
      multiply_inverse_result = sess.run(block.multiply_inverse(params))

    return multiply_result, multiply_inverse_result


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


def as_tensors(tensor_or_tuple):
  """Converts a potentially nested tuple of np.array to Tensors."""
  if isinstance(tensor_or_tuple, (tuple, list)):
    return tuple(as_tensors(t) for t in tensor_or_tuple)
  return ops.convert_to_tensor(tensor_or_tuple)

if __name__ == '__main__':
  test.main()
