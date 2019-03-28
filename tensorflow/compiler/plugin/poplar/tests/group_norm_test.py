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
"""Tests for the LSTM cell and layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import test_utils as tu

from tensorflow.python.client import session as session_lib
from tensorflow.compiler.plugin.poplar.ops import gen_popnn_ops
from tensorflow.python.platform import googletest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent

# These tests are implemented based on the Layers implementation of the group
# norm - tensorflow/contrib/layers/python/layers/normalization.py.

NAMED_GROUP_NORM_TESTCASES = (
    {
        "testcase_name": "xsmall",
        "batch_size": 1,
        "num_channels": 4,
        "num_groups": 2,
        "epsilon": 0.5,
        "dims": [1, 1]
    },
    {
        "testcase_name": "small",
        "batch_size": 2,
        "num_channels": 4,
        "num_groups": 2,
        "epsilon": 0.001,
        "dims": [4, 4]
    },
    {
        "testcase_name": "medium",
        "batch_size": 2,
        "num_channels": 8,
        "num_groups": 4,
        "epsilon": 0.0015,
        "dims": [128, 128]
    },
    {
        "testcase_name": "large",
        "batch_size": 4,
        "num_channels": 16,
        "num_groups": 4,
        "epsilon": 0.0015,
        "dims": [256, 256]
    },
    {
        "testcase_name": "number_of_groups_equals_channels",
        "batch_size": 4,
        "num_channels": 8,
        "num_groups": 8,
        "epsilon": 0.0015,
        "dims": [128, 128]
    },
    {
        "testcase_name": "number_of_groups_is_one",
        "batch_size": 1,
        "num_channels": 2,
        "num_groups": 1,
        "epsilon": 0.0015,
        "dims": [4, 4]
    },
)

training_abs_tolerance = 0.01
training_rel_tolerance = 1.0
dataType = np.float32


class GroupNormTest(test.TestCase, parameterized.TestCase):
  def _refGroupNormFwd(self,
                       inputs,
                       gamma,
                       beta,
                       groups,
                       mean=None,
                       inv_std_dev=None,
                       epsilon=0.0015,
                       data_format="NHWC"):
    if data_format == "NHWC":
      feature_index = 3
    elif data_format == "NCHW":
      feature_index = 1
    else:
      raise Exception("Unsupported data format " + data_format)

    num_channels = inputs.shape[feature_index]
    group_size = num_channels // groups
    original_shape = inputs.shape

    # Implementation detail - in Poplibs group norm, the groups are not
    # contiguous, but strided - we replicate that here
    # Move the channels to the first dimension for inputs, gamma and beta
    inputs = np.swapaxes(inputs, 0, feature_index)

    reshuffled_inputs = np.empty(inputs.shape, inputs.dtype)
    reshuffled_gamma = np.empty(gamma.shape, gamma.dtype)
    reshuffled_beta = np.empty(beta.shape, beta.dtype)

    for from_idx in range(num_channels):
      to_idx = (from_idx % groups) * group_size + from_idx // groups
      reshuffled_inputs[to_idx] = inputs[from_idx]
      reshuffled_gamma[to_idx] = gamma[from_idx]
      reshuffled_beta[to_idx] = beta[from_idx]
    inputs = np.swapaxes(reshuffled_inputs, 0, feature_index)
    gamma = reshuffled_gamma
    beta = reshuffled_beta

    if feature_index == 1:
      N, C, H, W = inputs.shape
      inputs = np.reshape(inputs, [N, groups, C // groups, H, W])
      gamma = np.reshape(gamma, [1, C, 1, 1])
      beta = np.reshape(beta, [1, C, 1, 1])
      moments_axes = (feature_index + 1, 3, 4)

      if mean is not None and inv_std_dev is not None:
        mean = np.reshape(mean, [N, groups, 1, 1, 1])
        inv_std_dev = np.reshape(inv_std_dev, [N, groups, 1, 1, 1])
    else:
      N, H, W, C = inputs.shape
      inputs = np.reshape(inputs, [N, H, W, groups, C // groups])
      gamma = np.reshape(gamma, [1, 1, 1, C])
      beta = np.reshape(beta, [1, 1, 1, C])
      moments_axes = (1, 2, feature_index + 1)

      if mean is not None and inv_std_dev is not None:
        mean = np.reshape(mean, [N, 1, 1, groups, 1])
        inv_std_dev = np.reshape(inv_std_dev, [N, 1, 1, groups, 1])

    if mean is None and inv_std_dev is None:
      mean = np.mean(inputs, moments_axes, dtype=np.float32, keepdims=True)
      variance = np.mean(
          np.power(inputs - mean, 2),
          moments_axes,
          dtype=np.float32,
          keepdims=True)
    else:
      variance = np.power(inv_std_dev, -2) - epsilon

    input_whitened = (inputs - mean) * np.power(variance + epsilon, -0.5)
    input_whitened = np.reshape(input_whitened, original_shape)
    output = input_whitened * gamma + beta

    # Undo the shuffle.
    output = np.swapaxes(output, 0, feature_index)

    reshuffled_output = np.empty(output.shape, output.dtype)
    for to_idx in range(num_channels):
      from_idx = (to_idx % groups) * group_size + to_idx // groups
      reshuffled_output[to_idx] = output[from_idx]
    inv_std_dev = np.power(variance + epsilon, -0.5)
    return (np.swapaxes(reshuffled_output, 0, feature_index),
            np.reshape(np.squeeze(mean), (mean.size)),
            np.reshape(np.squeeze(inv_std_dev), (inv_std_dev.size)))

  def _refGroupNormStatistics(self,
                              inputs,
                              groups,
                              epsilon=0.0015,
                              data_format="NHWC"):
    if data_format == "NHWC":
      feature_index = 3
    elif data_format == "NCHW":
      feature_index = 1
    else:
      raise Exception("Unsupported data format " + data_format)

    num_channels = inputs.shape[feature_index]
    group_size = num_channels // groups
    original_shape = inputs.shape

    # Implementation detail - in Poplibs group norm, the groups are not
    # contiguous, but strided - we replicate that here
    # Move the channels to the first dimension for inputs, gamma and beta
    inputs = np.swapaxes(inputs, 0, feature_index)

    reshuffled_inputs = np.empty(inputs.shape, inputs.dtype)

    for from_idx in range(num_channels):
      to_idx = (from_idx % groups) * group_size + from_idx // groups
      reshuffled_inputs[to_idx] = inputs[from_idx]
    inputs = np.swapaxes(reshuffled_inputs, 0, feature_index)

    if feature_index == 1:
      N, C, H, W = inputs.shape
      inputs = np.reshape(inputs, [N, groups, C // groups, H, W])
      moments_axes = (feature_index + 1, 3, 4)

    else:
      N, H, W, C = inputs.shape
      inputs = np.reshape(inputs, [N, H, W, groups, C // groups])
      moments_axes = (1, 2, feature_index + 1)

    mean = np.mean(inputs, moments_axes, dtype=np.float32, keepdims=True)
    variance = np.mean(
        np.power(inputs - mean, 2),
        moments_axes,
        dtype=np.float32,
        keepdims=True)
    inv_std_dev = np.power(variance + epsilon, -0.5)

    return (np.reshape(np.squeeze(mean), (mean.size)),
            np.reshape(np.squeeze(inv_std_dev), (inv_std_dev.size)))

  def _implGroupNormInf(self,
                        inputs,
                        gamma,
                        beta,
                        groups,
                        mean,
                        inv_std_dev,
                        epsilon=0.0015,
                        data_format="NHWC"):
    if data_format != "NHWC" and data_format != "NCHW":
      raise Exception("Unsupported data format " + data_format)

    with ops.device("/device:IPU:0"):
      pinputs = array_ops.placeholder(dataType, inputs.shape, name="inputs")
      pgamma = array_ops.placeholder(dataType, gamma.shape, name="gamma")
      pbeta = array_ops.placeholder(dataType, beta.shape, name="beta")
      pmean = array_ops.placeholder(dataType, mean.shape, name="mean")
      pinv_std_dev = array_ops.placeholder(
          dataType, inv_std_dev.shape, name="inv_std_dev")
      output = gen_popnn_ops.popnn_group_norm_inference(
          inputs=pinputs,
          gamma=pgamma,
          beta=pbeta,
          mean=pmean,
          inv_std_dev=pinv_std_dev,
          data_format=data_format,
          epsilon=epsilon,
          num_groups=groups)

    with session_lib.Session() as sess:
      fd = {
          pinputs: inputs,
          pgamma: gamma,
          pbeta: beta,
          pmean: mean,
          pinv_std_dev: inv_std_dev,
      }
      return sess.run(output, fd)

  def _implGroupNormTraining(self,
                             inputs,
                             gamma,
                             beta,
                             groups,
                             epsilon=0.0015,
                             data_format="NHWC"):
    if data_format != "NHWC" and data_format != "NCHW":
      raise Exception("Unsupported data format " + data_format)

    with ops.device("/device:IPU:0"):
      pinputs = array_ops.placeholder(dataType, inputs.shape, name="inputs")
      pgamma = array_ops.placeholder(dataType, gamma.shape, name="gamma")
      pbeta = array_ops.placeholder(dataType, beta.shape, name="beta")
      norm, mean, inv_std_dev = gen_popnn_ops.popnn_group_norm_training(
          inputs=pinputs,
          gamma=pgamma,
          beta=pbeta,
          data_format=data_format,
          epsilon=epsilon,
          num_groups=groups)

    with session_lib.Session() as sess:
      fd = {
          pinputs: inputs,
          pgamma: gamma,
          pbeta: beta,
      }
      return sess.run((norm, mean, inv_std_dev), fd)

  def _implGroupNormStatistics(self,
                               inputs,
                               groups,
                               epsilon=0.0015,
                               data_format="NHWC"):
    if data_format != "NHWC" and data_format != "NCHW":
      raise Exception("Unsupported data format " + data_format)

    with ops.device("/device:IPU:0"):
      pinputs = array_ops.placeholder(dataType, inputs.shape, name="inputs")
      mean, inv_std_dev = gen_popnn_ops.popnn_group_norm_statistics(
          inputs=pinputs,
          data_format=data_format,
          epsilon=epsilon,
          num_groups=groups)

    with session_lib.Session() as sess:
      fd = {
          pinputs: inputs,
      }
      return sess.run((mean, inv_std_dev), fd)

  @parameterized.named_parameters(*NAMED_GROUP_NORM_TESTCASES)
  def testGroupNormInference(self, batch_size, num_channels, num_groups, dims,
                             epsilon):
    np.random.seed(12)
    for data_format in ["NHWC", "NCHW"]:
      if data_format == "NHWC":
        acts_shape = [batch_size, dims[0], dims[1], num_channels]
      elif data_format == "NCHW":
        acts_shape = [batch_size, num_channels, dims[0], dims[1]]
      else:
        raise Exception("Unsupported data format " + data_format)

      gamma_beta_shape = [num_channels]
      mean_inv_std_dev_shape = [batch_size * num_groups]

      activations = np.random.rand(*acts_shape).astype(dataType)
      gamma = np.random.rand(*gamma_beta_shape).astype(dataType)
      beta = np.random.rand(*gamma_beta_shape).astype(dataType)
      mean = np.random.rand(*mean_inv_std_dev_shape).astype(dataType)
      inv_std_dev = np.random.rand(*mean_inv_std_dev_shape).astype(dataType)
      result = self._implGroupNormInf(
          activations,
          gamma,
          beta,
          num_groups,
          mean,
          inv_std_dev,
          epsilon=epsilon,
          data_format=data_format)

      expected, _, _ = self._refGroupNormFwd(
          activations,
          gamma,
          beta,
          num_groups,
          mean=mean,
          inv_std_dev=inv_std_dev,
          epsilon=epsilon,
          data_format=data_format)

      self.assertAllClose(expected, result)

  @parameterized.named_parameters(*NAMED_GROUP_NORM_TESTCASES)
  def testGroupNormTraining(self, batch_size, num_channels, num_groups, dims,
                            epsilon):
    np.random.seed(48)
    for data_format in ["NHWC", "NCHW"]:
      if data_format == "NHWC":
        acts_shape = [batch_size, dims[0], dims[1], num_channels]
      elif data_format == "NCHW":
        acts_shape = [batch_size, num_channels, dims[0], dims[1]]
      else:
        raise Exception("Unsupported data format " + data_format)

      gamma_beta_shape = [num_channels]

      activations = np.random.rand(*acts_shape).astype(dataType)
      gamma = np.random.rand(*gamma_beta_shape).astype(dataType)
      beta = np.random.rand(*gamma_beta_shape).astype(dataType)
      norm, mean, inv_std_dev = self._implGroupNormTraining(
          activations,
          gamma,
          beta,
          num_groups,
          epsilon=epsilon,
          data_format=data_format)

      expected_norm, expected_mean, expected_inv_std_dev = self._refGroupNormFwd(
          activations,
          gamma,
          beta,
          num_groups,
          epsilon=epsilon,
          data_format=data_format)
      self.assertAllClose(
          expected_mean,
          mean,
          rtol=training_rel_tolerance,
          atol=training_abs_tolerance)
      self.assertAllClose(
          expected_inv_std_dev,
          inv_std_dev,
          rtol=training_rel_tolerance,
          atol=training_abs_tolerance)
      self.assertAllClose(
          expected_norm,
          norm,
          rtol=training_rel_tolerance,
          atol=training_abs_tolerance)

  @parameterized.named_parameters(*NAMED_GROUP_NORM_TESTCASES)
  def testGroupNormStatistics(self, batch_size, num_channels, num_groups, dims,
                              epsilon):
    np.random.seed(48)
    for data_format in ["NHWC", "NCHW"]:
      if data_format == "NHWC":
        acts_shape = [batch_size, dims[0], dims[1], num_channels]
      elif data_format == "NCHW":
        acts_shape = [batch_size, num_channels, dims[0], dims[1]]
      else:
        raise Exception("Unsupported data format " + data_format)

      activations = np.random.rand(*acts_shape).astype(dataType)

      mean, inv_std_dev = self._implGroupNormStatistics(
          activations, num_groups, epsilon=epsilon, data_format=data_format)

      expected_mean, expected_inv_std_dev = self._refGroupNormStatistics(
          activations, num_groups, epsilon=epsilon, data_format=data_format)
      self.assertAllClose(
          expected_mean,
          mean,
          rtol=training_rel_tolerance,
          atol=training_abs_tolerance)
      self.assertAllClose(
          expected_inv_std_dev,
          inv_std_dev,
          rtol=training_rel_tolerance,
          atol=training_abs_tolerance)


if __name__ == "__main__":
  googletest.main()
