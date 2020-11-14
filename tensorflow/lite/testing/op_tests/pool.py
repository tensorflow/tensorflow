# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for pool operators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


def make_pool_tests(pool_op_in, allow_fully_quantize=False):
  """Make a set of tests to do average pooling.

  Args:
    pool_op_in: TensorFlow pooling operation to test  i.e. `tf.nn.avg_pool2d`.
    allow_fully_quantize: bool, whether fully_quantize is allowed.

  Returns:
    A function representing the true generator (after curried pool_op_in).
  """

  pool_op = pool_op_in

  def f(options, expected_tf_failures=0):
    """Actual function that generates examples.

    Args:
      options: An Options instance.
      expected_tf_failures: number of expected tensorflow failures.
    """

    # Chose a set of parameters
    test_parameters = [
        {
            "ksize": [[2, 1, 1, 2], [1, 1, 1, 1], [1, 1, 2, 1], [1, 10, 11, 1]],
            "strides": [[2, 1, 1, 2], [1, 1, 1, 1], [1, 1, 2, 1],
                        [1, 10, 11, 1]],
            # TODO(aselle): should add a degenerate shape (e.g. [1, 0, 1, 1]).
            "input_shape": [[], [1, 1, 1, 1], [1, 15, 14, 1], [3, 15, 14, 3]],
            "padding": ["SAME", "VALID"],
            "data_format": ["NHWC"],  # TODO(aselle): NCHW  would be good
            "fully_quantize": [False],
            "quant_16x8": [False]
        },
        {
            "ksize": [[2, 1, 1, 2], [1, 1, 1, 1], [1, 1, 2, 1], [1, 10, 11, 1]],
            "strides": [[2, 1, 1, 2], [1, 1, 1, 1], [1, 1, 2, 1],
                        [1, 10, 11, 1]],
            # TODO(aselle): should add a degenerate shape (e.g. [1, 0, 1, 1]).
            "input_shape": [[], [1, 1, 1, 1], [1, 15, 14, 1], [3, 15, 14, 3]],
            "padding": ["SAME", "VALID"],
            "data_format": ["NHWC"],  # TODO(aselle): NCHW  would be good
            "fully_quantize": [True],
            "quant_16x8": [False]
        },
        {
            "ksize": [[1, 1, 1, 1]],
            "strides": [[1, 1, 1, 1]],
            "input_shape": [[1, 1, 1, 1]],
            "padding": ["SAME", "VALID"],
            "data_format": ["NHWC"],
            "fully_quantize": [True],
            "quant_16x8": [True]
        }
    ]
    # test_parameters include fully_quantize option only when
    # allow_fully_quantize is True.
    if not allow_fully_quantize:
      test_parameters = [
          test_parameter for test_parameter in test_parameters
          if True not in test_parameter["fully_quantize"]
      ]

    def build_graph(parameters):
      input_tensor = tf.compat.v1.placeholder(
          dtype=tf.float32, name="input", shape=parameters["input_shape"])
      out = pool_op(
          input_tensor,
          ksize=parameters["ksize"],
          strides=parameters["strides"],
          data_format=parameters["data_format"],
          padding=parameters["padding"])
      return [input_tensor], [out]

    def build_inputs(parameters, sess, inputs, outputs):
      if allow_fully_quantize:
        input_values = create_tensor_data(
            tf.float32, parameters["input_shape"], min_value=-1, max_value=1)
      else:
        input_values = create_tensor_data(tf.float32, parameters["input_shape"])
      return [input_values], sess.run(
          outputs, feed_dict=dict(zip(inputs, [input_values])))

    make_zip_of_tests(
        options,
        test_parameters,
        build_graph,
        build_inputs,
        expected_tf_failures=expected_tf_failures)

  return f


def make_l2_pool(input_tensor, ksize, strides, padding, data_format):
  """Given an input perform a sequence of TensorFlow ops to produce l2pool."""
  return tf.sqrt(
      tf.nn.avg_pool(
          tf.square(input_tensor),
          ksize=ksize,
          strides=strides,
          padding=padding,
          data_format=data_format))


@register_make_test_function()
def make_l2_pool_tests(options):
  make_pool_tests(make_l2_pool)(options, expected_tf_failures=80)


@register_make_test_function()
def make_avg_pool_tests(options):
  make_pool_tests(
      tf.nn.avg_pool, allow_fully_quantize=True)(
          options, expected_tf_failures=160)


@register_make_test_function()
def make_max_pool_tests(options):
  make_pool_tests(
      tf.nn.max_pool, allow_fully_quantize=True)(
          options, expected_tf_failures=160)
