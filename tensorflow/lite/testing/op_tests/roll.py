# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for roll."""
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import ExtraTocoOptions
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function

test_parameters = [
    # Scalar axis.
    {
        "input_dtype": [tf.float32, tf.int32],
        "input_shape": [[2, 4, 5], [3, 8, 4]],
        "shift": [1, -3, 5],
        "axis": [0, 1, 2],
    },
    # 1-D axis.
    {
        "input_dtype": [tf.float32, tf.int32],
        "input_shape": [[2, 4, 5], [3, 8, 4]],
        "shift": [[1], [-3], [5]],
        "axis": [[0], [1], [2]],
    },
    # Multiple axis.
    {
        "input_dtype": [tf.float32, tf.int32],
        "input_shape": [[2, 4, 5], [3, 8, 4]],
        "shift": [[1, 3, 2], [3, -6, 5], [-5, 7, 8]],
        "axis": [[0, 1, 2]],
    },
    # Duplicate axis.
    {
        "input_dtype": [tf.float32],
        "input_shape": [[2, 4, 5], [3, 8, 4]],
        "shift": [[1, 3, -2]],
        "axis": [[0, 1, 1]],
    },
]


@register_make_test_function()
def make_roll_with_constant_tests(options):
  """Make a set of tests to do roll with constant shift and axis."""

  def build_graph(parameters):
    input_value = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    outs = tf.roll(
        input_value, shift=parameters["shift"], axis=parameters["axis"])
    return [input_value], [outs]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(parameters["input_dtype"],
                                     parameters["input_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)


@register_make_test_function()
def make_roll_tests(options):
  """Make a set of tests to do roll."""

  ext_test_parameters = test_parameters + [
      # Scalar axis.
      {
          "input_dtype": [tf.float32, tf.int32],
          "input_shape": [[None, 8, 4]],
          "shift": [-3, 5],
          "axis": [1, 2],
      }
  ]

  def set_dynamic_shape(shape):
    return [4 if x is None else x for x in shape]

  def get_shape(param):
    if np.isscalar(param):
      return []
    return [len(param)]

  def get_value(param, dtype):
    if np.isscalar(param):
      return np.dtype(dtype).type(param)
    return np.array(param).astype(dtype)

  def build_graph(parameters):
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    shift_tensor = tf.compat.v1.placeholder(
        dtype=tf.int64, name="shift", shape=get_shape(parameters["shift"]))
    axis_tensor = tf.compat.v1.placeholder(
        dtype=tf.int64, name="axis", shape=get_shape(parameters["axis"]))
    outs = tf.roll(input_tensor, shift_tensor, axis_tensor)
    return [input_tensor, shift_tensor, axis_tensor], [outs]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(
        parameters["input_dtype"], set_dynamic_shape(parameters["input_shape"]))
    shift_value = get_value(parameters["shift"], np.int64)
    axis_value = get_value(parameters["axis"], np.int64)
    return [input_value, shift_value, axis_value], sess.run(
        outputs,
        feed_dict=dict(zip(inputs, [input_value, shift_value, axis_value])))

  extra_toco_options = ExtraTocoOptions()
  extra_toco_options.allow_custom_ops = True
  make_zip_of_tests(options, ext_test_parameters, build_graph, build_inputs,
                    extra_toco_options)
