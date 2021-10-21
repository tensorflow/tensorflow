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
"""Test configs for fused_batch_norm."""
import numpy as np

import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_fused_batch_norm_tests(options):
  """Make a set of tests to do fused_batch_norm."""

  test_parameters = [{
      "dtype": [tf.float32],
      "input_shape": [[1, 1, 6, 2]],
      "epsilon": [0.001, 0.1],
      "is_training": [False],
  }]

  # Training support in MLIR converter.
  if options.use_experimental_converter:
    test_parameters = test_parameters + [
        {
            "dtype": [tf.float32],
            "input_shape": [[1, 1, 6, 2]],
            "epsilon": [0.001, 0.1],
            "is_training": [True],
        },
        {
            "dtype": [tf.float32],
            "input_shape": [[1, None, 6, 2]],
            "epsilon": [0.001, 0.1],
            "is_training": [True, False],
        },
    ]

  def build_graph(parameters):
    """Build the testing graph for fused batch normalization."""
    input_shape = parameters["input_shape"]
    scale_shape = input_shape[3]

    scale = create_tensor_data(parameters["dtype"], scale_shape)
    offset = create_tensor_data(parameters["dtype"], scale_shape)
    mean = create_tensor_data(parameters["dtype"], scale_shape)
    variance = create_tensor_data(parameters["dtype"], scale_shape)

    x = tf.compat.v1.placeholder(
        dtype=parameters["dtype"], name="x", shape=parameters["input_shape"])
    [x_norm, _, _] = tf.compat.v1.nn.fused_batch_norm(
        x,
        scale,
        offset,
        mean,
        variance,
        parameters["epsilon"],
        data_format="NHWC",
        is_training=parameters["is_training"])

    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    out = tf.add(input_tensor, x_norm)
    return [x, input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # Fill dynamic shape with a random number.
    input_shape = parameters["input_shape"]
    input_shape = [
        np.random.randint(1, 10) if v is None else v for v in input_shape
    ]

    input_values = [
        create_tensor_data(parameters["dtype"], input_shape),
        create_tensor_data(parameters["dtype"], input_shape)
    ]

    return input_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, input_values)))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
