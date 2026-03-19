# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for strided_slice operators."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


def _make_shape_to_strided_slice_test(options,
                                      test_parameters,
                                      expected_tf_failures=0):
  """Utility function to make shape_to_strided_slice_tests."""

  def build_graph(parameters):
    """Build graph for shape_stride_slice test."""
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["dynamic_input_shape"])
    begin = parameters["begin"]
    end = parameters["end"]
    strides = parameters["strides"]
    tensors = [input_tensor]
    out = tf.strided_slice(
        tf.shape(input=input_tensor),
        begin,
        end,
        strides,
        begin_mask=parameters["begin_mask"],
        end_mask=parameters["end_mask"])
    return tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    """Build inputs for stride_slice test."""
    input_values = create_tensor_data(
        parameters["dtype"],
        parameters["input_shape"],
        min_value=-1,
        max_value=1)
    values = [input_values]

    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=expected_tf_failures)


@register_make_test_function()
def make_shape_to_strided_slice_tests(options):
  """Make a set of tests to do shape op into strided_slice."""

  test_parameters = [
      # Test dynamic shape into strided slice quantization works.
      {
          "dtype": [tf.float32],
          "dynamic_input_shape": [[None, 2, 2, 5]],
          "input_shape": [[12, 2, 2, 5]],
          "strides": [[1]],
          "begin": [[0]],
          "end": [[1]],
          "begin_mask": [0],
          "end_mask": [0],
          "fully_quantize": [False, True],
          "dynamic_range_quantize": [False],
      },
      {
          "dtype": [tf.float32],
          "dynamic_input_shape": [[None, 2, 2, 5]],
          "input_shape": [[12, 2, 2, 5]],
          "strides": [[1]],
          "begin": [[0]],
          "end": [[1]],
          "begin_mask": [0],
          "end_mask": [0],
          "fully_quantize": [False],
          "dynamic_range_quantize": [True],
      },
  ]
  _make_shape_to_strided_slice_test(
      options, test_parameters, expected_tf_failures=0)
