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
"""Test configs for strided_slice operators."""
import numpy as np
import tensorflow as tf

from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import MAP_TF_TO_NUMPY_TYPE
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


def _make_strided_slice_tests(options, test_parameters, expected_tf_failures=0):
  """Utility function to make strided_slice_tests based on parameters."""

  def build_graph(parameters):
    """Build graph for stride_slice test."""
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    if parameters["constant_indices"]:
      begin = parameters["begin"]
      end = parameters["end"]
      strides = parameters["strides"]
      tensors = [input_tensor]
    else:
      begin = tf.compat.v1.placeholder(
          dtype=parameters["index_type"],
          name="begin",
          shape=[len(parameters["begin"])])
      end = tf.compat.v1.placeholder(
          dtype=parameters["index_type"],
          name="end",
          shape=[len(parameters["end"])])
      strides = None
      if parameters["strides"] is not None:
        strides = tf.compat.v1.placeholder(
            dtype=parameters["index_type"],
            name="strides",
            shape=[len(parameters["strides"])])
      tensors = [input_tensor, begin, end]
      if strides is not None:
        tensors.append(strides)

    kwargs = {}
    if parameters.get("ellipsis_mask", None):
      kwargs.update({"ellipsis_mask": parameters["ellipsis_mask"]})
    if parameters.get("new_axis_mask", None):
      kwargs.update({"new_axis_mask": parameters["new_axis_mask"]})

    out = tf.strided_slice(
        input_tensor,
        begin,
        end,
        strides,
        begin_mask=parameters["begin_mask"],
        end_mask=parameters["end_mask"],
        shrink_axis_mask=parameters["shrink_axis_mask"],
        **kwargs)
    return tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    """Build inputs for stride_slice test."""
    input_values = create_tensor_data(
        parameters["dtype"],
        parameters["input_shape"],
        min_value=-1,
        max_value=1)
    index_type = MAP_TF_TO_NUMPY_TYPE[parameters["index_type"]]
    values = [input_values]
    if not parameters["constant_indices"]:
      begin_values = np.array(parameters["begin"]).astype(index_type)
      end_values = np.array(parameters["end"]).astype(index_type)
      stride_values = (
          np.array(parameters["strides"]).astype(index_type)
          if parameters["strides"] is not None else None)
      values.append(begin_values)
      values.append(end_values)
      if stride_values is not None:
        values.append(stride_values)

    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=expected_tf_failures)


@register_make_test_function()
def make_strided_slice_tests(options):
  """Make a set of tests to do strided_slice."""

  # TODO(soroosh): add test/support for uint8.
  test_parameters = [
      # 4-D (basic cases with const/non-const indices).
      {
          "dtype": [tf.float32, tf.int32, tf.int64, tf.bool],
          "index_type": [tf.int32],
          "input_shape": [[12, 2, 2, 5]],
          "strides": [None, [2, 1, 3, 1]],
          "begin": [[0, 0, 0, 0]],
          "end": [[12, 2, 2, 5]],
          "begin_mask": [None],
          "end_mask": [None],
          "shrink_axis_mask": [None],
          "constant_indices": [False, True],
          "fully_quantize": [False],
      },
      # 4-D with non-trivial begin & end.
      {
          "dtype": [tf.float32],
          "index_type": [tf.int32],
          "input_shape": [[12, 2, 2, 5]],
          "begin": [[0, 0, 0, 0], [1, 0, 1, 0]],
          "end": [[8, 2, 2, 3], [12, 2, 2, 5]],
          "strides": [None, [2, 1, 3, 1]],
          "begin_mask": [None, 8],
          "end_mask": [None, 3],
          "shrink_axis_mask": [None, 15, -1],
          "constant_indices": [True],
          "fully_quantize": [False],
      },
      # Begin, end, strides dim are different from input shape
      {
          "dtype": [tf.float32],
          "index_type": [tf.int32],
          "input_shape": [[12, 2, 2, 5]],
          "begin": [[0]],
          "end": [[1]],
          "strides": [None, [1]],
          "begin_mask": [0],
          "end_mask": [0],
          "shrink_axis_mask": [1],
          "constant_indices": [True, False],
          "fully_quantize": [False],
      },
      # 2-D
      {
          "dtype": [tf.float32],
          "index_type": [tf.int32],
          "input_shape": [[2, 3]],
          "begin": [[0, 0]],
          "end": [[2, 2]],
          "strides": [None, [2, 2]],
          "begin_mask": [None, 1, 2],
          "end_mask": [None, 1, 2],
          "shrink_axis_mask": [None, 1, 2, 3, -1],
          "constant_indices": [False, True],
          "fully_quantize": [False],
      },
      # Negative strides
      {
          "dtype": [tf.float32],
          "index_type": [tf.int32],
          "input_shape": [[2, 3]],
          "begin": [[0, -1]],
          "end": [[2, -3]],
          "strides": [[1, -1]],
          "begin_mask": [None, 1, 2],
          "end_mask": [None, 1, 2],
          "shrink_axis_mask": [None, 1, 2, 3, -1],
          "constant_indices": [False],
          "fully_quantize": [False],
      },
      # 4-D (cases with const indices and batchsize of 1).
      {
          "dtype": [tf.float32],
          "index_type": [tf.int32],
          "input_shape": [[1, 2, 2, 5]],
          "strides": [None, [1, 1, 1, 1]],
          "begin": [[0, 0, 0, 0], [0, 1, 1, 3]],
          "end": [[1, 2, 2, 5], [1, 2, 2, 4]],
          "begin_mask": [None],
          "end_mask": [None],
          "shrink_axis_mask": [None],
          "constant_indices": [True],
          "fully_quantize": [True],
      },
      # Begin, end, strides dim are different from input shape
      {
          "dtype": [tf.float32],
          "index_type": [tf.int32],
          "input_shape": [[12, 2, 2, 5]],
          "begin": [[0]],
          "end": [[1]],
          "strides": [None, [1]],
          "begin_mask": [0],
          "end_mask": [0],
          "shrink_axis_mask": [1],
          "constant_indices": [True],
          "fully_quantize": [True],
      },
      {
          "dtype": [tf.float32],
          "index_type": [tf.int32],
          "input_shape": [[1, 1, 2]],
          "begin": [[1]],
          "end": [[0]],
          "strides": [[1]],
          "begin_mask": [0],
          "end_mask": [1],
          "shrink_axis_mask": [0],
          "constant_indices": [True, False],
          "fully_quantize": [False],
      },
      {
          "dtype": [tf.float32],
          "index_type": [tf.int32],
          "input_shape": [[1, 1, 2]],
          "begin": [[1, 0, 0]],
          "end": [[0, -1, -1]],
          "strides": [[1, 1, 1]],
          "begin_mask": [6],
          "end_mask": [7],
          "shrink_axis_mask": [0],
          "constant_indices": [True, False],
          "fully_quantize": [False],
      },
      # String input.
      {
          "dtype": [tf.string],
          "index_type": [tf.int32],
          "input_shape": [[12, 2, 2, 5]],
          "begin": [[0, 0, 0, 0]],
          "end": [[8, 2, 2, 3]],
          "strides": [[2, 1, 3, 1]],
          "begin_mask": [8],
          "end_mask": [3],
          "shrink_axis_mask": [None],
          "constant_indices": [True, False],
          "fully_quantize": [False],
      },
      # ellipsis_mask and new_axis_mask.
      {
          "dtype": [tf.float32],
          "index_type": [tf.int32],
          "input_shape": [[5, 5, 7, 7]],
          "begin": [[0, 0, 0, 0]],
          "end": [[2, 3, 4, 5]],
          "strides": [[1, 1, 1, 1]],
          "begin_mask": [0, 8],
          "end_mask": [0, 2],
          "shrink_axis_mask": [0, 4],
          "ellipsis_mask": [2, 4],
          "new_axis_mask": [1, 6],
          "constant_indices": [True],
          "fully_quantize": [False],
      },
      {
          "dtype": [tf.float32],
          "index_type": [tf.int32],
          "input_shape": [[5, 6, 7]],
          "begin": [[0, 0, 0]],
          "end": [[2, 3, 4]],
          "strides": [[1, 1, 1]],
          "begin_mask": [0],
          "end_mask": [0],
          "shrink_axis_mask": [0, 2],
          "ellipsis_mask": [2],
          "new_axis_mask": [1, 2, 3, 4, 5],
          "constant_indices": [False],
          "fully_quantize": [False],
      },
      # Shrink_axis and add_axis mask both set
      {
          "dtype": [tf.float32],
          "index_type": [tf.int32],
          "input_shape": [[6, 7, 8]],
          "begin": [[0, 0, 0, 0]],
          "end": [[2, 3, 4, 5]],
          "strides": [[1, 1, 1, 1]],
          "begin_mask": [0],
          "end_mask": [0],
          "new_axis_mask": [10],
          "shrink_axis_mask": [1],
          "constant_indices": [True],
          "fully_quantize": [False],
      },
  ]
  _make_strided_slice_tests(options, test_parameters, expected_tf_failures=29)


@register_make_test_function()
def make_strided_slice_1d_exhaustive_tests(options):
  """Make a set of exhaustive tests for 1D strided_slice."""
  test_parameters = [
      # 1-D Exhaustive
      {
          "dtype": [tf.float32],
          "index_type": [tf.int32],
          "input_shape": [[3]],
          "begin": [[-2], [-1], [0], [1], [2]],
          "end": [[-2], [-1], [0], [1], [2]],
          "strides": [[-2], [-1], [1], [2]],
          "begin_mask": [0, 1],
          "end_mask": [0, 1],
          "shrink_axis_mask": [0],
          "constant_indices": [False],
      },
  ]
  _make_strided_slice_tests(options, test_parameters)
