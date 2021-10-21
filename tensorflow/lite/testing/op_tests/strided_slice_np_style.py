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
"""Test configs for strided_slice_np_style."""
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


# TODO(b/137615945): Expand the test coverage of this one and remove the old
# ones.
@register_make_test_function()
def make_strided_slice_np_style_tests(options):
  """Make a set of tests to test strided_slice in np style."""

  test_parameters = [
      {
          "dtype": [tf.float32],
          "shape": [[12, 7], [33, 1]],
          "spec": [[slice(3, 7, 2), slice(None)],
                   [tf.newaxis,
                    slice(3, 7, 1), tf.newaxis,
                    slice(None)], [slice(1, 5, 1), slice(None)]],
      },
      # 1-D case
      {
          "dtype": [tf.float32],
          "shape": [[44]],
          "spec": [[slice(3, 7, 2)], [tf.newaxis, slice(None)]],
      },
      # Shrink mask.
      {
          "dtype": [tf.float32],
          "shape": [[21, 15, 7]],
          "spec": [[slice(3, 7, 2), slice(None), 2]],
      },
      # Ellipsis 3d.
      {
          "dtype": [tf.float32],
          "shape": [[21, 15, 7]],
          "spec": [[slice(3, 7, 2), Ellipsis],
                   [slice(1, 11, 3), Ellipsis,
                    slice(3, 7, 2)]],
      },
      # Ellipsis 4d.
      {
          "dtype": [tf.float32],
          "shape": [[21, 15, 7, 9]],
          "spec": [[slice(3, 7, 2), Ellipsis]],
      },
      # Ellipsis 5d.
      {
          "dtype": [tf.float32],
          "shape": [[11, 21, 15, 7, 9]],
          "spec": [[
              slice(3, 7, 2),
              slice(None),
              slice(None),
              slice(None),
              slice(None)
          ]],
      },
      # Ellipsis + Shrink Mask
      {
          "dtype": [tf.float32],
          "shape": [[22, 15, 7]],
          "spec": [
              [
                  2,  # shrink before ellipsis
                  Ellipsis
              ],
          ],
      },
      # Ellipsis + New Axis Mask
      {
          "dtype": [tf.float32],
          "shape": [[23, 15, 7]],
          "spec": [
              [
                  tf.newaxis,  # new_axis before ellipsis
                  slice(3, 7, 2),
                  slice(None),
                  Ellipsis
              ],
              [
                  tf.newaxis,  # new_axis after (and before) ellipsis
                  slice(3, 7, 2),
                  slice(None),
                  Ellipsis,
                  tf.newaxis
              ]
          ],
      },
  ]

  if options.use_experimental_converter:
    # The case when Ellipsis is expanded to multiple dimension is only supported
    # by MLIR converter (b/183902491).
    test_parameters = test_parameters + [
        # Ellipsis 3d.
        {
            "dtype": [tf.float32],
            "shape": [[21, 15, 7]],
            "spec": [[Ellipsis, slice(3, 7, 2)]],
        },
        # Ellipsis 4d.
        {
            "dtype": [tf.float32],
            "shape": [[21, 15, 7, 9]],
            "spec": [[Ellipsis, slice(3, 7, 2)],
                     [slice(1, 11, 3), Ellipsis,
                      slice(3, 7, 2)]],
        },
        # Ellipsis 5d.
        {
            "dtype": [tf.float32],
            "shape": [[11, 21, 15, 7, 9]],
            "spec": [[Ellipsis, slice(3, 7, 2)]],
        },
        # Ellipsis + Shrink Mask
        {
            "dtype": [tf.float32],
            "shape": [[22, 15, 7]],
            "spec": [[
                Ellipsis,  # shrink after ellipsis
                2
            ]],
        },
    ]

  def build_graph(parameters):
    """Build a simple graph with np style strided_slice."""
    input_value = tf.compat.v1.placeholder(
        dtype=parameters["dtype"], shape=parameters["shape"])
    out = input_value.__getitem__(parameters["spec"])
    return [input_value], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(parameters["dtype"], parameters["shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
