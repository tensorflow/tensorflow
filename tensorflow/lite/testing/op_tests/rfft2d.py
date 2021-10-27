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
"""Test configs for rfft2d."""
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import ExtraTocoOptions
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_rfft2d_tests(options):
  """Make a set of tests to do rfft2d."""

  test_parameters = [{
      "input_dtype": [tf.float32],
      "input_shape": [[8, 8], [3, 8, 8], [3, 1, 16]],
      "fft_length": [
          None, [4, 4], [4, 8], [8, 4], [8, 8], [8, 16], [16, 8], [16, 16],
          [1, 8], [1, 16]
      ]
  }]

  def build_graph(parameters):
    input_value = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    outs = tf.signal.rfft2d(input_value, fft_length=parameters["fft_length"])
    return [input_value], [outs]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(parameters["input_dtype"],
                                     parameters["input_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  extra_toco_options = ExtraTocoOptions()
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs,
                    extra_toco_options)
