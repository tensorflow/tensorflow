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
"""Test configs for irfft2d."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import ExtraTocoOptions
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_irfft2d_tests(options):
  """Make a set of tests to do irfft2d."""

  test_parameters = [{
      "input_dtype": [tf.complex64],
      "input_shape": [[4, 3]],
      "fft_length": [[4, 4], [2, 2], [2, 4]]
  }, {
      "input_dtype": [tf.complex64],
      "input_shape": [[3, 8, 5]],
      "fft_length": [[2, 4], [2, 8], [8, 8]]
  }, {
      "input_dtype": [tf.complex64],
      "input_shape": [[3, 1, 9]],
      "fft_length": [[1, 8], [1, 16]]
  }]

  def build_graph(parameters):
    input_value = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    outs = tf.signal.irfft2d(input_value, fft_length=parameters["fft_length"])
    return [input_value], [outs]

  def build_inputs(parameters, sess, inputs, outputs):
    rfft_length = []
    rfft_length.append(parameters["input_shape"][-2])
    rfft_length.append((parameters["input_shape"][-1] - 1) * 2)
    rfft_input = create_tensor_data(np.float32, parameters["input_shape"])
    rfft_result = np.fft.rfft2(rfft_input, rfft_length)

    return [rfft_result], sess.run(
        outputs, feed_dict=dict(zip(inputs, [rfft_result])))

  extra_toco_options = ExtraTocoOptions()
  extra_toco_options.allow_custom_ops = True
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs,
                    extra_toco_options)
