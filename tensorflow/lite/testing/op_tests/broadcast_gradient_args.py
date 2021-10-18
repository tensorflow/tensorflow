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
"""Test configs for broadcast_gradient_args."""
import numpy as np
import tensorflow as tf

from tensorflow.lite.testing.zip_test_utils import ExtraTocoOptions
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_broadcast_gradient_args_tests(options):
  """Make a set of tests to do broadcast_gradient_args."""

  test_parameters = [{
      'input_case': ['ALL_EQUAL', 'ONE_DIM', 'NON_BROADCASTABLE'],
      'dtype': [tf.dtypes.int32, tf.dtypes.int64],
  }]

  def build_graph(parameters):
    """Build the op testing graph."""
    input1 = tf.compat.v1.placeholder(dtype=parameters['dtype'], name='input1')
    input2 = tf.compat.v1.placeholder(dtype=parameters['dtype'], name='input2')
    output1, output2 = tf.raw_ops.BroadcastGradientArgs(s0=input1, s1=input2)
    return [input1, input2], [output1, output2]

  def build_inputs(parameters, sess, inputs, outputs):
    dtype = parameters['dtype'].as_numpy_dtype()

    if parameters['input_case'] == 'ALL_EQUAL':
      values = [
          np.array([2, 4, 1, 3], dtype=dtype),
          np.array([2, 4, 1, 3], dtype=dtype)
      ]
    elif parameters['input_case'] == 'ONE_DIM':
      values = [
          np.array([2, 4, 1, 3], dtype=dtype),
          np.array([2, 1, 1, 3], dtype=dtype)
      ]
    elif parameters['input_case'] == 'NON_BROADCASTABLE':
      values = [
          np.array([2, 4, 1, 3], dtype=dtype),
          np.array([2, 5, 1, 3], dtype=dtype)
      ]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  extra_toco_options = ExtraTocoOptions()
  extra_toco_options.allow_custom_ops = True
  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      extra_toco_options,
      expected_tf_failures=2)
