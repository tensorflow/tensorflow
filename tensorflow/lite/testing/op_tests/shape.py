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
"""Test configs for shape."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_shape_tests(options):
  """Make a set of tests to do shape."""

  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32],
      "input_shape": [[1, 4]],
      "new_shape": [[1, 4], [4, 1], [2, 2]],
      "out_type": [tf.int32, tf.int64],
  }]

  def build_graph(parameters):
    """Build the shape op testing graph."""
    # Note that we intentionally leave out the shape from the input placeholder
    # to prevent the Shape operation from being optimized out during conversion.
    # TODO(haoliang): Test shape op directly after we have better support for
    # dynamic input. Currently we need to introduce a Reshape op to prevent
    # shape being constant-folded.
    input_value = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        shape=parameters["input_shape"],
        name="input")
    shape_of_new_shape = [len(parameters["new_shape"])]
    new_shape = tf.compat.v1.placeholder(
        dtype=tf.int32, shape=shape_of_new_shape, name="new_shape")
    reshaped = tf.reshape(input_value, shape=new_shape)
    out = tf.shape(reshaped, out_type=parameters["out_type"])
    return [input_value, new_shape], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(parameters["input_dtype"],
                                     parameters["input_shape"])
    new_shape = np.array(parameters["new_shape"])
    return [input_value, new_shape], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value, new_shape])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
