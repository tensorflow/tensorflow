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
"""Test configs for cond."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
from tensorflow.python.framework import test_util


@register_make_test_function("make_cond_tests")
@test_util.enable_control_flow_v2
def make_cond_tests(options):
  """Make a set of tests to do relu1."""

  # Chose a set of parameters
  test_parameters = [{
      # Note: The `tf.string` test case also serves as a regression test to
      # ensure that branch subgraph with dynamically allocated inputs/outputs
      # are handled correctly.
      "dtype": [tf.float32, tf.string],
      "pred": [False, True],
  }]

  def build_graph(parameters):
    """Build the graph for cond tests."""
    input1 = tf.placeholder(dtype=parameters["dtype"], shape=(1,))
    input2 = tf.placeholder(dtype=parameters["dtype"], shape=(1,))
    # MLIR TFLite converter can't handle scalar inputs. This is a workaround
    # to input (1,) tensors and then reshape to scalar.
    # TODO(b/129003347): Remove the workaround after scalar inputs are
    # supported.
    pred = tf.placeholder(dtype=tf.bool, shape=(1,))
    pred_scalar = tf.reshape(pred, ())

    out = tf.cond(pred_scalar, lambda: input1, lambda: input2)
    return [input1, input2, pred], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_values = [
        create_tensor_data(parameters["dtype"], (1,)),
        create_tensor_data(parameters["dtype"], (1,)),
        np.array([parameters["pred"]], dtype=np.bool),
    ]
    return input_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, input_values)))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
