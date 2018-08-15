# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

import numpy as np
import tensorflow as tf
from tensorflow.contrib.lite.experimental.ops import ops
from tensorflow.contrib.lite.experimental.pb2lite.python import converter_wrapper as converter
from tensorflow.contrib.lite.python.interpreter import Interpreter
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.util.compat import as_bytes

ERROR_THRESHOLD = 1e-6


def _get_tf_operations():
  return tf.get_default_graph().get_operations()


class OpsTest(test_util.TensorFlowTestCase):

  def setUp(self):
    np.random.seed(0)

  def test_fully_connected(self):
    num_batches = 10
    num_input_channels = 20
    num_output_channels = 5

    # No operation initially.
    self.assertEqual(len(_get_tf_operations()), 0)

    # Create 1 operation (Placeholder for the input).
    input_shape = (num_batches, num_input_channels)
    x = tf.placeholder(tf.float32, shape=input_shape)
    self.assertEqual(len(_get_tf_operations()), 1)

    # Defining weights and bias as constants. It should add 2 more
    # nodes into the graph.
    weights_shape = (num_output_channels, num_input_channels)
    weights_value = np.random.rand(*weights_shape).astype(np.float32)
    weights = tf.constant(weights_value, dtype=tf.float32)
    bias_shape = (num_output_channels,)
    bias_value = np.random.rand(*bias_shape).astype(np.float32)
    bias = tf.constant(bias_value, dtype=tf.float32)
    self.assertEqual(len(_get_tf_operations()), 3)

    # Call the function to construct a TF Function node which is equivalent
    # to TFLite FULLY_CONNECTED node.
    output = ops.fully_connected(
        x, weights, bias, fused_activation_function='RELU')

    # Exactly one op should be added. It should be a function containing 2-3 ops
    # (matmul, add, relu).
    operations = _get_tf_operations()
    self.assertEqual(len(operations), 4)

    op = operations[-1]
    node_def = op.node_def
    # Note: `as_bytes` conversion is required for Python 3.
    self.assertEqual(node_def.attr['_tflite_function_name'].s,
                     as_bytes('FULLY_CONNECTED'))
    self.assertEqual(node_def.attr['_fused_activation_function'].s,
                     as_bytes('RELU'))

    # Try to run the TF session to get the output value.
    input_value = np.random.rand(*input_shape).astype(np.float32)
    with tf.Session() as sess:
      output_value = sess.run(output, feed_dict={x: input_value})
      graph_def = sess.graph_def

    # Convert the GraphDef to FlatBuffer.
    flatbuffer_data = converter.Convert(graph_def.SerializeToString())

    # Construct an interpreter with the FlatBuffer.
    interpreter = Interpreter(model_content=flatbuffer_data)

    # Invoke the interpreter.
    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']
    interpreter.resize_tensor_input(input_index, input_shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_index, input_value)
    interpreter.invoke()

    # Get the output from the interpreter, and compare it with the result from
    # TensorFlow.
    output_details = interpreter.get_output_details()
    tflite_output_value = interpreter.get_tensor(output_details[0]['index'])

    max_error = np.max(np.abs(tflite_output_value - output_value))

    self.assertTrue(max_error < ERROR_THRESHOLD)


if __name__ == '__main__':
  test.main()
