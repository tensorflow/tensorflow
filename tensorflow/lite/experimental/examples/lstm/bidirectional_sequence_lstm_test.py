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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tempfile
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.lite.experimental.examples.lstm.tflite_lstm import TFLiteLSTMCell
from tensorflow.lite.python.op_hint import convert_op_hints_to_stubs
from tensorflow.lite.python.op_hint import find_all_hinted_output_nodes
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.tools import optimize_for_inference_lib

# Number of steps to train model.
TRAIN_STEPS = 1

CONFIG = tf.ConfigProto(device_count={"GPU": 0})


class BidirectionalSequenceLstmTest(test_util.TensorFlowTestCase):

  def setUp(self):
    tf.reset_default_graph()
    # Import MNIST dataset
    self.mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # Define constants
    # Unrolled through 28 time steps
    self.time_steps = 28
    # Rows of 28 pixels
    self.n_input = 28
    # Learning rate for Adam optimizer
    self.learning_rate = 0.001
    # MNIST is meant to be classified in 10 classes(0-9).
    self.n_classes = 10
    # Batch size
    self.batch_size = 16
    # Lstm Units.
    self.num_units = 16

  def buildLstmLayer(self):
    return tf.nn.rnn_cell.MultiRNNCell([
        TFLiteLSTMCell(
            self.num_units, use_peepholes=True, forget_bias=0, name="rnn1"),
        TFLiteLSTMCell(self.num_units, num_proj=8, forget_bias=0, name="rnn2"),
        TFLiteLSTMCell(
            self.num_units // 2,
            use_peepholes=True,
            num_proj=8,
            forget_bias=0,
            name="rnn3"),
        TFLiteLSTMCell(self.num_units, forget_bias=0, name="rnn4")
    ])

  def buildModel(self, fw_lstm_layer, bw_lstm_layer):
    # Weights and biases for output softmax layer.
    out_weights = tf.Variable(
        tf.random_normal([self.num_units * 2, self.n_classes]))
    out_bias = tf.Variable(tf.random_normal([self.n_classes]))

    # input image placeholder
    x = tf.placeholder(
        "float", [None, self.time_steps, self.n_input], name="INPUT_IMAGE")

    lstm_input = tf.unstack(x, self.time_steps, 1)
    outputs, _, _ = tf.nn.static_bidirectional_rnn(
        fw_lstm_layer, bw_lstm_layer, lstm_input, dtype="float32")

    # Compute logits by multiplying outputs[-1] of shape [batch_size,num_units]
    # by the softmax layer's out_weight of shape [num_units,n_classes]
    # plus out_bias
    prediction = tf.matmul(outputs[-1], out_weights) + out_bias
    output_class = tf.nn.softmax(prediction, name="OUTPUT_CLASS")

    return x, prediction, output_class

  def trainModel(self, x, prediction, output_class, sess):
    # input label placeholder
    y = tf.placeholder("float", [None, self.n_classes])
    # Loss function
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # Optimization
    opt = tf.train.AdamOptimizer(
        learning_rate=self.learning_rate).minimize(loss)

    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)
    for _ in range(TRAIN_STEPS):
      batch_x, batch_y = self.mnist.train.next_batch(
          batch_size=self.batch_size, shuffle=False)

      batch_x = batch_x.reshape((self.batch_size, self.time_steps,
                                 self.n_input))
      sess.run(opt, feed_dict={x: batch_x, y: batch_y})

  def saveAndRestoreModel(self, fw_lstm_layer, bw_lstm_layer, sess, saver):
    model_dir = tempfile.mkdtemp()
    saver.save(sess, model_dir)

    # Reset the graph.
    tf.reset_default_graph()
    x, prediction, output_class = self.buildModel(fw_lstm_layer, bw_lstm_layer)

    new_sess = tf.Session(config=CONFIG)
    saver = tf.train.Saver()
    saver.restore(new_sess, model_dir)
    return x, prediction, output_class, new_sess

  def getInferenceResult(self, x, output_class, sess):
    b1, _ = self.mnist.train.next_batch(batch_size=1)
    sample_input = np.reshape(b1, (1, self.time_steps, self.n_input))

    expected_output = sess.run(output_class, feed_dict={x: sample_input})
    # It is important to keep all the ophint output nodes.
    hinted_outputs_nodes = find_all_hinted_output_nodes(sess)
    hinted_outputs_nodes.append(output_class.op.name)
    frozen_graph = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph_def, hinted_outputs_nodes)
    return sample_input, expected_output, frozen_graph

  def tfliteInvoke(self, graph, test_inputs, outputs):
    tf.reset_default_graph()
    # Turn the input into placeholder of shape 1
    tflite_input = tf.placeholder(
        "float", [1, self.time_steps, self.n_input], name="INPUT_IMAGE_LITE")
    tf.import_graph_def(graph, name="", input_map={"INPUT_IMAGE": tflite_input})
    with tf.Session() as sess:
      curr = sess.graph_def
      curr = convert_op_hints_to_stubs(graph_def=curr)

    curr = optimize_for_inference_lib.optimize_for_inference(
        curr, ["INPUT_IMAGE_LITE"], ["OUTPUT_CLASS"],
        [tf.float32.as_datatype_enum])

    converter = tf.lite.TFLiteConverter(curr, [tflite_input], [outputs])
    tflite = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite)

    try:
      interpreter.allocate_tensors()
    except ValueError:
      assert False

    input_index = (interpreter.get_input_details()[0]["index"])
    interpreter.set_tensor(input_index, test_inputs)
    interpreter.invoke()
    output_index = (interpreter.get_output_details()[0]["index"])
    result = interpreter.get_tensor(output_index)
    # Reset all variables so it will not pollute other inferences.
    interpreter.reset_all_variables()
    return result

  def testStaticRnnMultiRnnCell(self):
    sess = tf.Session(config=CONFIG)

    x, prediction, output_class = self.buildModel(self.buildLstmLayer(),
                                                  self.buildLstmLayer())
    self.trainModel(x, prediction, output_class, sess)

    saver = tf.train.Saver()
    x, prediction, output_class, new_sess = self.saveAndRestoreModel(
        self.buildLstmLayer(), self.buildLstmLayer(), sess, saver)

    test_inputs, expected_output, frozen_graph = self.getInferenceResult(
        x, output_class, new_sess)

    result = self.tfliteInvoke(frozen_graph, test_inputs, output_class)
    self.assertTrue(np.allclose(expected_output, result, rtol=1e-6, atol=1e-2))


if __name__ == "__main__":
  test.main()
