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

from tensorflow import flags

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.lite.python.op_hint import convert_op_hints_to_stubs
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.tools import optimize_for_inference_lib

FLAGS = flags.FLAGS

# Number of steps to train model.
TRAIN_STEPS = 1

CONFIG = tf.ConfigProto(device_count={"GPU": 0})


class UnidirectionalSequenceRnnTest(test_util.TensorFlowTestCase):

  def __init__(self, *args, **kwargs):
    super(UnidirectionalSequenceRnnTest, self).__init__(*args, **kwargs)
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
    # Rnn Units.
    self.num_units = 16

  def setUp(self):
    super(UnidirectionalSequenceRnnTest, self).setUp()
    # Import MNIST dataset
    data_dir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    self.mnist = input_data.read_data_sets(data_dir, one_hot=True)

  def buildRnnLayer(self):
    return tf.keras.layers.StackedRNNCells([
        tf.lite.experimental.nn.TfLiteRNNCell(self.num_units, name="rnn1"),
        tf.lite.experimental.nn.TfLiteRNNCell(self.num_units, name="rnn2")
    ])

  def buildModel(self, rnn_layer, is_dynamic_rnn):
    # Weights and biases for output softmax layer.
    out_weights = tf.Variable(
        tf.random_normal([self.num_units, self.n_classes]))
    out_bias = tf.Variable(tf.random_normal([self.n_classes]))

    # input image placeholder
    x = tf.placeholder(
        "float", [None, self.time_steps, self.n_input], name="INPUT_IMAGE")

    # x is shaped [batch_size,time_steps,num_inputs]
    if is_dynamic_rnn:
      rnn_input = tf.transpose(x, perm=[1, 0, 2])
      outputs, _ = tf.lite.experimental.nn.dynamic_rnn(
          rnn_layer, rnn_input, dtype="float32")
      outputs = tf.unstack(outputs, axis=0)
    else:
      rnn_input = tf.unstack(x, self.time_steps, 1)
      outputs, _ = tf.nn.static_rnn(rnn_layer, rnn_input, dtype="float32")

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
    sess.run(tf.global_variables_initializer())
    for _ in range(TRAIN_STEPS):
      batch_x, batch_y = self.mnist.train.next_batch(
          batch_size=self.batch_size, shuffle=False)

      batch_x = batch_x.reshape((self.batch_size, self.time_steps,
                                 self.n_input))
      sess.run(opt, feed_dict={x: batch_x, y: batch_y})

  def saveAndRestoreModel(self, rnn_layer, sess, saver, is_dynamic_rnn):
    """Saves and restores the model to mimic the most common use case.

    Args:
      rnn_layer: The rnn layer either a single rnn cell or a multi rnn cell.
      sess: Old session.
      saver: saver created by tf.train.Saver()
      is_dynamic_rnn: use dynamic_rnn or not.

    Returns:
      A tuple containing:

      - Input tensor of the restored model.
      - Prediction tensor of the restored model.
      - Output tensor, which is the softwmax result of the prediction tensor.
      - new session of the restored model.

    """
    model_dir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    saver.save(sess, model_dir)

    # Reset the graph.
    tf.reset_default_graph()
    x, prediction, output_class = self.buildModel(rnn_layer, is_dynamic_rnn)

    new_sess = tf.Session(config=CONFIG)
    saver = tf.train.Saver()
    saver.restore(new_sess, model_dir)
    return x, prediction, output_class, new_sess

  def getInferenceResult(self, x, output_class, sess):
    b1, _ = self.mnist.train.next_batch(batch_size=1)
    sample_input = np.reshape(b1, (1, self.time_steps, self.n_input))

    expected_output = sess.run(output_class, feed_dict={x: sample_input})
    frozen_graph = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph_def, [output_class.op.name])
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
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, test_inputs)
    interpreter.invoke()
    output_index = interpreter.get_output_details()[0]["index"]
    result = interpreter.get_tensor(output_index)
    # Reset all variables so it will not pollute other inferences.
    interpreter.reset_all_variables()
    return result

  def testStaticRnnMultiRnnCell(self):
    sess = tf.Session(config=CONFIG)

    x, prediction, output_class = self.buildModel(
        self.buildRnnLayer(), is_dynamic_rnn=False)
    self.trainModel(x, prediction, output_class, sess)

    saver = tf.train.Saver()
    x, prediction, output_class, new_sess = self.saveAndRestoreModel(
        self.buildRnnLayer(), sess, saver, is_dynamic_rnn=False)

    test_inputs, expected_output, frozen_graph = self.getInferenceResult(
        x, output_class, new_sess)

    result = self.tfliteInvoke(frozen_graph, test_inputs, output_class)
    self.assertTrue(np.allclose(expected_output, result, rtol=1e-6, atol=1e-2))

  @test_util.enable_control_flow_v2
  def testDynamicRnnMultiRnnCell(self):
    sess = tf.Session(config=CONFIG)

    x, prediction, output_class = self.buildModel(
        self.buildRnnLayer(), is_dynamic_rnn=True)
    self.trainModel(x, prediction, output_class, sess)

    saver = tf.train.Saver()

    x, prediction, output_class, new_sess = self.saveAndRestoreModel(
        self.buildRnnLayer(), sess, saver, is_dynamic_rnn=True)

    test_inputs, expected_output, frozen_graph = self.getInferenceResult(
        x, output_class, new_sess)

    result = self.tfliteInvoke(frozen_graph, test_inputs, output_class)
    self.assertTrue(np.allclose(expected_output, result, rtol=1e-6, atol=1e-2))


if __name__ == "__main__":
  test.main()
