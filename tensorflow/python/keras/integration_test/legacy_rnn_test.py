# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


class KerasNetworkTFRNNs(tf.keras.Model):

  def __init__(self, name=None):
    super(KerasNetworkTFRNNs, self).__init__(name=name)
    self._cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(1) for _ in range(2)])

  def call(self, inputs):
    return self._cell(inputs, self._cell.get_initial_state(inputs))


class KerasNetworkKerasRNNs(tf.keras.Model):

  def __init__(self, name=None):
    super(KerasNetworkKerasRNNs, self).__init__(name=name)
    self._cell = tf.keras.layers.StackedRNNCells(
        [tf.keras.layers.LSTMCell(1) for _ in range(2)])

  def call(self, inputs):
    return self._cell(inputs, self._cell.get_initial_state(inputs))


class LegacyRNNTest(tf.test.TestCase):

  def setUp(self):
    super(LegacyRNNTest, self).setUp()
    self._seed = 23489
    np.random.seed(self._seed)

  def testRNNWithKerasSimpleRNNCell(self):
    with self.cached_session() as sess:
      input_shape = 10
      output_shape = 5
      timestep = 4
      batch = 100
      (x_train, y_train), _ = get_test_data(
          train_samples=batch,
          test_samples=0,
          input_shape=(timestep, input_shape),
          num_classes=output_shape)
      y_train = tf.keras.utils.to_categorical(y_train)
      cell = tf.keras.layers.SimpleRNNCell(output_shape)

      inputs = tf.placeholder(
          tf.float32, shape=(None, timestep, input_shape))
      predict = tf.placeholder(
          tf.float32, shape=(None, output_shape))

      outputs, state = tf.nn.dynamic_rnn(
          cell, inputs, dtype=tf.float32)
      self.assertEqual(outputs.shape.as_list(), [None, timestep, output_shape])
      self.assertEqual(state.shape.as_list(), [None, output_shape])
      loss = tf.losses.softmax_cross_entropy(predict, state)
      train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

      sess.run([tf.global_variables_initializer()])
      _, outputs, state = sess.run(
          [train_op, outputs, state], {inputs: x_train, predict: y_train})

      self.assertEqual(len(outputs), batch)
      self.assertEqual(len(state), batch)

  def testRNNWithKerasGRUCell(self):
    with self.cached_session() as sess:
      input_shape = 10
      output_shape = 5
      timestep = 4
      batch = 100
      (x_train, y_train), _ = get_test_data(
          train_samples=batch,
          test_samples=0,
          input_shape=(timestep, input_shape),
          num_classes=output_shape)
      y_train = tf.keras.utils.to_categorical(y_train)
      cell = tf.keras.layers.GRUCell(output_shape)

      inputs = tf.placeholder(
          tf.float32, shape=(None, timestep, input_shape))
      predict = tf.placeholder(
          tf.float32, shape=(None, output_shape))

      outputs, state = tf.nn.dynamic_rnn(
          cell, inputs, dtype=tf.float32)
      self.assertEqual(outputs.shape.as_list(), [None, timestep, output_shape])
      self.assertEqual(state.shape.as_list(), [None, output_shape])
      loss = tf.losses.softmax_cross_entropy(predict, state)
      train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

      sess.run([tf.global_variables_initializer()])
      _, outputs, state = sess.run(
          [train_op, outputs, state], {inputs: x_train, predict: y_train})

      self.assertEqual(len(outputs), batch)
      self.assertEqual(len(state), batch)

  def testRNNWithKerasLSTMCell(self):
    with self.cached_session() as sess:
      input_shape = 10
      output_shape = 5
      timestep = 4
      batch = 100
      (x_train, y_train), _ = get_test_data(
          train_samples=batch,
          test_samples=0,
          input_shape=(timestep, input_shape),
          num_classes=output_shape)
      y_train = tf.keras.utils.to_categorical(y_train)
      cell = tf.keras.layers.LSTMCell(output_shape)

      inputs = tf.placeholder(
          tf.float32, shape=(None, timestep, input_shape))
      predict = tf.placeholder(
          tf.float32, shape=(None, output_shape))

      outputs, state = tf.nn.dynamic_rnn(
          cell, inputs, dtype=tf.float32)
      self.assertEqual(outputs.shape.as_list(), [None, timestep, output_shape])
      self.assertEqual(len(state), 2)
      self.assertEqual(state[0].shape.as_list(), [None, output_shape])
      self.assertEqual(state[1].shape.as_list(), [None, output_shape])
      loss = tf.losses.softmax_cross_entropy(predict, state[0])
      train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

      sess.run([tf.global_variables_initializer()])
      _, outputs, state = sess.run(
          [train_op, outputs, state], {inputs: x_train, predict: y_train})

      self.assertEqual(len(outputs), batch)
      self.assertEqual(len(state), 2)
      self.assertEqual(len(state[0]), batch)
      self.assertEqual(len(state[1]), batch)

  def testRNNWithStackKerasCell(self):
    with self.cached_session() as sess:
      input_shape = 10
      output_shape = 5
      timestep = 4
      batch = 100
      (x_train, y_train), _ = get_test_data(
          train_samples=batch,
          test_samples=0,
          input_shape=(timestep, input_shape),
          num_classes=output_shape)
      y_train = tf.keras.utils.to_categorical(y_train)
      cell = tf.keras.layers.StackedRNNCells(
          [tf.keras.layers.LSTMCell(2 * output_shape),
           tf.keras.layers.LSTMCell(output_shape)])

      inputs = tf.placeholder(
          tf.float32, shape=(None, timestep, input_shape))
      predict = tf.placeholder(
          tf.float32, shape=(None, output_shape))

      outputs, state = tf.nn.dynamic_rnn(
          cell, inputs, dtype=tf.float32)
      self.assertEqual(outputs.shape.as_list(), [None, timestep, output_shape])
      self.assertEqual(len(state), 2)
      state = tf.nest.flatten(state)
      self.assertEqual(len(state), 4)
      self.assertEqual(state[0].shape.as_list(), [None, 2 * output_shape])
      self.assertEqual(state[1].shape.as_list(), [None, 2 * output_shape])
      self.assertEqual(state[2].shape.as_list(), [None, output_shape])
      self.assertEqual(state[3].shape.as_list(), [None, output_shape])
      loss = tf.losses.softmax_cross_entropy(predict, state[2])
      train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

      sess.run([tf.global_variables_initializer()])
      _, outputs, state = sess.run(
          [train_op, outputs, state], {inputs: x_train, predict: y_train})

      self.assertEqual(len(outputs), batch)
      self.assertEqual(len(state), 4)
      for s in state:
        self.assertEqual(len(s), batch)

  def testStaticRNNWithKerasSimpleRNNCell(self):
    with self.cached_session() as sess:
      input_shape = 10
      output_shape = 5
      timestep = 4
      batch = 100
      (x_train, y_train), _ = get_test_data(
          train_samples=batch,
          test_samples=0,
          input_shape=(timestep, input_shape),
          num_classes=output_shape)
      x_train = np.transpose(x_train, (1, 0, 2))
      y_train = tf.keras.utils.to_categorical(y_train)
      cell = tf.keras.layers.SimpleRNNCell(output_shape)

      inputs = [tf.placeholder(
          tf.float32, shape=(None, input_shape))] * timestep
      predict = tf.placeholder(
          tf.float32, shape=(None, output_shape))

      outputs, state = tf.nn.static_rnn(
          cell, inputs, dtype=tf.float32)
      self.assertEqual(len(outputs), timestep)
      self.assertEqual(outputs[0].shape.as_list(), [None, output_shape])
      self.assertEqual(state.shape.as_list(), [None, output_shape])
      loss = tf.losses.softmax_cross_entropy(predict, state)
      train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

      sess.run([tf.global_variables_initializer()])
      feed_dict = {i: d for i, d in zip(inputs, x_train)}
      feed_dict[predict] = y_train
      _, outputs, state = sess.run(
          [train_op, outputs, state], feed_dict)

      self.assertEqual(len(outputs), timestep)
      self.assertEqual(len(outputs[0]), batch)
      self.assertEqual(len(state), batch)

  def testKerasAndTFRNNLayerOutputComparison(self):
    input_shape = 10
    output_shape = 5
    timestep = 4
    batch = 20
    (x_train, _), _ = get_test_data(
        train_samples=batch,
        test_samples=0,
        input_shape=(timestep, input_shape),
        num_classes=output_shape)
    fix_weights_generator = tf.keras.layers.SimpleRNNCell(output_shape)
    fix_weights_generator.build((None, input_shape))
    weights = fix_weights_generator.get_weights()

    with self.session(graph=tf.Graph()) as sess:
      inputs = tf.placeholder(
          tf.float32, shape=(None, timestep, input_shape))
      cell = tf.keras.layers.SimpleRNNCell(output_shape)
      tf_out, tf_state = tf.nn.dynamic_rnn(
          cell, inputs, dtype=tf.float32)
      cell.set_weights(weights)
      [tf_out, tf_state] = sess.run([tf_out, tf_state], {inputs: x_train})
    with self.session(graph=tf.Graph()) as sess:
      k_input = tf.keras.Input(shape=(timestep, input_shape),
                               dtype=tf.float32)
      cell = tf.keras.layers.SimpleRNNCell(output_shape)
      layer = tf.keras.layers.RNN(
          cell, return_sequences=True, return_state=True)
      keras_out = layer(k_input)
      cell.set_weights(weights)
      k_out, k_state = sess.run(keras_out, {k_input: x_train})
    self.assertAllClose(tf_out, k_out)
    self.assertAllClose(tf_state, k_state)

  def testSimpleRNNCellAndBasicRNNCellComparison(self):
    input_shape = 10
    output_shape = 5
    timestep = 4
    batch = 20
    (x_train, _), _ = get_test_data(
        train_samples=batch,
        test_samples=0,
        input_shape=(timestep, input_shape),
        num_classes=output_shape)
    fix_weights_generator = tf.keras.layers.SimpleRNNCell(output_shape)
    fix_weights_generator.build((None, input_shape))
    # The SimpleRNNCell contains 3 weights: kernel, recurrent_kernel, and bias
    # The BasicRNNCell contains 2 weight: kernel and bias, where kernel is
    # zipped [kernel, recurrent_kernel] in SimpleRNNCell.
    keras_weights = fix_weights_generator.get_weights()
    kernel, recurrent_kernel, bias = keras_weights
    tf_weights = [np.concatenate((kernel, recurrent_kernel)), bias]

    with self.session(graph=tf.Graph()) as sess:
      inputs = tf.placeholder(
          tf.float32, shape=(None, timestep, input_shape))
      cell = tf.keras.layers.SimpleRNNCell(output_shape)
      k_out, k_state = tf.nn.dynamic_rnn(
          cell, inputs, dtype=tf.float32)
      cell.set_weights(keras_weights)
      [k_out, k_state] = sess.run([k_out, k_state], {inputs: x_train})
    with self.session(graph=tf.Graph()) as sess:
      inputs = tf.placeholder(
          tf.float32, shape=(None, timestep, input_shape))
      cell = tf.nn.rnn_cell.BasicRNNCell(output_shape)
      tf_out, tf_state = tf.nn.dynamic_rnn(
          cell, inputs, dtype=tf.float32)
      cell.set_weights(tf_weights)
      [tf_out, tf_state] = sess.run([tf_out, tf_state], {inputs: x_train})

    self.assertAllClose(tf_out, k_out, atol=1e-5)
    self.assertAllClose(tf_state, k_state, atol=1e-5)

  def testRNNCellSerialization(self):
    for cell in [
        tf.nn.rnn_cell.LSTMCell(32, use_peepholes=True, cell_clip=True),
        tf.nn.rnn_cell.BasicLSTMCell(32, dtype=tf.float32),
        tf.nn.rnn_cell.BasicRNNCell(32, activation="relu", dtype=tf.float32),
        tf.nn.rnn_cell.GRUCell(32, dtype=tf.float32)
    ]:
      with self.cached_session():
        x = tf.keras.Input((None, 5))
        layer = tf.keras.layers.RNN(cell)
        y = layer(x)
        model = tf.keras.models.Model(x, y)
        model.compile(optimizer="rmsprop", loss="mse")

        # Test basic case serialization.
        x_np = np.random.random((6, 5, 5))
        y_np = model.predict(x_np)
        weights = model.get_weights()
        config = layer.get_config()
        # The custom_objects is important here since rnn_cell_impl is
        # not visible as a Keras layer, and also has a name conflict with
        # keras.LSTMCell and GRUCell.
        layer = tf.keras.layers.RNN.from_config(
            config,
            custom_objects={
                "BasicRNNCell": tf.nn.rnn_cell.BasicRNNCell,
                "GRUCell": tf.nn.rnn_cell.GRUCell,
                "LSTMCell": tf.nn.rnn_cell.LSTMCell,
                "BasicLSTMCell": tf.nn.rnn_cell.BasicLSTMCell
            })
        y = layer(x)
        model = tf.keras.models.Model(x, y)
        model.set_weights(weights)
        y_np_2 = model.predict(x_np)
        self.assertAllClose(y_np, y_np_2, atol=1e-4)

  def testRNNCellActsLikeKerasRNNCellInProperScope(self):
    with tf.layers.experimental.keras_style_scope():
      kn1 = KerasNetworkTFRNNs(name="kn1")
      kn2 = KerasNetworkKerasRNNs(name="kn2")

    z = tf.zeros((2, 3))

    kn1(z)  # pylint:disable=not-callable
    kn2(z)  # pylint:disable=not-callable

    # pylint: disable=protected-access
    self.assertTrue(all("kn1" in v.name for v in kn1._cell.variables))
    self.assertTrue(all("kn2" in v.name for v in kn2._cell.variables))

    with tf.layers.experimental.keras_style_scope():
      kn1_new = KerasNetworkTFRNNs(name="kn1_new")
      kn2_new = KerasNetworkKerasRNNs(name="kn2_new")

    kn2_new(z)  # pylint:disable=not-callable
    # Most importantly, this doesn't fail due to variable scope reuse issues.
    kn1_new(z)  # pylint:disable=not-callable

    self.assertTrue(all("kn1_new" in v.name for v in kn1_new._cell.variables))
    self.assertTrue(all("kn2_new" in v.name for v in kn2_new._cell.variables))


def get_test_data(train_samples,
                  test_samples,
                  input_shape,
                  num_classes):
  num_sample = train_samples + test_samples
  templates = 2 * num_classes * np.random.random((num_classes,) + input_shape)
  y = np.random.randint(0, num_classes, size=(num_sample,))
  x = np.zeros((num_sample,) + input_shape, dtype=np.float32)
  for i in range(num_sample):
    x[i] = templates[y[i]] + np.random.normal(loc=0, scale=1., size=input_shape)
  return ((x[:train_samples], y[:train_samples]),
          (x[train_samples:], y[train_samples:]))


if __name__ == "__main__":
  tf.test.main()
