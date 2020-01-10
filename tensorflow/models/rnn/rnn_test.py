"""Tests for rnn module."""

# pylint: disable=g-bad-import-order,unused-import
import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell


class Plus1RNNCell(rnn_cell.RNNCell):
  """RNN Cell generating (output, new_state) = (input + 1, state + 1)."""

  @property
  def output_size(self):
    return 5

  @property
  def state_size(self):
    return 5

  def __call__(self, input_, state):
    return (input_ + 1, state + 1)


class TestStateSaver(object):

  def __init__(self, batch_size, state_size):
    self._batch_size = batch_size
    self._state_size = state_size

  def State(self, _):
    return tf.zeros(tf.pack([self._batch_size, self._state_size]))

  def SaveState(self, _, state):
    self.saved_state = state
    return tf.identity(state)


class RNNTest(tf.test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  def testRNN(self):
    cell = Plus1RNNCell()
    batch_size = 2
    inputs = [tf.placeholder(tf.float32, shape=(batch_size, 5))] * 10
    outputs, states = rnn.rnn(cell, inputs, dtype=tf.float32)
    self.assertEqual(len(outputs), len(inputs))
    for out, inp in zip(outputs, inputs):
      self.assertEqual(out.get_shape(), inp.get_shape())
      self.assertEqual(out.dtype, inp.dtype)

    with self.test_session(use_gpu=False) as sess:
      input_value = np.random.randn(batch_size, 5)
      values = sess.run(outputs + [states[-1]],
                        feed_dict={inputs[0]: input_value})

      # Outputs
      for v in values[:-1]:
        self.assertAllClose(v, input_value + 1.0)

      # Final state
      self.assertAllClose(
          values[-1], 10.0*np.ones((batch_size, 5), dtype=np.float32))

  def testDropout(self):
    cell = Plus1RNNCell()
    full_dropout_cell = rnn_cell.DropoutWrapper(
        cell, input_keep_prob=1e-12, seed=0)
    batch_size = 2
    inputs = [tf.placeholder(tf.float32, shape=(batch_size, 5))] * 10
    with tf.variable_scope("share_scope"):
      outputs, states = rnn.rnn(cell, inputs, dtype=tf.float32)
    with tf.variable_scope("drop_scope"):
      dropped_outputs, _ = rnn.rnn(full_dropout_cell, inputs, dtype=tf.float32)
    self.assertEqual(len(outputs), len(inputs))
    for out, inp in zip(outputs, inputs):
      self.assertEqual(out.get_shape().as_list(), inp.get_shape().as_list())
      self.assertEqual(out.dtype, inp.dtype)

    with self.test_session(use_gpu=False) as sess:
      input_value = np.random.randn(batch_size, 5)
      values = sess.run(outputs + [states[-1]],
                        feed_dict={inputs[0]: input_value})
      full_dropout_values = sess.run(dropped_outputs,
                                     feed_dict={inputs[0]: input_value})

      for v in values[:-1]:
        self.assertAllClose(v, input_value + 1.0)
      for d_v in full_dropout_values[:-1]:  # Add 1.0 to dropped_out (all zeros)
        self.assertAllClose(d_v, np.ones_like(input_value))

  def testDynamicCalculation(self):
    cell = Plus1RNNCell()
    sequence_length = tf.placeholder(tf.int64)
    batch_size = 2
    inputs = [tf.placeholder(tf.float32, shape=(batch_size, 5))] * 10
    with tf.variable_scope("drop_scope"):
      dynamic_outputs, dynamic_states = rnn.rnn(
          cell, inputs, sequence_length=sequence_length, dtype=tf.float32)
    self.assertEqual(len(dynamic_outputs), len(inputs))
    self.assertEqual(len(dynamic_states), len(inputs))

    with self.test_session(use_gpu=False) as sess:
      input_value = np.random.randn(batch_size, 5)
      dynamic_values = sess.run(dynamic_outputs,
                                feed_dict={inputs[0]: input_value,
                                           sequence_length: [2, 3]})
      dynamic_state_values = sess.run(dynamic_states,
                                      feed_dict={inputs[0]: input_value,
                                                 sequence_length: [2, 3]})

      # fully calculated for t = 0, 1, 2
      for v in dynamic_values[:3]:
        self.assertAllClose(v, input_value + 1.0)
      for vi, v in enumerate(dynamic_state_values[:3]):
        self.assertAllEqual(v, 1.0 * (vi + 1) * np.ones((batch_size, 5)))
      # zeros for t = 3+
      for v in dynamic_values[3:]:
        self.assertAllEqual(v, np.zeros_like(input_value))
      for v in dynamic_state_values[3:]:
        self.assertAllEqual(v, np.zeros_like(input_value))


class LSTMTest(tf.test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  def _testNoProjNoSharding(self, use_gpu):
    num_units = 3
    input_size = 5
    batch_size = 2
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)
      cell = rnn_cell.LSTMCell(
          num_units, input_size, initializer=initializer)
      inputs = 10 * [
          tf.placeholder(tf.float32, shape=(batch_size, input_size))]
      outputs, _ = rnn.rnn(cell, inputs, dtype=tf.float32)
      self.assertEqual(len(outputs), len(inputs))
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units])

      tf.initialize_all_variables().run()
      input_value = np.random.randn(batch_size, input_size)
      sess.run(outputs, feed_dict={inputs[0]: input_value})

  def _testCellClipping(self, use_gpu):
    num_units = 3
    input_size = 5
    batch_size = 2
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)
      cell = rnn_cell.LSTMCell(
          num_units, input_size, use_peepholes=True,
          cell_clip=0.0, initializer=initializer)
      inputs = 10 * [
          tf.placeholder(tf.float32, shape=(batch_size, input_size))]
      outputs, _ = rnn.rnn(cell, inputs, dtype=tf.float32)
      self.assertEqual(len(outputs), len(inputs))
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units])

      tf.initialize_all_variables().run()
      input_value = np.random.randn(batch_size, input_size)
      values = sess.run(outputs, feed_dict={inputs[0]: input_value})

    for value in values:
      # if cell c is clipped to 0, tanh(c) = 0 => m==0
      self.assertAllEqual(value, np.zeros((batch_size, num_units)))

  def _testNoProjNoShardingSimpleStateSaver(self, use_gpu):
    num_units = 3
    input_size = 5
    batch_size = 2
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)
      state_saver = TestStateSaver(batch_size, 2*num_units)
      cell = rnn_cell.LSTMCell(
          num_units, input_size, use_peepholes=False, initializer=initializer)
      inputs = 10 * [
          tf.placeholder(tf.float32, shape=(batch_size, input_size))]
      with tf.variable_scope("share_scope"):
        outputs, states = rnn.state_saving_rnn(
            cell, inputs, state_saver=state_saver, state_name="save_lstm")
      self.assertEqual(len(outputs), len(inputs))
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units])

      tf.initialize_all_variables().run()
      input_value = np.random.randn(batch_size, input_size)
      (last_state_value, saved_state_value) = sess.run(
          [states[-1], state_saver.saved_state],
          feed_dict={inputs[0]: input_value})
      self.assertAllEqual(last_state_value, saved_state_value)

  def _testProjNoSharding(self, use_gpu):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)
      inputs = 10 * [
          tf.placeholder(tf.float32, shape=(None, input_size))]
      cell = rnn_cell.LSTMCell(
          num_units, input_size, use_peepholes=True,
          num_proj=num_proj, initializer=initializer)
      outputs, _ = rnn.rnn(cell, inputs, dtype=tf.float32)
      self.assertEqual(len(outputs), len(inputs))

      tf.initialize_all_variables().run()
      input_value = np.random.randn(batch_size, input_size)
      sess.run(outputs, feed_dict={inputs[0]: input_value})

  def _testProjSharding(self, use_gpu):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    num_proj_shards = 4
    num_unit_shards = 2
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)

      inputs = 10 * [
          tf.placeholder(tf.float32, shape=(None, input_size))]

      cell = rnn_cell.LSTMCell(
          num_units,
          input_size=input_size,
          use_peepholes=True,
          num_proj=num_proj,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          initializer=initializer)

      outputs, _ = rnn.rnn(cell, inputs, dtype=tf.float32)

      self.assertEqual(len(outputs), len(inputs))

      tf.initialize_all_variables().run()
      input_value = np.random.randn(batch_size, input_size)
      sess.run(outputs, feed_dict={inputs[0]: input_value})

  def _testDoubleInput(self, use_gpu):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    num_proj_shards = 4
    num_unit_shards = 2
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-1, 1, seed=self._seed)
      inputs = 10 * [tf.placeholder(tf.float64)]

      cell = rnn_cell.LSTMCell(
          num_units,
          input_size=input_size,
          use_peepholes=True,
          num_proj=num_proj,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          initializer=initializer)

      outputs, _ = rnn.rnn(
          cell, inputs, initial_state=cell.zero_state(batch_size, tf.float64))

      self.assertEqual(len(outputs), len(inputs))

      tf.initialize_all_variables().run()
      input_value = np.asarray(np.random.randn(batch_size, input_size),
                               dtype=np.float64)
      values = sess.run(outputs, feed_dict={inputs[0]: input_value})
      self.assertEqual(values[0].dtype, input_value.dtype)

  def _testShardNoShardEquivalentOutput(self, use_gpu):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    num_proj_shards = 4
    num_unit_shards = 2
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      inputs = 10 * [tf.placeholder(tf.float32)]
      initializer = tf.constant_initializer(0.001)

      cell_noshard = rnn_cell.LSTMCell(
          num_units, input_size,
          num_proj=num_proj,
          use_peepholes=True,
          initializer=initializer,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards)

      cell_shard = rnn_cell.LSTMCell(
          num_units, input_size, use_peepholes=True,
          initializer=initializer, num_proj=num_proj)

      with tf.variable_scope("noshard_scope"):
        outputs_noshard, states_noshard = rnn.rnn(
            cell_noshard, inputs, dtype=tf.float32)
      with tf.variable_scope("shard_scope"):
        outputs_shard, states_shard = rnn.rnn(
            cell_shard, inputs, dtype=tf.float32)

      self.assertEqual(len(outputs_noshard), len(inputs))
      self.assertEqual(len(outputs_noshard), len(outputs_shard))

      tf.initialize_all_variables().run()
      input_value = np.random.randn(batch_size, input_size)
      feeds = dict((x, input_value) for x in inputs)
      values_noshard = sess.run(outputs_noshard, feed_dict=feeds)
      values_shard = sess.run(outputs_shard, feed_dict=feeds)
      state_values_noshard = sess.run(states_noshard, feed_dict=feeds)
      state_values_shard = sess.run(states_shard, feed_dict=feeds)
      self.assertEqual(len(values_noshard), len(values_shard))
      self.assertEqual(len(state_values_noshard), len(state_values_shard))
      for (v_noshard, v_shard) in zip(values_noshard, values_shard):
        self.assertAllClose(v_noshard, v_shard, atol=1e-3)
      for (s_noshard, s_shard) in zip(state_values_noshard, state_values_shard):
        self.assertAllClose(s_noshard, s_shard, atol=1e-3)

  def _testDoubleInputWithDropoutAndDynamicCalculation(
      self, use_gpu):
    """Smoke test for using LSTM with doubles, dropout, dynamic calculation."""

    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    num_proj_shards = 4
    num_unit_shards = 2
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      sequence_length = tf.placeholder(tf.int64)
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)
      inputs = 10 * [tf.placeholder(tf.float64)]

      cell = rnn_cell.LSTMCell(
          num_units,
          input_size=input_size,
          use_peepholes=True,
          num_proj=num_proj,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          initializer=initializer)
      dropout_cell = rnn_cell.DropoutWrapper(cell, 0.5, seed=0)

      outputs, states = rnn.rnn(
          dropout_cell, inputs, sequence_length=sequence_length,
          initial_state=cell.zero_state(batch_size, tf.float64))

      self.assertEqual(len(outputs), len(inputs))
      self.assertEqual(len(outputs), len(states))

      tf.initialize_all_variables().run()
      input_value = np.asarray(np.random.randn(batch_size, input_size),
                               dtype=np.float64)
      values = sess.run(outputs, feed_dict={inputs[0]: input_value,
                                            sequence_length: [2, 3]})
      state_values = sess.run(states, feed_dict={inputs[0]: input_value,
                                                 sequence_length: [2, 3]})
      self.assertEqual(values[0].dtype, input_value.dtype)
      self.assertEqual(state_values[0].dtype, input_value.dtype)

  def testSharingWeightsWithReuse(self):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    with self.test_session(graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-1, 1, seed=self._seed)
      inputs = 10 * [
          tf.placeholder(tf.float32, shape=(None, input_size))]
      cell = rnn_cell.LSTMCell(
          num_units, input_size, use_peepholes=True,
          num_proj=num_proj, initializer=initializer)

      with tf.variable_scope("share_scope"):
        outputs0, _ = rnn.rnn(cell, inputs, dtype=tf.float32)
      with tf.variable_scope("share_scope", reuse=True):
        outputs1, _ = rnn.rnn(cell, inputs, dtype=tf.float32)
      with tf.variable_scope("diff_scope"):
        outputs2, _ = rnn.rnn(cell, inputs, dtype=tf.float32)

      tf.initialize_all_variables().run()
      input_value = np.random.randn(batch_size, input_size)
      output_values = sess.run(
          outputs0 + outputs1 + outputs2, feed_dict={inputs[0]: input_value})
      outputs0_values = output_values[:10]
      outputs1_values = output_values[10:20]
      outputs2_values = output_values[20:]
      self.assertEqual(len(outputs0_values), len(outputs1_values))
      self.assertEqual(len(outputs0_values), len(outputs2_values))
      for o1, o2, o3 in zip(outputs0_values, outputs1_values, outputs2_values):
        # Same weights used by both RNNs so outputs should be the same.
        self.assertAllEqual(o1, o2)
        # Different weights used so outputs should be different.
        self.assertTrue(np.linalg.norm(o1-o3) > 1e-6)

  def testSharingWeightsWithDifferentNamescope(self):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    with self.test_session(graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-1, 1, seed=self._seed)
      inputs = 10 * [
          tf.placeholder(tf.float32, shape=(None, input_size))]
      cell = rnn_cell.LSTMCell(
          num_units, input_size, use_peepholes=True,
          num_proj=num_proj, initializer=initializer)

      with tf.name_scope("scope0"):
        with tf.variable_scope("share_scope"):
          outputs0, _ = rnn.rnn(cell, inputs, dtype=tf.float32)
      with tf.name_scope("scope1"):
        with tf.variable_scope("share_scope", reuse=True):
          outputs1, _ = rnn.rnn(cell, inputs, dtype=tf.float32)

      tf.initialize_all_variables().run()
      input_value = np.random.randn(batch_size, input_size)
      output_values = sess.run(
          outputs0 + outputs1, feed_dict={inputs[0]: input_value})
      outputs0_values = output_values[:10]
      outputs1_values = output_values[10:]
      self.assertEqual(len(outputs0_values), len(outputs1_values))
      for out0, out1 in zip(outputs0_values, outputs1_values):
        self.assertAllEqual(out0, out1)

  def testNoProjNoShardingSimpleStateSaver(self):
    self._testNoProjNoShardingSimpleStateSaver(False)
    self._testNoProjNoShardingSimpleStateSaver(True)

  def testNoProjNoSharding(self):
    self._testNoProjNoSharding(False)
    self._testNoProjNoSharding(True)

  def testCellClipping(self):
    self._testCellClipping(False)
    self._testCellClipping(True)

  def testProjNoSharding(self):
    self._testProjNoSharding(False)
    self._testProjNoSharding(True)

  def testProjSharding(self):
    self._testProjSharding(False)
    self._testProjSharding(True)

  def testShardNoShardEquivalentOutput(self):
    self._testShardNoShardEquivalentOutput(False)
    self._testShardNoShardEquivalentOutput(True)

  def testDoubleInput(self):
    self._testDoubleInput(False)
    self._testDoubleInput(True)

  def testDoubleInputWithDropoutAndDynamicCalculation(self):
    self._testDoubleInputWithDropoutAndDynamicCalculation(False)
    self._testDoubleInputWithDropoutAndDynamicCalculation(True)


if __name__ == "__main__":
  tf.test.main()
