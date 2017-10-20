# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

import collections
import gc
import time

import numpy as np
import tensorflow as tf

from tensorflow.contrib.eager.python import tfe
from tensorflow.contrib.eager.python.examples import spinn
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util


def _generate_synthetic_snli_data_batch(sequence_length,
                                        batch_size,
                                        vocab_size):
  """Generate a fake batch of SNLI data for testing."""
  with tf.device("cpu:0"):
    labels = tf.random_uniform([batch_size], minval=1, maxval=4, dtype=tf.int64)
    prem = tf.random_uniform(
        (sequence_length, batch_size), maxval=vocab_size, dtype=tf.int64)
    prem_trans = tf.constant(np.array(
        [[3, 3, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3,
          2, 3, 3, 2, 2, 3, 3, 3, 2, 2, 2, 2,
          3, 2, 2]] * batch_size, dtype=np.int64).T)
    hypo = tf.random_uniform(
        (sequence_length, batch_size), maxval=vocab_size, dtype=tf.int64)
    hypo_trans = tf.constant(np.array(
        [[3, 3, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3,
          2, 3, 3, 2, 2, 3, 3, 3, 2, 2, 2, 2,
          3, 2, 2]] * batch_size, dtype=np.int64).T)
  if tfe.num_gpus():
    labels = labels.gpu()
    prem = prem.gpu()
    prem_trans = prem_trans.gpu()
    hypo = hypo.gpu()
    hypo_trans = hypo_trans.gpu()
  return labels, prem, prem_trans, hypo, hypo_trans


def _snli_classifier_config(d_embed, d_out):
  config_tuple = collections.namedtuple(
      "Config", ["d_hidden", "d_proj", "d_tracker", "predict",
                 "embed_dropout", "mlp_dropout", "n_mlp_layers", "d_mlp",
                 "d_out", "projection", "lr"])
  config = config_tuple(
      d_hidden=d_embed,
      d_proj=d_embed * 2,
      d_tracker=8,
      predict=False,
      embed_dropout=0.1,
      mlp_dropout=0.1,
      n_mlp_layers=2,
      d_mlp=32,
      d_out=d_out,
      projection=True,
      lr=2e-3)
  return config


class SpinnTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(SpinnTest, self).setUp()
    self._test_device = "gpu:0" if tfe.num_gpus() else "cpu:0"

  def testBundle(self):
    with tf.device(self._test_device):
      lstm_iter = [np.array([[0, 1], [2, 3]], dtype=np.float32),
                   np.array([[0, -1], [-2, -3]], dtype=np.float32),
                   np.array([[0, 2], [4, 6]], dtype=np.float32),
                   np.array([[0, -2], [-4, -6]], dtype=np.float32)]
      out = spinn._bundle(lstm_iter)

      self.assertEqual(2, len(out))
      self.assertEqual(tf.float32, out[0].dtype)
      self.assertEqual(tf.float32, out[1].dtype)
      self.assertAllEqual(np.array([[0, 2, 0, -2, 0, 4, 0, -4]]).T,
                          out[0].numpy())
      self.assertAllEqual(np.array([[1, 3, -1, -3, 2, 6, -2, -6]]).T,
                          out[1].numpy())

  def testUnbunbdle(self):
    with tf.device(self._test_device):
      state = [np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32),
               np.array([[0, -1, -2], [-3, -4, -5]], dtype=np.float32)]
      out = spinn._unbundle(state)

      self.assertEqual(2, len(out))
      self.assertEqual(tf.float32, out[0].dtype)
      self.assertEqual(tf.float32, out[1].dtype)
      self.assertAllEqual(np.array([[0, 1, 2, 0, -1, -2]]),
                          out[0].numpy())
      self.assertAllEqual(np.array([[3, 4, 5, -3, -4, -5]]),
                          out[1].numpy())

  def testReduce(self):
    with tf.device(self._test_device):
      batch_size = 3
      size = 10
      tracker_size = 8
      reducer = spinn.Reduce(size, tracker_size=tracker_size)

      left_in = []
      right_in = []
      tracking = []
      for _ in range(batch_size):
        left_in.append(tf.random_normal((1, size * 2)))
        right_in.append(tf.random_normal((1, size * 2)))
        tracking.append(tf.random_normal((1, tracker_size * 2)))

      out = reducer(left_in, right_in, tracking=tracking)
      self.assertEqual(batch_size, len(out))
      self.assertEqual(tf.float32, out[0].dtype)
      self.assertEqual((1, size * 2), out[0].shape)

  def testReduceTreeLSTM(self):
    with tf.device(self._test_device):
      size = 10
      tracker_size = 8
      reducer = spinn.Reduce(size, tracker_size=tracker_size)

      lstm_in = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                          [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]],
                         dtype=np.float32)
      c1 = np.array([[0, 1], [2, 3]], dtype=np.float32)
      c2 = np.array([[0, -1], [-2, -3]], dtype=np.float32)

      h, c = reducer._tree_lstm(c1, c2, lstm_in)
      self.assertEqual(tf.float32, h.dtype)
      self.assertEqual(tf.float32, c.dtype)
      self.assertEqual((2, 2), h.shape)
      self.assertEqual((2, 2), c.shape)

  def testTracker(self):
    with tf.device(self._test_device):
      batch_size = 2
      size = 10
      tracker_size = 8
      buffer_length = 18
      stack_size = 3

      tracker = spinn.Tracker(tracker_size, False)
      tracker.reset_state()

      # Create dummy inputs for testing.
      bufs = []
      buf = []
      for _ in range(buffer_length):
        buf.append(tf.random_normal((batch_size, size * 2)))
      bufs.append(buf)
      self.assertEqual(1, len(bufs))
      self.assertEqual(buffer_length, len(bufs[0]))
      self.assertEqual((batch_size, size * 2), bufs[0][0].shape)

      stacks = []
      stack = []
      for _ in range(stack_size):
        stack.append(tf.random_normal((batch_size, size * 2)))
      stacks.append(stack)
      self.assertEqual(1, len(stacks))
      self.assertEqual(3, len(stacks[0]))
      self.assertEqual((batch_size, size * 2), stacks[0][0].shape)

      for _ in range(2):
        out1, out2 = tracker(bufs, stacks)
        self.assertIsNone(out2)
        self.assertEqual(batch_size, len(out1))
        self.assertEqual(tf.float32, out1[0].dtype)
        self.assertEqual((1, tracker_size * 2), out1[0].shape)

        self.assertEqual(tf.float32, tracker.state.c.dtype)
        self.assertEqual((batch_size, tracker_size), tracker.state.c.shape)
        self.assertEqual(tf.float32, tracker.state.h.dtype)
        self.assertEqual((batch_size, tracker_size), tracker.state.h.shape)

  def testSPINN(self):
    with tf.device(self._test_device):
      embedding_dims = 10
      d_tracker = 8
      sequence_length = 15
      num_transitions = 27

      config_tuple = collections.namedtuple(
          "Config", ["d_hidden", "d_proj", "d_tracker", "predict"])
      config = config_tuple(
          embedding_dims, embedding_dims * 2, d_tracker, False)
      s = spinn.SPINN(config)

      # Create some fake data.
      buffers = tf.random_normal((sequence_length, 1, config.d_proj))
      transitions = np.array(
          [[3], [3], [2], [3], [3], [3], [2], [2], [2], [3], [3], [3],
           [2], [3], [3], [2], [2], [3], [3], [3], [2], [2], [2], [2],
           [3], [2], [2]], dtype=np.int32)
      self.assertEqual(tf.int32, transitions.dtype)
      self.assertEqual((num_transitions, 1), transitions.shape)

      out = s(buffers, transitions, training=True)
      self.assertEqual(tf.float32, out.dtype)
      self.assertEqual((1, embedding_dims), out.shape)

  def testSNLIClassifierAndTrainer(self):
    with tf.device(self._test_device):
      vocab_size = 40
      batch_size = 2
      d_embed = 10
      sequence_length = 15
      d_out = 4

      config = _snli_classifier_config(d_embed, d_out)

      # Create fake embedding matrix.
      embed = tf.random_normal((vocab_size, d_embed))

      model = spinn.SNLIClassifier(config, embed)
      trainer = spinn.SNLIClassifierTrainer(model, config.lr)

      (labels, prem, prem_trans, hypo,
       hypo_trans) = _generate_synthetic_snli_data_batch(sequence_length,
                                                         batch_size,
                                                         vocab_size)

      # Invoke model under non-training mode.
      logits = model(prem, prem_trans, hypo, hypo_trans, training=False)
      self.assertEqual(tf.float32, logits.dtype)
      self.assertEqual((batch_size, d_out), logits.shape)

      # Invoke model under training model.
      logits = model(prem, prem_trans, hypo, hypo_trans, training=True)
      self.assertEqual(tf.float32, logits.dtype)
      self.assertEqual((batch_size, d_out), logits.shape)

      # Calculate loss.
      loss1 = trainer.loss(labels, logits)
      self.assertEqual(tf.float32, loss1.dtype)
      self.assertEqual((), loss1.shape)

      loss2, logits = trainer.train_batch(
          labels, prem, prem_trans, hypo, hypo_trans)
      self.assertEqual(tf.float32, loss2.dtype)
      self.assertEqual((), loss2.shape)
      self.assertEqual(tf.float32, logits.dtype)
      self.assertEqual((batch_size, d_out), logits.shape)
      # Training on the batch should have led to a change in the loss value.
      self.assertNotEqual(loss1.numpy(), loss2.numpy())


class EagerSpinnSNLIClassifierBenchmark(test.Benchmark):

  def benchmarkEagerSpinnSNLIClassifier(self):
    test_device = "gpu:0" if tfe.num_gpus() else "cpu:0"
    with tf.device(test_device):
      burn_in_iterations = 2
      benchmark_iterations = 10

      vocab_size = 1000
      batch_size = 128
      sequence_length = 15
      d_embed = 200
      d_out = 4

      embed = tf.random_normal((vocab_size, d_embed))

      config = _snli_classifier_config(d_embed, d_out)
      model = spinn.SNLIClassifier(config, embed)
      trainer = spinn.SNLIClassifierTrainer(model, config.lr)

      (labels, prem, prem_trans, hypo,
       hypo_trans) = _generate_synthetic_snli_data_batch(sequence_length,
                                                         batch_size,
                                                         vocab_size)

      for _ in range(burn_in_iterations):
        trainer.train_batch(labels, prem, prem_trans, hypo, hypo_trans)

      gc.collect()
      start_time = time.time()
      for _ in xrange(benchmark_iterations):
        trainer.train_batch(labels, prem, prem_trans, hypo, hypo_trans)
      wall_time = time.time() - start_time
      # Named "examples"_per_sec to conform with other benchmarks.
      extras = {"examples_per_sec": benchmark_iterations / wall_time}
      self.report_benchmark(
          name="Eager_SPINN_SNLIClassifier_Benchmark",
          iters=benchmark_iterations,
          wall_time=wall_time,
          extras=extras)


if __name__ == "__main__":
  test.main()
