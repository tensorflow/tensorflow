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
import glob
import os
import shutil
import tempfile
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# pylint: disable=g-bad-import-order
import tensorflow.contrib.eager as tfe
from tensorflow.contrib.eager.python.examples.spinn import data
from third_party.examples.eager.spinn import spinn
from tensorflow.contrib.summary import summary_test_util
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util
from tensorflow.python.training import checkpoint_utils
# pylint: enable=g-bad-import-order


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


def _test_spinn_config(d_embed, d_out, logdir=None, inference_sentences=None):
  """Generate a config tuple for testing.

  Args:
    d_embed: Embedding dimensions.
    d_out: Model output dimensions.
    logdir: Optional logdir.
    inference_sentences: A 2-tuple of strings representing the sentences (with
      binary parsing result), e.g.,
      ("( ( The dog ) ( ( is running ) . ) )", "( ( The dog ) ( moves . ) )").

  Returns:
    A config tuple.
  """
  config_tuple = collections.namedtuple(
      "Config", ["d_hidden", "d_proj", "d_tracker", "predict",
                 "embed_dropout", "mlp_dropout", "n_mlp_layers", "d_mlp",
                 "d_out", "projection", "lr", "batch_size", "epochs",
                 "force_cpu", "logdir", "log_every", "dev_every", "save_every",
                 "lr_decay_every", "lr_decay_by", "inference_premise",
                 "inference_hypothesis"])

  inference_premise = inference_sentences[0] if inference_sentences else None
  inference_hypothesis = inference_sentences[1] if inference_sentences else None
  return config_tuple(
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
      lr=2e-2,
      batch_size=2,
      epochs=20,
      force_cpu=False,
      logdir=logdir,
      log_every=1,
      dev_every=2,
      save_every=2,
      lr_decay_every=1,
      lr_decay_by=0.75,
      inference_premise=inference_premise,
      inference_hypothesis=inference_hypothesis)


class SpinnTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(SpinnTest, self).setUp()
    self._test_device = "gpu:0" if tfe.num_gpus() else "cpu:0"
    self._temp_data_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self._temp_data_dir)
    super(SpinnTest, self).tearDown()

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

  def testReducer(self):
    with tf.device(self._test_device):
      batch_size = 3
      size = 10
      tracker_size = 8
      reducer = spinn.Reducer(size, tracker_size=tracker_size)

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
      reducer = spinn.Reducer(size, tracker_size=tracker_size)

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
      transitions = tf.constant(
          [[3], [3], [2], [3], [3], [3], [2], [2], [2], [3], [3], [3],
           [2], [3], [3], [2], [2], [3], [3], [3], [2], [2], [2], [2],
           [3], [2], [2]], dtype=tf.int64)
      self.assertEqual(tf.int64, transitions.dtype)
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

      config = _test_spinn_config(d_embed, d_out)

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

  def _create_test_data(self, snli_1_0_dir):
    fake_train_file = os.path.join(snli_1_0_dir, "snli_1.0_train.txt")
    os.makedirs(snli_1_0_dir)

    # Four sentences in total.
    with open(fake_train_file, "wt") as f:
      f.write("gold_label\tsentence1_binary_parse\tsentence2_binary_parse\t"
              "sentence1_parse\tsentence2_parse\tsentence1\tsentence2\t"
              "captionID\tpairID\tlabel1\tlabel2\tlabel3\tlabel4\tlabel5\n")
      f.write("neutral\t( ( Foo bar ) . )\t( ( foo . )\t"
              "DummySentence1Parse\tDummySentence2Parse\t"
              "Foo bar.\tfoo baz.\t"
              "4705552913.jpg#2\t4705552913.jpg#2r1n\t"
              "neutral\tentailment\tneutral\tneutral\tneutral\n")
      f.write("contradiction\t( ( Bar foo ) . )\t( ( baz . )\t"
              "DummySentence1Parse\tDummySentence2Parse\t"
              "Foo bar.\tfoo baz.\t"
              "4705552913.jpg#2\t4705552913.jpg#2r1n\t"
              "neutral\tentailment\tneutral\tneutral\tneutral\n")
      f.write("entailment\t( ( Quux quuz ) . )\t( ( grault . )\t"
              "DummySentence1Parse\tDummySentence2Parse\t"
              "Foo bar.\tfoo baz.\t"
              "4705552913.jpg#2\t4705552913.jpg#2r1n\t"
              "neutral\tentailment\tneutral\tneutral\tneutral\n")
      f.write("entailment\t( ( Quuz quux ) . )\t( ( garply . )\t"
              "DummySentence1Parse\tDummySentence2Parse\t"
              "Foo bar.\tfoo baz.\t"
              "4705552913.jpg#2\t4705552913.jpg#2r1n\t"
              "neutral\tentailment\tneutral\tneutral\tneutral\n")

    glove_dir = os.path.join(self._temp_data_dir, "glove")
    os.makedirs(glove_dir)
    glove_file = os.path.join(glove_dir, "glove.42B.300d.txt")

    words = [".", "foo", "bar", "baz", "quux", "quuz", "grault", "garply"]
    with open(glove_file, "wt") as f:
      for i, word in enumerate(words):
        f.write("%s " % word)
        for j in range(data.WORD_VECTOR_LEN):
          f.write("%.5f" % (i * 0.1))
          if j < data.WORD_VECTOR_LEN - 1:
            f.write(" ")
          else:
            f.write("\n")

    return fake_train_file

  def testInferSpinnWorks(self):
    """Test inference with the spinn model."""
    snli_1_0_dir = os.path.join(self._temp_data_dir, "snli/snli_1.0")
    self._create_test_data(snli_1_0_dir)

    vocab = data.load_vocabulary(self._temp_data_dir)
    word2index, embed = data.load_word_vectors(self._temp_data_dir, vocab)

    config = _test_spinn_config(
        data.WORD_VECTOR_LEN, 4,
        logdir=os.path.join(self._temp_data_dir, "logdir"),
        inference_sentences=("( foo ( bar . ) )", "( bar ( foo . ) )"))
    logits = spinn.train_or_infer_spinn(
        embed, word2index, None, None, None, config)
    self.assertEqual(tf.float32, logits.dtype)
    self.assertEqual((3,), logits.shape)

  def testInferSpinnThrowsErrorIfOnlyOneSentenceIsSpecified(self):
    snli_1_0_dir = os.path.join(self._temp_data_dir, "snli/snli_1.0")
    self._create_test_data(snli_1_0_dir)

    vocab = data.load_vocabulary(self._temp_data_dir)
    word2index, embed = data.load_word_vectors(self._temp_data_dir, vocab)

    config = _test_spinn_config(
        data.WORD_VECTOR_LEN, 4,
        logdir=os.path.join(self._temp_data_dir, "logdir"),
        inference_sentences=("( foo ( bar . ) )", None))
    with self.assertRaises(ValueError):
      spinn.train_or_infer_spinn(embed, word2index, None, None, None, config)

  def testTrainSpinn(self):
    """Test with fake toy SNLI data and GloVe vectors."""

    # 1. Create and load a fake SNLI data file and a fake GloVe embedding file.
    snli_1_0_dir = os.path.join(self._temp_data_dir, "snli/snli_1.0")
    fake_train_file = self._create_test_data(snli_1_0_dir)

    vocab = data.load_vocabulary(self._temp_data_dir)
    word2index, embed = data.load_word_vectors(self._temp_data_dir, vocab)

    train_data = data.SnliData(fake_train_file, word2index)
    dev_data = data.SnliData(fake_train_file, word2index)
    test_data = data.SnliData(fake_train_file, word2index)

    # 2. Create a fake config.
    config = _test_spinn_config(
        data.WORD_VECTOR_LEN, 4,
        logdir=os.path.join(self._temp_data_dir, "logdir"))

    # 3. Test training of a SPINN model.
    trainer = spinn.train_or_infer_spinn(
        embed, word2index, train_data, dev_data, test_data, config)

    # 4. Load train loss values from the summary files and verify that they
    #    decrease with training.
    summary_file = glob.glob(os.path.join(config.logdir, "events.out.*"))[0]
    events = summary_test_util.events_from_file(summary_file)
    train_losses = [event.summary.value[0].simple_value for event in events
                    if event.summary.value
                    and event.summary.value[0].tag == "train/loss"]
    self.assertEqual(config.epochs, len(train_losses))
    self.assertLess(train_losses[-1], train_losses[0])

    # 5. Verify that checkpoints exist and contains all the expected variables.
    self.assertTrue(glob.glob(os.path.join(config.logdir, "ckpt*")))
    ckpt_variable_names = [
        item[0] for item in checkpoint_utils.list_variables(config.logdir)]
    self.assertIn("global_step", ckpt_variable_names)
    for v in trainer.variables:
      variable_name = v.name[:v.name.index(":")] if ":" in v.name else v.name
      self.assertIn(variable_name, ckpt_variable_names)


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

      config = _test_spinn_config(d_embed, d_out)
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
