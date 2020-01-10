"""Tests for functional style sequence-to-sequence models."""
import math
import random

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq


class Seq2SeqTest(tf.test.TestCase):

  def testRNNDecoder(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        inp = [tf.constant(0.5, shape=[2, 2]) for _ in xrange(2)]
        _, enc_states = rnn.rnn(rnn_cell.GRUCell(2), inp, dtype=tf.float32)
        dec_inp = [tf.constant(0.4, shape=[2, 2]) for _ in xrange(3)]
        cell = rnn_cell.OutputProjectionWrapper(rnn_cell.GRUCell(2), 4)
        dec, mem = seq2seq.rnn_decoder(dec_inp, enc_states[-1], cell)
        sess.run([tf.initialize_all_variables()])
        res = sess.run(dec)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0].shape, (2, 4))

        res = sess.run(mem)
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0].shape, (2, 2))

  def testBasicRNNSeq2Seq(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        inp = [tf.constant(0.5, shape=[2, 2]) for _ in xrange(2)]
        dec_inp = [tf.constant(0.4, shape=[2, 2]) for _ in xrange(3)]
        cell = rnn_cell.OutputProjectionWrapper(rnn_cell.GRUCell(2), 4)
        dec, mem = seq2seq.basic_rnn_seq2seq(inp, dec_inp, cell)
        sess.run([tf.initialize_all_variables()])
        res = sess.run(dec)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0].shape, (2, 4))

        res = sess.run(mem)
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0].shape, (2, 2))

  def testTiedRNNSeq2Seq(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        inp = [tf.constant(0.5, shape=[2, 2]) for _ in xrange(2)]
        dec_inp = [tf.constant(0.4, shape=[2, 2]) for _ in xrange(3)]
        cell = rnn_cell.OutputProjectionWrapper(rnn_cell.GRUCell(2), 4)
        dec, mem = seq2seq.tied_rnn_seq2seq(inp, dec_inp, cell)
        sess.run([tf.initialize_all_variables()])
        res = sess.run(dec)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0].shape, (2, 4))

        res = sess.run(mem)
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0].shape, (2, 2))

  def testEmbeddingRNNDecoder(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        inp = [tf.constant(0.5, shape=[2, 2]) for _ in xrange(2)]
        cell = rnn_cell.BasicLSTMCell(2)
        _, enc_states = rnn.rnn(cell, inp, dtype=tf.float32)
        dec_inp = [tf.constant(i, tf.int32, shape=[2]) for i in xrange(3)]
        dec, mem = seq2seq.embedding_rnn_decoder(dec_inp, enc_states[-1],
                                                 cell, 4)
        sess.run([tf.initialize_all_variables()])
        res = sess.run(dec)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0].shape, (2, 2))

        res = sess.run(mem)
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0].shape, (2, 4))

  def testEmbeddingRNNSeq2Seq(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        enc_inp = [tf.constant(1, tf.int32, shape=[2]) for i in xrange(2)]
        dec_inp = [tf.constant(i, tf.int32, shape=[2]) for i in xrange(3)]
        cell = rnn_cell.BasicLSTMCell(2)
        dec, mem = seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp, cell, 2, 5)
        sess.run([tf.variables.initialize_all_variables()])
        res = sess.run(dec)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0].shape, (2, 5))

        res = sess.run(mem)
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0].shape, (2, 4))

        # Test externally provided output projection.
        w = tf.get_variable("proj_w", [2, 5])
        b = tf.get_variable("proj_b", [5])
        with tf.variable_scope("proj_seq2seq"):
          dec, _ = seq2seq.embedding_rnn_seq2seq(
              enc_inp, dec_inp, cell, 2, 5, output_projection=(w, b))
        sess.run([tf.variables.initialize_all_variables()])
        res = sess.run(dec)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0].shape, (2, 2))

        # Test that previous-feeding model ignores inputs after the first.
        dec_inp2 = [tf.constant(0, tf.int32, shape=[2]) for _ in xrange(3)]
        tf.get_variable_scope().reuse_variables()
        d1, _ = seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp, cell, 2, 5,
                                              feed_previous=True)
        d2, _ = seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp2, cell, 2, 5,
                                              feed_previous=True)
        d3, _ = seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp2, cell, 2, 5,
                                              feed_previous=tf.constant(True))
        res1 = sess.run(d1)
        res2 = sess.run(d2)
        res3 = sess.run(d3)
        self.assertAllClose(res1, res2)
        self.assertAllClose(res1, res3)

  def testEmbeddingTiedRNNSeq2Seq(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        enc_inp = [tf.constant(1, tf.int32, shape=[2]) for i in xrange(2)]
        dec_inp = [tf.constant(i, tf.int32, shape=[2]) for i in xrange(3)]
        cell = rnn_cell.BasicLSTMCell(2)
        dec, mem = seq2seq.embedding_tied_rnn_seq2seq(enc_inp, dec_inp, cell, 5)
        sess.run([tf.variables.initialize_all_variables()])
        res = sess.run(dec)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0].shape, (2, 5))

        res = sess.run(mem)
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0].shape, (2, 4))

        # Test externally provided output projection.
        w = tf.get_variable("proj_w", [2, 5])
        b = tf.get_variable("proj_b", [5])
        with tf.variable_scope("proj_seq2seq"):
          dec, _ = seq2seq.embedding_tied_rnn_seq2seq(
              enc_inp, dec_inp, cell, 5, output_projection=(w, b))
        sess.run([tf.variables.initialize_all_variables()])
        res = sess.run(dec)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0].shape, (2, 2))

        # Test that previous-feeding model ignores inputs after the first.
        dec_inp2 = [tf.constant(0, tf.int32, shape=[2]) for _ in xrange(3)]
        tf.get_variable_scope().reuse_variables()
        d1, _ = seq2seq.embedding_tied_rnn_seq2seq(enc_inp, dec_inp, cell, 5,
                                                   feed_previous=True)
        d2, _ = seq2seq.embedding_tied_rnn_seq2seq(enc_inp, dec_inp2, cell, 5,
                                                   feed_previous=True)
        d3, _ = seq2seq.embedding_tied_rnn_seq2seq(
            enc_inp, dec_inp2, cell, 5, feed_previous=tf.constant(True))
        res1 = sess.run(d1)
        res2 = sess.run(d2)
        res3 = sess.run(d3)
        self.assertAllClose(res1, res2)
        self.assertAllClose(res1, res3)

  def testAttentionDecoder1(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        cell = rnn_cell.GRUCell(2)
        inp = [tf.constant(0.5, shape=[2, 2]) for _ in xrange(2)]
        enc_outputs, enc_states = rnn.rnn(cell, inp, dtype=tf.float32)
        attn_states = tf.concat(1, [tf.reshape(e, [-1, 1, cell.output_size])
                                    for e in enc_outputs])
        dec_inp = [tf.constant(0.4, shape=[2, 2]) for _ in xrange(3)]
        dec, mem = seq2seq.attention_decoder(dec_inp, enc_states[-1],
                                             attn_states, cell, output_size=4)
        sess.run([tf.initialize_all_variables()])
        res = sess.run(dec)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0].shape, (2, 4))

        res = sess.run(mem)
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0].shape, (2, 2))

  def testAttentionDecoder2(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        cell = rnn_cell.GRUCell(2)
        inp = [tf.constant(0.5, shape=[2, 2]) for _ in xrange(2)]
        enc_outputs, enc_states = rnn.rnn(cell, inp, dtype=tf.float32)
        attn_states = tf.concat(1, [tf.reshape(e, [-1, 1, cell.output_size])
                                    for e in enc_outputs])
        dec_inp = [tf.constant(0.4, shape=[2, 2]) for _ in xrange(3)]
        dec, mem = seq2seq.attention_decoder(dec_inp, enc_states[-1],
                                             attn_states, cell, output_size=4,
                                             num_heads=2)
        sess.run([tf.initialize_all_variables()])
        res = sess.run(dec)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0].shape, (2, 4))

        res = sess.run(mem)
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0].shape, (2, 2))

  def testEmbeddingAttentionDecoder(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        inp = [tf.constant(0.5, shape=[2, 2]) for _ in xrange(2)]
        cell = rnn_cell.GRUCell(2)
        enc_outputs, enc_states = rnn.rnn(cell, inp, dtype=tf.float32)
        attn_states = tf.concat(1, [tf.reshape(e, [-1, 1, cell.output_size])
                                    for e in enc_outputs])
        dec_inp = [tf.constant(i, tf.int32, shape=[2]) for i in xrange(3)]
        dec, mem = seq2seq.embedding_attention_decoder(dec_inp, enc_states[-1],
                                                       attn_states, cell, 4,
                                                       output_size=3)
        sess.run([tf.initialize_all_variables()])
        res = sess.run(dec)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0].shape, (2, 3))

        res = sess.run(mem)
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0].shape, (2, 2))

  def testEmbeddingAttentionSeq2Seq(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        enc_inp = [tf.constant(1, tf.int32, shape=[2]) for i in xrange(2)]
        dec_inp = [tf.constant(i, tf.int32, shape=[2]) for i in xrange(3)]
        cell = rnn_cell.BasicLSTMCell(2)
        dec, mem = seq2seq.embedding_attention_seq2seq(
            enc_inp, dec_inp, cell, 2, 5)
        sess.run([tf.initialize_all_variables()])
        res = sess.run(dec)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0].shape, (2, 5))

        res = sess.run(mem)
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0].shape, (2, 4))

        # Test externally provided output projection.
        w = tf.get_variable("proj_w", [2, 5])
        b = tf.get_variable("proj_b", [5])
        with tf.variable_scope("proj_seq2seq"):
          dec, _ = seq2seq.embedding_attention_seq2seq(
              enc_inp, dec_inp, cell, 2, 5, output_projection=(w, b))
        sess.run([tf.variables.initialize_all_variables()])
        res = sess.run(dec)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0].shape, (2, 2))

        # Test that previous-feeding model ignores inputs after the first.
        dec_inp2 = [tf.constant(0, tf.int32, shape=[2]) for _ in xrange(3)]
        tf.get_variable_scope().reuse_variables()
        d1, _ = seq2seq.embedding_attention_seq2seq(
            enc_inp, dec_inp, cell, 2, 5, feed_previous=True)
        d2, _ = seq2seq.embedding_attention_seq2seq(
            enc_inp, dec_inp2, cell, 2, 5, feed_previous=True)
        d3, _ = seq2seq.embedding_attention_seq2seq(
            enc_inp, dec_inp2, cell, 2, 5, feed_previous=tf.constant(True))
        res1 = sess.run(d1)
        res2 = sess.run(d2)
        res3 = sess.run(d3)
        self.assertAllClose(res1, res2)
        self.assertAllClose(res1, res3)

  def testSequenceLoss(self):
    with self.test_session() as sess:
      output_classes = 5
      logits = [tf.constant(i + 0.5, shape=[2, 5]) for i in xrange(3)]
      targets = [tf.constant(i, tf.int32, shape=[2]) for i in xrange(3)]
      weights = [tf.constant(1.0, shape=[2]) for i in xrange(3)]

      average_loss_per_example = seq2seq.sequence_loss(
          logits, targets, weights, output_classes,
          average_across_timesteps=True,
          average_across_batch=True)
      res = sess.run(average_loss_per_example)
      self.assertAllClose(res, 1.60944)

      average_loss_per_sequence = seq2seq.sequence_loss(
          logits, targets, weights, output_classes,
          average_across_timesteps=False,
          average_across_batch=True)
      res = sess.run(average_loss_per_sequence)
      self.assertAllClose(res, 4.828314)

      total_loss = seq2seq.sequence_loss(
          logits, targets, weights, output_classes,
          average_across_timesteps=False,
          average_across_batch=False)
      res = sess.run(total_loss)
      self.assertAllClose(res, 9.656628)

  def testSequenceLossByExample(self):
    with self.test_session() as sess:
      output_classes = 5
      logits = [tf.constant(i + 0.5, shape=[2, output_classes])
                for i in xrange(3)]
      targets = [tf.constant(i, tf.int32, shape=[2]) for i in xrange(3)]
      weights = [tf.constant(1.0, shape=[2]) for i in xrange(3)]

      average_loss_per_example = seq2seq.sequence_loss_by_example(
          logits, targets, weights, output_classes,
          average_across_timesteps=True)
      res = sess.run(average_loss_per_example)
      self.assertAllClose(res, np.asarray([1.609438, 1.609438]))

      loss_per_sequence = seq2seq.sequence_loss_by_example(
          logits, targets, weights, output_classes,
          average_across_timesteps=False)
      res = sess.run(loss_per_sequence)
      self.assertAllClose(res, np.asarray([4.828314, 4.828314]))

  def testModelWithBuckets(self):
    """Larger tests that does full sequence-to-sequence model training."""
    # We learn to copy 10 symbols in 2 buckets: length 4 and length 8.
    classes = 10
    buckets = [(4, 4), (8, 8)]
    # We use sampled softmax so we keep output projection separate.
    w = tf.get_variable("proj_w", [24, classes])
    w_t = tf.transpose(w)
    b = tf.get_variable("proj_b", [classes])
    # Here comes a sample Seq2Seq model using GRU cells.
    def SampleGRUSeq2Seq(enc_inp, dec_inp, weights):
      """Example sequence-to-sequence model that uses GRU cells."""
      def GRUSeq2Seq(enc_inp, dec_inp):
        cell = rnn_cell.MultiRNNCell([rnn_cell.GRUCell(24)] * 2)
        return seq2seq.embedding_attention_seq2seq(
            enc_inp, dec_inp, cell, classes, classes, output_projection=(w, b))
      targets = [dec_inp[i+1] for i in xrange(len(dec_inp) - 1)] + [0]
      def SampledLoss(inputs, labels):
        labels = tf.reshape(labels, [-1, 1])
        return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, 8, classes)
      return seq2seq.model_with_buckets(enc_inp, dec_inp, targets, weights,
                                        buckets, classes, GRUSeq2Seq,
                                        softmax_loss_function=SampledLoss)
    # Now we construct the copy model.
    with self.test_session() as sess:
      tf.set_random_seed(111)
      batch_size = 32
      inp = [tf.placeholder(tf.int32, shape=[None]) for _ in xrange(8)]
      out = [tf.placeholder(tf.int32, shape=[None]) for _ in xrange(8)]
      weights = [tf.ones_like(inp[0], dtype=tf.float32) for _ in xrange(8)]
      with tf.variable_scope("root"):
        _, losses = SampleGRUSeq2Seq(inp, out, weights)
        updates = []
        params = tf.all_variables()
        optimizer = tf.train.AdamOptimizer(0.03, epsilon=1e-5)
        for i in xrange(len(buckets)):
          full_grads = tf.gradients(losses[i], params)
          grads, _ = tf.clip_by_global_norm(full_grads, 30.0)
          update = optimizer.apply_gradients(zip(grads, params))
          updates.append(update)
        sess.run([tf.initialize_all_variables()])
      for ep in xrange(3):
        log_perp = 0.0
        for _ in xrange(50):
          bucket = random.choice(range(len(buckets)))
          length = buckets[bucket][0]
          i = [np.array([np.random.randint(9) + 1 for _ in xrange(batch_size)],
                        dtype=np.int32) for _ in xrange(length)]
          # 0 is our "GO" symbol here.
          o = [np.array([0 for _ in xrange(batch_size)], dtype=np.int32)] + i
          feed = {}
          for l in xrange(length):
            feed[inp[l].name] = i[l]
            feed[out[l].name] = o[l]
          if length < 8:  # For the 4-bucket, we need the 5th as target.
            feed[out[length].name] = o[length]
          res = sess.run([updates[bucket], losses[bucket]], feed)
          log_perp += float(res[1])
        perp = math.exp(log_perp / 100)
        print "step %d avg. perp %f" % ((ep + 1)*50, perp)
      self.assertLess(perp, 2.5)

if __name__ == "__main__":
  tf.test.main()
