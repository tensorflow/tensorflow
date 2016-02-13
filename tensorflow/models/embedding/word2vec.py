# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Multi-threaded word2vec mini-batched skip-gram model.

Trains the model described in:
(Mikolov, et. al.) Efficient Estimation of Word Representations in Vector Space
ICLR 2013.
http://arxiv.org/abs/1301.3781
This model does traditional minibatching.

The key ops used are:
* placeholder for feeding in tensors for each example.
* embedding_lookup for fetching rows from the embedding matrix.
* sigmoid_cross_entropy_with_logits to calculate the loss.
* GradientDescentOptimizer for optimizing the loss.
* skipgram custom op that does input processing.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf

from tensorflow.models.embedding import gen_word2vec as word2vec

flags = tf.app.flags

flags.DEFINE_string("save_path", None, "Directory to write the model and "
                    "training summaries.")
flags.DEFINE_string("train_data", None, "Training text file. "
                    "E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string(
    "eval_data", None, "File consisting of analogies of four tokens."
    "embedding 2 - embedding 1 + embedding 3 should be close "
    "to embedding 4."
    "E.g. https://word2vec.googlecode.com/svn/trunk/questions-words.txt.")
flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.2, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 100,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 16,
                     "Number of training examples processed per step "
                     "(size of a minibatch).")
flags.DEFINE_integer("concurrent_steps", 12,
                     "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("min_count", 5,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-3,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
flags.DEFINE_boolean(
    "interactive", False,
    "If true, enters an IPython interactive session to play with the trained "
    "model. E.g., try model.analogy('france', 'paris', 'russia') and "
    "model.nearby(['proton', 'elephant', 'maxwell']")
flags.DEFINE_integer("statistics_interval", 5,
                     "Print statistics every n seconds.")
flags.DEFINE_integer("summary_interval", 5,
                     "Save training summary to file every n seconds (rounded "
                     "up to statistics interval.")
flags.DEFINE_integer("checkpoint_interval", 600,
                     "Checkpoint the model (i.e. save the parameters) every n "
                     "seconds (rounded up to statistics interval.")

FLAGS = flags.FLAGS


class Options(object):
  """Options used by our word2vec model."""

  def __init__(self):
    # Model options.

    # Embedding dimension.
    self.emb_dim = FLAGS.embedding_size

    # Training options.
    # The training text file.
    self.train_data = FLAGS.train_data

    # Number of negative samples per example.
    self.num_samples = FLAGS.num_neg_samples

    # The initial learning rate.
    self.learning_rate = FLAGS.learning_rate

    # Number of epochs to train. After these many epochs, the learning
    # rate decays linearly to zero and the training stops.
    self.epochs_to_train = FLAGS.epochs_to_train

    # Concurrent training steps.
    self.concurrent_steps = FLAGS.concurrent_steps

    # Number of examples for one training step.
    self.batch_size = FLAGS.batch_size

    # The number of words to predict to the left and right of the target word.
    self.window_size = FLAGS.window_size

    # The minimum number of word occurrences for it to be included in the
    # vocabulary.
    self.min_count = FLAGS.min_count

    # Subsampling threshold for word occurrence.
    self.subsample = FLAGS.subsample

    # How often to print statistics.
    self.statistics_interval = FLAGS.statistics_interval

    # How often to write to the summary file (rounds up to the nearest
    # statistics_interval).
    self.summary_interval = FLAGS.summary_interval

    # How often to write checkpoints (rounds up to the nearest statistics
    # interval).
    self.checkpoint_interval = FLAGS.checkpoint_interval

    # Where to write out summaries.
    self.save_path = FLAGS.save_path

    # Eval options.
    # The text file for eval.
    self.eval_data = FLAGS.eval_data


class Word2Vec(object):
  """Word2Vec model (Skipgram)."""

  def __init__(self, options, session):
    self._options = options
    self._session = session
    self._word2id = {}
    self._id2word = []
    self.build_graph()
    self.build_eval_graph()
    self.save_vocab()
    self._read_analogies()

  def _read_analogies(self):
    """Reads through the analogy question file.

    Returns:
      questions: a [n, 4] numpy array containing the analogy question's
                 word ids.
      questions_skipped: questions skipped due to unknown words.
    """
    questions = []
    questions_skipped = 0
    with open(self._options.eval_data, "rb") as analogy_f:
      for line in analogy_f:
        if line.startswith(b":"):  # Skip comments.
          continue
        words = line.strip().lower().split(b" ")
        ids = [self._word2id.get(w.strip()) for w in words]
        if None in ids or len(ids) != 4:
          questions_skipped += 1
        else:
          questions.append(np.array(ids))
    print("Eval analogy file: ", self._options.eval_data)
    print("Questions: ", len(questions))
    print("Skipped: ", questions_skipped)
    self._analogy_questions = np.array(questions, dtype=np.int32)

  def forward(self, examples, labels):
    """Build the graph for the forward pass."""
    opts = self._options

    # Declare all variables we need.
    # Embedding: [vocab_size, emb_dim]
    init_width = 0.5 / opts.emb_dim
    emb = tf.Variable(
        tf.random_uniform(
            [opts.vocab_size, opts.emb_dim], -init_width, init_width),
        name="emb")
    self._emb = emb

    # Softmax weight: [vocab_size, emb_dim]. Transposed.
    sm_w_t = tf.Variable(
        tf.zeros([opts.vocab_size, opts.emb_dim]),
        name="sm_w_t")

    # Softmax bias: [emb_dim].
    sm_b = tf.Variable(tf.zeros([opts.vocab_size]), name="sm_b")

    # Global step: scalar, i.e., shape [].
    self.global_step = tf.Variable(0, name="global_step")

    # Nodes to compute the nce loss w/ candidate sampling.
    labels_matrix = tf.reshape(
        tf.cast(labels,
                dtype=tf.int64),
        [opts.batch_size, 1])

    # Negative sampling.
    sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels_matrix,
        num_true=1,
        num_sampled=opts.num_samples,
        unique=True,
        range_max=opts.vocab_size,
        distortion=0.75,
        unigrams=opts.vocab_counts.tolist()))

    # Embeddings for examples: [batch_size, emb_dim]
    example_emb = tf.nn.embedding_lookup(emb, examples)

    # Weights for labels: [batch_size, emb_dim]
    true_w = tf.nn.embedding_lookup(sm_w_t, labels)
    # Biases for labels: [batch_size, 1]
    true_b = tf.nn.embedding_lookup(sm_b, labels)

    # Weights for sampled ids: [num_sampled, emb_dim]
    sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
    # Biases for sampled ids: [num_sampled, 1]
    sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

    # True logits: [batch_size, 1]
    true_logits = tf.reduce_sum(tf.mul(example_emb, true_w), 1) + true_b

    # Sampled logits: [batch_size, num_sampled]
    # We replicate sampled noise lables for all examples in the batch
    # using the matmul.
    sampled_b_vec = tf.reshape(sampled_b, [opts.num_samples])
    sampled_logits = tf.matmul(example_emb,
                               sampled_w,
                               transpose_b=True) + sampled_b_vec
    return true_logits, sampled_logits

  def nce_loss(self, true_logits, sampled_logits):
    """Build the graph for the NCE loss."""

    # cross-entropy(logits, labels)
    opts = self._options
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        true_logits, tf.ones_like(true_logits))
    sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        sampled_logits, tf.zeros_like(sampled_logits))

    # NCE-loss is the sum of the true and noise (sampled words)
    # contributions, averaged over the batch.
    nce_loss_tensor = (tf.reduce_sum(true_xent) +
                       tf.reduce_sum(sampled_xent)) / opts.batch_size
    return nce_loss_tensor

  def optimize(self, loss):
    """Build the graph to optimize the loss function."""

    # Optimizer nodes.
    # Linear learning rate decay.
    opts = self._options
    words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
    lr = opts.learning_rate * tf.maximum(
        0.0001, 1.0 - tf.cast(self._words, tf.float32) / words_to_train)
    self._lr = lr
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train = optimizer.minimize(loss,
                               global_step=self.global_step,
                               gate_gradients=optimizer.GATE_NONE)
    self._train = train

  def build_eval_graph(self):
    """Build the eval graph."""
    # Eval graph

    # Each analogy task is to predict the 4th word (d) given three
    # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
    # predict d=paris.

    # The eval feeds three vectors of word ids for a, b, c, each of
    # which is of size N, where N is the number of analogies we want to
    # evaluate in one batch.
    analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
    analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
    analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

    # Normalized word embeddings of shape [vocab_size, emb_dim].
    nemb = tf.nn.l2_normalize(self._emb, 1)

    # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
    # They all have the shape [N, emb_dim]
    a_emb = tf.gather(nemb, analogy_a)  # a's embs
    b_emb = tf.gather(nemb, analogy_b)  # b's embs
    c_emb = tf.gather(nemb, analogy_c)  # c's embs

    # We expect that d's embedding vectors on the unit hyper-sphere is
    # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
    target = c_emb + (b_emb - a_emb)

    # Compute cosine distance between each pair of target and vocab.
    # dist has shape [N, vocab_size].
    dist = tf.matmul(target, nemb, transpose_b=True)

    # For each question (row in dist), find the top 4 words.
    _, pred_idx = tf.nn.top_k(dist, 4)

    # Nodes for computing neighbors for a given word according to
    # their cosine distance.
    nearby_word = tf.placeholder(dtype=tf.int32)  # word id
    nearby_emb = tf.gather(nemb, nearby_word)
    nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
    nearby_val, nearby_idx = tf.nn.top_k(nearby_dist,
                                         min(1000, self._options.vocab_size))

    # Nodes in the construct graph which are used by training and
    # evaluation to run/feed/fetch.
    self._analogy_a = analogy_a
    self._analogy_b = analogy_b
    self._analogy_c = analogy_c
    self._analogy_pred_idx = pred_idx
    self._nearby_word = nearby_word
    self._nearby_val = nearby_val
    self._nearby_idx = nearby_idx

  def build_graph(self):
    """Build the graph for the full model."""
    opts = self._options
    # The training data. A text file.
    (words, counts, words_per_epoch, self._epoch, self._words, examples,
     labels) = word2vec.skipgram(filename=opts.train_data,
                                 batch_size=opts.batch_size,
                                 window_size=opts.window_size,
                                 min_count=opts.min_count,
                                 subsample=opts.subsample)
    (opts.vocab_words, opts.vocab_counts,
     opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
    opts.vocab_size = len(opts.vocab_words)
    print("Data file: ", opts.train_data)
    print("Vocab size: ", opts.vocab_size - 1, " + UNK")
    print("Words per epoch: ", opts.words_per_epoch)
    self._examples = examples
    self._labels = labels
    self._id2word = opts.vocab_words
    for i, w in enumerate(self._id2word):
      self._word2id[w] = i
    true_logits, sampled_logits = self.forward(examples, labels)
    loss = self.nce_loss(true_logits, sampled_logits)
    tf.scalar_summary("NCE loss", loss)
    self._loss = loss
    self.optimize(loss)

    # Properly initialize all variables.
    tf.initialize_all_variables().run()

    self.saver = tf.train.Saver()

  def save_vocab(self):
    """Save the vocabulary to a file so the model can be reloaded."""
    opts = self._options
    with open(os.path.join(opts.save_path, "vocab.txt"), "w") as f:
      for i in xrange(opts.vocab_size):
        f.write("%s %d\n" % (tf.compat.as_text(opts.vocab_words[i]),
                             opts.vocab_counts[i]))

  def _train_thread_body(self):
    initial_epoch, = self._session.run([self._epoch])
    while True:
      _, epoch = self._session.run([self._train, self._epoch])
      if epoch != initial_epoch:
        break

  def train(self):
    """Train the model."""
    opts = self._options

    initial_epoch, initial_words = self._session.run([self._epoch, self._words])

    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(opts.save_path,
                                            graph_def=self._session.graph_def)
    workers = []
    for _ in xrange(opts.concurrent_steps):
      t = threading.Thread(target=self._train_thread_body)
      t.start()
      workers.append(t)

    last_words, last_time, last_summary_time = initial_words, time.time(), 0
    last_checkpoint_time = 0
    while True:
      time.sleep(opts.statistics_interval)  # Reports our progress once a while.
      (epoch, step, loss, words, lr) = self._session.run(
          [self._epoch, self.global_step, self._loss, self._words, self._lr])
      now = time.time()
      last_words, last_time, rate = words, now, (words - last_words) / (
          now - last_time)
      print("Epoch %4d Step %8d: lr = %5.3f loss = %6.2f words/sec = %8.0f\r" %
            (epoch, step, lr, loss, rate), end="")
      sys.stdout.flush()
      if now - last_summary_time > opts.summary_interval:
        summary_str = self._session.run(summary_op)
        summary_writer.add_summary(summary_str, step)
        last_summary_time = now
      if now - last_checkpoint_time > opts.checkpoint_interval:
        self.saver.save(self._session,
                        opts.save_path + "model",
                        global_step=step.astype(int))
        last_checkpoint_time = now
      if epoch != initial_epoch:
        break

    for t in workers:
      t.join()

    return epoch

  def _predict(self, analogy):
    """Predict the top 4 answers for analogy questions."""
    idx, = self._session.run([self._analogy_pred_idx], {
        self._analogy_a: analogy[:, 0],
        self._analogy_b: analogy[:, 1],
        self._analogy_c: analogy[:, 2]
    })
    return idx

  def eval(self):
    """Evaluate analogy questions and reports accuracy."""

    # How many questions we get right at precision@1.
    correct = 0

    total = self._analogy_questions.shape[0]
    start = 0
    while start < total:
      limit = start + 2500
      sub = self._analogy_questions[start:limit, :]
      idx = self._predict(sub)
      start = limit
      for question in xrange(sub.shape[0]):
        for j in xrange(4):
          if idx[question, j] == sub[question, 3]:
            # Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
            correct += 1
            break
          elif idx[question, j] in sub[question, :3]:
            # We need to skip words already in the question.
            continue
          else:
            # The correct label is not the precision@1
            break
    print()
    print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total,
                                              correct * 100.0 / total))

  def analogy(self, w0, w1, w2):
    """Predict word w3 as in w0:w1 vs w2:w3."""
    wid = np.array([[self._word2id.get(w, 0) for w in [w0, w1, w2]]])
    idx = self._predict(wid)
    for c in [self._id2word[i] for i in idx[0, :]]:
      if c not in [w0, w1, w2]:
        return c
    return "unknown"

  def nearby(self, words, num=20):
    """Prints out nearby words given a list of words."""
    ids = np.array([self._word2id.get(x, 0) for x in words])
    vals, idx = self._session.run(
        [self._nearby_val, self._nearby_idx], {self._nearby_word: ids})
    for i in xrange(len(words)):
      print("\n%s\n=====================================" % (words[i]))
      for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
        print("%-20s %6.4f" % (self._id2word[neighbor], distance))


def _start_shell(local_ns=None):
  # An interactive shell is useful for debugging/development.
  import IPython
  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns)


def main(_):
  """Train a word2vec model."""
  if not FLAGS.train_data or not FLAGS.eval_data or not FLAGS.save_path:
    print("--train_data --eval_data and --save_path must be specified.")
    sys.exit(1)
  opts = Options()
  with tf.Graph().as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):
      model = Word2Vec(opts, session)
    for _ in xrange(opts.epochs_to_train):
      model.train()  # Process one epoch
      model.eval()  # Eval analogies.
    # Perform a final save.
    model.saver.save(session,
                     os.path.join(opts.save_path, "model.ckpt"),
                     global_step=model.global_step)
    if FLAGS.interactive:
      # E.g.,
      # [0]: model.analogy('france', 'paris', 'russia')
      # [1]: model.nearby(['proton', 'elephant', 'maxwell'])
      _start_shell(locals())


if __name__ == "__main__":
  tf.app.run()
