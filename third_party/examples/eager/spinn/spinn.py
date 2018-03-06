r"""Implementation of SPINN in TensorFlow eager execution.

SPINN: Stack-Augmented Parser-Interpreter Neural Network.

Ths file contains model definition and code for training the model.

The model definition is based on PyTorch implementation at:
  https://github.com/jekbradbury/examples/tree/spinn/snli

which was released under a BSD 3-Clause License at:
https://github.com/jekbradbury/examples/blob/spinn/LICENSE:

Copyright (c) 2017,
All rights reserved.

See ./LICENSE for more details.

Instructions for use:
* See `README.md` for details on how to prepare the SNLI and GloVe data.
* Suppose you have prepared the data at "/tmp/spinn-data", use the folloing
  command to train the model:

  ```bash
  python spinn.py --data_root /tmp/spinn-data --logdir /tmp/spinn-logs
  ```

  Checkpoints and TensorBoard summaries will be written to "/tmp/spinn-logs".

References:
* Bowman, S.R., Gauthier, J., Rastogi A., Gupta, R., Manning, C.D., & Potts, C.
  (2016). A Fast Unified Model for Parsing and Sentence Understanding.
  https://arxiv.org/abs/1603.06021
* Bradbury, J. (2017). Recursive Neural Networks with PyTorch.
  https://devblogs.nvidia.com/parallelforall/recursive-neural-networks-pytorch/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import os
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import tensorflow.contrib.eager as tfe
from tensorflow.contrib.eager.python.examples.spinn import data


def _bundle(lstm_iter):
  """Concatenate a list of Tensors along 1st axis and split result into two.

  Args:
    lstm_iter: A `list` of `N` dense `Tensor`s, each of which has the shape
      (R, 2 * M).

  Returns:
    A `list` of two dense `Tensor`s, each of which has the shape (N * R, M).
  """
  return tf.split(tf.concat(lstm_iter, 0), 2, axis=1)


def _unbundle(state):
  """Concatenate a list of Tensors along 2nd axis and split result.

  This is the inverse of `_bundle`.

  Args:
    state: A `list` of two dense `Tensor`s, each of which has the shape (R, M).

  Returns:
    A `list` of `R` dense `Tensors`, each of which has the shape (1, 2 * M).
  """
  return tf.split(tf.concat(state, 1), state[0].shape[0], axis=0)


class Reducer(tfe.Network):
  """A module that applies reduce operation on left and right vectors."""

  def __init__(self, size, tracker_size=None):
    super(Reducer, self).__init__()
    self.left = self.track_layer(tf.layers.Dense(5 * size, activation=None))
    self.right = self.track_layer(
        tf.layers.Dense(5 * size, activation=None, use_bias=False))
    if tracker_size is not None:
      self.track = self.track_layer(
          tf.layers.Dense(5 * size, activation=None, use_bias=False))
    else:
      self.track = None

  def call(self, left_in, right_in, tracking=None):
    """Invoke forward pass of the Reduce module.

    This method feeds a linear combination of `left_in`, `right_in` and
    `tracking` into a Tree LSTM and returns the output of the Tree LSTM.

    Args:
      left_in: A list of length L. Each item is a dense `Tensor` with
        the shape (1, n_dims). n_dims is the size of the embedding vector.
      right_in: A list of the same length as `left_in`. Each item should have
        the same shape as the items of `left_in`.
      tracking: Optional list of the same length as `left_in`. Each item is a
        dense `Tensor` with shape (1, tracker_size * 2). tracker_size is the
        size of the Tracker's state vector.

    Returns:
      Output: A list of length batch_size. Each item has the shape (1, n_dims).
    """
    left, right = _bundle(left_in), _bundle(right_in)
    lstm_in = self.left(left[0]) + self.right(right[0])
    if self.track and tracking:
      lstm_in += self.track(_bundle(tracking)[0])
    return _unbundle(self._tree_lstm(left[1], right[1], lstm_in))

  def _tree_lstm(self, c1, c2, lstm_in):
    a, i, f1, f2, o = tf.split(lstm_in, 5, axis=1)
    c = tf.tanh(a) * tf.sigmoid(i) + tf.sigmoid(f1) * c1 + tf.sigmoid(f2) * c2
    h = tf.sigmoid(o) * tf.tanh(c)
    return h, c


class Tracker(tfe.Network):
  """A module that tracks the history of the sentence with an LSTM."""

  def __init__(self, tracker_size, predict):
    """Constructor of Tracker.

    Args:
      tracker_size: Number of dimensions of the underlying `LSTMCell`.
      predict: (`bool`) Whether prediction mode is enabled.
    """
    super(Tracker, self).__init__()
    self._rnn = self.track_layer(tf.nn.rnn_cell.LSTMCell(tracker_size))
    self._state_size = tracker_size
    if predict:
      self._transition = self.track_layer(tf.layers.Dense(4))
    else:
      self._transition = None

  def reset_state(self):
    self.state = None

  def call(self, bufs, stacks):
    """Invoke the forward pass of the Tracker module.

    This method feeds the concatenation of the top two elements of the stacks
    into an LSTM cell and returns the resultant state of the LSTM cell.

    Args:
      bufs: A `list` of length batch_size. Each item is a `list` of
        max_sequence_len (maximum sequence length of the batch). Each item
        of the nested list is a dense `Tensor` of shape (1, d_proj), where
        d_proj is the size of the word embedding vector or the size of the
        vector space that the word embedding vector is projected to.
      stacks: A `list` of size batch_size. Each item is a `list` of
        variable length corresponding to the current height of the stack.
        Each item of the nested list is a dense `Tensor` of shape (1, d_proj).

    Returns:
      1. A list of length batch_size. Each item is a dense `Tensor` of shape
        (1, d_tracker * 2).
      2.  If under predict mode, result of applying a Dense layer on the
        first state vector of the RNN. Else, `None`.
    """
    buf = _bundle([buf[-1] for buf in bufs])[0]
    stack1 = _bundle([stack[-1] for stack in stacks])[0]
    stack2 = _bundle([stack[-2] for stack in stacks])[0]
    x = tf.concat([buf, stack1, stack2], 1)
    if self.state is None:
      batch_size = int(x.shape[0])
      zeros = tf.zeros((batch_size, self._state_size), dtype=tf.float32)
      self.state = [zeros, zeros]
    _, self.state = self._rnn(x, self.state)
    unbundled = _unbundle(self.state)
    if self._transition:
      return unbundled, self._transition(self.state[0])
    else:
      return unbundled, None


class SPINN(tfe.Network):
  """Stack-augmented Parser-Interpreter Neural Network.

  See https://arxiv.org/abs/1603.06021 for more details.
  """

  def __init__(self, config):
    """Constructor of SPINN.

    Args:
      config: A `namedtupled` with the following attributes.
        d_proj - (`int`) number of dimensions of the vector space to project the
          word embeddings to.
        d_tracker - (`int`) number of dimensions of the Tracker's state vector.
        d_hidden - (`int`) number of the dimensions of the hidden state, for the
          Reducer module.
        n_mlp_layers - (`int`) number of multi-layer perceptron layers to use to
          convert the output of the `Feature` module to logits.
        predict - (`bool`) Whether the Tracker will enabled predictions.
    """
    super(SPINN, self).__init__()
    self.config = config
    self.reducer = self.track_layer(Reducer(config.d_hidden, config.d_tracker))
    if config.d_tracker is not None:
      self.tracker = self.track_layer(Tracker(config.d_tracker, config.predict))
    else:
      self.tracker = None

  def call(self, buffers, transitions, training=False):
    """Invoke the forward pass of the SPINN model.

    Args:
      buffers: Dense `Tensor` of shape
        (max_sequence_len, batch_size, config.d_proj).
      transitions: Dense `Tensor` with integer values that represent the parse
        trees of the sentences. A value of 2 indicates "reduce"; a value of 3
        indicates "shift". Shape: (max_sequence_len * 2 - 3, batch_size).
      training: Whether the invocation is under training mode.

    Returns:
      Output `Tensor` of shape (batch_size, config.d_embed).
    """
    max_sequence_len, batch_size, d_proj = (int(x) for x in buffers.shape)

    # Split the buffers into left and right word items and put the initial
    # items in a stack.
    splitted = tf.split(
        tf.reshape(tf.transpose(buffers, [1, 0, 2]), [-1, d_proj]),
        max_sequence_len * batch_size, axis=0)
    buffers = [splitted[k:k + max_sequence_len]
               for k in xrange(0, len(splitted), max_sequence_len)]
    stacks = [[buf[0], buf[0]] for buf in buffers]

    if self.tracker:
      # Reset tracker state for new batch.
      self.tracker.reset_state()

    num_transitions = transitions.shape[0]

    # Iterate through transitions and perform the appropriate stack-pop, reduce
    # and stack-push operations.
    transitions = transitions.numpy()
    for i in xrange(num_transitions):
      trans = transitions[i]
      if self.tracker:
        # Invoke tracker to obtain the current tracker states for the sentences.
        tracker_states, trans_hypothesis = self.tracker(buffers, stacks)
        if trans_hypothesis:
          trans = tf.argmax(trans_hypothesis, axis=-1)
      else:
        tracker_states = itertools.repeat(None)
      lefts, rights, trackings = [], [], []
      for transition, buf, stack, tracking in zip(
          trans, buffers, stacks, tracker_states):
        if int(transition) == 3:  # Shift.
          stack.append(buf.pop())
        elif int(transition) == 2:  # Reduce.
          rights.append(stack.pop())
          lefts.append(stack.pop())
          trackings.append(tracking)

      if rights:
        reducer_output = self.reducer(lefts, rights, trackings)
        reduced = iter(reducer_output)

        for transition, stack in zip(trans, stacks):
          if int(transition) == 2:  # Reduce.
            stack.append(next(reduced))
    return _bundle([stack.pop() for stack in stacks])[0]


class SNLIClassifier(tfe.Network):
  """SNLI Classifier Model.

  A model aimed at solving the SNLI (Standford Natural Language Inference)
  task, using the SPINN model from above. For details of the task, see:
    https://nlp.stanford.edu/projects/snli/
  """

  def __init__(self, config, embed):
    """Constructor of SNLICLassifier.

    Args:
      config: A namedtuple containing required configurations for the model. It
        needs to have the following attributes.
        projection - (`bool`) whether the word vectors are to be projected onto
          another vector space (of `d_proj` dimensions).
        d_proj - (`int`) number of dimensions of the vector space to project the
          word embeddings to.
        embed_dropout - (`float`) dropout rate for the word embedding vectors.
        n_mlp_layers - (`int`) number of multi-layer perceptron (MLP) layers to
          use to convert the output of the `Feature` module to logits.
        mlp_dropout - (`float`) dropout rate of the MLP layers.
        d_out - (`int`) number of dimensions of the final output of the MLP
          layers.
        lr - (`float`) learning rate.
      embed: A embedding matrix of shape (vocab_size, d_embed).
    """
    super(SNLIClassifier, self).__init__()
    self.config = config
    self.embed = tf.constant(embed)

    self.projection = self.track_layer(tf.layers.Dense(config.d_proj))
    self.embed_bn = self.track_layer(tf.layers.BatchNormalization())
    self.embed_dropout = self.track_layer(
        tf.layers.Dropout(rate=config.embed_dropout))
    self.encoder = self.track_layer(SPINN(config))

    self.feature_bn = self.track_layer(tf.layers.BatchNormalization())
    self.feature_dropout = self.track_layer(
        tf.layers.Dropout(rate=config.mlp_dropout))

    self.mlp_dense = []
    self.mlp_bn = []
    self.mlp_dropout = []
    for _ in xrange(config.n_mlp_layers):
      self.mlp_dense.append(self.track_layer(tf.layers.Dense(config.d_mlp)))
      self.mlp_bn.append(
          self.track_layer(tf.layers.BatchNormalization()))
      self.mlp_dropout.append(
          self.track_layer(tf.layers.Dropout(rate=config.mlp_dropout)))
    self.mlp_output = self.track_layer(tf.layers.Dense(
        config.d_out,
        kernel_initializer=tf.random_uniform_initializer(minval=-5e-3,
                                                         maxval=5e-3)))

  def call(self,
           premise,
           premise_transition,
           hypothesis,
           hypothesis_transition,
           training=False):
    """Invoke the forward pass the SNLIClassifier model.

    Args:
      premise: The word indices of the premise sentences, with shape
        (max_prem_seq_len, batch_size).
      premise_transition: The transitions for the premise sentences, with shape
        (max_prem_seq_len * 2 - 3, batch_size).
      hypothesis: The word indices of the hypothesis sentences, with shape
        (max_hypo_seq_len, batch_size).
      hypothesis_transition: The transitions for the hypothesis sentences, with
        shape (max_hypo_seq_len * 2 - 3, batch_size).
      training: Whether the invocation is under training mode.

    Returns:
      The logits, as a dense `Tensor` of shape (batch_size, d_out), where d_out
      is the size of the output vector.
    """
    # Perform embedding lookup on the premise and hypothesis inputs, which have
    # the word-index format.
    premise_embed = tf.nn.embedding_lookup(self.embed, premise)
    hypothesis_embed = tf.nn.embedding_lookup(self.embed, hypothesis)

    if self.config.projection:
      # Project the embedding vectors to another vector space.
      premise_embed = self.projection(premise_embed)
      hypothesis_embed = self.projection(hypothesis_embed)

    # Perform batch normalization and dropout on the possibly projected word
    # vectors.
    premise_embed = self.embed_bn(premise_embed, training=training)
    hypothesis_embed = self.embed_bn(hypothesis_embed, training=training)
    premise_embed = self.embed_dropout(premise_embed, training=training)
    hypothesis_embed = self.embed_dropout(hypothesis_embed, training=training)

    # Run the batch-normalized and dropout-processed word vectors through the
    # SPINN encoder.
    premise = self.encoder(premise_embed, premise_transition,
                           training=training)
    hypothesis = self.encoder(hypothesis_embed, hypothesis_transition,
                              training=training)

    # Combine encoder outputs for premises and hypotheses into logits.
    # Then apply batch normalization and dropuout on the logits.
    logits = tf.concat(
        [premise, hypothesis, premise - hypothesis, premise * hypothesis], 1)
    logits = self.feature_dropout(
        self.feature_bn(logits, training=training), training=training)

    # Apply the multi-layer perceptron on the logits.
    for dense, bn, dropout in zip(
        self.mlp_dense, self.mlp_bn, self.mlp_dropout):
      logits = tf.nn.elu(dense(logits))
      logits = dropout(bn(logits, training=training), training=training)
    logits = self.mlp_output(logits)
    return logits


class SNLIClassifierTrainer(object):
  """A class that coordinates the training of an SNLIClassifier."""

  def __init__(self, snli_classifier, lr):
    """Constructor of SNLIClassifierTrainer.

    Args:
      snli_classifier: An instance of `SNLIClassifier`.
      lr: Learning rate.
    """
    self._model = snli_classifier
    # Create a custom learning rate Variable for the RMSProp optimizer, because
    # the learning rate needs to be manually decayed later (see
    # decay_learning_rate()).
    self._learning_rate = tfe.Variable(lr, name="learning_rate")
    self._optimizer = tf.train.RMSPropOptimizer(self._learning_rate,
                                                epsilon=1e-6)

  def loss(self, labels, logits):
    """Calculate the loss given a batch of data.

    Args:
      labels: The truth labels, with shape (batch_size,).
      logits: The logits output from the forward pass of the SNLIClassifier
        model, with shape (batch_size, d_out), where d_out is the output
        dimension size of the SNLIClassifier.

    Returns:
      The loss value, as a scalar `Tensor`.
    """
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits))

  def train_batch(self,
                  labels,
                  premise,
                  premise_transition,
                  hypothesis,
                  hypothesis_transition):
    """Train model on batch of data.

    Args:
      labels: The truth labels, with shape (batch_size,).
      premise: The word indices of the premise sentences, with shape
        (max_prem_seq_len, batch_size).
      premise_transition: The transitions for the premise sentences, with shape
        (max_prem_seq_len * 2 - 3, batch_size).
      hypothesis: The word indices of the hypothesis sentences, with shape
        (max_hypo_seq_len, batch_size).
      hypothesis_transition: The transitions for the hypothesis sentences, with
        shape (max_hypo_seq_len * 2 - 3, batch_size).

    Returns:
      1. loss value as a scalar `Tensor`.
      2. logits as a dense `Tensor` of shape (batch_size, d_out), where d_out is
        the output dimension size of the SNLIClassifier.
    """
    with tfe.GradientTape() as tape:
      tape.watch(self._model.variables)
      logits = self._model(premise,
                           premise_transition,
                           hypothesis,
                           hypothesis_transition,
                           training=True)
      loss = self.loss(labels, logits)
    gradients = tape.gradient(loss, self._model.variables)
    self._optimizer.apply_gradients(zip(gradients, self._model.variables),
                                    global_step=tf.train.get_global_step())
    return loss, logits

  def decay_learning_rate(self, decay_by):
    """Decay learning rate of the optimizer by factor decay_by."""
    self._learning_rate.assign(self._learning_rate * decay_by)
    print("Decayed learning rate of optimizer to: %s" %
          self._learning_rate.numpy())

  @property
  def learning_rate(self):
    return self._learning_rate

  @property
  def model(self):
    return self._model

  @property
  def variables(self):
    return (self._model.variables + [self.learning_rate] +
            self._optimizer.variables())


def _batch_n_correct(logits, label):
  """Calculate number of correct predictions in a batch.

  Args:
    logits: A logits Tensor of shape `(batch_size, num_categories)` and dtype
      `float32`.
    label: A labels Tensor of shape `(batch_size,)` and dtype `int64`

  Returns:
    Number of correct predictions.
  """
  return tf.reduce_sum(
      tf.cast((tf.equal(
          tf.argmax(logits, axis=1), label)), tf.float32)).numpy()


def _evaluate_on_dataset(snli_data, batch_size, trainer, use_gpu):
  """Run evaluation on a dataset.

  Args:
    snli_data: The `data.SnliData` to use in this evaluation.
    batch_size: The batch size to use during this evaluation.
    trainer: An instance of `SNLIClassifierTrainer to use for this
      evaluation.
    use_gpu: Whether GPU is being used.

  Returns:
    1. Average loss across all examples of the dataset.
    2. Average accuracy rate across all examples of the dataset.
  """
  mean_loss = tfe.metrics.Mean()
  accuracy = tfe.metrics.Accuracy()
  for label, prem, prem_trans, hypo, hypo_trans in _get_dataset_iterator(
      snli_data, batch_size):
    if use_gpu:
      label, prem, hypo = label.gpu(), prem.gpu(), hypo.gpu()
    logits = trainer.model(prem, prem_trans, hypo, hypo_trans, training=False)
    loss_val = trainer.loss(label, logits)
    batch_size = tf.shape(label)[0]
    mean_loss(loss_val, weights=batch_size.gpu() if use_gpu else batch_size)
    accuracy(tf.argmax(logits, axis=1), label)
  return mean_loss.result().numpy(), accuracy.result().numpy()


def _get_dataset_iterator(snli_data, batch_size):
  """Get a data iterator for a split of SNLI data.

  Args:
    snli_data: A `data.SnliData` object.
    batch_size: The desired batch size.

  Returns:
    A dataset iterator.
  """
  with tf.device("/device:CPU:0"):
    # Some tf.data ops, such as ShuffleDataset, are available only on CPU.
    dataset = tf.data.Dataset.from_generator(
        snli_data.get_generator(batch_size),
        (tf.int64, tf.int64, tf.int64, tf.int64, tf.int64))
    dataset = dataset.shuffle(snli_data.num_batches(batch_size))
    return tfe.Iterator(dataset)


def train_or_infer_spinn(embed,
                         word2index,
                         train_data,
                         dev_data,
                         test_data,
                         config):
  """Perform Training or Inference on a SPINN model.

  Args:
    embed: The embedding matrix as a float32 numpy array with shape
      [vocabulary_size, word_vector_len]. word_vector_len is the length of a
      word embedding vector.
    word2index: A `dict` mapping word to word index.
    train_data: An instance of `data.SnliData`, for the train split.
    dev_data: Same as above, for the dev split.
    test_data: Same as above, for the test split.
    config: A configuration object. See the argument to this Python binary for
      details.

  Returns:
    If `config.inference_premise ` and `config.inference_hypothesis` are not
      `None`, i.e., inference mode: the logits for the possible labels of the
      SNLI data set, as a `Tensor` of three floats.
    else:
      The trainer object.
  Raises:
    ValueError: if only one of config.inference_premise and
      config.inference_hypothesis is specified.
  """
  # TODO(cais): Refactor this function into separate one for training and
  #   inference.
  use_gpu = tfe.num_gpus() > 0 and not config.force_cpu
  device = "gpu:0" if use_gpu else "cpu:0"
  print("Using device: %s" % device)

  if ((config.inference_premise and not config.inference_hypothesis) or
      (not config.inference_premise and config.inference_hypothesis)):
    raise ValueError(
        "--inference_premise and --inference_hypothesis must be both "
        "specified or both unspecified, but only one is specified.")

  if config.inference_premise:
    # Inference mode.
    inference_sentence_pair = [
        data.encode_sentence(config.inference_premise, word2index),
        data.encode_sentence(config.inference_hypothesis, word2index)]
  else:
    inference_sentence_pair = None

  log_header = (
      "  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss"
      "     Accuracy  Dev/Accuracy")
  log_template = (
      "{:>6.0f} {:>5.0f} {:>9.0f} {:>5.0f}/{:<5.0f} {:>7.0f}% {:>8.6f} {} "
      "{:12.4f} {}")
  dev_log_template = (
      "{:>6.0f} {:>5.0f} {:>9.0f} {:>5.0f}/{:<5.0f} {:>7.0f}% {:>8.6f} "
      "{:8.6f} {:12.4f} {:12.4f}")

  summary_writer = tf.contrib.summary.create_file_writer(
      config.logdir, flush_millis=10000)

  with tf.device(device), \
       summary_writer.as_default(), \
       tf.contrib.summary.always_record_summaries():
    with tfe.restore_variables_on_create(
        tf.train.latest_checkpoint(config.logdir)):
      model = SNLIClassifier(config, embed)
      global_step = tf.train.get_or_create_global_step()
      trainer = SNLIClassifierTrainer(model, config.lr)

    if inference_sentence_pair:
      # Inference mode.
      with tfe.restore_variables_on_create(
          tf.train.latest_checkpoint(config.logdir)):
        prem, prem_trans = inference_sentence_pair[0]
        hypo, hypo_trans = inference_sentence_pair[1]
        hypo_trans = inference_sentence_pair[1][1]
        inference_logits = model(  # pylint: disable=not-callable
            tf.constant(prem), tf.constant(prem_trans),
            tf.constant(hypo), tf.constant(hypo_trans), training=False)
        inference_logits = inference_logits[0][1:]
        max_index = tf.argmax(inference_logits)
        print("\nInference logits:")
        for i, (label, logit) in enumerate(
            zip(data.POSSIBLE_LABELS, inference_logits)):
          winner_tag = " (winner)" if max_index == i else ""
          print("  {0:<16}{1:.6f}{2}".format(label + ":", logit, winner_tag))
      return inference_logits

    train_len = train_data.num_batches(config.batch_size)
    start = time.time()
    iterations = 0
    mean_loss = tfe.metrics.Mean()
    accuracy = tfe.metrics.Accuracy()
    print(log_header)
    for epoch in xrange(config.epochs):
      batch_idx = 0
      for label, prem, prem_trans, hypo, hypo_trans in _get_dataset_iterator(
          train_data, config.batch_size):
        if use_gpu:
          label, prem, hypo = label.gpu(), prem.gpu(), hypo.gpu()
          # prem_trans and hypo_trans are used for dynamic control flow and can
          # remain on CPU. Same in _evaluate_on_dataset().

        iterations += 1
        with tfe.restore_variables_on_create(
            tf.train.latest_checkpoint(config.logdir)):
          batch_train_loss, batch_train_logits = trainer.train_batch(
              label, prem, prem_trans, hypo, hypo_trans)
        batch_size = tf.shape(label)[0]
        mean_loss(batch_train_loss.numpy(),
                  weights=batch_size.gpu() if use_gpu else batch_size)
        accuracy(tf.argmax(batch_train_logits, axis=1), label)

        if iterations % config.save_every == 0:
          all_variables = trainer.variables + [global_step]
          saver = tfe.Saver(all_variables)
          saver.save(os.path.join(config.logdir, "ckpt"),
                     global_step=global_step)

        if iterations % config.dev_every == 0:
          dev_loss, dev_frac_correct = _evaluate_on_dataset(
              dev_data, config.batch_size, trainer, use_gpu)
          print(dev_log_template.format(
              time.time() - start,
              epoch, iterations, 1 + batch_idx, train_len,
              100.0 * (1 + batch_idx) / train_len,
              mean_loss.result(), dev_loss,
              accuracy.result() * 100.0, dev_frac_correct * 100.0))
          tf.contrib.summary.scalar("dev/loss", dev_loss)
          tf.contrib.summary.scalar("dev/accuracy", dev_frac_correct)
        elif iterations % config.log_every == 0:
          mean_loss_val = mean_loss.result()
          accuracy_val = accuracy.result()
          print(log_template.format(
              time.time() - start,
              epoch, iterations, 1 + batch_idx, train_len,
              100.0 * (1 + batch_idx) / train_len,
              mean_loss_val, " " * 8, accuracy_val * 100.0, " " * 12))
          tf.contrib.summary.scalar("train/loss", mean_loss_val)
          tf.contrib.summary.scalar("train/accuracy", accuracy_val)
          # Reset metrics.
          mean_loss = tfe.metrics.Mean()
          accuracy = tfe.metrics.Accuracy()

        batch_idx += 1
      if (epoch + 1) % config.lr_decay_every == 0:
        trainer.decay_learning_rate(config.lr_decay_by)

    test_loss, test_frac_correct = _evaluate_on_dataset(
        test_data, config.batch_size, trainer, use_gpu)
    print("Final test loss: %g; accuracy: %g%%" %
          (test_loss, test_frac_correct * 100.0))

  return trainer


def main(_):
  config = FLAGS

  # Load embedding vectors.
  vocab = data.load_vocabulary(FLAGS.data_root)
  word2index, embed = data.load_word_vectors(FLAGS.data_root, vocab)

  if not (config.inference_premise or config.inference_hypothesis):
    print("Loading train, dev and test data...")
    train_data = data.SnliData(
        os.path.join(FLAGS.data_root, "snli/snli_1.0/snli_1.0_train.txt"),
        word2index, sentence_len_limit=FLAGS.sentence_len_limit)
    dev_data = data.SnliData(
        os.path.join(FLAGS.data_root, "snli/snli_1.0/snli_1.0_dev.txt"),
        word2index, sentence_len_limit=FLAGS.sentence_len_limit)
    test_data = data.SnliData(
        os.path.join(FLAGS.data_root, "snli/snli_1.0/snli_1.0_test.txt"),
        word2index, sentence_len_limit=FLAGS.sentence_len_limit)
  else:
    train_data = None
    dev_data = None
    test_data = None

  train_or_infer_spinn(
      embed, word2index, train_data, dev_data, test_data, config)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description=
      "TensorFlow eager implementation of the SPINN SNLI classifier.")
  parser.add_argument("--data_root", type=str, default="/tmp/spinn-data",
                      help="Root directory in which the training data and "
                      "embedding matrix are found. See README.md for how to "
                      "generate such a directory.")
  parser.add_argument("--sentence_len_limit", type=int, default=-1,
                      help="Maximum allowed sentence length (# of words). "
                      "The default of -1 means unlimited.")
  parser.add_argument("--logdir", type=str, default="/tmp/spinn-logs",
                      help="Directory in which summaries will be written for "
                      "TensorBoard.")
  parser.add_argument("--inference_premise", type=str, default=None,
                      help="Premise sentence for inference. Must be "
                      "accompanied by --inference_hypothesis. If specified, "
                      "will override all training parameters and perform "
                      "inference.")
  parser.add_argument("--inference_hypothesis", type=str, default=None,
                      help="Hypothesis sentence for inference. Must be "
                      "accompanied by --inference_premise. If specified, will "
                      "override all training parameters and perform inference.")
  parser.add_argument("--epochs", type=int, default=50,
                      help="Number of epochs to train.")
  parser.add_argument("--batch_size", type=int, default=128,
                      help="Batch size to use during training.")
  parser.add_argument("--d_proj", type=int, default=600,
                      help="Dimensions to project the word embedding vectors "
                      "to.")
  parser.add_argument("--d_hidden", type=int, default=300,
                      help="Size of the hidden layer of the Tracker.")
  parser.add_argument("--d_out", type=int, default=4,
                      help="Output dimensions of the SNLIClassifier.")
  parser.add_argument("--d_mlp", type=int, default=1024,
                      help="Size of each layer of the multi-layer perceptron "
                      "of the SNLICLassifier.")
  parser.add_argument("--n_mlp_layers", type=int, default=2,
                      help="Number of layers in the multi-layer perceptron "
                      "of the SNLICLassifier.")
  parser.add_argument("--d_tracker", type=int, default=64,
                      help="Size of the tracker LSTM.")
  parser.add_argument("--log_every", type=int, default=50,
                      help="Print log and write TensorBoard summary every _ "
                      "training batches.")
  parser.add_argument("--lr", type=float, default=2e-3,
                      help="Initial learning rate.")
  parser.add_argument("--lr_decay_by", type=float, default=0.75,
                      help="The ratio to multiply the learning rate by every "
                      "time the learning rate is decayed.")
  parser.add_argument("--lr_decay_every", type=float, default=1,
                      help="Decay the learning rate every _ epoch(s).")
  parser.add_argument("--dev_every", type=int, default=1000,
                      help="Run evaluation on the dev split every _ training "
                      "batches.")
  parser.add_argument("--save_every", type=int, default=1000,
                      help="Save checkpoint every _ training batches.")
  parser.add_argument("--embed_dropout", type=float, default=0.08,
                      help="Word embedding dropout rate.")
  parser.add_argument("--mlp_dropout", type=float, default=0.07,
                      help="SNLIClassifier multi-layer perceptron dropout "
                      "rate.")
  parser.add_argument("--no-projection", action="store_false",
                      dest="projection",
                      help="Whether word embedding vectors are projected to "
                      "another set of vectors (see d_proj).")
  parser.add_argument("--predict_transitions", action="store_true",
                      dest="predict",
                      help="Whether the Tracker will perform prediction.")
  parser.add_argument("--force_cpu", action="store_true", dest="force_cpu",
                      help="Force use CPU-only regardless of whether a GPU is "
                      "available.")
  FLAGS, unparsed = parser.parse_known_args()

  tfe.run(main=main, argv=[sys.argv[0]] + unparsed)
