# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Text RNN model stored as a SavedModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("export_dir", None, "Directory to export SavedModel.")


class TextRnnModel(tf.train.Checkpoint):
  """Text RNN model.

  A full generative text RNN model that can train and decode sentences from a
  starting word.
  """

  def __init__(self, vocab, emb_dim, buckets, state_size):
    super(TextRnnModel, self).__init__()
    self._buckets = buckets
    self._lstm_cell = tf.keras.layers.LSTMCell(units=state_size)
    self._rnn_layer = tf.keras.layers.RNN(
        self._lstm_cell, return_sequences=True)
    self._embeddings = tf.Variable(tf.random.uniform(shape=[buckets, emb_dim]))
    self._logit_layer = tf.keras.layers.Dense(buckets)
    self._set_up_vocab(vocab)

  def _tokenize(self, sentences):
    # Perform a minimalistic text preprocessing by removing punctuation and
    # splitting on spaces.
    normalized_sentences = tf.strings.regex_replace(
        input=sentences, pattern=r"\pP", rewrite="")
    sparse_tokens = tf.strings.split(normalized_sentences, " ")

    # Deal with a corner case: there is one empty sentence.
    sparse_tokens, _ = tf.sparse.fill_empty_rows(sparse_tokens, tf.constant(""))
    # Deal with a corner case: all sentences are empty.
    sparse_tokens = tf.sparse.reset_shape(sparse_tokens)

    return (sparse_tokens.indices, sparse_tokens.values,
            sparse_tokens.dense_shape)

  def _set_up_vocab(self, vocab_tokens):
    # TODO(vbardiovsky): Currently there is no real vocabulary, because
    # saved_model serialization does not support trackable resources. Add a real
    # vocabulary when it does.
    vocab_list = ["UNK"] * self._buckets
    for vocab_token in vocab_tokens:
      index = self._words_to_indices(vocab_token).numpy()
      vocab_list[index] = vocab_token
    # This is a variable representing an inverse index.
    self._vocab_tensor = tf.Variable(vocab_list)

  def _indices_to_words(self, indices):
    return tf.gather(self._vocab_tensor, indices)

  def _words_to_indices(self, words):
    return tf.strings.to_hash_bucket(words, self._buckets)

  @tf.function(input_signature=[tf.TensorSpec([None], tf.dtypes.string)])
  def train(self, sentences):
    token_ids, token_values, token_dense_shape = self._tokenize(sentences)
    tokens_sparse = tf.sparse.SparseTensor(
        indices=token_ids, values=token_values, dense_shape=token_dense_shape)
    tokens = tf.sparse.to_dense(tokens_sparse, default_value="")

    sparse_lookup_ids = tf.sparse.SparseTensor(
        indices=tokens_sparse.indices,
        values=self._words_to_indices(tokens_sparse.values),
        dense_shape=tokens_sparse.dense_shape)
    lookup_ids = tf.sparse.to_dense(sparse_lookup_ids, default_value=0)

    # Targets are the next word for each word of the sentence.
    tokens_ids_seq = lookup_ids[:, 0:-1]
    tokens_ids_target = lookup_ids[:, 1:]

    tokens_prefix = tokens[:, 0:-1]

    # Mask determining which positions we care about for a loss: all positions
    # that have a valid non-terminal token.
    mask = tf.logical_and(
        tf.logical_not(tf.equal(tokens_prefix, "")),
        tf.logical_not(tf.equal(tokens_prefix, "<E>")))

    input_mask = tf.cast(mask, tf.int32)

    with tf.GradientTape() as t:
      sentence_embeddings = tf.nn.embedding_lookup(self._embeddings,
                                                   tokens_ids_seq)

      lstm_initial_state = self._lstm_cell.get_initial_state(
          sentence_embeddings)

      lstm_output = self._rnn_layer(
          inputs=sentence_embeddings, initial_state=lstm_initial_state)

      # Stack LSTM outputs into a batch instead of a 2D array.
      lstm_output = tf.reshape(lstm_output, [-1, self._lstm_cell.output_size])

      logits = self._logit_layer(lstm_output)

      targets = tf.reshape(tokens_ids_target, [-1])
      weights = tf.cast(tf.reshape(input_mask, [-1]), tf.float32)

      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=targets, logits=logits)

      # Final loss is the mean loss for all token losses.
      final_loss = tf.math.divide(
          tf.reduce_sum(tf.multiply(losses, weights)),
          tf.reduce_sum(weights),
          name="final_loss")

    watched = t.watched_variables()
    gradients = t.gradient(final_loss, watched)

    for w, g in zip(watched, gradients):
      w.assign_sub(g)

    return final_loss

  @tf.function
  def decode_greedy(self, sequence_length, first_word):
    initial_state = self._lstm_cell.get_initial_state(
        dtype=tf.float32, batch_size=1)

    sequence = [first_word]
    current_word = first_word
    current_id = tf.expand_dims(self._words_to_indices(current_word), 0)
    current_state = initial_state

    for _ in range(sequence_length):
      token_embeddings = tf.nn.embedding_lookup(self._embeddings, current_id)
      lstm_outputs, current_state = self._lstm_cell(token_embeddings,
                                                    current_state)
      lstm_outputs = tf.reshape(lstm_outputs, [-1, self._lstm_cell.output_size])
      logits = self._logit_layer(lstm_outputs)
      softmax = tf.nn.softmax(logits)

      next_ids = tf.math.argmax(softmax, axis=1)
      next_words = self._indices_to_words(next_ids)[0]

      current_id = next_ids
      current_word = next_words
      sequence.append(current_word)

    return sequence


def main(argv):
  del argv

  sentences = ["<S> hello there <E>", "<S> how are you doing today <E>"]
  vocab = [
      "<S>", "<E>", "hello", "there", "how", "are", "you", "doing", "today"
  ]

  module = TextRnnModel(vocab=vocab, emb_dim=10, buckets=100, state_size=128)

  for _ in range(100):
    _ = module.train(tf.constant(sentences))

  # We have to call this function explicitly if we want it exported, because it
  # has no input_signature in the @tf.function decorator.
  decoded = module.decode_greedy(
      sequence_length=10, first_word=tf.constant("<S>"))
  _ = [d.numpy() for d in decoded]

  tf.saved_model.save(module, FLAGS.export_dir)


if __name__ == "__main__":
  app.run(main)
