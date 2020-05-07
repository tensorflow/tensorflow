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
"""Text embedding model stored as a SavedModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl import app
from absl import flags

import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("export_dir", None, "Directory to export SavedModel.")


def write_vocabulary_file(vocabulary):
  """Write temporary vocab file for module construction."""
  tmpdir = tempfile.mkdtemp()
  vocabulary_file = os.path.join(tmpdir, "tokens.txt")
  with tf.io.gfile.GFile(vocabulary_file, "w") as f:
    for entry in vocabulary:
      f.write(entry + "\n")
  return vocabulary_file


class TextEmbeddingModel(tf.train.Checkpoint):
  """Text embedding model.

  A text embeddings model that takes a sentences on input and outputs the
  sentence embedding.
  """

  def __init__(self, vocabulary, emb_dim, oov_buckets):
    super(TextEmbeddingModel, self).__init__()
    self._oov_buckets = oov_buckets
    self._total_size = len(vocabulary) + oov_buckets
    # Assign the table initializer to this instance to ensure the asset
    # it depends on is saved with the SavedModel.
    self._table_initializer = tf.lookup.TextFileInitializer(
        write_vocabulary_file(vocabulary), tf.string,
        tf.lookup.TextFileIndex.WHOLE_LINE, tf.int64,
        tf.lookup.TextFileIndex.LINE_NUMBER)
    self._table = tf.lookup.StaticVocabularyTable(
        self._table_initializer, num_oov_buckets=self._oov_buckets)
    self.embeddings = tf.Variable(
        tf.random.uniform(shape=[self._total_size, emb_dim]))
    self.variables = [self.embeddings]
    self.trainable_variables = self.variables

  def _tokenize(self, sentences):
    # Perform a minimalistic text preprocessing by removing punctuation and
    # splitting on spaces.
    normalized_sentences = tf.strings.regex_replace(
        input=sentences, pattern=r"\pP", rewrite="")
    normalized_sentences = tf.reshape(normalized_sentences, [-1])
    sparse_tokens = tf.strings.split(normalized_sentences, " ").to_sparse()

    # Deal with a corner case: there is one empty sentence.
    sparse_tokens, _ = tf.sparse.fill_empty_rows(sparse_tokens, tf.constant(""))
    # Deal with a corner case: all sentences are empty.
    sparse_tokens = tf.sparse.reset_shape(sparse_tokens)
    sparse_token_ids = self._table.lookup(sparse_tokens.values)

    return (sparse_tokens.indices, sparse_token_ids, sparse_tokens.dense_shape)

  @tf.function(input_signature=[tf.TensorSpec([None], tf.dtypes.string)])
  def __call__(self, sentences):
    token_ids, token_values, token_dense_shape = self._tokenize(sentences)

    return tf.nn.safe_embedding_lookup_sparse(
        embedding_weights=self.embeddings,
        sparse_ids=tf.sparse.SparseTensor(token_ids, token_values,
                                          token_dense_shape),
        sparse_weights=None,
        combiner="sqrtn")


def main(argv):
  del argv

  vocabulary = ["cat", "is", "on", "the", "mat"]
  module = TextEmbeddingModel(vocabulary=vocabulary, emb_dim=10, oov_buckets=10)
  tf.saved_model.save(module, FLAGS.export_dir)


if __name__ == "__main__":
  app.run(main)
