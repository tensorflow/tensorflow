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
"""Benchmarks on Text classification with Transformer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras.benchmarks import benchmark_util


class TextWithTransformerBenchmark(tf.test.Benchmark):
  """Benchmarks for Text classification with Transformer
  using `tf.test.Benchmark`.
  """

  def __init__(self):
    super(TextWithTransformerBenchmark, self).__init__()
    self.max_feature = 20000
    self.max_len = 200
    (self.imdb_x, self.imdb_y), _ = tf.keras.datasets.imdb.load_data(
        num_words=self.max_feature)
    self.imdb_x = tf.keras.preprocessing.sequence.pad_sequences(
        self.imdb_x, maxlen=self.max_len)

  def _build_model(self):
    """Model from https://keras.io/examples/nlp/text_classification_with_transformer/."""
    embed_dim = 32
    num_heads = 2
    ff_dim = 32
    inputs = tf.keras.layers.Input(shape=(self.max_len,))
    embedding_layer = TokenAndPositionEmbedding(self.max_len, self.max_feature,
                                                embed_dim)
    x = embedding_layer(inputs)  #pylint: disable=not-callable
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)  #pylint: disable=not-callable
    x = tf.keras.layers.GlobalAvgPool1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(20, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

  # In each benchmark test, the required arguments for the
  # method `measure_performance` include:
  #   x: Input data, it could be Numpy or loaded from tfds.
  #   y: Target data. If `x` is a dataset or generator instance,
  #      `y` should not be specified.
  #   loss: Loss function for model.
  #   optimizer: Optimizer for model.
  #   Check more details in `measure_performance()` method of
  #   benchmark_util.
  def benchmark_text_classification_bs_128(self):
    """Measure performance with batch_size=128."""
    batch_size = 128
    metrics, wall_time, extras = benchmark_util.measure_performance(
        self._build_model,
        x=self.imdb_x,
        y=self.imdb_y,
        batch_size=batch_size,
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    metadata = benchmark_util.get_keras_examples_metadata(
        'transformer', batch_size)
    extras.update(metadata)
    self.report_benchmark(wall_time=wall_time, metrics=metrics, extras=extras)

  def benchmark_text_classification_bs_256(self):
    """Measure performance with batch_size=256."""
    batch_size = 256
    metrics, wall_time, extras = benchmark_util.measure_performance(
        self._build_model,
        x=self.imdb_x,
        y=self.imdb_y,
        batch_size=batch_size,
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    metadata = benchmark_util.get_keras_examples_metadata(
        'transformer', batch_size)
    extras.update(metadata)
    self.report_benchmark(wall_time=wall_time, metrics=metrics, extras=extras)

  def benchmark_text_classification_bs_512(self):
    """Measure performance with batch_size=512."""
    batch_size = 512
    metrics, wall_time, extras = benchmark_util.measure_performance(
        self._build_model,
        x=self.imdb_x,
        y=self.imdb_y,
        batch_size=batch_size,
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    metadata = benchmark_util.get_keras_examples_metadata(
        'transformer', batch_size)
    extras.update(metadata)
    self.report_benchmark(wall_time=wall_time, metrics=metrics, extras=extras)

  def benchmark_text_classification_bs_512_gpu_2(self):
    """Measure performance with batch_size=512, gpu=1 and

    distribution_strategy='mirrored'
    """
    batch_size = 512
    metrics, wall_time, extras = benchmark_util.measure_performance(
        self._build_model,
        x=self.imdb_x,
        y=self.imdb_y,
        batch_size=batch_size,
        num_gpus=2,
        distribution_strategy='mirrored',
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    metadata = benchmark_util.get_keras_examples_metadata(
        'transformer', batch_size)
    extras.update(metadata)
    self.report_benchmark(wall_time=wall_time, metrics=metrics, extras=extras)


class MultiHeadSelfAttention(tf.keras.layers.Layer):
  """Implement multi head self attention as a Keras layer."""

  def __init__(self, embed_dim, num_heads=8):
    super(MultiHeadSelfAttention, self).__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    if embed_dim % num_heads != 0:
      raise ValueError('embedding dimension = {embed_dim} should be divisible'
                       'by number of heads = {num_heads}')
    self.projection_dim = embed_dim // num_heads
    self.query_dense = tf.keras.layers.Dense(embed_dim)
    self.key_dense = tf.keras.layers.Dense(embed_dim)
    self.value_dense = tf.keras.layers.Dense(embed_dim)
    self.combine_heads = tf.keras.layers.Dense(embed_dim)

  def attention(self, query, key, value):
    score = tf.matmul(query, key, transpose_b=True)
    dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_score = score / tf.math.sqrt(dim_key)
    weights = tf.nn.softmax(scaled_score, axis=-1)
    output = tf.matmul(weights, value)
    return output, weights

  def separate_heads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, inputs):  #pylint: disable=arguments-differ
    # x.shape = [batch_size, seq_len, embedding_dim]
    batch_size = tf.shape(inputs)[0]
    query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
    key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
    value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
    query = self.separate_heads(
        query, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
    key = self.separate_heads(
        key, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
    value = self.separate_heads(
        value, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
    attention, _ = self.attention(query, key, value)
    attention = tf.transpose(
        attention, perm=[0, 2, 1,
                         3])  # (batch_size, seq_len, num_heads, projection_dim)
    concat_attention = tf.reshape(
        attention,
        (batch_size, -1, self.embed_dim))  # (batch_size, seq_len, embed_dim)
    output = self.combine_heads(
        concat_attention)  # (batch_size, seq_len, embed_dim)
    return output


class TransformerBlock(tf.keras.layers.Layer):
  """Implement a Transformer block as a layer."""

  def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
    super(TransformerBlock, self).__init__()
    self.att = MultiHeadSelfAttention(embed_dim, num_heads)
    self.ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation='relu'),
        tf.keras.layers.Dense(embed_dim)
    ])
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, inputs, training):  #pylint: disable=arguments-differ
    attn_output = self.att(inputs)  #pylint: disable=not-callable
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(inputs + attn_output)
    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output, training=training)
    return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
  """Implement embedding layer."""

  def __init__(self, maxlen, vocab_size, embed_dim):
    super(TokenAndPositionEmbedding, self).__init__()
    self.token_emb = tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embed_dim)
    self.pos_emb = tf.keras.layers.Embedding(
        input_dim=maxlen, output_dim=embed_dim)

  def call(self, x):  #pylint: disable=arguments-differ
    maxlen = tf.shape(x)[-1]
    positions = tf.range(start=0, limit=maxlen, delta=1)
    positions = self.pos_emb(positions)
    x = self.token_emb(x)
    return x + positions


if __name__ == '__main__':
  tf.test.main()
