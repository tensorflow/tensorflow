# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Common utilities for our Keras preprocessing intergration tests."""

import os

import tensorflow as tf
preprocessing = tf.keras.layers.experimental.preprocessing

BATCH_SIZE = 64
DS_SIZE = BATCH_SIZE * 16
STEPS = DS_SIZE / BATCH_SIZE
VOCAB_SIZE = 100


def make_dataset():
  """Make a simple structured dataset.

  The dataset contains three feature columns.
    - float_col: an unnormalized numeric column.
    - int_col: an column of integer IDs.
    - string_col: a column of fixed vocabulary terms.

  Returns:
    The dataset.
  """
  tf.random.set_seed(197011)
  floats = tf.random.uniform((DS_SIZE, 1), maxval=10, dtype="float64")
  # Generate a 100 unique integer values, but over a wide range to showcase a
  # common use case for IntegerLookup.
  ints = tf.random.uniform((DS_SIZE, 1), maxval=VOCAB_SIZE, dtype="int64")
  ints = ints * 1000
  # Use a fixed vocabulary of strings from 0 to 99, to showcase loading a
  # vocabulary from a file.
  strings = tf.random.uniform((DS_SIZE, 1), maxval=VOCAB_SIZE, dtype="int64")
  strings = tf.strings.as_string(strings)
  features = {"float_col": floats, "int_col": ints, "string_col": strings}
  # Random binary label.
  labels = tf.random.uniform((DS_SIZE, 1), maxval=2, dtype="int64")
  ds = tf.data.Dataset.from_tensor_slices((features, labels))
  return ds


def make_preprocessing_model(file_dir):
  """Make a standalone preprocessing model."""
  # The name of our keras.Input should match the column name in the dataset.
  float_in = tf.keras.Input(shape=(1,), dtype="float64", name="float_col")
  int_in = tf.keras.Input(shape=(1,), dtype="int64", name="int_col")
  string_in = tf.keras.Input(shape=(1,), dtype="string", name="string_col")

  # We need to batch a dataset before adapting.
  ds = make_dataset().batch(BATCH_SIZE)
  # Normalize floats by adapting the mean and variance of the input.
  normalization = preprocessing.Normalization()
  normalization.adapt(ds.map(lambda features, labels: features["float_col"]))
  float_out = normalization(float_in)
  # Lookup ints by adapting a vocab of interger IDs.
  int_lookup = preprocessing.IntegerLookup()
  int_lookup.adapt(ds.map(lambda features, labels: features["int_col"]))
  int_out = int_lookup(int_in)
  # Lookup strings from a fixed file based vocabulary.
  string_vocab = list(str(i) for i in range(VOCAB_SIZE))
  vocab_file = os.path.join(file_dir, "vocab_file.txt")
  with open(vocab_file, "w") as f:
    f.write("\n".join(string_vocab))
  string_lookup = preprocessing.StringLookup(vocabulary=vocab_file)
  string_out = string_lookup(string_in)

  return tf.keras.Model(
      inputs=(float_in, int_in, string_in),
      outputs=(float_out, int_out, string_out))


def make_training_model():
  """Make a trainable model for the preprocessed inputs."""
  float_in = tf.keras.Input(shape=(1,), dtype="float64", name="float_col")
  # After preprocessing, both the string and int column are integer ready for
  # embedding.
  int_in = tf.keras.Input(shape=(1,), dtype="int64", name="int_col")
  string_in = tf.keras.Input(shape=(1,), dtype="int64", name="string_col")

  # Feed the lookup layers into an embedding.
  int_embedding = tf.keras.layers.Embedding(VOCAB_SIZE + 1, 8, input_length=1)
  int_out = int_embedding(int_in)
  int_out = tf.keras.layers.Flatten()(int_out)
  string_embedding = tf.keras.layers.Embedding(
      VOCAB_SIZE + 1, 8, input_length=1)
  string_out = string_embedding(string_in)
  string_out = tf.keras.layers.Flatten()(string_out)

  # Concatenate outputs.
  concatate = tf.keras.layers.Concatenate()
  # Feed our preprocessed inputs into a simple MLP.
  x = concatate((float_in, int_out, string_out))
  x = tf.keras.layers.Dense(32, activation="relu")(x)
  x = tf.keras.layers.Dense(32, activation="relu")(x)
  outputs = tf.keras.layers.Dense(1, activation="softmax")(x)
  return tf.keras.Model(inputs=(float_in, int_in, string_in), outputs=outputs)
