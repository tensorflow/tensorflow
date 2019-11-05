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
"""Load and use text embedding module in a Dataset map function."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", None, "Directory to load SavedModel from.")


def train():
  """Build a Keras model and train with mock data."""
  module = tf.saved_model.load(FLAGS.model_dir)
  def _map_fn(features, labels):
    features = tf.expand_dims(features, 0)
    features = module(features)
    features = tf.squeeze(features, 0)
    return features, labels

  features = np.array(["my first sentence", "my second sentence"])
  labels = np.array([1, 0])
  dataset = tf.data.Dataset.from_tensor_slices((features, labels)).map(_map_fn)

  # Create the sequential keras model.
  l = tf.keras.layers
  model = tf.keras.Sequential()
  model.add(l.Dense(10, activation="relu"))
  model.add(l.Dense(1, activation="sigmoid"))

  model.compile(
      optimizer="adam",
      loss="binary_crossentropy",
      metrics=["accuracy"])

  model.fit_generator(generator=dataset.batch(10), epochs=5)

  # This is testing that a model using a SavedModel can be re-exported again,
  # e.g. to catch issues such as b/142231881.
  tf.saved_model.save(model, tempfile.mkdtemp())


def main(argv):
  del argv

  train()


if __name__ == "__main__":
  tf.enable_v2_behavior()
  app.run(main)
