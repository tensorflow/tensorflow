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
"""Load and use text embedding module in sequential Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.examples.saved_model.integration_tests import util

FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", None, "Directory to load SavedModel from.")


def train(fine_tuning):
  """Build a Keras model and train with mock data."""
  features = np.array(["my first sentence", "my second sentence"])
  labels = np.array([1, 0])
  dataset = tf.data.Dataset.from_tensor_slices((features, labels))

  module = tf.saved_model.load(FLAGS.model_dir)

  # Create the sequential keras model.
  l = tf.keras.layers
  model = tf.keras.Sequential()
  model.add(l.Reshape((), batch_input_shape=[None, 1], dtype=tf.string))
  model.add(util.CustomLayer(module, output_shape=[10], trainable=fine_tuning))
  model.add(l.Dense(100, activation="relu"))
  model.add(l.Dense(50, activation="relu"))
  model.add(l.Dense(1, activation="sigmoid"))

  model.compile(
      optimizer="adam",
      loss="binary_crossentropy",
      metrics=["accuracy"],
      # TODO(b/124446120): Remove after fixed.
      run_eagerly=True)

  model.fit_generator(generator=dataset.batch(1), epochs=5)


def main(argv):
  del argv

  train(fine_tuning=False)
  train(fine_tuning=True)


if __name__ == "__main__":
  app.run(main)
