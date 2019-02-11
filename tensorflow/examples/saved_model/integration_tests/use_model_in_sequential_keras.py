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
"""Load and use RNN model stored as a SavedModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import numpy as np

import tensorflow as tf
# TODO(vbardiovsky): Remove when load is available.
from tensorflow.python.saved_model.load import load

tf.saved_model.load = load

FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", None, "Directory to load SavedModel from.")


# TODO(vbardiovsky): We should just reuse Keras's Lambda layer, when that
# enables to get trainable variables.
class LambdaLayer(tf.keras.layers.Layer):
  """Lambda layer with output shape inference."""

  def __init__(self, func, **kwargs):
    self._func = func
    super(LambdaLayer, self).__init__(**kwargs)

  def call(self, x):
    result = self._func(x)
    # TODO(vbardiovsky): Polymorphic function should return shaped tensor.
    result.set_shape(self.compute_output_shape(x.shape))
    return result

  def compute_output_shape(self, input_shape):
    # TODO(vbardiovsky): We should be able to get the embedding dimension from
    # the restored model.
    return (input_shape[0], 10)


def main(argv):
  del argv

  features = np.array(["my first sentence", "my second sentence"])
  labels = np.array([1, 0])

  dataset = tf.data.Dataset.from_tensor_slices((features, labels))

  embedding = tf.saved_model.load(FLAGS.model_dir)

  # Create the sequential keras model.
  model = tf.keras.Sequential()
  model.add(LambdaLayer(embedding, batch_input_shape=[None], dtype=tf.string))
  model.add(tf.keras.layers.Dense(100, activation="relu"))
  model.add(tf.keras.layers.Dense(50, activation="relu"))
  model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
  model.compile(
      optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

  model.fit_generator(generator=dataset.batch(1), epochs=5)


if __name__ == "__main__":
  app.run(main)
