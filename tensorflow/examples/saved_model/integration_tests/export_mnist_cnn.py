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
"""Exports a convolutional feature extractor for MNIST in SavedModel format.

The feature extractor is a convolutional neural network plus a hidden layer
that gets trained as part of an MNIST classifier and then written to a
SavedModel (without the classification layer). From there, use_mnist_cnn.py
picks it up for transfer learning.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow as tf

from tensorflow.examples.saved_model.integration_tests import mnist_util

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'export_dir', None,
    'Directory of exported SavedModel.')
flags.DEFINE_integer(
    'epochs', 10,
    'Number of epochs to train.')
flags.DEFINE_bool(
    'fast_test_mode', False,
    'Shortcut training for running in unit tests.')


def make_feature_extractor():
  """Returns a Keras Model to compute a feature vector from MNIST images."""
  net = inp = tf.keras.Input(mnist_util.INPUT_SHAPE)
  net = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(net)
  net = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(net)
  net = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(net)
  # TODO(arnoegw): net = tf.keras.layers.Dropout(dropout_rate)(net)
  net = tf.keras.layers.Flatten()(net)
  # TODO(arnoegw): Add weight regularization.
  net = tf.keras.layers.Dense(128, activation='relu')(net)
  return tf.keras.Model(inputs=inp, outputs=net)


def make_classifier(feature_extractor, dropout_rate=0.5):
  """Returns a Keras Model to classify MNIST using feature_extractor."""
  net = inp = tf.keras.Input(mnist_util.INPUT_SHAPE)
  net = feature_extractor(net)
  net = tf.keras.layers.Dropout(dropout_rate)(net)
  net = tf.keras.layers.Dense(mnist_util.NUM_CLASSES, activation='softmax')(net)
  return tf.keras.Model(inputs=inp, outputs=net)


def wrap_keras_model_for_export(model):
  """Wraps `model` for saving and loading as SavedModel."""
  obj = tf.train.Checkpoint()
  obj.__call__ = tf.function(
      # TODO(arnoegw): Add args for training, hparams.
      lambda inputs: model(inputs),  # pylint: disable=unnecessary-lambda
      input_signature=[tf.TensorSpec(shape=(None,) + mnist_util.INPUT_SHAPE,
                                     dtype=tf.float32)])
  obj.trainable_variables = model.trainable_variables
  obj.variables = model.trainable_variables + model.non_trainable_variables
  return obj


def main(argv):
  del argv

  # Build a complete classifier model using a feature extractor.
  feature_extractor = make_feature_extractor()
  classifier = make_classifier(feature_extractor)

  # Train the complete model
  (x_train, y_train), (x_test, y_test) = mnist_util.load_reshaped_data(
      fake_tiny_data=FLAGS.fast_test_mode)
  classifier.compile(loss=tf.keras.losses.categorical_crossentropy,
                     optimizer=tf.keras.optimizers.SGD(),
                     metrics=['accuracy'])
  classifier.fit(x_train, y_train,
                 batch_size=128,
                 epochs=FLAGS.epochs,
                 verbose=1,
                 validation_data=(x_test, y_test))

  # Save the feature extractor to a framework-agnostic SavedModel for reuse.
  # Note that the feature_extractor object has not been compiled or fitted,
  # so it does not contain an optimizer and related state.
  exportable = wrap_keras_model_for_export(feature_extractor)
  tf.saved_model.save(exportable, FLAGS.export_dir)


if __name__ == '__main__':
  # tf.enable_v2_behavior()
  app.run(main)
