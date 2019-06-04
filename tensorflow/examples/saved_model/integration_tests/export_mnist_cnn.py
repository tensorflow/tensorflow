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
import tensorflow.compat.v2 as tf

from tensorflow.examples.saved_model.integration_tests import mnist_util
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect

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
flags.DEFINE_bool(
    'export_print_hparams', False,
    'If true, the exported function will print its effective hparams.')


def make_feature_extractor(l2_strength, dropout_rate):
  """Returns a Keras Model to compute a feature vector from MNIST images."""
  regularizer = lambda: tf.keras.regularizers.l2(l2_strength)
  net = inp = tf.keras.Input(mnist_util.INPUT_SHAPE)
  net = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv1',
                               kernel_regularizer=regularizer())(net)
  net = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2',
                               kernel_regularizer=regularizer())(net)
  net = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(net)
  net = tf.keras.layers.Dropout(dropout_rate, name='dropout1')(net)
  net = tf.keras.layers.Flatten(name='flatten')(net)
  net = tf.keras.layers.Dense(10, activation='relu', name='dense1',
                              kernel_regularizer=regularizer())(net)
  return tf.keras.Model(inputs=inp, outputs=net)


def set_feature_extractor_hparams(model, dropout_rate):
  model.get_layer('dropout1').rate = dropout_rate


def make_classifier(feature_extractor, l2_strength, dropout_rate=0.5):
  """Returns a Keras Model to classify MNIST using feature_extractor."""
  regularizer = lambda: tf.keras.regularizers.l2(l2_strength)
  net = inp = tf.keras.Input(mnist_util.INPUT_SHAPE)
  net = feature_extractor(net)
  net = tf.keras.layers.Dropout(dropout_rate)(net)
  net = tf.keras.layers.Dense(mnist_util.NUM_CLASSES, activation='softmax',
                              kernel_regularizer=regularizer())(net)
  return tf.keras.Model(inputs=inp, outputs=net)


def wrap_keras_model_for_export(model, batch_input_shape,
                                set_hparams, default_hparams):
  """Wraps `model` for saving and loading as SavedModel."""
  if default_hparams is None: default_hparams = {}
  hparam_keys = list(default_hparams.keys())
  hparam_defaults = tuple(default_hparams.values())
  # The goal is to save a function with this argspec...
  argspec = tf_inspect.FullArgSpec(
      args=(['inputs', 'training'] + hparam_keys),
      defaults=((False,) + hparam_defaults),
      varargs=None, varkw=None,
      kwonlyargs=[], kwonlydefaults=None,
      annotations={})
  # ...and this behavior:
  def call_fn(inputs, training, *args):
    if FLAGS.export_print_hparams:
      args = [tf.keras.backend.print_tensor(args[i], 'training=%s and %s='
                                            % (training, hparam_keys[i]))
              for i in range(len(args))]
    kwargs = dict(zip(hparam_keys, args))
    if kwargs: set_hparams(model, **kwargs)
    return model(inputs, training=training)
  # We cannot spell out `args` in def statement for call_fn, but since
  # tf.function uses tf_inspect, we can use tf_decorator to wrap it with
  # the desired argspec.
  def wrapped(*args, **kwargs):  # TODO(arnoegw): Can we use call_fn itself?
    return call_fn(*args, **kwargs)
  traced_call_fn = tf.function(autograph=False)(
      tf_decorator.make_decorator(call_fn, wrapped, decorator_argspec=argspec))
  # Now we need to trigger traces for
  # - training set to Python values True or False (hence two traces),
  # - tensor inputs of the expected nesting, shape and dtype,
  # - tensor-valued kwargs for hparams, with caller-side defaults.
  # Tracing with partially determined shapes requires an input signature,
  # so we initiate tracing from a helper function with only tensor inputs.
  @tf.function(autograph=False)
  def trigger_traces(inputs, **kwargs):
    return tuple(traced_call_fn(inputs, training=training, **kwargs)
                 for training in (True, False))
  inputs_spec = tf.TensorSpec(shape=batch_input_shape, dtype=tf.float32)
  hparams_spec = {name: tf.TensorSpec.from_tensor(tf.constant(value))
                  for name, value in default_hparams.items()}
  _ = trigger_traces.get_concrete_function(inputs_spec, **hparams_spec)

  # Assemble the output object.
  obj = tf.train.Checkpoint()
  obj.__call__ = traced_call_fn
  obj.trainable_variables = model.trainable_variables
  obj.variables = model.trainable_variables + model.non_trainable_variables
  obj.regularization_losses = [_get_traced_loss(model, i)
                               for i in range(len(model.losses))]
  return obj


def _get_traced_loss(model, i):
  """Returns tf.function for model.losses[i] with a trace for zero args.

  The intended usage is
    [_get_traced_loss(model, i) for i in range(len(model.losses))]
  This is better than
    [tf.function(lambda: model.losses[i], input_signature=[]) for i ...]
  because it avoids capturing a loop index in a lambda, and removes any
  chance of deferring the trace.

  Args:
    model: a Keras Model.
    i: an integer between from 0 up to but to len(model.losses).
  """
  f = tf.function(lambda: model.losses[i])
  _ = f.get_concrete_function()
  return f


def main(argv):
  del argv

  # Build a complete classifier model using a feature extractor.
  default_hparams = dict(dropout_rate=0.25)
  l2_strength = 0.01  # Not a hparam for inputs -> outputs.
  feature_extractor = make_feature_extractor(l2_strength=l2_strength,
                                             **default_hparams)
  classifier = make_classifier(feature_extractor, l2_strength=l2_strength)

  # Train the complete model.
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
  exportable = wrap_keras_model_for_export(feature_extractor,
                                           (None,) + mnist_util.INPUT_SHAPE,
                                           set_feature_extractor_hparams,
                                           default_hparams)
  tf.saved_model.save(exportable, FLAGS.export_dir)


if __name__ == '__main__':
  app.run(main)
