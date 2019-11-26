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
"""Deploys a SavedModel with an MNIST classifier to TFLite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow.examples.saved_model.integration_tests import mnist_util

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'saved_model_dir', None,
    'Directory of the SavedModel to deploy.')
flags.DEFINE_bool(
    'use_fashion_mnist', False,
    'Use Fashion MNIST (products) instead of the real MNIST (digits).')
flags.DEFINE_bool(
    'fast_test_mode', False,
    'Limit amount of test data for running in unit tests.')
flags.DEFINE_string(
    'tflite_output_file', None,
    'The filename of the .tflite model file to write (optional).')
flags.DEFINE_bool(
    'reload_as_keras_model', True,
    'Also test tf.keras.models.load_model() on --saved_model_dir.')


def main(argv):
  del argv

  # First convert the SavedModel in a pristine environment.
  converter = tf.lite.TFLiteConverter.from_saved_model(FLAGS.saved_model_dir)
  lite_model_content = converter.convert()
  # Here is how you can save it for actual deployment.
  if FLAGS.tflite_output_file:
    with open(FLAGS.tflite_output_file, 'wb') as outfile:
      outfile.write(lite_model_content)
  # For testing, the TFLite model can be executed like this.
  interpreter = tf.lite.Interpreter(model_content=lite_model_content)
  def lite_model(images):
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], images)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

  # Load the SavedModel again for use as a test baseline.
  imported = tf.saved_model.load(FLAGS.saved_model_dir)
  def tf_model(images):
    output_dict = imported.signatures['serving_default'](tf.constant(images))
    logits, = output_dict.values()  # Unpack single value.
    return logits

  # Compare model outputs on the test inputs.
  (_, _), (x_test, _) = mnist_util.load_reshaped_data(
      use_fashion_mnist=FLAGS.use_fashion_mnist,
      fake_tiny_data=FLAGS.fast_test_mode)
  for i, x in enumerate(x_test):
    x = x[None, ...]  # Make batch of size 1.
    y_lite = lite_model(x)
    y_tf = tf_model(x)
    # This numpy primitive uses plain `raise` and works outside tf.TestCase.
    # Model outputs are probabilities that sum to 1, so atol makes sense here.
    np.testing.assert_allclose(
        y_lite, y_tf, rtol=0, atol=1e-5,
        err_msg='Mismatch with TF Lite at test example %d' % i)

  # Test that the SavedModel loads correctly with v1 load APIs as well.
  with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session() as session:
    tf.compat.v1.saved_model.load(
        session,
        [tf.compat.v1.saved_model.SERVING],
        FLAGS.saved_model_dir)

  # The SavedModel actually was a Keras Model; test that it also loads as that.
  if FLAGS.reload_as_keras_model:
    keras_model = tf.keras.models.load_model(FLAGS.saved_model_dir)
    for i, x in enumerate(x_test):
      x = x[None, ...]  # Make batch of size 1.
      y_tf = tf_model(x)
      y_keras = keras_model(x)
      # This numpy primitive uses plain `raise` and works outside tf.TestCase.
      # Model outputs are probabilities that sum to 1, so atol makes sense here.
      np.testing.assert_allclose(
          y_tf, y_keras, rtol=0, atol=1e-5,
          err_msg='Mismatch with Keras at test example %d' % i)

if __name__ == '__main__':
  app.run(main)
