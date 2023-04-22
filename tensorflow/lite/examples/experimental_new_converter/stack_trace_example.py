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
"""CodeLab for displaying error stack trace w/ MLIR-based converter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from absl import app

import tensorflow as tf


def suppress_exception(f):
  def wrapped():
    try:
      f()
    except:  # pylint: disable=bare-except
      pass
  return wrapped


class TestModule(tf.Module):
  """The test model has unsupported op."""

  @tf.function(input_signature=[tf.TensorSpec(shape=[3, 3], dtype=tf.float32)])
  def model(self, x):
    y = tf.math.reciprocal(x)  # Not supported
    return y + y


# comment out the `@suppress_exception` to display the stack trace
@suppress_exception
def test_from_saved_model():
  """displaying stack trace when converting saved model."""
  test_model = TestModule()
  saved_model_path = '/tmp/test.saved_model'
  save_options = tf.saved_model.SaveOptions(save_debug_info=True)
  tf.saved_model.save(test_model, saved_model_path, options=save_options)

  # load the model and convert
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
  converter.convert()


# comment out the `@suppress_exception` to display the stack trace
# @suppress_exception
def test_from_concrete_function():
  """displaying stack trace when converting concrete function."""
  @tf.function(input_signature=[tf.TensorSpec(shape=[3, 3], dtype=tf.float32)])
  def model(x):
    y = tf.math.reciprocal(x)  # not supported
    return y + y

  func = model.get_concrete_function()
  converter = tf.lite.TFLiteConverter.from_concrete_functions([func], model)
  converter.convert()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  sys.stdout.write('==== Testing from_concrete_functions ====\n')
  test_from_concrete_function()

  sys.stdout.write('==== Testing from_saved_model ====\n')
  test_from_saved_model()


if __name__ == '__main__':
  app.run(main)
