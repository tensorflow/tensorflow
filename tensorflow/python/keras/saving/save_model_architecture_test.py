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
# pylint: disable=protected-access
"""Tests for saving/loading function for keras Model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.saving import model_architectures
from tensorflow.python.platform import test


@keras_parameterized.run_with_all_saved_model_formats
class TestModelArchitectures(keras_parameterized.TestCase):

  def _save_model_dir(self, dirname='saved_model'):
    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
    return os.path.join(temp_dir, dirname)

  def get_test_data(self, input_shape, target_shape):
    """Generate test dataset for testing."""
    if isinstance(input_shape, list):
      x = [
          np.random.random((2,) + input_shape[i][1:])
          for i in range(len(input_shape))
      ]
    else:
      x = np.random.random((2,) + input_shape[1:])

    if isinstance(target_shape, list):
      y = [
          np.random.random((2,) + target_shape[i][1:])
          for i in range(len(target_shape))
      ]
    else:
      y = np.random.random((2,) + target_shape[1:])

    return x, y

  def get_custom_objects(self):
    """Define custom_objects."""

    class CustomOpt(keras.optimizers.SGD):
      pass

    def custom_loss(y_true, y_pred):
      return keras.losses.mse(y_true, y_pred)

    return {'CustomOpt': CustomOpt,
            'custom_loss': custom_loss}

  @parameterized.named_parameters(*model_architectures.ALL_MODELS)
  def test_basic_saving_and_loading(self, model_fn):
    save_format = testing_utils.get_save_format()
    custom_objects = self.get_custom_objects()
    if 'subclassed_in_functional' in model_fn.__name__:
      subclass_custom_objects = {
          'MySubclassModel':
              model_architectures.MySubclassModel,
      }
      custom_objects.update(subclass_custom_objects)
    elif ('subclassed' in model_fn.__name__ and
          save_format in ['h5', 'hdf5', 'keras']):
      self.skipTest('Saving the model to HDF5 format requires the model to be '
                    'a Functional model or a Sequential model.')

    # TODO(b/147493902): Remove this skipTest once fixed.
    if ('stacked_rnn' in model_fn.__name__
        and save_format in ['h5', 'hdf5', 'keras']):
      self.skipTest('Stacked RNN model is not compatible with h5 save format.')

    saved_model_dir = self._save_model_dir()
    model_data = model_fn()
    model = model_data.model
    x_test, y_test = self.get_test_data(
        model_data.input_shape, model_data.target_shape)
    model.compile('rmsprop', 'mse')
    model.train_on_batch(x_test, y_test)

    # Save model.
    out1 = model.predict(x_test)
    keras.models.save_model(model, saved_model_dir, save_format=save_format)
    # Load model.
    loaded_model = keras.models.load_model(
        saved_model_dir,
        custom_objects=custom_objects)
    out2 = loaded_model.predict(x_test)

    self.assertAllClose(out1, out2, atol=1e-05)


if __name__ == '__main__':
  test.main()
