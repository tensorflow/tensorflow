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
"""Tests for Keras model saving code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.framework import test_util
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.saving import save
from tensorflow.python.platform import test
from tensorflow.python.saved_model import loader_impl

try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None


class TestSaveModel(test.TestCase):

  def setUp(self):
    self.model = testing_utils.get_small_sequential_mlp(1, 2, 3)
    self.subclassed_model = testing_utils.get_small_subclass_mlp(1, 2)

  def assert_h5_format(self, path):
    if h5py is not None:
      self.assertTrue(h5py.is_hdf5(path),
                      'Model saved at path {} is not a valid hdf5 file.'
                      .format(path))

  def assert_saved_model(self, path):
    loader_impl.parse_saved_model(path)

  @test_util.run_v2_only
  def test_save_format_defaults(self):
    path = os.path.join(self.get_temp_dir(), 'model_path')
    save.save_model(self.model, path)
    self.assert_saved_model(path)

  @test_util.run_v2_only
  def test_save_hdf5(self):
    path = os.path.join(self.get_temp_dir(), 'model')
    save.save_model(self.model, path, save_format='h5')
    self.assert_h5_format(path)
    with self.assertRaisesRegexp(
        NotImplementedError,
        'requires the model to be a Functional model or a Sequential model.'):
      save.save_model(self.subclassed_model, path, save_format='h5')

  @test_util.run_v2_only
  def test_save_tf(self):
    path = os.path.join(self.get_temp_dir(), 'model')
    save.save_model(self.model, path, save_format='tf')
    self.assert_saved_model(path)
    with self.assertRaisesRegexp(ValueError, 'input shapes have not been set'):
      save.save_model(self.subclassed_model, path, save_format='tf')
    self.subclassed_model.predict(np.random.random((3, 5)))
    save.save_model(self.subclassed_model, path, save_format='tf')
    self.assert_saved_model(path)

if __name__ == '__main__':
  test.main()
