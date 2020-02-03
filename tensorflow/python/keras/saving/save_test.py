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
import sys

import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.saving import model_config
from tensorflow.python.keras.saving import save
from tensorflow.python.ops import lookup_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import loader_impl

if sys.version_info >= (3, 4):
  import pathlib  # pylint:disable=g-import-not-at-top
try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None


class TestSaveModel(test.TestCase):

  def setUp(self):
    super(TestSaveModel, self).setUp()
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

  @test_util.run_v2_only
  def test_save_load_tf_string(self):
    path = os.path.join(self.get_temp_dir(), 'model')
    save.save_model(self.model, path, save_format='tf')
    save.load_model(path)

  @test_util.run_v2_only
  def test_save_load_tf_pathlib(self):
    if sys.version_info >= (3, 4):
      path = pathlib.Path(self.get_temp_dir()) / 'model'
      save.save_model(self.model, path, save_format='tf')
      save.load_model(path)

  @test_util.run_in_graph_and_eager_modes
  def test_saving_with_dense_features(self):
    cols = [
        feature_column_lib.numeric_column('a'),
        feature_column_lib.indicator_column(
            feature_column_lib.categorical_column_with_vocabulary_list(
                'b', ['one', 'two']))
    ]
    input_layers = {
        'a': keras.layers.Input(shape=(1,), name='a'),
        'b': keras.layers.Input(shape=(1,), name='b', dtype='string')
    }

    fc_layer = feature_column_lib.DenseFeatures(cols)(input_layers)
    output = keras.layers.Dense(10)(fc_layer)

    model = keras.models.Model(input_layers, output)

    model.compile(
        loss=keras.losses.MSE,
        optimizer='rmsprop',
        metrics=[keras.metrics.categorical_accuracy])

    config = model.to_json()
    loaded_model = model_config.model_from_json(config)

    inputs_a = np.arange(10).reshape(10, 1)
    inputs_b = np.arange(10).reshape(10, 1).astype('str')

    # Initialize tables for V1 lookup.
    if not context.executing_eagerly():
      self.evaluate(lookup_ops.tables_initializer())

    self.assertLen(loaded_model.predict({'a': inputs_a, 'b': inputs_b}), 10)

  @test_util.run_in_graph_and_eager_modes
  def test_saving_with_sequence_features(self):
    cols = [
        feature_column_lib.sequence_numeric_column('a'),
        feature_column_lib.indicator_column(
            feature_column_lib.sequence_categorical_column_with_vocabulary_list(
                'b', ['one', 'two']))
    ]
    input_layers = {
        'a':
            keras.layers.Input(shape=(None, 1), sparse=True, name='a'),
        'b':
            keras.layers.Input(
                shape=(None, 1), sparse=True, name='b', dtype='string')
    }

    fc_layer, _ = feature_column_lib.SequenceFeatures(cols)(input_layers)
    # TODO(tibell): Figure out the right dtype and apply masking.
    # sequence_length_mask = array_ops.sequence_mask(sequence_length)
    # x = keras.layers.GRU(32)(fc_layer, mask=sequence_length_mask)
    x = keras.layers.GRU(32)(fc_layer)
    output = keras.layers.Dense(10)(x)

    model = keras.models.Model(input_layers, output)

    model.compile(
        loss=keras.losses.MSE,
        optimizer='rmsprop',
        metrics=[keras.metrics.categorical_accuracy])

    config = model.to_json()
    loaded_model = model_config.model_from_json(config)

    batch_size = 10
    timesteps = 1

    values_a = np.arange(10, dtype=np.float32)
    indices_a = np.zeros((10, 3), dtype=np.int64)
    indices_a[:, 0] = np.arange(10)
    inputs_a = sparse_tensor.SparseTensor(indices_a, values_a,
                                          (batch_size, timesteps, 1))

    values_b = np.zeros(10, dtype=np.str)
    indices_b = np.zeros((10, 3), dtype=np.int64)
    indices_b[:, 0] = np.arange(10)
    inputs_b = sparse_tensor.SparseTensor(indices_b, values_b,
                                          (batch_size, timesteps, 1))

    # Initialize tables for V1 lookup.
    if not context.executing_eagerly():
      self.evaluate(lookup_ops.tables_initializer())

    self.assertLen(
        loaded_model.predict({
            'a': inputs_a,
            'b': inputs_b
        }, steps=1), batch_size)


if __name__ == '__main__':
  test.main()
