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

import collections
import os
import shutil
import sys
import tempfile
import warnings

from absl.testing import parameterized
import numpy as np
from six import string_types

from tensorflow.python import keras
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras import combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import losses
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.feature_column import dense_features
from tensorflow.python.keras.feature_column import sequence_feature_column as ksfc
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.saving import model_config
from tensorflow.python.keras.saving import save
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.training import training as training_module


if sys.version_info >= (3, 6):
  import pathlib  # pylint:disable=g-import-not-at-top
try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None


class TestSaveModel(test.TestCase, parameterized.TestCase):

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

  @testing_utils.run_v2_only
  def test_save_format_defaults(self):
    path = os.path.join(self.get_temp_dir(), 'model_path')
    save.save_model(self.model, path)
    self.assert_saved_model(path)

  @testing_utils.run_v2_only
  def test_save_format_defaults_pathlib(self):
    if sys.version_info < (3, 6):
      self.skipTest('pathlib is only available for python version >= 3.6')
    path = pathlib.Path(self.get_temp_dir()) / 'model_path'
    save.save_model(self.model, path)
    self.assert_saved_model(path)

  @testing_utils.run_v2_only
  def test_save_hdf5(self):
    path = os.path.join(self.get_temp_dir(), 'model')
    save.save_model(self.model, path, save_format='h5')
    self.assert_h5_format(path)
    with self.assertRaisesRegex(
        NotImplementedError,
        'requires the model to be a Functional model or a Sequential model.'):
      save.save_model(self.subclassed_model, path, save_format='h5')

  @testing_utils.run_v2_only
  def test_save_load_hdf5_pathlib(self):
    if sys.version_info < (3, 6):
      self.skipTest('pathlib is only available for python version >= 3.6')
    path = pathlib.Path(self.get_temp_dir()) / 'model'
    save.save_model(self.model, path, save_format='h5')
    save.load_model(path)

  @testing_utils.run_v2_only
  def test_save_tf(self):
    path = os.path.join(self.get_temp_dir(), 'model')
    save.save_model(self.model, path, save_format='tf')
    self.assert_saved_model(path)
    with self.assertRaisesRegex(ValueError, 'input shapes have not been set'):
      save.save_model(self.subclassed_model, path, save_format='tf')
    self.subclassed_model.predict(np.random.random((3, 5)))
    save.save_model(self.subclassed_model, path, save_format='tf')
    self.assert_saved_model(path)

  @testing_utils.run_v2_only
  def test_save_load_tf_string(self):
    path = os.path.join(self.get_temp_dir(), 'model')
    save.save_model(self.model, path, save_format='tf')
    save.load_model(path)

  @testing_utils.run_v2_only
  def test_save_load_tf_pathlib(self):
    if sys.version_info < (3, 6):
      self.skipTest('pathlib is only available for python version >= 3.6')
    path = pathlib.Path(self.get_temp_dir()) / 'model'
    save.save_model(self.model, path, save_format='tf')
    save.load_model(path)

  @testing_utils.run_v2_only
  def test_save_load_weights_tf_pathlib(self):
    if sys.version_info < (3, 6):
      self.skipTest('pathlib is only available for python version >= 3.6')
    path = pathlib.Path(self.get_temp_dir()) / 'model'
    self.model.save_weights(path, save_format='tf')
    self.model.load_weights(path)

  @testing_utils.run_v2_only
  def test_save_load_weights_hdf5_pathlib(self):
    if sys.version_info < (3, 6):
      self.skipTest('pathlib is only available for python version >= 3.6')
    path = pathlib.Path(self.get_temp_dir()) / 'model'
    self.model.save_weights(path, save_format='h5')
    self.model.load_weights(path)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
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

    fc_layer = dense_features.DenseFeatures(cols)(input_layers)
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

    with self.cached_session():
      # Initialize tables for V1 lookup.
      if not context.executing_eagerly():
        self.evaluate(lookup_ops.tables_initializer())

      self.assertLen(loaded_model.predict({'a': inputs_a, 'b': inputs_b}), 10)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
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

    fc_layer, _ = ksfc.SequenceFeatures(cols)(input_layers)
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

    with self.cached_session():
      # Initialize tables for V1 lookup.
      if not context.executing_eagerly():
        self.evaluate(lookup_ops.tables_initializer())

      self.assertLen(
          loaded_model.predict({
              'a': inputs_a,
              'b': inputs_b
          }, steps=1), batch_size)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_saving_h5_for_rnn_layers(self):
    # See https://github.com/tensorflow/tensorflow/issues/35731 for details.
    inputs = keras.Input([10, 91], name='train_input')
    rnn_layers = [
        keras.layers.LSTMCell(size, recurrent_dropout=0, name='rnn_cell%d' % i)
        for i, size in enumerate([512, 512])
    ]
    rnn_output = keras.layers.RNN(
        rnn_layers, return_sequences=True, name='rnn_layer')(inputs)
    pred_feat = keras.layers.Dense(91, name='prediction_features')(rnn_output)
    pred = keras.layers.Softmax()(pred_feat)
    model = keras.Model(inputs=[inputs], outputs=[pred, pred_feat])
    path = os.path.join(self.get_temp_dir(), 'model_path.h5')
    model.save(path)

    # Make sure the variable name is unique.
    self.assertNotEqual(rnn_layers[0].kernel.name,
                        rnn_layers[1].kernel.name)
    self.assertIn('rnn_cell1', rnn_layers[1].kernel.name)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_saving_optimizer_weights(self):

    class MyModel(keras.Model):

      def __init__(self):
        super(MyModel, self).__init__()
        self.layer = keras.layers.Dense(1)

      def call(self, x):
        return self.layer(x)

    path = os.path.join(self.get_temp_dir(), 'weights_path')
    x, y = np.ones((10, 10)), np.ones((10, 1))

    model = MyModel()
    model.compile('rmsprop', loss='bce')
    model.train_on_batch(x, y)
    model.reset_metrics()
    model.save_weights(path, save_format='tf')

    batch_loss = model.train_on_batch(x, y)

    new_model = MyModel()
    new_model.compile('rmsprop', loss='bce')
    new_model.train_on_batch(x, y)
    new_model.reset_metrics()

    new_model.load_weights(path)
    new_batch_loss = new_model.train_on_batch(x, y)

    self.assertAllClose(batch_loss, new_batch_loss)

  @combinations.generate(combinations.combine(mode=['eager', 'graph']))
  def test_save_include_optimizer_false(self):

    def get_variables(file_name):
      reader = training_module.load_checkpoint(
          os.path.join(file_name, 'variables/variables'))
      shape_from_key = reader.get_variable_to_shape_map()
      return sorted(shape_from_key.keys())

    path = os.path.join(self.get_temp_dir(), 'no_optimizer')
    x, y = np.ones((10, 10)), np.ones((10, 1))

    with self.cached_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(1))
      model.compile('adam', loss='mse')
      model.train_on_batch(x, y)
      model.save(path, save_format='tf', include_optimizer=False)

    variables = get_variables(path)
    for v in variables:
      self.assertNotIn('optimizer', v)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_saving_model_with_custom_object(self):
    with generic_utils.custom_object_scope(), self.cached_session():

      @generic_utils.register_keras_serializable()
      class CustomLoss(losses.MeanSquaredError):
        pass

      model = sequential.Sequential(
          [core.Dense(units=1, input_shape=(1,))])
      model.compile(optimizer='sgd', loss=CustomLoss())
      model.fit(np.zeros([10, 1]), np.zeros([10, 1]))

      temp_dir = self.get_temp_dir()
      filepath = os.path.join(temp_dir, 'saving')
      model.save(filepath)

      # Make sure the model can be correctly load back.
      _ = save.load_model(filepath, compile=True)


@keras_parameterized.run_with_all_saved_model_formats
class TestWholeModelSaving(keras_parameterized.TestCase):

  def _save_model_dir(self, dirname='saved_model'):
    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
    return os.path.join(temp_dir, dirname)

  def _assert_same_weights_and_metrics(self, model, loaded_model):
    """Checks that the loaded weights and metrics are the same as the original.

    Args:
      model: original model
      loaded_model: loaded model
    """
    self.assertAllClose(model.weights, loaded_model.weights)

    if loaded_model.optimizer:
      if testing_utils.get_save_format() == 'tf':
        # TODO(b/153110928): Keras TF format doesn't restore optimizer weights
        # currently.
        return
      self.assertAllClose(model.optimizer.weights,
                          loaded_model.optimizer.weights)

    # In V1/Graph mode, the model isn't built, so the metrics are not loaded
    # immediately (requires model to be called on some data before building
    # metrics).
    check_metrics = tf2.enabled() and context.executing_eagerly()

    if check_metrics:
      self.assertAllEqual([m.name for m in model.metrics],
                          [m.name for m in loaded_model.metrics])

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_save_and_load(self):
    saved_model_dir = self._save_model_dir()
    save_format = testing_utils.get_save_format()
    save_kwargs = testing_utils.get_save_kwargs()

    if ((save_format == 'h5' or not save_kwargs.get('save_traces', True)) and
        testing_utils.get_model_type() == 'subclass'):
      # HDF5 format currently does not allow saving subclassed models.
      # When saving with `save_traces=False`, the subclassed model must have a
      # get_config/from_config, which the autogenerated model does not have.
      return

    with self.cached_session():
      model = testing_utils.get_model_from_layers(
          [keras.layers.Dense(2),
           keras.layers.RepeatVector(3),
           keras.layers.TimeDistributed(keras.layers.Dense(3))],
          input_shape=(3,))
      model.compile(
          loss=keras.losses.MSE,
          optimizer=keras.optimizer_v2.rmsprop.RMSprop(lr=0.0001),
          metrics=[
              keras.metrics.categorical_accuracy,
              keras.metrics.CategoricalCrossentropy(
                  name='cce', label_smoothing=constant_op.constant(0.2)),
          ],
          weighted_metrics=[
              keras.metrics.categorical_crossentropy,
              keras.metrics.CategoricalCrossentropy(
                  name='cce', label_smoothing=constant_op.constant(0.2)),
          ],
          sample_weight_mode='temporal')

      x = np.random.random((1, 3))
      y = np.random.random((1, 3, 3))
      model.train_on_batch(x, y)

      out = model.predict(x)
      keras.models.save_model(
          model, saved_model_dir, save_format=save_format,
          **save_kwargs)

      loaded_model = keras.models.load_model(saved_model_dir)
      self._assert_same_weights_and_metrics(model, loaded_model)

      out2 = loaded_model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

      eval_out = model.evaluate(x, y)
      eval_out2 = loaded_model.evaluate(x, y)
      self.assertArrayNear(eval_out, eval_out2, 0.001)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_sequential_model_saving_without_input_shape(self):
    saved_model_dir = self._save_model_dir()
    save_format = testing_utils.get_save_format()
    with self.cached_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2))
      model.add(keras.layers.RepeatVector(3))
      model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
      model.compile(
          loss=keras.losses.MSE,
          optimizer='rmsprop',
          metrics=[
              keras.metrics.categorical_accuracy,
              keras.metrics.CategoricalAccuracy(name='cat_acc')
          ],
          weighted_metrics=[
              keras.metrics.categorical_accuracy,
              keras.metrics.CategoricalAccuracy(name='cat_acc2')
          ],
          sample_weight_mode='temporal')
      x = np.random.random((1, 3))
      y = np.random.random((1, 3, 3))
      model.train_on_batch(x, y)

      out = model.predict(x)
      model.save(saved_model_dir, save_format=save_format)

      new_model = keras.models.load_model(saved_model_dir)

      self._assert_same_weights_and_metrics(model, new_model)

      out2 = new_model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_sequential_model_saving_without_compile(self):
    saved_model_dir = self._save_model_dir()
    save_format = testing_utils.get_save_format()
    with self.cached_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.RepeatVector(3))
      model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))

      x = np.random.random((1, 3))
      out = model.predict(x)

      # Save the model without any compilation or training.
      keras.models.save_model(model, saved_model_dir, save_format=save_format)

      new_model = keras.models.load_model(saved_model_dir)
      self._assert_same_weights_and_metrics(model, new_model)

      out2 = new_model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

  def test_sequential_model_saving_2(self):
    saved_model_dir = self._save_model_dir()
    save_format = testing_utils.get_save_format()

    with ops.Graph().as_default(), self.cached_session():
      # test with custom optimizer, loss

      class CustomOp(optimizer_v1.RMSprop):
        pass

      def custom_loss(y_true, y_pred):
        return keras.losses.mse(y_true, y_pred)

      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.Dense(3))
      model.compile(loss=custom_loss, optimizer=CustomOp(), metrics=['acc'])

      x = np.random.random((1, 3))
      y = np.random.random((1, 3))
      model.train_on_batch(x, y)

      out = model.predict(x)
      keras.models.save_model(model, saved_model_dir, save_format=save_format)

      new_model = keras.models.load_model(
          saved_model_dir,
          custom_objects={'CustomOp': CustomOp,
                          'custom_loss': custom_loss})
      self._assert_same_weights_and_metrics(model, new_model)

      out2 = new_model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

  def test_saving_without_compilation(self):
    saved_model_dir = self._save_model_dir()
    save_format = testing_utils.get_save_format()
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2, input_shape=(3,)))
    model.add(keras.layers.Dense(3))
    model.compile(loss='mse', optimizer='sgd', metrics=['acc'])

    keras.models.save_model(model, saved_model_dir, save_format=save_format)
    model = keras.models.load_model(saved_model_dir)

  def test_saving_with_tf_optimizer(self):
    saved_model_dir = self._save_model_dir()
    save_format = testing_utils.get_save_format()

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2, input_shape=(3,)))
    model.add(keras.layers.Dense(3))
    model.compile(loss='mse',
                  optimizer=training_module.AdadeltaOptimizer(0.1),
                  metrics=['acc'])

    keras.models.save_model(model, saved_model_dir, save_format=save_format)
    model = keras.models.load_model(saved_model_dir)

  def test_saving_right_after_compilation(self):
    saved_model_dir = self._save_model_dir()
    save_format = testing_utils.get_save_format()
    with self.cached_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.Dense(3))
      model.compile(loss='mse', optimizer='sgd', metrics=['acc'])
      if not ops.executing_eagerly_outside_functions():
        model._make_train_function()
      keras.models.save_model(model, saved_model_dir, save_format=save_format)
      model = keras.models.load_model(saved_model_dir)

  def test_saving_lambda_numpy_array_arguments(self):
    saved_model_dir = self._save_model_dir()
    save_format = testing_utils.get_save_format()

    if h5py is None:
      self.skipTest('h5py required to run this test')

    mean = np.random.random((4, 2, 3))
    std = np.abs(np.random.random((4, 2, 3))) + 1e-5
    inputs = keras.layers.Input(shape=(4, 2, 3))
    output = keras.layers.Lambda(lambda image, mu, std: (image - mu) / std,
                                 arguments={'mu': mean, 'std': std})(inputs)
    model = keras.models.Model(inputs, output)
    model.compile(loss='mse', optimizer='sgd', metrics=['acc'])

    keras.models.save_model(model, saved_model_dir, save_format=save_format)

    model = keras.models.load_model(saved_model_dir)

    self.assertAllClose(mean, model.layers[1].arguments['mu'])
    self.assertAllClose(std, model.layers[1].arguments['std'])

  def test_saving_model_with_long_layer_names(self):
    saved_model_dir = self._save_model_dir()
    save_format = testing_utils.get_save_format()
    with self.cached_session():
      # This layer name will make the `layers_name` HDF5 attribute blow
      # out of proportion. Note that it fits into the internal HDF5
      # attribute memory limit on its own but because h5py converts
      # the list of layer names into numpy array, which uses the same
      # amount of memory for every item, it increases the memory
      # requirements substantially.
      x = keras.Input(shape=(2,), name='input_' + ('x' * (2**15)))
      f = x
      for i in range(4):
        f = keras.layers.Dense(2, name='dense_%d' % (i,))(f)
      model = keras.Model(inputs=[x], outputs=[f])
      model.compile(
          'adam', loss=keras.losses.MeanSquaredError(), metrics=['acc'])

      x = np.random.random((1, 2))
      y = np.random.random((1, 2))
      model.train_on_batch(x, y)
      out = model.predict(x)

      keras.models.save_model(model, saved_model_dir, save_format=save_format)
      model = keras.models.load_model(saved_model_dir)

      if save_format in ['tf', 'tensorflow']:
        return
      # Check that the HDF5 files contains chunked array
      # of layer names.
      with h5py.File(saved_model_dir, 'r') as h5file:
        num_names_arrays = len([attr for attr in h5file['model_weights'].attrs
                                if attr.startswith('layer_names')])
      # The chunking of layer names array should have happened.
      self.assertGreater(num_names_arrays, 0)
      out2 = model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

  def test_saving_model_with_long_weights_names(self):
    saved_model_dir = self._save_model_dir()
    save_format = testing_utils.get_save_format()

    with self.cached_session():
      x = keras.Input(shape=(2,), name='nested_model_input')
      f = x
      for i in range(4):
        f = keras.layers.Dense(2, name='nested_model_dense_%d' % (i,))(f)
      # This layer name will make the `weights_name`
      # HDF5 attribute blow out of proportion.
      f = keras.layers.Dense(2, name='nested_model_output' + ('x' * (2**14)))(f)
      nested_model = keras.Model(inputs=[x], outputs=[f], name='nested_model')

      x = keras.Input(shape=(2,), name='outer_model_input')
      f = nested_model(x)
      f = keras.layers.Dense(2, name='outer_model_output')(f)

      model = keras.Model(inputs=[x], outputs=[f])
      model.compile(loss='mse', optimizer='adam', metrics=['acc'])

      x = np.random.random((1, 2))
      y = np.random.random((1, 2))
      model.train_on_batch(x, y)
      out = model.predict(x)

      keras.models.save_model(model, saved_model_dir, save_format=save_format)
      model = keras.models.load_model(saved_model_dir)

      if save_format in ['h5', 'hdf5', 'keras']:
        # Check that the HDF5 files contains chunked array
        # of weight names.
        with h5py.File(saved_model_dir, 'r') as h5file:
          num_weight_arrays = len(
              [attr for attr in h5file['model_weights']['nested_model'].attrs
               if attr.startswith('weight_names')])
        # The chunking of layer names array should have happened.
        self.assertGreater(num_weight_arrays, 0)
      out2 = model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

  def test_model_saving_to_pre_created_h5py_file(self):
    saved_model_dir = self._save_model_dir()
    save_format = testing_utils.get_save_format()
    with ops.Graph().as_default(), self.cached_session():
      inputs = keras.Input(shape=(3,))
      x = keras.layers.Dense(2)(inputs)
      outputs = keras.layers.Dense(3)(x)

      model = keras.Model(inputs, outputs)
      model.compile(
          loss=keras.losses.MSE,
          optimizer=optimizer_v1.Adam(),
          metrics=[
              keras.metrics.categorical_accuracy,
              keras.metrics.CategoricalAccuracy()
          ])
      x = np.random.random((1, 3))
      y = np.random.random((1, 3))
      model.train_on_batch(x, y)

      out = model.predict(x)

      keras.models.save_model(model, saved_model_dir, save_format=save_format)
      loaded_model = keras.models.load_model(saved_model_dir)
      out1 = loaded_model.predict(x)
      self.assertAllClose(out, out1, atol=1e-05)
      if save_format in ['tf', 'tensorflow']:
        return

      # Test h5 format specifically
      fd, fname = tempfile.mkstemp('.h5')
      with h5py.File(fname, mode='r+') as h5file:
        keras.models.save_model(model, h5file)
        loaded_model = keras.models.load_model(h5file)
        out2 = loaded_model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

      # Test non-default options in h5
      with h5py.File(
          '_', driver='core', mode='w', backing_store=False) as h5file:
        keras.models.save_model(model, h5file)
        loaded_model = keras.models.load_model(h5file)
        out2 = loaded_model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

      # Cleanup
      os.close(fd)
      os.remove(fname)

  def test_model_saving_to_new_dir_path(self):
    saved_model_dir = os.path.join(self._save_model_dir(), 'newdir',
                                   'saved_model')
    save_format = testing_utils.get_save_format()

    with self.cached_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.RepeatVector(3))
      model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))

      x = np.random.random((1, 3))
      out = model.predict(x)

      keras.models.save_model(model, saved_model_dir, save_format=save_format)

      new_model = keras.models.load_model(saved_model_dir)
      self._assert_same_weights_and_metrics(model, new_model)

      out2 = new_model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

  def test_model_raise_exception_with_failed_saving(self):
    if h5py is None:
      self.skipTest('h5py required to run this test')

    saved_model_dir = self._save_model_dir()
    saved_model_path = os.path.join(saved_model_dir, 'saved_model.h5')

    with self.cached_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.RepeatVector(3))
      model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))

      with self.assertRaisesRegex(OSError, 'Unable to create file'):
        with h5py.File(saved_model_path, 'w'):
          keras.models.save_model(model, saved_model_path)

  def test_saving_constant_initializer_with_numpy(self):
    saved_model_dir = self._save_model_dir()
    save_format = testing_utils.get_save_format()

    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            2,
            input_shape=(3,),
            kernel_initializer=keras.initializers.Constant(np.ones((3, 2)))))
    model.add(keras.layers.Dense(3))
    model.compile(loss='mse', optimizer='sgd', metrics=['acc'])
    keras.models.save_model(model, saved_model_dir, save_format=save_format)
    model = keras.models.load_model(saved_model_dir)

  def test_saving_group_naming_h5py(self):
    # Test saving model with layer which name is prefix to a previous layer
    # name.

    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir)
    h5_path = os.path.join(temp_dir, 'test.h5')

    input_layer = keras.layers.Input((None, None, 3), name='test_input')
    x = keras.layers.Conv2D(1, 1, name='conv1/conv')(input_layer)
    x = keras.layers.Activation('relu', name='conv1')(x)
    model = keras.models.Model(inputs=input_layer, outputs=x)

    model.save_weights(h5_path)
    model.load_weights(h5_path)

  def test_primitive_attrs_contain_no_extraneous_strings(self):
    if h5py is None:
      self.skipTest('h5py required to run this test')

    saved_model_dir = self._save_model_dir()
    save_format = testing_utils.get_save_format()
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, input_shape=[2]))
    model.save(saved_model_dir, save_format=save_format)
    if save_format in ['tf', 'tensorflow']:
      return

    h5file = h5py.File(saved_model_dir, 'r')
    self.assertRegex(h5file.attrs['keras_version'], r'^[\d]+\.[\d]+\.[\S]+$')

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_functional_model_with_custom_loss_and_metric(self):
    def _make_model():
      inputs = keras.Input(shape=(4,))
      x = keras.layers.Dense(8, activation='relu')(inputs)
      outputs = keras.layers.Dense(3, activation='softmax')(x)
      model = keras.Model(inputs=inputs, outputs=outputs)
      custom_loss = keras.layers.Lambda(lambda x: keras.backend.sum(x * x))(x)
      model.add_loss(custom_loss)
      model.add_metric(custom_loss, aggregation='mean', name='custom_loss')
      return model

    saved_model_dir = self._save_model_dir()
    save_format = testing_utils.get_save_format()

    with self.cached_session():
      model = _make_model()
      model.compile(
          loss=keras.losses.SparseCategoricalCrossentropy(),
          optimizer=optimizers.gradient_descent_v2.SGD(),
          metrics=[keras.metrics.SparseCategoricalCrossentropy()])
      x = np.random.normal(size=(32, 4))
      y = np.random.randint(0, 3, size=32)
      model.train_on_batch(x, y)
      evaluation_results = model.evaluate(x, y)
      # Save and reload model.
      model.save(saved_model_dir, save_format=save_format)
      del model  # Prevent misuse.
      loaded_model = keras.models.load_model(saved_model_dir)
      loaded_model_eval_results = loaded_model.evaluate(x, y)
      # Assert all evaluation results are the same.
      self.assertAllClose(evaluation_results, loaded_model_eval_results, 1e-9)
      # Check correctness of the loss calculation.
      self.assertAllGreater(evaluation_results, 0.)
      evaluation_results = dict(
          zip(loaded_model.metrics_names, evaluation_results))
      self.assertNear(
          evaluation_results['sparse_categorical_crossentropy'] +
          evaluation_results['custom_loss'], evaluation_results['loss'], 1e-6)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_save_uncompiled_model_with_optimizer(self):
    with self.cached_session() as session:
      saved_model_dir = self._save_model_dir()
      save_format = testing_utils.get_save_format()
      model = keras.models.Sequential([keras.layers.Dense(1, input_shape=(3,))])
      # Set the model's optimizer but don't compile. This can happen if the
      # model is trained with a custom training loop.
      model.optimizer = keras.optimizer_v2.rmsprop.RMSprop(lr=0.0001)
      if not context.executing_eagerly():
        session.run([v.initializer for v in model.variables])
      model.save(saved_model_dir, save_format=save_format)

      if save_format in ['tf', 'tensorflow']:
        loaded = keras.models.load_model(saved_model_dir)
        self.assertIsInstance(loaded.optimizer,
                              keras.optimizer_v2.optimizer_v2.OptimizerV2)

  @combinations.generate(combinations.combine(mode=['eager']))
  def test_functional_model_with_getitem_op_layer(self):
    inp = keras.Input(shape=(8))

    out = inp[:]
    model = keras.Model(
        inputs=[inp],
        outputs=out)
    batch_size = 7
    x = array_ops.stack([
        math_ops.range(8) for _ in range(batch_size)])
    args = [x]
    expected = x[:]

    self.assertAllEqual(model(args), expected)
    self.assertAllEqual(model.predict(args, batch_size=batch_size), expected)

    # Make sure it can be successfully saved and loaded.
    save_format = testing_utils.get_save_format()
    saved_model_dir = self._save_model_dir()
    keras.models.save_model(model, saved_model_dir, save_format=save_format)

    loaded_model = keras.models.load_model(saved_model_dir)

    self.assertAllEqual(loaded_model(args), expected)
    self.assertAllEqual(loaded_model.predict(args, batch_size=batch_size),
                        expected)

  @combinations.generate(combinations.combine(mode=['eager', 'graph']))
  def test_custom_functional_registered(self):

    def _get_cls_definition():
      class CustomModel(keras.Model):

        def c(self):
          return 'c'

      return CustomModel

    cls = _get_cls_definition()
    self.assertEqual(cls.__bases__[0], keras.Model)

    with self.cached_session() as sess:
      input_ = keras.layers.Input(shape=(1,))
      output = keras.layers.Dense(1)(input_)
      model = cls(input_, output)
      # `cls` now inherits from `Functional` class.
      self.assertEqual(cls.__bases__[0], functional.Functional)

      if not context.executing_eagerly():
        sess.run([v.initializer for v in model.variables])

      save_format = testing_utils.get_save_format()
      saved_model_dir = self._save_model_dir()
      keras.models.save_model(model, saved_model_dir, save_format=save_format)

    loaded_model = keras.models.load_model(
        saved_model_dir, custom_objects={'CustomModel': cls})
    self.assertIsInstance(loaded_model, cls)

    # Check with "new" `CustomModel` class definition.
    new_cls = _get_cls_definition()
    # The new `CustomModel` class is *not* derived from `Functional`.
    self.assertEqual(new_cls.__bases__[0], keras.Model)
    reloaded_model = keras.models.load_model(
        saved_model_dir, custom_objects={'CustomModel': new_cls})
    self.assertIsInstance(reloaded_model, new_cls)

  @combinations.generate(combinations.combine(mode=['eager']))
  def test_shared_objects(self):
    class OuterLayer(keras.layers.Layer):

      def __init__(self, inner_layer):
        super(OuterLayer, self).__init__()
        self.inner_layer = inner_layer

      def call(self, inputs):
        return self.inner_layer(inputs)

      def get_config(self):
        return {
            'inner_layer': generic_utils.serialize_keras_object(
                self.inner_layer)
        }

      @classmethod
      def from_config(cls, config):
        return cls(generic_utils.deserialize_keras_object(
            config['inner_layer']))

    class InnerLayer(keras.layers.Layer):

      def __init__(self):
        super(InnerLayer, self).__init__()
        self.v = self.add_weight(name='v', shape=[], dtype=dtypes.float32)

      def call(self, inputs):
        return self.v + inputs

      @classmethod
      def from_config(cls, config):
        return cls()

    # Create a model with 2 output layers that share the same inner layer.
    inner_layer = InnerLayer()
    outer_layer_1 = OuterLayer(inner_layer)
    outer_layer_2 = OuterLayer(inner_layer)
    input_ = keras.Input(shape=(1,))
    model = keras.Model(
        inputs=input_, outputs=[outer_layer_1(input_), outer_layer_2(input_)])

    # Changes to the shared layer should affect both outputs.
    model.layers[1].inner_layer.v.assign(5)
    self.assertAllEqual(model(1), [6.0, 6.0])
    model.layers[1].inner_layer.v.assign(3)
    self.assertAllEqual(model(1), [4.0, 4.0])

    # After loading, changes to the shared layer should still affect both
    # outputs.
    def _do_assertions(loaded):
      loaded.layers[1].inner_layer.v.assign(5)
      self.assertAllEqual(loaded(1), [6.0, 6.0])
      loaded.layers[1].inner_layer.v.assign(3)
      self.assertAllEqual(loaded(1), [4.0, 4.0])
      loaded.layers[2].inner_layer.v.assign(5)
      self.assertAllEqual(loaded(1), [6.0, 6.0])
      loaded.layers[2].inner_layer.v.assign(3)
      self.assertAllEqual(loaded(1), [4.0, 4.0])

    # We'd like to make sure we only attach shared object IDs when strictly
    # necessary, so we'll recursively traverse the generated config to count
    # whether we have the exact number we expect.
    def _get_all_keys_recursive(dict_or_iterable):
      if isinstance(dict_or_iterable, dict):
        for key in dict_or_iterable.keys():
          yield key
        for key in _get_all_keys_recursive(dict_or_iterable.values()):
          yield key
      elif isinstance(dict_or_iterable, string_types):
        return
      else:
        try:
          for item in dict_or_iterable:
            for key in _get_all_keys_recursive(item):
              yield key
        # Not an iterable or dictionary
        except TypeError:
          return

    with generic_utils.CustomObjectScope({
        'OuterLayer': OuterLayer, 'InnerLayer': InnerLayer}):

      # Test saving and loading to disk
      save_format = testing_utils.get_save_format()
      saved_model_dir = self._save_model_dir()
      keras.models.save_model(model, saved_model_dir, save_format=save_format)
      loaded = keras.models.load_model(saved_model_dir)
      _do_assertions(loaded)

      # Test recreating directly from config
      config = model.get_config()
      key_count = collections.Counter(_get_all_keys_recursive(config))
      self.assertEqual(key_count[generic_utils.SHARED_OBJECT_KEY], 2)
      loaded = keras.Model.from_config(config)
      _do_assertions(loaded)

  @combinations.generate(combinations.combine(mode=['eager']))
  def test_shared_objects_wrapper(self):
    """Tests that shared layers wrapped with `Wrapper` restore correctly."""
    input_ = keras.Input(shape=(1,))
    unwrapped = keras.layers.Layer(name='unwrapped')
    wrapped = keras.layers.Wrapper(unwrapped, name='wrapped')
    model = keras.Model(inputs=input_,
                        outputs=[unwrapped(input_), wrapped(input_)])

    # Test recreating directly from config
    config = model.get_config()
    loaded = keras.Model.from_config(config)
    self.assertIs(loaded.layers[1], loaded.layers[2].layer)

    # Test saving and loading to disk
    save_format = testing_utils.get_save_format()
    saved_model_dir = self._save_model_dir()
    keras.models.save_model(model, saved_model_dir, save_format=save_format)
    loaded = keras.models.load_model(saved_model_dir)
    self.assertIs(loaded.layers[1], loaded.layers[2].layer)

  @combinations.generate(
      combinations.combine(mode=['graph', 'eager'], fit=[True, False]))
  def test_multi_output_metrics_name_stay_same(self, fit):
    """Tests that metric names don't change with each save/load cycle.

    e.g. "head_0_accuracy" should not become "head_0_head_0_accuracy" after
    saving and loading a model.

    Arguments:
      fit: Whether the model should be fit before saving.
    """
    # This doesn't work at all, so we can't check whether metric names are
    # correct.
    if not context.executing_eagerly() and not fit:
      self.skipTest('b/181767784')

    with self.cached_session():
      input_ = keras.Input((4,))
      model = keras.Model(
          input_,
          [keras.layers.Softmax(name='head_0')(keras.layers.Dense(3)(input_)),
           keras.layers.Softmax(name='head_1')(keras.layers.Dense(5)(input_))])
      metric = keras.metrics.BinaryAccuracy()
      model.compile(optimizer='rmsprop',
                    loss='mse',
                    metrics={'head_0': [metric, 'accuracy']})

      x = np.random.rand(2, 4)
      y = {'head_0': np.random.randint(2, size=(2, 3)),
           'head_1': np.random.randint(2, size=(2, 5))}

      # Make sure metrix prefixing works the same regardless of whether the user
      # has fit the model before saving.
      if fit:
        model.fit(x, y, verbose=0)

      # Save and reload.
      save_format = testing_utils.get_save_format()
      saved_model_dir = self._save_model_dir()
      keras.models.save_model(model, saved_model_dir, save_format=save_format)
      loaded = keras.models.load_model(saved_model_dir)

    # Make sure the metrics names from the model before saving match the loaded
    # model.
    self.assertSequenceEqual(model.metrics_names, loaded.metrics_names)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_warning_when_saving_invalid_custom_mask_layer(self):

    class MyMasking(keras.layers.Layer):

      def call(self, inputs):
        return inputs

      def compute_mask(self, inputs, mask=None):
        mask = math_ops.not_equal(inputs, 0)
        return mask

    class MyLayer(keras.layers.Layer):

      def call(self, inputs, mask=None):
        return array_ops.identity(inputs)

    samples = np.random.random((2, 2))
    model = keras.Sequential([MyMasking(), MyLayer()])
    model.predict(samples)
    with warnings.catch_warnings(record=True) as w:
      model.save(self._save_model_dir(), testing_utils.get_save_format())
    self.assertIn(generic_utils.CustomMaskWarning,
                  {warning.category for warning in w})

    # Test that setting up a custom mask correctly does not issue a warning.
    class MyCorrectMasking(keras.layers.Layer):

      def call(self, inputs):
        return inputs

      def compute_mask(self, inputs, mask=None):
        mask = math_ops.not_equal(inputs, 0)
        return mask

      # This get_config doesn't actually do anything because our mask is
      # static and doesn't need any external information to work. We do need a
      # dummy get_config method to prevent the warning from appearing, however.
      def get_config(self, *args, **kwargs):
        return {}

    model = keras.Sequential([MyCorrectMasking(), MyLayer()])
    model.predict(samples)
    with warnings.catch_warnings(record=True) as w:
      model.save(self._save_model_dir(), testing_utils.get_save_format())
    self.assertNotIn(generic_utils.CustomMaskWarning,
                     {warning.category for warning in w})


# Factory functions to create models that will be serialized inside a Network.
def _make_graph_network(input_size, output_size):
  inputs = keras.Input(input_size)
  x = keras.layers.Dense(8, activation='relu')(inputs)
  y = keras.layers.Dense(output_size)(x)
  return keras.Model(inputs=inputs, outputs=y)


def _make_sequential(input_size, output_size):
  del input_size
  return keras.Sequential([
      keras.layers.Dense(8, activation='relu'),
      keras.layers.Dense(output_size),
  ])


def _make_sequential_built(input_size, output_size):
  model = _make_sequential(input_size, output_size)
  model.build((None, input_size))
  return model


def _make_sequential_graph_network(input_size, output_size):
  return keras.Sequential([
      keras.layers.InputLayer(input_size),
      keras.layers.Dense(8, activation='relu'),
      keras.layers.Dense(output_size),
  ])


def _make_sequential_input_shape(input_size, output_size):
  return keras.Sequential([
      keras.layers.Dense(8, activation='relu', input_shape=(input_size,)),
      keras.layers.Dense(output_size),
  ])


class _make_subclassed(keras.Model):  # pylint: disable=invalid-name

  def __init__(self, input_size, output_size):
    super(_make_subclassed, self).__init__()
    self._config = {'input_size': input_size, 'output_size': output_size}
    self._hidden_layer = keras.layers.Dense(8, activation='relu', name='hidden')
    self._logits_layer = keras.layers.Dense(output_size, name='logits')

  def call(self, inputs):
    x = self._hidden_layer(inputs)
    return self._logits_layer(x)

  def get_config(self):
    return self._config

  @classmethod
  def from_config(cls, config):
    return cls(**config)


class _make_subclassed_built(_make_subclassed):  # pylint: disable=invalid-name

  def __init__(self, input_size, output_size):
    super(_make_subclassed_built, self).__init__(input_size, output_size)
    self.build((None, input_size))


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class TestWholeModelSavingWithNesting(test.TestCase, parameterized.TestCase):
  """Tests saving a whole model that contains other models."""

  @parameterized.named_parameters([
      ('graph_network', _make_graph_network),
      ('sequential', _make_sequential),
      ('sequential_built', _make_sequential_built),
      ('sequential_graph_network', _make_sequential_graph_network),
      ('sequential_input_shape', _make_sequential_input_shape),
      ('subclassed', _make_subclassed),
      ('subclassed_built', _make_subclassed_built),
  ])
  def test_functional(self, model_fn):
    """Tests serializing a model that uses a nested model to share weights."""
    if h5py is None:
      self.skipTest('h5py required to run this test')

    def _make_model():
      inputs = (keras.Input(shape=(4,), name='examples'),
                keras.Input(shape=(4,), name='neighbors'))
      base_model = model_fn(inputs[0].shape.as_list()[-1], 2)
      outputs = keras.layers.add([base_model(inputs[0]), base_model(inputs[1])])
      return keras.Model(inputs=inputs, outputs=outputs)

    with self.cached_session():
      x = (np.random.normal(size=(16, 4)).astype(np.float32),
           np.random.normal(size=(16, 4)).astype(np.float32))
      model = _make_model()
      predictions = model(x)
      # Save and reload.
      model_path = os.path.join(self.get_temp_dir(), 'model.h5')
      model.save(model_path)
      del model
      loaded_model = keras.models.load_model(
          model_path,
          custom_objects={
              '_make_subclassed': _make_subclassed,
              '_make_subclassed_built': _make_subclassed_built,
          },
          compile=False)
      self.assertAllClose(loaded_model(x), predictions, 1e-9)


if __name__ == '__main__':
  test.main()
