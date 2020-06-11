# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
#,============================================================================
"""Tests for model saving in the HDF5 format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.saving import hdf5_format
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import training as training_module
from tensorflow.python.training.tracking import util as trackable

try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class TestWeightSavingAndLoading(test.TestCase, parameterized.TestCase):

  @keras_parameterized.run_with_all_saved_model_formats
  def test_weight_loading(self):
    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir)
    saved_model_dir = os.path.join(temp_dir, 'saved_model')
    save_format = testing_utils.get_save_format()
    with self.cached_session():
      a = keras.layers.Input(shape=(2,))
      x = keras.layers.Dense(3)(a)
      b = keras.layers.Dense(1)(x)
      model = keras.models.Model(a, b)

      x = np.random.random((3, 2))
      ref_y = model.predict(x)
      weights = model.get_weights()
      model.set_weights(weights)
      y = model.predict(x)
      self.assertAllClose(ref_y, y)

      with self.assertRaises(ValueError):
        model.set_weights(weights[1:])
      with self.assertRaises(ValueError):
        model.set_weights(weights[::-1])

      model.save_weights(saved_model_dir, save_format=save_format)
      model.load_weights(saved_model_dir)
      y = model.predict(x)
      self.assertAllClose(ref_y, y)

  def test_weight_preprocessing(self):
    input_dim = 3
    output_dim = 3
    size = 2
    cases = [
        [
            (keras.layers.Bidirectional(keras.layers.SimpleRNN(2))),
            [np.random.random((2, 1)), np.random.random((2, 1))],
            (None, 3, 2),
        ],
        [
            (keras.layers.TimeDistributed(keras.layers.Dense(1))),
            [np.random.random((2, 1)), np.random.random((1,))],
            (None, 3, 2),
        ],
        [
            (keras.layers.Conv1D(output_dim, size, use_bias=False)),
            [np.random.random((output_dim, input_dim, size, 1))],
            (None, 4, input_dim),
        ],
        [
            (keras.layers.Conv2D(output_dim, size,
                                 use_bias=False, data_format='channels_first')),
            [np.random.random((output_dim, input_dim, size, size))],
            (None, input_dim, 4, 4),
        ],
        [
            (keras.layers.Conv2DTranspose(output_dim, size,
                                          use_bias=False,
                                          data_format='channels_first')),
            [np.random.random((output_dim, input_dim, size, size))],
            (None, input_dim, 4, 4),
        ],
        [
            (keras.layers.Conv2DTranspose(output_dim, size,
                                          use_bias=False,
                                          data_format='channels_last')),
            [np.random.random((size, size, input_dim, output_dim))],
            (None, 4, 4, input_dim),
        ],
        [
            (keras.layers.Conv3D(output_dim, size,
                                 use_bias=False, data_format='channels_first')),
            [np.random.random((output_dim, input_dim, size, size, size))],
            (None, input_dim, 4, 4, 4),
        ],
        [
            (keras.layers.GRUV1(output_dim)),
            [np.random.random((input_dim, output_dim)),
             np.random.random((output_dim, output_dim)),
             np.random.random((output_dim,)),
             np.random.random((input_dim, output_dim)),
             np.random.random((output_dim, output_dim)),
             np.random.random((output_dim,)),
             np.random.random((input_dim, output_dim)),
             np.random.random((output_dim, output_dim)),
             np.random.random((output_dim,))],
            (None, 4, input_dim),
        ],
        [
            (keras.layers.LSTMV1(output_dim)),
            [np.random.random((input_dim, output_dim)),
             np.random.random((output_dim, output_dim)),
             np.random.random((output_dim,)),
             np.random.random((input_dim, output_dim)),
             np.random.random((output_dim, output_dim)),
             np.random.random((output_dim,)),
             np.random.random((input_dim, output_dim)),
             np.random.random((output_dim, output_dim)),
             np.random.random((output_dim,)),
             np.random.random((input_dim, output_dim)),
             np.random.random((output_dim, output_dim)),
             np.random.random((output_dim,))],
            (None, 4, input_dim),
        ],
    ]
    for layer, weights, input_shape in cases:
      layer.build(input_shape)
      _ = hdf5_format.preprocess_weights_for_loading(
          layer, weights, original_keras_version='1')

    model = keras.models.Sequential([keras.layers.Dense(2, input_dim=2)])
    _ = hdf5_format.preprocess_weights_for_loading(
        model, model.weights, original_keras_version='1')

    x = keras.Input((2,))
    y = keras.layers.Dense(2)(x)
    model = keras.models.Model(x, y)
    _ = hdf5_format.preprocess_weights_for_loading(
        model, model.weights, original_keras_version='1')

  @parameterized.named_parameters(
      ('gru', keras.layers.GRU, {
          'units': 2,
          'input_shape': (3, 5)
      }),
      ('gru_with_reset_after', keras.layers.GRU, {
          'units': 2,
          'input_shape': (3, 5),
          'reset_after': True
      }),
      ('lstm', keras.layers.LSTM, {
          'units': 2,
          'input_shape': (3, 5)
      }),
      ('cudnngru', keras.layers.CuDNNGRU, {
          'units': 2,
          'input_shape': (3, 5)
      }),
      ('cudnnlstm', keras.layers.CuDNNLSTM, {
          'units': 2,
          'input_shape': (3, 5)
      }))
  def test_preprocess_weights_for_loading_rnn_should_be_idempotent(
      self, layer_class, layer_args):
    with self.cached_session():
      layer = layer_class(**layer_args)
      layer.build(input_shape=layer_args.get('input_shape'))
      weights1 = layer.get_weights()
      weights2 = hdf5_format.preprocess_weights_for_loading(
          layer, weights1)
      _ = [
          self.assertAllClose(x, y, rtol=1e-05)
          for (x, y) in zip(weights1, weights2)
      ]

  def test_sequential_weight_loading(self):
    if h5py is None:
      return

    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir)
    h5_path = os.path.join(temp_dir, 'test.h5')

    num_hidden = 5
    input_dim = 3
    batch_size = 5
    num_classes = 2

    with self.cached_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(num_hidden, input_dim=input_dim))
      model.add(keras.layers.Dense(num_classes))

      x = np.random.random((batch_size, input_dim))
      ref_y = model.predict(x)

      model.save_weights(h5_path)

      model = keras.models.Sequential()
      model.add(keras.layers.Dense(num_hidden, input_dim=input_dim))
      model.add(keras.layers.Dense(num_classes))
      model.load_weights(h5_path)
      y = model.predict(x)

      self.assertAllClose(y, ref_y)

  @keras_parameterized.run_with_all_saved_model_formats
  def test_nested_model_weight_loading(self):
    save_format = testing_utils.get_save_format()
    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir)
    saved_model_dir = os.path.join(temp_dir, 'saved_model')

    batch_size = 5
    shape = (None, None, 3)

    with self.cached_session():
      def gen_model():

        def seq_model():
          model = keras.models.Sequential([
              keras.layers.Conv2D(3, 1, input_shape=shape),
              keras.layers.BatchNormalization()])
          return model

        x = inner_inputs = keras.layers.Input((None, None, 3))
        x = seq_model()(x)
        x = seq_model()(x)
        inner_model = keras.models.Model(inner_inputs, x)

        inputs = keras.layers.Input(shape)
        return keras.models.Model(inputs, inner_model(inputs))

      model = gen_model()
      x = np.random.random((batch_size, 1, 1, 3))
      ref_y = model.predict(x)

      model.save_weights(saved_model_dir, save_format=save_format)

      model = gen_model()
      model.load_weights(saved_model_dir)
      y = model.predict(x)

      self.assertAllClose(y, ref_y)

  def test_sequential_weight_loading_group_name_with_incorrect_length(self):
    if h5py is None:
      return

    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir)
    h5_path = os.path.join(temp_dir, 'test.h5')

    num_hidden = 5
    input_dim = 3
    num_classes = 2
    with self.cached_session():
      ref_model = keras.models.Sequential()
      ref_model.add(keras.layers.Dense(num_hidden, input_dim=input_dim,
                                       name='d1'))
      ref_model.add(keras.layers.Dense(num_classes, name='d2'))
      ref_model.compile(loss=keras.losses.MSE,
                        optimizer='rmsprop',
                        metrics=[keras.metrics.categorical_accuracy])

      f_ref_model = h5py.File(h5_path, 'w')
      hdf5_format.save_weights_to_hdf5_group(f_ref_model, ref_model.layers)

      f_model = h5py.File(h5_path, 'r')
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(num_hidden, use_bias=False,
                                   input_dim=input_dim, name='d1'))
      model.add(keras.layers.Dense(num_classes, name='d2'))
      model.compile(loss=keras.losses.MSE,
                    optimizer='rmsprop',
                    metrics=[keras.metrics.categorical_accuracy])
      with self.assertRaisesRegexp(ValueError,
                                   r'Layer #0 \(named \"d1\"\) expects 1 '
                                   r'weight\(s\), but the saved weights have 2 '
                                   r'element\(s\)\.'):
        hdf5_format.load_weights_from_hdf5_group_by_name(f_model, model.layers)

      hdf5_format.load_weights_from_hdf5_group_by_name(
          f_model, model.layers, skip_mismatch=True)
      self.assertAllClose(keras.backend.get_value(ref_model.layers[1].kernel),
                          keras.backend.get_value(model.layers[1].kernel))

  def test_sequential_weight_loading_group_name_with_incorrect_shape(self):
    if h5py is None:
      return

    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir)
    h5_path = os.path.join(temp_dir, 'test.h5')

    num_hidden = 5
    input_dim = 3
    num_classes = 2
    with ops.Graph().as_default(), self.cached_session():
      ref_model = keras.models.Sequential()
      ref_model.add(keras.layers.Dense(num_hidden, input_dim=input_dim,
                                       name='d1'))
      ref_model.add(keras.layers.Dense(num_classes, name='d2'))
      ref_model.compile(loss=keras.losses.MSE,
                        optimizer=keras.optimizers.RMSprop(lr=0.0001),
                        metrics=[keras.metrics.categorical_accuracy])

      f_ref_model = h5py.File(h5_path, 'w')
      keras.backend.set_value(ref_model.layers[1].bias, [3.5] * num_classes)
      hdf5_format.save_weights_to_hdf5_group(f_ref_model, ref_model.layers)

      f_model = h5py.File(h5_path, 'r')
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(num_hidden + 5, input_dim=input_dim,
                                   name='d1'))
      model.add(keras.layers.Dense(num_classes, name='d2'))
      model.compile(loss=keras.losses.MSE,
                    optimizer=keras.optimizers.RMSprop(lr=0.0001),
                    metrics=[keras.metrics.categorical_accuracy])
      with self.assertRaisesRegexp(ValueError,
                                   r'Layer #0 \(named "d1"\), weight '
                                   r'<tf\.Variable \'d1_1\/kernel:0\' '
                                   r'shape=\(3, 10\) dtype=float32> has '
                                   r'shape \(3, 10\), but the saved weight has '
                                   r'shape \(3, 5\)\.'):
        hdf5_format.load_weights_from_hdf5_group_by_name(f_model, model.layers)

      hdf5_format.load_weights_from_hdf5_group_by_name(
          f_model, model.layers, skip_mismatch=True)
      self.assertAllClose([3.5] * num_classes,
                          keras.backend.get_value(model.layers[1].bias))


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
    check_metrics = tf2.enabled and context.executing_eagerly()

    if check_metrics:
      self.assertAllEqual([m.name for m in model.metrics],
                          [m.name for m in loaded_model.metrics])

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_save_and_load(self):
    saved_model_dir = self._save_model_dir()
    save_format = testing_utils.get_save_format()

    if save_format == 'h5' and testing_utils.get_model_type() == 'subclass':
      return  # HDF5 format currently does not allow saving classed models.

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
      keras.models.save_model(model, saved_model_dir, save_format=save_format)

      loaded_model = keras.models.load_model(saved_model_dir)
      self._assert_same_weights_and_metrics(model, loaded_model)

      out2 = loaded_model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

      eval_out = model.evaluate(x, y)
      eval_out2 = loaded_model.evaluate(x, y)
      self.assertArrayNear(eval_out, eval_out2, 0.001)

  @test_util.run_in_graph_and_eager_modes
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

  @test_util.run_in_graph_and_eager_modes
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

      class CustomOp(keras.optimizers.RMSprop):
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
          optimizer=keras.optimizers.Adam(),
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
      with h5py.File('_', driver='core',
                     backing_store=False) as h5file:
        keras.models.save_model(model, h5file)
        loaded_model = keras.models.load_model(h5file)
        out2 = loaded_model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

      # Cleanup
      os.close(fd)
      os.remove(fname)

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
    self.assertRegexpMatches(
        h5file.attrs['keras_version'], r'^[\d]+\.[\d]+\.[\S]+$')

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


class SubclassedModel(training.Model):

  def __init__(self):
    super(SubclassedModel, self).__init__()
    self.x_layer = keras.layers.Dense(3)
    self.b_layer = keras.layers.Dense(1)

  def call(self, a):
    return self.b_layer(self.x_layer(a))


class TestWeightSavingAndLoadingTFFormat(test.TestCase, parameterized.TestCase):

  def test_keras_optimizer_warning(self):
    graph = ops.Graph()
    with graph.as_default(), self.session(graph):
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.Dense(3))
      model.compile(loss='mse', optimizer=optimizers.Adam(), metrics=['acc'])
      if not ops.executing_eagerly_outside_functions():
        model._make_train_function()
      temp_dir = self.get_temp_dir()
      prefix = os.path.join(temp_dir, 'ckpt')
      with test.mock.patch.object(logging, 'warning') as mock_log:
        model.save_weights(prefix)
        self.assertRegexpMatches(
            str(mock_log.call_args),
            'Keras optimizer')

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_tensorflow_format_overwrite(self):
    with self.cached_session() as session:
      model = SubclassedModel()
      temp_dir = self.get_temp_dir()
      prefix = os.path.join(temp_dir, 'ckpt')

      x = constant_op.constant(np.random.random((3, 2)), dtype=dtypes.float32)
      executing_eagerly = context.executing_eagerly()
      model(x)  # pylint: disable=not-callable
      if not executing_eagerly:
        session.run([v.initializer for v in model.variables])
      model.save_weights(prefix, save_format='tensorflow')
      model.save_weights(prefix, save_format='tensorflow', overwrite=True)
      with self.assertRaises(EOFError):
        # Indirectly tests that the user is prompted
        model.save_weights(prefix, save_format='tensorflow', overwrite=False)

  def test_no_default_session(self):
    with ops.Graph().as_default():
      self.assertFalse(ops.get_default_session())
      data = np.random.random((1000, 32)).astype(np.float32)
      labels = np.random.random((1000, 10)).astype(np.float32)

      model = keras.models.Sequential([
          keras.layers.Dense(10, activation='softmax'),
          keras.layers.Dense(10, activation='softmax')])

      model.compile(optimizer=training_module.RMSPropOptimizer(0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

      model.fit(data, labels)
      fname = os.path.join(self.get_temp_dir(), 'weights', 'ckpt')
      model.save_weights(fname)
      model.load_weights(fname)

  def test_no_graph_pollution(self):
    with ops.get_default_graph().as_default():
      graph = ops.Graph()
      with graph.as_default(), self.session(graph) as session:
        model = SubclassedModel()
        temp_dir = self.get_temp_dir()
        prefix = os.path.join(temp_dir, 'ckpt')

        x = constant_op.constant(np.random.random((3, 2)), dtype=dtypes.float32)
        model(x)  # pylint: disable=not-callable
        session.run([v.initializer for v in model.variables])
        model.save_weights(prefix, save_format='tensorflow')
        op_count = len(graph.get_operations())
        model.save_weights(prefix, save_format='tensorflow')
        self.assertLen(graph.get_operations(), op_count)

        model.load_weights(prefix)
        op_count = len(graph.get_operations())
        model.load_weights(prefix)
        self.assertLen(graph.get_operations(), op_count)

  def _weight_loading_test_template(self, make_model_fn):
    with self.cached_session():
      model = make_model_fn()
      model.compile(
          loss='mse',
          optimizer=training_module.RMSPropOptimizer(0.1),
          metrics=['acc', keras.metrics.CategoricalAccuracy()])
      temp_dir = self.get_temp_dir()
      prefix = os.path.join(temp_dir, 'ckpt')
      train_x = np.random.random((3, 2))
      train_y = np.random.random((3,))
      x = constant_op.constant(train_x, dtype=dtypes.float32)

      model.train_on_batch(train_x, train_y)
      model.save_weights(prefix, save_format='tf')
      ref_y_before_train = model.predict(train_x)
      model.train_on_batch(train_x, train_y)
      ref_y_after_train = model.predict(train_x)
      for v in model.variables:
        self.evaluate(
            v.assign(random_ops.random_normal(shape=array_ops.shape(v))))

      self.addCleanup(shutil.rmtree, temp_dir)

      model.load_weights(prefix)
      self.assertAllClose(ref_y_before_train, self.evaluate(model(x)))

      # Test restore-on-create if this is a subclassed Model (graph Networks
      # will have already created their variables).
      load_model = make_model_fn()
      load_model.load_weights(prefix)
      self.assertAllClose(
          ref_y_before_train,
          self.evaluate(load_model(x)))
      load_model = make_model_fn()
      load_model.load_weights(prefix)
      # We need to run some of the restore ops for predict(), but not all
      # variables have been created yet (optimizer slot variables). Tests
      # incremental restore.
      load_model.predict(train_x)
      load_model.compile(
          loss='mse',
          optimizer=training_module.RMSPropOptimizer(0.1),
          metrics=['acc', keras.metrics.CategoricalAccuracy()])
      load_model.train_on_batch(train_x, train_y)
      self.assertAllClose(ref_y_after_train, self.evaluate(load_model(x)))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_weight_loading_graph_model(self):
    def _make_graph_model():
      a = keras.layers.Input(shape=(2,))
      x = keras.layers.Dense(3)(a)
      b = keras.layers.Dense(1)(x)
      return keras.models.Model(a, b)

    self._weight_loading_test_template(_make_graph_model)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_weight_loading_subclassed_model(self):
    self._weight_loading_test_template(SubclassedModel)

  def _new_layer_weight_loading_test_template(
      self, first_model_fn, second_model_fn):
    with self.cached_session() as session:
      model = first_model_fn()
      temp_dir = self.get_temp_dir()
      prefix = os.path.join(temp_dir, 'ckpt')

      x = constant_op.constant(np.random.random((3, 2)), dtype=dtypes.float32)
      executing_eagerly = context.executing_eagerly()
      ref_y_tensor = model(x)
      if not executing_eagerly:
        session.run([v.initializer for v in model.variables])
      ref_y = self.evaluate(ref_y_tensor)
      model.save_weights(prefix)
      self.assertEqual(
          prefix,
          checkpoint_management.latest_checkpoint(temp_dir))
      for v in model.variables:
        self.evaluate(
            v.assign(random_ops.random_normal(shape=array_ops.shape(v))))

      self.addCleanup(shutil.rmtree, temp_dir)

      second_model = second_model_fn()
      status = second_model.load_weights(prefix)
      second_model(x)
      status.run_restore_ops()
      second_model.save_weights(prefix)
      # Check that the second model's checkpoint loads into the original model
      status = model.load_weights(prefix)
      status.run_restore_ops(session)
      y = self.evaluate(model(x))
      self.assertAllClose(ref_y, y)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_weight_loading_graph_model_added_layer(self):
    def _save_graph_model():
      a = keras.layers.Input(shape=(2,))
      x = keras.layers.Dense(3, name='first')(a)
      b = keras.layers.Dense(1, name='second')(x)
      return keras.models.Model(a, b)
    def _restore_graph_model():
      a = keras.layers.Input(shape=(2,))
      x = keras.layers.Dense(3, name='first')(a)
      y = keras.layers.Dense(1, name='second')(x)
      b = keras.layers.Dense(3, name='secondjr')(y)
      return keras.models.Model(a, b)

    self._new_layer_weight_loading_test_template(
        _save_graph_model, _restore_graph_model)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_weight_loading_graph_model_added_no_weight_layer(self):
    def _save_graph_model():
      a = keras.layers.Input(shape=(2,))
      x = keras.layers.Dense(3, name='first')(a)
      b = keras.layers.Dense(1, name='second')(x)
      return keras.models.Model(a, b)
    def _restore_graph_model():
      a = keras.layers.Input(shape=(2,))
      x = keras.layers.Dense(3, name='first')(a)
      b = keras.layers.Dense(1, name='second')(x)
      y = keras.layers.Dropout(rate=0.1)(b)
      return keras.models.Model(a, y)

    self._new_layer_weight_loading_test_template(
        _save_graph_model, _restore_graph_model)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_weight_loading_subclassed_model_added_layer(self):

    class SubclassedModelRestore(training.Model):

      def __init__(self):
        super(SubclassedModelRestore, self).__init__()
        self.x_layer = keras.layers.Dense(3)
        self.y_layer = keras.layers.Dense(3)
        self.b_layer = keras.layers.Dense(1)

      def call(self, a):
        return self.b_layer(self.y_layer(self.x_layer(a)))

    self._new_layer_weight_loading_test_template(
        SubclassedModel, SubclassedModelRestore)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_incompatible_checkpoint(self):
    save_path = trackable.Checkpoint().save(
        os.path.join(self.get_temp_dir(), 'ckpt'))
    m = DummySubclassModel()
    with self.assertRaisesRegexp(AssertionError, 'Nothing to load'):
      m.load_weights(save_path)
    m.dense = keras.layers.Dense(2)
    m.dense(constant_op.constant([[1.]]))
    with self.assertRaisesRegexp(
        AssertionError, 'Nothing except the root object matched'):
      m.load_weights(save_path)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_directory_passed(self):
    with self.cached_session():
      m = DummySubclassModel()
      v = m.add_weight(name='v', shape=[])
      self.evaluate(v.assign(42.))
      prefix = os.path.join(self.get_temp_dir(),
                            '{}'.format(ops.uid()), 'ckpt/')
      m.save_weights(prefix)
      self.evaluate(v.assign(2.))
      m.load_weights(prefix)
      self.assertEqual(42., self.evaluate(v))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_relative_path(self):
    with self.cached_session():
      m = DummySubclassModel()
      v = m.add_weight(name='v', shape=[])
      os.chdir(self.get_temp_dir())

      prefix = 'ackpt'
      self.evaluate(v.assign(42.))
      m.save_weights(prefix)
      self.assertTrue(file_io.file_exists('ackpt.index'))
      self.evaluate(v.assign(1.))
      m.load_weights(prefix)
      self.assertEqual(42., self.evaluate(v))

      prefix = 'subdir/ackpt'
      self.evaluate(v.assign(43.))
      m.save_weights(prefix)
      self.assertTrue(file_io.file_exists('subdir/ackpt.index'))
      self.evaluate(v.assign(2.))
      m.load_weights(prefix)
      self.assertEqual(43., self.evaluate(v))

      prefix = 'ackpt/'
      self.evaluate(v.assign(44.))
      m.save_weights(prefix)
      self.assertTrue(file_io.file_exists('ackpt/.index'))
      self.evaluate(v.assign(3.))
      m.load_weights(prefix)
      self.assertEqual(44., self.evaluate(v))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_nonexistent_prefix_directory(self):
    with self.cached_session():
      m = DummySubclassModel()
      v = m.add_weight(name='v', shape=[])
      self.evaluate(v.assign(42.))
      prefix = os.path.join(self.get_temp_dir(),
                            '{}'.format(ops.uid()), 'bckpt')
      m.save_weights(prefix)
      self.evaluate(v.assign(2.))
      m.load_weights(prefix)
      self.assertEqual(42., self.evaluate(v))


class DummySubclassModel(training.Model):
  pass


if __name__ == '__main__':
  test.main()
