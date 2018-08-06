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
"""Tests for model saving."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile
from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras.engine import saving
from tensorflow.python.keras.engine import training
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training as training_module

try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None


class TestWeightSavingAndLoading(test.TestCase, parameterized.TestCase):

  def test_weight_loading(self):
    with self.test_session():
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

      temp_dir = self.get_temp_dir()
      self.addCleanup(shutil.rmtree, temp_dir)

      no_extension_path = os.path.join(temp_dir, 'test')
      model.save_weights(no_extension_path, save_format='tf')
      model.load_weights(no_extension_path)
      y = model.predict(x)
      self.assertAllClose(ref_y, y)

      if h5py is None:
        return  # Skip rest of test if H5py isn't available.

      h5_path = os.path.join(temp_dir, 'test.h5')
      model.save_weights(h5_path)
      model.load_weights(h5_path)
      y = model.predict(x)
      self.assertAllClose(ref_y, y)

      model.load_weights(h5_path, by_name=True)
      y = model.predict(x)
      self.assertAllClose(ref_y, y)

      model.save_weights(no_extension_path, save_format='hdf5')
      model.load_weights(no_extension_path)
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
            (keras.layers.GRU(output_dim)),
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
            (keras.layers.LSTM(output_dim)),
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
      _ = keras.engine.saving.preprocess_weights_for_loading(
          layer, weights, original_keras_version='1')

    model = keras.models.Sequential([keras.layers.Dense(2, input_dim=2)])
    _ = keras.engine.saving.preprocess_weights_for_loading(
        model, model.weights, original_keras_version='1')

    x = keras.Input((2,))
    y = keras.layers.Dense(2)(x)
    model = keras.models.Model(x, y)
    _ = keras.engine.saving.preprocess_weights_for_loading(
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
    with self.test_session():
      layer = layer_class(**layer_args)
      layer.build(input_shape=layer_args.get('input_shape'))
      weights1 = layer.get_weights()
      weights2 = keras.engine.saving.preprocess_weights_for_loading(
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

    with self.test_session():
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

  def test_sequential_weight_loading_group_name_with_incorrect_length(self):
    if h5py is None:
      return

    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir)
    h5_path = os.path.join(temp_dir, 'test.h5')

    num_hidden = 5
    input_dim = 3
    num_classes = 2
    with self.test_session():
      ref_model = keras.models.Sequential()
      ref_model.add(keras.layers.Dense(num_hidden, input_dim=input_dim,
                                       name='d1'))
      ref_model.add(keras.layers.Dense(num_classes, name='d2'))
      ref_model.compile(loss=keras.losses.MSE,
                        optimizer=keras.optimizers.RMSprop(lr=0.0001),
                        metrics=[keras.metrics.categorical_accuracy])

      f_ref_model = h5py.File(h5_path, 'w')
      saving.save_weights_to_hdf5_group(f_ref_model, ref_model.layers)

      f_model = h5py.File(h5_path, 'r')
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(num_hidden, use_bias=False,
                                   input_dim=input_dim, name='d1'))
      model.add(keras.layers.Dense(num_classes, name='d2'))
      model.compile(loss=keras.losses.MSE,
                    optimizer=keras.optimizers.RMSprop(lr=0.0001),
                    metrics=[keras.metrics.categorical_accuracy])
    with self.assertRaisesRegexp(ValueError,
                                 r'Layer #0 \(named \"d1\"\) expects 1 '
                                 r'weight\(s\), but the saved weights have 2 '
                                 r'element\(s\)\.'):
      saving.load_weights_from_hdf5_group_by_name(f_model, model.layers)

  def test_sequential_weight_loading_group_name_with_incorrect_shape(self):
    if h5py is None:
      return

    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir)
    h5_path = os.path.join(temp_dir, 'test.h5')

    num_hidden = 5
    input_dim = 3
    num_classes = 2
    with self.test_session():
      ref_model = keras.models.Sequential()
      ref_model.add(keras.layers.Dense(num_hidden, input_dim=input_dim,
                                       name='d1'))
      ref_model.add(keras.layers.Dense(num_classes, name='d2'))
      ref_model.compile(loss=keras.losses.MSE,
                        optimizer=keras.optimizers.RMSprop(lr=0.0001),
                        metrics=[keras.metrics.categorical_accuracy])

      f_ref_model = h5py.File(h5_path, 'w')
      saving.save_weights_to_hdf5_group(f_ref_model, ref_model.layers)

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
        saving.load_weights_from_hdf5_group_by_name(f_model, model.layers)


class TestWholeModelSaving(test.TestCase):

  def test_sequential_model_saving(self):
    if h5py is None:
      self.skipTest('h5py required to run this test')

    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.RepeatVector(3))
      model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
      model.compile(loss=keras.losses.MSE,
                    optimizer=keras.optimizers.RMSprop(lr=0.0001),
                    metrics=[keras.metrics.categorical_accuracy],
                    sample_weight_mode='temporal')
      x = np.random.random((1, 3))
      y = np.random.random((1, 3, 3))
      model.train_on_batch(x, y)

      out = model.predict(x)
      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)

      new_model = keras.models.load_model(fname)
      os.close(fd)
      os.remove(fname)

      out2 = new_model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

      # test that new updates are the same with both models
      x = np.random.random((1, 3))
      y = np.random.random((1, 3, 3))
      model.train_on_batch(x, y)
      new_model.train_on_batch(x, y)
      out = model.predict(x)
      out2 = new_model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

  def test_sequential_model_saving_without_compile(self):
    if h5py is None:
      self.skipTest('h5py required to run this test')

    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.RepeatVector(3))
      model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))

      x = np.random.random((1, 3))
      out = model.predict(x)
      fd, fname = tempfile.mkstemp('.h5')

      # Save the model without any compilation or training.
      keras.models.save_model(model, fname)

      new_model = keras.models.load_model(fname)
      os.close(fd)
      os.remove(fname)

      out2 = new_model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

  def test_sequential_model_saving_2(self):
    if h5py is None:
      self.skipTest('h5py required to run this test')

    with self.test_session():
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
      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)

      model = keras.models.load_model(
          fname,
          custom_objects={'CustomOp': CustomOp,
                          'custom_loss': custom_loss})
      os.close(fd)
      os.remove(fname)

      out2 = model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

  def test_functional_model_saving(self):
    if h5py is None:
      self.skipTest('h5py required to run this test')

    with self.test_session():
      inputs = keras.layers.Input(shape=(3,))
      x = keras.layers.Dense(2)(inputs)
      output = keras.layers.Dense(3)(x)

      model = keras.models.Model(inputs, output)
      model.compile(loss=keras.losses.MSE,
                    optimizer=keras.optimizers.RMSprop(lr=0.0001),
                    metrics=[keras.metrics.categorical_accuracy])
      x = np.random.random((1, 3))
      y = np.random.random((1, 3))
      model.train_on_batch(x, y)

      out = model.predict(x)
      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)

      model = keras.models.load_model(fname)
      os.close(fd)
      os.remove(fname)

      out2 = model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

  def test_saving_without_compilation(self):
    if h5py is None:
      self.skipTest('h5py required to run this test')

    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.Dense(3))
      model.compile(loss='mse', optimizer='sgd', metrics=['acc'])

      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)
      model = keras.models.load_model(fname)
      os.close(fd)
      os.remove(fname)

  def test_saving_with_tf_optimizer(self):
    if h5py is None:
      self.skipTest('h5py required to run this test')

    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.Dense(3))
      model.compile(loss='mse',
                    optimizer=training_module.AdadeltaOptimizer(0.1),
                    metrics=['acc'])

      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)
      model = keras.models.load_model(fname)
      os.close(fd)
      os.remove(fname)

  def test_saving_right_after_compilation(self):
    if h5py is None:
      self.skipTest('h5py required to run this test')

    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.Dense(3))
      model.compile(loss='mse', optimizer='sgd', metrics=['acc'])
      model._make_train_function()

      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)
      model = keras.models.load_model(fname)
      os.close(fd)
      os.remove(fname)

  def test_saving_lambda_numpy_array_arguments(self):
    with self.test_session():
      if h5py is None:
        self.skipTest('h5py required to run this test')

      mean = np.random.random((4, 2, 3))
      std = np.abs(np.random.random((4, 2, 3))) + 1e-5
      inputs = keras.layers.Input(shape=(4, 2, 3))
      output = keras.layers.Lambda(lambda image, mu, std: (image - mu) / std,
                                   arguments={'mu': mean, 'std': std})(inputs)
      model = keras.models.Model(inputs, output)
      model.compile(loss='mse', optimizer='sgd', metrics=['acc'])

      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)

      model = keras.models.load_model(fname)
      os.close(fd)
      os.remove(fname)

      self.assertAllClose(mean, model.layers[1].arguments['mu'])
      self.assertAllClose(std, model.layers[1].arguments['std'])

  def test_saving_model_with_long_layer_names(self):
    if h5py is None:
      self.skipTest('h5py required to run this test')

    with self.test_session():
      # This layer name will make the `layers_name` HDF5 attribute blow
      # out of proportion. Note that it fits into the internal HDF5
      # attribute memory limit on its own but because h5py converts
      # the list of layer names into numpy array, which uses the same
      # amout of memory for every item, it increases the memory
      # requirements substantially.
      x = keras.Input(shape=(2,), name='input_' + ('x' * (2**15)))
      f = x
      for i in range(4):
        f = keras.layers.Dense(2, name='dense_%d' % (i,))(f)
      model = keras.Model(inputs=[x], outputs=[f])
      model.compile(loss='mse', optimizer='adam', metrics=['acc'])

      x = np.random.random((1, 2))
      y = np.random.random((1, 2))
      model.train_on_batch(x, y)
      out = model.predict(x)

      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)
      model = keras.models.load_model(fname)

      # Check that the HDF5 files contains chunked array
      # of layer names.
      with h5py.File(fname, 'r') as h5file:
        num_names_arrays = len([attr for attr in h5file['model_weights'].attrs
                                if attr.startswith('layer_names')])
      # The chunking of layer names array should have happened.
      self.assertGreater(num_names_arrays, 0)
      out2 = model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

      # Cleanup
      os.close(fd)
      os.remove(fname)

  def test_saving_model_with_long_weights_names(self):
    if h5py is None:
      self.skipTest('h5py required to run this test')

    with self.test_session():
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

      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)
      model = keras.models.load_model(fname)

      # Check that the HDF5 files contains chunked array
      # of weight names.
      with h5py.File(fname, 'r') as h5file:
        num_weight_arrays = len(
            [attr for attr in h5file['model_weights']['nested_model'].attrs
             if attr.startswith('weight_names')])
      # The chunking of layer names array should have happened.
      self.assertGreater(num_weight_arrays, 0)
      out2 = model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

      # Cleanup
      os.close(fd)
      os.remove(fname)

  def test_model_saving_to_pre_created_h5py_file(self):
    if h5py is None:
      self.skipTest('h5py required to run this test')

    with self.test_session():
      inputs = keras.Input(shape=(3,))
      x = keras.layers.Dense(2)(inputs)
      outputs = keras.layers.Dense(3)(x)

      model = keras.Model(inputs, outputs)
      model.compile(loss=keras.losses.MSE,
                    optimizer=keras.optimizers.Adam(),
                    metrics=[keras.metrics.categorical_accuracy])
      x = np.random.random((1, 3))
      y = np.random.random((1, 3))
      model.train_on_batch(x, y)

      out = model.predict(x)
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


class SubclassedModel(training.Model):

  def __init__(self):
    super(SubclassedModel, self).__init__()
    self.x_layer = keras.layers.Dense(3)
    self.b_layer = keras.layers.Dense(1)

  def call(self, a):
    return self.b_layer(self.x_layer(a))


class TestWeightSavingAndLoadingTFFormat(test.TestCase):

  def test_keras_optimizer_warning(self):
    graph = ops.Graph()
    with graph.as_default(), self.test_session(graph):
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.Dense(3))
      model.compile(loss='mse', optimizer='adam', metrics=['acc'])
      model._make_train_function()
      temp_dir = self.get_temp_dir()
      prefix = os.path.join(temp_dir, 'ckpt')
      with test.mock.patch.object(logging, 'warning') as mock_log:
        model.save_weights(prefix)
        self.assertRegexpMatches(
            str(mock_log.call_args),
            'Keras optimizer')

  @test_util.run_in_graph_and_eager_modes
  def test_tensorflow_format_overwrite(self):
    with self.test_session() as session:
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
    with context.graph_mode():
      graph = ops.Graph()
      with graph.as_default(), self.test_session(graph) as session:
        model = SubclassedModel()
        temp_dir = self.get_temp_dir()
        prefix = os.path.join(temp_dir, 'ckpt')

        x = constant_op.constant(np.random.random((3, 2)), dtype=dtypes.float32)
        model(x)  # pylint: disable=not-callable
        session.run([v.initializer for v in model.variables])
        model.save_weights(prefix, save_format='tensorflow')
        op_count = len(graph.get_operations())
        model.save_weights(prefix, save_format='tensorflow')
        self.assertEqual(len(graph.get_operations()), op_count)

        model.load_weights(prefix)
        op_count = len(graph.get_operations())
        model.load_weights(prefix)
        self.assertEqual(len(graph.get_operations()), op_count)

  def _weight_loading_test_template(self, make_model_fn):
    with self.test_session():
      model = make_model_fn()
      model.compile(
          loss='mse',
          optimizer=training_module.RMSPropOptimizer(0.1),
          metrics=['acc'])
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
          metrics=['acc'])
      load_model.train_on_batch(train_x, train_y)
      self.assertAllClose(ref_y_after_train, self.evaluate(load_model(x)))

  @test_util.run_in_graph_and_eager_modes
  def test_weight_loading_graph_model(self):
    def _make_graph_model():
      a = keras.layers.Input(shape=(2,))
      x = keras.layers.Dense(3)(a)
      b = keras.layers.Dense(1)(x)
      return keras.models.Model(a, b)

    self._weight_loading_test_template(_make_graph_model)

  @test_util.run_in_graph_and_eager_modes
  def test_weight_loading_subclassed_model(self):
    self._weight_loading_test_template(SubclassedModel)

  def _new_layer_weight_loading_test_template(
      self, first_model_fn, second_model_fn, restore_init_fn):
    with self.test_session() as session:
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
      for v in model.variables:
        self.evaluate(
            v.assign(random_ops.random_normal(shape=array_ops.shape(v))))

      self.addCleanup(shutil.rmtree, temp_dir)

      second_model = second_model_fn()
      second_model.load_weights(prefix)
      second_model(x)
      self.evaluate(restore_init_fn(second_model))
      second_model.save_weights(prefix)
      # Check that the second model's checkpoint loads into the original model
      model.load_weights(prefix)
      y = self.evaluate(model(x))
      self.assertAllClose(ref_y, y)

  @test_util.run_in_graph_and_eager_modes
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
    def _restore_init_fn(restore_model):
      return [v.initializer for v in restore_model.layers[-1].variables]

    self._new_layer_weight_loading_test_template(
        _save_graph_model, _restore_graph_model,
        _restore_init_fn)

  @test_util.run_in_graph_and_eager_modes
  def test_weight_loading_graph_model_added_no_weight_layer(self):
    def _save_graph_model():
      a = keras.layers.Input(shape=(2,))
      x = keras.layers.Dense(3, name='first')(a)
      b = keras.layers.Dense(1, name='second')(x)
      return keras.models.Model(a, b)
    def _restore_graph_model():
      a = keras.layers.Input(shape=(2,))
      x = keras.layers.Dense(3, name='first')(a)
      y = keras.layers.Dropout(rate=0.1)(x)
      b = keras.layers.Dense(1, name='second')(y)
      return keras.models.Model(a, b)
    def _restore_init_fn(restore_model):
      del restore_model  # unused
      return []

    self._new_layer_weight_loading_test_template(
        _save_graph_model, _restore_graph_model,
        _restore_init_fn)

  @test_util.run_in_graph_and_eager_modes
  def test_weight_loading_subclassed_model_added_layer(self):

    class SubclassedModelRestore(training.Model):

      def __init__(self):
        super(SubclassedModelRestore, self).__init__()
        self.x_layer = keras.layers.Dense(3)
        self.y_layer = keras.layers.Dense(3)
        self.b_layer = keras.layers.Dense(1)

      def call(self, a):
        return self.b_layer(self.y_layer(self.x_layer(a)))

    def _restore_init_fn(restore_model):
      return [v.initializer for v in restore_model.y_layer.variables]

    self._new_layer_weight_loading_test_template(
        SubclassedModel, SubclassedModelRestore,
        _restore_init_fn)


if __name__ == '__main__':
  test.main()
