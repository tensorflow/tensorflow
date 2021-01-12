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
import uuid

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.saving import hdf5_format
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import training as training_module
from tensorflow.python.training.tracking import util as trackable

try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class TestWeightSavingAndLoading(test.TestCase, parameterized.TestCase):

  def _save_model_dir(self, dirname='saved_model'):
    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
    return os.path.join(temp_dir, dirname)

  @keras_parameterized.run_with_all_weight_formats
  def test_weight_loading(self):
    saved_model_dir = self._save_model_dir()
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

    h5_path = self._save_model_dir('test.h5')

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

  @keras_parameterized.run_with_all_saved_model_formats(
      exclude_formats=['tf_no_traces'])
  def test_nested_model_weight_loading(self):
    save_format = testing_utils.get_save_format()
    saved_model_dir = self._save_model_dir()

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

    h5_path = self._save_model_dir('test.h5')

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
      with self.assertRaisesRegex(
          ValueError, r'Layer #0 \(named \"d1\"\) expects 1 '
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

    h5_path = self._save_model_dir('test.h5')

    num_hidden = 5
    input_dim = 3
    num_classes = 2
    with ops.Graph().as_default(), self.cached_session():
      ref_model = keras.models.Sequential()
      ref_model.add(keras.layers.Dense(num_hidden, input_dim=input_dim,
                                       name='d1'))
      ref_model.add(keras.layers.Dense(num_classes, name='d2'))
      ref_model.compile(loss=keras.losses.MSE,
                        optimizer=optimizer_v1.RMSprop(lr=0.0001),
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
                    optimizer=optimizer_v1.RMSprop(lr=0.0001),
                    metrics=[keras.metrics.categorical_accuracy])
      with self.assertRaisesRegex(
          ValueError, r'Layer #0 \(named "d1"\), weight '
          r'<tf\.Variable \'d1_1\/kernel:0\' '
          r'shape=\(3, 10\) dtype=float32> has '
          r'shape \(3, 10\), but the saved weight has '
          r'shape \(3, 5\)\.'):
        hdf5_format.load_weights_from_hdf5_group_by_name(f_model, model.layers)

      hdf5_format.load_weights_from_hdf5_group_by_name(
          f_model, model.layers, skip_mismatch=True)
      self.assertAllClose([3.5] * num_classes,
                          keras.backend.get_value(model.layers[1].bias))

  @keras_parameterized.run_with_all_saved_model_formats(
      exclude_formats=['tf_no_traces'])
  @keras_parameterized.run_with_all_model_types
  def test_load_weights_from_saved_model(self):
    save_path = self._save_model_dir()
    save_format = testing_utils.get_save_format()

    if save_format == 'h5' and testing_utils.get_model_type() == 'subclass':
      # TODO(b/173646281): HDF5 format currently does not allow saving
      # subclassed models.
      return

    with self.cached_session():
      model = testing_utils.get_small_mlp(1, 4, input_dim=3)
      data = np.random.random((1, 3))
      labels = np.random.random((1, 4))
      model.compile(loss='mse', optimizer='rmsprop')
      model.fit(data, labels)
      model.save(save_path, save_format=save_format)
      new_model = testing_utils.get_small_mlp(1, 4, input_dim=3)
      if testing_utils.get_model_type() == 'subclass':
        # Call on test data to build the model.
        new_model.predict(data)
      new_model.load_weights(save_path)
      self.assertAllClose(model.weights, new_model.weights)


class SubclassedModel(training.Model):

  def __init__(self):
    super(SubclassedModel, self).__init__()
    self.x_layer = keras.layers.Dense(3)
    self.b_layer = keras.layers.Dense(1)

  def call(self, a):
    return self.b_layer(self.x_layer(a))


class TestWeightSavingAndLoadingTFFormat(test.TestCase, parameterized.TestCase):

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
    with self.assertRaisesRegex(AssertionError, 'Nothing to load'):
      m.load_weights(save_path)
    m.dense = keras.layers.Dense(2)
    m.dense(constant_op.constant([[1.]]))
    with self.assertRaisesRegex(AssertionError,
                                'Nothing except the root object matched'):
      m.load_weights(save_path)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_directory_passed(self):
    with self.cached_session():
      m = DummySubclassModel()
      v = m.add_weight(name='v', shape=[])
      self.evaluate(v.assign(42.))
      prefix = os.path.join(self.get_temp_dir(), str(uuid.uuid4()), 'ckpt/')
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
      self.assertTrue(file_io.file_exists_v2('ackpt.index'))
      self.evaluate(v.assign(1.))
      m.load_weights(prefix)
      self.assertEqual(42., self.evaluate(v))

      prefix = 'subdir/ackpt'
      self.evaluate(v.assign(43.))
      m.save_weights(prefix)
      self.assertTrue(file_io.file_exists_v2('subdir/ackpt.index'))
      self.evaluate(v.assign(2.))
      m.load_weights(prefix)
      self.assertEqual(43., self.evaluate(v))

      prefix = 'ackpt/'
      self.evaluate(v.assign(44.))
      m.save_weights(prefix)
      self.assertTrue(file_io.file_exists_v2('ackpt/.index'))
      self.evaluate(v.assign(3.))
      m.load_weights(prefix)
      self.assertEqual(44., self.evaluate(v))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_nonexistent_prefix_directory(self):
    with self.cached_session():
      m = DummySubclassModel()
      v = m.add_weight(name='v', shape=[])
      self.evaluate(v.assign(42.))
      prefix = os.path.join(self.get_temp_dir(), str(uuid.uuid4()), 'bckpt')
      m.save_weights(prefix)
      self.evaluate(v.assign(2.))
      m.load_weights(prefix)
      self.assertEqual(42., self.evaluate(v))


class DummySubclassModel(training.Model):
  pass


if __name__ == '__main__':
  test.main()
