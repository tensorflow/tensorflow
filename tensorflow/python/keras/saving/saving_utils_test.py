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
# ==============================================================================
"""Tests for saving utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np


from tensorflow.python import keras
from tensorflow.python import tf2
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save as save_lib
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import rmsprop


class TraceModelCallTest(keras_parameterized.TestCase):

  def _assert_all_close(self, expected, actual):
    if not context.executing_eagerly():
      with self.cached_session() as sess:
        K._initialize_variables(sess)
        self.assertAllClose(expected, actual)
    else:
      self.assertAllClose(expected, actual)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_trace_model_outputs(self):
    input_dim = 5 if testing_utils.get_model_type() == 'functional' else None
    model = testing_utils.get_small_mlp(10, 3, input_dim)
    inputs = array_ops.ones((8, 5))

    if input_dim is None:
      with self.assertRaisesRegexp(ValueError,
                                   'input shapes have not been set'):
        saving_utils.trace_model_call(model)
      model._set_inputs(inputs)

    fn = saving_utils.trace_model_call(model)
    signature_outputs = fn(inputs)
    if model.output_names:
      expected_outputs = {model.output_names[0]: model(inputs)}
    else:
      expected_outputs = {'output_1': model(inputs)}

    self._assert_all_close(expected_outputs, signature_outputs)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_trace_model_outputs_after_fitting(self):
    input_dim = 5 if testing_utils.get_model_type() == 'functional' else None
    model = testing_utils.get_small_mlp(10, 3, input_dim)
    model.compile(
        optimizer='sgd',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly())
    model.fit(
        x=np.random.random((8, 5)).astype(np.float32),
        y=np.random.random((8, 3)).astype(np.float32),
        epochs=2)

    inputs = array_ops.ones((8, 5))

    fn = saving_utils.trace_model_call(model)
    signature_outputs = fn(inputs)
    if model.output_names:
      expected_outputs = {model.output_names[0]: model(inputs)}
    else:
      expected_outputs = {'output_1': model(inputs)}

    self._assert_all_close(expected_outputs, signature_outputs)

  @keras_parameterized.run_with_all_model_types(exclude_models='sequential')
  @keras_parameterized.run_all_keras_modes
  def test_trace_multi_io_model_outputs(self):
    input_dim = 5
    num_classes = 3
    num_classes_b = 4
    input_a = keras.layers.Input(shape=(input_dim,), name='input_a')
    input_b = keras.layers.Input(shape=(input_dim,), name='input_b')

    dense = keras.layers.Dense(num_classes, name='dense')
    dense2 = keras.layers.Dense(num_classes_b, name='dense2')
    dropout = keras.layers.Dropout(0.5, name='dropout')
    branch_a = [input_a, dense]
    branch_b = [input_b, dense, dense2, dropout]

    model = testing_utils.get_multi_io_model(branch_a, branch_b)

    input_a_np = np.random.random((10, input_dim)).astype(np.float32)
    input_b_np = np.random.random((10, input_dim)).astype(np.float32)

    if testing_utils.get_model_type() == 'subclass':
      with self.assertRaisesRegexp(ValueError,
                                   'input shapes have not been set'):
        saving_utils.trace_model_call(model)

    model.compile(
        optimizer='sgd',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly())
    model.fit(x=[np.random.random((8, input_dim)).astype(np.float32),
                 np.random.random((8, input_dim)).astype(np.float32)],
              y=[np.random.random((8, num_classes)).astype(np.float32),
                 np.random.random((8, num_classes_b)).astype(np.float32)],
              epochs=2)

    fn = saving_utils.trace_model_call(model)
    signature_outputs = fn([input_a_np, input_b_np])
    outputs = model([input_a_np, input_b_np])
    if model.output_names:
      expected_outputs = {
          model.output_names[0]: outputs[0],
          model.output_names[1]: outputs[1]
      }
    else:
      expected_outputs = {'output_1': outputs[0], 'output_2': outputs[1]}
    self._assert_all_close(expected_outputs, signature_outputs)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_trace_features_layer(self):
    columns = [feature_column_lib.numeric_column('x')]
    model = sequential.Sequential([feature_column_lib.DenseFeatures(columns)])
    model_input = {'x': constant_op.constant([[1.]])}
    model.predict(model_input, steps=1)
    fn = saving_utils.trace_model_call(model)
    self.assertAllClose({'output_1': [[1.]]}, fn({'x': [[1.]]}))

    columns = [
        feature_column_lib.numeric_column('x'),
        feature_column_lib.numeric_column('y')
    ]
    model = sequential.Sequential([feature_column_lib.DenseFeatures(columns)])
    model_input = {'x': constant_op.constant([[1.]]),
                   'y': constant_op.constant([[2.]])}
    model.predict(model_input, steps=1)
    fn = saving_utils.trace_model_call(model)
    self.assertAllClose({'output_1': [[1., 2.]]},
                        fn({'x': [[1.]], 'y': [[2.]]}))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_specify_input_signature(self):
    model = testing_utils.get_small_sequential_mlp(10, 3, None)
    inputs = array_ops.ones((8, 5))

    with self.assertRaisesRegexp(ValueError, 'input shapes have not been set'):
      saving_utils.trace_model_call(model)

    fn = saving_utils.trace_model_call(
        model, [tensor_spec.TensorSpec(shape=[None, 5], dtype=dtypes.float32)])
    signature_outputs = fn(inputs)
    if model.output_names:
      expected_outputs = {model.output_names[0]: model(inputs)}
    else:
      expected_outputs = {'output_1': model(inputs)}
    self._assert_all_close(expected_outputs, signature_outputs)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_subclassed_model_with_input_signature(self):

    class Model(keras.Model):

      def __init__(self):
        super(Model, self).__init__()
        self.dense = keras.layers.Dense(3, name='dense')

      @def_function.function(
          input_signature=[[tensor_spec.TensorSpec([None, 5], dtypes.float32),
                            tensor_spec.TensorSpec([None], dtypes.float32)]],)
      def call(self, inputs, *args):
        x, y = inputs
        return self.dense(x) + y

    model = Model()
    fn = saving_utils.trace_model_call(model)
    x = array_ops.ones((8, 5), dtype=dtypes.float32)
    y = array_ops.ones((3,), dtype=dtypes.float32)
    expected_outputs = {'output_1': model([x, y])}
    signature_outputs = fn([x, y])
    self._assert_all_close(expected_outputs, signature_outputs)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_model_with_fixed_input_dim(self):
    """Ensure that the batch_dim is removed when saving.

    When serving or retraining, it is important to reset the batch dim.
    This can be an issue inside of tf.function. See b/132783590 for context.
    """
    model = testing_utils.get_small_mlp(10, 3, 5)

    loss_object = keras.losses.MeanSquaredError()
    optimizer = gradient_descent.SGD()

    @def_function.function
    def train_step(data, labels):
      with backprop.GradientTape() as tape:
        predictions = model(data)
        loss = loss_object(labels, predictions)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    x = np.random.random((8, 5))
    y = np.random.random((8, 3))

    train_step(x, y)

    fn = saving_utils.trace_model_call(model)
    self.assertEqual(fn.input_signature[0].shape.as_list(),
                     tensor_shape.TensorShape([None, 5]).as_list())


def _import_and_infer(save_dir, inputs):
  """Import a SavedModel into a TF 1.x-style graph and run `signature_key`."""
  graph = ops.Graph()
  with graph.as_default(), session_lib.Session() as session:
    model = loader.load(session, [tag_constants.SERVING], save_dir)
    signature = model.signature_def[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    assert set(inputs.keys()) == set(
        signature.inputs.keys()), ('expected {}, found {}'.format(
            signature.inputs.keys(), inputs.keys()))
    feed_dict = {}
    for arg_name in inputs.keys():
      feed_dict[graph.get_tensor_by_name(signature.inputs[arg_name].name)] = (
          inputs[arg_name])
    output_dict = {}
    for output_name, output_tensor_info in signature.outputs.items():
      output_dict[output_name] = graph.get_tensor_by_name(
          output_tensor_info.name)
    return session.run(output_dict, feed_dict=feed_dict)


@keras_parameterized.run_with_all_model_types
@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class ModelSaveTest(keras_parameterized.TestCase):

  def test_model_save(self):
    input_dim = 5
    model = testing_utils.get_small_mlp(10, 3, input_dim)
    inputs = array_ops.ones((8, 5))

    if testing_utils.get_model_type() == 'subclass':
      model._set_inputs(inputs)

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save_lib.save(model, save_dir)

    if model.output_names:
      output_name = model.output_names[0]
      input_name = model.input_names[0]
    else:
      output_name = 'output_1'
      input_name = 'input_1'

    self.assertAllClose({output_name: model.predict_on_batch(inputs)},
                        _import_and_infer(save_dir,
                                          {input_name: np.ones((8, 5))}))


@test_util.run_deprecated_v1  # Not used in v2.
class ExtractModelMetricsTest(keras_parameterized.TestCase):

  def test_extract_model_metrics(self):
    a = keras.layers.Input(shape=(3,), name='input_a')
    b = keras.layers.Input(shape=(3,), name='input_b')

    dense = keras.layers.Dense(4, name='dense')
    c = dense(a)
    d = dense(b)
    e = keras.layers.Dropout(0.5, name='dropout')(c)

    model = keras.models.Model([a, b], [d, e])
    extract_metrics = saving_utils.extract_model_metrics(model)
    self.assertEqual(None, extract_metrics)

    extract_metric_names = [
        'dense_binary_accuracy', 'dropout_binary_accuracy',
        'dense_mean_squared_error', 'dropout_mean_squared_error'
    ]
    if tf2.enabled():
      extract_metric_names.extend(['dense_mae', 'dropout_mae'])
    else:
      extract_metric_names.extend(
          ['dense_mean_absolute_error', 'dropout_mean_absolute_error'])

    model_metric_names = ['loss', 'dense_loss', 'dropout_loss'
                         ] + extract_metric_names
    model.compile(
        loss='mae',
        metrics=[
            keras.metrics.BinaryAccuracy(), 'mae',
            keras.metrics.mean_squared_error
        ],
        optimizer=rmsprop.RMSPropOptimizer(learning_rate=0.01))
    extract_metrics = saving_utils.extract_model_metrics(model)
    self.assertEqual(set(model_metric_names), set(model.metrics_names))
    self.assertEqual(set(extract_metric_names), set(extract_metrics.keys()))


if __name__ == '__main__':
  test.main()
