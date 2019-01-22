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
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
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
    expected_outputs = {model.output_names[0]: model(inputs)}

    self._assert_all_close(expected_outputs, signature_outputs)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_trace_model_outputs_after_fitting(self):
    input_dim = 5 if testing_utils.get_model_type() == 'functional' else None
    model = testing_utils.get_small_mlp(10, 3, input_dim)
    model.compile(optimizer='sgd', loss='mse')
    model.fit(x=np.random.random((8, 5)),
              y=np.random.random((8, 3)), epochs=2)

    inputs = array_ops.ones((8, 5))

    fn = saving_utils.trace_model_call(model)
    signature_outputs = fn(inputs)
    expected_outputs = {model.output_names[0]: model(inputs)}

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

    model.compile(optimizer='sgd', loss='mse')
    model.fit(x=[np.random.random((8, input_dim)).astype(np.float32),
                 np.random.random((8, input_dim)).astype(np.float32)],
              y=[np.random.random((8, num_classes)).astype(np.float32),
                 np.random.random((8, num_classes_b)).astype(np.float32)],
              epochs=2)

    fn = saving_utils.trace_model_call(model)
    signature_outputs = fn([input_a_np, input_b_np])
    outputs = model([input_a_np, input_b_np])
    expected_outputs = {model.output_names[0]: outputs[0],
                        model.output_names[1]: outputs[1]}

    self._assert_all_close(expected_outputs, signature_outputs)

  @keras_parameterized.run_all_keras_modes
  def test_specify_input_signature(self):
    model = testing_utils.get_small_sequential_mlp(10, 3, None)
    inputs = array_ops.ones((8, 5))

    with self.assertRaisesRegexp(ValueError, 'input shapes have not been set'):
      saving_utils.trace_model_call(model)

    fn = saving_utils.trace_model_call(
        model, [tensor_spec.TensorSpec(shape=[None, 5], dtype=dtypes.float32)])
    signature_outputs = fn(inputs)
    expected_outputs = {model.output_names[0]: model(inputs)}
    self._assert_all_close(expected_outputs, signature_outputs)

  @keras_parameterized.run_all_keras_modes
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


def _import_and_infer(save_dir, inputs):
  """Import a SavedModel into a TF 1.x-style graph and run `signature_key`."""
  graph = ops.Graph()
  with graph.as_default(), session_lib.Session() as session:
    model = loader.load(session, [tag_constants.SERVING], save_dir)
    signature = model.signature_def[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    assert set(inputs.keys()) == set(signature.inputs.keys())
    feed_dict = {}
    for arg_name in inputs.keys():
      feed_dict[graph.get_tensor_by_name(signature.inputs[arg_name].name)] = (
          inputs[arg_name])
    output_dict = {}
    for output_name, output_tensor_info in signature.outputs.items():
      output_dict[output_name] = graph.get_tensor_by_name(
          output_tensor_info.name)
    return session.run(output_dict, feed_dict=feed_dict)


class ModelSaveTest(keras_parameterized.TestCase):

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_model_save(self):
    input_dim = 5
    model = testing_utils.get_small_mlp(10, 3, input_dim)
    inputs = array_ops.ones((8, 5))

    if testing_utils.get_model_type() == 'subclass':
      model._set_inputs(inputs)

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save_lib.save(model, save_dir)

    self.assertAllClose(
        {model.output_names[0]: model.predict_on_batch(inputs)},
        _import_and_infer(save_dir, {model.input_names[0]: np.ones((8, 5))}))


class ExtractModelMetricsTest(test.TestCase):

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

    extract_metric_names = ['dense_loss', 'dropout_loss',
                            'dense_binary_accuracy', 'dropout_binary_accuracy']
    model_metric_names = ['loss'] + extract_metric_names
    model.compile(loss='mae',
                  metrics=[keras.metrics.BinaryAccuracy()],
                  optimizer=rmsprop.RMSPropOptimizer(learning_rate=0.01),
                  run_eagerly=None)
    extract_metrics = saving_utils.extract_model_metrics(model)
    self.assertEqual(set(model_metric_names), set(model.metrics_names))
    self.assertEqual(set(extract_metric_names), set(extract_metrics.keys()))

if __name__ == '__main__':
  test.main()
