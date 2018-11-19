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
"""Tests that show that DistributionStrategy works with canned Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import tempfile
from absl.testing import parameterized
import numpy as np
import six

from tensorflow.contrib.distribute.python import combinations
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import run_config
from tensorflow.python.estimator import training
from tensorflow.python.estimator.canned import dnn_linear_combined
from tensorflow.python.estimator.canned import prediction_keys
from tensorflow.python.estimator.export import export
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.feature_column import feature_column_lib as feature_column
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import distribution_strategy_context as ds_context


class KerasOptimizerV2IntegrationTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def dataset_input_fn(self, x, y, batch_size):

    def input_fn():
      dataset = dataset_ops.Dataset.from_tensor_slices((x, y))
      dataset = dataset.repeat(1).batch(batch_size)
      return dataset

    return input_fn

  @combinations.generate(
      combinations.combine(
          mode=['graph'],
          distribution=[
              combinations.one_device_strategy,
              combinations.mirrored_strategy_with_gpu_and_cpu,
              combinations.mirrored_strategy_with_two_gpus,
              combinations.core_mirrored_strategy_with_gpu_and_cpu,
              combinations.core_mirrored_strategy_with_two_gpus
          ],
          use_train_and_evaluate=[True, False]))
  def test_complete_flow_with_mode(self, distribution, use_train_and_evaluate):
    label_dimension = 2
    input_dimension = label_dimension
    batch_size = 10
    data = np.linspace(0., 2., batch_size * label_dimension, dtype=np.float32)
    data = data.reshape(batch_size, label_dimension)
    train_input_fn = self.dataset_input_fn(
        x={'x': data},
        y=data,
        batch_size=batch_size // len(distribution.worker_devices))
    eval_input_fn = self.dataset_input_fn(
        x={'x': data},
        y=data,
        batch_size=batch_size // len(distribution.worker_devices))
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': data}, batch_size=batch_size, shuffle=False)

    linear_feature_columns = [
        feature_column.numeric_column('x', shape=(input_dimension,))
    ]
    dnn_feature_columns = [
        feature_column.numeric_column('x', shape=(input_dimension,))
    ]
    feature_columns = linear_feature_columns + dnn_feature_columns
    session_config = config_pb2.ConfigProto(
        log_device_placement=True, allow_soft_placement=True)
    estimator = dnn_linear_combined.DNNLinearCombinedRegressor(
        linear_feature_columns=linear_feature_columns,
        dnn_hidden_units=(2, 2),
        dnn_feature_columns=dnn_feature_columns,
        label_dimension=label_dimension,
        model_dir=self._model_dir,
        dnn_optimizer=adam.Adam(0.001),
        linear_optimizer=adam.Adam(0.001),
        config=run_config.RunConfig(
            train_distribute=distribution,
            eval_distribute=distribution,
            session_config=session_config))

    num_steps = 2
    if use_train_and_evaluate:
      scores, _ = training.train_and_evaluate(
          estimator, training.TrainSpec(train_input_fn, max_steps=num_steps),
          training.EvalSpec(eval_input_fn))
    else:
      estimator.train(train_input_fn, steps=num_steps)
      scores = estimator.evaluate(eval_input_fn)

    self.assertIn('loss', six.iterkeys(scores))

    predictions = np.array([
        x[prediction_keys.PredictionKeys.PREDICTIONS]
        for x in estimator.predict(predict_input_fn)
    ])
    self.assertAllEqual((batch_size, label_dimension), predictions.shape)

    feature_spec = feature_column.make_parse_example_spec(feature_columns)
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)
    export_dir = estimator.export_savedmodel(tempfile.mkdtemp(),
                                             serving_input_receiver_fn)
    self.assertTrue(gfile.Exists(export_dir))

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)


def get_model():
  x = keras.layers.Input(shape=(3,), name='input')
  y = keras.layers.Dense(4, name='dense')(x)
  model = keras.Model(x, y)
  return model


class MirroredStrategyOptimizerV2Test(test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.combine(
      distribution=[
          combinations.mirrored_strategy_with_gpu_and_cpu,
          combinations.core_mirrored_strategy_with_gpu_and_cpu],
      mode=['graph']))
  def testKerasOptimizerWithUnequalInput(self, distribution):
    def create_fn():
      var = variables.Variable(
          2.0, name='var', aggregation=variable_scope.VariableAggregation.SUM)
      # grad for cpu is 1, grad for gpu is 2, avg grad is 1.5.
      loss = math_ops.cast(_replica_id() + 1, dtype=dtypes.float32) * var
      optimizer = adam.Adam(learning_rate=0.01, beta_1=0.2, beta_2=0.2)
      train_op = optimizer.minimize(loss, var_list=[var])
      m = optimizer.get_slot(var, 'm')
      v = optimizer.get_slot(var, 'v')
      return (var, m, v, train_op, optimizer.iterations)

    devices = ['/device:GPU:0', '/device:CPU:0']
    with distribution.scope():
      (var, m, v, op, counter) = distribution.call_for_each_replica(create_fn)
      self.evaluate(variables.global_variables_initializer())
      var_val = [2.0, 2.0, 2.0]
      self.assertAllClose(
          var_val,
          self.evaluate(
              [distribution.read_var(var),
               var.get(devices[0]),
               var.get(devices[1])]))
      self.assertAllClose([0, 0, 0],
                          self.evaluate([
                              distribution.read_var(counter),
                              counter.get(devices[0]),
                              counter.get(devices[1])
                          ]))

      train_op = distribution.unwrap(op)
      self.evaluate(train_op)
      # m(1) = beta1 * m(0) + (1-beta1) * grad = 0.2 * 0 + 0.8 * (1 + 2) / 2
      m_val = [1.2, 1.2, 1.2]
      # assert slot variables in both replicas are the same.
      self.assertAllClose(
          m_val,
          self.evaluate(
              [distribution.read_var(m),
               m.get(devices[0]),
               m.get(devices[1])]))
      # v(1) = beta2 * v(0) + (1-beta2) * grad^2 = 0.2 * 0 + 0.8 * 2.25
      v_val = [1.8, 1.8, 1.8]
      self.assertAllClose(
          v_val,
          self.evaluate(
              [distribution.read_var(v),
               v.get(devices[0]),
               v.get(devices[1])]))
      # var(1) = var(0) - lr * m(1) * sqrt(1 - beta2) / sqrt(v(1)) / (1 - beta1)
      #        = 2.0 - 0.01 * 1.2 * sqrt(0.8) / sqrt(1.8) / 0.8
      var_val = [1.99, 1.99, 1.99]
      self.assertAllClose(
          var_val,
          self.evaluate(
              [distribution.read_var(var),
               var.get(devices[0]),
               var.get(devices[1])]))
      self.assertAllClose([1, 1, 1],
                          self.evaluate([
                              distribution.read_var(counter),
                              counter.get(devices[0]),
                              counter.get(devices[1])
                          ]))

      self.evaluate(train_op)
      # m(2) = beta1 * m(1) + (1-beta1) * grad = 0.2 * 1.2 + 0.8 * 1.5
      m_val = [1.44, 1.44, 1.44]
      self.assertAllClose(
          m_val,
          self.evaluate(
              [distribution.read_var(m),
               m.get(devices[0]),
               m.get(devices[1])]))
      # v(2) = beta2 * v(1) + (1-beta2) * grad^2 = 0.2 * 1.8 + 0.8 * 2.25
      v_val = [2.16, 2.16, 2.16]
      self.assertAllClose(
          v_val,
          self.evaluate(
              [distribution.read_var(v),
               v.get(devices[0]),
               v.get(devices[1])]))
      self.assertAllClose([2, 2, 2],
                          self.evaluate([
                              distribution.read_var(counter),
                              counter.get(devices[0]),
                              counter.get(devices[1])
                          ]))

  @combinations.generate(combinations.combine(
      distribution=[
          combinations.mirrored_strategy_with_gpu_and_cpu,
          combinations.core_mirrored_strategy_with_gpu_and_cpu],
      mode=['graph']))
  def testOptimizerWithKerasModelAndNumpyArrays(self, distribution):

    with self.cached_session():
      model = get_model()
      optimizer = gradient_descent.SGD(0.001)
      loss = 'mse'
      metrics = ['mae']
      model.compile(optimizer, loss, metrics=metrics, distribute=distribution)

      inputs = np.zeros((64, 3), dtype=np.float32)
      targets = np.zeros((64, 4), dtype=np.float32)

      model.fit(
          inputs,
          targets,
          epochs=1,
          batch_size=2,
          verbose=0,
          validation_data=(inputs, targets))
      model.evaluate(inputs, targets)
      model.predict(inputs)


def _replica_id():
  replica_id = ds_context.get_replica_context().replica_id_in_sync_group
  if not isinstance(replica_id, ops.Tensor):
    replica_id = constant_op.constant(replica_id)
  return replica_id


if __name__ == '__main__':
  test.main()
