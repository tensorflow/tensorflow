# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
from tensorflow.contrib.optimizer_v2 import adagrad
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import test
from tensorflow.python.estimator import run_config
from tensorflow.python.estimator.canned import dnn_linear_combined
from tensorflow.python.estimator.canned import prediction_keys
from tensorflow.python.estimator.export import export
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.summary.writer import writer_cache


class DNNLinearCombinedClassifierIntegrationTest(test.TestCase,
                                                 parameterized.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def dataset_input_fn(self, x, y, batch_size, shuffle):

    def input_fn():
      dataset = dataset_ops.Dataset.from_tensor_slices((x, y))
      if shuffle:
        dataset = dataset.shuffle(batch_size)
      dataset = dataset.repeat(10).batch(batch_size)
      return dataset

    return input_fn

  @combinations.generate(
      combinations.combine(
          mode=['graph'],
          distribution=[
              combinations.one_device_strategy,
              combinations.mirrored_strategy_with_gpu_and_cpu,
              combinations.mirrored_strategy_with_two_gpus
          ]))
  def test_complete_flow_with_mode(self, distribution):
    label_dimension = 2
    input_dimension = label_dimension
    batch_size = 10
    data = np.linspace(0., 2., batch_size * label_dimension, dtype=np.float32)
    data = data.reshape(batch_size, label_dimension)
    train_input_fn = self.dataset_input_fn(
        x={'x': data},
        y=data,
        batch_size=batch_size // len(distribution.worker_devices),
        shuffle=True)
    eval_input_fn = self.dataset_input_fn(
        x={'x': data},
        y=data,
        batch_size=batch_size // len(distribution.worker_devices),
        shuffle=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': data}, batch_size=batch_size, shuffle=False)

    linear_feature_columns = [
        feature_column.numeric_column('x', shape=(input_dimension,))
    ]
    dnn_feature_columns = [
        feature_column.numeric_column('x', shape=(input_dimension,))
    ]
    feature_columns = linear_feature_columns + dnn_feature_columns
    estimator = dnn_linear_combined.DNNLinearCombinedRegressor(
        linear_feature_columns=linear_feature_columns,
        dnn_hidden_units=(2, 2),
        dnn_feature_columns=dnn_feature_columns,
        label_dimension=label_dimension,
        model_dir=self._model_dir,
        # TODO(isaprykin): Work around the colocate_with error.
        dnn_optimizer=adagrad.AdagradOptimizer(0.001),
        linear_optimizer=adagrad.AdagradOptimizer(0.001),
        config=run_config.RunConfig(
            train_distribute=distribution, eval_distribute=distribution))

    num_steps = 10
    estimator.train(train_input_fn, steps=num_steps)

    scores = estimator.evaluate(eval_input_fn)
    self.assertEqual(num_steps, scores[ops.GraphKeys.GLOBAL_STEP])
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


if __name__ == '__main__':
  test.main()
