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
"""Tests for Keras metrics serialization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import layers
from tensorflow.python.keras import metrics
from tensorflow.python.keras import optimizer_v2
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.util import nest

try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None


# Custom metric
class MyMeanAbsoluteError(metrics.MeanMetricWrapper):

  def __init__(self, name='my_mae', dtype=None):
    super(MyMeanAbsoluteError, self).__init__(_my_mae, name, dtype=dtype)


# Custom metric function
def _my_mae(y_true, y_pred):
  return keras.backend.mean(math_ops.abs(y_pred - y_true), axis=-1)


def _get_multi_io_model():
  inp_1 = layers.Input(shape=(1,), name='input_1')
  inp_2 = layers.Input(shape=(1,), name='input_2')
  d = testing_utils.Bias(name='output')
  out_1 = d(inp_1)
  out_2 = d(inp_2)
  return keras.Model([inp_1, inp_2], [out_1, out_2])


@keras_parameterized.run_all_keras_modes
@parameterized.named_parameters(
    dict(testcase_name='string', value=['mae']),
    dict(testcase_name='built_in_fn', value=[metrics.mae]),
    dict(testcase_name='built_in_class', value=[metrics.MeanAbsoluteError]),
    dict(testcase_name='custom_fn', value=[_my_mae]),
    dict(testcase_name='custom_class', value=[MyMeanAbsoluteError]),
    dict(testcase_name='list_of_strings', value=['mae', 'mae']),
    dict(
        testcase_name='list_of_built_in_fns', value=[metrics.mae, metrics.mae]),
    dict(
        testcase_name='list_of_built_in_classes',
        value=[metrics.MeanAbsoluteError, metrics.MeanAbsoluteError]),
    dict(testcase_name='list_of_custom_fns', value=[_my_mae, _my_mae]),
    dict(
        testcase_name='list_of_custom_classes',
        value=[MyMeanAbsoluteError, MyMeanAbsoluteError]),
    dict(testcase_name='list_of_string_and_list', value=['mae', ['mae']]),
    dict(
        testcase_name='list_of_built_in_fn_and_list',
        value=[metrics.mae, [metrics.mae]]),
    dict(
        testcase_name='list_of_built_in_class_and_list',
        value=[metrics.MeanAbsoluteError, [metrics.MeanAbsoluteError]]),
    dict(
        testcase_name='list_of_custom_fn_and_list', value=[_my_mae, [_my_mae]]),
    dict(
        testcase_name='list_of_custom_class_and_list',
        value=[MyMeanAbsoluteError, [MyMeanAbsoluteError]]),
    dict(
        testcase_name='list_of_lists_of_custom_fns',
        value=[[_my_mae], [_my_mae, 'mae']]),
    dict(
        testcase_name='list_of_lists_of_custom_classes',
        value=[[MyMeanAbsoluteError], [MyMeanAbsoluteError, 'mae']]),
    dict(
        testcase_name='dict_of_list_of_string',
        value={
            'output': ['mae'],
            'output_1': ['mae'],
        }),
    dict(
        testcase_name='dict_of_list_of_built_in_fn',
        value={
            'output': [metrics.mae],
            'output_1': [metrics.mae],
        }),
    dict(
        testcase_name='dict_of_list_of_built_in_class',
        value={
            'output': [metrics.MeanAbsoluteError],
            'output_1': [metrics.MeanAbsoluteError],
        }),
    dict(
        testcase_name='dict_of_list_of_custom_fn',
        value={
            'output': [_my_mae],
            'output_1': [_my_mae],
        }),
    dict(
        testcase_name='dict_of_list_of_custom_class',
        value={
            'output': [MyMeanAbsoluteError],
            'output_1': [MyMeanAbsoluteError],
        }),
    dict(
        testcase_name='dict_of_string',
        value={
            'output': 'mae',
            'output_1': 'mae',
        }),
    dict(
        testcase_name='dict_of_built_in_fn',
        value={
            'output': metrics.mae,
            'output_1': metrics.mae,
        }),
    dict(
        testcase_name='dict_of_built_in_class',
        value={
            'output': metrics.MeanAbsoluteError,
            'output_1': metrics.MeanAbsoluteError,
        }),
    dict(
        testcase_name='dict_of_custom_fn',
        value={
            'output': _my_mae,
            'output_1': _my_mae
        }),
    dict(
        testcase_name='dict_of_custom_class',
        value={
            'output': MyMeanAbsoluteError,
            'output_1': MyMeanAbsoluteError,
        }),
)
class MetricsSerialization(keras_parameterized.TestCase):

  def setUp(self):
    super(MetricsSerialization, self).setUp()
    tmpdir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, tmpdir)
    self.model_filename = os.path.join(tmpdir, 'tmp_model_metric.h5')
    self.x = np.array([[0.], [1.], [2.]], dtype='float32')
    self.y = np.array([[0.5], [2.], [3.5]], dtype='float32')
    self.w = np.array([1.25, 0.5, 1.25], dtype='float32')

  def test_serializing_model_with_metric_with_custom_object_scope(self, value):

    def get_instance(x):
      if isinstance(x, str):
        return x
      if isinstance(x, type) and issubclass(x, metrics.Metric):
        return x()
      return x

    metric_input = nest.map_structure(get_instance, value)
    weighted_metric_input = nest.map_structure(get_instance, value)

    with generic_utils.custom_object_scope({
        'MyMeanAbsoluteError': MyMeanAbsoluteError,
        '_my_mae': _my_mae,
        'Bias': testing_utils.Bias,
    }):
      model = _get_multi_io_model()
      model.compile(
          optimizer_v2.gradient_descent.SGD(0.1),
          'mae',
          metrics=metric_input,
          weighted_metrics=weighted_metric_input,
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())
      history = model.fit([self.x, self.x], [self.y, self.y],
                          batch_size=3,
                          epochs=3,
                          sample_weight=[self.w, self.w])

      # Assert training.
      self.assertAllClose(history.history['loss'], [2., 1.6, 1.2], 1e-3)
      eval_results = model.evaluate([self.x, self.x], [self.y, self.y],
                                    sample_weight=[self.w, self.w])

      if h5py is None:
        return
      model.save(self.model_filename)
      loaded_model = keras.models.load_model(self.model_filename)
      loaded_model.predict([self.x, self.x])
      loaded_eval_results = loaded_model.evaluate(
          [self.x, self.x], [self.y, self.y], sample_weight=[self.w, self.w])

      # Assert all evaluation results are the same.
      self.assertAllClose(eval_results, loaded_eval_results, 1e-9)

  def test_serializing_model_with_metric_with_custom_objects(self, value):

    def get_instance(x):
      if isinstance(x, str):
        return x
      if isinstance(x, type) and issubclass(x, metrics.Metric):
        return x()
      return x

    metric_input = nest.map_structure(get_instance, value)
    weighted_metric_input = nest.map_structure(get_instance, value)

    model = _get_multi_io_model()
    model.compile(
        optimizer_v2.gradient_descent.SGD(0.1),
        'mae',
        metrics=metric_input,
        weighted_metrics=weighted_metric_input,
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    history = model.fit([self.x, self.x], [self.y, self.y],
                        batch_size=3,
                        epochs=3,
                        sample_weight=[self.w, self.w])

    # Assert training.
    self.assertAllClose(history.history['loss'], [2., 1.6, 1.2], 1e-3)
    eval_results = model.evaluate([self.x, self.x], [self.y, self.y],
                                  sample_weight=[self.w, self.w])

    if h5py is None:
      return
    model.save(self.model_filename)
    loaded_model = keras.models.load_model(
        self.model_filename,
        custom_objects={
            'MyMeanAbsoluteError': MyMeanAbsoluteError,
            '_my_mae': _my_mae,
            'Bias': testing_utils.Bias,
        })
    loaded_model.predict([self.x, self.x])
    loaded_eval_results = loaded_model.evaluate([self.x, self.x],
                                                [self.y, self.y],
                                                sample_weight=[self.w, self.w])

    # Assert all evaluation results are the same.
    self.assertAllClose(eval_results, loaded_eval_results, 1e-9)


if __name__ == '__main__':
  test.main()
