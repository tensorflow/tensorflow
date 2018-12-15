# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=protected-access
"""Utils related to keras metrics.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.keras import metrics
from tensorflow.python.ops import metrics as metrics_module


def extract_model_metrics_as_v1_metrics(model):
  """Convert metrics from a Keras model to (value, update) ops.

  This is used for converting Keras models to Estimators and SavedModels.

  Args:
    model: A `tf.keras.Model` object.

  Returns:
    Dictionary mapping metric names to tuples of (value, update) ops. May return
    `None` if the model does not contain any metrics.
  """
  if not getattr(model, 'metrics', None):
    return None

  eval_metric_ops = {}

  def get_metric_name(metric):
    if isinstance(metric, metrics.Metric):
      return metric.name
    if callable(metric):
      return metric.__name__
    assert isinstance(metric, six.string_types)
    return metric

  # When each metric maps to an output
  if isinstance(model.metrics, dict):
    for i, output_name in enumerate(model.metrics.keys()):
      # `metric` is the user given metric value in `compile`. This can be
      # metric name (`acc`), metric function (binary_accuracy) or a metric
      # object (BinaryAccuracy()).
      metric = model.metrics[output_name]
      metric_name = get_metric_name(metric)
      # When some outputs use the same metric
      if list(model.metrics.values()).count(metric_name) > 1:
        metric_name += '_' + output_name
      if isinstance(metric, metrics.Metric):
        eval_metric_ops[metric_name] = metric
      else:
        eval_metric_ops[metric_name] = metrics_module.mean(
            model.metrics_tensors[i - len(model.metrics)])
  else:
    for i, metric in enumerate(model.metrics):
      metric_name = get_metric_name(metric)
      if isinstance(metric, metrics.Metric):
        eval_metric_ops[metric_name] = metric
      else:
        eval_metric_ops[metric_name] = metrics_module.mean(
            model.metrics_tensors[i])
  return eval_metric_ops
