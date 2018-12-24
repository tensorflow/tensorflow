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
"""Utils related to keras model saving."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import metrics


def extract_model_metrics(model):
  """Convert metrics from a Keras model to (value, update) ops.

  This is used for converting Keras models to Estimators and SavedModels.

  Args:
    model: A `tf.keras.Model` object.

  Returns:
    Dictionary mapping metric names to tuples of (value, update) ops. May return
    `None` if the model does not contain any metrics.
  """
  if not getattr(model, '_compile_metrics', None):
    return None

  # TODO(psv/kathywu): use this implementation in model to estimator flow.
  eval_metric_ops = {}
  for metric_name in model.metrics_names[1:]:  # Index 0 is `loss`.
    m = metrics.Mean()
    m(model._compile_metrics_tensors[metric_name])
    eval_metric_ops[metric_name] = m
  return eval_metric_ops
