# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Aliases for logit_fn builders used by canned (core) tf.Estimator's.

A logit_fn is an abstraction within model_fn that factors out the logit
construction logic.  Its output can be fed into Heads or otherwise composed.  It
should follow the following signature:

Args:
`features`: This is the first item returned from the `input_fn` passed to
            `train`, `evaluate`, and `predict`. This should be a single
            `Tensor` or `dict` of same, and is the only required argument.
`mode`: Optional. Specifies if this training, evaluation or prediction. See
        `ModeKeys`.
`params`: Optional `dict` of hyperparameters.  Will receive what is passed to
          Estimator in `params` parameter. This allows configuration of
          Estimators from hyperparameter tuning.
`config`: Optional configuration object. Will receive what is passed to
          Estimator in `config` parameter, or the default `config`. Allows
          updating things in your model_fn based on configuration such as
          `num_ps_replicas`, or `model_dir`.

Returns:
    A Tensor representing the logits.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.estimator.canned import dnn as dnn_core
from tensorflow.python.estimator.canned import linear as linear_core
from tensorflow.python.framework import ops
from tensorflow.python.util import function_utils

# pylint: disable=protected-access
dnn_logit_fn_builder = dnn_core._dnn_logit_fn_builder
linear_logit_fn_builder = linear_core._linear_logit_fn_builder
# pylint: enable=protected-access


def call_logit_fn(logit_fn, features, mode, params, config):
  """Calls logit_fn.

  A utility function that calls the provided logit_fn with the relevant subset
  of provided arguments.  Similar to tf.estimator._call_model_fn().

  Args:
    logit_fn: A logit_fn as defined above.
    features: The features dict.
    mode: TRAIN / EVAL / PREDICT ModeKeys.
    params: The hyperparameter dict.
    config: The configuration object.

  Returns:
    A logit Tensor, the output of logit_fn.

  Raises:
    ValueError: if logit_fn does not return a Tensor or a dictionary mapping
      strings to Tensors.
  """
  logit_fn_args = function_utils.fn_args(logit_fn)
  kwargs = {}
  if 'mode' in logit_fn_args:
    kwargs['mode'] = mode
  if 'params' in logit_fn_args:
    kwargs['params'] = params
  if 'config' in logit_fn_args:
    kwargs['config'] = config
  logit_fn_results = logit_fn(features=features, **kwargs)

  result_is_valid_dictionary = (
      isinstance(logit_fn_results, dict) and
      all([(isinstance(k, six.string_types) and isinstance(v, ops.Tensor))
           for k, v in six.iteritems(logit_fn_results)]))
  result_is_tensor = isinstance(logit_fn_results, ops.Tensor)

  if not (result_is_valid_dictionary or result_is_tensor):
    raise ValueError('logit_fn should return a Tensor or a dictionary mapping '
                     'strings to Tensors.  logit_fn returned: %s' %
                     logit_fn_results)

  return logit_fn_results
