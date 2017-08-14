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
            `Tensor` or `dict` of same.
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

from tensorflow.python.estimator.canned import dnn as dnn_core
from tensorflow.python.estimator.canned import linear as linear_core

# pylint: disable=protected-access
dnn_logit_fn_builder = dnn_core._dnn_logit_fn_builder
linear_logit_fn_builder = linear_core._linear_logit_fn_builder
# pylint: enable=protected-access
