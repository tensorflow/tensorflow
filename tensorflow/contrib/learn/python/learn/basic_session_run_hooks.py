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
"""Some common SessionRunHook classes (deprected).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.util.deprecation import deprecated_alias

# pylint: disable=invalid-name
LoggingTensorHook = deprecated_alias(
    'tf.contrib.learn.basic_session_run_hooks.LoggingTensorHook',
    'tf.train.LoggingTensorHook',
    basic_session_run_hooks.LoggingTensorHook)
StopAtStepHook = deprecated_alias(
    'tf.contrib.learn.basic_session_run_hooks.StopAtStepHook',
    'tf.train.StopAtStepHook',
    basic_session_run_hooks.StopAtStepHook)
CheckpointSaverHook = deprecated_alias(
    'tf.contrib.learn.basic_session_run_hooks.CheckpointSaverHook',
    'tf.train.CheckpointSaverHook',
    basic_session_run_hooks.CheckpointSaverHook)
StepCounterHook = deprecated_alias(
    'tf.contrib.learn.basic_session_run_hooks.StepCounterHook',
    'tf.train.StepCounterHook',
    basic_session_run_hooks.StepCounterHook)
NanLossDuringTrainingError = deprecated_alias(
    'tf.contrib.learn.basic_session_run_hooks.NanLossDuringTrainingError',
    'tf.train.NanLossDuringTrainingError',
    basic_session_run_hooks.NanLossDuringTrainingError)
NanTensorHook = deprecated_alias(
    'tf.contrib.learn.basic_session_run_hooks.NanTensorHook',
    'tf.train.NanTensorHook',
    basic_session_run_hooks.NanTensorHook)
SummarySaverHook = deprecated_alias(
    'tf.contrib.learn.basic_session_run_hooks.SummarySaverHook',
    'tf.train.SummarySaverHook',
    basic_session_run_hooks.SummarySaverHook)
# pylint: enable=invalid-name
