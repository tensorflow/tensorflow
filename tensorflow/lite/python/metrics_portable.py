# Lint as: python2, python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Python TFLite metrics helper."""
import os
from typing import Optional, Text

# pylint: disable=g-import-not-at-top
if not os.path.splitext(__file__)[0].endswith(
    os.path.join('tflite_runtime', 'metrics_portable')):
  # This file is part of tensorflow package.
  from tensorflow.lite.python import metrics_interface  # type: ignore
else:
  # This file is part of tflite_runtime package.
  from tflite_runtime import metrics_interface  # type: ignore
# pylint: enable=g-import-not-at-top


class TFLiteMetrics(metrics_interface.TFLiteMetricsInterface):
  """TFLite metrics helper."""

  def __init__(self,
               model_hash: Optional[Text] = None,
               model_path: Optional[Text] = None) -> None:
    pass

  def increase_counter_debugger_creation(self):
    pass

  def increase_counter_interpreter_creation(self):
    pass

  def increase_counter_converter_attempt(self):
    pass

  def increase_counter_converter_success(self):
    pass

  def set_converter_param(self, name, value):
    pass


class TFLiteConverterMetrics(TFLiteMetrics):
  """Similar to TFLiteMetrics but specialized for converter."""

  def __del__(self):
    pass

  def set_export_required(self):
    pass

  def export_metrics(self):
    pass
