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
from typing import Optional, Text

from tensorflow.lite.python import metrics_interface
from tensorflow.python.eager import monitoring


class TFLiteMetrics(metrics_interface.TFLiteMetricsInterface):
  """TFLite metrics helper for prod (borg) environment.

  Attributes:
    md5: A string containing the MD5 hash of the model binary.
    model_path: A string containing the path of the model for debugging
      purposes.
  """

  _counter_debugger_creation = monitoring.Counter(
      '/tensorflow/lite/quantization_debugger/created',
      'Counter for the number of debugger created.')

  _counter_interpreter_creation = monitoring.Counter(
      '/tensorflow/lite/interpreter/created',
      'Counter for number of interpreter created in Python.', 'language')

  def __init__(self,
               md5: Optional[Text] = None,
               model_path: Optional[Text] = None) -> None:
    del self  # Temporarily removing self until parameter logic is implemented.
    if md5 and not model_path or not md5 and model_path:
      raise ValueError('Both model metadata(md5, model_path) should be given '
                       'at the same time.')
    if md5:
      # TODO(b/180400857): Create stub once the service is implemented.
      pass

  def increase_counter_debugger_creation(self):
    self._counter_debugger_creation.get_cell().increase_by(1)

  def increase_counter_interpreter_creation(self):
    self._counter_interpreter_creation.get_cell('python').increase_by(1)
