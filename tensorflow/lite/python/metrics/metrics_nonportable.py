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
import uuid

from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.lite.python.metrics import metrics_interface
from tensorflow.lite.python.metrics.wrapper import metrics_wrapper
from tensorflow.python.eager import monitoring

_counter_debugger_creation = monitoring.Counter(
    '/tensorflow/lite/quantization_debugger/created',
    'Counter for the number of debugger created.')

_counter_interpreter_creation = monitoring.Counter(
    '/tensorflow/lite/interpreter/created',
    'Counter for number of interpreter created in Python.', 'language')

# The following are conversion metrics. Attempt and success are kept separated
# instead of using a single metric with a label because the converter may
# raise exceptions if conversion failed. That may lead to cases when we are
# unable to capture the conversion attempt. Increasing attempt count at the
# beginning of conversion process and the success count at the end is more
# suitable in these cases.
_counter_conversion_attempt = monitoring.Counter(
    '/tensorflow/lite/convert/attempt',
    'Counter for number of conversion attempts.')

_counter_conversion_success = monitoring.Counter(
    '/tensorflow/lite/convert/success',
    'Counter for number of successful conversions.')

_gauge_conversion_params = monitoring.StringGauge(
    '/tensorflow/lite/convert/params',
    'Gauge for keeping conversion parameters.', 'name')

_gauge_conversion_errors = monitoring.StringGauge(
    '/tensorflow/lite/convert/errors',
    'Gauge for collecting conversion errors. The value represents the error '
    'message.', 'component', 'subcomponent', 'op_name', 'error_code')

_gauge_conversion_latency = monitoring.IntGauge(
    '/tensorflow/lite/convert/latency', 'Conversion latency in ms.')


class TFLiteMetrics(metrics_interface.TFLiteMetricsInterface):
  """TFLite metrics helper for prod (borg) environment.

  Attributes:
    model_hash: A string containing the hash of the model binary.
    model_path: A string containing the path of the model for debugging
      purposes.
  """

  def __init__(self,
               model_hash: Optional[Text] = None,
               model_path: Optional[Text] = None) -> None:
    del self  # Temporarily removing self until parameter logic is implemented.
    if model_hash and not model_path or not model_hash and model_path:
      raise ValueError('Both model metadata(model_hash, model_path) should be '
                       'given at the same time.')
    if model_hash:
      # TODO(b/180400857): Create stub once the service is implemented.
      pass

  def increase_counter_debugger_creation(self):
    _counter_debugger_creation.get_cell().increase_by(1)

  def increase_counter_interpreter_creation(self):
    _counter_interpreter_creation.get_cell('python').increase_by(1)

  def increase_counter_converter_attempt(self):
    _counter_conversion_attempt.get_cell().increase_by(1)

  def increase_counter_converter_success(self):
    _counter_conversion_success.get_cell().increase_by(1)

  def set_converter_param(self, name, value):
    _gauge_conversion_params.get_cell(name).set(value)

  def set_converter_error(
      self, error_data: converter_error_data_pb2.ConverterErrorData):
    error_code_str = converter_error_data_pb2.ConverterErrorData.ErrorCode.Name(
        error_data.error_code)
    _gauge_conversion_errors.get_cell(
        error_data.component,
        error_data.subcomponent,
        error_data.operator.name,
        error_code_str,
    ).set(error_data.error_message)

  def set_converter_latency(self, value):
    _gauge_conversion_latency.get_cell().set(value)


class TFLiteConverterMetrics(TFLiteMetrics):
  """Similar to TFLiteMetrics but specialized for converter.

  A unique session id will be created for each new TFLiteConverterMetrics.
  """

  def __init__(self) -> None:
    super(TFLiteConverterMetrics, self).__init__()
    session_id = uuid.uuid4().hex
    self._metrics_exporter = metrics_wrapper.MetricsWrapper(session_id)
    self._exported = False

  def __del__(self):
    if not self._exported:
      self.export_metrics()

  def set_export_required(self):
    self._exported = False

  def export_metrics(self):
    self._metrics_exporter.ExportMetrics()
    self._exported = True
