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
"""TensorFlow monitoring APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import pywrap_tensorflow


def gauge(name, label, value):
  """Set the value of a Gauge metric.

  If the metric with this name does not exist, it will create a new metric.

  Args:
    name: metric name
    label: long label
    value: a int64 value
  """
  pywrap_tensorflow.TFE_MonitoringSetGauge(name, label, value)


def counter(name, label, value):
  """Increase a Counter metric by the given value.

  If the metric with this name does not exist, it will create a new metric.

  Args:
    name: metric name
    label: metric label
    value: a long value
  """
  pywrap_tensorflow.TFE_MonitoringAddCounter(name, label, value)


def sampler(name, label, value):
  """Add the value of a Sampler metric.

  If the metric with this name does not exist, it will create a new metric.

  Args:
    name: metric name
    label: metric label
    value: a double value
  """
  pywrap_tensorflow.TFE_MonitoringAddSampler(name, label, value)
