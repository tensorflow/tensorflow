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

import collections

from tensorflow.python import pywrap_tensorflow

_MetricMethod = collections.namedtuple('MetricMethod', 'create delete get_cell')
_counter_methods = [
    _MetricMethod(
        create=pywrap_tensorflow.TFE_MonitoringNewCounter0,
        delete=pywrap_tensorflow.TFE_MonitoringDeleteCounter0,
        get_cell=pywrap_tensorflow.TFE_MonitoringGetCellCounter0),
    _MetricMethod(
        create=pywrap_tensorflow.TFE_MonitoringNewCounter1,
        delete=pywrap_tensorflow.TFE_MonitoringDeleteCounter1,
        get_cell=pywrap_tensorflow.TFE_MonitoringGetCellCounter1),
    _MetricMethod(
        create=pywrap_tensorflow.TFE_MonitoringNewCounter2,
        delete=pywrap_tensorflow.TFE_MonitoringDeleteCounter2,
        get_cell=pywrap_tensorflow.TFE_MonitoringGetCellCounter2),
]


def gauge(name, label, value):
  """Set the value of a Gauge metric.

  If the metric with this name does not exist, it will create a new metric.

  Args:
    name: metric name
    label: long label
    value: a int64 value
  """
  pywrap_tensorflow.TFE_MonitoringSetGauge(name, label, value)


def sampler(name, label, value):
  """Add the value of a Sampler metric.

  If the metric with this name does not exist, it will create a new metric.

  Args:
    name: metric name
    label: metric label
    value: a double value
  """
  pywrap_tensorflow.TFE_MonitoringAddSampler(name, label, value)


class Metric(object):
  """The base class of metric."""

  def __init__(self, metric_name, metric_methods, label_length, *args):
    """Creates a new metric.

    Args:
      metric_name: name of the metric class.
      metric_methods: list of swig metric methods.
      label_length: length of label args.
      *args: the arguments to call create method.
    """
    self._metric_name = metric_name
    self._metric_methods = metric_methods
    self._label_length = label_length

    if label_length >= len(self._metric_methods):
      raise ValueError('Cannot create {} metric with label >= {}'.format(
          self._metric_name, len(self._metric_methods)))

    self._metric = self._metric_methods[self._label_length].create(*args)

  def __del__(self):
    self._metric_methods[self._label_length].delete(self._metric)

  def get_cell(self, *labels):
    """Retrieves the cell."""
    if len(labels) != self._label_length:
      raise ValueError('The {} expects taking {} labels'.format(
          self._metric_name, self._label_length))
    return self._metric_methods[self._label_length].get_cell(
        self._metric, *labels)


class CounterCell(object):
  """CounterCell stores each value of a Counter."""

  def __init__(self, cell):
    """Creates a new CounterCell.

    Args:
      cell: A c pointer of TFE_MonitoringCounterCell.
    """
    self._cell = cell

  def increase_by(self, value):
    """Atomically increments the value.

    Args:
      value: non-negative value.
    """
    pywrap_tensorflow.TFE_MonitoringCounterCellIncrementBy(self._cell, value)

  def value(self):
    """Retrieves the current value."""
    return pywrap_tensorflow.TFE_MonitoringCounterCellValue(self._cell)


class Counter(Metric):
  """A stateful class for updating a cumulative integer metric."""

  def __init__(self, name, description, *labels):
    """Creates a new Counter.

    Args:
      name: name of the new metric.
      description: description of the new metric.
      *labels: The label list of the new metrics
    """
    super(Counter, self).__init__('Counter', _counter_methods, len(labels),
                                  name, description, *labels)

  def get_cell(self, *labels):
    """Retrieves the cell."""
    return CounterCell(super(Counter, self).get_cell(*labels))
