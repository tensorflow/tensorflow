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

import collections
import functools
import time

from tensorflow.core.framework import summary_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import c_api_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export

_MetricMethod = collections.namedtuple('MetricMethod', 'create delete get_cell')
_counter_methods = [
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewCounter0,
        delete=pywrap_tfe.TFE_MonitoringDeleteCounter0,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellCounter0),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewCounter1,
        delete=pywrap_tfe.TFE_MonitoringDeleteCounter1,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellCounter1),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewCounter2,
        delete=pywrap_tfe.TFE_MonitoringDeleteCounter2,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellCounter2),
]
_int_gauge_methods = [
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewIntGauge0,
        delete=pywrap_tfe.TFE_MonitoringDeleteIntGauge0,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellIntGauge0),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewIntGauge1,
        delete=pywrap_tfe.TFE_MonitoringDeleteIntGauge1,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellIntGauge1),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewIntGauge2,
        delete=pywrap_tfe.TFE_MonitoringDeleteIntGauge2,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellIntGauge2),
]
_string_gauge_methods = [
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewStringGauge0,
        delete=pywrap_tfe.TFE_MonitoringDeleteStringGauge0,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellStringGauge0),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewStringGauge1,
        delete=pywrap_tfe.TFE_MonitoringDeleteStringGauge1,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellStringGauge1),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewStringGauge2,
        delete=pywrap_tfe.TFE_MonitoringDeleteStringGauge2,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellStringGauge2),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewStringGauge3,
        delete=pywrap_tfe.TFE_MonitoringDeleteStringGauge3,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellStringGauge3),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewStringGauge4,
        delete=pywrap_tfe.TFE_MonitoringDeleteStringGauge4,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellStringGauge4),
]
_bool_gauge_methods = [
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewBoolGauge0,
        delete=pywrap_tfe.TFE_MonitoringDeleteBoolGauge0,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellBoolGauge0),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewBoolGauge1,
        delete=pywrap_tfe.TFE_MonitoringDeleteBoolGauge1,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellBoolGauge1),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewBoolGauge2,
        delete=pywrap_tfe.TFE_MonitoringDeleteBoolGauge2,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellBoolGauge2),
]
_sampler_methods = [
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewSampler0,
        delete=pywrap_tfe.TFE_MonitoringDeleteSampler0,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellSampler0),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewSampler1,
        delete=pywrap_tfe.TFE_MonitoringDeleteSampler1,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellSampler1),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewSampler2,
        delete=pywrap_tfe.TFE_MonitoringDeleteSampler2,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellSampler2),
]


class Metric(object):
  """The base class of metric."""

  __slots__ = ["_metric", "_metric_name", "_metric_methods", "_label_length"]

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
    try:
      deleter = self._metric_methods[self._label_length].delete
      metric = self._metric
    except AttributeError:
      return

    if deleter is not None:
      deleter(metric)

  def get_cell(self, *labels):
    """Retrieves the cell."""
    if len(labels) != self._label_length:
      raise ValueError('The {} expects taking {} labels'.format(
          self._metric_name, self._label_length))
    return self._metric_methods[self._label_length].get_cell(
        self._metric, *labels)


class CounterCell(object):
  """CounterCell stores each value of a Counter."""

  __slots__ = ["_cell"]

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
    pywrap_tfe.TFE_MonitoringCounterCellIncrementBy(self._cell, value)

  def value(self):
    """Retrieves the current value."""
    return pywrap_tfe.TFE_MonitoringCounterCellValue(self._cell)


class Counter(Metric):
  """A stateful class for updating a cumulative integer metric.

  This class encapsulates a set of values (or a single value for a label-less
  metric). Each value is identified by a tuple of labels. The class allows the
  user to increment each value.
  """

  __slots__ = []

  def __init__(self, name, description, *labels):
    """Creates a new Counter.

    Args:
      name: name of the new metric.
      description: description of the new metric.
      *labels: The label list of the new metric.
    """
    super(Counter, self).__init__('Counter', _counter_methods, len(labels),
                                  name, description, *labels)

  def get_cell(self, *labels):
    """Retrieves the cell."""
    return CounterCell(super(Counter, self).get_cell(*labels))


class IntGaugeCell(object):
  """A single integer value stored in an `IntGauge`."""

  __slots__ = ["_cell"]

  def __init__(self, cell):
    """Creates a new IntGaugeCell.

    Args:
      cell: A c pointer of TFE_MonitoringIntGaugeCell.
    """
    self._cell = cell

  def set(self, value):
    """Atomically set the value.

    Args:
      value: integer value.
    """
    pywrap_tfe.TFE_MonitoringIntGaugeCellSet(self._cell, value)

  def value(self):
    """Retrieves the current value."""
    return pywrap_tfe.TFE_MonitoringIntGaugeCellValue(self._cell)


class IntGauge(Metric):
  """A stateful class for updating a gauge-like integer metric.

  This class encapsulates a set of integer values (or a single value for a
  label-less metric). Each value is identified by a tuple of labels. The class
  allows the user to set each value.
  """

  __slots__ = []

  def __init__(self, name, description, *labels):
    """Creates a new IntGauge.

    Args:
      name: name of the new metric.
      description: description of the new metric.
      *labels: The label list of the new metric.
    """
    super(IntGauge, self).__init__('IntGauge', _int_gauge_methods, len(labels),
                                   name, description, *labels)

  def get_cell(self, *labels):
    """Retrieves the cell."""
    return IntGaugeCell(super(IntGauge, self).get_cell(*labels))


class StringGaugeCell(object):
  """A single string value stored in an `StringGauge`."""

  __slots__ = ["_cell"]

  def __init__(self, cell):
    """Creates a new StringGaugeCell.

    Args:
      cell: A c pointer of TFE_MonitoringStringGaugeCell.
    """
    self._cell = cell

  def set(self, value):
    """Atomically set the value.

    Args:
      value: string value.
    """
    pywrap_tfe.TFE_MonitoringStringGaugeCellSet(self._cell, value)

  def value(self):
    """Retrieves the current value."""
    with c_api_util.tf_buffer() as buffer_:
      pywrap_tfe.TFE_MonitoringStringGaugeCellValue(self._cell, buffer_)
      value = pywrap_tf_session.TF_GetBuffer(buffer_).decode('utf-8')
    return value


class StringGauge(Metric):
  """A stateful class for updating a gauge-like string metric.

  This class encapsulates a set of string values (or a single value for a
  label-less metric). Each value is identified by a tuple of labels. The class
  allows the user to set each value.
  """

  __slots__ = []

  def __init__(self, name, description, *labels):
    """Creates a new StringGauge.

    Args:
      name: name of the new metric.
      description: description of the new metric.
      *labels: The label list of the new metric.
    """
    super(StringGauge, self).__init__('StringGauge', _string_gauge_methods,
                                      len(labels), name, description, *labels)

  def get_cell(self, *labels):
    """Retrieves the cell."""
    return StringGaugeCell(super(StringGauge, self).get_cell(*labels))


class BoolGaugeCell(object):
  """A single boolean value stored in an `BoolGauge`."""

  __slots__ = ["_cell"]

  def __init__(self, cell):
    """Creates a new BoolGaugeCell.

    Args:
      cell: A c pointer of TFE_MonitoringBoolGaugeCell.
    """
    self._cell = cell

  def set(self, value):
    """Atomically set the value.

    Args:
      value: bool value.
    """
    pywrap_tfe.TFE_MonitoringBoolGaugeCellSet(self._cell, value)

  def value(self):
    """Retrieves the current value."""
    return pywrap_tfe.TFE_MonitoringBoolGaugeCellValue(self._cell)


@tf_export("__internal__.monitoring.BoolGauge", v1=[])
class BoolGauge(Metric):
  """A stateful class for updating a gauge-like bool metric.

  This class encapsulates a set of boolean values (or a single value for a
  label-less metric). Each value is identified by a tuple of labels. The class
  allows the user to set each value.
  """

  __slots__ = []

  def __init__(self, name, description, *labels):
    """Creates a new BoolGauge.

    Args:
      name: name of the new metric.
      description: description of the new metric.
      *labels: The label list of the new metric.
    """
    super(BoolGauge, self).__init__('BoolGauge', _bool_gauge_methods,
                                    len(labels), name, description, *labels)

  def get_cell(self, *labels):
    """Retrieves the cell."""
    return BoolGaugeCell(super(BoolGauge, self).get_cell(*labels))


class SamplerCell(object):
  """SamplerCell stores each value of a Sampler."""

  __slots__ = ["_cell"]

  def __init__(self, cell):
    """Creates a new SamplerCell.

    Args:
      cell: A c pointer of TFE_MonitoringSamplerCell.
    """
    self._cell = cell

  def add(self, value):
    """Atomically add a sample.

    Args:
      value: float value.
    """
    pywrap_tfe.TFE_MonitoringSamplerCellAdd(self._cell, value)

  def value(self):
    """Retrieves the current distribution of samples.

    Returns:
      A HistogramProto describing the distribution of samples.
    """
    with c_api_util.tf_buffer() as buffer_:
      pywrap_tfe.TFE_MonitoringSamplerCellValue(self._cell, buffer_)
      proto_data = pywrap_tf_session.TF_GetBuffer(buffer_)
    histogram_proto = summary_pb2.HistogramProto()
    histogram_proto.ParseFromString(compat.as_bytes(proto_data))
    return histogram_proto


class Buckets(object):
  """Bucketing strategies for the samplers."""

  __slots__ = ["buckets"]

  def __init__(self, buckets):
    """Creates a new Buckets.

    Args:
      buckets: A c pointer of TFE_MonitoringBuckets.
    """
    self.buckets = buckets

  def __del__(self):
    pywrap_tfe.TFE_MonitoringDeleteBuckets(self.buckets)


class ExponentialBuckets(Buckets):
  """Exponential bucketing strategy.

  Sets up buckets of the form:
      [-DBL_MAX, ..., scale * growth^i,
       scale * growth_factor^(i + 1), ..., DBL_MAX].
  """

  __slots__ = []

  def __init__(self, scale, growth_factor, bucket_count):
    """Creates a new exponential Buckets.

    Args:
      scale: float
      growth_factor: float
      bucket_count: integer
    """
    super(ExponentialBuckets, self).__init__(
        pywrap_tfe.TFE_MonitoringNewExponentialBuckets(scale, growth_factor,
                                                       bucket_count))


class Sampler(Metric):
  """A stateful class for updating a cumulative histogram metric.

  This class encapsulates a set of histograms (or a single histogram for a
  label-less metric) configured with a list of increasing bucket boundaries.
  Each histogram is identified by a tuple of labels. The class allows the
  user to add a sample to each histogram value.
  """

  __slots__ = []

  def __init__(self, name, buckets, description, *labels):
    """Creates a new Sampler.

    Args:
      name: name of the new metric.
      buckets: bucketing strategy of the new metric.
      description: description of the new metric.
      *labels: The label list of the new metric.
    """
    super(Sampler, self).__init__('Sampler', _sampler_methods, len(labels),
                                  name, buckets.buckets, description, *labels)

  def get_cell(self, *labels):
    """Retrieves the cell."""
    return SamplerCell(super(Sampler, self).get_cell(*labels))


# Keeping track of current MonitoredTimer sections to prevent repetitive
# counting.
MonitoredTimerSections = []


class MonitoredTimer(object):
  """A context manager to measure the walltime and increment a Counter cell."""

  __slots__ = [
      "cell",
      "t",
      "monitored_section_name",
      "_counting",
      "_avoid_repetitive_counting",
  ]

  def __init__(
      self, cell, monitored_section_name=None, avoid_repetitive_counting=False
  ):
    """Creates a new MonitoredTimer.

    Args:
      cell: the cell associated with the time metric that will be inremented.
      monitored_section_name: name of action being monitored here.
      avoid_repetitive_counting: when set to True, if already in a monitored
        timer section with the same monitored_section_name, skip counting.
    """
    self.cell = cell
    self.monitored_section_name = monitored_section_name
    self._avoid_repetitive_counting = avoid_repetitive_counting
    self._counting = True

  def __enter__(self):
    if (
        self._avoid_repetitive_counting
        and self.monitored_section_name
        and self.monitored_section_name in MonitoredTimerSections
    ):
      self._counting = False
      return self

    self.t = time.time()
    if self.monitored_section_name:
      MonitoredTimerSections.append(self.monitored_section_name)

    return self

  def __exit__(self, exception_type, exception_value, traceback):
    del exception_type, exception_value, traceback
    if self._counting:
      micro_seconds = (time.time() - self.t) * 1000000
      self.cell.increase_by(int(micro_seconds))
      if self.monitored_section_name:
        MonitoredTimerSections.remove(self.monitored_section_name)


def monitored_timer(cell):
  """A function decorator for adding MonitoredTimer support.

  Args:
    cell: the cell associated with the time metric that will be inremented.
  Returns:
    A decorator that measure the function runtime and increment the specified
    counter cell.
  """

  def actual_decorator(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      with MonitoredTimer(cell):
        return func(*args, **kwargs)

    return wrapper

  return actual_decorator
