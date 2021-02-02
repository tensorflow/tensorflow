# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Metrics collecting utilities for single client training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from tensorflow.python.eager import monitoring
from tensorflow.python.util import tf_contextlib

enable_metrics = False
_METRICS_MAPPING = {}


def _init():
  """Initialize the metrics mapping."""
  global _METRICS_MAPPING

  # Time in seconds to bucket the distribution of execution time. Range from
  # 0.001s (i.e., 1ms) to 1000s.
  time_buckets = monitoring.ExponentialBuckets(0.001, 10, 6)

  function_tracing_sampler = monitoring.Sampler(
      '/tensorflow/api/ps_strategy/coordinator/function_tracing', time_buckets,
      'Sampler to track the time (in seconds) for tracing functions.')

  closure_execution_sampler = monitoring.Sampler(
      '/tensorflow/api/ps_strategy/coordinator/closure_execution',
      time_buckets,
      'Sampler to track the time (in seconds) for executing closures.')

  remote_value_fetch_sampler = monitoring.Sampler(
      '/tensorflow/api/ps_strategy/coordinator/remote_value_fetch',
      time_buckets,
      'Sampler to track the time (in seconds) for fetching remote_value.')

  _METRICS_MAPPING = {
      'function_tracing': function_tracing_sampler,
      'closure_execution': closure_execution_sampler,
      'remote_value_fetch': remote_value_fetch_sampler
  }


@tf_contextlib.contextmanager
def monitored_timer(metric_name, state_tracker=None):
  """Monitor the execution time and collect it into the specified metric."""
  if not enable_metrics:
    yield
  else:
    if not _METRICS_MAPPING:
      _init()
    start_time = time.time()
    start_state = state_tracker() if state_tracker else None
    yield
    duration_sec = time.time() - start_time
    # If a state_checker is provided, record the metric only if the end state is
    # different from the start state.
    if state_tracker is None or state_tracker() != start_state:
      metric = _METRICS_MAPPING[metric_name]
      metric.get_cell().add(duration_sec)


def get_metric_summary(metric_name):
  """Get summary for the specified metric."""
  metric = _METRICS_MAPPING[metric_name]
  histogram_proto = metric.get_cell().value()
  ret = dict()
  ret['min'] = histogram_proto.min
  ret['max'] = histogram_proto.max
  ret['num'] = histogram_proto.num
  ret['sum'] = histogram_proto.sum
  # TODO(haoyuzhang): consider reporting the distribution in buckets.
  return ret
