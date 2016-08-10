# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Test utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import os
import numpy as np
from tensorflow.core.framework import summary_pb2
from tensorflow.python.training import summary_io


def assert_summary(expected_tags, expected_simple_values, summary_proto):
  """Asserts summary contains the specified tags and values.

  Args:
    expected_tags: All tags in summary.
    expected_simple_values: Simply values for some tags.
    summary_proto: Summary to validate.

  Raises:
    ValueError: if expectations are not met.
  """
  actual_tags = set()
  for value in summary_proto.value:
    actual_tags.add(value.tag)
    if value.tag in expected_simple_values:
      expected = expected_simple_values[value.tag]
      actual = value.simple_value
      np.testing.assert_almost_equal(
          actual, expected, decimal=2, err_msg=value.tag)
  expected_tags = set(expected_tags)
  if expected_tags != actual_tags:
    raise ValueError('Expected tags %s, got %s.' % (expected_tags, actual_tags))


def to_summary_proto(summary_str):
  """Create summary based on latest stats.

  Args:
    summary_str: Serialized summary.
  Returns:
    summary_pb2.Summary.
  Raises:
    ValueError: if tensor is not a valid summary tensor.
  """
  summary = summary_pb2.Summary()
  summary.ParseFromString(summary_str)
  return summary


# TODO(ptucker): Move to a non-test package?
def latest_event_file(base_dir):
  """Find latest event file in `base_dir`.

  Args:
    base_dir: Base directory in which TF event flies are stored.
  Returns:
    File path, or `None` if none exists.
  """
  file_paths = glob.glob(os.path.join(base_dir, 'events.*'))
  return sorted(file_paths)[-1] if file_paths else None


def latest_events(base_dir):
  """Parse events from latest event file in base_dir.

  Args:
    base_dir: Base directory in which TF event flies are stored.
  Returns:
    Iterable of event protos.
  Raises:
    ValueError: if no event files exist under base_dir.
  """
  file_path = latest_event_file(base_dir)
  return summary_io.summary_iterator(file_path) if file_path else []


def latest_summaries(base_dir):
  """Parse summary events from latest event file in base_dir.

  Args:
    base_dir: Base directory in which TF event flies are stored.
  Returns:
    List of event protos.
  Raises:
    ValueError: if no event files exist under base_dir.
  """
  return [e for e in latest_events(base_dir) if e.HasField('summary')]


def simple_values_from_events(events, tags):
  """Parse summaries from events with simple_value.

  Args:
    events: List of tensorflow.Event protos.
    tags: List of string event tags corresponding to simple_value summaries.
  Returns:
    dict of tag:value.
  Raises:
   ValueError: if a summary with a specified tag does not contain simple_value.
  """
  step_by_tag = {}
  value_by_tag = {}
  for e in events:
    if e.HasField('summary'):
      for v in e.summary.value:
        tag = v.tag
        if tag in tags:
          if not v.HasField('simple_value'):
            raise ValueError('Summary for %s is not a simple_value.' % tag)
          # The events are mostly sorted in step order, but we explicitly check
          # just in case.
          if tag not in step_by_tag or e.step > step_by_tag[tag]:
            step_by_tag[tag] = e.step
            value_by_tag[tag] = v.simple_value
  return value_by_tag
