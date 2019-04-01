# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ========================================================================
"""A logger for profiling events."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path

from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary.writer import writer


class ProfileLogger(object):
  """For logging profiling events."""

  def _set_summary_dir(self, model_dir):
    """Sets the summary directory to be model_dir."""
    if model_dir is None:
      self._summary_dir = None
      self._summary_writer = None
      logging.warning('profile_logger: model_dir is None.'
                      'So nowhere to write summaries')
      return
    self._summary_dir = os.path.join(model_dir, 'profile')
    try:
      self._summary_writer = writer.FileWriter(
          logdir=self._summary_dir, filename_suffix='.profile_logger')
      logging.info('profile_logger(): set the summary directory to %s',
                   self._summary_dir)
    except Exception:  # pylint: disable=broad-except
      logging.warning('profile_logger(): failed to create %s',
                      self._summary_dir)
      self._summary_dir = None
      self._summary_writer = None

  def __init__(self, model_dir):
    self._set_summary_dir(model_dir)

  def log_event(self, event, phase):
    """Logs the given event to the summary directory."""

    event_name = 'profile/' + event + '_' + phase
    if self._summary_writer is None:
      logging.warning('profile_logger: cannot log event "%s" '
                      ' because of no summary directory', event_name)
      return

    # For now, we only need the event timestamp. No need to pass any value.
    s = Summary(value=[Summary.Value(tag=event_name,
                                     simple_value=0.0)])
    self._summary_writer.add_summary(s)
    self._summary_writer.flush()
    logging.info('profile_logger: log event "%s"', event_name)
