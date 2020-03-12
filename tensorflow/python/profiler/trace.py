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
# ==============================================================================
"""Trace allows the profiler to trace Python events."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.profiler.internal import _pywrap_traceme
from tensorflow.python.util.tf_export import tf_export


@tf_export('profiler.experimental.Trace', v1=[])
class Trace(object):
  """Context manager that generates a trace event in the profiler.

  A trace event will start when entering the context, and stop and save the
  result to the profiler when exiting the context. Open TensorBoard Profile tab
  and choose trace viewer to view the trace event in the timeline.

  Trace events are created only when the profiler is enabled. More information
  on how to use the profiler can be found at
  https://tensorflow.org/guide/profiler

  Example usage:
  ```python
  tf.profiler.experimental.start('logdir')
  for step in range(num_steps):
    # Creates a trace event for each training step with the step number.
    with tf.profiler.experimental.Trace("Train", step_num=step):
      train_fn()
  tf.profiler.experimental.stop()
  ```
  """

  def __init__(self, name, **kwargs):
    """Creates a trace event in the profiler.

    Args:
      name: The name of the trace event.
      **kwargs: Keyword arguments added to the trace event.
    """
    if _pywrap_traceme.TraceMe.IsEnabled():
      if kwargs:
        name += '#' + ','.join(key + '=' + str(value)
                               for key, value in six.iteritems(kwargs)) + '#'
      self._traceme = _pywrap_traceme.TraceMe(name)
    else:
      self._traceme = None

  def __enter__(self):
    if self._traceme:
      self._traceme.Enter()

  def __exit__(self, exc_type, exc_val, exc_tb):
    if self._traceme:
      self._traceme.Exit()
