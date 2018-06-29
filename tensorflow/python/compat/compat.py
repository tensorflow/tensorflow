# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for API compatibility between TensorFlow release versions.

See
@{$guide/version_compat#backward_and_partial_forward_compatibility}
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime

_FORWARD_COMPATIBILITY_HORIZON = datetime.date(2018, 8, 1)


def forward_compatible(year, month, day):
  """Return true if the forward compatibility window has expired.

  Forward-compatibility refers to scenarios where the producer of a TensorFlow
  model (a GraphDef or SavedModel) is compiled against a version of the
  TensorFlow library newer than what the consumer was compiled against. The
  "producer" is typically a Python program that constructs and trains a model
  while the "consumer" is typically another program that loads and serves the
  model.

  TensorFlow has been supporting a 3 week forward-compatibility window for
  programs compiled from source at HEAD.

  For example, consider the case where a new operation `MyNewAwesomeAdd` is
  created with the intent of replacing the implementation of an existing Python
  wrapper - `tf.add`.  The Python wrapper implementation should change from
  something like:

  ```python
  def add(inputs, name=None):
    return gen_math_ops.add(inputs, name)
  ```

  to:

  ```python
  from tensorflow.python.compat import compat

  def add(inputs, name=None):
    if compat.forward_compatible(year, month, day):
      # Can use the awesome new implementation.
      return gen_math_ops.my_new_awesome_add(inputs, name)
    # To maintain forward compatibiltiy, use the old implementation.
    return gen_math_ops.add(inputs, name)
  ```

  Where `year`, `month`, and `day` specify the date beyond which binaries
  that consume a model are expected to have been updated to include the
  new operations. This date is typically at least 3 weeks beyond the date
  the code that adds the new operation is committed.

  Args:
    year:  A year (e.g., 2018).
    month: A month (1 <= month <= 12) in year.
    day:   A day (1 <= day <= 31, or 30, or 29, or 28) in month.

  Returns:
    True if the caller can expect that serialized TensorFlow graphs produced
    can be consumed by programs that are compiled with the TensorFlow library
    source code after (year, month, day).
  """
  return _FORWARD_COMPATIBILITY_HORIZON > datetime.date(year, month, day)
