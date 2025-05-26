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

See [Version
Compatibility](https://tensorflow.org/guide/version_compat#backward_forward)
"""

import datetime
import os

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export


# This value changes every day with an automatic CL. It can be modified in code
# via `forward_compatibility_horizon()` or with the environment variable
# TF_FORWARD_COMPATIBILITY_DELTA_DAYS, which is added to the compatibility date.
_FORWARD_COMPATIBILITY_HORIZON = datetime.date(2025, 5, 26)
_FORWARD_COMPATIBILITY_DELTA_DAYS_VAR_NAME = "TF_FORWARD_COMPATIBILITY_DELTA_DAYS"
_FORWARD_COMPATIBILITY_DATE_NUMBER = None


def _date_to_date_number(year, month, day):
  return (year << 9) | (month << 5) | day


def _update_forward_compatibility_date_number(date_to_override=None):
  """Update the base date to compare in forward_compatible function."""

  global _FORWARD_COMPATIBILITY_DATE_NUMBER

  if date_to_override:
    date = date_to_override
  else:
    date = _FORWARD_COMPATIBILITY_HORIZON
    delta_days = os.getenv(_FORWARD_COMPATIBILITY_DELTA_DAYS_VAR_NAME)
    if delta_days:
      date += datetime.timedelta(days=int(delta_days))

  if date < _FORWARD_COMPATIBILITY_HORIZON:
    logging.warning("Trying to set the forward compatibility date to the past"
                    " date %s. This will be ignored by TensorFlow." % (date))
    return
  _FORWARD_COMPATIBILITY_DATE_NUMBER = _date_to_date_number(
      date.year, date.month, date.day)


_update_forward_compatibility_date_number()


@tf_export("compat.forward_compatible")
def forward_compatible(year, month, day):
  """Return true if the forward compatibility window has expired.

  See [Version
  compatibility](https://www.tensorflow.org/guide/versions#backward_and_partial_forward_compatibility).

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
    # To maintain forward compatibility, use the old implementation.
    return gen_math_ops.add(inputs, name)
  ```

  Where `year`, `month`, and `day` specify the date beyond which binaries
  that consume a model are expected to have been updated to include the
  new operations. This date is typically at least 3 weeks beyond the date
  the code that adds the new operation is committed.

  Args:
    year:  A year (e.g., 2018). Must be an `int`.
    month: A month (1 <= month <= 12) in year. Must be an `int`.
    day:   A day (1 <= day <= 31, or 30, or 29, or 28) in month. Must be an
      `int`.

  Returns:
    True if the caller can expect that serialized TensorFlow graphs produced
    can be consumed by programs that are compiled with the TensorFlow library
    source code after (year, month, day).
  """
  return _FORWARD_COMPATIBILITY_DATE_NUMBER > _date_to_date_number(
      year, month, day)


@tf_export("compat.forward_compatibility_horizon")
@tf_contextlib.contextmanager
def forward_compatibility_horizon(year, month, day):
  """Context manager for testing forward compatibility of generated graphs.

  See [Version
  compatibility](https://www.tensorflow.org/guide/versions#backward_and_partial_forward_compatibility).

  To ensure forward compatibility of generated graphs (see `forward_compatible`)
  with older binaries, new features can be gated with:

  ```python
  if compat.forward_compatible(year=2018, month=08, day=01):
    generate_graph_with_new_features()
  else:
    generate_graph_so_older_binaries_can_consume_it()
  ```

  However, when adding new features, one may want to unittest it before
  the forward compatibility window expires. This context manager enables
  such tests. For example:

  ```python
  from tensorflow.python.compat import compat

  def testMyNewFeature(self):
    with compat.forward_compatibility_horizon(2018, 08, 02):
       # Test that generate_graph_with_new_features() has an effect
  ```

  Args:
    year:  A year (e.g., 2018). Must be an `int`.
    month: A month (1 <= month <= 12) in year. Must be an `int`.
    day:   A day (1 <= day <= 31, or 30, or 29, or 28) in month. Must be an
      `int`.

  Yields:
    Nothing.
  """
  try:
    _update_forward_compatibility_date_number(datetime.date(year, month, day))
    yield
  finally:
    _update_forward_compatibility_date_number()
