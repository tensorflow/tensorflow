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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime

from tensorflow.python import tf2
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import variable_scope

from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export

_FORWARD_COMPATIBILITY_HORIZON = datetime.date(2018, 12, 20)


@tf_export("compat.forward_compatible")
def forward_compatible(year, month, day):
  """Return true if the forward compatibility window has expired.

  See [Version
  compatibility](https://tensorflow.org/guide/version_compat#backward_forward).

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


@tf_export("compat.forward_compatibility_horizon")
@tf_contextlib.contextmanager
def forward_compatibility_horizon(year, month, day):
  """Context manager for testing forward compatibility of generated graphs.

  See [Version
  compatibility](https://tensorflow.org/guide/version_compat#backward_forward).

  To ensure forward compatibility of generated graphs (see `forward_compatible`)
  with older binaries, new features can be gated with:

  ```python
  if compat.forward_compatible(year=2018, month=08, date=01):
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

  Args :
    year:  A year (e.g. 2018).
    month: A month (1 <= month <= 12) in year.
    day:   A day (1 <= day <= 31, or 30, or 29, or 28) in month.

  Yields:
    Nothing.
  """
  global _FORWARD_COMPATIBILITY_HORIZON
  try:
    old_compat_date = _FORWARD_COMPATIBILITY_HORIZON
    _FORWARD_COMPATIBILITY_HORIZON = datetime.date(year, month, day)
    yield
  finally:
    _FORWARD_COMPATIBILITY_HORIZON = old_compat_date


@tf_export(v1=["enable_v2_behavior"])
def enable_v2_behavior():
  """Enables TensorFlow 2.x behaviors.

  This function can be called at the beginning of the program (before `Tensors`,
  `Graphs` or other structures have been created, and before devices have been
  initialized. It switches all global behaviors that are different between
  TensorFlow 1.x and 2.x to behave as intended for 2.x.

  This function is called in the main TensorFlow `__init__.py` file, user should
  not need to call it, except during complex migrations.
  """
  tf2.enable()  # Switches TensorArrayV2 and control flow V2
  ops.enable_eager_execution()
  tensor_shape.enable_v2_tensorshape()  # Also switched by tf2
  variable_scope.enable_resource_variables()


@tf_export(v1=["disable_v2_behavior"])
def disable_v2_behavior():
  """Disables TensorFlow 2.x behaviors.

  This function can be called at the beginning of the program (before `Tensors`,
  `Graphs` or other structures have been created, and before devices have been
  initialized. It switches all global behaviors that are different between
  TensorFlow 1.x and 2.x to behave as intended for 1.x.

  User can call this function to disable 2.x behavior during complex migrations.
  """
  tf2.disable()  # Switches TensorArrayV2 and control flow V2
  ops.disable_eager_execution()
  tensor_shape.disable_v2_tensorshape()  # Also switched by tf2
  variable_scope.disable_resource_variables()


