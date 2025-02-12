# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Toggle to enable/disable resource variables."""

from tensorflow.python import tf2
from tensorflow.python.eager import monitoring
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


_api_usage_gauge = monitoring.BoolGauge(
    "/tensorflow/api/resource_variables",
    "Whether resource_variables_toggle.enable_resource_variables() is called.")

_DEFAULT_USE_RESOURCE = tf2.enabled()


@tf_export(v1=["enable_resource_variables"])
def enable_resource_variables() -> None:
  """Creates resource variables by default.

  Resource variables are improved versions of TensorFlow variables with a
  well-defined memory model. Accessing a resource variable reads its value, and
  all ops which access a specific read value of the variable are guaranteed to
  see the same value for that tensor. Writes which happen after a read (by
  having a control or data dependency on the read) are guaranteed not to affect
  the value of the read tensor, and similarly writes which happen before a read
  are guaranteed to affect the value. No guarantees are made about unordered
  read/write pairs.

  Calling tf.enable_resource_variables() lets you opt-in to this TensorFlow 2.0
  feature.
  """
  global _DEFAULT_USE_RESOURCE
  _DEFAULT_USE_RESOURCE = True
  logging.vlog(1, "Enabling resource variables")
  _api_usage_gauge.get_cell().set(True)


@deprecation.deprecated(
    None, "non-resource variables are not supported in the long term")
@tf_export(v1=["disable_resource_variables"])
def disable_resource_variables() -> None:
  """Opts out of resource variables.

  If your code needs tf.disable_resource_variables() to be called to work
  properly please file a bug.
  """
  global _DEFAULT_USE_RESOURCE
  _DEFAULT_USE_RESOURCE = False
  logging.vlog(1, "Disabling resource variables")
  _api_usage_gauge.get_cell().set(False)


@tf_export(v1=["resource_variables_enabled"])
def resource_variables_enabled() -> bool:
  """Returns `True` if resource variables are enabled.

  Resource variables are improved versions of TensorFlow variables with a
  well-defined memory model. Accessing a resource variable reads its value, and
  all ops which access a specific read value of the variable are guaranteed to
  see the same value for that tensor. Writes which happen after a read (by
  having a control or data dependency on the read) are guaranteed not to affect
  the value of the read tensor, and similarly writes which happen before a read
  are guaranteed to affect the value. No guarantees are made about unordered
  read/write pairs.

  Calling tf.enable_resource_variables() lets you opt-in to this TensorFlow 2.0
  feature.
  """
  return _DEFAULT_USE_RESOURCE
