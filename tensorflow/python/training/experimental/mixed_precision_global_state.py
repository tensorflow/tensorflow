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
"""Contains global variables related to mixed precision.

This is not part of mixed_precision.py to avoid a circular dependency.
mixed_precision.py depends on Session, and Session depends on this file.
"""

from tensorflow.python.util.tf_export import tf_export

# Whether the mixed precision graph rewrite has been enabled or not with
# `enable_mixed_precision_graph_rewrite`. Used to turn on auto_mixed_precision
# in ConfigProtos passed to Sessions.
_mixed_precision_graph_rewrite_is_enabled = False


# True if a Session has been created without the mixed precision graph rewrite
# being enabled. Used to give a warning if mixed precision is enabled after a
# Session has already been created.
_non_mixed_precision_session_created = False

# Whether the global tf.keras.mixed_precision.Policy uses mixed precision. Used
# to raise an error message if both a mixed Policy and the graph rewrite are
# used at the same time.
_using_mixed_precision_policy = False


@tf_export('__internal__.train.is_mixed_precision_graph_rewrite_enabled', v1=[])
def is_mixed_precision_graph_rewrite_enabled():
  return _mixed_precision_graph_rewrite_is_enabled


def set_mixed_precision_graph_rewrite_enabled(enabled):
  global _mixed_precision_graph_rewrite_is_enabled
  _mixed_precision_graph_rewrite_is_enabled = enabled


def non_mixed_precision_session_created():
  return _non_mixed_precision_session_created


def set_non_mixed_precision_session_created(created):
  global _non_mixed_precision_session_created
  _non_mixed_precision_session_created = created


def is_using_mixed_precision_policy():
  return _using_mixed_precision_policy


@tf_export('__internal__.train.set_using_mixed_precision_policy', v1=[])
def set_using_mixed_precision_policy(is_using):
  global _using_mixed_precision_policy
  _using_mixed_precision_policy = is_using
