# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Immutable mapping."""

import collections.abc


# WARNING: this class is used internally by extension types (tf.ExtensionType),
# and may be deleted if/when extension types transition to a different encoding
# in the future.
class ImmutableDict(collections.abc.Mapping):
  """Immutable `Mapping`."""
  # Note: keys, items, values, get, __eq__, and __ne__ are implemented by
  # the `Mapping` base class.

  def __init__(self, *args, **kwargs):
    self._dict = dict(*args, **kwargs)

  def __getitem__(self, key):
    return self._dict[key]

  def __contains__(self, key):
    return key in self._dict

  def __iter__(self):
    return iter(self._dict)

  def __len__(self):
    return len(self._dict)

  def __repr__(self):
    return f'ImmutableDict({self._dict})'

  # This suppresses a warning that tf.nest would otherwise generate.
  __supported_by_tf_nest__ = True
