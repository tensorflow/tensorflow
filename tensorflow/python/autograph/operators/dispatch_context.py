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
"""Structures that allow uniform control over the dispatch process."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


# TODO(mdan): This is where macro override controls fit.


class DispatchContext(collections.namedtuple(
    'DispatchContext',
    ('options',))):
  """Allows passing additional parameters to the specific implementations.

  Attributes:
    options: Optional dict of extra arguments that may be required by specific
      implementations.
  """

  def option(self, name):
    return self.options[name]


NO_CTX = DispatchContext(options={})
