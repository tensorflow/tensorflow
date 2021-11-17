# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Global configuration support."""

import enum


# TODO(mdan): For better performance, allow each rule to take a set names.


class Rule(object):
  """Base class for conversion rules."""

  def __init__(self, module_prefix):
    self._prefix = module_prefix

  def matches(self, module_name):
    return (module_name.startswith(self._prefix + '.') or
            module_name == self._prefix)


class Action(enum.Enum):
  NONE = 0
  CONVERT = 1
  DO_NOT_CONVERT = 2


class DoNotConvert(Rule):
  """Indicates that this module should be not converted."""

  def __str__(self):
    return 'DoNotConvert rule for {}'.format(self._prefix)

  def get_action(self, module):
    if self.matches(module.__name__):
      return Action.DO_NOT_CONVERT
    return Action.NONE


class Convert(Rule):
  """Indicates that this module should be converted."""

  def __str__(self):
    return 'Convert rule for {}'.format(self._prefix)

  def get_action(self, module):
    if self.matches(module.__name__):
      return Action.CONVERT
    return Action.NONE
