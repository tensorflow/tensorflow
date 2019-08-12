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
"""Thread-local context managers for AutoGraph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

import enum


stacks = threading.local()


def _control_ctx():
  if not hasattr(stacks, 'control_status'):
    stacks.control_status = [_default_control_status_ctx()]
  return stacks.control_status


def control_status_ctx():
  ret = _control_ctx()[-1]
  return ret


class Status(enum.Enum):
  UNSPECIFIED = 0
  ENABLED = 1
  DISABLED = 2


class ControlStatusCtx(object):
  """A context that tracks whether autograph is enabled by the user."""

  def __init__(self, status, options=None):
    self.status = status
    self.options = options

  def __enter__(self):
    _control_ctx().append(self)
    return self

  def __repr__(self):
    return '{}[status={}, options={}]'.format(
        self.__class__.__name__, self.status, self.options)

  def __exit__(self, unused_type, unused_value, unused_traceback):
    assert _control_ctx()[-1] is self
    _control_ctx().pop()


def _default_control_status_ctx():
  return ControlStatusCtx(status=Status.UNSPECIFIED)
