# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Miscellaneous utilities that don't fit anywhere else."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import sys
import types

import six


class BasicRef(object):
  """This shim emulates the nonlocal keyword in Py2-compatible source."""

  def __init__(self, init_value):
    self.value = init_value


def deprecated_py2_support(module_name):
  """Swaps calling module with a Py2-specific implementation. Noop in Py3."""
  if six.PY2:
    legacy_module = importlib.import_module(module_name + '_deprecated_py2')
    current_module = sys.modules[module_name]
    for name, target_val in legacy_module.__dict__.items():
      if isinstance(target_val, types.FunctionType):
        replacement = types.FunctionType(
            target_val.__code__, current_module.__dict__, target_val.__name__,
            target_val.__defaults__, target_val.__closure__)
        current_module.__dict__[name] = replacement
