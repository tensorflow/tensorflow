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
"""Imperative mode for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import *  # pylint: disable=wildcard-import
from tensorflow.contrib.imperative import imperative_mode


class _InteractiveMode(object):
  """Imperative mode suitable for interactive execution.

  This module has a global _InteractiveMode object that enables
  writing code as follows:

  ```python
  import tensorflow.contrib.imperative as tf
  print(tf.constant(42))
  ```
  """

  def __init__(self, target=None):
    if not target:
      target = train.Server.create_local_server().target
    self.target = target
    self.imperative_mode = imperative_mode.ImperativeMode(self.target)
    self.imperative_mode.__enter__()

  def new_step(self):
    return self.imperative_mode.new_step()


_default_interactive_mode = _InteractiveMode()


def new_step():
  return _default_interactive_mode.new_step()
