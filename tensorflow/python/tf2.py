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
"""Tools to help with the TensorFlow 2.0 transition.

This module is meant for TensorFlow internal implementation, not for users of
the TensorFlow library. For that see tf.compat instead.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

_force_enable = None


def enable():
  """Enables v2 behaviors."""
  global _force_enable
  _force_enable = True


def disable():
  """Disables v2 behaviors."""
  global _force_enable
  _force_enable = False


def enabled():
  """Returns True iff TensorFlow 2.0 behavior should be enabled."""
  if _force_enable is None:
    return os.getenv("TF2_BEHAVIOR", "0") != "0"

  return _force_enable
