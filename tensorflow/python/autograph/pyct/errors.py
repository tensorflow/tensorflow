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
"""Code transformation exceptions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class AutoGraphError(Exception):
  pass


class InternalError(AutoGraphError):

  def __init__(self, message, original_exc):
    super(InternalError, self).__init__()
    self.message = message
    self.original_exc = original_exc

  def __str__(self):
    return '{} during {}: {}'.format(
        type(self.original_exc).__name__, self.message, self.original_exc)


