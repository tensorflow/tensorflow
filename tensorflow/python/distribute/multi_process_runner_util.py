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
"""Util for multi-process runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.python.framework import errors_impl


@contextlib.contextmanager
def try_run_and_except_connection_error(test_obj):
  """Context manager to skip cases not considered failures by the tests."""
  # TODO(b/142074107): Remove this try-except once within-loop fault-tolerance
  # is supported. This is temporarily needed to avoid test flakiness.
  try:
    yield
  except errors_impl.UnavailableError as e:
    if ('Connection reset by peer' in str(e) or 'Socket closed' in str(e) or
        'failed to connect to all addresses' in str(e)):
      test_obj.skipTest(
          'Skipping connection error between processes: {}'.format(str(e)))
    else:
      raise
