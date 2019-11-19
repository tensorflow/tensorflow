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
"""OSS multi-process library to be implemented."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import unittest


class Process(object):
  """A process simulating a worker for testing multi-worker training."""

  def __init__(self, *args, **kwargs):
    del args, kwargs
    raise unittest.SkipTest(
        'TODO(b/141874796): Implement OSS version of `multi_process_lib`')


def get_user_data():
  """Returns the data commonly shared by parent process and subprocesses."""
  # TODO(b/141874796): Implement OSS version of `multi_process_lib`.
  pass


@contextlib.contextmanager
def context_manager():
  """No-op in OSS. This exists to maintain testing compatibility."""
  yield


def using_context_manager():
  """Whether the context manager is being used."""
  raise unittest.SkipTest(
      'TODO(b/141874796): Implement OSS version of `multi_process_lib`')
