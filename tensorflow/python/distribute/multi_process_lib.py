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

import multiprocessing as _multiprocessing
import os
import unittest

from tensorflow.python.platform import test


try:
  multiprocessing = _multiprocessing.get_context('forkserver')
except ValueError:
  # forkserver is not available on Windows.
  multiprocessing = _multiprocessing.get_context('spawn')


class Process(object):
  """A process simulating a worker for testing multi-worker training."""

  def __init__(self, *args, **kwargs):
    del args, kwargs
    raise unittest.SkipTest(
        'TODO(b/150264776): Implement OSS version of `multi_process_lib`')


def test_main():
  """Main function to be called within `__main__` of a test file."""
  os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
  test.main()


def initialized():
  """Returns whether the module is initialized."""
  return True
