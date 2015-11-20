# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Imports unittest as a replacement for testing.pybase.googletest."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import itertools
import os
import tempfile

# pylint: disable=wildcard-import
from unittest import *


unittest_main = main


# pylint: disable=invalid-name
# pylint: disable=undefined-variable
def main(*args, **kwargs):
  """Delegate to unittest.main after redefining testLoader."""
  if 'TEST_SHARD_STATUS_FILE' in os.environ:
    try:
      f = None
      try:
        f = open(os.environ['TEST_SHARD_STATUS_FILE'], 'w')
        f.write('')
      except IOError:
        sys.stderr.write('Error opening TEST_SHARD_STATUS_FILE (%s). Exiting.'
                         % os.environ['TEST_SHARD_STATUS_FILE'])
        sys.exit(1)
    finally:
      if f is not None: f.close()

  if ('TEST_TOTAL_SHARDS' not in os.environ or
      'TEST_SHARD_INDEX' not in os.environ):
    return unittest_main(*args, **kwargs)

  total_shards = int(os.environ['TEST_TOTAL_SHARDS'])
  shard_index = int(os.environ['TEST_SHARD_INDEX'])
  base_loader = TestLoader()

  delegate_get_names = base_loader.getTestCaseNames
  bucket_iterator = itertools.cycle(range(total_shards))

  def getShardedTestCaseNames(testCaseClass):
    filtered_names = []
    for testcase in sorted(delegate_get_names(testCaseClass)):
      bucket = next(bucket_iterator)
      if bucket == shard_index:
        filtered_names.append(testcase)
    return filtered_names

  # Override getTestCaseNames
  base_loader.getTestCaseNames = getShardedTestCaseNames

  kwargs['testLoader'] = base_loader
  unittest_main(*args, **kwargs)


def GetTempDir():
  first_frame = inspect.stack()[-1][0]
  temp_dir = os.path.join(
      tempfile.gettempdir(), os.path.basename(inspect.getfile(first_frame)))
  temp_dir = temp_dir.rstrip('.py')
  if not os.path.isdir(temp_dir):
    os.mkdir(temp_dir, 0o755)
  return temp_dir


def StatefulSessionAvailable():
  return False
