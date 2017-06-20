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
"""Tests for learn.utils.gc."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.utils import gc
from tensorflow.python.framework import test_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.util import compat


def _create_parser(base_dir):
  # create a simple parser that pulls the export_version from the directory.
  def parser(path):
    match = re.match("^" + compat.as_str_any(base_dir) + "/(\\d+)$",
                     compat.as_str_any(path.path))
    if not match:
      return None
    return path._replace(export_version=int(match.group(1)))

  return parser


class GcTest(test_util.TensorFlowTestCase):

  def testLargestExportVersions(self):
    paths = [gc.Path("/foo", 8), gc.Path("/foo", 9), gc.Path("/foo", 10)]
    newest = gc.largest_export_versions(2)
    n = newest(paths)
    self.assertEquals(n, [gc.Path("/foo", 9), gc.Path("/foo", 10)])

  def testLargestExportVersionsDoesNotDeleteZeroFolder(self):
    paths = [gc.Path("/foo", 0), gc.Path("/foo", 3)]
    newest = gc.largest_export_versions(2)
    n = newest(paths)
    self.assertEquals(n, [gc.Path("/foo", 0), gc.Path("/foo", 3)])

  def testModExportVersion(self):
    paths = [
        gc.Path("/foo", 4), gc.Path("/foo", 5), gc.Path("/foo", 6),
        gc.Path("/foo", 9)
    ]
    mod = gc.mod_export_version(2)
    self.assertEquals(mod(paths), [gc.Path("/foo", 4), gc.Path("/foo", 6)])
    mod = gc.mod_export_version(3)
    self.assertEquals(mod(paths), [gc.Path("/foo", 6), gc.Path("/foo", 9)])

  def testOneOfEveryNExportVersions(self):
    paths = [
        gc.Path("/foo", 0), gc.Path("/foo", 1), gc.Path("/foo", 3),
        gc.Path("/foo", 5), gc.Path("/foo", 6), gc.Path("/foo", 7),
        gc.Path("/foo", 8), gc.Path("/foo", 33)
    ]
    one_of = gc.one_of_every_n_export_versions(3)
    self.assertEquals(
        one_of(paths), [
            gc.Path("/foo", 3), gc.Path("/foo", 6), gc.Path("/foo", 8),
            gc.Path("/foo", 33)
        ])

  def testOneOfEveryNExportVersionsZero(self):
    # Zero is a special case since it gets rolled into the first interval.
    # Test that here.
    paths = [gc.Path("/foo", 0), gc.Path("/foo", 4), gc.Path("/foo", 5)]
    one_of = gc.one_of_every_n_export_versions(3)
    self.assertEquals(one_of(paths), [gc.Path("/foo", 0), gc.Path("/foo", 5)])

  def testUnion(self):
    paths = []
    for i in xrange(10):
      paths.append(gc.Path("/foo", i))
    f = gc.union(gc.largest_export_versions(3), gc.mod_export_version(3))
    self.assertEquals(
        f(paths), [
            gc.Path("/foo", 0), gc.Path("/foo", 3), gc.Path("/foo", 6),
            gc.Path("/foo", 7), gc.Path("/foo", 8), gc.Path("/foo", 9)
        ])

  def testNegation(self):
    paths = [
        gc.Path("/foo", 4), gc.Path("/foo", 5), gc.Path("/foo", 6),
        gc.Path("/foo", 9)
    ]
    mod = gc.negation(gc.mod_export_version(2))
    self.assertEquals(mod(paths), [gc.Path("/foo", 5), gc.Path("/foo", 9)])
    mod = gc.negation(gc.mod_export_version(3))
    self.assertEquals(mod(paths), [gc.Path("/foo", 4), gc.Path("/foo", 5)])

  def testPathsWithParse(self):
    base_dir = os.path.join(test.get_temp_dir(), "paths_parse")
    self.assertFalse(gfile.Exists(base_dir))
    for p in xrange(3):
      gfile.MakeDirs(os.path.join(base_dir, "%d" % p))
    # add a base_directory to ignore
    gfile.MakeDirs(os.path.join(base_dir, "ignore"))

    self.assertEquals(
        gc.get_paths(base_dir, _create_parser(base_dir)),
        [
            gc.Path(os.path.join(base_dir, "0"), 0),
            gc.Path(os.path.join(base_dir, "1"), 1),
            gc.Path(os.path.join(base_dir, "2"), 2)
        ])

  def testMixedStrTypes(self):
    temp_dir = compat.as_bytes(test.get_temp_dir())

    for sub_dir in ['str', b'bytes', u'unicode']:
      base_dir = os.path.join(
          (temp_dir if isinstance(sub_dir, bytes) else temp_dir.decode()),
          sub_dir)
      self.assertFalse(gfile.Exists(base_dir))
      gfile.MakeDirs(os.path.join(compat.as_str_any(base_dir), "42"))
      gc.get_paths(base_dir, _create_parser(base_dir))


if __name__ == "__main__":
  test.main()
