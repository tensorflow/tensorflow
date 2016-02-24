# -*- coding: utf-8 -*-
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

"""Tests for tensorflow.python.ops.io_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import tensorflow as tf


class IoOpsTest(tf.test.TestCase):

  def testReadFile(self):
    cases = ['', 'Some contents', 'Неки садржаји на српском']
    for contents in cases:
      contents = tf.compat.as_bytes(contents)
      temp = tempfile.NamedTemporaryFile(prefix='ReadFileTest')
      open(temp.name, 'wb').write(contents)
      with self.test_session():
        read = tf.read_file(temp.name)
        self.assertEqual([], read.get_shape())
        self.assertEqual(read.eval(), contents)

  def _subset(self, files, indices):
    return set(tf.compat.as_bytes(files[i].name)
               for i in range(len(files)) if i in indices)

  def testMatchingFiles(self):
    cases = ['ABcDEF.GH', 'ABzDEF.GH', 'ABasdfjklDEF.GH', 'AB3DEF.GH',
             'AB4DEF.GH', 'ABDEF.GH', 'XYZ']
    files = [tempfile.NamedTemporaryFile(prefix=c) for c in cases]

    with self.test_session():
      # Test exact match without wildcards.
      for f in files:
        self.assertEqual(tf.matching_files(f.name).eval(),
                         tf.compat.as_bytes(f.name))

      # We will look for files matching "ABxDEF.GH*" where "x" is some wildcard.
      pos = files[0].name.find(cases[0])
      pattern = files[0].name[:pos] + 'AB%sDEF.GH*'

      self.assertEqual(set(tf.matching_files(pattern % 'z').eval()),
                       self._subset(files, [1]))
      self.assertEqual(set(tf.matching_files(pattern % '?').eval()),
                       self._subset(files, [0, 1, 3, 4]))
      self.assertEqual(set(tf.matching_files(pattern % '*').eval()),
                       self._subset(files, [0, 1, 2, 3, 4, 5]))
      self.assertEqual(set(tf.matching_files(pattern % '[cxz]').eval()),
                       self._subset(files, [0, 1]))
      self.assertEqual(set(tf.matching_files(pattern % '[0-9]').eval()),
                       self._subset(files, [3, 4]))


if __name__ == '__main__':
  tf.test.main()
