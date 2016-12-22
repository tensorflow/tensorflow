# -*- coding: utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

import os
import tempfile

from tensorflow.python.ops import io_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class IoOpsTest(test.TestCase):

  def testReadFile(self):
    cases = ['', 'Some contents', 'Неки садржаји на српском']
    for contents in cases:
      contents = compat.as_bytes(contents)
      with tempfile.NamedTemporaryFile(
          prefix='ReadFileTest', dir=self.get_temp_dir(), delete=False) as temp:
        temp.write(contents)
      with self.test_session():
        read = io_ops.read_file(temp.name)
        self.assertEqual([], read.get_shape())
        self.assertEqual(read.eval(), contents)
      os.remove(temp.name)

  def testWriteFile(self):
    cases = ['', 'Some contents']
    for contents in cases:
      contents = compat.as_bytes(contents)
      with tempfile.NamedTemporaryFile(
          prefix='WriteFileTest', dir=self.get_temp_dir(),
          delete=False) as temp:
        pass
      with self.test_session() as sess:
        w = io_ops.write_file(temp.name, contents)
        sess.run(w)
        with open(temp.name, 'rb') as f:
          file_contents = f.read()
        self.assertEqual(file_contents, contents)
      os.remove(temp.name)

  def _subset(self, files, indices):
    return set(
        compat.as_bytes(files[i].name) for i in range(len(files))
        if i in indices)

  def testMatchingFiles(self):
    cases = [
        'ABcDEF.GH', 'ABzDEF.GH', 'ABasdfjklDEF.GH', 'AB3DEF.GH', 'AB4DEF.GH',
        'ABDEF.GH', 'XYZ'
    ]
    files = [
        tempfile.NamedTemporaryFile(
            prefix=c, dir=self.get_temp_dir(), delete=True) for c in cases
    ]

    with self.test_session():
      # Test exact match without wildcards.
      for f in files:
        self.assertEqual(
            io_ops.matching_files(f.name).eval(), compat.as_bytes(f.name))

      # We will look for files matching "ABxDEF.GH*" where "x" is some wildcard.
      pos = files[0].name.find(cases[0])
      pattern = files[0].name[:pos] + 'AB%sDEF.GH*'

      self.assertEqual(
          set(io_ops.matching_files(pattern % 'z').eval()),
          self._subset(files, [1]))
      self.assertEqual(
          set(io_ops.matching_files(pattern % '?').eval()),
          self._subset(files, [0, 1, 3, 4]))
      self.assertEqual(
          set(io_ops.matching_files(pattern % '*').eval()),
          self._subset(files, [0, 1, 2, 3, 4, 5]))
      # NOTE(mrry): Windows uses PathMatchSpec to match file patterns, which
      # does not support the following expressions.
      if os.name != 'nt':
        self.assertEqual(
            set(io_ops.matching_files(pattern % '[cxz]').eval()),
            self._subset(files, [0, 1]))
        self.assertEqual(
            set(io_ops.matching_files(pattern % '[0-9]').eval()),
            self._subset(files, [3, 4]))

    for f in files:
      f.close()


if __name__ == '__main__':
  test.main()
