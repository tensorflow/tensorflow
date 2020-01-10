"""Tests for tensorflow.python.ops.io_ops."""
# -*- coding: utf-8 -*-

import tempfile

import tensorflow.python.platform

import tensorflow as tf


class IoOpsTest(tf.test.TestCase):

  def testReadFile(self):
    cases = ['', 'Some contents', 'Неки садржаји на српском']
    for contents in cases:
      temp = tempfile.NamedTemporaryFile(prefix='ReadFileTest')
      open(temp.name, 'wb').write(contents)
      with self.test_session():
        read = tf.read_file(temp.name)
        self.assertEqual([], read.get_shape())
        self.assertEqual(read.eval(), contents)

  def _subset(self, files, indices):
    return set([files[i].name for i in range(len(files)) if i in indices])

  def testMatchingFiles(self):
    cases = ['ABcDEF.GH', 'ABzDEF.GH', 'ABasdfjklDEF.GH', 'AB3DEF.GH',
             'AB4DEF.GH', 'ABDEF.GH', 'XYZ']
    files = [tempfile.NamedTemporaryFile(prefix=c) for c in cases]

    with self.test_session():
      # Test exact match without wildcards.
      for f in files:
        self.assertEqual(tf.matching_files(f.name).eval(), f.name)

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
