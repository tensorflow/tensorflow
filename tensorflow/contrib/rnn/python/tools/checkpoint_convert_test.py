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
"""Unit tests for checkpoint converter."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import tempfile

from tensorflow.contrib.rnn.python.tools import checkpoint_convert
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib


class CheckpointConvertTest(test.TestCase):

  def setUp(self):
    self._old_ckpt_path = tempfile.mktemp()
    self._new_ckpt_path = tempfile.mktemp()
    ops.reset_default_graph()

  def tearDown(self):
    for file_name in glob.glob(self._old_ckpt_path + "*"):
      os.remove(file_name)
    for file_name in glob.glob(self._new_ckpt_path + "*"):
      os.remove(file_name)

  def testReplacementDictsContainUniqueAndNonEmptyVariableNames(self):
    for old_name in checkpoint_convert._RNN_NAME_REPLACEMENTS:
      new_name = checkpoint_convert._RNN_NAME_REPLACEMENTS[old_name]
      self.assertTrue(old_name)
      self.assertTrue(new_name)
      self.assertNotEqual(old_name, new_name)
    for old_name in checkpoint_convert._RNN_SHARDED_NAME_REPLACEMENTS:
      new_name = checkpoint_convert._RNN_SHARDED_NAME_REPLACEMENTS[old_name]
      self.assertTrue(old_name)
      self.assertTrue(new_name)
      self.assertNotEqual(old_name, new_name)

  def testConversionFromV2WithConvertedVariableNamesSucceeds(self):
    variables.Variable(10.0, name="a")
    for old_name in checkpoint_convert._RNN_NAME_REPLACEMENTS:
      variables.Variable(20.0, name=old_name)
    with session.Session() as sess:
      saver = saver_lib.Saver()
      sess.run(variables.global_variables_initializer())
      saver.save(sess, self._old_ckpt_path)

    new_var_map, conversion_map = checkpoint_convert.convert_names(
        self._old_ckpt_path, self._new_ckpt_path)
    self.assertTrue(glob.glob(self._new_ckpt_path + "*"))
    self.assertItemsEqual(
        ["a"] + list(checkpoint_convert._RNN_NAME_REPLACEMENTS.values()),
        new_var_map.keys())
    self.assertEqual(checkpoint_convert._RNN_NAME_REPLACEMENTS, conversion_map)

  def testConversionFromV2WithoutConvertedVariableNamesSucceeds(self):
    variables.Variable(10.0, name="a")
    with session.Session() as sess:
      saver = saver_lib.Saver()
      sess.run(variables.global_variables_initializer())
      saver.save(sess, self._old_ckpt_path)

    new_var_map, conversion_map = checkpoint_convert.convert_names(
        self._old_ckpt_path, self._new_ckpt_path)
    self.assertItemsEqual(["a"], new_var_map.keys())
    self.assertFalse(conversion_map)

  def testConversionToV1Succeeds(self):
    variables.Variable(10.0, name="a")
    variables.Variable(
        20.0, name=list(checkpoint_convert._RNN_NAME_REPLACEMENTS.keys())[-1])

    with session.Session() as sess:
      saver = saver_lib.Saver()
      sess.run(variables.global_variables_initializer())
      saver.save(sess, self._old_ckpt_path)

    new_var_map, conversion_map = checkpoint_convert.convert_names(
        self._old_ckpt_path, self._new_ckpt_path, write_v1_checkpoint=True)
    self.assertItemsEqual(
        ["a", list(checkpoint_convert._RNN_NAME_REPLACEMENTS.values())[-1]],
        new_var_map.keys())
    self.assertEqual(
        {list(checkpoint_convert._RNN_NAME_REPLACEMENTS.keys())[-1]:
         list(checkpoint_convert._RNN_NAME_REPLACEMENTS.values())[-1]},
        conversion_map)


if __name__ == "__main__":
  test.main()
