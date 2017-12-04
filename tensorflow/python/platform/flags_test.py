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
"""Sanity tests for tf.flags."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from absl import flags as absl_flags

from tensorflow.python.platform import flags


flags.DEFINE_string(
    flag_name='old_string', default_value='default', docstring='docstring')
flags.DEFINE_string(
    name='new_string', default='default', help='docstring')
flags.DEFINE_integer(
    flag_name='old_integer', default_value=1, docstring='docstring')
flags.DEFINE_integer(
    name='new_integer', default=1, help='docstring')
flags.DEFINE_float(
    flag_name='old_float', default_value=1.5, docstring='docstring')
flags.DEFINE_float(
    name='new_float', default=1.5, help='docstring')
flags.DEFINE_bool(
    flag_name='old_bool', default_value=True, docstring='docstring')
flags.DEFINE_bool(
    name='new_bool', default=True, help='docstring')
flags.DEFINE_boolean(
    flag_name='old_boolean', default_value=False, docstring='docstring')
flags.DEFINE_boolean(
    name='new_boolean', default=False, help='docstring')


class FlagsTest(unittest.TestCase):

  def test_global_flags_object(self):
    self.assertIs(flags.FLAGS, absl_flags.FLAGS)

  def test_keyword_arguments(self):
    test_cases = (
        ('old_string', 'default'),
        ('new_string', 'default'),
        ('old_integer', 1),
        ('new_integer', 1),
        ('old_float', 1.5),
        ('new_float', 1.5),
        ('old_bool', True),
        ('new_bool', True),
        ('old_boolean', False),
        ('new_boolean', False),
    )
    for flag_name, default_value in test_cases:
      self.assertEqual(default_value, absl_flags.FLAGS[flag_name].default)
      self.assertEqual('docstring', absl_flags.FLAGS[flag_name].help)


if __name__ == '__main__':
  unittest.main()
