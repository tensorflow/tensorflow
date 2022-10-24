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
import sys
import unittest

from absl import flags as absl_flags

from tensorflow.python.platform import flags
from tensorflow.python.platform import test


flags.DEFINE_string(  # pylint: disable=no-value-for-parameter
    flag_name='old_string',
    default_value='default',
    docstring='docstring')
flags.DEFINE_string(
    name='new_string', default='default', help='docstring')
flags.DEFINE_integer(  # pylint: disable=no-value-for-parameter
    flag_name='old_integer',
    default_value=1,
    docstring='docstring')
flags.DEFINE_integer(
    name='new_integer', default=1, help='docstring')
flags.DEFINE_float(  # pylint: disable=no-value-for-parameter
    flag_name='old_float',
    default_value=1.5,
    docstring='docstring')
flags.DEFINE_float(
    name='new_float', default=1.5, help='docstring')
flags.DEFINE_bool(  # pylint: disable=no-value-for-parameter
    flag_name='old_bool',
    default_value=True,
    docstring='docstring')
flags.DEFINE_bool(
    name='new_bool', default=True, help='docstring')
flags.DEFINE_boolean(  # pylint: disable=no-value-for-parameter
    flag_name='old_boolean',
    default_value=False,
    docstring='docstring')
flags.DEFINE_boolean(
    name='new_boolean', default=False, help='docstring')


class FlagsTest(unittest.TestCase):

  def setUp(self):
    self.original_flags = flags.FlagValues()
    self.wrapped_flags = flags._FlagValuesWrapper(self.original_flags)
    flags.DEFINE_string(
        'test', 'default', 'test flag', flag_values=self.wrapped_flags)

  def test_attribute_overrides(self):
    # Test that methods defined in absl.flags.FlagValues are the same as the
    # wrapped ones.
    self.assertEqual(flags.FLAGS.is_parsed, absl_flags.FLAGS.is_parsed)

  def test_getattr(self):
    self.assertFalse(self.wrapped_flags.is_parsed())
    with test.mock.patch.object(sys, 'argv', new=['program', '--test=new']):
      self.assertEqual('new', self.wrapped_flags.test)
    self.assertTrue(self.wrapped_flags.is_parsed())

  def test_setattr(self):
    self.assertEqual('default', self.wrapped_flags.test)
    self.wrapped_flags.test = 'new'
    self.assertEqual('new', self.wrapped_flags.test)

  def test_delattr(self):
    del self.wrapped_flags.test
    self.assertNotIn('test', self.wrapped_flags)
    with self.assertRaises(AttributeError):
      _ = self.wrapped_flags.test

  def test_dir(self):
    self.assertEqual(['test'], dir(self.wrapped_flags))

  def test_getitem(self):
    self.assertIs(self.original_flags['test'], self.wrapped_flags['test'])

  def test_setitem(self):
    flag = flags.Flag(flags.ArgumentParser(), flags.ArgumentSerializer(),
                      'fruit', 'apple', 'the fruit type')
    self.wrapped_flags['fruit'] = flag
    self.assertIs(self.original_flags['fruit'], self.wrapped_flags['fruit'])
    self.assertEqual('apple', self.wrapped_flags.fruit)

  def test_len(self):
    self.assertEqual(1, len(self.wrapped_flags))

  def test_iter(self):
    self.assertEqual(['test'], list(self.wrapped_flags))

  def test_str(self):
    self.assertEqual(str(self.wrapped_flags), str(self.original_flags))

  def test_call(self):
    self.wrapped_flags(['program', '--test=new'])
    self.assertEqual('new', self.wrapped_flags.test)

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
