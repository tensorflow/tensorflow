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

"""Tests for our flags implementation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from tensorflow.python.platform.default import _googletest as googletest

from tensorflow.python.platform.default import _flags as flags


flags.DEFINE_string("string_foo", "default_val", "HelpString")
flags.DEFINE_integer("int_foo", 42, "HelpString")
flags.DEFINE_float("float_foo", 42.0, "HelpString")

flags.DEFINE_boolean("bool_foo", True, "HelpString")
flags.DEFINE_boolean("bool_negation", True, "HelpString")
flags.DEFINE_boolean("bool_a", False, "HelpString")
flags.DEFINE_boolean("bool_c", False, "HelpString")
flags.DEFINE_boolean("bool_d", True, "HelpString")
flags.DEFINE_bool("bool_e", True, "HelpString")

FLAGS = flags.FLAGS

class FlagsTest(googletest.TestCase):

  def testString(self):
    res = FLAGS.string_foo
    self.assertEqual(res, "default_val")
    FLAGS.string_foo = "bar"
    self.assertEqual("bar", FLAGS.string_foo)

  def testBool(self):
    res = FLAGS.bool_foo
    self.assertTrue(res)
    FLAGS.bool_foo = False
    self.assertFalse(FLAGS.bool_foo)

  def testBoolCommandLines(self):
    # Specified on command line with no args, sets to True,
    # even if default is False.
    self.assertEqual(True, FLAGS.bool_a)

    # --no before the flag forces it to False, even if the
    # default is True
    self.assertEqual(False, FLAGS.bool_negation)

    # --bool_flag=True sets to True
    self.assertEqual(True, FLAGS.bool_c)

    # --bool_flag=False sets to False
    self.assertEqual(False, FLAGS.bool_d)

    # --bool_flag=gibberish sets to False
    self.assertEqual(False, FLAGS.bool_e)

  def testInt(self):
    res = FLAGS.int_foo
    self.assertEquals(res, 42)
    FLAGS.int_foo = -1
    self.assertEqual(-1, FLAGS.int_foo)

  def testFloat(self):
    res = FLAGS.float_foo
    self.assertEquals(42.0, res)
    FLAGS.float_foo = -1.0
    self.assertEqual(-1.0, FLAGS.float_foo)


if __name__ == "__main__":
  # Test command lines
  sys.argv.extend(["--bool_a", "--nobool_negation", "--bool_c=True",
                   "--bool_d=False", "--bool_e=gibberish", "--unknown_flag",
                   "and_argument"])

  # googletest.main() tries to interpret the above flags, so use the
  # direct functions instead.
  runner = googletest.TextTestRunner()
  itersuite = googletest.TestLoader().loadTestsFromTestCase(FlagsTest)
  runner.run(itersuite)
