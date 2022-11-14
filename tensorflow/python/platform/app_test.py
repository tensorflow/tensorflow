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

"""Tests for our flags implementation."""
import sys

from tensorflow.python.platform import app, flags, test

FLAGS = flags.FLAGS
flags.DEFINE_boolean("myflag", False, '')


class TestCMDArgs(test.TestCase):
  def testWith3AcceptableArgs(self):
    sys.argv.extend(["--myflag", "--passthrough", "extra"])
    with self.assertRaises(SystemExit) as cm:
      app.run()
    self.assertNotEqual(cm.exception.code, 1)

  def testWith2AcceptableArgs(self):
    sys.argv.extend(["--myflag", "--passthrough"])
    with self.assertRaises(SystemExit) as cm:
      app.run()
    self.assertEqual(cm.exception.code, 2)

  def testWith1AcceptableArg(self):
    sys.argv.extend(["--myflag"])
    with self.assertRaises(SystemExit) as cm:
      app.run()
    self.assertEqual(cm.exception.code, 2)

  def testWithoutPassThrough(self):
    sys.argv.extend(["--myflag", "--test", "extra"])
    with self.assertRaises(SystemExit) as cm:
      app.run()
    self.assertEqual(cm.exception.code, 2)

  def testWithoutExtra(self):
    sys.argv.extend(["--myflag", "--passthrough", "notextra"])
    with self.assertRaises(SystemExit) as cm:
      app.run()
    self.assertEqual(cm.exception.code, 2)

  def testWithoutArgs(self):
    with self.assertRaises(SystemExit) as cm:
      app.run()
    self.assertEqual(cm.exception.code, 2)


if __name__ == '__main__':
  test.main()
