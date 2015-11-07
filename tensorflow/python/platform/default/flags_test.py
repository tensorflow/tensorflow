"""Tests for our flags implementation."""
import sys

from tensorflow.python.platform.default import _googletest as googletest

from tensorflow.python.platform.default import _flags as flags


flags.DEFINE_string("string_foo", "default_val", "HelpString")
flags.DEFINE_boolean("bool_foo", True, "HelpString")
flags.DEFINE_integer("int_foo", 42, "HelpString")
flags.DEFINE_float("float_foo", 42.0, "HelpString")

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

  def testNoBool(self):
    FLAGS.bool_foo = True
    try:
      sys.argv.append("--nobool_foo")
      FLAGS._parse_flags()
      self.assertFalse(FLAGS.bool_foo)
    finally:
      sys.argv.pop()

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
  googletest.main()
