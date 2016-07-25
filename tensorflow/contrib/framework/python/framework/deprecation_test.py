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
"""tensor_util tests."""

# pylint: disable=unused-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.framework.python.framework import deprecation
from tensorflow.python.platform import tf_logging as logging


def _fn_with_doc(arg0, arg1):
  """fn doc.

  Args:
    arg0: Arg 0.
    arg1: Arg 0.

  Returns:
    Sum of args.
  """
  return arg0 + arg1


def _fn_no_doc(arg0, arg1):
  return arg0 + arg1


class DeprecationTest(tf.test.TestCase):

  def test_deprecated_illegal_args(self):
    instructions = "Instructions."
    with self.assertRaisesRegexp(ValueError, "date"):
      deprecation.deprecated(None, instructions)
    with self.assertRaisesRegexp(ValueError, "date"):
      deprecation.deprecated("", instructions)
    with self.assertRaisesRegexp(ValueError, "YYYY-MM-DD"):
      deprecation.deprecated("07-04-2016", instructions)
    date = "2016-07-04"
    with self.assertRaisesRegexp(ValueError, "instructions"):
      deprecation.deprecated(date, None)
    with self.assertRaisesRegexp(ValueError, "instructions"):
      deprecation.deprecated(date, "")

  @tf.test.mock.patch.object(logging, "warning", autospec=True)
  def test_static_fn_with_doc(self, mock_warning):
    date = "2016-07-04"
    instructions = "Update instructions."
    deprecated_fn = deprecation.deprecated(date, instructions)(_fn_with_doc)

    # Assert function docs are properly updated.
    self.assertEqual("_fn_with_doc", deprecated_fn.__name__)
    self.assertEqual(
        "fn doc. (deprecated)"
        "\n\nTHIS FUNCTION IS DEPRECATED. It will be removed after %s."
        "\nInstructions for updating:\n%s"
        "\n\n  Args:\n    arg0: Arg 0.\n    arg1: Arg 0."
        "\n\n  Returns:\n    Sum of args."
        "\n  " % (date, instructions),
        deprecated_fn.__doc__)
    self.assertEqual({}, deprecated_fn.__dict__)

    # Assert calling new fn issues log warning.
    self.assertEqual(3, _fn_with_doc(1, 2))
    self.assertEqual(0, mock_warning.call_count)
    self.assertEqual(3, deprecated_fn(1, 2))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"deprecated and will be removed after")
    self.assertTrue(set(args[1:]).issuperset(
        set(["_fn_with_doc", "__main__", date, instructions])))

  @tf.test.mock.patch.object(logging, "warning", autospec=True)
  def test_static_fn_no_doc(self, mock_warning):
    date = "2016-07-04"
    instructions = "Update instructions."
    deprecated_fn = deprecation.deprecated(date, instructions)(_fn_no_doc)

    # Assert function docs are properly updated.
    self.assertEqual("_fn_no_doc", deprecated_fn.__name__)
    self.assertEqual(
        "DEPRECATED FUNCTION"
        "\n\nTHIS FUNCTION IS DEPRECATED. It will be removed after %s."
        "\nInstructions for updating:\n%s" % (date, instructions),
        deprecated_fn.__doc__)
    self.assertEqual({}, deprecated_fn.__dict__)

    # Assert calling new fn issues log warning.
    self.assertEqual(3, _fn_no_doc(1, 2))
    self.assertEqual(0, mock_warning.call_count)
    self.assertEqual(3, deprecated_fn(1, 2))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"deprecated and will be removed after")
    self.assertTrue(set(args[1:]).issuperset(
        set(["_fn_no_doc", "__main__", date, instructions])))

  @tf.test.mock.patch.object(logging, "warning", autospec=True)
  def test_instance_fn_with_doc(self, mock_warning):

    class _Object(object):

      def __init(self):
        pass

      @deprecation.deprecated("2016-07-04", "Instructions.")
      def _fn(self, arg0, arg1):
        """fn doc.

        Args:
          arg0: Arg 0.
          arg1: Arg 0.

        Returns:
          Sum of args.
        """
        return arg0 + arg1

    # Assert function docs are properly updated.
    self.assertEqual(
        "fn doc. (deprecated)"
        "\n\nTHIS FUNCTION IS DEPRECATED. It will be removed after 2016-07-04."
        "\nInstructions for updating:\nInstructions."
        "\n\n        Args:\n          arg0: Arg 0.\n          arg1: Arg 0."
        "\n\n        Returns:\n          Sum of args.\n        ",
        getattr(_Object, "_fn").__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual(3, _Object()._fn(1, 2))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"deprecated and will be removed after")
    self.assertTrue(set(args[1:]).issuperset(
        set(["_fn", "__main__", "2016-07-04", "Instructions."])))

  @tf.test.mock.patch.object(logging, "warning", autospec=True)
  def test_instance_fn_no_doc(self, mock_warning):

    class _Object(object):

      def __init(self):
        pass

      @deprecation.deprecated("2016-07-04", "Instructions.")
      def _fn(self, arg0, arg1):
        return arg0 + arg1

    # Assert function docs are properly updated.
    self.assertEqual(
        "DEPRECATED FUNCTION"
        "\n\nTHIS FUNCTION IS DEPRECATED. It will be removed after 2016-07-04."
        "\nInstructions for updating:\nInstructions.",
        getattr(_Object, "_fn").__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual(3, _Object()._fn(1, 2))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"deprecated and will be removed after")
    self.assertTrue(set(args[1:]).issuperset(
        set(["_fn", "__main__", "2016-07-04", "Instructions."])))

  @tf.test.mock.patch.object(logging, "warning", autospec=True)
  def test_prop_with_doc(self, mock_warning):

    class _Object(object):

      def __init(self):
        pass

      @property
      @deprecation.deprecated("2016-07-04", "Instructions.")
      def _prop(self):
        """prop doc.

        Returns:
          String.
        """
        return "prop_with_doc"

    # Assert function docs are properly updated.
    self.assertEqual(
        "prop doc. (deprecated)"
        "\n\nTHIS FUNCTION IS DEPRECATED. It will be removed after 2016-07-04."
        "\nInstructions for updating:\nInstructions."
        "\n\n        Returns:\n          String.\n        ",
        getattr(_Object, "_prop").__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual("prop_with_doc", _Object()._prop)
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"deprecated and will be removed after")
    self.assertTrue(set(args[1:]).issuperset(
        set(["_prop", "__main__", "2016-07-04", "Instructions."])))

  @tf.test.mock.patch.object(logging, "warning", autospec=True)
  def test_prop_no_doc(self, mock_warning):

    class _Object(object):

      def __init(self):
        pass

      @property
      @deprecation.deprecated("2016-07-04", "Instructions.")
      def _prop(self):
        return "prop_no_doc"

    # Assert function docs are properly updated.
    self.assertEqual(
        "DEPRECATED FUNCTION"
        "\n\nTHIS FUNCTION IS DEPRECATED. It will be removed after 2016-07-04."
        "\nInstructions for updating:\nInstructions.",
        getattr(_Object, "_prop").__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual("prop_no_doc", _Object()._prop)
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"deprecated and will be removed after")
    self.assertTrue(set(args[1:]).issuperset(
        set(["_prop", "__main__", "2016-07-04", "Instructions."])))


if __name__ == "__main__":
  tf.test.main()
