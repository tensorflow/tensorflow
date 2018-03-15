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
"""Deprecation tests."""

# pylint: disable=unused-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation


class DeprecatedAliasTest(test.TestCase):

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_function_alias(self, mock_warning):
    deprecated_func = deprecation.deprecated_alias("deprecated.func",
                                                   "real.func",
                                                   logging.error)

    logging.error("fake error logged")
    self.assertEqual(0, mock_warning.call_count)
    deprecated_func("FAKE ERROR!")
    self.assertEqual(1, mock_warning.call_count)
    # Make sure the error points to the right file.
    self.assertRegexpMatches(mock_warning.call_args[0][1],
                             r"deprecation_test\.py:")
    deprecated_func("ANOTHER FAKE ERROR!")
    self.assertEqual(1, mock_warning.call_count)

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_class_alias(self, mock_warning):
    class MyClass(object):
      """My docstring."""

      init_args = []

      def __init__(self, arg):
        MyClass.init_args.append(arg)

    deprecated_cls = deprecation.deprecated_alias("deprecated.cls",
                                                  "real.cls",
                                                  MyClass)

    print(deprecated_cls.__name__)
    print(deprecated_cls.__module__)
    print(deprecated_cls.__doc__)

    MyClass("test")
    self.assertEqual(0, mock_warning.call_count)
    deprecated_cls("deprecated")
    self.assertEqual(1, mock_warning.call_count)
    # Make sure the error points to the right file.
    self.assertRegexpMatches(mock_warning.call_args[0][1],
                             r"deprecation_test\.py:")
    deprecated_cls("deprecated again")
    self.assertEqual(1, mock_warning.call_count)

    self.assertEqual(["test", "deprecated", "deprecated again"],
                     MyClass.init_args)


class DeprecationTest(test.TestCase):

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_deprecated_once(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated(date, instructions, warn_once=True)
    def _fn():
      pass

    _fn()
    self.assertEqual(1, mock_warning.call_count)
    _fn()
    self.assertEqual(1, mock_warning.call_count)

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_silence(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated(date, instructions, warn_once=False)
    def _fn():
      pass

    _fn()
    self.assertEqual(1, mock_warning.call_count)

    with deprecation.silence():
      _fn()
    self.assertEqual(1, mock_warning.call_count)

    _fn()
    self.assertEqual(2, mock_warning.call_count)

  def _assert_subset(self, expected_subset, actual_set):
    self.assertTrue(
        actual_set.issuperset(expected_subset),
        msg="%s is not a superset of %s." % (actual_set, expected_subset))

  def test_deprecated_illegal_args(self):
    instructions = "This is how you update..."
    with self.assertRaisesRegexp(ValueError, "YYYY-MM-DD"):
      deprecation.deprecated("", instructions)
    with self.assertRaisesRegexp(ValueError, "YYYY-MM-DD"):
      deprecation.deprecated("07-04-2016", instructions)
    date = "2016-07-04"
    with self.assertRaisesRegexp(ValueError, "instructions"):
      deprecation.deprecated(date, None)
    with self.assertRaisesRegexp(ValueError, "instructions"):
      deprecation.deprecated(date, "")

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_no_date(self, mock_warning):
    date = None
    instructions = "This is how you update..."

    @deprecation.deprecated(date, instructions)
    def _fn(arg0, arg1):
      """fn doc.

      Args:
        arg0: Arg 0.
        arg1: Arg 1.

      Returns:
        Sum of args.
      """
      return arg0 + arg1

    self.assertEqual(
        "fn doc. (deprecated)"
        "\n"
        "\nTHIS FUNCTION IS DEPRECATED. It will be removed in a future version."
        "\nInstructions for updating:\n%s"
        "\n"
        "\nArgs:"
        "\n  arg0: Arg 0."
        "\n  arg1: Arg 1."
        "\n"
        "\nReturns:"
        "\n  Sum of args." % instructions, _fn.__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(
        args[0], r"deprecated and will be removed")
    self._assert_subset(set(["in a future version", instructions]),
                        set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_static_fn_with_doc(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated(date, instructions)
    def _fn(arg0, arg1):
      """fn doc.

      Args:
        arg0: Arg 0.
        arg1: Arg 1.

      Returns:
        Sum of args.
      """
      return arg0 + arg1

    # Assert function docs are properly updated.
    self.assertEqual("_fn", _fn.__name__)
    self.assertEqual(
        "fn doc. (deprecated)"
        "\n"
        "\nTHIS FUNCTION IS DEPRECATED. It will be removed after %s."
        "\nInstructions for updating:\n%s"
        "\n"
        "\nArgs:"
        "\n  arg0: Arg 0."
        "\n  arg1: Arg 1."
        "\n"
        "\nReturns:"
        "\n  Sum of args." % (date, instructions), _fn.__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_static_fn_with_one_line_doc(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated(date, instructions)
    def _fn(arg0, arg1):
      """fn doc."""
      return arg0 + arg1

    # Assert function docs are properly updated.
    self.assertEqual("_fn", _fn.__name__)
    self.assertEqual(
        "fn doc. (deprecated)"
        "\n"
        "\nTHIS FUNCTION IS DEPRECATED. It will be removed after %s."
        "\nInstructions for updating:\n%s" % (date, instructions), _fn.__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_static_fn_no_doc(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated(date, instructions)
    def _fn(arg0, arg1):
      return arg0 + arg1

    # Assert function docs are properly updated.
    self.assertEqual("_fn", _fn.__name__)
    self.assertEqual(
        "DEPRECATED FUNCTION"
        "\n"
        "\nTHIS FUNCTION IS DEPRECATED. It will be removed after %s."
        "\nInstructions for updating:"
        "\n%s" % (date, instructions), _fn.__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_instance_fn_with_doc(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    class _Object(object):

      def __init(self):
        pass

      @deprecation.deprecated(date, instructions)
      def _fn(self, arg0, arg1):
        """fn doc.

        Args:
          arg0: Arg 0.
          arg1: Arg 1.

        Returns:
          Sum of args.
        """
        return arg0 + arg1

    # Assert function docs are properly updated.
    self.assertEqual(
        "fn doc. (deprecated)"
        "\n"
        "\nTHIS FUNCTION IS DEPRECATED. It will be removed after %s."
        "\nInstructions for updating:\n%s"
        "\n"
        "\nArgs:"
        "\n  arg0: Arg 0."
        "\n  arg1: Arg 1."
        "\n"
        "\nReturns:"
        "\n  Sum of args." % (date, instructions),
        getattr(_Object, "_fn").__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual(3, _Object()._fn(1, 2))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_instance_fn_with_one_line_doc(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    class _Object(object):

      def __init(self):
        pass

      @deprecation.deprecated(date, instructions)
      def _fn(self, arg0, arg1):
        """fn doc."""
        return arg0 + arg1

    # Assert function docs are properly updated.
    self.assertEqual(
        "fn doc. (deprecated)"
        "\n"
        "\nTHIS FUNCTION IS DEPRECATED. It will be removed after %s."
        "\nInstructions for updating:\n%s" % (date, instructions),
        getattr(_Object, "_fn").__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual(3, _Object()._fn(1, 2))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_instance_fn_no_doc(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    class _Object(object):

      def __init(self):
        pass

      @deprecation.deprecated(date, instructions)
      def _fn(self, arg0, arg1):
        return arg0 + arg1

    # Assert function docs are properly updated.
    self.assertEqual(
        "DEPRECATED FUNCTION"
        "\n"
        "\nTHIS FUNCTION IS DEPRECATED. It will be removed after %s."
        "\nInstructions for updating:"
        "\n%s" % (date, instructions), getattr(_Object, "_fn").__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual(3, _Object()._fn(1, 2))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  def test_prop_wrong_order(self):
    with self.assertRaisesRegexp(
        ValueError,
        "make sure @property appears before @deprecated in your source code"):
      # pylint: disable=unused-variable

      class _Object(object):

        def __init(self):
          pass

        @deprecation.deprecated("2016-07-04", "Instructions.")
        @property
        def _prop(self):
          return "prop_wrong_order"

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_prop_with_doc(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    class _Object(object):

      def __init(self):
        pass

      @property
      @deprecation.deprecated(date, instructions)
      def _prop(self):
        """prop doc.

        Returns:
          String.
        """
        return "prop_with_doc"

    # Assert function docs are properly updated.
    self.assertEqual(
        "prop doc. (deprecated)"
        "\n"
        "\nTHIS FUNCTION IS DEPRECATED. It will be removed after %s."
        "\nInstructions for updating:"
        "\n%s"
        "\n"
        "\nReturns:"
        "\n  String." % (date, instructions), getattr(_Object, "_prop").__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual("prop_with_doc", _Object()._prop)
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_prop_no_doc(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    class _Object(object):

      def __init(self):
        pass

      @property
      @deprecation.deprecated(date, instructions)
      def _prop(self):
        return "prop_no_doc"

    # Assert function docs are properly updated.
    self.assertEqual(
        "DEPRECATED FUNCTION"
        "\n"
        "\nTHIS FUNCTION IS DEPRECATED. It will be removed after %s."
        "\nInstructions for updating:"
        "\n%s" % (date, instructions), getattr(_Object, "_prop").__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual("prop_no_doc", _Object()._prop)
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))


class DeprecatedArgsTest(test.TestCase):

  def _assert_subset(self, expected_subset, actual_set):
    self.assertTrue(
        actual_set.issuperset(expected_subset),
        msg="%s is not a superset of %s." % (actual_set, expected_subset))

  def test_deprecated_illegal_args(self):
    instructions = "This is how you update..."
    date = "2016-07-04"
    with self.assertRaisesRegexp(ValueError, "YYYY-MM-DD"):
      deprecation.deprecated_args("", instructions, "deprecated")
    with self.assertRaisesRegexp(ValueError, "YYYY-MM-DD"):
      deprecation.deprecated_args("07-04-2016", instructions, "deprecated")
    with self.assertRaisesRegexp(ValueError, "instructions"):
      deprecation.deprecated_args(date, None, "deprecated")
    with self.assertRaisesRegexp(ValueError, "instructions"):
      deprecation.deprecated_args(date, "", "deprecated")
    with self.assertRaisesRegexp(ValueError, "argument"):
      deprecation.deprecated_args(date, instructions)

  def test_deprecated_missing_args(self):
    date = "2016-07-04"
    instructions = "This is how you update..."

    def _fn(arg0, arg1, deprecated=None):
      return arg0 + arg1 if deprecated else arg1 + arg0

    # Assert calls without the deprecated argument log nothing.
    with self.assertRaisesRegexp(ValueError, "not present.*\\['missing'\\]"):
      deprecation.deprecated_args(date, instructions, "missing")(_fn)

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_static_fn_with_doc(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_args(date, instructions, "deprecated")
    def _fn(arg0, arg1, deprecated=True):
      """fn doc.

      Args:
        arg0: Arg 0.
        arg1: Arg 1.
        deprecated: Deprecated!

      Returns:
        Sum of args.
      """
      return arg0 + arg1 if deprecated else arg1 + arg0

    # Assert function docs are properly updated.
    self.assertEqual("_fn", _fn.__name__)
    self.assertEqual(
        "fn doc. (deprecated arguments)"
        "\n"
        "\nSOME ARGUMENTS ARE DEPRECATED. They will be removed after %s."
        "\nInstructions for updating:\n%s"
        "\n"
        "\nArgs:"
        "\n  arg0: Arg 0."
        "\n  arg1: Arg 1."
        "\n  deprecated: Deprecated!"
        "\n"
        "\nReturns:"
        "\n  Sum of args." % (date, instructions), _fn.__doc__)

    # Assert calls without the deprecated argument log nothing.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(0, mock_warning.call_count)

    # Assert calls with the deprecated argument log a warning.
    self.assertEqual(3, _fn(1, 2, True))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_static_fn_with_one_line_doc(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_args(date, instructions, "deprecated")
    def _fn(arg0, arg1, deprecated=True):
      """fn doc."""
      return arg0 + arg1 if deprecated else arg1 + arg0

    # Assert function docs are properly updated.
    self.assertEqual("_fn", _fn.__name__)
    self.assertEqual(
        "fn doc. (deprecated arguments)"
        "\n"
        "\nSOME ARGUMENTS ARE DEPRECATED. They will be removed after %s."
        "\nInstructions for updating:\n%s" % (date, instructions), _fn.__doc__)

    # Assert calls without the deprecated argument log nothing.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(0, mock_warning.call_count)

    # Assert calls with the deprecated argument log a warning.
    self.assertEqual(3, _fn(1, 2, True))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_static_fn_no_doc(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_args(date, instructions, "deprecated")
    def _fn(arg0, arg1, deprecated=True):
      return arg0 + arg1 if deprecated else arg1 + arg0

    # Assert function docs are properly updated.
    self.assertEqual("_fn", _fn.__name__)
    self.assertEqual(
        "DEPRECATED FUNCTION ARGUMENTS"
        "\n"
        "\nSOME ARGUMENTS ARE DEPRECATED. They will be removed after %s."
        "\nInstructions for updating:"
        "\n%s" % (date, instructions), _fn.__doc__)

    # Assert calls without the deprecated argument log nothing.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(0, mock_warning.call_count)

    # Assert calls with the deprecated argument log a warning.
    self.assertEqual(3, _fn(1, 2, True))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_varargs(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_args(date, instructions, "deprecated")
    def _fn(arg0, arg1, *deprecated):
      return arg0 + arg1 if deprecated else arg1 + arg0

    # Assert calls without the deprecated argument log nothing.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(0, mock_warning.call_count)

    # Assert calls with the deprecated argument log a warning.
    self.assertEqual(3, _fn(1, 2, True, False))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_kwargs(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_args(date, instructions, "deprecated")
    def _fn(arg0, arg1, **deprecated):
      return arg0 + arg1 if deprecated else arg1 + arg0

    # Assert calls without the deprecated argument log nothing.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(0, mock_warning.call_count)

    # Assert calls with the deprecated argument log a warning.
    self.assertEqual(3, _fn(1, 2, a=True, b=False))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_positional_and_named(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_args(date, instructions, "d1", "d2")
    def _fn(arg0, d1=None, arg1=2, d2=None):
      return arg0 + arg1 if d1 else arg1 + arg0 if d2 else arg0 * arg1

    # Assert calls without the deprecated arguments log nothing.
    self.assertEqual(2, _fn(1, arg1=2))
    self.assertEqual(0, mock_warning.call_count)

    # Assert calls with the deprecated arguments log warnings.
    self.assertEqual(2, _fn(1, None, 2, d2=False))
    self.assertEqual(2, mock_warning.call_count)
    (args1, _) = mock_warning.call_args_list[0]
    self.assertRegexpMatches(args1[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions, "d1"]),
                        set(args1[1:]))
    (args2, _) = mock_warning.call_args_list[1]
    self.assertRegexpMatches(args2[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions, "d2"]),
                        set(args2[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_positional_and_named_with_ok_vals(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_args(date, instructions, ("d1", None),
                                 ("d2", "my_ok_val"))
    def _fn(arg0, d1=None, arg1=2, d2=None):
      return arg0 + arg1 if d1 else arg1 + arg0 if d2 else arg0 * arg1

    # Assert calls without the deprecated arguments log nothing.
    self.assertEqual(2, _fn(1, arg1=2))
    self.assertEqual(0, mock_warning.call_count)

    # Assert calls with the deprecated arguments log warnings.
    self.assertEqual(2, _fn(1, False, 2, d2=False))
    self.assertEqual(2, mock_warning.call_count)
    (args1, _) = mock_warning.call_args_list[0]
    self.assertRegexpMatches(args1[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions, "d1"]),
                        set(args1[1:]))
    (args2, _) = mock_warning.call_args_list[1]
    self.assertRegexpMatches(args2[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions, "d2"]),
                        set(args2[1:]))

    # Assert calls with the deprecated arguments don't log warnings if
    # the value matches the 'ok_val'.
    mock_warning.reset_mock()
    self.assertEqual(3, _fn(1, None, 2, d2="my_ok_val"))
    self.assertEqual(0, mock_warning.call_count)

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_deprecated_args_once(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_args(date, instructions, "arg", warn_once=True)
    def _fn(arg=0):  # pylint: disable=unused-argument
      pass

    _fn()
    self.assertEqual(0, mock_warning.call_count)
    _fn(arg=0)
    self.assertEqual(1, mock_warning.call_count)
    _fn(arg=1)
    self.assertEqual(1, mock_warning.call_count)

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_deprecated_multiple_args_once_each(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_args(date, instructions, "arg0", "arg1",
                                 warn_once=True)
    def _fn(arg0=0, arg1=0):  # pylint: disable=unused-argument
      pass

    _fn(arg0=0)
    self.assertEqual(1, mock_warning.call_count)
    _fn(arg0=0)
    self.assertEqual(1, mock_warning.call_count)
    _fn(arg1=0)
    self.assertEqual(2, mock_warning.call_count)
    _fn(arg0=0)
    self.assertEqual(2, mock_warning.call_count)
    _fn(arg1=0)
    self.assertEqual(2, mock_warning.call_count)


class DeprecatedArgValuesTest(test.TestCase):

  def _assert_subset(self, expected_subset, actual_set):
    self.assertTrue(
        actual_set.issuperset(expected_subset),
        msg="%s is not a superset of %s." % (actual_set, expected_subset))

  def test_deprecated_illegal_args(self):
    instructions = "This is how you update..."
    with self.assertRaisesRegexp(ValueError, "YYYY-MM-DD"):
      deprecation.deprecated_arg_values("", instructions, deprecated=True)
    with self.assertRaisesRegexp(ValueError, "YYYY-MM-DD"):
      deprecation.deprecated_arg_values(
          "07-04-2016", instructions, deprecated=True)
    date = "2016-07-04"
    with self.assertRaisesRegexp(ValueError, "instructions"):
      deprecation.deprecated_arg_values(date, None, deprecated=True)
    with self.assertRaisesRegexp(ValueError, "instructions"):
      deprecation.deprecated_arg_values(date, "", deprecated=True)
    with self.assertRaisesRegexp(ValueError, "argument", deprecated=True):
      deprecation.deprecated_arg_values(date, instructions)

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_static_fn_with_doc(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_arg_values(date, instructions, warn_once=False,
                                       deprecated=True)
    def _fn(arg0, arg1, deprecated=True):
      """fn doc.

      Args:
        arg0: Arg 0.
        arg1: Arg 1.
        deprecated: Deprecated!

      Returns:
        Sum of args.
      """
      return arg0 + arg1 if deprecated else arg1 + arg0

    # Assert function docs are properly updated.
    self.assertEqual("_fn", _fn.__name__)
    self.assertEqual(
        "fn doc. (deprecated arguments)"
        "\n"
        "\nSOME ARGUMENTS ARE DEPRECATED. They will be removed after %s."
        "\nInstructions for updating:\n%s"
        "\n"
        "\nArgs:"
        "\n  arg0: Arg 0."
        "\n  arg1: Arg 1."
        "\n  deprecated: Deprecated!"
        "\n"
        "\nReturns:"
        "\n  Sum of args." % (date, instructions), _fn.__doc__)

    # Assert calling new fn with non-deprecated value logs nothing.
    self.assertEqual(3, _fn(1, 2, deprecated=False))
    self.assertEqual(0, mock_warning.call_count)

    # Assert calling new fn with deprecated value issues log warning.
    self.assertEqual(3, _fn(1, 2, deprecated=True))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

    # Assert calling new fn with default deprecated value issues log warning.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(2, mock_warning.call_count)

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_static_fn_with_one_line_doc(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_arg_values(date, instructions, warn_once=False,
                                       deprecated=True)
    def _fn(arg0, arg1, deprecated=True):
      """fn doc."""
      return arg0 + arg1 if deprecated else arg1 + arg0

    # Assert function docs are properly updated.
    self.assertEqual("_fn", _fn.__name__)
    self.assertEqual(
        "fn doc. (deprecated arguments)"
        "\n"
        "\nSOME ARGUMENTS ARE DEPRECATED. They will be removed after %s."
        "\nInstructions for updating:\n%s" % (date, instructions), _fn.__doc__)

    # Assert calling new fn with non-deprecated value logs nothing.
    self.assertEqual(3, _fn(1, 2, deprecated=False))
    self.assertEqual(0, mock_warning.call_count)

    # Assert calling new fn with deprecated value issues log warning.
    self.assertEqual(3, _fn(1, 2, deprecated=True))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

    # Assert calling new fn with default deprecated value issues log warning.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(2, mock_warning.call_count)

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_static_fn_no_doc(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_arg_values(date, instructions, warn_once=False,
                                       deprecated=True)
    def _fn(arg0, arg1, deprecated=True):
      return arg0 + arg1 if deprecated else arg1 + arg0

    # Assert function docs are properly updated.
    self.assertEqual("_fn", _fn.__name__)
    self.assertEqual(
        "DEPRECATED FUNCTION ARGUMENTS"
        "\n"
        "\nSOME ARGUMENTS ARE DEPRECATED. They will be removed after %s."
        "\nInstructions for updating:"
        "\n%s" % (date, instructions), _fn.__doc__)

    # Assert calling new fn with non-deprecated value logs nothing.
    self.assertEqual(3, _fn(1, 2, deprecated=False))
    self.assertEqual(0, mock_warning.call_count)

    # Assert calling new fn issues log warning.
    self.assertEqual(3, _fn(1, 2, deprecated=True))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

    # Assert calling new fn with default deprecated value issues log warning.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(2, mock_warning.call_count)

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_deprecated_arg_values_once(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_arg_values(date, instructions, warn_once=True,
                                       deprecated=True)
    def _fn(deprecated):  # pylint: disable=unused-argument
      pass

    _fn(deprecated=False)
    self.assertEqual(0, mock_warning.call_count)
    _fn(deprecated=True)
    self.assertEqual(1, mock_warning.call_count)
    _fn(deprecated=True)
    self.assertEqual(1, mock_warning.call_count)

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_deprecated_multiple_arg_values_once_each(self, mock_warning):
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_arg_values(date, instructions, warn_once=True,
                                       arg0="forbidden", arg1="disallowed")
    def _fn(arg0, arg1):  # pylint: disable=unused-argument
      pass

    _fn(arg0="allowed", arg1="also allowed")
    self.assertEqual(0, mock_warning.call_count)
    _fn(arg0="forbidden", arg1="disallowed")
    self.assertEqual(2, mock_warning.call_count)
    _fn(arg0="forbidden", arg1="allowed")
    self.assertEqual(2, mock_warning.call_count)
    _fn(arg0="forbidden", arg1="disallowed")
    self.assertEqual(2, mock_warning.call_count)


class DeprecationArgumentsTest(test.TestCase):

  def testDeprecatedArgumentLookup(self):
    good_value = 3
    self.assertEqual(
        deprecation.deprecated_argument_lookup("val_new", good_value, "val_old",
                                               None), good_value)
    self.assertEqual(
        deprecation.deprecated_argument_lookup("val_new", None, "val_old",
                                               good_value), good_value)
    with self.assertRaisesRegexp(ValueError,
                                 "Cannot specify both 'val_old' and 'val_new'"):
      self.assertEqual(
          deprecation.deprecated_argument_lookup("val_new", good_value,
                                                 "val_old", good_value),
          good_value)

  def testRewriteArgumentDocstring(self):
    docs = """Add `a` and `b`

    Args:
      a: first arg
      b: second arg
    """
    new_docs = deprecation.rewrite_argument_docstring(
        deprecation.rewrite_argument_docstring(docs, "a", "left"), "b", "right")
    new_docs_ref = """Add `left` and `right`

    Args:
      left: first arg
      right: second arg
    """
    self.assertEqual(new_docs, new_docs_ref)


if __name__ == "__main__":
  test.main()
