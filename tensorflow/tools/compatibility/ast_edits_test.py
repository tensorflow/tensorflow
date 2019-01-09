# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for ast_edits which is used in tf upgraders.

All of the tests assume that we want to change from an API containing

    def f(a, b, kw1, kw2): ...
    def g(a, b, kw1, c, kw1_alias): ...
    def g2(a, b, kw1, c, d, kw1_alias): ...
    def h(a, kw1, kw2, kw1_alias, kw2_alias): ...

and the changes to the API consist of renaming, reordering, and/or removing
arguments. Thus, we want to be able to generate changes to produce each of the
following new APIs:

    def f(a, b, kw1, kw3): ...
    def f(a, b, kw2, kw1): ...
    def f(a, b, kw3, kw1): ...
    def g(a, b, kw1, c): ...
    def g(a, b, c, kw1): ...
    def g2(a, b, kw1, c, d): ...
    def g2(a, b, c, d, kw1): ...
    def h(a, kw1, kw2): ...

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib
from tensorflow.tools.compatibility import ast_edits


class NoUpdateSpec(ast_edits.APIChangeSpec):
  """A specification of an API change which doesn't change anything."""

  def __init__(self):
    self.function_handle = {}
    self.function_reorders = {}
    self.function_keyword_renames = {}
    self.symbol_renames = {}
    self.function_warnings = {}
    self.change_to_function = {}


class RenameKeywordSpec(NoUpdateSpec):
  """A specification where kw2 gets renamed to kw3.

  The new API is

    def f(a, b, kw1, kw3): ...

  """

  def __init__(self):
    NoUpdateSpec.__init__(self)
    self.update_renames()

  def update_renames(self):
    self.function_keyword_renames["f"] = {"kw2": "kw3"}


class ReorderKeywordSpec(NoUpdateSpec):
  """A specification where kw2 gets moved in front of kw1.

  The new API is

    def f(a, b, kw2, kw1): ...

  """

  def __init__(self):
    NoUpdateSpec.__init__(self)
    self.update_reorders()

  def update_reorders(self):
    # Note that these should be in the old order.
    self.function_reorders["f"] = ["a", "b", "kw1", "kw2"]


class ReorderAndRenameKeywordSpec(ReorderKeywordSpec, RenameKeywordSpec):
  """A specification where kw2 gets moved in front of kw1 and is changed to kw3.

  The new API is

    def f(a, b, kw3, kw1): ...

  """

  def __init__(self):
    ReorderKeywordSpec.__init__(self)
    RenameKeywordSpec.__init__(self)
    self.update_renames()
    self.update_reorders()


class RemoveDeprecatedAliasKeyword(NoUpdateSpec):
  """A specification where kw1_alias is removed in g.

  The new API is

    def g(a, b, kw1, c): ...
    def g2(a, b, kw1, c, d): ...

  """

  def __init__(self):
    NoUpdateSpec.__init__(self)
    self.function_keyword_renames["g"] = {"kw1_alias": "kw1"}
    self.function_keyword_renames["g2"] = {"kw1_alias": "kw1"}


class RemoveDeprecatedAliasAndReorderRest(RemoveDeprecatedAliasKeyword):
  """A specification where kw1_alias is removed in g.

  The new API is

    def g(a, b, c, kw1): ...
    def g2(a, b, c, d, kw1): ...

  """

  def __init__(self):
    RemoveDeprecatedAliasKeyword.__init__(self)
    # Note that these should be in the old order.
    self.function_reorders["g"] = ["a", "b", "kw1", "c"]
    self.function_reorders["g2"] = ["a", "b", "kw1", "c", "d"]


class RemoveMultipleKeywordArguments(NoUpdateSpec):
  """A specification where both keyword aliases are removed from h.

  The new API is

    def h(a, kw1, kw2): ...

  """

  def __init__(self):
    NoUpdateSpec.__init__(self)
    self.function_keyword_renames["h"] = {
        "kw1_alias": "kw1",
        "kw2_alias": "kw2",
    }


class TestAstEdits(test_util.TensorFlowTestCase):

  def _upgrade(self, spec, old_file_text):
    in_file = six.StringIO(old_file_text)
    out_file = six.StringIO()
    upgrader = ast_edits.ASTCodeUpgrader(spec)
    count, report, errors = (
        upgrader.process_opened_file("test.py", in_file,
                                     "test_out.py", out_file))
    return (count, report, errors), out_file.getvalue()

  def testNoTransformIfNothingIsSupplied(self):
    text = "f(a, b, kw1=c, kw2=d)\n"
    _, new_text = self._upgrade(NoUpdateSpec(), text)
    self.assertEqual(new_text, text)

    text = "f(a, b, c, d)\n"
    _, new_text = self._upgrade(NoUpdateSpec(), text)
    self.assertEqual(new_text, text)

  def testKeywordRename(self):
    """Test that we get the expected result if renaming kw2 to kw3."""
    text = "f(a, b, kw1=c, kw2=d)\n"
    expected = "f(a, b, kw1=c, kw3=d)\n"
    _, new_text = self._upgrade(RenameKeywordSpec(), text)
    self.assertEqual(new_text, expected)

    # No keywords specified, no reordering, so we should get input as output
    text = "f(a, b, c, d)\n"
    _, new_text = self._upgrade(RenameKeywordSpec(), text)
    self.assertEqual(new_text, text)

  def testKeywordReorder(self):
    """Test that we get the expected result if kw2 is now before kw1."""
    text = "f(a, b, kw1=c, kw2=d)\n"
    acceptable_outputs = [
        # No change is a valid output
        text,
        # Just reordering the kw.. args is also ok
        "f(a, b, kw2=d, kw1=c)\n",
        # Also cases where all arguments are fully specified are allowed
        "f(a=a, b=b, kw1=c, kw2=d)\n",
        "f(a=a, b=b, kw2=d, kw1=c)\n",
    ]
    _, new_text = self._upgrade(ReorderKeywordSpec(), text)
    self.assertIn(new_text, acceptable_outputs)

    # Keywords are reordered, so we should reorder arguments too
    text = "f(a, b, c, d)\n"
    acceptable_outputs = [
        "f(a, b, d, c)\n",
        "f(a=a, b=b, kw1=c, kw2=d)\n",
        "f(a=a, b=b, kw2=d, kw1=c)\n",
    ]
    _, new_text = self._upgrade(ReorderKeywordSpec(), text)
    self.assertIn(new_text, acceptable_outputs)

  def testKeywordReorderAndRename(self):
    """Test that we get the expected result if kw2 is renamed and moved."""
    text = "f(a, b, kw1=c, kw2=d)\n"
    acceptable_outputs = [
        "f(a, b, kw3=d, kw1=c)\n",
        "f(a=a, b=b, kw1=c, kw3=d)\n",
        "f(a=a, b=b, kw3=d, kw1=c)\n",
    ]
    _, new_text = self._upgrade(ReorderAndRenameKeywordSpec(), text)
    self.assertIn(new_text, acceptable_outputs)

    # Keywords are reordered, so we should reorder arguments too
    text = "f(a, b, c, d)\n"
    acceptable_outputs = [
        "f(a, b, d, c)\n",
        "f(a=a, b=b, kw1=c, kw3=d)\n",
        "f(a=a, b=b, kw3=d, kw1=c)\n",
    ]
    _, new_text = self._upgrade(ReorderAndRenameKeywordSpec(), text)
    self.assertIn(new_text, acceptable_outputs)

  def testRemoveDeprecatedKeywordAlias(self):
    """Test that we get the expected result if a keyword alias is removed."""
    text = "g(a, b, kw1=x, c=c)\n"
    acceptable_outputs = [
        # Not using deprecated alias, so original is ok
        text,
        "g(a=a, b=b, kw1=x, c=c)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertIn(new_text, acceptable_outputs)

    # No keyword used, should be no change
    text = "g(a, b, x, c)\n"
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertEqual(new_text, text)

    # If we used the alias, it should get renamed
    text = "g(a, b, kw1_alias=x, c=c)\n"
    acceptable_outputs = [
        "g(a, b, kw1=x, c=c)\n",
        "g(a, b, c=c, kw1=x)\n",
        "g(a=a, b=b, kw1=x, c=c)\n",
        "g(a=a, b=b, c=c, kw1=x)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertIn(new_text, acceptable_outputs)

    # It should get renamed even if it's last
    text = "g(a, b, c=c, kw1_alias=x)\n"
    acceptable_outputs = [
        "g(a, b, kw1=x, c=c)\n",
        "g(a, b, c=c, kw1=x)\n",
        "g(a=a, b=b, kw1=x, c=c)\n",
        "g(a=a, b=b, c=c, kw1=x)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertIn(new_text, acceptable_outputs)

  def testRemoveDeprecatedKeywordAndReorder(self):
    """Test for when a keyword alias is removed and args are reordered."""
    text = "g(a, b, kw1=x, c=c)\n"
    acceptable_outputs = [
        "g(a, b, c=c, kw1=x)\n",
        "g(a=a, b=b, kw1=x, c=c)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasAndReorderRest(), text)
    self.assertIn(new_text, acceptable_outputs)

    # Keywords are reordered, so we should reorder arguments too
    text = "g(a, b, x, c)\n"
    # Don't accept an output which doesn't reorder c and d
    acceptable_outputs = [
        "g(a, b, c, x)\n",
        "g(a=a, b=b, kw1=x, c=c)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasAndReorderRest(), text)
    self.assertIn(new_text, acceptable_outputs)

    # If we used the alias, it should get renamed
    text = "g(a, b, kw1_alias=x, c=c)\n"
    acceptable_outputs = [
        "g(a, b, kw1=x, c=c)\n",
        "g(a, b, c=c, kw1=x)\n",
        "g(a=a, b=b, kw1=x, c=c)\n",
        "g(a=a, b=b, c=c, kw1=x)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertIn(new_text, acceptable_outputs)

    # It should get renamed and reordered even if it's last
    text = "g(a, b, c=c, kw1_alias=x)\n"
    acceptable_outputs = [
        "g(a, b, kw1=x, c=c)\n",
        "g(a, b, c=c, kw1=x)\n",
        "g(a=a, b=b, kw1=x, c=c)\n",
        "g(a=a, b=b, c=c, kw1=x)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertIn(new_text, acceptable_outputs)

  def testRemoveDeprecatedKeywordAndReorder2(self):
    """Same as testRemoveDeprecatedKeywordAndReorder but on g2 (more args)."""
    text = "g2(a, b, kw1=x, c=c, d=d)\n"
    acceptable_outputs = [
        "g2(a, b, c=c, d=d, kw1=x)\n",
        "g2(a=a, b=b, kw1=x, c=c, d=d)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasAndReorderRest(), text)
    self.assertIn(new_text, acceptable_outputs)

    # Keywords are reordered, so we should reorder arguments too
    text = "g2(a, b, x, c, d)\n"
    # Don't accept an output which doesn't reorder c and d
    acceptable_outputs = [
        "g2(a, b, c, d, x)\n",
        "g2(a=a, b=b, kw1=x, c=c, d=d)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasAndReorderRest(), text)
    self.assertIn(new_text, acceptable_outputs)

    # If we used the alias, it should get renamed
    text = "g2(a, b, kw1_alias=x, c=c, d=d)\n"
    acceptable_outputs = [
        "g2(a, b, kw1=x, c=c, d=d)\n",
        "g2(a, b, c=c, d=d, kw1=x)\n",
        "g2(a=a, b=b, kw1=x, c=c, d=d)\n",
        "g2(a=a, b=b, c=c, d=d, kw1=x)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertIn(new_text, acceptable_outputs)

    # It should get renamed and reordered even if it's not in order
    text = "g2(a, b, d=d, c=c, kw1_alias=x)\n"
    acceptable_outputs = [
        "g2(a, b, kw1=x, c=c, d=d)\n",
        "g2(a, b, c=c, d=d, kw1=x)\n",
        "g2(a, b, d=d, c=c, kw1=x)\n",
        "g2(a=a, b=b, kw1=x, c=c, d=d)\n",
        "g2(a=a, b=b, c=c, d=d, kw1=x)\n",
        "g2(a=a, b=b, d=d, c=c, kw1=x)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertIn(new_text, acceptable_outputs)

  def testRemoveMultipleKeywords(self):
    """Remove multiple keywords at once."""
    # Not using deprecated keywords -> no rename
    text = "h(a, kw1=x, kw2=y)\n"
    _, new_text = self._upgrade(RemoveMultipleKeywordArguments(), text)
    self.assertEqual(new_text, text)

    # Using positional arguments (in proper order) -> no change
    text = "h(a, x, y)\n"
    _, new_text = self._upgrade(RemoveMultipleKeywordArguments(), text)
    self.assertEqual(new_text, text)

    # Use only the old names, in order
    text = "h(a, kw1_alias=x, kw2_alias=y)\n"
    acceptable_outputs = [
        "h(a, x, y)\n",
        "h(a, kw1=x, kw2=y)\n",
        "h(a=a, kw1=x, kw2=y)\n",
        "h(a, kw2=y, kw1=x)\n",
        "h(a=a, kw2=y, kw1=x)\n",
    ]
    _, new_text = self._upgrade(RemoveMultipleKeywordArguments(), text)
    self.assertIn(new_text, acceptable_outputs)

    # Use only the old names, in reverse order, should give one of same outputs
    text = "h(a, kw2_alias=y, kw1_alias=x)\n"
    _, new_text = self._upgrade(RemoveMultipleKeywordArguments(), text)
    self.assertIn(new_text, acceptable_outputs)

    # Mix old and new names
    text = "h(a, kw1=x, kw2_alias=y)\n"
    _, new_text = self._upgrade(RemoveMultipleKeywordArguments(), text)
    self.assertIn(new_text, acceptable_outputs)

  def testUnrestrictedFunctionWarnings(self):
    class FooWarningSpec(NoUpdateSpec):
      """Usages of function attribute foo() prints out a warning."""

      def __init__(self):
        NoUpdateSpec.__init__(self)
        self.function_warnings = {"*.foo": "not good"}

    texts = ["object.foo()", "get_object().foo()",
             "get_object().foo()", "object.foo().bar()"]
    for text in texts:
      (_, report, _), _ = self._upgrade(FooWarningSpec(), text)
      self.assertIn("not good", report)

    # Note that foo() won't result in a warning, because in this case foo is
    # not an attribute, but a name.
    false_alarms = ["foo", "foo()", "foo.bar()", "obj.run_foo()", "obj.foo"]
    for text in false_alarms:
      (_, report, _), _ = self._upgrade(FooWarningSpec(), text)
      self.assertNotIn("not good", report)


if __name__ == "__main__":
  test_lib.main()
