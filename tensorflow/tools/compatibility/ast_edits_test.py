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

    import foo as f

    def f(a, b, kw1, kw2): ...
    def g(a, b, kw1, c, kw1_alias): ...
    def g2(a, b, kw1, c, d, kw1_alias): ...
    def h(a, kw1, kw2, kw1_alias, kw2_alias): ...

and the changes to the API consist of renaming, reordering, and/or removing
arguments. Thus, we want to be able to generate changes to produce each of the
following new APIs:

    import bar as f

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

import ast
import os

import six

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib
from tensorflow.tools.compatibility import ast_edits


class ModuleDeprecationSpec(ast_edits.NoUpdateSpec):
  """A specification which deprecates 'a.b'."""

  def __init__(self):
    ast_edits.NoUpdateSpec.__init__(self)
    self.module_deprecations.update({"a.b": (ast_edits.ERROR, "a.b is evil.")})


class RenameKeywordSpec(ast_edits.NoUpdateSpec):
  """A specification where kw2 gets renamed to kw3.

  The new API is

    def f(a, b, kw1, kw3): ...

  """

  def __init__(self):
    ast_edits.NoUpdateSpec.__init__(self)
    self.update_renames()

  def update_renames(self):
    self.function_keyword_renames["f"] = {"kw2": "kw3"}


class ReorderKeywordSpec(ast_edits.NoUpdateSpec):
  """A specification where kw2 gets moved in front of kw1.

  The new API is

    def f(a, b, kw2, kw1): ...

  """

  def __init__(self):
    ast_edits.NoUpdateSpec.__init__(self)
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


class RemoveDeprecatedAliasKeyword(ast_edits.NoUpdateSpec):
  """A specification where kw1_alias is removed in g.

  The new API is

    def g(a, b, kw1, c): ...
    def g2(a, b, kw1, c, d): ...

  """

  def __init__(self):
    ast_edits.NoUpdateSpec.__init__(self)
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


class RemoveMultipleKeywordArguments(ast_edits.NoUpdateSpec):
  """A specification where both keyword aliases are removed from h.

  The new API is

    def h(a, kw1, kw2): ...

  """

  def __init__(self):
    ast_edits.NoUpdateSpec.__init__(self)
    self.function_keyword_renames["h"] = {
        "kw1_alias": "kw1",
        "kw2_alias": "kw2",
    }


class RenameImports(ast_edits.NoUpdateSpec):
  """Specification for renaming imports."""

  def __init__(self):
    ast_edits.NoUpdateSpec.__init__(self)
    self.import_renames = {
        "foo": ast_edits.ImportRename(
            "bar",
            excluded_prefixes=["foo.baz"])
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

  def testModuleDeprecation(self):
    text = "a.b.c(a.b.x)"
    (_, _, errors), new_text = self._upgrade(ModuleDeprecationSpec(), text)
    self.assertEqual(text, new_text)
    self.assertIn("Using member a.b.c", errors[0])
    self.assertIn("1:0", errors[0])
    self.assertIn("Using member a.b.c", errors[0])
    self.assertIn("1:6", errors[1])

  def testNoTransformIfNothingIsSupplied(self):
    text = "f(a, b, kw1=c, kw2=d)\n"
    _, new_text = self._upgrade(ast_edits.NoUpdateSpec(), text)
    self.assertEqual(new_text, text)

    text = "f(a, b, c, d)\n"
    _, new_text = self._upgrade(ast_edits.NoUpdateSpec(), text)
    self.assertEqual(new_text, text)

  def testKeywordRename(self):
    """Test that we get the expected result if renaming kw2 to kw3."""
    text = "f(a, b, kw1=c, kw2=d)\n"
    expected = "f(a, b, kw1=c, kw3=d)\n"
    (_, report, _), new_text = self._upgrade(RenameKeywordSpec(), text)
    self.assertEqual(new_text, expected)
    self.assertNotIn("Manual check required", report)

    # No keywords specified, no reordering, so we should get input as output
    text = "f(a, b, c, d)\n"
    (_, report, _), new_text = self._upgrade(RenameKeywordSpec(), text)
    self.assertEqual(new_text, text)
    self.assertNotIn("Manual check required", report)

    # Positional *args passed in that we cannot inspect, should warn
    text = "f(a, *args)\n"
    (_, report, _), _ = self._upgrade(RenameKeywordSpec(), text)
    self.assertNotIn("Manual check required", report)

    # **kwargs passed in that we cannot inspect, should warn
    text = "f(a, b, kw1=c, **kwargs)\n"
    (_, report, _), _ = self._upgrade(RenameKeywordSpec(), text)
    self.assertIn("Manual check required", report)

  def testKeywordReorderWithParens(self):
    """Test that we get the expected result if there are parens around args."""
    text = "f((a), ( ( b ) ))\n"
    acceptable_outputs = [
        # No change is a valid output
        text,
        # Also cases where all arguments are fully specified are allowed
        "f(a=(a), b=( ( b ) ))\n",
        # Making the parens canonical is ok
        "f(a=(a), b=((b)))\n",
    ]
    _, new_text = self._upgrade(ReorderKeywordSpec(), text)
    self.assertIn(new_text, acceptable_outputs)

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
    (_, report, _), new_text = self._upgrade(ReorderKeywordSpec(), text)
    self.assertIn(new_text, acceptable_outputs)
    self.assertNotIn("Manual check required", report)

    # Keywords are reordered, so we should reorder arguments too
    text = "f(a, b, c, d)\n"
    acceptable_outputs = [
        "f(a, b, d, c)\n",
        "f(a=a, b=b, kw1=c, kw2=d)\n",
        "f(a=a, b=b, kw2=d, kw1=c)\n",
    ]
    (_, report, _), new_text = self._upgrade(ReorderKeywordSpec(), text)
    self.assertIn(new_text, acceptable_outputs)
    self.assertNotIn("Manual check required", report)

    # Positional *args passed in that we cannot inspect, should warn
    text = "f(a, b, *args)\n"
    (_, report, _), _ = self._upgrade(ReorderKeywordSpec(), text)
    self.assertIn("Manual check required", report)

    # **kwargs passed in that we cannot inspect, should warn
    text = "f(a, b, kw1=c, **kwargs)\n"
    (_, report, _), _ = self._upgrade(ReorderKeywordSpec(), text)
    self.assertNotIn("Manual check required", report)

  def testKeywordReorderAndRename(self):
    """Test that we get the expected result if kw2 is renamed and moved."""
    text = "f(a, b, kw1=c, kw2=d)\n"
    acceptable_outputs = [
        "f(a, b, kw3=d, kw1=c)\n",
        "f(a=a, b=b, kw1=c, kw3=d)\n",
        "f(a=a, b=b, kw3=d, kw1=c)\n",
    ]
    (_, report, _), new_text = self._upgrade(
        ReorderAndRenameKeywordSpec(), text)
    self.assertIn(new_text, acceptable_outputs)
    self.assertNotIn("Manual check required", report)

    # Keywords are reordered, so we should reorder arguments too
    text = "f(a, b, c, d)\n"
    acceptable_outputs = [
        "f(a, b, d, c)\n",
        "f(a=a, b=b, kw1=c, kw3=d)\n",
        "f(a=a, b=b, kw3=d, kw1=c)\n",
    ]
    (_, report, _), new_text = self._upgrade(
        ReorderAndRenameKeywordSpec(), text)
    self.assertIn(new_text, acceptable_outputs)
    self.assertNotIn("Manual check required", report)

    # Positional *args passed in that we cannot inspect, should warn
    text = "f(a, *args, kw1=c)\n"
    (_, report, _), _ = self._upgrade(ReorderAndRenameKeywordSpec(), text)
    self.assertIn("Manual check required", report)

    # **kwargs passed in that we cannot inspect, should warn
    text = "f(a, b, kw1=c, **kwargs)\n"
    (_, report, _), _ = self._upgrade(ReorderAndRenameKeywordSpec(), text)
    self.assertIn("Manual check required", report)

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
    class FooWarningSpec(ast_edits.NoUpdateSpec):
      """Usages of function attribute foo() prints out a warning."""

      def __init__(self):
        ast_edits.NoUpdateSpec.__init__(self)
        self.function_warnings = {"*.foo": (ast_edits.WARNING, "not good")}

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

  def testFullNameNode(self):
    t = ast_edits.full_name_node("a.b.c")
    self.assertEqual(
        ast.dump(t),
        "Attribute(value=Attribute(value=Name(id='a', ctx=Load()), attr='b', "
        "ctx=Load()), attr='c', ctx=Load())")

  def testImport(self):
    # foo should be renamed to bar.
    text = "import foo as f"
    expected_text = "import bar as f"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

    text = "import foo"
    expected_text = "import bar as foo"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

    text = "import foo.test"
    expected_text = "import bar.test"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

    text = "import foo.test as t"
    expected_text = "import bar.test as t"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

    text = "import foo as f, a as b"
    expected_text = "import bar as f, a as b"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

  def testFromImport(self):
    # foo should be renamed to bar.
    text = "from foo import a"
    expected_text = "from bar import a"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

    text = "from foo.a import b"
    expected_text = "from bar.a import b"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

    text = "from foo import *"
    expected_text = "from bar import *"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

    text = "from foo import a, b"
    expected_text = "from bar import a, b"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

  def testImport_NoChangeNeeded(self):
    text = "import bar as b"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(text, new_text)

  def testFromImport_NoChangeNeeded(self):
    text = "from bar import a as b"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(text, new_text)

  def testExcludedImport(self):
    # foo.baz module is excluded from changes.
    text = "import foo.baz"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(text, new_text)

    text = "import foo.baz as a"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(text, new_text)

    text = "from foo import baz as a"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(text, new_text)

    text = "from foo.baz import a"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(text, new_text)

  def testMultipleImports(self):
    text = "import foo.bar as a, foo.baz as b, foo.baz.c, foo.d"
    expected_text = "import bar.bar as a, foo.baz as b, foo.baz.c, bar.d"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

    text = "from foo import baz, a, c"
    expected_text = """from foo import baz
from bar import a, c"""
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

  def testImportInsideFunction(self):
    text = """
def t():
  from c import d
  from foo import baz, a
  from e import y
"""
    expected_text = """
def t():
  from c import d
  from foo import baz
  from bar import a
  from e import y
"""
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

  def testUpgradeInplaceWithSymlink(self):
    if os.name == "nt":
      self.skipTest("os.symlink doesn't work uniformly on Windows.")

    upgrade_dir = os.path.join(self.get_temp_dir(), "foo")
    os.mkdir(upgrade_dir)
    file_a = os.path.join(upgrade_dir, "a.py")
    file_b = os.path.join(upgrade_dir, "b.py")

    with open(file_a, "a") as f:
      f.write("import foo as f")
    os.symlink(file_a, file_b)

    upgrader = ast_edits.ASTCodeUpgrader(RenameImports())
    upgrader.process_tree_inplace(upgrade_dir)

    self.assertTrue(os.path.islink(file_b))
    self.assertEqual(file_a, os.readlink(file_b))
    with open(file_a, "r") as f:
      self.assertEqual("import bar as f", f.read())

  def testUpgradeInPlaceWithSymlinkInDifferentDir(self):
    if os.name == "nt":
      self.skipTest("os.symlink doesn't work uniformly on Windows.")

    upgrade_dir = os.path.join(self.get_temp_dir(), "foo")
    other_dir = os.path.join(self.get_temp_dir(), "bar")
    os.mkdir(upgrade_dir)
    os.mkdir(other_dir)
    file_c = os.path.join(other_dir, "c.py")
    file_d = os.path.join(upgrade_dir, "d.py")

    with open(file_c, "a") as f:
      f.write("import foo as f")
    os.symlink(file_c, file_d)

    upgrader = ast_edits.ASTCodeUpgrader(RenameImports())
    upgrader.process_tree_inplace(upgrade_dir)

    self.assertTrue(os.path.islink(file_d))
    self.assertEqual(file_c, os.readlink(file_d))
    # File pointed to by symlink is in a different directory.
    # Therefore, it should not be upgraded.
    with open(file_c, "r") as f:
      self.assertEqual("import foo as f", f.read())

  def testUpgradeCopyWithSymlink(self):
    if os.name == "nt":
      self.skipTest("os.symlink doesn't work uniformly on Windows.")

    upgrade_dir = os.path.join(self.get_temp_dir(), "foo")
    output_dir = os.path.join(self.get_temp_dir(), "bar")
    os.mkdir(upgrade_dir)
    file_a = os.path.join(upgrade_dir, "a.py")
    file_b = os.path.join(upgrade_dir, "b.py")

    with open(file_a, "a") as f:
      f.write("import foo as f")
    os.symlink(file_a, file_b)

    upgrader = ast_edits.ASTCodeUpgrader(RenameImports())
    upgrader.process_tree(upgrade_dir, output_dir, copy_other_files=True)

    new_file_a = os.path.join(output_dir, "a.py")
    new_file_b = os.path.join(output_dir, "b.py")
    self.assertTrue(os.path.islink(new_file_b))
    self.assertEqual(new_file_a, os.readlink(new_file_b))
    with open(new_file_a, "r") as f:
      self.assertEqual("import bar as f", f.read())

  def testUpgradeCopyWithSymlinkInDifferentDir(self):
    if os.name == "nt":
      self.skipTest("os.symlink doesn't work uniformly on Windows.")

    upgrade_dir = os.path.join(self.get_temp_dir(), "foo")
    other_dir = os.path.join(self.get_temp_dir(), "bar")
    output_dir = os.path.join(self.get_temp_dir(), "baz")
    os.mkdir(upgrade_dir)
    os.mkdir(other_dir)
    file_a = os.path.join(other_dir, "a.py")
    file_b = os.path.join(upgrade_dir, "b.py")

    with open(file_a, "a") as f:
      f.write("import foo as f")
    os.symlink(file_a, file_b)

    upgrader = ast_edits.ASTCodeUpgrader(RenameImports())
    upgrader.process_tree(upgrade_dir, output_dir, copy_other_files=True)

    new_file_b = os.path.join(output_dir, "b.py")
    self.assertTrue(os.path.islink(new_file_b))
    self.assertEqual(file_a, os.readlink(new_file_b))
    with open(file_a, "r") as f:
      self.assertEqual("import foo as f", f.read())


if __name__ == "__main__":
  test_lib.main()
