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
"""Upgrader for Python scripts according to an API change specification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import os
import re
import shutil
import sys
import tempfile
import traceback

import pasta
import six

# Some regular expressions we will need for parsing
FIND_OPEN = re.compile(r"^\s*(\[).*$")
FIND_STRING_CHARS = re.compile(r"['\"]")


class APIChangeSpec(object):
  """This class defines the transformations that need to happen.

  This class must provide the following fields:

  * `function_keyword_renames`: maps function names to a map of old -> new
    argument names
  * `symbol_renames`: maps function names to new function names
  * `change_to_function`: a set of function names that have changed (for
    notifications)
  * `function_reorders`: maps functions whose argument order has changed to the
    list of arguments in the new order
  * `function_warnings`: maps full names of functions to warnings that will be
    printed out if the function is used. (e.g. tf.nn.convolution())
  * `function_transformers`: maps function names to custom handlers

  For an example, see `TFAPIChangeSpec`.
  """


class _PastaEditVisitor(ast.NodeVisitor):
  """AST Visitor that processes function calls.

  Updates function calls from old API version to new API version using a given
  change spec.
  """

  def __init__(self, api_change_spec):
    self._api_change_spec = api_change_spec
    self._log = []   # Holds 3-tuples: line, col, msg.
    self._errors = []  # Same structure as _log.
    self._stack = []  # Allow easy access to parents.

  # Overridden to maintain a stack of nodes to allow for parent access
  def visit(self, node):
    self._stack.append(node)
    super(_PastaEditVisitor, self).visit(node)
    self._stack.pop()

  @property
  def errors(self):
    return self._errors

  @property
  def log(self):
    return self._log

  def _format_log(self, log):
    text = ""
    for log_entry in log:
      text += "Line %d:%d: %s\n" % log_entry
    return text

  def log_text(self):
    return self._format_log(self.log)

  def add_log(self, lineno, col, msg):
    self._log.append((lineno, col, msg))
    print("Line %d:%d: %s" % (lineno, col, msg))

  def add_error(self, lineno, col, msg):
    # All errors are also added to the regular log.
    self.add_log(lineno, col, msg)
    self._errors.append((lineno, col, msg))

  def add_logs(self, logs):
    """Record a log and print it.

    The log should be a tuple (lineno, col_offset, msg), which will be printed
    and then recorded. It is part of the log available in the self.log property.

    Args:
      logs: The log to add. Must be a tuple (lineno, col_offset, msg).
    """
    self._log.extend(logs)
    for log in logs:
      print("Line %d:%d: %s" % log)

  def add_errors(self, errors):
    """Record an error and print it.

    The error must be a tuple (lineno, col_offset, msg), which will be printed
    and then recorded as both a log and an error. It is therefore part of the
    log available in the self.log as well as the self.errors property.

    Args:
      errors: The log to add. Must be a tuple (lineno, col_offset, msg).
    """
    self.add_logs(errors)
    self._errors.extend(errors)

  def _get_applicable_entries(self, transformer_field, full_name, name):
    """Get all list entries indexed by name that apply to full_name or name."""
    # Transformers are indexed to full name, name, or no name
    # as a performance optimization.
    function_transformers = getattr(self._api_change_spec,
                                    transformer_field, {})

    glob_name = "*." + name if name else None
    transformers = []
    if full_name in function_transformers:
      transformers.append(function_transformers[full_name])
    if glob_name in function_transformers:
      transformers.append(function_transformers[glob_name])
    if "*" in function_transformers:
      transformers.append(function_transformers["*"])
    return transformers

  def _get_applicable_dict(self, transformer_field, full_name, name):
    """Get all dict entries indexed by name that apply to full_name or name."""
    # Transformers are indexed to full name, name, or no name
    # as a performance optimization.
    function_transformers = getattr(self._api_change_spec,
                                    transformer_field, {})

    glob_name = "*." + name if name else None
    transformers = function_transformers.get("*", {}).copy()
    transformers.update(function_transformers.get(glob_name, {}))
    transformers.update(function_transformers.get(full_name, {}))
    return transformers

  def _get_full_name(self, node):
    """Traverse an Attribute node to generate a full name, e.g., "tf.foo.bar".

    This is the inverse of _full_name_node.

    Args:
      node: A Node of type Attribute.

    Returns:
      a '.'-delimited full-name or None if node was not Attribute or Name.
      i.e. `foo()+b).bar` returns None, while `a.b.c` would return "a.b.c".
    """
    curr = node
    items = []
    while not isinstance(curr, ast.Name):
      if not isinstance(curr, ast.Attribute):
        return None
      items.append(curr.attr)
      curr = curr.value
    items.append(curr.id)
    return ".".join(reversed(items))

  def _full_name_node(self, name, ctx=ast.Load()):
    """Make an Attribute or Name node for name.

    Translate a qualified name into nested Attribute nodes (and a Name node).

    Args:
      name: The name to translate to a node.
      ctx: What context this name is used in. Defaults to Load()

    Returns:
      A Name or Attribute node.
    """
    names = name.split(".")
    names.reverse()
    node = ast.Name(id=names.pop(), ctx=ast.Load())
    while names:
      node = ast.Attribute(value=node, attr=names.pop(), ctx=ast.Load())

    # Change outermost ctx to the one given to us (inner ones should be Load).
    node.ctx = ctx
    return node

  def _maybe_add_warning(self, node, full_name):
    """Adds an error to be printed about full_name at node."""
    function_warnings = self._api_change_spec.function_warnings
    if full_name in function_warnings:
      warning_message = function_warnings[full_name]
      warning_message = warning_message.replace("<function name>", full_name)
      self.add_error(node.lineno, node.col_offset,
                     "%s requires manual check: %s." % (full_name,
                                                        warning_message))
      return True
    else:
      return False

  def _maybe_add_call_warning(self, node, full_name, name):
    """Print a warning when specific functions are called with selected args.

    The function _print_warning_for_function matches the full name of the called
    function, e.g., tf.foo.bar(). This function matches the function name that
    is called, as long as the function is an attribute. For example,
    `tf.foo.bar()` and `foo.bar()` are matched, but not `bar()`.

    Args:
      node: ast.Call object
      full_name: The precomputed full name of the callable, if one exists, None
        otherwise.
      name: The precomputed name of the callable, if one exists, None otherwise.

    Returns:
      Whether an error was recorded.
    """
    # Only look for *.-warnings here, the other will be handled by the Attribute
    # visitor. Also, do not warn for bare functions, only if the call func is
    # an attribute.
    warned = False
    if isinstance(node.func, ast.Attribute):
      warned = self._maybe_add_warning(node, "*." + name)

    # All arg warnings are handled here, since only we have the args
    arg_warnings = self._get_applicable_dict("function_arg_warnings",
                                             full_name, name)

    used_args = [kw.arg for kw in node.keywords]
    for (kwarg, arg), warning in arg_warnings.items():
      if kwarg in used_args or len(node.args) > arg:
        warned = True
        warning_message = warning.replace("<function name>", full_name or name)
        self.add_error(node.lineno, node.col_offset,
                       "%s called with %s argument requires manual check: %s." %
                       (full_name or name, kwarg, warning_message))

    return warned

  def _maybe_rename(self, parent, node, full_name):
    """Replace node (Attribute or Name) with a node representing full_name."""
    new_name = self._api_change_spec.symbol_renames.get(full_name, None)
    if new_name:
      self.add_log(node.lineno, node.col_offset,
                   "Renamed %r to %r" % (full_name, new_name))
      new_node = self._full_name_node(new_name, node.ctx)
      ast.copy_location(new_node, node)
      pasta.ast_utils.replace_child(parent, node, new_node)
      return True
    else:
      return False

  def _maybe_change_to_function_call(self, parent, node, full_name):
    """Wraps node (typically, an Attribute or Expr) in a Call."""
    if full_name in self._api_change_spec.change_to_function:
      if not isinstance(parent, ast.Call):
        # ast.Call's constructor is really picky about how many arguments it
        # wants, and also, it changed between Py2 and Py3.
        if six.PY2:
          new_node = ast.Call(node, [], [], None, None)
        else:
          new_node = ast.Call(node, [], [])
        pasta.ast_utils.replace_child(parent, node, new_node)
        ast.copy_location(new_node, node)
        self.add_log(node.lineno, node.col_offset,
                     "Changed %r to a function call" % full_name)
        return True
    return False

  def _maybe_add_arg_names(self, node, full_name):
    """Make args into keyword args if function called full_name requires it."""
    function_reorders = self._api_change_spec.function_reorders

    if full_name in function_reorders:
      reordered = function_reorders[full_name]
      new_keywords = []
      for idx, arg in enumerate(node.args):
        if sys.version_info[:2] >= (3, 5) and isinstance(arg, ast.Starred):
          continue  # Can't move Starred to keywords
        keyword_arg = reordered[idx]
        keyword = ast.keyword(arg=keyword_arg, value=arg)
        new_keywords.append(keyword)

      if new_keywords:
        self.add_log(node.lineno, node.col_offset,
                     "Added keywords to args of function %r" % full_name)
        node.args = []
        node.keywords = new_keywords + (node.keywords or [])
        return True
    return False

  def _maybe_modify_args(self, node, full_name, name):
    """Rename keyword args if the function called full_name requires it."""
    renamed_keywords = self._get_applicable_dict("function_keyword_renames",
                                                 full_name, name)

    if not renamed_keywords:
      return False

    modified = False
    new_keywords = []
    for keyword in node.keywords:
      argkey = keyword.arg
      if argkey in renamed_keywords:
        modified = True
        if renamed_keywords[argkey] is None:
          lineno = getattr(keyword, "lineno", node.lineno)
          col_offset = getattr(keyword, "col_offset", node.col_offset)
          self.add_log(lineno, col_offset,
                       "Removed argument %s for function %s" % (
                           argkey, full_name or name))
        else:
          keyword.arg = renamed_keywords[argkey]
          lineno = getattr(keyword, "lineno", node.lineno)
          col_offset = getattr(keyword, "col_offset", node.col_offset)
          self.add_log(lineno, col_offset,
                       "Renamed keyword argument for %s from %s to %s" % (
                           full_name, argkey, renamed_keywords[argkey]))
          new_keywords.append(keyword)
      else:
        new_keywords.append(keyword)

    if modified:
      node.keywords = new_keywords
    return modified

  def visit_Call(self, node):  # pylint: disable=invalid-name
    """Handle visiting a call node in the AST.

    Args:
      node: Current Node
    """
    assert self._stack[-1] is node

    # Get the name for this call, so we can index stuff with it.
    full_name = self._get_full_name(node.func)
    if full_name:
      name = full_name.split(".")[-1]
    elif isinstance(node.func, ast.Name):
      name = node.func.id
    elif isinstance(node.func, ast.Attribute):
      name = node.func.attr
    else:
      name = None

    # Call standard transformers for this node.
    # Make sure warnings come first, since args or names triggering warnings
    # may be removed by the other transformations.
    self._maybe_add_call_warning(node, full_name, name)
    # Make all args into kwargs
    self._maybe_add_arg_names(node, full_name)
    # Argument name changes or deletions
    self._maybe_modify_args(node, full_name, name)

    # Call transformers. These have the ability to modify the node, and if they
    # do, will return the new node they created (or the same node if they just
    # changed it). The are given the parent, but we will take care of
    # integrating their changes into the parent if they return a new node.
    #
    # These are matched on the old name, since renaming is performed by the
    # Attribute visitor, which happens later.
    transformers = self._get_applicable_entries("function_transformers",
                                                full_name, name)

    parent = self._stack[-2]

    for transformer in transformers:
      logs = []
      errors = []
      new_node = transformer(parent, node, full_name, name, logs, errors)
      self.add_logs(logs)
      self.add_errors(errors)
      if new_node:
        if new_node is not node:
          pasta.ast_utils.replace_child(parent, node, new_node)
          node = new_node
          self._stack[-1] = node

    self.generic_visit(node)

  def visit_Attribute(self, node):  # pylint: disable=invalid-name
    """Handle bare Attributes i.e. [tf.foo, tf.bar]."""
    assert self._stack[-1] is node

    full_name = self._get_full_name(node)
    if full_name:
      parent = self._stack[-2]

      # Make sure the warning comes first, otherwise the name may have changed
      self._maybe_add_warning(node, full_name)

      # Once we did a modification, node is invalid and not worth inspecting
      # further. Also, we only perform modifications for simple nodes, so
      # There'd be no point in descending further.
      if self._maybe_rename(parent, node, full_name):
        return
      if self._maybe_change_to_function_call(parent, node, full_name):
        return

    self.generic_visit(node)


class ASTCodeUpgrader(object):
  """Handles upgrading a set of Python files using a given API change spec."""

  def __init__(self, api_change_spec):
    if not isinstance(api_change_spec, APIChangeSpec):
      raise TypeError("Must pass APIChangeSpec to ASTCodeUpgrader, got %s" %
                      type(api_change_spec))
    self._api_change_spec = api_change_spec

  def process_file(self, in_filename, out_filename):
    """Process the given python file for incompatible changes.

    Args:
      in_filename: filename to parse
      out_filename: output file to write to
    Returns:
      A tuple representing number of files processed, log of actions, errors
    """

    # Write to a temporary file, just in case we are doing an implace modify.
    # pylint: disable=g-backslash-continuation
    with open(in_filename, "r") as in_file, \
        tempfile.NamedTemporaryFile("w", delete=False) as temp_file:
      ret = self.process_opened_file(in_filename, in_file, out_filename,
                                     temp_file)
    # pylint: enable=g-backslash-continuation

    shutil.move(temp_file.name, out_filename)
    return ret

  def _format_errors(self, errors, in_filename):
    return ["%s:%d:%d: %s" % ((in_filename,) + error) for error in errors]

  def update_string_pasta(self, text, in_filename):
    """Updates a file using pasta."""
    try:
      t = pasta.parse(text)
    except (SyntaxError, ValueError, TypeError):
      log = "Failed to parse.\n\n" + traceback.format_exc()
      return 0, "", log, []

    visitor = _PastaEditVisitor(self._api_change_spec)
    visitor.visit(t)

    errors = self._format_errors(visitor.errors, in_filename)
    return 1, pasta.dump(t), visitor.log_text(), errors

  def _format_log(self, log, in_filename, out_filename):
    text = "-" * 80 + "\n"
    text += "Processing file %r\n outputting to %r\n" % (in_filename,
                                                         out_filename)
    text += "-" * 80 + "\n\n"
    text += log
    text += "-" * 80 + "\n\n"
    return text

  def process_opened_file(self, in_filename, in_file, out_filename, out_file):
    """Process the given python file for incompatible changes.

    This function is split out to facilitate StringIO testing from
    tf_upgrade_test.py.

    Args:
      in_filename: filename to parse
      in_file: opened file (or StringIO)
      out_filename: output file to write to
      out_file: opened file (or StringIO)
    Returns:
      A tuple representing number of files processed, log of actions, errors
    """
    lines = in_file.readlines()
    processed_file, new_file_content, log, process_errors = (
        self.update_string_pasta("".join(lines), in_filename))

    if out_file and processed_file:
      out_file.write(new_file_content)

    return (processed_file,
            self._format_log(log, in_filename, out_filename),
            process_errors)

  def process_tree(self, root_directory, output_root_directory,
                   copy_other_files, in_place):
    """Processes upgrades on an entire tree of python files in place.

    Note that only Python files. If you have custom code in other languages,
    you will need to manually upgrade those.

    Args:
      root_directory: Directory to walk and process.
      output_root_directory: Directory to use as base.
      copy_other_files: Copy files that are not touched by this converter.
      in_place: Allow the conversion of an entire directory in place.

    Returns:
      A tuple of files processed, the report string ofr all files, and errors
    """

    if output_root_directory == root_directory:
      if in_place:
        return self.process_tree_inplace(root_directory)
      else:
        print("In order to copy a directory in place the `--inplace` input "
              "arg must be set to `True`.")
        sys.exit(1)

    # make sure output directory doesn't exist
    if output_root_directory and os.path.exists(output_root_directory):
      print("Output directory %r must not already exist." %
            (output_root_directory))
      sys.exit(1)

    # make sure output directory does not overlap with root_directory
    norm_root = os.path.split(os.path.normpath(root_directory))
    norm_output = os.path.split(os.path.normpath(output_root_directory))
    if norm_root == norm_output:
      print("Output directory %r same as input directory %r" %
            (root_directory, output_root_directory))
      sys.exit(1)

    # Collect list of files to process (we do this to correctly handle if the
    # user puts the output directory in some sub directory of the input dir)
    files_to_process = []
    files_to_copy = []
    for dir_name, _, file_list in os.walk(root_directory):
      py_files = [f for f in file_list if f.endswith(".py")]
      copy_files = [f for f in file_list if not f.endswith(".py")]
      for filename in py_files:
        fullpath = os.path.join(dir_name, filename)
        fullpath_output = os.path.join(output_root_directory,
                                       os.path.relpath(fullpath,
                                                       root_directory))
        files_to_process.append((fullpath, fullpath_output))
      if copy_other_files:
        for filename in copy_files:
          fullpath = os.path.join(dir_name, filename)
          fullpath_output = os.path.join(output_root_directory,
                                         os.path.relpath(
                                             fullpath, root_directory))
          files_to_copy.append((fullpath, fullpath_output))

    file_count = 0
    tree_errors = []
    report = ""
    report += ("=" * 80) + "\n"
    report += "Input tree: %r\n" % root_directory
    report += ("=" * 80) + "\n"

    for input_path, output_path in files_to_process:
      output_directory = os.path.dirname(output_path)
      if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
      file_count += 1
      _, l_report, l_errors = self.process_file(input_path, output_path)
      tree_errors += l_errors
      report += l_report
    for input_path, output_path in files_to_copy:
      output_directory = os.path.dirname(output_path)
      if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
      shutil.copy(input_path, output_path)
    return file_count, report, tree_errors

  def process_tree_inplace(self, root_directory):
    """Process a directory of python files in place."""
    files_to_process = []
    for dir_name, _, file_list in os.walk(root_directory):
      py_files = [os.path.join(dir_name,
                               f) for f in file_list if f.endswith(".py")]
      files_to_process += py_files

    file_count = 0
    tree_errors = []
    report = ""
    report += ("=" * 80) + "\n"
    report += "Input tree: %r\n" % root_directory
    report += ("=" * 80) + "\n"

    for path in files_to_process:
      file_count += 1
      _, l_report, l_errors = self.process_file(path, path)
      tree_errors += l_errors
      report += l_report

    return file_count, report, tree_errors
