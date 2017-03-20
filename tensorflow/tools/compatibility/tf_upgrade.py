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
"""Upgrader for Python scripts from pre-1.0 TensorFlow to 1.0 TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import ast
import collections
import os
import shutil
import sys
import tempfile
import traceback


class APIChangeSpec(object):
  """List of maps that describe what changed in the API."""

  def __init__(self):
    # Maps from a function name to a dictionary that describes how to
    # map from an old argument keyword to the new argument keyword.
    self.function_keyword_renames = {
        "tf.count_nonzero": {
            "reduction_indices": "axis"
        },
        "tf.reduce_all": {
            "reduction_indices": "axis"
        },
        "tf.reduce_any": {
            "reduction_indices": "axis"
        },
        "tf.reduce_max": {
            "reduction_indices": "axis"
        },
        "tf.reduce_mean": {
            "reduction_indices": "axis"
        },
        "tf.reduce_min": {
            "reduction_indices": "axis"
        },
        "tf.reduce_prod": {
            "reduction_indices": "axis"
        },
        "tf.reduce_sum": {
            "reduction_indices": "axis"
        },
        "tf.reduce_logsumexp": {
            "reduction_indices": "axis"
        },
        "tf.expand_dims": {
            "dim": "axis"
        },
        "tf.argmax": {
            "dimension": "axis"
        },
        "tf.argmin": {
            "dimension": "axis"
        },
        "tf.reduce_join": {
            "reduction_indices": "axis"
        },
        "tf.sparse_concat": {
            "concat_dim": "axis"
        },
        "tf.sparse_split": {
            "split_dim": "axis"
        },
        "tf.sparse_reduce_sum": {
            "reduction_axes": "axis"
        },
        "tf.reverse_sequence": {
            "seq_dim": "seq_axis",
            "batch_dim": "batch_axis"
        },
        "tf.sparse_reduce_sum_sparse": {
            "reduction_axes": "axis"
        },
        "tf.squeeze": {
            "squeeze_dims": "axis"
        },
        "tf.split": {
            "split_dim": "axis",
            "num_split": "num_or_size_splits"
        },
        "tf.concat": {
            "concat_dim": "axis"
        },
    }

    # Mapping from function to the new name of the function
    self.function_renames = {
        "tf.inv": "tf.reciprocal",
        "tf.contrib.deprecated.scalar_summary": "tf.summary.scalar",
        "tf.contrib.deprecated.histogram_summary": "tf.summary.histogram",
        "tf.listdiff": "tf.setdiff1d",
        "tf.list_diff": "tf.setdiff1d",
        "tf.mul": "tf.multiply",
        "tf.neg": "tf.negative",
        "tf.sub": "tf.subtract",
        "tf.train.SummaryWriter": "tf.summary.FileWriter",
        "tf.scalar_summary": "tf.summary.scalar",
        "tf.histogram_summary": "tf.summary.histogram",
        "tf.audio_summary": "tf.summary.audio",
        "tf.image_summary": "tf.summary.image",
        "tf.merge_summary": "tf.summary.merge",
        "tf.merge_all_summaries": "tf.summary.merge_all",
        "tf.image.per_image_whitening": "tf.image.per_image_standardization",
        "tf.all_variables": "tf.global_variables",
        "tf.VARIABLES": "tf.GLOBAL_VARIABLES",
        "tf.initialize_all_variables": "tf.global_variables_initializer",
        "tf.initialize_variables": "tf.variables_initializer",
        "tf.initialize_local_variables": "tf.local_variables_initializer",
        "tf.batch_matrix_diag": "tf.matrix_diag",
        "tf.batch_band_part": "tf.band_part",
        "tf.batch_set_diag": "tf.set_diag",
        "tf.batch_matrix_transpose": "tf.matrix_transpose",
        "tf.batch_matrix_determinant": "tf.matrix_determinant",
        "tf.batch_matrix_inverse": "tf.matrix_inverse",
        "tf.batch_cholesky": "tf.cholesky",
        "tf.batch_cholesky_solve": "tf.cholesky_solve",
        "tf.batch_matrix_solve": "tf.matrix_solve",
        "tf.batch_matrix_triangular_solve": "tf.matrix_triangular_solve",
        "tf.batch_matrix_solve_ls": "tf.matrix_solve_ls",
        "tf.batch_self_adjoint_eig": "tf.self_adjoint_eig",
        "tf.batch_self_adjoint_eigvals": "tf.self_adjoint_eigvals",
        "tf.batch_svd": "tf.svd",
        "tf.batch_fft": "tf.fft",
        "tf.batch_ifft": "tf.ifft",
        "tf.batch_fft2d": "tf.fft2d",
        "tf.batch_ifft2d": "tf.ifft2d",
        "tf.batch_fft3d": "tf.fft3d",
        "tf.batch_ifft3d": "tf.ifft3d",
        "tf.select": "tf.where",
        "tf.complex_abs": "tf.abs",
        "tf.batch_matmul": "tf.matmul",
        "tf.pack": "tf.stack",
        "tf.unpack": "tf.unstack",
    }

    self.change_to_function = {
        "tf.ones_initializer",
        "tf.zeros_initializer",
    }

    # Functions that were reordered should be changed to the new keyword args
    # for safety, if positional arguments are used. If you have reversed the
    # positional arguments yourself, this could do the wrong thing.
    self.function_reorders = {
        "tf.split": ["axis", "num_or_size_splits", "value", "name"],
        "tf.sparse_split": ["axis", "num_or_size_splits", "value", "name"],
        "tf.concat": ["concat_dim", "values", "name"],
        "tf.svd": ["tensor", "compute_uv", "full_matrices", "name"],
        "tf.nn.softmax_cross_entropy_with_logits": [
            "logits", "labels", "dim", "name"],
        "tf.nn.sparse_softmax_cross_entropy_with_logits": [
            "logits", "labels", "name"],
        "tf.nn.sigmoid_cross_entropy_with_logits": [
            "logits", "labels", "name"]
    }

    # Specially handled functions.
    self.function_handle = {"tf.reverse": self._reverse_handler}

  @staticmethod
  def _reverse_handler(file_edit_recorder, node):
    # TODO(aselle): Could check for a literal list of bools and try to convert
    # them to indices.
    comment = ("ERROR: tf.reverse has had its argument semantics changed\n"
               "significantly the converter cannot detect this reliably, so you"
               "need to inspect this usage manually.\n")
    file_edit_recorder.add(comment,
                           node.lineno,
                           node.col_offset,
                           "tf.reverse",
                           "tf.reverse",
                           error="tf.reverse requires manual check.")


class FileEditTuple(collections.namedtuple(
    "FileEditTuple", ["comment", "line", "start", "old", "new"])):
  """Each edit that is recorded by a FileEditRecorder.

  Fields:
    comment: A description of the edit and why it was made.
    line: The line number in the file where the edit occurs (1-indexed).
    start: The line number in the file where the edit occurs (0-indexed).
    old: text string to remove (this must match what was in file).
    new: text string to add in place of `old`.
  """

  __slots__ = ()


class FileEditRecorder(object):
  """Record changes that need to be done to the file."""

  def __init__(self, filename):
    # all edits are lists of chars
    self._filename = filename

    self._line_to_edit = collections.defaultdict(list)
    self._errors = []

  def process(self, text):
    """Process a list of strings, each corresponding to the recorded changes.

    Args:
      text: A list of lines of text (assumed to contain newlines)
    Returns:
      A tuple of the modified text and a textual description of what is done.
    Raises:
      ValueError: if substitution source location does not have expected text.
    """

    change_report = ""

    # Iterate of each line
    for line, edits in self._line_to_edit.items():
      offset = 0
      # sort by column so that edits are processed in order in order to make
      # indexing adjustments cumulative for changes that change the string
      # length
      edits.sort(key=lambda x: x.start)

      # Extract each line to a list of characters, because mutable lists
      # are editable, unlike immutable strings.
      char_array = list(text[line - 1])

      # Record a description of the change
      change_report += "%r Line %d\n" % (self._filename, line)
      change_report += "-" * 80 + "\n\n"
      for e in edits:
        change_report += "%s\n" % e.comment
      change_report += "\n    Old: %s" % (text[line - 1])

      # Make underscore buffers for underlining where in the line the edit was
      change_list = [" "] * len(text[line - 1])
      change_list_new = [" "] * len(text[line - 1])

      # Iterate for each edit
      for e in edits:
        # Create effective start, end by accounting for change in length due
        # to previous edits
        start_eff = e.start + offset
        end_eff = start_eff + len(e.old)

        # Make sure the edit is changing what it should be changing
        old_actual = "".join(char_array[start_eff:end_eff])
        if old_actual != e.old:
          raise ValueError("Expected text %r but got %r" %
                           ("".join(e.old), "".join(old_actual)))
        # Make the edit
        char_array[start_eff:end_eff] = list(e.new)

        # Create the underline highlighting of the before and after
        change_list[e.start:e.start + len(e.old)] = "~" * len(e.old)
        change_list_new[start_eff:end_eff] = "~" * len(e.new)

        # Keep track of how to generate effective ranges
        offset += len(e.new) - len(e.old)

      # Finish the report comment
      change_report += "         %s\n" % "".join(change_list)
      text[line - 1] = "".join(char_array)
      change_report += "    New: %s" % (text[line - 1])
      change_report += "         %s\n\n" % "".join(change_list_new)
    return "".join(text), change_report, self._errors

  def add(self, comment, line, start, old, new, error=None):
    """Add a new change that is needed.

    Args:
      comment: A description of what was changed
      line: Line number (1 indexed)
      start: Column offset (0 indexed)
      old: old text
      new: new text
      error: this "edit" is something that cannot be fixed automatically
    Returns:
      None
    """

    self._line_to_edit[line].append(
        FileEditTuple(comment, line, start, old, new))
    if error:
      self._errors.append("%s:%d: %s" % (self._filename, line, error))


class TensorFlowCallVisitor(ast.NodeVisitor):
  """AST Visitor that finds TensorFlow Function calls.

  Updates function calls from old API version to new API version.
  """

  def __init__(self, filename, lines):
    self._filename = filename
    self._file_edit = FileEditRecorder(filename)
    self._lines = lines
    self._api_change_spec = APIChangeSpec()

  def process(self, lines):
    return self._file_edit.process(lines)

  def generic_visit(self, node):
    ast.NodeVisitor.generic_visit(self, node)

  def _rename_functions(self, node, full_name):
    function_renames = self._api_change_spec.function_renames
    try:
      new_name = function_renames[full_name]
      self._file_edit.add("Renamed function %r to %r" % (full_name,
                                                         new_name),
                          node.lineno, node.col_offset, full_name, new_name)
    except KeyError:
      pass

  def _get_attribute_full_path(self, node):
    """Traverse an attribute to generate a full name e.g. tf.foo.bar.

    Args:
      node: A Node of type Attribute.

    Returns:
      a '.'-delimited full-name or None if the tree was not a simple form.
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

  def _find_true_position(self, node):
    """Return correct line number and column offset for a given node.

    This is necessary mainly because ListComp's location reporting reports
    the next token after the list comprehension list opening.

    Args:
      node: Node for which we wish to know the lineno and col_offset
    """
    import re
    find_open = re.compile("^\s*(\\[).*$")
    find_string_chars = re.compile("['\"]")

    if isinstance(node, ast.ListComp):
      # Strangely, ast.ListComp returns the col_offset of the first token
      # after the '[' token which appears to be a bug. Workaround by
      # explicitly finding the real start of the list comprehension.
      line = node.lineno
      col = node.col_offset
      # loop over lines
      while 1:
        # Reverse the text to and regular expression search for whitespace
        text = self._lines[line-1]
        reversed_preceding_text = text[:col][::-1]
        # First find if a [ can be found with only whitespace between it and
        # col.
        m = find_open.match(reversed_preceding_text)
        if m:
          new_col_offset = col - m.start(1) - 1
          return line, new_col_offset
        else:
          if (reversed_preceding_text=="" or
             reversed_preceding_text.isspace()):
            line = line - 1
            prev_line = self._lines[line - 1]
            # TODO(aselle):
            # this is poor comment detection, but it is good enough for
            # cases where the comment does not contain string literal starting/
            # ending characters. If ast gave us start and end locations of the
            # ast nodes rather than just start, we could use string literal
            # node ranges to filter out spurious #'s that appear in string
            # literals.
            comment_start = prev_line.find("#")
            if comment_start ==  -1:
              col = len(prev_line) -1
            elif find_string_chars.search(prev_line[comment_start:]) is None:
              col = comment_start
            else:
              return None, None
          else:
            return None, None
    # Most other nodes return proper locations (with notably does not), but
    # it is not possible to use that in an argument.
    return node.lineno, node.col_offset


  def visit_Call(self, node):  # pylint: disable=invalid-name
    """Handle visiting a call node in the AST.

    Args:
      node: Current Node
    """


    # Find a simple attribute name path e.g. "tf.foo.bar"
    full_name = self._get_attribute_full_path(node.func)

    # Make sure the func is marked as being part of a call
    node.func.is_function_for_call = True

    if full_name and full_name.startswith("tf."):
      # Call special handlers
      function_handles = self._api_change_spec.function_handle
      if full_name in function_handles:
        function_handles[full_name](self._file_edit, node)

      # Examine any non-keyword argument and make it into a keyword argument
      # if reordering required.
      function_reorders = self._api_change_spec.function_reorders
      function_keyword_renames = (
          self._api_change_spec.function_keyword_renames)

      if full_name in function_reorders:
        reordered = function_reorders[full_name]
        for idx, arg in enumerate(node.args):
          lineno, col_offset = self._find_true_position(arg)
          if lineno is None or col_offset is None:
            self._file_edit.add(
                "Failed to add keyword %r to reordered function %r"
                % (reordered[idx], full_name), arg.lineno, arg.col_offset,
                "", "",
                error="A necessary keyword argument failed to be inserted.")
          else:
            keyword_arg = reordered[idx]
            if (full_name in function_keyword_renames and
                keyword_arg in function_keyword_renames[full_name]):
              keyword_arg = function_keyword_renames[full_name][keyword_arg]
            self._file_edit.add("Added keyword %r to reordered function %r"
                                % (reordered[idx], full_name), lineno,
                                col_offset, "", keyword_arg + "=")

      # Examine each keyword argument and convert it to the final renamed form
      renamed_keywords = ({} if full_name not in function_keyword_renames else
                          function_keyword_renames[full_name])
      for keyword in node.keywords:
        argkey = keyword.arg
        argval = keyword.value

        if argkey in renamed_keywords:
          argval_lineno, argval_col_offset = self._find_true_position(argval)
          if (argval_lineno is not None and argval_col_offset is not None):
            # TODO(aselle): We should scan backward to find the start of the
            # keyword key. Unfortunately ast does not give you the location of
            # keyword keys, so we are forced to infer it from the keyword arg
            # value.
            key_start = argval_col_offset - len(argkey) - 1
            key_end = key_start + len(argkey) + 1
            if self._lines[argval_lineno - 1][key_start:key_end] == argkey + "=":
              self._file_edit.add("Renamed keyword argument from %r to %r" %
                              (argkey, renamed_keywords[argkey]),
                              argval_lineno,
                              argval_col_offset - len(argkey) - 1,
                              argkey + "=", renamed_keywords[argkey] + "=")
              continue
          self._file_edit.add(
              "Failed to rename keyword argument from %r to %r" %
              (argkey, renamed_keywords[argkey]),
              argval.lineno,
              argval.col_offset - len(argkey) - 1,
              "", "",
              error="Failed to find keyword lexographically. Fix manually.")

    ast.NodeVisitor.generic_visit(self, node)

  def visit_Attribute(self, node):  # pylint: disable=invalid-name
    """Handle bare Attributes i.e. [tf.foo, tf.bar].

    Args:
      node: Node that is of type ast.Attribute
    """
    full_name = self._get_attribute_full_path(node)
    if full_name and full_name.startswith("tf."):
      self._rename_functions(node, full_name)
    if full_name in self._api_change_spec.change_to_function:
      if not hasattr(node, "is_function_for_call"):
        new_text = full_name + "()"
        self._file_edit.add("Changed %r to %r"%(full_name, new_text),
                            node.lineno, node.col_offset, full_name, new_text)

    ast.NodeVisitor.generic_visit(self, node)


class TensorFlowCodeUpgrader(object):
  """Class that handles upgrading a set of Python files to TensorFlow 1.0."""

  def __init__(self):
    pass

  def process_file(self, in_filename, out_filename):
    """Process the given python file for incompatible changes.

    Args:
      in_filename: filename to parse
      out_filename: output file to write to
    Returns:
      A tuple representing number of files processed, log of actions, errors
    """

    # Write to a temporary file, just in case we are doing an implace modify.
    with open(in_filename, "r") as in_file, \
        tempfile.NamedTemporaryFile("w", delete=False) as temp_file:
      ret = self.process_opened_file(
          in_filename, in_file, out_filename, temp_file)

    shutil.move(temp_file.name, out_filename)
    return ret

  # Broad exceptions are required here because ast throws whatever it wants.
  # pylint: disable=broad-except
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
    process_errors = []
    text = "-" * 80 + "\n"
    text += "Processing file %r\n outputting to %r\n" % (in_filename,
                                                         out_filename)
    text += "-" * 80 + "\n\n"

    parsed_ast = None
    lines = in_file.readlines()
    try:
      parsed_ast = ast.parse("".join(lines))
    except Exception:
      text += "Failed to parse %r\n\n" % in_filename
      text += traceback.format_exc()
    if parsed_ast:
      visitor = TensorFlowCallVisitor(in_filename, lines)
      visitor.visit(parsed_ast)
      out_text, new_text, process_errors = visitor.process(lines)
      text += new_text
      if out_file:
        out_file.write(out_text)
    text += "\n"
    return 1, text, process_errors
  # pylint: enable=broad-except

  def process_tree(self, root_directory, output_root_directory, copy_other_files):
    """Processes upgrades on an entire tree of python files in place.

    Note that only Python files. If you have custom code in other languages,
    you will need to manually upgrade those.

    Args:
      root_directory: Directory to walk and process.
      output_root_directory: Directory to use as base
    Returns:
      A tuple of files processed, the report string ofr all files, and errors
    """

    # make sure output directory doesn't exist
    if output_root_directory and os.path.exists(output_root_directory):
      print("Output directory %r must not already exist." % (
          output_root_directory))
      sys.exit(1)

    # make sure output directory does not overlap with root_directory
    norm_root = os.path.split(os.path.normpath(root_directory))
    norm_output = os.path.split(os.path.normpath(output_root_directory))
    if norm_root == norm_output:
      print("Output directory %r same as input directory %r" % (
          root_directory, output_root_directory))
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
        fullpath_output = os.path.join(
            output_root_directory, os.path.relpath(fullpath, root_directory))
        files_to_process.append((fullpath, fullpath_output))
      if copy_other_files:
        for filename in copy_files:
          fullpath = os.path.join(dir_name, filename)
          fullpath_output = os.path.join(
              output_root_directory, os.path.relpath(fullpath, root_directory))
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


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      formatter_class=argparse.RawDescriptionHelpFormatter,
      description="""Convert a TensorFlow Python file to 1.0

Simple usage:
  tf_convert.py --infile foo.py --outfile bar.py
  tf_convert.py --intree ~/code/old --outtree ~/code/new
""")
  parser.add_argument(
      "--infile",
      dest="input_file",
      help="If converting a single file, the name of the file "
      "to convert")
  parser.add_argument(
      "--outfile",
      dest="output_file",
      help="If converting a single file, the output filename.")
  parser.add_argument(
      "--intree",
      dest="input_tree",
      help="If converting a whole tree of files, the directory "
      "to read from (relative or absolute).")
  parser.add_argument(
      "--outtree",
      dest="output_tree",
      help="If converting a whole tree of files, the output "
      "directory (relative or absolute).")
  parser.add_argument(
      "--copyotherfiles",
      dest="copy_other_files",
      help=("If converting a whole tree of files, whether to "
            "copy the other files."),
      type=bool,
      default=False)
  parser.add_argument(
      "--reportfile",
      dest="report_filename",
      help=("The name of the file where the report log is "
            "stored."
            "(default: %(default)s)"),
      default="report.txt")
  args = parser.parse_args()

  upgrade = TensorFlowCodeUpgrader()
  report_text = None
  report_filename = args.report_filename
  files_processed = 0
  if args.input_file:
    files_processed, report_text, errors = upgrade.process_file(
        args.input_file, args.output_file)
    files_processed = 1
  elif args.input_tree:
    files_processed, report_text, errors = upgrade.process_tree(
        args.input_tree, args.output_tree, args.copy_other_files)
  else:
    parser.print_help()
  if report_text:
    open(report_filename, "w").write(report_text)
    print("TensorFlow 1.0 Upgrade Script")
    print("-----------------------------")
    print("Converted %d files\n" % files_processed)
    print("Detected %d errors that require attention" % len(errors))
    print("-" * 80)
    print("\n".join(errors))
    print("\nMake sure to read the detailed log %r\n" % report_filename)
