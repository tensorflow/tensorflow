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
"""Unit tests for source_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import os
import sys
import tempfile
import zipfile

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
# Import resource_variable_ops for the variables-to-tensor implicit conversion.
from tensorflow.python.ops import resource_variable_ops  # pylint: disable=unused-import
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.util import tf_inspect


def line_number_above():
  """Get lineno of the AST node immediately above this function's call site.

  It is assumed that there is no empty line(s) between the call site and the
  preceding AST node.

  Returns:
    The lineno of the preceding AST node, at the same level of the AST.
    If the preceding AST spans multiple lines:
      - In Python 3.8+, the lineno of the first line is returned.
      - In older Python versions, the lineno of the last line is returned.
  """
  # https://bugs.python.org/issue12458: In Python 3.8, traceback started
  # to return the lineno of the first line of a multi-line continuation block,
  # instead of that of the last line. Therefore, in Python 3.8+, we use `ast` to
  # get the lineno of the first line.
  call_site_lineno = tf_inspect.stack()[1][2]
  if sys.version_info < (3, 8):
    return call_site_lineno - 1
  else:
    with open(__file__, "rb") as f:
      source_text = f.read().decode("utf-8")
    source_tree = ast.parse(source_text)
    prev_node = _find_preceding_ast_node(source_tree, call_site_lineno)
    return prev_node.lineno


def _find_preceding_ast_node(node, lineno):
  """Find the ast node immediately before and not including lineno."""
  for i, child_node in enumerate(node.body):
    if child_node.lineno == lineno:
      return node.body[i - 1]
    if hasattr(child_node, "body"):
      found_node = _find_preceding_ast_node(child_node, lineno)
      if found_node:
        return found_node


class GuessIsTensorFlowLibraryTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.curr_file_path = os.path.normpath(os.path.abspath(__file__))

  def tearDown(self):
    ops.reset_default_graph()

  def testGuessedBaseDirIsProbablyCorrect(self):
    # In the non-pip world, code resides in "tensorflow/"
    # In the pip world, after virtual pip, code resides in "tensorflow_core/"
    # So, we have to check both of them
    self.assertIn(
        os.path.basename(source_utils._TENSORFLOW_BASEDIR),
        ["tensorflow", "tensorflow_core"])

  def testUnitTestFileReturnsFalse(self):
    self.assertFalse(
        source_utils.guess_is_tensorflow_py_library(self.curr_file_path))

  def testSourceUtilModuleReturnsTrue(self):
    self.assertTrue(
        source_utils.guess_is_tensorflow_py_library(source_utils.__file__))

  @test_util.run_deprecated_v1
  def testFileInPythonKernelsPathReturnsTrue(self):
    x = constant_op.constant(42.0, name="x")
    self.assertTrue(
        source_utils.guess_is_tensorflow_py_library(x.op.traceback[-1][0]))

  def testDebuggerExampleFilePathReturnsFalse(self):
    self.assertFalse(
        source_utils.guess_is_tensorflow_py_library(os.path.normpath(
            "site-packages/tensorflow/python/debug/examples/debug_mnist.py")))
    self.assertFalse(
        source_utils.guess_is_tensorflow_py_library(os.path.normpath(
            "site-packages/tensorflow/python/debug/examples/v1/example_v1.py")))
    self.assertFalse(
        source_utils.guess_is_tensorflow_py_library(os.path.normpath(
            "site-packages/tensorflow/python/debug/examples/v2/example_v2.py")))
    self.assertFalse(
        source_utils.guess_is_tensorflow_py_library(os.path.normpath(
            "site-packages/tensorflow/python/debug/examples/v3/example_v3.py")))

  def testNonPythonFileRaisesException(self):
    with self.assertRaisesRegexp(ValueError, r"is not a Python source file"):
      source_utils.guess_is_tensorflow_py_library(
          os.path.join(os.path.dirname(self.curr_file_path), "foo.cc"))


class SourceHelperTest(test_util.TensorFlowTestCase):

  def createAndRunGraphHelper(self):
    """Create and run a TensorFlow Graph to generate debug dumps.

    This is intentionally done in separate method, to make it easier to test
    the stack-top mode of source annotation.
    """

    self.dump_root = self.get_temp_dir()
    self.curr_file_path = os.path.abspath(
        tf_inspect.getfile(tf_inspect.currentframe()))

    # Run a simple TF graph to generate some debug dumps that can be used in
    # source annotation.
    with session.Session() as sess:
      self.u_init = constant_op.constant(
          np.array([[5.0, 3.0], [-1.0, 0.0]]), shape=[2, 2], name="u_init")
      self.u_init_line_number = line_number_above()

      self.u = variables.Variable(self.u_init, name="u")
      self.u_line_number = line_number_above()

      self.v_init = constant_op.constant(
          np.array([[2.0], [-1.0]]), shape=[2, 1], name="v_init")
      self.v_init_line_number = line_number_above()

      self.v = variables.Variable(self.v_init, name="v")
      self.v_line_number = line_number_above()

      self.w = math_ops.matmul(self.u, self.v, name="w")
      self.w_line_number = line_number_above()

      self.evaluate(self.u.initializer)
      self.evaluate(self.v.initializer)

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options, sess.graph, debug_urls=["file://%s" % self.dump_root])
      run_metadata = config_pb2.RunMetadata()
      sess.run(self.w, options=run_options, run_metadata=run_metadata)

      self.dump = debug_data.DebugDumpDir(
          self.dump_root, partition_graphs=run_metadata.partition_graphs)
      self.dump.set_python_graph(sess.graph)

  def setUp(self):
    self.createAndRunGraphHelper()
    self.helper_line_number = line_number_above()

  def tearDown(self):
    if os.path.isdir(self.dump_root):
      file_io.delete_recursively(self.dump_root)
    ops.reset_default_graph()

  def testAnnotateWholeValidSourceFileGivesCorrectResult(self):
    source_annotation = source_utils.annotate_source(self.dump,
                                                     self.curr_file_path)

    self.assertIn(self.u_init.op.name,
                  source_annotation[self.u_init_line_number])
    self.assertIn(self.u.op.name, source_annotation[self.u_line_number])
    self.assertIn(self.v_init.op.name,
                  source_annotation[self.v_init_line_number])
    self.assertIn(self.v.op.name, source_annotation[self.v_line_number])
    self.assertIn(self.w.op.name, source_annotation[self.w_line_number])

    # In the non-stack-top (default) mode, the helper line should be annotated
    # with all the ops as well.
    self.assertIn(self.u_init.op.name,
                  source_annotation[self.helper_line_number])
    self.assertIn(self.u.op.name, source_annotation[self.helper_line_number])
    self.assertIn(self.v_init.op.name,
                  source_annotation[self.helper_line_number])
    self.assertIn(self.v.op.name, source_annotation[self.helper_line_number])
    self.assertIn(self.w.op.name, source_annotation[self.helper_line_number])

  def testAnnotateWithStackTopGivesCorrectResult(self):
    source_annotation = source_utils.annotate_source(
        self.dump, self.curr_file_path, file_stack_top=True)

    self.assertIn(self.u_init.op.name,
                  source_annotation[self.u_init_line_number])
    self.assertIn(self.u.op.name, source_annotation[self.u_line_number])
    self.assertIn(self.v_init.op.name,
                  source_annotation[self.v_init_line_number])
    self.assertIn(self.v.op.name, source_annotation[self.v_line_number])
    self.assertIn(self.w.op.name, source_annotation[self.w_line_number])

    # In the stack-top mode, the helper line should not have been annotated.
    self.assertNotIn(self.helper_line_number, source_annotation)

  def testAnnotateSubsetOfLinesGivesCorrectResult(self):
    source_annotation = source_utils.annotate_source(
        self.dump,
        self.curr_file_path,
        min_line=self.u_line_number,
        max_line=self.u_line_number + 1)

    self.assertIn(self.u.op.name, source_annotation[self.u_line_number])
    self.assertNotIn(self.v_line_number, source_annotation)

  def testAnnotateDumpedTensorsGivesCorrectResult(self):
    source_annotation = source_utils.annotate_source(
        self.dump, self.curr_file_path, do_dumped_tensors=True)

    # Note: Constant Tensors u_init and v_init may not get dumped due to
    #   constant-folding.
    self.assertIn(self.u.name, source_annotation[self.u_line_number])
    self.assertIn(self.v.name, source_annotation[self.v_line_number])
    self.assertIn(self.w.name, source_annotation[self.w_line_number])

    self.assertNotIn(self.u.op.name, source_annotation[self.u_line_number])
    self.assertNotIn(self.v.op.name, source_annotation[self.v_line_number])
    self.assertNotIn(self.w.op.name, source_annotation[self.w_line_number])

    self.assertIn(self.u.name, source_annotation[self.helper_line_number])
    self.assertIn(self.v.name, source_annotation[self.helper_line_number])
    self.assertIn(self.w.name, source_annotation[self.helper_line_number])

  def testCallingAnnotateSourceWithoutPythonGraphRaisesException(self):
    self.dump.set_python_graph(None)
    with self.assertRaises(ValueError):
      source_utils.annotate_source(self.dump, self.curr_file_path)

  def testCallingAnnotateSourceOnUnrelatedSourceFileDoesNotError(self):
    # Create an unrelated source file.
    unrelated_source_path = tempfile.mktemp()
    with open(unrelated_source_path, "wt") as source_file:
      source_file.write("print('hello, world')\n")

    self.assertEqual({},
                     source_utils.annotate_source(self.dump,
                                                  unrelated_source_path))

    # Clean up unrelated source file.
    os.remove(unrelated_source_path)

  def testLoadingPythonSourceFileWithNonAsciiChars(self):
    source_path = tempfile.mktemp()
    with open(source_path, "wb") as source_file:
      source_file.write(u"print('\U0001f642')\n".encode("utf-8"))
    source_lines, _ = source_utils.load_source(source_path)
    self.assertEqual(source_lines, [u"print('\U0001f642')", u""])
    # Clean up unrelated source file.
    os.remove(source_path)

  def testLoadNonexistentNonParPathFailsWithIOError(self):
    bad_path = os.path.join(self.get_temp_dir(), "nonexistent.py")
    with self.assertRaisesRegexp(
        IOError, "neither exists nor can be loaded.*par.*"):
      source_utils.load_source(bad_path)

  def testLoadingPythonSourceFileInParFileSucceeds(self):
    # Create the .par file first.
    temp_file_path = os.path.join(self.get_temp_dir(), "model.py")
    with open(temp_file_path, "wb") as f:
      f.write(b"import tensorflow as tf\nx = tf.constant(42.0)\n")
    par_path = os.path.join(self.get_temp_dir(), "train_model.par")
    with zipfile.ZipFile(par_path, "w") as zf:
      zf.write(temp_file_path, os.path.join("tensorflow_models", "model.py"))

    source_path = os.path.join(par_path, "tensorflow_models", "model.py")
    source_lines, _ = source_utils.load_source(source_path)
    self.assertEqual(
        source_lines, ["import tensorflow as tf", "x = tf.constant(42.0)", ""])

  def testLoadingPythonSourceFileInParFileFailsRaisingIOError(self):
    # Create the .par file first.
    temp_file_path = os.path.join(self.get_temp_dir(), "model.py")
    with open(temp_file_path, "wb") as f:
      f.write(b"import tensorflow as tf\nx = tf.constant(42.0)\n")
    par_path = os.path.join(self.get_temp_dir(), "train_model.par")
    with zipfile.ZipFile(par_path, "w") as zf:
      zf.write(temp_file_path, os.path.join("tensorflow_models", "model.py"))

    source_path = os.path.join(par_path, "tensorflow_models", "nonexistent.py")
    with self.assertRaisesRegexp(
        IOError, "neither exists nor can be loaded.*par.*"):
      source_utils.load_source(source_path)


@test_util.run_v1_only("b/120545219")
class ListSourceAgainstDumpTest(test_util.TensorFlowTestCase):

  def createAndRunGraphWithWhileLoop(self):
    """Create and run a TensorFlow Graph with a while loop to generate dumps."""

    self.dump_root = self.get_temp_dir()
    self.curr_file_path = os.path.abspath(
        tf_inspect.getfile(tf_inspect.currentframe()))

    # Run a simple TF graph to generate some debug dumps that can be used in
    # source annotation.
    with session.Session() as sess:
      loop_body = lambda i: math_ops.add(i, 2)
      self.traceback_first_line = line_number_above()

      loop_cond = lambda i: math_ops.less(i, 16)

      i = constant_op.constant(10, name="i")
      loop = control_flow_ops.while_loop(loop_cond, loop_body, [i])

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options, sess.graph, debug_urls=["file://%s" % self.dump_root])
      run_metadata = config_pb2.RunMetadata()
      sess.run(loop, options=run_options, run_metadata=run_metadata)

      self.dump = debug_data.DebugDumpDir(
          self.dump_root, partition_graphs=run_metadata.partition_graphs)
      self.dump.set_python_graph(sess.graph)

  def setUp(self):
    self.createAndRunGraphWithWhileLoop()

  def tearDown(self):
    if os.path.isdir(self.dump_root):
      file_io.delete_recursively(self.dump_root)
    ops.reset_default_graph()

  def testGenerateSourceList(self):
    source_list = source_utils.list_source_files_against_dump(self.dump)

    # Assert that the file paths are sorted and unique.
    file_paths = [item[0] for item in source_list]
    self.assertEqual(sorted(file_paths), file_paths)
    self.assertEqual(len(set(file_paths)), len(file_paths))

    # Assert that each item of source_list has length 6.
    for item in source_list:
      self.assertTrue(isinstance(item, tuple))
      self.assertEqual(6, len(item))

    # The while loop body should have executed 3 times. The following table
    # lists the tensors and how many times each of them is dumped.
    #   Tensor name            # of times dumped:
    #   i:0                    1
    #   while/Enter:0          1
    #   while/Merge:0          4
    #   while/Merge:1          4
    #   while/Less/y:0         4
    #   while/Less:0           4
    #   while/LoopCond:0       4
    #   while/Switch:0         1
    #   while/Switch:1         3
    #   while/Identity:0       3
    #   while/Add/y:0          3
    #   while/Add:0            3
    #   while/NextIteration:0  3
    #   while/Exit:0           1
    # ----------------------------
    #   (Total)                39
    #
    # The total number of nodes is 12.
    # The total number of tensors is 14 (2 of the nodes have 2 outputs:
    #   while/Merge, while/Switch).

    _, is_tf_py_library, num_nodes, num_tensors, num_dumps, first_line = (
        source_list[file_paths.index(self.curr_file_path)])
    self.assertFalse(is_tf_py_library)
    self.assertEqual(12, num_nodes)
    self.assertEqual(14, num_tensors)
    self.assertEqual(39, num_dumps)
    self.assertEqual(self.traceback_first_line, first_line)

  def testGenerateSourceListWithNodeNameFilter(self):
    source_list = source_utils.list_source_files_against_dump(
        self.dump, node_name_regex_whitelist=r"while/Add.*")

    # Assert that the file paths are sorted.
    file_paths = [item[0] for item in source_list]
    self.assertEqual(sorted(file_paths), file_paths)
    self.assertEqual(len(set(file_paths)), len(file_paths))

    # Assert that each item of source_list has length 4.
    for item in source_list:
      self.assertTrue(isinstance(item, tuple))
      self.assertEqual(6, len(item))

    # Due to the node-name filtering the result should only contain 2 nodes
    # and 2 tensors. The total number of dumped tensors should be 6:
    #   while/Add/y:0          3
    #   while/Add:0            3
    _, is_tf_py_library, num_nodes, num_tensors, num_dumps, _ = (
        source_list[file_paths.index(self.curr_file_path)])
    self.assertFalse(is_tf_py_library)
    self.assertEqual(2, num_nodes)
    self.assertEqual(2, num_tensors)
    self.assertEqual(6, num_dumps)

  def testGenerateSourceListWithPathRegexFilter(self):
    curr_file_basename = os.path.basename(self.curr_file_path)
    source_list = source_utils.list_source_files_against_dump(
        self.dump,
        path_regex_whitelist=(
            ".*" + curr_file_basename.replace(".", "\\.") + "$"))

    self.assertEqual(1, len(source_list))
    (file_path, is_tf_py_library, num_nodes, num_tensors, num_dumps,
     first_line) = source_list[0]
    self.assertEqual(self.curr_file_path, file_path)
    self.assertFalse(is_tf_py_library)
    self.assertEqual(12, num_nodes)
    self.assertEqual(14, num_tensors)
    self.assertEqual(39, num_dumps)
    self.assertEqual(self.traceback_first_line, first_line)


if __name__ == "__main__":
  googletest.main()
