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

import inspect
import os
import shutil
import tempfile

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


def line_number_above():
  return inspect.stack()[1][2] - 1


class SourceHelperTest(test_util.TensorFlowTestCase):

  def createAndRunGraphHelper(self):
    """Create and run a TensorFlow Graph to generate debug dumps.

    This is intentionally done in separate method, to make it easier to test
    the stack-top mode of source annotation.
    """

    self.dump_root = self.get_temp_dir()
    self.curr_file_path = os.path.abspath(
        inspect.getfile(inspect.currentframe()))

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

      sess.run(self.u.initializer)
      sess.run(self.v.initializer)

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
      shutil.rmtree(self.dump_root)
    ops.reset_default_graph()

  def testAnnotateWholeValidSourceFileGivesCorrectResult(self):
    source_annotation = source_utils.annotate_source(self.dump,
                                                     self.curr_file_path)

    self.assertIn(self.u_init.op.name,
                  source_annotation[self.u_init_line_number])
    self.assertIn(self.u.op.name,
                  source_annotation[self.u_line_number])
    self.assertIn(self.v_init.op.name,
                  source_annotation[self.v_init_line_number])
    self.assertIn(self.v.op.name,
                  source_annotation[self.v_line_number])
    self.assertIn(self.w.op.name,
                  source_annotation[self.w_line_number])

    # In the non-stack-top (default) mode, the helper line should be annotated
    # with all the ops as well.
    self.assertIn(self.u_init.op.name,
                  source_annotation[self.helper_line_number])
    self.assertIn(self.u.op.name,
                  source_annotation[self.helper_line_number])
    self.assertIn(self.v_init.op.name,
                  source_annotation[self.helper_line_number])
    self.assertIn(self.v.op.name,
                  source_annotation[self.helper_line_number])
    self.assertIn(self.w.op.name,
                  source_annotation[self.helper_line_number])

  def testAnnotateWithStackTopGivesCorrectResult(self):
    source_annotation = source_utils.annotate_source(
        self.dump, self.curr_file_path, file_stack_top=True)

    self.assertIn(self.u_init.op.name,
                  source_annotation[self.u_init_line_number])
    self.assertIn(self.u.op.name,
                  source_annotation[self.u_line_number])
    self.assertIn(self.v_init.op.name,
                  source_annotation[self.v_init_line_number])
    self.assertIn(self.v.op.name,
                  source_annotation[self.v_line_number])
    self.assertIn(self.w.op.name,
                  source_annotation[self.w_line_number])

    # In the stack-top mode, the helper line should not have been annotated.
    self.assertNotIn(self.helper_line_number, source_annotation)

  def testAnnotateSubsetOfLinesGivesCorrectResult(self):
    source_annotation = source_utils.annotate_source(
        self.dump,
        self.curr_file_path,
        min_line=self.u_line_number,
        max_line=self.u_line_number + 1)

    self.assertIn(self.u.op.name,
                  source_annotation[self.u_line_number])
    self.assertNotIn(self.v_line_number, source_annotation)

  def testAnnotateDumpedTensorsGivesCorrectResult(self):
    source_annotation = source_utils.annotate_source(
        self.dump, self.curr_file_path, do_dumped_tensors=True)

    # Note: Constant Tensors u_init and v_init may not get dumped due to
    #   constant-folding.
    self.assertIn(self.u.name,
                  source_annotation[self.u_line_number])
    self.assertIn(self.v.name,
                  source_annotation[self.v_line_number])
    self.assertIn(self.w.name,
                  source_annotation[self.w_line_number])

    self.assertNotIn(self.u.op.name,
                     source_annotation[self.u_line_number])
    self.assertNotIn(self.v.op.name,
                     source_annotation[self.v_line_number])
    self.assertNotIn(self.w.op.name,
                     source_annotation[self.w_line_number])

    self.assertIn(self.u.name,
                  source_annotation[self.helper_line_number])
    self.assertIn(self.v.name,
                  source_annotation[self.helper_line_number])
    self.assertIn(self.w.name,
                  source_annotation[self.helper_line_number])

  def testCallingAnnotateSourceWithoutPythonGraphRaisesException(self):
    self.dump.set_python_graph(None)
    with self.assertRaises(ValueError):
      source_utils.annotate_source(self.dump, self.curr_file_path)

  def testCallingAnnotateSourceOnUnrelatedSourceFileDoesNotError(self):
    # Create an unrelated source file.
    unrelated_source_path = tempfile.mktemp()
    with open(unrelated_source_path, "wt") as source_file:
      source_file.write("print('hello, world')\n")

    self.assertEqual(
        {}, source_utils.annotate_source(self.dump, unrelated_source_path))

    # Clean up unrelated source file.
    os.remove(unrelated_source_path)


if __name__ == "__main__":
  googletest.main()
