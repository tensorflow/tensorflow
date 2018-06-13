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
# =============================================================================
"""Tests for tensorflow.python.training.saver.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import functools
import math
import os
import random
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np
import six

from google.protobuf.any_pb2 import Any
from google.protobuf import text_format

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import queue_runner_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import function
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import test_util
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import core
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary import summary
from tensorflow.python.training import adam
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.training import saver as saver_module
from tensorflow.python.training import saver_test_utils
from tensorflow.python.training import training_util
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.training.checkpointable import util as checkpointable_utils
from tensorflow.python.util import compat


class SaverTest(test.TestCase):

  def basicSaveRestore(self, variable_op):
    save_path = os.path.join(self.get_temp_dir(), "basic_save_restore")

    with self.test_session(graph=ops_lib.Graph()) as sess:
      # Build a graph with 2 parameter nodes, and Save and
      # Restore nodes for them.
      v0 = variable_op(10.0, name="v0")
      v1 = variable_op(20.0, name="v1")
      v2 = saver_test_utils.CheckpointedOp(name="v2")
      v2_init = v2.insert("k1", 30.0)

      # Initialize all variables
      if not context.executing_eagerly():
        self.evaluate([variables.global_variables_initializer(), v2_init])

        # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))
      self.assertEqual(b"k1", self.evaluate(v2.keys()))
      self.assertEqual(30.0, self.evaluate(v2.values()))

      # Save the initialized values in the file at "save_path"
      save = saver_module.Saver(
          {
              "v0": v0,
              "v1": v1,
              "v2": v2.saveable
          }, restore_sequentially=True)
      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

    # Start a second session.  In that session the parameter nodes
    # have not been initialized either.
    with self.test_session(graph=ops_lib.Graph()) as sess:
      v0 = variable_op(-1.0, name="v0")
      v1 = variable_op(-1.0, name="v1")
      v2 = saver_test_utils.CheckpointedOp(name="v2")

      # Assert that the variables are not initialized.
      if not context.executing_eagerly():
        self.assertEqual(
            len(variables.report_uninitialized_variables().eval()), 2)
        self.assertEqual(0, len(v2.keys().eval()))
        self.assertEqual(0, len(v2.values().eval()))
      # Restore the saved values in the parameter nodes.
      save = saver_module.Saver({"v0": v0, "v1": v1, "v2": v2.saveable})
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))
      self.assertEqual(b"k1", self.evaluate(v2.keys()))
      self.assertEqual(30.0, self.evaluate(v2.values()))

    # Build another graph with 2 nodes, initialized
    # differently, and a Restore node for them.
    with self.test_session(graph=ops_lib.Graph()) as sess:
      v0_2 = variable_op(1000.0, name="v0")
      v1_2 = variable_op(2000.0, name="v1")
      v2_2 = saver_test_utils.CheckpointedOp(name="v2")
      v2_init = v2_2.insert("k1000", 3000.0)

      # Check that the parameter nodes have been initialized.
      if not context.executing_eagerly():
        init_all_op = [variables.global_variables_initializer(), v2_init]
        self.evaluate(init_all_op)
        # TODO(xpan): Why _mutable_hash_table_v2 doesn't create empty
        # table as it claims in eager mode?
        self.assertEqual(b"k1000", self.evaluate(v2_2.keys()))
        self.assertEqual(3000.0, self.evaluate(v2_2.values()))
      self.assertEqual(1000.0, self.evaluate(v0_2))
      self.assertEqual(2000.0, self.evaluate(v1_2))

      # Restore the values saved earlier in the parameter nodes.
      save2 = saver_module.Saver({"v0": v0_2, "v1": v1_2, "v2": v2_2.saveable})
      save2.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, self.evaluate(v0_2))
      self.assertEqual(20.0, self.evaluate(v1_2))
      self.assertEqual(b"k1", self.evaluate(v2_2.keys()))
      self.assertEqual(30.0, self.evaluate(v2_2.values()))

  def testBasic(self):
    self.basicSaveRestore(variables.Variable)

  @test_util.run_in_graph_and_eager_modes()
  def testResourceBasic(self):
    self.basicSaveRestore(resource_variable_ops.ResourceVariable)

  def testResourceVariableReadOpsAddedDeterministically(self):
    graph_defs = []
    num_graphs = 10
    for _ in range(num_graphs):
      with ops_lib.Graph().as_default() as g:
        for i in range(20):
          resource_variable_ops.ResourceVariable(i, name="var%s" % i)
        saver_module.Saver()
        graph_defs.append(g.as_graph_def())
    for i in range(num_graphs - 1):
      self.assertEqual(graph_defs[i], graph_defs[i + 1])

  def testEagerBasic(self):
    with context.eager_mode():
      ckpt_prefix = os.path.join(self.get_temp_dir(), "ckpt")

      v1 = resource_variable_ops.ResourceVariable(3.14, name="v1")
      v2 = resource_variable_ops.ResourceVariable([1, 2], name="v2")
      save = saver_module.Saver([v1, v2])
      save.save(None, ckpt_prefix)

      v1.assign(0.0)
      v2.assign([0, 0])
      self.assertNear(0.0, self.evaluate(v1), 1e-5)
      self.assertAllEqual([0, 0], self.evaluate(v2))

      save.restore(None, ckpt_prefix)
      self.assertNear(3.14, self.evaluate(v1), 1e-5)
      self.assertAllEqual([1, 2], self.evaluate(v2))

  def testEagerGraphCompatibility(self):
    # Save from graph mode and restore from eager mode.
    graph_ckpt_prefix = os.path.join(self.get_temp_dir(), "graph_ckpt")
    with context.graph_mode():
      with self.test_session(graph=ops_lib.Graph()) as sess:
        # Create a graph model and save the checkpoint.
        w1 = resource_variable_ops.ResourceVariable(1.0, name="w1")
        w2 = resource_variable_ops.ResourceVariable(2.0, name="w2")
        graph_saver = saver_module.Saver([w1, w2])
        sess.run(variables.global_variables_initializer())
        graph_saver.save(sess, graph_ckpt_prefix)

    with context.eager_mode():
      ops_lib._default_graph_stack.reset()  # pylint: disable=protected-access
      ops_lib.reset_default_graph()

      w1 = resource_variable_ops.ResourceVariable(0.0, name="w1")
      w2 = resource_variable_ops.ResourceVariable(0.0, name="w2")

      graph_saver = saver_module.Saver([w1, w2])
      graph_saver.restore(None, graph_ckpt_prefix)

      self.assertAllEqual(self.evaluate(w1), 1.0)
      self.assertAllEqual(self.evaluate(w2), 2.0)

    # Save from eager mode and restore from graph mode.
    eager_ckpt_prefix = os.path.join(self.get_temp_dir(), "eager_ckpt")
    with context.eager_mode():
      ops_lib._default_graph_stack.reset()  # pylint: disable=protected-access
      ops_lib.reset_default_graph()

      w3 = resource_variable_ops.ResourceVariable(3.0, name="w3")
      w4 = resource_variable_ops.ResourceVariable(4.0, name="w4")

      graph_saver = saver_module.Saver([w3, w4])
      graph_saver.save(None, eager_ckpt_prefix)

    with context.graph_mode():
      with self.test_session(graph=ops_lib.Graph()) as sess:
        w3 = resource_variable_ops.ResourceVariable(0.0, name="w3")
        w4 = resource_variable_ops.ResourceVariable(0.0, name="w4")
        graph_saver = saver_module.Saver([w3, w4])
        sess.run(variables.global_variables_initializer())
        graph_saver.restore(sess, eager_ckpt_prefix)
        self.assertAllEqual(w3.eval(), 3.0)
        self.assertAllEqual(w4.eval(), 4.0)

  @test_util.run_in_graph_and_eager_modes()
  def testResourceSaveRestoreCachingDevice(self):
    save_path = os.path.join(self.get_temp_dir(), "resource_cache")
    with self.test_session(graph=ops_lib.Graph()) as sess:
      v = resource_variable_ops.ResourceVariable([1], caching_device="/cpu:0",
                                                 name="v")
      if context.executing_eagerly():
        sess = None
      else:
        self.evaluate(variables.global_variables_initializer())
      save = saver_module.Saver([v])
      save.save(sess, save_path)

      save2 = saver_module.Saver([v])
      save2.restore(sess, save_path)
      self.assertEquals(self.evaluate(v), [1])

  def testNoAdditionalOpsAddedBySaverForResourceVariablesOutsideSaveScope(self):
    with ops_lib.Graph().as_default() as g:
      v = resource_variable_ops.ResourceVariable(1.0, name="v")
      with ops_lib.name_scope("saver1"):
        saver_module.Saver()
      with ops_lib.name_scope("saver2"):
        saver_module.Saver({"name": v})
    ops_in_saver1_scope_but_not_save_scope = [
        op for op in g.get_operations()
        if (op.name.startswith("saver1/") and
            not op.name.startswith("saver1/save/"))]
    self.assertEqual(ops_in_saver1_scope_but_not_save_scope, [])
    ops_in_saver2_scope_but_not_save_scope = [
        op for op in g.get_operations()
        if (op.name.startswith("saver2/") and
            not op.name.startswith("saver2/save/"))]
    self.assertEqual(ops_in_saver2_scope_but_not_save_scope, [])

  def testSaveCopyRestoreWithSaveRelativePaths(self):
    """Save, copy checkpoint dir and restore from copied dir.

    This only works for save_relative_paths=True.
    """
    save_dir1 = os.path.join(self.get_temp_dir(), "save_dir1")
    os.mkdir(save_dir1)
    save_path1 = os.path.join(save_dir1, "save_copy_restore")

    # Build a graph with 2 parameter nodes, and Save and
    # Restore nodes for them.
    v0 = variables.Variable(10.0, name="v0")
    v1 = variables.Variable(20.0, name="v1")
    v2 = saver_test_utils.CheckpointedOp(name="v2")
    v2_init = v2.insert("k1", 30.0)
    save = saver_module.Saver(
        var_list={
            "v0": v0,
            "v1": v1,
            "v2": v2.saveable},
        restore_sequentially=True,
        save_relative_paths=True)
    init_all_op = [variables.global_variables_initializer(), v2_init]

    with self.test_session() as sess:
      # Initialize all variables
      sess.run(init_all_op)

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())
      self.assertEqual(b"k1", v2.keys().eval())
      self.assertEqual(30.0, v2.values().eval())

      # Save the initialized values in the file at "save_path"
      val = save.save(sess, save_path1)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path1, val)

    self.assertEqual(saver_module.latest_checkpoint(save_dir1), save_path1)
    save_dir2 = os.path.join(self.get_temp_dir(), "save_dir2")
    os.renames(save_dir1, save_dir2)
    save_path2 = os.path.join(save_dir2, "save_copy_restore")
    self.assertEqual(saver_module.latest_checkpoint(save_dir2), save_path2)

    # Start a second session.  In that session the parameter nodes
    # have not been initialized either.
    with self.test_session() as sess:
      v0 = variables.Variable(-1.0, name="v0")
      v1 = variables.Variable(-1.0, name="v1")
      v2 = saver_test_utils.CheckpointedOp(name="v2")
      save = saver_module.Saver({"v0": v0, "v1": v1, "v2": v2.saveable})

      # Assert that the variables are not initialized.
      self.assertEqual(
          len(variables.report_uninitialized_variables().eval()), 2)
      self.assertEqual(0, len(v2.keys().eval()))
      self.assertEqual(0, len(v2.values().eval()))

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path2)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())
      self.assertEqual(b"k1", v2.keys().eval())
      self.assertEqual(30.0, v2.values().eval())

  def testFilenameTensor(self):
    v0 = variables.Variable(0, name="v0")
    filename = b"somerandomfilename"
    save = saver_module.Saver({"v0": v0}, filename=filename)
    with self.test_session() as sess:
      tensor = sess.graph.get_tensor_by_name(
          save.saver_def.filename_tensor_name)
      self.assertEqual(sess.run(tensor), filename)

  def testInvalidPath(self):
    v0 = variables.Variable(0, name="v0")
    for ver in (saver_pb2.SaverDef.V1, saver_pb2.SaverDef.V2):
      with self.test_session() as sess:
        save = saver_module.Saver({"v0": v0}, write_version=ver)
        with self.assertRaisesRegexp(errors.NotFoundError,
                                     "Failed to find any matching files for"):
          save.restore(sess, "invalid path")

  def testInt64(self):
    save_path = os.path.join(self.get_temp_dir(), "int64")

    with self.test_session() as sess:
      # Build a graph with 1 node, and save and restore for them.
      v = variables.Variable(np.int64(15), name="v")
      save = saver_module.Saver({"v": v}, restore_sequentially=True)
      variables.global_variables_initializer().run()

      # Save the initialized values in the file at "save_path"
      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

      with self.test_session() as sess:
        v = variables.Variable(np.int64(-1), name="v")
        save = saver_module.Saver({"v": v})

      with self.assertRaisesWithPredicateMatch(
          errors_impl.OpError, lambda e: "uninitialized value v" in e.message):
        sess.run(v)

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(np.int64(15), v.eval())

  def testSomeErrors(self):
    with ops_lib.Graph().as_default():
      v0 = variables.Variable([10.0], name="v0")
      v1 = variables.Variable([20.0], name="v1")
      v2 = variables.Variable([20.0], name="v2")
      v2._set_save_slice_info(
          variables.Variable.SaveSliceInfo("v1", [1], [0], [1]))

      # By default the name used for "v2" will be "v1" and raise an error.
      with self.assertRaisesRegexp(ValueError, "same name: v1"):
        saver_module.Saver([v0, v1, v2])

      # The names are different and will work.
      saver_module.Saver({"vee1": v1, "other": [v2]})

      # Partitioned variables also cause name conflicts.
      p_v1 = variable_scope.get_variable(
          "p_v1",
          shape=[4, 5],
          partitioner=partitioned_variables.fixed_size_partitioner(
              num_shards=2))
      p_v2 = variable_scope.get_variable(
          "p_v2",
          shape=[4, 5],
          partitioner=partitioned_variables.fixed_size_partitioner(
              num_shards=2))
      p_v2._name = "p_v1"
      with self.assertRaisesRegexp(ValueError, "same name: p_v1"):
        saver_module.Saver([p_v1, p_v2])

  def testSameName(self):
    with ops_lib.Graph().as_default():
      v0 = variables.Variable([10.0], name="v0")
      v2 = saver_test_utils.CheckpointedOp(name="v2")

      # Saving one variable under two names raises an error.
      with self.assertRaisesRegexp(
          ValueError, "The same saveable will be restored with two names: v0"):
        saver_module.Saver({"v0": v0, "v0too": v0})

      # Ditto for custom saveables.
      with self.assertRaisesRegexp(
          ValueError, "The same saveable will be restored with two names: v2"):
        saver_module.Saver({"v2": v2.saveable, "v2too": v2.saveable})

      # Verify non-duplicate names work.
      saver_module.Saver({"v0": v0, "v2": v2.saveable})

  def testBasicsWithListOfVariables(self):
    save_path = os.path.join(self.get_temp_dir(), "basics_with_list")

    with self.test_session(graph=ops_lib.Graph()) as sess:
      # Build a graph with 2 parameter nodes, and Save and
      # Restore nodes for them.
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")
      v2 = saver_test_utils.CheckpointedOp(name="v2")
      v2_init = v2.insert("k1", 30.0)
      save = saver_module.Saver([v0, v1, v2.saveable])
      variables.global_variables_initializer().run()
      v2_init.run()

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())
      self.assertEqual(b"k1", v2.keys().eval())
      self.assertEqual(30.0, v2.values().eval())

      # Save the initialized values in the file at "save_path"
      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

    # Start a second session.  In that session the variables
    # have not been initialized either.
    with self.test_session(graph=ops_lib.Graph()) as sess:
      v0 = variables.Variable(-1.0, name="v0")
      v1 = variables.Variable(-1.0, name="v1")
      v2 = saver_test_utils.CheckpointedOp(name="v2")
      save = saver_module.Saver([v0, v1, v2.saveable])

      with self.assertRaisesWithPredicateMatch(
          errors_impl.OpError, lambda e: "uninitialized value v0" in e.message):
        sess.run(v0)
      with self.assertRaisesWithPredicateMatch(
          errors_impl.OpError, lambda e: "uninitialized value v1" in e.message):
        sess.run(v1)
      self.assertEqual(0, len(v2.keys().eval()))
      self.assertEqual(0, len(v2.values().eval()))

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())
      self.assertEqual(b"k1", v2.keys().eval())
      self.assertEqual(30.0, v2.values().eval())

    # Build another graph with 2 nodes, initialized
    # differently, and a Restore node for them.
    with self.test_session(graph=ops_lib.Graph()) as sess:
      v0_2 = variables.Variable(1000.0, name="v0")
      v1_2 = variables.Variable(2000.0, name="v1")
      v2_2 = saver_test_utils.CheckpointedOp(name="v2")
      save2 = saver_module.Saver([v0_2, v1_2, v2_2.saveable])
      v2_2.insert("k1000", 3000.0).run()
      variables.global_variables_initializer().run()

      # Check that the parameter nodes have been initialized.
      self.assertEqual(1000.0, v0_2.eval())
      self.assertEqual(2000.0, v1_2.eval())
      self.assertEqual(b"k1000", v2_2.keys().eval())
      self.assertEqual(3000.0, v2_2.values().eval())
      # Restore the values saved earlier in the parameter nodes.
      save2.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, v0_2.eval())
      self.assertEqual(20.0, v1_2.eval())
      self.assertEqual(b"k1", v2_2.keys().eval())
      self.assertEqual(30.0, v2_2.values().eval())

  def _SaveAndLoad(self, var_name, var_value, other_value, save_path):
    with self.test_session(graph=ops_lib.Graph()) as sess:
      var = resource_variable_ops.ResourceVariable(var_value, name=var_name)
      save = saver_module.Saver({var_name: var})
      if not context.executing_eagerly():
        self.evaluate(var.initializer)
      val = save.save(sess, save_path)
      self.assertEqual(save_path, val)
    with self.test_session(graph=ops_lib.Graph()) as sess:
      var = resource_variable_ops.ResourceVariable(other_value, name=var_name)
      save = saver_module.Saver({var_name: var})
      save.restore(sess, save_path)
      self.assertAllClose(var_value, self.evaluate(var))

  def testCacheRereadsFile(self):
    save_path = os.path.join(self.get_temp_dir(), "cache_rereads")
    # Save and reload one Variable named "var0".
    self._SaveAndLoad("var0", 0.0, 1.0, save_path)
    # Save and reload one Variable named "var1" in the same file.
    # The cached readers should know to re-read the file.
    self._SaveAndLoad("var1", 1.1, 2.2, save_path)

  def testAllowEmpty(self):
    save_path = os.path.join(self.get_temp_dir(), "allow_empty")
    with self.test_session() as sess:
      _ = constant_op.constant(1)
      save = saver_module.Saver(allow_empty=True)
      val = save.save(sess, save_path)
      self.assertIsNone(val)
    with self.test_session() as sess:
      save = saver_module.Saver(allow_empty=True)
      save.restore(sess, save_path)

  def testGPU(self):
    if not test.is_gpu_available():
      return
    save_path = os.path.join(self.get_temp_dir(), "gpu")
    with session.Session("", graph=ops_lib.Graph()) as sess:
      with sess.graph.device(test.gpu_device_name()):
        v0_1 = variables.Variable(123.45)
      save = saver_module.Saver({"v0": v0_1})
      variables.global_variables_initializer().run()
      save.save(sess, save_path)

    with session.Session("", graph=ops_lib.Graph()) as sess:
      with sess.graph.device(test.gpu_device_name()):
        v0_2 = variables.Variable(543.21)
      save = saver_module.Saver({"v0": v0_2})
      variables.global_variables_initializer().run()

  def testSharedServerOnGPU(self):
    if not test.is_gpu_available():
      return
    save_path = os.path.join(self.get_temp_dir(), "gpu")
    with session.Session("", graph=ops_lib.Graph()) as sess:
      with sess.graph.device(test.gpu_device_name()):
        v0_1 = variables.Variable(123.45)
      save = saver_module.Saver({"v0": v0_1}, sharded=True, allow_empty=True)
      variables.global_variables_initializer().run()
      save.save(sess, save_path)

    with session.Session("", graph=ops_lib.Graph()) as sess:
      with sess.graph.device(test.gpu_device_name()):
        v0_2 = variables.Variable(543.21)
      save = saver_module.Saver({"v0": v0_2}, sharded=True, allow_empty=True)
      variables.global_variables_initializer().run()

  def testVariables(self):
    save_path = os.path.join(self.get_temp_dir(), "variables")
    with session.Session("", graph=ops_lib.Graph()) as sess:
      one = variables.Variable(1.0)
      twos = variables.Variable([2.0, 2.0, 2.0])
      v2 = saver_test_utils.CheckpointedOp(name="v2")
      init = variables.global_variables_initializer()
      save = saver_module.Saver()
      init.run()
      v2.insert("k1", 3.0).run()
      save.save(sess, save_path)

    with session.Session("", graph=ops_lib.Graph()) as sess:
      one = variables.Variable(0.0)
      twos = variables.Variable([0.0, 0.0, 0.0])
      v2 = saver_test_utils.CheckpointedOp(name="v2")
      # Saver with no arg, defaults to 'all variables'.
      save = saver_module.Saver()
      save.restore(sess, save_path)
      self.assertAllClose(1.0, one.eval())
      self.assertAllClose([2.0, 2.0, 2.0], twos.eval())
      self.assertEqual(b"k1", v2.keys().eval())
      self.assertEqual(3.0, v2.values().eval())

  def testVarListShouldBeEmptyInDeferredBuild(self):
    with ops_lib.Graph().as_default():
      v = variables.Variable(1.0)
      with self.assertRaisesRegexp(ValueError, "defer_build"):
        saver_module.Saver([v], defer_build=True)

  def testBuildShouldBeCalledBeforeSaveInCaseOfDeferBuild(self):
    save_path = os.path.join(self.get_temp_dir(), "error_deferred_build")
    with ops_lib.Graph().as_default(), session.Session() as sess:
      variables.Variable(1.0)
      saver = saver_module.Saver(defer_build=True)
      with self.assertRaisesRegexp(RuntimeError, "build"):
        saver.save(sess, save_path)

  def testDeferredBuild(self):
    save_path = os.path.join(self.get_temp_dir(), "deferred_build")
    with session.Session("", graph=ops_lib.Graph()) as sess:
      one = variables.Variable(1.0)
      save = saver_module.Saver(defer_build=True)
      # if build is not deferred, saver cannot save the `twos`.
      twos = variables.Variable([2.0, 2.0, 2.0])
      init = variables.global_variables_initializer()
      save.build()
      init.run()
      save.save(sess, save_path)

    with session.Session("", graph=ops_lib.Graph()) as sess:
      one = variables.Variable(0.0)
      twos = variables.Variable([0.0, 0.0, 0.0])
      # Saver with no arg, defaults to 'all variables'.
      save = saver_module.Saver()
      save.restore(sess, save_path)
      self.assertAllClose(1.0, one.eval())
      self.assertAllClose([2.0, 2.0, 2.0], twos.eval())

  def testReshape(self):
    save_path = os.path.join(self.get_temp_dir(), "variables_reshape")
    with session.Session("", graph=ops_lib.Graph()) as sess:
      var = variables.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      init = variables.global_variables_initializer()
      save = saver_module.Saver()
      init.run()
      save.save(sess, save_path)

    # Error when restoring with default reshape=False
    with session.Session("", graph=ops_lib.Graph()) as sess:
      var = variables.Variable([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
      save = saver_module.Saver()
      with self.assertRaisesRegexp(
          errors_impl.InvalidArgumentError,
          "Assign requires shapes of both tensors to match."):
        save.restore(sess, save_path)

    # Restored to new shape with reshape=True
    with session.Session("", graph=ops_lib.Graph()) as sess:
      var = variables.Variable([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
      save = saver_module.Saver(reshape=True)
      save.restore(sess, save_path)
      self.assertAllClose([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], var.eval())

  @test_util.run_in_graph_and_eager_modes()
  def testSaveWithGlobalStep(self, pad_step_number=False):
    save_path = os.path.join(self.get_temp_dir(), "ckpt_with_global_step")
    global_step_int = 5
    # Save and reload one Variable named "var0".
    self._SaveAndLoad("var0", 0.0, 1.0, save_path)
    for use_tensor in [True, False]:
      with self.test_session(graph=ops_lib.Graph()):
        var = resource_variable_ops.ResourceVariable(1.0, name="var0")
        save = saver_module.Saver(
            {
                var._shared_name: var
            }, pad_step_number=pad_step_number)
        if context.executing_eagerly():
          sess = None
        else:
          self.evaluate(var.initializer)
          sess = ops_lib.get_default_session()
        if use_tensor:
          global_step = constant_op.constant(global_step_int)
          val = save.save(sess, save_path, global_step=global_step)
        else:
          val = save.save(sess, save_path, global_step=global_step_int)
        if pad_step_number:
          expected_save_path = "%s-%s" % (save_path,
                                          "{:08d}".format(global_step_int))
        else:
          expected_save_path = "%s-%d" % (save_path, global_step_int)
        self.assertEqual(expected_save_path, val)

  def testSaveWithGlobalStepWithPadding(self):
    self.testSaveWithGlobalStep(pad_step_number=True)

  def testSaveToNonexistingPath(self):
    file_io.write_string_to_file(
        os.path.join(self.get_temp_dir(), "actually_a_file"), "")
    paths = [
        os.path.join(self.get_temp_dir(), "nonexisting_dir/path"),
        os.path.join(self.get_temp_dir(), "other_nonexisting_dir/path1/path2"),
        os.path.join(self.get_temp_dir(), "actually_a_file/path"),
    ]

    for save_path in paths:
      # Build a graph with 2 parameter nodes, and Save and
      # Restore nodes for them.
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")
      save = saver_module.Saver({"v0": v0, "v1": v1}, restore_sequentially=True)
      init_all_op = variables.global_variables_initializer()

      # In the case where the parent directory doesn't exist, whether or not the
      # save succeeds or fails is implementation dependent.  Therefore we allow
      # both cases.
      try:
        with self.test_session() as sess:
          # Initialize all variables
          sess.run(init_all_op)

          # Check that the parameter nodes have been initialized.
          self.assertEqual(10.0, v0.eval())
          self.assertEqual(20.0, v1.eval())

          # Save the graph.
          save.save(sess, save_path)

        with self.test_session() as sess:
          # Restore the saved values in the parameter nodes.
          save.restore(sess, save_path)
          # Check that the parameter nodes have been restored.
          self.assertEqual(10.0, v0.eval())
          self.assertEqual(20.0, v1.eval())
      except ValueError as exc:
        error_msg_template = "Parent directory of {} doesn't exist, can't save."
        self.assertEqual(error_msg_template.format(save_path), str(exc))

  def testSaveToURI(self):
    # ParseURI functions don't work on Windows yet.
    # TODO(jhseu): Remove this check when it works.
    if os.name == "nt":
      self.skipTest("Local URI support doesn't work on Windows")
    save_path = "file://" + os.path.join(self.get_temp_dir(), "uri")

    # Build a graph with 2 parameter nodes, and Save and
    # Restore nodes for them.
    v0 = variables.Variable(10.0, name="v0")
    v1 = variables.Variable(20.0, name="v1")
    save = saver_module.Saver({"v0": v0, "v1": v1}, restore_sequentially=True)
    init_all_op = variables.global_variables_initializer()

    with self.test_session() as sess:
      # Initialize all variables
      sess.run(init_all_op)

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())
      save.save(sess, save_path)


class SaveRestoreShardedTest(test.TestCase):

  _WRITE_VERSION = saver_pb2.SaverDef.V1

  def _get_test_dir(self, dirname):
    test_dir = os.path.join(self.get_temp_dir(), dirname)
    gfile.MakeDirs(test_dir)
    return test_dir

  def testBasics(self):
    save_path = os.path.join(self.get_temp_dir(), "sharded_basics")

    # Build a graph with 2 parameter nodes on different devices.
    with session.Session(
        target="",
        config=config_pb2.ConfigProto(device_count={"CPU": 2})) as sess:
      with sess.graph.device("/cpu:0"):
        v0 = variables.Variable(10, name="v0")
        t0 = saver_test_utils.CheckpointedOp(name="t0")
      with sess.graph.device("/cpu:1"):
        v1 = variables.Variable(20, name="v1")
        t1 = saver_test_utils.CheckpointedOp(name="t1")
      save = saver_module.Saver(
          {
              "v0": v0,
              "v1": v1,
              "t0": t0.saveable,
              "t1": t1.saveable
          },
          write_version=self._WRITE_VERSION,
          sharded=True)
      variables.global_variables_initializer().run()
      t0.insert("k1", 30.0).run()
      t1.insert("k2", 40.0).run()
      val = save.save(sess, save_path)
      if save._write_version is saver_pb2.SaverDef.V1:
        self.assertEqual(save_path + "-?????-of-00002", val)
      else:
        self.assertEqual(save_path, val)
      meta_graph_filename = save._MetaGraphFilename(val)
      self.assertEqual(save_path + ".meta", meta_graph_filename)

    if save._write_version is saver_pb2.SaverDef.V1:
      # Restore different ops from shard 0 of the saved files.
      with session.Session(
          target="",
          config=config_pb2.ConfigProto(device_count={"CPU": 2})) as sess:
        with sess.graph.device("/cpu:0"):
          v0 = variables.Variable(111, name="v0")
          t0 = saver_test_utils.CheckpointedOp(name="t0")
        save = saver_module.Saver(
            {
                "v0": v0,
                "t0": t0.saveable
            },
            write_version=self._WRITE_VERSION,
            sharded=True)
        variables.global_variables_initializer().run()
        t0.insert("k11", 33.0).run()
        self.assertEqual(111, v0.eval())
        self.assertEqual(b"k11", t0.keys().eval())
        self.assertEqual(33.0, t0.values().eval())
        save.restore(sess, save_path + "-00000-of-00002")
        self.assertEqual(10, v0.eval())
        self.assertEqual(b"k1", t0.keys().eval())
        self.assertEqual(30.0, t0.values().eval())

      # Restore different ops from shard 1 of the saved files.
      with session.Session(
          target="",
          config=config_pb2.ConfigProto(device_count={"CPU": 2})) as sess:
        with sess.graph.device("/cpu:0"):
          v1 = variables.Variable(222)
          t1 = saver_test_utils.CheckpointedOp(name="t1")
        save = saver_module.Saver(
            {
                "v1": v1,
                "t1": t1.saveable
            },
            write_version=self._WRITE_VERSION,
            sharded=True)
        variables.global_variables_initializer().run()
        t1.insert("k22", 44.0).run()
        self.assertEqual(222, v1.eval())
        self.assertEqual(b"k22", t1.keys().eval())
        self.assertEqual(44.0, t1.values().eval())
        save.restore(sess, save_path + "-00001-of-00002")
        self.assertEqual(20, v1.eval())
        self.assertEqual(b"k2", t1.keys().eval())
        self.assertEqual(40.0, t1.values().eval())

    # Now try a restore with the sharded filename.
    with session.Session(
        target="",
        config=config_pb2.ConfigProto(device_count={"CPU": 2})) as sess:
      with sess.graph.device("/cpu:0"):
        v0 = variables.Variable(111, name="v0")
        t0 = saver_test_utils.CheckpointedOp(name="t0")
      with sess.graph.device("/cpu:1"):
        v1 = variables.Variable(222, name="v1")
        t1 = saver_test_utils.CheckpointedOp(name="t1")
      save = saver_module.Saver(
          {
              "v0": v0,
              "v1": v1,
              "t0": t0.saveable,
              "t1": t1.saveable
          },
          write_version=self._WRITE_VERSION,
          sharded=True)
      variables.global_variables_initializer().run()
      t0.insert("k11", 33.0).run()
      t1.insert("k22", 44.0).run()
      self.assertEqual(111, v0.eval())
      self.assertEqual(222, v1.eval())
      self.assertEqual(b"k11", t0.keys().eval())
      self.assertEqual(33.0, t0.values().eval())
      self.assertEqual(b"k22", t1.keys().eval())
      self.assertEqual(44.0, t1.values().eval())
      save_path = os.path.join(self.get_temp_dir(), "sharded_basics")
      if save._write_version is saver_pb2.SaverDef.V1:
        save.restore(sess, save_path + "-?????-of-?????")
      else:
        save.restore(sess, save_path)
      self.assertEqual(10, v0.eval())
      self.assertEqual(20, v1.eval())
      self.assertEqual(b"k1", t0.keys().eval())
      self.assertEqual(30.0, t0.values().eval())
      self.assertEqual(b"k2", t1.keys().eval())
      self.assertEqual(40.0, t1.values().eval())

    if save._write_version is saver_pb2.SaverDef.V1:
      self.assertEqual(
          saver_module.latest_checkpoint(self.get_temp_dir()),
          os.path.join(self.get_temp_dir(), "sharded_basics-?????-of-00002"))
    else:
      self.assertEqual(
          saver_module.latest_checkpoint(self.get_temp_dir()),
          os.path.join(self.get_temp_dir(), "sharded_basics"))

  def testSaverDef(self):
    with self.test_session():
      v0 = variables.Variable(123, name="v0")
      save = saver_module.Saver({"v0": v0}, sharded=True)
      sd = save.as_saver_def()
      self.assertTrue(sd.sharded)

  def _testPartitionedVariables(self, use_resource):
    var_full_shape = [10, 3]
    # Allows save/restore mechanism to work w/ different slicings.
    var_name = "my_var"
    saved_dir = self._get_test_dir("partitioned_variables")
    saved_path = os.path.join(saved_dir, "ckpt")

    call_saver_with_dict = False  # updated by test loop below

    def _save(slices=None, partitioner=None):
      with self.test_session(graph=ops_lib.Graph()) as sess:
        # Calls .eval() to return the ndarray that makes up the full variable.
        rnd = random_ops.random_uniform(var_full_shape).eval()

        if slices:
          assert not partitioner
          # TODO(apassos): make create_partitioned_variables take use_resource
          # option to make this test passable without creating a named
          # variable_scope.
          vs = partitioned_variables.create_partitioned_variables(
              var_full_shape, slices, rnd, name=var_name)
        elif partitioner:
          vs = [
              variable_scope.get_variable(
                  var_name,
                  shape=var_full_shape,
                  initializer=rnd,
                  partitioner=partitioner,
                  use_resource=use_resource)
          ]
        else:
          if use_resource:
            vs = [resource_variable_ops.ResourceVariable(rnd, name=var_name)]
          else:
            vs = [variables.Variable(rnd, name=var_name)]

        variables.global_variables_initializer().run()
        if call_saver_with_dict:
          saver = saver_module.Saver({var_name: (vs if slices else vs[0])})
        else:
          saver = saver_module.Saver(vs)
        actual_path = saver.save(sess, saved_path)
        self.assertEqual(saved_path, actual_path)

        return rnd

    def _restore(slices=None, partitioner=None):
      with self.test_session(graph=ops_lib.Graph()) as sess:
        if slices:
          assert not partitioner
          new_vs = partitioned_variables.create_partitioned_variables(
              var_full_shape,
              slices,
              array_ops.zeros(var_full_shape),  # != original contents.
              name=var_name)
        elif partitioner:
          new_vs = [
              variable_scope.get_variable(
                  var_name,
                  shape=var_full_shape,
                  initializer=array_ops.zeros(var_full_shape),
                  partitioner=partitioner)
          ]
        else:
          new_vs = [
              variables.Variable(
                  array_ops.zeros(
                      shape=var_full_shape),  # != original contents.
                  name=var_name)
          ]

        variables.global_variables_initializer().run()
        if call_saver_with_dict:
          saver = saver_module.Saver({
              var_name: (new_vs if slices else new_vs[0])
          })
        else:
          saver = saver_module.Saver(new_vs)
        saver.restore(sess, saved_path)

        if partitioner:
          return new_vs[0].as_tensor().eval()
        elif slices and slices[0] != 1:
          return array_ops.concat(new_vs, 0).eval()
        elif slices and slices[1] != 1:
          return array_ops.concat(new_vs, 1).eval()
        else:  # Non-sliced.
          return new_vs[0].eval()

    for call_saver_with_dict in {False, True}:
      # Save PartitionedVariable and restore into full variable.
      saved_full = _save(
          partitioner=partitioned_variables.fixed_size_partitioner(
              num_shards=2))
      restored_full = _restore()
      self.assertAllEqual(saved_full, restored_full)

      # Saves 10 horizontal parts of a partitioned variable.
      # Restores into a full variable, non-sliced.
      saved_full = _save(slices=[10, 1])
      restored_full = _restore()
      self.assertAllEqual(saved_full, restored_full)

      # Restores into a different number/orientation of slices.
      restored_full = _restore(slices=[2, 1])  # 2 horizon parts.
      self.assertAllEqual(saved_full, restored_full)
      restored_full = _restore(slices=[1, 3])  # 3 vertical parts.
      self.assertAllEqual(saved_full, restored_full)

      # Restores into a PartitionedVariable
      restored_full = _restore(
          partitioner=partitioned_variables.fixed_size_partitioner(
              num_shards=2))
      self.assertAllEqual(saved_full, restored_full)

      # Now, saves a full variable and restores in slices.
      saved_full = _save()
      restored_full = _restore(slices=[1, 3])
      self.assertAllEqual(saved_full, restored_full)

  def testPartitionedVariable(self):
    self._testPartitionedVariables(use_resource=False)

  def testPartitionedResourceVariable(self):
    self._testPartitionedVariables(use_resource=True)


class SaveRestoreShardedTestV2(SaveRestoreShardedTest):
  _WRITE_VERSION = saver_pb2.SaverDef.V2


class MaxToKeepTest(test.TestCase):

  def _get_test_dir(self, dirname):
    test_dir = os.path.join(self.get_temp_dir(), dirname)
    gfile.MakeDirs(test_dir)
    return test_dir

  def assertCheckpointState(self, model_checkpoint_path,
                            all_model_checkpoint_paths, save_dir):
    checkpoint_state = saver_module.get_checkpoint_state(save_dir)
    self.assertEqual(checkpoint_state.model_checkpoint_path,
                     model_checkpoint_path)
    self.assertEqual(checkpoint_state.all_model_checkpoint_paths,
                     all_model_checkpoint_paths)

  def testMaxToKeepEager(self):
    with context.eager_mode():
      save_dir = self._get_test_dir("max_to_keep_non_sharded")

      v = variable_scope.variable(10.0, name="v")
      save = saver_module.Saver({"v": v}, max_to_keep=2)
      self.evaluate(variables.global_variables_initializer())
      if not context.executing_eagerly():
        self.assertEqual([], save.last_checkpoints)

      s1 = save.save(None, os.path.join(save_dir, "s1"))
      self.assertEqual([s1], save.last_checkpoints)
      self.assertTrue(saver_module.checkpoint_exists(s1))
      self.assertCheckpointState(
          model_checkpoint_path=s1,
          all_model_checkpoint_paths=[s1],
          save_dir=save_dir)

      s2 = save.save(None, os.path.join(save_dir, "s2"))
      self.assertEqual([s1, s2], save.last_checkpoints)
      self.assertTrue(saver_module.checkpoint_exists(s1))
      self.assertTrue(saver_module.checkpoint_exists(s2))
      self.assertCheckpointState(
          model_checkpoint_path=s2,
          all_model_checkpoint_paths=[s1, s2],
          save_dir=save_dir)

      s3 = save.save(None, os.path.join(save_dir, "s3"))
      self.assertEqual([s2, s3], save.last_checkpoints)
      self.assertFalse(saver_module.checkpoint_exists(s1))
      self.assertTrue(saver_module.checkpoint_exists(s2))
      self.assertTrue(saver_module.checkpoint_exists(s3))
      self.assertCheckpointState(
          model_checkpoint_path=s3,
          all_model_checkpoint_paths=[s2, s3],
          save_dir=save_dir)

      # Create a second helper, identical to the first.
      save2 = saver_module.Saver({"v": v}, max_to_keep=2)
      save2.set_last_checkpoints(save.last_checkpoints)

      # Exercise the first helper.

      # Adding s2 again (old s2 is removed first, then new s2 appended)
      s2 = save.save(None, os.path.join(save_dir, "s2"))
      self.assertEqual([s3, s2], save.last_checkpoints)
      self.assertFalse(saver_module.checkpoint_exists(s1))
      self.assertTrue(saver_module.checkpoint_exists(s3))
      self.assertTrue(saver_module.checkpoint_exists(s2))
      self.assertCheckpointState(
          model_checkpoint_path=s2,
          all_model_checkpoint_paths=[s3, s2],
          save_dir=save_dir)

      # Adding s1 (s3 should now be deleted as oldest in list)
      s1 = save.save(None, os.path.join(save_dir, "s1"))
      self.assertEqual([s2, s1], save.last_checkpoints)
      self.assertFalse(saver_module.checkpoint_exists(s3))
      self.assertTrue(saver_module.checkpoint_exists(s2))
      self.assertCheckpointState(
          model_checkpoint_path=s1,
          all_model_checkpoint_paths=[s2, s1],
          save_dir=save_dir)

      s2 = save2.save(None, os.path.join(save_dir, "s2"))
      self.assertEqual([s3, s2], save2.last_checkpoints)
      # Created by the first helper.
      self.assertTrue(saver_module.checkpoint_exists(s1))
      # Deleted by the first helper.
      self.assertFalse(saver_module.checkpoint_exists(s3))

  def testNonSharded(self):
    save_dir = self._get_test_dir("max_to_keep_non_sharded")

    with self.test_session() as sess:
      v = variables.Variable(10.0, name="v")
      save = saver_module.Saver({"v": v}, max_to_keep=2)
      variables.global_variables_initializer().run()
      self.assertEqual([], save.last_checkpoints)

      s1 = save.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([s1], save.last_checkpoints)
      self.assertTrue(saver_module.checkpoint_exists(s1))
      self.assertCheckpointState(
          model_checkpoint_path=s1,
          all_model_checkpoint_paths=[s1],
          save_dir=save_dir)

      s2 = save.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([s1, s2], save.last_checkpoints)
      self.assertTrue(saver_module.checkpoint_exists(s1))
      self.assertTrue(saver_module.checkpoint_exists(s2))
      self.assertCheckpointState(
          model_checkpoint_path=s2,
          all_model_checkpoint_paths=[s1, s2],
          save_dir=save_dir)

      s3 = save.save(sess, os.path.join(save_dir, "s3"))
      self.assertEqual([s2, s3], save.last_checkpoints)
      self.assertFalse(saver_module.checkpoint_exists(s1))
      self.assertTrue(saver_module.checkpoint_exists(s2))
      self.assertTrue(saver_module.checkpoint_exists(s3))
      self.assertCheckpointState(
          model_checkpoint_path=s3,
          all_model_checkpoint_paths=[s2, s3],
          save_dir=save_dir)

      # Create a second helper, identical to the first.
      save2 = saver_module.Saver(saver_def=save.as_saver_def())
      save2.set_last_checkpoints(save.last_checkpoints)

      # Create a third helper, with the same configuration but no knowledge of
      # previous checkpoints.
      save3 = saver_module.Saver(saver_def=save.as_saver_def())

      # Exercise the first helper.

      # Adding s2 again (old s2 is removed first, then new s2 appended)
      s2 = save.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([s3, s2], save.last_checkpoints)
      self.assertFalse(saver_module.checkpoint_exists(s1))
      self.assertFalse(
          saver_module.checkpoint_exists(save._MetaGraphFilename(s1)))
      self.assertTrue(saver_module.checkpoint_exists(s3))
      self.assertTrue(
          saver_module.checkpoint_exists(save._MetaGraphFilename(s3)))
      self.assertTrue(saver_module.checkpoint_exists(s2))
      self.assertTrue(
          saver_module.checkpoint_exists(save._MetaGraphFilename(s2)))
      self.assertCheckpointState(
          model_checkpoint_path=s2,
          all_model_checkpoint_paths=[s3, s2],
          save_dir=save_dir)

      # Adding s1 (s3 should now be deleted as oldest in list)
      s1 = save.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([s2, s1], save.last_checkpoints)
      self.assertFalse(saver_module.checkpoint_exists(s3))
      self.assertFalse(
          saver_module.checkpoint_exists(save._MetaGraphFilename(s3)))
      self.assertTrue(saver_module.checkpoint_exists(s2))
      self.assertTrue(
          saver_module.checkpoint_exists(save._MetaGraphFilename(s2)))
      self.assertTrue(saver_module.checkpoint_exists(s1))
      self.assertTrue(
          saver_module.checkpoint_exists(save._MetaGraphFilename(s1)))
      self.assertCheckpointState(
          model_checkpoint_path=s1,
          all_model_checkpoint_paths=[s2, s1],
          save_dir=save_dir)

      # Exercise the second helper.

      # Adding s2 again (old s2 is removed first, then new s2 appended)
      s2 = save2.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([s3, s2], save2.last_checkpoints)
      # Created by the first helper.
      self.assertTrue(saver_module.checkpoint_exists(s1))
      self.assertTrue(
          saver_module.checkpoint_exists(save._MetaGraphFilename(s1)))
      # Deleted by the first helper.
      self.assertFalse(saver_module.checkpoint_exists(s3))
      self.assertFalse(
          saver_module.checkpoint_exists(save._MetaGraphFilename(s3)))
      self.assertTrue(saver_module.checkpoint_exists(s2))
      self.assertTrue(
          saver_module.checkpoint_exists(save._MetaGraphFilename(s2)))
      self.assertCheckpointState(
          model_checkpoint_path=s2,
          all_model_checkpoint_paths=[s3, s2],
          save_dir=save_dir)

      # Adding s1 (s3 should now be deleted as oldest in list)
      s1 = save2.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([s2, s1], save2.last_checkpoints)
      self.assertFalse(saver_module.checkpoint_exists(s3))
      self.assertFalse(
          saver_module.checkpoint_exists(save._MetaGraphFilename(s3)))
      self.assertTrue(saver_module.checkpoint_exists(s2))
      self.assertTrue(
          saver_module.checkpoint_exists(save._MetaGraphFilename(s2)))
      self.assertTrue(saver_module.checkpoint_exists(s1))
      self.assertTrue(
          saver_module.checkpoint_exists(save._MetaGraphFilename(s1)))
      self.assertCheckpointState(
          model_checkpoint_path=s1,
          all_model_checkpoint_paths=[s2, s1],
          save_dir=save_dir)

      # Exercise the third helper.

      # Adding s2 again (but helper is unaware of previous s2)
      s2 = save3.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([s2], save3.last_checkpoints)
      # Created by the first helper.
      self.assertTrue(saver_module.checkpoint_exists(s1))
      self.assertTrue(
          saver_module.checkpoint_exists(save._MetaGraphFilename(s1)))
      # Deleted by the first helper.
      self.assertFalse(saver_module.checkpoint_exists(s3))
      self.assertFalse(
          saver_module.checkpoint_exists(save._MetaGraphFilename(s3)))
      self.assertTrue(saver_module.checkpoint_exists(s2))
      self.assertTrue(
          saver_module.checkpoint_exists(save._MetaGraphFilename(s2)))
      # Even though the file for s1 exists, this saver isn't aware of it, which
      # is why it doesn't end up in the checkpoint state.
      self.assertCheckpointState(
          model_checkpoint_path=s2,
          all_model_checkpoint_paths=[s2],
          save_dir=save_dir)

      # Adding s1 (s3 should not be deleted because helper is unaware of it)
      s1 = save3.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([s2, s1], save3.last_checkpoints)
      self.assertFalse(saver_module.checkpoint_exists(s3))
      self.assertFalse(
          saver_module.checkpoint_exists(save._MetaGraphFilename(s3)))
      self.assertTrue(saver_module.checkpoint_exists(s2))
      self.assertTrue(
          saver_module.checkpoint_exists(save._MetaGraphFilename(s2)))
      self.assertTrue(saver_module.checkpoint_exists(s1))
      self.assertTrue(
          saver_module.checkpoint_exists(save._MetaGraphFilename(s1)))
      self.assertCheckpointState(
          model_checkpoint_path=s1,
          all_model_checkpoint_paths=[s2, s1],
          save_dir=save_dir)

  def testSharded(self):
    save_dir = self._get_test_dir("max_to_keep_sharded")

    with session.Session(
        target="",
        config=config_pb2.ConfigProto(device_count={"CPU": 2})) as sess:
      with sess.graph.device("/cpu:0"):
        v0 = variables.Variable(111, name="v0")
      with sess.graph.device("/cpu:1"):
        v1 = variables.Variable(222, name="v1")
      save = saver_module.Saver(
          {
              "v0": v0,
              "v1": v1
          }, sharded=True, max_to_keep=2)
      variables.global_variables_initializer().run()
      self.assertEqual([], save.last_checkpoints)

      s1 = save.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([s1], save.last_checkpoints)
      if save._write_version is saver_pb2.SaverDef.V1:
        self.assertEqual(2, len(gfile.Glob(s1)))
      else:
        self.assertEqual(4, len(gfile.Glob(s1 + "*")))

      self.assertTrue(gfile.Exists(save._MetaGraphFilename(s1)))

      s2 = save.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([s1, s2], save.last_checkpoints)
      if save._write_version is saver_pb2.SaverDef.V1:
        self.assertEqual(2, len(gfile.Glob(s1)))
      else:
        self.assertEqual(4, len(gfile.Glob(s1 + "*")))
      self.assertTrue(gfile.Exists(save._MetaGraphFilename(s1)))
      if save._write_version is saver_pb2.SaverDef.V1:
        self.assertEqual(2, len(gfile.Glob(s2)))
      else:
        self.assertEqual(4, len(gfile.Glob(s2 + "*")))
      self.assertTrue(gfile.Exists(save._MetaGraphFilename(s2)))

      s3 = save.save(sess, os.path.join(save_dir, "s3"))
      self.assertEqual([s2, s3], save.last_checkpoints)
      self.assertEqual(0, len(gfile.Glob(s1 + "*")))
      self.assertFalse(gfile.Exists(save._MetaGraphFilename(s1)))
      if save._write_version is saver_pb2.SaverDef.V1:
        self.assertEqual(2, len(gfile.Glob(s2)))
      else:
        self.assertEqual(4, len(gfile.Glob(s2 + "*")))
      self.assertTrue(gfile.Exists(save._MetaGraphFilename(s2)))
      if save._write_version is saver_pb2.SaverDef.V1:
        self.assertEqual(2, len(gfile.Glob(s3)))
      else:
        self.assertEqual(4, len(gfile.Glob(s3 + "*")))
      self.assertTrue(gfile.Exists(save._MetaGraphFilename(s3)))

  def testNoMaxToKeep(self):
    save_dir = self._get_test_dir("no_max_to_keep")
    save_dir2 = self._get_test_dir("max_to_keep_0")

    with self.test_session() as sess:
      v = variables.Variable(10.0, name="v")
      variables.global_variables_initializer().run()

      # Test max_to_keep being None.
      save = saver_module.Saver({"v": v}, max_to_keep=None)
      self.assertEqual([], save.last_checkpoints)
      s1 = save.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([], save.last_checkpoints)
      self.assertTrue(saver_module.checkpoint_exists(s1))
      s2 = save.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([], save.last_checkpoints)
      self.assertTrue(saver_module.checkpoint_exists(s2))

      # Test max_to_keep being 0.
      save2 = saver_module.Saver({"v": v}, max_to_keep=0)
      self.assertEqual([], save2.last_checkpoints)
      s1 = save2.save(sess, os.path.join(save_dir2, "s1"))
      self.assertEqual([], save2.last_checkpoints)
      self.assertTrue(saver_module.checkpoint_exists(s1))
      s2 = save2.save(sess, os.path.join(save_dir2, "s2"))
      self.assertEqual([], save2.last_checkpoints)
      self.assertTrue(saver_module.checkpoint_exists(s2))

  def testNoMetaGraph(self):
    save_dir = self._get_test_dir("no_meta_graph")

    with self.test_session() as sess:
      v = variables.Variable(10.0, name="v")
      save = saver_module.Saver({"v": v})
      variables.global_variables_initializer().run()

      s1 = save.save(sess, os.path.join(save_dir, "s1"), write_meta_graph=False)
      self.assertTrue(saver_module.checkpoint_exists(s1))
      self.assertFalse(gfile.Exists(save._MetaGraphFilename(s1)))


class KeepCheckpointEveryNHoursTest(test.TestCase):

  def _get_test_dir(self, dirname):
    test_dir = os.path.join(self.get_temp_dir(), dirname)
    gfile.MakeDirs(test_dir)
    return test_dir

  @test_util.run_in_graph_and_eager_modes()
  @test.mock.patch.object(saver_module, "time")
  def testNonSharded(self, mock_time):
    save_dir = self._get_test_dir("keep_checkpoint_every_n_hours")

    with self.test_session() as sess:
      v = variable_scope.variable([10.0], name="v")
      # Run the initializer NOW to avoid the 0.5s overhead of the first Run()
      # call, which throws the test timing off in fastbuild mode.
      self.evaluate(variables.global_variables_initializer())
      # Create a saver that will keep the last 2 checkpoints plus one every 0.7
      # seconds.
      start_time = time.time()
      mock_time.time.return_value = start_time
      save = saver_module.Saver(
          {
              "v": v
          }, max_to_keep=2, keep_checkpoint_every_n_hours=0.7 / 3600)
      self.assertEqual([], save.last_checkpoints)

      # Wait till 1 seconds have elapsed so s1 will be old enough to keep.
      # sleep may return early, don't trust it.
      mock_time.time.return_value = start_time + 1.0
      s1 = save.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([s1], save.last_checkpoints)

      s2 = save.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([s1, s2], save.last_checkpoints)

      # We now have 2 'last_checkpoints': [s1, s2].  The next call to Save(),
      # would normally delete s1, because max_to_keep is 2.  However, s1 is
      # older than 0.7s so we must keep it.
      s3 = save.save(sess, os.path.join(save_dir, "s3"))
      self.assertEqual([s2, s3], save.last_checkpoints)

      # s1 should still be here, we are Not checking now to reduce time
      # variance in the test.

      # We now have 2 'last_checkpoints': [s2, s3], and s1 on disk.  The next
      # call to Save(), will delete s2, because max_to_keep is 2, and because
      # we already kept the old s1. s2 is very close in time to s1 so it gets
      # deleted.
      s4 = save.save(sess, os.path.join(save_dir, "s4"))
      self.assertEqual([s3, s4], save.last_checkpoints)

      # Check that s1 is still here, but s2 is gone.
      self.assertTrue(saver_module.checkpoint_exists(s1))
      self.assertFalse(saver_module.checkpoint_exists(s2))
      self.assertTrue(saver_module.checkpoint_exists(s3))
      self.assertTrue(saver_module.checkpoint_exists(s4))


class SaveRestoreWithVariableNameMap(test.TestCase):

  def _testNonReshape(self, variable_op):
    save_path = os.path.join(self.get_temp_dir(), "non_reshape")

    with self.test_session(graph=ops_lib.Graph()) as sess:
      # Build a graph with 2 parameter nodes, and Save and
      # Restore nodes for them.
      v0 = variable_op(10.0, name="v0")
      v1 = variable_op(20.0, name="v1")
      save = saver_module.Saver({"save_prefix/v0": v0, "save_prefix/v1": v1})
      self.evaluate(variables.global_variables_initializer())

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))

      # Save the initialized values in the file at "save_path"
      # Use a variable name map to set the saved tensor names
      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

      # Verify that the original names are not in the Saved file
      save = saver_module.Saver({"v0": v0, "v1": v1})
      with self.assertRaisesOpError("not found in checkpoint"):
        save.restore(sess, save_path)

    # Verify that the mapped names are present in the Saved file and can be
    # Restored using remapped names.
    with self.test_session(graph=ops_lib.Graph()) as sess:
      v0 = variable_op(-1.0, name="v0")
      v1 = variable_op(-1.0, name="v1")

      if not context.executing_eagerly():
        with self.assertRaisesOpError("uninitialized"):
          self.evaluate(v0)
        with self.assertRaisesOpError("uninitialized"):
          self.evaluate(v1)

      save = saver_module.Saver({"save_prefix/v0": v0, "save_prefix/v1": v1})
      save.restore(sess, save_path)

      # Check that the parameter nodes have been restored.
      if not context.executing_eagerly():
        self.assertEqual(10.0, self.evaluate(v0))
        self.assertEqual(20.0, self.evaluate(v1))

    # Add a prefix to the node names in the current graph and Restore using
    # remapped names.
    with self.test_session(graph=ops_lib.Graph()) as sess:
      v0 = variable_op(-1.0, name="restore_prefix/v0")
      v1 = variable_op(-1.0, name="restore_prefix/v1")

      if not context.executing_eagerly():
        with self.assertRaisesOpError("uninitialized"):
          self.evaluate(v0)
        with self.assertRaisesOpError("uninitialized"):
          self.evaluate(v1)

      # Restore the saved values in the parameter nodes.
      save = saver_module.Saver({"save_prefix/v0": v0, "save_prefix/v1": v1})
      save.restore(sess, save_path)

      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))

  @test_util.run_in_graph_and_eager_modes()
  def testNonReshapeResourceVariable(self):
    self._testNonReshape(resource_variable_ops.ResourceVariable)

  def testNonReshapeVariable(self):
    self._testNonReshape(variables.Variable)


class LatestCheckpointWithRelativePaths(test.TestCase):

  @staticmethod
  @contextlib.contextmanager
  def tempWorkingDir(temppath):
    cwd = os.getcwd()
    os.chdir(temppath)
    try:
      yield
    finally:
      os.chdir(cwd)

  @staticmethod
  @contextlib.contextmanager
  def tempDir():
    tempdir = tempfile.mkdtemp()
    try:
      yield tempdir
    finally:
      shutil.rmtree(tempdir)

  def testNameCollision(self):
    # Make sure we have a clean directory to work in.
    with self.tempDir() as tempdir:
      # Jump to that directory until this test is done.
      with self.tempWorkingDir(tempdir):
        # Save training snapshots to a relative path.
        traindir = "train/"
        os.mkdir(traindir)
        # Collides with the default name of the checkpoint state file.
        filepath = os.path.join(traindir, "checkpoint")

        with self.test_session() as sess:
          unused_a = variables.Variable(0.0)  # So that Saver saves something.
          variables.global_variables_initializer().run()

          # Should fail.
          saver = saver_module.Saver(sharded=False)
          with self.assertRaisesRegexp(ValueError, "collides with"):
            saver.save(sess, filepath)

          # Succeeds: the file will be named "checkpoint-<step>".
          saver.save(sess, filepath, global_step=1)
          self.assertIsNotNone(saver_module.latest_checkpoint(traindir))

          # Succeeds: the file will be named "checkpoint-<i>-of-<n>".
          saver = saver_module.Saver(sharded=True)
          saver.save(sess, filepath)
          self.assertIsNotNone(saver_module.latest_checkpoint(traindir))

          # Succeeds: the file will be named "checkpoint-<step>-<i>-of-<n>".
          saver = saver_module.Saver(sharded=True)
          saver.save(sess, filepath, global_step=1)
          self.assertIsNotNone(saver_module.latest_checkpoint(traindir))

  def testRelativePath(self):
    # Make sure we have a clean directory to work in.
    with self.tempDir() as tempdir:

      # Jump to that directory until this test is done.
      with self.tempWorkingDir(tempdir):

        # Save training snapshots to a relative path.
        traindir = "train/"
        os.mkdir(traindir)

        filename = "snapshot"
        filepath = os.path.join(traindir, filename)

        with self.test_session() as sess:
          # Build a simple graph.
          v0 = variables.Variable(0.0)
          inc = v0.assign_add(1.0)

          save = saver_module.Saver({"v0": v0})

          # Record a short training history.
          variables.global_variables_initializer().run()
          save.save(sess, filepath, global_step=0)
          inc.eval()
          save.save(sess, filepath, global_step=1)
          inc.eval()
          save.save(sess, filepath, global_step=2)

        with self.test_session() as sess:
          # Build a new graph with different initialization.
          v0 = variables.Variable(-1.0)

          # Create a new saver.
          save = saver_module.Saver({"v0": v0})
          variables.global_variables_initializer().run()

          # Get the most recent checkpoint name from the training history file.
          name = saver_module.latest_checkpoint(traindir)
          self.assertIsNotNone(name)

          # Restore "v0" from that checkpoint.
          save.restore(sess, name)
          self.assertEqual(v0.eval(), 2.0)


class CheckpointStateTest(test.TestCase):

  def _get_test_dir(self, dirname):
    test_dir = os.path.join(self.get_temp_dir(), dirname)
    gfile.MakeDirs(test_dir)
    return test_dir

  def testAbsPath(self):
    save_dir = self._get_test_dir("abs_paths")
    abs_path = os.path.join(save_dir, "model-0")
    ckpt = saver_module.generate_checkpoint_state_proto(save_dir, abs_path)
    self.assertEqual(ckpt.model_checkpoint_path, abs_path)
    self.assertTrue(os.path.isabs(ckpt.model_checkpoint_path))
    self.assertEqual(len(ckpt.all_model_checkpoint_paths), 1)
    self.assertEqual(ckpt.all_model_checkpoint_paths[-1], abs_path)

  def testRelPath(self):
    train_dir = "train"
    model = os.path.join(train_dir, "model-0")
    # model_checkpoint_path should have no "train" directory part.
    new_rel_path = "model-0"
    ckpt = saver_module.generate_checkpoint_state_proto(train_dir, model)
    self.assertEqual(ckpt.model_checkpoint_path, new_rel_path)
    self.assertEqual(len(ckpt.all_model_checkpoint_paths), 1)
    self.assertEqual(ckpt.all_model_checkpoint_paths[-1], new_rel_path)

  def testAllModelCheckpointPaths(self):
    save_dir = self._get_test_dir("all_models_test")
    abs_path = os.path.join(save_dir, "model-0")
    for paths in [None, [], ["model-2"]]:
      ckpt = saver_module.generate_checkpoint_state_proto(
          save_dir, abs_path, all_model_checkpoint_paths=paths)
      self.assertEqual(ckpt.model_checkpoint_path, abs_path)
      self.assertTrue(os.path.isabs(ckpt.model_checkpoint_path))
      self.assertEqual(
          len(ckpt.all_model_checkpoint_paths), len(paths) if paths else 1)
      self.assertEqual(ckpt.all_model_checkpoint_paths[-1], abs_path)

  def testUpdateCheckpointState(self):
    save_dir = self._get_test_dir("update_checkpoint_state")
    os.chdir(save_dir)
    # Make a temporary train directory.
    train_dir = "train"
    os.mkdir(train_dir)
    abs_path = os.path.join(save_dir, "model-0")
    rel_path = os.path.join("train", "model-2")
    saver_module.update_checkpoint_state(
        train_dir, rel_path, all_model_checkpoint_paths=[abs_path, rel_path])
    ckpt = saver_module.get_checkpoint_state(train_dir)
    self.assertEqual(ckpt.model_checkpoint_path, rel_path)
    self.assertEqual(len(ckpt.all_model_checkpoint_paths), 2)
    self.assertEqual(ckpt.all_model_checkpoint_paths[-1], rel_path)
    self.assertEqual(ckpt.all_model_checkpoint_paths[0], abs_path)

  def testUpdateCheckpointStateSaveRelativePaths(self):
    save_dir = self._get_test_dir("update_checkpoint_state")
    os.chdir(save_dir)
    abs_path2 = os.path.join(save_dir, "model-2")
    rel_path2 = "model-2"
    abs_path0 = os.path.join(save_dir, "model-0")
    rel_path0 = "model-0"
    saver_module._update_checkpoint_state(  # pylint: disable=protected-access
        save_dir=save_dir,
        model_checkpoint_path=abs_path2,
        all_model_checkpoint_paths=[rel_path0, abs_path2],
        save_relative_paths=True)

    # File should contain relative paths.
    file_content = file_io.read_file_to_string(
        os.path.join(save_dir, "checkpoint"))
    ckpt = CheckpointState()
    text_format.Merge(file_content, ckpt)
    self.assertEqual(ckpt.model_checkpoint_path, rel_path2)
    self.assertEqual(len(ckpt.all_model_checkpoint_paths), 2)
    self.assertEqual(ckpt.all_model_checkpoint_paths[-1], rel_path2)
    self.assertEqual(ckpt.all_model_checkpoint_paths[0], rel_path0)

    # get_checkpoint_state should return absolute paths.
    ckpt = saver_module.get_checkpoint_state(save_dir)
    self.assertEqual(ckpt.model_checkpoint_path, abs_path2)
    self.assertEqual(len(ckpt.all_model_checkpoint_paths), 2)
    self.assertEqual(ckpt.all_model_checkpoint_paths[-1], abs_path2)
    self.assertEqual(ckpt.all_model_checkpoint_paths[0], abs_path0)

  def testCheckPointStateFailsWhenIncomplete(self):
    save_dir = self._get_test_dir("checkpoint_state_fails_when_incomplete")
    os.chdir(save_dir)
    ckpt_path = os.path.join(save_dir, "checkpoint")
    ckpt_file = open(ckpt_path, "w")
    ckpt_file.write("")
    ckpt_file.close()
    with self.assertRaises(ValueError):
      saver_module.get_checkpoint_state(save_dir)

  def testCheckPointCompletesRelativePaths(self):
    save_dir = self._get_test_dir("checkpoint_completes_relative_paths")
    os.chdir(save_dir)
    ckpt_path = os.path.join(save_dir, "checkpoint")
    ckpt_file = open(ckpt_path, "w")
    ckpt_file.write("""
        model_checkpoint_path: "./model.ckpt-687529"
        all_model_checkpoint_paths: "./model.ckpt-687500"
        all_model_checkpoint_paths: "./model.ckpt-687529"
        """)
    ckpt_file.close()
    ckpt = saver_module.get_checkpoint_state(save_dir)
    self.assertEqual(ckpt.model_checkpoint_path,
                     os.path.join(save_dir, "./model.ckpt-687529"))
    self.assertEqual(ckpt.all_model_checkpoint_paths[0],
                     os.path.join(save_dir, "./model.ckpt-687500"))
    self.assertEqual(ckpt.all_model_checkpoint_paths[1],
                     os.path.join(save_dir, "./model.ckpt-687529"))


class MetaGraphTest(test.TestCase):

  def _get_test_dir(self, dirname):
    test_dir = os.path.join(self.get_temp_dir(), dirname)
    gfile.MakeDirs(test_dir)
    return test_dir

  def testAddCollectionDef(self):
    test_dir = self._get_test_dir("good_collection")
    filename = os.path.join(test_dir, "metafile")
    with self.test_session():
      # Creates a graph.
      v0 = variables.Variable(1.0, name="v0")
      control_flow_ops.cond(
          math_ops.less(v0, 10), lambda: math_ops.add(v0, 1),
          lambda: math_ops.subtract(v0, 1))
      control_flow_ops.while_loop(lambda i: math_ops.less(i, 10),
                                  lambda i: math_ops.add(i, 1), [v0])
      var = variables.Variable(constant_op.constant(0, dtype=dtypes.int64))
      count_up_to = var.count_up_to(3)
      input_queue = data_flow_ops.FIFOQueue(
          30, dtypes.float32, shared_name="collection_queue")
      qr = queue_runner_impl.QueueRunner(input_queue, [count_up_to])
      variables.global_variables_initializer()
      # Creates a saver.
      save = saver_module.Saver({"v0": v0})
      # Adds a set of collections.
      ops_lib.add_to_collection("int_collection", 3)
      ops_lib.add_to_collection("float_collection", 3.5)
      ops_lib.add_to_collection("string_collection", "hello")
      ops_lib.add_to_collection("variable_collection", v0)
      # Add QueueRunners.
      queue_runner_impl.add_queue_runner(qr)
      # Adds user_defined proto in three formats: string, bytes and Any.
      queue_runner = queue_runner_pb2.QueueRunnerDef(queue_name="test_queue")
      ops_lib.add_to_collection("user_defined_string_collection",
                                str(queue_runner))
      ops_lib.add_to_collection("user_defined_bytes_collection",
                                queue_runner.SerializeToString())
      any_buf = Any()
      any_buf.Pack(queue_runner)
      ops_lib.add_to_collection("user_defined_any_collection", any_buf)

      # Generates MetaGraphDef.
      meta_graph_def = save.export_meta_graph(filename)
      self.assertTrue(meta_graph_def.HasField("saver_def"))
      self.assertTrue(meta_graph_def.HasField("graph_def"))
      self.assertTrue(meta_graph_def.HasField("meta_info_def"))
      self.assertNotEqual(meta_graph_def.meta_info_def.tensorflow_version, "")
      self.assertNotEqual(meta_graph_def.meta_info_def.tensorflow_git_version,
                          "")
      collection_def = meta_graph_def.collection_def
      self.assertEqual(len(collection_def), 12)

    with ops_lib.Graph().as_default():
      # Restores from MetaGraphDef.
      new_saver = saver_module.import_meta_graph(filename)
      # Generates a new MetaGraphDef.
      new_meta_graph_def = new_saver.export_meta_graph()
      # It should be the same as the original.

    test_util.assert_meta_graph_protos_equal(
        self, meta_graph_def, new_meta_graph_def)

  def testAddCollectionDefFails(self):
    with self.test_session():
      # Creates a graph.
      v0 = variables.Variable(10.0, name="v0")
      # Creates a saver.
      save = saver_module.Saver({"v0": v0})
      # Generates MetaGraphDef.
      meta_graph_def = meta_graph_pb2.MetaGraphDef()

      # Verifies that collection with unsupported key will not be added.
      ops_lib.add_to_collection(save, 3)
      save._add_collection_def(meta_graph_def, save)
      self.assertEqual(len(meta_graph_def.collection_def), 0)

      # Verifies that collection where item type does not match expected
      # type will not be added.
      ops_lib.add_to_collection("int_collection", 3)
      ops_lib.add_to_collection("int_collection", 3.5)
      save._add_collection_def(meta_graph_def, "int_collection")
      self.assertEqual(len(meta_graph_def.collection_def), 0)

  def _testMultiSaverCollectionSave(self, test_dir):
    filename = os.path.join(test_dir, "metafile")
    saver0_ckpt = os.path.join(test_dir, "saver0.ckpt")
    saver1_ckpt = os.path.join(test_dir, "saver1.ckpt")
    with self.test_session(graph=ops_lib.Graph()) as sess:
      # Creates a graph.
      v0 = variables.Variable([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], name="v0")
      v1 = variables.Variable(11.0, name="v1")
      # Creates 2 savers.
      saver0 = saver_module.Saver({"v0": v0}, name="saver0")
      saver1 = saver_module.Saver({"v1": v1}, name="saver1")
      ops_lib.add_to_collection("savers", saver0)
      ops_lib.add_to_collection("savers", saver1)
      variables.global_variables_initializer().run()
      # Saves to different checkpoints.
      saver0.save(sess, saver0_ckpt)
      saver1.save(sess, saver1_ckpt)
      # Generates MetaGraphDef.
      meta_graph_def = saver_module.export_meta_graph(filename)
      meta_graph_def0 = saver0.export_meta_graph()
      meta_graph_def1 = saver1.export_meta_graph()

      # Verifies that there is no saver_def in meta_graph_def.
      self.assertFalse(meta_graph_def.HasField("saver_def"))
      # Verifies that there is saver_def in meta_graph_def0 and 1.
      self.assertTrue(meta_graph_def0.HasField("saver_def"))
      self.assertTrue(meta_graph_def1.HasField("saver_def"))

      # Verifies SAVERS is saved as bytes_list for meta_graph_def.
      collection_def = meta_graph_def.collection_def["savers"]
      kind = collection_def.WhichOneof("kind")
      self.assertEqual(kind, "bytes_list")
      # Verifies that there are 2 entries in SAVERS collection.
      savers = getattr(collection_def, kind)
      self.assertEqual(2, len(savers.value))

      # Verifies SAVERS collection is saved as bytes_list for meta_graph_def0.
      collection_def = meta_graph_def0.collection_def["savers"]
      kind = collection_def.WhichOneof("kind")
      self.assertEqual(kind, "bytes_list")
      # Verifies that there are 2 entries in SAVERS collection.
      savers = getattr(collection_def, kind)
      self.assertEqual(2, len(savers.value))

  def _testMultiSaverCollectionRestore(self, test_dir):
    filename = os.path.join(test_dir, "metafile")
    saver0_ckpt = os.path.join(test_dir, "saver0.ckpt")
    saver1_ckpt = os.path.join(test_dir, "saver1.ckpt")
    with self.test_session(graph=ops_lib.Graph()) as sess:
      # Imports from meta_graph.
      saver_module.import_meta_graph(filename)
      # Retrieves SAVERS collection. Verifies there are 2 entries.
      savers = ops_lib.get_collection("savers")
      self.assertEqual(2, len(savers))
      # Retrieves saver0. Verifies that new_saver0 can restore v0, but not v1.
      new_saver0 = savers[0]
      new_saver0.restore(sess, saver0_ckpt)
      v0 = sess.graph.get_tensor_by_name("v0:0")
      v1 = sess.graph.get_tensor_by_name("v1:0")
      self.assertAllEqual([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], v0.eval())
      self.assertEqual([3, 2], v0.get_shape())
      self.assertEqual([], v1.get_shape())
      with self.assertRaisesWithPredicateMatch(
          errors_impl.OpError, lambda e: "uninitialized value v1" in e.message):
        sess.run(v1)
      # Retrieves saver1. Verifies that new_saver1 can restore v1.
      new_saver1 = savers[1]
      new_saver1.restore(sess, saver1_ckpt)
      v1 = sess.graph.get_tensor_by_name("v1:0")
      self.assertEqual(11.0, v1.eval())

  def testMultiSaverCollection(self):
    test_dir = self._get_test_dir("saver_collection")
    self._testMultiSaverCollectionSave(test_dir)
    self._testMultiSaverCollectionRestore(test_dir)

  def testClearExtraneousSavers(self):
    test_dir = self._get_test_dir("clear_extraneous_savers")
    filename = os.path.join(test_dir, "metafile")
    saver0_ckpt = os.path.join(test_dir, "saver0.ckpt")
    saver1_ckpt = os.path.join(test_dir, "saver1.ckpt")
    with self.test_session(graph=ops_lib.Graph()) as sess:
      # Creates a graph.
      v0 = variables.Variable([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], name="v0")
      v1 = variables.Variable(11.0, name="v1")

      # Creates 2 savers.
      saver0 = saver_module.Saver({"v0": v0}, name="saver0")
      saver1 = saver_module.Saver({"v1": v1}, name="saver1")
      ops_lib.add_to_collection("savers", saver0)
      ops_lib.add_to_collection("savers", saver1)
      variables.global_variables_initializer().run()

      # Saves to different checkpoints.
      saver0.save(sess, saver0_ckpt)
      saver1.save(sess, saver1_ckpt)

      # Generates MetaGraphDef.
      meta_graph_def = saver_module.export_meta_graph(filename)
      meta_graph_def0 = saver0.export_meta_graph()
      meta_graph_def1 = saver1.export_meta_graph(clear_extraneous_savers=True)

      # Verifies that there is no saver_def in meta_graph_def.
      self.assertFalse(meta_graph_def.HasField("saver_def"))
      # Verifies that there is saver_def in meta_graph_def0 and 1.
      self.assertTrue(meta_graph_def0.HasField("saver_def"))
      self.assertTrue(meta_graph_def1.HasField("saver_def"))

      # Verifies SAVERS is saved as bytes_list for meta_graph_def.
      collection_def = meta_graph_def.collection_def["savers"]
      kind = collection_def.WhichOneof("kind")
      self.assertEqual(kind, "bytes_list")

      # Verifies that there are 2 entries in SAVERS collection.
      savers = getattr(collection_def, kind)
      self.assertEqual(2, len(savers.value))

      # Verifies SAVERS collection is saved as bytes_list for meta_graph_def1.
      collection_def = meta_graph_def1.collection_def["savers"]
      kind = collection_def.WhichOneof("kind")
      self.assertEqual(kind, "bytes_list")

      # Verifies that there is 1 entry in SAVERS collection.
      savers = getattr(collection_def, kind)
      self.assertEqual(1, len(savers.value))

      # Verifies that saver0 graph nodes are omitted from the saver1 export
      self.assertEqual(29, len(meta_graph_def0.graph_def.node))
      self.assertEqual(19, len(meta_graph_def1.graph_def.node))

  def testBinaryAndTextFormat(self):
    test_dir = self._get_test_dir("binary_and_text")
    filename = os.path.join(test_dir, "metafile")
    with self.test_session(graph=ops_lib.Graph()):
      # Creates a graph.
      variables.Variable(10.0, name="v0")
      # Exports the graph as binary format.
      saver_module.export_meta_graph(filename, as_text=False)
    with self.test_session(graph=ops_lib.Graph()):
      # Imports the binary format graph.
      saver = saver_module.import_meta_graph(filename)
      self.assertIsNotNone(saver)
      # Exports the graph as text format.
      saver.export_meta_graph(filename, as_text=True)
    with self.test_session(graph=ops_lib.Graph()):
      # Imports the text format graph.
      saver_module.import_meta_graph(filename)
      # Writes wrong contents to the file.
      graph_io.write_graph(saver.as_saver_def(),
                           os.path.dirname(filename),
                           os.path.basename(filename))
    with self.test_session(graph=ops_lib.Graph()):
      # Import should fail.
      with self.assertRaisesWithPredicateMatch(IOError,
                                               lambda e: "Cannot parse file"):
        saver_module.import_meta_graph(filename)
      # Deletes the file
      gfile.Remove(filename)
      with self.assertRaisesWithPredicateMatch(IOError,
                                               lambda e: "does not exist"):
        saver_module.import_meta_graph(filename)

  def testSliceVariable(self):
    test_dir = self._get_test_dir("slice_saver")
    filename = os.path.join(test_dir, "metafile")
    with self.test_session():
      v1 = variables.Variable([20.0], name="v1")
      v2 = variables.Variable([20.0], name="v2")
      v2._set_save_slice_info(
          variables.Variable.SaveSliceInfo("v1", [1], [0], [1]))

      # The names are different and will work.
      slice_saver = saver_module.Saver({"first": v1, "second": v2})
      variables.global_variables_initializer().run()
      # Exports to meta_graph
      meta_graph_def = slice_saver.export_meta_graph(filename)

    with ops_lib.Graph().as_default():
      # Restores from MetaGraphDef.
      new_saver = saver_module.import_meta_graph(filename)
      self.assertIsNotNone(new_saver)
      # Generates a new MetaGraphDef.
      new_meta_graph_def = new_saver.export_meta_graph()
      # It should be the same as the original.
      test_util.assert_meta_graph_protos_equal(self, meta_graph_def,
                                               new_meta_graph_def)

  def _testGraphExtensionSave(self, test_dir):
    filename = os.path.join(test_dir, "metafile")
    saver0_ckpt = os.path.join(test_dir, "saver0.ckpt")
    # Creates an inference graph.
    # Hidden 1
    images = constant_op.constant(1.2, dtypes.float32, shape=[100, 28])
    with ops_lib.name_scope("hidden1"):
      weights = variables.Variable(
          random_ops.truncated_normal(
              [28, 128], stddev=1.0 / math.sqrt(float(28))),
          name="weights")
      # The use of control_flow_ops.cond here is purely for adding test coverage
      # the save and restore of control flow context (which doesn't make any
      # sense here from a machine learning perspective).  The typical biases is
      # a simple Variable without the conditions.
      biases = variables.Variable(
          control_flow_ops.cond(
              math_ops.less(random.random(), 0.5),
              lambda: array_ops.ones([128]), lambda: array_ops.zeros([128])),
          name="biases")
      hidden1 = nn_ops.relu(math_ops.matmul(images, weights) + biases)
    # Hidden 2
    with ops_lib.name_scope("hidden2"):
      weights = variables.Variable(
          random_ops.truncated_normal(
              [128, 32], stddev=1.0 / math.sqrt(float(128))),
          name="weights")

      # The use of control_flow_ops.while_loop here is purely for adding test
      # coverage the save and restore of control flow context (which doesn't
      # make any sense here from a machine learning perspective).  The typical
      # biases is a simple Variable without the conditions.
      def loop_cond(it, _):
        return it < 2

      def loop_body(it, biases):
        biases += constant_op.constant(0.1, shape=[32])
        return it + 1, biases

      _, biases = control_flow_ops.while_loop(
          loop_cond, loop_body,
          [constant_op.constant(0), variables.Variable(array_ops.zeros([32]))])
      hidden2 = nn_ops.relu(math_ops.matmul(hidden1, weights) + biases)
    # Linear
    with ops_lib.name_scope("softmax_linear"):
      weights = variables.Variable(
          random_ops.truncated_normal(
              [32, 10], stddev=1.0 / math.sqrt(float(32))),
          name="weights")
      biases = variables.Variable(array_ops.zeros([10]), name="biases")
      logits = math_ops.matmul(hidden2, weights) + biases
      ops_lib.add_to_collection("logits", logits)
    init_all_op = variables.global_variables_initializer()

    with self.test_session() as sess:
      # Initializes all the variables.
      sess.run(init_all_op)
      # Runs to logit.
      sess.run(logits)
      # Creates a saver.
      saver0 = saver_module.Saver()
      saver0.save(sess, saver0_ckpt)
      # Generates MetaGraphDef.
      saver0.export_meta_graph(filename)

  def _testGraphExtensionRestore(self, test_dir):
    filename = os.path.join(test_dir, "metafile")
    train_filename = os.path.join(test_dir, "train_metafile")
    saver0_ckpt = os.path.join(test_dir, "saver0.ckpt")
    with self.test_session(graph=ops_lib.Graph()) as sess:
      # Restores from MetaGraphDef.
      new_saver = saver_module.import_meta_graph(filename)
      # Generates a new MetaGraphDef.
      new_saver.export_meta_graph()
      # Restores from checkpoint.
      new_saver.restore(sess, saver0_ckpt)
      # Adds loss and train.
      labels = constant_op.constant(0, dtypes.int32, shape=[100], name="labels")
      batch_size = array_ops.size(labels)
      labels = array_ops.expand_dims(labels, 1)
      indices = array_ops.expand_dims(math_ops.range(0, batch_size), 1)
      concated = array_ops.concat([indices, labels], 1)
      onehot_labels = sparse_ops.sparse_to_dense(
          concated, array_ops.stack([batch_size, 10]), 1.0, 0.0)
      logits = ops_lib.get_collection("logits")[0]
      cross_entropy = nn_ops.softmax_cross_entropy_with_logits(
          labels=onehot_labels, logits=logits, name="xentropy")
      loss = math_ops.reduce_mean(cross_entropy, name="xentropy_mean")

      summary.scalar("loss", loss)
      # Creates the gradient descent optimizer with the given learning rate.
      optimizer = gradient_descent.GradientDescentOptimizer(0.01)

      # Runs train_op.
      train_op = optimizer.minimize(loss)
      ops_lib.add_to_collection("train_op", train_op)

      # Runs train_op.
      sess.run(train_op)

      # Generates MetaGraphDef.
      saver_module.export_meta_graph(train_filename)

  def _testRestoreFromTrainGraphWithControlContext(self, test_dir):
    train_filename = os.path.join(test_dir, "train_metafile")
    saver0_ckpt = os.path.join(test_dir, "saver0.ckpt")
    with self.test_session(graph=ops_lib.Graph()) as sess:
      # Restores from MetaGraphDef.
      new_saver = saver_module.import_meta_graph(train_filename)
      # Restores from checkpoint.
      new_saver.restore(sess, saver0_ckpt)
      train_op = ops_lib.get_collection("train_op")[0]
      sess.run(train_op)

  def testGraphExtension(self):
    test_dir = self._get_test_dir("graph_extension")
    self._testGraphExtensionSave(test_dir)
    self._testGraphExtensionRestore(test_dir)
    self._testRestoreFromTrainGraphWithControlContext(test_dir)

  def _testGradientSerDes(self, graph_fn):
    """Tests that gradients can be computed after exporting and importing.

    Builds a graph, exports it, and verifies that it can be imported and the
    gradient can be built and run correctly.

    Args:
      graph_fn: takes a single float Tensor argument as input, outputs a single
        Tensor
    """
    test_dir = self._get_test_dir("nested_control_flow")
    filename = os.path.join(test_dir, "metafile")
    saver_ckpt = os.path.join(test_dir, "saver.ckpt")

    # Create while loop using `outer_body_fn`.
    with ops_lib.Graph().as_default():
      var = variables.Variable(0.0)
      var_name = var.name
      output = graph_fn(var)
      output_name = output.name
      init_op = variables.global_variables_initializer()

      # Generate a MetaGraphDef containing the while loop.
      with session.Session() as sess:
        sess.run(init_op)
        sess.run(output)
        saver = saver_module.Saver()
        saver.save(sess, saver_ckpt)
        saver.export_meta_graph(filename)

      # Build and run the gradients of the while loop. We use this below to
      # verify that the gradients are correct with an imported MetaGraphDef.
      grad = gradients_impl.gradients([output], [var])
      # Turn off constant folding to avoid breaking testNestedControlFlowSerDes.
      # It appears that a missing control dependency in the gradient graph
      # causes the fetch node to not be triggered.
      no_constfold_config = config_pb2.ConfigProto()
      no_constfold_config.graph_options.rewrite_options.constant_folding = (
          rewriter_config_pb2.RewriterConfig.OFF)
      with session.Session(config=no_constfold_config) as sess:
        sess.run(init_op)
        expected_grad_value = sess.run(grad)

    # Restore the MetaGraphDef into a new Graph.
    with ops_lib.Graph().as_default():
      with session.Session() as sess:
        saver = saver_module.import_meta_graph(filename)
        saver.restore(sess, saver_ckpt)

      # Make sure we can still build gradients and get the same result.
      var = ops_lib.get_default_graph().get_tensor_by_name(var_name)
      output = ops_lib.get_default_graph().get_tensor_by_name(output_name)
      grad = gradients_impl.gradients([output], [var])

      init_op = variables.global_variables_initializer()

      with session.Session(config=no_constfold_config) as sess:
        sess.run(init_op)
        actual_grad_value = sess.run(grad)
        self.assertEqual(expected_grad_value, actual_grad_value)

  def _testWhileLoopAndGradientSerDes(self, outer_body_fn):
    # Build a while loop with `outer_body_fn`, export it, and verify that it can
    # be imported and the gradient can be built and run correctly.
    # pylint: disable=g-long-lambda
    return self._testGradientSerDes(
        lambda x: control_flow_ops.while_loop(
            lambda i, y: i < 5, outer_body_fn, [0, x])[1])
    # pylint: enable=g-long-lambda

  def testNestedWhileLoopsSerDes(self):
    # Test two simple nested while loops.
    def body(i, x):
      _, r = control_flow_ops.while_loop(lambda j, y: j < 3,
                                         lambda j, y: (j + 1, y + x),
                                         [0, 0.0])
      return i + 1, x + r
    self._testWhileLoopAndGradientSerDes(body)

  def testNestedControlFlowSerDes(self):
    # Test while loop in a cond in a while loop.
    # pylint: disable=g-long-lambda
    def body(i, x):
      cond_result = control_flow_ops.cond(
          i > 0,
          lambda: control_flow_ops.while_loop(
              lambda j, y: j < 3,
              lambda j, y: (j + 1, y + x),
              [0, 0.0])[1],
          lambda: x)
      return i + 1, cond_result
    # pylint: enable=g-long-lambda
    self._testWhileLoopAndGradientSerDes(body)

  def testNestedCondsSerDes(self):
    # Test conds in a cond.
    # pylint: disable=g-long-lambda
    self._testGradientSerDes(lambda x: control_flow_ops.cond(
        x > 0,
        lambda: control_flow_ops.cond(x > 3,
                                      lambda: array_ops.identity(x),
                                      lambda: math_ops.multiply(x, 2.0)),
        lambda: control_flow_ops.cond(x < -3,
                                      lambda: constant_op.constant(1.0),
                                      lambda: math_ops.multiply(x, -1.0))))
    # pylint: enable=g-long-lambda

  def testStrippedOpListDef(self):
    with self.test_session():
      # Creates a graph.
      v0 = variables.Variable(0.0)
      var = variables.Variable(10.0)
      math_ops.add(v0, var)

      @function.Defun(dtypes.float32)
      def minus_one(x):
        return x - 1

      minus_one(array_ops.identity(v0))
      save = saver_module.Saver({"v0": v0})
      variables.global_variables_initializer()

      # Generates MetaGraphDef.
      meta_graph_def = save.export_meta_graph()
      ops = [o.name for o in meta_graph_def.meta_info_def.stripped_op_list.op]
      if save._write_version is saver_pb2.SaverDef.V1:
        self.assertEqual(ops, [
            "Add", "Assign", "Const", "Identity", "NoOp", "RestoreV2",
            "SaveSlices", "Sub", "VariableV2"
        ])
      else:
        self.assertEqual(ops, [
            "Add", "Assign", "Const", "Identity", "NoOp", "RestoreV2", "SaveV2",
            "Sub", "VariableV2"
        ])

      # Test calling stripped_op_list_for_graph directly
      op_list = meta_graph.stripped_op_list_for_graph(meta_graph_def.graph_def)
      self.assertEqual(ops, [o.name for o in op_list.op])
      for o in op_list.op:
        self.assertEqual(o.summary, "")
        self.assertEqual(o.description, "")

  def testStripDefaultValuedAttrs(self):
    """Verifies that default valued attrs are stripped, unless disabled."""

    # With strip_default_attrs enabled, attributes "T" (float32) and "Tout"
    # (complex64) in the "Complex" op must be removed.
    with self.test_session():
      real_num = variables.Variable(1.0, dtype=dtypes.float32, name="real")
      imag_num = variables.Variable(2.0, dtype=dtypes.float32, name="imag")
      math_ops.complex(real_num, imag_num, name="complex")

      save = saver_module.Saver({"real_num": real_num, "imag_num": imag_num})
      variables.global_variables_initializer()

      meta_graph_def = save.export_meta_graph(strip_default_attrs=True)
      node_def = test_util.get_node_def_from_graph("complex",
                                                   meta_graph_def.graph_def)
      self.assertNotIn("T", node_def.attr)
      self.assertNotIn("Tout", node_def.attr)

    # With strip_default_attrs disabled, attributes "T" (float32) and "Tout"
    # (complex64) in the "Complex" op must *not* be removed, even if they map
    # to their defaults.
    with self.test_session(graph=ops_lib.Graph()):
      real_num = variables.Variable(1.0, dtype=dtypes.float32, name="real")
      imag_num = variables.Variable(2.0, dtype=dtypes.float32, name="imag")
      math_ops.complex(real_num, imag_num, name="complex")

      save = saver_module.Saver({"real_num": real_num, "imag_num": imag_num})
      variables.global_variables_initializer()

      meta_graph_def = save.export_meta_graph(strip_default_attrs=False)
      node_def = test_util.get_node_def_from_graph("complex",
                                                   meta_graph_def.graph_def)
      self.assertIn("T", node_def.attr)
      self.assertIn("Tout", node_def.attr)

  def testImportIntoNamescope(self):
    # Test that we can import a meta graph into a namescope.
    test_dir = self._get_test_dir("import_into_namescope")
    filename = os.path.join(test_dir, "ckpt")
    image = array_ops.placeholder(dtypes.float32, [None, 784], name="image")
    label = array_ops.placeholder(dtypes.float32, [None, 10], name="label")
    with session.Session() as sess:
      weights = variables.Variable(
          random_ops.random_uniform([784, 10]), name="weights")
      bias = variables.Variable(array_ops.zeros([10]), name="bias")
      logit = nn_ops.relu(math_ops.matmul(image, weights) + bias, name="logits")
      nn_ops.softmax(logit, name="prediction")
      cost = nn_ops.softmax_cross_entropy_with_logits(labels=label,
                                                      logits=logit, name="cost")
      adam.AdamOptimizer().minimize(cost, name="optimize")
      saver = saver_module.Saver()
      sess.run(variables.global_variables_initializer())
      saver.save(sess, filename)

    graph = ops_lib.Graph()
    with session.Session(graph=graph) as sess:
      new_saver = saver_module.import_meta_graph(
          filename + ".meta", graph=graph, import_scope="new_model")
      new_saver.restore(sess, filename)
      sess.run(["new_model/optimize"], {
          "new_model/image:0": np.random.random([1, 784]),
          "new_model/label:0": np.random.randint(
              10, size=[1, 10])
      })

  def testImportIntoNamescopeWithoutVariables(self):
    # Save a simple graph that contains no variables into a checkpoint.
    test_dir = self._get_test_dir("no_vars_graph")
    filename = os.path.join(test_dir, "ckpt")
    graph_1 = ops_lib.Graph()
    with session.Session(graph=graph_1) as sess:
      constant_op.constant([1, 2, 3], name="x")
      constant_op.constant([1, 2, 3], name="y")
      saver = saver_module.Saver(allow_empty=True)
      saver.save(sess, filename)

    # Create a fresh graph.
    graph_2 = ops_lib.Graph()
    with session.Session(graph=graph_2) as sess:
      # Restore the above checkpoint under scope "subgraph_1".
      new_saver_1 = saver_module.import_meta_graph(
          filename + ".meta", graph=graph_2, import_scope="subgraph_1")
      # There are no variables to restore, so import_meta_graph should not
      # return a Saver.
      self.assertIsNone(new_saver_1)

      # Create a variable in graph_2 under scope "my_scope".
      variables.Variable(array_ops.zeros([10]), name="my_scope/my_var")
      sess.run(variables.global_variables_initializer())
      # Restore the checkpoint into a different scope "subgraph_2".
      new_saver_2 = saver_module.import_meta_graph(
          filename + ".meta", graph=graph_2, import_scope="subgraph_2")
      # Because the variable does not live in scope "subgraph_2",
      # import_meta_graph should not attempt to restore the variable. So,
      # import_meta_graph still won't return a Saver instance.
      self.assertIsNone(new_saver_2)

      # However, if we restore the checkpoint under scope "my_scope",
      # import_meta_graph will detect the variable and return a Saver for
      # restoring it. This should happen even when the variable does not
      # originate from graph_1.
      new_saver_3 = saver_module.import_meta_graph(
          filename + ".meta", graph=graph_2, import_scope="my_scope")
      self.assertIsInstance(new_saver_3, saver_module.Saver)

  def testImportIntoImplicitNamescope(self):
    # Test that we can import a meta graph into an implicit namescope.
    test_dir = self._get_test_dir("import_into_namescope")
    filename = os.path.join(test_dir, "ckpt")
    image = array_ops.placeholder(dtypes.float32, [None, 784], name="image")
    label = array_ops.placeholder(dtypes.float32, [None, 10], name="label")
    with session.Session() as sess:
      weights = variables.Variable(
          random_ops.random_uniform([784, 10]), name="weights")
      bias = variables.Variable(array_ops.zeros([10]), name="bias")
      logit = nn_ops.relu(math_ops.matmul(image, weights) + bias, name="logits")
      nn_ops.softmax(logit, name="prediction")
      cost = nn_ops.softmax_cross_entropy_with_logits(labels=label,
                                                      logits=logit, name="cost")
      adam.AdamOptimizer().minimize(cost, name="optimize")
      saver = saver_module.Saver()
      sess.run(variables.global_variables_initializer())
      saver.save(sess, filename)

    graph = ops_lib.Graph()
    with session.Session(graph=graph) as sess:
      with ops_lib.name_scope("new_model"):
        new_saver = saver_module.import_meta_graph(
            filename + ".meta", graph=graph)

      new_saver.restore(sess, filename)
      sess.run(["new_model/optimize"], {
          "new_model/image:0": np.random.random([1, 784]),
          "new_model/label:0": np.random.randint(
              10, size=[1, 10])
      })

  def testClearDevicesOnImport(self):
    # Test that we import a graph without its devices and run successfully.
    with ops_lib.Graph().as_default():
      with ops_lib.device("/job:ps/replica:0/task:0/device:GPU:0"):
        image = array_ops.placeholder(dtypes.float32, [None, 784], name="image")
        label = array_ops.placeholder(dtypes.float32, [None, 10], name="label")
        weights = variables.Variable(
            random_ops.random_uniform([784, 10]), name="weights")
        bias = variables.Variable(array_ops.zeros([10]), name="bias")
        logit = nn_ops.relu(math_ops.matmul(image, weights) + bias)
        nn_ops.softmax(logit, name="prediction")
        cost = nn_ops.softmax_cross_entropy_with_logits(labels=label,
                                                        logits=logit)
        adam.AdamOptimizer().minimize(cost, name="optimize")
      meta_graph_def = saver_module.export_meta_graph()

    with session.Session(graph=ops_lib.Graph()) as sess:
      saver_module.import_meta_graph(
          meta_graph_def, clear_devices=False, import_scope="new_model")
      # Device refers to GPU, which is not available here.
      with self.assertRaises(errors_impl.InvalidArgumentError):
        sess.run(variables.global_variables_initializer())

    with session.Session(graph=ops_lib.Graph()) as sess:
      saver_module.import_meta_graph(
          meta_graph_def, clear_devices=True, import_scope="new_model")
      sess.run(variables.global_variables_initializer())
      sess.run(["new_model/optimize"], {
          "new_model/image:0": np.random.random([1, 784]),
          "new_model/label:0": np.random.randint(
              10, size=[1, 10])
      })

  def testClearDevicesOnExport(self):
    # Test that we export a graph without its devices and run successfully.
    with ops_lib.Graph().as_default():
      with ops_lib.device("/job:ps/replica:0/task:0/device:GPU:0"):
        image = array_ops.placeholder(dtypes.float32, [None, 784], name="image")
        label = array_ops.placeholder(dtypes.float32, [None, 10], name="label")
        weights = variables.Variable(
            random_ops.random_uniform([784, 10]), name="weights")
        bias = variables.Variable(array_ops.zeros([10]), name="bias")
        logit = nn_ops.relu(math_ops.matmul(image, weights) + bias)
        nn_ops.softmax(logit, name="prediction")
        cost = nn_ops.softmax_cross_entropy_with_logits(labels=label,
                                                        logits=logit)
        adam.AdamOptimizer().minimize(cost, name="optimize")
      meta_graph_def = saver_module.export_meta_graph(clear_devices=True)
      graph_io.write_graph(meta_graph_def, self.get_temp_dir(),
                           "meta_graph.pbtxt")

    with session.Session(graph=ops_lib.Graph()) as sess:
      saver_module.import_meta_graph(meta_graph_def, import_scope="new_model")
      sess.run(variables.global_variables_initializer())
      sess.run(["new_model/optimize"], {
          "new_model/image:0": np.random.random([1, 784]),
          "new_model/label:0": np.random.randint(
              10, size=[1, 10])
      })

  def testPreserveDatasetAndFunctions(self):
    with ops_lib.Graph().as_default() as g:
      dataset = dataset_ops.Dataset.range(10).map(lambda x: x * x)
      iterator = dataset.make_one_shot_iterator()
      next_element = iterator.get_next()
      _ = array_ops.identity(next_element, name="output")

      # Generate three MetaGraphDef protos using different code paths.
      meta_graph_def_simple = saver_module.export_meta_graph()
      meta_graph_def_devices_cleared = saver_module.export_meta_graph(
          clear_devices=True)
      meta_graph_def_from_graph_def = saver_module.export_meta_graph(
          clear_devices=True, graph_def=g.as_graph_def())

    for meta_graph_def in [meta_graph_def_simple,
                           meta_graph_def_devices_cleared,
                           meta_graph_def_from_graph_def]:
      with session.Session(graph=ops_lib.Graph()) as sess:
        saver_module.import_meta_graph(meta_graph_def, import_scope="new_model")
        sess.run(variables.global_variables_initializer())
        for i in range(10):
          self.assertEqual(i * i, sess.run("new_model/output:0"))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run("new_model/output:0")


class CheckpointReaderTest(test.TestCase):

  _WRITE_VERSION = saver_pb2.SaverDef.V1

  def testDebugString(self):
    # Builds a graph.
    v0 = variables.Variable(
        [[1, 2, 3], [4, 5, 6]], dtype=dtypes.float32, name="v0")
    v1 = variables.Variable(
        [[[1], [2]], [[3], [4]], [[5], [6]]], dtype=dtypes.float32, name="v1")
    init_all_op = variables.global_variables_initializer()
    save = saver_module.Saver(
        {
            "v0": v0,
            "v1": v1
        }, write_version=self._WRITE_VERSION)
    save_path = os.path.join(self.get_temp_dir(),
                             "ckpt_for_debug_string" + str(self._WRITE_VERSION))
    with self.test_session() as sess:
      sess.run(init_all_op)
      # Saves a checkpoint.
      save.save(sess, save_path)

      # Creates a reader.
      reader = pywrap_tensorflow.NewCheckpointReader(save_path)
      # Verifies that the tensors exist.
      self.assertTrue(reader.has_tensor("v0"))
      self.assertTrue(reader.has_tensor("v1"))
      debug_string = reader.debug_string()
      # Verifies that debug string contains the right strings.
      self.assertTrue(compat.as_bytes("v0 (DT_FLOAT) [2,3]") in debug_string)
      self.assertTrue(compat.as_bytes("v1 (DT_FLOAT) [3,2,1]") in debug_string)
      # Verifies get_variable_to_shape_map() returns the correct information.
      var_map = reader.get_variable_to_shape_map()
      self.assertEqual([2, 3], var_map["v0"])
      self.assertEqual([3, 2, 1], var_map["v1"])
      # Verifies get_tensor() returns the tensor value.
      v0_tensor = reader.get_tensor("v0")
      v1_tensor = reader.get_tensor("v1")
      self.assertAllEqual(v0.eval(), v0_tensor)
      self.assertAllEqual(v1.eval(), v1_tensor)
      # Verifies get_tensor() fails for non-existent tensors.
      with self.assertRaisesRegexp(errors.NotFoundError,
                                   "v3 not found in checkpoint"):
        reader.get_tensor("v3")

  def testNonexistentPath(self):
    with self.assertRaisesRegexp(errors.NotFoundError,
                                 "Unsuccessful TensorSliceReader"):
      pywrap_tensorflow.NewCheckpointReader("non-existent")


class CheckpointReaderForV2Test(CheckpointReaderTest):
  _WRITE_VERSION = saver_pb2.SaverDef.V2


class WriteGraphTest(test.TestCase):

  def _get_test_dir(self, dirname):
    test_dir = os.path.join(self.get_temp_dir(), dirname)
    gfile.MakeDirs(test_dir)
    return test_dir

  def testWriteGraph(self):
    test_dir = self._get_test_dir("write_graph_dir")
    variables.Variable([[1, 2, 3], [4, 5, 6]], dtype=dtypes.float32, name="v0")
    path = graph_io.write_graph(ops_lib.get_default_graph(),
                                os.path.join(test_dir, "l1"), "graph.pbtxt")
    truth = os.path.join(test_dir, "l1", "graph.pbtxt")
    self.assertEqual(path, truth)
    self.assertTrue(os.path.exists(path))

  def testRecursiveCreate(self):
    test_dir = self._get_test_dir("deep_dir")
    variables.Variable([[1, 2, 3], [4, 5, 6]], dtype=dtypes.float32, name="v0")
    path = graph_io.write_graph(ops_lib.get_default_graph().as_graph_def(),
                                os.path.join(test_dir, "l1", "l2", "l3"),
                                "graph.pbtxt")
    truth = os.path.join(test_dir, "l1", "l2", "l3", "graph.pbtxt")
    self.assertEqual(path, truth)
    self.assertTrue(os.path.exists(path))


class SaverUtilsTest(test.TestCase):

  def setUp(self):
    self._base_dir = os.path.join(self.get_temp_dir(), "saver_utils_test")
    gfile.MakeDirs(self._base_dir)

  def tearDown(self):
    gfile.DeleteRecursively(self._base_dir)

  def testCheckpointExists(self):
    for sharded in (False, True):
      for version in (saver_pb2.SaverDef.V2, saver_pb2.SaverDef.V1):
        with self.test_session(graph=ops_lib.Graph()) as sess:
          unused_v = variables.Variable(1.0, name="v")
          variables.global_variables_initializer().run()
          saver = saver_module.Saver(sharded=sharded, write_version=version)

          path = os.path.join(self._base_dir, "%s-%s" % (sharded, version))
          self.assertFalse(
              saver_module.checkpoint_exists(path))  # Not saved yet.

          ckpt_prefix = saver.save(sess, path)
          self.assertTrue(saver_module.checkpoint_exists(ckpt_prefix))

          ckpt_prefix = saver_module.latest_checkpoint(self._base_dir)
          self.assertTrue(saver_module.checkpoint_exists(ckpt_prefix))

  def testGetCheckpointMtimes(self):
    prefixes = []
    for version in (saver_pb2.SaverDef.V2, saver_pb2.SaverDef.V1):
      with self.test_session(graph=ops_lib.Graph()) as sess:
        unused_v = variables.Variable(1.0, name="v")
        variables.global_variables_initializer().run()
        saver = saver_module.Saver(write_version=version)
        prefixes.append(
            saver.save(sess, os.path.join(self._base_dir, str(version))))

    mtimes = saver_module.get_checkpoint_mtimes(prefixes)
    self.assertEqual(2, len(mtimes))
    self.assertTrue(mtimes[1] >= mtimes[0])


class ScopedGraphTest(test.TestCase):

  def _get_test_dir(self, dirname):
    test_dir = os.path.join(self.get_temp_dir(), dirname)
    gfile.MakeDirs(test_dir)
    return test_dir

  def _testScopedSave(self, test_dir, exported_filename, ckpt_filename):
    graph = ops_lib.Graph()
    with graph.as_default():
      # Creates an inference graph.
      # Hidden 1
      images = constant_op.constant(
          1.2, dtypes.float32, shape=[100, 28], name="images")
      with ops_lib.name_scope("hidden1"):
        weights1 = variables.Variable(
            random_ops.truncated_normal(
                [28, 128], stddev=1.0 / math.sqrt(float(28))),
            name="weights")
        # The use of control_flow_ops.cond here is purely for adding test
        # coverage the save and restore of control flow context (which doesn't
        # make any sense here from a machine learning perspective).  The typical
        # biases is a simple Variable without the conditions.
        biases1 = variables.Variable(
            control_flow_ops.cond(
                math_ops.less(random.random(), 0.5),
                lambda: array_ops.ones([128]), lambda: array_ops.zeros([128])),
            name="biases")
        hidden1 = nn_ops.relu(math_ops.matmul(images, weights1) + biases1)

      # Hidden 2
      with ops_lib.name_scope("hidden2"):
        weights2 = variables.Variable(
            random_ops.truncated_normal(
                [128, 32], stddev=1.0 / math.sqrt(float(128))),
            name="weights")

        # The use of control_flow_ops.while_loop here is purely for adding test
        # coverage the save and restore of control flow context (which doesn't
        # make any sense here from a machine learning perspective).  The typical
        # biases is a simple Variable without the conditions.
        def loop_cond(it, _):
          return it < 2

        def loop_body(it, biases2):
          biases2 += constant_op.constant(0.1, shape=[32])
          return it + 1, biases2

        _, biases2 = control_flow_ops.while_loop(loop_cond, loop_body, [
            constant_op.constant(0), variables.Variable(array_ops.zeros([32]))
        ])
        hidden2 = nn_ops.relu(math_ops.matmul(hidden1, weights2) + biases2)
      # Linear
      with ops_lib.name_scope("softmax_linear"):
        weights3 = variables.Variable(
            random_ops.truncated_normal(
                [32, 10], stddev=1.0 / math.sqrt(float(32))),
            name="weights")
        biases3 = variables.Variable(array_ops.zeros([10]), name="biases")
        logits = math_ops.matmul(hidden2, weights3) + biases3
        ops_lib.add_to_collection("logits", logits)

        # Adds user_defined proto in three formats: string, bytes and Any.
        # Any proto should just pass through.
        queue_runner = queue_runner_pb2.QueueRunnerDef(queue_name="test_queue")
        ops_lib.add_to_collection("user_defined_string_collection",
                                  str(queue_runner))
        ops_lib.add_to_collection("user_defined_bytes_collection",
                                  queue_runner.SerializeToString())
        any_buf = Any()
        any_buf.Pack(queue_runner)
        ops_lib.add_to_collection("user_defined_any_collection", any_buf)

      _, var_list = meta_graph.export_scoped_meta_graph(
          filename=os.path.join(test_dir, exported_filename),
          graph=ops_lib.get_default_graph(),
          export_scope="hidden1")
      self.assertEqual(["biases:0", "weights:0"], sorted(var_list.keys()))

    with self.test_session(graph=graph) as sess:
      sess.run(variables.global_variables_initializer())
      saver = saver_module.Saver(var_list=var_list, max_to_keep=1)
      saver.save(sess, os.path.join(test_dir, ckpt_filename), write_state=False)

  def _testScopedRestore(self, test_dir, exported_filename,
                         new_exported_filename, ckpt_filename):
    graph = ops_lib.Graph()
    # Create all the missing inputs.
    with graph.as_default():
      new_image = constant_op.constant(
          1.2, dtypes.float32, shape=[100, 28], name="images")
    var_list = meta_graph.import_scoped_meta_graph(
        os.path.join(test_dir, exported_filename),
        graph=graph,
        input_map={"$unbound_inputs_images": new_image},
        import_scope="new_hidden1")
    self.assertEqual(["biases:0", "weights:0"], sorted(var_list.keys()))
    hidden1 = graph.as_graph_element("new_hidden1/Relu:0")
    weights1 = graph.as_graph_element("new_hidden1/weights:0")
    biases1 = graph.as_graph_element("new_hidden1/biases:0")

    with graph.as_default():
      # Hidden 2
      with ops_lib.name_scope("hidden2"):
        weights = variables.Variable(
            random_ops.truncated_normal(
                [128, 32], stddev=1.0 / math.sqrt(float(128))),
            name="weights")

        # The use of control_flow_ops.while_loop here is purely for adding test
        # coverage the save and restore of control flow context (which doesn't
        # make any sense here from a machine learning perspective).  The typical
        # biases is a simple Variable without the conditions.
        def loop_cond(it, _):
          return it < 2

        def loop_body(it, biases):
          biases += constant_op.constant(0.1, shape=[32])
          return it + 1, biases

        _, biases = control_flow_ops.while_loop(loop_cond, loop_body, [
            constant_op.constant(0), variables.Variable(array_ops.zeros([32]))
        ])
        hidden2 = nn_ops.relu(math_ops.matmul(hidden1, weights) + biases)
      # Linear
      with ops_lib.name_scope("softmax_linear"):
        weights = variables.Variable(
            random_ops.truncated_normal(
                [32, 10], stddev=1.0 / math.sqrt(float(32))),
            name="weights")
        biases = variables.Variable(array_ops.zeros([10]), name="biases")
        logits = math_ops.matmul(hidden2, weights) + biases
        ops_lib.add_to_collection("logits", logits)

      # The rest of the variables.
      rest_variables = list(
          set(variables.global_variables()) - set(var_list.keys()))
      init_rest_op = variables.variables_initializer(rest_variables)

    with self.test_session(graph=graph) as sess:
      saver = saver_module.Saver(var_list=var_list, max_to_keep=1)
      saver.restore(sess, os.path.join(test_dir, ckpt_filename))
      # Verify that we have restored weights1 and biases1.
      sess.run([weights1, biases1])
      # Initialize the rest of the variables and run logits.
      sess.run(init_rest_op)
      sess.run(logits)

  # Verifies that we can save the subgraph under "hidden1" and restore it
  # into "new_hidden1" in the new graph.
  def testScopedSaveAndRestore(self):
    test_dir = self._get_test_dir("scoped_export_import")
    ckpt_filename = "ckpt"
    self._testScopedSave(test_dir, "exported_hidden1.pbtxt", ckpt_filename)
    self._testScopedRestore(test_dir, "exported_hidden1.pbtxt",
                            "exported_new_hidden1.pbtxt", ckpt_filename)

  # Verifies that we can copy the subgraph under "hidden1" and copy it
  # to different name scope in the same graph or different graph.
  def testCopyScopedGraph(self):
    test_dir = self._get_test_dir("scoped_copy")
    saver0_ckpt = os.path.join(test_dir, "saver0.ckpt")
    graph1 = ops_lib.Graph()
    with graph1.as_default():
      with ops_lib.name_scope("hidden1"):
        images = constant_op.constant(
            1.0, dtypes.float32, shape=[3, 2], name="images")
        weights1 = variables.Variable(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name="weights")
        biases1 = variables.Variable([0.1] * 3, name="biases")
        nn_ops.relu(math_ops.matmul(images, weights1) + biases1, name="relu")

    # Run the graph and save scoped checkpoint.
    with self.test_session(graph=graph1) as sess:
      sess.run(variables.global_variables_initializer())
      _, var_list_1 = meta_graph.export_scoped_meta_graph(
          export_scope="hidden1")
      saver = saver_module.Saver(var_list=var_list_1, max_to_keep=1)
      saver.save(sess, saver0_ckpt, write_state=False)

    expected = np.reshape([[5.0999999, 7.0999999, 9.10000038] * 3], (3, 3))

    # Verifies copy to the same graph with the same name fails.
    with graph1.as_default():
      with self.assertRaisesWithPredicateMatch(
          ValueError, lambda e: "need to be different" in str(e)):
        meta_graph.copy_scoped_meta_graph(
            from_scope="hidden1", to_scope="hidden1")

    # Verifies copy to the same graph.
    with graph1.as_default():
      var_list_2 = meta_graph.copy_scoped_meta_graph(
          from_scope="hidden1", to_scope="hidden2")

    with self.test_session(graph=graph1) as sess:
      saver1 = saver_module.Saver(var_list=var_list_1, max_to_keep=1)
      saver1.restore(sess, saver0_ckpt)
      saver2 = saver_module.Saver(var_list=var_list_2, max_to_keep=1)
      saver2.restore(sess, saver0_ckpt)
      self.assertAllClose(expected, sess.run("hidden1/relu:0"))
      self.assertAllClose(expected, sess.run("hidden2/relu:0"))

    # Verifies copy to differen graph.
    graph2 = ops_lib.Graph()
    new_var_list_1 = meta_graph.copy_scoped_meta_graph(
        from_scope="hidden1",
        to_scope="new_hidden1",
        from_graph=graph1,
        to_graph=graph2)

    with self.test_session(graph=graph2) as sess:
      saver3 = saver_module.Saver(var_list=new_var_list_1, max_to_keep=1)
      saver3.restore(sess, saver0_ckpt)
      self.assertAllClose(expected, sess.run("new_hidden1/relu:0"))

  def testExportGraphDefWithScope(self):
    test_dir = self._get_test_dir("export_graph_def")
    saver0_ckpt = os.path.join(test_dir, "saver0.ckpt")
    graph1 = ops_lib.Graph()
    with graph1.as_default():
      with ops_lib.name_scope("hidden1"):
        images = constant_op.constant(
            1.0, dtypes.float32, shape=[3, 2], name="images")
        weights1 = variables.Variable(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name="weights")
        biases1 = variables.Variable([0.1] * 3, name="biases")
        nn_ops.relu(math_ops.matmul(images, weights1) + biases1, name="relu")

    # Run the graph and save scoped checkpoint.
    with self.test_session(graph=graph1) as sess:
      sess.run(variables.global_variables_initializer())
      _, var_list_1 = meta_graph.export_scoped_meta_graph(
          graph_def=graph1.as_graph_def(), export_scope="hidden1")
      saver = saver_module.Saver(var_list=var_list_1, max_to_keep=1)
      saver.save(sess, saver0_ckpt, write_state=False)

    expected = np.reshape([[5.0999999, 7.0999999, 9.10000038] * 3], (3, 3))

    # Verifies that we can run successfully after restoring.
    graph2 = ops_lib.Graph()
    new_var_list_1 = meta_graph.copy_scoped_meta_graph(
        from_scope="hidden1",
        to_scope="new_hidden1",
        from_graph=graph1,
        to_graph=graph2)

    with self.test_session(graph=graph2) as sess:
      saver3 = saver_module.Saver(var_list=new_var_list_1, max_to_keep=1)
      saver3.restore(sess, saver0_ckpt)
      self.assertAllClose(expected, sess.run("new_hidden1/relu:0"))

  def testSerializeSaverWithScope(self):
    test_dir = self._get_test_dir("export_graph_def")
    saver1_ckpt = os.path.join(test_dir, "saver1.ckpt")
    saver2_ckpt = os.path.join(test_dir, "saver2.ckpt")
    graph = ops_lib.Graph()
    with graph.as_default():
      with ops_lib.name_scope("hidden1"):
        variable1 = variables.Variable([1.0], name="variable1")
        saver1 = saver_module.Saver(var_list=[variable1])
        graph.add_to_collection(ops_lib.GraphKeys.SAVERS, saver1)

      with ops_lib.name_scope("hidden2"):
        variable2 = variables.Variable([2.0], name="variable2")
      saver2 = saver_module.Saver(var_list=[variable2], name="hidden2/")
      graph.add_to_collection(ops_lib.GraphKeys.SAVERS, saver2)

    with self.test_session(graph=graph) as sess:
      variables.global_variables_initializer().run()
      saver1.save(sess, saver1_ckpt, write_state=False)
      saver2.save(sess, saver2_ckpt, write_state=False)

    graph1 = ops_lib.Graph()
    var_dict1 = meta_graph.copy_scoped_meta_graph(
        from_scope="hidden1",
        to_scope="new_hidden1",
        from_graph=graph,
        to_graph=graph1)
    self.assertEqual(1, len(var_dict1))

    saver_list1 = graph1.get_collection(ops_lib.GraphKeys.SAVERS)
    self.assertEqual(1, len(saver_list1))

    with self.test_session(graph=graph1) as sess:
      saver_list1[0].restore(sess, saver1_ckpt)
      self.assertEqual(1.0, var_dict1["variable1:0"].eval())

    graph2 = ops_lib.Graph()
    var_dict2 = meta_graph.copy_scoped_meta_graph(
        from_scope="hidden2",
        to_scope="new_hidden2",
        from_graph=graph,
        to_graph=graph2)
    self.assertEqual(1, len(var_dict2))

    saver_list2 = graph2.get_collection(ops_lib.GraphKeys.SAVERS)
    self.assertEqual(1, len(saver_list2))

    with self.test_session(graph=graph2) as sess:
      saver_list2[0].restore(sess, saver2_ckpt)
      self.assertEqual(2.0, var_dict2["variable2:0"].eval())


class _OwnsAVariableSimple(checkpointable.CheckpointableBase):
  """A Checkpointable object which can be saved using a tf.train.Saver."""

  def __init__(self):
    self.non_dep_variable = variable_scope.get_variable(
        name="non_dep_variable", initializer=6., use_resource=True)

  def _gather_saveables_for_checkpoint(self):
    return {checkpointable.VARIABLE_VALUE_KEY: self.non_dep_variable}

  # The Saver sorts by name before parsing, so we need a name property.
  @property
  def name(self):
    return self.non_dep_variable.name


class _MirroringSaveable(
    saver_module.BaseSaverBuilder.ResourceVariableSaveable):

  def __init__(self, primary_variable, mirrored_variable, name):
    self._primary_variable = primary_variable
    self._mirrored_variable = mirrored_variable
    super(_MirroringSaveable, self).__init__(
        self._primary_variable, "", name)

  def restore(self, restored_tensors, restored_shapes):
    """Restore the same value into both variables."""
    tensor, = restored_tensors
    return control_flow_ops.group(
        self._primary_variable.assign(tensor),
        self._mirrored_variable.assign(tensor))


class _OwnsMirroredVariables(checkpointable.CheckpointableBase):
  """A Checkpointable object which returns a more complex SaveableObject."""

  def __init__(self):
    self.non_dep_variable = variable_scope.get_variable(
        name="non_dep_variable", initializer=6., use_resource=True)
    self.mirrored = variable_scope.get_variable(
        name="mirrored", initializer=15., use_resource=True)

  def _gather_saveables_for_checkpoint(self):
    def _saveable_factory(name=self.non_dep_variable.name):
      return _MirroringSaveable(
          primary_variable=self.non_dep_variable,
          mirrored_variable=self.mirrored,
          name=name)
    return {checkpointable.VARIABLE_VALUE_KEY: _saveable_factory}

  # The Saver sorts by name before parsing, so we need a name property.
  @property
  def name(self):
    return self.non_dep_variable.name


class NonLayerCheckpointable(checkpointable.Checkpointable):

  def __init__(self):
    super(NonLayerCheckpointable, self).__init__()
    self.a_variable = checkpointable_utils.add_variable(
        self, name="a_variable", shape=[])


class MyModel(training.Model):
  """A concrete Model for testing."""

  def __init__(self):
    super(MyModel, self).__init__()
    self._named_dense = core.Dense(1, use_bias=True)
    self._second = core.Dense(1, use_bias=False)
    # We can still track Checkpointables which aren't Layers.
    self._non_layer = NonLayerCheckpointable()

  def call(self, values):
    ret = self._second(self._named_dense(values))
    return ret


class CheckpointableCompatibilityTests(test.TestCase):

  # TODO(allenl): Track down python3 reference cycles in these tests.
  @test_util.run_in_graph_and_eager_modes()
  def testNotSaveableButIsCheckpointable(self):
    v = _OwnsAVariableSimple()
    saver = saver_module.Saver(var_list=[v])
    test_dir = self.get_temp_dir()
    prefix = os.path.join(test_dir, "ckpt")
    self.evaluate(v.non_dep_variable.assign(42.))
    with self.test_session() as sess:
      save_path = saver.save(sess, prefix)
      self.evaluate(v.non_dep_variable.assign(43.))
      saver.restore(sess, save_path)
      self.assertEqual(42., self.evaluate(v.non_dep_variable))

  @test_util.run_in_graph_and_eager_modes()
  def testMoreComplexSaveableReturned(self):
    v = _OwnsMirroredVariables()
    saver = saver_module.Saver(var_list=[v])
    test_dir = self.get_temp_dir()
    prefix = os.path.join(test_dir, "ckpt")
    self.evaluate(v.non_dep_variable.assign(42.))
    with self.test_session() as sess:
      save_path = saver.save(sess, prefix)
      self.evaluate(v.non_dep_variable.assign(43.))
      self.evaluate(v.mirrored.assign(44.))
      saver.restore(sess, save_path)
      self.assertEqual(42., self.evaluate(v.non_dep_variable))
      self.assertEqual(42., self.evaluate(v.mirrored))

  def testSingleTensorEvaluation(self):

    class _CountingSaveable(saver_module.BaseSaverBuilder.SaveableObject):

      def __init__(self, name):
        self.eval_count = 0
        def _tensor():
          self.eval_count += 1
          return constant_op.constant([1.])
        dummy_op = constant_op.constant([2.])
        super(_CountingSaveable, self).__init__(
            dummy_op,
            [saver_module.BaseSaverBuilder.SaveSpec(
                _tensor, "", name, dtype=dummy_op.dtype)],
            name)

      def restore(self, restored_tensors, restored_shapes):
        """Restore the same value into both variables."""
        pass

    with context.eager_mode():
      v = _CountingSaveable("foo")
      saver = saver_module.Saver(var_list=[v])
      test_dir = self.get_temp_dir()
      prefix = os.path.join(test_dir, "ckpt")
      with self.test_session() as sess:
        save_path = saver.save(sess, prefix)
        self.assertEqual(1, v.eval_count)
        saver.restore(sess, save_path)
        self.assertEqual(1, v.eval_count)

  def _initialized_model(self):
    input_value = constant_op.constant([[3.]])
    model = MyModel()
    optimizer = adam.AdamOptimizer(0.001)
    optimizer_step = training_util.get_or_create_global_step()
    root_checkpointable = checkpointable_utils.Checkpoint(
        optimizer=optimizer, model=model, optimizer_step=optimizer_step)
    train_op = optimizer.minimize(
        functools.partial(model, input_value),
        global_step=optimizer_step)
    self.evaluate(checkpointable_utils.gather_initializers(
        root_checkpointable))
    self.evaluate(train_op)
    # A regular variable, a slot variable, and a non-slot Optimizer variable
    # with known values to check when loading.
    self.evaluate(model._named_dense.bias.assign([1.]))
    self.evaluate(optimizer.get_slot(
        var=model._named_dense.bias, name="m").assign([2.]))
    beta1_power, _ = optimizer._get_beta_accumulators()
    self.evaluate(beta1_power.assign(3.))
    return root_checkpointable

  def _set_sentinels(self, root_checkpointable):
    self.evaluate(root_checkpointable.model._named_dense.bias.assign([101.]))
    self.evaluate(
        root_checkpointable.optimizer.get_slot(
            var=root_checkpointable.model._named_dense.bias, name="m")
        .assign([102.]))
    beta1_power, _ = root_checkpointable.optimizer._get_beta_accumulators()
    self.evaluate(beta1_power.assign(103.))

  def _check_sentinels(self, root_checkpointable):
    self.assertAllEqual(
        [1.], self.evaluate(root_checkpointable.model._named_dense.bias))
    self.assertAllEqual([2.], self.evaluate(
        root_checkpointable.optimizer.get_slot(
            var=root_checkpointable.model._named_dense.bias, name="m")))
    beta1_power, _ = root_checkpointable.optimizer._get_beta_accumulators()
    self.assertAllEqual(3., self.evaluate(beta1_power))

  def testVariableNotFoundErrorRaised(self):
    # Restore does some tricky exception handling to figure out if it should
    # load an object-based checkpoint. Tests that the exception handling isn't
    # too broad.
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    a = resource_variable_ops.ResourceVariable(1., name="a")
    b = resource_variable_ops.ResourceVariable(1., name="b")
    a_saver = saver_module.Saver([a])
    b_saver = saver_module.Saver([b])
    with self.test_session() as sess:
      sess.run(a.initializer)
      save_path = a_saver.save(sess=sess, save_path=checkpoint_prefix)
      with self.assertRaisesRegexp(
          errors.NotFoundError, "Key b not found in checkpoint"):
        b_saver.restore(sess=sess, save_path=save_path)

  def testCheckpointNotFoundErrorRaised(self):
    # Restore does some tricky exception handling to figure out if it should
    # load an object-based checkpoint. Tests that the exception handling isn't
    # too broad.
    a = resource_variable_ops.ResourceVariable(1., name="a")
    saver = saver_module.Saver([a])
    with self.test_session() as sess:
      with self.assertRaisesRegexp(
          errors.NotFoundError,
          "Failed to find any matching files for path_which_does_not_exist"):
        saver.restore(sess=sess, save_path="path_which_does_not_exist")
      try:
        saver.restore(sess=sess, save_path="path_which_does_not_exist")
      except errors.NotFoundError:
        # Make sure we don't have a confusing "During handling of the above
        # exception" block in Python 3.
        # pylint: disable=no-value-for-parameter
        exception_string = "\n".join(
            traceback.format_exception(*sys.exc_info()))
        # pylint: enable=no-value-for-parameter
        self.assertNotIn("NewCheckpointReader", exception_string)

  def testLoadFromObjectBasedGraph(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    save_graph = ops_lib.Graph()
    with save_graph.as_default(), self.test_session(graph=save_graph) as sess:
      root = self._initialized_model()
      object_saver = checkpointable_utils.CheckpointableSaver(root)
      save_path = object_saver.save(file_prefix=checkpoint_prefix)

      # An incompatible object-based checkpoint to check error messages
      var = resource_variable_ops.ResourceVariable(1., name="a")
      self.evaluate(var.initializer)
      second_saver = checkpointable_utils.CheckpointableSaver(var)
      second_path = second_saver.save(file_prefix=os.path.join(
          checkpoint_directory, "second"))

    restore_graph = ops_lib.Graph()
    with restore_graph.as_default(), self.test_session(
        graph=restore_graph) as sess:
      root = self._initialized_model()
      self._set_sentinels(root)
      saver = saver_module.Saver()
      saver.restore(sess=sess, save_path=save_path)
      self._check_sentinels(root)
      before_second_restore_ops = restore_graph.get_operations()
      # Test that multiple restores do not pollute the graph
      saver.restore(sess=sess, save_path=save_path)
      self.assertEqual(before_second_restore_ops,
                       restore_graph.get_operations())
      with self.assertRaisesRegexp(errors.NotFoundError,
                                   "could not find a_variable"):
        saver.restore(sess=sess, save_path=second_path)

  def testLoadFromObjectBasedEager(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    save_graph = ops_lib.Graph()
    with save_graph.as_default(), self.test_session(graph=save_graph):
      root = self._initialized_model()
      object_saver = checkpointable_utils.CheckpointableSaver(root)
      save_path = object_saver.save(file_prefix=checkpoint_prefix)

    with context.eager_mode():
      root = self._initialized_model()
      self._set_sentinels(root)
      saver = saver_module.Saver(
          root.model.variables + root.optimizer.variables())
      saver.restore(sess=None, save_path=save_path)
      self._check_sentinels(root)


if __name__ == "__main__":
  test.main()
