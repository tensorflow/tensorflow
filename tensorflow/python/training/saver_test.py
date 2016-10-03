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

import math
import os.path
import time
import contextlib
import random
import shutil
import tempfile

import tensorflow as tf
import numpy as np
import six

from google.protobuf.any_pb2 import Any

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import queue_runner_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import gfile
from tensorflow.python.training import saver as saver_module
from tensorflow.python.util import compat


# pylint: disable=invalid-name
def _TestDir(test_name):
  test_dir = os.path.join(tf.test.get_temp_dir(), test_name)
  if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
  gfile.MakeDirs(test_dir)
  return test_dir
# pylint: enable=invalid-name


class CheckpointedOp(object):
  """Op with a custom checkpointing implementation.

  Defined as part of the test because the MutableHashTable Python code is
  currently in contrib.
  """

  def __init__(self, name):
    self._table_ref = gen_data_flow_ops._mutable_hash_table(
        key_dtype=tf.string, value_dtype=tf.float32, name=name)
    self._name = name
    self._saveable = CheckpointedOp.CustomSaveable(self, name)
    tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, self._saveable)

  @property
  def name(self):
    return self._name

  @property
  def saveable(self):
    return self._saveable

  def insert(self, keys, values):
    return gen_data_flow_ops._lookup_table_insert(self._table_ref, keys, values)

  def keys(self):
    return self._export()[0]

  def values(self):
    return self._export()[1]

  def _export(self):
    return gen_data_flow_ops._lookup_table_export(self._table_ref, tf.string,
                                                  tf.float32)

  class CustomSaveable(saver_module.BaseSaverBuilder.SaveableObject):

    def __init__(self, table, name):
      tensors = table._export()
      specs = [
          saver_module.BaseSaverBuilder.SaveSpec(tensors[0], "",
                                                 name + "-keys"),
          saver_module.BaseSaverBuilder.SaveSpec(tensors[1], "",
                                                 name + "-values")
      ]
      super(CheckpointedOp.CustomSaveable, self).__init__(table, specs, name)

    def restore(self, restore_tensors, shapes):
      return gen_data_flow_ops._lookup_table_import(self.op._table_ref,
                                                    restore_tensors[0],
                                                    restore_tensors[1])


class SaverTest(tf.test.TestCase):

  def testBasics(self):
    save_path = os.path.join(self.get_temp_dir(), "basics")

    # Build a graph with 2 parameter nodes, and Save and
    # Restore nodes for them.
    v0 = tf.Variable(10.0, name="v0")
    v1 = tf.Variable(20.0, name="v1")
    v2 = CheckpointedOp(name="v2")
    v2_init = v2.insert("k1", 30.0)
    save = tf.train.Saver(
        {"v0": v0,
         "v1": v1,
         "v2": v2.saveable}, restore_sequentially=True)
    init_all_op = [tf.initialize_all_variables(), v2_init]

    with self.test_session() as sess:
      # Initialize all variables
      sess.run(init_all_op)

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())
      self.assertEqual(b"k1", v2.keys().eval())
      self.assertEqual(30.0, v2.values().eval())

      # Save the initialized values in the file at "save_path"
      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

    # Start a second session.  In that session the parameter nodes
    # have not been initialized either.
    with self.test_session() as sess:
      v0 = tf.Variable(-1.0, name="v0")
      v1 = tf.Variable(-1.0, name="v1")
      v2 = CheckpointedOp(name="v2")
      save = tf.train.Saver({"v0": v0, "v1": v1, "v2": v2.saveable})

      with self.assertRaisesWithPredicateMatch(
          tf.OpError, lambda e: "uninitialized value v0" in e.message):
        sess.run(v0)
      with self.assertRaisesWithPredicateMatch(
          tf.OpError, lambda e: "uninitialized value v1" in e.message):
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
    with self.test_session() as sess:
      v0_2 = tf.Variable(1000.0, name="v0")
      v1_2 = tf.Variable(2000.0, name="v1")
      v2_2 = CheckpointedOp(name="v2")
      save2 = tf.train.Saver({"v0": v0_2, "v1": v1_2, "v2": v2_2.saveable})
      v2_2.insert("k1000", 3000.0).run()
      tf.initialize_all_variables().run()

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

  def testInvalidPath(self):
    v0 = tf.Variable(0, name="v0")
    with self.test_session() as sess:
      save = tf.train.Saver({"v0": v0})
      with self.assertRaisesRegexp(ValueError,
                                   "^Restore called with invalid save path.*"):
        save.restore(sess, "invalid path")

  def testInt64(self):
    save_path = os.path.join(self.get_temp_dir(), "int64")

    with self.test_session() as sess:
      # Build a graph with 1 node, and save and restore for them.
      v = tf.Variable(np.int64(15), name="v")
      save = tf.train.Saver({"v": v}, restore_sequentially=True)
      tf.initialize_all_variables().run()

      # Save the initialized values in the file at "save_path"
      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

      with self.test_session() as sess:
        v = tf.Variable(np.int64(-1), name="v")
        save = tf.train.Saver({"v": v})

      with self.assertRaisesWithPredicateMatch(
          tf.OpError, lambda e: "uninitialized value v" in e.message):
        sess.run(v)

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(np.int64(15), v.eval())

  def testSomeErrors(self):
    with tf.Graph().as_default():
      v0 = tf.Variable([10.0], name="v0")
      v1 = tf.Variable([20.0], name="v1")
      v2 = tf.Variable([20.0], name="v2")
      v2._set_save_slice_info(tf.Variable.SaveSliceInfo("v1", [1], [0], [1]))

      # By default the name used for "v2" will be "v1" and raise an error.
      with self.assertRaisesRegexp(ValueError, "same name: v1"):
        tf.train.Saver([v0, v1, v2])

      # The names are different and will work.
      tf.train.Saver({"vee1": v1, "other": [v2]})

  def testSameName(self):
    with tf.Graph().as_default():
      v0 = tf.Variable([10.0], name="v0")
      v2 = CheckpointedOp(name="v2")

      # Saving one variable under two names raises an error.
      with self.assertRaisesRegexp(
          ValueError, "The same saveable will be restored with two names: v0"):
        tf.train.Saver({"v0": v0, "v0too": v0})

      # Ditto for custom saveables.
      with self.assertRaisesRegexp(
          ValueError, "The same saveable will be restored with two names: v2"):
        tf.train.Saver({"v2": v2.saveable, "v2too": v2.saveable})

      # Verify non-duplicate names work.
      tf.train.Saver({"v0": v0, "v2": v2.saveable})

  def testBasicsWithListOfVariables(self):
    save_path = os.path.join(self.get_temp_dir(), "basics_with_list")

    with self.test_session(graph=tf.Graph()) as sess:
      # Build a graph with 2 parameter nodes, and Save and
      # Restore nodes for them.
      v0 = tf.Variable(10.0, name="v0")
      v1 = tf.Variable(20.0, name="v1")
      v2 = CheckpointedOp(name="v2")
      v2_init = v2.insert("k1", 30.0)
      save = tf.train.Saver([v0, v1, v2.saveable])
      tf.initialize_all_variables().run()
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
    with self.test_session(graph=tf.Graph()) as sess:
      v0 = tf.Variable(-1.0, name="v0")
      v1 = tf.Variable(-1.0, name="v1")
      v2 = CheckpointedOp(name="v2")
      save = tf.train.Saver([v0, v1, v2.saveable])

      with self.assertRaisesWithPredicateMatch(
          tf.OpError, lambda e: "uninitialized value v0" in e.message):
        sess.run(v0)
      with self.assertRaisesWithPredicateMatch(
          tf.OpError, lambda e: "uninitialized value v1" in e.message):
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
    with self.test_session(graph=tf.Graph()) as sess:
      v0_2 = tf.Variable(1000.0, name="v0")
      v1_2 = tf.Variable(2000.0, name="v1")
      v2_2 = CheckpointedOp(name="v2")
      save2 = tf.train.Saver([v0_2, v1_2, v2_2.saveable])
      v2_2.insert("k1000", 3000.0).run()
      tf.initialize_all_variables().run()

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
    with self.test_session() as sess:
      var = tf.Variable(var_value, name=var_name)
      save = tf.train.Saver({var_name: var})
      var.initializer.run()
      val = save.save(sess, save_path)
      self.assertEqual(save_path, val)
    with self.test_session() as sess:
      var = tf.Variable(other_value, name=var_name)
      save = tf.train.Saver({var_name: var})
      save.restore(sess, save_path)
      self.assertAllClose(var_value, var.eval())

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
      _ = tf.constant(1)
      save = tf.train.Saver(allow_empty=True)
      val = save.save(sess, save_path)
      self.assertIsNone(val)
    with self.test_session() as sess:
      save = tf.train.Saver(allow_empty=True)
      save.restore(sess, save_path)

  def testGPU(self):
    if not tf.test.is_gpu_available():
      return
    save_path = os.path.join(self.get_temp_dir(), "gpu")
    with tf.Session("", graph=tf.Graph()) as sess:
      with sess.graph.device("/gpu:0"):
        v0_1 = tf.Variable(123.45)
      save = tf.train.Saver({"v0": v0_1})
      tf.initialize_all_variables().run()
      save.save(sess, save_path)

    with tf.Session("", graph=tf.Graph()) as sess:
      with sess.graph.device("/gpu:0"):
        v0_2 = tf.Variable(543.21)
      save = tf.train.Saver({"v0": v0_2})
      tf.initialize_all_variables().run()
      self.assertAllClose(543.21, v0_2.eval())
      save.restore(sess, save_path)
      self.assertAllClose(123.45, v0_2.eval())

  def testVariables(self):
    save_path = os.path.join(self.get_temp_dir(), "variables")
    with tf.Session("", graph=tf.Graph()) as sess:
      one = tf.Variable(1.0)
      twos = tf.Variable([2.0, 2.0, 2.0])
      v2 = CheckpointedOp(name="v2")
      init = tf.initialize_all_variables()
      save = tf.train.Saver()
      init.run()
      v2.insert("k1", 3.0).run()
      save.save(sess, save_path)

    with tf.Session("", graph=tf.Graph()) as sess:
      one = tf.Variable(0.0)
      twos = tf.Variable([0.0, 0.0, 0.0])
      v2 = CheckpointedOp(name="v2")
      # Saver with no arg, defaults to 'all variables'.
      save = tf.train.Saver()
      save.restore(sess, save_path)
      self.assertAllClose(1.0, one.eval())
      self.assertAllClose([2.0, 2.0, 2.0], twos.eval())
      self.assertEqual(b"k1", v2.keys().eval())
      self.assertEqual(3.0, v2.values().eval())

  def testVarListShouldBeEmptyInDeferredBuild(self):
    with tf.Graph().as_default():
      v = tf.Variable(1.0)
      with self.assertRaisesRegexp(ValueError, "defer_build"):
        tf.train.Saver([v], defer_build=True)

  def testBuildShouldBeCalledBeforeSaveInCaseOfDeferBuild(self):
    save_path = os.path.join(self.get_temp_dir(), "error_deferred_build")
    with tf.Graph().as_default(), tf.Session() as sess:
      tf.Variable(1.0)
      saver = tf.train.Saver(defer_build=True)
      with self.assertRaisesRegexp(RuntimeError, "build"):
        saver.save(sess, save_path)

  def testDeferredBuild(self):
    save_path = os.path.join(self.get_temp_dir(), "deferred_build")
    with tf.Session("", graph=tf.Graph()) as sess:
      one = tf.Variable(1.0)
      save = tf.train.Saver(defer_build=True)
      # if build is not deferred, saver cannot save the `twos`.
      twos = tf.Variable([2.0, 2.0, 2.0])
      init = tf.initialize_all_variables()
      save.build()
      init.run()
      save.save(sess, save_path)

    with tf.Session("", graph=tf.Graph()) as sess:
      one = tf.Variable(0.0)
      twos = tf.Variable([0.0, 0.0, 0.0])
      # Saver with no arg, defaults to 'all variables'.
      save = tf.train.Saver()
      save.restore(sess, save_path)
      self.assertAllClose(1.0, one.eval())
      self.assertAllClose([2.0, 2.0, 2.0], twos.eval())

  def testReshape(self):
    save_path = os.path.join(self.get_temp_dir(), "variables")
    with tf.Session("", graph=tf.Graph()) as sess:
      var = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      init = tf.initialize_all_variables()
      save = tf.train.Saver()
      init.run()
      save.save(sess, save_path)

    # Error when restoring with default reshape=False
    with tf.Session("", graph=tf.Graph()) as sess:
      var = tf.Variable([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
      save = tf.train.Saver()
      with self.assertRaisesRegexp(
          tf.errors.InvalidArgumentError,
          "Assign requires shapes of both tensors to match."):
        save.restore(sess, save_path)

    # Restored to new shape with reshape=True
    with tf.Session("", graph=tf.Graph()) as sess:
      var = tf.Variable([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
      save = tf.train.Saver(reshape=True)
      save.restore(sess, save_path)
      self.assertAllClose([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], var.eval())

  def testSaveWithGlobalStep(self):
    save_path = os.path.join(self.get_temp_dir(), "ckpt_with_global_step")
    global_step_int = 5
    # Save and reload one Variable named "var0".
    self._SaveAndLoad("var0", 0.0, 1.0, save_path)
    for use_tensor in [True, False]:
      with self.test_session() as sess:
        var = tf.Variable(1.0, name="var0")
        save = tf.train.Saver({var.op.name: var})
        var.initializer.run()
        if use_tensor:
          global_step = tf.constant(global_step_int)
          val = save.save(sess, save_path, global_step=global_step)
        else:
          val = save.save(sess, save_path, global_step=global_step_int)
        expected_save_path = "%s-%d" % (save_path, global_step_int)
        self.assertEqual(expected_save_path, val)

  def testSaveToNonexistingPath(self):

    save_path = os.path.join(self.get_temp_dir(), "nonexisting_dir/path")

    # Build a graph with 2 parameter nodes, and Save and
    # Restore nodes for them.
    v0 = tf.Variable(10.0, name="v0")
    v1 = tf.Variable(20.0, name="v1")
    save = tf.train.Saver({"v0": v0, "v1": v1}, restore_sequentially=True)
    init_all_op = tf.initialize_all_variables()

    with self.test_session() as sess:
      # Initialize all variables
      sess.run(init_all_op)

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      # Assert saving fails when parent dir of save path doesn't exist
      with self.assertRaisesWithPredicateMatch(
        ValueError, lambda e:  "Parent directory of {} doesn't exist, can't save.".format(save_path) in str(e)):
        save.save(sess, save_path)


class SaveRestoreShardedTest(tf.test.TestCase):

  def testBasics(self):
    save_path = os.path.join(self.get_temp_dir(), "sharded")

    # Build a graph with 2 parameter nodes on different devices.
    with tf.Session(
        target="",
        config=tf.ConfigProto(device_count={"CPU": 2})) as sess:
      with sess.graph.device("/cpu:0"):
        v0 = tf.Variable(10, name="v0")
        t0 = CheckpointedOp(name="t0")
      with sess.graph.device("/cpu:1"):
        v1 = tf.Variable(20, name="v1")
        t1 = CheckpointedOp(name="t1")
      save = tf.train.Saver(
          {"v0": v0,
           "v1": v1,
           "t0": t0.saveable,
           "t1": t1.saveable},
          sharded=True)
      tf.initialize_all_variables().run()
      t0.insert("k1", 30.0).run()
      t1.insert("k2", 40.0).run()
      val = save.save(sess, save_path)
      self.assertEqual(save_path + "-?????-of-00002", val)
      meta_graph_filename = save._MetaGraphFilename(val)
      self.assertEqual(save_path + ".meta", meta_graph_filename)

    # Restore different ops from shard 0 of the saved files.
    with tf.Session(
        target="",
        config=tf.ConfigProto(device_count={"CPU": 2})) as sess:
      with sess.graph.device("/cpu:0"):
        v0 = tf.Variable(111, name="v0")
        t0 = CheckpointedOp(name="t0")
      save = tf.train.Saver({"v0": v0, "t0": t0.saveable}, sharded=True)
      tf.initialize_all_variables().run()
      t0.insert("k11", 33.0).run()
      self.assertEqual(111, v0.eval())
      self.assertEqual(b"k11", t0.keys().eval())
      self.assertEqual(33.0, t0.values().eval())
      save.restore(sess, save_path + "-00000-of-00002")
      self.assertEqual(10, v0.eval())
      self.assertEqual(b"k1", t0.keys().eval())
      self.assertEqual(30.0, t0.values().eval())

    # Restore different ops from shard 1 of the saved files.
    with tf.Session(
        target="",
        config=tf.ConfigProto(device_count={"CPU": 2})) as sess:
      with sess.graph.device("/cpu:0"):
        v1 = tf.Variable(222)
        t1 = CheckpointedOp(name="t1")
      save = tf.train.Saver({"v1": v1, "t1": t1.saveable}, sharded=True)
      tf.initialize_all_variables().run()
      t1.insert("k22", 44.0).run()
      self.assertEqual(222, v1.eval())
      self.assertEqual(b"k22", t1.keys().eval())
      self.assertEqual(44.0, t1.values().eval())
      save.restore(sess, save_path + "-00001-of-00002")
      self.assertEqual(20, v1.eval())
      self.assertEqual(b"k2", t1.keys().eval())
      self.assertEqual(40.0, t1.values().eval())

    # Now try a restore with the sharded filename.
    with tf.Session(
        target="",
        config=tf.ConfigProto(device_count={"CPU": 2})) as sess:
      with sess.graph.device("/cpu:0"):
        v0 = tf.Variable(111, name="v0")
        t0 = CheckpointedOp(name="t0")
      with sess.graph.device("/cpu:1"):
        v1 = tf.Variable(222, name="v1")
        t1 = CheckpointedOp(name="t1")
      save = tf.train.Saver(
          {"v0": v0,
           "v1": v1,
           "t0": t0.saveable,
           "t1": t1.saveable},
          sharded=True)
      tf.initialize_all_variables().run()
      t0.insert("k11", 33.0).run()
      t1.insert("k22", 44.0).run()
      self.assertEqual(111, v0.eval())
      self.assertEqual(222, v1.eval())
      self.assertEqual(b"k11", t0.keys().eval())
      self.assertEqual(33.0, t0.values().eval())
      self.assertEqual(b"k22", t1.keys().eval())
      self.assertEqual(44.0, t1.values().eval())
      save_path = os.path.join(self.get_temp_dir(), "sharded")
      save.restore(sess, save_path + "-?????-of-?????")
      self.assertEqual(10, v0.eval())
      self.assertEqual(20, v1.eval())
      self.assertEqual(b"k1", t0.keys().eval())
      self.assertEqual(30.0, t0.values().eval())
      self.assertEqual(b"k2", t1.keys().eval())
      self.assertEqual(40.0, t1.values().eval())

    self.assertEqual(
        tf.train.latest_checkpoint(self.get_temp_dir()),
        os.path.join(self.get_temp_dir(), "sharded-?????-of-00002"))

  def testSaverDef(self):
    with self.test_session():
      v0 = tf.Variable(123, name="v0")
      save = tf.train.Saver({"v0": v0}, sharded=True)
      sd = save.as_saver_def()
      self.assertTrue(sd.sharded)

  def testPartitionedVariables(self):
    var_full_shape = [10, 3]
    # Allows save/restore mechanism to work w/ different slicings.
    var_name = "my_var"
    saved_path = os.path.join(_TestDir("partitioned_variables"), "ckpt")

    def _save(slices):
      with self.test_session(graph=tf.Graph()) as sess:
        # Calls .eval() to return the ndarray that makes up the full variable.
        rnd = tf.random_uniform(var_full_shape).eval()

        if slices:
          vs = tf.create_partitioned_variables(var_full_shape,
                                               slices,
                                               rnd,
                                               name=var_name)
        else:
          vs = [tf.Variable(rnd, name=var_name)]

        tf.initialize_all_variables().run()
        saver = tf.train.Saver(vs)
        actual_path = saver.save(sess, saved_path)
        self.assertEqual(saved_path, actual_path)

        return rnd

    def _restore(slices):
      with self.test_session(graph=tf.Graph()) as sess:
        if slices:
          new_vs = tf.create_partitioned_variables(
              var_full_shape,
              slices,
              tf.zeros(var_full_shape),  # != original contents.
              name=var_name)
        else:
          new_vs = [tf.Variable(
              tf.zeros(shape=var_full_shape),  # != original contents.
              name=var_name)]

        tf.initialize_all_variables().run()
        saver = tf.train.Saver(new_vs)
        saver.restore(sess, saved_path)

        if slices and slices[0] != 1:
          return tf.concat(0, new_vs).eval()
        elif slices and slices[1] != 1:
          return tf.concat(1, new_vs).eval()
        else:  # Non-sliced.
          return new_vs[0].eval()

    # Saves 10 horizontal parts of a partitioned variable.
    # Restores into a full variable, non-sliced.
    saved_full = _save(slices=[10, 1])
    restored_full = _restore(slices=None)
    self.assertAllEqual(saved_full, restored_full)

    # Restores into a different number/orientation of slices.
    restored_full = _restore(slices=[2, 1])  # 2 horizon parts.
    self.assertAllEqual(saved_full, restored_full)
    restored_full = _restore(slices=[1, 3])  # 3 vertical parts.
    self.assertAllEqual(saved_full, restored_full)

    # Now, saves a full variable and restores in slices.
    saved_full = _save(slices=None)
    restored_full = _restore(slices=[1, 3])
    self.assertAllEqual(saved_full, restored_full)


class MaxToKeepTest(tf.test.TestCase):

  def testNonSharded(self):
    save_dir = _TestDir("max_to_keep_non_sharded")

    with self.test_session() as sess:
      v = tf.Variable(10.0, name="v")
      save = tf.train.Saver({"v": v}, max_to_keep=2)
      tf.initialize_all_variables().run()
      self.assertEqual([], save.last_checkpoints)

      s1 = save.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([s1], save.last_checkpoints)
      self.assertTrue(gfile.Exists(s1))

      s2 = save.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([s1, s2], save.last_checkpoints)
      self.assertTrue(gfile.Exists(s1))
      self.assertTrue(gfile.Exists(s2))

      s3 = save.save(sess, os.path.join(save_dir, "s3"))
      self.assertEqual([s2, s3], save.last_checkpoints)
      self.assertFalse(gfile.Exists(s1))
      self.assertTrue(gfile.Exists(s2))
      self.assertTrue(gfile.Exists(s3))

      # Create a second helper, identical to the first.
      save2 = tf.train.Saver(saver_def=save.as_saver_def())
      save2.set_last_checkpoints(save.last_checkpoints)

      # Create a third helper, with the same configuration but no knowledge of
      # previous checkpoints.
      save3 = tf.train.Saver(saver_def=save.as_saver_def())

      # Exercise the first helper.

      # Adding s2 again (old s2 is removed first, then new s2 appended)
      s2 = save.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([s3, s2], save.last_checkpoints)
      self.assertFalse(gfile.Exists(s1))
      self.assertFalse(gfile.Exists(save._MetaGraphFilename(s1)))
      self.assertTrue(gfile.Exists(s3))
      self.assertTrue(gfile.Exists(save._MetaGraphFilename(s3)))
      self.assertTrue(gfile.Exists(s2))
      self.assertTrue(gfile.Exists(save._MetaGraphFilename(s2)))

      # Adding s1 (s3 should now be deleted as oldest in list)
      s1 = save.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([s2, s1], save.last_checkpoints)
      self.assertFalse(gfile.Exists(s3))
      self.assertFalse(gfile.Exists(save._MetaGraphFilename(s3)))
      self.assertTrue(gfile.Exists(s2))
      self.assertTrue(gfile.Exists(save._MetaGraphFilename(s2)))
      self.assertTrue(gfile.Exists(s1))
      self.assertTrue(gfile.Exists(save._MetaGraphFilename(s1)))

      # Exercise the second helper.

      # Adding s2 again (old s2 is removed first, then new s2 appended)
      s2 = save2.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([s3, s2], save2.last_checkpoints)
      # Created by the first helper.
      self.assertTrue(gfile.Exists(s1))
      self.assertTrue(gfile.Exists(save._MetaGraphFilename(s1)))
      # Deleted by the first helper.
      self.assertFalse(gfile.Exists(s3))
      self.assertFalse(gfile.Exists(save._MetaGraphFilename(s3)))
      self.assertTrue(gfile.Exists(s2))
      self.assertTrue(gfile.Exists(save._MetaGraphFilename(s2)))

      # Adding s1 (s3 should now be deleted as oldest in list)
      s1 = save2.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([s2, s1], save2.last_checkpoints)
      self.assertFalse(gfile.Exists(s3))
      self.assertFalse(gfile.Exists(save._MetaGraphFilename(s3)))
      self.assertTrue(gfile.Exists(s2))
      self.assertTrue(gfile.Exists(save._MetaGraphFilename(s2)))
      self.assertTrue(gfile.Exists(s1))
      self.assertTrue(gfile.Exists(save._MetaGraphFilename(s1)))

      # Exercise the third helper.

      # Adding s2 again (but helper is unaware of previous s2)
      s2 = save3.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([s2], save3.last_checkpoints)
      # Created by the first helper.
      self.assertTrue(gfile.Exists(s1))
      self.assertTrue(gfile.Exists(save._MetaGraphFilename(s1)))
      # Deleted by the first helper.
      self.assertFalse(gfile.Exists(s3))
      self.assertFalse(gfile.Exists(save._MetaGraphFilename(s3)))
      self.assertTrue(gfile.Exists(s2))
      self.assertTrue(gfile.Exists(save._MetaGraphFilename(s2)))

      # Adding s1 (s3 should not be deleted because helper is unaware of it)
      s1 = save3.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([s2, s1], save3.last_checkpoints)
      self.assertFalse(gfile.Exists(s3))
      self.assertFalse(gfile.Exists(save._MetaGraphFilename(s3)))
      self.assertTrue(gfile.Exists(s2))
      self.assertTrue(gfile.Exists(save._MetaGraphFilename(s2)))
      self.assertTrue(gfile.Exists(s1))
      self.assertTrue(gfile.Exists(save._MetaGraphFilename(s1)))

  def testSharded(self):
    save_dir = _TestDir("max_to_keep_sharded")

    with tf.Session(
        target="",
        config=tf.ConfigProto(device_count={"CPU": 2})) as sess:
      with sess.graph.device("/cpu:0"):
        v0 = tf.Variable(111, name="v0")
      with sess.graph.device("/cpu:1"):
        v1 = tf.Variable(222, name="v1")
      save = tf.train.Saver({"v0": v0, "v1": v1}, sharded=True, max_to_keep=2)
      tf.initialize_all_variables().run()
      self.assertEqual([], save.last_checkpoints)

      s1 = save.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([s1], save.last_checkpoints)
      self.assertEqual(2, len(gfile.Glob(s1)))
      self.assertTrue(gfile.Exists(save._MetaGraphFilename(s1)))

      s2 = save.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([s1, s2], save.last_checkpoints)
      self.assertEqual(2, len(gfile.Glob(s1)))
      self.assertTrue(gfile.Exists(save._MetaGraphFilename(s1)))
      self.assertEqual(2, len(gfile.Glob(s2)))
      self.assertTrue(gfile.Exists(save._MetaGraphFilename(s2)))

      s3 = save.save(sess, os.path.join(save_dir, "s3"))
      self.assertEqual([s2, s3], save.last_checkpoints)
      self.assertEqual(0, len(gfile.Glob(s1)))
      self.assertFalse(gfile.Exists(save._MetaGraphFilename(s1)))
      self.assertEqual(2, len(gfile.Glob(s2)))
      self.assertTrue(gfile.Exists(save._MetaGraphFilename(s2)))
      self.assertEqual(2, len(gfile.Glob(s3)))
      self.assertTrue(gfile.Exists(save._MetaGraphFilename(s3)))

  def testNoMaxToKeep(self):
    save_dir = _TestDir("no_max_to_keep")
    save_dir2 = _TestDir("max_to_keep_0")

    with self.test_session() as sess:
      v = tf.Variable(10.0, name="v")
      tf.initialize_all_variables().run()

      # Test max_to_keep being None.
      save = tf.train.Saver({"v": v}, max_to_keep=None)
      self.assertEqual([], save.last_checkpoints)
      s1 = save.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([], save.last_checkpoints)
      self.assertTrue(gfile.Exists(s1))
      s2 = save.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([], save.last_checkpoints)
      self.assertTrue(gfile.Exists(s2))

      # Test max_to_keep being 0.
      save2 = tf.train.Saver({"v": v}, max_to_keep=0)
      self.assertEqual([], save2.last_checkpoints)
      s1 = save2.save(sess, os.path.join(save_dir2, "s1"))
      self.assertEqual([], save2.last_checkpoints)
      self.assertTrue(gfile.Exists(s1))
      s2 = save2.save(sess, os.path.join(save_dir2, "s2"))
      self.assertEqual([], save2.last_checkpoints)
      self.assertTrue(gfile.Exists(s2))

  def testNoMetaGraph(self):
    save_dir = _TestDir("no_meta_graph")

    with self.test_session() as sess:
      v = tf.Variable(10.0, name="v")
      save = tf.train.Saver({"v": v})
      tf.initialize_all_variables().run()

      s1 = save.save(sess, os.path.join(save_dir, "s1"),
                     write_meta_graph=False)
      self.assertTrue(gfile.Exists(s1))
      self.assertFalse(gfile.Exists(save._MetaGraphFilename(s1)))


class KeepCheckpointEveryNHoursTest(tf.test.TestCase):

  def testNonSharded(self):
    save_dir = _TestDir("keep_checkpoint_every_n_hours")

    with self.test_session() as sess:
      v = tf.Variable([10.0], name="v")
      # Run the initializer NOW to avoid the 0.5s overhead of the first Run()
      # call, which throws the test timing off in fastbuild mode.
      tf.initialize_all_variables().run()
      # Create a saver that will keep the last 2 checkpoints plus one every 0.7
      # seconds.
      start_time = time.time()
      save = tf.train.Saver({"v": v}, max_to_keep=2,
                            keep_checkpoint_every_n_hours=0.7 / 3600)
      self.assertEqual([], save.last_checkpoints)

      # Wait till 0.7 second have elapsed so s1 will be old enough to keep.
      time.sleep((time.time() + 0.7) - start_time)
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
      self.assertTrue(gfile.Exists(s1))
      self.assertFalse(gfile.Exists(s2))
      self.assertTrue(gfile.Exists(s3))
      self.assertTrue(gfile.Exists(s4))


class SaveRestoreWithVariableNameMap(tf.test.TestCase):

  def testNonReshape(self):
    save_path = os.path.join(self.get_temp_dir(), "basics")

    with self.test_session() as sess:
      # Build a graph with 2 parameter nodes, and Save and
      # Restore nodes for them.
      v0 = tf.Variable(10.0, name="v0")
      v1 = tf.Variable(20.0, name="v1")
      save = tf.train.Saver({"save_prefix/v0": v0, "save_prefix/v1": v1})
      tf.initialize_all_variables().run()

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      # Save the initialized values in the file at "save_path"
      # Use a variable name map to set the saved tensor names
      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

      # Verify that the original names are not in the Saved file
      save = tf.train.Saver({"v0": v0, "v1": v1})
      with self.assertRaisesOpError("not found in checkpoint"):
        save.restore(sess, save_path)

    # Verify that the mapped names are present in the Saved file and can be
    # Restored using remapped names.
    with self.test_session() as sess:
      v0 = tf.Variable(-1.0, name="v0")
      v1 = tf.Variable(-1.0, name="v1")

      with self.assertRaisesOpError("uninitialized value v0"):
        sess.run(v0)
      with self.assertRaisesOpError("uninitialized value v1"):
        sess.run(v1)

      save = tf.train.Saver({"save_prefix/v0": v0, "save_prefix/v1": v1})
      save.restore(sess, save_path)

      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

    # Add a prefix to the node names in the current graph and Restore using
    # remapped names.
    with self.test_session() as sess:
      v0 = tf.Variable(-1.0, name="restore_prefix/v0")
      v1 = tf.Variable(-1.0, name="restore_prefix/v1")

      with self.assertRaisesOpError("uninitialized value restore_prefix/v0"):
        sess.run(v0)
      with self.assertRaisesOpError("uninitialized value restore_prefix/v1"):
        sess.run(v1)

      # Restore the saved values in the parameter nodes.
      save = tf.train.Saver({"save_prefix/v0": v0, "save_prefix/v1": v1})
      save.restore(sess, save_path)

      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())


class LatestCheckpointWithRelativePaths(tf.test.TestCase):

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
          unused_a = tf.Variable(0.0)  # So that Saver saves something.
          tf.initialize_all_variables().run()

          # Should fail.
          saver = tf.train.Saver(sharded=False)
          with self.assertRaisesRegexp(ValueError, "collides with"):
            saver.save(sess, filepath)

          # Succeeds: the file will be named "checkpoint-<step>".
          saver.save(sess, filepath, global_step=1)
          self.assertIsNotNone(tf.train.latest_checkpoint(traindir))

          # Succeeds: the file will be named "checkpoint-<i>-of-<n>".
          saver = tf.train.Saver(sharded=True)
          saver.save(sess, filepath)
          self.assertIsNotNone(tf.train.latest_checkpoint(traindir))

          # Succeeds: the file will be named "checkpoint-<step>-<i>-of-<n>".
          saver = tf.train.Saver(sharded=True)
          saver.save(sess, filepath, global_step=1)
          self.assertIsNotNone(tf.train.latest_checkpoint(traindir))

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
          v0 = tf.Variable(0.0)
          inc = v0.assign_add(1.0)

          save = tf.train.Saver({"v0": v0})

          # Record a short training history.
          tf.initialize_all_variables().run()
          save.save(sess, filepath, global_step=0)
          inc.eval()
          save.save(sess, filepath, global_step=1)
          inc.eval()
          save.save(sess, filepath, global_step=2)

        with self.test_session() as sess:
          # Build a new graph with different initialization.
          v0 = tf.Variable(-1.0)

          # Create a new saver.
          save = tf.train.Saver({"v0": v0})
          tf.initialize_all_variables().run()

          # Get the most recent checkpoint name from the training history file.
          name = tf.train.latest_checkpoint(traindir)
          self.assertIsNotNone(name)

          # Restore "v0" from that checkpoint.
          save.restore(sess, name)
          self.assertEqual(v0.eval(), 2.0)


class CheckpointStateTest(tf.test.TestCase):

  def testAbsPath(self):
    save_dir = _TestDir("abs_paths")
    abs_path = os.path.join(save_dir, "model-0")
    ckpt = tf.train.generate_checkpoint_state_proto(save_dir, abs_path)
    self.assertEqual(ckpt.model_checkpoint_path, abs_path)
    self.assertTrue(os.path.isabs(ckpt.model_checkpoint_path))
    self.assertEqual(len(ckpt.all_model_checkpoint_paths), 1)
    self.assertEqual(ckpt.all_model_checkpoint_paths[-1], abs_path)

  def testRelPath(self):
    train_dir = "train"
    model = os.path.join(train_dir, "model-0")
    # model_checkpoint_path should have no "train" directory part.
    new_rel_path = "model-0"
    ckpt = tf.train.generate_checkpoint_state_proto(train_dir, model)
    self.assertEqual(ckpt.model_checkpoint_path, new_rel_path)
    self.assertEqual(len(ckpt.all_model_checkpoint_paths), 1)
    self.assertEqual(ckpt.all_model_checkpoint_paths[-1], new_rel_path)

  def testAllModelCheckpointPaths(self):
    save_dir = _TestDir("all_models_test")
    abs_path = os.path.join(save_dir, "model-0")
    for paths in [None, [], ["model-2"]]:
      ckpt = tf.train.generate_checkpoint_state_proto(
          save_dir,
          abs_path,
          all_model_checkpoint_paths=paths)
      self.assertEqual(ckpt.model_checkpoint_path, abs_path)
      self.assertTrue(os.path.isabs(ckpt.model_checkpoint_path))
      self.assertEqual(
          len(ckpt.all_model_checkpoint_paths), len(paths) if paths else 1)
      self.assertEqual(ckpt.all_model_checkpoint_paths[-1], abs_path)

  def testUpdateCheckpointState(self):
    save_dir = _TestDir("update_checkpoint_state")
    os.chdir(save_dir)
    # Make a temporary train directory.
    train_dir = "train"
    os.mkdir(train_dir)
    abs_path = os.path.join(save_dir, "model-0")
    rel_path = "train/model-2"
    tf.train.update_checkpoint_state(
        train_dir,
        rel_path,
        all_model_checkpoint_paths=[abs_path, rel_path])
    ckpt = tf.train.get_checkpoint_state(train_dir)
    self.assertEqual(ckpt.model_checkpoint_path, rel_path)
    self.assertEqual(len(ckpt.all_model_checkpoint_paths), 2)
    self.assertEqual(ckpt.all_model_checkpoint_paths[-1], rel_path)
    self.assertEqual(ckpt.all_model_checkpoint_paths[0], abs_path)

  def testCheckPointStateFailsWhenIncomplete(self):
    save_dir = _TestDir("checkpoint_state_fails_when_incomplete")
    os.chdir(save_dir)
    ckpt_path = os.path.join(save_dir, "checkpoint")
    ckpt_file = open(ckpt_path, "w")
    ckpt_file.write("")
    ckpt_file.close()
    with self.assertRaises(ValueError):
      tf.train.get_checkpoint_state(save_dir)

  def testCheckPointCompletesRelativePaths(self):
    save_dir = _TestDir("checkpoint_completes_relative_paths")
    os.chdir(save_dir)
    ckpt_path = os.path.join(save_dir, "checkpoint")
    ckpt_file = open(ckpt_path, "w")
    ckpt_file.write("""
        model_checkpoint_path: "./model.ckpt-687529"
        all_model_checkpoint_paths: "./model.ckpt-687500"
        all_model_checkpoint_paths: "./model.ckpt-687529"
        """)
    ckpt_file.close()
    ckpt = tf.train.get_checkpoint_state(save_dir)
    self.assertEqual(ckpt.model_checkpoint_path,
                     os.path.join(save_dir, "./model.ckpt-687529"))
    self.assertEqual(ckpt.all_model_checkpoint_paths[0],
                     os.path.join(save_dir, "./model.ckpt-687500"))
    self.assertEqual(ckpt.all_model_checkpoint_paths[1],
                     os.path.join(save_dir, "./model.ckpt-687529"))

class MetaGraphTest(tf.test.TestCase):

  def testNoVariables(self):
    test_dir = _TestDir("no_variables")
    filename = os.path.join(test_dir, "metafile")

    input_feed_value = -10  # Arbitrary input value for feed_dict.

    orig_graph = tf.Graph()
    with self.test_session(graph=orig_graph) as sess:
      # Create a minimal graph with zero variables.
      input_tensor = tf.placeholder(tf.float32, shape=[], name="input")
      offset = tf.constant(42, dtype=tf.float32, name="offset")
      output_tensor = tf.add(input_tensor, offset, name="add_offset")

      # Add input and output tensors to graph collections.
      tf.add_to_collection("input_tensor", input_tensor)
      tf.add_to_collection("output_tensor", output_tensor)

      output_value = sess.run(output_tensor, {input_tensor: input_feed_value})
      self.assertEqual(output_value, 32)

      # Generates MetaGraphDef.
      #
      # Note that this is calling the saver *module-level* export_meta_graph and
      # not the Saver.export_meta_graph instance-level method.
      meta_graph_def = saver_module.export_meta_graph(
          filename=filename,
          graph_def=tf.get_default_graph().as_graph_def(add_shapes=True),
          collection_list=["input_tensor", "output_tensor"],
          saver_def=None,
      )

    # Create a clean graph and import the MetaGraphDef nodes.
    new_graph = tf.Graph()
    with self.test_session(graph=new_graph) as sess:
      # Import the previously export meta graph.
      saver_instance = saver_module.import_meta_graph(filename)
      # The saver instance should be None since there are no graph variables
      # to be restored in this case.
      self.assertIsNone(saver_instance)

      # Re-exports the current graph state for comparison to the original.
      new_meta_graph_def = saver_module.export_meta_graph(filename + "_new")
      self.assertProtoEquals(meta_graph_def, new_meta_graph_def)

      # Ensures that we can still get a reference to our graph collections.
      new_input_tensor = tf.get_collection("input_tensor")[0]
      new_output_tensor = tf.get_collection("output_tensor")[0]
      # Verifies that the new graph computes the same result as the original.
      new_output_value = sess.run(
          new_output_tensor, {new_input_tensor: input_feed_value})
      self.assertEqual(new_output_value, output_value)

  def testAddCollectionDef(self):
    test_dir = _TestDir("good_collection")
    filename = os.path.join(test_dir, "metafile")
    with self.test_session():
      # Creates a graph.
      v0 = tf.Variable(1.0, name="v0")
      control_flow_ops.cond(tf.less(v0, 10),
                            lambda: tf.add(v0, 1),
                            lambda: tf.sub(v0, 1))
      control_flow_ops.while_loop(lambda i: tf.less(i, 10),
                                  lambda i: tf.add(i, 1),
                                  [v0])
      var = tf.Variable(tf.constant(0, dtype=tf.int64))
      count_up_to = var.count_up_to(3)
      input_queue = tf.FIFOQueue(30, tf.float32, shared_name="collection_queue")
      qr = tf.train.QueueRunner(input_queue, [count_up_to])
      tf.initialize_all_variables()
      # Creates a saver.
      save = tf.train.Saver({"v0": v0})
      # Adds a set of collections.
      tf.add_to_collection("int_collection", 3)
      tf.add_to_collection("float_collection", 3.5)
      tf.add_to_collection("string_collection", "hello")
      tf.add_to_collection("variable_collection", v0)
      # Add QueueRunners.
      tf.train.add_queue_runner(qr)
      # Adds user_defined proto in three formats: string, bytes and Any.
      queue_runner = queue_runner_pb2.QueueRunnerDef(queue_name="test_queue")
      tf.add_to_collection("user_defined_string_collection", str(queue_runner))
      tf.add_to_collection("user_defined_bytes_collection",
                           queue_runner.SerializeToString())
      any_buf = Any()
      any_buf.Pack(queue_runner)
      tf.add_to_collection("user_defined_any_collection", any_buf)

      # Generates MetaGraphDef.
      meta_graph_def = save.export_meta_graph(filename)
      self.assertTrue(meta_graph_def.HasField("saver_def"))
      self.assertTrue(meta_graph_def.HasField("graph_def"))
      collection_def = meta_graph_def.collection_def
      self.assertEqual(len(collection_def), 12)

    with tf.Graph().as_default():
      # Restores from MetaGraphDef.
      new_saver = tf.train.import_meta_graph(filename)
      # Generates a new MetaGraphDef.
      new_meta_graph_def = new_saver.export_meta_graph()
      # It should be the same as the original.
      self.assertProtoEquals(meta_graph_def, new_meta_graph_def)

  def testAddCollectionDefFails(self):
    with self.test_session():
      # Creates a graph.
      v0 = tf.Variable(10.0, name="v0")
      # Creates a saver.
      save = tf.train.Saver({"v0": v0})
      # Generates MetaGraphDef.
      meta_graph_def = meta_graph_pb2.MetaGraphDef()

      # Verifies that collection with unsupported key will not be added.
      tf.add_to_collection(save, 3)
      save._add_collection_def(meta_graph_def, save)
      self.assertEqual(len(meta_graph_def.collection_def), 0)

      # Verifies that collection where item type does not match expected
      # type will not be added.
      tf.add_to_collection("int_collection", 3)
      tf.add_to_collection("int_collection", 3.5)
      save._add_collection_def(meta_graph_def, "int_collection")
      self.assertEqual(len(meta_graph_def.collection_def), 0)

  def _testMultiSaverCollectionSave(self):
    test_dir = _TestDir("saver_collection")
    filename = os.path.join(test_dir, "metafile")
    saver0_ckpt = os.path.join(test_dir, "saver0.ckpt")
    saver1_ckpt = os.path.join(test_dir, "saver1.ckpt")
    with self.test_session(graph=tf.Graph()) as sess:
      # Creates a graph.
      v0 = tf.Variable([[1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0]], name="v0")
      v1 = tf.Variable(11.0, name="v1")
      # Creates 2 savers.
      saver0 = tf.train.Saver({"v0": v0}, name="saver0")
      saver1 = tf.train.Saver({"v1": v1}, name="saver1")
      tf.add_to_collection("savers", saver0)
      tf.add_to_collection("savers", saver1)
      tf.initialize_all_variables().run()
      # Saves to different checkpoints.
      saver0.save(sess, saver0_ckpt)
      saver1.save(sess, saver1_ckpt)
      # Generates MetaGraphDef.
      meta_graph_def = tf.train.export_meta_graph(filename)
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
      # Verifies that there are 3 entries in SAVERS collection.
      savers = getattr(collection_def, kind)
      self.assertEqual(2, len(savers.value))

  def _testMultiSaverCollectionRestore(self):
    test_dir = os.path.join(self.get_temp_dir(), "saver_collection")
    filename = os.path.join(test_dir, "metafile")
    saver0_ckpt = os.path.join(test_dir, "saver0.ckpt")
    saver1_ckpt = os.path.join(test_dir, "saver1.ckpt")
    with self.test_session(graph=tf.Graph()) as sess:
      # Imports from meta_graph.
      tf.train.import_meta_graph(filename)
      # Retrieves SAVERS collection. Verifies there are 2 entries.
      savers = tf.get_collection("savers")
      self.assertEqual(2, len(savers))
      # Retrieves saver0. Verifies that new_saver0 can restore v0, but not v1.
      new_saver0 = savers[0]
      new_saver0.restore(sess, saver0_ckpt)
      v0 = sess.graph.get_tensor_by_name("v0:0")
      v1 = sess.graph.get_tensor_by_name("v1:0")
      self.assertAllEqual([[1.0, 2.0],
                           [3.0, 4.0],
                           [5.0, 6.0]], v0.eval())
      self.assertEqual([3, 2], v0.get_shape())
      self.assertEqual([], v1.get_shape())
      with self.assertRaisesWithPredicateMatch(
          tf.OpError, lambda e: "uninitialized value v1" in e.message):
        sess.run(v1)
      # Retrieves saver1. Verifies that new_saver1 can restore v1.
      new_saver1 = savers[1]
      new_saver1.restore(sess, saver1_ckpt)
      v1 = sess.graph.get_tensor_by_name("v1:0")
      self.assertEqual(11.0, v1.eval())

  def testMultiSaverCollection(self):
    self._testMultiSaverCollectionSave()
    self._testMultiSaverCollectionRestore()

  def testBinaryAndTextFormat(self):
    test_dir = _TestDir("binary_and_text")
    filename = os.path.join(test_dir, "metafile")
    with self.test_session(graph=tf.Graph()):
      # Creates a graph.
      tf.Variable(10.0, name="v0")
      # Exports the graph as binary format.
      tf.train.export_meta_graph(filename, as_text=False)
    with self.test_session(graph=tf.Graph()):
      # Imports the binary format graph.
      saver = tf.train.import_meta_graph(filename)
      self.assertIsNotNone(saver)
      # Exports the graph as text format.
      saver.export_meta_graph(filename, as_text=True)
    with self.test_session(graph=tf.Graph()):
      # Imports the text format graph.
      tf.train.import_meta_graph(filename)
      # Writes wrong contents to the file.
      tf.train.write_graph(saver.as_saver_def(), os.path.dirname(filename),
                           os.path.basename(filename))
    with self.test_session(graph=tf.Graph()):
      # Import should fail.
      with self.assertRaisesWithPredicateMatch(
          IOError, lambda e: "Cannot parse file"):
        tf.train.import_meta_graph(filename)
      # Deletes the file
      gfile.Remove(filename)
      with self.assertRaisesWithPredicateMatch(
          IOError, lambda e: "does not exist"):
        tf.train.import_meta_graph(filename)

  def testSliceVariable(self):
    test_dir = _TestDir("slice_saver")
    filename = os.path.join(test_dir, "metafile")
    with self.test_session():
      v1 = tf.Variable([20.0], name="v1")
      v2 = tf.Variable([20.0], name="v2")
      v2._set_save_slice_info(tf.Variable.SaveSliceInfo("v1", [1], [0], [1]))

      # The names are different and will work.
      slice_saver = tf.train.Saver({"first": v1, "second": v2})
      tf.initialize_all_variables().run()
      # Exports to meta_graph
      meta_graph_def = slice_saver.export_meta_graph(filename)

    with tf.Graph().as_default():
      # Restores from MetaGraphDef.
      new_saver = tf.train.import_meta_graph(filename)
      self.assertIsNotNone(new_saver)
      # Generates a new MetaGraphDef.
      new_meta_graph_def = new_saver.export_meta_graph()
      # It should be the same as the original.
      self.assertProtoEquals(meta_graph_def, new_meta_graph_def)

  def _testGraphExtensionSave(self):
    test_dir = _TestDir("graph_extension")
    filename = os.path.join(test_dir, "metafile")
    saver0_ckpt = os.path.join(test_dir, "saver0.ckpt")
    # Creates an inference graph.
    # Hidden 1
    images = tf.constant(1.2, tf.float32, shape=[100, 28])
    with tf.name_scope("hidden1"):
      weights = tf.Variable(
          tf.truncated_normal([28, 128],
                              stddev=1.0 / math.sqrt(float(28))),
          name="weights")
      # The use of control_flow_ops.cond here is purely for adding test coverage
      # the save and restore of control flow context (which doesn't make any
      # sense here from a machine learning perspective).  The typical biases is
      # a simple Variable without the conditions.
      biases = tf.Variable(control_flow_ops.cond(tf.less(random.random(), 0.5),
                                                 lambda: tf.ones([128]),
                                                 lambda: tf.zeros([128])),
                           name="biases")
      hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Hidden 2
    with tf.name_scope("hidden2"):
      weights = tf.Variable(
          tf.truncated_normal([128, 32],
                              stddev=1.0 / math.sqrt(float(128))),
          name="weights")

      # The use of control_flow_ops.while_loop here is purely for adding test
      # coverage the save and restore of control flow context (which doesn't
      # make any sense here from a machine learning perspective).  The typical
      # biases is a simple Variable without the conditions.
      def loop_cond(it, _):
        return it < 2
      def loop_body(it, biases):
        biases += tf.constant(0.1, shape=[32])
        return it + 1, biases
      _, biases = control_flow_ops.while_loop(
          loop_cond, loop_body,
          [tf.constant(0), tf.Variable(tf.zeros([32]))])
      hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope("softmax_linear"):
      weights = tf.Variable(
          tf.truncated_normal([32, 10],
                              stddev=1.0 / math.sqrt(float(32))),
          name="weights")
      biases = tf.Variable(tf.zeros([10]),
                           name="biases")
      logits = tf.matmul(hidden2, weights) + biases
      tf.add_to_collection("logits", logits)
    init_all_op = tf.initialize_all_variables()

    with self.test_session() as sess:
      # Initializes all the variables.
      sess.run(init_all_op)
      # Runs to logit.
      sess.run(logits)
      # Creates a saver.
      saver0 = tf.train.Saver()
      saver0.save(sess, saver0_ckpt)
      # Generates MetaGraphDef.
      saver0.export_meta_graph(filename)

  def _testGraphExtensionRestore(self):
    test_dir = os.path.join(self.get_temp_dir(), "graph_extension")
    filename = os.path.join(test_dir, "metafile")
    train_filename = os.path.join(test_dir, "train_metafile")
    saver0_ckpt = os.path.join(test_dir, "saver0.ckpt")
    with self.test_session(graph=tf.Graph()) as sess:
      # Restores from MetaGraphDef.
      new_saver = tf.train.import_meta_graph(filename)
      # Generates a new MetaGraphDef.
      new_saver.export_meta_graph()
      # Restores from checkpoint.
      new_saver.restore(sess, saver0_ckpt)
      # Adds loss and train.
      labels = tf.constant(0, tf.int32, shape=[100], name="labels")
      batch_size = tf.size(labels)
      labels = tf.expand_dims(labels, 1)
      indices = tf.expand_dims(tf.range(0, batch_size), 1)
      concated = tf.concat(1, [indices, labels])
      onehot_labels = tf.sparse_to_dense(
          concated, tf.pack([batch_size, 10]), 1.0, 0.0)
      logits = tf.get_collection("logits")[0]
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                              onehot_labels,
                                                              name="xentropy")
      loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")

      tf.scalar_summary(loss.op.name, loss)
      # Creates the gradient descent optimizer with the given learning rate.
      optimizer = tf.train.GradientDescentOptimizer(0.01)

      # Runs train_op.
      train_op = optimizer.minimize(loss)
      tf.add_to_collection("train_op", train_op)

      # Runs train_op.
      sess.run(train_op)

      # Generates MetaGraphDef.
      tf.train.export_meta_graph(train_filename)

  def _testRestoreFromTrainGraphWithControlContext(self):
    test_dir = os.path.join(self.get_temp_dir(), "graph_extension")
    train_filename = os.path.join(test_dir, "train_metafile")
    saver0_ckpt = os.path.join(test_dir, "saver0.ckpt")
    with self.test_session(graph=tf.Graph()) as sess:
      # Restores from MetaGraphDef.
      new_saver = tf.train.import_meta_graph(train_filename)
      # Restores from checkpoint.
      new_saver.restore(sess, saver0_ckpt)
      train_op = tf.get_collection("train_op")[0]
      sess.run(train_op)

  def testGraphExtension(self):
    self._testGraphExtensionSave()
    self._testGraphExtensionRestore()
    self._testRestoreFromTrainGraphWithControlContext()

  def testStrippedOpListDef(self):
    with self.test_session():
      # Creates a graph.
      v0 = tf.Variable(0.0)
      var = tf.Variable(10.0)
      tf.add(v0, var)
      @function.Defun(tf.float32)
      def minus_one(x):
        return x - 1
      minus_one(tf.identity(v0))
      save = tf.train.Saver({"v0": v0})
      tf.initialize_all_variables()

      # Generates MetaGraphDef.
      meta_graph_def = save.export_meta_graph()
      ops = [o.name for o in meta_graph_def.meta_info_def.stripped_op_list.op]
      self.assertEqual(ops, ["Add", "Assign", "Const", "Identity", "NoOp",
                             "RestoreV2", "SaveSlices", "Sub", "Variable"])

      # Test calling stripped_op_list_for_graph directly
      op_list = tf.contrib.util.stripped_op_list_for_graph(
          meta_graph_def.graph_def)
      self.assertEqual(ops, [o.name for o in op_list.op])
      for o in op_list.op:
        self.assertEqual(o.summary, "")
        self.assertEqual(o.description, "")

  def testStrippedOpListNestedFunctions(self):
    with self.test_session():
      # Square two levels deep
      @function.Defun(tf.int32)
      def f0(x):
        return tf.square(x)
      @function.Defun(tf.int32)
      def f1(x):
        return f0(x)

      # At this point we've defined two functions but haven't called them, so
      # there should be no used ops.
      op_list = tf.contrib.util.stripped_op_list_for_graph(
          tf.get_default_graph().as_graph_def())
      self.assertEquals(len(op_list.op), 0)

      # If we call the function on a constant, there should be two ops
      _ = f1(tf.constant(7))
      op_list = tf.contrib.util.stripped_op_list_for_graph(
          tf.get_default_graph().as_graph_def())
      self.assertEquals(["Const", "Square"], [op.name for op in op_list.op])

  def testStrippedOpListRecursiveFunctions(self):
    # The function module doesn't support recursive functions, so we build a
    # recursive function situation by ourselves: A calls B calls A and Const.
    graph = graph_pb2.GraphDef()
    a = graph.library.function.add()
    b = graph.library.function.add()
    a.signature.name = "A"
    b.signature.name = "B"
    a.node.add().op = "B"
    b.node.add().op = "Const"
    b.node.add().op = "A"

    # Use A in the graph
    graph.node.add().op = "A"

    # The stripped op list should contain just Const.
    op_list = tf.contrib.util.stripped_op_list_for_graph(graph)
    self.assertEquals(["Const"], [op.name for op in op_list.op])


class CheckpointReaderTest(tf.test.TestCase):

  _WRITE_VERSION = saver_pb2.SaverDef.V1

  def testDebugString(self):
    # Builds a graph.
    v0 = tf.Variable([[1, 2, 3], [4, 5, 6]], dtype=tf.float32, name="v0")
    v1 = tf.Variable([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=tf.float32,
                     name="v1")
    init_all_op = tf.initialize_all_variables()
    save = tf.train.Saver(
        {"v0": v0,
         "v1": v1}, write_version=self._WRITE_VERSION)
    save_path = os.path.join(self.get_temp_dir(),
                             "ckpt_for_debug_string" + str(self._WRITE_VERSION))
    with self.test_session() as sess:
      sess.run(init_all_op)
      # Saves a checkpoint.
      save.save(sess, save_path)

      # Creates a reader.
      reader = tf.train.NewCheckpointReader(save_path)
      # Verifies that the tensors exist.
      self.assertTrue(reader.has_tensor("v0"))
      self.assertTrue(reader.has_tensor("v1"))
      debug_string = reader.debug_string()
      # Verifies that debug string contains the right strings.
      self.assertTrue(compat.as_bytes("v0 (DT_FLOAT) [2,3]") in debug_string)
      self.assertTrue(compat.as_bytes("v1 (DT_FLOAT) [3,2,1]") in debug_string)
      # Verifies get_variable_to_shape_map() returns the correct information.
      var_map = reader.get_variable_to_shape_map()
      self.assertEquals([2, 3], var_map["v0"])
      self.assertEquals([3, 2, 1], var_map["v1"])
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
      tf.train.NewCheckpointReader("non-existent")


class CheckpointReaderForV2Test(CheckpointReaderTest):

  _WRITE_VERSION = saver_pb2.SaverDef.V2


class WriteGraphTest(tf.test.TestCase):

  def testRecursiveCreate(self):
    test_dir = _TestDir("deep_dir")
    tf.Variable([[1, 2, 3], [4, 5, 6]], dtype=tf.float32, name="v0")
    tf.train.write_graph(tf.get_default_graph().as_graph_def(),
                         "/".join([test_dir, "l1/l2/l3"]), "graph.pbtxt")


if __name__ == "__main__":
  tf.test.main()
