"""Tests for tensorflow.ops.io_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time

import tensorflow.python.platform

import tensorflow as tf
import numpy as np
import six

from tensorflow.python.platform import gfile


class SaverTest(tf.test.TestCase):

  def testBasics(self):
    save_path = os.path.join(self.get_temp_dir(), "basics")

    with self.test_session() as sess:
      # Build a graph with 2 parameter nodes, and Save and
      # Restore nodes for them.
      v0 = tf.Variable(10.0, name="v0")
      v1 = tf.Variable(20.0, name="v1")
      save = tf.train.Saver({"v0": v0, "v1": v1}, restore_sequentially=True)
      tf.initialize_all_variables().run()

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      # Save the initialized values in the file at "save_path"
      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

    # Start a second session.  In that session the parameter nodes
    # have not been initialized either.
    with self.test_session() as sess:
      v0 = tf.Variable(-1.0, name="v0")
      v1 = tf.Variable(-1.0, name="v1")
      save = tf.train.Saver({"v0": v0, "v1": v1})

      with self.assertRaisesWithPredicateMatch(
          tf.OpError, lambda e: "uninitialized value v0" in e.message):
        sess.run(v0)
      with self.assertRaisesWithPredicateMatch(
          tf.OpError, lambda e: "uninitialized value v1" in e.message):
        sess.run(v1)

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

    # Build another graph with 2 nodes, initialized
    # differently, and a Restore node for them.
    with self.test_session() as sess:
      v0_2 = tf.Variable(1000.0, name="v0")
      v1_2 = tf.Variable(2000.0, name="v1")
      save2 = tf.train.Saver({"v0": v0_2, "v1": v1_2})
      tf.initialize_all_variables().run()

      # Check that the parameter nodes have been initialized.
      self.assertEqual(1000.0, v0_2.eval())
      self.assertEqual(2000.0, v1_2.eval())
      # Restore the values saved earlier in the parameter nodes.
      save2.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, v0_2.eval())
      self.assertEqual(20.0, v1_2.eval())

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
      v2._set_save_slice_info(tf.Variable.SaveSliceInfo("v1", ""))

      # By default the name used for "v2" will be "v1" and raise an error.
      with self.assertRaisesRegexp(ValueError, "same name: v1"):
        tf.train.Saver([v0, v1, v2])

      # The names are different and will work.
      tf.train.Saver({"vee1": v1, "other": [v2]})

  def testBasicsWithListOfVariables(self):
    save_path = os.path.join(self.get_temp_dir(), "basics_with_list")

    with self.test_session(graph=tf.Graph()) as sess:
      # Build a graph with 2 parameter nodes, and Save and
      # Restore nodes for them.
      v0 = tf.Variable(10.0, name="v0")
      v1 = tf.Variable(20.0, name="v1")
      save = tf.train.Saver([v0, v1])
      tf.initialize_all_variables().run()

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      # Save the initialized values in the file at "save_path"
      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

    # Start a second session.  In that session the variables
    # have not been initialized either.
    with self.test_session(graph=tf.Graph()) as sess:
      v0 = tf.Variable(-1.0, name="v0")
      v1 = tf.Variable(-1.0, name="v1")
      save = tf.train.Saver([v0, v1])

      with self.assertRaisesWithPredicateMatch(
          tf.OpError, lambda e: "uninitialized value v0" in e.message):
        sess.run(v0)
      with self.assertRaisesWithPredicateMatch(
          tf.OpError, lambda e: "uninitialized value v1" in e.message):
        sess.run(v1)

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

    # Build another graph with 2 nodes, initialized
    # differently, and a Restore node for them.
    with self.test_session(graph=tf.Graph()) as sess:
      v0_2 = tf.Variable(1000.0, name="v0")
      v1_2 = tf.Variable(2000.0, name="v1")
      save2 = tf.train.Saver([v0_2, v1_2])
      tf.initialize_all_variables().run()

      # Check that the parameter nodes have been initialized.
      self.assertEqual(1000.0, v0_2.eval())
      self.assertEqual(2000.0, v1_2.eval())
      # Restore the values saved earlier in the parameter nodes.
      save2.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, v0_2.eval())
      self.assertEqual(20.0, v1_2.eval())

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

  def testGPU(self):
    if not tf.test.IsBuiltWithCuda():
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
      init = tf.initialize_all_variables()
      save = tf.train.Saver(tf.all_variables())
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


class SaveRestoreShardedTest(tf.test.TestCase):

  def testBasics(self):
    save_path = os.path.join(self.get_temp_dir(), "sharded")

    # Build a graph with 2 parameter nodes on different devices.
    with tf.Session(
        target="",
        config=tf.ConfigProto(device_count={"CPU": 2})) as sess:
      with sess.graph.device("/cpu:0"):
        v0 = tf.Variable(10, name="v0")
      with sess.graph.device("/cpu:1"):
        v1 = tf.Variable(20, name="v1")
      save = tf.train.Saver({"v0": v0, "v1": v1}, sharded=True)
      tf.initialize_all_variables().run()
      val = save.save(sess, save_path)
      self.assertEqual(save_path + "-?????-of-00002", val)

    # Restore a different "v0" from shard 0 of the saved files.
    with tf.Session(
        target="",
        config=tf.ConfigProto(device_count={"CPU": 2})) as sess:
      with sess.graph.device("/cpu:0"):
        v0 = tf.Variable(111, name="v0")
      save = tf.train.Saver({"v0": v0}, sharded=True)
      tf.initialize_all_variables().run()
      self.assertEqual(111, v0.eval())
      save.restore(sess, save_path + "-00000-of-00002")
      self.assertEqual(10, v0.eval())

    # Restore a different "v1" from shard 1 of the saved files.
    with tf.Session(
        target="",
        config=tf.ConfigProto(device_count={"CPU": 2})) as sess:
      with sess.graph.device("/cpu:0"):
        v1 = tf.Variable(222)
      save = tf.train.Saver({"v1": v1}, sharded=True)
      tf.initialize_all_variables().run()
      self.assertEqual(222, v1.eval())
      save.restore(sess, save_path + "-00001-of-00002")
      self.assertEqual(20, v1.eval())

    # Now try a restore with the sharded filename.
    with tf.Session(
        target="",
        config=tf.ConfigProto(device_count={"CPU": 2})) as sess:
      with sess.graph.device("/cpu:0"):
        v0 = tf.Variable(111, name="v0")
      with sess.graph.device("/cpu:1"):
        v1 = tf.Variable(222, name="v1")
      save = tf.train.Saver({"v0": v0, "v1": v1}, sharded=True)
      tf.initialize_all_variables().run()
      self.assertEqual(111, v0.eval())
      self.assertEqual(222, v1.eval())
      save_path = os.path.join(self.get_temp_dir(), "sharded")
      save.restore(sess, save_path + "-?????-of-?????")
      self.assertEqual(10, v0.eval())
      self.assertEqual(20, v1.eval())

  def testSaverDef(self):
    with self.test_session():
      v0 = tf.Variable(123, name="v0")
      save = tf.train.Saver({"v0": v0}, sharded=True)
      sd = save.as_saver_def()
      self.assertTrue(sd.sharded)


class MaxToKeepTest(tf.test.TestCase):

  def testNonSharded(self):
    save_dir = os.path.join(self.get_temp_dir(), "max_to_keep_non_sharded")
    try:
      gfile.DeleteRecursively(save_dir)
    except gfile.GOSError as _:
      pass                      # Ignore
    gfile.MakeDirs(save_dir)

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
      self.assertTrue(gfile.Exists(s3))
      self.assertTrue(gfile.Exists(s2))

      # Adding s1 (s3 should now be deleted as oldest in list)
      s1 = save.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([s2, s1], save.last_checkpoints)
      self.assertFalse(gfile.Exists(s3))
      self.assertTrue(gfile.Exists(s2))
      self.assertTrue(gfile.Exists(s1))

      # Exercise the second helper.

      # Adding s2 again (old s2 is removed first, then new s2 appended)
      s2 = save2.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([s3, s2], save2.last_checkpoints)
      # Created by the first helper.
      self.assertTrue(gfile.Exists(s1))
      # Deleted by the first helper.
      self.assertFalse(gfile.Exists(s3))
      self.assertTrue(gfile.Exists(s2))

      # Adding s1 (s3 should now be deleted as oldest in list)
      s1 = save2.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([s2, s1], save2.last_checkpoints)
      self.assertFalse(gfile.Exists(s3))
      self.assertTrue(gfile.Exists(s2))
      self.assertTrue(gfile.Exists(s1))

      # Exercise the third helper.

      # Adding s2 again (but helper is unaware of previous s2)
      s2 = save3.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([s2], save3.last_checkpoints)
      # Created by the first helper.
      self.assertTrue(gfile.Exists(s1))
      # Deleted by the first helper.
      self.assertFalse(gfile.Exists(s3))
      self.assertTrue(gfile.Exists(s2))

      # Adding s1 (s3 should not be deleted because helper is unaware of it)
      s1 = save3.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([s2, s1], save3.last_checkpoints)
      self.assertFalse(gfile.Exists(s3))
      self.assertTrue(gfile.Exists(s2))
      self.assertTrue(gfile.Exists(s1))

  def testSharded(self):
    save_dir = os.path.join(self.get_temp_dir(), "max_to_keep_sharded")
    try:
      gfile.DeleteRecursively(save_dir)
    except gfile.GOSError as _:
      pass                      # Ignore
    gfile.MakeDirs(save_dir)

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
      self.assertEquals(2, len(gfile.Glob(s1)))

      s2 = save.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([s1, s2], save.last_checkpoints)
      self.assertEquals(2, len(gfile.Glob(s1)))
      self.assertEquals(2, len(gfile.Glob(s2)))

      s3 = save.save(sess, os.path.join(save_dir, "s3"))
      self.assertEqual([s2, s3], save.last_checkpoints)
      self.assertEquals(0, len(gfile.Glob(s1)))
      self.assertEquals(2, len(gfile.Glob(s2)))
      self.assertEquals(2, len(gfile.Glob(s3)))


class KeepCheckpointEveryNHoursTest(tf.test.TestCase):

  def testNonSharded(self):
    save_dir = os.path.join(self.get_temp_dir(),
                            "keep_checkpoint_every_n_hours")
    try:
      gfile.DeleteRecursively(save_dir)
    except gfile.GOSError as _:
      pass                      # Ignore
    gfile.MakeDirs(save_dir)

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


if __name__ == "__main__":
  tf.test.main()
