# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Tests for partitioned_variables.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


def _IotaInitializer(shape, dtype=tf.float32):
  assert dtype == tf.float32
  if len(shape) == 1:
    return range(shape[0])
  else:
    val = _IotaInitializer(shape[1:], dtype)
    return [[(10 ** i) * v for v in val] for i in range(shape[0])]


class PartitionedVariablesTestCase(tf.test.TestCase):

  def _TestSaveSpec(self, slices, expected_specs):
    self.assertEqual(len(expected_specs), len(slices))
    for i in xrange(len(expected_specs)):
      self.assertEquals(expected_specs[i], slices[i]._save_slice_info.spec)

  def testVecConstantInit(self):
    with self.test_session():
      rnd_par = tf.constant([1, 2, 3, 4])
      vs = tf.create_partitioned_variables([4], [4], rnd_par)
      tf.initialize_all_variables().run()
      val = tf.concat(0, vs).eval()
      rnd = rnd_par.eval()
      self.assertAllClose(rnd, val)
      self.assertEqual([tf.int32] * 4, [v.dtype.base_dtype for v in vs])
      self._TestSaveSpec(vs, ["4 0,1", "4 1,1", "4 2,1", "4 3,1"])

  def testConstantInit(self):
    with self.test_session():
      rnd_par = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
      vs = tf.create_partitioned_variables([2, 4], [1, 2], rnd_par)
      tf.initialize_all_variables().run()
      val = tf.concat(1, vs).eval()
      rnd = rnd_par.eval()
      self.assertAllClose(rnd, val)
      self.assertEqual([tf.int32] * 2, [v.dtype.base_dtype for v in vs])
      self._TestSaveSpec(vs, ["2 4 0,2:0,2", "2 4 0,2:2,2"])

  def testName(self):
    with self.test_session():
      rnd_par = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
      with tf.variable_scope("hi"):
        vs1 = tf.create_partitioned_variables([2, 4], [1, 2], rnd_par)
        vs2 = tf.create_partitioned_variables([2, 4], [1, 2], rnd_par)
      tf.initialize_all_variables().run()
      var1_name = vs1[0]._save_slice_info.full_name
      var2_name = vs2[0]._save_slice_info.full_name
      self.assertEqual("hi/PartitionedVariable", var1_name)
      self.assertEqual("hi/PartitionedVariable_1", var2_name)
      self.assertEqual(var1_name + "/part_0:0", vs1[0].name)
      self.assertEqual(var1_name + "/part_1:0", vs1[1].name)
      self.assertEqual(var2_name + "/part_0:0", vs2[0].name)
      self.assertEqual(var2_name + "/part_1:0", vs2[1].name)
    # Test same variable.
    with self.test_session():
      rnd_par = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
      with tf.variable_scope("hola") as vs:
        vs1 = tf.create_partitioned_variables([2, 4], [1, 2], rnd_par)
      with tf.variable_scope(vs, reuse=True):
        vs2 = tf.create_partitioned_variables([2, 4], [1, 2], rnd_par)
      tf.initialize_all_variables().run()
      var1_name = vs1[0]._save_slice_info.full_name
      var2_name = vs2[0]._save_slice_info.full_name
      self.assertEqual("hola/PartitionedVariable", var1_name)
      self.assertEqual("hola/PartitionedVariable", var2_name)
      self.assertEqual(var1_name + "/part_0:0", vs1[0].name)
      self.assertEqual(var1_name + "/part_1:0", vs1[1].name)
      self.assertEqual(var2_name + "/part_0:0", vs2[0].name)
      self.assertEqual(var2_name + "/part_1:0", vs2[1].name)
    # Test name_scope
    with self.test_session():
      rnd_par = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
      with tf.name_scope("ola"):
        vs1 = tf.create_partitioned_variables([2, 4], [1, 2], rnd_par)
        vs2 = tf.create_partitioned_variables([2, 4], [1, 2], rnd_par)
      tf.initialize_all_variables().run()
      var1_name = vs1[0]._save_slice_info.full_name
      var2_name = vs2[0]._save_slice_info.full_name
      # Currently, the name scope 'ola' has no effect.
      self.assertEqual("PartitionedVariable", var1_name)
      self.assertEqual("PartitionedVariable_1", var2_name)
      self.assertEqual(var1_name + "/part_0:0", vs1[0].name)
      self.assertEqual(var1_name + "/part_1:0", vs1[1].name)
      self.assertEqual(var2_name + "/part_0:0", vs2[0].name)
      self.assertEqual(var2_name + "/part_1:0", vs2[1].name)

  def testRandomInitValue(self):
    with self.test_session():
      rnd = tf.Variable(tf.random_uniform([200, 40]))
      vs = tf.create_partitioned_variables(
          rnd.get_shape(), [1, 10], rnd.initialized_value())
      tf.initialize_all_variables().run()
      val = tf.concat(1, vs).eval()
      rnd = rnd.eval()
      self.assertAllClose(rnd, val)
      self.assertEqual([tf.float32] * 10, [v.dtype.base_dtype for v in vs])
      self._TestSaveSpec(vs, ["200 40 0,200:0,4",
                              "200 40 0,200:4,4",
                              "200 40 0,200:8,4",
                              "200 40 0,200:12,4",
                              "200 40 0,200:16,4",
                              "200 40 0,200:20,4",
                              "200 40 0,200:24,4",
                              "200 40 0,200:28,4",
                              "200 40 0,200:32,4",
                              "200 40 0,200:36,4"])

  def testRandomInitUnevenPartitions(self):
    with self.test_session():
      rnd = tf.Variable(
          tf.random_uniform([20, 43], dtype=tf.float64))
      var_lists = [
          tf.create_partitioned_variables(
              rnd.get_shape(), [1, i],
              rnd.initialized_value())
          for i in xrange(1, 10)]
      tf.initialize_all_variables().run()
      rnd_val = rnd.eval()
      # Only check the slice save specs for the first 5 tf.
      save_specs = [
          # One slice
          ["20 43 0,20:0,43"],
          # Two slices
          ["20 43 0,20:0,22",
           "20 43 0,20:22,21"],
          # Three slices
          ["20 43 0,20:0,15",
           "20 43 0,20:15,14",
           "20 43 0,20:29,14"],
          # Four slices
          ["20 43 0,20:0,11",
           "20 43 0,20:11,11",
           "20 43 0,20:22,11",
           "20 43 0,20:33,10"],
          # Five slices
          ["20 43 0,20:0,9",
           "20 43 0,20:9,9",
           "20 43 0,20:18,9",
           "20 43 0,20:27,8",
           "20 43 0,20:35,8"]]
      for i, vs in enumerate(var_lists):
        var_val = tf.concat(1, vs).eval()
        self.assertAllClose(rnd_val, var_val)
        self.assertEqual(
            [tf.float64] * len(vs), [v.dtype.base_dtype for v in vs])
        if i < len(save_specs):
          self._TestSaveSpec(vs, save_specs[i])

  def testDegenerate(self):
    with self.test_session():
      rnd = tf.Variable(tf.random_uniform([10, 43]))
      vs = tf.create_partitioned_variables(
          rnd.get_shape(), [1, 1], rnd.initialized_value())
      tf.initialize_all_variables().run()
      val = tf.concat(0, vs).eval()
      rnd = rnd.eval()
      self.assertAllClose(rnd, val)
      self._TestSaveSpec(vs, ["10 43 0,10:0,43"])

  def testSliceSizeOne(self):
    with self.test_session():
      rnd = tf.Variable(tf.random_uniform([10, 43]))
      vs = tf.create_partitioned_variables(
          rnd.get_shape(), [10, 1], rnd.initialized_value())
      tf.initialize_all_variables().run()
      val = tf.concat(0, vs).eval()
      rnd = rnd.eval()
      self.assertAllClose(rnd, val)
      self._TestSaveSpec(vs, ["10 43 0,1:0,43",
                              "10 43 1,1:0,43",
                              "10 43 2,1:0,43",
                              "10 43 3,1:0,43",
                              "10 43 4,1:0,43",
                              "10 43 5,1:0,43",
                              "10 43 6,1:0,43",
                              "10 43 7,1:0,43",
                              "10 43 8,1:0,43",
                              "10 43 9,1:0,43"])

  def testIotaInitializer(self):
    self.assertAllClose([0., 1., 2., 3.], _IotaInitializer([4]))
    self.assertAllClose([[0., 1.], [0., 10.], [0., 100.], [0., 1000.]],
                        _IotaInitializer([4, 2]))
    with self.test_session():
      vs = tf.create_partitioned_variables([13, 5], [3, 1], _IotaInitializer)
      tf.initialize_all_variables().run()
      slice0 = _IotaInitializer([5, 5])
      slice1 = _IotaInitializer([4, 5])
      slice2 = _IotaInitializer([4, 5])
      val = tf.concat(0, vs).eval()
      self.assertAllClose(slice0 + slice1 + slice2, val)
      self._TestSaveSpec(vs, ["13 5 0,5:0,5",
                              "13 5 5,4:0,5",
                              "13 5 9,4:0,5"])

  def testRandomInitializer(self):
    # Sanity check that the slices uses a different seed when using a random
    # initializer function.
    with self.test_session():
      var0, var1 = tf.create_partitioned_variables(
          [20, 12], [1, 2], tf.random_uniform_initializer())
      tf.initialize_all_variables().run()
      val0, val1 = var0.eval().flatten(), var1.eval().flatten()
      self.assertTrue(np.linalg.norm(val0 - val1) > 1e-6)
    # Negative test that proves that slices have the same values if
    # the random initializer uses a seed.
    with self.test_session():
      var0, var1 = tf.create_partitioned_variables(
          [20, 12], [1, 2], tf.random_uniform_initializer(seed=201))
      tf.initialize_all_variables().run()
      val0, val1 = var0.eval().flatten(), var1.eval().flatten()
      self.assertAllClose(val0, val1)

  def testSomeErrors(self):
    with self.test_session():
      rnd = tf.Variable(tf.random_uniform([10, 43]))
      with self.assertRaises(ValueError):
        tf.create_partitioned_variables([10], [1, 1], rnd.initialized_value())
      with self.assertRaises(ValueError):
        tf.create_partitioned_variables([10, 20], [1], rnd.initialized_value())
      with self.assertRaises(ValueError):
        tf.create_partitioned_variables([10, 43], [1], rnd.initialized_value())
      with self.assertRaises(ValueError):
        tf.create_partitioned_variables(
            [10, 43], [1, 2, 3], rnd.initialized_value())
      with self.assertRaises(ValueError):
        tf.create_partitioned_variables(
            [10, 43], [11, 1], rnd.initialized_value())
      with self.assertRaises(ValueError):
        tf.create_partitioned_variables(
            [10, 43], [20, 1], rnd.initialized_value())
      with self.assertRaises(ValueError):
        tf.create_partitioned_variables(
            [10, 43], [1, 50], rnd.initialized_value())


if __name__ == "__main__":
  tf.test.main()
