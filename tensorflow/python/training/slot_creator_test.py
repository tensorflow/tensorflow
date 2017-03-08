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
# ==============================================================================

"""Functional test for slot_creator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.training import slot_creator


class SlotCreatorTest(tf.test.TestCase):

  def testCreateSlotFromVariable(self):
    with self.test_session():
      v = tf.Variable([1.0, 2.5], name="var")
      slot = slot_creator.create_slot(v, v.initialized_value(), name="slot")

      tf.global_variables_initializer().run()

      self.assertEqual(slot.op.name, "var/slot")
      self.assertEqual(slot.get_shape().as_list(), [2])
      self.assertEqual(slot.dtype.base_dtype, tf.float32)
      self.assertAllEqual(slot.eval(), [1.0, 2.5])

  def testCreateSlotFromTensor(self):
    with self.test_session():
      v = tf.constant([1.0, 2.5], name="const")
      slot = slot_creator.create_slot(v, v * 2, name="slot")

      tf.global_variables_initializer().run()

      self.assertEqual(slot.op.name, "const/slot")
      self.assertEqual(slot.get_shape().as_list(), [2])
      self.assertEqual(slot.dtype.base_dtype, tf.float32)
      self.assertAllEqual(slot.eval(), [2.0, 5.0])

  def testCreateZerosSlotFromVariable(self):
    with self.test_session():
      v = tf.Variable([1.0, 2.5], name="var")
      with tf.control_dependencies(None):
        slot = slot_creator.create_zeros_slot(v, name="slot", dtype=tf.float64)

      tf.global_variables_initializer().run()

      self.assertEqual(slot.op.name, "var/slot")
      self.assertEqual(slot.get_shape().as_list(), [2])
      self.assertEqual(slot.dtype.base_dtype, tf.float64)
      self.assertAllEqual(slot.eval(), [0.0, 0.0])

  def testCreateZerosSlotFromTensor(self):
    with self.test_session():
      v = tf.constant([1.0, 2.5], name="const")
      with tf.control_dependencies(None):
        slot = slot_creator.create_zeros_slot(v, name="slot")

      tf.global_variables_initializer().run()

      self.assertEqual(slot.op.name, "const/slot")
      self.assertEqual(slot.get_shape().as_list(), [2])
      self.assertEqual(slot.dtype.base_dtype, tf.float32)
      self.assertAllEqual(slot.eval(), [0.0, 0.0])

  def testCreateSlotFromVariableRespectsScope(self):
    # See discussion on #2740.
    with self.test_session():
      with tf.variable_scope("scope"):
        v = tf.Variable([1.0, 2.5], name="var")
        slot = slot_creator.create_slot(v, v.initialized_value(), name="slot")
        self.assertEqual(slot.op.name, "scope/scope/var/slot")

if __name__ == "__main__":
  tf.test.main()
