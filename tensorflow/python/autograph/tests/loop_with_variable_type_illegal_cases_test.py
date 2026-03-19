# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Loops with type changing variables."""

import re

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def while_with_variable_py_type():
  n = tf.constant(0, dtype=tf.int32)
  c = True
  while c:
    c = tf.constant(True)
  return n


def while_with_variable_dtype():
  n = tf.constant(0, dtype=tf.int32)
  while tf.constant(True):
    n = tf.constant(0, dtype=tf.float32)
  return n


def while_with_variable_dtype_and_early_stopping():
  n = tf.constant(0, dtype=tf.int32)
  while tf.constant(True):
    n = tf.constant(0, dtype=tf.float32)
    break
  return n


def for_with_variable_dtype(l):
  n = tf.constant(0, dtype=tf.int32)
  for _ in l:
    n = tf.constant(0, dtype=tf.float32)
  return n


def for_with_variable_dtype_and_early_stopping(l):
  n = tf.constant(0, dtype=tf.int32)
  for _ in l:
    n = tf.constant(0, dtype=tf.float32)
    break
  return n


def while_with_variable_shape():
  t = tf.constant([1])
  while tf.constant(True):
    t = tf.constant([1, 1])
  return t


def for_with_variable_shape(l):
  t = tf.constant([1])
  for _ in l:
    t = tf.constant([1, 1])
  return t


def while_with_shape_erasure():
  t = tf.constant([1])
  while tf.constant(True):
    t = tf.range(tf.random.uniform((), 2, 3, dtype=tf.int32))
  return t


def for_with_shape_erasure(l):
  t = tf.constant([1])
  for _ in l:
    t = tf.range(tf.random.uniform((), 2, 3, dtype=tf.int32))
  return t


def while_with_shape_invariant_violation():
  t = tf.constant([1])
  while tf.constant(True):
    tf.autograph.experimental.set_loop_options(
        shape_invariants=((t, tf.TensorShape([1])),))
    t = tf.range(tf.random.uniform((), 2, 3, dtype=tf.int32))
  return t


def for_with_shape_invariant_violation(l):
  t = tf.constant([1])
  for _ in l:
    tf.autograph.experimental.set_loop_options(
        shape_invariants=((t, tf.TensorShape([1])),))
    t = tf.range(tf.random.uniform((), 2, 3, dtype=tf.int32))
  return t


def while_with_variable_structure():
  s = {'a': tf.constant(0)}
  while tf.constant(True):
    s = tf.constant(7.0)
  return s


def for_with_variable_structure(l):
  s = [tf.constant(0)]
  for _ in l:
    s = s + [tf.constant(0)]
  return s


def _tf_range(l):
  return tf.range(len(l))


def _dataset(l):
  return tf.data.Dataset.from_tensor_slices(l)


def _dataset_iterator(l):
  return iter(tf.data.Dataset.from_tensor_slices(l))


def _distributed_dataset(l):
  ds = tf.data.Dataset.from_tensor_slices([l] * 2)
  return tf.distribute.MirroredStrategy().experimental_distribute_dataset(ds)


class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):

  def test_while_with_variable_py_type(self):
    with self.assertRaisesRegex(
        NotImplementedError,
        re.compile(
            r'.*condition of while loop started as non\-Tensor,'
            r' then changed to Tensor.*', re.DOTALL)):
      tf.function(while_with_variable_py_type)()

  def test_while_with_variable_dtype(self):
    with self.assertRaisesRegex(
        TypeError,
        "'n' has dtype int32 before the loop, but dtype float32 after"):
      tf.function(while_with_variable_dtype)()

  def test_while_with_variable_dtype_and_early_stopping(self):
    with self.assertRaisesRegex(
        TypeError,
        "'n' has dtype int32 before the loop, but dtype float32 after"):
      tf.function(while_with_variable_dtype_and_early_stopping)()

  @parameterized.parameters(
      (tf.constant,),
      (_tf_range,),
      (_dataset,),
      (_dataset_iterator,),
      (_distributed_dataset,),
  )
  def test_for_with_variable_dtype(self, type_):
    l = type_([1, 2, 3])
    with self.assertRaisesRegex(
        TypeError,
        "'n' has dtype int32 before the loop, but dtype float32 after"):
      tf.function(for_with_variable_dtype)(l)

  # Note: distributed datasets don't allow early stopping.
  @parameterized.parameters(
      (tf.constant,),
      (_tf_range,),
      (_dataset,),
      (_dataset_iterator,),
  )
  def test_for_with_variable_dtype_and_early_stopping(self, type_):
    l = type_([1, 2, 3])
    with self.assertRaisesRegex(
        TypeError,
        "'n' has dtype int32 before the loop, but dtype float32 after"):
      tf.function(for_with_variable_dtype_and_early_stopping)(l)

  def test_while_with_variable_shape(self):
    with self.assertRaisesRegex(
        ValueError,
        r"'t' has shape \(1,\) before the loop, but shape \(2,\) after"):
      tf.function(while_with_variable_shape)()

  # Note: datasets do allow variable shape.
  @parameterized.parameters(
      (tf.constant,),
      (_tf_range,),
      (_dataset_iterator,),
      (_distributed_dataset,),
  )
  def test_for_with_variable_shape(self, type_):
    l = type_([1, 2, 3])
    with self.assertRaisesRegex(
        ValueError,
        r"'t' has shape \(1,\) before the loop, but shape \(2,\) after"):
      tf.function(for_with_variable_shape)(l)

  def test_while_with_shape_erasure(self):
    with self.assertRaisesRegex(
        ValueError,
        r"'t' has shape \(1,\) before the loop, but shape \(None,\) after"):
      tf.function(while_with_shape_erasure)()

  # Note: datasets do allow variable shape.
  @parameterized.parameters(
      (tf.constant,),
      (_tf_range,),
      (_dataset_iterator,),
      (_distributed_dataset,),
  )
  def test_for_with_shape_erasure(self, type_):
    l = type_([1, 2, 3])
    with self.assertRaisesRegex(
        ValueError,
        r"'t' has shape \(1,\) before the loop, but shape \(None,\) after"):
      tf.function(for_with_shape_erasure)(l)

  def test_while_with_shape_invariant_violation(self):
    with self.assertRaisesRegex(
        ValueError,
        r"'t' has shape \(None,\) after one iteration, which does not conform"):
      tf.function(while_with_shape_invariant_violation)()

  # Note: dataset loops ignore shape invariants.
  @parameterized.parameters(
      (tf.constant,),
      (_tf_range,),
      (_dataset_iterator,),
      (_distributed_dataset,),
  )
  def test_for_with_shape_invariant_violation(self, type_):
    l = type_([1, 2, 3])
    with self.assertRaisesRegex(
        ValueError,
        r"'t' has shape \(None,\) after one iteration, which does not conform"):
      tf.function(for_with_shape_invariant_violation)(l)

  def test_while_with_variable_structure(self):
    with self.assertRaisesRegex(
        TypeError,
        "'s' does not have the same nested structure"):
      tf.function(while_with_variable_structure)()

  @parameterized.parameters(
      (tf.constant,),
      (_tf_range,),
      (_dataset,),
      (_dataset_iterator,),
      (_distributed_dataset,),
  )
  def test_for_with_variable_structure(self, type_):
    l = type_([1, 2, 3])
    with self.assertRaisesRegex(
        TypeError,
        "'s' does not have the same nested structure"):
      tf.function(for_with_variable_structure)(l)


if __name__ == '__main__':
  tf.test.main()
