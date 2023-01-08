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

import collections
import itertools

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def while_with_variable_shape_growing_vector(n):
  v = tf.constant([0, 0])
  i = 0
  while i < n:
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(v, tf.TensorShape([None]))])
    v = tf.concat((v, [i]), 0)
    i += 1
  return v


def for_with_variable_shape_growing_vector(l):
  v = tf.constant([0, 0])
  for i in l:
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(v, tf.TensorShape([None]))])
    v = tf.concat((v, [i]), 0)
  return v


def while_with_variable_shape_growing_matrix_rows(n):
  m = tf.constant([[0]])
  i = 0
  while i < n:
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(m, tf.TensorShape([None, 1]))])
    m = tf.concat((m, [[i]]), 0)
    i += 1
  return m


def for_with_variable_shape_growing_matrix_rows(l):
  m = tf.constant([[0]])
  for i in l:
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(m, tf.TensorShape([None, 1]))])
    m = tf.concat((m, [[i]]), 0)
  return m


def while_with_variable_shape_growing_matrix_cols(n):
  m = tf.constant([[0, 0]])
  i = 0
  while i < n:
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(m, tf.TensorShape([1, None]))])
    m = tf.concat((m, [[i]]), 1)
    i += 1
  return m


def for_with_variable_shape_growing_matrix_cols(l):
  m = tf.constant([[0, 0]])
  for i in l:
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(m, tf.TensorShape([1, None]))])
    m = tf.concat((m, [[i]]), 1)
  return m


def while_with_variable_shape_growing_matrix(n):
  m = tf.constant([[0, 0], [0, 0]])
  i = 0
  while i < n:
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(m, tf.TensorShape(None))])
    m = tf.pad(m, [[1, 1], [1, 1]], constant_values=i)
    i += 1
  return m


def for_with_variable_shape_growing_matrix(l):
  m = tf.constant([[0, 0], [0, 0]])
  for i in l:
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(m, tf.TensorShape(None))])
    m = tf.pad(m, [[1, 1], [1, 1]], constant_values=i)
  return m


def while_with_variable_shape_inside_if(n):
  v = tf.constant([0, 0])
  i = 0
  if n > 1:
    while i < n:
      tf.autograph.experimental.set_loop_options(
          shape_invariants=[(v, tf.TensorShape([None]))])
      v = tf.concat((v, [i]), 0)
      i += 1
  else:
    v = tf.constant([1, 2, 3])
  return v


def for_with_variable_shape_inside_if(n):
  v = tf.constant([0, 0])
  if n > 1:
    for i in range(n):
      tf.autograph.experimental.set_loop_options(
          shape_invariants=[(v, tf.TensorShape([None]))])
      v = tf.concat((v, [i]), 0)
      i += 1
  else:
    v = tf.constant([1, 2, 3])
  return v


def for_with_nested_variable_shape_inside_if(n):
  Test = collections.namedtuple('Test', ['var'])
  t = Test(var=tf.constant([0]))
  v = tf.constant([0, 0])
  if n > 1:
    for i in range(n):
      tf.autograph.experimental.set_loop_options(
          shape_invariants=[(v, tf.TensorShape([None]))])
      v = tf.concat((v, [i]), 0)
      t = Test(var=t.var + 1)
      i += 1
  else:
    v = tf.constant([1, 2, 3])
    t = Test(var=tf.constant([3]))
  return v


def while_with_variable_shape_and_break(n):
  v = tf.constant([0, 0])
  i = 0
  if n > 1:
    while i < n:
      tf.autograph.experimental.set_loop_options(
          shape_invariants=[(v, tf.TensorShape([None]))])
      v = tf.concat((v, [i]), 0)
      i += 1
      if i > 3:
        break
  else:
    v = tf.constant([1, 2, 3])
  return v


def for_with_variable_shape_and_break(n):
  v = tf.constant([0, 0])
  if n > 1:
    for i in range(n):
      tf.autograph.experimental.set_loop_options(
          shape_invariants=[(v, tf.TensorShape([None]))])
      v = tf.concat((v, [i]), 0)
      i += 1
      if i > 3:
        break
  else:
    v = tf.constant([1, 2, 3])
  return v


def while_with_composite_tensor_shape_invariant(n):
  v = tf.SparseTensor(
      indices=[[0, 0], [1, 1]], values=[1, 2], dense_shape=[3, 3])
  i = 0
  while i < n:
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(v, tf.TensorShape(None))])
    v = tf.sparse.expand_dims(v)
    i += 1
  return v


def for_with_composite_tensor_shape_invariant(l):
  v = tf.SparseTensor(
      indices=[[0, 0], [1, 1]], values=[1, 2], dense_shape=[3, 3])
  for _ in l:
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(v, tf.TensorShape(None))])
    v = tf.sparse.expand_dims(v)
  return v


def _int_dataset_range(n):
  return tf.data.Dataset.range(n).map(lambda x: tf.cast(x, tf.int32))


class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_while_with_variable_shape_growing_vector(self, n, type_):
    n = type_(n)
    self.assertFunctionMatchesEager(while_with_variable_shape_growing_vector, n)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (range, tf.range, tf.data.Dataset.range),
  ))
  def test_for_with_variable_shape_growing_vector(self, n, list_type):
    l = list_type(n)
    self.assertFunctionMatchesEager(for_with_variable_shape_growing_vector, l)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_while_with_variable_shape_growing_matrix_rows(self, n, type_):
    n = type_(n)
    self.assertFunctionMatchesEager(
        while_with_variable_shape_growing_matrix_rows, n)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (range, tf.range, _int_dataset_range),
  ))
  def test_for_with_variable_shape_growing_matrix_rows(self, l, type_):
    l = type_(l)
    self.assertFunctionMatchesEager(
        for_with_variable_shape_growing_matrix_rows, l)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_while_with_variable_shape_growing_matrix_cols(self, n, type_):
    n = type_(n)
    self.assertFunctionMatchesEager(
        while_with_variable_shape_growing_matrix_cols, n)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (range, tf.range, tf.data.Dataset.range),
  ))
  def test_for_with_variable_shape_growing_matrix_cols(self, l, type_):
    l = type_(l)
    self.assertFunctionMatchesEager(
        for_with_variable_shape_growing_matrix_cols, l)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_while_with_variable_shape_growing_matrix(self, n, type_):
    n = type_(n)
    self.assertFunctionMatchesEager(while_with_variable_shape_growing_matrix, n)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (range, tf.range, _int_dataset_range),
  ))
  def test_for_with_variable_shape_growing_matrix(self, n, type_):
    l = type_(n)
    self.assertFunctionMatchesEager(for_with_variable_shape_growing_matrix, l)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_while_with_variable_shape_inside_if(self, n, type_):
    n = type_(n)
    self.assertFunctionMatchesEager(while_with_variable_shape_inside_if, n)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_for_with_variable_shape_inside_if(self, n, type_):
    n = type_(n)
    self.assertFunctionMatchesEager(for_with_variable_shape_inside_if, n)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_for_with_nested_variable_shape_inside_if(self, n, type_):
    n = type_(n)
    self.assertFunctionMatchesEager(for_with_nested_variable_shape_inside_if, n)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_while_with_variable_shape_and_break(self, n, type_):
    n = type_(n)
    self.assertFunctionMatchesEager(while_with_variable_shape_and_break, n)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2, 5),
      (int, tf.constant),
  ))
  def test_for_with_variable_shape_and_break(self, n, type_):
    n = type_(n)
    self.assertFunctionMatchesEager(for_with_variable_shape_and_break, n)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2, 5),
      (int, tf.constant),
  ))
  def test_while_with_composite_tensor_shape_invariant(self, n, type_):
    n = type_(n)
    self.assertFunctionMatchesEager(
        while_with_composite_tensor_shape_invariant, n)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (range, tf.range, _int_dataset_range),
  ))
  def test_for_with_composite_tensor_shape_invariant(self, n, type_):
    l = type_(n)
    self.assertFunctionMatchesEager(
        for_with_composite_tensor_shape_invariant, l)


if __name__ == '__main__':
  tf.test.main()
