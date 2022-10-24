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
"""Basic list operations."""

import tensorflow as tf

from tensorflow.python import autograph as ag
from tensorflow.python.autograph.tests import reference_test_base


def type_not_annotated(n):
  l = []
  # TODO(mdan): Here, we ought to infer the dtype and shape when i is staged.
  for i in range(n):
    l.append(i)
  return ag.stack(l, strict=False)


def element_access():
  l = []
  l.append(1)
  l.append(2)
  l.append(3)
  ag.set_element_type(l, tf.int32)
  return 2 * l[1]


def element_update():
  l = []
  l.append(1)
  l.append(2)
  l.append(3)
  ag.set_element_type(l, tf.int32)
  l[1] = 5
  return ag.stack(l, strict=False)


def simple_fill(n):
  l = []
  ag.set_element_type(l, tf.int32)
  for i in range(n):
    l.append(i)
  return ag.stack(l, strict=False)


def nested_fill(m, n):
  mat = []
  ag.set_element_type(mat, tf.int32)
  for _ in range(m):
    l = []
    ag.set_element_type(l, tf.int32)
    for j in range(n):
      l.append(j)
    mat.append(ag.stack(l, strict=False))
  return ag.stack(mat, strict=False)


def read_write_loop(n):
  l = []
  l.append(1)
  l.append(1)
  ag.set_element_type(l, tf.int32)
  for i in range(2, n):
    l.append(l[i-1] + l[i-2])
    l[i-2] = -l[i-2]
  return ag.stack(l, strict=False)


def simple_empty(n):
  l = []
  l.append(1)
  l.append(2)
  l.append(3)
  l.append(4)
  ag.set_element_type(l, tf.int32, ())
  s = 0
  for _ in range(n):
    s += l.pop()
  return ag.stack(l, strict=False), s


def mutation(t, n):
  for i in range(n):
    t[i] = i
  return t


class ReferenceTest(reference_test_base.TestCase):

  def setUp(self):
    super(ReferenceTest, self).setUp()
    self.autograph_opts = tf.autograph.experimental.Feature.LISTS

  def test_tensor_mutation(self):
    self.assertConvertedMatchesNative(mutation, [0] * 10, 10)

  def test_basic(self):
    self.all_inputs_tensors = True
    self.assertFunctionMatchesEager(element_access)
    self.assertFunctionMatchesEager(element_update)

    # TODO(mdan): This should raise a compilation, not runtime, error.
    with self.assertRaisesRegex(
        ValueError,
        'cannot stack a list without knowing its element type; '
        'use set_element_type to annotate it'):
      self.function(type_not_annotated)(3)

    self.assertFunctionMatchesEager(simple_fill, 5)
    self.assertFunctionMatchesEager(nested_fill, 5, 3)
    self.assertFunctionMatchesEager(read_write_loop, 4)
    self.assertFunctionMatchesEager(simple_empty, 0)
    self.assertFunctionMatchesEager(simple_empty, 2)
    self.assertFunctionMatchesEager(simple_empty, 4)

    # TODO(mdan): Allow explicitly setting the element shape to mitigate these.
    # TODO(mdan): This should raise a friendlier runtime error.
    # The error should spell out that empty lists cannot be stacked.
    # Alternatively, we can also insert conditionals that construct a zero-sized
    # Tensor of the appropriate type and shape, but we first want to make sure
    # that doesn't degrade performance.
    with self.assertRaises(ValueError):
      self.function(simple_fill)(0)
    with self.assertRaises(ValueError):
      self.function(nested_fill)(0, 3)


if __name__ == '__main__':
  tf.test.main()
