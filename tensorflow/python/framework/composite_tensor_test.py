# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.framework.composite_tensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.util import nest


class TestCompositeTensor(composite_tensor.CompositeTensor):

  def __init__(self, *components):
    self._components = components

  def _to_components(self):
    return self._components

  @classmethod
  def _from_components(cls, components):
    return cls(*components)

  def _shape_invariant_to_components(self, shape=None):
    raise NotImplementedError('CompositeTensor._shape_invariant_to_components')

  def _is_graph_tensor(self):
    return True


class CompositeTensorTest(test_util.TensorFlowTestCase):

  def assertNestEqual(self, a, b, expand_composites=False):
    if isinstance(a, dict):
      self.assertIsInstance(b, dict)
      self.assertEqual(set(a), set(b))
      for key in a:
        self.assertNestEqual(a[key], b[key])
    elif isinstance(a, (list, tuple)):
      self.assertIsInstance(b, (list, tuple))
      self.assertEqual(len(a), len(b))
      for a_val, b_val in zip(a, b):
        self.assertNestEqual(a_val, b_val)
    elif expand_composites and isinstance(a, composite_tensor.CompositeTensor):
      self.assertIsInstance(b, composite_tensor.CompositeTensor)
      self.assertNestEqual(a._to_components(),
                           b._to_components())

  def testNestFlatten(self):
    st1 = sparse_tensor.SparseTensor([[0, 3], [7, 2]], [1, 2], [10, 10])
    st2 = sparse_tensor.SparseTensor([[1, 2, 3]], ['a'], [10, 10, 10])
    structure = [[st1], 'foo', {'y': [st2]}]
    x = nest.flatten(structure, expand_composites=True)
    self.assertEqual(len(x), 7)
    self.assertIs(x[0], st1.indices)
    self.assertIs(x[1], st1.values)
    self.assertIs(x[2], st1.dense_shape)
    self.assertEqual(x[3], 'foo')
    self.assertIs(x[4], st2.indices)
    self.assertIs(x[5], st2.values)
    self.assertIs(x[6], st2.dense_shape)

  def testNestPackSequenceAs(self):
    st1 = sparse_tensor.SparseTensor([[0, 3], [7, 2]], [1, 2], [10, 10])
    st2 = sparse_tensor.SparseTensor([[1, 2, 3]], ['a'], [10, 10, 10])
    structure1 = [[st1], 'foo', {'y': [st2]}]
    flat = [st2.indices, st2.values, st2.dense_shape, 'bar',
            st1.indices, st1.values, st1.dense_shape]
    result = nest.pack_sequence_as(structure1, flat, expand_composites=True)
    expected = [[st2], 'bar', {'y': [st1]}]
    self.assertNestEqual(expected, result)

  def testAssertSameStructure(self):
    st1 = sparse_tensor.SparseTensor([[0]], [0], [100])
    st2 = sparse_tensor.SparseTensor([[0, 3]], ['x'], [100, 100])
    test = TestCompositeTensor(st1.indices, st1.values, st1.dense_shape)
    nest.assert_same_structure(st1, st2, expand_composites=False)
    nest.assert_same_structure(st1, st2, expand_composites=True)
    nest.assert_same_structure(st1, test, expand_composites=False)
    with self.assertRaises(TypeError):
      nest.assert_same_structure(st1, test, expand_composites=True)


if __name__ == '__main__':
  googletest.main()
