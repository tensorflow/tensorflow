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


class SimpleCompositeTensor(composite_tensor.CompositeTensor):

  def __init__(self, *components, **kwargs):
    self._components = components
    self._metadata = kwargs.get('metadata', None)

  def _to_components(self):
    return self._components

  def _component_metadata(self):
    return self._metadata

  @classmethod
  def _from_components(cls, components, metadata):
    return cls(metadata=metadata, *components)

  def _shape_invariant_to_components(self, shape=None):
    raise NotImplementedError('CompositeTensor._shape_invariant_to_components')

  def _is_graph_tensor(self):
    return False

  def __repr__(self):
    return 'SimpleCompositeTensor%r' % (self._components,)

  def __eq__(self, other):
    return (isinstance(other, SimpleCompositeTensor) and
            self._components == other._components and
            self._metadata == other._metadata)


@test_util.run_all_in_graph_and_eager_modes
class CompositeTensorTest(test_util.TensorFlowTestCase):

  def assertNestEqual(self, a, b):
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
    elif isinstance(a, composite_tensor.CompositeTensor):
      self.assertIsInstance(b, composite_tensor.CompositeTensor)
      self.assertNestEqual(a._to_components(), b._to_components())
    else:
      self.assertAllEqual(a, b)

  def testNestFlatten(self):
    st1 = sparse_tensor.SparseTensor([[0, 3], [7, 2]], [1, 2], [10, 10])
    st2 = sparse_tensor.SparseTensor([[1, 2, 3]], ['a'], [10, 10, 10])
    structure = [[st1], 'foo', {'y': [st2]}]
    x = nest.flatten(structure, expand_composites=True)
    self.assertNestEqual(x, [
        st1.indices, st1.values, st1.dense_shape, 'foo', st2.indices,
        st2.values, st2.dense_shape
    ])

  def testNestPackSequenceAs(self):
    st1 = sparse_tensor.SparseTensor([[0, 3], [7, 2]], [1, 2], [10, 10])
    st2 = sparse_tensor.SparseTensor([[1, 2, 3]], ['a'], [10, 10, 10])
    structure1 = [[st1], 'foo', {'y': [st2]}]
    flat = [
        st2.indices, st2.values, st2.dense_shape, 'bar', st1.indices,
        st1.values, st1.dense_shape
    ]
    result = nest.pack_sequence_as(structure1, flat, expand_composites=True)
    expected = [[st2], 'bar', {'y': [st1]}]
    self.assertNestEqual(expected, result)

  def testNestAssertSameStructure(self):
    st1 = sparse_tensor.SparseTensor([[0]], [0], [100])
    st2 = sparse_tensor.SparseTensor([[0, 3]], ['x'], [100, 100])
    test = SimpleCompositeTensor(st1.indices, st1.values, st1.dense_shape)
    nest.assert_same_structure(st1, st2, expand_composites=False)
    nest.assert_same_structure(st1, st2, expand_composites=True)
    nest.assert_same_structure(st1, test, expand_composites=False)
    with self.assertRaises(TypeError):
      nest.assert_same_structure(st1, test, expand_composites=True)

  def testNestAssertSameStructureWithMetadata(self):
    ct1 = SimpleCompositeTensor([1, 2, 3], metadata=(100,))
    ct2 = SimpleCompositeTensor([4, 5, 6], metadata=(100,))
    ct3 = SimpleCompositeTensor([7, 8, 9], metadata=(200,))

    nest.assert_same_structure(ct1, ct2, expand_composites=True)
    with self.assertRaisesRegexp(ValueError,
                                 'CompositeTensors have different metadata'):
      nest.assert_same_structure(ct1, ct3, expand_composites=True)

  def testNestMapStructure(self):
    structure = [[SimpleCompositeTensor(1, 2, 3)], 100, {
        'y': SimpleCompositeTensor(SimpleCompositeTensor(4, 5), 6)
    }]

    def func(x):
      return x + 10

    result = nest.map_structure(func, structure, expand_composites=True)
    expected = [[SimpleCompositeTensor(11, 12, 13)], 110, {
        'y': SimpleCompositeTensor(SimpleCompositeTensor(14, 15), 16)
    }]
    self.assertEqual(result, expected)

  def testNestMapStructureWithPaths(self):
    structure = [[SimpleCompositeTensor(1, 2, 3)], 100, {
        'y': SimpleCompositeTensor(SimpleCompositeTensor(4, 5), 6)
    }]

    def func(path, x):
      return '%s:%s' % (path, x)

    result = nest.map_structure_with_paths(
        func, structure, expand_composites=True)
    expected = [[SimpleCompositeTensor('0/0/0:1', '0/0/1:2', '0/0/2:3')],
                '1:100', {
                    'y':
                        SimpleCompositeTensor(
                            SimpleCompositeTensor('2/y/0/0:4', '2/y/0/1:5'),
                            '2/y/1:6')
                }]
    self.assertEqual(result, expected)

  def testNestMapStructureWithTuplePaths(self):
    structure = [[SimpleCompositeTensor(1, 2, 3)], 100, {
        'y': SimpleCompositeTensor(SimpleCompositeTensor(4, 5), 6)
    }]

    def func(path, x):
      return (path, x)

    result = nest.map_structure_with_tuple_paths(
        func, structure, expand_composites=True)
    expected = [[
        SimpleCompositeTensor(((0, 0, 0), 1), ((0, 0, 1), 2), ((0, 0, 2), 3))
    ], ((1,), 100), {
        'y':
            SimpleCompositeTensor(
                SimpleCompositeTensor(((2, 'y', 0, 0), 4), ((2, 'y', 0, 1), 5)),
                ((2, 'y', 1), 6))
    }]
    self.assertEqual(result, expected)

  def testNestAssertShallowStructure(self):
    s1 = [[SimpleCompositeTensor(1, 2, 3)], 100, {
        'y': SimpleCompositeTensor(5, 6)
    }]
    s2 = [[SimpleCompositeTensor(1, 2, 3)], 100, {
        'y': SimpleCompositeTensor(SimpleCompositeTensor(4, 5), 6)
    }]
    nest.assert_shallow_structure(s1, s2, expand_composites=False)
    nest.assert_shallow_structure(s1, s2, expand_composites=True)
    nest.assert_shallow_structure(s2, s1, expand_composites=False)
    with self.assertRaises(TypeError):
      nest.assert_shallow_structure(s2, s1, expand_composites=True)

  def testNestFlattenUpTo(self):
    s1 = [[SimpleCompositeTensor(1, 2, 3)], 100, {
        'y': SimpleCompositeTensor(5, 6)
    }]
    s2 = [[SimpleCompositeTensor(1, 2, 3)], 100, {
        'y': SimpleCompositeTensor(SimpleCompositeTensor(4, 5), 6)
    }]
    result1 = nest.flatten_up_to(s1, s2, expand_composites=True)
    expected1 = [1, 2, 3, 100, SimpleCompositeTensor(4, 5), 6]
    self.assertEqual(result1, expected1)

    result2 = nest.flatten_up_to(s1, s2, expand_composites=False)
    expected2 = [
        SimpleCompositeTensor(1, 2, 3), 100,
        SimpleCompositeTensor(SimpleCompositeTensor(4, 5), 6)
    ]
    self.assertEqual(result2, expected2)

  def testNestFlattenWithTuplePathsUpTo(self):
    s1 = [[SimpleCompositeTensor(1, 2, 3)], 100, {
        'y': SimpleCompositeTensor(5, 6)
    }]
    s2 = [[SimpleCompositeTensor(1, 2, 3)], 100, {
        'y': SimpleCompositeTensor(SimpleCompositeTensor(4, 5), 6)
    }]
    result1 = nest.flatten_with_tuple_paths_up_to(
        s1, s2, expand_composites=True)
    expected1 = [((0, 0, 0), 1), ((0, 0, 1), 2), ((0, 0, 2), 3), ((1,), 100),
                 ((2, 'y', 0), SimpleCompositeTensor(4, 5)), ((2, 'y', 1), 6)]
    self.assertEqual(result1, expected1)

    result2 = nest.flatten_with_tuple_paths_up_to(
        s1, s2, expand_composites=False)
    expected2 = [((0, 0), SimpleCompositeTensor(1, 2, 3)), ((1,), 100),
                 ((2, 'y'),
                  SimpleCompositeTensor(SimpleCompositeTensor(4, 5), 6))]
    self.assertEqual(result2, expected2)

  def testNestMapStructureUpTo(self):
    s1 = [[SimpleCompositeTensor(1, 2, 3)], 100, {
        'y': SimpleCompositeTensor(5, 6)
    }]
    s2 = [[SimpleCompositeTensor(1, 2, 3)], 100, {
        'y': SimpleCompositeTensor(SimpleCompositeTensor(4, 5), 6)
    }]

    def func(x):
      return x + 10 if isinstance(x, int) else x

    result = nest.map_structure_up_to(s1, func, s2, expand_composites=True)
    expected = [[SimpleCompositeTensor(11, 12, 13)], 110, {
        'y': SimpleCompositeTensor(SimpleCompositeTensor(4, 5), 16)
    }]
    self.assertEqual(result, expected)

  def testNestMapStructureWithTuplePathsUpTo(self):
    s1 = [[SimpleCompositeTensor(1, 2, 3)], 100, {
        'y': SimpleCompositeTensor(5, 6)
    }]
    s2 = [[SimpleCompositeTensor(1, 2, 3)], 100, {
        'y': SimpleCompositeTensor(SimpleCompositeTensor(4, 5), 6)
    }]

    def func(path, x):
      return (path, x)

    result = nest.map_structure_with_tuple_paths_up_to(
        s1, func, s2, expand_composites=True)
    expected = [[
        SimpleCompositeTensor(((0, 0, 0), 1), ((0, 0, 1), 2), ((0, 0, 2), 3))
    ], ((1,), 100), {
        'y':
            SimpleCompositeTensor(((2, 'y', 0), SimpleCompositeTensor(4, 5)),
                                  ((2, 'y', 1), 6))
    }]
    self.assertEqual(result, expected)

  def testNestGetTraverseShallowStructure(self):
    pass

  def testNestYieldFlatPaths(self):
    structure = [[SimpleCompositeTensor(1, 2, 3)], 100, {
        'y': SimpleCompositeTensor(SimpleCompositeTensor(4, 5), 6)
    }]
    result1 = list(nest.yield_flat_paths(structure, expand_composites=True))
    expected1 = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (1,), (2, 'y', 0, 0),
                 (2, 'y', 0, 1), (2, 'y', 1)]
    self.assertEqual(result1, expected1)

    result2 = list(nest.yield_flat_paths(structure, expand_composites=False))
    expected2 = [(0, 0), (1,), (2, 'y')]
    self.assertEqual(result2, expected2)

  def testNestFlattenWithJoinedStringPaths(self):
    structure = [[SimpleCompositeTensor(1, 2, 3)], 100, {
        'y': SimpleCompositeTensor(SimpleCompositeTensor(4, 5), 6)
    }]
    result1 = nest.flatten_with_joined_string_paths(
        structure, expand_composites=True)
    expected1 = [('0/0/0', 1), ('0/0/1', 2), ('0/0/2', 3), ('1', 100),
                 ('2/y/0/0', 4), ('2/y/0/1', 5), ('2/y/1', 6)]
    self.assertEqual(result1, expected1)

    result2 = nest.flatten_with_joined_string_paths(
        structure, expand_composites=False)
    expected2 = [('0/0', SimpleCompositeTensor(1, 2, 3)), ('1', 100),
                 ('2/y', SimpleCompositeTensor(SimpleCompositeTensor(4, 5), 6))]
    self.assertEqual(result2, expected2)

  def testNestFlattenWithTuplePaths(self):
    structure = [[SimpleCompositeTensor(1, 2, 3)], 100, {
        'y': SimpleCompositeTensor(SimpleCompositeTensor(4, 5), 6)
    }]
    result1 = nest.flatten_with_tuple_paths(structure, expand_composites=True)
    expected1 = [((0, 0, 0), 1), ((0, 0, 1), 2), ((0, 0, 2), 3), ((1,), 100),
                 ((2, 'y', 0, 0), 4), ((2, 'y', 0, 1), 5), ((2, 'y', 1), 6)]
    self.assertEqual(result1, expected1)

    result2 = nest.flatten_with_tuple_paths(structure, expand_composites=False)
    expected2 = [((0, 0), SimpleCompositeTensor(1, 2, 3)), ((1,), 100),
                 ((2, 'y'),
                  SimpleCompositeTensor(SimpleCompositeTensor(4, 5), 6))]
    self.assertEqual(result2, expected2)


if __name__ == '__main__':
  googletest.main()
