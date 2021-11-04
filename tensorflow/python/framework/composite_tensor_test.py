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

import gc
import sys
import weakref

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest
from tensorflow.python.util import nest


class CTSpec(type_spec.TypeSpec):
  """A generic CompositeTensor TypeSpec, used for constructing tests."""

  def __init__(self, component_specs, metadata=None):
    self.component_specs = component_specs
    self.metadata = metadata

  value_type = property(lambda self: CT)
  _component_specs = property(lambda self: self.component_specs)

  def _serialize(self):
    return (self.component_specs, self.metadata)

  def _to_components(self, value):
    return value.components

  def _from_components(self, tensor_list):
    return CT(tensor_list, self.metadata)


class CT(composite_tensor.CompositeTensor):
  """A generic CompositeTensor, used for constructing tests."""
  _type_spec_class = CTSpec

  def __init__(self, components, metadata=None):
    if isinstance(components, list):
      components = tuple(components)
    self.components = components
    self.metadata = metadata

  @property
  def _type_spec(self):
    component_specs = nest.map_structure(type_spec.type_spec_from_value,
                                         self.components)
    return self._type_spec_class(component_specs, self.metadata)

  def __repr__(self):
    return '%s(%r, %r)' % (type(self).__name__, self.components, self.metadata)

  def __eq__(self, other):
    return (type(self) is type(other) and
            self.components == other.components and
            self.metadata == other.metadata)


# Another test CompositeTensor class.  `tf.nest` should treat different CT
# classes as different structure types (e.g. for assert_same_structure).
class CTSpec2(CTSpec):
  pass


class CT2(CT):
  _type_spec_class = CTSpec2


@test_util.run_all_in_graph_and_eager_modes
class CompositeTensorTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.parameters([
      {'structure': CT(0),
       'expected': [0],
       'paths': [('CT',)]},
      {'structure': CT('a'),
       'expected': ['a'],
       'paths': [('CT',)]},
      {'structure': CT(['a', 'b', 'c']),
       'expected': ['a', 'b', 'c'],
       'paths': [('CT', 0), ('CT', 1), ('CT', 2)]},
      {'structure': CT({'x': 'a', 'y': 'b', 'z': 'c'}),
       'expected': ['a', 'b', 'c'],
       'paths': [('CT', 'x'), ('CT', 'y'), ('CT', 'z')]},
      {'structure': [{'k1': CT('a')}, CT(['b', {'x': CT({'y': 'c'})}])],
       'expected': ['a', 'b', 'c'],
       'paths': [(0, 'k1', 'CT'), (1, 'CT', 0), (1, 'CT', 1, 'x', 'CT', 'y')]},
      {'structure': CT(0),
       'expand_composites': False,
       'expected': [CT(0)],
       'paths': [()]},
      {'structure': [{'k1': CT('a')}, CT(['b', {'x': CT({'y': 'c'})}])],
       'expand_composites': False,
       'expected': [CT('a'), CT(['b', {'x': CT({'y': 'c'})}])],
       'paths': [(0, 'k1'), (1,)]},
  ])  # pyformat: disable
  def testNestFlatten(self, structure, expected, paths, expand_composites=True):
    result = nest.flatten(structure, expand_composites=expand_composites)
    self.assertEqual(result, expected)

    result_with_paths = nest.flatten_with_tuple_paths(
        structure, expand_composites=expand_composites)
    self.assertEqual(result_with_paths, list(zip(paths, expected)))

    string_paths = ['/'.join(str(p) for p in path) for path in paths]  # pylint: disable=g-complex-comprehension
    result_with_string_paths = nest.flatten_with_joined_string_paths(
        structure, expand_composites=expand_composites)
    self.assertEqual(result_with_string_paths,
                     list(zip(string_paths, expected)))

    flat_paths_result = list(
        nest.yield_flat_paths(structure, expand_composites=expand_composites))
    self.assertEqual(flat_paths_result, paths)

  @parameterized.parameters([
      {'s1': [1, 2, 3],
       's2': [CT(['a', 'b']), 'c', 'd'],
       'expand_composites': False,
       'expected': [CT(['a', 'b']), 'c', 'd'],
       'paths': [(0,), (1,), (2,)]},
      {'s1': [CT([1, 2, 3])],
       's2': [5],
       'expand_composites': False,
       'expected': [5],
       'paths': [(0,)]},
      {'s1': [[CT([9, 9, 9])], 999, {'y': CT([9, 9])}],
       's2': [[CT([1, 2, 3])], 100, {'y': CT([CT([4, 5]), 6])}],
       'expand_composites': False,
       'expected': [CT([1, 2, 3]), 100, CT([CT([4, 5]), 6])],
       'paths': [(0, 0), (1,), (2, 'y')]},
      {'s1': [[CT([9, 9, 9])], 999, {'y': CT([CT([9, 9]), 9])}],
       's2': [[CT([1, 2, 3])], 100, {'y': CT([5, 6])}],
       'expand_composites': False,
       'expected': [CT([1, 2, 3]), 100, CT([5, 6])],
       'paths': [(0, 0), (1,), (2, 'y')]},
  ])  # pyformat: disable
  def testNestFlattenUpTo(self, s1, s2, expected, paths,
                          expand_composites=True):
    result = nest.flatten_up_to(s1, s2, expand_composites=expand_composites)
    self.assertEqual(expected, result)

    result_with_paths = nest.flatten_with_tuple_paths_up_to(
        s1, s2, expand_composites=expand_composites)
    self.assertEqual(result_with_paths, list(zip(paths, expected)))

  @parameterized.parameters([
      {'structure': CT(0),
       'sequence': [5],
       'expected': CT(5)},
      {'structure': CT(['a', 'b', 'c']),
       'sequence': ['A', CT(['b']), {'x': 'y'}],
       'expected': CT(['A', CT(['b']), {'x': 'y'}])},
      {'structure': [{'k1': CT('a')}, CT(['b', {'x': CT({'y': 'c'})}])],
       'sequence': ['A', 'B', 'C'],
       'expected': [{'k1': CT('A')}, CT(['B', {'x': CT({'y': 'C'})}])]},
      {'structure': [{'k1': CT('a')}, CT(['b', {'x': CT({'y': 'c'})}])],
       'sequence': ['A', 'B'],
       'expand_composites': False,
       'expected': [{'k1': 'A'}, 'B']},
      {'structure': CT(0, metadata='abc'),
       'sequence': [5],
       'expected': CT(5, metadata='abc')},
  ])  # pyformat: disable
  def testNestPackSequenceAs(self,
                             structure,
                             sequence,
                             expected,
                             expand_composites=True):
    result = nest.pack_sequence_as(
        structure, sequence, expand_composites=expand_composites)
    self.assertEqual(result, expected)

  @parameterized.parameters([
      {'s1': CT('abc'), 's2': CT('xyz')},
      {'s1': CT(['a', 'b', 'c']), 's2': CT(['d', 'e', 'f'])},
      {'s1': [1, CT([10]), CT(200, metadata='xyz')],
       's2': [8, CT([55]), CT(100, metadata='xyz')]},
  ])  # pyformat: disable
  def testNestAssertSameStructure(self, s1, s2, expand_composites=True):
    nest.assert_same_structure(s1, s2, expand_composites=expand_composites)
    nest.assert_shallow_structure(s1, s2, expand_composites=expand_composites)

  @parameterized.parameters([
      {'s1': CT(0), 's2': CT(['x'])},
      {'s1': CT([1]), 's2': CT([1, 2])},
      {'s1': CT({'x': 1}), 's2': CT({'y': 1})},
      {'s1': CT(0), 's2': CT(0, metadata='xyz')},
      {'s1': CT(0, metadata='xyz'), 's2': CT(0)},
      {'s1': CT(0, metadata='xyz'), 's2': CT(0, metadata='abc')},
      {'s1': CT(['a', 'b', 'c']), 's2': CT(['d', 'e'])},
      {'s1': [1, CT(['a']), CT('b', metadata='xyz')],
       's2': [8, CT([55, 66]), CT(100, metadata='abc')]},
      {'s1': CT(0), 's2': CT2(0), 'error': TypeError},
  ])  # pyformat: disable
  def testNestAssertSameStructureCompositeMismatch(self,
                                                   s1,
                                                   s2,
                                                   error=ValueError):
    # s1 and s2 have the same structure if expand_composites=False; but
    # different structures if expand_composites=True.
    nest.assert_same_structure(s1, s2, expand_composites=False)
    nest.assert_shallow_structure(s1, s2, expand_composites=False)
    with self.assertRaises(error):  # pylint: disable=g-error-prone-assert-raises
      nest.assert_same_structure(s1, s2, expand_composites=True)

  @parameterized.parameters([
      # Note: there are additional test cases in testNestAssertSameStructure.
      {'s1': [1], 's2': [CT(1)]},
      {'s1': [[CT([1, 2, 3])], 100, {'y': CT([5, 6])}],
       's2': [[CT([1, 2, 3])], 100, {'y': CT([CT([4, 5]), 6])}],
       'expand_composites': False},
      {'s1': [[CT([1, 2, 3])], 100, {'y': CT([CT([4, 5]), 6])}],
       's2': [[CT([1, 2, 3])], 100, {'y': CT([5, 6])}],
       'expand_composites': False},
  ])  # pyformat: disable
  def testNestAssertShallowStructure(self, s1, s2, expand_composites=True):
    nest.assert_shallow_structure(s1, s2, expand_composites=expand_composites)

  @parameterized.parameters([
      # Note: there are additional test cases in
      # testNestAssertSameStructureCompositeMismatch.
      {'s1': [[CT([1, 2, 3])], 100, {'y': CT([CT([4, 5]), 6])}],
       's2': [[CT([1, 2, 3])], 100, {'y': CT([5, 6])}]},
      {'s1': CT([1, 2, 3]),
       's2': [1, 2, 3],
       'check_types': False},
  ])  # pyformat: disable
  def testNestAssertShallowStructureCompositeMismatch(self,
                                                      s1,
                                                      s2,
                                                      check_types=True):
    with self.assertRaises((TypeError, ValueError)):  # pylint: disable=g-error-prone-assert-raises
      nest.assert_shallow_structure(
          s1, s2, expand_composites=True, check_types=check_types)

  @parameterized.parameters([
      {'structure': CT(1, metadata=2),
       'expected': CT(11, metadata=2)},
      {'structure': CT({'x': 1, 'y': [2, 3]}, metadata=2),
       'expected': CT({'x': 11, 'y': [12, 13]}, metadata=2)},
      {'structure': [[CT([1, 2, 3])], 100, {'y': CT([CT([4, 5]), 6])}],
       'expected': [[CT([11, 12, 13])], 110, {'y': CT([CT([14, 15]), 16])}]},
  ])  # pyformat: disable
  def testNestMapStructure(self, structure, expected, expand_composites=True):
    func = lambda x: x + 10
    result = nest.map_structure(
        func, structure, expand_composites=expand_composites)
    self.assertEqual(result, expected)

  @parameterized.parameters([
      {'s1': [[CT([1, 2, 3])], 100, {'y': 4}],
       's2': [[CT([1, 2, 3])], 100, {'y': CT([CT([4, 5]), 6])}],
       'expected': [[CT([11, 12, 13])], 110, {'y': CT([CT([4, 5]), 6])}]}
  ])  # pyformat: disable
  def testNestMapStructureUpTo(self, s1, s2, expected):
    func = lambda x: x + 10 if isinstance(x, int) else x
    result = nest.map_structure_up_to(s1, func, s2, expand_composites=True)
    self.assertEqual(result, expected)

  @parameterized.parameters([
      {'structure': CT('a'),
       'expected': CT('CT:a')},
      {'structure': CT(['a', 'b']),
       'expected': CT(['CT/0:a', 'CT/1:b'])},
      {'structure': [[CT([1, 2, 3])], 100, {'y': CT([CT([4, 5]), 6])}],
       'expected': [
           [CT(['0/0/CT/0:1', '0/0/CT/1:2', '0/0/CT/2:3'])],
           '1:100',
           {'y': CT([CT(['2/y/CT/0/CT/0:4', '2/y/CT/0/CT/1:5']),
                     '2/y/CT/1:6'])}]},
  ])  # pyformat: disable
  def testNestMapStructureWithPaths(self,
                                    structure,
                                    expected,
                                    expand_composites=True):

    def func1(path, x):
      return '%s:%s' % (path, x)

    result = nest.map_structure_with_paths(
        func1, structure, expand_composites=expand_composites)
    self.assertEqual(result, expected)

    # Use the same test cases for map_structure_with_tuple_paths.
    def func2(tuple_path, x):
      return '%s:%s' % ('/'.join(str(v) for v in tuple_path), x)

    result = nest.map_structure_with_tuple_paths(
        func2, structure, expand_composites=expand_composites)
    self.assertEqual(result, expected)

  @parameterized.parameters([
      {'s1': [[CT([1, 2, 3])], 100, {'y': [4, 5]}],
       's2': [[CT([1, 2, 3])], 100, {'y': [CT([4, 5]), 6]}],
       'expected': [
           [CT(['0/0/CT/0:1', '0/0/CT/1:2', '0/0/CT/2:3'])],
           ('1:100'),
           {'y': ['2/y/0:CT((4, 5), None)', '2/y/1:6']}]},
  ])  # pyformat: disable
  def testNestMapStructureWithTuplePathsUpTo(self, s1, s2, expected):

    def func(tuple_path, x):
      return '%s:%s' % ('/'.join(str(v) for v in tuple_path), x)

    result = nest.map_structure_with_tuple_paths_up_to(
        s1, func, s2, expand_composites=True)
    self.assertEqual(result, expected)

  def testNestGetTraverseShallowStructure(self):
    func = lambda t: not (isinstance(t, CT) and t.metadata == 'B')
    structure = [CT([1, 2], metadata='A'), CT([CT(3)], metadata='B')]

    result = nest.get_traverse_shallow_structure(
        func, structure, expand_composites=True)
    expected = [CT([True, True], metadata='A'), False]
    self.assertEqual(result, expected)

  def testMemoryIsFreed(self):
    # Note: we use `np.array` values for CT and `set` values for
    # metadata because we need to construct weakrefs to them.  Other builtin
    # types, such as `list` and `tuple`, do not support weakrefs.
    ct1 = CT(np.array([1, 2]), set(['no', 'leaks']))
    ct2 = CT(np.array([3, 4]), set(['no', 'leaks']))
    ct3 = CT(np.array([5, 6]), set(['other', 'metadata']))

    # Note: map_structure exercises flatten, pack_sequence_as, and
    # assert_same_structure.
    func = lambda x, y: x + y
    ct4 = nest.map_structure(func, ct1, ct2, expand_composites=True)

    # Check that the exception-raising path in assert_same_structure
    # doesn't leak any objects.
    with self.assertRaises(ValueError):
      nest.map_structure(func, ct2, ct3, expand_composites=True)
    if hasattr(sys, 'exc_clear'):
      sys.exc_clear()  # Remove any references in exception stack traces.

    refs = []
    for ct in [ct1, ct2, ct3, ct4]:
      refs.append(weakref.ref(ct))
      refs.append(weakref.ref(ct.components))
      refs.append(weakref.ref(ct.metadata))
    del ct  # pylint: disable=undefined-loop-variable

    for ref in refs:
      self.assertIsNotNone(ref())

    del ct1, ct2, ct3, ct4
    gc.collect()
    for ref in refs:
      self.assertIsNone(ref())

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters([
      ('IndexedSlicesNoDenseShape', lambda: indexed_slices.IndexedSlices(
          constant_op.constant([1, 2, 3]), constant_op.constant([2, 8, 4]))),
      ('IndexedSlicesInt32DenseShape', lambda: indexed_slices.IndexedSlices(
          constant_op.constant([1, 2, 3]), constant_op.constant([2, 8, 4]),
          constant_op.constant([10], dtypes.int32))),
      ('IndexedSlicesInt64DenseShape', lambda: indexed_slices.IndexedSlices(
          constant_op.constant([[1, 2], [3, 4]]), constant_op.constant([2, 8]),
          constant_op.constant([10, 2], dtypes.int64))),
      ('RaggedTensorRaggedRank1',
       lambda: ragged_factory_ops.constant([[1, 2], [3]])),
      ('RaggedTensorRaggedRank2',
       lambda: ragged_factory_ops.constant([[[1, 2], [3]], [[6, 7, 8]]])),
      ('SparseTensor',
       lambda: sparse_tensor.SparseTensor([[3], [7]], ['a', 'b'], [10])),
      ('Nested structure', lambda: {
          'a':
              indexed_slices.IndexedSlices(
                  constant_op.constant([1, 2, 3]),
                  constant_op.constant([2, 8, 4])),
          'b': [
              ragged_factory_ops.constant([[1, 2], [3]]),
              sparse_tensor.SparseTensor([[3], [7]], ['a', 'b'], [10])
          ]
      }),
  ])
  def testAssertSameStructureWithValueAndTypeSpec(self, value_func):
    value = value_func()
    spec = nest.map_structure(type_spec.type_spec_from_value, value,
                              expand_composites=False)
    nest.assert_same_structure(value, spec, expand_composites=True)


if __name__ == '__main__':
  googletest.main()
