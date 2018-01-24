# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator
import re
import textwrap

import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin

from tensorflow.contrib.labeled_tensor.python.ops import _typecheck as tc
from tensorflow.contrib.labeled_tensor.python.ops import core
from tensorflow.contrib.labeled_tensor.python.ops import test_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test as test_lib


class AxisTest(test_lib.TestCase):

  def setUp(self):
    d_7 = tensor_shape.Dimension(7)
    p_rgb = ['red', 'green', 'blue']

    self.i_7 = core.Axis('7', d_7)
    self.i_7p = core.Axis('7prime', d_7)
    self.i_rgb = core.Axis('rgb', p_rgb)
    self.i_range = core.Axis('range', range(7))
    self.i_unknown = core.Axis('unknown', None)

  def test_equality(self):

    axes = [self.i_7, self.i_7p, self.i_rgb, self.i_range, self.i_unknown]
    for i, axis_0 in enumerate(axes):
      for j, axis_1 in enumerate(axes):
        if i == j:
          self.assertEqual(axis_0, axis_1)
        else:
          self.assertNotEqual(axis_0, axis_1)

  def test_axis_value(self):
    self.assertEqual(self.i_7.value, tensor_shape.Dimension(7))
    self.assertTrue(self.i_range.value == tuple(range(7)))

  def test_axis_input(self):
    axes = [self.i_7, self.i_7p, self.i_rgb, self.i_range, self.i_unknown]
    for axis in axes:
      self.assertEqual(axis, core.Axis(axis.name, axis.value))

  def test_axis_value_input(self):
    axis = self.i_range
    for value in [range(7), list(range(7)), np.arange(7)]:
      self.assertEqual(axis, core.Axis(axis.name, value))

  def test_size(self):
    self.assertEqual(len(self.i_7), 7)
    self.assertEqual(len(self.i_rgb), 3)
    self.assertEqual(len(self.i_range), 7)
    self.assertEqual(self.i_unknown.size, None)

  def test_concat_single(self):
    red = core.Axis('rgb', ['red'])

    self.assertEqual(core.concat_axes([red]), red)

  def test_concat_many(self):
    red = core.Axis('rgb', ['red'])
    green = core.Axis('rgb', ['green'])
    blue = core.Axis('rgb', ['blue'])
    red_green_blue = core.Axis('rgb', ['red', 'green', 'blue'])

    self.assertEqual(core.concat_axes([red, green, blue]), red_green_blue)

  def test_concat_different_names(self):
    red = core.Axis('red', ['red'])
    green = core.Axis('green', ['red'])
    with self.assertRaises(ValueError):
      core.concat_axes([red, green])

  def test_concat_unknown(self):
    red = core.Axis('rgb', None)
    green = core.Axis('rgb', None)
    self.assertEqual(core.concat_axes([red, green]), red)

  def test_repr(self):
    self.assertEqual("Axis('7', Dimension(7))", repr(self.i_7))

  def test_invalid_input(self):
    with self.assertRaises(TypeError):
      core.Axis('foo', [{}])
    with self.assertRaises(ValueError):
      core.Axis('foo', [1, 2, 3, 1])
    red = core.Axis('foo', ['red'])
    with self.assertRaises(tc.Error):
      core.concat_axes([red, 1])

  def test_as_axis(self):
    self.assertEqual(self.i_7, core.as_axis(('7', 7)))
    self.assertEqual(self.i_7, core.as_axis(self.i_7))


class AxesTest(test_lib.TestCase):

  def setUp(self):
    d_7 = tensor_shape.Dimension(7)
    d_8 = tensor_shape.Dimension(8)
    p_rgb = ['red', 'green', 'blue']
    p_range = range(7)

    self.i_8 = core.Axis('8', d_8)

    self.a0 = core.Axes([('d7', d_7)])
    self.a1 = core.Axes([('d7', d_7)])
    self.a2 = core.Axes([('d7', d_7), ('rgb', p_rgb)])
    self.a3 = core.Axes([('8', d_8), ('range', p_range)])

  def test_equality(self):
    self.assertEqual(self.a0, self.a0)
    self.assertEqual(self.a0, self.a1)
    self.assertNotEqual(self.a0, self.a2)

  def test_repr(self):
    self.assertEqual("Axes([('d7', Dimension(7))])", repr(self.a0))

  def test_remove(self):
    a = self.a3.remove('range')
    self.assertEqual(a, core.Axes([self.i_8]))
    with self.assertRaises(KeyError):
      self.a3.remove('foobar')

  def test_typecheck_error_message(self):
    pattern = ('List(Union(labeled_tensor.Axis, Tuple(..., '
               'Union(Union(numpy.ndarray, %s, list, tuple), '
               'Optional(Union(tensorflow.Dimension, int))))))' %
               range.__name__)
    regexp = re.escape(pattern).replace(re.escape('...'), '.*')
    with self.assertRaisesRegexp(tc.Error, 'allowed type ' + regexp):
      core.Axes(None)


class LabeledTensorTest(test_util.Base):

  def setUp(self):
    tensor = array_ops.ones([7, 3, 8, 1])
    a0 = ('x', range(7))
    a1 = ('channel', ['red', 'green', 'blue'])
    a2 = ('y', 8)
    a3 = ('z', tensor_shape.Dimension(1))

    self.lt = core.LabeledTensor(tensor, [a0, a1, a2, a3])

  def test_repr(self):
    pattern = textwrap.dedent("""\
    <LabeledTensor '...' shape=(7, 3, 8, 1) dtype=float32
     axes=[('x', ...),
           ('channel', ...),
           ('y', Dimension(8)),
           ('z', Dimension(1))]>""")
    regexp = re.escape(pattern).replace(re.escape('...'), '.*')
    self.assertRegexpMatches(repr(self.lt), regexp)

  def test_reuse_existing_axes(self):
    alt_lt = core.LabeledTensor(self.lt.tensor, self.lt.axes)
    self.assertLabeledTensorsEqual(alt_lt, self.lt)

  def test_reuse_existing_axis_objects(self):
    alt_lt = core.LabeledTensor(self.lt.tensor, self.lt.axes.values())
    self.assertLabeledTensorsEqual(alt_lt, self.lt)

  def test_indexing_scalars(self):
    actual = self.lt[:, :, :, 0]
    expected = core.LabeledTensor(self.lt.tensor[:, :, :, 0],
                                  list(self.lt.axes.values())[:-1])
    self.assertLabeledTensorsEqual(actual, expected)

    actual = self.lt[1, :, :, 0]
    expected = core.LabeledTensor(self.lt.tensor[1, :, :, 0],
                                  list(self.lt.axes.values())[1:-1])
    self.assertLabeledTensorsEqual(actual, expected)

    actual = self.lt[1, 2, :, 0]
    expected = core.LabeledTensor(self.lt.tensor[1, 2, :, 0],
                                  list(self.lt.axes.values())[2:-1])
    self.assertLabeledTensorsEqual(actual, expected)

  def test_indexing_1d(self):
    lt_1d = self.lt[1, 2, :, 0]
    actual = lt_1d[3]
    expected = core.LabeledTensor(lt_1d.tensor[3], [])
    self.assertLabeledTensorsEqual(actual, expected)

  def test_indexing_slices(self):
    actual = self.lt[:3, :, :, :]
    axes = [('x', range(3))] + list(self.lt.axes.values())[1:]
    expected = core.LabeledTensor(self.lt.tensor[:3, :, :, :], axes)
    self.assertLabeledTensorsEqual(actual, expected)

  def test_invalid_indexing(self):
    with self.assertRaises(ValueError):
      self.lt[0]  # pylint: disable=pointless-statement
    with self.assertRaises(ValueError):
      self.lt[:, :, :, :, 0]  # pylint: disable=pointless-statement

  def test_unknown_size(self):
    tensor = array_ops.placeholder(dtypes.string, [None])
    actual = core.LabeledTensor(tensor, ['x'])
    self.assertIsNone(actual.axes['x'].size)
    self.assertIsNone(actual.axes['x'].value.value)

  def test_eq(self):
    self.assertEqual(self.lt, self.lt)
    self.assertNotEqual(self.lt, self.lt.tensor)
    self.assertNotEqual(self.lt.tensor, self.lt)

  def test_hash(self):
    lt1 = self.lt
    lt2 = core.LabeledTensor(self.lt.tensor, self.lt.axes)
    self.assertEqual(lt1, lt2)
    self.assertEqual(hash(lt1), hash(lt2))

  def test_name(self):
    self.assertEqual(self.lt.name, self.lt.tensor.name)

  def test_dtype(self):
    self.assertEqual(self.lt.dtype, self.lt.tensor.dtype)

  def test_get_shape(self):
    self.assertEqual(self.lt.get_shape(), self.lt.tensor.get_shape())

  def test_convert_to_tensor(self):
    expected = self.lt.tensor
    actual = ops.convert_to_tensor(self.lt)
    self.assertIs(expected, actual)


class Base(test_util.Base):

  def setUp(self):
    self.x_size = 7
    self.channel_size = 3
    self.z_size = 4
    self.probs_size = 11

    tensor = math_ops.range(0, self.x_size * self.channel_size * self.z_size *
                            self.probs_size)
    tensor = array_ops.reshape(
        tensor, [self.x_size, self.channel_size, self.z_size, self.probs_size])
    a0 = ('x', range(self.x_size))
    a1 = ('channel', ['red', 'green', 'blue'])
    a2 = 'z'
    a3 = ('probs', np.linspace(0.0, 1.0, self.probs_size))

    self.tensor = tensor
    self.a0 = a0
    self.a1 = a1
    self.a2 = a2
    self.a3 = a3
    self.original_lt = core.LabeledTensor(tensor, [a0, a1, a2, a3])

    self.x_probs_lt = core.slice_function(self.original_lt,
                                          {'z': 0,
                                           'channel': 0})
    self.channel_probs_lt = core.slice_function(self.original_lt,
                                                {'x': 3,
                                                 'z': 0})


class IdentityTest(Base):

  def test_name(self):
    identity_lt = core.identity(self.original_lt)
    self.assertIn('lt_identity', identity_lt.name)


class SliceFunctionTest(Base):

  def test_name(self):
    select_lt = core.slice_function(self.original_lt, {'channel': 1})
    self.assertIn('lt_slice', select_lt.name)

  def test_scalar(self):
    select_lt = core.slice_function(self.original_lt, {'channel': 1})
    golden_lt = core.LabeledTensor(self.tensor[:, 1, :, :],
                                   [self.a0, self.a2, self.a3])

    self.assertLabeledTensorsEqual(select_lt, golden_lt)

  def test_slice(self):
    select_lt = core.slice_function(self.original_lt, {'channel': slice(0, 2)})

    a1_sliced = ('channel', ['red', 'green'])
    golden_lt = core.LabeledTensor(self.tensor[:, :2, :, :],
                                   [self.a0, a1_sliced, self.a2, self.a3])

    self.assertLabeledTensorsEqual(select_lt, golden_lt)

  def test_slices(self):
    select_lt = core.slice_function(
        self.original_lt, {'x': slice(1, 5),
                           'channel': slice(1, None)})

    a0_sliced = ('x', range(1, 5))
    a1_sliced = ('channel', ['green', 'blue'])
    golden_lt = core.LabeledTensor(self.tensor[1:5, 1:, :, :],
                                   [a0_sliced, a1_sliced, self.a2, self.a3])

    self.assertLabeledTensorsEqual(select_lt, golden_lt)

  def test_slice_unlabeled(self):
    select_lt = core.slice_function(self.original_lt, {'z': slice(1, 3)})

    a2_sliced = 'z'
    golden_lt = core.LabeledTensor(self.tensor[:, :, 1:3, :],
                                   [self.a0, self.a1, a2_sliced, self.a3])

    self.assertLabeledTensorsEqual(select_lt, golden_lt)

  def test_slice_unknown_shape(self):
    lt = core.LabeledTensor(
        array_ops.placeholder(dtypes.float32, [None, 1]), ['x', 'y'])
    sliced_lt = core.slice_function(lt, {'y': 0})
    self.assertEqual(list(sliced_lt.axes.values()), [lt.axes['x']])


class TransposeTest(Base):

  def test_name(self):
    transpose_lt = core.transpose(self.original_lt,
                                  self.original_lt.axes.keys())
    self.assertIn('lt_transpose', transpose_lt.name)

  def test_identity(self):
    transpose_lt = core.transpose(self.original_lt,
                                  self.original_lt.axes.keys())
    golden_lt = self.original_lt

    self.assertLabeledTensorsEqual(transpose_lt, golden_lt)

  def test(self):
    transpose_lt = core.transpose(self.original_lt,
                                  ['z', 'channel', 'x', 'probs'])
    golden_lt = core.LabeledTensor(
        array_ops.transpose(self.tensor, [2, 1, 0, 3]),
        [self.a2, self.a1, self.a0, self.a3])

    self.assertLabeledTensorsEqual(transpose_lt, golden_lt)

  def test_default_axis_order(self):
    transpose_lt = core.transpose(self.original_lt)
    golden_lt = core.LabeledTensor(
        array_ops.transpose(self.tensor, [3, 2, 1, 0]),
        list(reversed(list(self.original_lt.axes.values()))))

    self.assertLabeledTensorsEqual(transpose_lt, golden_lt)

  def test_invalid_input(self):
    with self.assertRaises(ValueError):
      core.transpose(self.original_lt, ['channel', 'x', 'probs'])
    with self.assertRaises(ValueError):
      core.transpose(self.original_lt, ['z', 'foo', 'x', 'probs'])


class ExpandDimsTest(Base):

  def test_name(self):
    expand_lt = core.expand_dims(self.original_lt, self.original_lt.axes.keys())
    self.assertIn('lt_expand', expand_lt.name)

  def test_identity(self):
    expand_lt = core.expand_dims(self.original_lt, self.original_lt.axes.keys())
    golden_lt = self.original_lt

    self.assertLabeledTensorsEqual(expand_lt, golden_lt)

  def test(self):
    expand_lt = core.expand_dims(
        self.original_lt, ['foo', 'x', 'bar', 'channel', 'z', 'probs', 'grok'])
    golden_lt = core.LabeledTensor(
        array_ops.reshape(self.tensor, [
            1, self.x_size, 1, self.channel_size, self.z_size, self.probs_size,
            1
        ]), ['foo', self.a0, 'bar', self.a1, self.a2, self.a3, 'grok'])

    self.assertLabeledTensorsEqual(expand_lt, golden_lt)

  def test_label(self):
    expand_lt = core.expand_dims(self.original_lt, [
        'x',
        'channel',
        ('foo', 'bar'),
        'z',
        'probs',
    ])
    golden_lt = core.LabeledTensor(
        array_ops.reshape(
            self.tensor,
            [self.x_size, self.channel_size, 1, self.z_size, self.probs_size]),
        [self.a0, self.a1, ('foo', ['bar']), self.a2, self.a3])

    self.assertLabeledTensorsEqual(expand_lt, golden_lt)

  def test_unknown_dimension(self):
    orig_lt = core.LabeledTensor(
        array_ops.placeholder(dtypes.float32, [None]), ['x'])
    expand_lt = core.expand_dims(orig_lt, ['x', 'y'])
    self.assertEqual(expand_lt.axes, core.Axes([('x', None), ('y', 1)]))

  def test_invalid_input(self):
    with self.assertRaises(core.AxisOrderError):
      core.expand_dims(self.original_lt,
                       ['foo', 'not_x', 'bar', 'channel', 'z', 'probs', 'grok'])
    with self.assertRaises(core.AxisOrderError):
      core.expand_dims(self.original_lt,
                       ['foo', 'z', 'bar', 'channel', 'x', 'probs', 'grok'])


class AxisOrderScopeTest(Base):

  def test(self):
    xyz = ['x', 'y', 'z']
    abc = ['a', 'b', 'c']

    self.assertIsNone(core.get_axis_order())

    with core.axis_order_scope(xyz):
      self.assertEqual(core.get_axis_order(), xyz)

      with core.axis_order_scope():
        self.assertIsNone(core.get_axis_order())

        with core.axis_order_scope(abc):
          self.assertEqual(core.get_axis_order(), abc)

        self.assertIsNone(core.get_axis_order())

      self.assertEqual(core.get_axis_order(), xyz)

    self.assertIsNone(core.get_axis_order())


class CheckAxisOrderTest(Base):

  def test_passes(self):
    axis_order = ['w', 'x', 'y', 'z']

    lt = core.LabeledTensor(array_ops.ones((1, 1, 1, 1)), axis_order)
    core.check_axis_order(lt, axis_order)

    lt = core.LabeledTensor(array_ops.ones((1, 1, 1)), axis_order[1:])
    core.check_axis_order(lt, axis_order)

    lt = core.LabeledTensor(array_ops.ones((1, 1, 1)), axis_order[:-1])
    core.check_axis_order(lt, axis_order)

  def test_invalid(self):
    axis_order = ['w', 'x', 'y', 'z']
    lt = core.LabeledTensor(array_ops.ones((1, 1, 1, 1)), axis_order)
    with self.assertRaises(core.AxisOrderError):
      core.check_axis_order(lt)
    with self.assertRaises(core.AxisOrderError):
      core.check_axis_order(lt, axis_order[:-1])
    with self.assertRaises(core.AxisOrderError):
      core.check_axis_order(lt, axis_order[::-1])

  def test_scope(self):
    axis_order = ['w', 'x', 'y', 'z']
    lt = core.LabeledTensor(array_ops.ones((1, 1, 1, 1)), axis_order)
    with core.axis_order_scope(axis_order):
      core.check_axis_order(lt)


class ImposeAxisOrderTest(Base):

  def test_identity(self):
    axis_order = ['w', 'x', 'y', 'z']
    lt = core.LabeledTensor(
        array_ops.reshape(math_ops.range(24), (1, 2, 3, 4)), axis_order)
    actual = core.impose_axis_order(lt, axis_order)
    self.assertLabeledTensorsEqual(lt, actual)

    lt = core.LabeledTensor(
        array_ops.reshape(math_ops.range(6), (1, 2, 3)), axis_order[:3])
    actual = core.impose_axis_order(lt, axis_order)
    self.assertLabeledTensorsEqual(lt, actual)

  def test_reverse(self):
    axis_order = ['w', 'x', 'y', 'z']

    lt = core.LabeledTensor(
        array_ops.reshape(math_ops.range(24), (1, 2, 3, 4)), axis_order)
    actual = core.impose_axis_order(lt, axis_order[::-1])
    expected = core.transpose(lt, axis_order[::-1])
    self.assertLabeledTensorsEqual(expected, actual)

    lt = core.LabeledTensor(
        array_ops.reshape(math_ops.range(6), (1, 2, 3)), axis_order[:3])
    actual = core.impose_axis_order(lt, axis_order[::-1])
    expected = core.transpose(lt, ['y', 'x', 'w'])
    self.assertLabeledTensorsEqual(expected, actual)

  def test_scope(self):
    axis_order = ['w', 'x', 'y', 'z']

    lt = core.LabeledTensor(
        array_ops.reshape(math_ops.range(24), (1, 2, 3, 4)), axis_order)
    expected = core.transpose(lt, axis_order[::-1])
    with core.axis_order_scope(axis_order[::-1]):
      actual = core.impose_axis_order(lt)
    self.assertLabeledTensorsEqual(expected, actual)

  def test_invalid(self):
    lt = core.LabeledTensor(
        array_ops.reshape(math_ops.range(2), (1, 2)), ['x', 'y'])
    with self.assertRaises(ValueError):
      core.impose_axis_order(lt)
    with self.assertRaises(ValueError):
      core.impose_axis_order(lt, ['x'])


class FindConsistentOrderingTest(Base):

  def test(self):
    cases = [
        ([], [], []),
        (['x'], [], ['x']),
        ([], ['x'], ['x']),
        (['x'], ['x'], ['x']),
        (['x'], ['y'], ['x', 'y']),
        (['y'], ['x'], ['y', 'x']),
        (['x', 'y'], ['x', 'y'], ['x', 'y']),
        (['x', 'y'], ['y', 'x'], None),
        (['x', 'y'], ['y', 'z'], ['x', 'y', 'z']),
        (['x', 'z'], ['y', 'z'], ['x', 'y', 'z']),
        (['x', 'y'], ['x', 'z'], ['x', 'y', 'z']),
        (['w', 'x'], ['y', 'z'], ['w', 'x', 'y', 'z']),
        (['x', 'y', 'z'], ['z', 'x'], None),
        (['x', 'y', 'z'], ['x'], ['x', 'y', 'z']),
        ([], ['x', 'y', 'z'], ['x', 'y', 'z']),
    ]
    for a, b, expected in cases:
      actual = core._find_consistent_ordering(a, b)
      msg = ('unexpected ordering between %r and %r:\nexpected: %r\nactual: %r'
             % (a, b, expected, actual))
      self.assertEqual(expected, actual, msg=msg)


class AlignTest(Base):

  def test_name(self):
    align_lt_0, align_lt_1, _ = core.align(self.original_lt, self.original_lt)
    self.assertIn('lt_align', align_lt_0.name)
    self.assertIn('/0', align_lt_0.name)
    self.assertIn('lt_align', align_lt_1.name)
    self.assertIn('/1', align_lt_1.name)

  def test_identical_shaped_inputs(self):
    offset_tensor = self.original_lt.tensor + 1
    offset_lt = core.LabeledTensor(offset_tensor, self.original_lt.axes)

    align_lt, align_offset_lt, broadcast_axes = core.align(self.original_lt,
                                                           offset_lt)

    self.assertLabeledTensorsEqual(align_lt, self.original_lt)
    self.assertLabeledTensorsEqual(align_offset_lt, offset_lt)
    self.assertEqual(broadcast_axes, self.original_lt.axes)

  def test_different_inputs(self):
    # The correct axis ordering is ['x', 'channel', 'probs'].
    align_x_probs_lt, align_channel_probs_lt, broadcast_axes = core.align(
        self.x_probs_lt, self.channel_probs_lt)

    x_probs_golden_lt = core.LabeledTensor(
        array_ops.reshape(self.x_probs_lt.tensor,
                          [self.x_size, 1, self.probs_size]),
        [self.a0, 'channel', self.a3])

    self.assertLabeledTensorsEqual(align_x_probs_lt, x_probs_golden_lt)

    channel_probs_golden_lt = core.LabeledTensor(
        array_ops.reshape(self.channel_probs_lt.tensor,
                          [1, self.channel_size, self.probs_size]),
        ['x', self.a1, self.a3])

    self.assertLabeledTensorsEqual(align_channel_probs_lt,
                                   channel_probs_golden_lt)

    self.assertEqual(broadcast_axes, core.Axes([self.a0, self.a1, self.a3]))

  def test_axis_order_scope(self):
    xz_lt = core.LabeledTensor(array_ops.ones((2, 3)), ['x', 'z'])
    yz_lt = core.LabeledTensor(array_ops.ones((4, 3)), ['y', 'z'])

    _, _, broadcast_axes = core.align(xz_lt, yz_lt)
    self.assertEqual(list(broadcast_axes.keys()), ['x', 'y', 'z'])

    _, _, broadcast_axes = core.align(yz_lt, xz_lt)
    self.assertEqual(list(broadcast_axes.keys()), ['y', 'x', 'z'])

    with core.axis_order_scope(['x', 'y', 'z']):
      _, _, broadcast_axes = core.align(yz_lt, xz_lt)
      self.assertEqual(list(broadcast_axes.keys()), ['x', 'y', 'z'])

    with core.axis_order_scope(['x', 'y']):
      with self.assertRaises(core.AxisOrderError):
        core.align(xz_lt, yz_lt)
      with self.assertRaises(core.AxisOrderError):
        core.align(yz_lt, xz_lt)

  def test_invalid_input(self):
    lt_0 = core.LabeledTensor(array_ops.zeros([5]), [('a', range(5))])
    lt_1 = core.LabeledTensor(array_ops.zeros([5]), [('a', range(1, 6))])
    with self.assertRaises(ValueError):
      core.align(lt_0, lt_1)


class ConvertToLabeledTensorTest(Base):

  # TODO(shoyer): Simplify these tests once we can reuse labeled tensors in
  # assertLabeledTensorsEqual.

  def test_labeled_tensor(self):
    actual = core.convert_to_labeled_tensor(self.original_lt)
    self.assertLabeledTensorsEqual(actual, self.original_lt)

  def test_python_scalar(self):
    actual = core.convert_to_labeled_tensor(42)
    golden_lt = core.LabeledTensor(ops.convert_to_tensor(42), [])
    self.assertLabeledTensorsEqual(actual, golden_lt)

  def test_numpy_array(self):
    actual = core.convert_to_labeled_tensor(np.array(42))
    golden_lt = core.LabeledTensor(ops.convert_to_tensor(42), [])
    self.assertLabeledTensorsEqual(actual, golden_lt)

  def test_tensor(self):
    actual = core.convert_to_labeled_tensor(constant_op.constant(42))
    golden_lt = core.LabeledTensor(ops.convert_to_tensor(42), [])
    self.assertLabeledTensorsEqual(actual, golden_lt)

  def test_invalid_input(self):
    with self.assertRaises(ValueError):
      core.convert_to_labeled_tensor(math_ops.range(5))
    with self.assertRaises(ValueError):
      core.convert_to_labeled_tensor(np.array([1, 2]))


class DocStringCheckMixin(object):
  # requires self.ops to be defined

  def test_function_docstring_and_name(self):
    for op_name, _, _, lt_op in self.ops:
      if lt_op is not None:
        self.assertIn('tf.%s' % op_name, lt_op.__doc__)
        self.assertEqual(op_name, lt_op.__name__)


class UnaryOpsTestsMixin(object):
  # requires self.ops and self.test_lt to be defined

  def test_core_op(self):
    for op_name, _, tf_op, lt_op in self.ops:
      if tf_op is not None:
        golden_lt = core.LabeledTensor(
            tf_op(self.test_lt.tensor), self.test_lt.axes)
        actual_lt = lt_op(self.test_lt)
        self.assertIn(op_name, actual_lt.name)
        self.assertLabeledTensorsEqual(golden_lt, actual_lt)

  def test_infix(self):
    for op_name, infix_op, _, _ in self.ops:
      if infix_op is not None:
        expected_lt = core.LabeledTensor(
            infix_op(self.test_lt.tensor), self.test_lt.axes)
        actual_lt = infix_op(self.test_lt)
        self.assertIn(op_name, actual_lt.name)
        self.assertLabeledTensorsEqual(expected_lt, actual_lt)


class CoreUnaryOpsTest(Base, DocStringCheckMixin, UnaryOpsTestsMixin):

  def setUp(self):
    super(CoreUnaryOpsTest, self).setUp()

    self.ops = [
        ('abs', operator.abs, math_ops.abs, core.abs_function),
        ('neg', operator.neg, math_ops.negative, core.neg),
        # TODO(shoyer): add unary + to core TensorFlow
        ('pos', None, None, None),
        ('sign', None, math_ops.sign, core.sign),
        ('reciprocal', None, math_ops.reciprocal, core.reciprocal),
        ('square', None, math_ops.square, core.square),
        ('round', None, math_ops.round, core.round_function),
        ('sqrt', None, math_ops.sqrt, core.sqrt),
        ('rsqrt', None, math_ops.rsqrt, core.rsqrt),
        ('log', None, math_ops.log, core.log),
        ('exp', None, math_ops.exp, core.exp),
        ('log', None, math_ops.log, core.log),
        ('ceil', None, math_ops.ceil, core.ceil),
        ('floor', None, math_ops.floor, core.floor),
        ('cos', None, math_ops.cos, core.cos),
        ('sin', None, math_ops.sin, core.sin),
        ('tan', None, math_ops.tan, core.tan),
        ('acos', None, math_ops.acos, core.acos),
        ('asin', None, math_ops.asin, core.asin),
        ('atan', None, math_ops.atan, core.atan),
        ('lgamma', None, math_ops.lgamma, core.lgamma),
        ('digamma', None, math_ops.digamma, core.digamma),
        ('erf', None, math_ops.erf, core.erf),
        ('erfc', None, math_ops.erfc, core.erfc),
        ('lgamma', None, math_ops.lgamma, core.lgamma),
    ]
    total_size = np.prod([v.size for v in self.original_lt.axes.values()])
    self.test_lt = core.LabeledTensor(
        math_ops.cast(self.original_lt, dtypes.float32) / total_size,
        self.original_lt.axes)


class LogicalNotTest(Base, DocStringCheckMixin, UnaryOpsTestsMixin):

  def setUp(self):
    super(LogicalNotTest, self).setUp()
    self.ops = [('logical_not', operator.invert, math_ops.logical_not,
                 core.logical_not),]
    self.test_lt = self.original_lt < 10


class BinaryOpsTestsMixin(object):
  # requires self.ops, self.test_lt_1, self.test_lt_2, self.test_lt_1_broadcast
  # and self.test_lt_2_broadcast to be defined

  def test_core_op(self):
    for op_name, _, tf_op, lt_op in self.ops:
      golden_tensor = tf_op(self.test_lt_1_broadcast, self.test_lt_2_broadcast)
      golden_lt = core.LabeledTensor(golden_tensor, self.broadcast_axes)
      actual_lt = lt_op(self.test_lt_1, self.test_lt_2)
      self.assertIn(op_name, actual_lt.name)
      self.assertLabeledTensorsEqual(golden_lt, actual_lt)

  def test_infix(self):
    for op_name, infix_op, _, lt_op in self.ops:
      if infix_op is not None:
        expected_lt = lt_op(self.test_lt_1, self.test_lt_2)
        actual_lt = infix_op(self.test_lt_1, self.test_lt_2)
        self.assertIn(op_name, actual_lt.name)
        self.assertLabeledTensorsEqual(expected_lt, actual_lt)


class CoreBinaryOpsTest(Base, DocStringCheckMixin, BinaryOpsTestsMixin):

  def setUp(self):
    super(CoreBinaryOpsTest, self).setUp()

    self.x_probs_broadcast_tensor = array_ops.reshape(
        self.x_probs_lt.tensor, [self.x_size, 1, self.probs_size])

    self.channel_probs_broadcast_tensor = array_ops.reshape(
        self.channel_probs_lt.tensor, [1, self.channel_size, self.probs_size])

    # == and != are not element-wise for tf.Tensor, so they shouldn't be
    # elementwise for LabeledTensor, either.
    self.ops = [
        ('add', operator.add, math_ops.add, core.add),
        ('sub', operator.sub, math_ops.subtract, core.sub),
        ('mul', operator.mul, math_ops.multiply, core.mul),
        ('div', operator.truediv, math_ops.div, core.div),
        ('mod', operator.mod, math_ops.mod, core.mod),
        ('pow', operator.pow, math_ops.pow, core.pow_function),
        ('equal', None, math_ops.equal, core.equal),
        ('less', operator.lt, math_ops.less, core.less),
        ('less_equal', operator.le, math_ops.less_equal, core.less_equal),
        ('not_equal', None, math_ops.not_equal, core.not_equal),
        ('greater', operator.gt, math_ops.greater, core.greater),
        ('greater_equal', operator.ge, math_ops.greater_equal,
         core.greater_equal),
    ]
    self.test_lt_1 = self.x_probs_lt
    self.test_lt_2 = self.channel_probs_lt
    self.test_lt_1_broadcast = self.x_probs_broadcast_tensor
    self.test_lt_2_broadcast = self.channel_probs_broadcast_tensor
    self.broadcast_axes = [self.a0, self.a1, self.a3]

  def test_reflexive(self):
    labeled_tensor = self.x_probs_lt + 1  # all elements must be >0 for division
    for op_name, infix_op, _, lt_op in self.ops:
      if infix_op is not None:
        expected_lt = lt_op(2, labeled_tensor)
        actual_lt = infix_op(2, labeled_tensor)
        # Python uses greater for the reflexive version of less (and vise-versa)
        if 'less' in op_name:
          op_name = op_name.replace('less', 'greater')
        elif 'greater' in op_name:
          op_name = op_name.replace('greater', 'less')
        self.assertIn(op_name, actual_lt.name)
        self.assertLabeledTensorsEqual(expected_lt, actual_lt)


class LogicalBinaryOpsTest(Base, DocStringCheckMixin, BinaryOpsTestsMixin):

  def setUp(self):
    super(LogicalBinaryOpsTest, self).setUp()

    self.ops = [
        ('logical_and', operator.and_, math_ops.logical_and, core.logical_and),
        ('logical_or', operator.or_, math_ops.logical_or, core.logical_or),
        ('logical_xor', operator.xor, math_ops.logical_xor, core.logical_xor),
    ]
    self.test_lt_1 = self.original_lt < 10
    self.test_lt_2 = self.original_lt < 5
    self.test_lt_1_broadcast = self.test_lt_1.tensor
    self.test_lt_2_broadcast = self.test_lt_2.tensor
    self.broadcast_axes = self.test_lt_1.axes


class FloatBinaryOpsTest(Base, DocStringCheckMixin, BinaryOpsTestsMixin):

  def setUp(self):
    super(FloatBinaryOpsTest, self).setUp()

    self.ops = [
        ('igamma', None, math_ops.igamma, core.igamma),
        ('igammac', None, math_ops.igammac, core.igammac),
        ('zeta', None, math_ops.zeta, core.zeta),
        ('polygamma', None, math_ops.polygamma, core.polygamma),
        ('maximum', None, math_ops.maximum, core.maximum),
        ('minimum', None, math_ops.minimum, core.minimum),
        ('squared_difference', None, math_ops.squared_difference,
         core.squared_difference),
    ]
    total_size = np.prod([v.size for v in self.original_lt.axes.values()])
    test_lt = core.LabeledTensor(
        math_ops.cast(self.original_lt, dtypes.float32) / total_size,
        self.original_lt.axes)
    self.test_lt_1 = test_lt
    self.test_lt_2 = 1.0 - test_lt
    self.test_lt_1_broadcast = self.test_lt_1.tensor
    self.test_lt_2_broadcast = self.test_lt_2.tensor
    self.broadcast_axes = self.test_lt_1.axes


if __name__ == '__main__':
  test_lib.main()
