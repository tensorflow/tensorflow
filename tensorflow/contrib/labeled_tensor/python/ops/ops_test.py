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

import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.labeled_tensor.python.ops import core
from tensorflow.contrib.labeled_tensor.python.ops import ops
from tensorflow.contrib.labeled_tensor.python.ops import test_util


class Base(test_util.Base):

  def setUp(self):
    super(Base, self).setUp()

    self.x_size = 7
    self.channel_size = 3
    self.z_size = 4
    self.probs_size = 11

    tensor = tf.range(0, self.x_size * self.channel_size * self.z_size *
                      self.probs_size)
    tensor = tf.reshape(tensor, [self.x_size, self.channel_size, self.z_size,
                                 self.probs_size])
    a0 = ('x', range(self.x_size))
    a1 = ('channel', ['red', 'green', 'blue'])
    a2 = 'z'
    a3 = ('probs', np.linspace(0.0, 1.0, self.probs_size))

    self.tensor = tensor
    self.a0 = a0
    self.a1 = a1
    self.a2 = a2
    self.a2_resolved = ('z', self.z_size)
    self.a3 = a3
    self.original_lt = core.LabeledTensor(tensor, [a0, a1, a2, a3])

    self.x_probs_lt = core.slice_function(self.original_lt, {'z': 0})
    self.x_probs_lt = ops.select(self.x_probs_lt, {'channel': 'red'})
    self.channel_probs_lt = core.slice_function(self.original_lt, {'x': 3,
                                                                   'z': 0})


class SelectTest(Base):

  def test_name(self):
    select_lt = ops.select(self.original_lt, {'channel': 'green'})
    self.assertIn('lt_select', select_lt.name)

  def test_scalar(self):
    select_lt = ops.select(self.original_lt, {'channel': 'green'})
    golden_lt = core.LabeledTensor(self.tensor[:, 1, :, :], [self.a0, self.a2,
                                                             self.a3])
    self.assertLabeledTensorsEqual(select_lt, golden_lt)

  def test_slice(self):
    select_lt = ops.select(self.original_lt, {'channel': slice('red', 'green')})
    a1_sliced = ('channel', ['red', 'green'])
    golden_lt = core.LabeledTensor(self.tensor[:, :2, :, :],
                                   [self.a0, a1_sliced, self.a2, self.a3])
    self.assertLabeledTensorsEqual(select_lt, golden_lt)

  def test_slices(self):
    select_lt = ops.select(self.original_lt, {'x': slice(1, 4),
                                              'channel': slice('green', None)})

    a0_sliced = ('x', range(1, 5))
    a1_sliced = ('channel', ['green', 'blue'])
    golden_lt = core.LabeledTensor(self.tensor[1:5, 1:, :, :],
                                   [a0_sliced, a1_sliced, self.a2, self.a3])
    self.assertLabeledTensorsEqual(select_lt, golden_lt)

  def test_list(self):
    select_lt = ops.select(self.original_lt, {'channel': ['red', 'green']})
    a1_sliced = ('channel', ['red', 'green'])
    golden_lt = core.LabeledTensor(self.tensor[:, :2, :, :],
                                   [self.a0, a1_sliced, self.a2, self.a3])
    self.assertLabeledTensorsEqual(select_lt, golden_lt)

  def test_list_one_item(self):
    select_lt = ops.select(self.original_lt, {'channel': ['red']})
    a1_sliced = ('channel', ['red'])
    golden_lt = core.LabeledTensor(self.tensor[:, :1, :, :],
                                   [self.a0, a1_sliced, self.a2, self.a3])
    self.assertLabeledTensorsEqual(select_lt, golden_lt)

  def test_list_zero_items(self):
    select_lt = ops.select(self.original_lt, {'channel': []})
    golden_lt = core.LabeledTensor(self.tensor[:, :0, :, :],
                                   [self.a0, 'channel', self.a2, self.a3])
    self.assertLabeledTensorsEqual(select_lt, golden_lt)

  def test_scalars(self):
    select_lt = ops.select(self.original_lt, {'x': 1, 'channel': 'green'})
    golden_lt = core.LabeledTensor(self.tensor[1, 1, :, :],
                                   [self.a2, self.a3])
    self.assertLabeledTensorsEqual(select_lt, golden_lt)

  def test_invalid_input(self):
    with self.assertRaises(ValueError):
      ops.select(self.original_lt, {'foo': 1})
    with self.assertRaises(ValueError):
      ops.select(self.original_lt, {'z': 1})
    with self.assertRaises(KeyError):
      ops.select(self.original_lt, {'channel': 'purple'})
    with self.assertRaises(KeyError):
      ops.select(self.original_lt, {'channel': ['red', 'purple']})
    with self.assertRaises(NotImplementedError):
      ops.select(self.original_lt, {'channel': ['red'], 'x': [1]})
    with self.assertRaises(NotImplementedError):
      ops.select(self.original_lt, {'channel': ['red'], 'x': 1})
    with self.assertRaises(NotImplementedError):
      ops.select(self.original_lt, {'channel': slice('red', 'green', 2)})


class ConcatTest(Base):

  def setUp(self):
    super(ConcatTest, self).setUp()

    self.red_lt = ops.select(self.original_lt, {'channel': ['red']})
    self.green_lt = ops.select(self.original_lt, {'channel': ['green']})
    self.blue_lt = ops.select(self.original_lt, {'channel': ['blue']})

  def test_name(self):
    concat_lt = ops.concat([self.red_lt, self.blue_lt], 'channel')
    self.assertIn('lt_concat', concat_lt.name)

  def test(self):
    concat_lt = ops.concat([self.red_lt, self.green_lt], 'channel')
    golden_lt = ops.select(self.original_lt, {'channel': ['red', 'green']})

    self.assertLabeledTensorsEqual(concat_lt, golden_lt)

  def test_transposed(self):
    green_transposed = core.transpose(self.green_lt,
                                      ['probs', 'channel', 'z', 'x'])
    with self.assertRaises(ValueError):
      ops.concat([self.red_lt, green_transposed], 'channel')

  def test_invalid_input(self):
    with self.assertRaises(ValueError):
      ops.concat([], 'channel')
    with self.assertRaises(ValueError):
      ops.concat([self.red_lt, self.red_lt], 'channel')
    with self.assertRaises(ValueError):
      ops.concat([self.red_lt, self.red_lt], 'foo')


class PackTest(Base):

  def test_name(self):
    pack_lt = ops.pack([self.original_lt, self.original_lt], 'batch')
    self.assertIn('lt_pack', pack_lt.name)

  def test(self):
    pack_lt = ops.pack([self.original_lt, self.original_lt], 'batch')
    golden_lt = core.LabeledTensor(
        tf.stack([self.original_lt.tensor, self.original_lt.tensor]),
        ['batch', self.a0, self.a1, self.a2, self.a3])

    self.assertLabeledTensorsEqual(pack_lt, golden_lt)

  def test_axis(self):
    pack_lt = ops.pack([self.original_lt, self.original_lt],
                       new_axis='batch',
                       axis_position=4)
    golden_lt = core.LabeledTensor(
        tf.stack(
            [self.original_lt.tensor, self.original_lt.tensor], axis=4),
        [self.a0, self.a1, self.a2, self.a3, 'batch'])

    self.assertLabeledTensorsEqual(pack_lt, golden_lt)

  def test_invalid_input(self):
    with self.assertRaises(ValueError):
      ops.pack([self.original_lt, self.original_lt], 'channel')


class UnpackTest(Base):

  def test_name(self):
    unpack_lts = ops.unpack(self.original_lt)
    for t in unpack_lts:
      self.assertIn('lt_unpack', t.name)

  def test(self):
    unpack_lt = ops.unpack(self.original_lt)[0]
    golden_lt = core.LabeledTensor(
        tf.unstack(self.original_lt.tensor)[0], [self.a1, self.a2, self.a3])

    self.assertLabeledTensorsEqual(unpack_lt, golden_lt)

  def test_axis(self):
    unpack_lt = ops.unpack(self.original_lt, axis_name='z')[0]
    golden_lt = core.LabeledTensor(
        tf.unstack(
            self.original_lt.tensor, axis=2)[0], [self.a0, self.a1, self.a3])

    self.assertLabeledTensorsEqual(unpack_lt, golden_lt)

  def test_invalid_input(self):
    with self.assertRaises(ValueError):
      ops.unpack(self.original_lt, axis_name='not_found')


class ReshapeTest(Base):

  def test_name(self):
    reshape_lt = ops.reshape(self.original_lt, ['channel'], ['foo'])
    self.assertIn('lt_reshape', reshape_lt.name)

  def test_identity(self):
    reshape_lt = ops.reshape(self.original_lt, self.original_lt.axes.keys(),
                             self.original_lt.axes.values())
    self.assertLabeledTensorsEqual(reshape_lt, self.original_lt)

  def test_known_size(self):
    new_dim_size = self.channel_size * self.z_size * self.probs_size
    reshape_lt = ops.reshape(self.original_lt, ['channel', 'z', 'probs'],
                             [('new_dim', new_dim_size)])
    golden_lt = core.LabeledTensor(
        tf.reshape(self.original_lt.tensor, [self.x_size, -1]),
        [self.original_lt.axes['x'], 'new_dim'])
    self.assertLabeledTensorsEqual(reshape_lt, golden_lt)

  def test_unknown_size(self):
    reshape_lt = ops.reshape(self.original_lt, ['channel', 'z', 'probs'],
                             ['new_dim'])
    golden_lt = core.LabeledTensor(
        tf.reshape(self.original_lt.tensor, [self.x_size, -1]),
        [self.original_lt.axes['x'], 'new_dim'])
    self.assertLabeledTensorsEqual(reshape_lt, golden_lt)

  def test_unknown_dimension(self):
    orig_lt = core.LabeledTensor(tf.placeholder(tf.float32, [None]), ['x'])
    reshape_lt = ops.reshape(orig_lt, ['x'], ['y', ('z', 1)])
    self.assertEqual(reshape_lt.axes, core.Axes([('y', None), ('z', 1)]))
    with self.test_session() as sess:
      result = sess.run(reshape_lt, feed_dict={orig_lt.tensor: [1, 2]})
      np.testing.assert_array_equal(result, [[1], [2]])

  def test_with_labels(self):
    new_dim_size = self.channel_size * self.z_size * self.probs_size
    reshape_lt = ops.reshape(self.original_lt, ['channel', 'z', 'probs'],
                             [('new_dim', range(new_dim_size))])
    golden_lt = core.LabeledTensor(
        tf.reshape(self.original_lt.tensor, [self.x_size, -1]),
        [self.original_lt.axes['x'], ('new_dim', range(new_dim_size))])
    self.assertLabeledTensorsEqual(reshape_lt, golden_lt)

  def test_invalid_input(self):
    with self.assertRaisesRegexp(ValueError, 'not contained in the set'):
      ops.reshape(self.original_lt, ['foo'], ['bar'])
    with self.assertRaisesRegexp(core.AxisOrderError,
                                 'not a slice of axis names'):
      ops.reshape(self.original_lt, ['probs', 'z'], ['bar'])
    with self.assertRaisesRegexp(ValueError, 'at most one axis in new_axes'):
      ops.reshape(self.original_lt, ['probs'], ['foo', 'bar'])


class RenameAxisTest(Base):

  def test_name(self):
    rename_axis_lt = ops.rename_axis(self.original_lt, 'channel', 'foo')
    self.assertIn('lt_rename_axis', rename_axis_lt.name)

  def test_identity(self):
    rename_axis_lt = ops.rename_axis(self.original_lt, 'channel', 'channel')
    self.assertLabeledTensorsEqual(rename_axis_lt, self.original_lt)

  def test_new_name(self):
    rename_axis_lt = ops.rename_axis(self.original_lt, 'channel', 'foo')
    expected_axes = [(name if name != 'channel' else 'foo', axis.value)
                     for name, axis in self.original_lt.axes.items()]
    expected_lt = core.LabeledTensor(self.original_lt.tensor, expected_axes)
    self.assertLabeledTensorsEqual(rename_axis_lt, expected_lt)

  def test_invalid_input(self):
    with self.assertRaisesRegexp(ValueError, 'not contained in the set'):
      ops.rename_axis(self.original_lt, 'foo', 'bar')


class BatchTest(Base):

  def setUp(self):
    super(BatchTest, self).setUp()

    tensors = []
    for i in range(10):
      offset_lt = core.LabeledTensor(tf.constant(i), [])
      tensors.append(core.add(self.original_lt, offset_lt))
    self.pack_lt = ops.pack(tensors, 'batch')

  def test_name(self):
    batch_ops = ops.batch([self.pack_lt, self.pack_lt],
                          batch_size=2,
                          enqueue_many=True)
    for bo in batch_ops:
      self.assertIn('lt_batch', bo.name)

  def test_enqueue_many(self):
    [batch_2_op] = ops.batch([self.pack_lt], batch_size=2, enqueue_many=True)
    self.assertEqual(len(batch_2_op.axes['batch']), 2)

    [batch_10_op] = ops.batch([batch_2_op], batch_size=10, enqueue_many=True)

    self.assertLabeledTensorsEqual(self.pack_lt, batch_10_op)

  def test_no_enqueue_many(self):
    [batch_2_op] = ops.batch([self.original_lt], batch_size=2)
    self.assertEqual(len(batch_2_op.axes['batch']), 2)

    [batch_10_op] = ops.batch([batch_2_op], batch_size=10, enqueue_many=True)

    self.assertLabeledTensorsEqual(
        ops.pack(10 * [self.original_lt], 'batch'), batch_10_op)

  def test_invalid_input(self):
    with self.assertRaises(ValueError):
      ops.batch([self.original_lt], 3, enqueue_many=True)

  def test_allow_smaller_final_batch(self):
    [batch_2_op] = ops.batch([self.original_lt], batch_size=2,
                             allow_smaller_final_batch=True)
    self.assertEqual(batch_2_op.axes['batch'].size, None)


class ShuffleBatchTest(Base):

  def setUp(self):
    super(ShuffleBatchTest, self).setUp()

    tensors = []
    for i in range(10):
      offset_lt = core.LabeledTensor(tf.constant(i), [])
      tensors.append(core.add(self.original_lt, offset_lt))
    self.pack_lt = ops.pack(tensors, 'batch')

  def test_name(self):
    batch_lts = ops.shuffle_batch([self.pack_lt, self.pack_lt],
                                  batch_size=2,
                                  enqueue_many=True)
    for blt in batch_lts:
      self.assertIn('lt_shuffle_batch', blt.name)

  def test_enqueue_many(self):
    [batch_2_lt] = ops.shuffle_batch([self.pack_lt],
                                     batch_size=2,
                                     enqueue_many=True,
                                     min_after_dequeue=8,
                                     seed=0)
    self.assertEqual(len(batch_2_lt.axes['batch']), 2)

    [batch_10_lt] = ops.batch([batch_2_lt], batch_size=10, enqueue_many=True)

    self.assertEqual(batch_10_lt.axes, self.pack_lt.axes)
    [batch_10, pack] = self.eval([batch_10_lt.tensor, self.pack_lt.tensor])
    self.assertFalse((batch_10 == pack).all())

  def test_allow_smaller_final_batch(self):
    [batch_2_op] = ops.shuffle_batch([self.original_lt], batch_size=2,
                                     allow_smaller_final_batch=True)
    self.assertEqual(batch_2_op.axes['batch'].size, None)


class RandomCropTest(Base):

  def test_name(self):
    crop_lt = ops.random_crop(self.original_lt, {'probs': 3})
    self.assertIn('lt_random_crop', crop_lt.name)

  def test_single(self):
    crop_lt = ops.random_crop(self.original_lt, {'probs': 3})

    self.assertEqual(
        core.Axes([self.a0, self.a1, self.a2_resolved, ('probs', 3)]),
        crop_lt.axes)

  def test_double(self):
    crop_lt = ops.random_crop(self.original_lt, {'probs': 3, 'channel': 2})

    self.assertEqual(
        core.Axes([self.a0, ('channel', 2), self.a2_resolved, ('probs', 3)]),
        crop_lt.axes)

  def test_size1(self):
    crop_lt = ops.random_crop(self.original_lt, {'probs': 1})

    self.assertEqual(
        core.Axes([self.a0, self.a1, self.a2_resolved, ('probs', 1)]),
        crop_lt.axes)

  def test_different_seeds(self):
    crop_0_lt = ops.random_crop(self.original_lt, {'probs': 3,
                                                   'channel': 2},
                                seed=0)
    crop_1_lt = ops.random_crop(self.original_lt, {'probs': 3,
                                                   'channel': 2},
                                seed=1)

    self.assertEqual(crop_0_lt.axes, crop_1_lt.axes)
    [crop_0, crop_1] = self.eval([crop_0_lt.tensor, crop_1_lt.tensor])
    self.assertFalse((crop_0 == crop_1).all())

  def test_identical_seeds(self):
    crop_0_lt = ops.random_crop(self.original_lt, {'probs': 3,
                                                   'channel': 2},
                                seed=0)
    crop_1_lt = ops.random_crop(self.original_lt, {'probs': 3,
                                                   'channel': 2},
                                seed=0)

    self.assertLabeledTensorsEqual(crop_0_lt, crop_1_lt)

  def test_crop_idempotent(self):
    crop_0_lt = ops.random_crop(self.original_lt, {'probs': 3,
                                                   'channel': 2},
                                seed=0)
    crop_1_lt = ops.random_crop(crop_0_lt, {'probs': 3, 'channel': 2}, seed=1)

    self.assertLabeledTensorsEqual(crop_0_lt, crop_1_lt)

  def test_invalid_input(self):
    with self.assertRaises(ValueError):
      ops.random_crop(self.original_lt, {'foobar': 2})


class MapFnTest(Base):

  def test_name(self):
    map_lt = ops.map_fn(core.identity, self.original_lt)
    self.assertIn('lt_map_fn', map_lt.name)

  def test_identity(self):
    map_lt = ops.map_fn(core.identity, self.original_lt)
    self.assertLabeledTensorsEqual(map_lt, self.original_lt)

  def test_callable_object(self):

    class Identity(object):

      def __call__(self, other):
        return other

    map_lt = ops.map_fn(Identity(), self.original_lt)
    self.assertLabeledTensorsEqual(map_lt, self.original_lt)

  def test_slice(self):
    map_lt = ops.map_fn(lambda t: core.slice_function(t, {'channel': 1}),
                        self.original_lt)
    slice_lt = core.slice_function(self.original_lt, {'channel': 1})
    self.assertLabeledTensorsEqual(map_lt, slice_lt)


class SqueezeTest(Base):

  def setUp(self):
    super(SqueezeTest, self).setUp()

    self.squeezable_lt = core.slice_function(self.original_lt,
                                             {'channel': slice(0, 1),
                                              'probs': slice(0, 1)})

  def test_name(self):
    squeeze_lt = ops.squeeze(self.squeezable_lt)
    self.assertIn('lt_squeeze', squeeze_lt.name)

  def test_none(self):
    none_lt = ops.squeeze(self.squeezable_lt, None)
    axes_lt = ops.squeeze(self.squeezable_lt, ['channel', 'probs'])
    self.assertLabeledTensorsEqual(none_lt, axes_lt)

  def test(self):
    squeeze_lt = ops.squeeze(self.squeezable_lt, ['probs'])
    golden_lt = core.slice_function(self.squeezable_lt, {'probs': 0})
    self.assertLabeledTensorsEqual(squeeze_lt, golden_lt)

  def test_invalid_input(self):
    with self.assertRaises(ValueError):
      ops.squeeze(self.original_lt, ['channel'])
    with self.assertRaises(ValueError):
      ops.squeeze(self.squeezable_lt, ['foo'])


class MatMulTest(Base):

  def test_name(self):
    x_lt = core.LabeledTensor(tf.ones((3,)), ['x'])
    matmul_lt = ops.matmul(x_lt, x_lt)
    self.assertIn('lt_matmul', matmul_lt.name)

  def test_vector_vector(self):
    x_lt = core.LabeledTensor(tf.range(3), ['x'])
    matmul_lt = ops.matmul(x_lt, x_lt)
    golden_lt = core.convert_to_labeled_tensor(5)
    self.assertLabeledTensorsEqual(matmul_lt, golden_lt)

  def test_matrix_vector(self):
    xy_lt = core.LabeledTensor(tf.reshape(tf.range(6), (2, 3)), ['x', 'y'])
    y_lt = core.LabeledTensor(tf.range(3), ['y'])

    matmul_lt = ops.matmul(xy_lt, y_lt)
    golden_lt = core.LabeledTensor(
        tf.matmul(xy_lt.tensor, tf.reshape(y_lt.tensor, (-1, 1)))[:, 0], ['x'])
    self.assertLabeledTensorsEqual(matmul_lt, golden_lt)

    matmul_lt = ops.matmul(y_lt, xy_lt)
    self.assertLabeledTensorsEqual(matmul_lt, golden_lt)

  def test_matrix_matrix(self):
    xy_lt = core.LabeledTensor(tf.reshape(tf.range(6), (2, 3)), ['x', 'y'])
    yz_lt = core.LabeledTensor(tf.reshape(tf.range(12), (3, 4)), ['y', 'z'])

    matmul_lt = ops.matmul(xy_lt, yz_lt)
    golden_lt = core.LabeledTensor(
        tf.matmul(xy_lt.tensor, yz_lt.tensor), ['x', 'z'])
    self.assertLabeledTensorsEqual(matmul_lt, golden_lt)

    transpose = lambda x: core.transpose(x, list(x.axes.keys())[::-1])

    matmul_lt = ops.matmul(xy_lt, transpose(yz_lt))
    self.assertLabeledTensorsEqual(matmul_lt, golden_lt)

    matmul_lt = ops.matmul(transpose(xy_lt), yz_lt)
    self.assertLabeledTensorsEqual(matmul_lt, golden_lt)

    matmul_lt = ops.matmul(transpose(xy_lt), transpose(yz_lt))
    self.assertLabeledTensorsEqual(matmul_lt, golden_lt)

    matmul_lt = ops.matmul(yz_lt, xy_lt)
    self.assertLabeledTensorsEqual(matmul_lt, transpose(golden_lt))

  def test_matrix_matrix_axis_order(self):
    xy_lt = core.LabeledTensor(tf.reshape(tf.range(6), (2, 3)), ['x', 'y'])
    yz_lt = core.LabeledTensor(tf.reshape(tf.range(12), (3, 4)), ['y', 'z'])

    golden_lt = core.LabeledTensor(
        tf.matmul(xy_lt.tensor, yz_lt.tensor), ['x', 'z'])

    with core.axis_order_scope(['x', 'y', 'z']):

      matmul_lt = ops.matmul(xy_lt, yz_lt)
      self.assertLabeledTensorsEqual(matmul_lt, golden_lt)

      matmul_lt = ops.matmul(yz_lt, xy_lt)
      self.assertLabeledTensorsEqual(matmul_lt, golden_lt)

  def test_invalid(self):
    scalar_lt = core.LabeledTensor(tf.ones(()), [])
    x_lt = core.LabeledTensor(tf.ones((2,)), ['x'])
    x2_lt = core.LabeledTensor(tf.ones((3,)), ['x'])
    y_lt = core.LabeledTensor(tf.ones((3,)), ['y'])
    xy_lt = core.LabeledTensor(tf.ones((2, 3)), ['x', 'y'])
    xyz_lt = core.LabeledTensor(tf.ones((2, 3, 1)), ['x', 'y', 'z'])

    with self.assertRaisesRegexp(ValueError, 'inputs with at least rank'):
      ops.matmul(x_lt, scalar_lt)

    with self.assertRaises(NotImplementedError):
      ops.matmul(x_lt, xyz_lt)

    with self.assertRaisesRegexp(ValueError, 'exactly one axis in common'):
      ops.matmul(x_lt, y_lt)

    with self.assertRaises(NotImplementedError):
      ops.matmul(xy_lt, xy_lt)

    with self.assertRaisesRegexp(ValueError, 'does not match'):
      ops.matmul(x_lt, x2_lt)


class ReduceSumTest(Base):

  def test_name(self):
    sum_lt = ops.reduce_sum(self.original_lt, {'channel'})
    self.assertIn('lt_reduce_sum', sum_lt.name)

  def test_drop_axis(self):
    sum_lt = ops.reduce_sum(self.original_lt, {'channel'})
    golden_lt = core.LabeledTensor(
        tf.reduce_sum(self.original_lt.tensor, 1), [self.a0, self.a2, self.a3])
    self.assertLabeledTensorsEqual(sum_lt, golden_lt)

  def test_drop_scalar_axis(self):
    sum_lt = ops.reduce_sum(self.original_lt, 'channel')
    golden_lt = core.LabeledTensor(
        tf.reduce_sum(self.original_lt.tensor, 1), [self.a0, self.a2, self.a3])
    self.assertLabeledTensorsEqual(sum_lt, golden_lt)

  def test_keep_axis(self):
    sum_lt = ops.reduce_sum(self.original_lt, {('channel', 'hihowareyou')})
    golden_lt = core.LabeledTensor(
        tf.reduce_sum(self.original_lt.tensor,
                      1, keep_dims=True),
        [self.a0, ('channel', ['hihowareyou']), self.a2, self.a3])
    self.assertLabeledTensorsEqual(sum_lt, golden_lt)

  def test_keep_scalar_axis(self):
    sum_lt = ops.reduce_sum(self.original_lt, ('channel', 'hihowareyou'))
    golden_lt = core.LabeledTensor(
        tf.reduce_sum(self.original_lt.tensor,
                      1, keep_dims=True),
        [self.a0, ('channel', ['hihowareyou']), self.a2, self.a3])
    self.assertLabeledTensorsEqual(sum_lt, golden_lt)

  def test_scalar(self):
    scalar_lt = core.LabeledTensor(tf.constant(42), [])
    reduce_lt = ops.reduce_sum(scalar_lt, [])
    self.assertLabeledTensorsEqual(reduce_lt, scalar_lt)

  def test_empty_list(self):
    reduce_lt = ops.reduce_sum(self.original_lt, [])
    self.assertLabeledTensorsEqual(reduce_lt, self.original_lt)

  def test_none(self):
    sum_lt = ops.reduce_sum(self.original_lt)
    golden_lt = core.LabeledTensor(tf.reduce_sum(self.original_lt.tensor), [])
    self.assertLabeledTensorsEqual(sum_lt, golden_lt)

  def test_function_docstring_and_name(self):
    self.assertIn('tf.reduce_sum', ops.reduce_sum.__doc__)
    self.assertEqual('reduce_sum', ops.reduce_sum.__name__)


class ReduceMeanTest(Base):

  def test_name(self):
    actual_lt = ops.reduce_mean(self.original_lt, {'channel'})
    self.assertIn('lt_reduce_mean', actual_lt.name)

  def test(self):
    actual_lt = ops.reduce_mean(self.original_lt, {'channel'})
    golden_lt = core.LabeledTensor(
        tf.reduce_mean(self.original_lt.tensor, 1), [self.a0, self.a2, self.a3])
    self.assertLabeledTensorsEqual(actual_lt, golden_lt)


class ReduceProdTest(Base):

  def test_name(self):
    result_lt = ops.reduce_prod(self.original_lt, {'channel'})
    self.assertIn('lt_reduce_prod', result_lt.name)

  def test(self):
    result_lt = ops.reduce_prod(self.original_lt, {'channel'})
    golden_lt = core.LabeledTensor(
        tf.reduce_prod(self.original_lt.tensor, 1), [self.a0, self.a2, self.a3])
    self.assertLabeledTensorsEqual(result_lt, golden_lt)


class ReduceMinTest(Base):

  def test_name(self):
    result_lt = ops.reduce_min(self.original_lt, {'channel'})
    self.assertIn('lt_reduce_min', result_lt.name)

  def test(self):
    result_lt = ops.reduce_min(self.original_lt, {'channel'})
    golden_lt = core.LabeledTensor(
        tf.reduce_min(self.original_lt.tensor, 1), [self.a0, self.a2, self.a3])
    self.assertLabeledTensorsEqual(result_lt, golden_lt)


class ReduceMaxTest(Base):

  def test_name(self):
    result_lt = ops.reduce_max(self.original_lt, {'channel'})
    self.assertIn('lt_reduce_max', result_lt.name)

  def test(self):
    result_lt = ops.reduce_max(self.original_lt, {'channel'})
    golden_lt = core.LabeledTensor(
        tf.reduce_max(self.original_lt.tensor, 1), [self.a0, self.a2, self.a3])
    self.assertLabeledTensorsEqual(result_lt, golden_lt)


class BaseReduceBoolean(Base):

  def setUp(self):
    super(BaseReduceBoolean, self).setUp()
    self.bool_tensor = tf.cast(self.original_lt.tensor > 5, tf.bool)
    self.bool_lt = core.LabeledTensor(self.bool_tensor, self.original_lt.axes)


class ReduceAllTest(BaseReduceBoolean):

  def test_name(self):
    result_lt = ops.reduce_all(self.bool_lt, {'channel'})
    self.assertIn('lt_reduce_all', result_lt.name)

  def test(self):
    result_lt = ops.reduce_all(self.bool_lt, {'channel'})
    golden_lt = core.LabeledTensor(
        tf.reduce_all(self.bool_tensor, 1), [self.a0, self.a2, self.a3])
    self.assertLabeledTensorsEqual(result_lt, golden_lt)


class ReduceAnyTest(BaseReduceBoolean):

  def test_name(self):
    result_lt = ops.reduce_any(self.bool_lt, {'channel'})
    self.assertIn('lt_reduce_any', result_lt.name)

  def test(self):
    result_lt = ops.reduce_any(self.bool_lt, {'channel'})
    golden_lt = core.LabeledTensor(
        tf.reduce_any(self.bool_tensor, 1), [self.a0, self.a2, self.a3])
    self.assertLabeledTensorsEqual(result_lt, golden_lt)


class TileTest(Base):

  def test_name(self):
    tile_lt = ops.tile(self.original_lt, {'z': 2})
    self.assertIn('lt_tile', tile_lt.name)

  def test(self):
    for multiple in [2, tf.constant(2)]:
      tile_lt = ops.tile(self.original_lt, {'z': multiple})
      golden_op = tf.tile(self.original_lt.tensor, [1, 1, multiple, 1])
      golden_axes = ['z' if axis.name == 'z' else axis
                     for axis in self.original_lt.axes.values()]
      golden_lt = core.LabeledTensor(golden_op, golden_axes)
      self.assertLabeledTensorsEqual(tile_lt, golden_lt)

  def test_invalid_input(self):
    with self.assertRaisesRegexp(ValueError, 'are not contained in the set'):
      ops.tile(self.original_lt, {'foo': 5})
    with self.assertRaisesRegexp(ValueError, 'axes with tick labels'):
      ops.tile(self.original_lt, {'x': 5})


class PadTest(Base):

  def test_name(self):
    pad_lt = ops.pad(self.original_lt, {'x': (1, 1),
                                        'channel': ([], ['alpha'])})
    self.assertIn('lt_pad', pad_lt.name)

  def test(self):
    pad_lt = ops.pad(self.original_lt, {'x': (1, 1),
                                        'channel': ([], ['alpha'])})

    golden_op = tf.pad(self.original_lt.tensor, [[1, 1], [0, 1], [0, 0],
                                                 [0, 0]])
    golden_axes = [('x', self.x_size + 2),
                   ('channel', ['red', 'green', 'blue', 'alpha']), self.a2,
                   self.a3]
    golden_lt = core.LabeledTensor(golden_op, golden_axes)
    self.assertLabeledTensorsEqual(pad_lt, golden_lt)

  def test_invalid_input(self):
    with self.assertRaisesRegexp(ValueError, 'are not contained in the set'):
      ops.pad(self.original_lt, {'foo': (1, 1), 'channel': ([], ['alpha'])})


class ConstantTest(Base):

  def test_name(self):
    constant_lt = ops.constant(1)
    self.assertIn('lt_constant', constant_lt.name)

  def test_scalar(self):
    constant_lt = ops.constant(1)
    golden_lt = core.LabeledTensor(tf.constant(1), [])
    self.assertLabeledTensorsEqual(constant_lt, golden_lt)

  def test_infer_shape(self):
    constant_lt = ops.constant([1, 2], axes=['x'])
    golden_lt = core.LabeledTensor(tf.constant([1, 2]), ['x'])
    self.assertLabeledTensorsEqual(constant_lt, golden_lt)

  def test_specify_shape(self):
    constant_lt = ops.constant(1, axes=[('x', 3)])
    golden_lt = core.LabeledTensor(tf.constant(1, shape=(3,)), ['x'])
    self.assertLabeledTensorsEqual(constant_lt, golden_lt)

  def test_existing_axes(self):
    golden_lt = core.LabeledTensor(tf.constant([1, 2]), ['x'])
    constant_lt = ops.constant([1, 2], axes=golden_lt.axes)
    self.assertLabeledTensorsEqual(constant_lt, golden_lt)


class ZerosLikeTest(Base):

  def test_name(self):
    like_lt = ops.zeros_like(self.original_lt)
    self.assertIn('lt_zeros_like', like_lt.name)

  def test(self):
    like_lt = ops.zeros_like(self.original_lt)
    golden_lt = core.LabeledTensor(
        tf.zeros_like(self.original_lt.tensor), self.original_lt.axes)
    self.assertLabeledTensorsEqual(like_lt, golden_lt)


class OnesLikeTest(Base):

  def test_name(self):
    like_lt = ops.ones_like(self.original_lt)
    self.assertIn('lt_ones_like', like_lt.name)

  def test(self):
    like_lt = ops.ones_like(self.original_lt)
    golden_lt = core.LabeledTensor(
        tf.ones_like(self.original_lt.tensor), self.original_lt.axes)
    self.assertLabeledTensorsEqual(like_lt, golden_lt)


class CastTest(Base):

  def test_name(self):
    cast_lt = ops.cast(self.original_lt, tf.float16)
    self.assertIn('lt_cast', cast_lt.name)

  def test(self):
    cast_lt = ops.cast(self.original_lt, tf.float16)
    golden_lt = core.LabeledTensor(
        tf.cast(self.original_lt.tensor, tf.float16), self.original_lt.axes)
    self.assertLabeledTensorsEqual(cast_lt, golden_lt)


class VerifyTensorAllFiniteTest(Base):

  def setUp(self):
    super(VerifyTensorAllFiniteTest, self).setUp()

    self.finite_lt = core.LabeledTensor(tf.constant(42.0), [])
    self.nan_lt = core.LabeledTensor(tf.constant(np.nan), [])

    self.checked_finite_lt = ops.verify_tensor_all_finite(self.finite_lt, '')
    self.checked_nan_lt = ops.verify_tensor_all_finite(self.nan_lt, '')

  def test_name(self):
    self.assertIn('lt_verify_tensor_all_finite', self.checked_finite_lt.name)
    self.assertIn('lt_verify_tensor_all_finite', self.checked_nan_lt.name)

  def test_finite(self):
    self.assertLabeledTensorsEqual(self.finite_lt, self.checked_finite_lt)

  def test_nan(self):
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                 'Tensor had NaN values'):
      self.eval([self.checked_nan_lt])


class BooleanMaskTest(Base):

  def test_name(self):
    mask = core.LabeledTensor(tf.range(7) > 3, [self.a0])
    masked_lt = ops.boolean_mask(self.original_lt, mask)
    self.assertIn('lt_boolean_mask', masked_lt.name)

  def test(self):
    mask = core.LabeledTensor(tf.range(7) > 3, [self.a0])
    masked_lt = ops.boolean_mask(self.original_lt, mask)
    golden_lt = core.LabeledTensor(
        tf.boolean_mask(self.original_lt.tensor, mask.tensor),
        ['x', self.a1, self.a2, self.a3])
    self.assertLabeledTensorsEqual(masked_lt, golden_lt)

  def test_invalid_rank(self):
    mask = core.LabeledTensor(tf.ones((7, 3)) > 3, [self.a0, self.a1])
    with self.assertRaises(NotImplementedError):
      ops.boolean_mask(self.original_lt, mask)

  def test_mismatched_axis(self):
    mask = core.LabeledTensor(tf.range(7) > 3, ['foo'])
    with self.assertRaisesRegexp(ValueError, 'not equal'):
      ops.boolean_mask(self.original_lt, mask)


class WhereTest(Base):

  def test_name(self):
    condition = core.LabeledTensor(tf.range(5) < 3, ['x'])
    where_lt = ops.where(condition, condition, condition)
    self.assertIn('lt_where', where_lt.name)

  def test(self):
    condition = core.LabeledTensor(tf.range(5) < 3, ['x'])
    x = core.LabeledTensor(tf.ones(5), ['x'])
    y = core.LabeledTensor(tf.zeros(5), ['x'])
    where_lt = ops.where(condition, x, y)

    golden_lt = core.LabeledTensor(
        tf.concat_v2([tf.ones(3), tf.zeros(2)], 0), ['x'])
    self.assertLabeledTensorsEqual(where_lt, golden_lt)

  def test_mismatched_axes(self):
    condition = core.LabeledTensor(tf.range(5) < 3, ['x'])
    with self.assertRaisesRegexp(ValueError, 'equal axes'):
      ops.where(condition, condition[:3], condition)
    with self.assertRaisesRegexp(ValueError, 'equal axes'):
      ops.where(condition, condition, condition[:3])


if __name__ == '__main__':
  tf.test.main()
