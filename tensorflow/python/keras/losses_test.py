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
"""Tests for Keras loss functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.platform import test

try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None

ALL_LOSSES = [keras.losses.mean_squared_error,
              keras.losses.mean_absolute_error,
              keras.losses.mean_absolute_percentage_error,
              keras.losses.mean_squared_logarithmic_error,
              keras.losses.squared_hinge,
              keras.losses.hinge,
              keras.losses.categorical_crossentropy,
              keras.losses.binary_crossentropy,
              keras.losses.kullback_leibler_divergence,
              keras.losses.poisson,
              keras.losses.cosine_similarity,
              keras.losses.logcosh,
              keras.losses.categorical_hinge]


class _MSEMAELoss(object):
  """Loss function with internal state, for testing serialization code."""

  def __init__(self, mse_fraction):
    self.mse_fraction = mse_fraction

  def __call__(self, y_true, y_pred, sample_weight=None):
    return (self.mse_fraction * keras.losses.mse(y_true, y_pred) +
            (1 - self.mse_fraction) * keras.losses.mae(y_true, y_pred))

  def get_config(self):
    return {'mse_fraction': self.mse_fraction}


class KerasLossesTest(test.TestCase):

  def test_objective_shapes_3d(self):
    with self.cached_session():
      y_a = keras.backend.variable(np.random.random((5, 6, 7)))
      y_b = keras.backend.variable(np.random.random((5, 6, 7)))
      for obj in ALL_LOSSES:
        objective_output = obj(y_a, y_b)
        self.assertListEqual(objective_output.shape.as_list(), [5, 6])

  def test_objective_shapes_2d(self):
    with self.cached_session():
      y_a = keras.backend.variable(np.random.random((6, 7)))
      y_b = keras.backend.variable(np.random.random((6, 7)))
      for obj in ALL_LOSSES:
        objective_output = obj(y_a, y_b)
        self.assertListEqual(objective_output.shape.as_list(), [
            6,
        ])

  def test_cce_one_hot(self):
    with self.cached_session():
      y_a = keras.backend.variable(np.random.randint(0, 7, (5, 6)))
      y_b = keras.backend.variable(np.random.random((5, 6, 7)))
      objective_output = keras.losses.sparse_categorical_crossentropy(y_a, y_b)
      assert keras.backend.eval(objective_output).shape == (5, 6)

      y_a = keras.backend.variable(np.random.randint(0, 7, (6,)))
      y_b = keras.backend.variable(np.random.random((6, 7)))
      objective_output = keras.losses.sparse_categorical_crossentropy(y_a, y_b)
      assert keras.backend.eval(objective_output).shape == (6,)

  @test_util.run_in_graph_and_eager_modes
  def test_categorical_crossentropy_loss(self):
    target = keras.backend.variable(np.random.randint(0, 1, (5, 1)))
    logits = keras.backend.variable(np.random.random((5, 1)))
    softmax_output = keras.backend.softmax(logits)
    output_from_logit = keras.losses.categorical_crossentropy(
        target, logits, from_logits=True)
    output_from_softmax = keras.losses.categorical_crossentropy(
        target, softmax_output)
    np.testing.assert_allclose(
        keras.backend.eval(output_from_logit),
        keras.backend.eval(output_from_softmax), atol=1e-5)

  @test_util.run_in_graph_and_eager_modes
  def test_categorical_crossentropy_loss_with_unknown_rank_tensor(self):
    t = keras.backend.placeholder()
    p = keras.backend.placeholder()
    o = keras.losses.categorical_crossentropy(t, p)

    t_val = ops.convert_to_tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    p_val = ops.convert_to_tensor([[.9, .05, .05], [.05, .89, .06],
                                   [.05, .01, .94]])
    f = keras.backend.function([t, p], o)

    result = f([t_val, p_val])
    self.assertArrayNear(result, [.105, .116, .062], 1e-3)

    # from logits
    p_val = ops.convert_to_tensor([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
    o = keras.losses.categorical_crossentropy(t, p, from_logits=True)
    f = keras.backend.function([t, p], o)

    result = f([t_val, p_val])
    self.assertArrayNear(result, [.002, 0, .17], 1e-3)

  @test_util.run_in_graph_and_eager_modes
  def test_sparse_categorical_crossentropy_loss(self):
    target = keras.backend.variable(np.random.randint(0, 1, (5, 1)))
    logits = keras.backend.variable(np.random.random((5, 1)))
    softmax_output = keras.backend.softmax(logits)
    output_from_logit = keras.losses.sparse_categorical_crossentropy(
        target, logits, from_logits=True)
    output_from_softmax = keras.losses.sparse_categorical_crossentropy(
        target, softmax_output)
    np.testing.assert_allclose(
        keras.backend.eval(output_from_logit),
        keras.backend.eval(output_from_softmax), atol=1e-5)

  @test_util.run_in_graph_and_eager_modes
  def test_sparse_categorical_crossentropy_loss_with_unknown_rank_tensor(self):
    t = keras.backend.placeholder()
    p = keras.backend.placeholder()
    o = keras.losses.sparse_categorical_crossentropy(t, p)

    t_val = ops.convert_to_tensor([0, 1, 2])
    p_val = ops.convert_to_tensor([[.9, .05, .05], [.05, .89, .06],
                                   [.05, .01, .94]])
    f = keras.backend.function([t, p], o)

    result = f([t_val, p_val])
    self.assertArrayNear(result, [.105, .116, .062], 1e-3)

    # from logits
    p_val = ops.convert_to_tensor([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
    o = keras.losses.sparse_categorical_crossentropy(t, p, from_logits=True)
    f = keras.backend.function([t, p], o)

    result = f([t_val, p_val])
    self.assertArrayNear(result, [.002, 0, .17], 1e-3)

  @test_util.run_in_graph_and_eager_modes
  def test_binary_crossentropy_loss(self):
    target = keras.backend.variable(np.random.randint(0, 1, (5, 1)))
    logits = keras.backend.variable(np.random.random((5, 1)))
    sigmoid_output = keras.backend.sigmoid(logits)
    output_from_logit = keras.losses.binary_crossentropy(
        target, logits, from_logits=True)
    output_from_sigmoid = keras.losses.binary_crossentropy(
        target, sigmoid_output)
    np.testing.assert_allclose(
        keras.backend.eval(output_from_logit),
        keras.backend.eval(output_from_sigmoid), atol=1e-5)

  def test_get_bce(self):
    bce_fn = keras.losses.get('bce')
    self.assertEqual(bce_fn, keras.losses.binary_crossentropy)

  def test_serialization(self):
    fn = keras.losses.get('mse')
    config = keras.losses.serialize(fn)
    new_fn = keras.losses.deserialize(config)
    self.assertEqual(fn, new_fn)

  def test_categorical_hinge(self):
    y_pred = keras.backend.variable(np.array([[0.3, 0.2, 0.1],
                                              [0.1, 0.2, 0.7]]))
    y_true = keras.backend.variable(np.array([[0, 1, 0], [1, 0, 0]]))
    expected_loss = ((0.3 - 0.2 + 1) + (0.7 - 0.1 + 1)) / 2.0
    loss = keras.backend.eval(keras.losses.categorical_hinge(y_true, y_pred))
    self.assertAllClose(expected_loss, np.mean(loss))

  def test_serializing_loss_class(self):
    orig_loss_class = _MSEMAELoss(0.3)
    with generic_utils.custom_object_scope({'_MSEMAELoss': _MSEMAELoss}):
      serialized = keras.losses.serialize(orig_loss_class)

    with generic_utils.custom_object_scope({'_MSEMAELoss': _MSEMAELoss}):
      deserialized = keras.losses.deserialize(serialized)
    assert isinstance(deserialized, _MSEMAELoss)
    assert deserialized.mse_fraction == 0.3

  def test_serializing_model_with_loss_class(self):
    tmpdir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, tmpdir)
    model_filename = os.path.join(tmpdir, 'custom_loss.h5')

    with self.cached_session():
      with generic_utils.custom_object_scope({'_MSEMAELoss': _MSEMAELoss}):
        loss = _MSEMAELoss(0.3)
        inputs = keras.layers.Input((2,))
        outputs = keras.layers.Dense(1, name='model_output')(inputs)
        model = keras.models.Model(inputs, outputs)
        model.compile(optimizer='sgd', loss={'model_output': loss})
        model.fit(np.random.rand(256, 2), np.random.rand(256, 1))

        if h5py is None:
          return

        model.save(model_filename)

      with generic_utils.custom_object_scope({'_MSEMAELoss': _MSEMAELoss}):
        loaded_model = keras.models.load_model(model_filename)
        loaded_model.predict(np.random.rand(128, 2))

  def test_loss_wrapper(self):
    loss_fn = keras.losses.get('mse')
    mse_obj = keras.losses.LossFunctionWrapper(loss_fn, name=loss_fn.__name__)

    self.assertEqual(mse_obj.name, 'mean_squared_error')
    self.assertEqual(mse_obj.reduction, losses_utils.ReductionV2.AUTO)

    y_true = constant_op.constant([[1., 9.], [2., 5.]])
    y_pred = constant_op.constant([[4., 8.], [12., 3.]])
    sample_weight = constant_op.constant([1.2, 0.5])
    loss = mse_obj(y_true, y_pred, sample_weight=sample_weight)

    # mse = [((4 - 1)^2 + (8 - 9)^2) / 2, ((12 - 2)^2 + (3 - 5)^2) / 2]
    # mse = [5, 52]
    # weighted_mse = [5 * 1.2, 52 * 0.5] = [6, 26]
    # reduced_weighted_mse = (6 + 26) / 2 =
    self.assertAllClose(self.evaluate(loss), 16, 1e-2)

  def test_invalid_reduction(self):
    with self.assertRaisesRegexp(ValueError, 'Invalid Reduction Key Foo.'):
      keras.losses.MeanSquaredError(reduction='Foo')

    mse_obj = keras.losses.MeanSquaredError()
    y = constant_op.constant([1])
    mse_obj.reduction = 'Bar'
    with self.assertRaisesRegexp(ValueError, 'Invalid Reduction Key Bar.'):
      mse_obj(y, y)


@test_util.run_all_in_graph_and_eager_modes
class MeanSquaredErrorTest(test.TestCase):

  def test_config(self):
    mse_obj = keras.losses.MeanSquaredError(
        reduction=losses_utils.ReductionV2.SUM, name='mse_1')
    self.assertEqual(mse_obj.name, 'mse_1')
    self.assertEqual(mse_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_all_correct_unweighted(self):
    mse_obj = keras.losses.MeanSquaredError()
    y_true = constant_op.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
    loss = mse_obj(y_true, y_true)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_unweighted(self):
    mse_obj = keras.losses.MeanSquaredError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mse_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), 49.5, 3)

  def test_scalar_weighted(self):
    mse_obj = keras.losses.MeanSquaredError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mse_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), 113.85, 3)

  def test_sample_weighted(self):
    mse_obj = keras.losses.MeanSquaredError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = mse_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 767.8 / 6, 3)

  def test_timestep_weighted(self):
    mse_obj = keras.losses.MeanSquaredError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3, 1),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([3, 6, 5, 0, 4, 2], shape=(2, 3))
    loss = mse_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 587 / 6, 3)

  def test_zero_weighted(self):
    mse_obj = keras.losses.MeanSquaredError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mse_obj(y_true, y_pred, sample_weight=0)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_invalid_sample_weight(self):
    mse_obj = keras.losses.MeanSquaredError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3], shape=(2, 3, 1))
    sample_weight = constant_op.constant([3, 6, 5, 0], shape=(2, 2))
    with self.assertRaisesRegexp(ValueError,
                                 'weights can not be broadcast to values'):
      mse_obj(y_true, y_pred, sample_weight=sample_weight)

  def test_no_reduction(self):
    mse_obj = keras.losses.MeanSquaredError(
        reduction=losses_utils.ReductionV2.NONE)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mse_obj(y_true, y_pred, sample_weight=2.3)
    loss = self.evaluate(loss)
    self.assertArrayNear(loss, [84.3333, 143.3666], 1e-3)

  def test_sum_reduction(self):
    mse_obj = keras.losses.MeanSquaredError(
        reduction=losses_utils.ReductionV2.SUM)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mse_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), 227.69998, 3)


@test_util.run_all_in_graph_and_eager_modes
class MeanAbsoluteErrorTest(test.TestCase):

  def test_config(self):
    mae_obj = keras.losses.MeanAbsoluteError(
        reduction=losses_utils.ReductionV2.SUM, name='mae_1')
    self.assertEqual(mae_obj.name, 'mae_1')
    self.assertEqual(mae_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_all_correct_unweighted(self):
    mae_obj = keras.losses.MeanAbsoluteError()
    y_true = constant_op.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
    loss = mae_obj(y_true, y_true)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_unweighted(self):
    mae_obj = keras.losses.MeanAbsoluteError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mae_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), 5.5, 3)

  def test_scalar_weighted(self):
    mae_obj = keras.losses.MeanAbsoluteError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mae_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), 12.65, 3)

  def test_sample_weighted(self):
    mae_obj = keras.losses.MeanAbsoluteError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = mae_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 81.4 / 6, 3)

  def test_timestep_weighted(self):
    mae_obj = keras.losses.MeanAbsoluteError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3, 1),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([3, 6, 5, 0, 4, 2], shape=(2, 3))
    loss = mae_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 83 / 6, 3)

  def test_zero_weighted(self):
    mae_obj = keras.losses.MeanAbsoluteError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mae_obj(y_true, y_pred, sample_weight=0)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_invalid_sample_weight(self):
    mae_obj = keras.losses.MeanAbsoluteError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3], shape=(2, 3, 1))
    sample_weight = constant_op.constant([3, 6, 5, 0], shape=(2, 2))
    with self.assertRaisesRegexp(ValueError,
                                 'weights can not be broadcast to values'):
      mae_obj(y_true, y_pred, sample_weight=sample_weight)

  def test_no_reduction(self):
    mae_obj = keras.losses.MeanAbsoluteError(
        reduction=losses_utils.ReductionV2.NONE)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mae_obj(y_true, y_pred, sample_weight=2.3)
    loss = self.evaluate(loss)
    self.assertArrayNear(loss, [10.7333, 14.5666], 1e-3)

  def test_sum_reduction(self):
    mae_obj = keras.losses.MeanAbsoluteError(
        reduction=losses_utils.ReductionV2.SUM)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mae_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), 25.29999, 3)


@test_util.run_all_in_graph_and_eager_modes
class MeanAbsolutePercentageErrorTest(test.TestCase):

  def test_config(self):
    mape_obj = keras.losses.MeanAbsolutePercentageError(
        reduction=losses_utils.ReductionV2.SUM, name='mape_1')
    self.assertEqual(mape_obj.name, 'mape_1')
    self.assertEqual(mape_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_all_correct_unweighted(self):
    mape_obj = keras.losses.MeanAbsolutePercentageError()
    y_true = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mape_obj(y_true, y_true)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_unweighted(self):
    mape_obj = keras.losses.MeanAbsolutePercentageError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mape_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), 211.8518, 3)

  def test_scalar_weighted(self):
    mape_obj = keras.losses.MeanAbsolutePercentageError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mape_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), 487.259, 3)

  def test_sample_weighted(self):
    mape_obj = keras.losses.MeanAbsolutePercentageError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = mape_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 422.8888, 3)

  def test_timestep_weighted(self):
    mape_obj = keras.losses.MeanAbsolutePercentageError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3, 1),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([3, 6, 5, 0, 4, 2], shape=(2, 3))
    loss = mape_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 694.4445, 3)

  def test_zero_weighted(self):
    mape_obj = keras.losses.MeanAbsolutePercentageError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mape_obj(y_true, y_pred, sample_weight=0)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_no_reduction(self):
    mape_obj = keras.losses.MeanAbsolutePercentageError(
        reduction=losses_utils.ReductionV2.NONE)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mape_obj(y_true, y_pred, sample_weight=2.3)
    loss = self.evaluate(loss)
    self.assertArrayNear(loss, [621.8518, 352.6666], 1e-3)


@test_util.run_all_in_graph_and_eager_modes
class MeanSquaredLogarithmicErrorTest(test.TestCase):

  def test_config(self):
    msle_obj = keras.losses.MeanSquaredLogarithmicError(
        reduction=losses_utils.ReductionV2.SUM, name='mape_1')
    self.assertEqual(msle_obj.name, 'mape_1')
    self.assertEqual(msle_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_unweighted(self):
    msle_obj = keras.losses.MeanSquaredLogarithmicError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = msle_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), 1.4370, 3)

  def test_scalar_weighted(self):
    msle_obj = keras.losses.MeanSquaredLogarithmicError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = msle_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), 3.3051, 3)

  def test_sample_weighted(self):
    msle_obj = keras.losses.MeanSquaredLogarithmicError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = msle_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 3.7856, 3)

  def test_timestep_weighted(self):
    msle_obj = keras.losses.MeanSquaredLogarithmicError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3, 1),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([3, 6, 5, 0, 4, 2], shape=(2, 3))
    loss = msle_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 2.6473, 3)

  def test_zero_weighted(self):
    msle_obj = keras.losses.MeanSquaredLogarithmicError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = msle_obj(y_true, y_pred, sample_weight=0)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)


@test_util.run_all_in_graph_and_eager_modes
class CosineSimilarityTest(test.TestCase):

  def l2_norm(self, x, axis):
    epsilon = 1e-12
    square_sum = np.sum(np.square(x), axis=axis, keepdims=True)
    x_inv_norm = 1 / np.sqrt(np.maximum(square_sum, epsilon))
    return np.multiply(x, x_inv_norm)

  def setup(self, axis=1):
    self.np_y_true = np.asarray([[1, 9, 2], [-5, -2, 6]], dtype=np.float32)
    self.np_y_pred = np.asarray([[4, 8, 12], [8, 1, 3]], dtype=np.float32)

    y_true = self.l2_norm(self.np_y_true, axis)
    y_pred = self.l2_norm(self.np_y_pred, axis)
    self.expected_loss = np.sum(np.multiply(y_true, y_pred), axis=(axis,))

    self.y_true = constant_op.constant(self.np_y_true)
    self.y_pred = constant_op.constant(self.np_y_pred)

  def test_config(self):
    cosine_obj = keras.losses.CosineSimilarity(
        axis=2, reduction=losses_utils.ReductionV2.SUM, name='cosine_loss')
    self.assertEqual(cosine_obj.name, 'cosine_loss')
    self.assertEqual(cosine_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_unweighted(self):
    self.setup()
    cosine_obj = keras.losses.CosineSimilarity()
    loss = cosine_obj(self.y_true, self.y_pred)
    expected_loss = -np.mean(self.expected_loss)
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_scalar_weighted(self):
    self.setup()
    cosine_obj = keras.losses.CosineSimilarity()
    sample_weight = 2.3
    loss = cosine_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
    expected_loss = -np.mean(self.expected_loss * sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_sample_weighted(self):
    self.setup()
    cosine_obj = keras.losses.CosineSimilarity()
    sample_weight = np.asarray([1.2, 3.4])
    loss = cosine_obj(
        self.y_true,
        self.y_pred,
        sample_weight=constant_op.constant(sample_weight))
    expected_loss = -np.mean(self.expected_loss * sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_timestep_weighted(self):
    self.setup()
    cosine_obj = keras.losses.CosineSimilarity()
    np_y_true = self.np_y_true.reshape((2, 3, 1))
    np_y_pred = self.np_y_pred.reshape((2, 3, 1))
    sample_weight = np.asarray([3, 6, 5, 0, 4, 2]).reshape((2, 3))

    y_true = self.l2_norm(np_y_true, 2)
    y_pred = self.l2_norm(np_y_pred, 2)
    expected_loss = np.sum(np.multiply(y_true, y_pred), axis=(2,))

    y_true = constant_op.constant(np_y_true)
    y_pred = constant_op.constant(np_y_pred)
    loss = cosine_obj(
        y_true, y_pred, sample_weight=constant_op.constant(sample_weight))

    expected_loss = -np.mean(expected_loss * sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_zero_weighted(self):
    self.setup()
    cosine_obj = keras.losses.CosineSimilarity()
    loss = cosine_obj(self.y_true, self.y_pred, sample_weight=0)
    self.assertAlmostEqual(self.evaluate(loss), 0., 3)

  def test_axis(self):
    self.setup(axis=1)
    cosine_obj = keras.losses.CosineSimilarity(axis=1)
    loss = cosine_obj(self.y_true, self.y_pred)
    expected_loss = -np.mean(self.expected_loss)
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)


@test_util.run_all_in_graph_and_eager_modes
class BinaryCrossentropyTest(test.TestCase):

  def test_config(self):
    bce_obj = keras.losses.BinaryCrossentropy(
        reduction=losses_utils.ReductionV2.SUM, name='bce_1')
    self.assertEqual(bce_obj.name, 'bce_1')
    self.assertEqual(bce_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_all_correct_unweighted(self):
    y_true = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                  dtype=dtypes.float32)
    bce_obj = keras.losses.BinaryCrossentropy()
    loss = bce_obj(y_true, y_true)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

    # Test with logits.
    logits = constant_op.constant([[100.0, -100.0, -100.0],
                                   [-100.0, 100.0, -100.0],
                                   [-100.0, -100.0, 100.0]])
    bce_obj = keras.losses.BinaryCrossentropy(from_logits=True)
    loss = bce_obj(y_true, logits)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_unweighted(self):
    y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
    y_pred = np.asarray([1, 1, 1, 0], dtype=np.float32).reshape([2, 2])
    bce_obj = keras.losses.BinaryCrossentropy()
    loss = bce_obj(y_true, y_pred)

    # EPSILON = 1e-7, y = y_true, y` = y_pred, Y_MAX = 0.9999999
    # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
    # y` = [Y_MAX, Y_MAX, Y_MAX, EPSILON]

    # Loss = -(y log(y` + EPSILON) + (1 - y) log(1 - y` + EPSILON))
    #      = [-log(Y_MAX + EPSILON), -log(1 - Y_MAX + EPSILON),
    #         -log(Y_MAX + EPSILON), -log(1)]
    #      = [0, 15.33, 0, 0]
    # Reduced loss = 15.33 / 4

    self.assertAlmostEqual(self.evaluate(loss), 3.833, 3)

    # Test with logits.
    y_true = constant_op.constant([[1, 0, 1], [0, 1, 1]])
    logits = constant_op.constant([[100.0, -100.0, 100.0],
                                   [100.0, 100.0, -100.0]])
    bce_obj = keras.losses.BinaryCrossentropy(from_logits=True)
    loss = bce_obj(y_true, logits)

    # Loss = max(x, 0) - x * z + log(1 + exp(-abs(x)))
    #            (where x = logits and z = y_true)
    #      = [((100 - 100 * 1 + log(1 + exp(-100))) +
    #          (0 + 100 * 0 + log(1 + exp(-100))) +
    #          (100 - 100 * 1 + log(1 + exp(-100))),
    #         ((100 - 100 * 0 + log(1 + exp(-100))) +
    #          (100 - 100 * 1 + log(1 + exp(-100))) +
    #          (0 + 100 * 1 + log(1 + exp(-100))))]
    #      = [(0 + 0 + 0) / 3, 200 / 3]
    # Reduced loss = (0 + 66.666) / 2

    self.assertAlmostEqual(self.evaluate(loss), 33.333, 3)

  def test_scalar_weighted(self):
    bce_obj = keras.losses.BinaryCrossentropy()
    y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
    y_pred = np.asarray([1, 1, 1, 0], dtype=np.float32).reshape([2, 2])
    loss = bce_obj(y_true, y_pred, sample_weight=2.3)

    # EPSILON = 1e-7, y = y_true, y` = y_pred, Y_MAX = 0.9999999
    # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
    # y` = [Y_MAX, Y_MAX, Y_MAX, EPSILON]

    # Loss = -(y log(y` + EPSILON) + (1 - y) log(1 - y` + EPSILON))
    #      = [-log(Y_MAX + EPSILON), -log(1 - Y_MAX + EPSILON),
    #         -log(Y_MAX + EPSILON), -log(1)]
    #      = [0, 15.33, 0, 0]
    # Weighted loss = [0, 15.33 * 2.3, 0, 0]
    # Reduced loss = 15.33 * 2.3 / 4

    self.assertAlmostEqual(self.evaluate(loss), 8.817, 3)

    # Test with logits.
    y_true = constant_op.constant([[1, 0, 1], [0, 1, 1]])
    logits = constant_op.constant([[100.0, -100.0, 100.0],
                                   [100.0, 100.0, -100.0]])
    bce_obj = keras.losses.BinaryCrossentropy(from_logits=True)
    loss = bce_obj(y_true, logits, sample_weight=2.3)

    # Loss = max(x, 0) - x * z + log(1 + exp(-abs(x)))
    #            (where x = logits and z = y_true)
    # Loss = [(0 + 0 + 0) / 3, 200 / 3]
    # Weighted loss = [0 * 2.3, 66.666 * 2.3]
    # Reduced loss = (0 + 66.666 * 2.3) / 2

    self.assertAlmostEqual(self.evaluate(loss), 76.667, 3)

  def test_sample_weighted(self):
    bce_obj = keras.losses.BinaryCrossentropy()
    y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
    y_pred = np.asarray([1, 1, 1, 0], dtype=np.float32).reshape([2, 2])
    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = bce_obj(y_true, y_pred, sample_weight=sample_weight)

    # EPSILON = 1e-7, y = y_true, y` = y_pred, Y_MAX = 0.9999999
    # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
    # y` = [Y_MAX, Y_MAX, Y_MAX, EPSILON]

    # Loss = -(y log(y` + EPSILON) + (1 - y) log(1 - y` + EPSILON))
    #      = [-log(Y_MAX + EPSILON), -log(1 - Y_MAX + EPSILON),
    #         -log(Y_MAX + EPSILON), -log(1)]
    #      = [0, 15.33, 0, 0]
    # Reduced loss = 15.33 * 1.2 / 4

    self.assertAlmostEqual(self.evaluate(loss), 4.6, 3)

    # Test with logits.
    y_true = constant_op.constant([[1, 0, 1], [0, 1, 1]])
    logits = constant_op.constant([[100.0, -100.0, 100.0],
                                   [100.0, 100.0, -100.0]])
    weights = constant_op.constant([4, 3])
    bce_obj = keras.losses.BinaryCrossentropy(from_logits=True)
    loss = bce_obj(y_true, logits, sample_weight=weights)

    # Loss = max(x, 0) - x * z + log(1 + exp(-abs(x)))
    #            (where x = logits and z = y_true)
    # Loss = [(0 + 0 + 0)/3, 200 / 3]
    # Weighted loss = [0 * 4, 66.666 * 3]
    # Reduced loss = (0 + 66.666 * 3) / 2

    self.assertAlmostEqual(self.evaluate(loss), 100, 3)

  def test_no_reduction(self):
    y_true = constant_op.constant([[1, 0, 1], [0, 1, 1]])
    logits = constant_op.constant([[100.0, -100.0, 100.0],
                                   [100.0, 100.0, -100.0]])
    bce_obj = keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=losses_utils.ReductionV2.NONE)
    loss = bce_obj(y_true, logits)

    # Loss = max(x, 0) - x * z + log(1 + exp(-abs(x)))
    #            (where x = logits and z = y_true)
    # Loss = [(0 + 0 + 0)/3, (200)/3]

    self.assertAllClose((0., 66.6666), self.evaluate(loss), 3)

  def test_label_smoothing(self):
    logits = constant_op.constant([[100.0, -100.0, -100.0]])
    y_true = constant_op.constant([[1, 0, 1]])
    label_smoothing = 0.1
    # Loss: max(x, 0) - x * z + log(1 + exp(-abs(x)))
    #            (where x = logits and z = y_true)
    # Label smoothing: z' = z * (1 - L) + 0.5L
    #                  1  = 1 - 0.5L
    #                  0  = 0.5L
    # Applying the above two fns to the given input:
    # (100 - 100 * (1 - 0.5 L)  + 0 +
    #  0   + 100 * (0.5 L)      + 0 +
    #  0   + 100 * (1 - 0.5 L)  + 0) * (1/3)
    #  = (100 + 50L) * 1/3
    bce_obj = keras.losses.BinaryCrossentropy(
        from_logits=True, label_smoothing=label_smoothing)
    loss = bce_obj(y_true, logits)
    expected_value = (100.0 + 50.0 * label_smoothing) / 3.0
    self.assertAlmostEqual(self.evaluate(loss), expected_value, 3)


@test_util.run_all_in_graph_and_eager_modes
class CategoricalCrossentropyTest(test.TestCase):

  def test_config(self):
    cce_obj = keras.losses.CategoricalCrossentropy(
        reduction=losses_utils.ReductionV2.SUM, name='bce_1')
    self.assertEqual(cce_obj.name, 'bce_1')
    self.assertEqual(cce_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_all_correct_unweighted(self):
    y_true = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                  dtype=dtypes.int64)
    y_pred = constant_op.constant([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                                  dtype=dtypes.float32)
    cce_obj = keras.losses.CategoricalCrossentropy()
    loss = cce_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

    # Test with logits.
    logits = constant_op.constant([[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]])
    cce_obj = keras.losses.CategoricalCrossentropy(from_logits=True)
    loss = cce_obj(y_true, logits)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_unweighted(self):
    cce_obj = keras.losses.CategoricalCrossentropy()
    y_true = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = constant_op.constant(
        [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]], dtype=dtypes.float32)
    loss = cce_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), .3239, 3)

    # Test with logits.
    logits = constant_op.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
    cce_obj = keras.losses.CategoricalCrossentropy(from_logits=True)
    loss = cce_obj(y_true, logits)
    self.assertAlmostEqual(self.evaluate(loss), .0573, 3)

  def test_scalar_weighted(self):
    cce_obj = keras.losses.CategoricalCrossentropy()
    y_true = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = constant_op.constant(
        [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]], dtype=dtypes.float32)
    loss = cce_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), .7449, 3)

    # Test with logits.
    logits = constant_op.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
    cce_obj = keras.losses.CategoricalCrossentropy(from_logits=True)
    loss = cce_obj(y_true, logits, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), .1317, 3)

  def test_sample_weighted(self):
    cce_obj = keras.losses.CategoricalCrossentropy()
    y_true = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = constant_op.constant(
        [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]], dtype=dtypes.float32)
    sample_weight = constant_op.constant([[1.2], [3.4], [5.6]], shape=(3, 1))
    loss = cce_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 1.0696, 3)

    # Test with logits.
    logits = constant_op.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
    cce_obj = keras.losses.CategoricalCrossentropy(from_logits=True)
    loss = cce_obj(y_true, logits, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 0.31829, 3)

  def test_no_reduction(self):
    y_true = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    logits = constant_op.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
    cce_obj = keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=losses_utils.ReductionV2.NONE)
    loss = cce_obj(y_true, logits)
    self.assertAllClose((0.001822, 0.000459, 0.169846), self.evaluate(loss), 3)

  def test_label_smoothing(self):
    logits = constant_op.constant([[100.0, -100.0, -100.0]])
    y_true = constant_op.constant([[1, 0, 0]])
    label_smoothing = 0.1
    # Softmax Cross Entropy Loss: -\sum_i p_i \log q_i
    # where for a softmax activation
    # \log q_i = x_i - \log \sum_j \exp x_j
    #          = x_i - x_max - \log \sum_j \exp (x_j - x_max)
    # For our activations, [100, -100, -100]
    # \log ( exp(0) + exp(-200) + exp(-200) ) = 0
    # so our log softmaxes become: [0, -200, -200]
    # Label smoothing: z' = z * (1 - L) + L/n
    #                  1  = 1 - L + L/n
    #                  0  = L/n
    # Applying the above two fns to the given input:
    # -0 * (1 - L + L/n) + 200 * L/n + 200 * L/n = 400 L/n
    cce_obj = keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=label_smoothing)
    loss = cce_obj(y_true, logits)
    expected_value = 400.0 * label_smoothing / 3.0
    self.assertAlmostEqual(self.evaluate(loss), expected_value, 3)


@test_util.run_all_in_graph_and_eager_modes
class SparseCategoricalCrossentropyTest(test.TestCase):

  def test_config(self):
    cce_obj = keras.losses.SparseCategoricalCrossentropy(
        reduction=losses_utils.ReductionV2.SUM, name='scc')
    self.assertEqual(cce_obj.name, 'scc')
    self.assertEqual(cce_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_all_correct_unweighted(self):
    y_true = constant_op.constant([[0], [1], [2]], dtype=dtypes.int64)
    y_pred = constant_op.constant([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                                  dtype=dtypes.float32)
    cce_obj = keras.losses.SparseCategoricalCrossentropy()
    loss = cce_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

    # Test with logits.
    logits = constant_op.constant([[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]])
    cce_obj = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = cce_obj(y_true, logits)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_unweighted(self):
    cce_obj = keras.losses.SparseCategoricalCrossentropy()
    y_true = constant_op.constant([0, 1, 2])
    y_pred = constant_op.constant(
        [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]], dtype=dtypes.float32)
    loss = cce_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), .3239, 3)

    # Test with logits.
    logits = constant_op.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
    cce_obj = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = cce_obj(y_true, logits)
    self.assertAlmostEqual(self.evaluate(loss), .0573, 3)

  def test_scalar_weighted(self):
    cce_obj = keras.losses.SparseCategoricalCrossentropy()
    y_true = constant_op.constant([[0], [1], [2]])
    y_pred = constant_op.constant(
        [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]], dtype=dtypes.float32)
    loss = cce_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), .7449, 3)

    # Test with logits.
    logits = constant_op.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
    cce_obj = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = cce_obj(y_true, logits, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), .1317, 3)

  def test_sample_weighted(self):
    cce_obj = keras.losses.SparseCategoricalCrossentropy()
    y_true = constant_op.constant([[0], [1], [2]])
    y_pred = constant_op.constant(
        [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]], dtype=dtypes.float32)
    sample_weight = constant_op.constant([[1.2], [3.4], [5.6]], shape=(3, 1))
    loss = cce_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 1.0696, 3)

    # Test with logits.
    logits = constant_op.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
    cce_obj = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = cce_obj(y_true, logits, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 0.31829, 3)

  def test_no_reduction(self):
    y_true = constant_op.constant([[0], [1], [2]])
    logits = constant_op.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
    cce_obj = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=losses_utils.ReductionV2.NONE)
    loss = cce_obj(y_true, logits)
    self.assertAllClose((0.001822, 0.000459, 0.169846), self.evaluate(loss), 3)


@test_util.run_all_in_graph_and_eager_modes
class HingeTest(test.TestCase):

  def test_config(self):
    hinge_obj = keras.losses.Hinge(
        reduction=losses_utils.ReductionV2.SUM, name='hinge_loss')
    self.assertEqual(hinge_obj.name, 'hinge_loss')
    self.assertEqual(hinge_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_unweighted(self):
    hinge_obj = keras.losses.Hinge()
    y_true = constant_op.constant([[0, 1, 0, 1], [0, 0, 1, 1]])
    y_pred = constant_op.constant([[-0.3, 0.2, -0.1, 1.6],
                                   [-0.25, -1., 0.5, 0.6]])

    # loss = max(0, 1-y_true * y_pred), where y_true is -1/1

    # y_true = [[-1, 1, -1, 1], [-1, -1, 1, 1]]
    # y_true * y_pred = [[0.3, 0.2, 0.1, 1.6], [0.25, 1, 0.5, 0.6]]
    # 1 - y_true * y_pred = [[0.7, 0.8, 0.9, -0.6], [0.75, 0, 0.5, 0.4]]
    # loss = [(0.7 + 0.8 + 0.9 + 0) / 4, (0.75 + 0 + 0.5 + 0.4) / 4]
    #      = [0.6, 0.4125]
    # reduced loss = (0.6 + 0.4125) / 2

    loss = hinge_obj(y_true, y_pred)
    self.assertAllClose(0.506, self.evaluate(loss), atol=1e-3)

  def test_scalar_weighted(self):
    hinge_obj = keras.losses.Hinge()
    y_true = constant_op.constant([[0, 1, 0, 1], [0, 0, 1, 1]])
    y_pred = constant_op.constant([[-0.3, 0.2, -0.1, 1.6],
                                   [-0.25, -1., 0.5, 0.6]])

    # loss = max(0, 1-y_true * y_pred), where y_true is -1/1

    # y_true = [[-1, 1, -1, 1], [-1, -1, 1, 1]]
    # y_true * y_pred = [[0.3, 0.2, 0.1, 1.6], [0.25, 1, 0.5, 0.6]]
    # 1 - y_true * y_pred = [[0.7, 0.8, 0.9, -0.6], [0.75, 0, 0.5, 0.4]]
    # loss = [(0.7 + 0.8 + 0.9 + 0) / 4, (0.75 + 0 + 0.5 + 0.4) / 4]
    #      = [0.6, 0.4125]
    # weighted_loss = [0.6 * 2.3, 0.4125 * 2.3]
    # reduced loss = (0.6 + 0.4125) * 2.3 / 2

    loss = hinge_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), 1.164, 3)

    # Verify we get the same output when the same input is given
    loss_2 = hinge_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAllClose(self.evaluate(loss), self.evaluate(loss_2), 1e-3)

  def test_sample_weighted(self):
    hinge_obj = keras.losses.Hinge()
    y_true = constant_op.constant([[0, 1, 0, 1], [0, 0, 1, 1]])
    y_pred = constant_op.constant([[-0.3, 0.2, -0.1, 1.6],
                                   [-0.25, -1., 0.5, 0.6]])

    # loss = max(0, 1-y_true * y_pred), where y_true is -1/1

    # y_true = [[-1, 1, -1, 1], [-1, -1, 1, 1]]
    # y_true * y_pred = [[0.3, 0.2, 0.1, 1.6], [0.25, 1, 0.5, 0.6]]
    # 1 - y_true * y_pred = [[0.7, 0.8, 0.9, -0.6], [0.75, 0, 0.5, 0.4]]
    # loss = [(0.7 + 0.8 + 0.9 + 0) / 4, (0.75 + 0 + 0.5 + 0.4) / 4]
    #      = [0.6, 0.4125]
    # weighted loss = [0.6 * 1.2, 0.4125 * 3.4]
    # reduced loss = (0.6 * 1.2 + 0.4125 * 3.4) / 2

    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(self.evaluate(loss), 1.061, 1e-3)

  def test_timestep_weighted(self):
    hinge_obj = keras.losses.Hinge()
    y_true = constant_op.constant([[0, 1, 0, 1], [0, 0, 1, 1]], shape=(2, 4, 1))
    y_pred = constant_op.constant(
        [[-0.3, 0.2, -0.1, 1.6], [-0.25, -1., 0.5, 0.6]], shape=(2, 4, 1))
    sample_weight = constant_op.constant([3, 6, 5, 0, 4, 2, 1, 3], shape=(2, 4))

    # loss = max(0, 1-y_true * y_pred), where y_true is -1/1

    # y_true = [[[-1], [1], [-1], [1]], [[-1], [-1], [1], [1]]]
    # y_true * y_pred = [[[0.3], [0.2], [0.1], [1.6]],
    #                    [[0.25], [1], [0.5], [0.6]]]
    # 1 - y_true * y_pred = [[[0.7], [0.8], [0.9], [-0.6]],
    #                        [[0.75], [0], [0.5], [0.4]]]
    # loss = [[0.7, 0.8, 0.9, 0], [0.75, 0, 0.5, 0.4]]
    # weighted loss    = [[2.1, 4.8, 4.5, 0], [3, 0, 0.5, 1.2]]
    # reduced loss = (2.1 + 4.8 + 4.5 + 0 + 3 + 0 + 0.5 + 1.2) / 8

    loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(self.evaluate(loss), 2.012, 1e-3)

  def test_zero_weighted(self):
    hinge_obj = keras.losses.Hinge()
    y_true = constant_op.constant([[0, 1, 0, 1], [0, 0, 1, 1]])
    y_pred = constant_op.constant([[-0.3, 0.2, -0.1, 1.6],
                                   [-0.25, -1., 0.5, 0.6]])
    loss = hinge_obj(y_true, y_pred, sample_weight=0)
    self.assertAllClose(self.evaluate(loss), 0., 1e-3)


@test_util.run_all_in_graph_and_eager_modes
class SquaredHingeTest(test.TestCase):

  def test_config(self):
    sq_hinge_obj = keras.losses.SquaredHinge(
        reduction=losses_utils.ReductionV2.SUM, name='sq_hinge_loss')
    self.assertEqual(sq_hinge_obj.name, 'sq_hinge_loss')
    self.assertEqual(sq_hinge_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_unweighted(self):
    sq_hinge_obj = keras.losses.SquaredHinge()
    y_true = constant_op.constant([[0, 1, 0, 1], [0, 0, 1, 1]])
    y_pred = constant_op.constant([[-0.3, 0.2, -0.1, 1.6],
                                   [-0.25, -1., 0.5, 0.6]])

    # loss = max(0, 1-y_true * y_pred), where y_true is -1/1

    # y_true = [[-1, 1, -1, 1], [-1, -1, 1, 1]]
    # y_true * y_pred = [[0.3, 0.2, 0.1, 1.6], [0.25, 1, 0.5, 0.6]]
    # 1 - y_true * y_pred = [[0.7, 0.8, 0.9, -0.6], [0.75, 0, 0.5, 0.4]]
    # max(0, 1 - y_true * y_pred) = [[0.7, 0.8, 0.9, 0], [0.75, 0, 0.5, 0.4]]
    # squared(max(0, 1 - y_true * y_pred)) = [[0.49, 0.64, 0.81, 0],
    #                                         [0.5625, 0, 0.25, 0.16]]
    # loss = [(0.49 + 0.64 + 0.81 + 0) / 4, (0.5625 + 0 + 0.25 + 0.16) / 4]
    #      = [0.485, 0.2431]
    # reduced loss = (0.485 + 0.2431) / 2

    loss = sq_hinge_obj(y_true, y_pred)
    self.assertAllClose(self.evaluate(loss), 0.364, 1e-3)

  def test_scalar_weighted(self):
    sq_hinge_obj = keras.losses.SquaredHinge()
    y_true = constant_op.constant([[0, 1, 0, 1], [0, 0, 1, 1]])
    y_pred = constant_op.constant([[-0.3, 0.2, -0.1, 1.6],
                                   [-0.25, -1., 0.5, 0.6]])

    # loss = max(0, 1-y_true * y_pred), where y_true is -1/1

    # y_true = [[-1, 1, -1, 1], [-1, -1, 1, 1]]
    # y_true * y_pred = [[0.3, 0.2, 0.1, 1.6], [0.25, 1, 0.5, 0.6]]
    # 1 - y_true * y_pred = [[0.7, 0.8, 0.9, -0.6], [0.75, 0, 0.5, 0.4]]
    # max(0, 1 - y_true * y_pred) = [[0.7, 0.8, 0.9, 0], [0.75, 0, 0.5, 0.4]]
    # squared(max(0, 1 - y_true * y_pred)) = [[0.49, 0.64, 0.81, 0],
    #                                         [0.5625, 0, 0.25, 0.16]]
    # loss = [(0.49 + 0.64 + 0.81 + 0) / 4, (0.5625 + 0 + 0.25 + 0.16) / 4]
    #      = [0.485, 0.2431]
    # weighted loss = [0.485 * 2.3, 0.2431 * 2.3]
    # reduced loss = (0.485 + 0.2431) * 2.3 / 2

    loss = sq_hinge_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAllClose(self.evaluate(loss), 0.837, 1e-3)

    # Verify we get the same output when the same input is given
    loss_2 = sq_hinge_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), self.evaluate(loss_2), 3)

  def test_sample_weighted(self):
    sq_hinge_obj = keras.losses.SquaredHinge()
    y_true = constant_op.constant([[0, 1, 0, 1], [0, 0, 1, 1]])
    y_pred = constant_op.constant([[-0.3, 0.2, -0.1, 1.6],
                                   [-0.25, -1., 0.5, 0.6]])

    # loss = max(0, 1-y_true * y_pred), where y_true is -1/1

    # y_true = [[-1, 1, -1, 1], [-1, -1, 1, 1]]
    # y_true * y_pred = [[0.3, 0.2, 0.1, 1.6], [0.25, 1, 0.5, 0.6]]
    # 1 - y_true * y_pred = [[0.7, 0.8, 0.9, -0.6], [0.75, 0, 0.5, 0.4]]
    # max(0, 1 - y_true * y_pred) = [[0.7, 0.8, 0.9, 0], [0.75, 0, 0.5, 0.4]]
    # squared(max(0, 1 - y_true * y_pred)) = [[0.49, 0.64, 0.81, 0],
    #                                         [0.5625, 0, 0.25, 0.16]]
    # loss = [(0.49 + 0.64 + 0.81 + 0) / 4, (0.5625 + 0 + 0.25 + 0.16) / 4]
    #      = [0.485, 0.2431]
    # weighted loss = [0.485 * 1.2, 0.2431 * 3.4]
    # reduced loss = (0.485 * 1.2 + 0.2431 * 3.4) / 2

    sample_weight = constant_op.constant([1.2, 3.4])
    loss = sq_hinge_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(self.evaluate(loss), 0.704, 1e-3)

  def test_timestep_weighted(self):
    sq_hinge_obj = keras.losses.SquaredHinge()
    y_true = constant_op.constant([[0, 1, 0, 1], [0, 0, 1, 1]], shape=(2, 4, 1))
    y_pred = constant_op.constant(
        [[-0.3, 0.2, -0.1, 1.6], [-0.25, -1., 0.5, 0.6]], shape=(2, 4, 1))
    sample_weight = constant_op.constant([3, 6, 5, 0, 4, 2, 1, 3], shape=(2, 4))

    # loss = max(0, 1-y_true * y_pred), where y_true is -1/1

    # y_true = [[[-1], [1], [-1], [1]], [[-1], [-1], [1], [1]]]
    # y_true * y_pred = [[[0.3], [0.2], [0.1], [1.6]],
    #                    [[0.25], [1], [0.5], [0.6]]]
    # 1 - y_true * y_pred = [[[0.7], [0.8], [0.9], [-0.6]],
    #                        [[0.75], [0], [0.5], [0.4]]]
    # loss = [[0.49, 0.64, 0.81, 0], [0.5625, 0, 0.25, 0.16]]
    # weighted loss    = [[1.47, 3.84, 4.05, 0], [2.25, 0, 0.25, 0.48]]
    # reduced loss = (1.47 + 3.84 + 4.05 + 0 + 2.25 + 0 + 0.25 + 0.48) / 8

    loss = sq_hinge_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(self.evaluate(loss), 1.542, 1e-3)

  def test_zero_weighted(self):
    sq_hinge_obj = keras.losses.SquaredHinge()
    y_true = constant_op.constant([[0, 1, 0, 1], [0, 0, 1, 1]])
    y_pred = constant_op.constant([[-0.3, 0.2, -0.1, 1.6],
                                   [-0.25, -1., 0.5, 0.6]])
    loss = sq_hinge_obj(y_true, y_pred, sample_weight=0)
    self.assertAllClose(self.evaluate(loss), 0., 1e-3)


@test_util.run_all_in_graph_and_eager_modes
class CategoricalHingeTest(test.TestCase):

  def test_config(self):
    cat_hinge_obj = keras.losses.CategoricalHinge(
        reduction=losses_utils.ReductionV2.SUM, name='cat_hinge_loss')
    self.assertEqual(cat_hinge_obj.name, 'cat_hinge_loss')
    self.assertEqual(cat_hinge_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_unweighted(self):
    cat_hinge_obj = keras.losses.CategoricalHinge()
    y_true = constant_op.constant([1, 9, 2, -5], shape=(2, 2))
    y_pred = constant_op.constant([4, 8, 12, 8],
                                  shape=(2, 2),
                                  dtype=dtypes.float32)
    loss = cat_hinge_obj(y_true, y_pred)

    # pos = reduce_sum(y_true * y_pred) = [1*4+8*9, 12*2+8*-5] = [76, -16]
    # neg = reduce_max((1. - y_true) * y_pred) = [[0, -64], [-12, 48]] = [0, 48]
    # cat_hinge = max(0., neg - pos + 1.) = [0, 65]
    # reduced_loss = (0 + 65)/2 = 32.5
    self.assertAlmostEqual(self.evaluate(loss), 32.5, 3)

  def test_scalar_weighted(self):
    cat_hinge_obj = keras.losses.CategoricalHinge()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = cat_hinge_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), 83.95, 3)

    # Verify we get the same output when the same input is given
    loss_2 = cat_hinge_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), self.evaluate(loss_2), 3)

  def test_sample_weighted(self):
    cat_hinge_obj = keras.losses.CategoricalHinge()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = cat_hinge_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 124.1, 3)

  def test_timestep_weighted(self):
    cat_hinge_obj = keras.losses.CategoricalHinge()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3, 1),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([3, 6, 5, 0, 4, 2], shape=(2, 3))
    loss = cat_hinge_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 4.0, 3)

  def test_zero_weighted(self):
    cat_hinge_obj = keras.losses.CategoricalHinge()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = cat_hinge_obj(y_true, y_pred, sample_weight=0)
    self.assertAlmostEqual(self.evaluate(loss), 0., 3)


@test_util.run_all_in_graph_and_eager_modes
class LogCoshTest(test.TestCase):

  def setup(self):
    y_pred = np.asarray([1, 9, 2, -5, -2, 6]).reshape((2, 3))
    y_true = np.asarray([4, 8, 12, 8, 1, 3]).reshape((2, 3))

    self.batch_size = 6
    error = y_pred - y_true
    self.expected_losses = np.log((np.exp(error) + np.exp(-error)) / 2)

    self.y_pred = constant_op.constant(y_pred, dtype=dtypes.float32)
    self.y_true = constant_op.constant(y_true)

  def test_config(self):
    logcosh_obj = keras.losses.LogCosh(
        reduction=losses_utils.ReductionV2.SUM, name='logcosh_loss')
    self.assertEqual(logcosh_obj.name, 'logcosh_loss')
    self.assertEqual(logcosh_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_unweighted(self):
    self.setup()
    logcosh_obj = keras.losses.LogCosh()

    loss = logcosh_obj(self.y_true, self.y_pred)
    expected_loss = np.sum(self.expected_losses) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_scalar_weighted(self):
    self.setup()
    logcosh_obj = keras.losses.LogCosh()
    sample_weight = 2.3

    loss = logcosh_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
    expected_loss = sample_weight * np.sum(
        self.expected_losses) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

    # Verify we get the same output when the same input is given
    loss_2 = logcosh_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), self.evaluate(loss_2), 3)

  def test_sample_weighted(self):
    self.setup()
    logcosh_obj = keras.losses.LogCosh()

    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = logcosh_obj(self.y_true, self.y_pred, sample_weight=sample_weight)

    expected_loss = np.multiply(
        self.expected_losses,
        np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3)))
    expected_loss = np.sum(expected_loss) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_timestep_weighted(self):
    self.setup()
    logcosh_obj = keras.losses.LogCosh()
    y_true = np.asarray([1, 9, 2, -5, -2, 6]).reshape(2, 3, 1)
    y_pred = np.asarray([4, 8, 12, 8, 1, 3]).reshape(2, 3, 1)
    error = y_pred - y_true
    expected_losses = np.log((np.exp(error) + np.exp(-error)) / 2)
    sample_weight = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3, 1))

    y_pred = constant_op.constant(y_pred, dtype=dtypes.float32)
    y_true = constant_op.constant(y_true)
    loss = logcosh_obj(
        y_true,
        y_pred,
        sample_weight=constant_op.constant(sample_weight, shape=(2, 3)))
    expected_loss = np.sum(expected_losses * sample_weight) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_zero_weighted(self):
    self.setup()
    logcosh_obj = keras.losses.LogCosh()
    sample_weight = 0
    loss = logcosh_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 0., 3)


@test_util.run_all_in_graph_and_eager_modes
class PoissonTest(test.TestCase):

  def setup(self):
    self.np_y_pred = np.asarray([1, 9, 2, 5, 2, 6]).reshape((2, 3))
    self.np_y_true = np.asarray([4, 8, 12, 8, 1, 3]).reshape((2, 3))

    self.batch_size = 6
    self.expected_losses = self.np_y_pred - np.multiply(self.np_y_true,
                                                        np.log(self.np_y_pred))

    self.y_pred = constant_op.constant(self.np_y_pred, dtype=dtypes.float32)
    self.y_true = constant_op.constant(self.np_y_true)

  def test_config(self):
    poisson_obj = keras.losses.Poisson(
        reduction=losses_utils.ReductionV2.SUM, name='poisson')
    self.assertEqual(poisson_obj.name, 'poisson')
    self.assertEqual(poisson_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_unweighted(self):
    self.setup()
    poisson_obj = keras.losses.Poisson()

    loss = poisson_obj(self.y_true, self.y_pred)
    expected_loss = np.sum(self.expected_losses) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_scalar_weighted(self):
    self.setup()
    poisson_obj = keras.losses.Poisson()
    sample_weight = 2.3
    loss = poisson_obj(self.y_true, self.y_pred, sample_weight=sample_weight)

    expected_loss = sample_weight * np.sum(
        self.expected_losses) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

    # Verify we get the same output when the same input is given
    loss_2 = poisson_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), self.evaluate(loss_2), 3)

  def test_sample_weighted(self):
    self.setup()
    poisson_obj = keras.losses.Poisson()

    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = poisson_obj(self.y_true, self.y_pred, sample_weight=sample_weight)

    expected_loss = np.multiply(
        self.expected_losses,
        np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3)))
    expected_loss = np.sum(expected_loss) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_timestep_weighted(self):
    self.setup()
    poisson_obj = keras.losses.Poisson()
    y_true = self.np_y_true.reshape(2, 3, 1)
    y_pred = self.np_y_pred.reshape(2, 3, 1)
    sample_weight = np.asarray([3, 6, 5, 0, 4, 2]).reshape(2, 3, 1)
    expected_losses = y_pred - np.multiply(y_true, np.log(y_pred))

    y_pred = constant_op.constant(y_pred, dtype=dtypes.float32)
    y_true = constant_op.constant(y_true)

    loss = poisson_obj(
        y_true,
        y_pred,
        sample_weight=constant_op.constant(sample_weight, shape=(2, 3)))
    expected_loss = np.sum(expected_losses * sample_weight) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_zero_weighted(self):
    self.setup()
    poisson_obj = keras.losses.Poisson()
    loss = poisson_obj(self.y_true, self.y_pred, sample_weight=0)
    self.assertAlmostEqual(self.evaluate(loss), 0., 3)


@test_util.run_all_in_graph_and_eager_modes
class KLDivergenceTest(test.TestCase):

  def setup(self):
    self.np_y_pred = np.asarray([.4, .9, .12, .36, .3, .4]).reshape((2, 3))
    self.np_y_true = np.asarray([.5, .8, .12, .7, .43, .8]).reshape((2, 3))

    self.batch_size = 2
    self.expected_losses = np.multiply(self.np_y_true,
                                       np.log(self.np_y_true / self.np_y_pred))

    self.y_pred = constant_op.constant(self.np_y_pred, dtype=dtypes.float32)
    self.y_true = constant_op.constant(self.np_y_true)

  def test_config(self):
    k_obj = keras.losses.KLDivergence(
        reduction=losses_utils.ReductionV2.SUM, name='kld')
    self.assertEqual(k_obj.name, 'kld')
    self.assertEqual(k_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_unweighted(self):
    self.setup()
    k_obj = keras.losses.KLDivergence()

    loss = k_obj(self.y_true, self.y_pred)
    expected_loss = np.sum(self.expected_losses) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_scalar_weighted(self):
    self.setup()
    k_obj = keras.losses.KLDivergence()
    sample_weight = 2.3

    loss = k_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
    expected_loss = sample_weight * np.sum(
        self.expected_losses) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

    # Verify we get the same output when the same input is given
    loss_2 = k_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), self.evaluate(loss_2), 3)

  def test_sample_weighted(self):
    self.setup()
    k_obj = keras.losses.KLDivergence()
    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = k_obj(self.y_true, self.y_pred, sample_weight=sample_weight)

    expected_loss = np.multiply(
        self.expected_losses,
        np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape(2, 3))
    expected_loss = np.sum(expected_loss) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_timestep_weighted(self):
    self.setup()
    k_obj = keras.losses.KLDivergence()
    y_true = self.np_y_true.reshape(2, 3, 1)
    y_pred = self.np_y_pred.reshape(2, 3, 1)
    sample_weight = np.asarray([3, 6, 5, 0, 4, 2]).reshape(2, 3)
    expected_losses = np.sum(
        np.multiply(y_true, np.log(y_true / y_pred)), axis=-1)

    y_pred = constant_op.constant(y_pred, dtype=dtypes.float32)
    y_true = constant_op.constant(y_true)
    loss = k_obj(
        y_true, y_pred, sample_weight=constant_op.constant(sample_weight))

    num_timesteps = 3
    expected_loss = np.sum(expected_losses * sample_weight) / (
        self.batch_size * num_timesteps)
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_zero_weighted(self):
    self.setup()
    k_obj = keras.losses.KLDivergence()
    loss = k_obj(self.y_true, self.y_pred, sample_weight=0)
    self.assertAlmostEqual(self.evaluate(loss), 0., 3)


@test_util.run_all_in_graph_and_eager_modes
class HuberLossTest(test.TestCase):

  def huber_loss(self, y_true, y_pred, delta=1.0):
    error = y_pred - y_true
    abs_error = np.abs(error)

    quadratic = np.minimum(abs_error, delta)
    linear = np.subtract(abs_error, quadratic)
    return np.add(
        np.multiply(0.5, np.multiply(quadratic, quadratic)),
        np.multiply(delta, linear))

  def setup(self, delta=1.0):
    self.np_y_pred = np.asarray([.9, .2, .2, .8, .4, .6]).reshape((2, 3))
    self.np_y_true = np.asarray([1., 0., 1., 1., 0., 0.]).reshape((2, 3))

    self.batch_size = 6
    self.expected_losses = self.huber_loss(self.np_y_true, self.np_y_pred,
                                           delta)

    self.y_pred = constant_op.constant(self.np_y_pred)
    self.y_true = constant_op.constant(self.np_y_true)

  def test_config(self):
    h_obj = keras.losses.Huber(
        reduction=losses_utils.ReductionV2.SUM, name='huber')
    self.assertEqual(h_obj.name, 'huber')
    self.assertEqual(h_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_all_correct(self):
    self.setup()
    h_obj = keras.losses.Huber()
    loss = h_obj(self.y_true, self.y_true)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_unweighted(self):
    self.setup()
    h_obj = keras.losses.Huber()
    loss = h_obj(self.y_true, self.y_pred)
    actual_loss = np.sum(self.expected_losses) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), actual_loss, 3)

  def test_scalar_weighted(self):
    self.setup()
    h_obj = keras.losses.Huber()
    sample_weight = 2.3
    loss = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
    actual_loss = sample_weight * np.sum(self.expected_losses) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), actual_loss, 3)

    # Verify we get the same output when the same input is given
    loss_2 = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), self.evaluate(loss_2), 3)

  def test_sample_weighted(self):
    self.setup()
    h_obj = keras.losses.Huber()
    sample_weight = constant_op.constant((1.2, 3.4), shape=(2, 1))

    loss = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
    actual_loss = np.multiply(
        self.expected_losses,
        np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3)))
    actual_loss = np.sum(actual_loss) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), actual_loss, 3)

  def test_timestep_weighted(self):
    self.setup()
    h_obj = keras.losses.Huber()
    y_pred = self.np_y_pred.reshape((2, 3, 1))
    y_true = self.np_y_true.reshape((2, 3, 1))
    expected_losses = self.huber_loss(y_true, y_pred)

    y_pred = constant_op.constant(y_pred)
    y_true = constant_op.constant(y_true)
    sample_weight = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3, 1))
    loss = h_obj(
        y_true,
        y_pred,
        sample_weight=constant_op.constant(sample_weight, shape=(2, 3)))
    actual_loss = np.multiply(expected_losses, sample_weight)
    actual_loss = np.sum(actual_loss) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), actual_loss, 3)

  def test_zero_weighted(self):
    self.setup()
    h_obj = keras.losses.Huber()
    sample_weight = 0
    loss = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 0., 3)

  def test_non_default_delta(self):
    self.setup(delta=0.8)
    h_obj = keras.losses.Huber(delta=0.8)
    sample_weight = 2.3
    loss = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
    actual_loss = sample_weight * np.sum(self.expected_losses) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), actual_loss, 3)


if __name__ == '__main__':
  test.main()
