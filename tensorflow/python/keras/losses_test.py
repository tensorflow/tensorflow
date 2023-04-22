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

from absl.testing import parameterized
import numpy as np

from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import combinations
from tensorflow.python.keras import losses
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test

ALL_LOSSES = [
    losses.mean_squared_error, losses.mean_absolute_error,
    losses.mean_absolute_percentage_error,
    losses.mean_squared_logarithmic_error, losses.squared_hinge, losses.hinge,
    losses.categorical_crossentropy, losses.binary_crossentropy,
    losses.kl_divergence, losses.poisson,
    losses.cosine_similarity, losses.log_cosh, losses.categorical_hinge
]


class KerasLossesTest(test.TestCase, parameterized.TestCase):

  def test_objective_shapes_3d(self):
    with self.cached_session():
      y_a = backend.variable(np.random.random((5, 6, 7)))
      y_b = backend.variable(np.random.random((5, 6, 7)))
      for obj in ALL_LOSSES:
        objective_output = obj(y_a, y_b)
        self.assertListEqual(objective_output.shape.as_list(), [5, 6])

  def test_objective_shapes_2d(self):
    with self.cached_session():
      y_a = backend.variable(np.random.random((6, 7)))
      y_b = backend.variable(np.random.random((6, 7)))
      for obj in ALL_LOSSES:
        objective_output = obj(y_a, y_b)
        self.assertListEqual(objective_output.shape.as_list(), [
            6,
        ])

  def test_cce_one_hot(self):
    with self.cached_session():
      y_a = backend.variable(np.random.randint(0, 7, (5, 6)))
      y_b = backend.variable(np.random.random((5, 6, 7)))
      objective_output = losses.sparse_categorical_crossentropy(y_a, y_b)
      assert backend.eval(objective_output).shape == (5, 6)

      y_a = backend.variable(np.random.randint(0, 7, (6,)))
      y_b = backend.variable(np.random.random((6, 7)))
      objective_output = losses.sparse_categorical_crossentropy(y_a, y_b)
      assert backend.eval(objective_output).shape == (6,)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_categorical_crossentropy_loss(self):
    target = backend.variable(np.random.randint(0, 1, (5, 1)))
    logits = backend.variable(np.random.random((5, 1)))
    softmax_output = backend.softmax(logits)
    output_from_logit = losses.categorical_crossentropy(
        target, logits, from_logits=True)
    output_from_softmax = losses.categorical_crossentropy(
        target, softmax_output)
    np.testing.assert_allclose(
        backend.eval(output_from_logit),
        backend.eval(output_from_softmax),
        atol=1e-5)

    axis = 0
    output_from_logit_axis = losses.categorical_crossentropy(
        target, logits, from_logits=True, axis=axis)
    output_from_softmax_axis = losses.categorical_crossentropy(
        target, softmax_output, axis=axis)

    np.testing.assert_allclose(
        backend.eval(output_from_logit_axis),
        backend.eval(output_from_softmax_axis),
        atol=1e-5)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_categorical_crossentropy_loss_with_unknown_rank_tensor(self):
    t = backend.placeholder()
    p = backend.placeholder()
    o = losses.categorical_crossentropy(t, p)

    t_val = ops.convert_to_tensor_v2_with_dispatch([[1., 0., 0.], [0., 1., 0.],
                                                    [0., 0., 1.]])
    p_val = ops.convert_to_tensor_v2_with_dispatch([[.9, .05, .05],
                                                    [.05, .89, .06],
                                                    [.05, .01, .94]])
    f = backend.function([t, p], o)

    result = f([t_val, p_val])
    self.assertArrayNear(result, [.105, .116, .062], 1e-3)

    # from logits
    p_val = ops.convert_to_tensor_v2_with_dispatch([[8., 1., 1.], [0., 9., 1.],
                                                    [2., 3., 5.]])
    o = losses.categorical_crossentropy(t, p, from_logits=True)
    f = backend.function([t, p], o)

    result = f([t_val, p_val])
    self.assertArrayNear(result, [.002, 0, .17], 1e-3)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_sparse_categorical_crossentropy_loss(self):
    target = backend.variable(np.random.randint(0, 1, (5, 1)))
    logits = backend.variable(np.random.random((5, 1)))
    softmax_output = backend.softmax(logits)
    output_from_logit = losses.sparse_categorical_crossentropy(
        target, logits, from_logits=True)
    output_from_softmax = losses.sparse_categorical_crossentropy(
        target, softmax_output)
    np.testing.assert_allclose(
        backend.eval(output_from_logit),
        backend.eval(output_from_softmax),
        atol=1e-5)

  @combinations.generate(combinations.combine(mode=['graph']))
  def test_sparse_categorical_crossentropy_loss_with_unknown_rank_tensor(self):
    # This test only runs in graph because the TF op layer is not supported yet
    # for sparse ops.
    t = backend.placeholder()
    p = backend.placeholder()
    o = losses.sparse_categorical_crossentropy(t, p)

    t_val = ops.convert_to_tensor_v2_with_dispatch([0, 1, 2])
    p_val = ops.convert_to_tensor_v2_with_dispatch([[.9, .05, .05],
                                                    [.05, .89, .06],
                                                    [.05, .01, .94]])
    f = backend.function([t, p], o)

    result = f([t_val, p_val])
    self.assertArrayNear(result, [.105, .116, .062], 1e-3)

    # from logits
    p_val = ops.convert_to_tensor_v2_with_dispatch([[8., 1., 1.], [0., 9., 1.],
                                                    [2., 3., 5.]])
    o = losses.sparse_categorical_crossentropy(t, p, from_logits=True)
    f = backend.function([t, p], o)

    result = f([t_val, p_val])
    self.assertArrayNear(result, [.002, 0, .17], 1e-3)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_binary_crossentropy_loss(self):
    target = backend.variable(np.random.randint(0, 1, (5, 1)))
    logits = backend.variable(np.random.random((5, 1)))
    sigmoid_output = backend.sigmoid(logits)
    output_from_logit = losses.binary_crossentropy(
        target, logits, from_logits=True)
    output_from_sigmoid = losses.binary_crossentropy(target, sigmoid_output)
    np.testing.assert_allclose(
        backend.eval(output_from_logit),
        backend.eval(output_from_sigmoid),
        atol=1e-5)

    axis = 0
    output_from_logit_axis = losses.binary_crossentropy(
        target, logits, from_logits=True, axis=axis)
    output_from_sigmoid_axis = losses.binary_crossentropy(
        target, sigmoid_output, axis=axis)

    np.testing.assert_allclose(
        backend.eval(output_from_logit_axis),
        backend.eval(output_from_sigmoid_axis),
        atol=1e-5)

  def test_get_bce(self):
    bce_fn = losses.get('bce')
    self.assertEqual(bce_fn, losses.binary_crossentropy)

  def test_serialization(self):
    fn = losses.get('mse')
    config = losses.serialize(fn)
    new_fn = losses.deserialize(config)
    self.assertEqual(fn, new_fn)

  def test_categorical_hinge(self):
    y_pred = backend.variable(np.array([[0.3, 0.2, 0.1], [0.1, 0.2, 0.7]]))
    y_true = backend.variable(np.array([[0, 1, 0], [1, 0, 0]]))
    expected_loss = ((0.3 - 0.2 + 1) + (0.7 - 0.1 + 1)) / 2.0
    loss = backend.eval(losses.categorical_hinge(y_true, y_pred))
    self.assertAllClose(expected_loss, np.mean(loss))

  def test_loss_wrapper(self):
    loss_fn = losses.get('mse')
    mse_obj = losses.LossFunctionWrapper(loss_fn, name=loss_fn.__name__)

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

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_loss_wrapper_autograph(self):
    # Test that functions with control flow wrapped in a LossFunctionWrapper
    # get autographed when in a tf.function
    def loss_fn(y_true, y_pred):
      mse_loss_fn = losses.get('mse')
      if math_ops.reduce_mean(y_true) > 0:
        return mse_loss_fn(y_true, y_pred)
      else:
        return mse_loss_fn(y_true, y_pred)

    mse_obj = losses.LossFunctionWrapper(loss_fn)

    y_true = constant_op.constant([[1., 9.], [2., 5.]])
    y_pred = constant_op.constant([[4., 8.], [12., 3.]])
    sample_weight = constant_op.constant([1.2, 0.5])

    @def_function.function
    def tf_functioned_loss_fn(y_true, y_pred, sample_weight=None):
      return mse_obj(y_true, y_pred, sample_weight=sample_weight)

    loss = tf_functioned_loss_fn(y_true, y_pred, sample_weight=sample_weight)

    # mse = [((4 - 1)^2 + (8 - 9)^2) / 2, ((12 - 2)^2 + (3 - 5)^2) / 2]
    # mse = [5, 52]
    # weighted_mse = [5 * 1.2, 52 * 0.5] = [6, 26]
    # reduced_weighted_mse = (6 + 26) / 2 =
    self.assertAllClose(self.evaluate(loss), 16, 1e-2)

  def test_invalid_reduction(self):
    with self.assertRaisesRegex(ValueError, 'Invalid Reduction Key Foo.'):
      losses.MeanSquaredError(reduction='Foo')

    mse_obj = losses.MeanSquaredError()
    y = constant_op.constant([1])
    mse_obj.reduction = 'Bar'
    with self.assertRaisesRegex(ValueError, 'Invalid Reduction Key Bar.'):
      mse_obj(y, y)

  def test_deserialization_error(self):
    with self.assertRaisesRegex(ValueError, 'Could not interpret loss'):
      losses.get(0)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_binary_crossentropy_uses_cached_logits(self):
    logits = constant_op.constant([[-30., 30.]])
    y_pred = activations.sigmoid(logits)
    self.assertTrue(hasattr(y_pred, '_keras_logits'))
    y_true = constant_op.constant([[0., 1.]])
    loss = losses.binary_crossentropy(y_true, y_pred)[0]
    # Check that logits are used. If y_pred is used directly, loss will
    # collapse to 0 from underflow.
    self.assertNotEqual(self.evaluate(loss), 0.)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_categorical_crossentropy_uses_cached_logits(self):
    logits = constant_op.constant([[-5., 0., 5.]])
    y_pred = activations.softmax(logits)
    self.assertTrue(hasattr(y_pred, '_keras_logits'))
    y_true = constant_op.constant([[0., 0., 1.]])
    loss = losses.categorical_crossentropy(y_true, logits, from_logits=True)[0]
    # Check that logits are used. If y_pred is used directly, loss will
    # collapse to 0 from underflow.
    self.assertNotEqual(self.evaluate(loss), 0.)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_sparse_categorical_crossentropy_uses_cached_logits(self):
    logits = constant_op.constant([[-5., 0., 5.]])
    y_pred = activations.softmax(logits)
    self.assertTrue(hasattr(y_pred, '_keras_logits'))
    y_true = constant_op.constant([2])
    loss = losses.sparse_categorical_crossentropy(
        y_true, logits, from_logits=True)[0]
    # Check that logits are used. If y_pred is used directly, loss will
    # collapse to 0 from underflow.
    self.assertNotEqual(self.evaluate(loss), 0.)

  @combinations.generate(combinations.combine(mode=['eager']))
  def test_loss_not_autographed_in_eager(self):

    class MyLoss(losses.Loss):

      def call(self, y_true, y_pred):
        return y_true - y_pred

    loss = MyLoss()
    y_true = constant_op.constant([[0., 0., 0.]])
    y_pred = constant_op.constant([[1., 1., 1.]])

    def tf_convert(fn, _):
      assert False, 'Function should not be autographed.'
      return fn

    with test.mock.patch.object(autograph, 'tf_convert', tf_convert):
      loss(y_true, y_pred)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class MeanSquaredErrorTest(test.TestCase):

  def test_config(self):
    mse_obj = losses.MeanSquaredError(
        reduction=losses_utils.ReductionV2.SUM, name='mse_1')
    self.assertEqual(mse_obj.name, 'mse_1')
    self.assertEqual(mse_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_all_correct_unweighted(self):
    mse_obj = losses.MeanSquaredError()
    y_true = constant_op.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
    loss = mse_obj(y_true, y_true)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_unweighted(self):
    mse_obj = losses.MeanSquaredError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mse_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), 49.5, 3)

  def test_scalar_weighted(self):
    mse_obj = losses.MeanSquaredError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mse_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), 113.85, 3)

  def test_sample_weighted(self):
    mse_obj = losses.MeanSquaredError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = mse_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 767.8 / 6, 3)

  def test_ragged_tensors(self):
    mse_obj = losses.MeanSquaredError()

    y_true = ragged_factory_ops.constant([[1., 1., 9.], [2., 5.]])
    y_pred = ragged_factory_ops.constant([[4., 1., 8.], [12., 3.]])
    sample_weight = constant_op.constant([1.2, 0.5])
    loss = mse_obj(y_true, y_pred, sample_weight=sample_weight)

    # mse = [((4 - 1)^2 + (8 - 9)^2) / 3, ((12 - 2)^2 + (3 - 5)^2) / 2]
    # mse = [3.(3), 52]
    # weighted_mse = [3.(3) * 1.2, 52 * 0.5] = [4, 26]
    # reduced_weighted_mse = (4 + 26) / 2 =
    self.assertAllClose(self.evaluate(loss), 15, 1e-2)

  def test_timestep_weighted(self):
    mse_obj = losses.MeanSquaredError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3, 1),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([3, 6, 5, 0, 4, 2], shape=(2, 3))
    loss = mse_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 587 / 6, 3)

  def test_zero_weighted(self):
    mse_obj = losses.MeanSquaredError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mse_obj(y_true, y_pred, sample_weight=0)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_invalid_sample_weight(self):
    mse_obj = losses.MeanSquaredError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3], shape=(2, 3, 1))
    sample_weight = constant_op.constant([3, 6, 5, 0], shape=(2, 2))
    with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError),
                                (r'Incompatible shapes: \[2,3\] vs. \[2,2\]|'
                                 'Dimensions must be equal')):
      mse_obj(y_true, y_pred, sample_weight=sample_weight)

  def test_no_reduction(self):
    mse_obj = losses.MeanSquaredError(reduction=losses_utils.ReductionV2.NONE)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mse_obj(y_true, y_pred, sample_weight=2.3)
    loss = self.evaluate(loss)
    self.assertArrayNear(loss, [84.3333, 143.3666], 1e-3)

  def test_sum_reduction(self):
    mse_obj = losses.MeanSquaredError(reduction=losses_utils.ReductionV2.SUM)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mse_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), 227.69998, 3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class MeanAbsoluteErrorTest(test.TestCase):

  def test_config(self):
    mae_obj = losses.MeanAbsoluteError(
        reduction=losses_utils.ReductionV2.SUM, name='mae_1')
    self.assertEqual(mae_obj.name, 'mae_1')
    self.assertEqual(mae_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_all_correct_unweighted(self):
    mae_obj = losses.MeanAbsoluteError()
    y_true = constant_op.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
    loss = mae_obj(y_true, y_true)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_unweighted(self):
    mae_obj = losses.MeanAbsoluteError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mae_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), 5.5, 3)

  def test_scalar_weighted(self):
    mae_obj = losses.MeanAbsoluteError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mae_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), 12.65, 3)

  def test_sample_weighted(self):
    mae_obj = losses.MeanAbsoluteError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = mae_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 81.4 / 6, 3)

  def test_timestep_weighted(self):
    mae_obj = losses.MeanAbsoluteError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3, 1),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([3, 6, 5, 0, 4, 2], shape=(2, 3))
    loss = mae_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 83 / 6, 3)

  def test_zero_weighted(self):
    mae_obj = losses.MeanAbsoluteError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mae_obj(y_true, y_pred, sample_weight=0)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_invalid_sample_weight(self):
    mae_obj = losses.MeanAbsoluteError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3], shape=(2, 3, 1))
    sample_weight = constant_op.constant([3, 6, 5, 0], shape=(2, 2))
    with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError),
                                (r'Incompatible shapes: \[2,3\] vs. \[2,2\]|'
                                 'Dimensions must be equal')):
      mae_obj(y_true, y_pred, sample_weight=sample_weight)

  def test_no_reduction(self):
    mae_obj = losses.MeanAbsoluteError(reduction=losses_utils.ReductionV2.NONE)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mae_obj(y_true, y_pred, sample_weight=2.3)
    loss = self.evaluate(loss)
    self.assertArrayNear(loss, [10.7333, 14.5666], 1e-3)

  def test_sum_reduction(self):
    mae_obj = losses.MeanAbsoluteError(reduction=losses_utils.ReductionV2.SUM)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mae_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), 25.29999, 3)

  def test_ragged_tensor(self):
    mae_obj = losses.MeanAbsoluteError()
    y_true = ragged_factory_ops.constant([[1, 9, 2], [-5, -2]],
                                         dtype=dtypes.float32)
    y_pred = ragged_factory_ops.constant([[4, 8, 12], [8, 1]],
                                         dtype=dtypes.float32)
    # loss = [14/3, 16/2]
    sample_weight = constant_op.constant([1.2, 1.0], shape=(2, 1))
    loss = mae_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 6.8, 5)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class MeanAbsolutePercentageErrorTest(test.TestCase):

  def test_config(self):
    mape_obj = losses.MeanAbsolutePercentageError(
        reduction=losses_utils.ReductionV2.SUM, name='mape_1')
    self.assertEqual(mape_obj.name, 'mape_1')
    self.assertEqual(mape_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_all_correct_unweighted(self):
    mape_obj = losses.MeanAbsolutePercentageError()
    y_true = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mape_obj(y_true, y_true)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_unweighted(self):
    mape_obj = losses.MeanAbsolutePercentageError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mape_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), 211.8518, 3)

  def test_scalar_weighted(self):
    mape_obj = losses.MeanAbsolutePercentageError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mape_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), 487.259, 3)

  def test_sample_weighted(self):
    mape_obj = losses.MeanAbsolutePercentageError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = mape_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 422.8888, 3)

  def test_ragged_tensors(self):
    mape_obj = losses.MeanAbsolutePercentageError()
    y_true = ragged_factory_ops.constant([[1, 9, 2], [-5, -2]])
    y_pred = ragged_factory_ops.constant([[4, 8, 12], [8, 1]],
                                         dtype=dtypes.float32)
    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = mape_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 510.7222, 3)

  def test_timestep_weighted(self):
    mape_obj = losses.MeanAbsolutePercentageError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3, 1),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([3, 6, 5, 0, 4, 2], shape=(2, 3))
    loss = mape_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 694.4445, 3)

  def test_zero_weighted(self):
    mape_obj = losses.MeanAbsolutePercentageError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mape_obj(y_true, y_pred, sample_weight=0)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_no_reduction(self):
    mape_obj = losses.MeanAbsolutePercentageError(
        reduction=losses_utils.ReductionV2.NONE)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mape_obj(y_true, y_pred, sample_weight=2.3)
    loss = self.evaluate(loss)
    self.assertArrayNear(loss, [621.8518, 352.6666], 1e-3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class MeanSquaredLogarithmicErrorTest(test.TestCase):

  def test_config(self):
    msle_obj = losses.MeanSquaredLogarithmicError(
        reduction=losses_utils.ReductionV2.SUM, name='mape_1')
    self.assertEqual(msle_obj.name, 'mape_1')
    self.assertEqual(msle_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_unweighted(self):
    msle_obj = losses.MeanSquaredLogarithmicError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = msle_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), 1.4370, 3)

  def test_scalar_weighted(self):
    msle_obj = losses.MeanSquaredLogarithmicError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = msle_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), 3.3051, 3)

  def test_sample_weighted(self):
    msle_obj = losses.MeanSquaredLogarithmicError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = msle_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 3.7856, 3)

  def test_timestep_weighted(self):
    msle_obj = losses.MeanSquaredLogarithmicError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3, 1),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([3, 6, 5, 0, 4, 2], shape=(2, 3))
    loss = msle_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 2.6473, 3)

  def test_zero_weighted(self):
    msle_obj = losses.MeanSquaredLogarithmicError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = msle_obj(y_true, y_pred, sample_weight=0)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_ragged_tensors(self):
    msle_obj = losses.MeanSquaredLogarithmicError()
    y_true = ragged_factory_ops.constant([[1, 9, 2], [-5, -2]])
    # log(max(y_true, 0) + 1): [[0.69314, 2.3025, 1.0986], [0., 0.]]
    y_pred = ragged_factory_ops.constant([[4, 8, 12], [8, 1]],
                                         dtype=dtypes.float32)
    # log(max(y_pred, 0) + 1): [[1.6094, 2.1972, 2.5649], [2.1972, 0.6932]]
    # per batch loss: [1.0002, 2.6541]
    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = msle_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 5.1121, 3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
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
    cosine_obj = losses.CosineSimilarity(
        axis=2, reduction=losses_utils.ReductionV2.SUM, name='cosine_loss')
    self.assertEqual(cosine_obj.name, 'cosine_loss')
    self.assertEqual(cosine_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_unweighted(self):
    self.setup()
    cosine_obj = losses.CosineSimilarity()
    loss = cosine_obj(self.y_true, self.y_pred)
    expected_loss = -np.mean(self.expected_loss)
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_scalar_weighted(self):
    self.setup()
    cosine_obj = losses.CosineSimilarity()
    sample_weight = 2.3
    loss = cosine_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
    expected_loss = -np.mean(self.expected_loss * sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_sample_weighted(self):
    self.setup()
    cosine_obj = losses.CosineSimilarity()
    sample_weight = np.asarray([1.2, 3.4])
    loss = cosine_obj(
        self.y_true,
        self.y_pred,
        sample_weight=constant_op.constant(sample_weight))
    expected_loss = -np.mean(self.expected_loss * sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_timestep_weighted(self):
    self.setup()
    cosine_obj = losses.CosineSimilarity()
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
    cosine_obj = losses.CosineSimilarity()
    loss = cosine_obj(self.y_true, self.y_pred, sample_weight=0)
    self.assertAlmostEqual(self.evaluate(loss), 0., 3)

  def test_axis(self):
    self.setup(axis=1)
    cosine_obj = losses.CosineSimilarity(axis=1)
    loss = cosine_obj(self.y_true, self.y_pred)
    expected_loss = -np.mean(self.expected_loss)
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class BinaryCrossentropyTest(test.TestCase):

  def test_config(self):
    bce_obj = losses.BinaryCrossentropy(
        reduction=losses_utils.ReductionV2.SUM, name='bce_1')
    self.assertEqual(bce_obj.name, 'bce_1')
    self.assertEqual(bce_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_all_correct_unweighted(self):
    y_true = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                  dtype=dtypes.float32)
    bce_obj = losses.BinaryCrossentropy()
    loss = bce_obj(y_true, y_true)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

    # Test with logits.
    logits = constant_op.constant([[100.0, -100.0, -100.0],
                                   [-100.0, 100.0, -100.0],
                                   [-100.0, -100.0, 100.0]])
    bce_obj = losses.BinaryCrossentropy(from_logits=True)
    loss = bce_obj(y_true, logits)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_unweighted(self):
    y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
    y_pred = np.asarray([1, 1, 1, 0], dtype=np.float32).reshape([2, 2])
    bce_obj = losses.BinaryCrossentropy()
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
    bce_obj = losses.BinaryCrossentropy(from_logits=True)
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
    bce_obj = losses.BinaryCrossentropy()
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
    bce_obj = losses.BinaryCrossentropy(from_logits=True)
    loss = bce_obj(y_true, logits, sample_weight=2.3)

    # Loss = max(x, 0) - x * z + log(1 + exp(-abs(x)))
    #            (where x = logits and z = y_true)
    # Loss = [(0 + 0 + 0) / 3, 200 / 3]
    # Weighted loss = [0 * 2.3, 66.666 * 2.3]
    # Reduced loss = (0 + 66.666 * 2.3) / 2

    self.assertAlmostEqual(self.evaluate(loss), 76.667, 3)

  def test_sample_weighted(self):
    bce_obj = losses.BinaryCrossentropy()
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
    bce_obj = losses.BinaryCrossentropy(from_logits=True)
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
    bce_obj = losses.BinaryCrossentropy(
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
    bce_obj = losses.BinaryCrossentropy(
        from_logits=True, label_smoothing=label_smoothing)
    loss = bce_obj(y_true, logits)
    expected_value = (100.0 + 50.0 * label_smoothing) / 3.0
    self.assertAlmostEqual(self.evaluate(loss), expected_value, 3)

  def test_ragged_tensors(self):
    bce_obj = losses.BinaryCrossentropy()
    y_true = ragged_factory_ops.constant([[1, 0, 1], [0]])
    y_pred = ragged_factory_ops.constant([[1, 1, 1], [0]], dtype=dtypes.float32)
    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = bce_obj(y_true, y_pred, sample_weight=sample_weight)

    # per batch loss = [ sum([0, 15.33, 0]) / 3, 0. ]
    #                = [ 5.11, 0]
    # Reduced loss = 5.11 * 1.2 / 2

    self.assertAlmostEqual(self.evaluate(loss), 3.0666, 3)

    # Test with logits.
    y_true = ragged_factory_ops.constant([[1, 0, 1], [0, 1]])
    logits = ragged_factory_ops.constant([[100.0, -100.0, 100.0],
                                          [100.0, 100.0]])
    weights = constant_op.constant([4, 3])
    bce_obj = losses.BinaryCrossentropy(from_logits=True)
    loss = bce_obj(y_true, logits, sample_weight=weights)

    # Loss = max(x, 0) - x * z + log(1 + exp(-abs(x)))
    #            (where x = logits and z = y_true)
    # Loss = [(0 + 0 + 0)/3, 100 / 2]
    # Weighted loss = [0 * 4, 50 * 3]
    # Reduced loss = (0 + 50 * 3) / 2

    self.assertAlmostEqual(self.evaluate(loss), 75., 3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class CategoricalCrossentropyTest(test.TestCase):

  def test_config(self):
    cce_obj = losses.CategoricalCrossentropy(
        reduction=losses_utils.ReductionV2.SUM, name='bce_1')
    self.assertEqual(cce_obj.name, 'bce_1')
    self.assertEqual(cce_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_all_correct_unweighted(self):
    y_true = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                  dtype=dtypes.int64)
    y_pred = constant_op.constant([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                                  dtype=dtypes.float32)
    cce_obj = losses.CategoricalCrossentropy()
    loss = cce_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

    # Test with logits.
    logits = constant_op.constant([[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]])
    cce_obj = losses.CategoricalCrossentropy(from_logits=True)
    loss = cce_obj(y_true, logits)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_unweighted(self):
    cce_obj = losses.CategoricalCrossentropy()
    y_true = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = constant_op.constant(
        [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]], dtype=dtypes.float32)
    loss = cce_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), .3239, 3)

    # Test with logits.
    logits = constant_op.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
    cce_obj = losses.CategoricalCrossentropy(from_logits=True)
    loss = cce_obj(y_true, logits)
    self.assertAlmostEqual(self.evaluate(loss), .0573, 3)

  def test_scalar_weighted(self):
    cce_obj = losses.CategoricalCrossentropy()
    y_true = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = constant_op.constant(
        [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]], dtype=dtypes.float32)
    loss = cce_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), .7449, 3)

    # Test with logits.
    logits = constant_op.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
    cce_obj = losses.CategoricalCrossentropy(from_logits=True)
    loss = cce_obj(y_true, logits, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), .1317, 3)

  def test_sample_weighted(self):
    cce_obj = losses.CategoricalCrossentropy()
    y_true = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = constant_op.constant(
        [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]], dtype=dtypes.float32)
    sample_weight = constant_op.constant([[1.2], [3.4], [5.6]], shape=(3, 1))
    loss = cce_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 1.0696, 3)

    # Test with logits.
    logits = constant_op.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
    cce_obj = losses.CategoricalCrossentropy(from_logits=True)
    loss = cce_obj(y_true, logits, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 0.31829, 3)

  def test_no_reduction(self):
    y_true = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    logits = constant_op.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
    cce_obj = losses.CategoricalCrossentropy(
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
    cce_obj = losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=label_smoothing)
    loss = cce_obj(y_true, logits)
    expected_value = 400.0 * label_smoothing / 3.0
    self.assertAlmostEqual(self.evaluate(loss), expected_value, 3)

  def test_shape_mismatch(self):
    y_true = constant_op.constant([[0], [1], [2]])
    y_pred = constant_op.constant([[.9, .05, .05], [.5, .89, .6],
                                   [.05, .01, .94]])

    cce_obj = losses.CategoricalCrossentropy()
    with self.assertRaisesRegex(ValueError, 'Shapes .+ are incompatible'):
      cce_obj(y_true, y_pred)

  def test_ragged_tensors(self):
    cce_obj = losses.CategoricalCrossentropy()
    y_true = ragged_factory_ops.constant([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1]]])
    y_pred = ragged_factory_ops.constant(
        [[[.9, .05, .05], [.5, .89, .6]], [[.05, .01, .94]]],
        dtype=dtypes.float32)
    # batch losses [[0.1054, 0.8047], [0.0619]]
    sample_weight = constant_op.constant([[1.2], [3.4]], shape=(2, 1))
    loss = cce_obj(y_true, y_pred, sample_weight=sample_weight)
    # sum([0.1054, 0.8047, 0.0619]) / 3
    self.assertAlmostEqual(self.evaluate(loss), 0.4341, 3)

    # Test with logits.
    logits = ragged_factory_ops.constant([[[8., 1., 1.], [0., 9., 1.]],
                                          [[2., 3., 5.]]])
    cce_obj = losses.CategoricalCrossentropy(from_logits=True)
    # batch losses [[0.0018, 0.0004], [0.1698]]
    loss = cce_obj(y_true, logits, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 0.1934, 3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class SparseCategoricalCrossentropyTest(test.TestCase):

  def test_config(self):
    cce_obj = losses.SparseCategoricalCrossentropy(
        reduction=losses_utils.ReductionV2.SUM, name='scc')
    self.assertEqual(cce_obj.name, 'scc')
    self.assertEqual(cce_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_all_correct_unweighted(self):
    y_true = constant_op.constant([[0], [1], [2]], dtype=dtypes.int64)
    y_pred = constant_op.constant([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                                  dtype=dtypes.float32)
    cce_obj = losses.SparseCategoricalCrossentropy()
    loss = cce_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

    # Test with logits.
    logits = constant_op.constant([[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]])
    cce_obj = losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = cce_obj(y_true, logits)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_unweighted(self):
    cce_obj = losses.SparseCategoricalCrossentropy()
    y_true = constant_op.constant([0, 1, 2])
    y_pred = constant_op.constant(
        [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]], dtype=dtypes.float32)
    loss = cce_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), .3239, 3)

    # Test with logits.
    logits = constant_op.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
    cce_obj = losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = cce_obj(y_true, logits)
    self.assertAlmostEqual(self.evaluate(loss), .0573, 3)

  def test_scalar_weighted(self):
    cce_obj = losses.SparseCategoricalCrossentropy()
    y_true = constant_op.constant([[0], [1], [2]])
    y_pred = constant_op.constant(
        [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]], dtype=dtypes.float32)
    loss = cce_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), .7449, 3)

    # Test with logits.
    logits = constant_op.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
    cce_obj = losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = cce_obj(y_true, logits, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), .1317, 3)

  def test_sample_weighted(self):
    cce_obj = losses.SparseCategoricalCrossentropy()
    y_true = constant_op.constant([[0], [1], [2]])
    y_pred = constant_op.constant(
        [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]], dtype=dtypes.float32)
    sample_weight = constant_op.constant([[1.2], [3.4], [5.6]], shape=(3, 1))
    loss = cce_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 1.0696, 3)

    # Test with logits.
    logits = constant_op.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
    cce_obj = losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = cce_obj(y_true, logits, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 0.31829, 3)

  def test_no_reduction(self):
    y_true = constant_op.constant([[0], [1], [2]])
    logits = constant_op.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
    cce_obj = losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=losses_utils.ReductionV2.NONE)
    loss = cce_obj(y_true, logits)
    self.assertAllClose((0.001822, 0.000459, 0.169846), self.evaluate(loss), 3)

  def test_non_tensor(self):
    # Test case for GitHub issue 33394.
    cce_obj = losses.SparseCategoricalCrossentropy()
    y_true = [[0], [1], [2]]
    y_pred = [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]]
    loss = cce_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), .7449, 3)

  def test_ragged_tensors(self):
    cce_obj = losses.SparseCategoricalCrossentropy()
    y_true = ragged_factory_ops.constant([[0, 1], [2]])
    y_pred = ragged_factory_ops.constant(
        [[[.9, .05, .05], [.5, .89, .6]], [[.05, .01, .94]]],
        dtype=dtypes.float32)
    # batch losses [[0.1054, 0.8047], [0.0619]]
    sample_weight = constant_op.constant([[1.2], [3.4]], shape=(2, 1))
    loss = cce_obj(y_true, y_pred, sample_weight=sample_weight)
    # sum([0.1054, 0.8047, 0.0619]) / 3
    self.assertAlmostEqual(self.evaluate(loss), 0.4341, 3)

    # Test with logits.
    logits = ragged_factory_ops.constant([[[8., 1., 1.], [0., 9., 1.]],
                                          [[2., 3., 5.]]])
    cce_obj = losses.SparseCategoricalCrossentropy(from_logits=True)
    # batch losses [[0.0018, 0.0004], [0.1698]]
    loss = cce_obj(y_true, logits, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 0.1934, 3)

  def test_ragged_tensors_rank_1(self):
    cce_obj = losses.SparseCategoricalCrossentropy()
    y_true = ragged_factory_ops.constant([[0, 1], [2]])
    y_pred = ragged_factory_ops.constant(
        [[[.9, .05, .05], [.5, .89, .6]], [[.05, .01, .94]]],
        ragged_rank=1,
        dtype=dtypes.float32)
    # batch losses [[0.1054, 0.8047], [0.0619]]
    sample_weight = constant_op.constant([[1.2], [3.4]], shape=(2, 1))
    loss = cce_obj(y_true, y_pred, sample_weight=sample_weight)
    # sum([0.1054, 0.8047, 0.0619]) / 3
    self.assertAlmostEqual(self.evaluate(loss), 0.4341, 3)

    # Test with logits.
    logits = ragged_factory_ops.constant(
        [[[8., 1., 1.], [0., 9., 1.]], [[2., 3., 5.]]], ragged_rank=1)
    cce_obj = losses.SparseCategoricalCrossentropy(from_logits=True)
    # batch losses [[0.0018, 0.0004], [0.1698]]
    loss = cce_obj(y_true, logits, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 0.1934, 3)

  def test_ragged_tensors_3d(self):
    # shape [2, 1, None]
    y_true = ragged_factory_ops.constant([[[1, 1]], [[0]]])
    # shape [2, 1, None, 2]
    y_pred = ragged_factory_ops.constant([[[[0.1, 0.9], [0.1, 0.9]]],
                                          [[[0.9, 0.1]]]])
    cce_obj = losses.SparseCategoricalCrossentropy()
    loss = cce_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), 0.1054, 3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class HingeTest(test.TestCase):

  def test_config(self):
    hinge_obj = losses.Hinge(
        reduction=losses_utils.ReductionV2.SUM, name='hinge_loss')
    self.assertEqual(hinge_obj.name, 'hinge_loss')
    self.assertEqual(hinge_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_unweighted(self):
    hinge_obj = losses.Hinge()
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
    hinge_obj = losses.Hinge()
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
    hinge_obj = losses.Hinge()
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
    hinge_obj = losses.Hinge()
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
    hinge_obj = losses.Hinge()
    y_true = constant_op.constant([[0, 1, 0, 1], [0, 0, 1, 1]])
    y_pred = constant_op.constant([[-0.3, 0.2, -0.1, 1.6],
                                   [-0.25, -1., 0.5, 0.6]])
    loss = hinge_obj(y_true, y_pred, sample_weight=0)
    self.assertAllClose(self.evaluate(loss), 0., 1e-3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class SquaredHingeTest(test.TestCase):

  def test_config(self):
    sq_hinge_obj = losses.SquaredHinge(
        reduction=losses_utils.ReductionV2.SUM, name='sq_hinge_loss')
    self.assertEqual(sq_hinge_obj.name, 'sq_hinge_loss')
    self.assertEqual(sq_hinge_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_unweighted(self):
    sq_hinge_obj = losses.SquaredHinge()
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
    sq_hinge_obj = losses.SquaredHinge()
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
    sq_hinge_obj = losses.SquaredHinge()
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
    sq_hinge_obj = losses.SquaredHinge()
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
    sq_hinge_obj = losses.SquaredHinge()
    y_true = constant_op.constant([[0, 1, 0, 1], [0, 0, 1, 1]])
    y_pred = constant_op.constant([[-0.3, 0.2, -0.1, 1.6],
                                   [-0.25, -1., 0.5, 0.6]])
    loss = sq_hinge_obj(y_true, y_pred, sample_weight=0)
    self.assertAllClose(self.evaluate(loss), 0., 1e-3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class CategoricalHingeTest(test.TestCase):

  def test_config(self):
    cat_hinge_obj = losses.CategoricalHinge(
        reduction=losses_utils.ReductionV2.SUM, name='cat_hinge_loss')
    self.assertEqual(cat_hinge_obj.name, 'cat_hinge_loss')
    self.assertEqual(cat_hinge_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_unweighted(self):
    cat_hinge_obj = losses.CategoricalHinge()
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
    cat_hinge_obj = losses.CategoricalHinge()
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
    cat_hinge_obj = losses.CategoricalHinge()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = cat_hinge_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 124.1, 3)

  def test_timestep_weighted(self):
    cat_hinge_obj = losses.CategoricalHinge()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3, 1),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([3, 6, 5, 0, 4, 2], shape=(2, 3))
    loss = cat_hinge_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 4.0, 3)

  def test_zero_weighted(self):
    cat_hinge_obj = losses.CategoricalHinge()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = cat_hinge_obj(y_true, y_pred, sample_weight=0)
    self.assertAlmostEqual(self.evaluate(loss), 0., 3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
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
    logcosh_obj = losses.LogCosh(
        reduction=losses_utils.ReductionV2.SUM, name='logcosh_loss')
    self.assertEqual(logcosh_obj.name, 'logcosh_loss')
    self.assertEqual(logcosh_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_unweighted(self):
    self.setup()
    logcosh_obj = losses.LogCosh()

    loss = logcosh_obj(self.y_true, self.y_pred)
    expected_loss = np.sum(self.expected_losses) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_scalar_weighted(self):
    self.setup()
    logcosh_obj = losses.LogCosh()
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
    logcosh_obj = losses.LogCosh()

    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = logcosh_obj(self.y_true, self.y_pred, sample_weight=sample_weight)

    expected_loss = np.multiply(
        self.expected_losses,
        np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3)))
    expected_loss = np.sum(expected_loss) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_timestep_weighted(self):
    self.setup()
    logcosh_obj = losses.LogCosh()
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
    logcosh_obj = losses.LogCosh()
    sample_weight = 0
    loss = logcosh_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 0., 3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
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
    poisson_obj = losses.Poisson(
        reduction=losses_utils.ReductionV2.SUM, name='poisson')
    self.assertEqual(poisson_obj.name, 'poisson')
    self.assertEqual(poisson_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_unweighted(self):
    self.setup()
    poisson_obj = losses.Poisson()

    loss = poisson_obj(self.y_true, self.y_pred)
    expected_loss = np.sum(self.expected_losses) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_scalar_weighted(self):
    self.setup()
    poisson_obj = losses.Poisson()
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
    poisson_obj = losses.Poisson()

    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = poisson_obj(self.y_true, self.y_pred, sample_weight=sample_weight)

    expected_loss = np.multiply(
        self.expected_losses,
        np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3)))
    expected_loss = np.sum(expected_loss) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_timestep_weighted(self):
    self.setup()
    poisson_obj = losses.Poisson()
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
    poisson_obj = losses.Poisson()
    loss = poisson_obj(self.y_true, self.y_pred, sample_weight=0)
    self.assertAlmostEqual(self.evaluate(loss), 0., 3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
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
    k_obj = losses.KLDivergence(
        reduction=losses_utils.ReductionV2.SUM, name='kld')
    self.assertEqual(k_obj.name, 'kld')
    self.assertEqual(k_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_unweighted(self):
    self.setup()
    k_obj = losses.KLDivergence()

    loss = k_obj(self.y_true, self.y_pred)
    expected_loss = np.sum(self.expected_losses) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_scalar_weighted(self):
    self.setup()
    k_obj = losses.KLDivergence()
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
    k_obj = losses.KLDivergence()
    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = k_obj(self.y_true, self.y_pred, sample_weight=sample_weight)

    expected_loss = np.multiply(
        self.expected_losses,
        np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape(2, 3))
    expected_loss = np.sum(expected_loss) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_timestep_weighted(self):
    self.setup()
    k_obj = losses.KLDivergence()
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
    k_obj = losses.KLDivergence()
    loss = k_obj(self.y_true, self.y_pred, sample_weight=0)
    self.assertAlmostEqual(self.evaluate(loss), 0., 3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
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
    h_obj = losses.Huber(reduction=losses_utils.ReductionV2.SUM, name='huber')
    self.assertEqual(h_obj.name, 'huber')
    self.assertEqual(h_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_all_correct(self):
    self.setup()
    h_obj = losses.Huber()
    loss = h_obj(self.y_true, self.y_true)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_unweighted(self):
    self.setup()
    h_obj = losses.Huber()
    loss = h_obj(self.y_true, self.y_pred)
    actual_loss = np.sum(self.expected_losses) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), actual_loss, 3)

  def test_scalar_weighted(self):
    self.setup()
    h_obj = losses.Huber()
    sample_weight = 2.3
    loss = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
    actual_loss = sample_weight * np.sum(self.expected_losses) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), actual_loss, 3)

    # Verify we get the same output when the same input is given
    loss_2 = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), self.evaluate(loss_2), 3)

  def test_sample_weighted(self):
    self.setup()
    h_obj = losses.Huber()
    sample_weight = constant_op.constant((1.2, 3.4), shape=(2, 1))

    loss = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
    actual_loss = np.multiply(
        self.expected_losses,
        np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3)))
    actual_loss = np.sum(actual_loss) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), actual_loss, 3)

  def test_timestep_weighted(self):
    self.setup()
    h_obj = losses.Huber()
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
    h_obj = losses.Huber()
    sample_weight = 0
    loss = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 0., 3)

  def test_non_default_delta(self):
    self.setup(delta=0.8)
    h_obj = losses.Huber(delta=0.8)
    sample_weight = 2.3
    loss = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
    actual_loss = sample_weight * np.sum(self.expected_losses) / self.batch_size
    self.assertAlmostEqual(self.evaluate(loss), actual_loss, 3)

  def test_loss_with_non_default_dtype(self):
    # Test case for GitHub issue:
    # https://github.com/tensorflow/tensorflow/issues/39004
    self.setup()
    h_obj = losses.Huber()
    try:
      backend.set_floatx('float64')
      loss = h_obj(self.y_true, self.y_true)
      self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)
    finally:
      backend.set_floatx('float32')


class BinaryTruePositivesViaControlFlow(losses.Loss):

  def __init__(self, reduction=losses_utils.ReductionV2.AUTO):
    super().__init__(reduction=reduction)

  def call(self, y_true, y_pred):
    y_true = math_ops.cast(y_true, dtypes.bool)
    y_pred = math_ops.cast(y_pred, dtypes.bool)

    result = constant_op.constant(0.0)
    for i in range(len(y_true)):
      for j in range(len(y_true[i])):
        if y_true[i][j] and y_pred[i][j]:
          result = result + 1
    return result


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class CustomLossTest(test.TestCase):

  def test_autograph(self):
    y_true = constant_op.constant([[0, 0.9, 0, 1, 0], [0, 0, 1, 1, 1],
                                   [1, 1, 1, 1, 0], [0, 0, 0, 0, 1.5]])
    y_pred = constant_op.constant([[0, 0, 1, 5, 0], [1, 1, 1, 1, 1],
                                   [0, 1, 0, 1, 0], [1, 10, 1, 1, 1]])

    @def_function.function
    def loss_fn(y_true, y_pred):
      loss_obj = BinaryTruePositivesViaControlFlow()
      return loss_obj(y_true, y_pred)

    loss = loss_fn(y_true, y_pred)
    self.assertAllEqual(
        self.evaluate(loss),
        7.0,
    )


if __name__ == '__main__':
  test.main()
