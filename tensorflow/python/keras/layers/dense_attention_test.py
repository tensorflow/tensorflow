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
"""Tests dense attention layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.layers import dense_attention
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class BaseDenseAttentionTest(test.TestCase):

  def test_one_dim_with_mask(self):
    # Scores tensor of shape [1, 1, 1]
    scores = np.array([[[1.1]]], dtype=np.float32)
    # Value tensor of shape [1, 1, 1]
    v = np.array([[[1.6]]], dtype=np.float32)
    # Scores mask tensor of shape [1, 1, 1]
    scores_mask = np.array([[[True]]], dtype=np.bool_)
    actual = dense_attention.BaseDenseAttention()._apply_scores(
        scores=scores, value=v, scores_mask=scores_mask)

    # Expected tensor of shape [1, 1, 1].
    # expected000 = softmax(scores)[0, 0] * 1.6 = 1.6
    expected = np.array([[[1.6]]], dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_one_dim_no_mask(self):
    # Scores tensor of shape [1, 1, 1]
    scores = np.array([[[1.1]]], dtype=np.float32)
    # Value tensor of shape [1, 1, 1]
    v = np.array([[[1.6]]], dtype=np.float32)
    actual = dense_attention.BaseDenseAttention()._apply_scores(
        scores=scores, value=v)

    # Expected tensor of shape [1, 1, 1].
    # expected000 = softmax(scores)[0, 0] * 1.6 = 1.6
    expected = np.array([[[1.6]]], dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_multi_dim_with_mask(self):
    # Scores tensor of shape [1, 1, 3]
    scores = np.array([[[1., 0., 1.]]], dtype=np.float32)
    # Value tensor of shape [1, 3, 1]
    v = np.array([[[1.6], [0.7], [-0.8]]], dtype=np.float32)
    # Scores mask tensor of shape [1, 1, 3]
    scores_mask = np.array([[[True, True, False]]], dtype=np.bool_)
    actual = dense_attention.BaseDenseAttention()._apply_scores(
        scores=scores, value=v, scores_mask=scores_mask)

    # Expected attention distribution = softmax(scores) with zeros in
    # positions where v_mask == False.
    # => attention_distribution000 = exp(1)/(exp(1) + exp(0)) = 0.73105857863
    #    attention_distribution001 = exp(0)/(exp(1) + exp(0)) = 0.26894142137
    #    attention_distribution002 = 0
    #
    # Expected tensor of shape [1, 1, 1].
    # expected000 = 0.73105857863 * 1.6 + 0.26894142137 * 0.7 - 0 * 0.8
    #             = 1.35795272077
    expected = np.array([[[1.35795272077]]], dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_multi_dim_no_mask(self):
    # Scores tensor of shape [1, 1, 3]
    scores = np.array([[[1., 0., 1.]]], dtype=np.float32)
    # Value tensor of shape [1, 3, 1]
    v = np.array([[[1.6], [0.7], [-0.8]]], dtype=np.float32)
    actual = dense_attention.BaseDenseAttention()._apply_scores(
        scores=scores, value=v)

    # Expected attention distribution = softmax(scores).
    # => attention_distribution000 = exp(1)/(exp(1) + exp(0) + exp(1))
    #                              = 0.42231879825
    #    attention_distribution001 = exp(0)/(exp(1) + exp(0) + exp(1))
    #                              = 0.15536240349
    #    attention_distribution002 = exp(1)/(exp(1) + exp(0) + exp(1))
    #                              = 0.42231879825
    #
    # Expected tensor of shape [1, 1, 1].
    # expected000 = 0.42231879825 * 1.6 + 0.15536240349 * 0.7
    #               - 0.42231879825 * 0.8
    #             = 0.44660872104
    expected = np.array([[[0.44660872104]]], dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_one_dim_batch_size_two(self):
    # Scores tensor of shape [2, 1, 1]
    scores = np.array([[[1.1]], [[2.1]]], dtype=np.float32)
    # Value tensor of shape [2, 1, 1]
    v = np.array([[[1.6]], [[2.6]]], dtype=np.float32)
    # Scpres mask tensor of shape [2, 1, 1]
    scores_mask = np.array([[[True]], [[True]]], dtype=np.bool_)
    actual = dense_attention.BaseDenseAttention()._apply_scores(
        scores=scores, value=v, scores_mask=scores_mask)

    # Expected tensor of shape [2, 1, 1].
    # expected000 = softmax(scores)[0, 0] * 1.6 = 1.6
    # expected100 = softmax(scores)[1, 0] * 2.6 = 2.6
    expected = np.array([[[1.6]], [[2.6]]], dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_shape_with_dropout(self):
    # scores: Scores float tensor of shape `[batch_size, tq, tv]`.
    # value: Value tensor of shape `[batch_size, tv, dim]`.
    batch_size = 4
    tq = 5
    tv = 6
    dim = 7
    scores = np.ones((batch_size, tq, tv))
    value = np.ones((batch_size, tv, dim))
    actual = dense_attention.BaseDenseAttention(dropout=0.1)._apply_scores(
        scores=scores, value=value, training=False)

    # Expected Tensor of shape `[batch_size, tq, dim]`.
    expected_shape = [batch_size, tq, dim]
    self.assertAllEqual(expected_shape, array_ops.shape(actual))

  def test_serialization(self):
    # Test serialization with causal
    layer = dense_attention.BaseDenseAttention(causal=True)

    config = keras.layers.serialize(layer)
    new_layer = keras.layers.deserialize(config)
    self.assertEqual(new_layer.causal, True)

    config = layer.get_config()
    new_layer = dense_attention.BaseDenseAttention.from_config(config)
    self.assertEqual(new_layer.causal, True)


@test_util.run_all_in_graph_and_eager_modes
class AttentionTest(test.TestCase):

  def test_calculate_scores_one_dim(self):
    # Query tensor of shape [1, 1, 1]
    q = np.array([[[1.1]]], dtype=np.float32)
    # Key tensor of shape [1, 1, 1]
    k = np.array([[[1.6]]], dtype=np.float32)
    attention_layer = dense_attention.Attention()
    attention_layer.build(input_shape=([1, 1, 1], [1, 1, 1]))
    actual = attention_layer._calculate_scores(query=q, key=k)

    # Expected tensor of shape [1, 1, 1].
    # expected000 = 1.1*1.6 = 1.76
    expected = np.array([[[1.76]]], dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_calculate_scores_multi_dim(self):
    # Query tensor of shape [1, 2, 4]
    q = np.array(
        [[[1., 1.1, 1.2, 1.3], [2., 2.1, 2.2, 2.3]]], dtype=np.float32)
    # Key tensor of shape [1, 3, 4]
    k = np.array(
        [[[1.5, 1.6, 1.7, 1.8], [2.5, 2.6, 2.7, 2.8], [3.5, 3.6, 3.7, 3.8]]],
        dtype=np.float32)
    attention_layer = dense_attention.Attention()
    attention_layer.build(input_shape=([1, 2, 4], [1, 3, 4]))
    actual = attention_layer._calculate_scores(query=q, key=k)

    # Expected tensor of shape [1, 2, 3].
    # expected000 = 1.*1.5+1.1*1.6+1.2*1.7+1.3*1.8 = 7.64
    # expected001 = 1.*2.5+1.1*2.6+1.2*2.7+1.3*2.8 = 12.24
    # expected002 = 1.*3.5+1.1*3.6+1.2*3.7+1.3*3.8 = 16.84
    # expected010 = 2.*1.5+2.1*1.6+2.2*1.7+2.3*1.8 = 14.24
    # expected011 = 2.*2.5+2.1*2.6+2.2*2.7+2.3*2.8 = 22.84
    # expected012 = 2.*3.5+2.1*3.6+2.2*3.7+2.3*3.8 = 31.44
    expected = np.array(
        [[[7.64, 12.24, 16.84], [14.24, 22.84, 31.44]]], dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_calculate_scores_one_dim_batch_size_two(self):
    # Query tensor of shape [2, 1, 1]
    q = np.array([[[1.1]], [[2.1]]], dtype=np.float32)
    # Key tensor of shape [2, 1, 1]
    k = np.array([[[1.6]], [[2.6]]], dtype=np.float32)
    attention_layer = dense_attention.Attention()
    attention_layer.build(input_shape=([2, 1, 1], [2, 1, 1]))
    actual = attention_layer._calculate_scores(query=q, key=k)

    # Expected tensor of shape [2, 1, 1].
    # expected000 = 1.1*1.6 = 1.76
    # expected100 = 2.1*2.6 = 5.46
    expected = np.array([[[1.76]], [[5.46]]], dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_calculate_scores_one_dim_with_scale(self):
    """Tests that scores are multiplied by scale."""
    # Query tensor of shape [1, 1, 1]
    q = np.array([[[1.1]]], dtype=np.float32)
    # Key tensor of shape [1, 1, 1]
    k = np.array([[[1.6]]], dtype=np.float32)
    attention_layer = dense_attention.Attention(use_scale=True)
    attention_layer.build(input_shape=([1, 1, 1], [1, 1, 1]))
    attention_layer.scale = -2.
    actual = attention_layer._calculate_scores(query=q, key=k)

    # Expected tensor of shape [1, 1, 1].
    # expected000 = -2*1.1*1.6 = -3.52
    expected = np.array([[[-3.52]]], dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_shape(self):
    # Query tensor of shape [1, 2, 4]
    q = np.array(
        [[[1., 1.1, 1.2, 1.3], [2., 2.1, 2.2, 2.3]]], dtype=np.float32)
    # Value tensor of shape [1, 3, 4]
    v = np.array(
        [[[1.5, 1.6, 1.7, 1.8], [2.5, 2.6, 2.7, 2.8], [3.5, 3.6, 3.7, 3.8]]],
        dtype=np.float32)
    # Value mask tensor of shape [1, 3]
    v_mask = np.array([[True, True, False]], dtype=np.bool_)
    attention_layer = dense_attention.Attention()
    actual = attention_layer([q, v], mask=[None, v_mask])

    expected_shape = [1, 2, 4]
    self.assertAllEqual(expected_shape, array_ops.shape(actual))

  def test_shape_with_key(self):
    # Query tensor of shape [1, 2, 4]
    q = np.array(
        [[[1., 1.1, 1.2, 1.3], [2., 2.1, 2.2, 2.3]]], dtype=np.float32)
    # Value tensor of shape [1, 3, 4]
    v = np.array(
        [[[1.5, 1.6, 1.7, 1.8], [2.5, 2.6, 2.7, 2.8], [3.5, 3.6, 3.7, 3.8]]],
        dtype=np.float32)
    # Key tensor of shape [1, 3, 4]
    k = np.array(
        [[[1.5, 1.6, 1.7, 1.8], [2.5, 2.6, 2.7, 2.8], [3.5, 3.6, 3.7, 3.8]]],
        dtype=np.float32)
    # Value mask tensor of shape [1, 3]
    v_mask = np.array([[True, True, False]], dtype=np.bool_)
    attention_layer = dense_attention.Attention()
    actual = attention_layer([q, v, k], mask=[None, v_mask])

    expected_shape = [1, 2, 4]
    self.assertAllEqual(expected_shape, array_ops.shape(actual))

  def test_multi_dim(self):
    # Query tensor of shape [1, 1, 1]
    q = np.array([[[1.1]]], dtype=np.float32)
    # Value tensor of shape [1, 3, 1]
    v = np.array([[[1.6], [0.7], [-0.8]]], dtype=np.float32)
    # Value mask tensor of shape [1, 3]
    v_mask = np.array([[True, True, False]], dtype=np.bool_)
    attention_layer = dense_attention.Attention()
    actual = attention_layer([q, v], mask=[None, v_mask])

    # Expected scores of shape [1, 1, 3]
    # scores = [[[1.1*1.6, 1.1*0.7, -1.1*0.8]]] = [[[1.76, 0.77, -0.88]]]
    # Expected attention distribution = softmax(scores) with zeros in
    # positions where v_mask == False.
    # => attention_distribution000 = exp(1.76)/(exp(1.76) + exp(0.77))
    #                              = 0.72908792234
    #    attention_distribution001 = exp(0.77)/(exp(1.76) + exp(0.77))
    #                              = 0.27091207765
    #    attention_distribution002 = 0
    #
    # Expected tensor of shape [1, 1, 1].
    # expected000 = 0.72908792234 * 1.6 + 0.27091207765 * 0.7 - 0 * 0.8
    #             = 1.3561791301
    expected = np.array([[[1.3561791301]]], dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_multi_dim_with_key(self):
    # Query tensor of shape [1, 1, 1]
    q = np.array([[[1.1]]], dtype=np.float32)
    # Value tensor of shape [1, 3, 1]
    v = np.array([[[0.5], [0.8], [-0.3]]], dtype=np.float32)
    # Key tensor of shape [1, 3, 1]
    k = np.array([[[1.6], [0.7], [-0.8]]], dtype=np.float32)
    # Value mask tensor of shape [1, 3]
    v_mask = np.array([[True, True, False]], dtype=np.bool_)
    attention_layer = dense_attention.Attention()
    actual = attention_layer([q, v, k], mask=[None, v_mask])

    # Expected scores of shape [1, 1, 3]
    # scores = [[[1.1*1.6, 1.1*0.7, -1.1*0.8]]] = [[[1.76, 0.77, -0.88]]]
    # Expected attention distribution = softmax(scores) with zeros in
    # positions where v_mask == False.
    # => attention_distribution000 = exp(1.76)/(exp(1.76) + exp(0.77))
    #                              = 0.72908792234
    #    attention_distribution001 = exp(0.77)/(exp(1.76) + exp(0.77))
    #                              = 0.27091207765
    #    attention_distribution002 = 0
    #
    # Expected tensor of shape [1, 1, 1].
    # expected000 = 0.72908792234 * 0.5 + 0.27091207765 * 0.8 - 0 * 0.3
    #             = 0.58127362329
    expected = np.array([[[0.58127362329]]], dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_multi_dim_with_query_mask(self):
    # Query tensor of shape [1, 2, 1]
    q = np.array([[[1.1], [-0.5]]], dtype=np.float32)
    # Value tensor of shape [1, 3, 1]
    v = np.array([[[1.6], [0.7], [-0.8]]], dtype=np.float32)
    # Query mask tensor of shape [1, 2]
    q_mask = np.array([[True, False]], dtype=np.bool_)
    # Value mask tensor of shape [1, 3]
    v_mask = np.array([[True, True, False]], dtype=np.bool_)
    attention_layer = dense_attention.Attention()
    actual = attention_layer([q, v], mask=[q_mask, v_mask])

    # Expected scores of shape [1, 2, 3]
    # scores = [[[1.1*1.6, 1.1*0.7, -1.1*0.8], [-0.5*1.6, -0.5*0.7, 0.5*0.8]]]
    #        = [[[1.76, 0.77, -0.88], [-0.8, -0.35, 0.4]]]
    # Expected attention distribution = softmax(scores) with zeros in
    # positions where v_mask == False.
    # => attention_distribution000 = exp(1.76)/(exp(1.76) + exp(0.77))
    #                              = 0.72908792234
    #    attention_distribution001 = exp(0.77)/(exp(1.76) + exp(0.77))
    #                              = 0.27091207765
    #    attention_distribution002 = 0
    # => attention_distribution010 = exp(-0.8)/(exp(-0.8) + exp(-0.35))
    #                              = 0.38936076605
    #    attention_distribution011 = exp(-0.35)/(exp(-0.8) + exp(-0.35))
    #                              = 0.61063923394
    #    attention_distribution012 = 0
    #
    # Expected tensor of shape [1, 2, 1] with zeros where  q_mask == False.
    # expected000 = 0.72908792234 * 1.6 + 0.27091207765 * 0.7 - 0 * 0.8
    #             = 1.3561791301
    # expected000 = 0
    expected = np.array([[[1.3561791301], [0.]]], dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_scale_None(self):
    """Tests that scale is None by default."""
    attention_layer = dense_attention.Attention()
    attention_layer.build(input_shape=([1, 1, 1], [1, 1, 1]))
    self.assertIsNone(attention_layer.scale)

  def test_scale_init_eager(self):
    """Tests that scale initializes to 1 when use_scale=True."""
    with context.eager_mode():
      attention_layer = dense_attention.Attention(use_scale=True)
      attention_layer.build(input_shape=([1, 1, 1], [1, 1, 1]))
      self.assertAllClose(1., attention_layer.scale.value())

  @test_util.deprecated_graph_mode_only
  def test_scale_init_graph(self):
    """Tests that scale initializes to 1 when use_scale=True."""
    with self.cached_session() as sess:
      attention_layer = dense_attention.Attention(use_scale=True)
      attention_layer.build(input_shape=([1, 1, 1], [1, 1, 1]))
      sess.run(attention_layer.scale.initializer)
      self.assertAllClose(1., attention_layer.scale.value())

  def test_self_attention_causal(self):
    # Query-value tensor of shape [1, 3, 1]
    q = np.array([[[0.5], [0.8], [-0.3]]], dtype=np.float32)
    attention_layer = dense_attention.Attention(causal=True)
    actual = attention_layer([q, q])

    # Expected scores of shape [1, 3, 3]
    # scores = [[0.25, 0.4, -0.15], [0.4, 0.64, -0.24], [-0.15, -0.24, 0.09]]
    # Expected attention distribution = softmax(scores) lower triangular
    # => attention_distribution00 = [1., 0., 0.]
    #    attention_distribution01
    #      = [exp(0.4), exp(0.64), 0.] / (exp(0.4) + exp(0.64))
    #      = [0.44028635073, 0.55971364926, 0.]
    #    attention_distribution02
    #      = [exp(-0.15), exp(-0.24), exp(0.09)]
    #        / (exp(-0.15) + exp(-0.24) + exp(0.09))
    #      = [0.31395396638, 0.28693232061, 0.399113713]
    #
    # Expected tensor of shape [1, 3, 1].
    # expected000 = 0.5
    # expected010 = 0.44028635073 * 0.5 + 0.55971364926 * 0.8
    #             = 0.66791409477
    # expected020 = 0.31395396638 * 0.5 +0.28693232061 * 0.8 -0.399113713 * 0.3
    #             = 0.26678872577
    expected = np.array(
        [[[0.5], [0.66791409477], [0.26678872577]]], dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_inputs_not_list(self):
    attention_layer = dense_attention.Attention()
    q = np.array([[[1.1]]], dtype=np.float32)
    with self.assertRaisesRegexp(
        ValueError, 'Attention layer must be called on a list of inputs'):
      attention_layer(q)

  def test_inputs_too_short(self):
    attention_layer = dense_attention.Attention()
    q = np.array([[[1.1]]], dtype=np.float32)
    with self.assertRaisesRegexp(
        ValueError,
        'Attention layer accepts inputs list of length 2 or 3'):
      attention_layer([q])

  def test_inputs_too_long(self):
    attention_layer = dense_attention.Attention()
    q = np.array([[[1.1]]], dtype=np.float32)
    with self.assertRaisesRegexp(
        ValueError,
        'Attention layer accepts inputs list of length 2 or 3'):
      attention_layer([q, q, q, q])

  def test_mask_not_list(self):
    attention_layer = dense_attention.Attention()
    q = np.array([[[1.1]]], dtype=np.float32)
    mask = np.array([[True]], dtype=np.bool_)
    with self.assertRaisesRegexp(
        ValueError, 'Attention layer mask must be a list'):
      attention_layer([q, q], mask=mask)

  def test_mask_too_short(self):
    attention_layer = dense_attention.Attention()
    q = np.array([[[1.1]]], dtype=np.float32)
    mask = np.array([[True]], dtype=np.bool_)
    with self.assertRaisesRegexp(
        ValueError, 'Attention layer mask must be a list of length 2'):
      attention_layer([q, q], mask=[mask])

  def test_mask_too_long(self):
    attention_layer = dense_attention.Attention()
    q = np.array([[[1.1]]], dtype=np.float32)
    mask = np.array([[True]], dtype=np.bool_)
    with self.assertRaisesRegexp(
        ValueError, 'Attention layer mask must be a list of length 2'):
      attention_layer([q, q], mask=[mask, mask, mask])

  def test_override_mask(self):
    attention_layer = dense_attention.Attention()
    q = core.Masking()(np.array([[[1.1]]], dtype=np.float32))
    mask = np.array([[False]], dtype=np.bool_)
    actual = attention_layer([q, q], mask=[mask, mask])
    self.assertAllClose([[[0]]], actual)

  def test_implicit_mask(self):
    attention_layer = dense_attention.Attention()
    q = core.Masking(1.1)(np.array([[[1.1], [1]]], dtype=np.float32))
    v = core.Masking(1.2)(np.array([[[1.2], [1]]], dtype=np.float32))
    actual = attention_layer([q, v])
    self.assertAllClose([[[0], [1]]], actual)

  def test_serialization(self):
    # Test serialization with use_scale
    layer = dense_attention.Attention(use_scale=True)

    config = keras.layers.serialize(layer)
    new_layer = keras.layers.deserialize(config)
    self.assertEqual(new_layer.use_scale, True)

    config = layer.get_config()
    new_layer = dense_attention.Attention.from_config(config)
    self.assertEqual(new_layer.use_scale, True)


@test_util.run_all_in_graph_and_eager_modes
class AdditiveAttentionTest(test.TestCase):

  def test_calculate_scores_one_dim(self):
    # Query tensor of shape [1, 1, 1]
    q = np.array([[[1.1]]], dtype=np.float32)
    # Key tensor of shape [1, 1, 1]
    k = np.array([[[1.6]]], dtype=np.float32)
    attention_layer = dense_attention.AdditiveAttention()
    attention_layer.build(input_shape=([1, 1, 1], [1, 1, 1]))
    # Scale tensor of shape [1]
    attention_layer.scale = np.array([[[0.5]]], dtype=np.float32)
    actual = attention_layer._calculate_scores(query=q, key=k)

    # Expected tensor of shape [1, 1, 1].
    # expected000 = 0.5 * tanh(1.1 + 1.6) = 0.49550372683
    expected = np.array([[[0.49550372683]]], dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_calculate_scores_multi_dim(self):
    # Query tensor of shape [1, 2, 4]
    q = np.array(
        [[[1., 1.1, 1.2, 1.3], [2., 2.1, 2.2, 2.3]]], dtype=np.float32)
    # Key tensor of shape [1, 3, 4]
    k = np.array(
        [[[1.5, 1.6, 1.7, 1.8], [2.5, 2.6, 2.7, 2.8], [3.5, 3.6, 3.7, 3.8]]],
        dtype=np.float32)
    attention_layer = dense_attention.AdditiveAttention()
    attention_layer.build(input_shape=([1, 2, 4], [1, 3, 4]))
    # Scale tensor of shape [4]
    attention_layer.scale = np.array([[[0.5, 0.6, 0.7, 0.8]]], dtype=np.float32)
    actual = attention_layer._calculate_scores(query=q, key=k)

    # pylint:disable=line-too-long
    # expected000 = 0.5*tanh(1.+1.5) + 0.6*tanh(1.1+1.6) + 0.7*tanh(1.2+1.7) + 0.8*tanh(1.3+1.8) = 2.58044532581
    # expected001 = 0.5*tanh(1.+2.5) + 0.6*tanh(1.1+2.6) + 0.7*tanh(1.2+2.7) + 0.8*tanh(1.3+2.8) = 2.59734317449
    # expected002 = 0.5*tanh(1.+3.5) + 0.6*tanh(1.1+3.6) + 0.7*tanh(1.2+3.7) + 0.8*tanh(1.3+3.8) = 2.59964024652
    # expected010 = 0.5*tanh(2.+1.5) + 0.6*tanh(2.1+1.6) + 0.7*tanh(2.2+1.7) + 0.8*tanh(2.3+1.8) = 2.59734317449
    # expected011 = 0.5*tanh(2.+2.5) + 0.6*tanh(2.1+2.6) + 0.7*tanh(2.2+2.7) + 0.8*tanh(2.3+2.8) = 2.59964024652
    # expected012 = 0.5*tanh(2.+3.5) + 0.6*tanh(2.1+3.6) + 0.7*tanh(2.2+3.7) + 0.8*tanh(2.3+3.8) = 2.59995130916
    # pylint:enable=line-too-long
    expected = np.array(
        [[[2.58044532581, 2.59734317449, 2.59964024652],
          [2.59734317449, 2.59964024652, 2.59995130916]]],
        dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_calculate_scores_one_dim_batch_size_two(self):
    # Query tensor of shape [2, 1, 1]
    q = np.array([[[1.1]], [[2.1]]], dtype=np.float32)
    # Key tensor of shape [2, 1, 1]
    k = np.array([[[1.6]], [[2.6]]], dtype=np.float32)
    attention_layer = dense_attention.AdditiveAttention()
    attention_layer.build(input_shape=([2, 1, 1], [2, 1, 1]))
    # Scale tensor of shape [1]
    attention_layer.scale = np.array([[[0.5]]], dtype=np.float32)
    actual = attention_layer._calculate_scores(query=q, key=k)

    # Expected tensor of shape [2, 1, 1].
    # expected000 = 0.5 * tanh(1.1 + 1.6) = 0.49550372683
    # expected100 = 0.5 * tanh(2.1 + 2.6) = 0.49991728277
    expected = np.array(
        [[[0.49550372683]], [[0.49991728277]]], dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_shape(self):
    # Query tensor of shape [1, 2, 4]
    q = np.array(
        [[[1., 1.1, 1.2, 1.3], [2., 2.1, 2.2, 2.3]]], dtype=np.float32)
    # Value tensor of shape [1, 3, 4]
    v = np.array(
        [[[1.5, 1.6, 1.7, 1.8], [2.5, 2.6, 2.7, 2.8], [3.5, 3.6, 3.7, 3.8]]],
        dtype=np.float32)
    # Value mask tensor of shape [1, 3]
    v_mask = np.array([[True, True, False]], dtype=np.bool_)
    attention_layer = dense_attention.AdditiveAttention()
    actual = attention_layer([q, v], mask=[None, v_mask])

    expected_shape = [1, 2, 4]
    self.assertAllEqual(expected_shape, array_ops.shape(actual))

  def test_shape_no_scale(self):
    # Query tensor of shape [1, 2, 4]
    q = np.array(
        [[[1., 1.1, 1.2, 1.3], [2., 2.1, 2.2, 2.3]]], dtype=np.float32)
    # Value tensor of shape [1, 3, 4]
    v = np.array(
        [[[1.5, 1.6, 1.7, 1.8], [2.5, 2.6, 2.7, 2.8], [3.5, 3.6, 3.7, 3.8]]],
        dtype=np.float32)
    # Value mask tensor of shape [1, 3]
    v_mask = np.array([[True, True, False]], dtype=np.bool_)
    attention_layer = dense_attention.AdditiveAttention(use_scale=False)
    actual = attention_layer([q, v], mask=[None, v_mask])

    expected_shape = [1, 2, 4]
    self.assertAllEqual(expected_shape, array_ops.shape(actual))

  def test_shape_with_key(self):
    # Query tensor of shape [1, 2, 4]
    q = np.array(
        [[[1., 1.1, 1.2, 1.3], [2., 2.1, 2.2, 2.3]]], dtype=np.float32)
    # Value tensor of shape [1, 3, 4]
    v = np.array(
        [[[1.5, 1.6, 1.7, 1.8], [2.5, 2.6, 2.7, 2.8], [3.5, 3.6, 3.7, 3.8]]],
        dtype=np.float32)
    # Key tensor of shape [1, 3, 4]
    k = np.array(
        [[[1.5, 1.6, 1.7, 1.8], [2.5, 2.6, 2.7, 2.8], [3.5, 3.6, 3.7, 3.8]]],
        dtype=np.float32)
    # Value mask tensor of shape [1, 3]
    v_mask = np.array([[True, True, False]], dtype=np.bool_)
    attention_layer = dense_attention.AdditiveAttention()
    actual = attention_layer([q, v, k], mask=[None, v_mask])

    expected_shape = [1, 2, 4]
    self.assertAllEqual(expected_shape, array_ops.shape(actual))

  def test_multi_dim(self):
    # Query tensor of shape [1, 1, 1]
    q = np.array([[[1.1]]], dtype=np.float32)
    # Value tensor of shape [1, 3, 1]
    v = np.array([[[1.6], [0.7], [-0.8]]], dtype=np.float32)
    # Value mask tensor of shape [1, 3]
    v_mask = np.array([[True, True, False]], dtype=np.bool_)
    attention_layer = dense_attention.AdditiveAttention()
    attention_layer.build(input_shape=([1, 1, 1], [1, 3, 1]))
    # Scale tensor of shape [1]
    attention_layer.scale = np.array([[[0.5]]], dtype=np.float32)
    actual = attention_layer([q, v], mask=[None, v_mask])

    # pylint:disable=line-too-long
    # Expected scores of shape [1, 1, 3]
    # scores = [[[0.5 * tanh(1.1 + 1.6), 0.5 * tanh(1.1 + 0.7), 0.5 * tanh(1.1 - 0.8)]]]
    #        = [[[0.49550372683, 0.47340300642, 0.14565630622]]]
    # Expected attention distribution = softmax(scores) with zeros in
    # positions where v_mask == False.
    # => attention_distribution000
    #      = exp(0.49550372683)/(exp(0.49550372683) + exp(0.47340300642))
    #      = 0.50552495521
    #    attention_distribution001
    #      = exp(0.47340300642)/(exp(0.49550372683) + exp(0.47340300642))
    #      = 0.49447504478
    #    attention_distribution002 = 0
    #
    # Expected tensor of shape [1, 1, 1].
    # expected000 = 0.50552495521 * 1.6 + 0.49447504478 * 0.7 - 0 * 0.8
    #             = 1.15497245968
    # pylint:enable=line-too-long
    expected = np.array([[[1.15497245968]]], dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_multi_dim_with_key(self):
    # Query tensor of shape [1, 1, 1]
    q = np.array([[[1.1]]], dtype=np.float32)
    # Value tensor of shape [1, 3, 1]
    v = np.array([[[0.5], [0.8], [-0.3]]], dtype=np.float32)
    # Key tensor of shape [1, 3, 1]
    k = np.array([[[1.6], [0.7], [-0.8]]], dtype=np.float32)
    # Value mask tensor of shape [1, 3]
    v_mask = np.array([[True, True, False]], dtype=np.bool_)
    attention_layer = dense_attention.AdditiveAttention()
    attention_layer.build(input_shape=([1, 1, 1], [1, 3, 1]))
    # Scale tensor of shape [1]
    attention_layer.scale = np.array([[[0.5]]], dtype=np.float32)
    actual = attention_layer([q, v, k], mask=[None, v_mask])

    # pylint:disable=line-too-long
    # Expected scores of shape [1, 1, 3]
    # scores = [[[0.5 * tanh(1.1 + 1.6), 0.5 * tanh(1.1 + 0.7), 0.5 * tanh(1.1 - 0.8)]]]
    #        = [[[0.49550372683, 0.47340300642, 0.14565630622]]]
    # Expected attention distribution = softmax(scores) with zeros in
    # positions where v_mask == False.
    # => attention_distribution000
    #        = exp(0.49550372683)/(exp(0.49550372683) + exp(0.47340300642))
    #        = 0.50552495521
    #    attention_distribution001
    #        = exp(0.47340300642)/(exp(0.49550372683) + exp(0.47340300642))
    #        = 0.49447504478
    #    attention_distribution002 = 0
    #
    # Expected tensor of shape [1, 1, 1].
    # expected000 = 0.50552495521 * 0.5 + 0.49447504478 * 0.8 - 0 * 0.3
    #             = 0.64834251342
    # pylint:enable=line-too-long
    expected = np.array([[[0.64834251342]]], dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_multi_dim_with_query_mask(self):
    # Query tensor of shape [1, 2, 1]
    q = np.array([[[1.1], [-0.5]]], dtype=np.float32)
    # Value tensor of shape [1, 3, 1]
    v = np.array([[[1.6], [0.7], [-0.8]]], dtype=np.float32)
    # Query mask tensor of shape [1, 2]
    q_mask = np.array([[True, False]], dtype=np.bool_)
    # Value mask tensor of shape [1, 3]
    v_mask = np.array([[True, True, False]], dtype=np.bool_)
    attention_layer = dense_attention.AdditiveAttention()
    attention_layer.build(input_shape=([1, 1, 1], [1, 3, 1]))
    # Scale tensor of shape [1]
    attention_layer.scale = np.array([[[0.5]]], dtype=np.float32)
    actual = attention_layer([q, v], mask=[q_mask, v_mask])

    # pylint:disable=line-too-long
    # Expected scores of shape [1, 2, 3]
    # scores = [[[0.5 * tanh(1.1 + 1.6), 0.5 * tanh(1.1 + 0.7), 0.5 * tanh(1.1 - 0.8)],
    #            [0.5 * tanh(-0.5 + 1.6), 0.5 * tanh(-0.5 + 0.7), 0.5 * tanh(-0.5 - 0.8)]]]
    #        = [[[0.49550372683, 0.47340300642, 0.14565630622],
    #            [0.40024951088, 0.09868766011, -0.43086157965]]]
    # Expected attention distribution = softmax(scores) with zeros in
    # positions where v_mask == False.
    # => attention_distribution000
    #        = exp(0.49550372683)/(exp(0.49550372683) + exp(0.47340300642))
    #        = 0.50552495521
    #    attention_distribution001
    #        = exp(0.47340300642)/(exp(0.49550372683) + exp(0.47340300642))
    #        = 0.49447504478
    #    attention_distribution002 = 0
    # => attention_distribution010
    #        = exp(0.40024951088)/(exp(0.40024951088) + exp(0.09868766011))
    #        = 0.57482427975
    #    attention_distribution011
    #        = exp(0.09868766011)/(exp(0.40024951088) + exp(0.09868766011))
    #        = 0.42517572025
    #    attention_distribution012 = 0
    #
    # Expected tensor of shape [1, 2, 1] with zeros where  q_mask == False.
    # expected000 = 0.50552495521 * 1.6 + 0.49447504478 * 0.7 - 0 * 0.8
    #             = 1.15497245968
    # expected000 = 0
    # pylint:enable=line-too-long
    expected = np.array([[[1.15497245968], [0.]]], dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_serialization(self):
    # Test serialization with use_scale
    layer = dense_attention.AdditiveAttention(use_scale=True)

    config = keras.layers.serialize(layer)
    new_layer = keras.layers.deserialize(config)
    self.assertEqual(new_layer.use_scale, True)

    config = layer.get_config()
    new_layer = dense_attention.AdditiveAttention.from_config(config)
    self.assertEqual(new_layer.use_scale, True)


@test_util.run_all_in_graph_and_eager_modes
class LowerTriangularMaskTest(test.TestCase):

  def test_square_shape(self):
    actual = dense_attention._lower_triangular_mask([3, 3])
    expected = np.array(
        [[True, False, False], [True, True, False], [True, True, True]],
        dtype=np.bool_)
    self.assertAllEqual(expected, actual)

  def test_orthogonal_shape(self):
    actual = dense_attention._lower_triangular_mask([3, 2])
    expected = np.array(
        [[True, False], [True, True], [True, True]], dtype=np.bool_)
    self.assertAllEqual(expected, actual)

  def test_three_dim(self):
    actual = dense_attention._lower_triangular_mask([1, 3, 3])
    expected = np.array(
        [[[True, False, False], [True, True, False], [True, True, True]]],
        dtype=np.bool_)
    self.assertAllEqual(expected, actual)


if __name__ == '__main__':
  test.main()
