# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

"""Tests for boise_offset_converter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_text as tf_text

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class OffsetsToBoiseTagsTest(tf.test.TestCase):

  def test_ragged_input_2d(self):
    token_begin_offsets = tf.ragged.constant([[0, 4, 8, 12, 17], [0, 4, 8, 12]])
    token_end_offsets = tf.ragged.constant([[3, 7, 11, 16, 20], [3, 7, 11, 16]])
    span_begin_offsets = tf.ragged.constant([[4], [12]])
    span_end_offsets = tf.ragged.constant([[16], [16]])
    span_type = tf.ragged.constant([['animal'], ['loc']])

    result = tf_text.offsets_to_boise_tags(token_begin_offsets,
                                           token_end_offsets,
                                           span_begin_offsets, span_end_offsets,
                                           span_type)
    expected = tf.ragged.constant(
        [['O', 'B-animal', 'I-animal', 'E-animal', 'O'],
         ['O', 'O', 'O', 'S-loc']])
    self.assertAllEqual(result, expected)

  def test_ragged_input_2d_empty_span(self):
    token_begin_offsets = tf.ragged.constant([[0, 4, 8, 12, 17], [1],
                                              [0, 4, 8, 12]])
    token_end_offsets = tf.ragged.constant([[3, 7, 11, 16, 20], [2],
                                            [3, 7, 11, 16]])
    span_begin_offsets = tf.ragged.constant([[4], [], [12]])
    span_end_offsets = tf.ragged.constant([[16], [], [16]])
    span_type = tf.ragged.constant([['animal'], [], ['loc']])

    result = tf_text.offsets_to_boise_tags(token_begin_offsets,
                                           token_end_offsets,
                                           span_begin_offsets, span_end_offsets,
                                           span_type)
    expected = tf.ragged.constant(
        [['O', 'B-animal', 'I-animal', 'E-animal', 'O'], ['O'],
         ['O', 'O', 'O', 'S-loc']])
    self.assertAllEqual(result, expected)

  def test_ragged_input_2d_empty_token(self):
    token_begin_offsets = tf.ragged.constant([[0, 4, 8, 12, 17], [],
                                              [0, 4, 8, 12]])
    token_end_offsets = tf.ragged.constant([[3, 7, 11, 16, 20], [],
                                            [3, 7, 11, 16]])
    span_begin_offsets = tf.ragged.constant([[4], [1], [12]])
    span_end_offsets = tf.ragged.constant([[16], [2], [16]])
    span_type = tf.ragged.constant([['animal'], ['per'], ['loc']])

    result = tf_text.offsets_to_boise_tags(token_begin_offsets,
                                           token_end_offsets,
                                           span_begin_offsets, span_end_offsets,
                                           span_type)
    expected = tf.ragged.constant(
        [['O', 'B-animal', 'I-animal', 'E-animal', 'O'], [],
         ['O', 'O', 'O', 'S-loc']])
    self.assertAllEqual(result, expected)

  def test_ragged_input_2d_strict_boundary_mode(self):
    token_begin_offsets = tf.ragged.constant([[0, 4, 8, 12, 17], [0, 4, 8, 12]])
    token_end_offsets = tf.ragged.constant([[3, 7, 11, 16, 20], [3, 7, 11, 16]])
    span_begin_offsets = tf.ragged.constant([[4], [12]])
    span_end_offsets = tf.ragged.constant([[15], [16]])
    span_type = tf.ragged.constant([['animal'], ['loc']])

    result = tf_text.offsets_to_boise_tags(
        token_begin_offsets,
        token_end_offsets,
        span_begin_offsets,
        span_end_offsets,
        span_type,
        use_strict_boundary_mode=True)
    expected = tf.ragged.constant([['O', 'B-animal', 'I-animal', 'O', 'O'],
                                   ['O', 'O', 'O', 'S-loc']])
    self.assertAllEqual(result, expected)

  def test_ragged_input_1d(self):
    token_begin_offsets = tf.ragged.constant([0, 4, 8, 12, 17])
    token_end_offsets = tf.ragged.constant([3, 7, 11, 16, 20])
    span_begin_offsets = tf.ragged.constant([4])
    span_end_offsets = tf.ragged.constant([16])
    span_type = tf.ragged.constant(['animal'])

    result = tf_text.offsets_to_boise_tags(token_begin_offsets,
                                           token_end_offsets,
                                           span_begin_offsets, span_end_offsets,
                                           span_type)
    expected = tf.ragged.constant(
        ['O', 'B-animal', 'I-animal', 'E-animal', 'O'])
    self.assertAllEqual(result, expected)

  def test_ragged_input_0d(self):
    token_begin_offsets = tf.ragged.constant(0)
    token_end_offsets = tf.ragged.constant(3)
    span_begin_offsets = tf.ragged.constant(0)
    span_end_offsets = tf.ragged.constant(3)
    span_type = tf.ragged.constant('animal')

    result = tf_text.offsets_to_boise_tags(token_begin_offsets,
                                           token_end_offsets,
                                           span_begin_offsets, span_end_offsets,
                                           span_type)
    expected = tf.ragged.constant('S-animal')
    self.assertAllEqual(result, expected)

  def test_tensor_input_2d(self):
    token_begin_offsets = tf.constant([[0, 4, 8, 12, 17], [0, 4, 8, 12, 17]])
    token_end_offsets = tf.constant([[3, 7, 11, 16, 20], [3, 7, 11, 16, 19]])
    span_begin_offsets = tf.constant([[4], [12]])
    span_end_offsets = tf.constant([[16], [16]])
    span_type = tf.constant([['animal'], ['loc']])

    result = tf_text.offsets_to_boise_tags(token_begin_offsets,
                                           token_end_offsets,
                                           span_begin_offsets, span_end_offsets,
                                           span_type)
    expected = tf.ragged.constant(
        [['O', 'B-animal', 'I-animal', 'E-animal', 'O'],
         ['O', 'O', 'O', 'S-loc', 'O']])
    self.assertAllEqual(result, expected)

  def test_tensor_input_1d(self):
    token_begin_offsets = tf.constant([0, 4, 8, 12, 17])
    token_end_offsets = tf.constant([3, 7, 11, 16, 20])
    span_begin_offsets = tf.constant([4])
    span_end_offsets = tf.constant([16])
    span_type = tf.constant(['animal'])

    result = tf_text.offsets_to_boise_tags(token_begin_offsets,
                                           token_end_offsets,
                                           span_begin_offsets, span_end_offsets,
                                           span_type)
    expected = tf.ragged.constant(
        ['O', 'B-animal', 'I-animal', 'E-animal', 'O'])
    self.assertAllEqual(result, expected)

  def test_tensor_input_0d(self):
    token_begin_offsets = tf.constant(0)
    token_end_offsets = tf.constant(3)
    span_begin_offsets = tf.constant(0)
    span_end_offsets = tf.constant(3)
    span_type = tf.constant('animal')

    result = tf_text.offsets_to_boise_tags(token_begin_offsets,
                                           token_end_offsets,
                                           span_begin_offsets, span_end_offsets,
                                           span_type)
    expected = tf.ragged.constant('S-animal')
    self.assertAllEqual(result, expected)

  def test_ragged_input_tensor_type_error(self):
    token_begin_offsets = tf.ragged.constant([[0, 4, 8, 12, 17], [0, 4, 8, 12]])
    token_end_offsets = tf.ragged.constant([[3, 7, 11, 16, 20], [3, 7, 11, 16]])
    span_begin_offsets = tf.ragged.constant([[4], [12]])
    span_end_offsets = tf.ragged.constant([[16], [16]])
    span_type = tf.constant([['animal'], ['loc']])
    with self.assertRaises(ValueError):
      tf_text.offsets_to_boise_tags(token_begin_offsets, token_end_offsets,
                                    span_begin_offsets, span_end_offsets,
                                    span_type)

  def test_ragged_input_tensor_token_shape_error(self):
    token_begin_offsets = tf.ragged.constant([
        [0, 4, 8, 12, 17],
        [0, 4, 8, 12, 20],
        [0, 1, 12]  # this should cause error
    ])
    token_end_offsets = tf.ragged.constant([[3, 7, 11, 16, 20], [3, 7, 11, 16]])
    span_begin_offsets = tf.ragged.constant([[4], [12]])
    span_end_offsets = tf.ragged.constant([[16], [16]])
    span_type = tf.ragged.constant([['animal'], ['loc']])
    with self.assertRaises(ValueError):
      tf_text.offsets_to_boise_tags(token_begin_offsets, token_end_offsets,
                                    span_begin_offsets, span_end_offsets,
                                    span_type)

  # TODO(luyaoxu): uncomment the following two tests after figuring out why
  # assertRaises is not catching ValueError, while ValueError is raised.
  # def test_ragged_input_tensor_span_type_shape_error(self):
  #   token_begin_offsets = tf.ragged.constant([[0, 4, 8, 12, 17], [0, 4, 8]])
  #   token_end_offsets = tf.ragged.constant([[3, 7, 11, 16, 20], [3, 7, 11]])
  #   span_begin_offsets = tf.ragged.constant([[4], [12]])
  #   span_end_offsets = tf.ragged.constant([[16], [16]])
  #   span_type = tf.ragged.constant([['animal', 'this should cause error'],
  #                                   ['loc']])
  #   with self.assertRaises(ValueError):
  #     tf_text.offsets_to_boise_tags(token_begin_offsets, token_end_offsets,
  #                                   span_begin_offsets, span_end_offsets,
  #                                   span_type)

  # def test_ragged_input_tensor_row_splits_mismatch_error(self):
  #   token_begin_offsets = tf.ragged.constant([
  #       [0, 4, 8, 12, 17, 20],
  #       [0, 4],  # row lengths different from end offsets, causing error
  #   ])
  #   token_end_offsets = tf.ragged.constant([[3, 7, 11, 16, 20], [3, 7, 11]])
  #   span_begin_offsets = tf.ragged.constant([[4], [12]])
  #   span_end_offsets = tf.ragged.constant([[16], [16]])
  #   span_type = tf.ragged.constant([['animal'], ['loc']])
  #   with self.assertRaises(ValueError):
  #     tf_text.offsets_to_boise_tags(token_begin_offsets, token_end_offsets,
  #                                   span_begin_offsets, span_end_offsets,
  #                                   span_type)

  # TODO(luyaoxu): uncomment when the op is supported in tf.lite.
  # def testTfLite(self):
  #   """Checks TFLite conversion and inference."""

  #   class Model(tf.keras.Model):

  #     # def __init__(self, **kwargs):
  #     #   super().__init__(**kwargs)

  #     @tf.function(input_signature=[
  #         tf.TensorSpec(
  #             shape=[None], dtype=tf.int32, name='token_begin_offsets'),
  #         tf.TensorSpec(shape=[None], dtype=tf.int32,
  #                       name='token_end_offsets'),
  #         tf.TensorSpec(
  #             shape=[None], dtype=tf.int32, name='span_begin_offsets'),
  #         tf.TensorSpec(shape=[None], dtype=tf.int32,
  #                       name='span_end_offsets'),
  #         tf.TensorSpec(shape=[None], dtype=tf.string, name='span_type')
  #     ])
  #     def call(self, token_begin_offsets, token_end_offsets,
  #              span_begin_offsets,
  #              span_end_offsets, span_type):
  #       return {
  #           'result':
  #               tf_text.offsets_to_boise_tags(token_begin_offsets,
  #                                             token_end_offsets,
  #                                             span_begin_offsets,
  #                                             span_end_offsets, span_type)
  #       }

  #   # Test input data.
  #   # input_data = np.array(['Some minds are better kept apart'])
  #   token_begin_offsets = np.array([0, 4, 8, 12, 17], dtype=np.int32)
  #   token_end_offsets = np.array([3, 7, 11, 16, 20], dtype=np.int32)
  #   span_begin_offsets = np.array([4], dtype=np.int32)
  #   span_end_offsets = np.array([16], dtype=np.int32)
  #   span_type = np.array(['animal'])

  #   # Define a model.
  #   model = Model()
  #   # Do TF inference.
  #   tf_result = model(
  #       tf.constant(token_begin_offsets), tf.constant(token_end_offsets),
  #       tf.constant(span_begin_offsets), tf.constant(span_end_offsets),
  #       tf.constant(span_type))['result']

  #   # Convert to TFLite.
  #   converter = tf.lite.TFLiteConverter.from_keras_model(model)
  #   converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
  #   converter.allow_custom_ops = True
  #   tflite_model = converter.convert()

  #   # Do TFLite inference.
  #   interp = interpreter.InterpreterWithCustomOps(
  #       model_content=tflite_model,
  #       custom_op_registerers=tf_text.tflite_registrar.SELECT_TFTEXT_OPS)
  #   print(interp.get_signature_list())
  #   func_under_test = interp.get_signature_runner('serving_default')
  #   output = func_under_test(
  #       token_begin_offsets=token_begin_offsets,
  #       token_end_offsets=token_end_offsets,
  #       span_begin_offsets=span_begin_offsets,
  #       span_end_offsets=span_end_offsets,
  #       span_type=span_type)
  #   if tf.executing_eagerly():
  #     tflite_result = output['result']
  #   else:
  #     tflite_result = output['output_1']

  #   # Assert the results are identical.
  #   self.assertAllEqual(tflite_result, tf_result)


@test_util.run_all_in_graph_and_eager_modes
class BoiseTagsToOffsetsTest(tf.test.TestCase):

  def test_ragged_input_2d(self):
    token_begin_offsets = tf.ragged.constant([[0, 4, 8, 12, 17], [0, 4, 8, 12]])
    token_end_offsets = tf.ragged.constant([[3, 7, 11, 16, 20], [3, 7, 11, 16]])
    boise_tags = tf.ragged.constant(
        [['O', 'B-animal', 'I-animal', 'E-animal', 'O'],
         ['B-person', 'E-person', 'O', 'S-loc']])

    expected_span_begin_offsets = tf.ragged.constant([[4], [0, 12]])
    expected_span_end_offsets = tf.ragged.constant([[16], [7, 16]])
    expected_span_type = tf.ragged.constant([['animal'], ['person', 'loc']])
    (span_begin_offsets, span_end_offsets, span_type) = (
        tf_text.boise_tags_to_offsets(token_begin_offsets, token_end_offsets,
                                      boise_tags))

    self.assertAllEqual(span_begin_offsets, expected_span_begin_offsets)
    self.assertAllEqual(span_end_offsets, expected_span_end_offsets)
    self.assertAllEqual(span_type, expected_span_type)

  def test_ragged_input_2d_empty_span(self):
    token_begin_offsets = tf.ragged.constant([[0, 4, 8, 12, 17], [1],
                                              [0, 4, 8, 12]])
    token_end_offsets = tf.ragged.constant([[3, 7, 11, 16, 20], [2],
                                            [3, 7, 11, 16]])
    boise_tags = tf.ragged.constant(
        [['O', 'B-animal', 'I-animal', 'E-animal', 'O'], ['O'],
         ['O', 'O', 'O', 'S-loc']])

    expected_span_begin_offsets = tf.ragged.constant([[4], [], [12]])
    expected_span_end_offsets = tf.ragged.constant([[16], [], [16]])
    expected_span_type = tf.ragged.constant([['animal'], [], ['loc']])
    (span_begin_offsets, span_end_offsets, span_type) = (
        tf_text.boise_tags_to_offsets(token_begin_offsets, token_end_offsets,
                                      boise_tags))

    self.assertAllEqual(span_begin_offsets, expected_span_begin_offsets)
    self.assertAllEqual(span_end_offsets, expected_span_end_offsets)
    self.assertAllEqual(span_type, expected_span_type)

  def test_ragged_input_2d_empty_token(self):
    token_begin_offsets = tf.ragged.constant([[0, 4, 8, 12, 17], [],
                                              [0, 4, 8, 12]])
    token_end_offsets = tf.ragged.constant([[3, 7, 11, 16, 20], [],
                                            [3, 7, 11, 16]])
    boise_tags = tf.ragged.constant(
        [['O', 'B-animal', 'I-animal', 'E-animal', 'O'], [],
         ['O', 'O', 'O', 'S-loc']])

    expected_span_begin_offsets = tf.ragged.constant([[4], [], [12]])
    expected_span_end_offsets = tf.ragged.constant([[16], [], [16]])
    expected_span_type = tf.ragged.constant([['animal'], [], ['loc']])
    (span_begin_offsets, span_end_offsets, span_type) = (
        tf_text.boise_tags_to_offsets(token_begin_offsets, token_end_offsets,
                                      boise_tags))

    self.assertAllEqual(span_begin_offsets, expected_span_begin_offsets)
    self.assertAllEqual(span_end_offsets, expected_span_end_offsets)
    self.assertAllEqual(span_type, expected_span_type)

  def test_ragged_input_1d(self):
    token_begin_offsets = tf.ragged.constant([0, 4, 8, 12, 17])
    token_end_offsets = tf.ragged.constant([3, 7, 11, 16, 20])
    boise_tags = tf.ragged.constant(
        ['O', 'B-animal', 'I-animal', 'E-animal', 'O'])

    expected_span_begin_offsets = tf.ragged.constant([4])
    expected_span_end_offsets = tf.ragged.constant([16])
    expected_span_type = tf.ragged.constant(['animal'])
    (span_begin_offsets, span_end_offsets, span_type) = (
        tf_text.boise_tags_to_offsets(token_begin_offsets, token_end_offsets,
                                      boise_tags))

    self.assertAllEqual(span_begin_offsets, expected_span_begin_offsets)
    self.assertAllEqual(span_end_offsets, expected_span_end_offsets)
    self.assertAllEqual(span_type, expected_span_type)

  def test_ragged_input_0d(self):
    token_begin_offsets = tf.ragged.constant(0)
    token_end_offsets = tf.ragged.constant(3)
    boise_tags = tf.ragged.constant('S-animal')

    expected_span_begin_offsets = tf.ragged.constant(0)
    expected_span_end_offsets = tf.ragged.constant(3)
    expected_span_type = tf.ragged.constant('animal')
    (span_begin_offsets, span_end_offsets, span_type) = (
        tf_text.boise_tags_to_offsets(token_begin_offsets, token_end_offsets,
                                      boise_tags))

    self.assertAllEqual(span_begin_offsets, expected_span_begin_offsets)
    self.assertAllEqual(span_end_offsets, expected_span_end_offsets)
    self.assertAllEqual(span_type, expected_span_type)

  def test_tensor_input_2d(self):
    token_begin_offsets = tf.constant([[0, 4, 8, 12, 17], [0, 4, 8, 12, 17]])
    token_end_offsets = tf.constant([[3, 7, 11, 16, 20], [3, 7, 11, 16, 19]])
    boise_tags = tf.constant([['O', 'B-animal', 'I-animal', 'E-animal', 'O'],
                              ['O', 'O', 'O', 'S-loc', 'O']])

    expected_span_begin_offsets = tf.ragged.constant([[4], [12]])
    expected_span_end_offsets = tf.ragged.constant([[16], [16]])
    expected_span_type = tf.ragged.constant([['animal'], ['loc']])
    (span_begin_offsets, span_end_offsets, span_type) = (
        tf_text.boise_tags_to_offsets(token_begin_offsets, token_end_offsets,
                                      boise_tags))

    self.assertAllEqual(span_begin_offsets, expected_span_begin_offsets)
    self.assertAllEqual(span_end_offsets, expected_span_end_offsets)
    self.assertAllEqual(span_type, expected_span_type)

  def test_tensor_input_1d(self):
    token_begin_offsets = tf.constant([0, 4, 8, 12, 17])
    token_end_offsets = tf.constant([3, 7, 11, 16, 20])
    boise_tags = tf.constant(['O', 'B-animal', 'I-animal', 'E-animal', 'O'])

    expected_span_begin_offsets = tf.constant([4])
    expected_span_end_offsets = tf.constant([16])
    expected_span_type = tf.constant(['animal'])
    (span_begin_offsets, span_end_offsets, span_type) = (
        tf_text.boise_tags_to_offsets(token_begin_offsets, token_end_offsets,
                                      boise_tags))

    self.assertAllEqual(span_begin_offsets, expected_span_begin_offsets)
    self.assertAllEqual(span_end_offsets, expected_span_end_offsets)
    self.assertAllEqual(span_type, expected_span_type)

  def test_tensor_input_0d(self):
    token_begin_offsets = tf.constant(0)
    token_end_offsets = tf.constant(3)
    boise_tags = tf.constant('S-animal')

    expected_span_begin_offsets = tf.constant(0)
    expected_span_end_offsets = tf.constant(3)
    expected_span_type = tf.constant('animal')
    (span_begin_offsets, span_end_offsets, span_type) = (
        tf_text.boise_tags_to_offsets(token_begin_offsets, token_end_offsets,
                                      boise_tags))

    self.assertAllEqual(span_begin_offsets, expected_span_begin_offsets)
    self.assertAllEqual(span_end_offsets, expected_span_end_offsets)
    self.assertAllEqual(span_type, expected_span_type)

  def test_ragged_input_tensor_type_error(self):
    token_begin_offsets = tf.ragged.constant([[0, 4, 8, 12, 17],
                                              [0, 4, 8, 12, 17]])
    token_end_offsets = tf.ragged.constant([[3, 7, 11, 16, 20],
                                            [3, 7, 11, 16, 20]])
    # not same as token offsets tensor type, causing error
    boise_tags = tf.constant([['O', 'B-animal', 'I-animal', 'E-animal', 'O'],
                              ['O', 'O', 'O', 'S-loc', 'O']])
    with self.assertRaises(ValueError):
      tf_text.boise_tags_to_offsets(token_begin_offsets, token_end_offsets,
                                    boise_tags)


if __name__ == '__main__':
  test.main()
