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

# encoding=utf-8
"""Tests for ngram ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text

from tensorflow.lite.python import interpreter
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow_text.core.pybinds import tflite_registrar
from tensorflow_text.python.ops import ngrams_op


@test_util.run_all_in_graph_and_eager_modes
class NgramsOpTest(test_util.TensorFlowTestCase):

  def testSumReduction(self):
    test_data = constant_op.constant([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    op = ngrams_op.ngrams(
        test_data, width=2, axis=1, reduction_type=ngrams_op.Reduction.SUM)
    expected_values = [[3.0, 5.0], [30.0, 50.0]]

    self.assertAllEqual(expected_values, op)

  def testRaggedSumReduction(self):
    test_data = ragged_factory_ops.constant([[1.0, 2.0, 3.0, 4.0],
                                             [10.0, 20.0, 30.0]])
    op = ngrams_op.ngrams(
        test_data, width=2, axis=1, reduction_type=ngrams_op.Reduction.SUM)
    expected_values = [[3.0, 5.0, 7.0], [30.0, 50.0]]

    self.assertAllEqual(expected_values, op)

  def testRaggedSumReductionAxisZero(self):
    test_data = ragged_factory_ops.constant([[1.0, 2.0, 3.0, 4.0],
                                             [10.0, 20.0, 30.0, 40.0]])
    op = ngrams_op.ngrams(
        test_data, width=2, axis=0, reduction_type=ngrams_op.Reduction.SUM)
    expected_values = [[11.0, 22.0, 33.0, 44.0]]

    self.assertAllEqual(expected_values, op)

  def testMeanReduction(self):
    test_data = constant_op.constant([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    op = ngrams_op.ngrams(
        test_data, width=2, axis=1, reduction_type=ngrams_op.Reduction.MEAN)
    expected_values = [[1.5, 2.5], [15.0, 25.0]]

    self.assertAllEqual(expected_values, op)

  def testRaggedMeanReduction(self):
    test_data = ragged_factory_ops.constant([[1.0, 2.0, 3.0, 4.0],
                                             [10.0, 20.0, 30.0]])
    op = ngrams_op.ngrams(
        test_data, width=2, axis=-1, reduction_type=ngrams_op.Reduction.MEAN)
    expected_values = [[1.5, 2.5, 3.5], [15.0, 25.0]]

    self.assertAllEqual(expected_values, op)

  def testStringJoinReduction(self):
    test_data = constant_op.constant([[b"a", b"b", b"c"],
                                      [b"dd", b"ee", b"ff"]])
    op = ngrams_op.ngrams(
        test_data,
        width=2,
        axis=-1,
        reduction_type=ngrams_op.Reduction.STRING_JOIN,
        string_separator=b"|")
    expected_values = [[b"a|b", b"b|c"], [b"dd|ee", b"ee|ff"]]

    self.assertAllEqual(expected_values, op)

  def testStringJoinReductionAxisZero(self):
    test_data = constant_op.constant([b"a", b"b", b"c"])
    op = ngrams_op.ngrams(
        test_data,
        width=2,
        axis=-1,  # The -1 axis is the zero axis here.
        reduction_type=ngrams_op.Reduction.STRING_JOIN,
        string_separator=b"|")
    expected_values = [b"a|b", b"b|c"]

    self.assertAllEqual(expected_values, op)

  def testRaggedStringJoinReduction(self):
    test_data = ragged_factory_ops.constant([[b"a", b"b", b"c"],
                                             [b"dd", b"ee"]])
    op = ngrams_op.ngrams(
        test_data,
        width=2,
        axis=-1,
        reduction_type=ngrams_op.Reduction.STRING_JOIN,
        string_separator=b"|")
    expected_values = [[b"a|b", b"b|c"], [b"dd|ee"]]

    self.assertAllEqual(expected_values, op)

  def testReductionWithNegativeAxis(self):
    test_data = constant_op.constant([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    op = ngrams_op.ngrams(
        test_data, width=2, axis=-1, reduction_type=ngrams_op.Reduction.SUM)
    expected_values = [[3.0, 5.0], [30.0, 50.0]]

    self.assertAllEqual(expected_values, op)

  def testReductionOnInnerAxis(self):
    test_data = constant_op.constant([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
                                      [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]]])
    op = ngrams_op.ngrams(
        test_data, width=2, axis=-2, reduction_type=ngrams_op.Reduction.SUM)
    expected_values = [[[11.0, 22.0, 33.0]], [[44.0, 55.0, 66.0]]]

    self.assertAllEqual(expected_values, op)

  def testRaggedReductionOnInnerAxis(self):
    test_data = ragged_factory_ops.constant([[[1.0, 2.0, 3.0, 4.0],
                                              [10.0, 20.0, 30.0, 40.0]],
                                             [[100.0, 200.0], [300.0, 400.0]]])
    op = ngrams_op.ngrams(
        test_data, width=2, axis=-2, reduction_type=ngrams_op.Reduction.SUM)
    expected_values = [[[11.0, 22.0, 33.0, 44.0]], [[400.0, 600.0]]]

    self.assertAllEqual(expected_values, op)

  def testReductionOnAxisWithInsufficientValuesReturnsEmptySet(self):
    test_data = constant_op.constant([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    op = ngrams_op.ngrams(
        test_data, width=4, axis=-1, reduction_type=ngrams_op.Reduction.SUM)
    expected_values = [[], []]

    self.assertAllEqual(expected_values, op)

  def testRaggedReductionOnAxisWithInsufficientValuesReturnsEmptySet(self):
    test_data = ragged_factory_ops.constant([[1.0, 2.0, 3.0],
                                             [10.0, 20.0, 30.0, 40.0]])
    op = ngrams_op.ngrams(
        test_data, width=4, axis=1, reduction_type=ngrams_op.Reduction.SUM)
    expected_values = [[], [100.0]]

    self.assertAllEqual(expected_values, op)

  def testStringJoinReductionFailsWithImproperAxis(self):
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r".*requires that ngrams' 'axis' parameter be -1."):
      _ = ngrams_op.ngrams(
          data=[],
          width=2,
          axis=0,
          reduction_type=ngrams_op.Reduction.STRING_JOIN)

  def testUnspecifiedReductionTypeFails(self):
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                r"reduction_type must be specified."):
      _ = ngrams_op.ngrams(data=[], width=2, axis=0)

  def testBadReductionTypeFails(self):
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                r"reduction_type must be a Reduction."):
      _ = ngrams_op.ngrams(data=[], width=2, axis=0, reduction_type="SUM")


class NgramsV2OpTest(test_util.TensorFlowTestCase):

  @test_util.with_forward_compatibility_horizons([2022, 11, 30])
  def testSumReduction(self):
    test_data = constant_op.constant([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    op = ngrams_op.ngrams(
        test_data, width=2, axis=1, reduction_type=ngrams_op.Reduction.SUM)
    expected_values = [[3.0, 5.0], [30.0, 50.0]]

    self.assertAllEqual(expected_values, op)

  @test_util.with_forward_compatibility_horizons([2022, 11, 30])
  def testRaggedSumReduction(self):
    test_data = ragged_factory_ops.constant([[1.0, 2.0, 3.0, 4.0],
                                             [10.0, 20.0, 30.0]])
    op = ngrams_op.ngrams(
        test_data, width=2, axis=1, reduction_type=ngrams_op.Reduction.SUM)
    expected_values = [[3.0, 5.0, 7.0], [30.0, 50.0]]

    self.assertAllEqual(expected_values, op)

  @test_util.with_forward_compatibility_horizons([2022, 11, 30])
  def testRaggedSumReductionAxisZero(self):
    test_data = ragged_factory_ops.constant([[1.0, 2.0, 3.0, 4.0],
                                             [10.0, 20.0, 30.0, 40.0]])
    op = ngrams_op.ngrams(
        test_data, width=2, axis=0, reduction_type=ngrams_op.Reduction.SUM)
    expected_values = [[11.0, 22.0, 33.0, 44.0]]

    self.assertAllEqual(expected_values, op)

  @test_util.with_forward_compatibility_horizons([2022, 11, 30])
  def testMeanReduction(self):
    test_data = constant_op.constant([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    op = ngrams_op.ngrams(
        test_data, width=2, axis=1, reduction_type=ngrams_op.Reduction.MEAN)
    expected_values = [[1.5, 2.5], [15.0, 25.0]]

    self.assertAllEqual(expected_values, op)

  @test_util.with_forward_compatibility_horizons([2022, 11, 30])
  def testRaggedMeanReduction(self):
    test_data = ragged_factory_ops.constant([[1.0, 2.0, 3.0, 4.0],
                                             [10.0, 20.0, 30.0]])
    op = ngrams_op.ngrams(
        test_data, width=2, axis=-1, reduction_type=ngrams_op.Reduction.MEAN)
    expected_values = [[1.5, 2.5, 3.5], [15.0, 25.0]]

    self.assertAllEqual(expected_values, op)

  @test_util.with_forward_compatibility_horizons([2022, 11, 30])
  def testStringJoinReduction(self):
    test_data = constant_op.constant([["a", "b", "c"], ["dd", "ee", "ff"]])
    op = ngrams_op.ngrams(
        test_data,
        width=2,
        axis=-1,
        reduction_type=ngrams_op.Reduction.STRING_JOIN,
        string_separator="|")
    expected_values = [[b"a|b", b"b|c"], [b"dd|ee", b"ee|ff"]]

    self.assertAllEqual(expected_values, op)

  @test_util.with_forward_compatibility_horizons([2022, 11, 30])
  def testStringJoinReductionRank3(self):
    test_data = constant_op.constant([[["a", "b", "c"], ["z", "y", "x"]],
                                      [["dd", "ee", "ff"], ["zz", "yy", "xx"]]])
    op = ngrams_op.ngrams(
        test_data,
        width=2,
        axis=-1,
        reduction_type=ngrams_op.Reduction.STRING_JOIN,
        string_separator="|")
    expected_values = [[[b"a|b", b"b|c"], [b"z|y", b"y|x"]],
                       [[b"dd|ee", b"ee|ff"], [b"zz|yy", b"yy|xx"]]]

    self.assertAllEqual(expected_values, op)

  @test_util.with_forward_compatibility_horizons([2022, 11, 30])
  def testStringJoinReductionAxisZero(self):
    test_data = constant_op.constant(["a", "b", "c"])
    op = ngrams_op.ngrams(
        test_data,
        width=2,
        axis=-1,  # The -1 axis is the zero axis here.
        reduction_type=ngrams_op.Reduction.STRING_JOIN,
        string_separator="|")
    expected_values = [b"a|b", b"b|c"]

    self.assertAllEqual(expected_values, op)

  @test_util.with_forward_compatibility_horizons([2022, 11, 30])
  def testRaggedStringJoinReduction(self):
    test_data = ragged_factory_ops.constant([["a", "b", "c"], ["dd", "ee"]])
    op = ngrams_op.ngrams(
        test_data,
        width=2,
        axis=-1,
        reduction_type=ngrams_op.Reduction.STRING_JOIN,
        string_separator="|")
    expected_values = [[b"a|b", b"b|c"], [b"dd|ee"]]

    self.assertAllEqual(expected_values, op)

  @test_util.with_forward_compatibility_horizons([2022, 11, 30])
  def testRaggedDeepStringJoinReduction(self):
    test_data = ragged_factory_ops.constant([[[["a", "b", "c"]],
                                              [["dd", "ee"]]],
                                             [[["f", "g"], ["h", "i", "j"]],
                                              [["k", "l"]]]])
    op = ngrams_op.ngrams(
        test_data,
        width=2,
        axis=-1,
        reduction_type=ngrams_op.Reduction.STRING_JOIN,
        string_separator="|")
    expected_values = [[[[b"a|b", b"b|c"]], [[b"dd|ee"]]],
                       [[[b"f|g"], [b"h|i", b"i|j"]], [[b"k|l"]]]]

    self.assertAllEqual(expected_values, op)

  @test_util.with_forward_compatibility_horizons([2022, 11, 30])
  def testDoubleRaggedStringJoinReduction(self):
    test_data = tf.constant([["a b c"], ["d e"]])
    t = tf_text.WhitespaceTokenizer()
    test_data = t.tokenize(test_data)
    op = ngrams_op.ngrams(
        test_data,
        width=2,
        axis=-1,
        reduction_type=ngrams_op.Reduction.STRING_JOIN,
        string_separator="|")
    expected_values = [[[b"a|b", b"b|c"]], [[b"d|e"]]]

    self.assertAllEqual(expected_values, op)

  @test_util.with_forward_compatibility_horizons([2022, 11, 30])
  def testReductionWithNegativeAxis(self):
    test_data = constant_op.constant([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    op = ngrams_op.ngrams(
        test_data, width=2, axis=-1, reduction_type=ngrams_op.Reduction.SUM)
    expected_values = [[3.0, 5.0], [30.0, 50.0]]

    self.assertAllEqual(expected_values, op)

  @test_util.with_forward_compatibility_horizons([2022, 11, 30])
  def testReductionOnInnerAxis(self):
    test_data = constant_op.constant([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
                                      [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]]])
    op = ngrams_op.ngrams(
        test_data, width=2, axis=-2, reduction_type=ngrams_op.Reduction.SUM)
    expected_values = [[[11.0, 22.0, 33.0]], [[44.0, 55.0, 66.0]]]

    self.assertAllEqual(expected_values, op)

  @test_util.with_forward_compatibility_horizons([2022, 11, 30])
  def testRaggedReductionOnInnerAxis(self):
    test_data = ragged_factory_ops.constant([[[1.0, 2.0, 3.0, 4.0],
                                              [10.0, 20.0, 30.0, 40.0]],
                                             [[100.0, 200.0], [300.0, 400.0]]])
    op = ngrams_op.ngrams(
        test_data, width=2, axis=-2, reduction_type=ngrams_op.Reduction.SUM)
    expected_values = [[[11.0, 22.0, 33.0, 44.0]], [[400.0, 600.0]]]

    self.assertAllEqual(expected_values, op)

  @test_util.with_forward_compatibility_horizons([2022, 11, 30])
  def testReductionOnAxisWithInsufficientValuesReturnsEmptySet(self):
    test_data = constant_op.constant([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    op = ngrams_op.ngrams(
        test_data, width=4, axis=-1, reduction_type=ngrams_op.Reduction.SUM)
    expected_values = [[], []]

    self.assertAllEqual(expected_values, op)

  @test_util.with_forward_compatibility_horizons([2022, 11, 30])
  def testRaggedReductionOnAxisWithInsufficientValuesReturnsEmptySet(self):
    test_data = ragged_factory_ops.constant([[1.0, 2.0, 3.0],
                                             [10.0, 20.0, 30.0, 40.0]])
    op = ngrams_op.ngrams(
        test_data, width=4, axis=1, reduction_type=ngrams_op.Reduction.SUM)
    expected_values = [[], [100.0]]

    self.assertAllEqual(expected_values, op)

  @test_util.with_forward_compatibility_horizons([2022, 11, 30])
  def testStringJoinReductionFailsWithImproperAxis(self):
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r".*requires that ngrams' 'axis' parameter be -1."):
      _ = ngrams_op.ngrams(
          data=[],
          width=2,
          axis=0,
          reduction_type=ngrams_op.Reduction.STRING_JOIN)

  @test_util.with_forward_compatibility_horizons([2022, 11, 30])
  def testUnspecifiedReductionTypeFails(self):
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                r"reduction_type must be specified."):
      _ = ngrams_op.ngrams(data=[], width=2, axis=0)

  @test_util.with_forward_compatibility_horizons([2022, 11, 30])
  def testBadReductionTypeFails(self):
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                r"reduction_type must be a Reduction."):
      _ = ngrams_op.ngrams(data=[], width=2, axis=0, reduction_type="SUM")

  @test_util.with_forward_compatibility_horizons([2022, 11, 30])
  def testTfLite(self):
    """Checks TFLite conversion and inference."""

    class NgramModel(tf.keras.Model):

      def call(self, input_tensor, **kwargs):
        return ngrams_op.ngrams(input_tensor, width=2, axis=-1,
                                reduction_type=ngrams_op.Reduction.STRING_JOIN,
                                string_separator="|")

    # Test input data.
    input_data = np.array(["a", "b", "c"])

    # Define a Keras model.
    model = NgramModel()
    # Do TF.Text inference.
    tf_result = model(tf.constant(input_data))

    # Convert to TFLite.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    # Do TFLite inference.
    op = tflite_registrar.AddNgramsStringJoin
    interp = interpreter.InterpreterWithCustomOps(
        model_content=tflite_model,
        custom_op_registerers=[op])
    input_details = interp.get_input_details()
    interp.resize_tensor_input(input_details[0]["index"], tf.shape(input_data))
    interp.allocate_tensors()
    interp.set_tensor(input_details[0]["index"], input_data)
    interp.invoke()
    output_details = interp.get_output_details()
    tflite_result = interp.get_tensor(output_details[0]["index"])

    # Assert the results are identical.
    self.assertAllEqual(tflite_result, tf_result)

  @test_util.with_forward_compatibility_horizons([2022, 11, 30])
  def testTfLiteRagged(self):
    """Checks TFLite conversion and inference."""

    class NgramModel(tf.keras.Model):

      def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tf_text.WhitespaceTokenizer()

      @tf.function(input_signature=[
          tf.TensorSpec(shape=[1], dtype=tf.string, name="input")
      ])
      def call(self, input_tensor):
        input_tensor = self.tokenizer.tokenize(input_tensor)
        x = ngrams_op.ngrams(input_tensor, width=2, axis=-1,
                             reduction_type=ngrams_op.Reduction.STRING_JOIN,
                             string_separator="|")
        return {"result": x.flat_values}

    # Test input data.
    input_data = np.array(["foo bar"])

    # Define a Keras model.
    model = NgramModel()
    # Do TF.Text inference.
    tf_result = model(tf.constant(input_data))["result"]
    print(tf_result.shape)

    # Convert to TFLite.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    # Do TFLite inference.
    interp = interpreter.InterpreterWithCustomOps(
        model_content=tflite_model,
        custom_op_registerers=tf_text.tflite_registrar.SELECT_TFTEXT_OPS)
    print(interp.get_signature_list())
    tn = interp.get_signature_runner("serving_default")
    output = tn(input=input_data)
    if tf.executing_eagerly():
      tflite_result = output["result"]
    else:
      tflite_result = output["output_1"]

    # Assert the results are identical.
    self.assertAllEqual(tflite_result, tf_result)


if __name__ == "__main__":
  test.main()
