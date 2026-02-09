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

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for sentencepiece_tokenizer."""

import time

from absl import flags
import numpy as np
import tensorflow.compat.v2 as tf  # pylint: disable=g-direct-tensorflow-import
import tensorflow_text as tf_text

from tensorflow.lite.python import interpreter as interpreter_wrapper  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.platform import gfile
from tensorflow_text.python.ops import fast_sentencepiece_tokenizer as sentencepiece_tokenizer

FLAGS = flags.FLAGS

SENTENCEPIECE_MODEL_FILE = (
    "tensorflow_text/python/ops/test_data/"
    "fast_sentencepiece.model")


def _get_sentencepiece_model():
  return gfile.GFile(SENTENCEPIECE_MODEL_FILE, "rb").read()


class FastSentencepieceTokenizerTest(tf.test.TestCase):

  def setUp(self):
    super(FastSentencepieceTokenizerTest, self).setUp()
    self.sentencepiece_model = _get_sentencepiece_model()

  def test_tftext_sentencepiece_tokenizer(self):
    """Check that the new tokenizer produces the same result that the tftext one."""
    tftext_sp = tf_text.SentencepieceTokenizer(self.sentencepiece_model)
    opt_sp = sentencepiece_tokenizer.FastSentencepieceTokenizer(
        self.sentencepiece_model)

    input_text = [
        u" ", u"to be or not to be", u"ignored by length text1",
        u"ignored by length text2"
    ]
    tftext_tokenized = tftext_sp.tokenize(input_text)
    opt_tokenized = opt_sp.tokenize(input_text)
    self.assertAllEqual(tftext_tokenized, opt_tokenized)

  def test_tftext_sentencepiece_detokenizer(self):
    """Check that the new tokenizer produces the same result that the tftext one."""
    tftext_sp = tf_text.SentencepieceTokenizer(self.sentencepiece_model)
    opt_sp = sentencepiece_tokenizer.FastSentencepieceTokenizer(
        self.sentencepiece_model)

    input_text = [
        u" ", u"to be or not to be", u"ignored by length text1",
        u"ignored by length text2"
    ]
    tftext_tokenized = tftext_sp.tokenize(input_text)

    # Check detokenizer
    tftext_detokenized = tftext_sp.detokenize(tftext_tokenized)
    opt_detokenized = opt_sp.detokenize(tftext_tokenized)
    self.assertAllEqual(tftext_detokenized, opt_detokenized)

  def test_tftext_sentencepiece_tokenizer_bos_eos(self):
    """Check that the new tokenizer produces the same result that the tftext one with bos and eos."""
    tftext_sp = tf_text.SentencepieceTokenizer(
        self.sentencepiece_model, add_bos=True, add_eos=True)
    opt_sp = sentencepiece_tokenizer.FastSentencepieceTokenizer(
        self.sentencepiece_model, add_bos=True, add_eos=True)

    input_text = [
        u" ", u"to be or not to be", u"ignored by length text1",
        u"ignored by length text2"
    ]
    tftext_tokenized = tftext_sp.tokenize(input_text)
    opt_tokenized = opt_sp.tokenize(input_text)
    self.assertAllEqual(tftext_tokenized, opt_tokenized)

  def test_tflite_opt_sentence_tokenizer(self):
    """Check that can convert a Keras model to TFLite and it produces the same result for tokenization."""

    class TokenizerLayer(tf.keras.layers.Layer):

      def __init__(self, sentencepiece_model, **kwargs):
        super(TokenizerLayer, self).__init__(**kwargs)
        self.sp = sentencepiece_tokenizer.FastSentencepieceTokenizer(
            sentencepiece_model)

      def call(self, input_tensor, **kwargs):
        return self.sp.tokenize(input_tensor).flat_values

    model = tf.keras.models.Sequential(
        [TokenizerLayer(self.sentencepiece_model)])
    input_data = np.array([[
        u" ", u"to be or not to be", u"ignored by length text1",
        u"ignored by length text2"
    ]])
    tf_result = model.predict(input_data)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_ops = supported_ops
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    interpreter = interpreter_wrapper.InterpreterWithCustomOps(
        model_content=tflite_model,
        custom_op_registerers=tf_text.tflite_registrar.SELECT_TFTEXT_OPS)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    expected_result = [
        13, 36, 83, 131, 13, 36, 4, 3127, 152, 130, 30, 2424, 168, 1644, 1524,
        4, 3127, 152, 130, 30, 2424, 168, 1644, 636
    ]
    self.assertAllEqual(tf_result, expected_result)
    self.assertAllEqual(
        interpreter.get_tensor(output_details[0]["index"]), expected_result)

  def test_tflite_opt_sentence_detokenizer(self):
    """Check that can convert a Keras model to TFLite and it produces the same result for tokenization."""

    class DeTokenizerLayer(tf.keras.layers.Layer):

      def __init__(self, sentencepiece_model, **kwargs):
        super(DeTokenizerLayer, self).__init__(**kwargs)
        self.sp = sentencepiece_tokenizer.FastSentencepieceTokenizer(
            sentencepiece_model)

      def call(self, input_tensor, **kwargs):
        return self.sp.detokenize(input_tensor)

    model = tf.keras.models.Sequential(
        [DeTokenizerLayer(self.sentencepiece_model)])
    input_data = np.array([[
        13, 36, 83, 131, 13, 36, 4, 3127, 152, 130, 30, 2424, 168, 1644, 1524,
        4, 3127, 152, 130, 30, 2424, 168, 1644, 636
    ]],
                          dtype=np.int32)
    tf_result = model.predict(input_data)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_ops = supported_ops
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    interpreter = interpreter_wrapper.InterpreterWithCustomOps(
        model_content=tflite_model,
        custom_op_registerers=tf_text.tflite_registrar.SELECT_TFTEXT_OPS)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    expected_result = [
        "to be or not to be ignored by length text1 ignored by length text2"
    ]
    self.assertAllEqual(tf_result, expected_result)
    self.assertAllEqual(
        interpreter.get_tensor(output_details[0]["index"]), expected_result)

  def test_tflite_opt_sentence_tokenizer_vocab_size(self):
    """Check that can convert a Keras model to TFLite and it produces the same result for vocabulary size."""

    class TokenizerLayer(tf.keras.layers.Layer):

      def __init__(self, sentencepiece_model, **kwargs):
        super(TokenizerLayer, self).__init__(**kwargs)
        self.sp = sentencepiece_tokenizer.FastSentencepieceTokenizer(
            sentencepiece_model)

      def call(self, input_tensor, **kwargs):
        return self.sp.vocab_size()

    model = tf.keras.models.Sequential(
        [TokenizerLayer(self.sentencepiece_model)])
    input_data = np.array([[""]])
    tf_result = model.predict(input_data)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_ops = supported_ops
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    interpreter = interpreter_wrapper.InterpreterWithCustomOps(
        model_content=tflite_model,
        custom_op_registerers=tf_text.tflite_registrar.SELECT_TFTEXT_OPS)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    expected_result = 4000
    self.assertEqual(tf_result, expected_result)
    self.assertAllEqual(
        interpreter.get_tensor(output_details[0]["index"]), expected_result)


class FastSentencepieceTokenizerBenchmark(tf.test.Benchmark):

  def benchmarkTokenizer(self):
    sp_model = _get_sentencepiece_model()
    test_text = [
        ("This week we celebrate the casts and creatives who have come "
         "together to bring us our favorite."),
        "More Stacks products demonstrated commitment to excellent support.",
        "Test, test, test."
    ]

    tftext_sp = tf_text.SentencepieceTokenizer(sp_model)
    opt_sp = sentencepiece_tokenizer.FastSentencepieceTokenizer(sp_model)
    iter_number = 1000
    start = time.time()
    for _ in range(iter_number):
      _ = opt_sp.tokenize(test_text)
    self.report_benchmark(
        iters=iter_number, wall_time=time.time() - start, name="opt")
    start = time.time()
    for _ in range(iter_number):
      _ = tftext_sp.tokenize(test_text)
    self.report_benchmark(
        iters=iter_number, wall_time=time.time() - start, name="tf.text")


if __name__ == "__main__":
  tf.test.main()
