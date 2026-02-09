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

# coding=utf-8
"""Tests for fast_bert_normalizer op in tensorflow_text."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text

from tensorflow.lite.python import interpreter
from tensorflow.python.framework import test_util
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow_text.core.pybinds import pywrap_fast_bert_normalizer_model_builder
from tensorflow_text.python.ops import fast_bert_normalizer
from tensorflow_text.python.ops import normalize_ops


def _Utf8(char):
  return char.encode("utf-8")


def _OriginalNormalize(text_input, lower_case_nfd_strip_accents):
  """Based on tensorflow_text/python/ops/bert_tokenizer.py.

  Args:
    text_input: the input tensor.
    lower_case_nfd_strip_accents: bool - If true, a preprocessing step is added
      to lowercase the text, which also applies NFD normalization and strips
      accents from characters.

  Returns:
    The normalized text.
  """
  if lower_case_nfd_strip_accents:
    text_input = normalize_ops.case_fold_utf8(text_input)
    text_input = normalize_ops.normalize_utf8(text_input, "NFD")
    text_input = string_ops.regex_replace(text_input, r"\p{Mn}", "")
  # strip out control characters
  text_input = string_ops.regex_replace(text_input, r"\p{Cc}|\p{Cf}", " ")
  return text_input


@parameterized.parameters([
    dict(lower_case_nfd_strip_accents=True),
    dict(lower_case_nfd_strip_accents=False),
])
@test_util.run_all_in_graph_and_eager_modes
class FastBertNormalizeTest(test.TestCase, parameterized.TestCase):

  def test_normal(self, lower_case_nfd_strip_accents):
    txt = [
        " TExt to loWERcase! ",
        "Punctuation and digits: -*/+$#%@%$123456789#^$*%&",
        "Non-latin UTF8 chars: ΘͽʦȺЩ",
        "Accented chars: ĎÔPQRŔSŠoóôpqrŕsštťuúvwxyý",
        "Non-UTF8-letters: e.g. ◆, ♥, and the emoji symbol ( ͡° ͜ʖ ͡°)",
        "Folded: ßς",
        "",
        u"Unicode with combining marks: \u1e9b\u0323",
        u"Control chars: Te\u0010xt",
    ]
    expected = _OriginalNormalize(txt, lower_case_nfd_strip_accents)

    # Test with building the model buffer on-the-fly.
    text_normalizer_lower_case_nfd_strip_accents = (
        fast_bert_normalizer.FastBertNormalizer(
            lower_case_nfd_strip_accents=lower_case_nfd_strip_accents))
    self.assertAllEqual(
        expected, text_normalizer_lower_case_nfd_strip_accents.normalize(txt))

    # Test with loading the model buffer.
    model_buffer = (
        pywrap_fast_bert_normalizer_model_builder
        .build_fast_bert_normalizer_model(lower_case_nfd_strip_accents))
    text_normalizer_lower_case_nfd_strip_accents = (
        fast_bert_normalizer.FastBertNormalizer(model_buffer=model_buffer))
    self.assertAllEqual(
        expected, text_normalizer_lower_case_nfd_strip_accents.normalize(txt))

  def test_one_string_ragged(self, lower_case_nfd_strip_accents):
    txt = ragged_factory_ops.constant([[" TExt ", "to", " loWERcase! "],
                                       [" TExt to loWERcase! "]])
    expected = _OriginalNormalize(txt, lower_case_nfd_strip_accents)
    text_normalizer_lower_case_nfd_strip_accents = (
        fast_bert_normalizer.FastBertNormalizer(
            lower_case_nfd_strip_accents=lower_case_nfd_strip_accents))
    self.assertAllEqual(
        expected, text_normalizer_lower_case_nfd_strip_accents.normalize(txt))

  def test_lowercase_nfd_strip_accent_empty_string(
      self, lower_case_nfd_strip_accents):
    txt = [
        "",
    ]
    expected = _OriginalNormalize(txt, lower_case_nfd_strip_accents)
    text_normalizer_lower_case_nfd_strip_accents = (
        fast_bert_normalizer.FastBertNormalizer(
            lower_case_nfd_strip_accents=lower_case_nfd_strip_accents))
    self.assertAllEqual(
        expected, text_normalizer_lower_case_nfd_strip_accents.normalize(txt))


@test_util.run_all_in_graph_and_eager_modes
class FastBertNormalizeWithOffsetsMapTest(parameterized.TestCase,
                                          test.TestCase):

  @parameterized.parameters([
      # Test one string and rank = 1 offset input
      dict(
          txt_input=[u"\u1e9b\u0323"],
          lower_case_nfd_strip_accents=True,
          expected_normalized_txt=[u"s"],
          expected_offsets=[[0, 5]],
      ),
      # Test multiple strings and rank = 1 offset input
      dict(
          txt_input=[u"\u1e9b\u0323", u"same", u"LOW\u0010er"],
          lower_case_nfd_strip_accents=True,
          expected_normalized_txt=["s", "same", "low er"],
          expected_offsets=[[0, 5], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6]],
      ),
      # Test multiple strings ragged tensor and rank > 1 offset input
      dict(
          txt_input=[[u"\u1e9b\u0323", u"same"], [], ["", "", ""],
                     [u"LOW\u0010er"]],
          lower_case_nfd_strip_accents=True,
          expected_normalized_txt=[[b"s", b"same"], [], [b"", b"", b""],
                                   [b"low er"]],
          expected_offsets=[[[0, 5], [0, 1, 2, 3, 4]], [], [[0], [0], [0]],
                            [[0, 1, 2, 3, 4, 5, 6]]],
      ),
  ])
  def test_tensor_input(self, txt_input, lower_case_nfd_strip_accents,
                        expected_normalized_txt, expected_offsets):
    normalizer = fast_bert_normalizer.FastBertNormalizer(
        lower_case_nfd_strip_accents)
    normalized_txt, offsets = normalizer.normalize_with_offsets(
        ragged_factory_ops.constant(txt_input))
    self.assertAllEqual(normalized_txt, expected_normalized_txt)
    self.assertAllEqual(offsets, expected_offsets)


@parameterized.parameters([
    # Test 0. No lower_case_nfd_strip_accents.
    dict(
        lower_case_nfd_strip_accents=False,
        text_inputs=[u"same", u"\u1e9b\u0323", u"LOWer"],
    ),
    # Test 1. No lower_case_nfd_strip_accents.
    dict(
        lower_case_nfd_strip_accents=True,
        text_inputs=[u"same", u"\u1e9b\u0323", u"LOWer"],
    ),
])
class FastBertNormalizerKerasModelTest(test.TestCase):

  def DISABELD_testKerasAndTflite(self, lower_case_nfd_strip_accents,
                                  text_inputs):
    """Checks TFLite conversion and inference."""

    class TextNormalizerModel(tf.keras.Model):

      def __init__(self, lower_case_nfd_strip_accents=False, **kwargs):
        super().__init__(**kwargs)
        self.normalizer = fast_bert_normalizer.FastBertNormalizer(
            lower_case_nfd_strip_accents=lower_case_nfd_strip_accents)

      def call(self, input_tensor, **kwargs):
        return self.normalizer.normalize(input_tensor)

    # Define a Keras model.
    model = TextNormalizerModel(
        lower_case_nfd_strip_accents=lower_case_nfd_strip_accents)
    # Build the Keras model by doing inference.
    model(np.array(text_inputs))

    # Convert to TFLite.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    # Setup the TFLite interpreter.
    interp = interpreter.InterpreterWithCustomOps(
        model_content=tflite_model,
        custom_op_registerers=tf_text.tflite_registrar.SELECT_TFTEXT_OPS)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    # Do TF.Text and TFLite inference.
    # TFLite only supports batch_size=1.
    for text in text_inputs:
      test_data = np.array([text])  # Construct a single-element input array.
      tf_result = model(test_data)
      interp.set_tensor(input_details[0]["index"], test_data)
      interp.invoke()
      tflite_result = interp.get_tensor(output_details[0]["index"])

      # Assert the results are identical.
      self.assertAllEqual(tflite_result, tf_result)


if __name__ == "__main__":
  test.main()
