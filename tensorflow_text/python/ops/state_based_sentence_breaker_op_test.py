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

"""Tests for sentence_breaking_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text

from tensorflow.lite.python import interpreter
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test
from tensorflow_text.python.ops import state_based_sentence_breaker_op


class SentenceFragmenterTestCasesV2(test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      # pyformat: disable
      dict(
          test_description="Test acronyms",
          doc=["Welcome to the U.S. don't be surprised."],
          expected_fragment_text=[
              [b"Welcome to the U.S.", b"don't be surprised."]
          ],
      ),
      dict(
          test_description="Test batch containing acronyms",
          doc=["Welcome to the U.S. don't be surprised.", "I.B.M. yo"],
          expected_fragment_text=[
              [b"Welcome to the U.S.", b"don't be surprised."],
              [b"I.B.M.", b"yo"]
          ],
      ),
      dict(
          test_description="Test when rank > 1.",
          doc=[["Welcome to the U.S. don't be surprised."], ["I.B.M. yo"]],
          expected_fragment_text=[
              [[b"Welcome to the U.S.", b"don't be surprised."]],
              [[b"I.B.M.", b"yo"]]
          ],
      ),
      dict(
          test_description="Test semicolons",
          doc=["Welcome to the US; don't be surprised."],
          expected_fragment_text=[[b"Welcome to the US; don't be surprised."]],
      ),
      dict(
          test_description="Basic test",
          doc=["Hello. Foo bar!"],
          expected_fragment_text=[[b"Hello.", b"Foo bar!"]],
      ),
      dict(
          test_description="Basic ellipsis test",
          doc=["Hello...foo bar"],
          expected_fragment_text=[[b"Hello...", b"foo bar"]],
      ),
      dict(
          test_description="Parentheses and ellipsis test",
          doc=["Hello (who are you...) foo bar"],
          expected_fragment_text=[[b"Hello (who are you...)", b"foo bar"]],
      ),
      dict(
          test_description="Punctuation after parentheses test",
          doc=["Hello (who are you)? Foo bar!"],
          expected_fragment_text=[[b"Hello (who are you)?", b"Foo bar!"]],
      ),
      dict(
          test_description="MidFragment Parentheses test",
          doc=["Hello (who are you) world? Foo bar"],
          expected_fragment_text=[[b"Hello (who are you) world?", b"Foo bar"]],
      ),
      dict(
          test_description="Many final punctuation test",
          doc=["Hello!!!!! Who are you??"],
          expected_fragment_text=[[b"Hello!!!!!", b"Who are you??"]],
      ),
      dict(
          test_description="Test emoticons within text",
          doc=["Hello world :) Oh, hi :-O"],
          expected_fragment_text=[[b"Hello world :)", b"Oh, hi :-O"]],
      ),
      dict(
          test_description="Test emoticons with punctuation following",
          doc=["Hello world :)! Hi."],
          expected_fragment_text=[[b"Hello world :)!", b"Hi."]],
      ),
      dict(
          test_description="Test emoticon list",
          doc=[b":) :-\\ (=^..^=) |-O"],
          expected_fragment_text=[[b":)", b":-\\", b"(=^..^=)", b"|-O"]],
      ),
      dict(
          test_description="Test emoticon batch",
          doc=[":)", ":-\\", "(=^..^=)", "|-O"],
          expected_fragment_text=[[b":)"], [b":-\\"], [b"(=^..^=)"], [b"|-O"]],
      ),
      dict(
          test_description="Test tensor inputs w/ shape [2, 1]",
          doc=[["Welcome to the U.S. don't be surprised. We like it here."],
               ["I.B.M. yo"]],
          expected_fragment_text=[
              [[b"Welcome to the U.S.", b"don't be surprised.",
                b"We like it here."]],
              [[b"I.B.M.", b"yo"]]
          ],
      ),
      # pyformat: enable
  ])
  def testStateBasedSentenceBreaker(self, test_description, doc,
                                    expected_fragment_text):
    input = constant_op.constant(doc)  # pylint: disable=redefined-builtin
    sentence_breaker = (
        state_based_sentence_breaker_op.StateBasedSentenceBreaker())
    fragment_text, fragment_starts, fragment_ends = (
        sentence_breaker.break_sentences_with_offsets(input))

    texts, starts, ends = self.evaluate(
        (fragment_text, fragment_starts, fragment_ends))
    self.assertAllEqual(expected_fragment_text, fragment_text)
    for d, text, start, end in zip(doc, texts.to_list(), starts.to_list(),
                                   ends.to_list()):
      # broadcast d to match start/end's shape
      start = constant_op.constant(start)
      end = constant_op.constant(end)
      d = array_ops.broadcast_to(d, start.shape)
      self.assertAllEqual(string_ops.substr(d, start, end - start), text)

  def testTfLite(self):
    """Checks TFLite conversion and inference."""

    class Model(tf.keras.Model):

      def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sb = tf_text.StateBasedSentenceBreaker()

      @tf.function(input_signature=[
          tf.TensorSpec(shape=[None], dtype=tf.string, name="input")
      ])
      def call(self, input_tensor):
        return {"result": self.sb.break_sentences(input_tensor).flat_values}
    # Test input data.
    input_data = np.array(["Some minds are better kept apart"])

    # Define a model.
    model = Model()
    # Do TF inference.
    tf_result = model(tf.constant(input_data))["result"]

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
    split = interp.get_signature_runner("serving_default")
    output = split(input=input_data)
    if tf.executing_eagerly():
      tflite_result = output["result"]
    else:
      tflite_result = output["output_1"]

    # Assert the results are identical.
    self.assertAllEqual(tflite_result, tf_result)

if __name__ == "__main__":
  test.main()
