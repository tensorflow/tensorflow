# encoding: utf-8
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
"""Text processor tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

from tensorflow.contrib.learn.python.learn.preprocessing import CategoricalVocabulary
from tensorflow.contrib.learn.python.learn.preprocessing import text
from tensorflow.python.platform import test


class TextTest(test.TestCase):
  """Text processor tests."""

  def testTokenizer(self):
    words = text.tokenizer(
        ["a b c", "a\nb\nc", "a, b - c", "фыв выф", "你好 怎么样"])
    self.assertEqual(
        list(words), [["a", "b", "c"], ["a", "b", "c"], ["a", "b", "-", "c"],
                      ["фыв", "выф"], ["你好", "怎么样"]])

  def testByteProcessor(self):
    processor = text.ByteProcessor(max_document_length=8)
    inp = ["abc", "фыва", "фыва", b"abc", "12345678901234567890"]
    res = list(processor.fit_transform(inp))
    self.assertAllEqual(res, [[97, 98, 99, 0, 0, 0, 0, 0],
                              [209, 132, 209, 139, 208, 178, 208, 176],
                              [209, 132, 209, 139, 208, 178, 208, 176],
                              [97, 98, 99, 0, 0, 0, 0, 0],
                              [49, 50, 51, 52, 53, 54, 55, 56]])
    res = list(processor.reverse(res))
    self.assertAllEqual(res, ["abc", "фыва", "фыва", "abc", "12345678"])

  def testVocabularyProcessor(self):
    vocab_processor = text.VocabularyProcessor(
        max_document_length=4, min_frequency=1)
    tokens = vocab_processor.fit_transform(["a b c", "a\nb\nc", "a, b - c"])
    self.assertAllEqual(
        list(tokens), [[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 3]])

  def testVocabularyProcessorSaveRestore(self):
    filename = test.get_temp_dir() + "test.vocab"
    vocab_processor = text.VocabularyProcessor(
        max_document_length=4, min_frequency=1)
    tokens = vocab_processor.fit_transform(["a b c", "a\nb\nc", "a, b - c"])
    vocab_processor.save(filename)
    new_vocab = text.VocabularyProcessor.restore(filename)
    tokens = new_vocab.transform(["a b c"])
    self.assertAllEqual(list(tokens), [[1, 2, 3, 0]])

  def testExistingVocabularyProcessor(self):
    vocab = CategoricalVocabulary()
    vocab.get("A")
    vocab.get("B")
    vocab.freeze()
    vocab_processor = text.VocabularyProcessor(
        max_document_length=4, vocabulary=vocab, tokenizer_fn=list)
    tokens = vocab_processor.fit_transform(["ABC", "CBABAF"])
    self.assertAllEqual(list(tokens), [[1, 2, 0, 0], [0, 2, 1, 2]])


if __name__ == "__main__":
  test.main()
