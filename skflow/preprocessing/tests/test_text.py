# encoding: utf-8

#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import tensorflow as tf

from skflow.preprocessing import text


class TextTest(tf.test.TestCase):

    def testTokenizer(self):
        words = text.tokenizer(["a b c", "a\nb\nc", "a, b - c",
                                u"фыв выф", u"你好 怎么样"])
        self.assertEqual(list(words),
                         [["a", "b", "c"],
                          ["a", "b", "c"],
                          ["a", "b", "-", "c"],
                          [u"фыв", u"выф"],
                          [u"你好", u"怎么样"]])

    def testByteProcessor(self):
        processor = text.ByteProcessor(max_document_length=8)
        res = processor.transform(["abc", "фыва", u"фыва", b"abc",
                                   u"12345678901234567890"])
        self.assertAllClose(list(res),
                            [[97, 98, 99, 0, 0, 0, 0, 0],
                            [209, 132, 209, 139, 208, 178, 208, 176],
                            [209, 132, 209, 139, 208, 178, 208, 176],
                            [97, 98, 99, 0, 0, 0, 0, 0],
                            [49, 50, 51, 52, 53, 54, 55, 56]])

    def testVocabularyProcessor(self):
        vocab_processor = text.VocabularyProcessor(
            max_document_length=4,
            min_frequency=1)
        tokens = vocab_processor.fit_transform(
            ["a b c", "a\nb\nc", "a, b - c"])
        self.assertAllClose(list(tokens),
                            [[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 3]])

if __name__ == "__main__":
    tf.test.main()
