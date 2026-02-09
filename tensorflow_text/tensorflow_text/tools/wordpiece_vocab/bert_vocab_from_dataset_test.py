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

"""Tests for tensorflow_text.python.tools.wordpiece_vocab.wordpiece_tokenizer_learner_lib."""

from absl.testing import absltest
import tensorflow as tf

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset


class BertVocabFromDatasetTest(absltest.TestCase):

  def test_smoke(self):
    ds = tf.data.Dataset.from_tensor_slices([
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do',
        'eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut',
        'enim ad minim veniam, quis nostrud exercitation ullamco laboris',
        'nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in',
        'reprehenderit in voluptate velit esse cillum dolore eu fugiat',
        'nulla pariatur. Excepteur sint occaecat cupidatat non proident,',
        'sunt in culpa qui officia deserunt mollit anim id est laborum.',
    ]).repeat(10).batch(3)

    reserved = ['<START>', '<END>']
    vocab = bert_vocab_from_dataset.bert_vocab_from_dataset(
        ds,
        vocab_size=65,
        reserved_tokens=reserved,
        bert_tokenizer_params={'lower_case': True})

    self.assertContainsSubset(reserved, vocab)
    self.assertContainsSubset(['a', 'b', 'c'], vocab)
    self.assertContainsSubset(['dolore', 'dolor'], vocab)
    self.assertContainsSubset(['##q', '##r', '##s'], vocab)


if __name__ == '__main__':
  absltest.main()
