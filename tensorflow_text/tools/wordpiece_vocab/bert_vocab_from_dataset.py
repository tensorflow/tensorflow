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

"""Build a bert vocab from a `tf.data.Dataset`."""
from typing import List

from tensorflow_text.python.ops import bert_tokenizer
from tensorflow_text.tools.wordpiece_vocab import wordpiece_tokenizer_learner_lib as learner


def bert_vocab_from_dataset(dataset,
                            vocab_size: int,
                            reserved_tokens: List[str],
                            bert_tokenizer_params=None,
                            learn_params=None) -> List[str]:
  """Generate a Bert wordpiece vocabulary from a `tf.data.Dataset` of texts.

  ```
  import tensorflow_text as text

  vocab = bert_vocab_from_dataset(dataset, vocab_size, reserved_tokens,
                                  bert_tokenizer_params, learn_params)
  bert_tokenizer = text.BertTokenizer(vocab, **bert_tokenizer_params)
  token_ids = bert_tokenizer.tokenize(text)
  ```

  This uses Bert's splitting algorithm to split the text into words before
  generating the subword vocabulary from the resulting words.

  The resulting vocabulary _can_ be used directly with a
  `text.WordpieceTokenizer`, but note that the vocabulary will be sub-optimal or
  **broken** if you split the text into words a different way.

  ```
  wordpiece_tokenizer = text.WordpieceTokenizer(vocab, ...)
  words = split(text)
  token_ids = wordpiece_tokenizer.tokenize(words)
  ```

  Args:
    dataset: `A tf.data.Dataset` containing string-tensor elements.
    vocab_size: The target vocabulary size. This is the maximum size.
    reserved_tokens: A list of tokens that must be included in the vocabulary.
    bert_tokenizer_params: The `text.BertTokenizer` arguments relavant for to
      vocabulary-generation:
      * `lower_case`
      * `keep_whitespace`
      * `normalization_form`
      * `preserve_unused_token`

      See `BertTokenizer` for details. You should use the same values for
      these to both generate the vocabulary and build the `BertTokenizer`.
    learn_params: A dict of additional key word arguments for the the vocabulary
      learning function. See `wordpiece_tokenizer_learner_lib.learn` for
      details.

  Returns:
    A list strings containing the vocabulary.

  Raises:
    TypeError: If the dataset contains structured elements instead of single
      tensors.
  """
  if bert_tokenizer_params is None:
    bert_tokenizer_params = {}

  if learn_params is None:
    learn_params = {}

  element_spec = dataset.element_spec

  try:
    element_spec.shape
  except AttributeError:
    raise TypeError("The dataset should contain single-tensor elements.")

  tokenizer = bert_tokenizer.BasicTokenizer(**bert_tokenizer_params)
  words_dataset = dataset.map(tokenizer.tokenize)
  word_counts = learner.count_words(words_dataset)

  vocab = learner.learn(word_counts, vocab_size, reserved_tokens,
                        **learn_params)

  return vocab
