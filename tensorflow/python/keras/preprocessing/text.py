# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for text input preprocessing.
"""
# pylint: disable=invalid-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_preprocessing import text

from tensorflow.python.keras.preprocessing.text_dataset import text_dataset_from_directory  # pylint: disable=unused-import
from tensorflow.python.util.tf_export import keras_export

hashing_trick = text.hashing_trick
Tokenizer = text.Tokenizer


@keras_export('keras.preprocessing.text.text_to_word_sequence')
def text_to_word_sequence(input_text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True,
                          split=' '):
  """Converts a text to a sequence of words (or tokens).

  This function transforms a string of text into a list of words
  while ignoring `filters` which include punctuations by default.

  >>> sample_text = 'This is a sample sentence.'
  >>> tf.keras.preprocessing.text.text_to_word_sequence(sample_text)
  ['this', 'is', 'a', 'sample', 'sentence']

  Arguments:
      input_text: Input text (string).
      filters: list (or concatenation) of characters to filter out, such as
          punctuation. Default: `'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\\t\\n'`,
            includes basic punctuation, tabs, and newlines.
      lower: boolean. Whether to convert the input to lowercase.
      split: str. Separator for word splitting.

  Returns:
      A list of words (or tokens).
  """
  return text.text_to_word_sequence(
      input_text, filters=filters, lower=lower, split=split)


@keras_export('keras.preprocessing.text.one_hot')
def one_hot(input_text,
            n,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '):
  r"""One-hot encodes a text into a list of word indexes of size `n`.

  This function receives as input a string of text and returns a
  list of encoded integers each corresponding to a word (or token)
  in the given input string.

  Arguments:
      input_text: Input text (string).
      n: int. Size of vocabulary.
      filters: list (or concatenation) of characters to filter out, such as
        punctuation. Default:
        ```
        '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n
        ```,
        includes basic punctuation, tabs, and newlines.
      lower: boolean. Whether to set the text to lowercase.
      split: str. Separator for word splitting.

  Returns:
      List of integers in `[1, n]`. Each integer encodes a word
      (unicity non-guaranteed).
  """
  return text.one_hot(input_text, n, filters=filters, lower=lower, split=split)


# text.tokenizer_from_json is only available if keras_preprocessing >= 1.1.0
try:
  tokenizer_from_json = text.tokenizer_from_json
  keras_export('keras.preprocessing.text.tokenizer_from_json')(
      tokenizer_from_json)
except AttributeError:
  pass

keras_export('keras.preprocessing.text.hashing_trick')(hashing_trick)
keras_export('keras.preprocessing.text.Tokenizer')(Tokenizer)
