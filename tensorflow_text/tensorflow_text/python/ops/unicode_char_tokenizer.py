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

"""Tokenizer implementation for character-based models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import monitoring
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow_text.python.ops.tokenization import Detokenizer
from tensorflow_text.python.ops.tokenization import TokenizerWithOffsets


_tf_text_unicode_char_tokenizer_create_counter = monitoring.Counter(
    "/nlx/api/python/unicode_char_tokenizer_create_counter",
    "Counter for number of UnicodeCharTokenizers created in Python.")


class UnicodeCharTokenizer(TokenizerWithOffsets, Detokenizer):
  """Tokenizes a tensor of UTF-8 strings on Unicode character boundaries.


  Resulting tokens are integers (unicode codepoints). Scalar input will
  produce a `Tensor` output containing the codepoints. Tensor inputs will
  produce `RaggedTensor` outputs.

  Example:

  >>> tokenizer = tf_text.UnicodeCharTokenizer()
  >>> tokens = tokenizer.tokenize("abc")
  >>> print(tokens)
  tf.Tensor([97 98 99], shape=(3,), dtype=int32)

  >>> tokens = tokenizer.tokenize(["abc", "de"])
  >>> print(tokens)
  <tf.RaggedTensor [[97, 98, 99], [100, 101]]>

  Note: any remaining illegal and special UTF-8 characters (like BOM
  characters) in the input string will not be treated specially by the tokenizer
  and show up in the output tokens. These should be normalized out before
  or after tokenization if they are unwanted in the application.

  >>> t = ["abc" + chr(0xfffe) + chr(0x1fffe) ]
  >>> tokens = tokenizer.tokenize(t)
  >>> print(tokens.to_list())
  [[97, 98, 99, 65534, 131070]]

  Passing malformed UTF-8 will result in unpredictable behavior. Make sure
  inputs conform to UTF-8.
  """

  def __init__(self):
    """Initializes a new instance."""
    super(UnicodeCharTokenizer, self).__init__()
    _tf_text_unicode_char_tokenizer_create_counter.get_cell().increase_by(1)

  def tokenize(self, input):  # pylint: disable=redefined-builtin
    """Tokenizes a tensor of UTF-8 strings on Unicode character boundaries.

    Input strings are split on character boundaries using
    unicode_decode_with_offsets.

    Args:
      input: A `RaggedTensor`or `Tensor` of UTF-8 strings with any shape.

    Returns:
      A `RaggedTensor` of tokenized text. The returned shape is the shape of the
      input tensor with an added ragged dimension for tokens (characters) of
      each string.
    """
    (tokens, _, _) = self.tokenize_with_offsets(input)
    return tokens

  def tokenize_with_offsets(self, input):  # pylint: disable=redefined-builtin
    """Tokenizes a tensor of UTF-8 strings to Unicode characters.

    Example:

    >>> tokenizer = tf_text.UnicodeCharTokenizer()
    >>> tokens = tokenizer.tokenize_with_offsets("a"+chr(8364)+chr(10340))
    >>> print(tokens[0])
    tf.Tensor([   97  8364 10340], shape=(3,), dtype=int32)
    >>> print(tokens[1])
    tf.Tensor([0 1 4], shape=(3,), dtype=int64)
    >>> print(tokens[2])
    tf.Tensor([1 4 7], shape=(3,), dtype=int64)

    The `start_offsets` and `end_offsets` are in byte indices of the original
    string. When calling with multiple string inputs, the offset indices will
    be relative to the individual source strings:

    >>> toks = tokenizer.tokenize_with_offsets(["a"+chr(8364), "b"+chr(10300) ])
    >>> print(toks[0])
    <tf.RaggedTensor [[97, 8364], [98, 10300]]>
    >>> print(toks[1])
    <tf.RaggedTensor [[0, 1], [0, 1]]>
    >>> print(toks[2])
    <tf.RaggedTensor [[1, 4], [1, 4]]>

    Args:
      input: A `RaggedTensor`or `Tensor` of UTF-8 strings with any shape.

    Returns:
      A tuple `(tokens, start_offsets, end_offsets)` where:

        * `tokens`: A `RaggedTensor` of code points (integer type).
        * `start_offsets`: A `RaggedTensor` of the tokens' starting byte offset.
        * `end_offsets`: A `RaggedTensor` of the tokens' ending byte offset.
    """
    name = None
    with ops.name_scope(name, "UnicodeCharTokenize", [input]):
      input_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(input)
      (codepoints, byte_start_offsets) = (
          ragged_string_ops.unicode_decode_with_offsets(input_tensor, "UTF-8"))
      strlens = math_ops.cast(
          array_ops.expand_dims(string_ops.string_length(input_tensor), -1),
          dtypes.int64)
      # Adjust strlens to set 0-length strings to empty array (there will be no
      # tokens in this case).
      final_ends = ragged_array_ops.boolean_mask(strlens, strlens > 0)
      byte_end_offsets = array_ops.concat(
          [byte_start_offsets[..., 1:], final_ends], -1)
      return codepoints, byte_start_offsets, byte_end_offsets

  def detokenize(self, input, name=None):  # pylint: disable=redefined-builtin
    """Detokenizes input codepoints (integers) to UTF-8 strings.

    Example:

    >>> tokenizer = tf_text.UnicodeCharTokenizer()
    >>> tokens = tokenizer.tokenize(["abc", "de"])
    >>> s = tokenizer.detokenize(tokens)
    >>> print(s)
    tf.Tensor([b'abc' b'de'], shape=(2,), dtype=string)

    Args:
      input: A `RaggedTensor` or `Tensor` of codepoints (ints) with a rank of at
        least 1.
      name: The name argument that is passed to the op function.

    Returns:
      A N-1 dimensional string tensor of the text corresponding to the UTF-8
      codepoints in the input.
    """
    name = None
    with ops.name_scope(name, "UnicodeCharTokenize", [input, self]):
      input_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(input)
      return ragged_string_ops.unicode_encode(input_tensor, "UTF-8")
