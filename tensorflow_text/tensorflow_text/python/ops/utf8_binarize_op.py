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

"""UTF-8 binarization op (RetVec-inspired)."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack

# pylint: disable=g-bad-import-order,unused-import
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_utf8_binarize_op = load_library.load_op_library(resource_loader.get_path_to_datafile('_utf8_binarize_op.so'))


def utf8_binarize(tokens, word_length=16, bits_per_char=24,
                  replacement_char=65533, name=None):
  """Decode UTF8 tokens into code points and return their bits.

  See the [RetVec paper](https://arxiv.org/abs/2302.09207) for details.

  Example:

  >>> code_points = utf8_binarize("hello", word_length=3, bits_per_char=4)
  >>> print(code_points.numpy())
  [0. 0. 0. 1. 1. 0. 1. 0. 0. 0. 1. 1.]

  The codepoints are encoded bitwise in the little-endian order.
  The inner dimension of the output is always `word_length * bits_per_char`,
  because extra characters are truncated / missing characters are padded,
  and `bits_per_char` lowest bits of each codepoint is stored.

  Decoding errors (which in applications are often replaced with the character
  U+65533 "REPLACEMENT CHARACTER") are represented with `replacement_char`'s
  `bits_per_char` lowest bits.

  Args:
    tokens: A `Tensor` of tokens (strings) with any shape.
    word_length: Number of Unicode characters to process per word (the rest are
                 silently ignored; the output is zero-padded).
    bits_per_char: The number of lowest bits of the Unicode codepoint to encode.
    replacement_char: The Unicode codepoint to use on decoding errors.
    name: The op name (optional).

  Returns:
    A tensor of floating-point zero and one values corresponding to the bits
    of the token characters' Unicode code points.
    Shape: `[<shape of `tokens`>, word_length * bits_per_char]`.
  """
  with ops.name_scope(name, "Utf8Binarize", [tokens]):
    original_tokens_tensor = ops.convert_to_tensor(tokens)
    tokens_tensor = original_tokens_tensor
    shape = tokens_tensor.shape
    if shape.ndims is None:
      raise ValueError("Rank of `tokens` must be statically known.")
    if shape.ndims == 0:
      tokens_tensor = array_ops_stack.stack([tokens_tensor])
    elif shape.ndims > 1:
      tokens_tensor = array_ops.reshape(tokens_tensor, [-1])
    binarizations = gen_utf8_binarize_op.tf_text_utf8_binarize(
        tokens_tensor,
        word_length,
        bits_per_char=bits_per_char,
        replacement_char=replacement_char
    )
    if shape.ndims != 1:
      inner_dimension = word_length * bits_per_char
      computed_shape = (
          (inner_dimension,) if shape.ndims == 0
          else array_ops.concat([array_ops.shape(original_tokens_tensor),
                                 [inner_dimension]], axis=0))
      binarizations = array_ops.reshape(binarizations, computed_shape)
    return binarizations
