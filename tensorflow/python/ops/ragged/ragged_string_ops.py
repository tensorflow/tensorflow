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
"""Ragged operations for working with string Tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import compat as util_compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export


map_fn_lib = LazyLoader("map_fn_lib", globals(),
                        "tensorflow.python.ops.map_fn")


@tf_export("strings.bytes_split")
@dispatch.add_dispatch_support
def string_bytes_split(input, name=None):  # pylint: disable=redefined-builtin
  """Split string elements of `input` into bytes.

  Examples:

  >>> tf.strings.bytes_split('hello').numpy()
  array([b'h', b'e', b'l', b'l', b'o'], dtype=object)
  >>> tf.strings.bytes_split(['hello', '123'])
  <tf.RaggedTensor [[b'h', b'e', b'l', b'l', b'o'], [b'1', b'2', b'3']]>

  Note that this op splits strings into bytes, not unicode characters.  To
  split strings into unicode characters, use `tf.strings.unicode_split`.

  See also: `tf.io.decode_raw`, `tf.strings.split`, `tf.strings.unicode_split`.

  Args:
    input: A string `Tensor` or `RaggedTensor`: the strings to split.  Must
      have a statically known rank (`N`).
    name: A name for the operation (optional).

  Returns:
    A `RaggedTensor` of rank `N+1`: the bytes that make up the source strings.
  """
  with ops.name_scope(name, "StringsByteSplit", [input]):
    input = ragged_tensor.convert_to_tensor_or_ragged_tensor(input,
                                                             name="input")
    if isinstance(input, ragged_tensor.RaggedTensor):
      return input.with_flat_values(string_bytes_split(input.flat_values))

    rank = input.shape.ndims
    if rank is None:
      raise ValueError("input must have a statically-known rank.")

    if rank == 0:
      return string_bytes_split(array_ops.stack([input]))[0]
    elif rank == 1:
      indices, values, shape = gen_string_ops.string_split(
          input, delimiter="", skip_empty=False)
      return ragged_tensor.RaggedTensor.from_value_rowids(
          values=values, value_rowids=indices[:, 0], nrows=shape[0],
          validate=False)
    else:
      return string_bytes_split(ragged_tensor.RaggedTensor.from_tensor(input))


# pylint: disable=redefined-builtin
@tf_export("strings.unicode_encode")
@dispatch.add_dispatch_support
def unicode_encode(input,
                   output_encoding,
                   errors="replace",
                   replacement_char=65533,
                   name=None):
  r"""Encodes each sequence of Unicode code points in `input` into a string.

  `result[i1...iN]` is the string formed by concatenating the Unicode
  codepoints `input[1...iN, :]`, encoded using `output_encoding`.

  Args:
    input: An `N+1` dimensional potentially ragged integer tensor with shape
      `[D1...DN, num_chars]`.
    output_encoding: Unicode encoding that should be used to encode each
      codepoint sequence.  Can be `"UTF-8"`, `"UTF-16-BE"`, or `"UTF-32-BE"`.
    errors: Specifies the response when an invalid codepoint is encountered
      (optional). One of:
            * `'replace'`: Replace invalid codepoint with the
              `replacement_char`. (default)
            * `'ignore'`: Skip invalid codepoints.
            * `'strict'`: Raise an exception for any invalid codepoint.
    replacement_char: The replacement character codepoint to be used in place of
      any invalid input when `errors='replace'`. Any valid unicode codepoint may
      be used. The default value is the default unicode replacement character
      which is 0xFFFD (U+65533).
    name: A name for the operation (optional).

  Returns:
    A `N` dimensional `string` tensor with shape `[D1...DN]`.

  #### Example:

  >>> input = tf.ragged.constant(
  ...     [[71, 246, 246, 100, 110, 105, 103, 104, 116], [128522]])
  >>> print(unicode_encode(input, 'UTF-8'))
  tf.Tensor([b'G\xc3\xb6\xc3\xb6dnight' b'\xf0\x9f\x98\x8a'],
            shape=(2,), dtype=string)
  """
  with ops.name_scope(name, "UnicodeEncode", [input]):
    input_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(input)
    if input_tensor.shape.ndims is None:
      raise ValueError("Rank of input_tensor must be statically known.")
    if ragged_tensor.is_ragged(input_tensor):
      if input_tensor.flat_values.shape.ndims > 1:
        # If the flat_values of our ragged tensor is multi-dimensional, we can
        # process it separately and our output will have the same nested splits
        # as our input.
        return input_tensor.with_flat_values(
            unicode_encode(input_tensor.flat_values, output_encoding, errors,
                           replacement_char))
      elif input_tensor.ragged_rank > 1:
        # Recursively process the values of the ragged tensor.
        return input_tensor.with_values(
            unicode_encode(input_tensor.values, output_encoding, errors,
                           replacement_char))
      else:
        # Our ragged tensor is of the correct shape (rank 1 flat_values tensor
        # with ragged_rank of 1) so we can process it as normal.
        return gen_string_ops.unicode_encode(
            input_values=input_tensor.values,
            input_splits=input_tensor.row_splits,
            output_encoding=output_encoding,
            errors=errors,
            replacement_char=replacement_char)
    else:
      if input_tensor.shape.ndims == 2:
        # The input tensor is of the correct 2-D shape, it's just not ragged.
        return unicode_encode(
            ragged_tensor.RaggedTensor.from_tensor(input_tensor),
            output_encoding, errors, replacement_char)
      elif input_tensor.shape.ndims > 2:
        # We need to initially flatten the input tensor to 2-D, and then can
        # reshape the output of our processed flattened tensor.
        flat_input_tensor = array_ops.reshape(
            input_tensor,
            array_ops.stack([-1, array_ops.shape(input_tensor)[-1]]))
        flat_output_tensor = unicode_encode(flat_input_tensor, output_encoding,
                                            errors, replacement_char)
        return array_ops.reshape(flat_output_tensor, input_tensor.shape[:-1])
      elif input_tensor.shape.ndims == 0:
        raise ValueError("input_tensor's rank must be at least 1.")
      else:
        # Our input tensor is rank 1, so we create a ragged tensor with an added
        # dimension to create the correct input shape & type, and then remove
        # the additional dimension from the output and return the string scalar.
        ragged_input_tensor = ragged_tensor.RaggedTensor.from_row_splits(
            input_tensor,
            array_ops.stack(
                [0, array_ops.shape(input_tensor, out_type=dtypes.int32)[0]]),
            validate=False)
        output_tensor = unicode_encode(ragged_input_tensor, output_encoding,
                                       errors, replacement_char)
        return array_ops.reshape(output_tensor, [])


# pylint: disable=redefined-builtin
@tf_export("strings.unicode_decode")
@dispatch.add_dispatch_support
def unicode_decode(input,
                   input_encoding,
                   errors="replace",
                   replacement_char=0xFFFD,
                   replace_control_characters=False,
                   name=None):
  r"""Decodes each string in `input` into a sequence of Unicode code points.

  `result[i1...iN, j]` is the Unicode codepoint for the `j`th character in
  `input[i1...iN]`, when decoded using `input_encoding`.

  Args:
    input: An `N` dimensional potentially ragged `string` tensor with shape
      `[D1...DN]`.  `N` must be statically known.
    input_encoding: String name for the unicode encoding that should be used to
      decode each string.
    errors: Specifies the response when an input string can't be converted
      using the indicated encoding. One of:
      * `'strict'`: Raise an exception for any illegal substrings.
      * `'replace'`: Replace illegal substrings with `replacement_char`.
      * `'ignore'`: Skip illegal substrings.
    replacement_char: The replacement codepoint to be used in place of invalid
      substrings in `input` when `errors='replace'`; and in place of C0 control
      characters in `input` when `replace_control_characters=True`.
    replace_control_characters: Whether to replace the C0 control characters
      `(U+0000 - U+001F)` with the `replacement_char`.
    name: A name for the operation (optional).

  Returns:
    A `N+1` dimensional `int32` tensor with shape `[D1...DN, (num_chars)]`.
    The returned tensor is a `tf.Tensor` if `input` is a scalar, or a
    `tf.RaggedTensor` otherwise.

  #### Example:

  >>> input = [s.encode('utf8') for s in (u'G\xf6\xf6dnight', u'\U0001f60a')]
  >>> tf.strings.unicode_decode(input, 'UTF-8').to_list()
  [[71, 246, 246, 100, 110, 105, 103, 104, 116], [128522]]
  """
  with ops.name_scope(name, "UnicodeDecode", [input]):
    return _unicode_decode(input, input_encoding, errors, replacement_char,
                           replace_control_characters, with_offsets=False)


@tf_export("strings.unicode_decode_with_offsets")
@dispatch.add_dispatch_support
def unicode_decode_with_offsets(input,
                                input_encoding,
                                errors="replace",
                                replacement_char=0xFFFD,
                                replace_control_characters=False,
                                name=None):
  r"""Decodes each string into a sequence of code points with start offsets.

  This op is similar to `tf.strings.decode(...)`, but it also returns the
  start offset for each character in its respective string.  This information
  can be used to align the characters with the original byte sequence.

  Returns a tuple `(codepoints, start_offsets)` where:

  * `codepoints[i1...iN, j]` is the Unicode codepoint for the `j`th character
    in `input[i1...iN]`, when decoded using `input_encoding`.
  * `start_offsets[i1...iN, j]` is the start byte offset for the `j`th
    character in `input[i1...iN]`, when decoded using `input_encoding`.

  Args:
    input: An `N` dimensional potentially ragged `string` tensor with shape
      `[D1...DN]`.  `N` must be statically known.
    input_encoding: String name for the unicode encoding that should be used to
      decode each string.
    errors: Specifies the response when an input string can't be converted
      using the indicated encoding. One of:
      * `'strict'`: Raise an exception for any illegal substrings.
      * `'replace'`: Replace illegal substrings with `replacement_char`.
      * `'ignore'`: Skip illegal substrings.
    replacement_char: The replacement codepoint to be used in place of invalid
      substrings in `input` when `errors='replace'`; and in place of C0 control
      characters in `input` when `replace_control_characters=True`.
    replace_control_characters: Whether to replace the C0 control characters
      `(U+0000 - U+001F)` with the `replacement_char`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `N+1` dimensional tensors `(codepoints, start_offsets)`.

    * `codepoints` is an `int32` tensor with shape `[D1...DN, (num_chars)]`.
    * `offsets` is an `int64` tensor with shape `[D1...DN, (num_chars)]`.

    The returned tensors are `tf.Tensor`s if `input` is a scalar, or
    `tf.RaggedTensor`s otherwise.

  #### Example:

  >>> input = [s.encode('utf8') for s in (u'G\xf6\xf6dnight', u'\U0001f60a')]
  >>> result = tf.strings.unicode_decode_with_offsets(input, 'UTF-8')
  >>> result[0].to_list()  # codepoints
  [[71, 246, 246, 100, 110, 105, 103, 104, 116], [128522]]
  >>> result[1].to_list()  # offsets
  [[0, 1, 3, 5, 6, 7, 8, 9, 10], [0]]

  """
  with ops.name_scope(name, "UnicodeDecodeWithOffsets", [input]):
    return _unicode_decode(input, input_encoding, errors, replacement_char,
                           replace_control_characters, with_offsets=True)


@tf_export("strings.unicode_split")
@dispatch.add_dispatch_support
def unicode_split(input,
                  input_encoding,
                  errors="replace",
                  replacement_char=0xFFFD,
                  name=None):
  r"""Splits each string in `input` into a sequence of Unicode code points.

  `result[i1...iN, j]` is the substring of `input[i1...iN]` that encodes its
  `j`th character, when decoded using `input_encoding`.

  Args:
    input: An `N` dimensional potentially ragged `string` tensor with shape
      `[D1...DN]`.  `N` must be statically known.
    input_encoding: String name for the unicode encoding that should be used to
      decode each string.
    errors: Specifies the response when an input string can't be converted
      using the indicated encoding. One of:
      * `'strict'`: Raise an exception for any illegal substrings.
      * `'replace'`: Replace illegal substrings with `replacement_char`.
      * `'ignore'`: Skip illegal substrings.
    replacement_char: The replacement codepoint to be used in place of invalid
      substrings in `input` when `errors='replace'`.
    name: A name for the operation (optional).

  Returns:
    A `N+1` dimensional `int32` tensor with shape `[D1...DN, (num_chars)]`.
    The returned tensor is a `tf.Tensor` if `input` is a scalar, or a
    `tf.RaggedTensor` otherwise.

  #### Example:

  >>> input = [s.encode('utf8') for s in (u'G\xf6\xf6dnight', u'\U0001f60a')]
  >>> tf.strings.unicode_split(input, 'UTF-8').to_list()
  [[b'G', b'\xc3\xb6', b'\xc3\xb6', b'd', b'n', b'i', b'g', b'h', b't'],
   [b'\xf0\x9f\x98\x8a']]
  """
  with ops.name_scope(name, "UnicodeSplit", [input]):
    codepoints = _unicode_decode(input, input_encoding, errors,
                                 replacement_char, False, with_offsets=False)
    return unicode_encode(
        ragged_array_ops.expand_dims(codepoints, -1),
        output_encoding=input_encoding,
        errors=errors,
        replacement_char=replacement_char)


@tf_export("strings.unicode_split_with_offsets")
@dispatch.add_dispatch_support
def unicode_split_with_offsets(input,
                               input_encoding,
                               errors="replace",
                               replacement_char=0xFFFD,
                               name=None):
  r"""Splits each string into a sequence of code points with start offsets.

  This op is similar to `tf.strings.decode(...)`, but it also returns the
  start offset for each character in its respective string.  This information
  can be used to align the characters with the original byte sequence.

  Returns a tuple `(chars, start_offsets)` where:

  * `chars[i1...iN, j]` is the substring of `input[i1...iN]` that encodes its
    `j`th character, when decoded using `input_encoding`.
  * `start_offsets[i1...iN, j]` is the start byte offset for the `j`th
    character in `input[i1...iN]`, when decoded using `input_encoding`.

  Args:
    input: An `N` dimensional potentially ragged `string` tensor with shape
      `[D1...DN]`.  `N` must be statically known.
    input_encoding: String name for the unicode encoding that should be used to
      decode each string.
    errors: Specifies the response when an input string can't be converted
      using the indicated encoding. One of:
      * `'strict'`: Raise an exception for any illegal substrings.
      * `'replace'`: Replace illegal substrings with `replacement_char`.
      * `'ignore'`: Skip illegal substrings.
    replacement_char: The replacement codepoint to be used in place of invalid
      substrings in `input` when `errors='replace'`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `N+1` dimensional tensors `(codepoints, start_offsets)`.

    * `codepoints` is an `int32` tensor with shape `[D1...DN, (num_chars)]`.
    * `offsets` is an `int64` tensor with shape `[D1...DN, (num_chars)]`.

    The returned tensors are `tf.Tensor`s if `input` is a scalar, or
    `tf.RaggedTensor`s otherwise.

  #### Example:

  >>> input = [s.encode('utf8') for s in (u'G\xf6\xf6dnight', u'\U0001f60a')]
  >>> result = tf.strings.unicode_split_with_offsets(input, 'UTF-8')
  >>> result[0].to_list()  # character substrings
  [[b'G', b'\xc3\xb6', b'\xc3\xb6', b'd', b'n', b'i', b'g', b'h', b't'],
   [b'\xf0\x9f\x98\x8a']]
  >>> result[1].to_list()  # offsets
  [[0, 1, 3, 5, 6, 7, 8, 9, 10], [0]]

  """
  with ops.name_scope(name, "UnicodeSplitWithOffsets", [input]):
    codepoints, offsets = _unicode_decode(input, input_encoding, errors,
                                          replacement_char, False,
                                          with_offsets=True)
    chars = unicode_encode(
        ragged_array_ops.expand_dims(codepoints, -1),
        output_encoding=input_encoding,
        errors=errors,
        replacement_char=replacement_char)
    return chars, offsets


def _unicode_decode(input, input_encoding, errors, replacement_char,
                    replace_control_characters, with_offsets):
  """Decodes each string into a sequence of codepoints."""
  input = ragged_tensor.convert_to_tensor_or_ragged_tensor(input, name="input")
  input_ndims = input.shape.ndims
  if input_ndims is None:
    raise ValueError("Rank of `input` must be statically known.")

  if input_ndims > 1:
    # Convert to a ragged tensor with ragged_rank = input_ndims - 1.
    if not ragged_tensor.is_ragged(input):
      input = ragged_tensor.RaggedTensor.from_tensor(
          input, ragged_rank=input_ndims - 1)
    elif input.ragged_rank < input_ndims - 1:
      input = input.with_flat_values(
          ragged_tensor.RaggedTensor.from_tensor(
              input.flat_values,
              ragged_rank=input_ndims - input.ragged_rank - 1))

  # Reshape the input to a flat vector, and apply the gen_string_ops op.
  if ragged_tensor.is_ragged(input):
    flat_input = array_ops.reshape(input.flat_values, [-1])
  else:
    flat_input = array_ops.reshape(input, [-1])

  if with_offsets:
    decode_op = gen_string_ops.unicode_decode_with_offsets
  else:
    decode_op = gen_string_ops.unicode_decode
  flat_result = decode_op(
      input=flat_input,
      input_encoding=input_encoding,
      errors=errors,
      replacement_char=replacement_char,
      replace_control_characters=replace_control_characters)

  if input_ndims == 0:
    codepoints = flat_result.char_values
    if with_offsets:
      offsets = flat_result.char_to_byte_starts
  else:
    codepoints = ragged_tensor.RaggedTensor.from_row_splits(
        flat_result.char_values, flat_result.row_splits, validate=False)
    if input_ndims > 1:
      codepoints = input.with_flat_values(codepoints)
    if with_offsets:
      offsets = ragged_tensor.RaggedTensor.from_row_splits(
          flat_result.char_to_byte_starts, flat_result.row_splits,
          validate=False)
      if input_ndims > 1:
        offsets = input.with_flat_values(offsets)

  if with_offsets:
    return codepoints, offsets
  else:
    return codepoints


@tf_export("strings.split", v1=[])
@dispatch.add_dispatch_support
def string_split_v2(input, sep=None, maxsplit=-1, name=None):  # pylint: disable=redefined-builtin
  """Split elements of `input` based on `sep` into a `RaggedTensor`.

  Let N be the size of `input` (typically N will be the batch size). Split each
  element of `input` based on `sep` and return a `RaggedTensor` containing the 
  split tokens. Empty tokens are ignored.

  Example:

  >>> tf.strings.split('hello world').numpy()
   array([b'hello', b'world'], dtype=object)
  >>> tf.strings.split(['hello world', 'a b c'])
  <tf.RaggedTensor [[b'hello', b'world'], [b'a', b'b', b'c']]>

  If `sep` is given, consecutive delimiters are not grouped together and are
  deemed to delimit empty strings. For example, `input` of `"1<>2<><>3"` and
  `sep` of `"<>"` returns `["1", "2", "", "3"]`. If `sep` is None or an empty
  string, consecutive whitespace are regarded as a single separator, and the
  result will contain no empty strings at the start or end if the string has
  leading or trailing whitespace.

  Note that the above mentioned behavior matches python's str.split.

  Args:
    input: A string `Tensor` of rank `N`, the strings to split.  If
      `rank(input)` is not known statically, then it is assumed to be `1`.
    sep: `0-D` string `Tensor`, the delimiter string.
    maxsplit: An `int`. If `maxsplit > 0`, limit of the split of the result.
    name: A name for the operation (optional).

  Raises:
    ValueError: If sep is not a string.

  Returns:
    A `RaggedTensor` of rank `N+1`, the strings split according to the
    delimiter.
  """
  with ops.name_scope(name, "StringSplit", [input]):
    input = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        input, dtype=dtypes.string, name="input")
    if isinstance(input, ragged_tensor.RaggedTensor):
      return input.with_flat_values(
          string_split_v2(input.flat_values, sep, maxsplit))

    rank = input.shape.ndims
    if rank == 0:
      return string_split_v2(array_ops.stack([input]), sep, maxsplit)[0]
    elif rank == 1 or rank is None:
      sparse_result = string_ops.string_split_v2(
          input, sep=sep, maxsplit=maxsplit)
      return ragged_tensor.RaggedTensor.from_value_rowids(
          values=sparse_result.values,
          value_rowids=sparse_result.indices[:, 0],
          nrows=sparse_result.dense_shape[0],
          validate=False)
    else:
      return string_split_v2(
          ragged_tensor.RaggedTensor.from_tensor(input), sep, maxsplit)


@tf_export(v1=["string_split"])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None,
                             "delimiter is deprecated, please use sep instead.",
                             "delimiter")
def string_split(source, sep=None, skip_empty=True, delimiter=None,
                 result_type="SparseTensor", name=None):  # pylint: disable=invalid-name
  """Split elements of `source` based on `delimiter`.

  Let N be the size of `source` (typically N will be the batch size). Split each
  element of `source` based on `delimiter` and return a `SparseTensor`
  or `RaggedTensor` containing the split tokens. Empty tokens are ignored.

  If `sep` is an empty string, each element of the `source` is split
  into individual strings, each containing one byte. (This includes splitting
  multibyte sequences of UTF-8.) If delimiter contains multiple bytes, it is
  treated as a set of delimiters with each considered a potential split point.

  Examples:

  >>> print(tf.compat.v1.string_split(['hello world', 'a b c']))
  SparseTensor(indices=tf.Tensor( [[0 0] [0 1] [1 0] [1 1] [1 2]], ...),
               values=tf.Tensor([b'hello' b'world' b'a' b'b' b'c'], ...),
               dense_shape=tf.Tensor([2 3], shape=(2,), dtype=int64))

  >>> print(tf.compat.v1.string_split(['hello world', 'a b c'],
  ...     result_type="RaggedTensor"))
  <tf.RaggedTensor [[b'hello', b'world'], [b'a', b'b', b'c']]>

  Args:
    source: `1-D` string `Tensor`, the strings to split.
    sep: `0-D` string `Tensor`, the delimiter character, the string should
      be length 0 or 1. Default is ' '.
    skip_empty: A `bool`. If `True`, skip the empty strings from the result.
    delimiter: deprecated alias for `sep`.
    result_type: The tensor type for the result: one of `"RaggedTensor"` or
      `"SparseTensor"`.
    name: A name for the operation (optional).

  Raises:
    ValueError: If delimiter is not a string.

  Returns:
    A `SparseTensor` or `RaggedTensor` of rank `2`, the strings split according
    to the delimiter.  The first column of the indices corresponds to the row
    in `source` and the second column corresponds to the index of the split
    component in this row.
  """
  with ops.name_scope(name, "StringSplit", [source]):
    sparse_result = string_ops.string_split(
        source, sep=sep, skip_empty=skip_empty, delimiter=delimiter)
    if result_type == "SparseTensor":
      return sparse_result
    elif result_type == "RaggedTensor":
      return ragged_tensor.RaggedTensor.from_value_rowids(
          values=sparse_result.values,
          value_rowids=sparse_result.indices[:, 0],
          nrows=sparse_result.dense_shape[0],
          validate=False)
    else:
      raise ValueError("result_type must be 'RaggedTensor' or 'SparseTensor'.")


# In TensorFlow 1.x, "tf.strings.split" uses the new signature (with maxsplit),
# but we need to add the result_type argument.
@tf_export(v1=["strings.split"])
@dispatch.add_dispatch_support
def strings_split_v1(input=None, sep=None, maxsplit=-1,  # pylint: disable=redefined-builtin
                     result_type="SparseTensor", source=None, name=None):
  """Split elements of `input` based on `sep`.

  Let N be the size of `input` (typically N will be the batch size). Split each
  element of `input` based on `sep` and return a `SparseTensor` or
  `RaggedTensor` containing the split tokens. Empty tokens are ignored.

  Examples:

  >>> print(tf.compat.v1.strings.split(['hello world', 'a b c']))
  SparseTensor(indices=tf.Tensor( [[0 0] [0 1] [1 0] [1 1] [1 2]], ...),
               values=tf.Tensor([b'hello' b'world' b'a' b'b' b'c'], ...),
               dense_shape=tf.Tensor([2 3], shape=(2,), dtype=int64))

  >>> print(tf.compat.v1.strings.split(['hello world', 'a b c'],
  ...     result_type="RaggedTensor"))
  <tf.RaggedTensor [[b'hello', b'world'], [b'a', b'b', b'c']]>

  If `sep` is given, consecutive delimiters are not grouped together and are
  deemed to delimit empty strings. For example, `input` of `"1<>2<><>3"` and
  `sep` of `"<>"` returns `["1", "2", "", "3"]`. If `sep` is None or an empty
  string, consecutive whitespace are regarded as a single separator, and the
  result will contain no empty strings at the start or end if the string has
  leading or trailing whitespace.

  Note that the above mentioned behavior matches python's str.split.

  Args:
    input: A string `Tensor` of rank `N`, the strings to split.  If
      `rank(input)` is not known statically, then it is assumed to be `1`.
    sep: `0-D` string `Tensor`, the delimiter character.
    maxsplit: An `int`. If `maxsplit > 0`, limit of the split of the result.
    result_type: The tensor type for the result: one of `"RaggedTensor"` or
      `"SparseTensor"`.
    source: alias for "input" argument.
    name: A name for the operation (optional).

  Raises:
    ValueError: If sep is not a string.

  Returns:
    A `SparseTensor` or `RaggedTensor` of rank `N+1`, the strings split
    according to the delimiter.
  """
  input = deprecation.deprecated_argument_lookup(
      "input", input, "source", source)
  with ops.name_scope(name, "StringSplit", [input]):
    input = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        input, dtype=dtypes.string, name="input")

    if input.shape.rank == 0:
      input = array_ops.expand_dims(input, 0)

    if result_type == "SparseTensor":
      if input.shape.rank == 1:
        return string_ops.string_split_v2(input, sep=sep, maxsplit=maxsplit)
      else:
        return string_split_v2(input, sep=sep, maxsplit=maxsplit).to_sparse()
    elif result_type == "RaggedTensor":
      return string_split_v2(input, sep=sep, maxsplit=maxsplit)
    else:
      raise ValueError("result_type must be 'RaggedTensor' or 'SparseTensor'.")


def reduce_join(inputs, axis=None, keepdims=None, separator="", name=None):
  """For docs, see: _RAGGED_REDUCE_DOCSTRING."""
  return ragged_math_ops.ragged_reduce_aggregate(
      string_ops.reduce_join, string_ops.unsorted_segment_join, inputs, axis,
      keepdims, separator, name or "RaggedSegmentJoin")


@tf_export("strings.ngrams")
@dispatch.add_dispatch_support
def ngrams(data,
           ngram_width,
           separator=" ",
           pad_values=None,
           padding_width=None,
           preserve_short_sequences=False,
           name=None):
  """Create a tensor of n-grams based on `data`.

  Creates a tensor of n-grams based on `data`. The n-grams are created by
  joining windows of `width` adjacent strings from the inner axis of `data`
  using `separator`.

  The input data can be padded on both the start and end of the sequence, if
  desired, using the `pad_values` argument. If set, `pad_values` should contain
  either a tuple of strings or a single string; the 0th element of the tuple
  will be used to pad the left side of the sequence and the 1st element of the
  tuple will be used to pad the right side of the sequence. The `padding_width`
  arg controls how many padding values are added to each side; it defaults to
  `ngram_width-1`.

  If this op is configured to not have padding, or if it is configured to add
  padding with `padding_width` set to less than ngram_width-1, it is possible
  that a sequence, or a sequence plus padding, is smaller than the ngram
  width. In that case, no ngrams will be generated for that sequence. This can
  be prevented by setting `preserve_short_sequences`, which will cause the op
  to always generate at least one ngram per non-empty sequence.

  Examples:

  >>> tf.strings.ngrams(["A", "B", "C", "D"], 2).numpy()
  array([b'A B', b'B C', b'C D'], dtype=object)
  >>> tf.strings.ngrams(["TF", "and", "keras"], 1).numpy()
  array([b'TF', b'and', b'keras'], dtype=object)

  Args:
    data: A Tensor or RaggedTensor containing the source data for the ngrams.
    ngram_width: The width(s) of the ngrams to create. If this is a list or
      tuple, the op will return ngrams of all specified arities in list order.
      Values must be non-Tensor integers greater than 0.
    separator: The separator string used between ngram elements. Must be a
      string constant, not a Tensor.
    pad_values: A tuple of (left_pad_value, right_pad_value), a single string,
      or None. If None, no padding will be added; if a single string, then that
      string will be used for both left and right padding. Values must be Python
      strings.
    padding_width: If set, `padding_width` pad values will be added to both
      sides of each sequence. Defaults to `ngram_width`-1. Must be greater than
      0. (Note that 1-grams are never padded, regardless of this value.)
    preserve_short_sequences: If true, then ensure that at least one ngram is
      generated for each input sequence.  In particular, if an input sequence is
      shorter than `min(ngram_width) + 2*pad_width`, then generate a single
      ngram containing the entire sequence.  If false, then no ngrams are
      generated for these short input sequences.
    name: The op name.

  Returns:
    A RaggedTensor of ngrams. If `data.shape=[D1...DN, S]`, then
    `output.shape=[D1...DN, NUM_NGRAMS]`, where
    `NUM_NGRAMS=S-ngram_width+1+2*padding_width`.

  Raises:
    TypeError: if `pad_values` is set to an invalid type.
    ValueError: if `pad_values`, `padding_width`, or `ngram_width` is set to an
      invalid value.
  """

  with ops.name_scope(name, "StringNGrams", [data]):
    if pad_values is None:
      left_pad = ""
      right_pad = ""
    elif isinstance(pad_values, (list, tuple)):
      if (not isinstance(pad_values[0], util_compat.bytes_or_text_types) or
          not isinstance(pad_values[1], util_compat.bytes_or_text_types)):
        raise TypeError(
            "pad_values must be a string, tuple of strings, or None.")
      left_pad = pad_values[0]
      right_pad = pad_values[1]
    else:
      if not isinstance(pad_values, util_compat.bytes_or_text_types):
        raise TypeError(
            "pad_values must be a string, tuple of strings, or None.")
      left_pad = pad_values
      right_pad = pad_values

    if padding_width is not None and padding_width < 1:
      raise ValueError("padding_width must be greater than 0.")

    if padding_width is not None and pad_values is None:
      raise ValueError("pad_values must be provided if padding_width is set.")

    data = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        data, name="data", dtype=dtypes.string)

    # preserve the shape of the data if it is a tensor
    to_tensor = False
    if isinstance(data, ops.Tensor):
      dense_shape = array_ops.concat([array_ops.shape(data)[:-1], [-1]], axis=0)
      to_tensor = True

    if not isinstance(data, ragged_tensor.RaggedTensor):
      if data.shape.ndims is None:
        raise ValueError("Rank of data must be known.")
      elif data.shape.ndims == 0:
        raise ValueError("Data must have rank>0")
      elif data.shape.ndims == 1:
        rt = ragged_tensor.RaggedTensor.from_row_starts(
            data, [0], validate=False)
        return ngrams(rt, ngram_width, separator, pad_values, padding_width,
                      preserve_short_sequences, name)[0]
      else:
        data = ragged_tensor.RaggedTensor.from_tensor(
            data, ragged_rank=data.shape.ndims - 1)

    if data.ragged_rank > 1:
      output = data.with_values(
          ngrams(data.values, ngram_width, separator, pad_values, padding_width,
                 preserve_short_sequences, name))
      return array_ops.reshape(output.flat_values,
                               dense_shape) if to_tensor else output

    if pad_values is None:
      padding_width = 0

    if pad_values is not None and padding_width is None:
      padding_width = -1

    if not isinstance(ngram_width, (list, tuple)):
      ngram_widths = [ngram_width]
    else:
      ngram_widths = ngram_width
    for width in ngram_widths:
      if width < 1:
        raise ValueError("All ngram_widths must be greater than 0. Got %s" %
                         ngram_width)

    output, output_splits = gen_string_ops.string_n_grams(
        data=data.flat_values,
        data_splits=data.row_splits,
        separator=separator,
        ngram_widths=ngram_widths,
        left_pad=left_pad,
        right_pad=right_pad,
        pad_width=padding_width,
        preserve_short_sequences=preserve_short_sequences)

    # if the input is Dense tensor, the output should also be a dense tensor
    output = ragged_tensor.RaggedTensor.from_row_splits(
        values=output, row_splits=output_splits, validate=False)
    return array_ops.reshape(output.flat_values,
                             dense_shape) if to_tensor else output


def string_format(template, inputs, placeholder="{}", summarize=3, name=None):
  """Version of tf.strings.format that handles RaggedTensors."""
  if tensor_util.is_tf_type(inputs) or ragged_tensor.is_ragged(inputs):
    inputs = [inputs]

  split_template = template.split(placeholder)
  if len(inputs) != len(split_template) - 1:
    raise ValueError("num placeholders in template and num inputs must match"
                     ": {} vs {}".format(len(split_template) - 1, len(inputs)))

  with ops.name_scope(name, "StringFormat", [inputs]):
    output_pieces = [constant_op.constant(split_template[0])]
    for i, input in enumerate(inputs):
      if ragged_tensor.is_ragged(input):
        output_pieces.append(ragged_tensor_to_string(input, summarize))
      else:
        output_pieces.append(string_ops.string_format(
            "{}", [input], summarize=summarize))
      output_pieces.append(constant_op.constant(split_template[i + 1]))
    if len(output_pieces) == 1:
      return output_pieces[0]
    else:
      return string_ops.reduce_join(output_pieces)


def ragged_tensor_to_string(rt, summarize=None):
  """Returns a scalar string tensor with the contents of a RaggedTensor.

  Requires that `rt.shape.rank` is not `None`.

  Note: this converts the entire `RaggedTensor` into a single string scalar.
  If you want to convert individual elements, use `tf.strings.as_string(rt)`.

  >>> rt1 = tf.ragged.constant([[1, 2, 3], [4, 5]])
  >>> ragged_tensor_to_string(rt1).numpy()
  b'[[1, 2, 3], [4, 5]]'

  >>> rt2 = tf.ragged.constant([[['a'], ['b', 'c']], [['d', 'e', 'f'], []]])
  >>> ragged_tensor_to_string(rt2).numpy()
  b"[[['a'], ['b', 'c']], [['d', 'e', 'f'], []]]"

  >>> rt3 = tf.ragged.constant([[1], [2, 3, 4, 5, 6], [], [], [7], [8, 9]])
  >>> ragged_tensor_to_string(rt3, summarize=2).numpy()
  b'[[1], [2, 3, ..., 5, 6], ..., [7], [8, 9]]'

  Args:
    rt: The RaggedTensor that should be converted to a string.
    summarize: If specified, then only the first and last `summarize` elements
      within each dimension are included in the string. If `-1` or `None`, then
      all elements are included.
  """
  if (summarize is not None and summarize != -1 and
      not (isinstance(summarize, int) and summarize > 0)):
    raise ValueError("Expected summarize to be -1 or a positive int, got %r" %
                     summarize)
  with ops.name_scope(None, "AsString", [rt]):
    rt = ragged_tensor.convert_to_tensor_or_ragged_tensor(rt)
    if rt.shape.rank is None:
      raise ValueError("RaggedTensor to_string requires that rt.shape.rank "
                       "is not None.")
    # Convert all elements of `rt` to strings.
    if rt.dtype == dtypes.string:
      escaped = string_ops.regex_replace(rt.flat_values, r"(['\\])", r"\\\1")
      str_t = rt.with_flat_values("'" + escaped + "'")
    else:
      str_t = rt.with_flat_values(string_ops.as_string(rt.flat_values))

    return _ragged_tensor_to_string(str_t, summarize)


def _ragged_tensor_to_string(string_tensor, summarize):
  """Returns a scalar string tensor with the contents of `string_tensor`.

  Args:
    string_tensor: A potentially ragged tensor with dtype=string.
    summarize: Include only the first and last `summarize` elements of each
      dimension.  If `-1` or `None`, then include all elements.

  Returns:
    A scalar string Tensor.
  """
  if string_tensor.shape.rank == 1:
    pieces = string_tensor
  else:
    pieces = map_fn_lib.map_fn(
        lambda s: _ragged_tensor_to_string(s, summarize),
        string_tensor,
        fn_output_signature=tensor_spec.TensorSpec(None, dtypes.string))
  if summarize not in (-1, None):
    pieces = control_flow_ops.cond(
        _nrows(string_tensor) <= 2 * summarize,
        lambda: pieces,
        lambda: array_ops.concat(  # pylint: disable=g-long-lambda
            [pieces[:summarize], ["..."], pieces[-summarize:]],
            axis=0))
  return "[" + string_ops.reduce_join(pieces, separator=", ") + "]"


def _nrows(tensor, out_type=dtypes.int32):
  if isinstance(tensor, ragged_tensor.RaggedTensor):
    return tensor.nrows(out_type=out_type)
  else:
    return array_ops.shape(tensor, out_type=out_type)[0]
