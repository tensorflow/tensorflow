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

"""Operations for working with string Tensors.

See the [Strings](https://tensorflow.org/api_guides/python/string_ops) guide.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.compat import compat
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat as util_compat

# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_string_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
# pylint: enable=wildcard-import


# pylint: disable=redefined-builtin
def regex_full_match(input, pattern, name=None):
  r"""Match elements of `input` with regex `pattern`.

  Args:
    input: string `Tensor`, the source strings to process.
    pattern: string or scalar string `Tensor`, regular expression to use,
      see more details at https://github.com/google/re2/wiki/Syntax
    name: Name of the op.

  Returns:
    bool `Tensor` of the same shape as `input` with match results.
  """
  # TODO(b/112455102): Remove compat.forward_compatible once past the horizon.
  if not compat.forward_compatible(2018, 11, 10):
    return gen_string_ops.regex_full_match(
        input=input, pattern=pattern, name=name)
  if isinstance(pattern, util_compat.bytes_or_text_types):
    # When `pattern` is static through the life of the op we can
    # use a version which performs the expensive regex compilation once at
    # creation time.
    return gen_string_ops.static_regex_full_match(
        input=input, pattern=pattern, name=name)
  return gen_string_ops.regex_full_match(
      input=input, pattern=pattern, name=name)

regex_full_match.__doc__ = gen_string_ops.regex_full_match.__doc__

# Expose regex_full_match in strings namespace
tf_export("strings.regex_full_match")(regex_full_match)


def regex_replace(source, pattern, rewrite, replace_global=True):
  r"""Replace elements of `source` matching regex `pattern` with `rewrite`.

  Args:
    source: string `Tensor`, the source strings to process.
    pattern: string or scalar string `Tensor`, regular expression to use,
      see more details at https://github.com/google/re2/wiki/Syntax
    rewrite: string or scalar string `Tensor`, value to use in match
      replacement, supports backslash-escaped digits (\1 to \9) can be to insert
      text matching corresponding parenthesized group.
    replace_global: `bool`, if `True` replace all non-overlapping matches,
      else replace only the first match.

  Returns:
    string `Tensor` of the same shape as `source` with specified replacements.
  """
  # TODO(b/112455102): Remove compat.forward_compatible once past the horizon.
  if not compat.forward_compatible(2018, 10, 10):
    return gen_string_ops.regex_replace(
        input=source, pattern=pattern,
        rewrite=rewrite, replace_global=replace_global)
  if (isinstance(pattern, util_compat.bytes_or_text_types) and
      isinstance(rewrite, util_compat.bytes_or_text_types)):
    # When `pattern` and `rewrite` are static through the life of the op we can
    # use a version which performs the expensive regex compilation once at
    # creation time.
    return gen_string_ops.static_regex_replace(
        input=source, pattern=pattern,
        rewrite=rewrite, replace_global=replace_global)
  return gen_string_ops.regex_replace(
      input=source, pattern=pattern,
      rewrite=rewrite, replace_global=replace_global)


@tf_export("string_split")
def string_split(source, delimiter=" ", skip_empty=True):  # pylint: disable=invalid-name
  """Split elements of `source` based on `delimiter` into a `SparseTensor`.

  Let N be the size of source (typically N will be the batch size). Split each
  element of `source` based on `delimiter` and return a `SparseTensor`
  containing the split tokens. Empty tokens are ignored.

  If `delimiter` is an empty string, each element of the `source` is split
  into individual strings, each containing one byte. (This includes splitting
  multibyte sequences of UTF-8.) If delimiter contains multiple bytes, it is
  treated as a set of delimiters with each considered a potential split point.

  For example:
  N = 2, source[0] is 'hello world' and source[1] is 'a b c', then the output
  will be

  st.indices = [0, 0;
                0, 1;
                1, 0;
                1, 1;
                1, 2]
  st.shape = [2, 3]
  st.values = ['hello', 'world', 'a', 'b', 'c']

  Args:
    source: `1-D` string `Tensor`, the strings to split.
    delimiter: `0-D` string `Tensor`, the delimiter character, the string should
      be length 0 or 1.
    skip_empty: A `bool`. If `True`, skip the empty strings from the result.

  Raises:
    ValueError: If delimiter is not a string.

  Returns:
    A `SparseTensor` of rank `2`, the strings split according to the delimiter.
    The first column of the indices corresponds to the row in `source` and the
    second column corresponds to the index of the split component in this row.
  """
  delimiter = ops.convert_to_tensor(delimiter, dtype=dtypes.string)
  source = ops.convert_to_tensor(source, dtype=dtypes.string)

  indices, values, shape = gen_string_ops.string_split(
      source, delimiter=delimiter, skip_empty=skip_empty)
  indices.set_shape([None, 2])
  values.set_shape([None])
  shape.set_shape([2])
  return sparse_tensor.SparseTensor(indices, values, shape)


@tf_export("strings.split")
def string_split_v2(source, sep=None, maxsplit=-1):
  """Split elements of `source` based on `sep` into a `SparseTensor`.

  Let N be the size of source (typically N will be the batch size). Split each
  element of `source` based on `sep` and return a `SparseTensor`
  containing the split tokens. Empty tokens are ignored.

  For example, N = 2, source[0] is 'hello world' and source[1] is 'a b c',
  then the output will be

  st.indices = [0, 0;
                0, 1;
                1, 0;
                1, 1;
                1, 2]
  st.shape = [2, 3]
  st.values = ['hello', 'world', 'a', 'b', 'c']

  If `sep` is given, consecutive delimiters are not grouped together and are
  deemed to delimit empty strings. For example, source of `"1<>2<><>3"` and
  sep of `"<>"` returns `["1", "2", "", "3"]`. If `sep` is None or an empty
  string, consecutive whitespace are regarded as a single separator, and the
  result will contain no empty strings at the startor end if the string has
  leading or trailing whitespace.

  Note that the above mentioned behavior matches python's str.split.

  Args:
    source: `1-D` string `Tensor`, the strings to split.
    sep: `0-D` string `Tensor`, the delimiter character.
    maxsplit: An `int`. If `maxsplit > 0`, limit of the split of the result.

  Raises:
    ValueError: If sep is not a string.

  Returns:
    A `SparseTensor` of rank `2`, the strings split according to the delimiter.
    The first column of the indices corresponds to the row in `source` and the
    second column corresponds to the index of the split component in this row.
  """
  if sep is None:
    sep = ""
  sep = ops.convert_to_tensor(sep, dtype=dtypes.string)
  source = ops.convert_to_tensor(source, dtype=dtypes.string)

  indices, values, shape = gen_string_ops.string_split_v2(
      source, sep=sep, maxsplit=maxsplit)
  indices.set_shape([None, 2])
  values.set_shape([None])
  shape.set_shape([2])
  return sparse_tensor.SparseTensor(indices, values, shape)


def _reduce_join_reduction_dims(x, axis, reduction_indices):
  """Returns range(rank(x) - 1, 0, -1) if reduction_indices is None."""
  # TODO(aselle): Remove this after deprecation
  if reduction_indices is not None:
    if axis is not None:
      raise ValueError("Can't specify both 'axis' and 'reduction_indices'.")
    axis = reduction_indices
  if axis is not None:
    return axis
  else:
    # Fast path: avoid creating Rank and Range ops if ndims is known.
    if x.get_shape().ndims is not None:
      return constant_op.constant(
          np.arange(x.get_shape().ndims - 1, -1, -1), dtype=dtypes.int32)

    # Otherwise, we rely on Range and Rank to do the right thing at run-time.
    return math_ops.range(array_ops.rank(x) - 1, -1, -1)


@tf_export("reduce_join")
def reduce_join(inputs, axis=None,
                keep_dims=False,
                separator="",
                name=None,
                reduction_indices=None):
  inputs_t = ops.convert_to_tensor(inputs)
  reduction_indices = _reduce_join_reduction_dims(
      inputs_t, axis, reduction_indices)
  return gen_string_ops.reduce_join(
      inputs=inputs_t,
      reduction_indices=reduction_indices,
      keep_dims=keep_dims,
      separator=separator,
      name=name)


reduce_join.__doc__ = deprecation.rewrite_argument_docstring(
    gen_string_ops.reduce_join.__doc__, "reduction_indices", "axis")

ops.NotDifferentiable("RegexReplace")
ops.NotDifferentiable("StringToHashBucket")
ops.NotDifferentiable("StringToHashBucketFast")
ops.NotDifferentiable("StringToHashBucketStrong")
ops.NotDifferentiable("ReduceJoin")
ops.NotDifferentiable("StringJoin")
ops.NotDifferentiable("StringSplit")
ops.NotDifferentiable("AsString")
ops.NotDifferentiable("EncodeBase64")
ops.NotDifferentiable("DecodeBase64")
