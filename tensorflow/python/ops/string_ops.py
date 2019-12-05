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

"""Operations for working with string Tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import math_ops

# go/tf-wildcard-import
# pylint: disable=wildcard-import
# pylint: disable=g-bad-import-order
from tensorflow.python.ops.gen_string_ops import *
from tensorflow.python.util import compat as util_compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
# pylint: enable=g-bad-import-order
# pylint: enable=wildcard-import


# pylint: disable=redefined-builtin
@tf_export("strings.regex_full_match")
@dispatch.add_dispatch_support
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
  if isinstance(pattern, util_compat.bytes_or_text_types):
    # When `pattern` is static through the life of the op we can
    # use a version which performs the expensive regex compilation once at
    # creation time.
    return gen_string_ops.static_regex_full_match(
        input=input, pattern=pattern, name=name)
  return gen_string_ops.regex_full_match(
      input=input, pattern=pattern, name=name)

regex_full_match.__doc__ = gen_string_ops.regex_full_match.__doc__


@tf_export(
    "strings.regex_replace", v1=["strings.regex_replace", "regex_replace"])
@deprecation.deprecated_endpoints("regex_replace")
@dispatch.add_dispatch_support
def regex_replace(input, pattern, rewrite, replace_global=True, name=None):
  r"""Replace elements of `input` matching regex `pattern` with `rewrite`.

  Args:
    input: string `Tensor`, the source strings to process.
    pattern: string or scalar string `Tensor`, regular expression to use,
      see more details at https://github.com/google/re2/wiki/Syntax
    rewrite: string or scalar string `Tensor`, value to use in match
      replacement, supports backslash-escaped digits (\1 to \9) can be to insert
      text matching corresponding parenthesized group.
    replace_global: `bool`, if `True` replace all non-overlapping matches,
      else replace only the first match.
    name: A name for the operation (optional).

  Returns:
    string `Tensor` of the same shape as `input` with specified replacements.
  """
  if (isinstance(pattern, util_compat.bytes_or_text_types) and
      isinstance(rewrite, util_compat.bytes_or_text_types)):
    # When `pattern` and `rewrite` are static through the life of the op we can
    # use a version which performs the expensive regex compilation once at
    # creation time.
    return gen_string_ops.static_regex_replace(
        input=input, pattern=pattern,
        rewrite=rewrite, replace_global=replace_global,
        name=name)
  return gen_string_ops.regex_replace(
      input=input, pattern=pattern,
      rewrite=rewrite, replace_global=replace_global,
      name=name)


@tf_export("strings.format")
def string_format(template, inputs, placeholder="{}", summarize=3, name=None):
  r"""Formats a string template using a list of tensors.

  Formats a string template using a list of tensors, abbreviating tensors by
  only printing the first and last `summarize` elements of each dimension
  (recursively). If formatting only one tensor into a template, the tensor does
  not have to be wrapped in a list.

  Example:
    Formatting a single-tensor template:
    ```python
    sess = tf.compat.v1.Session()
    with sess.as_default():
        tensor = tf.range(10)
        formatted = tf.strings.format("tensor: {}, suffix", tensor)
        out = sess.run(formatted)
        expected = "tensor: [0 1 2 ... 7 8 9], suffix"

        assert(out.decode() == expected)
    ```

    Formatting a multi-tensor template:
    ```python
    sess = tf.compat.v1.Session()
    with sess.as_default():
        tensor_one = tf.reshape(tf.range(100), [10, 10])
        tensor_two = tf.range(10)
        formatted = tf.strings.format("first: {}, second: {}, suffix",
          (tensor_one, tensor_two))

        out = sess.run(formatted)
        expected = ("first: [[0 1 2 ... 7 8 9]\n"
              " [10 11 12 ... 17 18 19]\n"
              " [20 21 22 ... 27 28 29]\n"
              " ...\n"
              " [70 71 72 ... 77 78 79]\n"
              " [80 81 82 ... 87 88 89]\n"
              " [90 91 92 ... 97 98 99]], second: [0 1 2 ... 7 8 9], suffix")

        assert(out.decode() == expected)
    ```

  Args:
    template: A string template to format tensor values into.
    inputs: A list of `Tensor` objects, or a single Tensor.
      The list of tensors to format into the template string. If a solitary
      tensor is passed in, the input tensor will automatically be wrapped as a
      list.
    placeholder: An optional `string`. Defaults to `{}`.
      At each placeholder occurring in the template, a subsequent tensor
      will be inserted.
    summarize: An optional `int`. Defaults to `3`.
      When formatting the tensors, show the first and last `summarize`
      entries of each tensor dimension (recursively). If set to -1, all
      elements of the tensor will be shown.
    name: A name for the operation (optional).

  Returns:
    A scalar `Tensor` of type `string`.

  Raises:
    ValueError: if the number of placeholders does not match the number of
      inputs.
  """
  # If there is only one tensor to format, we will automatically wrap it in a
  # list to simplify the user experience
  if tensor_util.is_tensor(inputs):
    inputs = [inputs]
  if template.count(placeholder) != len(inputs):
    raise ValueError("%s placeholder(s) in template does not match %s tensor(s)"
                     " provided as input" % (template.count(placeholder),
                                             len(inputs)))

  return gen_string_ops.string_format(inputs,
                                      template=template,
                                      placeholder=placeholder,
                                      summarize=summarize,
                                      name=name)


# Note: tf.strings.split is exported in ragged/ragged_string_ops.py, which
# defines a wrapper for this function.
def string_split(source, sep=None, skip_empty=True, delimiter=None):  # pylint: disable=invalid-name
  """Split elements of `source` based on `delimiter` into a `SparseTensor`.

  Let N be the size of source (typically N will be the batch size). Split each
  element of `source` based on `delimiter` and return a `SparseTensor`
  containing the split tokens. Empty tokens are ignored.

  If `sep` is an empty string, each element of the `source` is split
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
    sep: `0-D` string `Tensor`, the delimiter character, the string should
      be length 0 or 1. Default is ' '.
    skip_empty: A `bool`. If `True`, skip the empty strings from the result.
    delimiter: deprecated alias for `sep`.

  Raises:
    ValueError: If delimiter is not a string.

  Returns:
    A `SparseTensor` of rank `2`, the strings split according to the delimiter.
    The first column of the indices corresponds to the row in `source` and the
    second column corresponds to the index of the split component in this row.
  """
  delimiter = deprecation.deprecated_argument_lookup(
      "sep", sep, "delimiter", delimiter)

  if delimiter is None:
    delimiter = " "
  delimiter = ops.convert_to_tensor(delimiter, dtype=dtypes.string)
  source = ops.convert_to_tensor(source, dtype=dtypes.string)

  indices, values, shape = gen_string_ops.string_split(
      source, delimiter=delimiter, skip_empty=skip_empty)
  indices.set_shape([None, 2])
  values.set_shape([None])
  shape.set_shape([2])
  return sparse_tensor.SparseTensor(indices, values, shape)


# Note: tf.strings.split is exported in ragged/ragged_string_ops.py, which
# defines a wrapper for this function.
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
  result will contain no empty strings at the start or end if the string has
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


def _reduce_join_reduction_dims(x, axis):
  """Returns range(rank(x) - 1, 0, -1) if axis is None; or axis otherwise."""
  if axis is not None:
    return axis
  else:
    # Fast path: avoid creating Rank and Range ops if ndims is known.
    if x.get_shape().ndims is not None:
      return constant_op.constant(
          np.arange(x.get_shape().ndims - 1, -1, -1), dtype=dtypes.int32)

    # Otherwise, we rely on Range and Rank to do the right thing at run-time.
    return math_ops.range(array_ops.rank(x) - 1, -1, -1)


@tf_export(v1=["strings.reduce_join", "reduce_join"])
@deprecation.deprecated_args(None,
                             "keep_dims is deprecated, use keepdims instead",
                             "keep_dims")
@deprecation.deprecated_endpoints("reduce_join")
def reduce_join(inputs, axis=None,  # pylint: disable=missing-docstring
                keep_dims=None,
                separator="",
                name=None,
                reduction_indices=None,
                keepdims=None):
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  if keep_dims is None:
    keep_dims = False
  axis = deprecation.deprecated_argument_lookup("axis", axis,
                                                "reduction_indices",
                                                reduction_indices)
  return reduce_join_v2(
      inputs=inputs,
      axis=axis,
      keepdims=keepdims,
      separator=separator,
      name=name)


@tf_export("strings.reduce_join", v1=[])
@dispatch.add_dispatch_support
def reduce_join_v2(  # pylint: disable=missing-docstring
    inputs,
    axis=None,
    keepdims=False,
    separator="",
    name=None):
  """Joins all strings into a single string, or joins along an axis.

  >>> tf.strings.reduce_join([['abc','123'],
  ...                         ['def','456']]).numpy()
  b'abc123def456'
  >>> tf.strings.reduce_join([['abc','123'],
  ...                         ['def','456']], axis=-1).numpy()
  array([b'abc123', b'def456'], dtype=object)
  >>> tf.strings.reduce_join([['abc','123'],
  ...                         ['def','456']],
  ...                        axis=-1,
  ...                        separator=" ").numpy()
  array([b'abc 123', b'def 456'], dtype=object)

  Args:
    inputs: A `tf.string` tensor.
    axis: Which axis to join along. The default behavior is to join all
      elements, producing a scalar.
    keepdims: If true, retains reduced dimensions with length 1.
    separator: a string added between each string being joined.
    name: A name for the operation (optional).

  Returns:
    A `tf.string` tensor.
  """
  with ops.name_scope(None, "ReduceJoin", [inputs, axis]):
    inputs_t = ops.convert_to_tensor(inputs)
    axis = _reduce_join_reduction_dims(inputs_t, axis)
    return gen_string_ops.reduce_join(
        inputs=inputs_t,
        reduction_indices=axis,
        keep_dims=keepdims,
        separator=separator,
        name=name)

reduce_join.__doc__ = reduce_join_v2.__doc__


# This wrapper provides backwards compatibility for code that predates the
# unit argument and that passed 'name' as a positional argument.
@tf_export(v1=["strings.length"])
@dispatch.add_dispatch_support
def string_length(input, name=None, unit="BYTE"):
  return gen_string_ops.string_length(input, unit=unit, name=name)


@tf_export("strings.length", v1=[])
@dispatch.add_dispatch_support
def string_length_v2(input, unit="BYTE", name=None):
  return string_length(input, name, unit)


string_length.__doc__ = gen_string_ops.string_length.__doc__


@tf_export(v1=["substr"])
@deprecation.deprecated(None, "Use `tf.strings.substr` instead of `tf.substr`.")
def substr_deprecated(input, pos, len, name=None, unit="BYTE"):
  return substr(input, pos, len, name=name, unit=unit)

substr_deprecated.__doc__ = gen_string_ops.substr.__doc__


@tf_export(v1=["strings.substr"])
@dispatch.add_dispatch_support
def substr(input, pos, len, name=None, unit="BYTE"):
  return gen_string_ops.substr(input, pos, len, unit=unit, name=name)

substr.__doc__ = gen_string_ops.substr.__doc__


@tf_export("strings.substr", v1=[])
@dispatch.add_dispatch_support
def substr_v2(input, pos, len, unit="BYTE", name=None):
  return gen_string_ops.substr(input, pos, len, unit=unit, name=name)

substr_v2.__doc__ = gen_string_ops.substr.__doc__


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


@tf_export("strings.to_number", v1=[])
@dispatch.add_dispatch_support
def string_to_number(input, out_type=dtypes.float32, name=None):
  r"""Converts each string in the input Tensor to the specified numeric type.

  (Note that int32 overflow results in an error while float overflow
  results in a rounded value.)

  Args:
    input: A `Tensor` of type `string`.
    out_type: An optional `tf.DType` from: `tf.float32, tf.float64, tf.int32,
      tf.int64`. Defaults to `tf.float32`.
      The numeric type to interpret each string in `string_tensor` as.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
  """
  return gen_parsing_ops.string_to_number(input, out_type, name)


@tf_export(v1=["strings.to_number", "string_to_number"])
def string_to_number_v1(
    string_tensor=None,
    out_type=dtypes.float32,
    name=None,
    input=None):
  string_tensor = deprecation.deprecated_argument_lookup(
      "input", input, "string_tensor", string_tensor)
  return gen_parsing_ops.string_to_number(string_tensor, out_type, name)

string_to_number_v1.__doc__ = gen_parsing_ops.string_to_number.__doc__


@tf_export("strings.to_hash_bucket", v1=[])
@dispatch.add_dispatch_support
def string_to_hash_bucket(input, num_buckets, name=None):
  # pylint: disable=line-too-long
  r"""Converts each string in the input Tensor to its hash mod by a number of buckets.

  The hash function is deterministic on the content of the string within the
  process.

  Note that the hash function may change from time to time.
  This functionality will be deprecated and it's recommended to use
  `tf.strings.to_hash_bucket_fast()` or `tf.strings.to_hash_bucket_strong()`.

  Args:
    input: A `Tensor` of type `string`.
    num_buckets: An `int` that is `>= 1`. The number of buckets.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  # pylint: enable=line-too-long
  return gen_string_ops.string_to_hash_bucket(input, num_buckets, name)


@tf_export(v1=["strings.to_hash_bucket", "string_to_hash_bucket"])
def string_to_hash_bucket_v1(
    string_tensor=None,
    num_buckets=None,
    name=None,
    input=None):
  string_tensor = deprecation.deprecated_argument_lookup(
      "input", input, "string_tensor", string_tensor)
  return gen_string_ops.string_to_hash_bucket(string_tensor, num_buckets, name)

string_to_hash_bucket_v1.__doc__ = gen_string_ops.string_to_hash_bucket.__doc__
