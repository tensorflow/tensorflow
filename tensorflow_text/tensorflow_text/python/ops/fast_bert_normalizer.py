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

"""Ops to normalize text for BERT tokenization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import monitoring
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow_text.core.pybinds import pywrap_fast_bert_normalizer_model_builder

# pylint: disable=g-bad-import-order
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_fast_bert_normalizer = load_library.load_op_library(resource_loader.get_path_to_datafile('_fast_bert_normalizer.so'))

_tf_text_fast_bert_normalizer_op_create_counter = monitoring.Counter(
    '/nlx/api/python/fast_bert_normalizer_create_counter',
    'Counter for number of FastBertNormalizers created in Python.')


class FastBertNormalizer(module.Module):
  """Normalizes a tensor of UTF-8 strings."""

  def __init__(self, lower_case_nfd_strip_accents=False, model_buffer=None):
    """Initializes the FastBertNormalizer.

    Two ways to initialize:
      * use a precompiled `model_buffer`.
      * use `lower_case_nfd_strip_accents`.

    Args:
      lower_case_nfd_strip_accents: (optional). - If true, it first lowercases
        the text, applies NFD normalization, strips accents characters, and then
        replaces control characters with whitespaces. - If false, it only
        replaces control characters with whitespaces.
      model_buffer: (optional) bytes object (or a uint8 tf.Tenosr) that contains
        the fast bert normalizer model in flatbuffer format (see
        fast_bert_normalizer_model.fbs). If not `None`, all other arguments are
        ignored.
    """
    super(FastBertNormalizer, self).__init__()
    _tf_text_fast_bert_normalizer_op_create_counter.get_cell().increase_by(1)

    if model_buffer is None:
      model_buffer = (
          pywrap_fast_bert_normalizer_model_builder
          .build_fast_bert_normalizer_model(lower_case_nfd_strip_accents))
    # Use uint8 tensor as a buffer for the model to avoid any possible changes,
    # for example truncation by '\0'.
    if isinstance(model_buffer, tensor.Tensor):
      self._model = model_buffer
    else:
      self._model = constant_op.constant(list(model_buffer), dtype=dtypes.uint8)

  def normalize(self, input):  # pylint: disable=redefined-builtin
    r"""Tokenizes a tensor of UTF-8 strings.

    ### Example:

    >>> texts = [["They're", "the", "Greatest", "\xC0bc"]]
    >>> normalizer = FastBertNormalizer(lower_case_nfd_strip_accents=True)
    >>> normalizer.normalize(texts)
    <tf.RaggedTensor [[b"they're", b'the', b'greatest', b'abc']]>

    Args:
      input: An N-dimensional `Tensor` or `RaggedTensor` of UTF-8 strings.

    Returns:
      An N-dimensional `Tensor` or `RaggedTensor` of UTF-8 strings.
    """
    normalized_texts, _ = self._normalize_with_offsets_helper(
        input, get_offsets=False)
    return normalized_texts

  def normalize_with_offsets(self, input):  # pylint: disable=redefined-builtin
    r"""Normalizes a tensor of UTF-8 strings and returns offsets map.

    ### Example:

    >>> texts = ["They're", "the", "Greatest", "\xC0bc"]
    >>> normalizer = FastBertNormalizer(lower_case_nfd_strip_accents=True)
    >>> normalized_text, offsets = (
    ...   normalizer.normalize_with_offsets(texts))
    >>> normalized_text
    <tf.Tensor: shape=(4,), dtype=string, numpy=array([b"they're", b'the',
    b'greatest', b'abc'], dtype=object)>
    >>> offsets
    <tf.RaggedTensor [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3], [0, 1, 2, 3, 4, 5,
    6, 7, 8], [0, 2, 3, 4]]>

    Args:
      input: An N-dimensional `Tensor` or `RaggedTensor` of UTF-8 strings.

    Returns:
      A tuple `(normalized_texts, offsets)` where:

      normalized_texts: is a `Tensor` or `RaggedTensor`.
      offsets: is a `RaggedTensor` of the byte offsets from the output
        to the input. For example, if the input is `input[i1...iN]` with `N`
        strings, `offsets[i1...iN, k]` is the byte offset in `inputs[i1...iN]`
        for the `kth` byte in `normalized_texts[i1...iN]`. Note that
        `offsets[i1...iN, ...]` also covers the position following the last byte
        in `normalized_texts[i1...iN]`, so that we know the byte offset position
        in `input[i1...iN]` that corresponds to the end of
        `normalized_texts[i1...iN]`.

    """
    return self._normalize_with_offsets_helper(input, get_offsets=True)

  def _normalize_with_offsets_helper(self, input, get_offsets):  # pylint: disable=redefined-builtin
    r"""The actual function of normalization.

    Args:
      input: An N-dimensional `Tensor` or `RaggedTensor` of UTF-8 strings.
      get_offsets: bool. Whether to compute offsets or not.

    Returns:
      A tuple `(texts, offsets)` where:
        texts: is a `Tensor` or `RaggedTensor`.
        offsets: If `get_offsets`=True, it is a `RaggedTensor` of the
        byte offsets from the output to the input. Otherwise, it is None.
    """
    name = None
    with ops.name_scope(name, 'FastBertNormalize', [input, self._model]):
      # Check that the types are expected and the ragged rank is appropriate.
      input = ragged_tensor.convert_to_tensor_or_ragged_tensor(input)
      rank = input.shape.ndims
      if rank is None:
        raise ValueError('input must have a known rank.')

      if rank == 0:
        normalized_texts, offsets = self._normalize_with_offsets_helper(
            array_ops_stack.stack([input]), get_offsets)
        return (normalized_texts.values,
                offsets.values if get_offsets else None)

      elif rank > 1:
        if not ragged_tensor.is_ragged(input):
          input = ragged_tensor.RaggedTensor.from_tensor(
              input, ragged_rank=rank - 1)
        normalized_texts, offsets = self._normalize_with_offsets_helper(
            input.flat_values, get_offsets)
        if get_offsets:
          offsets = offsets.with_row_splits_dtype(
              input.row_splits.dtype)
        return (input.with_flat_values(normalized_texts),
                input.with_flat_values(offsets)
                if get_offsets else None)

      normalized_texts, offsets, row_splits = (
          gen_fast_bert_normalizer.fast_bert_normalize(
              input_values=input,
              fast_bert_normalizer_model=self._model,
              get_offsets=get_offsets))

      if get_offsets:
        offsets = RaggedTensor.from_row_splits(
            offsets, row_splits, validate=False)

      return normalized_texts, offsets
