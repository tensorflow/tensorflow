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

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

"""Python class that implements Sentencepiece tokenizer.

It follows TF.text designers design.

"""
import tensorflow.compat.v2 as tf  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops.ragged import ragged_tensor  # pylint: disable=g-direct-tensorflow-import
from tensorflow_text.core.pybinds import pywrap_model_converter

# pylint: disable=g-bad-import-order
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_fast_sentencepiece_tokenizer = load_library.load_op_library(resource_loader.get_path_to_datafile('_fast_sentencepiece_tokenizer.so'))


class FastSentencepieceTokenizer:
  """Sentencepiece tokenizer with tf.text interface."""

  def __init__(self, model, reverse=False, add_bos=False, add_eos=False):
    converted_model = pywrap_model_converter.convert_sentencepiece_model(model)
    converted_model_detokenizer = pywrap_model_converter.convert_sentencepiece_model_for_decoder(
        model)
    # Use uint8 tensor as a buffer for the model to avoid any possible changes,
    # for example truncation by '\0'.
    self._converted_model = tf.constant(list(converted_model), dtype=tf.uint8)
    self._converted_model_detokenizer = tf.constant(
        list(converted_model_detokenizer), dtype=tf.uint8)
    self._vocab_size = pywrap_model_converter.get_vocabulary_size(
        converted_model)
    self._reverse = reverse
    self._add_bos = add_bos
    self._add_eos = add_eos

  def tokenize(self, inputs):
    """The main tokenization function."""
    input_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(inputs)
    if input_tensor.shape.ndims is None:
      raise ValueError("Rank of input_tensor must be statically known.")
    if ragged_tensor.is_ragged(input_tensor):
      # Ensure that input has row_split_dtype is int32
      input_tensor = input_tensor.with_row_splits_dtype(tf.int32)
      # Recursively process the values of the ragged tensor.
      tokens = self.tokenize(input_tensor.flat_values)
      return input_tensor.with_flat_values(tokens)
    else:
      if input_tensor.shape.ndims > 1:
        # Convert the input tensor to ragged and process it.
        return self.tokenize(
            tf.RaggedTensor.from_tensor(
                input_tensor, row_splits_dtype=tf.int32))
      elif input_tensor.shape.ndims == 0:
        tokens = self.tokenize(tf.stack([input_tensor]))
        return tokens.values
      else:
        # Our rank 1 tensor is the correct shape, so we can process it as
        # normal.
        (output_values, row_splits) = (
            gen_fast_sentencepiece_tokenizer
            .tf_text_fast_sentencepiece_tokenize(
                self._converted_model, input_tensor, 0, 0, self._add_bos,
                self._add_eos, self._reverse))
        tokens = tf.RaggedTensor.from_nested_row_splits(
            flat_values=output_values,
            nested_row_splits=[row_splits],
            validate=False)
        return tokens

  def detokenize(self, input):  # pylint: disable=redefined-builtin
    """Detokenizes tokens into preprocessed text.

    Args:
      input: A `RaggedTensor` or `Tensor` with int32 encoded text with rank >=
        1.

    Returns:
      A N-1 dimensional string Tensor or RaggedTensor of the detokenized text.
    """
    input_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(input)
    if input_tensor.shape.ndims is None:
      raise ValueError("Rank of input_tensor must be statically known.")
    if input_tensor.shape.ndims == 0:
      raise ValueError("Rank of input_tensor must be at least 1.")
    if ragged_tensor.is_ragged(input_tensor):
      if input_tensor.flat_values.shape.ndims > 1:
        # If the flat_values of our ragged tensor is multi-dimensional, we can
        # process it separately and our output will have the same nested
        # splits as our input.
        tokens = self.detokenize(input_tensor.flat_values)
        return input_tensor.with_flat_values(tokens)
      elif input_tensor.ragged_rank > 1:
        # Recursively process the values of the ragged tensor.
        tokens = self.detokenize(input_tensor.values)
        return input_tensor.with_values(tokens)
      else:
        return gen_fast_sentencepiece_tokenizer.tf_text_fast_sentencepiece_detokenize(
            self._converted_model_detokenizer, input_tensor.flat_values,
            input_tensor.row_splits)
    else:
      if input_tensor.shape.ndims > 1:
        # Convert the input tensor to ragged and process it.
        return self.detokenize(
            tf.RaggedTensor.from_tensor(
                input_tensor, row_splits_dtype=tf.int32))
      else:
        tokens = self.detokenize(tf.stack([input_tensor]))
        return tf.reshape(tokens, [])

  def vocab_size(self):
    """Returns size of the vocabulary in Sentencepiece model."""
    return self._vocab_size
