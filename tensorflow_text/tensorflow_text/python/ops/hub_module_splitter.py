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

"""Splitter that uses a Hub module."""

from tensorflow.python.eager import monitoring
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import load
from tensorflow_text.python.ops.splitter import SplitterWithOffsets

_tf_text_hub_module_splitter_create_counter = monitoring.Counter(
    '/nlx/api/python/hub_module_splitter_create_counter',
    'Counter for number of HubModuleSplitters created in Python.')


class HubModuleSplitter(SplitterWithOffsets):
  r"""Splitter that uses a Hub module.

  The TensorFlow graph from the module performs the real work.  The Python code
  from this class handles the details of interfacing with that module, as well
  as the support for ragged tensors and high-rank tensors.

  The Hub module should be supported by `hub.load()
  <https://www.tensorflow.org/hub/api_docs/python/hub/load>`_ If a v1 module, it
  should have a graph variant with an empty set of tags; we consider that graph
  variant to be the module and ignore everything else.  The module should have a
  signature named `default` that takes a `text` input (a rank-1 tensor of
  strings to split into pieces) and returns a dictionary of tensors, let's say
  `output_dict`, such that:

  * `output_dict['num_pieces']` is a rank-1 tensor of integers, where
  num_pieces[i] is the number of pieces that text[i] was split into.

  * `output_dict['pieces']` is a rank-1 tensor of strings containing all pieces
  for text[0] (in order), followed by all pieces for text[1] (in order) and so
  on.

  * `output_dict['starts']` is a rank-1 tensor of integers with the byte offsets
  where the pieces start (relative to the beginning of the corresponding input
  string).

  * `output_dict['end']` is a rank-1 tensor of integers with the byte offsets
  right after the end of the tokens (relative to the beginning of the
  corresponding input string).

  The output dictionary may contain other tensors (e.g., for debugging) but this
  class is not using them.

  Example:

  import tensorflow_hub as hub
  HUB_MODULE = "https://tfhub.dev/google/zh_segmentation/1"
  segmenter = HubModuleSplitter(hub.resolve(HUB_MODULE))
  segmenter.split(["新华社北京"])
  <tf.RaggedTensor [[b'\xe6\x96\xb0\xe5\x8d\x8e\xe7\xa4\xbe',
                     b'\xe5\x8c\x97\xe4\xba\xac']]>

  You can also use this tokenizer to return the split strings and their offsets:

  import tensorflow_hub as hub
  HUB_MODULE = "https://tfhub.dev/google/zh_segmentation/1"
  segmenter = HubModuleSplitter(hub.resolve(HUB_MODULE))
  pieces, starts, ends = segmenter.split_with_offsets(["新华社北京"])
  print("pieces: %s starts: %s ends: %s" % (pieces, starts, ends))
  pieces: <tf.RaggedTensor [[b'\xe6\x96\xb0\xe5\x8d\x8e\xe7\xa4\xbe',
                             b'\xe5\x8c\x97\xe4\xba\xac']]>
  starts: <tf.RaggedTensor [[0, 9]]>
  ends: <tf.RaggedTensor [[9, 15]]>


  Currently, this class also supports an older API, which uses slightly
  different key names for the output dictionary.  For new Hub modules, please
  use the API described above.
  """

  def __init__(self, hub_module_handle):
    """Initializes a new HubModuleSplitter instance.

    Args:
      hub_module_handle: A string handle accepted by tf.saved_model.load().
        Supported cases include a local path to a directory containing a module.
        If a model is stored on https://tfhub.dev, call hub.resolve() to
        download the model locally. The module should implement the signature
        described in the docstring for this class.
    """
    super(HubModuleSplitter, self).__init__()
    empty_tags = set()
    self._hub_module = load.load(hub_module_handle, tags=empty_tags)
    self._hub_module_signature = self._hub_module.signatures['default']
    _tf_text_hub_module_splitter_create_counter.get_cell().increase_by(1)

  def _predict_pieces(self, input_strs):
    output_dict = self._hub_module_signature(text=input_strs)
    if 'tokens' in output_dict:
      # Use the legacy hub module API.  That API was originally intended only
      # for tokenization, hence the 'token'-heavy string literals:
      pieces = output_dict['tokens']
      num_pieces = output_dict['num_tokens']
      starts = output_dict['starts']
      ends = output_dict['ends']
    else:
      pieces = output_dict['pieces']
      num_pieces = output_dict['num_pieces']
      starts = output_dict['starts']
      ends = output_dict['ends']

    pieces = ragged_tensor.RaggedTensor.from_row_lengths(
        pieces, row_lengths=num_pieces)
    starts = ragged_tensor.RaggedTensor.from_row_lengths(
        starts, row_lengths=num_pieces)
    ends = ragged_tensor.RaggedTensor.from_row_lengths(
        ends, row_lengths=num_pieces)
    return pieces, starts, ends

  def split_with_offsets(self, input_strs):
    """Splits a tensor of UTF-8 strings into pieces with [start,end) offsets.

    Args:
      input_strs: An N-dimensional `Tensor` or `RaggedTensor` of UTF-8 strings.

    Returns:
      A tuple `(pieces, start_offsets, end_offsets)` where:
        * `pieces` is a `RaggedTensor` of strings where `pieces[i1...iN, j]` is
          the string content of the `j-th` piece in `input_strs[i1...iN]`
        * `start_offsets` is a `RaggedTensor` of int64s where
          `start_offsets[i1...iN, j]` is the byte offset for the start of the
          `j-th` piece in `input_strs[i1...iN]`.
        * `end_offsets` is a `RaggedTensor` of int64s where
          `end_offsets[i1...iN, j]` is the byte offset immediately after the
          end of the `j-th` piece in `input_strs[i...iN]`.
    """
    input_strs = ragged_tensor.convert_to_tensor_or_ragged_tensor(input_strs)
    rank = input_strs.shape.ndims
    if rank is None:
      raise ValueError('input must have a known rank.')

    # Currently, the hub_module accepts only rank 1 input tensors, and outputs
    # rank 2 pieces/starts/ends.  To handle input of different ranks (0, 2, 3,
    # etc), we first convert the input into a rank 1 tensor, then run the
    # module, and finally convert the output back to the expected shape.
    if rank == 0:
      # Build a rank 1 input batch with one string.
      input_batch = array_ops_stack.stack([input_strs])
      # [1, (number pieces)]
      pieces, starts, ends = self._predict_pieces(input_batch)
      return pieces.flat_values, starts.flat_values, ends.flat_values
    elif rank == 1:
      return self._predict_pieces(input_strs)
    else:
      if not ragged_tensor.is_ragged(input_strs):
        input_strs = ragged_tensor.RaggedTensor.from_tensor(
            input_strs, ragged_rank=rank - 1)

      # [number strings, (number pieces)]
      pieces, starts, ends = self._predict_pieces(input_strs.flat_values)
      pieces = input_strs.with_flat_values(pieces)
      starts = input_strs.with_flat_values(starts)
      ends = input_strs.with_flat_values(ends)
    return pieces, starts, ends

  def split(self, input_strs):
    """Splits a tensor of UTF-8 strings into pieces.

    Args:
      input_strs: An N-dimensional `Tensor` or `RaggedTensor` of UTF-8 strings.

    Returns:
      A `RaggedTensor` of segmented text. The returned shape is the shape of the
      input tensor with an added ragged dimension for the pieces of each string.
    """
    pieces, _, _ = self.split_with_offsets(input_strs)
    return pieces
