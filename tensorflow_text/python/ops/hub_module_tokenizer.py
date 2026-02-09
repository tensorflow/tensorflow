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

"""Tokenizer that uses a Hub module."""

from tensorflow.python.eager import monitoring
from tensorflow_text.python.ops import hub_module_splitter
from tensorflow_text.python.ops.tokenization import TokenizerWithOffsets

_tf_text_hub_module_tokenizer_create_counter = monitoring.Counter(
    '/nlx/api/python/hub_module_tokenizer_create_counter',
    'Counter for number of HubModuleTokenizers created in Python.')


class HubModuleTokenizer(TokenizerWithOffsets):
  r"""Tokenizer that uses a Hub module.

  This class is just a wrapper around an internal HubModuleSplitter.  It offers
  the same functionality, but with 'token'-based method names: e.g., one can use
  tokenize() instead of the more general and less informatively named split().

  Example:

  import tensorflow_hub as hub
  HUB_MODULE = "https://tfhub.dev/google/zh_segmentation/1"
  segmenter = HubModuleTokenizer(hub.resolve(HUB_MODULE))
  segmenter.tokenize(["新华社北京"])
  <tf.RaggedTensor [[b'\xe6\x96\xb0\xe5\x8d\x8e\xe7\xa4\xbe',
                     b'\xe5\x8c\x97\xe4\xba\xac']]>

  You can also use this tokenizer to return the split strings and their offsets:

  import tensorflow_hub as hub
  HUB_MODULE = "https://tfhub.dev/google/zh_segmentation/1"
  segmenter = HubModuleTokenizer(hub.resolve(HUB_MODULE))
  pieces, starts, ends = segmenter.tokenize_with_offsets(["新华社北京"])
  print("pieces: %s starts: %s ends: %s" % (pieces, starts, ends))
  pieces: <tf.RaggedTensor [[b'\xe6\x96\xb0\xe5\x8d\x8e\xe7\xa4\xbe',
                             b'\xe5\x8c\x97\xe4\xba\xac']]>
  starts: <tf.RaggedTensor [[0, 9]]>
  ends: <tf.RaggedTensor [[9, 15]]>

  """

  def __init__(self, hub_module_handle):
    """Initializes a new HubModuleTokenizer instance.

    Args:
      hub_module_handle: A string handle accepted by hub.load().  Supported
        cases include (1) a local path to a directory containing a module, and
        (2) a handle to a module uploaded to e.g., https://tfhub.dev
    """
    super(HubModuleTokenizer, self).__init__()
    self._splitter = hub_module_splitter.HubModuleSplitter(hub_module_handle)

  def tokenize_with_offsets(self, input_strs):
    """Tokenizes a tensor of UTF-8 strings into words with [start,end) offsets.

    Args:
      input_strs: An N-dimensional `Tensor` or `RaggedTensor` of UTF-8 strings.

    Returns:
      A tuple `(tokens, start_offsets, end_offsets)` where:
        * `tokens` is a `RaggedTensor` of strings where `tokens[i1...iN, j]` is
          the string content of the `j-th` token in `input_strs[i1...iN]`
        * `start_offsets` is a `RaggedTensor` of int64s where
          `start_offsets[i1...iN, j]` is the byte offset for the start of the
          `j-th` token in `input_strs[i1...iN]`.
        * `end_offsets` is a `RaggedTensor` of int64s where
          `end_offsets[i1...iN, j]` is the byte offset immediately after the
          end of the `j-th` token in `input_strs[i...iN]`.
    """
    return self._splitter.split_with_offsets(input_strs)

  def tokenize(self, input_strs):
    """Tokenizes a tensor of UTF-8 strings into words.

    Args:
      input_strs: An N-dimensional `Tensor` or `RaggedTensor` of UTF-8 strings.

    Returns:
      A `RaggedTensor` of segmented text. The returned shape is the shape of the
      input tensor with an added ragged dimension for tokens of each string.
    """
    return self._splitter.split(input_strs)
