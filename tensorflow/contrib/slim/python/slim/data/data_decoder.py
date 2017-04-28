# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Contains helper functions and classes necessary for decoding data.

While data providers read data from disk, sstables or other formats, data
decoders decode the data (if necessary). A data decoder is provided with a
serialized or encoded piece of data as well as a list of items and
returns a set of tensors, each of which correspond to the requested list of
items extracted from the data:

  def Decode(self, data, items):
    ...

For example, if data is a compressed map, the implementation might be:

  def Decode(self, data, items):
    decompressed_map = _Decompress(data)
    outputs = []
    for item in items:
      outputs.append(decompressed_map[item])
    return outputs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class DataDecoder(object):
  """An abstract class which is used to decode data for a provider."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def decode(self, data, items):
    """Decodes the data to returns the tensors specified by the list of items.

    Args:
      data: A possibly encoded data format.
      items: A list of strings, each of which indicate a particular data type.

    Returns:
      A list of `Tensors`, whose length matches the length of `items`, where
      each `Tensor` corresponds to each item.

    Raises:
      ValueError: If any of the items cannot be satisfied.
    """
    pass

  @abc.abstractmethod
  def list_items(self):
    """Lists the names of the items that the decoder can decode.

    Returns:
      A list of string names.
    """
    pass
