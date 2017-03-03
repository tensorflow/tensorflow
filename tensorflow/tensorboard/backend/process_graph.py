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
"""Graph post-processing logic. Used by both TensorBoard and mldash."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util import compat


def prepare_graph_for_ui(graph, limit_attr_size=1024,
                         large_attrs_key='_too_large_attrs'):
  """Prepares (modifies in-place) the graph to be served to the front-end.

  For now, it supports filtering out attributes that are
  too large to be shown in the graph UI.

  Args:
    graph: The GraphDef proto message.
    limit_attr_size: Maximum allowed size in bytes, before the attribute
        is considered large. Default is 1024 (1KB). Must be > 0 or None.
        If None, there will be no filtering.
    large_attrs_key: The attribute key that will be used for storing attributes
        that are too large. Default is '_too_large_attrs'. Must be != None if
        `limit_attr_size` is != None.

  Raises:
    ValueError: If `large_attrs_key is None` while `limit_attr_size != None`.
    ValueError: If `limit_attr_size` is defined, but <= 0.
  """
  # Check input for validity.
  if limit_attr_size is not None:
    if large_attrs_key is None:
      raise ValueError('large_attrs_key must be != None when limit_attr_size'
                       '!= None.')

    if limit_attr_size <= 0:
      raise ValueError('limit_attr_size must be > 0, but is %d' %
                       limit_attr_size)

  # Filter only if a limit size is defined.
  if limit_attr_size is not None:
    for node in graph.node:
      # Go through all the attributes and filter out ones bigger than the
      # limit.
      keys = list(node.attr.keys())
      for key in keys:
        size = node.attr[key].ByteSize()
        if size > limit_attr_size or size < 0:
          del node.attr[key]
          # Add the attribute key to the list of "too large" attributes.
          # This is used in the info card in the graph UI to show the user
          # that some attributes are too large to be shown.
          node.attr[large_attrs_key].list.s.append(compat.as_bytes(key))

