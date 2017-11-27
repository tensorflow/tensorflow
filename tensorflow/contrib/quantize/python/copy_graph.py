# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Utility to copy a tf.Graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.training import saver as saver_lib


def CopyGraph(graph):
  """Return a copy of graph."""
  meta_graph = saver_lib.export_meta_graph(
      graph=graph, collection_list=graph.get_all_collection_keys())
  graph_copy = ops.Graph()
  with graph_copy.as_default():
    _ = saver_lib.import_meta_graph(meta_graph)
  return graph_copy
