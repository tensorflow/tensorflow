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
"""Test utilities for tf.contrib.signal."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training import saver


def grappler_optimize(graph, fetches=None, rewriter_config=None):
  """Tries to optimize the provided graph using grappler.

  Args:
    graph: A @{tf.Graph} instance containing the graph to optimize.
    fetches: An optional list of `Tensor`s to fetch (i.e. not optimize away).
      Grappler uses the 'train_op' collection to look for fetches, so if not
      provided this collection should be non-empty.
    rewriter_config: An optional @{tf.RewriterConfig} to use when rewriting the
      graph.

  Returns:
    A @{tf.GraphDef} containing the rewritten graph.
  """
  if rewriter_config is None:
    rewriter_config = rewriter_config_pb2.RewriterConfig()
  if fetches is not None:
    for fetch in fetches:
      graph.add_to_collection('train_op', fetch)
  metagraph = saver.export_meta_graph(graph_def=graph.as_graph_def())
  return tf_optimizer.OptimizeGraph(rewriter_config, metagraph)
