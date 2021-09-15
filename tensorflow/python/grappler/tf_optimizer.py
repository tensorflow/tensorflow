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
# =============================================================================
"""Provides a proper python API for the symbols exported through swig."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.grappler import _pywrap_tf_optimizer as tf_opt
from tensorflow.python.grappler import cluster as gcluster

_OPTIMIZE_GRAPH_CLUSTER_LOCK = threading.Lock()


def OptimizeGraph(config_proto,
                  metagraph,
                  verbose=True,
                  graph_id=b'graph_to_optimize',
                  cluster=None,
                  strip_default_attributes=False):
  """Optimize the provided metagraph.

  For best results, the signature_def field in `metagraph` should be populated
  with information about input (feed) and output (fetch) tensors.

  Args:
    config_proto: a ConfigProto protobuf.
    metagraph: a MetagraphDef protobuf.
    verbose: whether to log optimization results.
    graph_id: a string identifying this graph.
    cluster: a grappler cluster object representing hardware resources
        available to run this graph.
    strip_default_attributes: whether graph node attributes having default
        values should be removed after all the optimization passes. This
        option is useful if the resulting graph will be executed by an older
        process that might not know some of the recently added attributes.
  """
  if not isinstance(config_proto, config_pb2.ConfigProto):
    raise TypeError('Argument `config_proto` should be a tf.ConfigProto, '
                    f'received type: {type(config_proto).__name__}')
  if cluster is not None:
    out_graph = tf_opt.TF_OptimizeGraph(cluster.tf_cluster,
                                        config_proto.SerializeToString(),
                                        metagraph.SerializeToString(), verbose,
                                        graph_id, strip_default_attributes)
  else:
    # Currently Grappler assumes no more than 1 sessions alive globally.
    # See comments on SingleMachine::Provision(), hence we use the following
    # lock to prevent concurrent access to the following code.
    with _OPTIMIZE_GRAPH_CLUSTER_LOCK:
      cluster = gcluster.Cluster()
      try:
        out_graph = tf_opt.TF_OptimizeGraph(cluster.tf_cluster,
                                            config_proto.SerializeToString(),
                                            metagraph.SerializeToString(),
                                            verbose, graph_id,
                                            strip_default_attributes)
      finally:
        # Force the cleanup instead of waiting on python GC to cleanup the
        # temporary cluster we've created. Otherwise subsequent calls might
        # not have a clean slate because GC may not have run yet.
        cluster.Shutdown()
  return graph_pb2.GraphDef().FromString(out_graph)
