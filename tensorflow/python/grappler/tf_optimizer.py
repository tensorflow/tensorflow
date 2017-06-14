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

from tensorflow.core.framework import graph_pb2
from tensorflow.python import pywrap_tensorflow as tf_opt
from tensorflow.python.framework import errors


def OptimizeGraph(rewriter_config, metagraph, graph_id=b'graph_to_optimize'):
  """Optimize the provided metagraph."""
  with errors.raise_exception_on_not_ok_status() as status:
    ret_from_swig = tf_opt.TF_OptimizeGraph(rewriter_config.SerializeToString(),
                                            metagraph.SerializeToString(),
                                            graph_id, status)
  if ret_from_swig is None:
    return None
  out_graph = graph_pb2.GraphDef().FromString(ret_from_swig)
  return out_graph
