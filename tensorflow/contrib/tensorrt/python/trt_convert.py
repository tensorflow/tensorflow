# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Exposes the Python wrapper conversion to trt_graph."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,wildcard-import, line-too-long
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl as _impl
from tensorflow.contrib.tensorrt.wrap_conversion import trt_convert
from tensorflow.python.util import compat
import tensorflow as tf
from tensorflow.python.grappler import tf_optimizer
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops


def CreateInferenceGraph(input_graph_def, outputs,max_batch_size=1,max_workspace_size=2<<20):
  """Python wrapper for the TRT transormation.


  Args:
    input_graph_def: GraphDef object containing a model to be transformed.
    outputs: List of node names for the model outputs.
    max_batch_size: max size for the input batch
    max_workspace_size: parameter to control memory allocation (in Bytes)

  Returns:
    New GraphDef with TRTEngineOps placed in graph replacing subgraphs.
  """

  # with errors.raise_exception_on_not_ok_status() as status:
  #   output_graph_def_string = trt_convert(
  #       input_graph_def_string,outputs,
  #       max_batch_size,max_workspace_size, status)
  g = tf.Graph()
  with g.as_default():
    tf.import_graph_def(input_graph_def, name="")
  rewriter_config = rewriter_config_pb2.RewriterConfig()
  rewriter_config.optimizers.append('layout')
  rewriter_config.optimizers.append('constfold')

  # mark output nodes as fetch
  train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
  for node_name in outputs:
    out_node = g.get_operation_by_name(node_name)
    for i in range(0,len(out_node.outputs)):
      train_op.append(out_node.outputs[0])

  # constant folding
  mg = meta_graph.create_meta_graph_def(graph=g)
  meta_graph.add_collection_def(mg, ops.GraphKeys.TRAIN_OP)
  optimized_graph_def_str = \
    tf_optimizer.OptimizeGraph(rewriter_config, mg).SerializeToString()

  # TODO(sami): Fix this when we can return status from C++ library
  # There is a problem with the TF internal library setup that doesn't allow us to return a status object from C++.
  # Thus we return a  pair or strings where first one is encoded status and the second one is the
  # transformed graphs protobuf string.
  out = trt_convert(
      optimized_graph_def_str ,outputs,
      max_batch_size,max_workspace_size)
  status = out[0]
  output_graph_def_string = out[1]
  del optimized_graph_def_str #save some memory
  if len(status) < 2:
    raise _impl.UnknownError(None,None,status)
  if status[:2] != "OK":
    msg=status.split(";")
    if len(msg) == 1:
      raise RuntimeError("Status message is malformed {}".format(status))
    raise _impl._make_specific_exception(None,None,";".join(msg[1:]), int(msg[0]))
  output_graph_def = graph_pb2.GraphDef()
  output_graph_def.ParseFromString(output_graph_def_string)
  del output_graph_def_string #save some memory
  return output_graph_def
