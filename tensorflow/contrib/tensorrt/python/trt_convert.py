# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

# pylint: disable=unused-import,line-too-long
import six as _six
from tensorflow.contrib.tensorrt.wrap_conversion import calib_convert
from tensorflow.contrib.tensorrt.wrap_conversion import get_linked_tensorrt_version
from tensorflow.contrib.tensorrt.wrap_conversion import get_loaded_tensorrt_version
from tensorflow.contrib.tensorrt.wrap_conversion import is_tensorrt_enabled
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import errors_impl as _impl
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import saver
# pylint: enable=unused-import,line-too-long


def create_inference_graph(input_graph_def,
                           outputs,
                           max_batch_size=1,
                           max_workspace_size_bytes=2 << 20,
                           precision_mode="FP32",
                           minimum_segment_size=3,
                           is_dynamic_op=False,
                           maximum_cached_engines=1,
                           cached_engine_batches=[]):
  """Python wrapper for the TRT transformation.

  Args:
    input_graph_def: GraphDef object containing a model to be transformed.
    outputs: list of tensors or node names for the model outputs.
    max_batch_size: max size for the input batch
    max_workspace_size_bytes: parameter to control memory allocation (in Bytes)
    precision_mode: one of 'FP32', 'FP16' and 'INT8'
    minimum_segment_size: the minimum number of nodes required for a subgraph to
      be replaced by TRTEngineOp.
    is_dynamic_op: whether to generate dynamic TRT ops which will build the TRT
      network and engine at run time.
    maximum_cached_engines: max number of cached TRT engines in dynamic TRT ops.
    cached_engine_batches: batch sizes used to pre-create cached engines.

  Returns:
    New GraphDef with TRTEngineOps placed in graph replacing subgraphs.

  Raises:
    ValueError: if the provided precision mode is invalid.
    RuntimeError: if the returned status message is malformed.
  """
  supported_precision_modes = {"FP32": 0, "FP16": 1, "INT8": 2}
  if precision_mode.upper() not in supported_precision_modes:
    raise ValueError(("precision mode '{}' is not supported."
                      "It should be one of {}").format(
                          precision_mode, "{'FP32', 'FP16', 'INT8'}"))
  mode = supported_precision_modes[precision_mode.upper()]
  compiled_version = get_linked_tensorrt_version()
  loaded_version = get_loaded_tensorrt_version()
  version_mismatch = False
  if loaded_version[0] < compiled_version[0]:
    tf_logging.error(
        "TensorRT version mismatch. Tensorflow was compiled against " +
        "TensorRT %s but library loaded from environment is TensorRT %s" %
        (".".join([str(x) for x in compiled_version]),
         ".".join([str(x) for x in loaded_version])) +
        ". Please make sure that correct version of TensorRT " +
        "is available in the system and added to ldconfig or LD_LIBRARY_PATH")
    raise RuntimeError("Incompatible TensorRT library version")
  for i in zip(loaded_version, compiled_version):
    if i[0] != i[1]:
      tf_logging.warn("TensorRT mismatch. Compiled against version " +
                      "%s, but loaded %s. Things may not work" %
                      (".".join([str(x) for x in compiled_version]),
                       ".".join([str(x) for x in loaded_version])))
      version_mismatch = True
      break
  if not version_mismatch:
    tf_logging.info("Running against TensorRT version %s" % ".".join(
        [str(x) for x in loaded_version]))

  def py2bytes(inp):
    return inp

  def py3bytes(inp):
    return inp.encode("utf-8", errors="surrogateescape")

  def py2string(inp):
    return inp

  def py3string(inp):
    return inp.decode("utf-8")

  if _six.PY2:
    to_bytes = py2bytes
    to_string = py2string
  else:
    to_bytes = py3bytes
    to_string = py3string

  # Create MetaGraphDef
  graph = ops.Graph()
  with graph.as_default():
    importer.import_graph_def(input_graph_def, name="")
  meta_graph = saver.export_meta_graph(
      graph_def=graph.as_graph_def(), graph=graph)
  if outputs:
    output_collection = meta_graph_pb2.CollectionDef()
    output_list = output_collection.node_list.value
    for i in outputs:
      if isinstance(i, ops.Tensor):
        output_list.append(to_bytes(i.name))
      else:
        output_list.append(to_bytes(i))
    meta_graph.collection_def["train_op"].CopyFrom(output_collection)

  # Create RewriterConfig.
  rewriter_cfg = rewriter_config_pb2.RewriterConfig()
  rewriter_cfg.optimizers.extend(["constfold", "layout"])
  optimizer = rewriter_cfg.custom_optimizers.add()
  optimizer.name = "TensorRTOptimizer"
  optimizer.parameter_map["minimum_segment_size"].i = minimum_segment_size
  optimizer.parameter_map["max_batch_size"].i = max_batch_size
  optimizer.parameter_map["is_dynamic_op"].b = is_dynamic_op
  optimizer.parameter_map[
      "max_workspace_size_bytes"].i = max_workspace_size_bytes
  optimizer.parameter_map["precision_mode"].s = to_bytes(precision_mode)
  optimizer.parameter_map["maximum_cached_engines"].i = maximum_cached_engines
  if cached_engine_batches:
    optimizer.parameter_map["cached_engine_batches"].list.extend(
        cached_engine_batches)

  return tf_optimizer.OptimizeGraph(
      rewriter_cfg, meta_graph, graph_id=b"tf_graph")


def calib_graph_to_infer_graph(calibration_graph_def, is_dynamic_op=False):
  """Convert an existing calibration graph to inference graph.

  Args:
    calibration_graph_def: the calibration GraphDef object with calibration data
    is_dynamic_op: whether to create dynamic static engines from calibration
  Returns:
    New GraphDef with TRTEngineOps placed in graph replacing calibration nodes.
  Raises:
    RuntimeError: if the returned status message is malformed.
  """

  def py2string(inp):
    return inp

  def py3string(inp):
    return inp.decode("utf-8")

  if _six.PY2:
    to_string = py2string
  else:
    to_string = py3string
  is_calib_graph = False
  for n in calibration_graph_def.node:
    if n.op == "TRTEngineOp":
      is_calib_graph = is_calib_graph or not n.attr["calibration_data"].s
  if not is_calib_graph:
    tf_logging.error(
        "Not a calib graph. Doesn't seem to contain any calibration nodes.")
    return None
  graph_str = calibration_graph_def.SerializeToString()
  out = calib_convert(graph_str, is_dynamic_op)
  status = to_string(out[0])
  output_graph_def_string = out[1]
  del graph_str  # Save some memory
  if len(status) < 2:
    raise _impl.UnknownError(None, None, status)
  if status[:2] != "OK":
    msg = status.split(";")
    if len(msg) == 1:
      raise RuntimeError("Status message is malformed {}".format(status))
    # pylint: disable=protected-access
    raise _impl._make_specific_exception(None, None, ";".join(msg[1:]),
                                         int(msg[0]))
    # pylint: enable=protected-access
  output_graph_def = graph_pb2.GraphDef()
  output_graph_def.ParseFromString(output_graph_def_string)
  del output_graph_def_string  # Save some memory
  return output_graph_def
