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
from tensorflow.contrib.tensorrt.wrap_conversion import trt_convert
from tensorflow.contrib.tensorrt.wrap_conversion import get_loaded_tensorrt_version
from tensorflow.contrib.tensorrt.wrap_conversion import get_linked_tensorrt_version
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl as _impl
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import compat

# pylint: enable=unused-import,line-too-long


# TODO(skama): get outputs from session when implemented as c++
# optimization pass
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

  out_names = []
  for i in outputs:
    if isinstance(i, ops.Tensor):
      out_names.append(to_bytes(i.name))
    else:
      out_names.append(to_bytes(i))

  input_graph_def_str = input_graph_def.SerializeToString()

  # TODO(sami): Fix this when we can return status from C++ library
  # There is a problem with the TF internal library setup that doesn't
  # allow us to return a status object from C++.  Thus we return a
  # pair or strings where first one is encoded status and the second
  # one is the transformed graphs protobuf string.
  out = trt_convert(input_graph_def_str, out_names, max_batch_size,
                    max_workspace_size_bytes, mode, minimum_segment_size,
                    is_dynamic_op, maximum_cached_engines,
                    cached_engine_batches)
  status = to_string(out[0])
  output_graph_def_string = out[1]
  del input_graph_def_str  # Save some memory
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


def calib_graph_to_infer_graph(calibration_graph_def):
  """Convert an existing calibration graph to inference graph.

  Args:
    calibration_graph_def: the calibration GraphDef object with calibration data
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
      is_calib_graph = len(n.attr["calibration_data"].s) == 0
  if not is_calib_graph:
    tf_logging.error(
        "Not a calib graph. Doesn't seem to contain any calibration nodes.")
    return None
  graph_str = calibration_graph_def.SerializeToString()
  out = calib_convert(graph_str)
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
