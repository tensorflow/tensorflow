# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

import collections
import os
import re

from packaging import version

from tensorflow.compiler.tf2tensorrt import _pywrap_py_utils
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import dtypes


def disable_non_trt_optimizers_in_rewriter_config(rewriter_config):
  """Modifies rewriter_config to disable all non-TRT optimizations."""
  off = rewriter_config_pb2.RewriterConfig.OFF

  rewriter_config.arithmetic_optimization = off
  rewriter_config.auto_mixed_precision = off
  rewriter_config.auto_parallel.enable = False
  rewriter_config.constant_folding = off
  rewriter_config.debug_stripper = off
  rewriter_config.dependency_optimization = off
  # This one needs to be ON to allow TF-TRT
  rewriter_config.disable_meta_optimizer = False
  rewriter_config.disable_model_pruning = True
  rewriter_config.function_optimization = off
  rewriter_config.implementation_selector = off
  rewriter_config.layout_optimizer = off
  rewriter_config.loop_optimization = off
  rewriter_config.memory_optimization = (
      rewriter_config_pb2.RewriterConfig.NO_MEM_OPT)
  rewriter_config.min_graph_nodes = -1
  rewriter_config.pin_to_host_optimization = off
  rewriter_config.remapping = off
  rewriter_config.scoped_allocator_optimization = off
  rewriter_config.shape_optimization = off


def version_tuple_to_string(ver_tuple):
  assert isinstance(ver_tuple, tuple)
  assert len(ver_tuple) == 3

  ver_tuple = [str(x) for x in ver_tuple]
  return ".".join(ver_tuple)


def _is_tensorrt_version_greater_equal(trt_ver, target_ver):
  trt_ver = version.Version(version_tuple_to_string(trt_ver))
  target_ver = version.Version(version_tuple_to_string(target_ver))

  return trt_ver >= target_ver


def is_linked_tensorrt_version_greater_equal(major, minor=0, patch=0):
  ver = _pywrap_py_utils.get_linked_tensorrt_version()
  return _is_tensorrt_version_greater_equal(ver, (major, minor, patch))


def is_loaded_tensorrt_version_greater_equal(major, minor=0, patch=0):
  ver = _pywrap_py_utils.get_loaded_tensorrt_version()
  return _is_tensorrt_version_greater_equal(ver, (major, minor, patch))


def is_experimental_feature_activated(feature_name):
  """Determines if a TF-TRT experimental feature is enabled.

  This helper function checks if an experimental feature was enabled using
  the environment variable `TF_TRT_EXPERIMENTAL_FEATURES=feature_1,feature_2`.

  Args:
    feature_name: Name of the feature being tested for activation.
  """

  return (feature_name
          in os.environ.get("TF_TRT_EXPERIMENTAL_FEATURES",
                            default="").split(","))


def _convert_dtype_id_to_str(dtype):
  """Helper function to convert a dtype id to a corresponding string name."""
  if isinstance(dtype, int):
    return dtypes._TYPE_TO_STRING[dtype]
  else:
    return [dtypes._TYPE_TO_STRING[d] for d in dtype]


def get_node_compute_dtype(node):
  """Returns the compute DType of a GraphDef Node."""
  # Note: Order is important, by default TF Node compute dtype is mentioned
  # under `T` key, unless these nodes are one of ["TRTEngineOP", "Cast", "Plh"].
  for type_key in [
      "precision_mode",  # TRTEngineOp
      "DstT",  # Cast Nodes
      "dtype",  # Placeholder
      "T",  # Everything Else
  ]:
    try:
      precision_val = node.attr[type_key]
      if type_key == "precision_mode":
        precision_val = precision_val.s.decode("utf-8")
        if precision_val == "":
          continue
        if precision_val == "FP32":
          return "float32"
        elif precision_val == "FP16":
          return "float16"
        elif precision_val == "INT8":
          return "int8"
        else:
          return "unknown"
      else:
        return _convert_dtype_id_to_str(precision_val.type)
    except Exception as e:
      continue


def get_node_io_shapes(node, key):
  """Returns the input/output shapes of a GraphDef Node."""
  out_shape = []
  for shape in node.attr[key].list.shape:
    out_shape.append([dim.size for dim in shape.dim])
  return out_shape


def get_trtengineop_io_dtypes(node, key):
  """Returns the input/output dtypes of a TRTEngineOp."""
  return _convert_dtype_id_to_str(node.attr[key].list.type)


def get_trtengineop_io_nodes_count(node, key):
  """Returns the number of input/output nodes of a TRTEngineOp."""
  return len(node.attr[key].list.type)


def get_trtengineop_node_op_count(graphdef, node_name):
  """Counts the number of nodes and OP types of a given TRTEngineOp."""
  ops_in_engine = collections.defaultdict(int)
  for func in graphdef.library.function:
    if f"{node_name}_native_segment" == func.signature.name:
      node_count = len(func.node_def)
      for node in func.node_def:
        ops_in_engine[node.op] += 1
      break
  return node_count, ops_in_engine


class DTypeIndex(dict):
  """Helper class to create an index of dtypes with incremental values."""

  def get_dtype_index(self, dtype):
    if dtype not in self:
      self[dtype] = len(self) + 1
    return self[dtype]


def draw_graphdef_as_graphviz(graphdef, dot_output_filename):
  """Exports a GraphDef to GraphViz format.

  - Step 1: Drawing Each Node of the compute GraphDef.
  - Step 2: Create nodes for each collected dtype in the graph.
  - Step 3: Creating invisible links to align properly the legend.

  Each node consequently mentions:
  - Op Type
  - Compute Dtype
  - Compute Device
  """

  dtype_index = DTypeIndex()

  with open(dot_output_filename, "w") as f:
    print("digraph tftrt_converted_graph {", file=f)

    print("  graph [fontsize=10 fontname=\"Verdana\"];", file=f)
    # ColorScheme Documentation: https://graphviz.org/doc/info/colors.html
    print(
        "  node [style=filled height=0.55 colorscheme=set312 shape=box];",
        file=f)

    # Step 1: Parsing the graph and drawing OPs one by one.
    print("\n  subgraph tensorflow_graph {", file=f)
    print("    node [width=1.35];", file=f)
    nodes_with_no_inputs = []
    for node in graphdef.node:
      output_name = node.name

      node_precision = get_node_compute_dtype(node)
      color_idx = dtype_index.get_dtype_index(node_precision)

      device_key = node.device.split("/")[-1]
      if not device_key:
        device_key = "device:Unspecified"

      if node.op == "TRTEngineOp":
        node_count, _ = get_trtengineop_node_op_count(graphdef, output_name)
        node_label = f"{output_name} [{node_count}]"
      else:
        node_label = f"{node.op}"

      # Note: double space before <br/> is necessary for formatting.
      node_label = f"<b>{node_label}</b>  <br/><i>{device_key}</i>"

      print(
          f"    \"{output_name}\" [label=<{node_label}> "
          f"fillcolor={color_idx}];",
          file=f)

      if len(node.input):
        for input_full_name in node.input:
          parts = input_full_name.split(":")
          input_name = re.sub(r"^\^", "", parts[0])
          print(f"  \"{input_name}\" -> \"{output_name}\";", file=f)
      else:
        nodes_with_no_inputs.append(output_name)
    print("  }", file=f)

    # Step 2: Creating the DType Nodes previously found in Step 1.
    print("\n  subgraph cluster_legend {", file=f)
    print("    label=\"Compute Dtype Legend\";", file=f)
    print("    margin=\"30\";", file=f)
    print("    node [width=2];", file=f)

    for dtype, color_idx in dtype_index.items():
      print(
          f"    {dtype} [fillcolor={color_idx} label=<<b>{dtype}</b>>];",
          file=f)

    print("  }", file=f)

    # Step 3: Alignement of the legend with the graph.
    print("\n  edge[style=\"invisible\", dir=\"none\"];", file=f)
    for dtype in dtype_index.keys():
      for node_name in nodes_with_no_inputs:
        print(f"  \"{dtype}\" -> \"{node_name}\"", file=f)

    print("}", file=f)

  print("\n===================================================================")
  print(f"Graph Visualization Exported to: `{dot_output_filename}`.")
  print("We recommend using https://edotor.net/ to visualize the .dot file.")
  print("You can also use `graphviz` utility to convert them to PNG format:")
  print("  - `sudo apt install -y graphviz`")
  print("  - `dot -Tpng <input_filename>.dot -o <output_filename>.png`")
  print("===================================================================\n")
