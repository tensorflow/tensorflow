#!/usr/bin/env python
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
"""This tool creates an html visualization of a TensorFlow Lite graph.

Example usage:

python visualize.py foo.tflite foo.html
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import re
import sys
import numpy as np

from tensorflow.lite.python import schema_py_generated as schema_fb

# A CSS description for making the visualizer
_CSS = """
<html>
<head>
<style>
body {font-family: sans-serif; background-color: #fa0;}
table {background-color: #eca;}
th {background-color: black; color: white;}
h1 {
  background-color: ffaa00;
  padding:5px;
  color: black;
}

svg {
  margin: 10px;
  border: 2px;
  border-style: solid;
  border-color: black;
  background: white;
}

div {
  border-radius: 5px;
  background-color: #fec;
  padding:5px;
  margin:5px;
}

.tooltip {color: blue;}
.tooltip .tooltipcontent  {
    visibility: hidden;
    color: black;
    background-color: yellow;
    padding: 5px;
    border-radius: 4px;
    position: absolute;
    z-index: 1;
}
.tooltip:hover .tooltipcontent {
    visibility: visible;
}

.edges line {
  stroke: #333;
}

text {
  font-weight: bold;
}

.nodes text {
  color: black;
  pointer-events: none;
  font-family: sans-serif;
  font-size: 11px;
}
</style>

<script src="https://d3js.org/d3.v4.min.js"></script>

</head>
<body>
"""

_D3_HTML_TEMPLATE = """
  <script>
    function buildGraph() {
      // Build graph data
      var graph = %s;

      var svg = d3.select("#subgraph%d")
      var width = svg.attr("width");
      var height = svg.attr("height");
      // Make the graph scrollable.
      svg = svg.call(d3.zoom().on("zoom", function() {
        svg.attr("transform", d3.event.transform);
      })).append("g");


      var color = d3.scaleOrdinal(d3.schemeDark2);

      var simulation = d3.forceSimulation()
          .force("link", d3.forceLink().id(function(d) {return d.id;}))
          .force("charge", d3.forceManyBody())
          .force("center", d3.forceCenter(0.5 * width, 0.5 * height));

      var edge = svg.append("g").attr("class", "edges").selectAll("line")
        .data(graph.edges).enter().append("path").attr("stroke","black").attr("fill","none")

      // Make the node group
      var node = svg.selectAll(".nodes")
        .data(graph.nodes)
        .enter().append("g")
        .attr("x", function(d){return d.x})
        .attr("y", function(d){return d.y})
        .attr("transform", function(d) {
          return "translate( " + d.x + ", " + d.y + ")"
        })
        .attr("class", "nodes")
          .call(d3.drag()
              .on("start", function(d) {
                if(!d3.event.active) simulation.alphaTarget(1.0).restart();
                d.fx = d.x;d.fy = d.y;
              })
              .on("drag", function(d) {
                d.fx = d3.event.x; d.fy = d3.event.y;
              })
              .on("end", function(d) {
                if (!d3.event.active) simulation.alphaTarget(0);
                d.fx = d.fy = null;
              }));
      // Within the group, draw a box for the node position and text
      // on the side.

      var node_width = 150;
      var node_height = 30;

      node.append("rect")
          .attr("r", "5px")
          .attr("width", node_width)
          .attr("height", node_height)
          .attr("rx", function(d) { return d.group == 1 ? 1 : 10; })
          .attr("stroke", "#000000")
          .attr("fill", function(d) { return d.group == 1 ? "#dddddd" : "#000000"; })
      node.append("text")
          .text(function(d) { return d.name; })
          .attr("x", 5)
          .attr("y", 20)
          .attr("fill", function(d) { return d.group == 1 ? "#000000" : "#eeeeee"; })
      // Setup force parameters and update position callback


      var node = svg.selectAll(".nodes")
        .data(graph.nodes);

      // Bind the links
      var name_to_g = {}
      node.each(function(data, index, nodes) {
        console.log(data.id)
        name_to_g[data.id] = this;
      });

      function proc(w, t) {
        return parseInt(w.getAttribute(t));
      }
      edge.attr("d", function(d) {
        function lerp(t, a, b) {
          return (1.0-t) * a + t * b;
        }
        var x1 = proc(name_to_g[d.source],"x") + node_width /2;
        var y1 = proc(name_to_g[d.source],"y") + node_height;
        var x2 = proc(name_to_g[d.target],"x") + node_width /2;
        var y2 = proc(name_to_g[d.target],"y");
        var s = "M " + x1 + " " + y1
            + " C " + x1 + " " + lerp(.5, y1, y2)
            + " " + x2 + " " + lerp(.5, y1, y2)
            + " " + x2  + " " + y2
      return s;
    });

  }
  buildGraph()
</script>
"""


def TensorTypeToName(tensor_type):
  """Converts a numerical enum to a readable tensor type."""
  for name, value in schema_fb.TensorType.__dict__.items():
    if value == tensor_type:
      return name
  return None


def BuiltinCodeToName(code):
  """Converts a builtin op code enum to a readable name."""
  for name, value in schema_fb.BuiltinOperator.__dict__.items():
    if value == code:
      return name
  return None


def NameListToString(name_list):
  """Converts a list of integers to the equivalent ASCII string."""
  if isinstance(name_list, str):
    return name_list
  else:
    result = ""
    if name_list is not None:
      for val in name_list:
        result = result + chr(int(val))
    return result


class OpCodeMapper(object):
  """Maps an opcode index to an op name."""

  def __init__(self, data):
    self.code_to_name = {}
    for idx, d in enumerate(data["operator_codes"]):
      self.code_to_name[idx] = BuiltinCodeToName(d["builtin_code"])
      if self.code_to_name[idx] == "CUSTOM":
        self.code_to_name[idx] = NameListToString(d["custom_code"])

  def __call__(self, x):
    if x not in self.code_to_name:
      s = "<UNKNOWN>"
    else:
      s = self.code_to_name[x]
    return "%s (%d)" % (s, x)


class DataSizeMapper(object):
  """For buffers, report the number of bytes."""

  def __call__(self, x):
    if x is not None:
      return "%d bytes" % len(x)
    else:
      return "--"


class TensorMapper(object):
  """Maps a list of tensor indices to a tooltip hoverable indicator of more."""

  def __init__(self, subgraph_data):
    self.data = subgraph_data

  def __call__(self, x):
    html = ""
    html += "<span class='tooltip'><span class='tooltipcontent'>"
    for i in x:
      tensor = self.data["tensors"][i]
      html += str(i) + " "
      html += NameListToString(tensor["name"]) + " "
      html += TensorTypeToName(tensor["type"]) + " "
      html += (repr(tensor["shape"]) if "shape" in tensor else "[]")
      html += (repr(tensor["shape_signature"])
               if "shape_signature" in tensor else "[]") + "<br>"
    html += "</span>"
    html += repr(x)
    html += "</span>"
    return html


def GenerateGraph(subgraph_idx, g, opcode_mapper):
  """Produces the HTML required to have a d3 visualization of the dag."""

  def TensorName(idx):
    return "t%d" % idx

  def OpName(idx):
    return "o%d" % idx

  edges = []
  nodes = []
  first = {}
  second = {}
  pixel_mult = 200  # TODO(aselle): multiplier for initial placement
  width_mult = 170  # TODO(aselle): multiplier for initial placement
  for op_index, op in enumerate(g["operators"]):

    for tensor_input_position, tensor_index in enumerate(op["inputs"]):
      if tensor_index not in first:
        first[tensor_index] = ((op_index - 0.5 + 1) * pixel_mult,
                               (tensor_input_position + 1) * width_mult)
      edges.append({
          "source": TensorName(tensor_index),
          "target": OpName(op_index)
      })
    for tensor_output_position, tensor_index in enumerate(op["outputs"]):
      if tensor_index not in second:
        second[tensor_index] = ((op_index + 0.5 + 1) * pixel_mult,
                                (tensor_output_position + 1) * width_mult)
      edges.append({
          "target": TensorName(tensor_index),
          "source": OpName(op_index)
      })

    nodes.append({
        "id": OpName(op_index),
        "name": opcode_mapper(op["opcode_index"]),
        "group": 2,
        "x": pixel_mult,
        "y": (op_index + 1) * pixel_mult
    })
  for tensor_index, tensor in enumerate(g["tensors"]):
    initial_y = (
        first[tensor_index] if tensor_index in first else
        second[tensor_index] if tensor_index in second else (0, 0))

    nodes.append({
        "id": TensorName(tensor_index),
        "name": "%r (%d)" % (getattr(tensor, "shape", []), tensor_index),
        "group": 1,
        "x": initial_y[1],
        "y": initial_y[0]
    })
  graph_str = json.dumps({"nodes": nodes, "edges": edges})

  html = _D3_HTML_TEMPLATE % (graph_str, subgraph_idx)
  return html


def GenerateTableHtml(items, keys_to_print, display_index=True):
  """Given a list of object values and keys to print, make an HTML table.

  Args:
    items: Items to print an array of dicts.
    keys_to_print: (key, display_fn). `key` is a key in the object. i.e.
      items[0][key] should exist. display_fn is the mapping function on display.
      i.e. the displayed html cell will have the string returned by
      `mapping_fn(items[0][key])`.
    display_index: add a column which is the index of each row in `items`.

  Returns:
    An html table.
  """
  html = ""
  # Print the list of  items
  html += "<table><tr>\n"
  html += "<tr>\n"
  if display_index:
    html += "<th>index</th>"
  for h, mapper in keys_to_print:
    html += "<th>%s</th>" % h
  html += "</tr>\n"
  for idx, tensor in enumerate(items):
    html += "<tr>\n"
    if display_index:
      html += "<td>%d</td>" % idx
    # print tensor.keys()
    for h, mapper in keys_to_print:
      val = tensor[h] if h in tensor else None
      val = val if mapper is None else mapper(val)
      html += "<td>%s</td>\n" % val

    html += "</tr>\n"
  html += "</table>\n"
  return html


def CamelCaseToSnakeCase(camel_case_input):
  """Converts an identifier in CamelCase to snake_case."""
  s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_case_input)
  return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def FlatbufferToDict(fb, preserve_as_numpy):
  """Converts a hierarchy of FB objects into a nested dict.

  We avoid transforming big parts of the flat buffer into python arrays. This
  speeds conversion from ten minutes to a few seconds on big graphs.

  Args:
    fb: a flat buffer structure. (i.e. ModelT)
    preserve_as_numpy: true if all downstream np.arrays should be preserved.
      false if all downstream np.array should become python arrays
  Returns:
    A dictionary representing the flatbuffer rather than a flatbuffer object.
  """
  if isinstance(fb, int) or isinstance(fb, float) or isinstance(fb, str):
    return fb
  elif hasattr(fb, "__dict__"):
    result = {}
    for attribute_name in dir(fb):
      attribute = fb.__getattribute__(attribute_name)
      if not callable(attribute) and attribute_name[0] != "_":
        snake_name = CamelCaseToSnakeCase(attribute_name)
        preserve = True if attribute_name == "buffers" else preserve_as_numpy
        result[snake_name] = FlatbufferToDict(attribute, preserve)
    return result
  elif isinstance(fb, np.ndarray):
    return fb if preserve_as_numpy else fb.tolist()
  elif hasattr(fb, "__len__"):
    return [FlatbufferToDict(entry, preserve_as_numpy) for entry in fb]
  else:
    return fb


def CreateDictFromFlatbuffer(buffer_data):
  model_obj = schema_fb.Model.GetRootAsModel(buffer_data, 0)
  model = schema_fb.ModelT.InitFromObj(model_obj)
  return FlatbufferToDict(model, preserve_as_numpy=False)


def CreateHtmlFile(tflite_input, html_output):
  """Given a tflite model in `tflite_input` file, produce html description."""

  # Convert the model into a JSON flatbuffer using flatc (build if doesn't
  # exist.
  if not os.path.exists(tflite_input):
    raise RuntimeError("Invalid filename %r" % tflite_input)
  if tflite_input.endswith(".tflite") or tflite_input.endswith(".bin"):
    with open(tflite_input, "rb") as file_handle:
      file_data = bytearray(file_handle.read())
    data = CreateDictFromFlatbuffer(file_data)
  elif tflite_input.endswith(".json"):
    data = json.load(open(tflite_input))
  else:
    raise RuntimeError("Input file was not .tflite or .json")
  html = ""
  html += _CSS
  html += "<h1>TensorFlow Lite Model</h2>"

  data["filename"] = tflite_input  # Avoid special case
  toplevel_stuff = [("filename", None), ("version", None),
                    ("description", None)]

  html += "<table>\n"
  for key, mapping in toplevel_stuff:
    if not mapping:
      mapping = lambda x: x
    html += "<tr><th>%s</th><td>%s</td></tr>\n" % (key, mapping(data.get(key)))
  html += "</table>\n"

  # Spec on what keys to display
  buffer_keys_to_display = [("data", DataSizeMapper())]
  operator_keys_to_display = [("builtin_code", BuiltinCodeToName),
                              ("custom_code", NameListToString),
                              ("version", None)]

  # Update builtin code fields.
  for idx, d in enumerate(data["operator_codes"]):
    d["builtin_code"] = max(d["builtin_code"], d["deprecated_builtin_code"])

  for subgraph_idx, g in enumerate(data["subgraphs"]):
    # Subgraph local specs on what to display
    html += "<div class='subgraph'>"
    tensor_mapper = TensorMapper(g)
    opcode_mapper = OpCodeMapper(data)
    op_keys_to_display = [("inputs", tensor_mapper), ("outputs", tensor_mapper),
                          ("builtin_options", None),
                          ("opcode_index", opcode_mapper)]
    tensor_keys_to_display = [("name", NameListToString),
                              ("type", TensorTypeToName), ("shape", None),
                              ("shape_signature", None), ("buffer", None),
                              ("quantization", None)]

    html += "<h2>Subgraph %d</h2>\n" % subgraph_idx

    # Inputs and outputs.
    html += "<h3>Inputs/Outputs</h3>\n"
    html += GenerateTableHtml([{
        "inputs": g["inputs"],
        "outputs": g["outputs"]
    }], [("inputs", tensor_mapper), ("outputs", tensor_mapper)],
                              display_index=False)

    # Print the tensors.
    html += "<h3>Tensors</h3>\n"
    html += GenerateTableHtml(g["tensors"], tensor_keys_to_display)

    # Print the ops.
    html += "<h3>Ops</h3>\n"
    html += GenerateTableHtml(g["operators"], op_keys_to_display)

    # Visual graph.
    html += "<svg id='subgraph%d' width='1600' height='900'></svg>\n" % (
        subgraph_idx,)
    html += GenerateGraph(subgraph_idx, g, opcode_mapper)
    html += "</div>"

  # Buffers have no data, but maybe in the future they will
  html += "<h2>Buffers</h2>\n"
  html += GenerateTableHtml(data["buffers"], buffer_keys_to_display)

  # Operator codes
  html += "<h2>Operator Codes</h2>\n"
  html += GenerateTableHtml(data["operator_codes"], operator_keys_to_display)

  html += "</body></html>\n"

  with open(html_output, "w") as output_file:
    output_file.write(html)


def main(argv):
  try:
    tflite_input = argv[1]
    html_output = argv[2]
  except IndexError:
    print("Usage: %s <input tflite> <output html>" % (argv[0]))
  else:
    CreateHtmlFile(tflite_input, html_output)


if __name__ == "__main__":
  main(sys.argv)
