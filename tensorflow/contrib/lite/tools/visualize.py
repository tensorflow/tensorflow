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
import sys

# Schema to use for flatbuffers
_SCHEMA = "third_party/tensorflow/contrib/lite/schema/schema.fbs"

# Where the binary will be once built in for the flatc converter
_BINARY = "third_party/flatbuffers/flatc"

# A CSS description for making the visualizer
_CSS = """
<html>
<head>
<style>
body {font-family: sans-serif; background-color: #ffaa00;}
table {background-color: #eeccaa;}
th {background-color: black; color: white;}
h1 {
  background-color: ffaa00;
  padding:5px;
  color: black;
}

div {
  border-radius: 5px;
  background-color: #ffeecc;
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
  stroke: #333333;
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
    // Build graph data
    var graph = %s;

    var svg = d3.select("#subgraph%d");
    var width = svg.attr("width");
    var height = svg.attr("height");
    var color = d3.scaleOrdinal(d3.schemeCategory20);

    var simulation = d3.forceSimulation()
        .force("link", d3.forceLink().id(function(d) {return d.id;}))
        .force("charge", d3.forceManyBody())
        .force("center", d3.forceCenter(0.5 * width, 0.5 * height));


    function buildGraph() {
      var edge = svg.append("g").attr("class", "edges").selectAll("line")
        .data(graph.edges).enter().append("line")
      // Make the node group
      var node = svg.selectAll(".nodes")
        .data(graph.nodes)
        .enter().append("g")
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
      // Within the group, draw a circle for the node position and text
      // on the side.
      node.append("circle")
          .attr("r", "5px")
          .attr("fill", function(d) { return color(d.group); })
      node.append("text")
          .attr("dx", 8).attr("dy", 5).text(function(d) { return d.name; });
      // Setup force parameters and update position callback
      simulation.nodes(graph.nodes).on("tick", forceSimulationUpdated);
      simulation.force("link").links(graph.edges);

      function forceSimulationUpdated() {
        // Update edges.
        edge.attr("x1", function(d) {return d.source.x;})
            .attr("y1", function(d) {return d.source.y;})
            .attr("x2", function(d) {return d.target.x;})
            .attr("y2", function(d) {return d.target.y;});
        // Update node positions
        node.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
      }
    }
  buildGraph()
</script>
"""


class OpCodeMapper(object):
  """Maps an opcode index to an op name."""

  def __init__(self, data):
    self.code_to_name = {}
    for idx, d in enumerate(data["operator_codes"]):
      self.code_to_name[idx] = d["builtin_code"]

  def __call__(self, x):
    if x not in self.code_to_name:
      s = "<UNKNOWN>"
    else:
      s = self.code_to_name[x]
    return "%s (opcode=%d)" % (s, x)


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
      html += tensor["name"] + " "
      html += str(tensor["type"]) + " "
      html += repr(tensor["shape"]) + "<br>"
    html += "</span>"
    html += repr(x)
    html += "</span>"
    return html


def GenerateGraph(subgraph_idx, g, opcode_mapper):
  """Produces the HTML required to have a d3 visualization of the dag."""
  def TensorName(idx):
    return "t%d"%idx
  def OpName(idx):
    return "o%d"%idx
  edges = []
  nodes = []
  first = {}
  pixel_mult = 50  # TODO(aselle): multiplier for initial placement
  for op_index, op in enumerate(g["operators"]):
    for tensor_input_position, tensor_index in enumerate(op["inputs"]):
      if tensor_index not in first:
        first[tensor_index] = (
            op_index*pixel_mult,
            tensor_input_position*pixel_mult - pixel_mult/2)
      edges.append(
          {"source": TensorName(tensor_index), "target": OpName(op_index)})
    for tensor_index in op["outputs"]:
      edges.append(
          {"target": TensorName(tensor_index), "source": OpName(op_index)})
    nodes.append({"id": OpName(op_index),
                  "name": opcode_mapper(op["opcode_index"]),
                  "group": 2,
                  "x": pixel_mult,
                  "y": op_index * pixel_mult})
  for tensor_index, tensor in enumerate(g["tensors"]):
    initial_y = (first[tensor_index] if tensor_index in first
                 else len(g["operators"]))

    nodes.append({"id": TensorName(tensor_index),
                  "name": "%s (%d)" % (tensor["name"], tensor_index),
                  "group": 1,
                  "x": 2,
                  "y": initial_y})
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
      html += "<td>%s</td>\n"%val

    html += "</tr>\n"
  html += "</table>\n"
  return html


def CreateHtmlFile(tflite_input, html_output):
  """Given a tflite model in `tflite_input` file, produce html description."""

  # Convert the model into a JSON flatbuffer using flatc (build if doesn't
  # exist.
  if  not os.path.exists(tflite_input):
    raise RuntimeError("Invalid filename %r" % tflite_input)
  if tflite_input.endswith(".tflite") or tflite_input.endswith(".bin"):

    # Run convert
    cmd = (_BINARY + " -t "
           "--strict-json --defaults-json -o /tmp {schema} -- {input}".format(
               input=tflite_input, schema=_SCHEMA))
    print(cmd)
    os.system(cmd)
    real_output = ("/tmp/"+ os.path.splitext(os.path.split(tflite_input)[-1])[0]
                   + ".json")

    data = json.load(open(real_output))
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
    if not mapping: mapping = lambda x: x
    html += "<tr><th>%s</th><td>%s</td></tr>\n" % (key, mapping(data[key]))
  html += "</table>\n"

  # Spec on what keys to display
  buffer_keys_to_display = [("data", DataSizeMapper())]
  operator_keys_to_display = [("builtin_code", None)]

  for subgraph_idx, g in enumerate(data["subgraphs"]):
    # Subgraph local specs on what to display
    html += "<div class='subgraph'>"
    tensor_mapper = TensorMapper(g)
    opcode_mapper = OpCodeMapper(data)
    op_keys_to_display = [
        ("inputs", tensor_mapper), ("outputs", tensor_mapper),
        ("builtin_options", None), ("opcode_index", opcode_mapper)]
    tensor_keys_to_display = [
        ("name", None), ("type", None), ("shape", None), ("buffer", None),
        ("quantization", None)]

    html += "<h2>Subgraph %d</h2>\n" % subgraph_idx

    # Inputs and outputs.
    html += "<h3>Inputs/Outputs</h3>\n"
    html += GenerateTableHtml([{"inputs": g["inputs"],
                                "outputs": g["outputs"]}],
                              [("inputs", tensor_mapper),
                               ("outputs", tensor_mapper)],
                              display_index=False)

    # Print the tensors.
    html += "<h3>Tensors</h3>\n"
    html += GenerateTableHtml(g["tensors"], tensor_keys_to_display)

    # Print the ops.
    html += "<h3>Ops</h3>\n"
    html += GenerateTableHtml(g["operators"], op_keys_to_display)

    # Visual graph.
    html += "<svg id='subgraph%d' width='960' height='1600'></svg>\n" % (
        subgraph_idx,)
    html += GenerateGraph(subgraph_idx, g, opcode_mapper)
    html += "</div>"

  # Buffers have no data, but maybe in the future they will
  html += "<h2>Buffers</h2>\n"
  html += GenerateTableHtml(data["buffers"], buffer_keys_to_display)

  # Operator codes
  html += "<h2>Operator Codes</h2>\n"
  html += GenerateTableHtml(data["operator_codes"],
                            operator_keys_to_display)

  html += "</body></html>\n"

  open(html_output, "w").write(html)


def main(argv):
  try:
    tflite_input = argv[1]
    html_output = argv[2]
  except IndexError:
    print ("Usage: %s <input tflite> <output html>" % (argv[0]))
  else:
    CreateHtmlFile(tflite_input, html_output)

if __name__ == "__main__":
  main(sys.argv)

