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
# ==============================================================================
r"""obfuscate graph names
"""

from __future__ import absolute_import, division, print_function

from google.protobuf.text_format import Parse
from tensorflow.contrib.min_quantize.quantized_pb2 import QuantizedGraph
from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.python.platform import app, flags as flags_lib

from tensorflow.contrib.min_quantize.obfuscate_lib import obfuscate_graph_def, obfuscate_quantized_graph

flags = flags_lib
FLAGS = flags.FLAGS

flags.DEFINE_string("input", None, """input graph path""")
flags.DEFINE_boolean("text_proto", False, """graph in text proto format""")
flags.DEFINE_boolean("quantized", False, """input is quantized graph""")
flags.DEFINE_string("output_graph", None, """output graph path""")
flags.DEFINE_string("output_mapping", None, """output mapping path""")
flags.DEFINE_multi_string("keep", [], """node name keeps, could be node name or node_name:desired_name""")


def main(unused_args):
  # params
  in_path = FLAGS.input  # type: str
  in_is_text = FLAGS.text_proto  # type: bool
  quantized = FLAGS.quantized  # type: bool
  out_path = FLAGS.output_graph  # type: str
  out_mapping_path = FLAGS.output_mapping  # type: str
  keeps = [s if ':' not in s else tuple(s.split(':')) for s in FLAGS.keep]

  # validate param
  if in_path is None or len(in_path) == 0:
    raise RuntimeError("in_path must be provided")

  if out_path is None or len(out_path) == 0:
    raise RuntimeError("output must be provided")

  if out_mapping_path is None or len(out_mapping_path) == 0:
    raise RuntimeError("output_mapping must be provided")

  # read graph
  if quantized:
    in_graph = QuantizedGraph()
  else:
    in_graph = GraphDef()
  if in_is_text:
    with open(in_path, "r") as fp:
      Parse(fp.read(), in_graph)
  else:
    with open(in_path, "rb") as fp:
      in_graph.ParseFromString(fp.read())

  # obfuscate
  if quantized:
    obfuscated, mapping = obfuscate_quantized_graph(in_graph, keeps)
  else:
    obfuscated, mapping = obfuscate_graph_def(in_graph, keeps)

  # write graph
  with open(out_path, "wb") as fp:
    fp.write(obfuscated.SerializeToString())

  # write mapping
  with open(out_mapping_path, "w") as fp:
    for k, v in mapping.items():
      fp.write("{}:{}\n".format(k, v))


if __name__ == "__main__":
  app.run()
