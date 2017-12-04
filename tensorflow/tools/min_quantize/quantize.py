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
r"""quantize graph
"""

from __future__ import absolute_import, division, print_function

from google.protobuf.text_format import Parse
from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.python.platform import app, flags as flags_lib

from tensorflow.contrib.min_quantize.quantize_lib import quantize_graph_def

flags = flags_lib
FLAGS = flags.FLAGS

flags.DEFINE_string("input", None, """input graph path""")
flags.DEFINE_string("output", None, """output graph path""")
flags.DEFINE_boolean("text_proto", False, """graph in text proto format""")
flags.DEFINE_multi_string("skip", [], """nodes without quantize""")
flags.DEFINE_multi_string("output_node", [], """output node names""")


def main(unused_args):
  # params
  in_path = FLAGS.input  # type: str
  in_is_text = FLAGS.text_proto  # type: bool
  out_path = FLAGS.output  # type: str
  skip = FLAGS.skip  # type: list
  output_nodes = FLAGS.output_node # type: list

  # validate param
  if in_path is None or len(in_path) == 0:
    raise RuntimeError("in_path must be provided")

  if out_path is None or len(out_path) == 0:
    raise RuntimeError("output must be provided")

  # read graph
  in_graph = GraphDef()
  if in_is_text:
    with open(in_path, "r") as fp:
      Parse(fp.read(), in_graph)
  else:
    with open(in_path, "rb") as fp:
      in_graph.ParseFromString(fp.read())

  # quantize
  quantized = quantize_graph_def(in_graph, set(skip))

  # write
  with open(out_path, "wb") as fp:
    fp.write(quantized.SerializeToString())


if __name__ == "__main__":
  app.run()
