# pylint: disable=g-bad-file-header
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
r"""Removes parts of a graph that are only needed for training.

There are several common transformations that can be applied to GraphDefs
created to train a model, that help reduce the amount of computation needed when
the network is used only for inference. These include:

 - Removing training-only operations like checkpoint saving.

 - Stripping out parts of the graph that are never reached.

 - Removing debug operations like CheckNumerics.

 - Folding batch normalization ops into the pre-calculated weights.

 - Fusing common operations into unified versions.

This script takes either a frozen binary GraphDef file (where the weight
variables have been converted into constants by the freeze_graph script), or a
text GraphDef proto file (the weight variables are stored in a separate
checkpoint file), and outputs a new GraphDef with the optimizations applied.

If the input graph is a text graph file, make sure to include the node that
restores the variable weights in output_names. That node is usually named
"restore_all".

An example of command-line usage is:

bazel build tensorflow/python/tools:optimize_for_inference && \
bazel-bin/tensorflow/python/tools/optimize_for_inference \
--input=frozen_inception_graph.pb \
--output=optimized_inception_graph.pb \
--frozen_graph=True \
--input_names=Mul \
--output_names=softmax


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.platform import app
from tensorflow.python.platform import flags as flags_lib
from tensorflow.python.platform import gfile
from tensorflow.python.tools import optimize_for_inference_lib

flags = flags_lib
FLAGS = flags.FLAGS
flags.DEFINE_string("input", "", """TensorFlow 'GraphDef' file to load.""")
flags.DEFINE_string("output", "", """File to save the output graph to.""")
flags.DEFINE_string("input_names", "", """Input node names, comma separated.""")
flags.DEFINE_string("output_names", "",
                    """Output node names, comma separated.""")
flags.DEFINE_boolean("frozen_graph", True,
                     """If true, the input graph is a binary frozen GraphDef
                     file; if false, it is a text GraphDef proto file.""")
flags.DEFINE_integer("placeholder_type_enum", dtypes.float32.as_datatype_enum,
                     """The AttrValue enum to use for placeholders.""")


def main(unused_args):
  if not gfile.Exists(FLAGS.input):
    print("Input graph file '" + FLAGS.input + "' does not exist!")
    return -1

  input_graph_def = graph_pb2.GraphDef()
  with gfile.Open(FLAGS.input, "r") as f:
    data = f.read()
    if FLAGS.frozen_graph:
      input_graph_def.ParseFromString(data)
    else:
      text_format.Merge(data.decode("utf-8"), input_graph_def)

  output_graph_def = optimize_for_inference_lib.optimize_for_inference(
      input_graph_def,
      FLAGS.input_names.split(","),
      FLAGS.output_names.split(","), FLAGS.placeholder_type_enum)

  if FLAGS.frozen_graph:
    f = gfile.FastGFile(FLAGS.output, "w")
    f.write(output_graph_def.SerializeToString())
  else:
    graph_io.write_graph(output_graph_def,
                         os.path.dirname(FLAGS.output),
                         os.path.basename(FLAGS.output))
  return 0


if __name__ == "__main__":
  app.run()
