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

This script takes a frozen GraphDef file (where the weight variables have been
converted into constants by the freeze_graph script) and outputs a new GraphDef
with the optimizations applied.

An example of command-line usage is:

bazel build tensorflow/python/tools:optimize_for_inference && \
bazel-bin/tensorflow/python/tools/optimize_for_inference \
--input=frozen_inception_graph.pb \
--output=optimized_inception_graph.pb \
--input_names=Mul \
--output_names=softmax

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.tools import optimize_for_inference_lib

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("input", "", """TensorFlow 'GraphDef' file to load.""")
flags.DEFINE_string("output", "", """File to save the output graph to.""")
flags.DEFINE_string("input_names", "",
                    """Input node names, comma separated.""")
flags.DEFINE_string("output_names", "",
                    """Output node names, comma separated.""")
flags.DEFINE_integer("placeholder_type_enum",
                     tf.float32.as_datatype_enum,
                     """The AttrValue enum to use for placeholders.""")


def main(unused_args):
  if not tf.gfile.Exists(FLAGS.input):
    print("Input graph file '" + FLAGS.input + "' does not exist!")
    return -1

  input_graph_def = tf.GraphDef()
  with tf.gfile.Open(FLAGS.input, "r") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

  output_graph_def = optimize_for_inference_lib.optimize_for_inference(
      input_graph_def, FLAGS.input_names.split(","),
      FLAGS.output_names.split(","), FLAGS.placeholder_type_enum)

  f = tf.gfile.FastGFile(FLAGS.output, "w")
  f.write(output_graph_def.SerializeToString())

  return 0


if __name__ == "__main__":
  tf.app.run()
