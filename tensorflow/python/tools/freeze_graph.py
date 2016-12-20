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
"""Converts checkpoint variables into Const ops in a standalone GraphDef file.

This script is designed to take a GraphDef proto, a SaverDef proto, and a set of
variable values stored in a checkpoint file, and output a GraphDef with all of
the variable ops converted into const ops containing the values of the
variables.

It's useful to do this when we need to load a single file in C++, especially in
environments like mobile or embedded where we may not have access to the
RestoreTensor ops and file loading calls that they rely on.

An example of command-line usage is:
bazel build tensorflow/python/tools:freeze_graph && \
bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=some_graph_def.pb \
--input_checkpoint=model.ckpt-8361242 \
--output_graph=/tmp/frozen_graph.pb --output_node_names=softmax

You can also look at freeze_graph_test.py for an example of how to use it.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.framework import graph_util


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("input_graph", "",
                           """TensorFlow 'GraphDef' file to load.""")
tf.app.flags.DEFINE_string("input_saver", "",
                           """TensorFlow saver file to load.""")
tf.app.flags.DEFINE_string("input_checkpoint", "",
                           """TensorFlow variables file to load.""")
tf.app.flags.DEFINE_string("output_graph", "",
                           """Output 'GraphDef' file name.""")
tf.app.flags.DEFINE_boolean("input_binary", False,
                            """Whether the input files are in binary format.""")
tf.app.flags.DEFINE_string("output_node_names", "",
                           """The name of the output nodes, comma separated.""")
tf.app.flags.DEFINE_string("restore_op_name", "save/restore_all",
                           """The name of the master restore operator.""")
tf.app.flags.DEFINE_string("filename_tensor_name", "save/Const:0",
                           """The name of the tensor holding the save path.""")
tf.app.flags.DEFINE_boolean("clear_devices", True,
                            """Whether to remove device specifications.""")
tf.app.flags.DEFINE_string("initializer_nodes", "", "comma separated list of "
                           "initializer nodes to run before freezing.")
tf.app.flags.DEFINE_string("variable_names_blacklist", "", "comma separated "
                           "list of variables to skip converting to constants ")


def freeze_graph(input_graph, input_saver, input_binary, input_checkpoint,
                 output_node_names, restore_op_name, filename_tensor_name,
                 output_graph, clear_devices, initializer_nodes):
  """Converts all variables in a graph and checkpoint into constants."""

  if not tf.gfile.Exists(input_graph):
    print("Input graph file '" + input_graph + "' does not exist!")
    return -1

  if input_saver and not tf.gfile.Exists(input_saver):
    print("Input saver file '" + input_saver + "' does not exist!")
    return -1

  # 'input_checkpoint' may be a prefix if we're using Saver V2 format
  if not tf.train.checkpoint_exists(input_checkpoint):
    print("Input checkpoint '" + input_checkpoint + "' doesn't exist!")
    return -1

  if not output_node_names:
    print("You need to supply the name of a node to --output_node_names.")
    return -1

  input_graph_def = tf.GraphDef()
  mode = "rb" if input_binary else "r"
  with tf.gfile.FastGFile(input_graph, mode) as f:
    if input_binary:
      input_graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read().decode("utf-8"), input_graph_def)
  # Remove all the explicit device specifications for this node. This helps to
  # make the graph more portable.
  if clear_devices:
    for node in input_graph_def.node:
      node.device = ""
  _ = tf.import_graph_def(input_graph_def, name="")

  with tf.Session() as sess:
    if input_saver:
      with tf.gfile.FastGFile(input_saver, mode) as f:
        saver_def = tf.train.SaverDef()
        if input_binary:
          saver_def.ParseFromString(f.read())
        else:
          text_format.Merge(f.read(), saver_def)
        saver = tf.train.Saver(saver_def=saver_def)
        saver.restore(sess, input_checkpoint)
    else:
      sess.run([restore_op_name], {filename_tensor_name: input_checkpoint})
      if initializer_nodes:
        sess.run(initializer_nodes)

    variable_names_blacklist = (FLAGS.variable_names_blacklist.split(",") if
                                FLAGS.variable_names_blacklist else None)
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(","),
        variable_names_blacklist=variable_names_blacklist)

  with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
  print("%d ops in the final graph." % len(output_graph_def.node))


def main(unused_args):
  freeze_graph(FLAGS.input_graph, FLAGS.input_saver, FLAGS.input_binary,
               FLAGS.input_checkpoint, FLAGS.output_node_names,
               FLAGS.restore_op_name, FLAGS.filename_tensor_name,
               FLAGS.output_graph, FLAGS.clear_devices, FLAGS.initializer_nodes)

if __name__ == "__main__":
  tf.app.run()
