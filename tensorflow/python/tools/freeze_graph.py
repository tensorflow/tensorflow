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
r"""Converts checkpoint variables into Const ops in a standalone GraphDef file.

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

import argparse
import sys

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.training import saver as saver_lib

FLAGS = None


def freeze_graph_with_def_protos(
    input_graph_def,
    input_saver_def,
    input_checkpoint,
    output_node_names,
    restore_op_name,
    filename_tensor_name,
    output_graph,
    clear_devices,
    initializer_nodes,
    variable_names_blacklist=""):
  """Converts all variables in a graph and checkpoint into constants."""
  del restore_op_name, filename_tensor_name  # Unused by updated loading code.

  # 'input_checkpoint' may be a prefix if we're using Saver V2 format
  if not saver_lib.checkpoint_exists(input_checkpoint):
    print("Input checkpoint '" + input_checkpoint + "' doesn't exist!")
    return -1

  if not output_node_names:
    print("You need to supply the name of a node to --output_node_names.")
    return -1

  # Remove all the explicit device specifications for this node. This helps to
  # make the graph more portable.
  if clear_devices:
    for node in input_graph_def.node:
      node.device = ""

  _ = importer.import_graph_def(input_graph_def, name="")

  with session.Session() as sess:
    if input_saver_def:
      saver = saver_lib.Saver(saver_def=input_saver_def)
      saver.restore(sess, input_checkpoint)
    else:
      var_list = {}
      reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in var_to_shape_map:
        try:
          tensor = sess.graph.get_tensor_by_name(key + ":0")
        except KeyError:
          # This tensor doesn't exist in the graph (for example it's
          # 'global_step' or a similar housekeeping element) so skip it.
          continue
        var_list[key] = tensor
      saver = saver_lib.Saver(var_list=var_list)
      saver.restore(sess, input_checkpoint)
      if initializer_nodes:
        sess.run(initializer_nodes)

    variable_names_blacklist = (variable_names_blacklist.split(",") if
                                variable_names_blacklist else None)
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        output_node_names.split(","),
        variable_names_blacklist=variable_names_blacklist)

  with gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
  print("%d ops in the final graph." % len(output_graph_def.node))


def _parse_input_graph_proto(input_graph, input_binary):
  """Parser input tensorflow graph into GraphDef proto."""
  if not gfile.Exists(input_graph):
    print("Input graph file '" + input_graph + "' does not exist!")
    return -1
  input_graph_def = graph_pb2.GraphDef()
  mode = "rb" if input_binary else "r"
  with gfile.FastGFile(input_graph, mode) as f:
    if input_binary:
      input_graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), input_graph_def)
  return input_graph_def


def _parse_input_saver_proto(input_saver, input_binary):
  """Parser input tensorflow Saver into SaverDef proto."""
  if not gfile.Exists(input_saver):
    print("Input saver file '" + input_saver + "' does not exist!")
    return -1
  mode = "rb" if input_binary else "r"
  with gfile.FastGFile(input_saver, mode) as f:
    saver_def = saver_pb2.SaverDef()
    if input_binary:
      saver_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), saver_def)
  return saver_def


def freeze_graph(input_graph,
                 input_saver,
                 input_binary,
                 input_checkpoint,
                 output_node_names,
                 restore_op_name,
                 filename_tensor_name,
                 output_graph,
                 clear_devices,
                 initializer_nodes,
                 variable_names_blacklist=""):
  """Converts all variables in a graph and checkpoint into constants."""
  input_graph_def = _parse_input_graph_proto(input_graph, input_binary)
  input_saver_def = None
  if input_saver:
    input_saver_def = _parse_input_saver_proto(input_saver, input_binary)
  freeze_graph_with_def_protos(
      input_graph_def,
      input_saver_def,
      input_checkpoint,
      output_node_names,
      restore_op_name,
      filename_tensor_name,
      output_graph,
      clear_devices,
      initializer_nodes,
      variable_names_blacklist)


def main(unused_args):
  freeze_graph(FLAGS.input_graph, FLAGS.input_saver, FLAGS.input_binary,
               FLAGS.input_checkpoint, FLAGS.output_node_names,
               FLAGS.restore_op_name, FLAGS.filename_tensor_name,
               FLAGS.output_graph, FLAGS.clear_devices, FLAGS.initializer_nodes,
               FLAGS.variable_names_blacklist)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--input_graph",
      type=str,
      default="",
      help="TensorFlow \'GraphDef\' file to load.")
  parser.add_argument(
      "--input_saver",
      type=str,
      default="",
      help="TensorFlow saver file to load.")
  parser.add_argument(
      "--input_checkpoint",
      type=str,
      default="",
      help="TensorFlow variables file to load.")
  parser.add_argument(
      "--output_graph",
      type=str,
      default="",
      help="Output \'GraphDef\' file name.")
  parser.add_argument(
      "--input_binary",
      nargs="?",
      const=True,
      type="bool",
      default=False,
      help="Whether the input files are in binary format.")
  parser.add_argument(
      "--output_node_names",
      type=str,
      default="",
      help="The name of the output nodes, comma separated.")
  parser.add_argument(
      "--restore_op_name",
      type=str,
      default="save/restore_all",
      help="The name of the master restore operator.")
  parser.add_argument(
      "--filename_tensor_name",
      type=str,
      default="save/Const:0",
      help="The name of the tensor holding the save path.")
  parser.add_argument(
      "--clear_devices",
      nargs="?",
      const=True,
      type="bool",
      default=True,
      help="Whether to remove device specifications.")
  parser.add_argument(
      "--initializer_nodes",
      type=str,
      default="",
      help="comma separated list of initializer nodes to run before freezing.")
  parser.add_argument(
      "--variable_names_blacklist",
      type=str,
      default="",
      help="""\
      comma separated list of variables to skip converting to constants\
      """)
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
