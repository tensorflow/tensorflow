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
import argparse
import re
import sys
from typing import List, Optional, Union

from absl import app

from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import importer
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import saver as saver_lib


def _has_no_variables(sess: session.Session) -> bool:
  """Determines if the graph has any variables.

  Args:
    sess: TensorFlow Session.

  Returns:
    Bool.
  """
  for op in sess.graph.get_operations():
    if op.type.startswith("Variable") or op.type.endswith("VariableOp"):
      return False
  return True


def freeze_graph_with_def_protos(
    input_graph_def: Optional[graph_pb2.GraphDef],
    input_saver_def: Optional[saver_pb2.SaverDef],
    input_checkpoint: Optional[str],
    output_node_names: str,
    restore_op_name: Optional[str],
    filename_tensor_name: Optional[str],
    output_graph: str,
    clear_devices: bool,
    initializer_nodes: str,
    variable_names_whitelist: str = "",
    variable_names_denylist: str = "",
    input_meta_graph_def: Optional[meta_graph_pb2.MetaGraphDef] = None,
    input_saved_model_dir: Optional[str] = None,
    saved_model_tags: Optional[List[str]] = None,
    checkpoint_version: int = saver_pb2.SaverDef.V2,
) -> graph_pb2.GraphDef:
  """Converts all variables in a graph and checkpoint into constants.

  Args:
    input_graph_def: A `GraphDef`.
    input_saver_def: A `SaverDef` (optional).
    input_checkpoint: The prefix of a V1 or V2 checkpoint, with V2 taking
      priority.  Typically the result of `Saver.save()` or that of
      `tf.train.latest_checkpoint()`, regardless of sharded/non-sharded or
      V1/V2.
    output_node_names: The name(s) of the output nodes, comma separated.
    restore_op_name: Unused.
    filename_tensor_name: Unused.
    output_graph: String where to write the frozen `GraphDef`.
    clear_devices: A Bool whether to remove device specifications.
    initializer_nodes: Comma separated string of initializer nodes to run before
      freezing.
    variable_names_whitelist: The set of variable names to convert (optional, by
      default, all variables are converted).
    variable_names_denylist: The set of variable names to omit converting to
      constants (optional).
    input_meta_graph_def: A `MetaGraphDef` (optional),
    input_saved_model_dir: Path to the dir with TensorFlow 'SavedModel' file and
      variables (optional).
    saved_model_tags: Group of comma separated tag(s) of the MetaGraphDef to
      load, in string format (optional).
    checkpoint_version: Tensorflow variable file format (saver_pb2.SaverDef.V1
      or saver_pb2.SaverDef.V2)

  Returns:
    Location of the output_graph_def.
  """
  del restore_op_name, filename_tensor_name  # Unused by updated loading code.

  # 'input_checkpoint' may be a prefix if we're using Saver V2 format
  if not input_saved_model_dir and not checkpoint_management.checkpoint_exists(
      input_checkpoint
  ):
    raise ValueError(
        "Input checkpoint '" + input_checkpoint + "' doesn't exist!"
    )

  if not output_node_names:
    raise ValueError(
        "You need to supply the name of a node to --output_node_names."
    )

  # Remove all the explicit device specifications for this node. This helps to
  # make the graph more portable.
  if clear_devices:
    if input_meta_graph_def:
      for node in input_meta_graph_def.graph_def.node:
        node.device = ""
    elif input_graph_def:
      for node in input_graph_def.node:
        node.device = ""

  if input_graph_def:
    _ = importer.import_graph_def(input_graph_def, name="")
  with session.Session() as sess:
    if input_saver_def:
      saver = saver_lib.Saver(
          saver_def=input_saver_def, write_version=checkpoint_version
      )
      saver.restore(sess, input_checkpoint)
    elif input_meta_graph_def:
      restorer = saver_lib.import_meta_graph(
          input_meta_graph_def, clear_devices=True
      )
      restorer.restore(sess, input_checkpoint)
      if initializer_nodes:
        sess.run(initializer_nodes.replace(" ", "").split(","))
    elif input_saved_model_dir:
      if saved_model_tags is None:
        saved_model_tags = []
      loader.load(sess, saved_model_tags, input_saved_model_dir)
    else:
      var_list = {}
      reader = py_checkpoint_reader.NewCheckpointReader(input_checkpoint)
      var_to_shape_map = reader.get_variable_to_shape_map()

      # List of all partition variables. Because the condition is heuristic
      # based, the list could include false positives.
      all_partition_variable_names = [
          tensor.name.split(":")[0]
          for op in sess.graph.get_operations()
          for tensor in op.values()
          if re.search(r"/part_\d+/", tensor.name)
      ]
      has_partition_var = False

      for key in var_to_shape_map:
        try:
          tensor = sess.graph.get_tensor_by_name(key + ":0")
          if any(key in name for name in all_partition_variable_names):
            has_partition_var = True
        except KeyError:
          # This tensor doesn't exist in the graph (for example it's
          # 'global_step' or a similar housekeeping element) so skip it.
          continue
        var_list[key] = tensor

      try:
        saver = saver_lib.Saver(
            var_list=var_list, write_version=checkpoint_version
        )
      except TypeError as e:
        # `var_list` is required to be a map of variable names to Variable
        # tensors. Partition variables are Identity tensors that cannot be
        # handled by Saver.
        if has_partition_var:
          raise ValueError(
              "Models containing partition variables cannot be converted "
              "from checkpoint files. Please pass in a SavedModel using "
              "the flag --input_saved_model_dir."
          )
        # Models that have been frozen previously do not contain Variables.
        elif _has_no_variables(sess):
          raise ValueError(
              "No variables were found in this model. It is likely the model "
              "was frozen previously. You cannot freeze a graph twice."
          )
        else:
          raise e

      saver.restore(sess, input_checkpoint)
      if initializer_nodes:
        sess.run(initializer_nodes.replace(" ", "").split(","))

    variable_names_whitelist = (
        variable_names_whitelist.replace(" ", "").split(",")
        if variable_names_whitelist
        else None
    )
    variable_names_denylist = (
        variable_names_denylist.replace(" ", "").split(",")
        if variable_names_denylist
        else None
    )

    if input_meta_graph_def:
      output_graph_def = convert_to_constants.convert_variables_to_constants(
          sess,
          input_meta_graph_def.graph_def,
          output_node_names.replace(" ", "").split(","),
          variable_names_whitelist=variable_names_whitelist,
          variable_names_blacklist=variable_names_denylist,
      )
    else:
      output_graph_def = convert_to_constants.convert_variables_to_constants(
          sess,
          input_graph_def,
          output_node_names.replace(" ", "").split(","),
          variable_names_whitelist=variable_names_whitelist,
          variable_names_blacklist=variable_names_denylist,
      )

  # Write GraphDef to file if output path has been given.
  if output_graph:
    with gfile.GFile(output_graph, "wb") as f:
      f.write(output_graph_def.SerializeToString(deterministic=True))

  return output_graph_def


def _parse_input_graph_proto(
    input_graph: str, input_binary: bool
) -> graph_pb2.GraphDef:
  """Parses input tensorflow graph into GraphDef proto."""
  if not gfile.Exists(input_graph):
    raise IOError("Input graph file '" + input_graph + "' does not exist!")
  input_graph_def = graph_pb2.GraphDef()
  mode = "rb" if input_binary else "r"
  with gfile.GFile(input_graph, mode) as f:
    if input_binary:
      input_graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), input_graph_def)
  return input_graph_def


def _parse_input_meta_graph_proto(
    input_graph: str, input_binary: bool
) -> meta_graph_pb2.MetaGraphDef:
  """Parses input tensorflow graph into MetaGraphDef proto."""
  if not gfile.Exists(input_graph):
    raise IOError("Input meta graph file '" + input_graph + "' does not exist!")
  input_meta_graph_def = meta_graph_pb2.MetaGraphDef()
  mode = "rb" if input_binary else "r"
  with gfile.GFile(input_graph, mode) as f:
    if input_binary:
      input_meta_graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), input_meta_graph_def)
  print("Loaded meta graph file '" + input_graph)
  return input_meta_graph_def


def _parse_input_saver_proto(input_saver, input_binary):
  """Parses input tensorflow Saver into SaverDef proto."""
  if not gfile.Exists(input_saver):
    raise IOError("Input saver file '" + input_saver + "' does not exist!")
  mode = "rb" if input_binary else "r"
  with gfile.GFile(input_saver, mode) as f:
    saver_def = saver_pb2.SaverDef()
    if input_binary:
      saver_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), saver_def)
  return saver_def


def freeze_graph(
    input_graph: Optional[str],
    input_saver: str,
    input_binary: bool,
    input_checkpoint: Optional[str],
    output_node_names: str,
    restore_op_name: Optional[str],
    filename_tensor_name: Optional[str],
    output_graph: str,
    clear_devices: bool,
    initializer_nodes: str,
    variable_names_whitelist: str = "",
    variable_names_denylist: str = "",
    input_meta_graph: Union[None, bool, str] = None,
    input_saved_model_dir: Optional[str] = None,
    saved_model_tags: str = tag_constants.SERVING,
    checkpoint_version: int = saver_pb2.SaverDef.V2,
) -> graph_pb2.GraphDef:
  """Converts all variables in a graph and checkpoint into constants.

  Args:
    input_graph: A `GraphDef` file to load.
    input_saver: A TensorFlow Saver file.
    input_binary: A Bool. True means input_graph is .pb, False indicates .pbtxt.
    input_checkpoint: The prefix of a V1 or V2 checkpoint, with V2 taking
      priority.  Typically the result of `Saver.save()` or that of
      `tf.train.latest_checkpoint()`, regardless of sharded/non-sharded or
      V1/V2.
    output_node_names: The name(s) of the output nodes, comma separated.
    restore_op_name: Unused.
    filename_tensor_name: Unused.
    output_graph: String where to write the frozen `GraphDef`.
    clear_devices: A Bool whether to remove device specifications.
    initializer_nodes: Comma separated list of initializer nodes to run before
      freezing.
    variable_names_whitelist: The set of variable names to convert (optional, by
      default, all variables are converted),
    variable_names_denylist: The set of variable names to omit converting to
      constants (optional).
    input_meta_graph: A `MetaGraphDef` file to load (optional).
    input_saved_model_dir: Path to the dir with TensorFlow 'SavedModel' file and
      variables (optional).
    saved_model_tags: Group of comma separated tag(s) of the MetaGraphDef to
      load, in string format.
    checkpoint_version: Tensorflow variable file format (saver_pb2.SaverDef.V1
      or saver_pb2.SaverDef.V2).

  Returns:
    String that is the location of frozen GraphDef.
  """
  input_graph_def = None
  if input_saved_model_dir:
    input_graph_def = saved_model_utils.get_meta_graph_def(
        input_saved_model_dir, saved_model_tags
    ).graph_def
  elif input_graph:
    input_graph_def = _parse_input_graph_proto(input_graph, input_binary)
  input_meta_graph_def = None
  if input_meta_graph:
    input_meta_graph_def = _parse_input_meta_graph_proto(
        input_meta_graph, input_binary
    )
  input_saver_def = None
  if input_saver:
    input_saver_def = _parse_input_saver_proto(input_saver, input_binary)
  return freeze_graph_with_def_protos(
      input_graph_def,
      input_saver_def,
      input_checkpoint,
      output_node_names,
      restore_op_name,
      filename_tensor_name,
      output_graph,
      clear_devices,
      initializer_nodes,
      variable_names_whitelist,
      variable_names_denylist,
      input_meta_graph_def,
      input_saved_model_dir,
      [tag for tag in saved_model_tags.replace(" ", "").split(",") if tag],
      checkpoint_version=checkpoint_version,
  )


def main(unused_args, flags):
  if flags.checkpoint_version == 1:
    checkpoint_version = saver_pb2.SaverDef.V1
  elif flags.checkpoint_version == 2:
    checkpoint_version = saver_pb2.SaverDef.V2
  else:
    raise ValueError(
        "Invalid checkpoint version (must be '1' or '2'): %d"
        % flags.checkpoint_version
    )
  freeze_graph(
      flags.input_graph,
      flags.input_saver,
      flags.input_binary,
      flags.input_checkpoint,
      flags.output_node_names,
      flags.restore_op_name,
      flags.filename_tensor_name,
      flags.output_graph,
      flags.clear_devices,
      flags.initializer_nodes,
      flags.variable_names_whitelist,
      flags.variable_names_denylist,
      flags.input_meta_graph,
      flags.input_saved_model_dir,
      flags.saved_model_tags,
      checkpoint_version,
  )


def run_main():
  """Main function of freeze_graph."""
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--input_graph",
      type=str,
      default="",
      help="TensorFlow 'GraphDef' file to load.",
  )
  parser.add_argument(
      "--input_saver",
      type=str,
      default="",
      help="TensorFlow saver file to load.",
  )
  parser.add_argument(
      "--input_checkpoint",
      type=str,
      default="",
      help="TensorFlow variables file to load.",
  )
  parser.add_argument(
      "--checkpoint_version",
      type=int,
      default=2,
      help="Tensorflow variable file format",
  )
  parser.add_argument(
      "--output_graph",
      type=str,
      default="",
      help="Output 'GraphDef' file name.",
  )
  parser.add_argument(
      "--input_binary",
      nargs="?",
      const=True,
      type="bool",
      default=False,
      help="Whether the input files are in binary format.",
  )
  parser.add_argument(
      "--output_node_names",
      type=str,
      default="",
      help="The name of the output nodes, comma separated.",
  )
  parser.add_argument(
      "--restore_op_name",
      type=str,
      default="save/restore_all",
      help="""\
      The name of the master restore operator. Deprecated, unused by updated \
      loading code.
      """,
  )
  parser.add_argument(
      "--filename_tensor_name",
      type=str,
      default="save/Const:0",
      help="""\
      The name of the tensor holding the save path. Deprecated, unused by \
      updated loading code.
      """,
  )
  parser.add_argument(
      "--clear_devices",
      nargs="?",
      const=True,
      type="bool",
      default=True,
      help="Whether to remove device specifications.",
  )
  parser.add_argument(
      "--initializer_nodes",
      type=str,
      default="",
      help="Comma separated list of initializer nodes to run before freezing.",
  )
  parser.add_argument(
      "--variable_names_whitelist",
      type=str,
      default="",
      help="""\
      Comma separated list of variables to convert to constants. If specified, \
      only those variables will be converted to constants.\
      """,
  )
  parser.add_argument(
      "--variable_names_denylist",
      type=str,
      default="",
      help="""\
      Comma separated list of variables to skip converting to constants.\
      """,
  )
  parser.add_argument(
      "--input_meta_graph",
      type=str,
      default="",
      help="TensorFlow 'MetaGraphDef' file to load.",
  )
  parser.add_argument(
      "--input_saved_model_dir",
      type=str,
      default="",
      help="Path to the dir with TensorFlow 'SavedModel' file and variables.",
  )
  parser.add_argument(
      "--saved_model_tags",
      type=str,
      default="serve",
      help="""\
      Group of tag(s) of the MetaGraphDef to load, in string format,\
      separated by \',\'. For tag-set contains multiple tags, all tags \
      must be passed in.\
      """,
  )
  flags, unparsed = parser.parse_known_args()

  my_main = lambda unused_args: main(unused_args, flags)
  app.run(main=my_main, argv=[sys.argv[0]] + unparsed)


if __name__ == "__main__":
  run_main()
