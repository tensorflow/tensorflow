# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Importer for an exported TensorFlow model.

This module provides a function to create a SessionBundle containing both the
Session and MetaGraph.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensorflow.contrib.session_bundle import constants
from tensorflow.contrib.session_bundle import manifest_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.lib.io import file_io


def load_session_bundle_from_path(export_dir, target="", config=None):
  """Load session bundle from the given path.

  The function reads input from the export_dir, constructs the graph data to the
  default graph and restores the parameters for the session created.

  Args:
    export_dir: the directory that contains files exported by exporter.
    target: The execution engine to connect to. See target in tf.Session()
    config: A ConfigProto proto with configuration options. See config in
    tf.Session()

  Returns:
    session: a tensorflow session created from the variable files.
    meta_graph: a meta graph proto saved in the exporter directory.

  Raises:
    RuntimeError: if the required files are missing or contain unrecognizable
    fields, i.e. the exported model is invalid.
  """
  meta_graph_filename = os.path.join(export_dir,
                                     constants.META_GRAPH_DEF_FILENAME)
  if not file_io.file_exists(meta_graph_filename):
    raise RuntimeError("Expected meta graph file missing %s" %
                       meta_graph_filename)
  variables_filename = os.path.join(export_dir,
                                    constants.VARIABLES_FILENAME)
  if not file_io.file_exists(variables_filename):
    variables_filename = os.path.join(
        export_dir, constants.VARIABLES_FILENAME_PATTERN)
    if not file_io.get_matching_files(variables_filename):
      # If graph_util.convert_variables_to_constants() is called on a model
      # it won't have any variables, and that's OK.
      #
      # TODO(yxshi): verify that the graph_def in fact does not have any
      # reachable variables.
      variables_filename = None
  assets_dir = os.path.join(export_dir, constants.ASSETS_DIRECTORY)

  # Reads meta graph file.
  meta_graph_def = meta_graph_pb2.MetaGraphDef()
  meta_graph_def.ParseFromString(file_io.read_file_to_string(
      meta_graph_filename))

  collection_def = meta_graph_def.collection_def
  graph_def = tf.GraphDef()
  if constants.GRAPH_KEY in collection_def:
    # Use serving graph_def in MetaGraphDef collection_def if exists
    graph_def_any = collection_def[constants.GRAPH_KEY].any_list.value
    if len(graph_def_any) != 1:
      raise RuntimeError(
          "Expected exactly one serving GraphDef in : %s" % meta_graph_def)
    else:
      graph_def_any[0].Unpack(graph_def)
      # Replace the graph def in meta graph proto.
      meta_graph_def.graph_def.CopyFrom(graph_def)

  tf.reset_default_graph()
  sess = tf.Session(target, graph=None, config=config)
  # Import the graph.
  saver = tf.train.import_meta_graph(meta_graph_def)
  # Restore the session.
  if variables_filename:
    saver.restore(sess, variables_filename)

  init_op_tensor = None
  if constants.INIT_OP_KEY in collection_def:
    init_ops = collection_def[constants.INIT_OP_KEY].node_list.value
    if len(init_ops) != 1:
      raise RuntimeError(
          "Expected exactly one serving init op in : %s" % meta_graph_def)
    init_op_tensor = tf.get_collection(constants.INIT_OP_KEY)[0]

  # Create asset input tensor list.
  asset_tensor_dict = {}
  if constants.ASSETS_KEY in collection_def:
    assets_any = collection_def[constants.ASSETS_KEY].any_list.value
    for asset in assets_any:
      asset_pb = manifest_pb2.AssetFile()
      asset.Unpack(asset_pb)
      asset_tensor_dict[asset_pb.tensor_binding.tensor_name] = os.path.join(
          assets_dir, asset_pb.filename)

  if init_op_tensor:
    # Run the init op.
    sess.run(fetches=[init_op_tensor], feed_dict=asset_tensor_dict)

  return sess, meta_graph_def
