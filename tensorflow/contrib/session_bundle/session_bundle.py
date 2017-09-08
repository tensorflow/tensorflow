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

from tensorflow.contrib.session_bundle import constants
from tensorflow.contrib.session_bundle import manifest_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.util.deprecation import deprecated


@deprecated("2017-06-30",
            "No longer supported. Switch to SavedModel immediately.")
def maybe_session_bundle_dir(export_dir):
  """Checks if the model path contains session bundle model.

  Args:
    export_dir: string path to model checkpoint, for example 'model/00000123'

  Returns:
    true if path contains session bundle model files, ie META_GRAPH_DEF_FILENAME
  """

  meta_graph_filename = os.path.join(export_dir,
                                     constants.META_GRAPH_DEF_FILENAME)
  return file_io.file_exists(meta_graph_filename)


@deprecated("2017-06-30",
            "No longer supported. Switch to SavedModel immediately.")
def load_session_bundle_from_path(export_dir,
                                  target="",
                                  config=None,
                                  meta_graph_def=None):
  """Load session bundle from the given path.

  The function reads input from the export_dir, constructs the graph data to the
  default graph and restores the parameters for the session created.

  Args:
    export_dir: the directory that contains files exported by exporter.
    target: The execution engine to connect to. See target in tf.Session()
    config: A ConfigProto proto with configuration options. See config in
    tf.Session()
    meta_graph_def: optional object of type MetaGraphDef. If this object is
    present, then it is used instead of parsing MetaGraphDef from export_dir.

  Returns:
    session: a tensorflow session created from the variable files.
    meta_graph: a meta graph proto saved in the exporter directory.

  Raises:
    RuntimeError: if the required files are missing or contain unrecognizable
    fields, i.e. the exported model is invalid.
  """
  if not meta_graph_def:
    meta_graph_filename = os.path.join(export_dir,
                                       constants.META_GRAPH_DEF_FILENAME)
    if not file_io.file_exists(meta_graph_filename):
      raise RuntimeError("Expected meta graph file missing %s" %
                         meta_graph_filename)
    # Reads meta graph file.
    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    meta_graph_def.ParseFromString(
        file_io.read_file_to_string(meta_graph_filename, binary_mode=True))

  variables_filename = ""
  variables_filename_list = []
  checkpoint_sharded = False

  variables_index_filename = os.path.join(export_dir,
                                          constants.VARIABLES_INDEX_FILENAME_V2)
  checkpoint_v2 = file_io.file_exists(variables_index_filename)

  # Find matching checkpoint files.
  if checkpoint_v2:
    # The checkpoint is in v2 format.
    variables_filename_pattern = os.path.join(
        export_dir, constants.VARIABLES_FILENAME_PATTERN_V2)
    variables_filename_list = file_io.get_matching_files(
        variables_filename_pattern)
    checkpoint_sharded = True
  else:
    variables_filename = os.path.join(export_dir, constants.VARIABLES_FILENAME)
    if file_io.file_exists(variables_filename):
      variables_filename_list = [variables_filename]
    else:
      variables_filename = os.path.join(export_dir,
                                        constants.VARIABLES_FILENAME_PATTERN)
      variables_filename_list = file_io.get_matching_files(variables_filename)
      checkpoint_sharded = True

  # Prepare the files to restore a session.
  if not variables_filename_list:
    restore_files = ""
  elif checkpoint_v2 or not checkpoint_sharded:
    # For checkpoint v2 or v1 with non-sharded files, use "export" to restore
    # the session.
    restore_files = constants.VARIABLES_FILENAME
  else:
    restore_files = constants.VARIABLES_FILENAME_PATTERN

  assets_dir = os.path.join(export_dir, constants.ASSETS_DIRECTORY)

  collection_def = meta_graph_def.collection_def
  graph_def = graph_pb2.GraphDef()
  if constants.GRAPH_KEY in collection_def:
    # Use serving graph_def in MetaGraphDef collection_def if exists
    graph_def_any = collection_def[constants.GRAPH_KEY].any_list.value
    if len(graph_def_any) != 1:
      raise RuntimeError("Expected exactly one serving GraphDef in : %s" %
                         meta_graph_def)
    else:
      graph_def_any[0].Unpack(graph_def)
      # Replace the graph def in meta graph proto.
      meta_graph_def.graph_def.CopyFrom(graph_def)

  ops.reset_default_graph()
  sess = session.Session(target, graph=None, config=config)
  # Import the graph.
  saver = saver_lib.import_meta_graph(meta_graph_def)
  # Restore the session.
  if restore_files:
    saver.restore(sess, os.path.join(export_dir, restore_files))

  init_op_tensor = None
  if constants.INIT_OP_KEY in collection_def:
    init_ops = collection_def[constants.INIT_OP_KEY].node_list.value
    if len(init_ops) != 1:
      raise RuntimeError("Expected exactly one serving init op in : %s" %
                         meta_graph_def)
    init_op_tensor = ops.get_collection(constants.INIT_OP_KEY)[0]

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
