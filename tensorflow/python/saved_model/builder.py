## Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""SavedModel builder.

Builds a SavedModel that can be saved to storage, is language neutral, and
enables systems to produce, consume, or transform TensorFlow Models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from google.protobuf.any_pb2 import Any

from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import constants
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import compat


class SavedModelBuilder(object):
  """Builds the `SavedModel` protocol buffer and saves variables and assets.

  The `SavedModelBuilder` class provides functionality to build a `SavedModel`
  protocol buffer. Specifically, this allows multiple meta graphs to be saved as
  part of a single language-neutral `SavedModel`, while sharing variables and
  assets.

  To build a SavedModel, the first meta graph must be saved with variables.
  Subsequent meta graphs will simply be saved with their graph definitions. If
  assets need to be saved and written or copied to disk, they must be provided
  as part of the first meta graph to be saved. Subsequent meta graphs can
  provide a subset of the initial assets to be added to the SavedModel
  definition.

  Each meta graph added to the SavedModel must be annotated with tags. The tags
  provide a means to identify the specific meta graph to load and restore, along
  with the shared set of variables and assets.

  Typical usage for the `SavedModelBuilder`:
  ```python
  ...
  builder = saved_model_builder.SavedModelBuilder(export_dir)

  with tf.Session(graph=tf.Graph()) as sess:
    ...
    builder.add_meta_graph_and_variables(sess,
                                    ["foo-tag"],
                                    signature_def_map=foo_signatures,
                                    asset_collection=foo_assets)
  ...

  with tf.Session(graph=tf.Graph()) as sess:
    ...
    builder.add_meta_graph(["bar-tag", "baz-tag"])
  ...

  builder.save()
  ```
  """

  def __init__(self, export_dir):
    self._saved_model = saved_model_pb2.SavedModel()
    self._saved_model.saved_model_schema_version = (
        constants.SAVED_MODEL_SCHEMA_VERSION)

    self._export_dir = export_dir
    if not file_io.file_exists(export_dir):
      file_io.recursive_create_dir(self._export_dir)

    # Boolean to track whether variables and assets corresponding to the
    # SavedModel have been saved. Specifically, the first meta graph to be added
    # MUST use the add_meta_graph_and_variables() API. Subsequent add operations
    # on the SavedModel MUST use the add_meta_graph() API which does not save
    # weights.
    self._has_saved_variables = False

  def _asset_path_from_tensor(self, path_tensor):
    """Returns the filepath value stored in constant `path_tensor`.

    Args:
      path_tensor: Tensor of a file-path.

    Returns:
      The string value i.e. path of the tensor, if valid.

    Raises:
      TypeError if tensor does not match expected op type, dtype or value.
    """
    if not isinstance(path_tensor, ops.Tensor):
      raise TypeError("Asset path tensor must be a Tensor.")
    if path_tensor.op.type != "Const":
      raise TypeError("Asset path tensor must be of type constant.")
    if path_tensor.dtype != dtypes.string:
      raise TypeError("Asset path tensor must be of dtype string.")
    str_values = path_tensor.op.get_attr("value").string_val
    if len(str_values) != 1:
      raise TypeError("Asset path tensor must be a scalar.")
    return str_values[0]

  def _add_asset_to_collection(self, asset_filename, asset_tensor):
    """Builds an asset proto and adds it to the asset collection of the graph.

    Args:
      asset_filename: The filename of the asset to be added.
      asset_tensor: The asset tensor used to populate the tensor info of the
          asset proto.
    """
    asset_proto = meta_graph_pb2.AssetFileDef()
    asset_proto.filename = asset_filename
    asset_proto.tensor_info.name = asset_tensor.name

    asset_any_proto = Any()
    asset_any_proto.Pack(asset_proto)
    ops.add_to_collection(constants.ASSETS_KEY, asset_any_proto)

  def _save_and_write_assets(self, assets_collection_to_add=None):
    """Saves asset to the meta graph and writes asset files to disk.

    Args:
      assets_collection_to_add: The collection where the asset paths are setup.
    """
    asset_source_filepath_list = self._save_assets(assets_collection_to_add)

    # Return if there are no assets to write.
    if len(asset_source_filepath_list) is 0:
      tf_logging.info("No assets to write.")
      return

    assets_destination_dir = os.path.join(
        compat.as_bytes(self._export_dir),
        compat.as_bytes(constants.ASSETS_DIRECTORY))

    if not file_io.file_exists(assets_destination_dir):
      file_io.recursive_create_dir(assets_destination_dir)

    # Copy each asset from source path to destination path.
    for asset_source_filepath in asset_source_filepath_list:
      asset_source_filename = os.path.basename(asset_source_filepath)

      asset_destination_filepath = os.path.join(
          compat.as_bytes(assets_destination_dir),
          compat.as_bytes(asset_source_filename))
      file_io.copy(
          asset_source_filepath, asset_destination_filepath, overwrite=True)

    tf_logging.info("Assets written to: %s", assets_destination_dir)

  def _save_assets(self, assets_collection_to_add=None):
    """Saves assets to the meta graph.

    Args:
      assets_collection_to_add: The collection where the asset paths are setup.

    Returns:
      The list of filepaths to the assets in the assets collection.

    Raises:
      ValueError: Indicating an invalid filepath tensor.
    """
    asset_source_filepath_list = []

    if assets_collection_to_add is None:
      tf_logging.info("No assets to save.")
      return asset_source_filepath_list

    # Iterate over the supplied asset collection, build the `AssetFile` proto
    # and add them to the collection with key `constants.ASSETS_KEY`, in the
    # graph.
    for asset_tensor in assets_collection_to_add:
      asset_source_filepath = self._asset_path_from_tensor(asset_tensor)
      if not asset_source_filepath:
        raise ValueError("Invalid asset filepath tensor %s" % asset_tensor)

      asset_source_filename = os.path.basename(asset_source_filepath)

      # Build `AssetFile` proto and add it to the asset collection in the graph.
      self._add_asset_to_collection(asset_source_filename, asset_tensor)

      asset_source_filepath_list.append(asset_source_filepath)

    tf_logging.info("Assets added to graph.")
    return asset_source_filepath_list

  def _tag_and_add_meta_graph(self, meta_graph_def, tags, signature_def_map):
    """Tags the meta graph def and adds it to the SavedModel.

    Tags the meta graph def with the supplied tags, adds signature defs to it if
    provided and appends the meta graph def to the SavedModel proto.

    Args:
      meta_graph_def: The meta graph def to add to the SavedModel.
      tags: The set of tags to annotate the meta graph def with.
      signature_def_map: The map of signature defs to be added to the meta graph
          def.
    """
    for tag in tags:
      meta_graph_def.meta_info_def.tags.append(tag)

    if signature_def_map is not None:
      for key in signature_def_map:
        meta_graph_def.signature_def[key].CopyFrom(signature_def_map[key])

    proto_meta_graph_def = self._saved_model.meta_graphs.add()
    proto_meta_graph_def.CopyFrom(meta_graph_def)

  def add_meta_graph(self, tags, signature_def_map=None,
                     assets_collection=None):
    """Adds the current meta graph to the SavedModel.

    Creates a Saver in the current scope and uses the Saver to export the meta
    graph def. Invoking this API requires the `add_meta_graph_and_variables()`
    API to have been invoked before.

    Args:
      tags: The set of tags to annotate the meta graph def with.
      signature_def_map: The map of signature defs to be added to the meta graph
          def.
      assets_collection: Assets collection to be saved with SavedModel. Note
          that this collection should be a subset of the assets saved as part of
          the first meta graph in the SavedModel.

    Raises:
      AssertionError: If the variables for the SavedModel have not been saved
          yet.
    """
    if not self._has_saved_variables:
      raise AssertionError(
          "Variables and assets have not been saved yet. "
          "Please invoke `add_meta_graph_and_variables()` first.")

    # Save asset files, if any.
    self._save_assets(assets_collection)

    saver = tf_saver.Saver(variables.all_variables())
    meta_graph_def = saver.export_meta_graph()

    # Tag the meta graph def and add it to the SavedModel.
    self._tag_and_add_meta_graph(meta_graph_def, tags, signature_def_map)

  def add_meta_graph_and_variables(self,
                                   sess,
                                   tags,
                                   signature_def_map=None,
                                   assets_collection=None):
    """Adds the current meta graph to the SavedModel and saves variables.

    Creates a Saver to save the variables from the provided session. Exports the
    corresponding meta graph def. This function assumes that the variables to be
    saved have been initialized. For a given `SavedModelBuilder`, this API must
    be called exactly once and for the first meta graph to save. For subsequent
    meta graph defs to be added, the `add_meta_graph()` API must be used.

    Args:
      sess: The TensorFlow session from which to save the meta graph and
        variables.
      tags: The set of tags with which to save the meta graph.
      signature_def_map: The map of signature def map to add to the meta graph
        def.
      assets_collection: Assets collection to be saved with SavedModel.
    """
    if self._has_saved_variables:
      raise AssertionError("Variables and assets have already been saved. "
                           "Please invoke `add_meta_graph()` instead.")

    # Save asset files and write them to disk, if any.
    self._save_and_write_assets(assets_collection)

    export_path = os.path.join(
        compat.as_text(self._export_dir),
        compat.as_text(constants.VARIABLES_FILENAME))

    # Save the variables and export meta graph def.
    saver = tf_saver.Saver(variables.all_variables())
    saver.save(sess, export_path, write_meta_graph=False)
    meta_graph_def = saver.export_meta_graph()

    # Tag the meta graph def and add it to the SavedModel.
    self._tag_and_add_meta_graph(meta_graph_def, tags, signature_def_map)

    # Mark this instance of SavedModel as having saved variables, such that
    # subsequent attempts to save variables will fail.
    self._has_saved_variables = True

  def save(self, as_text=False):
    """Writes a `SavedModel` protocol buffer to disk.

    The function writes the SavedModel protocol buffer to the export directory
    in serialized format.

    Args:
      as_text: Writes the SavedModel protocol buffer in text format to disk.

    Returns:
      The path to which the SavedModel protocol buffer was written.
    """
    if not file_io.file_exists(self._export_dir):
      file_io.recursive_create_dir(self._export_dir)

    if as_text:
      path = os.path.join(
          compat.as_bytes(self._export_dir),
          compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT))
      file_io.write_string_to_file(path, str(self._saved_model))
    else:
      path = os.path.join(
          compat.as_bytes(self._export_dir),
          compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))
      file_io.write_string_to_file(path, self._saved_model.SerializeToString())
    tf_logging.info("SavedModel written to: %s", path)

    return path
