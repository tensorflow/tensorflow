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
"""SavedModel builder implementation."""

import functools
import os

from google.protobuf.any_pb2 import Any

from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export

# API label for SavedModel metrics.
_SAVE_BUILDER_LABEL = "save_v1_builder"


# Base class for the SavedModelBuilder that is only used by Tensorflow
# internally. Please use tf.compat.v1.saved_model.SavedModelBuilder instead.
@tf_export("__internal__.saved_model.SavedModelBuilder", v1=[])
class _SavedModelBuilder(object):
  """Builds the `SavedModel` protocol buffer and saves variables and assets.

  The `SavedModelBuilder` class provides the functionality to build a
  `SavedModel` protocol buffer. Specifically, this allows multiple meta
  graphs to be saved as part of a single language-neutral `SavedModel`,
  while sharing variables and assets.

  To build a SavedModel, the first meta graph must be saved with variables.
  Subsequent meta graphs will simply be saved with their graph definitions. If
  assets need to be saved and written or copied to disk, they can be provided
  when the meta graph def is added. If multiple meta graph defs are associated
  an asset of the same name, only the first version is retained.

  Each meta graph added to the SavedModel must be annotated with tags. The tags
  provide a means to identify the specific meta graph to load and restore, along
  with the shared set of variables and assets.

  Typical usage for the `SavedModelBuilder`:

  ```python
  ...
  builder = tf.compat.v1.saved_model.Builder(export_dir)

  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    ...
    builder.add_meta_graph_and_variables(sess,
                                    ["foo-tag"],
                                    signature_def_map=foo_signatures,
                                    assets_list=foo_assets)
  ...

  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    ...
    builder.add_meta_graph(["bar-tag", "baz-tag"])
  ...

  builder.save()
  ```

  Note: This function will only be available through the v1 compatibility
  library as tf.compat.v1.saved_model.builder.SavedModelBuilder or
  tf.compat.v1.saved_model.Builder. Tensorflow 2.0 will introduce a new
  object-based method of creating SavedModels.
  """

  def __init__(self, export_dir):
    self._saved_model = saved_model_pb2.SavedModel()
    self._saved_model.saved_model_schema_version = (
        constants.SAVED_MODEL_SCHEMA_VERSION)

    self._export_dir = export_dir
    if file_io.file_exists(export_dir):
      if file_io.list_directory(export_dir):
        raise AssertionError(
            f"Export directory {export_dir} already exists, and isn't empty. "
            "Please choose a different export directory, or delete all the "
            "contents of the specified directory.")
    else:
      file_io.recursive_create_dir(self._export_dir)

    # Boolean to track whether variables and assets corresponding to the
    # SavedModel have been saved. Specifically, the first meta graph to be added
    # MUST use the add_meta_graph_and_variables() API. Subsequent add operations
    # on the SavedModel MUST use the add_meta_graph() API which does not save
    # weights.
    self._has_saved_variables = False

  def _save_and_write_assets(self, meta_graph_def, assets_list=None):
    """Saves asset to the meta graph and writes asset files to disk.

    Args:
      meta_graph_def: The meta graph def to which the assets will be added.
      assets_list: The list where the asset paths are setup.
    """
    # Creates a function that adds assets into the meta graph def.
    write_fn = functools.partial(_add_asset_to_metagraph, meta_graph_def)
    asset_filename_map = _maybe_save_assets(write_fn, assets_list)

    # Return if there are no assets to write.
    if not asset_filename_map:
      tf_logging.info("No assets to write.")
      return

    # Copy assets from source path to destination path.
    copy_assets_to_destination_dir(asset_filename_map, self._export_dir)

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

  def _validate_tensor_info(self, tensor_info):
    """Validates the `TensorInfo` proto.

    Checks if the `encoding` (`name` or `coo_sparse` or `type_spec`) and
    `dtype` fields exist and are non-empty.

    Args:
      tensor_info: `TensorInfo` protocol buffer to validate.

    Raises:
      AssertionError: If the `encoding` or `dtype` fields of the supplied
          `TensorInfo` proto are not populated.
    """
    if tensor_info is None:
      raise AssertionError(
          "All TensorInfo protos used in the SignatureDefs must have the name "
          "and dtype fields set.")
    if tensor_info.WhichOneof("encoding") is None:
      # TODO(soergel) validate each of the fields of coo_sparse
      raise AssertionError(
          f"Invalid `tensor_info`: {tensor_info}. All TensorInfo protos used "
          "in the SignatureDefs must have one of the 'encoding' fields (e.g., "
          "name or coo_sparse) set.")
    if tensor_info.WhichOneof("encoding") == "composite_tensor":
      for component in tensor_info.composite_tensor.components:
        self._validate_tensor_info(component)
    elif tensor_info.dtype == types_pb2.DT_INVALID:
      raise AssertionError(
          f"Invalid `tensor_info`: {tensor_info}. All TensorInfo protos used in"
          " the SignatureDefs must have the dtype field set.")

  def _validate_signature_def_map(self, signature_def_map):
    """Validates the `SignatureDef` entries in the signature def map.

    Validation of entries in the signature def map includes ensuring that the
    `name` and `dtype` fields of the TensorInfo protos of the `inputs` and
    `outputs` of each `SignatureDef` are populated. Also ensures that reserved
    SignatureDef keys for the initialization and train ops are not used.

    Args:
      signature_def_map: The map of signature defs to be validated.

    Raises:
      AssertionError: If a TensorInfo is not valid.
      KeyError: If a reserved signature key is used in the map.
    """
    for signature_def_key in signature_def_map:
      signature_def = signature_def_map[signature_def_key]
      inputs = signature_def.inputs
      outputs = signature_def.outputs
      for inputs_key in inputs:
        self._validate_tensor_info(inputs[inputs_key])
      for outputs_key in outputs:
        self._validate_tensor_info(outputs[outputs_key])
    if constants.INIT_OP_SIGNATURE_KEY in signature_def_map:
      raise KeyError(
          f"SignatureDef map key \"{constants.INIT_OP_SIGNATURE_KEY}\" is "
          "reserved for initialization. Please use a different key.")
    if constants.TRAIN_OP_SIGNATURE_KEY in signature_def_map:
      raise KeyError(
          f"SignatureDef map key \"{constants.TRAIN_OP_SIGNATURE_KEY}\" is "
          f"reserved for the train op. Please use a different key.")

  def _maybe_create_saver(self, saver=None):
    """Creates a sharded saver if one does not already exist."""
    if not saver:
      # Initialize a saver to generate a sharded output for all saveables in the
      # current scope.
      saver = tf_saver.Saver(
          variables._all_saveable_objects(),  # pylint: disable=protected-access
          sharded=True,
          write_version=saver_pb2.SaverDef.V2,
          allow_empty=True)
    return saver

  def add_meta_graph(self,
                     tags,
                     signature_def_map=None,
                     assets_list=None,
                     clear_devices=False,
                     init_op=None,
                     train_op=None,
                     saver=None):
    """Adds the current meta graph to the SavedModel.

    Creates a Saver in the current scope and uses the Saver to export the meta
    graph def. Invoking this API requires the `add_meta_graph_and_variables()`
    API to have been invoked before.

    Args:
      tags: The set of tags to annotate the meta graph def with.
      signature_def_map: The map of signature defs to be added to the meta graph
          def.
      assets_list: Assets to be saved with SavedModel. Note
          that this list should be a subset of the assets saved as part of
          the first meta graph in the SavedModel.
      clear_devices: Set to true if the device info on the default graph should
          be cleared.
      init_op: Op or group of ops to execute when the graph is loaded. Note
          that when the init_op is specified it is run after the restore op at
          load-time.
      train_op: Op or group of opts that trains the model when run. This will
        not be run automatically when the graph is loaded, instead saved in
        a SignatureDef accessible through the exported MetaGraph.
      saver: An instance of tf.compat.v1.train.Saver that will be used to export
        the metagraph. If None, a sharded Saver that restores all variables will
        be used.

    Raises:
      AssertionError: If the variables for the SavedModel have not been saved
          yet, or if the graph already contains one or more legacy init ops.
    """
    if not self._has_saved_variables:
      raise AssertionError(
          "Graph state including variables and assets has not been saved yet. "
          "Please invoke `add_meta_graph_and_variables()` first.")

    # Validate the signature def map to ensure all included TensorInfos are
    # properly populated.
    signature_def_map = signature_def_map or {}
    self._validate_signature_def_map(signature_def_map)

    # Create a SignatureDef pointing to the graph initialization op, which will
    # be added to the MetaGraphDef.
    _add_op_to_signature_def_map(signature_def_map, init_op,
                                 constants.INIT_OP_SIGNATURE_KEY)
    _add_op_to_signature_def_map(signature_def_map, train_op,
                                 constants.TRAIN_OP_SIGNATURE_KEY)

    saver = self._maybe_create_saver(saver)

    # The graph almost certainly previously contained at least one Saver, and
    # possibly several (e.g. one for loading a pretrained embedding, and another
    # for the model weights).  Removing the preexisting ones was the
    # motivation for the clear_extraneous_savers option, but it turns out that
    # there are edge cases where that option breaks the graph.  Until that is
    # resolved, we just leave the option set to False for now.
    # TODO(soergel): Reinstate clear_extraneous_savers=True when possible.
    meta_graph_def = saver.export_meta_graph(
        clear_devices=clear_devices, strip_default_attrs=True)

    # Save asset files and write them to disk, if any.
    self._save_and_write_assets(meta_graph_def, assets_list)

    # Tag the meta graph def and add it to the SavedModel.
    self._tag_and_add_meta_graph(meta_graph_def, tags, signature_def_map)

  def add_meta_graph_and_variables(self,
                                   sess,
                                   tags,
                                   signature_def_map=None,
                                   assets_list=None,
                                   clear_devices=False,
                                   init_op=None,
                                   train_op=None,
                                   strip_default_attrs=False,
                                   saver=None):
    # pylint: disable=line-too-long
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
      assets_list: Assets to be saved with SavedModel.
      clear_devices: Set to true if the device info on the default graph should
          be cleared.
      init_op: Op or group of ops to execute when the graph is loaded. Note
          that when the init_op is specified it is run after the restore op at
          load-time.
      train_op: Op or group of ops that trains the model when run. This will
        not be run automatically when the graph is loaded, instead saved in
        a SignatureDef accessible through the exported MetaGraph.
      strip_default_attrs: Boolean. If `True`, default-valued attributes will be
        removed from the NodeDefs. For a detailed guide, see
        [Stripping Default-Valued Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).
      saver: An instance of tf.compat.v1.train.Saver that will be used to export the
        metagraph and save variables. If None, a sharded Saver that restores
        all variables will be used.

    """
    # pylint: enable=line-too-long
    if self._has_saved_variables:
      raise AssertionError("Graph state including variables and assets has "
                           "already been saved. Please invoke "
                           "`add_meta_graph()` instead.")

    # Validate the signature def map to ensure all included TensorInfos are
    # properly populated.
    signature_def_map = signature_def_map or {}
    self._validate_signature_def_map(signature_def_map)

    # Create a SignatureDef pointing to the graph initialization op, which will
    # be added to the MetaGraphDef.
    _add_op_to_signature_def_map(signature_def_map, init_op,
                                 constants.INIT_OP_SIGNATURE_KEY)
    _add_op_to_signature_def_map(signature_def_map, train_op,
                                 constants.TRAIN_OP_SIGNATURE_KEY)

    saved_model_utils.get_or_create_variables_dir(self._export_dir)
    variables_path = saved_model_utils.get_variables_path(self._export_dir)

    saver = self._maybe_create_saver(saver)

    # Save the variables. Also, disable writing the checkpoint state proto. The
    # file is not used during SavedModel loading. In addition, since a
    # SavedModel can be copied or moved, this avoids the checkpoint state to
    # become outdated.
    saver.save(sess, variables_path, write_meta_graph=False, write_state=False)

    # Export the meta graph def.

    # The graph almost certainly previously contained at least one Saver, and
    # possibly several (e.g. one for loading a pretrained embedding, and another
    # for the model weights).  Removing the preexisting ones was the
    # motivation for the clear_extraneous_savers option, but it turns out that
    # there are edge cases where that option breaks the graph.  Until that is
    # resolved, we just leave the option set to False for now.
    # TODO(soergel): Reinstate clear_extraneous_savers=True when possible.
    meta_graph_def = saver.export_meta_graph(
        clear_devices=clear_devices, strip_default_attrs=strip_default_attrs)

    # Save asset files and write them to disk, if any.
    self._save_and_write_assets(meta_graph_def, assets_list)

    # Tag the meta graph def and add it to the SavedModel.
    self._tag_and_add_meta_graph(meta_graph_def, tags, signature_def_map)

    # Mark this instance of SavedModel as having saved variables, such that
    # subsequent attempts to save variables will fail.
    self._has_saved_variables = True

  def save(self, as_text=False):
    """Writes a `SavedModel` protocol buffer to disk.

    The function writes the SavedModel protocol buffer to the export directory
    in a serialized format.

    Args:
      as_text: Writes the SavedModel protocol buffer in text format to
        disk. Protocol buffers in text format are useful for debugging, but
        parsing fails when it encounters an unknown field and so is not forward
        compatible. This means changes to TensorFlow may prevent deployment of
        new text format SavedModels to existing serving binaries. Do not deploy
        `as_text` SavedModels to production.

    Returns:
      The path to which the SavedModel protocol buffer was written.
    """
    metrics.IncrementWriteApi(_SAVE_BUILDER_LABEL)
    if not file_io.file_exists(self._export_dir):
      file_io.recursive_create_dir(self._export_dir)

    if as_text:
      path = file_io.join(
          compat.as_bytes(self._export_dir),
          compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT))
      file_io.write_string_to_file(path, str(self._saved_model))
    else:
      path = file_io.join(
          compat.as_bytes(self._export_dir),
          compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))
      file_io.write_string_to_file(
          path, self._saved_model.SerializeToString(deterministic=True))
    tf_logging.info("SavedModel written to: %s", compat.as_text(path))
    metrics.IncrementWrite(write_version="1")
    return path


@tf_export(v1=["saved_model.Builder", "saved_model.builder.SavedModelBuilder"])  # pylint: disable=missing-docstring
class SavedModelBuilder(_SavedModelBuilder):
  __doc__ = _SavedModelBuilder.__doc__.replace("assets_list",
                                               "assets_collection")

  def __init__(self, export_dir):
    super(SavedModelBuilder, self).__init__(export_dir=export_dir)

  def _add_collections(self, assets_collection, main_op, train_op):
    """Add asset and op collections to be saved."""
    # Save asset files and write them to disk, if any.
    self._save_and_write_assets(assets_collection)

    self._maybe_add_main_op(main_op)

    self._add_train_op(train_op)

  def _save_and_write_assets(self, assets_collection_to_add=None):
    """Saves asset to the meta graph and writes asset files to disk.

    Args:
      assets_collection_to_add: The collection where the asset paths are setup.
    """
    # Add assets to the collection with key `saved_model.ASSETS_KEY`, in the
    # graph.
    asset_filename_map = _maybe_save_assets(_add_asset_to_collection,
                                            assets_collection_to_add)

    # Return if there are no assets to write.
    if not asset_filename_map:
      tf_logging.info("No assets to write.")
      return

    # Copy assets from source path to destination path.
    copy_assets_to_destination_dir(asset_filename_map, self._export_dir)

  def _maybe_add_main_op(self, main_op):
    """Adds main op to the SavedModel.

    Args:
      main_op: Main op to run as part of graph initialization. If None, no main
        op will be added to the graph.

    Raises:
      TypeError: If the main op is provided but is not of type `Operation`.
      ValueError: if the Graph already contains an init op.
    """
    if main_op is None:
      return

    if not isinstance(main_op, ops.Operation):
      raise TypeError(f"Expected {main_op} to be an Operation but got type "
                      f"{type(main_op)} instead.")

    # Validate that no other init ops have been added to this graph already.
    # We check main_op and legacy_init_op for thoroughness and explicitness.
    for init_op_key in (constants.MAIN_OP_KEY, constants.LEGACY_INIT_OP_KEY):
      if ops.get_collection(init_op_key):
        raise ValueError(
            "Graph already contains one or more main ops under the "
            f"collection {init_op_key}.")

    ops.add_to_collection(constants.MAIN_OP_KEY, main_op)

  def _add_train_op(self, train_op):
    """Add train op to the SavedModel.

    Note that this functionality is in development, and liable to be
    moved elsewhere.

    Args:
      train_op: Op or group of ops that are used for training. These are stored
        as a collection with key TRAIN_OP_KEY, but not executed.

    Raises:
      TypeError if Train op is not of type `Operation`.
    """
    if train_op is not None:
      if (not isinstance(train_op, ops.Tensor) and
          not isinstance(train_op, ops.Operation)):
        raise TypeError(f"`train_op` {train_op} needs to be a Tensor or Op.")
      ops.add_to_collection(constants.TRAIN_OP_KEY, train_op)

  @deprecated_args(None,
                   "Pass your op to the equivalent parameter main_op instead.",
                   "legacy_init_op")
  def add_meta_graph(self,
                     tags,
                     signature_def_map=None,
                     assets_collection=None,
                     legacy_init_op=None,
                     clear_devices=False,
                     main_op=None,
                     strip_default_attrs=False,
                     saver=None):
    if not self._has_saved_variables:
      raise AssertionError(
          "Graph state including variables and assets has not been saved yet. "
          "Please invoke `add_meta_graph_and_variables()` first.")

    # Validate the signature def map to ensure all included TensorInfos are
    # properly populated.
    signature_def_map = signature_def_map or {}
    self._validate_signature_def_map(signature_def_map)

    # legacy_init_op is deprecated, and going away in TF 2.0.
    # Re-mapping to main_op, as treatment is identical regardless.
    main_op = main_op if main_op is not None else legacy_init_op

    # Add assets and ops
    self._add_collections(assets_collection, main_op, None)

    saver = self._maybe_create_saver(saver)

    # The graph almost certainly previously contained at least one Saver, and
    # possibly several (e.g. one for loading a pretrained embedding, and another
    # for the model weights).  Removing the preexisting ones was the
    # motivation for the clear_extraneous_savers option, but it turns out that
    # there are edge cases where that option breaks the graph.  Until that is
    # resolved, we just leave the option set to False for now.
    # TODO(soergel): Reinstate clear_extraneous_savers=True when possible.
    meta_graph_def = saver.export_meta_graph(
        clear_devices=clear_devices, strip_default_attrs=strip_default_attrs)

    # Tag the meta graph def and add it to the SavedModel.
    self._tag_and_add_meta_graph(meta_graph_def, tags, signature_def_map)

  @deprecated_args(None,
                   "Pass your op to the equivalent parameter main_op instead.",
                   "legacy_init_op")
  def add_meta_graph_and_variables(self,
                                   sess,
                                   tags,
                                   signature_def_map=None,
                                   assets_collection=None,
                                   legacy_init_op=None,
                                   clear_devices=False,
                                   main_op=None,
                                   strip_default_attrs=False,
                                   saver=None):
    if self._has_saved_variables:
      raise AssertionError("Graph state including variables and assets has "
                           "already been saved. Please invoke "
                           "`add_meta_graph()` instead.")

    # Validate the signature def map to ensure all included TensorInfos are
    # properly populated.
    signature_def_map = signature_def_map or {}
    self._validate_signature_def_map(signature_def_map)

    # legacy_init_op is deprecated, and going away in TF 2.0.
    # Re-mapping to main_op, as treatment is identical regardless.
    main_op = main_op or legacy_init_op

    # Add assets and ops
    self._add_collections(assets_collection, main_op, None)

    saved_model_utils.get_or_create_variables_dir(self._export_dir)
    variables_path = saved_model_utils.get_variables_path(self._export_dir)

    saver = self._maybe_create_saver(saver)

    # Save the variables. Also, disable writing the checkpoint state proto. The
    # file is not used during SavedModel loading. In addition, since a
    # SavedModel can be copied or moved, this avoids the checkpoint state to
    # become outdated.
    saver.save(sess, variables_path, write_meta_graph=False, write_state=False)

    # Export the meta graph def.

    # The graph almost certainly previously contained at least one Saver, and
    # possibly several (e.g. one for loading a pretrained embedding, and another
    # for the model weights).  Removing the preexisting ones was the
    # motivation for the clear_extraneous_savers option, but it turns out that
    # there are edge cases where that option breaks the graph.  Until that is
    # resolved, we just leave the option set to False for now.
    # TODO(soergel): Reinstate clear_extraneous_savers=True when possible.
    meta_graph_def = saver.export_meta_graph(
        clear_devices=clear_devices, strip_default_attrs=strip_default_attrs)

    # Tag the meta graph def and add it to the SavedModel.
    self._tag_and_add_meta_graph(meta_graph_def, tags, signature_def_map)

    # Mark this instance of SavedModel as having saved variables, such that
    # subsequent attempts to save variables will fail.
    self._has_saved_variables = True

  add_meta_graph.__doc__ = _SavedModelBuilder.add_meta_graph.__doc__.replace(
      "assets_list", "assets_collection")
  add_meta_graph_and_variables.__doc__ = \
      _SavedModelBuilder.add_meta_graph_and_variables.__doc__.replace(
          "assets_list", "assets_collection")


def _maybe_save_assets(write_fn, assets_to_add=None):
  """Saves assets to the meta graph.

  Args:
    write_fn: A function callback that writes assets into meta graph.
    assets_to_add: The list where the asset paths are setup.

  Returns:
    A dict of asset basenames for saving to the original full path to the asset.

  Raises:
    ValueError: Indicating an invalid filepath tensor.
  """
  # Map of target file names to original filenames
  asset_filename_map = {}

  if assets_to_add is None:
    tf_logging.info("No assets to save.")
    return asset_filename_map

  # Iterate over the supplied assets, build the `AssetFile` proto and add them
  # to the meta graph.
  for asset_tensor in assets_to_add:
    asset_source_filepath = _asset_path_from_tensor(asset_tensor)
    if not asset_source_filepath:
      raise ValueError(f"Asset filepath tensor {asset_tensor} in is invalid.")

    asset_filename = get_asset_filename_to_add(
        asset_source_filepath, asset_filename_map)

    # Call the passed-in function that builds AssetFileDef proto and adds it
    # to either the collection or asset_file_def field of the meta graph.
    # Note that this should be done even when the file is a duplicate of an
    # already-added file, as the tensor reference should still exist.
    write_fn(asset_filename, asset_tensor)

    # In the cases where we are adding a duplicate, this will result in the
    # last of the filepaths being the one used for copying the file to the
    # SavedModel. Since the files in question are the same, it doesn't matter
    # either way.
    asset_filename_map[asset_filename] = asset_source_filepath

  tf_logging.info("Assets added to graph.")
  return asset_filename_map


def get_asset_filename_to_add(asset_filepath, asset_filename_map):
  """Get a unique basename to add to the SavedModel if this file is unseen.

  Assets come from users as full paths, and we save them out to the
  SavedModel as basenames. In some cases, the basenames collide. Here,
  we dedupe asset basenames by first checking if the file is the same,
  and, if different, generate and return an index-suffixed basename
  that can be used to add the asset to the SavedModel.

  Args:
    asset_filepath: the full path to the asset that is being saved
    asset_filename_map: a dict of filenames used for saving the asset in
      the SavedModel to full paths from which the filenames were derived.

  Returns:
    Uniquified filename string if the file is not a duplicate, or the original
    filename if the file has already been seen and saved.
  """
  asset_filename = os.path.basename(asset_filepath)

  if asset_filename not in asset_filename_map:
    # This is an unseen asset. Safe to add.
    return asset_filename

  other_asset_filepath = asset_filename_map[asset_filename]
  if other_asset_filepath == asset_filepath:
    # This is the same file, stored twice in the list. No need
    # to make unique.
    return asset_filename

  # Else, asset_filename is in the map, and the filepath is different. Dedupe.
  if not file_io.filecmp(asset_filepath, other_asset_filepath):
    # Files are different; dedupe filenames.
    return _get_unique_asset_filename(asset_filename, asset_filename_map)

  # Files are the same; don't make unique.
  return asset_filename


def _get_unique_asset_filename(asset_filename, asset_filename_map):
  i = 1
  unique_filename = asset_filename
  while unique_filename in asset_filename_map:
    unique_filename = compat.as_bytes("_").join(
        [compat.as_bytes(asset_filename), compat.as_bytes(str(i))])
    i += 1
  return unique_filename


def _asset_path_from_tensor(path_tensor):
  """Returns the filepath value stored in constant `path_tensor`.

  Args:
    path_tensor: Tensor of a file-path.

  Returns:
    The string value i.e. path of the tensor, if valid.

  Raises:
    TypeError if tensor does not match expected op type, dtype or value.
  """
  if not isinstance(path_tensor, ops.Tensor):
    raise TypeError(f"Asset path tensor {path_tensor} must be a Tensor.")
  if path_tensor.op.type != "Const":
    raise TypeError(f"Asset path tensor {path_tensor} must be of type constant."
                    f"Has type {path_tensor.op.type} instead.")
  if path_tensor.dtype != dtypes.string:
    raise TypeError(f"Asset path tensor {path_tensor}` must be of dtype string."
                    f"Has type {path_tensor.dtype} instead.")
  str_values = path_tensor.op.get_attr("value").string_val
  if len(str_values) != 1:
    raise TypeError(f"Asset path tensor {path_tensor} must be a scalar.")
  return str_values[0]


def _add_asset_to_metagraph(meta_graph_def, asset_filename, asset_tensor):
  """Builds an asset proto and adds it to the meta graph def.

  Args:
    meta_graph_def: The meta graph def to which the asset will be added.
    asset_filename: The filename of the asset to be added.
    asset_tensor: The asset tensor used to populate the tensor info of the asset
      proto.
  """
  asset_proto = meta_graph_def.asset_file_def.add()
  asset_proto.filename = asset_filename
  asset_proto.tensor_info.name = asset_tensor.name


def copy_assets_to_destination_dir(asset_filename_map, destination_dir):
  """Copy all assets from source path to destination path."""
  assets_destination_dir = saved_model_utils.get_or_create_assets_dir(
      destination_dir)

  # Copy each asset from source path to destination path.
  for asset_basename, asset_source_filepath in asset_filename_map.items():
    asset_destination_filepath = file_io.join(
        compat.as_bytes(assets_destination_dir),
        compat.as_bytes(asset_basename))

    # Copy asset file to the destination.
    file_io.copy(
        asset_source_filepath, asset_destination_filepath, overwrite=True)

  tf_logging.info("Assets written to: %s",
                  compat.as_text(assets_destination_dir))


def _add_asset_to_collection(asset_filename, asset_tensor):
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


def _add_op_to_signature_def_map(signature_def_map, op, key):
  if op is not None:
    signature_def_map[key] = signature_def_utils.op_signature_def(op, key)
