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
"""Loader implementation for SavedModel with hermetic, language-neutral exports.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from google.protobuf import message
from google.protobuf import text_format

from tensorflow.core.protobuf import graph_debug_info_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


def parse_saved_model_with_debug_info(export_dir):
  """Reads the savedmodel as well as the graph debug info.

  Args:
    export_dir: Directory containing the SavedModel and GraphDebugInfo files.

  Returns:
    `SavedModel` and `GraphDebugInfo` protocol buffers.

  Raises:
    IOError: If the saved model file does not exist, or cannot be successfully
    parsed. Missing graph debug info file is fine.
  """
  saved_model = _parse_saved_model(export_dir)

  debug_info_path = os.path.join(
      saved_model_utils.get_debug_dir(export_dir),
      constants.DEBUG_INFO_FILENAME_PB)
  debug_info = graph_debug_info_pb2.GraphDebugInfo()
  if file_io.file_exists(debug_info_path):
    with file_io.FileIO(debug_info_path, "rb") as debug_file:
      try:
        debug_info.ParseFromString(debug_file.read())
      except message.DecodeError as e:
        raise IOError("Cannot parse file %s: %s." % (debug_info_path, str(e)))

  return (saved_model, debug_info)


def parse_saved_model(export_dir):
  """Reads the savedmodel.pb or savedmodel.pbtxt file containing `SavedModel`.

  Args:
    export_dir: String or Pathlike, path to the directory containing the
    SavedModel file.

  Returns:
    A `SavedModel` protocol buffer.

  Raises:
    IOError: If the file does not exist, or cannot be successfully parsed.
  """
  # Build the path to the SavedModel in pbtxt format.
  path_to_pbtxt = os.path.join(
      compat.as_bytes(compat.path_to_str(export_dir)),
      compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT))
  # Build the path to the SavedModel in pb format.
  path_to_pb = os.path.join(
      compat.as_bytes(compat.path_to_str(export_dir)),
      compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))

  # Parse the SavedModel protocol buffer.
  saved_model = saved_model_pb2.SavedModel()
  if file_io.file_exists(path_to_pb):
    try:
      file_content = file_io.FileIO(path_to_pb, "rb").read()
      saved_model.ParseFromString(file_content)
      return saved_model
    except message.DecodeError as e:
      raise IOError("Cannot parse file %s: %s." % (path_to_pb, str(e)))
  elif file_io.file_exists(path_to_pbtxt):
    try:
      file_content = file_io.FileIO(path_to_pbtxt, "rb").read()
      text_format.Merge(file_content.decode("utf-8"), saved_model)
      return saved_model
    except text_format.ParseError as e:
      raise IOError("Cannot parse file %s: %s." % (path_to_pbtxt, str(e)))
  else:
    raise IOError("SavedModel file does not exist at: %s/{%s|%s}" %
                  (export_dir,
                   constants.SAVED_MODEL_FILENAME_PBTXT,
                   constants.SAVED_MODEL_FILENAME_PB))


# TODO(b/120594573): Make this symbol also available as private, so that
# tensorflow_transform and tensorflow_estimator do not break.
_parse_saved_model = parse_saved_model


def get_asset_tensors(export_dir, meta_graph_def_to_load, import_scope=None):
  """Gets the asset tensors, if defined in the meta graph def to load.

  Args:
    export_dir: Directory where the SavedModel is located.
    meta_graph_def_to_load: The meta graph def from the SavedModel to be loaded.
    import_scope: Optional `string` -- if specified, prepend this followed by
        '/' to all returned asset tensor names.

  Returns:
    A dictionary of asset tensors, keyed by the name of the asset tensor. The
    value in the map corresponds to the absolute path of the asset file.
  """
  # Collection-def that may contain the assets key.
  collection_def = meta_graph_def_to_load.collection_def

  asset_tensor_dict = {}
  asset_protos = []

  if meta_graph_def_to_load.asset_file_def:
    asset_protos = meta_graph_def_to_load.asset_file_def
  elif constants.ASSETS_KEY in collection_def:
    assets_any_proto = collection_def[constants.ASSETS_KEY].any_list.value
    for asset_any_proto in assets_any_proto:
      asset_proto = meta_graph_pb2.AssetFileDef()
      asset_any_proto.Unpack(asset_proto)
      asset_protos.append(asset_proto)

  # Location of the assets for SavedModel.
  assets_directory = os.path.join(
      compat.as_bytes(export_dir), compat.as_bytes(constants.ASSETS_DIRECTORY))
  # Process each asset and add it to the asset tensor dictionary.
  for asset_proto in asset_protos:
    tensor_name = asset_proto.tensor_info.name
    if import_scope:
      tensor_name = "%s/%s" % (import_scope, tensor_name)
    asset_tensor_dict[tensor_name] = os.path.join(
        compat.as_bytes(assets_directory),
        compat.as_bytes(asset_proto.filename))

  return asset_tensor_dict


def _get_main_op_tensor(
    meta_graph_def_to_load, init_op_key=constants.MAIN_OP_KEY):
  """Gets the main op tensor, if one exists.

  Args:
    meta_graph_def_to_load: The meta graph def from the SavedModel to be loaded.
    init_op_key: name of the collection to check; should be one of MAIN_OP_KEY
      or the deprecated LEGACY_INIT_OP_KEY

  Returns:
    The main op tensor, if it exists and `None` otherwise.

  Raises:
    RuntimeError: If the collection def corresponding to the main op key has
        other than exactly one tensor.
  """
  # TODO(kathywu): Rename this method to _get_op_from_collection when
  # dependency from SavedModelEstimator is removed.
  collection_def = meta_graph_def_to_load.collection_def
  init_op = None
  if init_op_key in collection_def:
    init_op_list = collection_def[init_op_key].node_list.value
    if len(init_op_list) != 1:
      raise RuntimeError("Expected exactly one SavedModel init op. "
                         "Found: {}".format(init_op_list))
    init_op = ops.get_collection(init_op_key)[0]
  return init_op


def _get_op_from_collection(meta_graph_def, op_key):
  return _get_main_op_tensor(meta_graph_def, op_key)


def _get_op_from_signature_def(meta_graph_def, op_signature_key, import_scope):
  """Retrieve op stored in the imported meta graph's signature def."""
  if op_signature_key in meta_graph_def.signature_def:
    return signature_def_utils.load_op_from_signature_def(
        meta_graph_def.signature_def[op_signature_key], op_signature_key,
        import_scope)
  else:
    return None


def get_init_op(meta_graph_def, import_scope=None):
  return (_get_op_from_signature_def(
      meta_graph_def, constants.INIT_OP_SIGNATURE_KEY, import_scope) or
          _get_op_from_collection(meta_graph_def, constants.MAIN_OP_KEY) or
          _get_op_from_collection(meta_graph_def, constants.LEGACY_INIT_OP_KEY))


def get_train_op(meta_graph_def, import_scope=None):
  train_op = _get_op_from_signature_def(
      meta_graph_def, constants.TRAIN_OP_SIGNATURE_KEY, import_scope)
  if train_op is None:
    train_op = _get_op_from_collection(meta_graph_def, constants.TRAIN_OP_KEY)
  return train_op


@tf_export(v1=[
    "saved_model.contains_saved_model",
    "saved_model.maybe_saved_model_directory",
    "saved_model.loader.maybe_saved_model_directory"
])
@deprecation.deprecated_endpoints(
    "saved_model.loader.maybe_saved_model_directory")
def maybe_saved_model_directory(export_dir):
  """Checks whether the provided export directory could contain a SavedModel.

  Note that the method does not load any data by itself. If the method returns
  `false`, the export directory definitely does not contain a SavedModel. If the
  method returns `true`, the export directory may contain a SavedModel but
  provides no guarantee that it can be loaded.

  Args:
    export_dir: Absolute string path to possible export location. For example,
                '/my/foo/model'.

  Returns:
    True if the export directory contains SavedModel files, False otherwise.
  """
  txt_path = os.path.join(export_dir, constants.SAVED_MODEL_FILENAME_PBTXT)
  pb_path = os.path.join(export_dir, constants.SAVED_MODEL_FILENAME_PB)
  return file_io.file_exists(txt_path) or file_io.file_exists(pb_path)


@tf_export("saved_model.contains_saved_model", v1=[])
def contains_saved_model(export_dir):
  """Checks whether the provided export directory could contain a SavedModel.

  Note that the method does not load any data by itself. If the method returns
  `false`, the export directory definitely does not contain a SavedModel. If the
  method returns `true`, the export directory may contain a SavedModel but
  provides no guarantee that it can be loaded.

  Args:
    export_dir: Absolute string path to possible export location. For example,
                '/my/foo/model'.

  Returns:
    True if the export directory contains SavedModel files, False otherwise.
  """
  return maybe_saved_model_directory(export_dir)


@tf_export(v1=["saved_model.load", "saved_model.loader.load"])
@deprecation.deprecated(
    None,
    "This function will only be available through the v1 compatibility "
    "library as tf.compat.v1.saved_model.loader.load or "
    "tf.compat.v1.saved_model.load. There will be a new function for importing "
    "SavedModels in Tensorflow 2.0.")
def load(sess, tags, export_dir, import_scope=None, **saver_kwargs):
  """Loads the model from a SavedModel as specified by tags.

  Args:
    sess: The TensorFlow session to restore the variables.
    tags: Set of string tags to identify the required MetaGraphDef. These should
        correspond to the tags used when saving the variables using the
        SavedModel `save()` API.
    export_dir: Directory in which the SavedModel protocol buffer and variables
        to be loaded are located.
    import_scope: Optional `string` -- if specified, prepend this string
        followed by '/' to all loaded tensor names. This scope is applied to
        tensor instances loaded into the passed session, but it is *not* written
        through to the static `MetaGraphDef` protocol buffer that is returned.
    **saver_kwargs: Optional keyword arguments passed through to Saver.

  Returns:
    The `MetaGraphDef` protocol buffer loaded in the provided session. This
    can be used to further extract signature-defs, collection-defs, etc.

  Raises:
    RuntimeError: MetaGraphDef associated with the tags cannot be found.
  """
  loader = SavedModelLoader(export_dir)
  return loader.load(sess, tags, import_scope, **saver_kwargs)


class SavedModelLoader(object):
  """Load graphs and restore variable values from a `SavedModel`."""

  def __init__(self, export_dir):
    """Creates a `SavedModelLoader`.

    Args:
      export_dir: Directory in which the SavedModel protocol buffer and
        variables to be loaded are located.
    """
    self._export_dir = export_dir
    self._variables_path = saved_model_utils.get_variables_path(export_dir)
    self._saved_model = parse_saved_model(export_dir)

  @property
  def export_dir(self):
    """Directory containing the SavedModel."""
    return self._export_dir

  @property
  def variables_path(self):
    """Path to variable checkpoint files."""
    return self._variables_path

  @property
  def saved_model(self):
    """SavedModel object parsed from the export directory."""
    return self._saved_model

  def get_meta_graph_def_from_tags(self, tags):
    """Return MetaGraphDef with the exact specified tags.

    Args:
      tags: A list or set of string tags that identify the MetaGraphDef.

    Returns:
      MetaGraphDef with the same tags.

    Raises:
      RuntimeError: if no metagraphs were found with the associated tags.
    """
    found_match = False
    available_tags = []
    for meta_graph_def in self._saved_model.meta_graphs:
      available_tags.append(set(meta_graph_def.meta_info_def.tags))
      if set(meta_graph_def.meta_info_def.tags) == set(tags):
        meta_graph_def_to_load = meta_graph_def
        found_match = True
        break

    if not found_match:
      raise RuntimeError(
          "MetaGraphDef associated with tags " + str(tags).strip("[]") +
          " could not be found in SavedModel. To inspect available tag-sets in"
          " the SavedModel, please use the SavedModel CLI: `saved_model_cli`"
          "\navailable_tags: " + str(available_tags))
    return meta_graph_def_to_load

  def load_graph(self, graph, tags, import_scope=None, **saver_kwargs):
    """Load ops and nodes from SavedModel MetaGraph into graph.

    Args:
      graph: tf.Graph object.
      tags: a set of string tags identifying a MetaGraphDef.
      import_scope: Optional `string` -- if specified, prepend this string
        followed by '/' to all loaded tensor names. This scope is applied to
        tensor instances loaded into the passed session, but it is *not* written
        through to the static `MetaGraphDef` protocol buffer that is returned.
      **saver_kwargs: keyword arguments to pass to tf.train.import_meta_graph.

    Returns:
      A tuple of
        * Saver defined by the MetaGraph, which can be used to restore the
          variable values.
        * List of `Operation`/`Tensor` objects returned from
          `tf.import_graph_def` (may be `None`).
    """
    meta_graph_def = self.get_meta_graph_def_from_tags(tags)
    with graph.as_default():
      return tf_saver._import_meta_graph_with_return_elements(  # pylint: disable=protected-access
          meta_graph_def, import_scope=import_scope, **saver_kwargs)

  def restore_variables(self, sess, saver, import_scope=None):
    """Restore SavedModel variable values into the session.

    Args:
      sess: tf.compat.v1.Session to restore variable values.
      saver: a tf.compat.v1.train.Saver object. Can be None if there are no
        variables in graph. This may be the saver returned by the load_graph()
        function, or a default `tf.compat.v1.train.Saver()`.
      import_scope: Optional `string` -- if specified, prepend this string
        followed by '/' to all loaded tensor names. This scope is applied to
        tensor instances loaded into the passed session, but it is *not* written
        through to the static `MetaGraphDef` protocol buffer that is returned.

    Raises:
      ValueError: if no saver was passed to the saver argument, and there are
        variables in the graph.
    """
    with sess.graph.as_default():
      if (saver is None and
          not variables._all_saveable_objects(scope=import_scope)):  # pylint: disable=protected-access
        tf_logging.info("The specified SavedModel has no variables; no "
                        "checkpoints were restored.")
      elif isinstance(saver, tf_saver.Saver):
        saver.restore(sess, self._variables_path)
      else:
        raise ValueError(
            "No tf.train.Saver object was passed to the function "
            "SavedModelLoader.restore_variables. Since there are variables in "
            "the graph, a saver is required.")

  def run_init_ops(self, sess, tags, import_scope=None):
    """Run initialization ops defined in the `MetaGraphDef`.

    Args:
      sess: tf.compat.v1.Session to restore variable values.
      tags: a set of string tags identifying a MetaGraphDef.
      import_scope: Optional `string` -- if specified, prepend this string
        followed by '/' to all loaded tensor names. This scope is applied to
        tensor instances loaded into the passed session, but it is *not* written
        through to the static `MetaGraphDef` protocol buffer that is returned.
    """
    meta_graph_def = self.get_meta_graph_def_from_tags(tags)
    with sess.graph.as_default():
      # Get asset tensors, if any.
      asset_tensors_dictionary = get_asset_tensors(
          self._export_dir, meta_graph_def, import_scope=import_scope)

      init_op = get_init_op(meta_graph_def, import_scope)
      if init_op is not None:
        sess.run(fetches=[init_op], feed_dict=asset_tensors_dictionary)

  def load(self, sess, tags, import_scope=None, **saver_kwargs):
    """Load the MetaGraphDef graph and restore variable values into the session.

    Args:
      sess: tf.compat.v1.Session to restore variable values.
      tags: a set of string tags identifying a MetaGraphDef.
      import_scope: Optional `string` -- if specified, prepend this string
        followed by '/' to all loaded tensor names. This scope is applied to
        tensor instances loaded into the passed session, but it is *not* written
        through to the static `MetaGraphDef` protocol buffer that is returned.
      **saver_kwargs: keyword arguments to pass to tf.train.import_meta_graph.

    Returns:
      `MetagraphDef` proto of the graph that was loaded.
    """
    with sess.graph.as_default():
      saver, _ = self.load_graph(sess.graph, tags, import_scope,
                                 **saver_kwargs)
      self.restore_variables(sess, saver, import_scope)
      self.run_init_ops(sess, tags, import_scope)
    return self.get_meta_graph_def_from_tags(tags)
