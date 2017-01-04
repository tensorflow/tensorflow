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
"""Loader functionality for SavedModel with hermetic, language-neutral exports.

Load and restore capability for a SavedModel, which may include multiple meta
graph defs. Each SavedModel is associated with a single checkpoint. Each meta
graph def is saved with one or more tags, which are used to identify the exact
meta graph def to load.

The `load` operation requires the session in which to restore the graph
definition and variables, the tags used to identify the meta graph def to
load and the location of the SavedModel.

Upon a load, the subset of variables and assets supplied as part of the specific
meta graph def, will be restored into the supplied session. The values of the
variables though will correspond to the saved values from the first meta graph
added to the SavedModel using `add_meta_graph_and_variables(...)` in
`builder.py`.

Typical usage:
```python
...
builder = saved_model_builder.SavedModelBuilder(export_dir)

with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph_and_variables(sess,
                                       ["foo-tag"],
                                       signature_def_map=foo_signatures,
                                       assets_collection=foo_assets)
...

with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph(["bar-tag", "baz-tag"],
                         assets_collection=bar_baz_assets)
...

builder.save()

...
with tf.Session(graph=tf.Graph()) as sess:
  loader.load(sess, ["foo-tag"], export_dir)
  ...

```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from google.protobuf import text_format

from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import constants
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import compat


def _parse_saved_model(export_dir):
  """Reads the savedmodel.pb or savedmodel.pbtxt file containing `SavedModel`.

  Args:
    export_dir: Directory containing the SavedModel file.

  Returns:
    A `SavedModel` protocol buffer.

  Raises:
    IOError: If the file does not exist, or cannot be successfully parsed.
  """
  # Build the path to the SavedModel in pbtxt format.
  path_to_pbtxt = os.path.join(
      compat.as_bytes(export_dir),
      compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT))
  # Build the path to the SavedModel in pb format.
  path_to_pb = os.path.join(
      compat.as_bytes(export_dir),
      compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))

  # Ensure that the SavedModel exists at either path.
  if not file_io.file_exists(path_to_pbtxt) and not file_io.file_exists(
      path_to_pb):
    raise IOError("SavedModel file does not exist at: %s" % export_dir)

  saved_model = saved_model_pb2.SavedModel()

  # Parse the SavedModel protocol buffer.
  try:
    file_content = file_io.read_file_to_string(path_to_pb)
    saved_model.ParseFromString(file_content)
    return saved_model
  except Exception:  # pylint: disable=broad-except
    # Pass for exceptions in order to try reading the file in text format.
    pass

  try:
    file_content = file_io.read_file_to_string(path_to_pbtxt)
    text_format.Merge(file_content.decode("utf-8"), saved_model)
  except text_format.ParseError as e:
    raise IOError("Cannot parse file %s: %s." % (path_to_pbtxt, str(e)))
  return saved_model


def _get_asset_tensors(export_dir, meta_graph_def_to_load):
  """Gets the asset tensors, if defined in the meta graph def to load.

  Args:
    export_dir: Directory where the SavedModel is located.
    meta_graph_def_to_load: The meta graph def from the SavedModel to be loaded.

  Returns:
    A dictionary of asset tensors, keyed by the name of the asset tensor. The
    value in the map corresponds to the absolute path of the asset file.
  """
  # Collection-def that may contain the assets key.
  collection_def = meta_graph_def_to_load.collection_def

  asset_tensor_dict = {}
  if constants.ASSETS_KEY in collection_def:
    # Location of the assets for SavedModel.
    assets_directory = os.path.join(
        compat.as_bytes(export_dir),
        compat.as_bytes(constants.ASSETS_DIRECTORY))
    assets_any_proto = collection_def[constants.ASSETS_KEY].any_list.value
    # Process each asset and add it to the asset tensor dictionary.
    for asset_any_proto in assets_any_proto:
      asset_proto = meta_graph_pb2.AssetFileDef()
      asset_any_proto.Unpack(asset_proto)
      asset_tensor_dict[asset_proto.tensor_info.name] = os.path.join(
          compat.as_bytes(assets_directory),
          compat.as_bytes(asset_proto.filename))
  return asset_tensor_dict


def _get_main_op_tensor(meta_graph_def_to_load):
  """Gets the main op tensor, if one exists.

  Args:
    meta_graph_def_to_load: The meta graph def from the SavedModel to be loaded.

  Returns:
    The main op tensor, if it exists and `None` otherwise.

  Raises:
    RuntimeError: If the collection def corresponding to the main op key has
        other than exactly one tensor.
  """
  collection_def = meta_graph_def_to_load.collection_def
  main_op_tensor = None
  if constants.MAIN_OP_KEY in collection_def:
    main_ops = collection_def[constants.MAIN_OP_KEY].node_list.value
    if len(main_ops) != 1:
      raise RuntimeError("Expected exactly one SavedModel main op.")
    main_op_tensor = ops.get_collection(constants.MAIN_OP_KEY)[0]
  return main_op_tensor


def _get_legacy_init_op_tensor(meta_graph_def_to_load):
  """Gets the legacy init op tensor, if one exists.

  Args:
    meta_graph_def_to_load: The meta graph def from the SavedModel to be loaded.

  Returns:
    The legacy init op tensor, if it exists and `None` otherwise.

  Raises:
    RuntimeError: If the collection def corresponding to the legacy init op key
        has other than exactly one tensor.
  """
  collection_def = meta_graph_def_to_load.collection_def
  legacy_init_op_tensor = None
  if constants.LEGACY_INIT_OP_KEY in collection_def:
    legacy_init_ops = collection_def[
        constants.LEGACY_INIT_OP_KEY].node_list.value
    if len(legacy_init_ops) != 1:
      raise RuntimeError("Expected exactly one legacy serving init op.")
    legacy_init_op_tensor = ops.get_collection(constants.LEGACY_INIT_OP_KEY)[0]
  return legacy_init_op_tensor


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
  return (file_io.file_exists(txt_path) or file_io.file_exists(pb_path))


def load(sess, tags, export_dir):
  """Loads the model from a SavedModel as specified by tags.

  Args:
    sess: The TensorFlow session to restore the variables.
    tags: Set of string tags to identify the required MetaGraphDef. These should
        correspond to the tags used when saving the variables using the
        SavedModel `save()` API.
    export_dir: Directory in which the SavedModel protocol buffer and variables
        to be loaded are located.

  Returns:
    The `MetaGraphDef` protocol buffer loaded in the provided session. This
    can be used to further extract signature-defs, collection-defs, etc.

  Raises:
    RuntimeError: MetaGraphDef associated with the tags cannot be found.
  """
  # Build the SavedModel protocol buffer and find the requested meta graph def.
  saved_model = _parse_saved_model(export_dir)
  found_match = False
  for meta_graph_def in saved_model.meta_graphs:
    if set(meta_graph_def.meta_info_def.tags) == set(tags):
      meta_graph_def_to_load = meta_graph_def
      found_match = True
      break

  if not found_match:
    raise RuntimeError("MetaGraphDef associated with tags " + str(tags).strip(
        "[]") + " could not be found in SavedModel")

  # Build a saver by importing the meta graph def to load.
  saver = tf_saver.import_meta_graph(meta_graph_def_to_load)

  # Build the checkpoint path where the variables are located.
  variables_path = os.path.join(
      compat.as_bytes(export_dir),
      compat.as_bytes(constants.VARIABLES_DIRECTORY),
      compat.as_bytes(constants.VARIABLES_FILENAME))

  # Restore the variables using the built saver in the provided session.
  saver.restore(sess, variables_path)

  # Get asset tensors, if any.
  asset_tensors_dictionary = _get_asset_tensors(export_dir,
                                                meta_graph_def_to_load)

  main_op_tensor = _get_main_op_tensor(meta_graph_def_to_load)
  if main_op_tensor is not None:
    sess.run(fetches=[main_op_tensor], feed_dict=asset_tensors_dictionary)
  else:
    legacy_init_op_tensor = _get_legacy_init_op_tensor(meta_graph_def_to_load)
    if legacy_init_op_tensor is not None:
      sess.run(fetches=[legacy_init_op_tensor],
               feed_dict=asset_tensors_dictionary)

  return meta_graph_def_to_load
