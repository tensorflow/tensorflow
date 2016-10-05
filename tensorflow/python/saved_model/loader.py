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
added to the SavedModel using `add_graph_and_variables(...)` in `builder.py`.

TODO(sukritiramesh): Add support for a single init or main op to run upon load.

Typical usage:
```python
...
builder = saved_model_builder.SavedModelBuilder(export_dir)

with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_graph_and_variables(sess,
                                  ["foo-tag"],
                                  signature_def_map=foo_signatures,
                                  asset_collection=foo_assets)
...

with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_graph(["bar-tag", "baz-tag"])
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
from tensorflow.core.protobuf import saved_model_pb2
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
    The `MetaGraphDef` protocol buffer loaloadded in the provided session. This
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
      compat.as_bytes(constants.VARIABLES_FILENAME_SHARDED))

  # Restore the variables using the built saver in the provided session.
  saver.restore(sess, variables_path)

  # Return the meta graph def that was loaded into the session.
  return meta_graph_def_to_load
