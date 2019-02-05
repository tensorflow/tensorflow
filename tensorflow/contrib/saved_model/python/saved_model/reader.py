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
"""SavedModel functionality to read a SavedModel from disk."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from google.protobuf import message
from google.protobuf import text_format
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import constants
from tensorflow.python.util import compat


def read_saved_model(saved_model_dir):
  """Reads the savedmodel.pb or savedmodel.pbtxt file containing `SavedModel`.

  Args:
    saved_model_dir: Directory containing the SavedModel file.

  Returns:
    A `SavedModel` protocol buffer.

  Raises:
    IOError: If the file does not exist, or cannot be successfully parsed.
  """
  # Build the path to the SavedModel in pbtxt format.
  path_to_pbtxt = os.path.join(
      compat.as_bytes(saved_model_dir),
      compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT))
  # Build the path to the SavedModel in pb format.
  path_to_pb = os.path.join(
      compat.as_bytes(saved_model_dir),
      compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))

  # Ensure that the SavedModel exists at either path.
  if not file_io.file_exists(path_to_pbtxt) and not file_io.file_exists(
      path_to_pb):
    raise IOError("SavedModel file does not exist at: %s" % saved_model_dir)

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
                  (saved_model_dir, constants.SAVED_MODEL_FILENAME_PBTXT,
                   constants.SAVED_MODEL_FILENAME_PB))


def get_saved_model_tag_sets(saved_model_dir):
  """Retrieves all the tag-sets available in the SavedModel.

  Args:
    saved_model_dir: Directory containing the SavedModel.

  Returns:
    String representation of all tag-sets in the SavedModel.
  """
  saved_model = read_saved_model(saved_model_dir)
  all_tags = []
  for meta_graph_def in saved_model.meta_graphs:
    all_tags.append(list(meta_graph_def.meta_info_def.tags))
  return all_tags
