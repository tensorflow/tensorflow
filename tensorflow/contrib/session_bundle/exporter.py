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
"""Export a TensorFlow model.

See: go/tf-exporter
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import six

from google.protobuf.any_pb2 import Any

from tensorflow.contrib.session_bundle import constants
from tensorflow.contrib.session_bundle import gc
from tensorflow.contrib.session_bundle import manifest_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import training_util
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated


@deprecated("2017-06-30", "Please use SavedModel instead.")
def gfile_copy_callback(files_to_copy, export_dir_path):
  """Callback to copy files using `gfile.Copy` to an export directory.

  This method is used as the default `assets_callback` in `Exporter.init` to
  copy assets from the `assets_collection`. It can also be invoked directly to
  copy additional supplementary files into the export directory (in which case
  it is not a callback).

  Args:
    files_to_copy: A dictionary that maps original file paths to desired
      basename in the export directory.
    export_dir_path: Directory to copy the files to.
  """
  logging.info("Write assets into: %s using gfile_copy.", export_dir_path)
  gfile.MakeDirs(export_dir_path)
  for source_filepath, basename in files_to_copy.items():
    new_path = os.path.join(
        compat.as_bytes(export_dir_path), compat.as_bytes(basename))
    logging.info("Copying asset %s to path %s.", source_filepath, new_path)

    if gfile.Exists(new_path):
      # Guard against being restarted while copying assets, and the file
      # existing and being in an unknown state.
      # TODO(b/28676216): Do some file checks before deleting.
      logging.info("Removing file %s.", new_path)
      gfile.Remove(new_path)
    gfile.Copy(source_filepath, new_path)


@deprecated("2017-06-30", "Please use SavedModel instead.")
def regression_signature(input_tensor, output_tensor):
  """Creates a regression signature.

  Args:
    input_tensor: Tensor specifying the input to a graph.
    output_tensor: Tensor specifying the output of a graph.

  Returns:
    A Signature message.
  """
  signature = manifest_pb2.Signature()
  signature.regression_signature.input.tensor_name = input_tensor.name
  signature.regression_signature.output.tensor_name = output_tensor.name
  return signature


@deprecated("2017-06-30", "Please use SavedModel instead.")
def classification_signature(input_tensor,
                             classes_tensor=None,
                             scores_tensor=None):
  """Creates a classification signature.

  Args:
    input_tensor: Tensor specifying the input to a graph.
    classes_tensor: Tensor specifying the output classes of a graph.
    scores_tensor: Tensor specifying the scores of the output classes.

  Returns:
    A Signature message.
  """
  signature = manifest_pb2.Signature()
  signature.classification_signature.input.tensor_name = input_tensor.name
  if classes_tensor is not None:
    signature.classification_signature.classes.tensor_name = classes_tensor.name
  if scores_tensor is not None:
    signature.classification_signature.scores.tensor_name = scores_tensor.name
  return signature


@deprecated("2017-06-30", "Please use SavedModel instead.")
def generic_signature(name_tensor_map):
  """Creates a generic signature of name to Tensor name.

  Args:
    name_tensor_map: Map from logical name to Tensor.

  Returns:
    A Signature message.
  """
  signature = manifest_pb2.Signature()
  for name, tensor in six.iteritems(name_tensor_map):
    signature.generic_signature.map[name].tensor_name = tensor.name
  return signature


class Exporter(object):
  """Exporter helps package a TensorFlow model for serving.

  Args:
    saver: Saver object.
  """

  def __init__(self, saver):
    # Makes a copy of the saver-def and disables garbage-collection, since the
    # exporter enforces garbage-collection independently. Specifically, since
    # the exporter performs atomic copies of the saver output, it is required
    # that garbage-collection via the underlying saver be disabled.
    saver_def = saver.as_saver_def()
    saver_def.ClearField("max_to_keep")
    self._saver = tf_saver.Saver(saver_def=saver_def)
    self._has_init = False
    self._assets_to_copy = {}

  @deprecated("2017-06-30", "Please use SavedModel instead.")
  def init(self,
           graph_def=None,
           init_op=None,
           clear_devices=False,
           default_graph_signature=None,
           named_graph_signatures=None,
           assets_collection=None,
           assets_callback=gfile_copy_callback):
    """Initialization.

    Args:
      graph_def: A GraphDef message of the graph to be used in inference.
        GraphDef of default graph is used when None.
      init_op: Op to be used in initialization.
      clear_devices: If device info of the graph should be cleared upon export.
      default_graph_signature: Default signature of the graph.
      named_graph_signatures: Map of named input/output signatures of the graph.
      assets_collection: A collection of constant asset filepath tensors. If set
        the assets will be exported into the asset directory.
      assets_callback: callback with two argument called during export with the
        list of files to copy and the asset path.
    Raises:
      RuntimeError: if init is called more than once.
      TypeError: if init_op is not an Operation or None.
      ValueError: if asset file path tensors are not non-empty constant string
        scalar tensors.
    """
    # Avoid Dangerous default value []
    if named_graph_signatures is None:
      named_graph_signatures = {}
    assets = []
    if assets_collection:
      for asset_tensor in assets_collection:
        asset_filepath = self._file_path_value(asset_tensor)
        if not asset_filepath:
          raise ValueError("invalid asset filepath tensor %s" % asset_tensor)
        basename = os.path.basename(asset_filepath)
        assets.append((basename, asset_tensor))
        self._assets_to_copy[asset_filepath] = basename

    if self._has_init:
      raise RuntimeError("init should be called only once")
    self._has_init = True

    if graph_def or clear_devices:
      copy = graph_pb2.GraphDef()
      if graph_def:
        copy.CopyFrom(graph_def)
      else:
        copy.CopyFrom(ops.get_default_graph().as_graph_def())
      if clear_devices:
        for node in copy.node:
          node.device = ""
      graph_any_buf = Any()
      graph_any_buf.Pack(copy)
      ops.add_to_collection(constants.GRAPH_KEY, graph_any_buf)

    if init_op:
      if not isinstance(init_op, ops.Operation):
        raise TypeError("init_op needs to be an Operation: %s" % init_op)
      ops.add_to_collection(constants.INIT_OP_KEY, init_op)

    signatures_proto = manifest_pb2.Signatures()
    if default_graph_signature:
      signatures_proto.default_signature.CopyFrom(default_graph_signature)
    for signature_name, signature in six.iteritems(named_graph_signatures):
      signatures_proto.named_signatures[signature_name].CopyFrom(signature)
    signatures_any_buf = Any()
    signatures_any_buf.Pack(signatures_proto)
    ops.add_to_collection(constants.SIGNATURES_KEY, signatures_any_buf)

    for filename, tensor in assets:
      asset = manifest_pb2.AssetFile()
      asset.filename = filename
      asset.tensor_binding.tensor_name = tensor.name
      asset_any_buf = Any()
      asset_any_buf.Pack(asset)
      ops.add_to_collection(constants.ASSETS_KEY, asset_any_buf)

    self._assets_callback = assets_callback

  @deprecated("2017-06-30", "Please use SavedModel instead.")
  def export(self,
             export_dir_base,
             global_step_tensor,
             sess=None,
             exports_to_keep=None):
    """Exports the model.

    Args:
      export_dir_base: A string path to the base export dir.
      global_step_tensor: An Tensor or tensor name providing the
        global step counter to append to the export directory path and set
        in the manifest version.
      sess: A Session to use to save the parameters.
      exports_to_keep: a gc.Path filter function used to determine the set of
        exports to keep. If set to None, all versions will be kept.

    Returns:
      The string path to the exported directory.

    Raises:
      RuntimeError: if init is not called.
      RuntimeError: if the export would overwrite an existing directory.
    """
    if not self._has_init:
      raise RuntimeError("init must be called first")

    # Export dir must not end with / or it will break exports to keep. Strip /.
    if export_dir_base.endswith("/"):
      export_dir_base = export_dir_base[:-1]

    global_step = training_util.global_step(sess, global_step_tensor)
    export_dir = os.path.join(
        compat.as_bytes(export_dir_base),
        compat.as_bytes(constants.VERSION_FORMAT_SPECIFIER % global_step))

    # Prevent overwriting on existing exports which could lead to bad/corrupt
    # storage and loading of models. This is an important check that must be
    # done before any output files or directories are created.
    if gfile.Exists(export_dir):
      raise RuntimeError("Overwriting exports can cause corruption and are "
                         "not allowed. Duplicate export dir: %s" % export_dir)

    # Output to a temporary directory which is atomically renamed to the final
    # directory when complete.
    tmp_export_dir = compat.as_text(export_dir) + "-tmp"
    gfile.MakeDirs(tmp_export_dir)

    self._saver.save(sess,
                     os.path.join(
                         compat.as_text(tmp_export_dir),
                         compat.as_text(constants.EXPORT_BASE_NAME)),
                     meta_graph_suffix=constants.EXPORT_SUFFIX_NAME)

    # Run the asset callback.
    if self._assets_callback and self._assets_to_copy:
      assets_dir = os.path.join(
          compat.as_bytes(tmp_export_dir),
          compat.as_bytes(constants.ASSETS_DIRECTORY))
      gfile.MakeDirs(assets_dir)
      self._assets_callback(self._assets_to_copy, assets_dir)

    # TODO(b/27794910): Delete *checkpoint* file before rename.
    gfile.Rename(tmp_export_dir, export_dir)

    if exports_to_keep:
      # create a simple parser that pulls the export_version from the directory.
      def parser(path):
        match = re.match("^" + export_dir_base + "/(\\d{8})$", path.path)
        if not match:
          return None
        return path._replace(export_version=int(match.group(1)))

      paths_to_delete = gc.negation(exports_to_keep)
      for p in paths_to_delete(gc.get_paths(export_dir_base, parser=parser)):
        gfile.DeleteRecursively(p.path)

    return export_dir

  def _file_path_value(self, path_tensor):
    """Returns the filepath value stored in constant `path_tensor`."""
    if not isinstance(path_tensor, ops.Tensor):
      raise TypeError("tensor is not a Tensor")
    if path_tensor.op.type != "Const":
      raise TypeError("Only constants tensor are supported")
    if path_tensor.dtype != dtypes.string:
      raise TypeError("File paths should be string")
    str_value = path_tensor.op.get_attr("value").string_val
    if len(str_value) != 1:
      raise TypeError("Only scalar tensors are supported")
    return str_value[0]
