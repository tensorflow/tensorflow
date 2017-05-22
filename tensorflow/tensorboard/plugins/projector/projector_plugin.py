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
"""The Embedding Projector plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import imghdr
import math
import os
import numpy as np

from six import BytesIO
from werkzeug import wrappers
from google.protobuf import json_format
from google.protobuf import text_format
from tensorflow.python.client import session
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import image_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader
from tensorflow.python.summary import plugin_asset
from tensorflow.python.training.saver import checkpoint_exists
from tensorflow.python.training.saver import latest_checkpoint
from tensorflow.tensorboard.backend.http_util import Respond
from tensorflow.tensorboard.plugins.base_plugin import TBPlugin
from tensorflow.tensorboard.plugins.projector import projector_config_pb2

# The prefix of routes provided by this plugin.
_PLUGIN_PREFIX_ROUTE = 'projector'

PROJECTOR_FILENAME = 'projector_config.pbtxt'
_PLUGIN_NAME = 'org_tensorflow_tensorboard_projector'
_PLUGINS_DIR = 'plugins'

# Number of tensors in the LRU cache.
_TENSOR_CACHE_CAPACITY = 1

# HTTP routes.
CONFIG_ROUTE = '/info'
TENSOR_ROUTE = '/tensor'
METADATA_ROUTE = '/metadata'
RUNS_ROUTE = '/runs'
BOOKMARKS_ROUTE = '/bookmarks'
SPRITE_IMAGE_ROUTE = '/sprite_image'

_IMGHDR_TO_MIMETYPE = {
    'bmp': 'image/bmp',
    'gif': 'image/gif',
    'jpeg': 'image/jpeg',
    'png': 'image/png'
}
_DEFAULT_IMAGE_MIMETYPE = 'application/octet-stream'


class LRUCache(object):
  """LRU cache. Used for storing the last used tensor."""

  def __init__(self, size):
    if size < 1:
      raise ValueError('The cache size must be >=1')
    self._size = size
    self._dict = collections.OrderedDict()

  def get(self, key):
    try:
      value = self._dict.pop(key)
      self._dict[key] = value
      return value
    except KeyError:
      return None

  def set(self, key, value):
    if value is None:
      raise ValueError('value must be != None')
    try:
      self._dict.pop(key)
    except KeyError:
      if len(self._dict) >= self._size:
        self._dict.popitem(last=False)
    self._dict[key] = value


class EmbeddingMetadata(object):
  """Metadata container for an embedding.

  The metadata holds different columns with values used for visualization
  (color by, label by) in the "Embeddings" tab in TensorBoard.
  """

  def __init__(self, num_points):
    """Constructs a metadata for an embedding of the specified size.

    Args:
      num_points: Number of points in the embedding.
    """
    self.num_points = num_points
    self.column_names = []
    self.name_to_values = {}

  def add_column(self, column_name, column_values):
    """Adds a named column of metadata values.

    Args:
      column_name: Name of the column.
      column_values: 1D array/list/iterable holding the column values. Must be
          of length `num_points`. The i-th value corresponds to the i-th point.

    Raises:
      ValueError: If `column_values` is not 1D array, or of length `num_points`,
          or the `name` is already used.
    """
    # Sanity checks.
    if isinstance(column_values, list) and isinstance(column_values[0], list):
      raise ValueError('"column_values" must be a flat list, but we detected '
                       'that its first entry is a list')

    if isinstance(column_values, np.ndarray) and column_values.ndim != 1:
      raise ValueError('"column_values" should be of rank 1, '
                       'but is of rank %d' % column_values.ndim)
    if len(column_values) != self.num_points:
      raise ValueError('"column_values" should be of length %d, but is of '
                       'length %d' % (self.num_points, len(column_values)))
    if column_name in self.name_to_values:
      raise ValueError('The column name "%s" is already used' % column_name)

    self.column_names.append(column_name)
    self.name_to_values[column_name] = column_values


class ProjectorPluginAsset(plugin_asset.PluginAsset):
  """Provides a registry for assets needed by the Projector plugin."""
  plugin_name = _PLUGIN_NAME

  def __init__(self):
    self._config = projector_config_pb2.ProjectorConfig()
    self._assets = {}
    self._used_names = set()

  def add_metadata_for_embedding_variable(self,
                                          var_name,
                                          metadata=None,
                                          thumbnails=None,
                                          thumbnail_dim=None):
    """Adds metadata for an embedding variable stored in a checkpoint file.

    Args:
      var_name: Name of the embedding variable.
      metadata: Optional. A `Metadata` container mapping column header names to
          the values of that column.
      thumbnails: Optional. A 4D `ndarray` or a list of 3D `ndarray`s. Each
          3D array represents the pixels [height, width, channels] of a single
          thumbnail. The i-th image corresponds to the i-th row (data point) of
          the embedding variable.
      thumbnail_dim: Required if `thumbnails` is provided. A tuple
          (height, width) of a single thumbnail in the sprite.

    Raises:
      ValueError: If the name of the variable was previously used in this
          object, or both `metadata` and `thumbnails` are None.
    """

    if metadata is None and thumbnails is None:
      raise ValueError('At least one of (`metadata`, `thumbnails`) must be '
                       'provided')
    self._convert_embedding_to_assets(var_name, None, metadata, thumbnails,
                                      thumbnail_dim)

  def add_embedding(self,
                    name,
                    values,
                    metadata=None,
                    thumbnails=None,
                    thumbnail_dim=None):
    """Adds an embedding asset to be visualized by the Embedding Projector.

    Args:
      name: Name of the embedding.
      values: 2D `ndarray` of shape [numPoints, dimensionality]
          containing the embedding values. The i-th row corresponds to the i-th
          data point.
      metadata: Optional. A `Metadata` container mapping column header names to
          the values of that column.
      thumbnails: Optional. A 4D `ndarray` or a list of 3D `ndarray`s. Each
          3D array represents the pixels [height, width, channels] of a single
          thumbnail. The i-th image corresponds to the i-th row (data point) of
          the `values` matrix.
      thumbnail_dim: Required if `thumbnails` is provided. A tuple
          (height, width) of a single thumbnail in the sprite.

    Raises:
      ValueError: If the name of the embedding was previously used in this
          object, or `values` is not a 2D array.
    """

    # Sanity checks.
    if values.ndim != 2:
      raise ValueError('`values` must be a 2D array, but is '
                       '%d-D' % values.ndim)
    self._convert_embedding_to_assets(name, values, metadata, thumbnails,
                                      thumbnail_dim)

  def _convert_embedding_to_assets(self,
                                   name,
                                   values=None,
                                   metadata=None,
                                   thumbnails=None,
                                   thumbnail_dim=None):
    """Converts the data associated with embeddings into serializable assets."""

    if name in self._used_names:
      raise ValueError('The name "%s" was previously used' % name)
    if thumbnails is not None and not thumbnail_dim:
      raise ValueError('`thumbnail_dim` is required when `thumbnails` is '
                       'provided')
    if thumbnail_dim is not None:
      if not isinstance(thumbnail_dim, (list, tuple, np.ndarray)):
        raise ValueError('`thumbnail_dim` must be either a list, tuple or '
                         '`ndarray`')
      if len(thumbnail_dim) != 2:
        raise ValueError('`thumbnail_dim` must be of length 2, '
                         'but is of length %d' % len(thumbnail_dim))
    if metadata:
      if values is not None and len(values) != metadata.num_points:
        raise ValueError('First dimension of `values` "%d" must match '
                         '`metadata.num_points` "%d"' % (len(values),
                                                         metadata.num_points))
      if not metadata.column_names:
        raise ValueError('The provided metadata has no columns. Did you forget '
                         'to add a column?')

    self._used_names.add(name)
    embedding_info = self._config.embeddings.add()
    embedding_info.tensor_name = name

    if values is not None:
      bytes_io = BytesIO()
      np.savetxt(bytes_io, values, fmt='%.6g', delimiter='\t')
      fname = '{}_values.tsv'.format(name)
      embedding_info.tensor_path = fname
      embedding_info.tensor_shape.extend(values.shape)
      self._assets[fname] = bytes_io.getvalue()

    if metadata:
      metadata_tsv_lines = []
      should_have_header = len(metadata.column_names) > 1
      if should_have_header:
        metadata_tsv_lines.append('\t'.join(metadata.column_names))

      for i in range(metadata.num_points):
        row = [
            metadata.name_to_values[col_name][i]
            for col_name in metadata.column_names
        ]
        metadata_tsv_lines.append('\t'.join(map(str, row)))
      fname = '{}_metadata.tsv'.format(name)
      embedding_info.metadata_path = fname
      self._assets[fname] = '\n'.join(metadata_tsv_lines) + '\n'

    if thumbnails is not None:
      fname = '{}_sprite.png'.format(name)
      embedding_info.sprite.image_path = fname
      embedding_info.sprite.single_image_dim.extend(thumbnail_dim)
      self._assets[fname] = _make_sprite_image(thumbnails, thumbnail_dim)

  def assets(self):
    self._assets[PROJECTOR_FILENAME] = text_format.MessageToString(self._config)
    return self._assets


def _read_tensor_tsv_file(fpath):
  with file_io.FileIO(fpath, 'r') as f:
    tensor = []
    for line in f:
      if line:
        tensor.append(list(map(float, line.rstrip('\n').split('\t'))))
  return np.array(tensor, dtype='float32')


def _assets_dir_to_logdir(assets_dir):
  sub_path = os.path.sep + _PLUGINS_DIR + os.path.sep
  if sub_path in assets_dir:
    two_parents_up = os.pardir + os.path.sep + os.pardir
    return os.path.abspath(os.path.join(assets_dir, two_parents_up))
  return assets_dir


def _latest_checkpoints_changed(configs, run_path_pairs):
  """Returns true if the latest checkpoint has changed in any of the runs."""
  for run_name, assets_dir in run_path_pairs:
    if run_name not in configs:
      config = projector_config_pb2.ProjectorConfig()
      config_fpath = os.path.join(assets_dir, PROJECTOR_FILENAME)
      if file_io.file_exists(config_fpath):
        file_content = file_io.read_file_to_string(config_fpath)
        text_format.Merge(file_content, config)
    else:
      config = configs[run_name]

    # See if you can find a checkpoint file in the logdir.
    logdir = _assets_dir_to_logdir(assets_dir)
    ckpt_path = _find_latest_checkpoint(logdir)
    if not ckpt_path:
      continue
    if config.model_checkpoint_path != ckpt_path:
      return True
  return False


def _parse_positive_int_param(request, param_name):
  """Parses and asserts a positive (>0) integer query parameter.

  Args:
    request: The Werkzeug Request object
    param_name: Name of the parameter.

  Returns:
    Param, or None, or -1 if parameter is not a positive integer.
  """
  param = request.args.get(param_name)
  if not param:
    return None
  try:
    param = int(param)
    if param <= 0:
      raise ValueError()
    return param
  except ValueError:
    return -1


def _rel_to_abs_asset_path(fpath, config_fpath):
  fpath = os.path.expanduser(fpath)
  if not os.path.isabs(fpath):
    return os.path.join(os.path.dirname(config_fpath), fpath)
  return fpath


class ProjectorPlugin(TBPlugin):
  """Embedding projector."""

  plugin_name = _PLUGIN_PREFIX_ROUTE

  def __init__(self):
    self._handlers = None
    self.readers = {}
    self.run_paths = None
    self.logdir = None
    self._configs = None
    self.old_num_run_paths = None
    self.multiplexer = None
    self.tensor_cache = LRUCache(_TENSOR_CACHE_CAPACITY)

  def get_plugin_apps(self, multiplexer, logdir):
    self.multiplexer = multiplexer
    self.run_paths = multiplexer.RunPaths()
    self.logdir = logdir
    self._handlers = {
        RUNS_ROUTE: self._serve_runs,
        CONFIG_ROUTE: self._serve_config,
        TENSOR_ROUTE: self._serve_tensor,
        METADATA_ROUTE: self._serve_metadata,
        BOOKMARKS_ROUTE: self._serve_bookmarks,
        SPRITE_IMAGE_ROUTE: self._serve_sprite_image
    }
    return self._handlers

  def is_active(self):
    """Determines whether this plugin is active.

    This plugin is only active if any run has an embedding.

    Returns:
      A boolean. Whether this plugin is active.
    """
    return bool(self.configs)

  @property
  def configs(self):
    """Returns a map of run paths to `ProjectorConfig` protos."""
    run_path_pairs = list(self.run_paths.items())
    self._append_plugin_asset_directories(run_path_pairs)
    # If there are no summary event files, the projector should still work,
    # treating the `logdir` as the model checkpoint directory.
    if not run_path_pairs:
      run_path_pairs.append(('.', self.logdir))
    if (self._run_paths_changed() or
        _latest_checkpoints_changed(self._configs, run_path_pairs)):
      self.readers = {}
      self._configs, self.config_fpaths = self._read_latest_config_files(
          run_path_pairs)
      self._augment_configs_with_checkpoint_info()
    return self._configs

  def _run_paths_changed(self):
    num_run_paths = len(list(self.run_paths.keys()))
    if num_run_paths != self.old_num_run_paths:
      self.old_num_run_paths = num_run_paths
      return True
    return False

  def _augment_configs_with_checkpoint_info(self):
    for run, config in self._configs.items():
      for embedding in config.embeddings:
        # Normalize the name of the embeddings.
        if embedding.tensor_name.endswith(':0'):
          embedding.tensor_name = embedding.tensor_name[:-2]
        # Find the size of embeddings associated with a tensors file.
        if embedding.tensor_path and not embedding.tensor_shape:
          fpath = _rel_to_abs_asset_path(embedding.tensor_path,
                                         self.config_fpaths[run])
          tensor = self.tensor_cache.get(embedding.tensor_name)
          if tensor is None:
            tensor = _read_tensor_tsv_file(fpath)
            self.tensor_cache.set(embedding.tensor_name, tensor)
          embedding.tensor_shape.extend([len(tensor), len(tensor[0])])

      reader = self._get_reader_for_run(run)
      if not reader:
        continue
      # Augment the configuration with the tensors in the checkpoint file.
      special_embedding = None
      if config.embeddings and not config.embeddings[0].tensor_name:
        special_embedding = config.embeddings[0]
        config.embeddings.remove(special_embedding)
      var_map = reader.get_variable_to_shape_map()
      for tensor_name, tensor_shape in var_map.items():
        if len(tensor_shape) != 2:
          continue
        embedding = self._get_embedding(tensor_name, config)
        if not embedding:
          embedding = config.embeddings.add()
          embedding.tensor_name = tensor_name
          if special_embedding:
            embedding.metadata_path = special_embedding.metadata_path
            embedding.bookmarks_path = special_embedding.bookmarks_path
        if not embedding.tensor_shape:
          embedding.tensor_shape.extend(tensor_shape)

    # Remove configs that do not have any valid (2D) tensors.
    runs_to_remove = []
    for run, config in self._configs.items():
      if not config.embeddings:
        runs_to_remove.append(run)
    for run in runs_to_remove:
      del self._configs[run]
      del self.config_fpaths[run]

  def _read_latest_config_files(self, run_path_pairs):
    """Reads and returns the projector config files in every run directory."""
    configs = {}
    config_fpaths = {}
    for run_name, assets_dir in run_path_pairs:
      config = projector_config_pb2.ProjectorConfig()
      config_fpath = os.path.join(assets_dir, PROJECTOR_FILENAME)
      if file_io.file_exists(config_fpath):
        file_content = file_io.read_file_to_string(config_fpath)
        text_format.Merge(file_content, config)
      has_tensor_files = False
      for embedding in config.embeddings:
        if embedding.tensor_path:
          if not embedding.tensor_name:
            embedding.tensor_name = os.path.basename(embedding.tensor_path)
          has_tensor_files = True
          break

      if not config.model_checkpoint_path:
        # See if you can find a checkpoint file in the logdir.
        logdir = _assets_dir_to_logdir(assets_dir)
        ckpt_path = _find_latest_checkpoint(logdir)
        if not ckpt_path and not has_tensor_files:
          continue
        if ckpt_path:
          config.model_checkpoint_path = ckpt_path

      # Sanity check for the checkpoint file.
      if (config.model_checkpoint_path and
          not checkpoint_exists(config.model_checkpoint_path)):
        logging.warning('Checkpoint file "%s" not found',
                        config.model_checkpoint_path)
        continue
      configs[run_name] = config
      config_fpaths[run_name] = config_fpath
    return configs, config_fpaths

  def _get_reader_for_run(self, run):
    if run in self.readers:
      return self.readers[run]

    config = self._configs[run]
    reader = None
    if config.model_checkpoint_path:
      try:
        reader = NewCheckpointReader(config.model_checkpoint_path)
      except Exception:  # pylint: disable=broad-except
        logging.warning('Failed reading "%s"', config.model_checkpoint_path)
    self.readers[run] = reader
    return reader

  def _get_metadata_file_for_tensor(self, tensor_name, config):
    embedding_info = self._get_embedding(tensor_name, config)
    if embedding_info:
      return embedding_info.metadata_path
    return None

  def _get_bookmarks_file_for_tensor(self, tensor_name, config):
    embedding_info = self._get_embedding(tensor_name, config)
    if embedding_info:
      return embedding_info.bookmarks_path
    return None

  def _canonical_tensor_name(self, tensor_name):
    if ':' not in tensor_name:
      return tensor_name + ':0'
    else:
      return tensor_name

  def _get_embedding(self, tensor_name, config):
    if not config.embeddings:
      return None
    for info in config.embeddings:
      if (self._canonical_tensor_name(info.tensor_name) ==
          self._canonical_tensor_name(tensor_name)):
        return info
    return None

  def _append_plugin_asset_directories(self, run_path_pairs):
    for run, assets in self.multiplexer.PluginAssets(_PLUGIN_NAME).items():
      if PROJECTOR_FILENAME not in assets:
        continue
      assets_dir = os.path.join(self.run_paths[run], _PLUGINS_DIR, _PLUGIN_NAME)
      assets_path_pair = (run, os.path.abspath(assets_dir))
      run_path_pairs.append(assets_path_pair)

  @wrappers.Request.application
  def _serve_runs(self, request):
    """Returns a list of runs that have embeddings."""
    return Respond(request, list(self.configs.keys()), 'application/json')

  @wrappers.Request.application
  def _serve_config(self, request):
    run = request.args.get('run')
    if run is None:
      return Respond(request, 'query parameter "run" is required', 'text/plain',
                     400)
    if run not in self.configs:
      return Respond(request, 'Unknown run: "%s"' % run, 'text/plain', 400)

    config = self.configs[run]
    return Respond(request,
                   json_format.MessageToJson(config), 'application/json')

  @wrappers.Request.application
  def _serve_metadata(self, request):
    run = request.args.get('run')
    if run is None:
      return Respond(request, 'query parameter "run" is required', 'text/plain',
                     400)

    name = request.args.get('name')
    if name is None:
      return Respond(request, 'query parameter "name" is required',
                     'text/plain', 400)

    num_rows = _parse_positive_int_param(request, 'num_rows')
    if num_rows == -1:
      return Respond(request, 'query parameter num_rows must be integer > 0',
                     'text/plain', 400)

    if run not in self.configs:
      return Respond(request, 'Unknown run: "%s"' % run, 'text/plain', 400)

    config = self.configs[run]
    fpath = self._get_metadata_file_for_tensor(name, config)
    if not fpath:
      return Respond(
          request,
          'No metadata file found for tensor "%s" in the config file "%s"' %
          (name, self.config_fpaths[run]), 'text/plain', 400)
    fpath = _rel_to_abs_asset_path(fpath, self.config_fpaths[run])
    if not file_io.file_exists(fpath) or file_io.is_directory(fpath):
      return Respond(request, '"%s" not found, or is not a file' % fpath,
                     'text/plain', 400)

    num_header_rows = 0
    with file_io.FileIO(fpath, 'r') as f:
      lines = []
      # Stream reading the file with early break in case the file doesn't fit in
      # memory.
      for line in f:
        lines.append(line)
        if len(lines) == 1 and '\t' in lines[0]:
          num_header_rows = 1
        if num_rows and len(lines) >= num_rows + num_header_rows:
          break
    return Respond(request, ''.join(lines), 'text/plain')

  @wrappers.Request.application
  def _serve_tensor(self, request):
    run = request.args.get('run')
    if run is None:
      return Respond(request, 'query parameter "run" is required', 'text/plain',
                     400)

    name = request.args.get('name')
    if name is None:
      return Respond(request, 'query parameter "name" is required',
                     'text/plain', 400)

    num_rows = _parse_positive_int_param(request, 'num_rows')
    if num_rows == -1:
      return Respond(request, 'query parameter num_rows must be integer > 0',
                     'text/plain', 400)

    if run not in self.configs:
      return Respond(request, 'Unknown run: "%s"' % run, 'text/plain', 400)

    config = self.configs[run]

    tensor = self.tensor_cache.get(name)
    if tensor is None:
      # See if there is a tensor file in the config.
      embedding = self._get_embedding(name, config)

      if embedding and embedding.tensor_path:
        fpath = _rel_to_abs_asset_path(embedding.tensor_path,
                                       self.config_fpaths[run])
        if not file_io.file_exists(fpath):
          return Respond(request,
                         'Tensor file "%s" does not exist' % fpath,
                         'text/plain', 400)
        tensor = _read_tensor_tsv_file(fpath)
      else:
        reader = self._get_reader_for_run(run)
        if not reader or not reader.has_tensor(name):
          return Respond(request,
                         'Tensor "%s" not found in checkpoint dir "%s"' %
                         (name, config.model_checkpoint_path), 'text/plain',
                         400)
        try:
          tensor = reader.get_tensor(name)
        except errors.InvalidArgumentError as e:
          return Respond(request, str(e), 'text/plain', 400)

      self.tensor_cache.set(name, tensor)

    if num_rows:
      tensor = tensor[:num_rows]
    if tensor.dtype != 'float32':
      tensor = tensor.astype(dtype='float32', copy=False)
    data_bytes = tensor.tobytes()
    return Respond(request, data_bytes, 'application/octet-stream')

  @wrappers.Request.application
  def _serve_bookmarks(self, request):
    run = request.args.get('run')
    if not run:
      return Respond(request, 'query parameter "run" is required', 'text/plain',
                     400)

    name = request.args.get('name')
    if name is None:
      return Respond(request, 'query parameter "name" is required',
                     'text/plain', 400)

    if run not in self.configs:
      return Respond(request, 'Unknown run: "%s"' % run, 'text/plain', 400)

    config = self.configs[run]
    fpath = self._get_bookmarks_file_for_tensor(name, config)
    if not fpath:
      return Respond(
          request,
          'No bookmarks file found for tensor "%s" in the config file "%s"' %
          (name, self.config_fpaths[run]), 'text/plain', 400)
    fpath = _rel_to_abs_asset_path(fpath, self.config_fpaths[run])
    if not file_io.file_exists(fpath) or file_io.is_directory(fpath):
      return Respond(request, '"%s" not found, or is not a file' % fpath,
                     'text/plain', 400)

    bookmarks_json = None
    with file_io.FileIO(fpath, 'rb') as f:
      bookmarks_json = f.read()
    return Respond(request, bookmarks_json, 'application/json')

  @wrappers.Request.application
  def _serve_sprite_image(self, request):
    run = request.args.get('run')
    if not run:
      return Respond(request, 'query parameter "run" is required', 'text/plain',
                     400)

    name = request.args.get('name')
    if name is None:
      return Respond(request, 'query parameter "name" is required',
                     'text/plain', 400)

    if run not in self.configs:
      return Respond(request, 'Unknown run: "%s"' % run, 'text/plain', 400)

    config = self.configs[run]
    embedding_info = self._get_embedding(name, config)

    if not embedding_info or not embedding_info.sprite.image_path:
      return Respond(
          request,
          'No sprite image file found for tensor "%s" in the config file "%s"' %
          (name, self.config_fpaths[run]), 'text/plain', 400)

    fpath = os.path.expanduser(embedding_info.sprite.image_path)
    fpath = _rel_to_abs_asset_path(fpath, self.config_fpaths[run])
    if not file_io.file_exists(fpath) or file_io.is_directory(fpath):
      return Respond(request, '"%s" does not exist or is directory' % fpath,
                     'text/plain', 400)
    f = file_io.FileIO(fpath, 'rb')
    encoded_image_string = f.read()
    f.close()
    image_type = imghdr.what(None, encoded_image_string)
    mime_type = _IMGHDR_TO_MIMETYPE.get(image_type, _DEFAULT_IMAGE_MIMETYPE)
    return Respond(request, encoded_image_string, mime_type)


def _find_latest_checkpoint(dir_path):
  try:
    ckpt_path = latest_checkpoint(dir_path)
    if not ckpt_path:
      # Check the parent directory.
      ckpt_path = latest_checkpoint(os.path.join(dir_path, os.pardir))
    return ckpt_path
  except errors.NotFoundError:
    return None


def _make_sprite_image(thumbnails, thumbnail_dim):
  """Constructs a sprite image from thumbnails and returns the png bytes."""
  if len(thumbnails) < 1:
    raise ValueError('The length of "thumbnails" must be >= 1')

  if isinstance(thumbnails, np.ndarray) and thumbnails.ndim != 4:
    raise ValueError('"thumbnails" should be of rank 4, '
                     'but is of rank %d' % thumbnails.ndim)
  if isinstance(thumbnails, list):
    if not isinstance(thumbnails[0], np.ndarray) or thumbnails[0].ndim != 3:
      raise ValueError('Each element of "thumbnails" must be a 3D `ndarray`')
    thumbnails = np.array(thumbnails)

  with ops.Graph().as_default():
    s = session.Session()
    resized_images = image_ops.resize_images(thumbnails, thumbnail_dim).eval(
        session=s)
    images_per_row = int(math.ceil(math.sqrt(len(thumbnails))))
    thumb_height = thumbnail_dim[0]
    thumb_width = thumbnail_dim[1]
    master_height = images_per_row * thumb_height
    master_width = images_per_row * thumb_width
    num_channels = thumbnails.shape[3]
    master = np.zeros([master_height, master_width, num_channels])
    for idx, image in enumerate(resized_images):
      left_idx = idx % images_per_row
      top_idx = int(math.floor(idx / images_per_row))
      left_start = left_idx * thumb_width
      left_end = left_start + thumb_width
      top_start = top_idx * thumb_height
      top_end = top_start + thumb_height
      master[top_start:top_end, left_start:left_end, :] = image

    return image_ops.encode_png(master).eval(session=s)
