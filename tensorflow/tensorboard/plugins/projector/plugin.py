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

import imghdr
import os

from google.protobuf import json_format
from google.protobuf import text_format
from tensorflow.contrib.tensorboard.plugins.projector import PROJECTOR_FILENAME
from tensorflow.contrib.tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader
from tensorflow.python.training.saver import latest_checkpoint
from tensorflow.tensorboard.plugins.base_plugin import TBPlugin

# HTTP routes.
INFO_ROUTE = '/info'
TENSOR_ROUTE = '/tensor'
METADATA_ROUTE = '/metadata'
RUNS_ROUTE = '/runs'
BOOKMARKS_ROUTE = '/bookmarks'
SPRITE_IMAGE_ROUTE = '/sprite_image'

# Limit for the number of points we send to the browser.
LIMIT_NUM_POINTS = 100000

_IMGHDR_TO_MIMETYPE = {
    'bmp': 'image/bmp',
    'gif': 'image/gif',
    'jpeg': 'image/jpeg',
    'png': 'image/png'
}
_DEFAULT_IMAGE_MIMETYPE = 'application/octet-stream'


class ProjectorPlugin(TBPlugin):
  """Embedding projector."""

  def get_plugin_handlers(self, run_paths, logdir):
    self.configs, self.config_fpaths = self._read_config_files(run_paths,
                                                               logdir)
    self.readers = {}

    return {
        RUNS_ROUTE: self._serve_runs,
        INFO_ROUTE: self._serve_info,
        TENSOR_ROUTE: self._serve_tensor,
        METADATA_ROUTE: self._serve_metadata,
        BOOKMARKS_ROUTE: self._serve_bookmarks,
        SPRITE_IMAGE_ROUTE: self._serve_sprite_image
    }

  def _read_config_files(self, run_paths, logdir):
    # If there are no summary event files, the projector can still work,
    # thus treating the `logdir` as the model checkpoint directory.
    if not run_paths:
      run_paths['.'] = logdir

    configs = {}
    config_fpaths = {}
    for run_name, logdir in run_paths.items():
      config_fpath = os.path.join(logdir, PROJECTOR_FILENAME)
      if not file_io.file_exists(config_fpath):
        # Skip runs that have no config file.
        continue
      # Read the config file.
      file_content = file_io.read_file_to_string(config_fpath).decode('utf-8')
      config = ProjectorConfig()
      text_format.Merge(file_content, config)

      if not config.model_checkpoint_path:
        # See if you can find a checkpoint file in the logdir.
        ckpt_path = latest_checkpoint(logdir)
        if not ckpt_path:
          # Or in the parent of logdir.
          ckpt_path = latest_checkpoint(os.path.join('../', logdir))
          if not ckpt_path:
            logging.warning('Cannot find model checkpoint in %s', logdir)
            continue
        config.model_checkpoint_path = ckpt_path

      # Sanity check for the checkpoint file.
      if not file_io.file_exists(config.model_checkpoint_path):
        logging.warning('Checkpoint file %s not found',
                        config.model_checkpoint_path)
        continue
      configs[run_name] = config
      config_fpaths[run_name] = config_fpath
    return configs, config_fpaths

  def _get_reader_for_run(self, run):
    if run in self.readers:
      return self.readers[run]

    config = self.configs[run]
    reader = NewCheckpointReader(config.model_checkpoint_path)
    self.readers[run] = reader
    return reader

  def _get_metadata_file_for_tensor(self, tensor_name, config):
    embedding_info = self._get_embedding_info_for_tensor(tensor_name, config)
    if embedding_info:
      return embedding_info.metadata_path
    return None

  def _get_bookmarks_file_for_tensor(self, tensor_name, config):
    embedding_info = self._get_embedding_info_for_tensor(tensor_name, config)
    if embedding_info:
      return embedding_info.bookmarks_path
    return None

  def _canonical_tensor_name(self, tensor_name):
    if ':' not in tensor_name:
      return tensor_name + ':0'
    else:
      return tensor_name

  def _get_embedding_info_for_tensor(self, tensor_name, config):
    if not config.embeddings:
      return None
    for info in config.embeddings:
      if (self._canonical_tensor_name(info.tensor_name) ==
          self._canonical_tensor_name(tensor_name)):
        return info
    return None

  def _serve_runs(self, query_params):
    """Returns a list of runs that have embeddings."""
    self.handler.respond(list(self.configs.keys()), 'application/json')

  def _serve_info(self, query_params):
    run = query_params.get('run')
    if run is None:
      self.handler.respond('query parameter "run" is required',
                           'text/plain', 400)
      return
    if run not in self.configs:
      self.handler.respond('Unknown run: %s' % run, 'text/plain', 400)
      return

    config = self.configs[run]
    reader = self._get_reader_for_run(run)
    var_map = reader.get_variable_to_shape_map()

    for tensor_name, tensor_shape in var_map.items():
      if len(tensor_shape) != 2:
        continue
      info = self._get_embedding_info_for_tensor(tensor_name, config)
      if not info:
        info = config.embeddings.add()
        info.tensor_name = tensor_name
      if not info.tensor_shape:
        info.tensor_shape.extend(tensor_shape)

    self.handler.respond(json_format.MessageToDict(config), 'application/json')

  def _serve_metadata(self, query_params):
    run = query_params.get('run')
    if run is None:
      self.handler.respond('query parameter "run" is required',
                           'text/plain', 400)
      return

    name = query_params.get('name')
    if name is None:
      self.handler.respond('query parameter "name" is required',
                           'text/plain', 400)
      return
    if run not in self.configs:
      self.handler.respond('Unknown run: %s' % run, 'text/plain', 400)
      return

    config = self.configs[run]
    fpath = self._get_metadata_file_for_tensor(name, config)
    if not fpath:
      self.handler.respond(
          'Not metadata file found for tensor %s in the config file %s' %
          (name, self.config_fpaths[run]), 'text/plain', 400)
      return
    if not file_io.file_exists(fpath) or file_io.is_directory(fpath):
      self.handler.respond('%s is not a file' % fpath, 'text/plain', 400)
      return

    num_header_rows = 0
    with file_io.FileIO(fpath, 'r') as f:
      lines = []
      # Stream reading the file with early break in case the file doesn't fit in
      # memory.
      for line in f:
        lines.append(line)
        if len(lines) == 1 and '\t' in lines[0]:
          num_header_rows = 1
        if len(lines) >= LIMIT_NUM_POINTS + num_header_rows:
          break
    self.handler.respond(''.join(lines), 'text/plain')

  def _serve_tensor(self, query_params):
    run = query_params.get('run')
    if run is None:
      self.handler.respond('query parameter "run" is required',
                           'text/plain', 400)
      return

    name = query_params.get('name')
    if name is None:
      self.handler.respond('query parameter "name" is required',
                           'text/plain', 400)
      return

    if run not in self.configs:
      self.handler.respond('Unknown run: %s' % run, 'text/plain', 400)
      return

    reader = self._get_reader_for_run(run)
    config = self.configs[run]
    if not reader.has_tensor(name):
      self.handler.respond('Tensor %s not found in checkpoint dir %s' %
                           (name, config.model_checkpoint_path),
                           'text/plain', 400)
      return
    tensor = reader.get_tensor(name)
    # Sample the tensor
    tensor = tensor[:LIMIT_NUM_POINTS]
    # Stream it as TSV.
    tsv = '\n'.join(['\t'.join([str(val) for val in row]) for row in tensor])
    self.handler.respond(tsv, 'text/tab-separated-values')

  def _serve_bookmarks(self, query_params):
    run = query_params.get('run')
    if not run:
      self.handler.respond('query parameter "run" is required', 'text/plain',
                           400)
      return

    name = query_params.get('name')
    if name is None:
      self.handler.respond('query parameter "name" is required', 'text/plain',
                           400)
      return

    if run not in self.configs:
      self.handler.respond('Unknown run: %s' % run, 'text/plain', 400)
      return

    config = self.configs[run]
    fpath = self._get_bookmarks_file_for_tensor(name, config)
    if not fpath:
      self.handler.respond(
          'No bookmarks file found for tensor %s in the config file %s' %
          (name, self.config_fpaths[run]), 'text/plain', 400)
      return
    if not file_io.file_exists(fpath) or file_io.is_directory(fpath):
      self.handler.respond('%s is not a file' % fpath, 'text/plain', 400)
      return

    bookmarks_json = None
    with file_io.FileIO(fpath, 'r') as f:
      bookmarks_json = f.read()
    self.handler.respond(bookmarks_json, 'application/json')

  def _serve_sprite_image(self, query_params):
    run = query_params.get('run')
    if not run:
      self.handler.respond('query parameter "run" is required', 'text/plain',
                           400)
      return

    name = query_params.get('name')
    if name is None:
      self.handler.respond('query parameter "name" is required', 'text/plain',
                           400)
      return

    if run not in self.configs:
      self.handler.respond('Unknown run: %s' % run, 'text/plain', 400)
      return

    config = self.configs[run]
    embedding_info = self._get_embedding_info_for_tensor(name, config)

    if not embedding_info or not embedding_info.sprite.image_path:
      self.handler.respond(
          'No sprite image file found for tensor %s in the config file %s' %
          (name, self.config_fpaths[run]), 'text/plain', 400)
      return

    fpath = embedding_info.sprite.image_path
    if not file_io.file_exists(fpath) or file_io.is_directory(fpath):
      self.handler.respond('%s does not exist or is directory' % fpath,
                           'text/plain', 400)
      return
    f = file_io.FileIO(fpath, 'r')
    encoded_image_string = f.read()
    f.close()
    image_type = imghdr.what(None, encoded_image_string)
    mime_type = _IMGHDR_TO_MIMETYPE.get(image_type, _DEFAULT_IMAGE_MIMETYPE)
    self.handler.respond(encoded_image_string, mime_type)
