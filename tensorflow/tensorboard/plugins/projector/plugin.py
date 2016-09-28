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

import os

import six
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

# Limit for the number of points we send to the browser.
LIMIT_NUM_POINTS = 50000


class ProjectorPlugin(TBPlugin):
  """Embedding projector."""

  def get_plugin_handlers(self, run_paths):
    self.configs, self.config_fpaths = self._read_config_files(run_paths)
    self.readers = {}

    return {
        RUNS_ROUTE: self._serve_runs,
        INFO_ROUTE: self._serve_info,
        TENSOR_ROUTE: self._serve_tensor,
        METADATA_ROUTE: self._serve_metadata,
    }

  def _read_config_files(self, run_paths):
    configs = {}
    config_fpaths = {}
    for run_name, logdir in six.iteritems(run_paths):
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
    if not config.embedding:
      return None
    for info in config.embedding:
      if info.tensor_name == tensor_name:
        return info.metadata_path
    return None

  def _serve_runs(self, query_params):
    """Returns a list of runs that have embeddings."""
    self.handler.send_json_response(list(self.configs.keys()))

  def _serve_info(self, query_params):
    run = query_params.get('run')
    if run is None:
      self.handler.send_error(400, 'query parameter "run" is required')
      return
    if run not in self.configs:
      self.handler.send_error(400, 'Unknown run: %s' % run)
      return

    config = self.configs[run]
    reader = self._get_reader_for_run(run)
    var_map = reader.get_variable_to_shape_map()
    embedding_map = {name: {
        'name': name,
        'shape': shape,
        'metadataFile': self._get_metadata_file_for_tensor(name, config)
    }
                     for name, shape in six.iteritems(var_map)
                     if len(shape) == 2}
    self.handler.send_json_response({
        'tensors': embedding_map,
        'checkpointFile': config.model_checkpoint_path,
    })

  def _serve_metadata(self, query_params):
    run = query_params.get('run')
    if run is None:
      self.handler.send_error(400, 'query parameter "run" is required')
      return

    name = query_params.get('name')
    if name is None:
      self.handler.send_error(400, 'query parameter "name" is required')
      return
    if run not in self.configs:
      self.handler.send_error(400, 'Unknown run: %s' % run)
      return

    config = self.configs[run]
    fpath = self._get_metadata_file_for_tensor(name, config)
    if not fpath:
      self.handler.send_error(
          400, 'Not metadata file found for tensor %s in the config file %s' %
          (name, self.config_fpaths[run]))
      return
    if not file_io.file_exists(fpath) or file_io.is_directory(fpath):
      self.handler.send_error(400, '%s is not a file' % fpath)
      return

    with file_io.FileIO(fpath, 'r') as f:
      lines = []
      for line in f:
        lines.append(line)
        if len(lines) >= LIMIT_NUM_POINTS:
          break
    self.handler.send_gzip_response(''.join(lines), 'text/plain')

  def _serve_tensor(self, query_params):
    run = query_params.get('run')
    if run is None:
      self.handler.send_error(400, 'query parameter "run" is required')
      return

    name = query_params.get('name')
    if name is None:
      self.handler.send_error(400, 'query parameter "name" is required')
      return

    if run not in self.configs:
      self.handler.send_error(400, 'Unknown run: %s' % run)
      return

    reader = self._get_reader_for_run(run)
    config = self.configs[run]
    if not reader.has_tensor(name):
      self.handler.send_error(400, 'Tensor %s not found in checkpoint dir %s' %
                              (name, config.model_checkpoint_path))
      return
    tensor = reader.get_tensor(name)
    # Sample the tensor
    tensor = tensor[:LIMIT_NUM_POINTS]
    # Stream it as TSV.
    tsv = '\n'.join(['\t'.join([str(val) for val in row]) for row in tensor])
    self.handler.send_gzip_response(tsv, 'text/plain')
