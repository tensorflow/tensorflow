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
"""Public API for the Embedding Projector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from google.protobuf import text_format
from tensorflow.contrib.tensorboard.plugins.projector.projector_config_pb2 import EmbeddingInfo
from tensorflow.contrib.tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
from tensorflow.python.lib.io import file_io

PROJECTOR_FILENAME = 'projector_config.pbtxt'


def visualize_embeddings(summary_writer, config):
  """Stores a config file used by the embedding projector.

  Args:
    summary_writer: The summary writer used for writting events.
    config: `tf.contrib.tensorboard.plugins.projector.ProjectorConfig`
      proto that holds the configuration for the projector such as paths to
      checkpoint files and metadata files for the embeddings. If
      `config.model_checkpoint_path` is none, it defaults to the
      `logdir` used by the summary_writer.

  Raises:
    ValueError: If the summary writer does not have a `logdir`.
  """
  logdir = summary_writer.get_logdir()

  # Sanity checks.
  if logdir is None:
    raise ValueError('Summary writer must have a logdir')

  # Saving the config file in the logdir.
  config_pbtxt = text_format.MessageToString(config)
  file_io.write_string_to_file(
      os.path.join(logdir, PROJECTOR_FILENAME), config_pbtxt)
