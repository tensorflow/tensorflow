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
"""API tests for the projector plugin in TensorBoard."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

from google.protobuf import text_format

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib.tensorboard.plugins.projector import projector_config_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary.writer import writer as writer_lib


class ProjectorApiTest(test.TestCase):

  def testVisualizeEmbeddings(self):
    # Create a dummy configuration.
    config = projector_config_pb2.ProjectorConfig()
    config.model_checkpoint_path = 'test'
    emb1 = config.embeddings.add()
    emb1.tensor_name = 'tensor1'
    emb1.metadata_path = 'metadata1'

    # Call the API method to save the configuration to a temporary dir.
    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir)
    writer = writer_lib.FileWriter(temp_dir)
    projector.visualize_embeddings(writer, config)

    # Read the configuratin from disk and make sure it matches the original.
    with gfile.GFile(os.path.join(temp_dir, 'projector_config.pbtxt')) as f:
      config2 = projector_config_pb2.ProjectorConfig()
      text_format.Parse(f.read(), config2)
      self.assertEqual(config, config2)


if __name__ == '__main__':
  test.main()
