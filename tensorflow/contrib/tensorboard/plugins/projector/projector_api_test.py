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
import tensorflow as tf

from google.protobuf import text_format


class ProjectorApiTest(tf.test.TestCase):

  def testVisualizeEmbeddings(self):
    # Create a dummy configuration.
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    config.model_checkpoint_path = 'test'
    emb1 = config.embedding.add()
    emb1.tensor_name = 'tensor1'
    emb1.metadata_path = 'metadata1'

    # Call the API method to save the configuration to a temporary dir.
    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir)
    writer = tf.train.SummaryWriter(temp_dir)
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer,
                                                                  config)

    # Read the configuratin from disk and make sure it matches the original.
    with tf.gfile.GFile(os.path.join(temp_dir, 'projector_config.pbtxt')) as f:
      config2 = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
      text_format.Parse(f.read(), config2)
      self.assertEqual(config, config2)
