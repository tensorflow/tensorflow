# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Integration tests for the Embedding Projector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import io
import json
import os
import numpy as np

from werkzeug import test as werkzeug_test
from werkzeug import wrappers

from google.protobuf import text_format
from tensorflow.contrib.tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib
from tensorflow.tensorboard.backend import application
from tensorflow.tensorboard.backend.event_processing import event_multiplexer
from tensorflow.tensorboard.plugins.projector import projector_plugin


class ProjectorAppTest(test.TestCase):

  def setUp(self):
    self.log_dir = self.get_temp_dir()

  def testRunsWithValidCheckpoint(self):
    self._GenerateProjectorTestData()
    self._SetupWSGIApp()
    run_json = self._GetJson('/data/plugin/projector/runs')
    self.assertEqual(run_json, ['.'])

  def testRunsWithNoCheckpoint(self):
    self._SetupWSGIApp()
    run_json = self._GetJson('/data/plugin/projector/runs')
    self.assertEqual(run_json, [])

  def testRunsWithInvalidModelCheckpointPath(self):
    checkpoint_file = os.path.join(self.log_dir, 'checkpoint')
    f = open(checkpoint_file, 'w')
    f.write('model_checkpoint_path: "does_not_exist"\n')
    f.write('all_model_checkpoint_paths: "does_not_exist"\n')
    f.close()
    self._SetupWSGIApp()

    run_json = self._GetJson('/data/plugin/projector/runs')
    self.assertEqual(run_json, [])

  def testInfoWithValidCheckpoint(self):
    self._GenerateProjectorTestData()
    self._SetupWSGIApp()

    info_json = self._GetJson('/data/plugin/projector/info?run=.')
    self.assertItemsEqual(info_json['embeddings'], [{
        'tensorShape': [1, 2],
        'tensorName': 'var1'
    }, {
        'tensorShape': [10, 10],
        'tensorName': 'var2'
    }, {
        'tensorShape': [100, 100],
        'tensorName': 'var3'
    }])

  def testTensorWithValidCheckpoint(self):
    self._GenerateProjectorTestData()
    self._SetupWSGIApp()

    url = '/data/plugin/projector/tensor?run=.&name=var1'
    tensor_bytes = self._Get(url).data
    tensor = np.reshape(np.fromstring(tensor_bytes, dtype='float32'), [1, 2])
    expected_tensor = np.array([[6, 6]], dtype='float32')
    self.assertTrue(np.array_equal(tensor, expected_tensor))

  def _SetupWSGIApp(self):
    multiplexer = event_multiplexer.EventMultiplexer(
        size_guidance=application.DEFAULT_SIZE_GUIDANCE,
        purge_orphaned_data=True)
    projector = projector_plugin.ProjectorPlugin()
    projector.get_plugin_apps(multiplexer, self.log_dir)
    plugins = {'projector': projector}
    wsgi_app = application.TensorBoardWSGIApp(
        self.log_dir, plugins, multiplexer, reload_interval=0)
    self.server = werkzeug_test.Client(wsgi_app, wrappers.BaseResponse)

  def _Get(self, path):
    return self.server.get(path)

  def _GetJson(self, path):
    response = self.server.get(path)
    data = response.data
    if response.headers.get('Content-Encoding') == 'gzip':
      data = gzip.GzipFile('', 'rb', 9, io.BytesIO(data)).read()
    return json.loads(data.decode('utf-8'))

  def _GenerateProjectorTestData(self):
    config_path = os.path.join(self.log_dir, 'projector_config.pbtxt')
    config = ProjectorConfig()
    embedding = config.embeddings.add()
    # Add an embedding by its canonical tensor name.
    embedding.tensor_name = 'var1:0'
    config_pbtxt = text_format.MessageToString(config)
    with gfile.GFile(config_path, 'w') as f:
      f.write(config_pbtxt)

    # Write a checkpoint with some dummy variables.
    with ops.Graph().as_default():
      sess = session.Session()
      checkpoint_path = os.path.join(self.log_dir, 'model')
      variable_scope.get_variable(
          'var1', [1, 2], initializer=init_ops.constant_initializer(6.0))
      variable_scope.get_variable('var2', [10, 10])
      variable_scope.get_variable('var3', [100, 100])
      sess.run(variables.global_variables_initializer())
      saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V1)
      saver.save(sess, checkpoint_path)


if __name__ == '__main__':
  test.main()
