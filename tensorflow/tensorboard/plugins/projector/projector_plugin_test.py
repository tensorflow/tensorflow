# -*- coding: utf-8 -*-
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
import tensorflow as tf

from werkzeug import test as werkzeug_test
from werkzeug import wrappers

from google.protobuf import text_format

from tensorflow.tensorboard.backend import application
from tensorflow.tensorboard.backend.event_processing import event_multiplexer
from tensorflow.tensorboard.plugins.projector import projector_config_pb2
from tensorflow.tensorboard.plugins.projector import projector_plugin


class ProjectorAppTest(tf.test.TestCase):

  def setUp(self):
    self.log_dir = self.get_temp_dir()

  def testRunsWithValidCheckpoint(self):
    self._GenerateProjectorTestData()
    self._SetupWSGIApp()
    run_json = self._GetJson('/data/plugin/projector/runs')
    self.assertTrue(run_json)

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

  def testRunsWithInvalidModelCheckpointPathInConfig(self):
    config_path = os.path.join(self.log_dir, 'projector_config.pbtxt')
    config = projector_config_pb2.ProjectorConfig()
    config.model_checkpoint_path = 'does_not_exist'
    embedding = config.embeddings.add()
    embedding.tensor_name = 'var1'
    with tf.gfile.GFile(config_path, 'w') as f:
      f.write(text_format.MessageToString(config))
    self._SetupWSGIApp()

    run_json = self._GetJson('/data/plugin/projector/runs')
    self.assertEqual(run_json, [])

  def testInfoWithValidCheckpointNoEventsData(self):
    self._GenerateProjectorTestData()
    self._SetupWSGIApp()

    info_json = self._GetJson('/data/plugin/projector/info?run=.')
    self.assertItemsEqual(info_json['embeddings'], [{
        'tensorShape': [1, 2],
        'tensorName': 'var1',
        'bookmarksPath': 'bookmarks.json'
    }, {
        'tensorShape': [10, 10],
        'tensorName': 'var2'
    }, {
        'tensorShape': [100, 100],
        'tensorName': 'var3'
    }])

  def testInfoWithValidCheckpointAndEventsData(self):
    self._GenerateProjectorTestData()
    self._GenerateEventsData()
    self._SetupWSGIApp()

    run_json = self._GetJson('/data/plugin/projector/runs')
    self.assertTrue(run_json)
    run = run_json[0]
    info_json = self._GetJson('/data/plugin/projector/info?run=%s' % run)
    self.assertItemsEqual(info_json['embeddings'], [{
        'tensorShape': [1, 2],
        'tensorName': 'var1',
        'bookmarksPath': 'bookmarks.json'
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
    expected_tensor = np.array([[6, 6]], dtype=np.float32)
    self._AssertTensorResponse(tensor_bytes, expected_tensor)

  def testBookmarksRequestMissingRunAndName(self):
    self._GenerateProjectorTestData()
    self._SetupWSGIApp()

    url = '/data/plugin/projector/bookmarks'
    self.assertEqual(self._Get(url).status_code, 400)

  def testBookmarksRequestMissingName(self):
    self._GenerateProjectorTestData()
    self._SetupWSGIApp()

    url = '/data/plugin/projector/bookmarks?run=.'
    self.assertEqual(self._Get(url).status_code, 400)

  def testBookmarksRequestMissingRun(self):
    self._GenerateProjectorTestData()
    self._SetupWSGIApp()

    url = '/data/plugin/projector/bookmarks?name=var1'
    self.assertEqual(self._Get(url).status_code, 400)

  def testBookmarksUnknownRun(self):
    self._GenerateProjectorTestData()
    self._SetupWSGIApp()

    url = '/data/plugin/projector/bookmarks?run=unknown&name=var1'
    self.assertEqual(self._Get(url).status_code, 400)

  def testBookmarksUnknownName(self):
    self._GenerateProjectorTestData()
    self._SetupWSGIApp()

    url = '/data/plugin/projector/bookmarks?run=.&name=unknown'
    self.assertEqual(self._Get(url).status_code, 400)

  def testBookmarks(self):
    self._GenerateProjectorTestData()
    self._SetupWSGIApp()

    url = '/data/plugin/projector/bookmarks?run=.&name=var1'
    bookmark = self._GetJson(url)
    self.assertEqual(bookmark, {'a': 'b'})

  def testEndpointsNoAssets(self):
    g = tf.Graph()

    fw = tf.summary.FileWriter(self.log_dir, graph=g)
    fw.close()

    self._SetupWSGIApp()
    run_json = self._GetJson('/data/plugin/projector/runs')
    self.assertEqual(run_json, [])

  def _AssertTensorResponse(self, tensor_bytes, expected_tensor):
    tensor = np.reshape(np.fromstring(tensor_bytes, dtype=np.float32),
                        expected_tensor.shape)
    self.assertTrue(np.array_equal(tensor, expected_tensor))

  def testPluginIsActive(self):
    self._GenerateProjectorTestData()
    self._SetupWSGIApp()

    # Embedding data is available.
    self.assertTrue(self.plugin.is_active())

  def testPluginIsNotActive(self):
    self._SetupWSGIApp()

    # Embedding data is not available.
    self.assertFalse(self.plugin.is_active())

  def _SetupWSGIApp(self):
    multiplexer = event_multiplexer.EventMultiplexer(
        size_guidance=application.DEFAULT_SIZE_GUIDANCE,
        purge_orphaned_data=True)
    self.plugin = projector_plugin.ProjectorPlugin()
    wsgi_app = application.TensorBoardWSGIApp(
        self.log_dir, [self.plugin], multiplexer, reload_interval=0)
    self.server = werkzeug_test.Client(wsgi_app, wrappers.BaseResponse)

  def _Get(self, path):
    return self.server.get(path)

  def _GetJson(self, path):
    response = self.server.get(path)
    data = response.data
    if response.headers.get('Content-Encoding') == 'gzip':
      data = gzip.GzipFile('', 'rb', 9, io.BytesIO(data)).read()
    return json.loads(data.decode('utf-8'))

  def _GenerateEventsData(self):
    fw = tf.summary.FileWriter(self.log_dir)
    event = tf.Event(
        wall_time=1,
        step=1,
        summary=tf.Summary(value=[tf.Summary.Value(tag='s1', simple_value=0)]))
    fw.add_event(event)
    fw.close()

  def _GenerateProjectorTestData(self):
    config_path = os.path.join(self.log_dir, 'projector_config.pbtxt')
    config = projector_config_pb2.ProjectorConfig()
    embedding = config.embeddings.add()
    # Add an embedding by its canonical tensor name.
    embedding.tensor_name = 'var1:0'

    with tf.gfile.GFile(os.path.join(self.log_dir, 'bookmarks.json'), 'w') as f:
      f.write('{"a": "b"}')
    embedding.bookmarks_path = 'bookmarks.json'

    config_pbtxt = text_format.MessageToString(config)
    with tf.gfile.GFile(config_path, 'w') as f:
      f.write(config_pbtxt)

    # Write a checkpoint with some dummy variables.
    with tf.Graph().as_default():
      sess = tf.Session()
      checkpoint_path = os.path.join(self.log_dir, 'model')
      tf.get_variable('var1', [1, 2], initializer=tf.constant_initializer(6.0))
      tf.get_variable('var2', [10, 10])
      tf.get_variable('var3', [100, 100])
      sess.run(tf.global_variables_initializer())
      saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
      saver.save(sess, checkpoint_path)


class MetadataColumnsTest(tf.test.TestCase):

  def testLengthDoesNotMatch(self):
    metadata = projector_plugin.EmbeddingMetadata(10)

    with self.assertRaises(ValueError):
      metadata.add_column('Labels', [''] * 11)

  def testValuesNot1D(self):
    metadata = projector_plugin.EmbeddingMetadata(3)
    values = np.array([[1, 2, 3]])

    with self.assertRaises(ValueError):
      metadata.add_column('Labels', values)

  def testMultipleColumnsRetrieval(self):
    metadata = projector_plugin.EmbeddingMetadata(3)
    metadata.add_column('Sizes', [1, 2, 3])
    metadata.add_column('Labels', ['a', 'b', 'c'])
    self.assertEqual(metadata.column_names, ['Sizes', 'Labels'])
    self.assertEqual(metadata.name_to_values['Labels'], ['a', 'b', 'c'])
    self.assertEqual(metadata.name_to_values['Sizes'], [1, 2, 3])

  def testValuesAreListofLists(self):
    metadata = projector_plugin.EmbeddingMetadata(3)
    values = [[1, 2, 3], [4, 5, 6]]
    with self.assertRaises(ValueError):
      metadata.add_column('Labels', values)

  def testStringListRetrieval(self):
    metadata = projector_plugin.EmbeddingMetadata(3)
    metadata.add_column('Labels', ['a', 'B', 'c'])
    self.assertEqual(metadata.name_to_values['Labels'], ['a', 'B', 'c'])
    self.assertEqual(metadata.column_names, ['Labels'])

  def testNumericListRetrieval(self):
    metadata = projector_plugin.EmbeddingMetadata(3)
    metadata.add_column('Labels', [1, 2, 3])
    self.assertEqual(metadata.name_to_values['Labels'], [1, 2, 3])

  def testNumericNdArrayRetrieval(self):
    metadata = projector_plugin.EmbeddingMetadata(3)
    metadata.add_column('Labels', np.array([1, 2, 3]))
    self.assertEqual(metadata.name_to_values['Labels'].tolist(), [1, 2, 3])

  def testStringNdArrayRetrieval(self):
    metadata = projector_plugin.EmbeddingMetadata(2)
    metadata.add_column('Labels', np.array(['a', 'b']))
    self.assertEqual(metadata.name_to_values['Labels'].tolist(), ['a', 'b'])

  def testDuplicateColumnName(self):
    metadata = projector_plugin.EmbeddingMetadata(2)
    metadata.add_column('Labels', np.array(['a', 'b']))
    with self.assertRaises(ValueError):
      metadata.add_column('Labels', np.array(['a', 'b']))


class LRUCacheTest(tf.test.TestCase):

  def testInvalidSize(self):
    with self.assertRaises(ValueError):
      projector_plugin.LRUCache(0)

  def testSimpleGetAndSet(self):
    cache = projector_plugin.LRUCache(1)
    value = cache.get('a')
    self.assertIsNone(value)
    cache.set('a', 10)
    self.assertEqual(cache.get('a'), 10)

  def testErrorsWhenSettingNoneAsValue(self):
    cache = projector_plugin.LRUCache(1)
    with self.assertRaises(ValueError):
      cache.set('a', None)

  def testLRUReplacementPolicy(self):
    cache = projector_plugin.LRUCache(2)
    cache.set('a', 1)
    cache.set('b', 2)
    cache.set('c', 3)
    self.assertIsNone(cache.get('a'))
    self.assertEqual(cache.get('b'), 2)
    self.assertEqual(cache.get('c'), 3)

    # Make 'b' the most recently used.
    cache.get('b')
    cache.set('d', 4)

    # Make sure 'c' got replaced with 'd'.
    self.assertIsNone(cache.get('c'))
    self.assertEqual(cache.get('b'), 2)
    self.assertEqual(cache.get('d'), 4)


if __name__ == '__main__':
  tf.test.main()
