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

from werkzeug import test as werkzeug_test
from werkzeug import wrappers
from google.protobuf import text_format
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary import plugin_asset
from tensorflow.python.summary.writer import writer
from tensorflow.python.training import saver as saver_lib
from tensorflow.tensorboard.backend import application
from tensorflow.tensorboard.backend.event_processing import event_multiplexer
from tensorflow.tensorboard.plugins.projector import projector_config_pb2
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
    plugin = projector_plugin.ProjectorPlugin()
    wsgi_app = application.TensorBoardWSGIApp(
        self.log_dir, [plugin], multiplexer, reload_interval=0)
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
    config = projector_config_pb2.ProjectorConfig()
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


class MetadataColumnsTest(test.TestCase):

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


class ProjectorPluginAssetTest(test.TestCase):

  def testNoAssets(self):
    manager = plugin_asset.get_plugin_asset(
        projector_plugin.ProjectorPluginAsset)
    self.assertEqual(manager.assets(), {'projector_config.pbtxt': ''})

  def testAddEmbeddingNoMetadata(self):
    manager = plugin_asset.get_plugin_asset(
        projector_plugin.ProjectorPluginAsset)
    manager.add_embedding('test', np.array([[1, 2, 3.1]]))

    config = projector_config_pb2.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = 'test'
    embedding.tensor_shape.extend([1, 3])
    embedding.tensor_path = 'test_values.tsv'
    expected_config_pbtxt = text_format.MessageToString(config)

    self.assertEqual(manager.assets(), {
        'projector_config.pbtxt': expected_config_pbtxt,
        'test_values.tsv': b'1\t2\t3.1\n'
    })

  def testAddEmbeddingIncorrectRank(self):
    manager = plugin_asset.get_plugin_asset(
        projector_plugin.ProjectorPluginAsset)
    with self.assertRaises(ValueError):
      manager.add_embedding('test', np.array([1, 2, 3.1]))

  def testAddEmbeddingWithTwoMetadataColumns(self):
    manager = plugin_asset.get_plugin_asset(
        projector_plugin.ProjectorPluginAsset)

    metadata = projector_plugin.EmbeddingMetadata(3)
    metadata.add_column('labels', ['a', 'b', 'друг јазик'])
    metadata.add_column('sizes', [10, 20, 30])
    manager.add_embedding('test', np.array([[1], [2], [3]]), metadata)

    config = projector_config_pb2.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = 'test'
    embedding.tensor_shape.extend([3, 1])
    embedding.tensor_path = 'test_values.tsv'
    embedding.metadata_path = 'test_metadata.tsv'
    expected_config_pbtxt = text_format.MessageToString(config)

    self.assertEqual(manager.assets(), {
        'projector_config.pbtxt': expected_config_pbtxt,
        'test_values.tsv': b'1\n2\n3\n',
        'test_metadata.tsv': 'labels\tsizes\na\t10\nb\t20\nдруг јазик\t30\n'
    })

  def testAddEmbeddingWithOneMetadataColumn(self):
    manager = plugin_asset.get_plugin_asset(
        projector_plugin.ProjectorPluginAsset)

    metadata = projector_plugin.EmbeddingMetadata(3)
    metadata.add_column('labels', ['a', 'b', 'c'])
    manager.add_embedding('test', np.array([[1], [2], [3]]), metadata)

    config = projector_config_pb2.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = 'test'
    embedding.tensor_shape.extend([3, 1])
    embedding.tensor_path = 'test_values.tsv'
    embedding.metadata_path = 'test_metadata.tsv'
    expected_config_pbtxt = text_format.MessageToString(config)

    self.assertEqual(manager.assets(), {
        'projector_config.pbtxt': expected_config_pbtxt,
        'test_values.tsv': b'1\n2\n3\n',
        'test_metadata.tsv': 'a\nb\nc\n'
    })

  def testAddEmbeddingWithThumbnails(self):
    manager = plugin_asset.get_plugin_asset(
        projector_plugin.ProjectorPluginAsset)

    image1 = np.array([[[1, 2, 3], [4, 5, 6]],
                       [[7, 8, 9], [10, 11, 12]]])
    image2 = np.array([[[10, 20, 30], [40, 50, 60]],
                       [[70, 80, 90], [100, 110, 120]]])
    manager.add_embedding(
        'test',
        np.array([[1], [2], [3]]),
        thumbnails=[image1, image2],
        thumbnail_dim=[2, 2])

    assets = manager.assets()

    config = projector_config_pb2.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = 'test'
    embedding.tensor_shape.extend([3, 1])
    embedding.tensor_path = 'test_values.tsv'
    embedding.sprite.image_path = 'test_sprite.png'
    embedding.sprite.single_image_dim.extend([2, 2])
    expected_config_pbtxt = text_format.MessageToString(config)

    self.assertEqual(assets['projector_config.pbtxt'], expected_config_pbtxt)
    self.assertEqual(assets['test_values.tsv'], b'1\n2\n3\n')

    png_bytes = assets['test_sprite.png']
    with ops.Graph().as_default():
      s = session.Session()
      image_array = image_ops.decode_png(png_bytes).eval(session=s).tolist()
    expected_master_image = [
        [[1, 2, 3], [4, 5, 6], [10, 20, 30], [40, 50, 60]],
        [[7, 8, 9], [10, 11, 12], [70, 80, 90], [100, 110, 120]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    ]
    self.assertEqual(image_array, expected_master_image)

  def testAddEmbeddingWithSpriteImageButNoThumbnailDim(self):
    manager = plugin_asset.get_plugin_asset(
        projector_plugin.ProjectorPluginAsset)

    thumbnails = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    with self.assertRaises(ValueError):
      manager.add_embedding(
          'test', np.array([[1], [2], [3]]), thumbnails=thumbnails)

  def testAddEmbeddingThumbnailDimNotAList(self):
    manager = plugin_asset.get_plugin_asset(
        projector_plugin.ProjectorPluginAsset)

    thumbnails = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    with self.assertRaises(ValueError):
      manager.add_embedding(
          'test', np.array([[1], [2], [3]]), thumbnails=thumbnails,
          thumbnail_dim=4)

  def testAddEmbeddingThumbnailDimNotOfLength2(self):
    manager = plugin_asset.get_plugin_asset(
        projector_plugin.ProjectorPluginAsset)

    thumbnails = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    with self.assertRaises(ValueError):
      manager.add_embedding(
          'test', np.array([[1], [2], [3]]), thumbnails=thumbnails,
          thumbnail_dim=[4])

  def testAddEmbeddingWithMetadataOfIncorrectLength(self):
    manager = plugin_asset.get_plugin_asset(
        projector_plugin.ProjectorPluginAsset)

    metadata = projector_plugin.EmbeddingMetadata(3)
    metadata.add_column('labels', ['a', 'b', 'c'])
    # values has length 2, while metadata has length 3.
    values = np.array([[1], [2]])

    with self.assertRaises(ValueError):
      manager.add_embedding('test', values, metadata)

  def testAddMetadataForVariableButNoColumns(self):
    manager = plugin_asset.get_plugin_asset(
        projector_plugin.ProjectorPluginAsset)
    metadata = projector_plugin.EmbeddingMetadata(3)
    with self.assertRaises(ValueError):
      manager.add_metadata_for_embedding_variable('test', metadata)

  def testAddMetadataForVariable(self):
    manager = plugin_asset.get_plugin_asset(
        projector_plugin.ProjectorPluginAsset)
    metadata = projector_plugin.EmbeddingMetadata(3)
    metadata.add_column('Labels', ['a', 'b', 'c'])
    manager.add_metadata_for_embedding_variable('test', metadata)

    config = projector_config_pb2.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = 'test'
    embedding.metadata_path = 'test_metadata.tsv'
    expected_config_pbtxt = text_format.MessageToString(config)

    self.assertEqual(manager.assets(), {
        'projector_config.pbtxt': expected_config_pbtxt,
        'test_metadata.tsv': 'a\nb\nc\n'
    })

  def testAddMetadataForVariableAtLeastOneParamIsRequired(self):
    manager = plugin_asset.get_plugin_asset(
        projector_plugin.ProjectorPluginAsset)
    with self.assertRaises(ValueError):
      manager.add_metadata_for_embedding_variable('test')

  def testNoAssetsProperSerializationOnDisk(self):
    logdir = self.get_temp_dir()
    plugin_dir = os.path.join(logdir, writer._PLUGINS_DIR,
                              projector_plugin.ProjectorPluginAsset.plugin_name)

    with ops.Graph().as_default() as g:
      plugin_asset.get_plugin_asset(projector_plugin.ProjectorPluginAsset)
      fw = writer.FileWriter(logdir)
      fw.add_graph(g)

    with gfile.Open(os.path.join(plugin_dir, 'projector_config.pbtxt')) as f:
      content = f.read()
    self.assertEqual(content, '')

  def testNoReferenceToPluginNoSerializationOnDisk(self):
    logdir = self.get_temp_dir()
    plugin_dir = os.path.join(logdir, writer._PLUGINS_DIR,
                              projector_plugin.ProjectorPluginAsset.plugin_name)

    with ops.Graph().as_default() as g:
      fw = writer.FileWriter(logdir)
      fw.add_graph(g)

    self.assertFalse(
        gfile.Exists(plugin_dir),
        'The projector plugin directory should not exist.')

if __name__ == '__main__':
  test.main()
