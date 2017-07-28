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
"""Tests for MetaGraphDef Transform Tool."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google.protobuf.any_pb2 import Any
from tensorflow.contrib.meta_graph_transform import meta_graph_transform
from tensorflow.core.framework import function_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.training import saver
from tensorflow.python.util import compat


def _make_asset_file_def_any(node_name):
  asset_file_def = meta_graph_pb2.AssetFileDef()
  asset_file_def.tensor_info.name = node_name
  any_message = Any()
  any_message.Pack(asset_file_def)
  return any_message


class MetaGraphTransformTest(test.TestCase):

  def test_meta_graph_transform(self):

    with ops.Graph().as_default():
      with tf_session.Session(''):
        a = array_ops.placeholder(dtypes.int64, [1], name='a')
        b = array_ops.placeholder(dtypes.int64, [1], name='b')
        c = array_ops.placeholder(dtypes.int64, [1], name='c')
        _ = a * b
        _ = b * c
        base_meta_graph_def = saver.export_meta_graph()

    with ops.Graph().as_default():
      with tf_session.Session(''):
        a = array_ops.placeholder(dtypes.int64, [1], name='a')
        b = array_ops.placeholder(dtypes.int64, [1], name='b')
        _ = a * b
        meta_info_def = meta_graph_pb2.MetaGraphDef.MetaInfoDef()
        meta_info_def.tags.append('tag_ab')

        expected_meta_graph_def = saver.export_meta_graph(
            meta_info_def=meta_info_def)
        # Graph rewriter clears versions field, so we expect that.
        expected_meta_graph_def.graph_def.ClearField('versions')
        # Graph rewriter adds an empty library field, so we expect that.
        expected_meta_graph_def.graph_def.library.CopyFrom(
            function_pb2.FunctionDefLibrary())

    input_names = ['a', 'b']
    output_names = ['mul:0']
    transforms = ['strip_unused_nodes']
    tags = ['tag_ab']
    print('AAAAAA: {}'.format(base_meta_graph_def))
    transformed_meta_graph_def = meta_graph_transform.meta_graph_transform(
        base_meta_graph_def, input_names, output_names, transforms, tags)

    self.assertEqual(expected_meta_graph_def, transformed_meta_graph_def)

  def test_add_pruned_collection_node(self):
    collection_name = 'node_collection'
    base_meta_graph_def = meta_graph_pb2.MetaGraphDef()
    base_meta_graph_def.collection_def[collection_name].node_list.value.extend(
        ['node1', 'node2', 'node3', 'node4'])

    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    removed_op_names = ['node2', 'node4', 'node5']
    meta_graph_transform._add_pruned_collection(
        base_meta_graph_def, meta_graph_def, collection_name, removed_op_names)

    collection = meta_graph_def.collection_def[collection_name]

    expected_nodes = ['node1', 'node3']
    self.assertEqual(expected_nodes, collection.node_list.value)

  def test_add_pruned_collection_int(self):
    collection_name = 'int_collection'
    base_meta_graph_def = meta_graph_pb2.MetaGraphDef()
    base_meta_graph_def.collection_def[collection_name].int64_list.value[:] = (
        [10, 20, 30, 40])

    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    removed_op_names = ['node2', 'node4', 'node5']
    meta_graph_transform._add_pruned_collection(
        base_meta_graph_def, meta_graph_def, collection_name, removed_op_names)

    collection = meta_graph_def.collection_def[collection_name]

    expected_ints = [10, 20, 30, 40]
    self.assertEqual(expected_ints, collection.int64_list.value)

  def test_add_pruned_collection_proto_in_any_list(self):
    collection_name = 'proto_collection'
    base_meta_graph_def = meta_graph_pb2.MetaGraphDef()
    base_meta_graph_def.collection_def[collection_name].any_list.value.extend(
        [_make_asset_file_def_any('node1'),
         _make_asset_file_def_any('node2'),
         _make_asset_file_def_any('node3'),
         _make_asset_file_def_any('node4')])

    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    removed_op_names = ['node2', 'node4', 'node5']
    meta_graph_transform._add_pruned_collection(
        base_meta_graph_def, meta_graph_def, collection_name, removed_op_names)

    collection = meta_graph_def.collection_def[collection_name]

    expected_protos = [_make_asset_file_def_any('node1'),
                       _make_asset_file_def_any('node3')]
    self.assertEqual(expected_protos, collection.any_list.value[:])

  def test_add_pruned_collection_proto_in_bytes_list(self):
    collection_name = 'proto_collection'
    base_meta_graph_def = meta_graph_pb2.MetaGraphDef()
    base_meta_graph_def.collection_def[collection_name].bytes_list.value.extend(
        [compat.as_bytes(compat.as_str_any(_make_asset_file_def_any('node1'))),
         compat.as_bytes(compat.as_str_any(_make_asset_file_def_any('node2'))),
         compat.as_bytes(compat.as_str_any(_make_asset_file_def_any('node3'))),
         compat.as_bytes(compat.as_str_any(_make_asset_file_def_any('node4')))])

    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    removed_op_names = ['node2', 'node4', 'node5']
    meta_graph_transform._add_pruned_collection(
        base_meta_graph_def, meta_graph_def, collection_name, removed_op_names)

    collection = meta_graph_def.collection_def[collection_name]

    expected_values = [
        compat.as_bytes(compat.as_str_any(_make_asset_file_def_any('node1'))),
        compat.as_bytes(compat.as_str_any(_make_asset_file_def_any('node3')))]
    self.assertEqual(expected_values, collection.bytes_list.value[:])

  def test_add_pruned_saver(self):
    base_meta_graph_def = meta_graph_pb2.MetaGraphDef()

    base_meta_graph_def.saver_def.filename_tensor_name = 'node1'
    base_meta_graph_def.saver_def.save_tensor_name = 'node3'
    base_meta_graph_def.saver_def.restore_op_name = 'node6'

    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    removed_op_names = ['node2', 'node4', 'node5']
    meta_graph_transform._add_pruned_saver(base_meta_graph_def,
                                           meta_graph_def,
                                           removed_op_names)

    # TODO(b/63447631): For now the saver is just copied unchanged
    self.assertEqual(base_meta_graph_def.saver_def, meta_graph_def.saver_def)

  def test_add_pruned_signature(self):
    base_meta_graph_def = meta_graph_pb2.MetaGraphDef()

    signature_name_keep = 'test_signature_keep'
    base_sig_keep = base_meta_graph_def.signature_def[signature_name_keep]
    base_sig_keep.inputs['input_1'].name = 'input_1'
    base_sig_keep.outputs['output_1'].name = 'output_1'

    signature_name_remove = 'test_signature_remove'
    base_sig_remove = base_meta_graph_def.signature_def[signature_name_remove]
    base_sig_remove.inputs['node2'].name = 'node2'
    base_sig_remove.outputs['output_1'].name = 'output_1'

    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    removed_op_names = ['node2', 'node4', 'node5']
    meta_graph_transform._add_pruned_signature(base_meta_graph_def,
                                               meta_graph_def,
                                               signature_name_keep,
                                               removed_op_names)
    meta_graph_transform._add_pruned_signature(base_meta_graph_def,
                                               meta_graph_def,
                                               signature_name_remove,
                                               removed_op_names)

    self.assertTrue(signature_name_keep in meta_graph_def.signature_def)
    sig_keep = meta_graph_def.signature_def[signature_name_keep]
    self.assertEqual(base_sig_keep, sig_keep)

    self.assertFalse(signature_name_remove in meta_graph_def.signature_def)


if __name__ == '__main__':
  test.main()
