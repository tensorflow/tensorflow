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
"""Tests for data input for speech commands."""

import os.path

from tensorflow.examples.speech_commands import freeze
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.platform import test


class FreezeTest(test.TestCase):

  def _getConcreteFunction(self, **kwargs):
    _, serve_fn = freeze.create_inference_graph(
        wanted_words='a,b,c,d',
        sample_rate=16000,
        clip_duration_ms=1000.0,
        clip_stride_ms=30.0,
        window_size_ms=30.0,
        window_stride_ms=10.0,
        feature_bin_count=40,
        model_architecture='conv',
        **kwargs)
    return serve_fn.get_concrete_function()

  def testCreateInferenceGraphWithMfcc(self):
    concrete_fn = self._getConcreteFunction(preprocess='mfcc')
    self.assertIsNotNone(concrete_fn.graph.get_tensor_by_name('wav_data:0'))
    self.assertIsNotNone(
        concrete_fn.graph.get_tensor_by_name('decoded_sample_data:0'))
    self.assertIsNotNone(
        concrete_fn.graph.get_tensor_by_name('labels_softmax:0'))
    ops = [node.op for node in concrete_fn.graph.as_graph_def().node]
    self.assertEqual(1, ops.count('Mfcc'))

  def testCreateInferenceGraphWithoutMfcc(self):
    concrete_fn = self._getConcreteFunction(preprocess='average')
    self.assertIsNotNone(concrete_fn.graph.get_tensor_by_name('wav_data:0'))
    self.assertIsNotNone(
        concrete_fn.graph.get_tensor_by_name('decoded_sample_data:0'))
    self.assertIsNotNone(
        concrete_fn.graph.get_tensor_by_name('labels_softmax:0'))
    ops = [node.op for node in concrete_fn.graph.as_graph_def().node]
    self.assertEqual(0, ops.count('Mfcc'))

  def testCreateInferenceGraphWithMicro(self):
    concrete_fn = self._getConcreteFunction(preprocess='micro')
    self.assertIsNotNone(concrete_fn.graph.get_tensor_by_name('wav_data:0'))
    self.assertIsNotNone(
        concrete_fn.graph.get_tensor_by_name('decoded_sample_data:0'))
    self.assertIsNotNone(
        concrete_fn.graph.get_tensor_by_name('labels_softmax:0'))

  def testFeatureBinCount(self):
    _, serve_fn = freeze.create_inference_graph(
        wanted_words='a,b,c,d',
        sample_rate=16000,
        clip_duration_ms=1000.0,
        clip_stride_ms=30.0,
        window_size_ms=30.0,
        window_stride_ms=10.0,
        feature_bin_count=80,
        model_architecture='conv',
        preprocess='average')
    concrete_fn = serve_fn.get_concrete_function()
    self.assertIsNotNone(concrete_fn.graph.get_tensor_by_name('wav_data:0'))
    self.assertIsNotNone(
        concrete_fn.graph.get_tensor_by_name('decoded_sample_data:0'))
    self.assertIsNotNone(
        concrete_fn.graph.get_tensor_by_name('labels_softmax:0'))
    ops = [node.op for node in concrete_fn.graph.as_graph_def().node]
    self.assertEqual(0, ops.count('Mfcc'))

  def testCreateSavedModel(self):
    tmp_dir = self.get_temp_dir()
    saved_model_path = os.path.join(tmp_dir, 'saved_model')
    model, serve_fn = freeze.create_inference_graph(
        wanted_words='a,b,c,d',
        sample_rate=16000,
        clip_duration_ms=1000.0,
        clip_stride_ms=30.0,
        window_size_ms=30.0,
        window_stride_ms=10.0,
        feature_bin_count=40,
        model_architecture='conv',
        preprocess='micro')
    concrete_fn = serve_fn.get_concrete_function()
    freeze.save_saved_model(saved_model_path, model, concrete_fn)
    self.assertTrue(os.path.exists(saved_model_path))

  def testFreezeGraphDef(self):
    _, serve_fn = freeze.create_inference_graph(
        wanted_words='a,b,c,d',
        sample_rate=16000,
        clip_duration_ms=1000.0,
        clip_stride_ms=30.0,
        window_size_ms=30.0,
        window_stride_ms=10.0,
        feature_bin_count=40,
        model_architecture='conv',
        preprocess='mfcc')
    concrete_fn = serve_fn.get_concrete_function()
    frozen_func = convert_to_constants.convert_variables_to_constants_v2(
        concrete_fn)
    node_names = [node.name for node in frozen_func.graph.as_graph_def().node]
    self.assertIn('wav_data', node_names)
    self.assertIn('labels_softmax', node_names)
    # The trained weights should have been folded into Const nodes rather
    # than remaining as ReadVariableOp/VarHandleOp nodes.
    ops = [node.op for node in frozen_func.graph.as_graph_def().node]
    self.assertNotIn('VarHandleOp', ops)


if __name__ == '__main__':
  test.main()
