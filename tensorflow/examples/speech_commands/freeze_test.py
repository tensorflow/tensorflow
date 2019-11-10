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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.speech_commands import freeze
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class FreezeTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testCreateInferenceGraphWithMfcc(self):
    with self.cached_session() as sess:
      freeze.create_inference_graph(
          wanted_words='a,b,c,d',
          sample_rate=16000,
          clip_duration_ms=1000.0,
          clip_stride_ms=30.0,
          window_size_ms=30.0,
          window_stride_ms=10.0,
          feature_bin_count=40,
          model_architecture='conv',
          preprocess='mfcc')
      self.assertIsNotNone(sess.graph.get_tensor_by_name('wav_data:0'))
      self.assertIsNotNone(
          sess.graph.get_tensor_by_name('decoded_sample_data:0'))
      self.assertIsNotNone(sess.graph.get_tensor_by_name('labels_softmax:0'))
      ops = [node.op for node in sess.graph_def.node]
      self.assertEqual(1, ops.count('Mfcc'))

  @test_util.run_deprecated_v1
  def testCreateInferenceGraphWithoutMfcc(self):
    with self.cached_session() as sess:
      freeze.create_inference_graph(
          wanted_words='a,b,c,d',
          sample_rate=16000,
          clip_duration_ms=1000.0,
          clip_stride_ms=30.0,
          window_size_ms=30.0,
          window_stride_ms=10.0,
          feature_bin_count=40,
          model_architecture='conv',
          preprocess='average')
      self.assertIsNotNone(sess.graph.get_tensor_by_name('wav_data:0'))
      self.assertIsNotNone(
          sess.graph.get_tensor_by_name('decoded_sample_data:0'))
      self.assertIsNotNone(sess.graph.get_tensor_by_name('labels_softmax:0'))
      ops = [node.op for node in sess.graph_def.node]
      self.assertEqual(0, ops.count('Mfcc'))

  @test_util.run_deprecated_v1
  def testCreateInferenceGraphWithMicro(self):
    with self.cached_session() as sess:
      freeze.create_inference_graph(
          wanted_words='a,b,c,d',
          sample_rate=16000,
          clip_duration_ms=1000.0,
          clip_stride_ms=30.0,
          window_size_ms=30.0,
          window_stride_ms=10.0,
          feature_bin_count=40,
          model_architecture='conv',
          preprocess='micro')
      self.assertIsNotNone(sess.graph.get_tensor_by_name('wav_data:0'))
      self.assertIsNotNone(
          sess.graph.get_tensor_by_name('decoded_sample_data:0'))
      self.assertIsNotNone(sess.graph.get_tensor_by_name('labels_softmax:0'))

  @test_util.run_deprecated_v1
  def testFeatureBinCount(self):
    with self.cached_session() as sess:
      freeze.create_inference_graph(
          wanted_words='a,b,c,d',
          sample_rate=16000,
          clip_duration_ms=1000.0,
          clip_stride_ms=30.0,
          window_size_ms=30.0,
          window_stride_ms=10.0,
          feature_bin_count=80,
          model_architecture='conv',
          preprocess='average')
      self.assertIsNotNone(sess.graph.get_tensor_by_name('wav_data:0'))
      self.assertIsNotNone(
          sess.graph.get_tensor_by_name('decoded_sample_data:0'))
      self.assertIsNotNone(sess.graph.get_tensor_by_name('labels_softmax:0'))
      ops = [node.op for node in sess.graph_def.node]
      self.assertEqual(0, ops.count('Mfcc'))


if __name__ == '__main__':
  test.main()
