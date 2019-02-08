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
"""Tests for WAVE file labeling tool."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.examples.speech_commands import label_wav
from tensorflow.python.platform import test


class LabelWavTest(test.TestCase):

  def _getWavData(self):
    with self.cached_session() as sess:
      sample_data = tf.zeros([1000, 2])
      wav_encoder = contrib_audio.encode_wav(sample_data, 16000)
      wav_data = self.evaluate(wav_encoder)
    return wav_data

  def _saveTestWavFile(self, filename, wav_data):
    with open(filename, "wb") as f:
      f.write(wav_data)

  def testLabelWav(self):
    tmp_dir = self.get_temp_dir()
    wav_data = self._getWavData()
    wav_filename = os.path.join(tmp_dir, "wav_file.wav")
    self._saveTestWavFile(wav_filename, wav_data)
    input_name = "test_input"
    output_name = "test_output"
    graph_filename = os.path.join(tmp_dir, "test_graph.pb")
    with tf.Session() as sess:
      tf.placeholder(tf.string, name=input_name)
      tf.zeros([1, 3], name=output_name)
      with open(graph_filename, "wb") as f:
        f.write(sess.graph.as_graph_def().SerializeToString())
    labels_filename = os.path.join(tmp_dir, "test_labels.txt")
    with open(labels_filename, "w") as f:
      f.write("a\nb\nc\n")
    label_wav.label_wav(wav_filename, labels_filename, graph_filename,
                        input_name + ":0", output_name + ":0", 3)


if __name__ == "__main__":
  test.main()
