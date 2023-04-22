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
"""Tests for test file generation for speech commands."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.examples.speech_commands import generate_streaming_test_wav
from tensorflow.python.platform import test


class GenerateStreamingTestWavTest(test.TestCase):

  def testMixInAudioSample(self):
    track_data = np.zeros([10000])
    sample_data = np.ones([1000])
    generate_streaming_test_wav.mix_in_audio_sample(
        track_data, 2000, sample_data, 0, 1000, 1.0, 100, 100)
    self.assertNear(1.0, track_data[2500], 0.0001)
    self.assertNear(0.0, track_data[3500], 0.0001)


if __name__ == "__main__":
  test.main()
