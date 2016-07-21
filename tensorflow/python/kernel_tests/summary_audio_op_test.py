# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for summary sound op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


class SummaryAudioOpTest(tf.test.TestCase):

  def _AsSummary(self, s):
    summ = tf.Summary()
    summ.ParseFromString(s)
    return summ

  def _CheckProto(self, audio_summ, sample_rate, num_channels, length_frames):
    """Verify that the non-audio parts of the audio_summ proto match shape."""
    # Only the first 3 sounds are returned.
    for v in audio_summ.value:
      v.audio.ClearField("encoded_audio_string")
    expected = "\n".join("""
        value {
          tag: "snd/audio/%d"
          audio { content_type: "audio/wav" sample_rate: %d
                  num_channels: %d length_frames: %d }
        }""" % (i, sample_rate, num_channels, length_frames) for i in xrange(3))
    self.assertProtoEquals(expected, audio_summ)

  def testAudioSummary(self):
    np.random.seed(7)
    with self.test_session() as sess:
      num_frames = 7
      for channels in 1, 2, 5, 8:
        shape = (4, num_frames, channels)
        # Generate random audio in the range [-1.0, 1.0).
        const = 2.0 * np.random.random(shape) - 1.0

        # Summarize
        sample_rate = 8000
        summ = tf.audio_summary("snd",
                                const,
                                max_outputs=3,
                                sample_rate=sample_rate)
        value = sess.run(summ)
        self.assertEqual([], summ.get_shape())
        audio_summ = self._AsSummary(value)

        # Check the rest of the proto
        self._CheckProto(audio_summ, sample_rate, channels, num_frames)


if __name__ == "__main__":
  tf.test.main()
