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
"""Tests for summary V1 audio op."""

import numpy as np

from tensorflow.core.framework import summary_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_summary_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.platform import test
from tensorflow.python.summary import summary


class SummaryV1AudioOpTest(test.TestCase):

  def _AsSummary(self, s):
    summ = summary_pb2.Summary()
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
        }""" % (i, sample_rate, num_channels, length_frames) for i in range(3))
    self.assertProtoEquals(expected, audio_summ)

  def testAudioSummary(self):
    np.random.seed(7)
    for channels in (1, 2, 5, 8):
      with self.session(graph=ops.Graph()) as sess:
        num_frames = 7
        shape = (4, num_frames, channels)
        # Generate random audio in the range [-1.0, 1.0).
        const = 2.0 * np.random.random(shape) - 1.0

        # Summarize
        sample_rate = 8000
        summ = summary.audio(
            "snd", const, max_outputs=3, sample_rate=sample_rate)
        value = self.evaluate(summ)
        self.assertEqual([], summ.get_shape())
        audio_summ = self._AsSummary(value)

        # Check the rest of the proto
        self._CheckProto(audio_summ, sample_rate, channels, num_frames)

  def testAudioSummaryTensorDimensionValidation(self):
    """Test that WriteAudioSummary rejects tensors with less than 2 dims."""
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops_v2.create_file_writer_v2(logdir)
      with writer.as_default():
        # Test scalar tensor (0D) - should fail
        with self.assertRaisesRegex(
            errors.InvalidArgumentError, "at least 2"):
          gen_summary_ops.write_audio_summary(
              writer=writer._resource,
              step=0,
              tag="audio_test",
              tensor=np.float32(1.0),  # scalar
              sample_rate=16000.0,
              max_outputs=3)

        # Test 1D tensor - should fail
        with self.assertRaisesRegex(
            errors.InvalidArgumentError, "at least 2"):
          gen_summary_ops.write_audio_summary(
              writer=writer._resource,
              step=0,
              tag="audio_test",
              tensor=np.array([1.0, 2.0], dtype=np.float32),  # 1D
              sample_rate=16000.0,
              max_outputs=3)

        # Test 2D tensor - should succeed
        gen_summary_ops.write_audio_summary(
            writer=writer._resource,
            step=0,
            tag="audio_test",
            tensor=np.zeros((1, 100), dtype=np.float32),  # 2D
            sample_rate=16000.0,
            max_outputs=3)


if __name__ == "__main__":
  test.main()
