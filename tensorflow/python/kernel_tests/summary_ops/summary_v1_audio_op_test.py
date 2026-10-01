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

from typing import Union

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

  def _WriteAudioSummary(
      self,
      tensor: np.ndarray,
      step: Union[int, np.ndarray] = 0,
      tag: Union[str, np.ndarray] = "audio_test",
      sample_rate: Union[float, np.ndarray] = 16000.0,
  ) -> None:
    """Helper to call write_audio_summary with standard test parameters."""
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops_v2.create_file_writer_v2(logdir)
      try:
        gen_summary_ops.write_audio_summary(
            writer=writer._resource,
            step=step,
            tag=tag,
            tensor=tensor,
            sample_rate=sample_rate,
            max_outputs=3)
      finally:
        writer.close()

  def testWriteAudioSummaryRejectsScalarTensor(self):
    scalar_tensor = np.array(1.0, dtype=np.float32)
    with self.assertRaisesRegex(errors.InvalidArgumentError, "2 or 3"):
      self._WriteAudioSummary(scalar_tensor)

  def testWriteAudioSummaryRejects1DTensor(self):
    one_d_tensor = np.array([1.0, 2.0], dtype=np.float32)
    with self.assertRaisesRegex(errors.InvalidArgumentError, "2 or 3"):
      self._WriteAudioSummary(one_d_tensor)

  def testWriteAudioSummaryAccepts2DTensor(self):
    two_d_tensor = np.zeros((1, 100), dtype=np.float32)
    # no exception should be raised
    self._WriteAudioSummary(two_d_tensor)

  def testWriteAudioSummaryAccepts3DTensor(self):
    three_d_tensor = np.zeros((1, 100, 2), dtype=np.float32)
    # no exception should be raised
    self._WriteAudioSummary(three_d_tensor)

  def testWriteAudioSummaryAcceptsEmptyTensor(self):
    # Test length_frames = 0
    empty_tensor_frames = np.zeros((1, 0), dtype=np.float32)
    self._WriteAudioSummary(empty_tensor_frames)

    # Test batch_size = 0
    empty_tensor_batch = np.zeros((0, 100), dtype=np.float32)
    self._WriteAudioSummary(empty_tensor_batch)

  def testWriteAudioSummaryRejectsZeroChannels(self):
    empty_tensor_channels = np.zeros((1, 100, 0), dtype=np.float32)
    with self.assertRaisesRegex(errors.InvalidArgumentError, "num_channels"):
      self._WriteAudioSummary(empty_tensor_channels)

  def testWriteAudioSummaryRejects4DTensor(self):
    four_d_tensor = np.zeros((1, 10, 2, 2), dtype=np.float32)
    with self.assertRaisesRegex(errors.InvalidArgumentError, "2 or 3"):
      self._WriteAudioSummary(four_d_tensor)

  def testWriteAudioSummaryRejectsNonScalarInputs(self):
    tensor = np.zeros((1, 100), dtype=np.float32)
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "step must be a scalar"):
      self._WriteAudioSummary(tensor, step=np.array([1, 2], dtype=np.int64))
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "tag must be a scalar"):
      self._WriteAudioSummary(tensor, tag=np.array(["a", "b"]))
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "sample_rate must be a scalar"):
      self._WriteAudioSummary(
          tensor, sample_rate=np.array([1.0, 2.0], dtype=np.float32))


if __name__ == "__main__":
  test.main()
