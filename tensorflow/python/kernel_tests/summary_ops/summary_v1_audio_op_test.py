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
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
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
      sample_rate: Union[float, np.ndarray] = 16000.0,
  ) -> None:
    """Writes `tensor` with summary_ops_v2.audio in graph or eager mode."""
    writer = summary_ops_v2.create_file_writer_v2(self.get_temp_dir())
    self.evaluate(writer.init())
    try:
      with writer.as_default(), summary_ops_v2.always_record_summaries():
        self.evaluate(
            summary_ops_v2.audio(
                "audio_test",
                tensor,
                sample_rate=sample_rate,
                max_outputs=3,
                step=step))
    finally:
      self.evaluate(writer.close())

  @test_util.run_in_graph_and_eager_modes
  def testWriteAudioSummaryRejectsScalarTensor(self):
    scalar_tensor = np.array(1.0, dtype=np.float32)
    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                "2 or 3"):
      self._WriteAudioSummary(scalar_tensor)

  @test_util.run_in_graph_and_eager_modes
  def testWriteAudioSummaryRejects1DTensor(self):
    one_d_tensor = np.array([1.0, 2.0], dtype=np.float32)
    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                "2 or 3"):
      self._WriteAudioSummary(one_d_tensor)

  @test_util.run_in_graph_and_eager_modes
  def testWriteAudioSummaryAccepts2DTensor(self):
    two_d_tensor = np.zeros((1, 100), dtype=np.float32)
    # no exception should be raised
    self._WriteAudioSummary(two_d_tensor)

  @test_util.run_in_graph_and_eager_modes
  def testWriteAudioSummaryAccepts3DTensor(self):
    three_d_tensor = np.zeros((1, 100, 2), dtype=np.float32)
    # no exception should be raised
    self._WriteAudioSummary(three_d_tensor)

  @test_util.run_in_graph_and_eager_modes
  def testWriteAudioSummaryAcceptsEmptyTensor(self):
    # Test length_frames = 0
    empty_tensor_frames = np.zeros((1, 0), dtype=np.float32)
    self._WriteAudioSummary(empty_tensor_frames)

    # Test batch_size = 0
    empty_tensor_batch = np.zeros((0, 100), dtype=np.float32)
    self._WriteAudioSummary(empty_tensor_batch)

  @test_util.run_in_graph_and_eager_modes
  def testWriteAudioSummaryRejectsZeroChannels(self):
    empty_tensor_channels = np.zeros((1, 100, 0), dtype=np.float32)
    # The empty tensor can have a null data pointer (seen on CUDA builds), and
    # EncodeAudioAsS16LEWav checks for that before it checks num_channels.
    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                "num_channels|audio is null"):
      self._WriteAudioSummary(empty_tensor_channels)

  @test_util.run_in_graph_and_eager_modes
  def testWriteAudioSummaryRejects4DTensor(self):
    four_d_tensor = np.zeros((1, 10, 2, 2), dtype=np.float32)
    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                "2 or 3"):
      self._WriteAudioSummary(four_d_tensor)

  @test_util.run_in_graph_and_eager_modes
  def testWriteAudioSummaryRejectsNonScalarInputs(self):
    tensor = np.zeros((1, 100), dtype=np.float32)
    # summary_ops_v2 rejects a step with more than one element in Python.
    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                "step`? must be a scalar"):
      self._WriteAudioSummary(tensor, step=np.array([1, 2], dtype=np.int64))
    # A one-element step passes that check and reaches the kernel.
    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                "step must be a scalar"):
      self._WriteAudioSummary(tensor, step=np.array([5], dtype=np.int64))
    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                "sample_rate must be a scalar"):
      self._WriteAudioSummary(
          tensor, sample_rate=np.array([1.0, 2.0], dtype=np.float32))

  @test_util.run_in_graph_and_eager_modes
  def testWriteAudioSummaryRejectsNonScalarTag(self):
    # summary_ops_v2.audio builds the tag from a name string, so only the raw
    # op can pass a non-scalar tag.
    writer = summary_ops_v2.create_file_writer_v2(self.get_temp_dir())
    self.evaluate(writer.init())
    try:
      with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                  "tag must be a scalar"):
        self.evaluate(
            gen_summary_ops.write_audio_summary(
                writer=writer._resource,
                step=0,
                tag=np.array(["a", "b"]),
                tensor=np.zeros((1, 100), dtype=np.float32),
                sample_rate=16000.0,
                max_outputs=3))
    finally:
      self.evaluate(writer.close())


if __name__ == "__main__":
  test.main()
