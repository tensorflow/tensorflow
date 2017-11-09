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
"""Unit tests for eager execution SummaryWriter."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

import numpy as np

from tensorflow.contrib.eager.python import summary_writer
from tensorflow.core.util import event_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import tf_record
from tensorflow.python.platform import gfile


class SummaryWriterTest(test.TestCase):

  def setUp(self):
    super(SummaryWriterTest, self).setUp()
    self._test_device = "gpu:0" if context.num_gpus() else "cpu:0"
    self._tmp_logdir = tempfile.mkdtemp()
    with context.device(self._test_device):
      # Use max_queue=0 so that summaries are immediately flushed to filesystem,
      # making testing easier.
      self._writer = summary_writer.SummaryWriter(self._tmp_logdir, max_queue=0)

  def tearDown(self):
    if os.path.isdir(self._tmp_logdir):
      shutil.rmtree(self._tmp_logdir)
    super(SummaryWriterTest, self).tearDown()

  def _readLastEvent(self, logdir=None):
    if not logdir:
      logdir = self._tmp_logdir
    files = [f for f in gfile.ListDirectory(logdir)
             if not gfile.IsDirectory(os.path.join(logdir, f))]
    file_path = os.path.join(logdir, files[0])
    records = list(tf_record.tf_record_iterator(file_path))
    event = event_pb2.Event()
    event.ParseFromString(records[-1])
    return event

  def testGlobalStep(self):
    with context.device(self._test_device):
      orig_step = self._writer.global_step
      self._writer.step()
      self.assertEqual(orig_step + 1, self._writer.global_step)
      self.assertEqual(orig_step + 1, self._writer.global_step)
      self._writer.step()
      self._writer.step()
      self.assertEqual(orig_step + 3, self._writer.global_step)

  def testGenericSummary(self):
    with context.device(self._test_device):
      x = constant_op.constant(1337.0)
      with context.device("cpu:0"):
        metadata = constant_op.constant("foo")
      self._writer.generic("x", x, metadata)
      event = self._readLastEvent()
      self.assertEqual("x", event.summary.value[0].tag)

  def testScalarSummary(self):
    with context.device(self._test_device):
      x = constant_op.constant(1337.0)
      self._writer.scalar("x", x)
      event = self._readLastEvent()
      self.assertTrue("x", event.summary.value[0].tag)
      self.assertEqual(1337.0, event.summary.value[0].simple_value)

  def testHistogramSummary(self):
    with context.device(self._test_device):
      y = constant_op.constant([1.0, 3.0, 3.0, 7.0])
      self._writer.histogram("y", y)
      event = self._readLastEvent()
      self.assertEqual("y", event.summary.value[0].tag)
      self.assertTrue(event.summary.value[0].histo)

  def testImageSummary(self):
    with context.device(self._test_device):
      a = constant_op.constant([[10.0, 20.0], [-20.0, -10.0]])
      self._writer.histogram("image1", a)
      event = self._readLastEvent()
      self.assertEqual("image1", event.summary.value[0].tag)
      self.assertTrue(event.summary.value[0].image)

  def testAudioSummary(self):
    with context.device(self._test_device):
      w = constant_op.constant(np.random.rand(3, 10, 2), dtype=dtypes.float32)
      fs = constant_op.constant(44100.0, dtype=dtypes.float32)
      max_outputs = 1
      self._writer.audio("audio1", w, fs, max_outputs)
      event = self._readLastEvent()
      self.assertTrue(event.summary.value[0].audio)

  def testTwoSummaryWritersGlobalStepsWorkWithoutCrosstalk(self):
    tmp_logdir2 = os.path.join(self._tmp_logdir, "_writer2_")
    writer2 = summary_writer.SummaryWriter(tmp_logdir2, max_queue=0)

    self.assertEqual(0, writer2.global_step)
    self._writer.step()
    self.assertEqual(0, writer2.global_step)
    writer2.step()
    writer2.step()
    writer2.step()
    self.assertEqual(3, writer2.global_step)

    x = constant_op.constant(1337.0)
    writer_orig_step = self._writer.global_step
    self._writer.step()
    self._writer.scalar("x", x)

    event = self._readLastEvent()
    self.assertEqual(writer_orig_step + 1, event.step)

    writer2.scalar("x", x)
    event = self._readLastEvent(tmp_logdir2)
    self.assertEqual(3, event.step)

    self._writer.step()
    self._writer.scalar("x", x)

    event = self._readLastEvent()
    self.assertEqual(writer_orig_step + 2, event.step)


# TODO(cais): Add performance benchmark for SummaryWriter.


if __name__ == "__main__":
  test.main()
