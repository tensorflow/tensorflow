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
"""Generate some standard test data for debugging TensorBoard.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bisect
import math
import os
import os.path
import random
import shutil

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.summary.writer import writer as writer_lib

tf.flags.DEFINE_string("target", None, """The directoy where serialized data
will be written""")

flags.DEFINE_boolean("overwrite", False, """Whether to remove and overwrite
TARGET if it already exists.""")

FLAGS = tf.flags.FLAGS

# Hardcode a start time and reseed so script always generates the same data.
_start_time = 0
random.seed(0)


def _MakeHistogramBuckets():
  v = 1E-12
  buckets = []
  neg_buckets = []
  while v < 1E20:
    buckets.append(v)
    neg_buckets.append(-v)
    v *= 1.1
  # Should include DBL_MAX, but won't bother for test data.
  return neg_buckets[::-1] + [0] + buckets


def _MakeHistogram(values):
  """Convert values into a histogram proto using logic from histogram.cc."""
  limits = _MakeHistogramBuckets()
  counts = [0] * len(limits)
  for v in values:
    idx = bisect.bisect_left(limits, v)
    counts[idx] += 1

  limit_counts = [(limits[i], counts[i]) for i in xrange(len(limits))
                  if counts[i]]
  bucket_limit = [lc[0] for lc in limit_counts]
  bucket = [lc[1] for lc in limit_counts]
  sum_sq = sum(v * v for v in values)
  return summary_pb2.HistogramProto(
      min=min(values),
      max=max(values),
      num=len(values),
      sum=sum(values),
      sum_squares=sum_sq,
      bucket_limit=bucket_limit,
      bucket=bucket)


def WriteScalarSeries(writer, tag, f, n=5):
  """Write a series of scalar events to writer, using f to create values."""
  step = 0
  wall_time = _start_time
  for i in xrange(n):
    v = f(i)
    value = summary_pb2.Summary.Value(tag=tag, simple_value=v)
    summary = summary_pb2.Summary(value=[value])
    event = event_pb2.Event(wall_time=wall_time, step=step, summary=summary)
    writer.add_event(event)
    step += 1
    wall_time += 10


def WriteHistogramSeries(writer, tag, mu_sigma_tuples, n=20):
  """Write a sequence of normally distributed histograms to writer."""
  step = 0
  wall_time = _start_time
  for [mean, stddev] in mu_sigma_tuples:
    data = [random.normalvariate(mean, stddev) for _ in xrange(n)]
    histo = _MakeHistogram(data)
    summary = summary_pb2.Summary(
        value=[summary_pb2.Summary.Value(
            tag=tag, histo=histo)])
    event = event_pb2.Event(wall_time=wall_time, step=step, summary=summary)
    writer.add_event(event)
    step += 10
    wall_time += 100


def WriteImageSeries(writer, tag, n_images=1):
  """Write a few dummy images to writer."""
  step = 0
  session = session_lib.Session()
  p = array_ops.placeholder("uint8", (1, 4, 4, 3))
  s = logging_ops.image_summary(tag, p)
  for _ in xrange(n_images):
    im = np.random.random_integers(0, 255, (1, 4, 4, 3))
    summ = session.run(s, feed_dict={p: im})
    writer.add_summary(summ, step)
    step += 20
  session.close()


def WriteAudioSeries(writer, tag, n_audio=1):
  """Write a few dummy audio clips to writer."""
  step = 0
  session = session_lib.Session()

  min_frequency_hz = 440
  max_frequency_hz = 880
  sample_rate = 4000
  duration_frames = sample_rate * 0.5  # 0.5 seconds.
  frequencies_per_run = 1
  num_channels = 2

  p = array_ops.placeholder("float32", (frequencies_per_run, duration_frames,
                                        num_channels))
  s = logging_ops.audio_summary(tag, p, sample_rate)

  for _ in xrange(n_audio):
    # Generate a different frequency for each channel to show stereo works.
    frequencies = np.random.random_integers(
        min_frequency_hz,
        max_frequency_hz,
        size=(frequencies_per_run, num_channels))
    tiled_frequencies = np.tile(frequencies, (1, duration_frames))
    tiled_increments = np.tile(
        np.arange(0, duration_frames),
        (num_channels, 1)).T.reshape(1, duration_frames * num_channels)
    tones = np.sin(2.0 * np.pi * tiled_frequencies * tiled_increments /
                   sample_rate)
    tones = tones.reshape(frequencies_per_run, duration_frames, num_channels)

    summ = session.run(s, feed_dict={p: tones})
    writer.add_summary(summ, step)
    step += 20
  session.close()


def GenerateTestData(path):
  """Generates the test data directory."""
  run1_path = os.path.join(path, "run1")
  os.makedirs(run1_path)
  writer1 = writer_lib.FileWriter(run1_path)
  WriteScalarSeries(writer1, "foo/square", lambda x: x * x)
  WriteScalarSeries(writer1, "bar/square", lambda x: x * x)
  WriteScalarSeries(writer1, "foo/sin", math.sin)
  WriteScalarSeries(writer1, "foo/cos", math.cos)
  WriteHistogramSeries(writer1, "histo1", [[0, 1], [0.3, 1], [0.5, 1], [0.7, 1],
                                           [1, 1]])
  WriteImageSeries(writer1, "im1")
  WriteImageSeries(writer1, "im2")
  WriteAudioSeries(writer1, "au1")

  run2_path = os.path.join(path, "run2")
  os.makedirs(run2_path)
  writer2 = writer_lib.FileWriter(run2_path)
  WriteScalarSeries(writer2, "foo/square", lambda x: x * x * 2)
  WriteScalarSeries(writer2, "bar/square", lambda x: x * x * 3)
  WriteScalarSeries(writer2, "foo/cos", lambda x: math.cos(x) * 2)
  WriteHistogramSeries(writer2, "histo1", [[0, 2], [0.3, 2], [0.5, 2], [0.7, 2],
                                           [1, 2]])
  WriteHistogramSeries(writer2, "histo2", [[0, 1], [0.3, 1], [0.5, 1], [0.7, 1],
                                           [1, 1]])
  WriteImageSeries(writer2, "im1")
  WriteAudioSeries(writer2, "au2")

  graph_def = graph_pb2.GraphDef()
  node1 = graph_def.node.add()
  node1.name = "a"
  node1.op = "matmul"
  node2 = graph_def.node.add()
  node2.name = "b"
  node2.op = "matmul"
  node2.input.extend(["a:0"])

  writer1.add_graph(graph_def)
  node3 = graph_def.node.add()
  node3.name = "c"
  node3.op = "matmul"
  node3.input.extend(["a:0", "b:0"])
  writer2.add_graph(graph_def)
  writer1.close()
  writer2.close()


def main(unused_argv=None):
  target = FLAGS.target
  if not target:
    print("The --target flag is required.")
    return -1
  if os.path.exists(target):
    if FLAGS.overwrite:
      if os.path.isdir(target):
        shutil.rmtree(target)
      else:
        os.remove(target)
    else:
      print("Refusing to overwrite target %s without --overwrite" % target)
      return -2
  GenerateTestData(target)


if __name__ == "__main__":
  app.run()
