# Copyright 2015 Google Inc. All Rights Reserved.
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

import base64
import bisect
import math
import os
import os.path
import random
import shutil

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

tf.flags.DEFINE_string("target", None, """The directoy where serialized data
will be written""")

tf.flags.DEFINE_boolean("overwrite", False, """Whether to remove and overwrite
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
  return tf.HistogramProto(min=min(values),
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
    value = tf.Summary.Value(tag=tag, simple_value=v)
    summary = tf.Summary(value=[value])
    event = tf.Event(wall_time=wall_time, step=step, summary=summary)
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
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=histo)])
    event = tf.Event(wall_time=wall_time, step=step, summary=summary)
    writer.add_event(event)
    step += 10
    wall_time += 100


def WriteImageSeries(writer, tag, n_images=1):
  """Write a few dummy images to writer."""
  # 1x1 transparent GIF.
  encoded_image = base64.b64decode(
      "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")
  image_value = tf.Summary.Image(height=1,
                                 width=1,
                                 colorspace=1,
                                 encoded_image_string=encoded_image)
  step = 0
  wall_time = _start_time
  for _ in xrange(n_images):
    value = tf.Summary.Value(tag=tag, image=image_value)
    event = tf.Event(wall_time=wall_time,
                     step=step,
                     summary=tf.Summary(value=[value]))
    writer.add_event(event)
    step += 20
    wall_time += 200


def GenerateTestData(path):
  """Generates the test data directory."""
  run1_path = os.path.join(path, "run1")
  os.makedirs(run1_path)
  writer = tf.train.SummaryWriter(run1_path)
  WriteScalarSeries(writer, "cross_entropy (1)", lambda x: x*x)
  WriteHistogramSeries(writer, "histo1", [[0, 1], [0.3, 1], [0.5, 1], [0.7, 1],
                                          [1, 1]])
  WriteImageSeries(writer, "im1")
  writer.close()


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
  tf.app.run()
