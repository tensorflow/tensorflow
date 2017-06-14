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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.tensorboard.plugins.distributions import compressor


def _make_expected_value(*values):
  return [compressor.CompressedHistogramValue(bp, val) for bp, val in values]


class HistcompTest(tf.test.TestCase):

  def testExample(self):
    bps = (0, 2500, 5000, 7500, 10000)
    proto = tf.HistogramProto(
        min=1,
        max=2,
        num=3,
        sum=4,
        sum_squares=5,
        bucket_limit=[1, 2, 3],
        bucket=[0, 3, 0])
    self.assertEqual(
        _make_expected_value(
            (0, 1.0),
            (2500, 1.25),
            (5000, 1.5),
            (7500, 1.75),
            (10000, 2.0)),
        compressor.CompressHistogram(proto, bps))

  def testAnotherExample(self):
    bps = (0, 2500, 5000, 7500, 10000)
    proto = tf.HistogramProto(
        min=-2,
        max=3,
        num=4,
        sum=5,
        sum_squares=6,
        bucket_limit=[2, 3, 4],
        bucket=[1, 3, 0])
    self.assertEqual(
        _make_expected_value(
            (0, -2),
            (2500, 2),
            (5000, 2 + 1 / 3),
            (7500, 2 + 2 / 3),
            (10000, 3)),
        compressor.CompressHistogram(proto, bps))

  def testEmpty(self):
    bps = (0, 2500, 5000, 7500, 10000)
    proto = tf.HistogramProto(
        min=None,
        max=None,
        num=0,
        sum=0,
        sum_squares=0,
        bucket_limit=[1, 2, 3],
        bucket=[0, 0, 0])
    self.assertEqual(
        _make_expected_value(
            (0, 0),
            (2500, 0),
            (5000, 0),
            (7500, 0),
            (10000, 0)),
        compressor.CompressHistogram(proto, bps))

  def testUgly(self):
    bps = (0, 668, 1587, 3085, 5000, 6915, 8413, 9332, 10000)
    proto = tf.HistogramProto(
        min=0.0,
        max=1.0,
        num=960.0,
        sum=64.0,
        sum_squares=64.0,
        bucket_limit=[0.0, 1e-12, 0.917246389039776, 1.0089710279437536,
                      1.7976931348623157e+308],
        bucket=[0.0, 896.0, 0.0, 64.0, 0.0])
    vals = compressor.CompressHistogram(proto, bps)
    self.assertEquals(tuple(v.basis_point for v in vals), bps)
    self.assertAlmostEqual(vals[0].value, 0.0)
    self.assertAlmostEqual(vals[1].value, 7.157142857142856e-14)
    self.assertAlmostEqual(vals[2].value, 1.7003571428571426e-13)
    self.assertAlmostEqual(vals[3].value, 3.305357142857143e-13)
    self.assertAlmostEqual(vals[4].value, 5.357142857142857e-13)
    self.assertAlmostEqual(vals[5].value, 7.408928571428571e-13)
    self.assertAlmostEqual(vals[6].value, 9.013928571428571e-13)
    self.assertAlmostEqual(vals[7].value, 9.998571428571429e-13)
    self.assertAlmostEqual(vals[8].value, 1.0)


if __name__ == '__main__':
  tf.test.main()
