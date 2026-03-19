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
"""Tests for the actual serialized proto output of the V1 tf.summary ops.

The tensor, audio, and image ops have dedicated tests in adjacent files. The
overall tf.summary API surface also has its own tests in summary_test.py that
check calling the API methods but not the exact serialized proto output.
"""

from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import logging_ops
from tensorflow.python.platform import test
from tensorflow.python.summary import summary


class SummaryV1OpsTest(test.TestCase):

  def _AsSummary(self, s):
    summ = summary_pb2.Summary()
    summ.ParseFromString(s)
    return summ

  def testScalarSummary(self):
    with self.cached_session() as sess:
      const = constant_op.constant([10.0, 20.0])
      summ = logging_ops.scalar_summary(["c1", "c2"], const, name="mysumm")
      value = self.evaluate(summ)
    self.assertEqual([], summ.get_shape())
    self.assertProtoEquals("""
      value { tag: "c1" simple_value: 10.0 }
      value { tag: "c2" simple_value: 20.0 }
      """, self._AsSummary(value))

  def testScalarSummaryDefaultName(self):
    with self.cached_session() as sess:
      const = constant_op.constant([10.0, 20.0])
      summ = logging_ops.scalar_summary(["c1", "c2"], const)
      value = self.evaluate(summ)
    self.assertEqual([], summ.get_shape())
    self.assertProtoEquals("""
      value { tag: "c1" simple_value: 10.0 }
      value { tag: "c2" simple_value: 20.0 }
      """, self._AsSummary(value))

  @test_util.run_deprecated_v1
  def testMergeSummary(self):
    with self.cached_session() as sess:
      const = constant_op.constant(10.0)
      summ1 = summary.histogram("h", const)
      summ2 = logging_ops.scalar_summary("c", const)
      merge = summary.merge([summ1, summ2])
      value = self.evaluate(merge)
    self.assertEqual([], merge.get_shape())
    self.assertProtoEquals("""
      value {
        tag: "h"
        histo {
          min: 10.0
          max: 10.0
          num: 1.0
          sum: 10.0
          sum_squares: 100.0
          bucket_limit: 9.93809490288
          bucket_limit: 10.9319043932
          bucket_limit: 1.7976931348623157e+308
          bucket: 0.0
          bucket: 1.0
          bucket: 0.0
        }
      }
      value { tag: "c" simple_value: 10.0 }
    """, self._AsSummary(value))

  def testMergeAllSummaries(self):
    with ops.Graph().as_default():
      const = constant_op.constant(10.0)
      summ1 = summary.histogram("h", const)
      summ2 = summary.scalar("o", const, collections=["foo_key"])
      summ3 = summary.scalar("c", const)
      merge = summary.merge_all()
      self.assertEqual("MergeSummary", merge.op.type)
      self.assertEqual(2, len(merge.op.inputs))
      self.assertEqual(summ1, merge.op.inputs[0])
      self.assertEqual(summ3, merge.op.inputs[1])
      merge = summary.merge_all("foo_key")
      self.assertEqual("MergeSummary", merge.op.type)
      self.assertEqual(1, len(merge.op.inputs))
      self.assertEqual(summ2, merge.op.inputs[0])
      self.assertTrue(summary.merge_all("bar_key") is None)


if __name__ == "__main__":
  test.main()
