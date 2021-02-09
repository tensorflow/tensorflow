# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Keras metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras import metrics
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def _labeled_dataset_fn():
  # First four batches of x: labels, predictions -> (labels == predictions)
  #  0: 0, 0 -> True;   1: 1, 1 -> True;   2: 2, 2 -> True;   3: 3, 0 -> False
  #  4: 4, 1 -> False;  5: 0, 2 -> False;  6: 1, 0 -> False;  7: 2, 1 -> False
  #  8: 3, 2 -> False;  9: 4, 0 -> False; 10: 0, 1 -> False; 11: 1, 2 -> False
  # 12: 2, 0 -> False; 13: 3, 1 -> False; 14: 4, 2 -> False; 15: 0, 0 -> True
  return dataset_ops.Dataset.range(1000).map(
      lambda x: {"labels": x % 5, "predictions": x % 3}).batch(
          4, drop_remainder=True)


def _boolean_dataset_fn():
  # First four batches of labels, predictions: {TP, FP, TN, FN}
  # with a threshold of 0.5:
  #   T, T -> TP;  F, T -> FP;   T, F -> FN
  #   F, F -> TN;  T, T -> TP;   F, T -> FP
  #   T, F -> FN;  F, F -> TN;   T, T -> TP
  #   F, T -> FP;  T, F -> FN;   F, F -> TN
  return dataset_ops.Dataset.from_tensor_slices({
      "labels": [True, False, True, False],
      "predictions": [True, True, False, False]}).repeat().batch(
          3, drop_remainder=True)


def _threshold_dataset_fn():
  # First four batches of labels, predictions: {TP, FP, TN, FN}
  # with a threshold of 0.5:
  #   True, 1.0 -> TP;  False, .75 -> FP;   True, .25 -> FN
  #  False, 0.0 -> TN;   True, 1.0 -> TP;  False, .75 -> FP
  #   True, .25 -> FN;  False, 0.0 -> TN;   True, 1.0 -> TP
  #  False, .75 -> FP;   True, .25 -> FN;  False, 0.0 -> TN
  return dataset_ops.Dataset.from_tensor_slices({
      "labels": [True, False, True, False],
      "predictions": [1.0, 0.75, 0.25, 0.]}).repeat().batch(
          3, drop_remainder=True)


def _regression_dataset_fn():
  return dataset_ops.Dataset.from_tensor_slices({
      "labels": [1., .5, 1., 0.],
      "predictions": [1., .75, .25, 0.]}).repeat()


def all_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.default_strategy,
          strategy_combinations.one_device_strategy,
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.mirrored_strategy_with_two_gpus,
      ],
      mode=["graph"])


def tpu_combinations():
  return combinations.combine(
      distribution=[strategy_combinations.tpu_strategy,],
      mode=["graph"])


class KerasMetricsTest(test.TestCase, parameterized.TestCase):

  def _test_metric(self, distribution, dataset_fn, metric_init_fn, expected_fn):
    with ops.Graph().as_default(), distribution.scope():
      metric = metric_init_fn()

      iterator = distribution.make_input_fn_iterator(lambda _: dataset_fn())
      updates = distribution.experimental_local_results(
          distribution.run(metric, args=(iterator.get_next(),)))
      batches_per_update = distribution.num_replicas_in_sync

      self.evaluate(iterator.initializer)
      self.evaluate([v.initializer for v in metric.variables])

      batches_consumed = 0
      for i in range(4):
        batches_consumed += batches_per_update
        self.evaluate(updates)
        self.assertAllClose(expected_fn(batches_consumed),
                            self.evaluate(metric.result()),
                            0.001,
                            msg="After update #" + str(i+1))
        if batches_consumed >= 4:  # Consume 4 input batches in total.
          break

  @ds_combinations.generate(all_combinations() + tpu_combinations())
  def testMean(self, distribution):
    def _dataset_fn():
      return dataset_ops.Dataset.range(1000).map(math_ops.to_float).batch(
          4, drop_remainder=True)

    def _expected_fn(num_batches):
      # Mean(0..3) = 1.5, Mean(0..7) = 3.5, Mean(0..11) = 5.5, etc.
      return num_batches * 2 - 0.5

    self._test_metric(distribution, _dataset_fn, metrics.Mean, _expected_fn)


if __name__ == "__main__":
  test.main()
