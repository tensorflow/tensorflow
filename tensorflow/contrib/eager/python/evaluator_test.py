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
"""Tests for class Evaluator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.eager.python import evaluator
from tensorflow.contrib.eager.python import metrics
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import test


class IdentityModel(object):

  def eval_data(self, d):
    return d


class PrefixLModel(object):

  def eval_data(self, d):
    return {"l_" + key: d[key] for key in d}


class SimpleEvaluator(evaluator.Evaluator):

  def __init__(self, model):
    super(SimpleEvaluator, self).__init__(model)
    self.mean = self.add_metric(metrics.Mean("mean"))

  def call(self, eval_data):
    self.mean(eval_data)


class DelegatingEvaluator(evaluator.Evaluator):

  def __init__(self, model):
    super(DelegatingEvaluator, self).__init__(model)
    self.sub = self.add_evaluator("inner", SimpleEvaluator(model))
    self.mean = self.add_metric(metrics.Mean("outer-mean"))

  def call(self, eval_data):
    # Keys here come from PrefixLModel, which adds "l_".
    self.mean(eval_data["l_outer"])
    self.sub.call(eval_data["l_inner"])


# pylint: disable=not-callable
class EvaluatorTest(test.TestCase):

  def testSimple(self):
    e = SimpleEvaluator(IdentityModel())
    e(3.0)
    e([5.0, 7.0, 9.0])
    results = e.all_metric_results()
    self.assertEqual(set(["mean"]), set(results.keys()))
    self.assertEqual(6.0, results["mean"].numpy())

  def testComposition(self):
    e = DelegatingEvaluator(PrefixLModel())
    e({"inner": 2.0, "outer": 100.0})
    e({"inner": 4.0, "outer": 1000.0})
    results = e.all_metric_results()
    self.assertEqual(set(["inner/mean", "outer-mean"]), set(results.keys()))
    self.assertEqual(3.0, results["inner/mean"].numpy())
    self.assertEqual(550.0, results["outer-mean"].numpy())

  def testMetricVariables(self):
    e = DelegatingEvaluator(PrefixLModel())
    e({"inner": 2.0, "outer": 100.0})
    prefix_count = {}
    for v in e.metric_variables:
      p = v.name.split("/")[0]
      prefix_count[p] = prefix_count.get(p, 0) + 1
    self.assertEqual({"outer_mean": 2, "mean": 2}, prefix_count)

  def testDataset(self):
    e = SimpleEvaluator(IdentityModel())
    ds = dataset_ops.Dataset.from_tensor_slices([3.0, 5.0, 7.0, 9.0])
    results = e.evaluate_on_dataset(ds)
    self.assertEqual(set(["mean"]), set(results.keys()))
    self.assertEqual(6.0, results["mean"].numpy())

  def testModelProperty(self):
    m = IdentityModel()
    e = SimpleEvaluator(m)
    self.assertIs(m, e.model)

  def testMetricsProperty(self):
    e = DelegatingEvaluator(PrefixLModel())
    names = set([m.name for m in e.metrics])
    self.assertEqual(set(["outer-mean", "mean"]), names)


class SparseSoftmaxEvaluatorTest(test.TestCase):

  def testSimple(self):
    e = evaluator.SparseSoftmaxEvaluator(IdentityModel())
    e({e.loss_key: 1.0, e.label_key: 5, e.predicted_class_key: 5})
    e({e.loss_key: [0.0, 3.0, 4.0],
       e.label_key: [1, 2, 3],
       e.predicted_class_key: [1, 1, 3]})
    results = e.all_metric_results()
    self.assertEqual(set(["Avg_Loss", "Accuracy"]), set(results.keys()))
    self.assertEqual(2.0, results["Avg_Loss"].numpy())
    self.assertEqual(0.75, results["Accuracy"].numpy())


if __name__ == "__main__":
  test.main()
