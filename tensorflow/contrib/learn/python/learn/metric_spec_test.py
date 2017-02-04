# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for MetricSpec."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import sys

# pylint: disable=g-bad-todo,g-import-not-at-top
# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
from tensorflow.python.platform import test


class MetricSpecTest(test.TestCase):

  def test_named_args_with_weights(self):
    features = {"f1": "f1_value", "f2": "f2_value"}
    labels_ = {"l1": "l1_value", "l2": "l2_value"}
    predictions_ = {"p1": "p1_value", "p2": "p2_value"}

    def _fn0(predictions, labels, weights=None):
      self.assertEqual("p1_value", predictions)
      self.assertEqual("l1_value", labels)
      self.assertEqual("f2_value", weights)
      return "metric_fn_result"

    def _fn1(predictions, targets, weights=None):
      self.assertEqual("p1_value", predictions)
      self.assertEqual("l1_value", targets)
      self.assertEqual("f2_value", weights)
      return "metric_fn_result"

    def _fn2(prediction, label, weight=None):
      self.assertEqual("p1_value", prediction)
      self.assertEqual("l1_value", label)
      self.assertEqual("f2_value", weight)
      return "metric_fn_result"

    def _fn3(prediction, target, weight=None):
      self.assertEqual("p1_value", prediction)
      self.assertEqual("l1_value", target)
      self.assertEqual("f2_value", weight)
      return "metric_fn_result"

    for fn in (_fn0, _fn1, _fn2, _fn3):
      spec = MetricSpec(
          metric_fn=fn, prediction_key="p1", label_key="l1", weight_key="f2")
      self.assertEqual(
          "metric_fn_result",
          spec.create_metric_ops(features, labels_, predictions_))

  def test_no_args(self):
    def _fn():
      self.fail("Expected failure before metric_fn.")

    spec = MetricSpec(metric_fn=_fn)
    with self.assertRaises(TypeError):
      spec.create_metric_ops(
          {"f1": "f1_value"}, "labels_value", "predictions_value")

  def test_kwargs(self):
    features = {"f1": "f1_value"}
    labels_ = "labels_value"
    predictions_ = "predictions_value"

    def _fn(**kwargs):
      self.assertEqual({}, kwargs)
      return "metric_fn_result"

    spec = MetricSpec(metric_fn=_fn)
    with self.assertRaises(TypeError):
      spec.create_metric_ops(features, labels_, predictions_)

  def test_named_labels_no_predictions(self):
    features = {"f1": "f1_value"}
    labels_ = "labels_value"
    predictions_ = "predictions_value"

    def _fn(labels):
      self.assertEqual(labels_, labels)
      return "metric_fn_result"

    spec = MetricSpec(metric_fn=_fn)
    with self.assertRaises(TypeError):
      spec.create_metric_ops(features, labels_, predictions_)

  def test_named_labels_no_predictions_with_kwargs(self):
    features = {"f1": "f1_value"}
    labels_ = "labels_value"
    predictions_ = "predictions_value"

    def _fn(labels, **kwargs):
      self.assertEqual(labels_, labels)
      self.assertEqual({}, kwargs)
      return "metric_fn_result"

    spec = MetricSpec(metric_fn=_fn)
    with self.assertRaises(TypeError):
      spec.create_metric_ops(features, labels_, predictions_)

  def test_no_named_predictions_named_labels_first_arg(self):
    features = {"f1": "f1_value"}
    labels_ = "labels_value"
    predictions_ = "predictions_value"

    def _fn(labels, predictions_by_another_name):
      self.assertEqual(predictions_, predictions_by_another_name)
      self.assertEqual(labels_, labels)
      return "metric_fn_result"

    spec = MetricSpec(metric_fn=_fn)
    self.assertEqual(
        "metric_fn_result",
        spec.create_metric_ops(features, labels_, predictions_))

  def test_no_named_predictions_named_labels_second_arg(self):
    features = {"f1": "f1_value"}
    labels_ = "labels_value"
    predictions_ = "predictions_value"

    def _fn(predictions_by_another_name, labels):
      self.assertEqual(predictions_, predictions_by_another_name)
      self.assertEqual(labels_, labels)
      return "metric_fn_result"

    spec = MetricSpec(metric_fn=_fn)
    self.assertEqual(
        "metric_fn_result",
        spec.create_metric_ops(features, labels_, predictions_))

  def test_no_named_labels(self):
    features = {"f1": "f1_value"}
    labels_ = "labels_value"
    predictions_ = "predictions_value"

    def _fn(predictions):
      self.assertEqual(predictions_, predictions)
      return "metric_fn_result"

    spec = MetricSpec(metric_fn=_fn)
    self.assertEqual(
        "metric_fn_result",
        spec.create_metric_ops(features, labels_, predictions_))

  def test_no_named_labels_or_predictions_1arg(self):
    features = {"f1": "f1_value"}
    labels_ = "labels_value"
    predictions_ = "predictions_value"

    def _fn(a):
      self.assertEqual(predictions_, a)
      return "metric_fn_result"

    spec = MetricSpec(metric_fn=_fn)
    self.assertEqual(
        "metric_fn_result",
        spec.create_metric_ops(features, labels_, predictions_))

  def test_no_named_labels_or_predictions_2args(self):
    features = {"f1": "f1_value"}
    labels_ = "labels_value"
    predictions_ = "predictions_value"

    def _fn(a, b):
      del a, b
      self.fail("Expected failure before metric_fn.")

    spec = MetricSpec(metric_fn=_fn)
    with self.assertRaises(TypeError):
      spec.create_metric_ops(features, labels_, predictions_)

  def test_named_args_no_weights(self):
    features = {"f1": "f1_value", "f2": "f2_value"}
    labels_ = {"l1": "l1_value", "l2": "l2_value"}
    predictions_ = {"p1": "p1_value", "p2": "p2_value"}

    def _fn0(predictions, labels):
      self.assertEqual("p1_value", predictions)
      self.assertEqual("l1_value", labels)
      return "metric_fn_result"

    def _fn1(predictions, targets):
      self.assertEqual("p1_value", predictions)
      self.assertEqual("l1_value", targets)
      return "metric_fn_result"

    def _fn2(prediction, label):
      self.assertEqual("p1_value", prediction)
      self.assertEqual("l1_value", label)
      return "metric_fn_result"

    def _fn3(prediction, target):
      self.assertEqual("p1_value", prediction)
      self.assertEqual("l1_value", target)
      return "metric_fn_result"

    for fn in (_fn0, _fn1, _fn2, _fn3):
      spec = MetricSpec(metric_fn=fn, prediction_key="p1", label_key="l1")
      self.assertEqual(
          "metric_fn_result",
          spec.create_metric_ops(features, labels_, predictions_))

  def test_predictions_dict_no_key(self):
    features = {"f1": "f1_value", "f2": "f2_value"}
    labels = {"l1": "l1_value", "l2": "l2_value"}
    predictions = {"p1": "p1_value", "p2": "p2_value"}

    def _fn(predictions, labels, weights=None):
      del labels, predictions, weights
      self.fail("Expected failure before metric_fn.")

    spec = MetricSpec(metric_fn=_fn, label_key="l1", weight_key="f2")
    with self.assertRaisesRegexp(
        ValueError,
        "MetricSpec without specified prediction_key requires predictions"
        " tensor or single element dict"):
      spec.create_metric_ops(features, labels, predictions)

  def test_labels_dict_no_key(self):
    features = {"f1": "f1_value", "f2": "f2_value"}
    labels = {"l1": "l1_value", "l2": "l2_value"}
    predictions = {"p1": "p1_value", "p2": "p2_value"}

    def _fn(labels, predictions, weights=None):
      del labels, predictions, weights
      self.fail("Expected failure before metric_fn.")

    spec = MetricSpec(metric_fn=_fn, prediction_key="p1", weight_key="f2")
    with self.assertRaisesRegexp(
        ValueError,
        "MetricSpec without specified label_key requires labels tensor or"
        " single element dict"):
      spec.create_metric_ops(features, labels, predictions)

  def test_single_prediction(self):
    features = {"f1": "f1_value", "f2": "f2_value"}
    labels_ = {"l1": "l1_value", "l2": "l2_value"}
    predictions_ = "p1_value"

    def _fn(predictions, labels, weights=None):
      self.assertEqual(predictions_, predictions)
      self.assertEqual("l1_value", labels)
      self.assertEqual("f2_value", weights)
      return "metric_fn_result"

    spec = MetricSpec(metric_fn=_fn, label_key="l1", weight_key="f2")
    self.assertEqual(
        "metric_fn_result",
        spec.create_metric_ops(features, labels_, predictions_))

  def test_single_label(self):
    features = {"f1": "f1_value", "f2": "f2_value"}
    labels_ = "l1_value"
    predictions_ = {"p1": "p1_value", "p2": "p2_value"}

    def _fn(predictions, labels, weights=None):
      self.assertEqual("p1_value", predictions)
      self.assertEqual(labels_, labels)
      self.assertEqual("f2_value", weights)
      return "metric_fn_result"

    spec = MetricSpec(metric_fn=_fn, prediction_key="p1", weight_key="f2")
    self.assertEqual(
        "metric_fn_result",
        spec.create_metric_ops(features, labels_, predictions_))

  def test_single_predictions_with_key(self):
    features = {"f1": "f1_value", "f2": "f2_value"}
    labels = {"l1": "l1_value", "l2": "l2_value"}
    predictions = "p1_value"

    def _fn(predictions, labels, weights=None):
      del labels, predictions, weights
      self.fail("Expected failure before metric_fn.")

    spec = MetricSpec(
        metric_fn=_fn, prediction_key="p1", label_key="l1", weight_key="f2")
    with self.assertRaisesRegexp(
        ValueError,
        "MetricSpec with prediction_key specified requires predictions dict"):
      spec.create_metric_ops(features, labels, predictions)

  def test_single_labels_with_key(self):
    features = {"f1": "f1_value", "f2": "f2_value"}
    labels = "l1_value"
    predictions = {"p1": "p1_value", "p2": "p2_value"}

    def _fn(predictions, labels, weights=None):
      del labels, predictions, weights
      self.fail("Expected failure before metric_fn.")

    spec = MetricSpec(
        metric_fn=_fn, prediction_key="p1", label_key="l1", weight_key="f2")
    with self.assertRaisesRegexp(
        ValueError, "MetricSpec with label_key specified requires labels dict"):
      spec.create_metric_ops(features, labels, predictions)

  def test_str(self):
    def _metric_fn(labels, predictions, weights=None):
      return predictions, labels, weights

    string = str(MetricSpec(
        metric_fn=_metric_fn,
        label_key="my_label",
        prediction_key="my_prediction",
        weight_key="my_weight"))
    self.assertIn("_metric_fn", string)
    self.assertIn("my_label", string)
    self.assertIn("my_prediction", string)
    self.assertIn("my_weight", string)

  def test_partial_str(self):

    def custom_metric(predictions, labels, stuff, weights=None):
      return predictions, labels, weights, stuff

    string = str(MetricSpec(
        metric_fn=functools.partial(custom_metric, stuff=5),
        label_key="my_label",
        prediction_key="my_prediction",
        weight_key="my_weight"))
    self.assertIn("custom_metric", string)
    self.assertIn("my_label", string)
    self.assertIn("my_prediction", string)
    self.assertIn("my_weight", string)

  def test_partial(self):
    features = {"f1": "f1_value", "f2": "f2_value"}
    labels = {"l1": "l1_value"}
    predictions = {"p1": "p1_value", "p2": "p2_value"}

    def custom_metric(predictions, labels, stuff, weights=None):
      self.assertEqual("p1_value", predictions)
      self.assertEqual("l1_value", labels)
      self.assertEqual("f2_value", weights)
      if stuff:
        return "metric_fn_result"
      raise ValueError("No stuff.")

    spec = MetricSpec(
        metric_fn=functools.partial(custom_metric, stuff=5),
        label_key="l1",
        prediction_key="p1",
        weight_key="f2")
    self.assertEqual(
        "metric_fn_result",
        spec.create_metric_ops(features, labels, predictions))

    spec = MetricSpec(
        metric_fn=functools.partial(custom_metric, stuff=None),
        prediction_key="p1", label_key="l1", weight_key="f2")
    with self.assertRaisesRegexp(ValueError, "No stuff."):
      spec.create_metric_ops(features, labels, predictions)

  def test_label_key_without_label_arg(self):
    def _fn0(predictions, weights=None):
      del predictions, weights
      self.fail("Expected failure before metric_fn.")

    def _fn1(prediction, weight=None):
      del prediction, weight
      self.fail("Expected failure before metric_fn.")

    for fn in (_fn0, _fn1):
      with self.assertRaisesRegexp(ValueError, "label.*missing"):
        MetricSpec(metric_fn=fn, label_key="l1")

  def test_weight_key_without_weight_arg(self):
    def _fn0(predictions, labels):
      del predictions, labels
      self.fail("Expected failure before metric_fn.")

    def _fn1(prediction, label):
      del prediction, label
      self.fail("Expected failure before metric_fn.")

    def _fn2(predictions, targets):
      del predictions, targets
      self.fail("Expected failure before metric_fn.")

    def _fn3(prediction, target):
      del prediction, target
      self.fail("Expected failure before metric_fn.")

    for fn in (_fn0, _fn1, _fn2, _fn3):
      with self.assertRaisesRegexp(ValueError, "weight.*missing"):
        MetricSpec(metric_fn=fn, weight_key="f2")

  def test_multiple_label_args(self):
    def _fn0(predictions, labels, targets):
      del predictions, labels, targets
      self.fail("Expected failure before metric_fn.")

    def _fn1(prediction, label, target):
      del prediction, label, target
      self.fail("Expected failure before metric_fn.")

    for fn in (_fn0, _fn1):
      with self.assertRaisesRegexp(ValueError, "provide only one of.*label"):
        MetricSpec(metric_fn=fn)

  def test_multiple_prediction_args(self):
    def _fn(predictions, prediction, labels):
      del predictions, prediction, labels
      self.fail("Expected failure before metric_fn.")

    with self.assertRaisesRegexp(ValueError, "provide only one of.*prediction"):
      MetricSpec(metric_fn=_fn)

  def test_multiple_weight_args(self):
    def _fn(predictions, labels, weights=None, weight=None):
      del predictions, labels, weights, weight
      self.fail("Expected failure before metric_fn.")

    with self.assertRaisesRegexp(ValueError, "provide only one of.*weight"):
      MetricSpec(metric_fn=_fn)

if __name__ == "__main__":
  test.main()
