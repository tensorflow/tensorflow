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
"""Tests for tf 2.0 upgrader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import os
import tempfile

import six
import tensorflow as tf
# OSS TF V2 import placeholder.

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export
from tensorflow.python.util import tf_inspect
from tensorflow.tools.common import public_api
from tensorflow.tools.common import traverse
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import tf_upgrade_v2


def get_symbol_for_name(root, name):
  name_parts = name.split(".")
  symbol = root
  # Iterate starting with second item since 1st item is "tf.".
  for part in name_parts[1:]:
    symbol = getattr(symbol, part)
  return symbol


def get_args(symbol):
  if hasattr(inspect, "signature"):
    signature = inspect.signature(symbol)
    # Ignore *args and **kwargs for now.
    return [param.name for param in signature.parameters.values()
            if param.kind == param.POSITIONAL_OR_KEYWORD]
  return tf_inspect.getargspec(symbol)[0]


def get_func_and_args_from_str(call_str):
  """Parse call string to get function and argument names.

  Args:
    call_str: Call string must be in the form:
              `tf.foo(arg1=val1, arg2=val2, ...)`.

  Returns:
    (function_name, list of arg names) tuple.
  """
  open_paren_index = call_str.find("(")
  close_paren_index = call_str.rfind(")")

  function_name = call_str[:call_str.find("(")]
  args = call_str[open_paren_index+1:close_paren_index].split(",")
  args = [arg.split("=")[0].strip() for arg in args]
  args = [arg for arg in args if arg]  # filter out empty strings
  return function_name, args


class TestUpgrade(test_util.TensorFlowTestCase):
  """Test various APIs that have been changed in 2.0.

  We also test whether a converted file is executable. test_file_v1_10.py
  aims to exhaustively test that API changes are convertible and actually
  work when run with current TensorFlow.
  """

  @classmethod
  def setUpClass(cls):
    cls.v2_symbols = {}
    cls.v1_symbols = {}
    if hasattr(tf.compat, "v2"):

      def symbol_collector(unused_path, unused_parent, children):
        for child in children:
          _, attr = tf_decorator.unwrap(child[1])
          api_names_v2 = tf_export.get_v2_names(attr)
          for name in api_names_v2:
            cls.v2_symbols["tf." + name] = attr

      visitor = public_api.PublicAPIVisitor(symbol_collector)
      traverse.traverse(tf.compat.v2, visitor)

    if hasattr(tf.compat, "v1"):

      def symbol_collector_v1(unused_path, unused_parent, children):
        for child in children:
          _, attr = tf_decorator.unwrap(child[1])
          api_names_v1 = tf_export.get_v1_names(attr)
          for name in api_names_v1:
            cls.v1_symbols["tf." + name] = attr

      visitor = public_api.PublicAPIVisitor(symbol_collector_v1)
      traverse.traverse(tf.compat.v1, visitor)

  def _upgrade(self, old_file_text):
    in_file = six.StringIO(old_file_text)
    out_file = six.StringIO()
    upgrader = ast_edits.ASTCodeUpgrader(tf_upgrade_v2.TFAPIChangeSpec())
    count, report, errors = (
        upgrader.process_opened_file("test.py", in_file,
                                     "test_out.py", out_file))
    return count, report, errors, out_file.getvalue()

  def testParseError(self):
    _, report, unused_errors, unused_new_text = self._upgrade(
        "import tensorflow as tf\na + \n")
    self.assertTrue(report.find("Failed to parse") != -1)

  def testReport(self):
    text = "tf.angle(a)\n"
    _, report, unused_errors, unused_new_text = self._upgrade(text)
    # This is not a complete test, but it is a sanity test that a report
    # is generating information.
    self.assertTrue(report.find("Renamed function `tf.angle` to "
                                "`tf.math.angle`"))

  def testRename(self):
    text = "tf.conj(a)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.math.conj(a)\n")
    text = "tf.rsqrt(tf.log_sigmoid(3.8))\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.math.rsqrt(tf.math.log_sigmoid(3.8))\n")

  def testAllAPI(self):
    if not hasattr(tf.compat, "v2"):
      return

    # Converts all symbols in the v1 namespace to the v2 namespace, raising
    # an error if the target of the conversion is not in the v2 namespace.
    # Please regenerate the renames file or edit any manual renames if this
    # test fails.
    def conversion_visitor(unused_path, unused_parent, children):
      for child in children:
        _, attr = tf_decorator.unwrap(child[1])
        api_names = tf_export.get_v1_names(attr)
        for name in api_names:
          _, _, _, text = self._upgrade("tf." + name)
          if (text and
              not text.startswith("tf.compat.v1") and
              text not in self.v2_symbols and
              # Builds currently install old version of estimator that doesn't
              # have some 2.0 symbols.
              not text.startswith("tf.estimator")):
            self.assertFalse(
                True, "Symbol %s generated from %s not in v2 API" % (
                    text, name))

    visitor = public_api.PublicAPIVisitor(conversion_visitor)
    visitor.do_not_descend_map["tf"].append("contrib")
    visitor.private_map["tf.compat"] = ["v1", "v2"]
    traverse.traverse(tf.compat.v1, visitor)

  def testAllAPIV1(self):
    collect = True
    v1_symbols = set([])

    # Symbols which may be generated by the conversion script which do not exist
    # in TF 1.x. This should be a very short list of symbols which are
    # experimental in 1.x but stable for 2.x.
    whitelisted_v2_only_symbols = set(["tf.saved_model.save"])

    # Converts all symbols in the v1 namespace to the v2 namespace, raising
    # an error if the target of the conversion is not in the v1 namespace.
    def conversion_visitor(unused_path, unused_parent, children):
      for child in children:
        _, attr = tf_decorator.unwrap(child[1])
        api_names = tf_export.get_v1_names(attr)
        for name in api_names:
          if collect:
            v1_symbols.add("tf." + name)
          else:
            _, _, _, text = self._upgrade("tf." + name)
            if (text and
                not text.startswith("tf.compat.v1") and
                not text.startswith("tf.estimator") and
                text not in v1_symbols and
                text not in whitelisted_v2_only_symbols):
              self.assertFalse(
                  True, "Symbol %s generated from %s not in v1 API" % (
                      text, name))

    visitor = public_api.PublicAPIVisitor(conversion_visitor)
    visitor.do_not_descend_map["tf"].append("contrib")
    visitor.private_map["tf.compat"] = ["v1", "v2"]
    traverse.traverse(tf.compat.v1, visitor)
    collect = False
    traverse.traverse(tf.compat.v1, visitor)

  def testV1KeywordArgNames(self):
    all_keyword_renames = (
        tf_upgrade_v2.TFAPIChangeSpec().function_keyword_renames)

    # Visitor that verifies V1 argument names.
    def arg_test_visitor(unused_path, unused_parent, children):
      for child in children:
        _, attr = tf_decorator.unwrap(child[1])
        names_v1 = tf_export.get_v1_names(attr)

        for name in names_v1:
          name = "tf.%s" % name
          if name not in all_keyword_renames:
            continue
          arg_names_v1 = tf_inspect.getargspec(attr)[0]
          keyword_renames = all_keyword_renames[name]
          self.assertEqual(type(keyword_renames), dict)

          # Assert that v1 function has valid v1 argument names.
          for from_name, _ in keyword_renames.items():
            self.assertIn(
                from_name, arg_names_v1,
                "%s not found in %s arguments: %s" %
                (from_name, name, str(arg_names_v1)))

    visitor = public_api.PublicAPIVisitor(arg_test_visitor)
    visitor.do_not_descend_map["tf"].append("contrib")
    visitor.private_map["tf.compat"] = ["v1", "v2"]
    traverse.traverse(tf.compat.v1, visitor)

  def testV2KeywordArgNames(self):
    # This test converts a call of the form:
    # tf.foo(arg1=0, arg2=1, ...)
    # to 2.0. Then, checks that converted function has valid argument names.
    if not hasattr(tf.compat, "v2"):
      return
    v2_arg_exceptions = {
        "verify_shape_is_now_always_true",
        # These arguments should not be used, they just specify
        # that a function takes named arguments.
        "keyword_required",
        "_sentinel",
    }
    v1_name_exceptions = {
        "tf.print",  # requires print_function import
    }
    function_warnings = (
        tf_upgrade_v2.TFAPIChangeSpec().function_warnings)
    function_transformers = (
        tf_upgrade_v2.TFAPIChangeSpec().function_transformers)
    keyword_renames = (
        tf_upgrade_v2.TFAPIChangeSpec().function_keyword_renames)

    # Visitor that converts to V2 and checks V2 argument names.
    def conversion_visitor(unused_path, unused_parent, children):
      for child in children:
        _, attr = tf_decorator.unwrap(child[1])
        if not tf_inspect.isfunction(attr):
          continue
        names_v1 = tf_export.get_v1_names(attr)
        arg_names_v1 = get_args(attr)

        for name in names_v1:
          tf_name = "tf.%s" % name
          if tf_name in function_warnings or tf_name in function_transformers:
            continue  # These require manual change
          if tf_name in v1_name_exceptions:
            continue
          # Assert that arg names after converting to v2 are present in
          # v2 function.
          # 1. First, create an input of the form:
          #    tf.foo(arg1=val1, arg2=val2, ...)
          args = ",".join(
              ["%s=%d" % (from_name, from_index)
               for from_index, from_name in enumerate(arg_names_v1)])
          text_input = "%s(%s)" % (tf_name, args)
          # 2. Convert the input to V2.
          _, _, _, text = self._upgrade(text_input)
          new_function_name, new_args = get_func_and_args_from_str(text)
          if new_function_name == "tf.compat.v1.%s" % name:
            if tf_name in keyword_renames:
              # If we rename arguments, new function must be available in 2.0.
              # We should not be using compat.v1 in this case.
              self.assertFalse(
                  "Function '%s' is not in 2.0 when converting\n%s\nto\n%s" %
                  (new_function_name, text_input, text))
            continue
          # 3. Verify V2 function and arguments.
          args_v2 = get_args(self.v2_symbols[new_function_name])
          args_v2.extend(v2_arg_exceptions)
          for new_arg in new_args:
            self.assertIn(
                new_arg, args_v2,
                "Invalid argument '%s' in 2.0 when converting\n%s\nto\n%s.\n"
                "Supported arguments: %s" % (
                    new_arg, text_input, text, str(args_v2)))
          # 4. Verify that the argument exists in v1 as well.
          if new_function_name in set(["tf.nn.ctc_loss",
                                       "tf.saved_model.save"]):
            continue
          args_v1 = get_args(self.v1_symbols[new_function_name])
          args_v1.extend(v2_arg_exceptions)
          for new_arg in new_args:
            self.assertIn(
                new_arg, args_v1,
                "Invalid argument '%s' in 1.0 when converting\n%s\nto\n%s.\n"
                "Supported arguments: %s" % (
                    new_arg, text_input, text, str(args_v1)))

    visitor = public_api.PublicAPIVisitor(conversion_visitor)
    visitor.do_not_descend_map["tf"].append("contrib")
    visitor.private_map["tf.compat"] = ["v1", "v2"]
    traverse.traverse(tf.compat.v1, visitor)

  def testPositionsMatchArgGiven(self):
    full_dict = tf_upgrade_v2.TFAPIChangeSpec().function_arg_warnings
    method_names = full_dict.keys()
    for method_name in method_names:
      args = full_dict[method_name].keys()
      # special case for optimizer methods
      if method_name.startswith("*."):
        method = method_name.replace("*", "tf.train.Optimizer")
      else:
        method = method_name
      method = get_symbol_for_name(tf, method)
      arg_spec = tf_inspect.getfullargspec(method)
      for (arg, pos) in args:
        # to deal with the self argument on methods on objects
        if method_name.startswith("*."):
          pos += 1
        self.assertEqual(arg_spec[0][pos], arg)

  def testReorderFileNeedsUpdate(self):
    reordered_function_names = (
        tf_upgrade_v2.TFAPIChangeSpec().reordered_function_names)
    function_reorders = (
        tf_upgrade_v2.TFAPIChangeSpec().function_reorders)

    added_names_message = """Some function names in
self.reordered_function_names are not in reorders_v2.py.
Please run the following commands to update reorders_v2.py:
bazel build tensorflow/tools/compatibility/update:generate_v2_reorders_map
bazel-bin/tensorflow/tools/compatibility/update/generate_v2_reorders_map
"""
    removed_names_message = """%s in self.reorders_v2 does not match
any name in self.reordered_function_names.
Please run the following commands to update reorders_v2.py:
bazel build tensorflow/tools/compatibility/update:generate_v2_reorders_map
bazel-bin/tensorflow/tools/compatibility/update/generate_v2_reorders_map
"""
    self.assertTrue(
        reordered_function_names.issubset(function_reorders),
        added_names_message)
    # function_reorders should contain reordered_function_names
    # and their TensorFlow V1 aliases.
    for name in function_reorders:
      # get other names for this function
      attr = get_symbol_for_name(tf.compat.v1, name)
      _, attr = tf_decorator.unwrap(attr)
      v1_names = tf_export.get_v1_names(attr)
      self.assertTrue(v1_names)
      v1_names = ["tf.%s" % n for n in v1_names]
      # check if any other name is in
      self.assertTrue(
          any(n in reordered_function_names for n in v1_names),
          removed_names_message % name)

  def testRenameConstant(self):
    text = "tf.MONOLITHIC_BUILD\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.sysconfig.MONOLITHIC_BUILD\n")
    text = "some_call(tf.MONOLITHIC_BUILD)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "some_call(tf.sysconfig.MONOLITHIC_BUILD)\n")

  def testRenameArgs(self):
    text = ("tf.nn.pool(input_a, window_shape_a, pooling_type_a, padding_a, "
            "dilation_rate_a, strides_a, name_a, data_format_a)\n")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text,
                     ("tf.nn.pool(input=input_a, window_shape=window_shape_a,"
                      " pooling_type=pooling_type_a, padding=padding_a, "
                      "dilations=dilation_rate_a, strides=strides_a, "
                      "name=name_a, data_format=data_format_a)\n"))

  def testReorder(self):
    text = "tf.boolean_mask(a, b, c, d)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text,
                     "tf.boolean_mask(tensor=a, mask=b, name=c, axis=d)\n")

  def testLearningRateDecay(self):
    for decay in ["tf.train.exponential_decay",
                  "tf.train.polynomial_decay", "tf.train.natural_exp_decay",
                  "tf.train.inverse_time_decay", "tf.train.cosine_decay",
                  "tf.train.cosine_decay_restarts",
                  "tf.train.linear_cosine_decay",
                  "tf.train.noisy_linear_cosine_decay",
                  "tf.train.piecewise_constant_decay",
                 ]:

      text = "%s(a, b)\n" % decay
      _, report, unused_errors, _ = self._upgrade(text)
      self.assertIn("switch to the schedules in "
                    "`tf.keras.optimizers.schedules`", report)

  def testMetrics(self):
    metrics = [
        "accuracy",
        "auc",
        "average_precision_at_k",
        "false_negatives",
        "false_negatives_at_thresholds",
        "false_positives",
        "false_positives_at_thresholds",
        "mean",
        "mean_absolute_error",
        "mean_cosine_distance",
        "mean_iou",
        "mean_per_class_accuracy",
        "mean_relative_error",
        "mean_squared_error",
        "mean_tensor",
        "percentage_below",
        "precision",
        "precision_at_k",
        "precision_at_thresholds",
        "precision_at_top_k",
        "recall",
        "recall_at_k",
        "recall_at_thresholds",
        "recall_at_top_k",
        "root_mean_squared_error",
        "sensitivity_at_specificity",
        "sparse_average_precision_at_k",
        "sparse_precision_at_k",
        "specificity_at_sensitivity",
        "true_negatives",
        "true_negatives_at_thresholds",
        "true_positives",
        "true_positives_at_thresholds",
    ]
    for m in metrics:
      text = "tf.metrics." + m + "(a, b)"
      _, report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual("tf.compat.v1.metrics." + m + "(a, b)", new_text)
      self.assertIn(
          "tf.metrics have been replaced with object oriented versions", report)

  def testLosses(self):
    losses = [
        "absolute_difference",
        "add_loss",
        "compute_weighted_loss",
        "cosine_distance",
        "get_losses",
        "get_regularization_loss",
        "get_regularization_losses",
        "get_total_loss",
        "hinge_loss",
        "huber_loss",
        "log_loss",
        "mean_pairwise_squared_error",
        "mean_squared_error",
        "sigmoid_cross_entropy",
        "softmax_cross_entropy",
        "sparse_softmax_cross_entropy",
    ]
    for l in losses:
      text = "tf.losses." + l + "(a, b)"
      _, report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual("tf.compat.v1.losses." + l + "(a, b)", new_text)
      self.assertIn(
          "tf.losses have been replaced with object oriented versions", report)

  def testEstimatorLossReductionChange(self):
    classes = [
        "LinearClassifier", "LinearRegressor", "DNNLinearCombinedClassifier",
        "DNNLinearCombinedRegressor", "DNNRegressor", "DNNClassifier",
        "BaselineClassifier", "BaselineRegressor"
    ]
    for c in classes:
      ns = "tf.estimator." + c
      text = ns + "(a, b)"
      _, report, errors, new_text = self._upgrade(text)
      self.assertEqual(text, new_text)
      self.assertIn("%s requires manual check" % ns, errors[0])
      self.assertIn("loss_reduction has been changed", report)

  def testExtractGlimpse(self):
    text = ("tf.image.extract_glimpse(x, size, off, False, "
            "False, False, name=\"foo\")\n")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.image.extract_glimpse(x, size, off, False, "
        "False, 'uniform' if (False) else 'gaussian', name=\"foo\")\n",
    )

    text = ("tf.image.extract_glimpse(x, size, off, centered=False, "
            "normalized=False, uniform_noise=True if uniform_noise else "
            "False, name=\"foo\")\n")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.image.extract_glimpse(x, size, off, centered=False, "
        "normalized=False, noise='uniform' if (True if uniform_noise else "
        "False) else 'gaussian', name=\"foo\")\n",
    )

    text = ("tf.image.extract_glimpse(x,\n"
            "                         size,\n"
            "                         off,\n"
            "                         centered=True,\n"
            "                         normalized=True, # Stuff before\n"
            "                         uniform_noise=False,\n"
            "                         name=\"foo\")# Stuff after\n")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text, "tf.image.extract_glimpse(x,\n"
        "                         size,\n"
        "                         off,\n"
        "                         centered=True,\n"
        "                         normalized=True, # Stuff before\n"
        "                         noise='uniform' if (False) else 'gaussian',\n"
        "                         name=\"foo\")# Stuff after\n")

    text = "tf.image.extract_glimpse(x)\n"
    _, unused_report, errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, text)
    self.assertEqual(errors, [])

  def testDropout(self):
    text = "tf.nn.dropout(x, keep_prob, name=\"foo\")\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.nn.dropout(x, 1 - (keep_prob), name=\"foo\")\n",
    )

    text = "tf.nn.dropout(x, keep_prob=.4, name=\"foo\")\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.nn.dropout(x, rate=1 - (.4), name=\"foo\")\n",
    )

    text = (
        "tf.nn.dropout(x,  # Stuff before\n"
        "              keep_prob=.4,  # Stuff after\n"
        "              name=\"foo\")\n"
    )
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.nn.dropout(x,  # Stuff before\n"
        "              rate=1 - (.4),  # Stuff after\n"
        "              name=\"foo\")\n",
    )

    text = "tf.nn.dropout(x)\n"
    _, unused_report, errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, text)
    self.assertIn("tf.nn.dropout called without arguments", errors[0])

  def testDropoutExpr(self):
    text = "tf.nn.dropout(x, 1 - func(3 + 4.), name=\"foo\")\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.nn.dropout(x, 1 - (1 - func(3 + 4.)), name=\"foo\")\n",
    )

  def testMathCountNonZeroChanges(self):
    text = (
        "tf.math.count_nonzero(input_tensor=input, dtype=dtype, name=name, "
        "reduction_indices=axis, keep_dims=keepdims)\n"
        )
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    expected_text = (
        "tf.math.count_nonzero(input=input, dtype=dtype, name=name, "
        "axis=axis, keepdims=keepdims)\n"
        )
    self.assertEqual(new_text, expected_text)

  def testCountNonZeroChanges(self):
    text = (
        "tf.count_nonzero(input_tensor=input, dtype=dtype, name=name, "
        "reduction_indices=axis, keep_dims=keepdims)\n"
        )
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    expected_text = (
        "tf.math.count_nonzero(input=input, dtype=dtype, name=name, "
        "axis=axis, keepdims=keepdims)\n"
        )
    self.assertEqual(new_text, expected_text)

  def testRandomMultinomialToRandomCategorical(self):
    text = (
        "tf.random.multinomial(logits, samples, seed, name, output_dtype)\n"
        )
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    expected_text = (
        "tf.random.categorical(logits=logits, num_samples=samples, seed=seed, "
        "name=name, dtype=output_dtype)\n"
        )
    self.assertEqual(new_text, expected_text)

    text = (
        "tf.multinomial(logits, samples, seed, name, output_dtype)\n"
        )
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    expected_text = (
        "tf.random.categorical(logits=logits, num_samples=samples, seed=seed, "
        "name=name, dtype=output_dtype)\n"
        )
    self.assertEqual(new_text, expected_text)

  def testRandomPoissonConversion(self):
    text1 = "tf.random_poisson(lam, shape, dtype)"
    text2 = "tf.random.poisson(lam, shape, dtype)"
    expected_text = "tf.random.poisson(lam=lam, shape=shape, dtype=dtype)"
    _, unused_report, unused_errors, new_text1 = self._upgrade(text1)
    self.assertEqual(new_text1, expected_text)
    _, unused_report, unused_errors, new_text2 = self._upgrade(text2)
    self.assertEqual(new_text2, expected_text)

  def testConvolutionOpUpdate(self):
    text = (
        "tf.nn.convolution(input, filter, padding, strides, dilation_rate, "
        "name, data_format)"
    )
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    expected_text = (
        "tf.nn.convolution(input=input, filters=filter, padding=padding, "
        "strides=strides, dilations=dilation_rate, name=name, "
        "data_format=data_format)"
    )
    self.assertEqual(new_text, expected_text)

  def test_substr(self):
    text = "tf.substr(input, pos, len, name, unit)\n"
    _, unused_report, errors, new_text = self._upgrade(text)
    self.assertEqual("tf.strings.substr(input=input, pos=pos, len=len, "
                     "name=name, unit=unit)\n", new_text)
    self.assertEqual(errors, [])

  def testColocateGradientsWithOps(self):
    text = "tf.gradients(yx=a, foo=False)\n"
    _, unused_report, errors, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)
    self.assertEqual(errors, [])

    text = "tf.gradients(yx=a, colocate_gradients_with_ops=False)\n"
    _, report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual("tf.gradients(yx=a)\n", new_text)
    self.assertIn("tf.gradients no longer takes", report)

    text = "tf.gradients(y, x, grad_ys, name, colocate, gate)\n"
    expected = ("tf.gradients(ys=y, xs=x, grad_ys=grad_ys, name=name, "
                "gate_gradients=gate)\n")
    _, unused_report, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def testColocateGradientsWithOpsMinimize(self):
    text = "optimizer.minimize(a, foo=False)\n"
    _, unused_report, errors, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)
    self.assertEqual(errors, [])

    text = "optimizer.minimize(a, colocate_gradients_with_ops=False)\n"
    _, report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual("optimizer.minimize(a)\n", new_text)
    self.assertIn("Optimizer.minimize no longer takes", report)

  def testColocateGradientsWithOpsComputeGradients(self):
    text = "optimizer.compute_gradients(a, foo=False)\n"
    _, unused_report, errors, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)
    self.assertEqual(errors, [])

    text = "optimizer.compute_gradients(a, colocate_gradients_with_ops=False)\n"
    _, report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual("optimizer.compute_gradients(a)\n", new_text)
    self.assertIn("Optimizer.compute_gradients no longer takes", report)

  def testExportSavedModelRename(self):
    text = "self.est.export_savedmodel(path)"
    _, report, unused_errors, unused_new_text = self._upgrade(text)
    self.assertIn(
        "rename the method export_savedmodel() to export_saved_model()",
        report)

  def testArgmin(self):
    text = "tf.argmin(input, name=n, dimension=1, output_type=type)"
    expected_text = "tf.argmin(input=input, name=n, axis=1, output_type=type)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = "tf.argmin(input, 0)"
    expected_text = "tf.argmin(input=input, axis=0)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = "tf.arg_min(input, 0)"
    expected_text = "tf.argmin(input, 0)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testArgmax(self):
    text = "tf.argmax(input, name=n, dimension=1, output_type=type)"
    expected_text = "tf.argmax(input=input, name=n, axis=1, output_type=type)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = "tf.argmax(input, 0)"
    expected_text = "tf.argmax(input=input, axis=0)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = "tf.arg_max(input, 0)"
    expected_text = "tf.argmax(input, 0)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testEstimatorInputs(self):
    text = "tf.estimator.inputs.numpy_input_fn(0)"
    expected_text = "tf.compat.v1.estimator.inputs.numpy_input_fn(0)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = "tf.estimator.inputs.pandas_input_fn(0)"
    expected_text = "tf.compat.v1.estimator.inputs.pandas_input_fn(0)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testBatchToSpace(self):
    text = "tf.batch_to_space_nd(input, block_shape, crops, name)"
    expected_text = "tf.batch_to_space(input, block_shape, crops, name)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = "tf.batch_to_space(input, crops, block_size, name)"
    expected_text = (
        "tf.batch_to_space(input=input, crops=crops, block_shape=block_size, "
        "name=name)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = "tf.manip.batch_to_space_nd(input, block_shape, crops, name)"
    expected_text = "tf.batch_to_space(input, block_shape, crops, name)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testExtractImagePatches(self):
    text = (
        "tf.extract_image_patches(images, ksizes=ksizes, strides=strides,"
        "rates=rates, padding=padding, name=name)")
    expected_text = (
        "tf.image.extract_image_patches(images, sizes=ksizes, strides=strides,"
        "rates=rates, padding=padding, name=name)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testKerasSavedModel(self):
    text = (
        "tf.contrib.saved_model.save_keras_model(model, './saved_models')\n"
        "tf.contrib.saved_model.load_keras_model(saved_model_path)\n")
    expected_text = (
        "tf.keras.experimental.export(model, './saved_models')\n"
        "tf.keras.experimental.load_from_saved_model(saved_model_path)\n")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testStatelessMultinomial(self):
    text = (
        "tf.random.stateless_multinomial(logits, num_samples, seed, "
        "output_dtype=dtype, name=name)")
    expected_text = (
        "tf.random.stateless_categorical(logits, num_samples, seed, "
        "dtype=dtype, name=name)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testSoftMaxCrossEntropyWithLogitsV2(self):
    text = (
        "tf.nn.softmax_cross_entropy_with_logits_v2("
        "labels=labels, logits=logits, dim=2)")
    expected_text = (
        "tf.nn.softmax_cross_entropy_with_logits("
        "labels=labels, logits=logits, axis=2)")
    _, unused_report, errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    self.assertFalse(errors)

  def testSoftMaxCrossEntropyWithLogits(self):
    text = ("tf.nn.softmax_cross_entropy_with_logits("
            "labels=labels, logits=logits, dim=2)")
    expected_text = (
        "tf.nn.softmax_cross_entropy_with_logits("
        "labels=tf.stop_gradient(labels), logits=logits, axis=2)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = ("tf.nn.softmax_cross_entropy_with_logits("
            "labels=foo(bar))")
    expected_text = ("tf.nn.softmax_cross_entropy_with_logits("
                     "labels=tf.stop_gradient(foo(bar)))")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def testSparseMatmul(self):
    text = ("tf.sparse_matmul(a, b, c, d, e, f, g)\n")
    expected_text = ("tf.linalg.matmul(a=a, b=b, transpose_a=c, transpose_b=d, "
                     "a_is_sparse=e, b_is_sparse=f, name=g)\n")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testWeightedMoments(self):
    text = "tf.nn.weighted_moments(x, axes, freq, name, kd)"
    expected_text = (
        "tf.nn.weighted_moments(x=x, axes=axes, frequency_weights=freq, "
        "name=name, keepdims=kd)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testSparseAdd(self):
    text = "tf.sparse.add(a, b, t)"
    expected_text = "tf.sparse.add(a=a, b=b, threshold=t)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testSparseConcat(self):
    text = "tf.sparse.concat(ax, inp, name, exp, concat)"
    expected_text = (
        "tf.sparse.concat(axis=ax, sp_inputs=inp, name=name, "
        "expand_nonconcat_dims=exp, axis=concat)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testSeparableConv2D(self):
    text = "tf.nn.separable_conv2d(inp, d, pt, strides, pad, rate, name, fmt)"
    expected_text = (
        "tf.nn.separable_conv2d(input=inp, depthwise_filter=d, "
        "pointwise_filter=pt, strides=strides, padding=pad, "
        "dilations=rate, name=name, data_format=fmt)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testConv2D(self):
    text = (
        "tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu, "
        "data_format)")
    expected_text = (
        "tf.nn.conv2d(input=input, filters=filter, strides=strides, "
        "padding=padding, data_format=data_format)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = (
        "tf.nn.conv2d(input, filter=filter, strides=strides, padding=padding, "
        "use_cudnn_on_gpu=use_cudnn_on_gpu)")
    expected_text = ("tf.nn.conv2d(input=input, filters=filter, "
                     "strides=strides, padding=padding)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testConv2DBackpropFilter(self):
    text = (
        "tf.nn.conv2d_backprop_filter(input, filter_sizes, out_backprop, "
        "strides, padding, use_cudnn_on_gpu, data_format)")
    expected_text = (
        "tf.nn.conv2d_backprop_filter(input=input, filter_sizes=filter_sizes, "
        "out_backprop=out_backprop, strides=strides, padding=padding, "
        "data_format=data_format)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testConv2DBackpropInput(self):
    text = (
        "tf.nn.conv2d_backprop_input(input_sizes, filter, out_backprop, "
        "strides, padding, use_cudnn_on_gpu, data_format)")
    expected_text = (
        "tf.nn.conv2d_backprop_input(input_sizes=input_sizes, filters=filter, "
        "out_backprop=out_backprop, strides=strides, padding=padding, "
        "data_format=data_format)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testSpacetoBatch(self):
    text = "tf.space_to_batch_nd(input, shape, paddings, name)"
    expected_text = "tf.space_to_batch(input, shape, paddings, name)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = "tf.nn.space_to_batch(input, paddings, block_size, name)"
    expected_text = (
        "tf.space_to_batch(input=input, paddings=paddings, "
        "block_shape=block_size, name=name)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testInTopK(self):
    text = "tf.math.in_top_k(a, b, c, n)"
    expected_text = (
        "tf.math.in_top_k(predictions=a, targets=b, k=c, name=n)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testDepthToSpace(self):
    text = "tf.nn.depth_to_space(input, block_size, name, data_format)"
    expected_text = (
        "tf.nn.depth_to_space(input=input, block_size=block_size, "
        "name=name, data_format=data_format)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testEmbeddingLookup(self):
    text = ("tf.nn.embedding_lookup(params, ids, partition_strategy, name, "
            "validate_indices, max_norm)")
    expected_text = ("tf.nn.embedding_lookup(params=params, ids=ids, "
                     "partition_strategy=partition_strategy, name=name, "
                     "max_norm=max_norm)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testEmbeddingLookupSparse(self):
    text = ("tf.nn.embedding_lookup_sparse(params, sp_ids, sp_weights, "
            "partition_strategy, name, combiner, max_norm)")
    expected_text = ("tf.nn.embedding_lookup_sparse(params=params, "
                     "sp_ids=sp_ids, sp_weights=sp_weights, "
                     "partition_strategy=partition_strategy, name=name, "
                     "combiner=combiner, max_norm=max_norm)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testNnInTopK(self):
    text = "tf.nn.in_top_k(predictions, targets, k, name)"
    expected_text = ("tf.nn.in_top_k(predictions=predictions, "
                     "targets=targets, k=k, name=name)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testSpaceToDepth(self):
    text = "tf.nn.space_to_depth(input, block_size, name, data_format)"
    expected_text = ("tf.nn.space_to_depth(input=input, block_size=block_size, "
                     "name=name, data_format=data_format)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testPrint(self):
    # tf.print() cannot be parsed unless we import print_function
    text = """from __future__ import print_function
tf.print()
tf.print('abc')
"""
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, text)  # Text should stay the same

  def testSparseSplit(self):
    text = (
        "tf.sparse_split(sp_input=sp_input, num_split=num_split, axis=axis, "
        "name=name)")
    expected_text = (
        "tf.sparse.split(sp_input=sp_input, num_split=num_split, axis=axis, "
        "name=name)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = (
        "tf.sparse_split(sp_input=sp_input, num_split=num_split, "
        "name=name, split_dim=axis)")
    expected_text = (
        "tf.sparse.split(sp_input=sp_input, num_split=num_split, "
        "name=name, axis=axis)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = (
        "tf.sparse.split(sp_input=sp_input, num_split=num_split, "
        "name=name, split_dim=axis)")
    expected_text = (
        "tf.sparse.split(sp_input=sp_input, num_split=num_split, "
        "name=name, axis=axis)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testIterators(self):
    for (text, expected) in [
        ("(expr + yielding(data)).make_one_shot_iterator()",
         "tf.compat.v1.data.make_one_shot_iterator((expr + yielding(data)))"),
        ("dataset.make_one_shot_iterator()",
         "tf.compat.v1.data.make_one_shot_iterator(dataset)"),
        ("dataset.make_one_shot_iterator(shared_name=foo)",
         "tf.compat.v1.data.make_one_shot_iterator(dataset, shared_name=foo)"),
        ("dataset.make_one_shot_iterator(x, y, z)",
         "tf.compat.v1.data.make_one_shot_iterator(dataset, x, y, z)"),
        ("dataset.make_initializable_iterator()",
         "tf.compat.v1.data.make_initializable_iterator(dataset)"),
        ("ds.make_initializable_iterator(shared_name=foo)",
         "tf.compat.v1.data.make_initializable_iterator(ds, shared_name=foo)"),
        ("dataset.make_initializable_iterator(x, y, z)",
         "tf.compat.v1.data.make_initializable_iterator(dataset, x, y, z)"),
        ("tf.data.make_one_shot_iterator(dataset)",
         "tf.compat.v1.data.make_one_shot_iterator(dataset)"),
        ("tf.data.make_one_shot_iterator(dataset, shared_name=foo)",
         "tf.compat.v1.data.make_one_shot_iterator(dataset, shared_name=foo)"),
        ("tf.data.make_one_shot_iterator(dataset, x, y, z)",
         "tf.compat.v1.data.make_one_shot_iterator(dataset, x, y, z)"),
        ("tf.data.make_initializable_iterator(dataset)",
         "tf.compat.v1.data.make_initializable_iterator(dataset)"),
        ("tf.data.make_initializable_iterator(ds, shared_name=foo)",
         "tf.compat.v1.data.make_initializable_iterator(ds, shared_name=foo)"),
        ("tf.data.make_initializable_iterator(dataset, x, y, z)",
         "tf.compat.v1.data.make_initializable_iterator(dataset, x, y, z)"),
        ("tf.compat.v1.data.make_one_shot_iterator(dataset)",
         "tf.compat.v1.data.make_one_shot_iterator(dataset)"),
        ("tf.compat.v1.data.make_one_shot_iterator(dataset, shared_name=foo)",
         "tf.compat.v1.data.make_one_shot_iterator(dataset, shared_name=foo)"),
        ("tf.compat.v1.data.make_one_shot_iterator(dataset, x, y, z)",
         "tf.compat.v1.data.make_one_shot_iterator(dataset, x, y, z)"),
        ("tf.compat.v1.data.make_initializable_iterator(dataset)",
         "tf.compat.v1.data.make_initializable_iterator(dataset)"),
        ("tf.compat.v1.data.make_initializable_iterator(ds, shared_name=foo)",
         "tf.compat.v1.data.make_initializable_iterator(ds, shared_name=foo)"),
        ("tf.compat.v1.data.make_initializable_iterator(dataset, x, y, z)",
         "tf.compat.v1.data.make_initializable_iterator(dataset, x, y, z)")]:
      _, unused_report, unused_errors, actual = self._upgrade(text)
      self.assertEqual(actual, expected)

  def testCast(self):
    for (name, dtype) in [("int32", "int32"),
                          ("int64", "int64"),
                          ("float", "float32"),
                          ("double", "float64"),
                          ("complex64", "complex64"),
                          ("complex128", "complex128"),
                          ("bfloat16", "bfloat16")]:
      text = "tf.to_%s(x, name='test')" % name
      expected_text = "tf.cast(x, name='test', dtype=tf.%s)" % dtype
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(expected_text, new_text)

  def testCastPositionalSecondArgument(self):
    for (name, dtype) in [("int32", "int32"),
                          ("int64", "int64"),
                          ("float", "float32"),
                          ("double", "float64"),
                          ("complex64", "complex64"),
                          ("complex128", "complex128"),
                          ("bfloat16", "bfloat16")]:
      text = "tf.to_%s(x, 'test')" % name
      expected_text = "tf.cast(x, name='test', dtype=tf.%s)" % dtype
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(expected_text, new_text)

  def testImageResize(self):
    for method in ["bilinear", "area", "bicubic", "nearest_neighbor"]:
      text = "tf.image.resize_%s(i, s)" % method
      expected_text = ("tf.image.resize(i, s, "
                       "method=tf.image.ResizeMethod.%s)" % method.upper())
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(expected_text, new_text)

  def testImageResizeExtraPositionalArgs(self):
    for method in ["bilinear", "area", "bicubic", "nearest_neighbor"]:
      text = "tf.image.resize_%s(i, s, a, p)" % method
      expected_text = ["tf.image.resize(i, s, ", "align_corners=a, ",
                       "preserve_aspect_ratio=p, ",
                       "method=tf.image.ResizeMethod.%s)" % method.upper()]
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      for s in expected_text:
        self.assertIn(s, new_text)

  def testCond(self):
    text = "tf.cond(a, b, c, True)"
    expected_text = "tf.cond(pred=a, true_fn=b, false_fn=c)"
    _, unused_report, errors, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)
    self.assertIn("tf.cond", errors[0])
    self.assertIn("requires manual check", errors[0])

  def testParens(self):
    text = """
def _log_prob(self, x):
  return tf.reduce_logsumexp(
      (self.mixture_distribution.logits + self.distribution.log_prob(
          x[..., tf.newaxis])),
          axis=-1)"""
    expected_text = """
def _log_prob(self, x):
  return tf.reduce_logsumexp(
      input_tensor=(self.mixture_distribution.logits + self.distribution.log_prob(
          x[..., tf.newaxis])),
          axis=-1)"""
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def testAssertStatements(self):
    for name in ["assert_greater", "assert_equal", "assert_none_equal",
                 "assert_less", "assert_negative", "assert_positive",
                 "assert_non_negative", "assert_non_positive", "assert_near",
                 "assert_less", "assert_less_equal", "assert_greater",
                 "assert_greater_equal", "assert_integer", "assert_type",
                 "assert_scalar"]:
      text = "tf.%s(a)" % name
      expected_text = "tf.compat.v1.%s(a)" % name
      _, report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(expected_text, new_text)
      self.assertIn("%s has been" % name, report)

      text = "tf.debugging.%s(a)" % name
      expected_text = "tf.compat.v1.debugging.%s(a)" % name
      _, report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(expected_text, new_text)
      self.assertIn("%s has been" % name, report)

  def testAssertRankStatements(self):
    for name in ["assert_rank", "assert_rank_at_least", "assert_rank_in"]:
      text = "tf.%s(a)" % name
      expected_text = "tf.compat.v1.%s(a)" % name
      _, report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(expected_text, new_text)
      self.assertIn("%s has been" % name, report)

      text = "tf.debugging.%s(a)" % name
      expected_text = "tf.compat.v1.debugging.%s(a)" % name
      _, report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(expected_text, new_text)
      self.assertIn("%s has been" % name, report)

  def test_assert_equal_graph_def(self):
    text = "tf.test.assert_equal_graph_def(a, b, checkpoint_v2=x)"
    expected = "tf.test.assert_equal_graph_def(actual=a, expected=b)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_is_tensor_upgrade(self):
    text = "tf.contrib.framework.is_tensor(x)"
    expected = "tf.is_tensor(x)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_sample_distorted_bounding_box(self):
    # pylint: disable=line-too-long
    text = "tf.image.sample_distorted_bounding_box(a, b, c, d, e, f, g, h, i, j)"
    expected = "tf.image.sample_distorted_bounding_box(image_size=a, bounding_boxes=b, seed=c, min_object_covered=e, aspect_ratio_range=f, area_range=g, max_attempts=h, use_image_if_no_bounding_boxes=i, name=j)"
    # pylint: enable=line-too-long
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_contrib_framework_argsort(self):
    text = "tf.contrib.framework.argsort"
    expected = "tf.argsort"
    # pylint: enable=line-too-long
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_flags_bare(self):
    _, _, errors, _ = self._upgrade("tf.flags")
    self.assertIn("tf.flags has been removed", errors[0])

  def test_flags_flags(self):
    _, _, errors, _ = self._upgrade("tf.flags.FLAGS")
    self.assertIn("tf.flags has been removed", errors[0])

  def test_max_pool_2d(self):
    text = "tf.nn.max_pool(value=4)"
    expected_text = "tf.nn.max_pool2d(input=4)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)


class TestUpgradeFiles(test_util.TensorFlowTestCase):

  def testInplace(self):
    """Check to make sure we don't have a file system race."""
    temp_file = tempfile.NamedTemporaryFile("w", delete=False)
    original = "tf.conj(a)\n"
    upgraded = "tf.math.conj(a)\n"
    temp_file.write(original)
    temp_file.close()
    upgrader = ast_edits.ASTCodeUpgrader(tf_upgrade_v2.TFAPIChangeSpec())
    upgrader.process_file(temp_file.name, temp_file.name)
    self.assertAllEqual(open(temp_file.name).read(), upgraded)
    os.unlink(temp_file.name)


if __name__ == "__main__":
  test_lib.main()
