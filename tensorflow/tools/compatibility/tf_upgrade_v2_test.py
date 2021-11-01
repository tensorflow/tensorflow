# Lint as: python2, python3
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

import inspect
import os
import tempfile

from absl.testing import parameterized
import six
import tensorflow.compat.v1 as tf
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
  name_parts = six.ensure_str(name).split(".")
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
  open_paren_index = six.ensure_str(call_str).find("(")
  close_paren_index = call_str.rfind(")")

  function_name = call_str[:six.ensure_str(call_str).find("(")]
  args = six.ensure_str(call_str[open_paren_index +
                                 1:close_paren_index]).split(",")
  args = [six.ensure_str(arg).split("=")[0].strip() for arg in args]
  args = [arg for arg in args if arg]  # filter out empty strings
  return function_name, args


class TestUpgrade(test_util.TensorFlowTestCase, parameterized.TestCase):
  """Test various APIs that have been changed in 2.0.

  We also test whether a converted file is executable. test_file_v1_10.py
  aims to exhaustively test that API changes are convertible and actually
  work when run with current TensorFlow.
  """

  @classmethod
  def setUpClass(cls):
    super(TestUpgrade, cls).setUpClass()
    cls.v2_symbols = {}
    cls.v1_symbols = {}
    if hasattr(tf.compat, "v2"):

      def symbol_collector(unused_path, unused_parent, children):
        for child in children:
          _, attr = tf_decorator.unwrap(child[1])
          api_names_v2 = tf_export.get_v2_names(attr)
          for name in api_names_v2:
            cls.v2_symbols["tf." + six.ensure_str(name)] = attr

      visitor = public_api.PublicAPIVisitor(symbol_collector)
      visitor.private_map["tf.compat"] = ["v1", "v2"]
      traverse.traverse(tf.compat.v2, visitor)

    if hasattr(tf.compat, "v1"):

      def symbol_collector_v1(unused_path, unused_parent, children):
        for child in children:
          _, attr = tf_decorator.unwrap(child[1])
          api_names_v1 = tf_export.get_v1_names(attr)
          for name in api_names_v1:
            cls.v1_symbols["tf." + six.ensure_str(name)] = attr

      visitor = public_api.PublicAPIVisitor(symbol_collector_v1)
      visitor.private_map["tf.compat"] = ["v1", "v2"]
      traverse.traverse(tf.compat.v1, visitor)

  def _upgrade(self,
               old_file_text,
               import_rename=False,
               upgrade_compat_v1_import=False):
    in_file = six.StringIO(old_file_text)
    out_file = six.StringIO()
    upgrader = ast_edits.ASTCodeUpgrader(
        tf_upgrade_v2.TFAPIChangeSpec(
            import_rename, upgrade_compat_v1_import=upgrade_compat_v1_import))
    count, report, errors = (
        upgrader.process_opened_file("test.py", in_file,
                                     "test_out.py", out_file))
    return count, report, errors, out_file.getvalue()

  def _upgrade_multiple(self, old_file_texts):
    upgrader = ast_edits.ASTCodeUpgrader(tf_upgrade_v2.TFAPIChangeSpec())
    results = []
    for old_file_text in old_file_texts:
      in_file = six.StringIO(old_file_text)
      out_file = six.StringIO()
      count, report, errors = (
          upgrader.process_opened_file("test.py", in_file,
                                       "test_out.py", out_file))
      results.append([count, report, errors, out_file.getvalue()])
    return results

  def testParseError(self):
    _, report, unused_errors, unused_new_text = self._upgrade(
        "import tensorflow as tf\na + \n")
    self.assertNotEqual(six.ensure_str(report).find("Failed to parse"), -1)

  def testReport(self):
    text = "tf.angle(a)\n"
    _, report, unused_errors, unused_new_text = self._upgrade(text)
    # This is not a complete test, but it is a sanity test that a report
    # is generating information.
    self.assertTrue(
        six.ensure_str(report).find("Renamed function `tf.angle` to "
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
          _, _, _, text = self._upgrade("tf." + six.ensure_str(name))
          if (text and
              not text.startswith("tf.compat.v1") and
              not text.startswith("tf.compat.v2") and
              text not in self.v2_symbols and
              # Ignore any symbol that contains __internal__
              "__internal__" not in text and
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

    # Converts all symbols in the v1 namespace to the v2 namespace, raising
    # an error if the target of the conversion is not in the v1 namespace.
    def conversion_visitor(unused_path, unused_parent, children):
      for child in children:
        _, attr = tf_decorator.unwrap(child[1])
        api_names = tf_export.get_v1_names(attr)
        for name in api_names:
          if collect:
            v1_symbols.add("tf." + six.ensure_str(name))
          else:
            _, _, _, text = self._upgrade("tf." + six.ensure_str(name))
            if (text and
                not text.startswith("tf.compat.v1") and
                not text.startswith("tf.compat.v2") and
                not text.startswith("tf.estimator") and
                text not in v1_symbols):
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
          if "__internal__" in new_function_name:
            # Skip the tf.__internal__ and tf.keras.__internal__ API.
            continue
          if new_function_name == "tf.compat.v1.%s" % name:
            if tf_name in keyword_renames:
              # If we rename arguments, new function must be available in 2.0.
              # We should not be using compat.v1 in this case.
              self.fail(
                  "Function '%s' is not in 2.0 when converting\n%s\nto\n%s" %
                  (new_function_name, text_input, text))
            continue
          if new_function_name.startswith("tf.compat.v2"):
            self.assertIn(new_function_name.replace("tf.compat.v2.", "tf."),
                          self.v2_symbols)
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
    method_names = list(full_dict.keys())
    for method_name in method_names:
      args = list(full_dict[method_name].keys())
      if "contrib" in method_name:
        # Skip descending and fetching contrib methods during test. These are
        # not available in the repo anymore.
        continue
      elif six.ensure_str(method_name).startswith("*."):
        # special case for optimizer methods
        method = six.ensure_str(method_name).replace("*", "tf.train.Optimizer")
      else:
        method = method_name

      method = get_symbol_for_name(tf, method)
      arg_spec = tf_inspect.getfullargspec(method)
      for (arg, pos) in args:
        # to deal with the self argument on methods on objects
        if six.ensure_str(method_name).startswith("*."):
          pos += 1
        self.assertEqual(arg_spec[0][pos], arg)

  def testReorderFileNeedsUpdate(self):
    reordered_function_names = (
        tf_upgrade_v2.TFAPIChangeSpec().reordered_function_names)
    function_reorders = (
        tf_upgrade_v2.TFAPIChangeSpec().function_reorders)
    manual_function_reorders = (
        tf_upgrade_v2.TFAPIChangeSpec().manual_function_reorders)

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
      if name in manual_function_reorders:
        continue
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

  def verify_compat_v1_rename_correctness(self, values, ns_prefix=""):
    if ns_prefix:
      ns_prefix += "."
    for v in values:
      text = "tf." + ns_prefix + v + "(a, b)"
      _, _, _, new_text = self._upgrade(text)
      self.assertEqual("tf.compat.v1." + ns_prefix + v + "(a, b)", new_text)

  def testInitializers(self):
    initializers = [
        "zeros",
        "ones",
        "constant",
        "random_uniform",
        "random_normal",
        "truncated_normal",
        "variance_scaling",
        "orthogonal",
        "glorot_uniform",
        "glorot_normal",
        "identity",
        "lecun_normal",
        "lecun_uniform",
        "he_normal",
        "he_uniform",
    ]
    self.verify_compat_v1_rename_correctness(
        initializers, ns_prefix="initializers")

    initializers = [
        "zeros_initializer",
        "ones_initializer",
        "constant_initializer",
        "random_uniform_initializer",
        "random_normal_initializer",
        "truncated_normal_initializer",
        "variance_scaling_initializer",
        "orthogonal_initializer",
        "glorot_uniform_initializer",
        "glorot_normal_initializer",
    ]
    self.verify_compat_v1_rename_correctness(initializers)

    initializers = [
        "zeros",
        "ones",
        "Ones",
        "Zeros",
        "constant",
        "Constant",
        "VarianceScaling",
        "Orthogonal",
        "orthogonal",
        "Identity",
        "identity",
        "glorot_uniform",
        "glorot_normal",
        "lecun_normal",
        "lecun_uniform",
        "he_normal",
        "he_uniform",
        "TruncatedNormal",
        "truncated_normal",
        "RandomUniform",
        "uniform",
        "random_uniform",
        "RandomNormal",
        "normal",
        "random_normal",
    ]
    self.verify_compat_v1_rename_correctness(
        initializers, ns_prefix="keras.initializers")

  def testContribXavierInitializer(self):
    for contrib_alias in ["tf.contrib.", "contrib_"]:
      text = contrib_alias + "layers.xavier_initializer()\n"
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(
          new_text,
          "tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, "
          "mode=\"fan_avg\", "
          "distribution=\"uniform\")\n",
      )

      text = "slim.xavier_initializer(True or False)\n"
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(
          new_text,
          "tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, "
          "mode=\"fan_avg\", "
          "distribution=(\"uniform\" if True or False else "
          "\"truncated_normal\"))\n",
      )

      text = "slim.xavier_initializer(uniform=(True or False))\n"
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(
          new_text,
          "tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, "
          "mode=\"fan_avg\", "
          "distribution=(\"uniform\" if True or False else "
          "\"truncated_normal\"))\n",
      )

      text = contrib_alias + "layers.xavier_initializer_conv2d(False, 12)\n"
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(
          new_text,
          "tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, "
          "mode=\"fan_avg\", "
          "distribution=(\"uniform\" if False else \"truncated_normal\"), "
          "seed=12)\n",
      )

      text = (contrib_alias + "layers.xavier_initializer_conv2d("
              "False, 12, tf.float32)\n")
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(
          new_text,
          "tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, "
          "mode=\"fan_avg\", "
          "distribution=(\"uniform\" if False else \"truncated_normal\"), "
          "seed=12, "
          "dtype=tf.float32)\n",
      )

      text = (contrib_alias + "layers.xavier_initializer("
              "False, 12, dtypes=tf.float32)\n")
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(
          new_text,
          "tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, "
          "mode=\"fan_avg\", "
          "distribution=(\"uniform\" if False else \"truncated_normal\"), "
          "seed=12, "
          "dtypes=tf.float32)\n",
      )

  def testVarianceScalingInitializer(self):
    text = ("tf.contrib.layers.variance_scaling_initializer("
            "mode=(\"FAN\" + \"_AVG\"))\n")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0, "
        "mode=(\"FAN\" + \"_AVG\").lower())\n",
    )

    text = ("slim.variance_scaling_initializer("
            "uniform=(True or False), mode=(\"FAN\" + \"_AVG\"))\n")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0, "
        "distribution=(\"uniform\" if True or False else \"truncated_normal\"),"
        " mode=(\"FAN\" + \"_AVG\").lower())\n",
    )

    text = "tf.contrib.layers.variance_scaling_initializer(factor=1.0)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0)\n",
    )

    text = ("tf.contrib.layers.variance_scaling_initializer("
            "12.0, \"FAN_AVG\", True, dtypes=tf.float32)\n")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.compat.v1.keras.initializers.VarianceScaling(12.0, "
        "(\"FAN_AVG\").lower(), "
        "(\"uniform\" if True else \"truncated_normal\"), "
        "dtypes=tf.float32)\n",
    )

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
      text = ns + "()"
      expected_text = ns + "(loss_reduction=tf.keras.losses.Reduction.SUM)"
      _, report, errors, new_text = self._upgrade(text)
      self.assertEqual(expected_text, new_text)

      text = ns + "(loss_reduction=TEST)"
      expected_text = ns + "(loss_reduction=TEST)"
      _, report, errors, new_text = self._upgrade(text)
      self.assertEqual(text, new_text)
    text = "tf.estimator.BaselineClassifier(m, c, w, v, o, c, lr)"
    expected_text = (
        "tf.compat.v1.estimator.BaselineClassifier("
        "model_dir=m, n_classes=c, weight_column=w, label_vocabulary=v, "
        "optimizer=o, config=c, loss_reduction=lr)")
    _, report, errors, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "tf.estimator.BaselineClassifier(model_dir=model_dir)"
    expected_text = ("tf.estimator.BaselineClassifier(" +
                     "model_dir=model_dir, "
                     "loss_reduction=tf.keras.losses.Reduction.SUM)")
    _, report, errors, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def testBaseEstimatorPartitioner(self):
    classes = ["LinearEstimator", "DNNLinearCombinedEstimator", "DNNEstimator"]
    for c in classes:
      ns = "tf.estimator." + c
      suffix = "(input_layer_partitioner=TEST)"
      text = ns + suffix
      expected_text = "tf.compat.v1.estimator." + c + suffix
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(new_text, expected_text)

  def testCannedEstimatorPartitioner(self):
    classes = [
        "LinearClassifier", "LinearRegressor", "DNNLinearCombinedClassifier",
        "DNNLinearCombinedRegressor", "DNNRegressor", "DNNClassifier"
    ]

    for c in classes:
      ns = "tf.estimator." + c
      suffix = "(input_layer_partitioner=TEST)"
      text = ns + suffix
      suffix = ("(input_layer_partitioner=TEST, "
                "loss_reduction=tf.keras.losses.Reduction.SUM)")
      expected_text = "tf.compat.v1.estimator." + c + suffix
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(new_text, expected_text)

  def testBaseEstimatorOptimizer(self):
    classes = ["BaselineEstimator", "LinearEstimator", "DNNEstimator"]
    for c in classes:
      ns = "tf.estimator." + c
      suffix = "(optimizer=TEST)"
      text = ns + suffix
      expected_text = "tf.compat.v1.estimator." + c + suffix
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(new_text, expected_text)

  def testDNNLinearCombinedEstimatorOptimizer(self):
    classes = ["DNNLinearCombinedEstimator"]
    for c in classes:
      ns = "tf.estimator." + c
      suffix = "(dnn_optimizer=TEST, linear_optimizer=Test)"
      text = ns + suffix
      expected_text = "tf.compat.v1.estimator." + c + suffix
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(new_text, expected_text)

  def testCannedEstimatorOptimizer(self):
    classes = [
        "BaselineClassifier", "BaselineRegressor", "LinearClassifier",
        "LinearRegressor", "DNNRegressor", "DNNClassifier"
    ]

    for c in classes:
      ns = "tf.estimator." + c
      suffix = "(optimizer=TEST)"
      text = ns + suffix
      suffix = ("(optimizer=TEST, "
                "loss_reduction=tf.keras.losses.Reduction.SUM)")
      expected_text = "tf.compat.v1.estimator." + c + suffix
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(new_text, expected_text)

  def testDNNLinearCombinedOptimizer(self):
    classes = [
        "DNNLinearCombinedClassifier",
        "DNNLinearCombinedRegressor",
    ]
    for c in classes:
      ns = "tf.estimator." + c
      suffix = "(dnn_optimizer=TEST, linear_optimizer=Test)"
      text = ns + suffix
      suffix = ("(dnn_optimizer=TEST, linear_optimizer=Test, "
                "loss_reduction=tf.keras.losses.Reduction.SUM)")
      expected_text = "tf.compat.v1.estimator." + c + suffix
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(new_text, expected_text)

  def testBaseEstimatorPartitionerAndOptimizer(self):
    classes = ["LinearEstimator", "DNNEstimator"]
    for c in classes:
      ns = "tf.estimator." + c
      suffix = "(input_layer_partitioner=TEST, optimizer=TEST)"
      text = ns + suffix
      expected_text = "tf.compat.v1.estimator." + c + suffix
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(new_text, expected_text)

  def testDNNLinearCombinedEstimatorPartitionerAndOptimizer(self):
    classes = ["DNNLinearCombinedEstimator"]
    for c in classes:
      ns = "tf.estimator." + c
      suffix = ("(input_layer_partitioner=TEST, dnn_optimizer=TEST, "
                "linear_optimizer=TEST)")
      text = ns + suffix
      expected_text = "tf.compat.v1.estimator." + c + suffix
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(new_text, expected_text)

  def testCannedEstimatorPartitionerAndOptimizer(self):
    classes = [
        "LinearClassifier", "LinearRegressor", "DNNRegressor", "DNNClassifier"
    ]

    for c in classes:
      ns = "tf.estimator." + c
      suffix = "(input_layer_partitioner=TEST, optimizer=TEST)"
      text = ns + suffix
      suffix = ("(input_layer_partitioner=TEST, optimizer=TEST, "
                "loss_reduction=tf.keras.losses.Reduction.SUM)")
      expected_text = "tf.compat.v1.estimator." + c + suffix
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(new_text, expected_text)

  def testDNNLinearCombinedPartitionerAndOptimizer(self):
    classes = [
        "DNNLinearCombinedClassifier",
        "DNNLinearCombinedRegressor",
    ]

    for c in classes:
      ns = "tf.estimator." + c
      suffix = ("(input_layer_partitioner=TEST, dnn_optimizer=TEST, "
                "linear_optimizer=TEST)")
      text = ns + suffix
      suffix = ("(input_layer_partitioner=TEST, dnn_optimizer=TEST, "
                "linear_optimizer=TEST, "
                "loss_reduction=tf.keras.losses.Reduction.SUM)")
      expected_text = "tf.compat.v1.estimator." + c + suffix
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(new_text, expected_text)

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
        "tf.nn.dropout(x, rate=1 - (keep_prob), name=\"foo\")\n",
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
        "tf.nn.dropout(x, rate=1 - (1 - func(3 + 4.)), name=\"foo\")\n",
    )

  def testContribL1(self):
    text = "tf.contrib.layers.l1_regularizer(scale)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.keras.regularizers.l1(scale)\n",
    )
    self.assertNotIn("Dropping scope", unused_report)

    text = "tf.contrib.layers.l1_regularizer(scale, scope)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.keras.regularizers.l1(scale)\n",
    )
    self.assertIn("Dropping scope", unused_report)

    text = (
        "slim.l1_regularizer(  # Stuff before\n"
        "                    scale=.4,"
        "                    scope=\"foo\")\n"
    )
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.keras.regularizers.l1(  # Stuff before\n"
        "                    l=.4)\n",
    )
    self.assertIn("Dropping scope", unused_report)

  def testContribL2(self):
    text = "tf.contrib.layers.l2_regularizer(scale)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.keras.regularizers.l2(0.5 * (scale))\n",
    )
    self.assertNotIn("Dropping scope", unused_report)

    text = "tf.contrib.layers.l2_regularizer(scale, scope)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.keras.regularizers.l2(0.5 * (scale))\n",
    )
    self.assertIn("Dropping scope", unused_report)

    text = (
        "slim.l2_regularizer(  # Stuff before\n"
        "                    scale=.4,"
        "                    scope=\"foo\")\n"
    )
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.keras.regularizers.l2(  # Stuff before\n"
        "                    l=0.5 * (.4))\n",
    )
    self.assertIn("Dropping scope", unused_report)

  def testContribL2Expr(self):
    text = "tf.contrib.layers.l2_regularizer(1 - func(3 + 4.), scope=\"foo\")\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.keras.regularizers.l2(0.5 * (1 - func(3 + 4.)))\n",
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

  def testColocateGradientsWithHessians(self):
    text = "tf.hessians(ys=a, xs=b, colocate_gradients_with_ops=False)\n"
    _, report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual("tf.hessians(ys=a, xs=b)\n", new_text)
    self.assertIn("tf.hessians no longer takes", report)

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

  def testAutograph(self):
    text = "tf.autograph.to_graph(f, True, arg_values=None, arg_types=None)"
    expected_text = "tf.autograph.to_graph(f, True)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = ("tf.autograph.to_code"
            "(f, False, arg_values=None, arg_types=None, indentation=' ')")
    expected_text = "tf.autograph.to_code(f, False)"
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
        "tf.image.extract_patches(images, sizes=ksizes, strides=strides,"
        "rates=rates, padding=padding, name=name)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testKerasSavedModel(self):
    text = (
        "tf.contrib.saved_model.save_keras_model(model, './saved_models')\n"
        "tf.contrib.saved_model.load_keras_model(saved_model_path)\n")
    expected_text = (
        "tf.compat.v1.keras.experimental.export_saved_model(model, "
        "'./saved_models')\ntf.compat.v1.keras.experimental."
        "load_from_saved_model(saved_model_path)\n"
    )
    _, report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)
    expected_info = "Please use model.save"
    self.assertIn(expected_info, report)

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

  def testSoftMaxCrossEntropyWithLogitsDoesntNest(self):
    text = ("tf.nn.softmax_cross_entropy_with_logits("
            "labels=tf.stop_gradient(labels), logits=logits, dim=2)")
    expected_text = (
        "tf.nn.softmax_cross_entropy_with_logits("
        "labels=tf.stop_gradient(labels), logits=logits, axis=2)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = ("tf.nn.softmax_cross_entropy_with_logits("
            "labels=tf.stop_gradient(foo(bar)))")
    expected_text = ("tf.nn.softmax_cross_entropy_with_logits("
                     "labels=tf.stop_gradient(foo(bar)))")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = ("tf.nn.softmax_cross_entropy_with_logits("
            "labels=foo())")
    expected_text = ("tf.nn.softmax_cross_entropy_with_logits("
                     "labels=tf.stop_gradient(foo()))")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = ("tf.nn.softmax_cross_entropy_with_logits("
            "labels=foo().zz())")
    expected_text = ("tf.nn.softmax_cross_entropy_with_logits("
                     "labels=tf.stop_gradient(foo().zz()))")
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
        "tf.compat.v1.nn.conv2d_backprop_filter(input, filter_sizes, "
        "out_backprop, strides, padding, use_cudnn_on_gpu, data_format)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testConv2DBackpropInput(self):
    text = (
        "tf.nn.conv2d_backprop_input(input_sizes, filter, out_backprop, "
        "strides, padding, use_cudnn_on_gpu, data_format)")
    expected_text = (
        "tf.nn.conv2d_transpose(output_shape=input_sizes, filters=filter, "
        "input=out_backprop, strides=strides, padding=padding, "
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

  def testStructure(self):
    for (text, expected) in [
        ("tf.data.experimental.DatasetStructure", "tf.data.DatasetSpec"),
        ("tf.data.experimental.OptionalStructure", "tf.OptionalSpec"),
        ("tf.data.experimental.RaggedTensorStructure", "tf.RaggedTensorSpec"),
        ("tf.data.experimental.SparseTensorStructure", "tf.SparseTensorSpec"),
        ("tf.data.experimental.Structure", "tf.TypeSpec"),
        ("tf.data.experimental.TensorArrayStructure", "tf.TensorArraySpec"),
        ("tf.data.experimental.TensorStructure", "tf.TensorSpec"),
    ]:
      _, unused_report, unused_errors, actual = self._upgrade(text)
      self.assertEqual(actual, expected)

  def testMapAndBatch(self):
    suffix = ".data.experimental.map_and_batch_with_legacy_function(args)"
    text = "tf" + suffix
    expected = "tf.compat.v1" + suffix
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
      expected_text = [
          "tf.image.resize(i, s, ", "preserve_aspect_ratio=p, ",
          "method=tf.image.ResizeMethod.%s)" % method.upper()
      ]
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
    for name in [
        "assert_greater", "assert_equal", "assert_none_equal", "assert_less",
        "assert_negative", "assert_positive", "assert_non_negative",
        "assert_non_positive", "assert_near", "assert_less",
        "assert_less_equal", "assert_greater", "assert_greater_equal",
        "assert_scalar"
    ]:
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
    text = ("tf.test.assert_equal_graph_def(a, b, checkpoint_v2=x, "
            "hash_table_shared_name=y)")
    expected = "tf.test.assert_equal_graph_def(actual=a, expected=b)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_is_tensor_upgrade(self):
    text = "tf.contrib.framework.is_tensor(x)"
    expected = "tf.is_tensor(x)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_is_tensor_direct_import_upgrade(self):
    text = "contrib_framework.is_tensor(x)"
    expected = "tf.is_tensor(x)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_CriticalSection_upgrade(self):
    text = "tf.contrib.framework.CriticalSection(shared_name='blah')"
    expected = "tf.CriticalSection(shared_name='blah')"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_sample_distorted_bounding_box(self):
    # pylint: disable=line-too-long
    text = "tf.image.sample_distorted_bounding_box(a, b, c, d, e, f, g, h, i, j)"
    expected = "tf.image.sample_distorted_bounding_box(image_size=a, bounding_boxes=b, seed=c, min_object_covered=e, aspect_ratio_range=f, area_range=g, max_attempts=h, use_image_if_no_bounding_boxes=i, name=j)"
    # pylint: enable=line-too-long
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_contrib_initialize(self):
    text = "tf.contrib.summary.initialize"
    expected = "tf.compat.v1.summary.initialize"
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
    self.assertIn("tf.flags and tf.app.flags have been removed", errors[0])

  def test_flags_flags(self):
    _, _, errors, _ = self._upgrade("tf.flags.FLAGS")
    self.assertIn("tf.flags and tf.app.flags have been removed", errors[0])

  def test_contrib_estimator_head_deprecation(self):
    for contrib_alias in ["tf.contrib.", "contrib_"]:
      api_symbols = ["binary_classification_head", "logistic_regression_head",
                     "multi_class_head", "multi_head", "multi_label_head",
                     "poisson_regression_head", "regression_head"]
      for symbol in api_symbols:
        text = contrib_alias + "estimator." + symbol
        _, report, _, _ = self._upgrade(text)
        self.assertIn("`tf.contrib.estimator.*_head` has been deprecated",
                      report)

  def test_contrib_layers_layer_norm_deprecation(self):
    for contrib_alias in ["tf.contrib.", "contrib_"]:
      _, report, _, _ = self._upgrade(contrib_alias + "layers.layer_norm")
      self.assertIn(
          "`tf.contrib.layers.layer_norm` has been deprecated", report)

  def test_contrib_rnn_deprecation(self):
    _, report, _, _ = self._upgrade("tf.contrib.rnn")
    self.assertIn("tf.contrib.rnn.* has been deprecated", report)

  def test_contrib_cudnn_rnn_deprecation(self):
    _, report, _, _ = self._upgrade("tf.contrib.cudnn_rnn")
    self.assertIn("tf.contrib.cudnn_rnn.* has been deprecated", report)

  def test_max_pool_2d(self):
    text = "tf.nn.max_pool(value=4)"
    expected_text = "tf.nn.max_pool2d(input=4)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def test_contrib_estimator_early_stopping(self):
    for contrib_alias in ["tf.contrib.", "contrib_"]:
      api_symbols = [
          "make_early_stopping_hook", "stop_if_higher_hook",
          "stop_if_lower_hook",
          "stop_if_no_decrease_hook", "stop_if_no_increase_hook"
      ]
      for symbol in api_symbols:
        text = contrib_alias + "estimator." + symbol
        expected_text = "tf.estimator.experimental." + symbol
        _, _, _, new_text = self._upgrade(text)
        self.assertEqual(expected_text, new_text)

  def test_contrib_rnn_cell(self):
    api_symbols = ["RNNCell", "BasicLSTMCell", "BasicRNNCell", "GRUCell",
                   "LSTMCell", "MultiRNNCell"]
    for symbol in api_symbols:
      text = "tf.contrib.rnn." + symbol
      expected_text = "tf.compat.v1.nn.rnn_cell." + symbol
      _, _, _, new_text = self._upgrade(text)
      self.assertEqual(expected_text, new_text)

  def test_contrib_rnn_function(self):
    api_symbols = ["static_rnn", "static_state_saving_rnn",
                   "static_bidirectional_rnn"]
    for symbol in api_symbols:
      text = "tf.contrib.rnn." + symbol
      expected_text = "tf.compat.v1.nn." + symbol
      _, _, _, new_text = self._upgrade(text)
      self.assertEqual(expected_text, new_text)

  def test_contrib_summary_generic(self):
    text = "tf.contrib.summary.generic('foo', myval, meta, 'fam', 42)"
    expected = ("tf.compat.v2.summary.write(tag='foo', data=myval, "
                "metadata=meta, step=42)")
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    # Arg errors come in alphabetical order of arguments, not appearance order.
    self.assertIn("'family' argument", errors[0])
    self.assertIn("'name' argument", errors[1])
    self.assertIn("tf.compat.v2.summary.*", errors[2])

  def test_contrib_summary_audio(self):
    text = "tf.contrib.summary.audio('foo', myval, 44100, 3, 'fam', 42)"
    expected = ("tf.compat.v2.summary.audio(name='foo', data=myval, "
                "sample_rate=44100, max_outputs=3, step=42)")
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("'family' argument", errors[0])
    self.assertIn("tf.compat.v2.summary.*", errors[1])

  def test_contrib_summary_histogram(self):
    text = "tf.contrib.summary.histogram('foo', myval, 'fam', 42)"
    expected = ("tf.compat.v2.summary.histogram(name='foo', data=myval, "
                "step=42)")
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("'family' argument", errors[0])
    self.assertIn("tf.compat.v2.summary.*", errors[1])

  def test_contrib_summary_image(self):
    text = "tf.contrib.summary.image('foo', myval, red, 3, 'fam', 42)"
    expected = ("tf.compat.v2.summary.image(name='foo', data=myval, "
                "max_outputs=3, step=42)")
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("'bad_color' argument", errors[0])
    self.assertIn("'family' argument", errors[1])
    self.assertIn("tf.compat.v2.summary.*", errors[2])

  def test_contrib_summary_scalar(self):
    text = "tf.contrib.summary.scalar('foo', myval, 'fam', 42)"
    expected = ("tf.compat.v2.summary.scalar(name='foo', data=myval, "
                "step=42)")
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("'family' argument", errors[0])
    self.assertIn("tf.compat.v2.summary.*", errors[1])

  def test_contrib_summary_generic_nostep(self):
    text = "tf.contrib.summary.generic('foo', myval)"
    expected = ("tf.compat.v2.summary.write(tag='foo', data=myval, "
                "step=tf.compat.v1.train.get_or_create_global_step())")
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("'name' argument", errors[0])
    self.assertIn("'step' argument", errors[1])
    self.assertIn("tf.compat.v2.summary.*", errors[2])

  def test_contrib_summary_audio_nostep(self):
    text = "tf.contrib.summary.audio('foo', myval, 44100)"
    expected = ("tf.compat.v2.summary.audio(name='foo', data=myval, "
                "sample_rate=44100, "
                "step=tf.compat.v1.train.get_or_create_global_step())")
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("'step' argument", errors[0])
    self.assertIn("tf.compat.v2.summary.*", errors[1])

  def test_contrib_summary_histogram_nostep(self):
    text = "tf.contrib.summary.histogram('foo', myval)"
    expected = ("tf.compat.v2.summary.histogram(name='foo', data=myval, "
                "step=tf.compat.v1.train.get_or_create_global_step())")
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("'step' argument", errors[0])
    self.assertIn("tf.compat.v2.summary.*", errors[1])

  def test_contrib_summary_image_nostep(self):
    text = "tf.contrib.summary.image('foo', myval)"
    expected = ("tf.compat.v2.summary.image(name='foo', data=myval, "
                "step=tf.compat.v1.train.get_or_create_global_step())")
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("'step' argument", errors[0])
    self.assertIn("tf.compat.v2.summary.*", errors[1])

  def test_contrib_summary_scalar_nostep(self):
    text = "tf.contrib.summary.scalar('foo', myval)"
    expected = ("tf.compat.v2.summary.scalar(name='foo', data=myval, "
                "step=tf.compat.v1.train.get_or_create_global_step())")
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("'step' argument", errors[0])
    self.assertIn("tf.compat.v2.summary.*", errors[1])

  def test_contrib_summary_graph(self):
    text = "tf.contrib.summary.graph(my_graph)"
    _, _, errors, _ = self._upgrade(text)
    expected_error = "tf.compat.v2.summary.trace"
    self.assertIn(expected_error, errors[0])

  def test_contrib_summary_import_event(self):
    text = "tf.contrib.summary.import_event(my_event)"
    _, _, errors, _ = self._upgrade(text)
    expected_error = "tf.compat.v2.summary.experimental.write_raw_pb"
    self.assertIn(expected_error, errors[0])

  def test_contrib_summary_flush(self):
    text = "tf.contrib.summary.flush(writer=foo)"
    expected = "tf.compat.v2.summary.flush(writer=foo)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_contrib_summary_create_file_writer(self):
    text = ("tf.contrib.summary.create_file_writer('my_logdir', 0, 1000, "
            "'.foo', 'shared-name')")
    expected = ("tf.compat.v2.summary.create_file_writer(logdir='my_logdir', "
                "max_queue=0, flush_millis=1000, filename_suffix='.foo')")
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("'name' argument", errors[0])
    self.assertIn("no longer re-uses existing event files", errors[1])

  def test_contrib_summary_always_record_summaries(self):
    text = "tf.contrib.summary.always_record_summaries()"
    expected = "tf.compat.v2.summary.record_if(True)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_contrib_summary_never_record_summaries(self):
    text = "tf.contrib.summary.never_record_summaries()"
    expected = "tf.compat.v2.summary.record_if(False)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_contrib_summary_record_summaries_every_n_global_steps(self):
    text = "tf.contrib.summary.record_summaries_every_n_global_steps(10)"
    _, _, errors, _ = self._upgrade(text)
    expected_error = "replaced by a call to tf.compat.v2.summary.record_if()"
    self.assertIn(expected_error, errors[0])

  def test_contrib_summary_all_summary_ops(self):
    text = "tf.contrib.summary.all_summary_ops()"
    expected = "tf.compat.v1.summary.all_v2_summary_ops()"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_contrib_summary_full_example(self):
    deindent = lambda n, s: "\n".join(line[n:] for line in s.split("\n"))
    text = deindent(4, """
    import tensorflow as tf
    tf.enable_eager_execution()
    writer = tf.contrib.summary.create_file_writer(
        "/tmp/migration_test", flush_millis=1000)
    with writer.as_default(), tf.contrib.summary.always_record_summaries():
      tf.contrib.summary.scalar("loss", 0.42)
      tf.contrib.summary.histogram("weights", [1.0, 2.0], step=7)
      tf.contrib.summary.flush()
    """)
    expected = deindent(4, """
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    writer = tf.compat.v2.summary.create_file_writer(
        logdir="/tmp/migration_test", flush_millis=1000)
    with writer.as_default(), tf.compat.v2.summary.record_if(True):
      tf.compat.v2.summary.scalar(name="loss", data=0.42, step=tf.compat.v1.train.get_or_create_global_step())
      tf.compat.v2.summary.histogram(name="weights", data=[1.0, 2.0], step=7)
      tf.compat.v2.summary.flush()
    """)
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_summary_api_warning(self):
    text = "tf.summary.scalar('foo', 42)"
    _, report, _, _ = self._upgrade(text)
    expected_info = "TF 1.x summary API cannot be automatically migrated"
    self.assertIn(expected_info, report)

  def test_avg_pool_2d(self):
    text = "tf.nn.avg_pool(value=4)"
    expected_text = "tf.nn.avg_pool2d(input=4)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def test_saved_model_load(self):
    text = "tf.saved_model.load(sess, ['foo_graph'])"
    expected = "tf.compat.v1.saved_model.load(sess, ['foo_graph'])"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_saved_model_load_v2(self):
    text = "tf.saved_model.load_v2('/tmp/blah')"
    expected = "tf.compat.v2.saved_model.load('/tmp/blah')"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_app_flags(self):
    text = "flags = tf.app.flags"
    expected = "flags = tf.compat.v1.app.flags"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_uniform_unit_scaling_initializer(self):
    text = "tf.uniform_unit_scaling_initializer(0.5)"
    expected_text = ("tf.compat.v1.keras.initializers.VarianceScaling("
                     "scale=0.5, distribution=\"uniform\")")
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "tf.initializers.uniform_unit_scaling(0.5)"
    expected_text = ("tf.compat.v1.keras.initializers.VarianceScaling("
                     "scale=0.5, distribution=\"uniform\")")
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def test_name_scope(self):
    text = "tf.name_scope(None, default_name, [some, values])"
    expected_text = "tf.name_scope(name=default_name)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "tf.name_scope(default_name=default_name, values=stuff)"
    expected_text = "tf.name_scope(name=default_name)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "tf.name_scope(name=n, default_name=d, values=s)"
    expected_text = "tf.compat.v1.name_scope(name=n, default_name=d, values=s)"
    _, report, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)
    self.assertIn("`name` passed to `name_scope`", report)

    text = "tf.name_scope(name=None, values=stuff)"
    _, _, errors, _ = self._upgrade(text)
    self.assertIn("name_scope call with neither name nor default_name",
                  errors[0])

  @parameterized.parameters(
      # Rename parameter: delimiter -> sep and add .to_sparse()
      ["tf.string_split('test', delimiter=' ')",
       "tf.strings.split(input='test', sep=' ').to_sparse()"],
      # Rename parameter: source -> input
      ["tf.strings.split(source='test1')",
       "tf.strings.split(input='test1').to_sparse()"],
      # Use compat.v1 for skip_empty parameter.
      ["tf.string_split('test', ' ', True)",
       "tf.compat.v1.string_split(source='test', sep=' ', skip_empty=True)"],
      ["tf.string_split('test', ' ', skip_empty=False)",
       "tf.strings.split(input='test', sep=' ').to_sparse()"],
      # Split behavior for sep=None changed.  (In particular, it now splits on
      # all whitespace, not just the space character)
      ["tf.string_split(x)",
       "tf.compat.v1.string_split(source=x)"],
      # Split behavior for sep='' changed:
      ["tf.string_split(x, '')",
       "tf.strings.bytes_split(input=x).to_sparse()"],
      ["tf.string_split(x, sep='')",
       "tf.strings.bytes_split(input=x).to_sparse()"],
      ["tf.string_split(x, delimiter='')",
       "tf.strings.bytes_split(input=x).to_sparse()"],
      ["tf.string_split(x, '', result_type='RaggedTensor')",
       "tf.strings.bytes_split(input=x)"],
      # If sep is a variable, we can't tell if it's empty:
      ["tf.string_split(x, sep)",
       "tf.compat.v1.string_split(source=x, sep=sep)"],
      # If sep is a non-empty string literal, then we don't need compat.v1.
      ["tf.string_split(x, 'non-empty-sep')",
       "tf.strings.split(input=x, sep='non-empty-sep').to_sparse()"],
      # Add to_sparse unless result_type is RaggedTensor:
      ["tf.string_split(x, ' ')",
       "tf.strings.split(input=x, sep=' ').to_sparse()"],
      ["tf.string_split(x, ' ', result_type='SparseTensor')",
       "tf.strings.split(input=x, sep=' ').to_sparse()"],
      ["tf.string_split(x, ' ', result_type='RaggedTensor')",
       "tf.strings.split(input=x, sep=' ')"],
      ["tf.string_split(x, ' ', result_type=x)",
       "tf.compat.v1.string_split(source=x, sep=' ', result_type=x)"],
  )  # pyformat: disable
  # TODO(b/129398290)
  def DISABLED_test_string_split(self, text, expected_text):
    """Tests for transforming from tf.string_split."""
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  @parameterized.parameters(
      # Add to_sparse unless result_type is RaggedTensor:
      ["tf.strings.split(x, sep)",
       "tf.strings.split(x, sep).to_sparse()"],
      ["tf.strings.split(x, sep, result_type='SparseTensor')",
       "tf.strings.split(x, sep).to_sparse()"],
      ["tf.strings.split(x, sep, result_type='RaggedTensor')",
       "tf.strings.split(x, sep)"],
      ["tf.strings.split(x, sep, result_type=x)",
       "tf.compat.v1.strings.split(x, sep, result_type=x)"],
  )  # pyformat: disable
  def test_strings_split(self, text, expected_text):
    """Tests for transforming from tf.strings.split."""
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def test_sdca_to_raw_ops(self):
    text = "tf.train.sdca_fprint(input_tensor)"
    expected_text = "tf.raw_ops.SdcaFprint(input=input_tensor)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "tf.train.sdca_fprint(input, name=n)"
    expected_text = "tf.raw_ops.SdcaFprint(input=input, name=n)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "tf.train.sdca_shrink_l1(w, l, ll)"
    expected_text = "tf.raw_ops.SdcaShrinkL1(weights=w, l1=l, l2=ll)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = (
        "tf.train.sdca_optimizer(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o)")
    expected_text = (
        "tf.raw_ops.SdcaOptimizer(sparse_example_indices=a, "
        "sparse_feature_indices=b, sparse_feature_values=c, dense_features=d, "
        "example_weights=e, example_labels=f, sparse_indices=g, "
        "sparse_weights=h, dense_weights=i, example_state_data=j, loss_type=k, "
        "l1=l, l2=m, num_loss_partitions=n, num_inner_iterations=o)")
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def test_contrib_to_addons_move(self):
    small_mapping = {
        "tf.contrib.layers.poincare_normalize":
            "tfa.layers.PoincareNormalize",
        "tf.contrib.layers.maxout":
            "tfa.layers.Maxout",
        "tf.contrib.layers.group_norm":
            "tfa.layers.GroupNormalization",
        "tf.contrib.layers.instance_norm":
            "tfa.layers.InstanceNormalization",
    }
    for symbol, replacement in small_mapping.items():
      text = "{}('stuff', *args, **kwargs)".format(symbol)
      _, report, _, _ = self._upgrade(text)
      self.assertIn(replacement, report)

  def testXlaExperimental(self):
    text = "tf.xla.experimental.jit_scope(0)"
    expected_text = "tf.xla.experimental.jit_scope(0)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = "tf.xla.experimental.compile(0)"
    expected_text = "tf.xla.experimental.compile(0)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testNnErosion2d(self):
    text = "tf.nn.erosion2d(v, k, s, r, p)"
    expected_text = "tf.nn.erosion2d(v, k, s, r, p, data_format='NHWC')"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testNnDilation2d(self):
    text = "tf.nn.dilation2d(v, k, s, r, p)"
    expected_text = "tf.nn.dilation2d(v, k, s, r, p, data_format='NHWC')"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testPywrapTensorflowWarning(self):
    text = "tf.pywrap_tensorflow.foo()"
    expected = "tf.pywrap_tensorflow.foo()"
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("`tf.pywrap_tensorflow` will not be distributed", errors[0])

  def testKerasSaveModelFormat(self):
    text = "tf.keras.models.save_model(model, path)"
    expected_text = "tf.keras.models.save_model(model, path, save_format='h5')"
    _, report, _, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)
    self.assertNotIn(
        "saves to the Tensorflow SavedModel format by default", report)

    _, report, _, _ = self._upgrade("model.save(path)")
    self.assertIn(
        "saves to the Tensorflow SavedModel format by default", report)

  def test_distribute_strategy(self):
    text = "tf.contrib.distribute.CrossDeviceOps()"
    expected = "tf.distribute.CrossDeviceOps()"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

    text = "tf.contrib.distribute.MirroredStrategy"
    expected = "tf.contrib.distribute.MirroredStrategy"
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("migrated to tf.distribute.MirroredStrategy", errors[0])

    text = "tf.distribute.MirroredStrategy"
    expected = "tf.distribute.MirroredStrategy"
    _, report, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("tf.distribute.MirroredStrategy API has changed", report)
    self.assertIn("make_dataset_iterator->experimental_distribute_dataset",
                  report)

    text = "tf.contrib.distribute.TPUStrategy"
    expected = "tf.contrib.distribute.TPUStrategy"
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("migrated to tf.distribute.TPUStrategy",
                  errors[0])

    text = "tf.contrib.distribute.foo"
    expected = "tf.contrib.distribute.foo"
    _, report, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("tf.contrib.distribute.* have been migrated", report)

  def test_decode_raw(self):
    text = "tf.io.decode_raw(bytes=[1,2,3], output_dtype=tf.int32)"
    expected_text = (
        "tf.io.decode_raw(input_bytes=[1,2,3], output_dtype=tf.int32)")
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def testRecomputeGrad(self):
    text = "tf.contrib.layers.recompute_grad()"
    expected = "tf.recompute_grad()"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_load_variable(self):
    text = "tf.contrib.framework.load_variable('a')"
    expected_text = (
        "tf.train.load_variable('a')")
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)
    text = "tf.contrib.framework.load_variable(checkpoint_dir='a')"
    expected_text = (
        "tf.train.load_variable(ckpt_dir_or_file='a')")
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def test_import_rename_analysis(self):
    old_symbol = "tf.conj(a)"
    new_symbol = "tf.math.conj(a)"

    import_header = "import tensorflow as tf\n"
    text = import_header + old_symbol
    expected_text = "import tensorflow.compat.v2 as tf\n" + new_symbol
    _, unused_report, unused_errors, new_text = self._upgrade(
        text, import_rename=True)
    self.assertEqual(new_text, expected_text)

    import_header = "import tensorflow as tf, other_import as y\n"
    text = import_header + old_symbol
    new_import_header = "import tensorflow.compat.v2 as tf, other_import as y\n"
    expected_text = new_import_header + new_symbol
    _, unused_report, unused_errors, new_text = self._upgrade(
        text, import_rename=True)
    self.assertEqual(new_text, expected_text)

    import_header = ("import tensorflow as tf\n"
                     "import tensorflow.compat.v1 as tf_v1\n"
                     "import tensorflow.compat.v2 as tf_v2\n")
    text = import_header + old_symbol
    expected_header = ("import tensorflow.compat.v2 as tf\n"
                       "import tensorflow.compat.v1 as tf_v1\n"
                       "import tensorflow.compat.v2 as tf_v2\n")
    expected_text = expected_header + new_symbol
    _, _, _, new_text = self._upgrade(text, import_rename=True)
    self.assertEqual(new_text, expected_text)

    import_header = ("import tensorflow.compat.v1 as tf\n"
                     "import tensorflow.compat.v1 as tf_v1\n"
                     "import tensorflow.compat.v2 as tf_v2\n")
    text = import_header + old_symbol
    expected_header = ("import tensorflow.compat.v2 as tf\n"
                       "import tensorflow.compat.v1 as tf_v1\n"
                       "import tensorflow.compat.v2 as tf_v2\n")
    expected_text = expected_header + new_symbol
    _, _, _, new_text = self._upgrade(
        text, import_rename=True, upgrade_compat_v1_import=True)
    self.assertEqual(new_text, expected_text)

    import_header = ("import tensorflow.compat.v1 as tf\n"
                     "import tensorflow.compat.v1 as tf_v1\n"
                     "import tensorflow.compat.v2 as tf_v2\n")
    text = import_header + old_symbol
    expected_header = ("import tensorflow as tf\n"
                       "import tensorflow.compat.v1 as tf_v1\n"
                       "import tensorflow.compat.v2 as tf_v2\n")
    expected_text = expected_header + new_symbol
    _, _, _, new_text = self._upgrade(
        text, import_rename=False, upgrade_compat_v1_import=True)
    self.assertEqual(new_text, expected_text)

    import_header = "from tensorflow import foo\n"
    text = import_header + old_symbol
    expected_text = "from tensorflow.compat.v2 import foo\n" + new_symbol
    _, unused_report, unused_errors, new_text = self._upgrade(
        text, import_rename=True)
    self.assertEqual(new_text, expected_text)

    import_header = "from tensorflow import *\n"
    text = import_header + old_symbol
    expected_text = "from tensorflow.compat.v2 import *\n" + new_symbol
    _, unused_report, unused_errors, new_text = self._upgrade(
        text, import_rename=True)
    self.assertEqual(new_text, expected_text)

    import_header = "from tensorflow.foo import bar\n"
    text = import_header + old_symbol
    expected_text = "from tensorflow.compat.v2.foo import bar\n" + new_symbol
    _, unused_report, unused_errors, new_text = self._upgrade(
        text, import_rename=True)
    self.assertEqual(new_text, expected_text)

    import_header = ("from tensorflow import foo as tf\n"
                     "from tensorflow.compat import v1 as tf_v1\n"
                     "from tensorflow.compat import v2 as tf_v2\n")
    text = import_header + old_symbol
    expected_header = ("from tensorflow.compat.v2 import foo as tf\n"
                       "from tensorflow.compat import v1 as tf_v1\n"
                       "from tensorflow.compat import v2 as tf_v2\n")
    expected_text = expected_header + new_symbol
    _, _, _, new_text = self._upgrade(text, import_rename=True)
    self.assertEqual(new_text, expected_text)

  def test_import_analysis(self):
    old_symbol = "tf.conj(a)"
    new_symbol = "tf.math.conj(a)"

    # We upgrade the base un-versioned tensorflow aliased as tf
    import_header = "import tensorflow as tf\n"
    text = import_header + old_symbol
    expected_text = import_header + new_symbol
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    import_header = ("import tensorflow as tf\n"
                     "import tensorflow.compat.v1 as tf_v1\n"
                     "import tensorflow.compat.v2 as tf_v2\n")
    text = import_header + old_symbol
    expected_text = import_header + new_symbol
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    # We don't handle unaliased tensorflow imports currently,
    # So the upgrade script show log errors
    import_header = "import tensorflow\n"
    text = import_header + old_symbol
    expected_text = import_header + old_symbol
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)
    self.assertIn("unaliased `import tensorflow`", "\n".join(errors))

    # Upgrading explicitly-versioned tf code is unsafe, but we don't
    # need to throw errors when we detect explicitly-versioned tf.
    import_header = "import tensorflow.compat.v1 as tf\n"
    text = import_header + old_symbol
    expected_text = import_header + old_symbol
    _, report, errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)
    self.assertIn("`tensorflow.compat.v1` was directly imported as `tf`",
                  report)
    self.assertEmpty(errors)

    import_header = "from tensorflow.compat import v1 as tf\n"
    text = import_header + old_symbol
    expected_text = import_header + old_symbol
    _, report, errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)
    self.assertIn("`tensorflow.compat.v1` was directly imported as `tf`",
                  report)
    self.assertEmpty(errors)

    import_header = "from tensorflow.compat import v1 as tf, v2 as tf2\n"
    text = import_header + old_symbol
    expected_text = import_header + old_symbol
    _, report, errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)
    self.assertIn("`tensorflow.compat.v1` was directly imported as `tf`",
                  report)
    self.assertEmpty(errors)

    import_header = "import tensorflow.compat.v2 as tf\n"
    text = import_header + old_symbol
    expected_text = import_header + old_symbol
    _, report, errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)
    self.assertIn("`tensorflow.compat.v2` was directly imported as `tf`",
                  report)
    self.assertEmpty(errors)

    import_header = "from tensorflow.compat import v1 as tf1, v2 as tf\n"
    text = import_header + old_symbol
    expected_text = import_header + old_symbol
    _, report, errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)
    self.assertIn("`tensorflow.compat.v2` was directly imported as `tf`",
                  report)
    self.assertEmpty(errors)

  def test_api_spec_reset_between_files(self):
    for old_symbol, new_symbol in [
        ("tf.conj(a)", "tf.math.conj(a)"),
        ("tf.to_int32(x)", "tf.cast(x, dtype=tf.int32)")]:

      ## Test that the api spec is reset in between files:
      import_header = "import tensorflow.compat.v2 as tf\n"
      text_a = import_header + old_symbol
      expected_text_a = import_header + old_symbol
      text_b = old_symbol
      expected_text_b = new_symbol
      results = self._upgrade_multiple([text_a, text_b])
      result_a, result_b = results[0], results[1]
      self.assertEqual(result_a[3], expected_text_a)
      self.assertEqual(result_b[3], expected_text_b)

  def test_model_to_estimator_checkpoint_warning(self):
    text = "tf.keras.estimator.model_to_estimator(model)"
    _, report, _, _ = self._upgrade(text)
    expected_info = "will save object-based checkpoints"
    self.assertIn(expected_info, report)

  def test_keras_experimental_export_warning(self):
    text = "tf.keras.experimental.export_saved_model"
    _, report, _, _ = self._upgrade(text)
    expected_info = "Please use model.save"
    self.assertIn(expected_info, report)


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

  def testInplaceNoOutputChangeOnErrorHandling(self):
    """In place file should not be modified when parsing error is handled."""
    temp_file = tempfile.NamedTemporaryFile("w", delete=False)
    original = "print 'a' \n"
    upgraded = "print 'a' \n"
    temp_file.write(original)
    temp_file.close()
    upgrader = ast_edits.ASTCodeUpgrader(tf_upgrade_v2.TFAPIChangeSpec())
    upgrader.process_file(
        temp_file.name, temp_file.name, no_change_to_outfile_on_error=True)
    self.assertAllEqual(open(temp_file.name).read(), upgraded)
    os.unlink(temp_file.name)

  def testInplaceEmptyOutputOnError(self):
    """In place file becomes empty when parsing error is not handled."""
    temp_file = tempfile.NamedTemporaryFile("w", delete=False)
    original = "print 'a' \n"
    upgraded = ""
    temp_file.write(original)
    temp_file.close()
    upgrader = ast_edits.ASTCodeUpgrader(tf_upgrade_v2.TFAPIChangeSpec())
    upgrader.process_file(temp_file.name, temp_file.name)
    self.assertAllEqual(open(temp_file.name).read(), upgraded)
    os.unlink(temp_file.name)


if __name__ == "__main__":
  test_lib.main()
