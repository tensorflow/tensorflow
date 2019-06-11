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
"""Facilities for creating multiple test combinations.

Here is an example of testing various optimizers in Eager and Graph mode:

class AdditionExample(test.TestCase, parameterized.TestCase):
  @combinations.generate(
     combinations.combine(mode=["graph", "eager"],
                          optimizer=[AdamOptimizer(),
                                     GradientDescentOptimizer()]))
  def testOptimizer(self, optimizer):
    ... f(optimizer)...

This will run `testOptimizer` 4 times with the specified optimizers: 2 in
Eager and 2 in Graph mode.
The test will be provided with arguments that match the arguments of combine
by name.  It is necessary to request all arguments, except for `mode`, which is
optional.

`combine()` function is available for creating a cross product of various
options.  `times()` function exists for creating a product of N `combine()`-ed
results.  See below.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import sys
import types
import unittest

from absl.testing import parameterized
import six

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_inspect


GPU_TEST = "test_gpu" in sys.argv[0]
TPU_TEST = "test_tpu" in sys.argv[0]


# TODO(rchao): Rename `distribution` parameter to `strategy` or
# `distribute_strategy`
def generate(combinations):
  """A decorator for generating test cases of a test method or a test class.

  Args:
    combinations: a list of dictionaries created using combine() and times().

  Restrictions:
   -- the "mode" argument can be either "eager" or "graph".  It's "graph" by
      default.
   -- arguments of the test method must match by name to get the corresponding
      value of the combination.  Tests must accept all arguments except the
      "mode", "required_tpu" and "required_gpus".
   -- "distribution" argument is special and optional.  It is meant for passing
      instances of DistributionStrategy.  Each instance is to be passed as via
      `NamedDistribution`.  If using "distribution", "required_gpus" and
      "required_tpu" should be specified via the NamedDistribution instance,
      rather than as separate arguments.
   -- "required_tpu" argument is special and optional.  If not `None`, then the
      test will be skipped if TPUs aren't available.
   -- "required_gpus" argument is special and optional.  If not `None`, then the
      test will be skipped if the specified number of GPUs aren't available.

  Returns:
    a decorator that will cause the test method or the test class to be run
    under the specified conditions.

  Raises:
    ValueError - if "mode" argument wasn't either "eager" or "graph" or if other
      arguments were not accepted by the test method.
  """

  def decorator(test_method_or_class):
    """The decorator to be returned."""

    # Generate good test names that can be used with --test_filter.
    named_combinations = []
    for combination in combinations:
      # We use OrderedDicts in `combine()` and `times()` to ensure stable
      # order of keys in each dictionary.
      assert isinstance(combination, OrderedDict)
      name = "".join([
          "_{}_{}".format(
              "".join(filter(str.isalnum, key)),
              "".join(filter(str.isalnum, str(value))))
          for key, value in combination.items()
      ])
      named_combinations.append(
          OrderedDict(
              list(combination.items()) + [("testcase_name",
                                            "_test{}".format(name))]))

    if isinstance(test_method_or_class, type):
      class_object = test_method_or_class
      class_object._test_method_ids = test_method_ids = {}
      for name, test_method in six.iteritems(class_object.__dict__.copy()):
        if (name.startswith(unittest.TestLoader.testMethodPrefix) and
            isinstance(test_method, types.FunctionType)):
          delattr(class_object, name)
          methods = {}
          parameterized._update_class_dict_for_param_test_case(
              class_object.__name__, methods, test_method_ids, name,
              parameterized._ParameterizedTestIter(
                  _augment_with_special_arguments(test_method),
                  named_combinations, parameterized._NAMED, name))
          for method_name, method in six.iteritems(methods):
            setattr(class_object, method_name, method)

      return class_object
    else:
      test_method = _augment_with_special_arguments(test_method_or_class)
      return parameterized.named_parameters(*named_combinations)(test_method)

  return decorator


def _augment_with_special_arguments(test_method):
  def decorated(self, **kwargs):
    """A wrapped test method that treats some arguments in a special way."""
    mode = kwargs.pop("mode", "graph")

    distribution = kwargs.get("distribution", None)
    required_tpu = kwargs.pop("required_tpu", False)
    required_gpus = kwargs.pop("required_gpus", None)

    if distribution:
      assert required_gpus is None, (
          "Do not use `required_gpus` and `distribution` together.")
      assert required_tpu is False, (
          "Do not use `required_tpu` and `distribution` together.")
      required_gpus = distribution.required_gpus
      required_tpu = distribution.required_tpu

    maybe_skip_test(self, required_tpu, required_gpus)

    # At this point, `kwargs` doesn't have `required_gpus` or `required_tpu`
    # that the user might have specified.  `kwargs` still has `mode`, which
    # the test is allowed to accept or ignore.
    requested_arguments = tf_inspect.getfullargspec(test_method).args
    missing_arguments = set(list(kwargs.keys()) + ["self"]).difference(
        set(requested_arguments + ["mode"]))
    if missing_arguments:
      raise ValueError("The test is missing arguments {} .".format(
          missing_arguments))

    kwargs_to_pass = {}
    for arg in requested_arguments:
      if arg == "self":
        kwargs_to_pass[arg] = self
      elif arg == "mode":
        kwargs_to_pass[arg] = mode
      else:
        kwargs_to_pass[arg] = kwargs[arg]

    if mode == "eager":
      with context.eager_mode():
        if distribution:
          kwargs_to_pass["distribution"] = distribution.strategy
        test_method(**kwargs_to_pass)
    elif mode == "graph":
      with ops.Graph().as_default(), context.graph_mode():
        if distribution:
          kwargs_to_pass["distribution"] = distribution.strategy
        test_method(**kwargs_to_pass)
    else:
      raise ValueError(
          "'mode' has to be either 'eager' or 'graph' and not {}".format(
              mode))
  return decorated


def maybe_skip_test(test_case, is_tpu_required, num_gpus_required):
  if is_tpu_required and not TPU_TEST:
    test_case.skipTest("Test requires a TPU, but it's not available.")
  if not is_tpu_required and TPU_TEST:
    test_case.skipTest("Test that doesn't require a TPU.")

  if not num_gpus_required:
    if GPU_TEST:
      test_case.skipTest("Test that doesn't require GPUs.")
  elif context.num_gpus() < num_gpus_required:
    # TODO(priyag): Consider allowing tests in graph mode using soft
    # placement.
    test_case.skipTest(
        "{} GPUs are not available for this test. {} GPUs are available".format(
            num_gpus_required, context.num_gpus()))


def combine(**kwargs):
  """Generate combinations based on its keyword arguments.

  Two sets of returned combinations can be concatenated using +.  Their product
  can be computed using `times()`.

  Args:
    **kwargs: keyword arguments of form `option=[possibilities, ...]`
         or `option=the_only_possibility`.

  Returns:
    a list of dictionaries for each combination. Keys in the dictionaries are
    the keyword argument names.  Each key has one value - one of the
    corresponding keyword argument values.
  """
  if not kwargs:
    return [OrderedDict()]

  sort_by_key = lambda k: k[0]
  kwargs = OrderedDict(sorted(kwargs.items(), key=sort_by_key))
  first = list(kwargs.items())[0]

  rest = dict(list(kwargs.items())[1:])
  rest_combined = combine(**rest)

  key = first[0]
  values = first[1]
  if not isinstance(values, list):
    values = [values]

  return [
      OrderedDict(sorted(list(combined.items()) + [(key, v)], key=sort_by_key))
      for v in values
      for combined in rest_combined
  ]


def times(*combined):
  """Generate a product of N sets of combinations.

  times(combine(a=[1,2]), combine(b=[3,4])) == combine(a=[1,2], b=[3,4])

  Args:
    *combined: N lists of dictionaries that specify combinations.

  Returns:
    a list of dictionaries for each combination.

  Raises:
    ValueError: if some of the inputs have overlapping keys.
  """
  assert combined

  if len(combined) == 1:
    return combined[0]

  first = combined[0]
  rest_combined = times(*combined[1:])

  combined_results = []
  for a in first:
    for b in rest_combined:
      if set(a.keys()).intersection(set(b.keys())):
        raise ValueError("Keys need to not overlap: {} vs {}".format(
            a.keys(), b.keys()))

      combined_results.append(OrderedDict(list(a.items()) + list(b.items())))
  return combined_results


class NamedObject(object):
  """A class that translates an object into a good test name."""

  def __init__(self, name, obj):
    self._name = name
    self._obj = obj

  def __getattr__(self, name):
    return getattr(self._obj, name)

  def __call__(self, *args, **kwargs):
    return self._obj(*args, **kwargs)

  def __repr__(self):
    return self._name


class NamedDistribution(object):
  """Translates DistributionStrategy and its data into a good name."""

  def __init__(self, name, distribution_fn, required_gpus=None,
               required_tpu=False):
    self._distribution_fn = distribution_fn
    self._name = name
    self._required_gpus = required_gpus
    self._required_tpu = required_tpu

  def __repr__(self):
    return self._name

  @property
  def strategy(self):
    return self._distribution_fn()

  @property
  def required_gpus(self):
    return self._required_gpus

  @property
  def required_tpu(self):
    return self._required_tpu


class NamedDistributionPair(object):
  """NamedDistribution but for a pair of strategies."""

  def __init__(self, named_distribution_1, named_distribution_2):
    self._named_distribution_1 = named_distribution_1
    self._named_distribution_2 = named_distribution_2
    self._name = str(named_distribution_1) + "_" + str(named_distribution_2)

    if not named_distribution_1.required_gpus:
      self._required_gpus = named_distribution_2.required_gpus
    elif not named_distribution_2.required_gpus:
      self._required_gpus = named_distribution_1.required_gpus
    else:
      self._required_gpus = max(self._named_distribution_1.required_gpus,
                                self._named_distribution_2.required_gpus)

    self._required_tpu = (
        self._named_distribution_1.required_tpu or
        self._named_distribution_2.required_tpu)

  def __repr__(self):
    return self._name

  @property
  def strategy_1(self):
    return self._named_distribution_1._distribution_fn()

  @property
  def strategy_2(self):
    return self._named_distribution_2._distribution_fn()

  @property
  def num_gpus_required(self):
    return self._required_gpus

  @property
  def is_tpu_required(self):
    return self._required_tpu
