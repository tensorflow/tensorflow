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
from absl.testing import parameterized

from tensorflow.contrib.distribute.python import mirrored_strategy
from tensorflow.contrib.distribute.python import one_device_strategy
from tensorflow.contrib.distribute.python import tpu_strategy
from tensorflow.contrib.optimizer_v2 import adam as adam_v2
from tensorflow.contrib.optimizer_v2 import gradient_descent as gradient_descent_v2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.training import adam
from tensorflow.python.training import distribute as distribute_lib
from tensorflow.python.training import gradient_descent
from tensorflow.python.util import tf_inspect


GPU_TEST = "test_gpu" in sys.argv[0]
TPU_TEST = "test_tpu" in sys.argv[0]


def generate(combinations):
  """A decorator for generating test cases of a test method or a test class.

  Args:
    combinations: a list of dictionaries created using combine() and times().

  Restrictions:
   -- there should always be a "mode" argument.  Accepted values are "eager"
      and "graph".
   -- arguments of the test method must match by name to get the corresponding
      value of the combination.  Tests must accept all arguments (except "mode",
      which is optional).
   -- distribution argument is special.  It is meant for passing instances of
      DistributionStrategy.  Each instance is to be passed as `(<int>,
      <DistributionStrategy>)` tuple, where <int> is the number of required
      GPUs.  If the required number of GPUs for the DistributionStrategy isn't
      available then the test case is going to be skipped.

  Returns:
    a decorator that will cause the test method to be run under the specified
    conditions.

  Raises:
    ValueError - if "mode" argument wasn't either "eager" or "graph.
  """

  def decorator(test_function):
    """The decorator to be returned."""

    # Generate good test names that can be used with --test_filter.
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
      combination.update({"testcase_name": "_test{}".format(name)})

    @parameterized.named_parameters(*combinations)
    def decorated(self, **kwargs):
      """A wrapped test method that sets up `test_function`."""
      assert "mode" in kwargs
      mode = kwargs["mode"]

      if "distribution" in kwargs:
        distribution = kwargs["distribution"]
        kwargs["distribution"] = distribution.strategy
        if distribution.required_tpu and not TPU_TEST:
          self.skipTest("Test requires a TPU, but it's not available.")
        if not distribution.required_tpu and TPU_TEST:
          self.skipTest("Test that doesn't require a TPU.")

        if not distribution.required_gpus:
          if GPU_TEST:
            self.skipTest("Test that doesn't require GPUs.")
        elif context.num_gpus() < distribution.required_gpus:
          self.skipTest(
              "{} GPUs are not available for this test. {} GPUs are available".
              format(distribution.required_gpus, context.num_gpus()))

      requested_arguments = tf_inspect.getfullargspec(test_function).args
      missing_arguments = set(list(kwargs.keys()) + ["self"]).difference(
          set(requested_arguments + ["mode"]))
      if missing_arguments:
        raise ValueError("The test is missing arguments {} .".format(
            missing_arguments))

      kwargs_to_pass = {}
      for arg in requested_arguments:
        if arg == "self":
          kwargs_to_pass[arg] = self
        else:
          kwargs_to_pass[arg] = kwargs[arg]

      if mode == "eager":
        with context.eager_mode(), ops.Graph().as_default():
          test_function(**kwargs_to_pass)
      elif mode == "graph":
        with context.graph_mode(), ops.Graph().as_default():
          test_function(**kwargs_to_pass)
      else:
        raise ValueError(
            "'mode' has to be either 'eager' or 'graph' and not {}".format(
                mode))

    return decorated
  return decorator


def combine(**kwargs):
  """Generate combinations based on its keyword arguments.

  Two sets of returned combinations can be concatenated using +.  Their product
  can be computed using `times()`.

  Args:
    **kwargs: keyword arguments of form `option=[possibilities, ...]`.

  Returns:
    a list of dictionaries for each combination. Keys in the dictionaries are
    the keyword argument names.  Each key has one value - one of the
    corresponding keyword argument values.
  """
  if not kwargs:
    return [OrderedDict()]

  sort_by_key = lambda k: k[0][0]
  kwargs = OrderedDict(sorted(kwargs.items(), key=sort_by_key))
  first = list(kwargs.items())[0]

  rest = dict(list(kwargs.items())[1:])
  rest_combined = combine(**rest)

  key = first[0]
  values = first[1]

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

  def __init__(self, name, distribution, required_gpus=None,
               required_tpu=False):
    self._distribution = distribution
    self._name = name
    self._required_gpus = required_gpus
    self._required_tpu = required_tpu

  def __repr__(self):
    return self._name

  @property
  def strategy(self):
    return self._distribution

  @property
  def required_gpus(self):
    return self._required_gpus

  @property
  def required_tpu(self):
    return self._required_tpu


default_strategy = NamedDistribution(
    "Default",
    distribute_lib._default_distribution_strategy,  # pylint: disable=protected-access
    required_gpus=None)
one_device_strategy = NamedDistribution(
    "OneDeviceCPU", one_device_strategy.OneDeviceStrategy("/cpu:0"),
    required_gpus=None)
tpu_strategy_single_iteration = NamedDistribution(
    "TPUSingleIteration",
    tpu_strategy.TPUStrategy(iterations_per_step=1),
    required_tpu=True)
tpu_strategy = NamedDistribution(
    "TPU", tpu_strategy.TPUStrategy(), required_tpu=True)
# Note that we disable prefetching for testing since prefetching makes
# the input non-deterministic.
mirrored_strategy_with_gpu_and_cpu = NamedDistribution(
    "MirroredCPUAndGPU",
    mirrored_strategy.MirroredStrategy(
        ["/gpu:0", "/cpu:0"], prefetch_on_device=False),
    required_gpus=1)
mirrored_strategy_with_two_gpus = NamedDistribution(
    "Mirrored2GPUs",
    mirrored_strategy.MirroredStrategy(
        ["/gpu:0", "/gpu:1"], prefetch_on_device=False),
    required_gpus=2)

adam_optimizer_v1_fn = NamedObject(
    "AdamV1", lambda: adam.AdamOptimizer(0.2, epsilon=1))
gradient_descent_optimizer_v1_fn = NamedObject(
    "GradientDescentV1", lambda: gradient_descent.GradientDescentOptimizer(0.2))

adam_optimizer_v2_fn = NamedObject(
    "AdamV2", lambda: adam_v2.AdamOptimizer(0.2, epsilon=1))
gradient_descent_optimizer_v2_fn = NamedObject(
    "GradientDescentV2",
    lambda: gradient_descent_v2.GradientDescentOptimizer(0.2))

graph_and_eager_modes = ["graph", "eager"]


def distributions_and_v1_optimizers():
  """A common set of combination with DistributionStrategies and Optimizers."""
  return combine(
      distribution=[
          one_device_strategy, mirrored_strategy_with_gpu_and_cpu,
          mirrored_strategy_with_two_gpus
      ],
      optimizer_fn=[adam_optimizer_v1_fn, gradient_descent_optimizer_v1_fn])


def distributions_and_v2_optimizers():
  """DistributionStrategies and V2 Optimizers."""
  return combine(
      distribution=[
          one_device_strategy, mirrored_strategy_with_gpu_and_cpu,
          mirrored_strategy_with_two_gpus
      ],
      optimizer_fn=[adam_optimizer_v2_fn, gradient_descent_optimizer_v2_fn])
