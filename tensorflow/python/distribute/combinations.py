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
"""This module customizes `test_combinations` for `tf.distribute.Strategy`.

Additionally it provides `generate()`, `combine()` and `times()` with
`tf.distribute.Strategy` customizations as a default.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys
import types
import unittest

import six

from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations as framework_combinations
from tensorflow.python.framework import test_combinations as combinations_lib
from tensorflow.python.platform import flags
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect

FLAGS = flags.FLAGS


# TODO(rchao): Rename `distribution` parameter to `strategy` or
# `distribute_strategy` in all tests.
class DistributionParameter(combinations_lib.ParameterModifier):
  """Transforms arguments of type `NamedDistribution`.

  Convert all arguments of type `NamedDistribution` to the value of their
  `strategy` property.
  """

  def modified_arguments(self, kwargs, requested_parameters):
    del requested_parameters
    distribution_arguments = {}
    for k, v in kwargs.items():
      if isinstance(v, NamedDistribution):
        distribution_arguments[k] = v.strategy
    return distribution_arguments


class ClusterParameters(combinations_lib.ParameterModifier):
  """Adds cluster parameters if a `NamedDistribution` has it.

  It needs to be before DistributionParameter.
  """

  def modified_arguments(self, kwargs, requested_parameters):
    strategy = None
    for _, v in kwargs.items():
      if isinstance(v, NamedDistribution):
        if strategy is not None and _num_total_workers(v.has_chief,
                                                       v.num_workers) > 1:
          raise ValueError("Only support one NamedDistribution for multi worker"
                           "tests.")
        strategy = v
    # Always set cluster parameters if they're requested. So that generate()
    # works when there's no startegy in the combinations.
    update = {}
    if "has_chief" in requested_parameters:
      update["has_chief"] = strategy.has_chief if strategy else False
    if "num_workers" in requested_parameters:
      update["num_workers"] = strategy.num_workers if strategy else 1
    return update


class NamedGPUCombination(combinations_lib.TestCombination):
  """Enable tests to request GPU hardware and skip non-GPU combinations.

  This class expects test_combinations to be generated with `NamedDistribution`
  wrapping instances of `tf.distribute.Strategy`.

  Optionally, the `required_gpus` argument is supported.  GPU hardware is
  required, if its value is `True` or > 0.

  Attributes:
    GPU_TEST: The environment is considered to have GPU hardware available if
              the name of the program contains "test_gpu" or "test_xla_gpu".
  """

  GPU_TEST = re.search(r"(test_gpu|test_xla_gpu)$", sys.argv[0])

  def should_execute_combination(self, kwargs):
    distributions = [
        v for v in kwargs.values() if isinstance(v, NamedDistribution)
    ]
    required_gpus = kwargs.get("required_gpus", None)

    if distributions and required_gpus:
      raise ValueError("Do not use `required_gpus` and arguments of type "
                       "NamedDistribution together.")

    number_of_required_gpus = max([required_gpus or 0] +
                                  [d.required_gpus or 0 for d in distributions])

    if not number_of_required_gpus and GPUCombination.GPU_TEST:
      return (False, "Test that doesn't require GPUs.")
    elif (number_of_required_gpus > 0
          and context.num_gpus() < number_of_required_gpus):
      return (False, ("Only {} of {} required GPUs are available.".format(
          context.num_gpus(), number_of_required_gpus)))
    else:
      return (True, None)

  def parameter_modifiers(self):
    return [combinations_lib.OptionalParameter("required_gpus")]


class GPUCombination(NamedGPUCombination):
  """NamedGPUCombination that passes `tf.distribute.Strategy` to the tests."""

  def parameter_modifiers(self):
    return [
        ClusterParameters(),
        DistributionParameter(),
    ] + NamedGPUCombination.parameter_modifiers(self)


class NamedTPUCombination(combinations_lib.TestCombination):
  """Allow to request TPU hardware and skip non-TPU combinations.

  This class expects test_combinations to be generated with `NamedDistribution`
  wrapping instances of `tf.distribute.Strategy`.

  Optionally, the `required_tpus` parameter is supported.  TPU hardware is
  required, if its argument is `True` or > 0.

  Optionally, the `use_cloud_tpu` parameter is supported. If TPU hardware is
  required by `required_tpus`, it specifically must be a Cloud TPU (specified
  with `--tpu`) if `use_cloud_tpu` is `True`.

  Attributes:
    TPU_TEST: The environment is considered to have TPU hardware available if
              the name of the program contains "test_tpu".
  """

  TPU_TEST = "test_tpu" in sys.argv[0]

  def should_execute_combination(self, kwargs):
    distributions = [
        v for v in kwargs.values() if isinstance(v, NamedDistribution)
    ]
    # TODO(isaprykin): Migrate all tests away from using 'required_tpu' in favor
    # of 'required_tpus'.
    if "required_tpus" in kwargs and "required_tpu" in kwargs:
      raise ValueError("Do not use `required_tpu`.  Both `required_tpus` and "
                       "`required_tpu` were specified.")
    required_tpus = kwargs.get("required_tpus", None) or kwargs.get(
        "required_tpu", None)

    if distributions and required_tpus:
      raise ValueError("Do not use `required_tpus` and arguments of type "
                       "NamedDistribution together.")

    # TODO(isaprykin): Add support for a particular number of TPUs.  Right now
    # it's binary.
    number_of_required_tpus = max([required_tpus or 0] +
                                  [d.required_tpu or 0 for d in distributions])
    use_cloud_tpu = any([kwargs.get("use_cloud_tpu")] +
                        [d.use_cloud_tpu for d in distributions])
    tpu = hasattr(FLAGS, "tpu") and FLAGS.tpu or ""

    if not number_of_required_tpus and TPUCombination.TPU_TEST:
      return (False, "Test that doesn't require TPUs.")
    if number_of_required_tpus and not TPUCombination.TPU_TEST:
      return (False, "Test requires a TPU, but it's not available.")
    if use_cloud_tpu and not tpu:
      return (False, "Test requires a Cloud TPU, but none specified.")
    if not use_cloud_tpu and tpu:
      return (False, "Test requires local TPU, but Cloud TPU specified.")
    return (True, None)

  def parameter_modifiers(self):
    return [
        combinations_lib.OptionalParameter("required_tpus"),
        combinations_lib.OptionalParameter("required_tpu"),
        combinations_lib.OptionalParameter("use_cloud_tpu"),
    ]


class TPUCombination(NamedTPUCombination):
  """NamedTPUCombination that passes `tf.distribute.Strategy` to the tests."""

  def parameter_modifiers(self):
    return [
        ClusterParameters(),
        DistributionParameter(),
    ] + NamedTPUCombination.parameter_modifiers(self)


class NamedDistribution(object):
  """Wraps a `tf.distribute.Strategy` and adds a name for test titles."""

  def __init__(self,
               name,
               distribution_fn,
               required_gpus=None,
               required_tpu=False,
               use_cloud_tpu=False,
               has_chief=False,
               num_workers=1):
    """Initialize NamedDistribution.

    Args:
      name: Name that will be a part of the name of the test case.
      distribution_fn: A callable that creates a `tf.distribute.Strategy`.
      required_gpus: The number of GPUs that the strategy requires.
      required_tpu: Whether the strategy requires TPU.
      use_cloud_tpu: Whether the strategy requires cloud TPU.
      has_chief: Whether the strategy requires a chief worker.
      num_workers: The number of workers that the strategy requires.
    """
    object.__init__(self)
    self._name = name
    self._distribution_fn = distribution_fn
    self.required_gpus = required_gpus
    self.required_tpu = required_tpu
    self.use_cloud_tpu = use_cloud_tpu
    self.has_chief = has_chief
    self.num_workers = num_workers

  @property
  def strategy(self):
    return self._distribution_fn()

  def __repr__(self):
    return self._name


def concat(*combined):
  """Concats combinations."""
  result = []
  for one in combined:
    result += one
  return result


def generate(combinations, test_combinations=()):
  # pylint: disable=g-doc-args,g-doc-return-or-yield
  """Distributed adapter of `framework.combinations_lib.generate`.

  All tests with distributed strategy should use this one instead of
  `framework.test_combinations.generate`. This function has support of strategy
  combinations, GPU/TPU and multi worker support.

  See `framework.test_combinations_lib.generate` for usage.
  """
  # pylint: enable=g-doc-args,g-doc-return-or-yield
  default_combinations = (
      framework_combinations.EagerGraphCombination(),
      framework_combinations.TFVersionCombination(),
      GPUCombination(),
      TPUCombination(),
  )
  # We apply our own decoration to handle multi worker tests before applying
  # framework.test_combinations.generate. The order is important since we need
  # framework.test_combinations.generate to apply all parameter modifiers first.
  combination_decorator = combinations_lib.generate(
      combinations, test_combinations=default_combinations + test_combinations)

  def decorator(test_method_or_class):
    if isinstance(test_method_or_class, type):
      # If it's a test class.
      class_object = test_method_or_class
      # Decorate each test method with _multi_worker_test.
      for name, test_method in six.iteritems(class_object.__dict__.copy()):
        if (name.startswith(unittest.TestLoader.testMethodPrefix) and
            isinstance(test_method, types.FunctionType)):
          setattr(class_object, name, _multi_worker_test(test_method))
      return combination_decorator(class_object)
    else:
      return combination_decorator(_multi_worker_test(test_method_or_class))

  return decorator


combine = combinations_lib.combine
times = combinations_lib.times
NamedObject = combinations_lib.NamedObject


def main():
  """Tests must call this main()."""
  return multi_process_runner.test_main()


# Identifies whether we're in the main process or worker processes.
# `_multi_worker_test` decoration behaves differently in the main processs and
# the worker processes. See the documentation of _multi_worker_test for detail.
_running_in_worker = False


def _test_runner(test_id):
  """Executes the test with the given test_id.

  This is a simple wrapper around TestRunner to be used with
  multi_process_runner. Similar to test.main(), but it executes only one test
  specified by test_id and returns whether the test succeeds. If the test fails,
  the function prints failures and errors to stdout.

  Args:
    test_id: TestCase.id()

  Returns:
    A boolean indicates whether the test succeeds.
  """
  global _running_in_worker
  # No need to restore the value of _running_in_worker since it should always be
  # True in worker processes.
  _running_in_worker = True
  test = unittest.defaultTestLoader.loadTestsFromName(test_id)
  runner = unittest.TextTestRunner()
  result = runner.run(test)
  # Print failures and errors to stdout and multi_process_runner will collect
  # them and stream back to the main process.
  for _, msg in result.failures + result.errors:
    print(msg)
  # Return expected failures as failures, so that the main process can get
  # them and fail as expected.
  if result.expectedFailures:
    return False
  return result.wasSuccessful()


def _multi_worker_test(test_method):
  """Decorate test_method so that it runs in each worker.

  We use `multi_process_runner` to simulate multiple workers. Since we run the
  this function in the main process and all worker processes, this decoration
  behaves differently in the main process and worker procssses. In the main
  process, it spawns subprocesses and runs the test on each of them; in a worker
  process, it executes test in the same way as a normal test, e.g.
  setUp()/tearDown() are called before/after the test.

  Args:
    test_method: a function which must be a test method.

  Returns:
    Decorated `test_method`. Note that the decorated function has additional
    arguments.
  """

  def decorator(self, has_chief, num_workers, **kwargs):
    if _num_total_workers(has_chief, num_workers) == 1 or _running_in_worker:
      # We're in worker process or the test is for single worker. Either case we
      # execute the test method directly instead of spawning subprocesses.
      test_method(self, **kwargs)
      return

    # We're in the main process. We spawn subprocesses and run the *test* on
    # each of them. Note that we're not directly executing test_method passed to
    # _multi_worker_test, because we need setUp()/tearDown() to be called and
    # all the decorations on the test method. The conceptual call stack is:
    #   [main process]test.main()
    #     [main process]test_runner.run(test)
    #       [main process]wrapper by combinations.generate()
    #         [main process]_multi_worker_test.decorator()
    #           # A sub process goes through the same code path as the main
    #           # process.
    #           [sub process]_test_runner()
    #             [sub process]test_runner.run(test)
    #               [sub process]wrapper by combinations.generate()
    #                 [sub process]_multi_worker_test.decorator()
    #                   # _running_in_worker is True
    #                   [sub process]test_method()
    test_id = self.id()
    cluster_spec = multi_worker_test_base.create_cluster_spec(
        has_chief=has_chief, num_workers=num_workers, num_ps=0, has_eval=False)
    result = multi_process_runner.run(
        _test_runner, cluster_spec, args=(test_id,))
    for was_successful in result.return_value:
      if not was_successful:
        raise AssertionError("some worker failed, see logs for details")

  argspec = tf_inspect.getfullargspec(test_method)
  decorator_args = (argspec.args or []) + ["has_chief", "num_workers"]
  decorator_argspec = argspec._replace(args=decorator_args)
  return tf_decorator.make_decorator(
      test_method, decorator, decorator_argspec=decorator_argspec)


def _num_total_workers(has_chief, num_workers):
  """Returns the number of workers including the chief."""
  if has_chief:
    return num_workers + 1
  return num_workers
