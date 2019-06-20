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
"""This module customizes `test_combinations` for Tensorflow.

Additionally it provides `generate()`, `combine()` and `times()` with Tensorflow
customizations as a default.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import sys

from tensorflow.python.distribute import test_combinations
from tensorflow.python.eager import context
from tensorflow.python.framework import ops


# TODO(rchao): Rename `distribution` parameter to `strategy` or
# `distribute_strategy` in all tests.
class DistributionParameter(test_combinations.ParameterModifier):
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


class NamedGPUCombination(test_combinations.TestCombination):
  """Enable tests to request GPU hardware and skip non-GPU combinations.

  This class expects test_combinations to be genarated with `NamedDistribution`
  wrapping instances of `tf.distribute.Strategy`.

  Optionally, the `required_gpus` argument is supported.  GPU hardware is
  required, if its value is `True` or > 0.

  Attributes:
    GPU_TEST: The environment is considered to have GPU hardware available if
              the name of the program contains "test_gpu".
  """

  GPU_TEST = "test_gpu" in sys.argv[0]

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
    elif context.num_gpus() < number_of_required_gpus:
      return (False, ("Only {} of {} required GPUs are available.".format(
          context.num_gpus(), number_of_required_gpus)))
    else:
      return (True, None)

  def parameter_modifiers(self):
    return [test_combinations.OptionalParameter("required_gpus")]


class GPUCombination(NamedGPUCombination):
  """NamedGPUCombination that passes `tf.distribute.Strategy` to the tests."""

  def parameter_modifiers(self):
    return [DistributionParameter()
           ] + NamedGPUCombination.parameter_modifiers(self)


class NamedTPUCombination(test_combinations.TestCombination):
  """Allow to request TPU hardware and skip non-TPU combinations.

  This class expects test_combinations to be genarated with `NamedDistribution`
  wrapping instances of `tf.distribute.Strategy`.

  Optionally, the `required_tpus` parameter is supported.  TPU hardware is
  required, if its argument is `True` or > 0.

  Attributes:
    TPU_TEST: The environment is considered to have GPU hardware available if
              the name of the program contains "test_gpu".
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

    if not number_of_required_tpus and TPUCombination.TPU_TEST:
      return (False, "Test that doesn't require TPUs.")
    elif number_of_required_tpus and not TPUCombination.TPU_TEST:
      return (False, "Test requires a TPU, but it's not available.")
    else:
      return (True, None)

  def parameter_modifiers(self):
    return [
        test_combinations.OptionalParameter("required_tpus"),
        test_combinations.OptionalParameter("required_tpu")
    ]


class TPUCombination(NamedTPUCombination):
  """NamedTPUCombination that passes `tf.distribute.Strategy` to the tests."""

  def parameter_modifiers(self):
    return [DistributionParameter()
           ] + NamedTPUCombination.parameter_modifiers(self)


class EagerGraphCombination(test_combinations.TestCombination):
  """Run the test in Graph or Eager mode.  Graph is the default.

  The optional `mode` parameter controls the test's execution mode.  Its
  accepted values are "graph" or "eager" literals.
  """

  def context_managers(self, kwargs):
    # TODO(isaprykin): Switch the default to eager.
    mode = kwargs.pop("mode", "graph")
    if mode == "eager":
      return [context.eager_mode()]
    elif mode == "graph":
      return [ops.Graph().as_default(), context.graph_mode()]
    else:
      raise ValueError(
          "'mode' has to be either 'eager' or 'graph' and not {}".format(mode))

  def parameter_modifiers(self):
    return [test_combinations.OptionalParameter("mode")]


class NamedDistribution(object):
  """Wraps a `tf.distribute.Strategy` and adds a name for test titles."""

  def __init__(self, name, distribution_fn, required_gpus=None,
               required_tpu=False):
    object.__init__(self)
    self._name = name
    self._distribution_fn = distribution_fn
    self._required_gpus = required_gpus
    self._required_tpu = required_tpu

  @property
  def strategy(self):
    return self._distribution_fn()

  @property
  def required_gpus(self):
    return self._required_gpus

  @property
  def required_tpu(self):
    return self._required_tpu

  def __repr__(self):
    return self._name


generate = functools.partial(
    test_combinations.generate,
    test_combinations=(EagerGraphCombination(), GPUCombination(),
                       TPUCombination()))
combine = test_combinations.combine
times = test_combinations.times
NamedObject = test_combinations.NamedObject
