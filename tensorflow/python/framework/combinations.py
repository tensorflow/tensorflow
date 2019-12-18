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

from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_combinations


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


class TFVersionCombination(test_combinations.TestCombination):
  """Control the execution of the test in TF1.x and TF2.

  If TF2 is enabled then a test with TF1 test is going to be skipped and vice
  versa.

  Test targets continuously run in TF2 thanks to the tensorflow.v2 TAP target.
  A test can be run in TF2 with bazel by passing --test_env=TF2_BEHAVIOR=1.
  """

  def should_execute_combination(self, kwargs):
    tf_api_version = kwargs.pop("tf_api_version", None)
    if tf_api_version == 1 and tf2.enabled():
      return (False, "Skipping a TF1.x test when TF2 is enabled.")
    elif tf_api_version == 2 and not tf2.enabled():
      return (False, "Skipping a TF2 test when TF2 is not enabled.")
    return (True, None)

  def parameter_modifiers(self):
    return [test_combinations.OptionalParameter("tf_api_version")]


generate = functools.partial(
    test_combinations.generate,
    test_combinations=(EagerGraphCombination(), TFVersionCombination()))
combine = test_combinations.combine
times = test_combinations.times
NamedObject = test_combinations.NamedObject
