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

from tensorflow.python.framework import test_combinations
from tensorflow.python.eager import context
from tensorflow.python.framework import ops


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


generate = functools.partial(
    test_combinations.generate,
    test_combinations=(EagerGraphCombination(),))
combine = test_combinations.combine
times = test_combinations.times
NamedObject = test_combinations.NamedObject
