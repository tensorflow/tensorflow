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
"""Test utility."""

import numpy as np

from tensorflow.python.ops import variables
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops
from tensorflow.python.platform import test
from tensorflow.python.util import nest


class PForTestCase(test.TestCase):
  """Base class for test cases."""

  def _run_targets(self, targets1, targets2=None, run_init=True):
    targets1 = nest.flatten(targets1)
    targets2 = ([] if targets2 is None else nest.flatten(targets2))
    assert len(targets1) == len(targets2) or not targets2
    if run_init:
      init = variables.global_variables_initializer()
      self.evaluate(init)
    return self.evaluate(targets1 + targets2)

  # TODO(agarwal): Allow tests to pass down tolerances.
  def run_and_assert_equal(self, targets1, targets2, rtol=1e-4, atol=1e-5):
    outputs = self._run_targets(targets1, targets2)
    outputs = nest.flatten(outputs)  # flatten SparseTensorValues
    n = len(outputs) // 2
    for i in range(n):
      if outputs[i + n].dtype != np.object_:
        self.assertAllClose(outputs[i + n], outputs[i], rtol=rtol, atol=atol)
      else:
        self.assertAllEqual(outputs[i + n], outputs[i])

  def _test_loop_fn(self,
                    loop_fn,
                    iters,
                    parallel_iterations=None,
                    fallback_to_while_loop=False,
                    rtol=1e-4,
                    atol=1e-5):
    t1 = pfor_control_flow_ops.pfor(
        loop_fn,
        iters=iters,
        fallback_to_while_loop=fallback_to_while_loop,
        parallel_iterations=parallel_iterations)
    loop_fn_dtypes = nest.map_structure(lambda x: x.dtype, t1)
    t2 = pfor_control_flow_ops.for_loop(loop_fn, loop_fn_dtypes, iters=iters,
                                        parallel_iterations=parallel_iterations)

    def _check_shape(a, b):
      msg = (
          "Inferred static shapes are different between two loops:"
          f" {a.shape} vs {b.shape}."
      )
      # TODO(b/268146947): should assert bool(a.shape) == bool(b.shape),
      # since both should be either defined or undefined. But it does not work.
      if b.shape:
        self.assertEqual(a.shape.as_list()[0], b.shape.as_list()[0], msg)
        # TODO(b/268146947): self.assertShapeEqual(a, b, msg) does not work.

    nest.map_structure(_check_shape, t1, t2)
    self.run_and_assert_equal(t1, t2, rtol=rtol, atol=atol)
