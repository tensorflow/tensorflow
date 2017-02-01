# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for contrib.compiler.jit."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys


# TODO(keveman): #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes  # pylint: disable=g-import-not-at-top
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)


# pylint: disable=g-import-not-at-top
from tensorflow.contrib.compiler import jit
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
# pylint: enable=g-import-not-at-top


_REGISTERED_OPS = op_def_registry.get_registered_ops()


def enable_jit_nonstateful(node_def):
  try:
    return not _REGISTERED_OPS[node_def.op].is_stateful
  except KeyError:
    raise ValueError("Unregistered op being created: %s" % node_def)


class JITTest(test.TestCase):

  def compute(self, use_jit, compute_fn):
    random_seed.set_random_seed(1234)
    with self.test_session(graph=ops.Graph()) as sess:
      with jit.experimental_jit_scope(use_jit):
        r = compute_fn()
      sess.run(variables.global_variables_initializer())
      return (r, sess.run(r))

  def testJITCreateOpsLambda(self):
    """Test several ways of customizing the compilation attribute."""
    def create_ops():
      with variable_scope.variable_scope(
          "root",
          initializer=init_ops.random_uniform_initializer(
              -0.1, 0.1, seed=2)):
        inputs = random_ops.random_uniform((1,), seed=1)
        return inputs
    v_false_1_t, v_false_1 = self.compute(False, create_ops)
    _, v_false_2 = self.compute(False, create_ops)
    v_true_1_t, v_true_1 = self.compute(enable_jit_nonstateful, create_ops)
    _, v_true_2 = self.compute(enable_jit_nonstateful, create_ops)
    v_all_true_t, _ = self.compute(True, create_ops)
    self.assertEqual(False, v_false_1_t.op.get_attr("_XlaCompile"))
    v_true_1_t_sampler_op = v_true_1_t.graph.get_operation_by_name(
        "root/random_uniform/RandomUniform")
    v_all_true_t_sampler_op = v_all_true_t.graph.get_operation_by_name(
        "root/random_uniform/RandomUniform")

    self.assertEqual(False, v_true_1_t_sampler_op.get_attr("_XlaCompile"))
    self.assertEqual(True, v_all_true_t_sampler_op.get_attr("_XlaCompile"))

    self.assertEqual(True, v_true_1_t.op.get_attr("_XlaCompile"))
    self.assertEqual(True, v_all_true_t.op.get_attr("_XlaCompile"))

    # Additionally ensure that where no JIT compilation happens on the
    # random_uniform op, the output values are identical to the case
    # where no JIT compilation happens anywhere.
    self.assertAllClose(v_false_1, v_false_2)
    self.assertAllClose(v_true_1, v_true_2)
    self.assertAllClose(v_false_1, v_true_1)

  def testJITVariableSeed(self):
    """Test that the stateful initializer is not marked for compilation.

    XLA does not currently support seeded initialization and XLA initializers
    therefore return different values than non-XLA counterparts.  Here
    we ensure that if we can disable JIT compilation for the initializers and
    get the same variable values as if no JIT compilation happened.
    """
    def create_ops():
      with variable_scope.variable_scope(
          "root",
          initializer=init_ops.random_uniform_initializer(
              -0.1, 0.1, seed=2)):
        inputs = variable_scope.get_variable("var", (1,))
        return inputs
    _, v_false_1 = self.compute(False, create_ops)
    _, v_false_2 = self.compute(False, create_ops)
    _, v_true_1 = self.compute(enable_jit_nonstateful, create_ops)
    _, v_true_2 = self.compute(enable_jit_nonstateful, create_ops)
    self.assertAllClose(v_false_1, v_false_2)
    self.assertAllClose(v_true_1, v_true_2)
    self.assertAllClose(v_false_1, v_true_1)


if __name__ == "__main__":
  test.main()
