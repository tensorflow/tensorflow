# Copyright 2024 The OpenXLA Authors.
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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from xla.backends.cpu import testlib as testlib_cpu
from xla.backends.cpu.testlib import utilities
from xla.codegen import testlib as testlib_base
from xla.codegen.testlib import utilities as base_utilities
from xla.python import xla_extension

create_literal = base_utilities.create_literal_from_np
HloInstruction = testlib_base.HloInstruction


def create_trivial_add_computation(
    dtype: np.dtype,
) -> testlib_base.HloComputation:
  scalar_shape = xla_extension.Shape.scalar_shape(dtype)
  param_0 = testlib_base.HloInstruction.create_parameter(
      0,
      scalar_shape,
      "sub_computation_param_0",
  )
  param_1 = testlib_base.HloInstruction.create_parameter(
      1,
      scalar_shape,
      "sub_computation_param_1",
  )
  root = testlib_base.HloInstruction.create_binary(
      scalar_shape,
      testlib_base.HloOpcode.add,
      param_0,
      param_1,
  )
  return testlib_base.build_hlo_computation(
      root,
      param_0,
      param_1,
  )


class CallKernelTest(parameterized.TestCase):

  def test_basic_call(self):
    dtype = np.dtype(np.float32)

    lhs_literal = base_utilities.create_scalar_literal(1.0, dtype)
    lhs_parameter = testlib_base.HloInstruction.create_parameter(
        0, lhs_literal.shape(), "lhs"
    )

    rhs_literal = base_utilities.create_scalar_literal(2.0, dtype)
    rhs_parameter = testlib_base.HloInstruction.create_parameter(
        1, rhs_literal.shape(), "rhs"
    )

    result_literal = base_utilities.create_scalar_literal(0.0, dtype)

    add_computation = create_trivial_add_computation(dtype)

    call_instruction = testlib_base.HloInstruction.create_call(
        result_literal.shape(),
        [lhs_parameter, rhs_parameter],
        add_computation,
    )

    hlo_module, buffer_assignment = utilities.build_hlo_module(
        call_instruction,
        lhs_parameter,
        rhs_parameter,
        extra_computations=[add_computation],
    )

    jit_compiler = testlib_cpu.JitCompiler(hlo_module.get_config())

    call_emitter = testlib_cpu.CallKernelEmitter(
        hlo_module.get_root_instruction(),
        buffer_assignment,
        jit_compiler.get_target_machine(),
    )

    kernel_definition = call_emitter.emit_kernel_definition()
    self.assertIsNotNone(kernel_definition)

    runner = testlib_cpu.KernelRunner.create(kernel_definition, jit_compiler)
    runner.call([lhs_literal, rhs_literal, result_literal])
    self.assertEqual(np.asarray(result_literal).item(), 3.0)


if __name__ == "__main__":
  absltest.main()
