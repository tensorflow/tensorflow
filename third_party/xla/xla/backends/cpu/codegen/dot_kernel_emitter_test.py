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
import numpy as np

from xla.backends.cpu import testlib as testlib_cpu
from xla.backends.cpu.testlib import utilities
from xla.codegen import testlib as testlib_base
from xla.codegen.testlib import utilities as base_utilities

create_literal = base_utilities.create_literal_from_np
HloInstruction = testlib_base.HloInstruction
HloOpcode = testlib_base.HloOpcode


class DotKernelRunnerTest(absltest.TestCase):

  def test_dot_kernel_emitter(self):
    lhs_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    rhs_np = np.array([[5, 6], [7, 8]], dtype=np.float32)

    lhs_literal = create_literal(lhs_np)
    rhs_literal = create_literal(rhs_np)

    output_literal = create_literal(np.ndarray((2, 2), dtype=np.float32))

    lhs_param = HloInstruction.create_parameter(0, lhs_literal.shape(), "lhs")
    rhs_param = HloInstruction.create_parameter(1, rhs_literal.shape(), "rhs")

    dot_dimension_numbers = testlib_base.DotDimensionNumbers([1], [0], [], [])
    hlo_op = HloInstruction.create_dot(
        output_literal.shape(), lhs_param, rhs_param, dot_dimension_numbers
    )

    hlo_module, buffer_assignment = utilities.build_hlo_module(
        hlo_op, lhs_param, rhs_param
    )
    jit_compiler = testlib_cpu.JitCompiler()

    emitter = testlib_cpu.DotKernelEmitter(
        hlo_module.get_root_instruction(),
        buffer_assignment,
        jit_compiler.get_target_machine(),
    )

    runner = testlib_cpu.KernelRunner.create(
        emitter.emit_kernel_definition(), jit_compiler
    )

    runner.call([lhs_literal, rhs_literal, output_literal])
    np.testing.assert_equal(np.asarray(output_literal), lhs_np @ rhs_np)


if __name__ == "__main__":
  absltest.main()
