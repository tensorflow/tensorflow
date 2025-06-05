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
from xla.codegen.testlib import utilities as base_utilities


class LoopFusionTest(parameterized.TestCase):

  def test_basic_add_sub(self):
    dtype = np.dtype(np.float32)

    hlo = """
      HloModule test_module

      fusion_computation {
        %param_0 = f32[100, 200] parameter(0)
        %param_1 = f32[100, 200] parameter(1)
        %param_2 = f32[100, 200] parameter(2)
        %add = f32[100, 200] add(%param_0, %param_1)
        %sub = f32[100, 200] subtract(%add, %param_2)
        ROOT %tuple = (f32[100, 200], f32[100, 200]) tuple(%add, %sub)
      }

      ENTRY main {
        %param_0 = f32[100, 200] parameter(0)
        %param_1 = f32[100, 200] parameter(1)
        %param_2 = f32[100, 200] parameter(2)
        ROOT %wrapped_fusion = (f32[100, 200], f32[100, 200])
                               fusion(%param_0, %param_1, %param_2),
                               kind=kLoop, calls=%fusion_computation
      }
    """

    hlo_module, buffer_assignment = utilities.parse_hlo_module(hlo)
    jit_compiler = testlib_cpu.JitCompiler(hlo_module.get_config())
    mlir_context = testlib_cpu.MLIRContext()
    kernel_definition = testlib_cpu.emit_fusion_kernel(
        mlir_context, hlo_module.get_root_instruction(), buffer_assignment
    )

    kernel_runner = testlib_cpu.KernelRunner.create(
        kernel_definition, jit_compiler
    )
    operand_shape = (100, 200)

    param_0 = base_utilities.create_literal_from_np(
        np.random.rand(*operand_shape).astype(dtype)
    )
    param_1 = base_utilities.create_literal_from_np(
        np.random.rand(*operand_shape).astype(dtype)
    )
    param_2 = base_utilities.create_literal_from_np(
        np.random.rand(*operand_shape).astype(dtype)
    )

    result_0 = base_utilities.create_literal_from_np(
        np.zeros(operand_shape, dtype)
    )
    result_1 = base_utilities.create_literal_from_np(
        np.zeros(operand_shape, dtype)
    )

    kernel_runner.call([param_0, param_1, param_2, result_0, result_1])

    np.testing.assert_array_almost_equal(
        np.asarray(result_0), np.add(param_0, param_1)
    )
    np.testing.assert_array_almost_equal(
        np.asarray(result_1), np.subtract(np.add(param_0, param_1), param_2)
    )


if __name__ == "__main__":
  absltest.main()
