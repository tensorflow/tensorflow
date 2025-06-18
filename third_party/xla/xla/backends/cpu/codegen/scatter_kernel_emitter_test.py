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


def create_sclar_add_computation(
    dtype: np.dtype,
) -> testlib_base.HloComputation:
  scalar_shape = xla_extension.Shape.scalar_shape(dtype)
  param_0 = testlib_base.HloInstruction.create_parameter(
      0,
      scalar_shape,
      "update_param_0",
  )
  param_1 = testlib_base.HloInstruction.create_parameter(
      1,
      scalar_shape,
      "update_param_1",
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


def create_scatter_runner(
    operand: xla_extension.Literal,
    scatter_indicies: xla_extension.Literal,
    updates: xla_extension.Literal,
    scatter_dimension_numbers: testlib_base.ScatterDimensionNumbers,
) -> testlib_cpu.KernelRunner:
  dtype = np.dtype(np.float32)

  operand_parameter = testlib_base.HloInstruction.create_parameter(
      0, operand.shape(), "operand"
  )
  scatter_indicies_parameter = testlib_base.HloInstruction.create_parameter(
      1, scatter_indicies.shape(), "scatter_indicies"
  )
  updates_parameter = testlib_base.HloInstruction.create_parameter(
      2, updates.shape(), "updates"
  )

  update_computation = create_sclar_add_computation(dtype)

  scatter_instruction = testlib_base.HloInstruction.create_scatter(
      operand.shape(),
      operand_parameter,
      scatter_indicies_parameter,
      updates_parameter,
      update_computation,
      scatter_dimension_numbers,
      False,
      False,
  )

  hlo_module, _ = utilities.build_hlo_module(
      scatter_instruction,
      operand_parameter,
      scatter_indicies_parameter,
      updates_parameter,
      extra_computations=[update_computation],
  )

  hlo_module = testlib_cpu.run_fusion_wrapper_pass(hlo_module)
  hlo_module, buffer_assignment = utilities.annotate_hlo_module(hlo_module)

  scatter_emitter = testlib_cpu.ScatterKernelEmitter(
      hlo_module.get_root_instruction(), buffer_assignment
  )
  kernel_definition = scatter_emitter.emit_kernel_definition()

  jit_compiler = testlib_cpu.JitCompiler(hlo_module.get_config())

  return testlib_cpu.KernelRunner.create(kernel_definition, jit_compiler)


class ScatterKernelTest(parameterized.TestCase):

  def test_basic_scatter(self):
    dtype = np.dtype(np.float32)

    scatter_dimension_numbers = testlib_base.ScatterDimensionNumbers(
        update_window_dims=[1, 2],
        inserted_window_dims=[],
        scatter_dims_to_operand_dims=[0, 1],
        index_vector_dim=1,
    )

    operand_shape = [3, 4]
    operand = base_utilities.create_literal_from_np(
        np.zeros(operand_shape, dtype)
    )

    # Repeat the last dimension to test the update computation.
    scatter_indicies = base_utilities.create_literal_from_np(
        np.array([[0, 0], [2, 0], [1, 0], [1, 1], [2, 0]], dtype=np.int32)
    )

    updates = base_utilities.create_literal_from_np(
        np.arange(15, dtype=dtype).reshape([5, 1, 3])
    )

    runner = create_scatter_runner(
        operand, scatter_indicies, updates, scatter_dimension_numbers
    )

    runner.call([operand, scatter_indicies, updates])

    operand_np = np.asarray(operand)
    updates_np = np.asarray(updates).reshape([5, 3])

    np.testing.assert_array_equal(
        operand_np[0, :], np.concatenate((updates_np[0, :], [0]))
    )
    np.testing.assert_array_equal(
        operand_np[1, :],
        np.concatenate((updates_np[2, :], [0]))
        + (np.concatenate(([0], updates_np[3, :]))),
    )
    np.testing.assert_array_equal(
        operand_np[2, :],
        np.concatenate((updates_np[1, :] + updates_np[4, :], [0])),
    )

  def test_out_of_bounds_scatter(self):
    dtype = np.dtype(np.float32)

    scatter_dimension_numbers = testlib_base.ScatterDimensionNumbers(
        update_window_dims=[1, 2],
        inserted_window_dims=[],
        scatter_dims_to_operand_dims=[0, 1],
        index_vector_dim=1,
    )

    operand_shape = [3, 4]
    operand = base_utilities.create_literal_from_np(
        np.zeros(operand_shape, dtype)
    )

    # Offset of 2 combined with window size of 3 results in oob.
    scatter_indicies = base_utilities.create_literal_from_np(
        np.array([[0, 2]], dtype=np.int32)
    )

    updates = base_utilities.create_literal_from_np(
        np.arange(3, dtype=dtype).reshape([1, 1, 3])
    )

    runner = create_scatter_runner(
        operand, scatter_indicies, updates, scatter_dimension_numbers
    )

    runner.call([operand, scatter_indicies, updates])

    np.testing.assert_array_equal(
        np.asarray(operand), np.zeros(operand_shape, dtype)
    )


if __name__ == "__main__":
  absltest.main()
