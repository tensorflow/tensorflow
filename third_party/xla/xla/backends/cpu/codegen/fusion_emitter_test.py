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


import itertools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from xla.backends.cpu import testlib as testlib_cpu
from xla.backends.cpu.testlib import utilities
from xla.codegen import testlib as base_testlib
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
        mlir_context,
        hlo_module.get_root_instruction(),
        buffer_assignment,
        False,
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

  def test_convert_f32_bf16_f32(self):
    dtype = np.dtype(np.float32)

    hlo = """
      HloModule test_module

      fusion_computation {
        %param = f32[300, 500] parameter(0)
        %convert_bf16 = bf16[300, 500] convert(%param)
        ROOT %convert_f32 = f32[300, 500] convert(%convert_bf16)
      }

      ENTRY main {
        %param = f32[300, 500] parameter(0)
        ROOT %wrapped_fusion = f32[300, 500]
                               fusion(%param),
                               kind=kLoop, calls=%fusion_computation
      }
    """

    hlo_module, buffer_assignment = utilities.parse_hlo_module(hlo)
    jit_compiler = testlib_cpu.JitCompiler(hlo_module.get_config())
    mlir_context = testlib_cpu.MLIRContext()
    kernel_definition = testlib_cpu.emit_fusion_kernel(
        mlir_context,
        hlo_module.get_root_instruction(),
        buffer_assignment,
        False,
    )

    kernel_runner = testlib_cpu.KernelRunner.create(
        kernel_definition, jit_compiler
    )
    operand_shape = (300, 500)

    # np.rand produces values in the range [0, 1).
    # Bias and scale it to get more representative values.
    param = base_utilities.create_literal_from_np(
        (np.random.rand(*operand_shape).astype(dtype) - 0.5) * 1e9
    )
    # Also make sure inf is preserved.
    np.asarray(param)[0, 0] = np.inf

    result = base_utilities.create_literal_from_np(
        np.zeros(operand_shape, dtype)
    )

    kernel_runner.call([param, result])

    # Based on the 7-bit fraction of bf16.
    bf_16_rtol = 2**-8
    np.testing.assert_allclose(result, param, rtol=bf_16_rtol)

  # Check that user provided nans and nans produced by internal ops are
  # preserved.
  def test_convert_f32_bf16_f32_nan(self):
    dtype = np.dtype(np.float32)

    hlo = """
      HloModule test_module

      fusion_computation {
        %param = f32[3] parameter(0)
        %div = f32[3] divide(%param, %param)
        %convert_bf16 = bf16[3] convert(%div)
        ROOT %convert_f32 = f32[3] convert(%convert_bf16)
      }

      ENTRY main {
        %param = f32[3] parameter(0)
        ROOT %wrapped_fusion = f32[3]
                               fusion(%param),
                               kind=kLoop, calls=%fusion_computation
      }
    """

    hlo_module, buffer_assignment = utilities.parse_hlo_module(hlo)
    jit_compiler = testlib_cpu.JitCompiler(hlo_module.get_config())
    mlir_context = testlib_cpu.MLIRContext()
    kernel_definition = testlib_cpu.emit_fusion_kernel(
        mlir_context,
        hlo_module.get_root_instruction(),
        buffer_assignment,
        False,
    )

    kernel_runner = testlib_cpu.KernelRunner.create(
        kernel_definition, jit_compiler
    )
    operand_shape = (3,)

    param = base_utilities.create_literal_from_np(
        np.ndarray(operand_shape, dtype)
    )
    np_param = np.asarray(param)
    np_param[0] = 1
    np_param[1] = 0
    np_param[2] = np.nan

    result = base_utilities.create_literal_from_np(
        np.zeros(operand_shape, dtype)
    )

    kernel_runner.call([param, result])

    np_result = np.asarray(result)

    self.assertEqual(np_result[0], 1)
    self.assertTrue(np.isnan(np_result[1]))
    self.assertTrue(np.isnan(np_result[2]))

  # Check that a constant with a layout is respected.
  def test_constant_with_layout(self):
    dtype = np.dtype(np.float32)

    hlo = """
      HloModule test_module

      fusion_computation {
        %constant = f32[2, 2]{0, 1} constant({{0, 1}, {2, 3}})
        ROOT %result = f32[2, 2]{1, 0} copy(%constant)
      }

      ENTRY main {
        ROOT %wrapped_fusion =
          f32[2, 2] fusion(), kind=kLoop, calls=%fusion_computation
      }
    """

    hlo_module, buffer_assignment = utilities.parse_hlo_module(hlo)
    jit_compiler = testlib_cpu.JitCompiler(hlo_module.get_config())
    mlir_context = testlib_cpu.MLIRContext()
    kernel_definition = testlib_cpu.emit_fusion_kernel(
        mlir_context,
        hlo_module.get_root_instruction(),
        buffer_assignment,
        False,
    )

    kernel_runner = testlib_cpu.KernelRunner.create(
        kernel_definition, jit_compiler
    )
    shape = (2, 2)

    result = base_utilities.create_literal_from_np(np.zeros(shape, dtype))

    kernel_runner.call([result])

    np_result = np.asarray(result)

    self.assertEqual(np_result[0, 0], 0)
    self.assertEqual(np_result[0, 1], 1)
    self.assertEqual(np_result[1, 0], 2)
    self.assertEqual(np_result[1, 1], 3)


class FusionEmitterTest(parameterized.TestCase):

  def test_exp_nan_dce(self):
    dtype = np.dtype(np.float64)

    hlo = """
      HloModule test_module

      fusion_computation {
        %param = f64[3] parameter(0)
        %multiplier = f64[3] constant({1, 0, nan})
        %mult = f64[3] multiply(%param, %multiplier)
        ROOT %exp = f64[3] exponential(%mult) 
      }

      ENTRY main {
        %param = f64[3] parameter(0)
        ROOT %wrapped_fusion = f64[3]
                               fusion(%param),
                               kind=kLoop, calls=%fusion_computation
      }
    """

    hlo_module, buffer_assignment = utilities.parse_hlo_module(hlo)
    jit_compiler = testlib_cpu.JitCompiler(hlo_module.get_config())
    mlir_context = testlib_cpu.MLIRContext()
    kernel_definition = testlib_cpu.emit_fusion_kernel(
        mlir_context,
        hlo_module.get_root_instruction(),
        buffer_assignment,
        False,
    )

    kernel_runner = testlib_cpu.KernelRunner.create(
        kernel_definition, jit_compiler
    )
    operand_shape = (3,)

    param = base_utilities.create_literal_from_np(
        np.ndarray(operand_shape, dtype)
    )
    np_param = np.asarray(param)
    np_param[0] = 1
    np_param[1] = 0
    np_param[2] = np.nan

    result = base_utilities.create_literal_from_np(
        np.zeros(operand_shape, dtype)
    )

    kernel_runner.call([param, result])

    np_result = np.asarray(result)

    self.assertAlmostEqual(np_result[0], np.exp(1))
    self.assertAlmostEqual(np_result[1], np.exp(0))
    self.assertTrue(np.isnan(np_result[2]))


def _make_random_array(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
  if dtype == np.dtype(np.bool_):
    return np.random.rand(*shape) > 0.5
  if np.issubdtype(dtype, np.complexfloating):
    real = np.random.rand(*shape) * 10
    imag = np.random.rand(*shape) * 10
    return (real + 1j * imag).astype(dtype)
  return (np.random.rand(*shape) * 10).astype(dtype)


class ConcatenateFusionTest(parameterized.TestCase):

  @parameterized.product(
      layout_mode=["default", "non_default", "mismatched"],
      dtype=[
          np.dtype(np.bool_),
          np.dtype(np.uint8),
          np.dtype(np.uint16),
          np.dtype(np.uint32),
          np.dtype(np.uint64),
          np.dtype(np.int8),
          np.dtype(np.int16),
          np.dtype(np.int32),
          np.dtype(np.int64),
          np.dtype(np.float16),
          np.dtype(np.float32),
          np.dtype(np.float64),
          np.dtype(np.complex64),
          np.dtype(np.complex128),
      ],
      concat_dimension=[0, 1, 2],
  )
  def test_concatenate(self, layout_mode, dtype, concat_dimension):
    num_inputs = 5
    shape = (4, 4, 4)
    np_inputs = [_make_random_array(shape, dtype) for _ in range(num_inputs)]
    if layout_mode == "mismatched":
      default_layout = [0, 1, 2]
      input_literals = [
          base_utilities.create_literal_from_np(
              input_array, np.roll(default_layout, idx).tolist()
          )
          for idx, input_array in enumerate(np_inputs)
      ]
      output_layout = [2, 1, 0]
    elif layout_mode == "non_default":
      output_layout = [0, 2, 1]
      input_literals = [
          base_utilities.create_literal_from_np(input_array, output_layout)
          for input_array in np_inputs
      ]
    else:
      output_layout = [2, 1, 0]
      input_literals = [
          base_utilities.create_literal_from_np(input_array, output_layout)
          for input_array in np_inputs
      ]

    expected_output = np.concatenate(np_inputs, axis=concat_dimension)
    output_literal = base_utilities.create_literal_from_np(
        np.zeros_like(expected_output), output_layout
    )

    hlo_parameters = [
        base_testlib.HloInstruction.create_parameter(
            idx, literal.shape(), f"input_{idx}"
        )
        for [idx, literal] in enumerate(input_literals)
    ]

    hlo_op = base_testlib.HloInstruction.create_concatenate(
        output_literal.shape(), hlo_parameters, concat_dimension
    )

    hlo_module, _ = utilities.build_hlo_module(hlo_op, *hlo_parameters)
    hlo_module = testlib_cpu.run_fusion_wrapper_pass(hlo_module)
    hlo_module, buffer_assignment = utilities.annotate_hlo_module(hlo_module)
    jit_compiler = testlib_cpu.JitCompiler(hlo_module.get_config())
    mlir_context = testlib_cpu.MLIRContext()

    kernel_definition = testlib_cpu.emit_fusion_kernel(
        mlir_context,
        hlo_module.get_root_instruction(),
        buffer_assignment,
        False,
    )
    runner = testlib_cpu.KernelRunner.create(kernel_definition, jit_compiler)
    runner.call(list(itertools.chain(input_literals, [output_literal])))

    np.testing.assert_array_equal(np.asarray(output_literal), expected_output)

  @parameterized.parameters(2, 4)
  def test_concatenate_uneven_workgroup_split(self, num_partitions):
    hlo = f"""
      HloModule test_module

      fused_computation {{
        %p0 = f32[5,2,3]{{2,1,0}} parameter(0)
        %p1 = f32[5,4,3]{{2,1,0}} parameter(1)
        ROOT %concat = f32[5,6,3]{{2,1,0}} concatenate(%p0, %p1), dimensions={{1}}
      }}

      ENTRY main {{
        %p0 = f32[5,2,3]{{2,1,0}} parameter(0)
        %p1 = f32[5,4,3]{{2,1,0}} parameter(1)
        ROOT %wrapped_concat = f32[5,6,3]{{2,1,0}}
                               fusion(%p0, %p1),
                               kind=kLoop, calls=%fused_computation,
                               backend_config={{"outer_dimension_partitions":["{num_partitions}"]}}
      }}
    """
    hlo_module, buffer_assignment = utilities.parse_hlo_module(hlo)
    jit_compiler = testlib_cpu.JitCompiler(hlo_module.get_config())
    mlir_context = testlib_cpu.MLIRContext()
    kernel_definition = testlib_cpu.emit_fusion_kernel(
        mlir_context,
        hlo_module.get_root_instruction(),
        buffer_assignment,
        False,
    )
    mlir_source = str(kernel_definition.source())
    self.assertIn("xla.copy_slice", mlir_source)
    self.assertIn("xla.workgroup_id x", mlir_source)
    self.assertIn(
        f"xla.range = [0 : index, {num_partitions - 1} : index]", mlir_source
    )

    np_p0 = np.arange(1, 5 * 2 * 3 + 1, dtype=np.float32).reshape(5, 2, 3)
    np_p1 = np.arange(101, 101 + 5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)
    expected_output = np.concatenate([np_p0, np_p1], axis=1)

    p0_literal = base_utilities.create_literal_from_np(np_p0)
    p1_literal = base_utilities.create_literal_from_np(np_p1)
    output_literal = base_utilities.create_literal_from_np(
        np.zeros_like(expected_output)
    )

    runner = testlib_cpu.KernelRunner.create(kernel_definition, jit_compiler)
    runner.call([p0_literal, p1_literal, output_literal])

    np.testing.assert_array_equal(np.asarray(output_literal), expected_output)


if __name__ == "__main__":
  absltest.main()
