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

from collections.abc import Callable, Iterable

from absl.testing import absltest
import numpy as np

from xla.backends.cpu import testlib as cpu_testlib
from xla.codegen import testlib as base_testlib
from xla.codegen.testlib import utilities as testlib_utilities

create_literal = testlib_utilities.create_literal_from_np


def compare_kernel(
    ir: str,
    kernel_name: str,
    num_workgroups: int,
    input_shapes: Iterable[tuple[int, ...]],
    output_shape: tuple[int, ...],
    dtype,
    expected_output: Callable[[np.ndarray, ...], np.ndarray],
    exact: bool = True,
) -> None:
  mlir_emitter = cpu_testlib.MlirTestKernelEmitter(
      ir, kernel_name, (num_workgroups, 1, 1)
  )
  kernel_definition = mlir_emitter.emit_kernel_definition()

  runner = cpu_testlib.KernelRunner.create(
      kernel_definition,
      cpu_testlib.JitCompiler(base_testlib.HloModuleConfig()),
  )
  inputs = [np.random.rand(*shape).astype(dtype) for shape in input_shapes]

  input_tensors = [create_literal(input) for input in inputs]
  output_tensor = create_literal(np.zeros(output_shape, dtype=dtype))
  runner.call(input_tensors + [output_tensor])

  if exact:
    np.testing.assert_array_equal(
        np.asarray(output_tensor), expected_output(*inputs)
    )
  else:
    np.testing.assert_array_almost_equal(
        np.asarray(output_tensor), expected_output(*inputs)
    )


class XtileLoweringTest(absltest.TestCase):

  def test_slice(self):
    # Check that masked extract / insert works.
    ir = """
      module @tiled_slice {
        xtile.entry_func @tiled_slice(
            %input: memref<5x5xf32>,
            %output: memref<5x5xf32>,
            %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:1, tiles_per_workgroup:1>} {
          %offset = arith.constant 0 : index
          %input_tile = xtile.extract %input[%offset, %offset][64, 64][1, 1] : memref<5x5xf32> -> tensor<64x64xf32>
          %transposed_tile = stablehlo.transpose %input_tile, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
          xtile.insert %transposed_tile into %output[%offset, %offset][64, 64][1, 1] : tensor<64x64xf32> -> memref<5x5xf32>
          xtile.return
        }
      }
    """

    compare_kernel(
        ir,
        "tiled_slice",
        1,
        [(5, 5)],
        (5, 5),
        np.float32,
        lambda arg: arg.transpose(),
    )

  def test_transpose(self):
    ir = """
      module @tiled_transpose {
        xtile.entry_func @tiled_transpose(
            %input: memref<4096x4096xf32>,
            %output: memref<4096x4096xf32>,
            %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:262144, tiles_per_workgroup:32768>} {
          %offset_0 = xla.apply_indexing #xla.indexing_map<"(tid) -> ((tid mod 512) * 8), domain: tid in [0, 262144]">(%tile_id)
          %offset_1 = xla.apply_indexing #xla.indexing_map<"(tid) -> ((tid floordiv 512) * 8), domain: tid in [0, 262144]">(%tile_id)
          %input_tile = xtile.extract %input[%offset_0, %offset_1][8, 8][1, 1] : memref<4096x4096xf32> -> tensor<8x8xf32>
          %transposed_tile = stablehlo.transpose %input_tile, dims = [1, 0] : (tensor<8x8xf32>) -> tensor<8x8xf32>
          xtile.insert %transposed_tile into %output[%offset_1, %offset_0][8, 8][1, 1] : tensor<8x8xf32> -> memref<4096x4096xf32>
          xtile.return
        }
      }
    """

    compare_kernel(
        ir,
        "tiled_transpose",
        8,
        [(4096, 4096)],
        (4096, 4096),
        np.float32,
        lambda arg: arg.transpose(),
    )

  def test_add_tranpose(self):
    ir = """
      module @add_tranpose {
        xtile.entry_func @add_tranpose(
            %input: memref<4096x4096xf32>,
            %output: memref<4096x4096xf32>,
            %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:262144, tiles_per_workgroup:32768>} {
          %offset_0 = xla.apply_indexing #xla.indexing_map<"(tid) -> ((tid mod 512) * 8), domain: tid in [0, 262144]">(%tile_id)
          %offset_1 = xla.apply_indexing #xla.indexing_map<"(tid) -> ((tid floordiv 512) * 8), domain: tid in [0, 262144]">(%tile_id)
          %input_tile_0 = xtile.extract %input[%offset_0, %offset_1][8, 8][1, 1] : memref<4096x4096xf32> -> tensor<8x8xf32>
          %input_tile_1 = xtile.extract %input[%offset_1, %offset_0][8, 8][1, 1] : memref<4096x4096xf32> -> tensor<8x8xf32>
          %transposed_tile = stablehlo.transpose %input_tile_0, dims = [1, 0] : (tensor<8x8xf32>) -> tensor<8x8xf32>
          %added_tile = arith.addf %input_tile_1, %transposed_tile : tensor<8x8xf32>
          xtile.insert %added_tile into %output[%offset_1, %offset_0][8, 8][1, 1] : tensor<8x8xf32> -> memref<4096x4096xf32>
          xtile.return
        }
      }
    """

    compare_kernel(
        ir,
        "add_tranpose",
        8,
        [(4096, 4096)],
        (4096, 4096),
        np.float32,
        lambda arg: arg + arg.transpose(),
    )

  def test_dot_single_tile(self):
    ir = """
      module @dot_single_tile {
        xtile.entry_func @dot_single_tile(
            %lhs: memref<8x16xf32>,
            %rhs: memref<16x8xf32>,
            %output: memref<8x8xf32>,
            %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:1, tiles_per_workgroup:1>} {
          %offset = arith.constant 0 : index
          %lhs_tile = xtile.extract %lhs[%offset, %offset][8, 16][1, 1] : memref<8x16xf32> -> tensor<8x16xf32>
          %rhs_tile = xtile.extract %rhs[%offset, %offset][16, 8][1, 1] : memref<16x8xf32> -> tensor<16x8xf32>
          %result = stablehlo.dot_general %lhs_tile, %rhs_tile, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
          xtile.insert %result into %output[%offset, %offset][8, 8][1, 1] : tensor<8x8xf32> -> memref<8x8xf32>
          xtile.return
        }
      }
    """

    compare_kernel(
        ir,
        "dot_single_tile",
        1,
        [(8, 16), (16, 8)],
        (8, 8),
        np.float32,
        lambda lhs, rhs: lhs @ rhs,
        False,
    )

  def test_dot_fusion_single_tile(self):
    ir = """
      module @dot_fusion_single_tile {
        xtile.entry_func @dot_fusion_single_tile(
            %lhs_0: memref<8x16xf32>,
            %lhs_1: memref<8x16xf32>,
            %rhs: memref<16x1xf32>,
            %output: memref<8x1xf32>,
            %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:1, tiles_per_workgroup:1>} {
          %offset = arith.constant 0 : index
          %lhs_0_tile = xtile.extract %lhs_0[%offset, %offset][8, 16][1, 1] : memref<8x16xf32> -> tensor<8x16xf32>
          %lhs_1_tile = xtile.extract %lhs_1[%offset, %offset][8, 16][1, 1] : memref<8x16xf32> -> tensor<8x16xf32>
          %add_lhs = arith.addf %lhs_0_tile, %lhs_1_tile : tensor<8x16xf32>
          %rhs_tile = xtile.extract %rhs[%offset, %offset][16, 1][1, 1] : memref<16x1xf32> -> tensor<16xf32>
          %result = stablehlo.dot_general %add_lhs, %rhs_tile, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16xf32>) -> tensor<8xf32>
          %tanh_result = math.tanh %result : tensor<8xf32>
          xtile.insert %tanh_result into %output[%offset, %offset][8, 1][1, 1] : tensor<8xf32> -> memref<8x1xf32>
          xtile.return
        }
      }
    """

    compare_kernel(
        ir,
        "dot_fusion_single_tile",
        1,
        [(8, 16), (8, 16), (16, 1)],
        (8, 1),
        np.float32,
        lambda lhs_0, lhs_1, rhs: np.tanh((lhs_0 + lhs_1) @ rhs),
        False,
    )


if __name__ == "__main__":
  absltest.main()
