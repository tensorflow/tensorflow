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

from xla.backends.cpu import testlib as cpu_testlib
from xla.codegen import testlib as base_testlib
from xla.codegen.testlib import utilities as testlib_utilities

create_literal = testlib_utilities.create_literal_from_np


class LLvmKernelRunnerTest(absltest.TestCase):

  def test_llvm_ir_kernel_runner(self):
    ir = """
        %struct.XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
        %struct.XLA_CPU_KernelArg = type { ptr, i64 }
        ; c = a + b (per thread)
        define ptr @LlvmAddI32(ptr noundef %call_frame_ptr) {
          %args_gep = getelementptr inbounds %struct.XLA_CPU_KernelCallFrame,
                          ptr %call_frame_ptr, i32 0, i32 3
          %args_ptr = load ptr, ptr %args_gep, align 8
          %arg1_gep = getelementptr inbounds %struct.XLA_CPU_KernelArg, ptr %args_ptr, i64 1
          %arg2_gep = getelementptr inbounds %struct.XLA_CPU_KernelArg, ptr %args_ptr, i64 2
          %arg0_ptr = load ptr, ptr %args_ptr, align 8
          %arg1_ptr = load ptr, ptr %arg1_gep, align 8
          %arg2_ptr = load ptr, ptr %arg2_gep, align 8
          %thread_gep = getelementptr inbounds %struct.XLA_CPU_KernelCallFrame, ptr %call_frame_ptr, i32 0, i32 1
          %thread_ptr = load ptr, ptr %thread_gep, align 8
          %thread_idx = load i64, ptr %thread_ptr, align 8
          %a_ptr = getelementptr inbounds i32, ptr %arg0_ptr, i64 %thread_idx
          %a = load i32, ptr %a_ptr, align 4
          %b_ptr = getelementptr inbounds i32, ptr %arg1_ptr, i64 %thread_idx
          %b = load i32, ptr %b_ptr, align 4
          %c = add nsw i32 %a, %b
          %result_ptr = getelementptr inbounds i32, ptr %arg2_ptr, i64 %thread_idx
          store i32 %c, ptr %result_ptr, align 4
          ret ptr null
        }
    """
    llvm_emitter = cpu_testlib.LlvmIrKernelEmitter(ir, "LlvmAddI32", (4, 1, 1))

    kernel_definition = llvm_emitter.emit_kernel_definition()

    runner = cpu_testlib.KernelRunner.create(
        kernel_definition,
        cpu_testlib.JitCompiler(base_testlib.HloModuleConfig()),
    )
    a = create_literal(np.array([1, 2, 3, 4], dtype=np.int32))
    b = create_literal(np.array([5, 6, 7, 8], dtype=np.int32))
    c = create_literal(np.array([0, 0, 0, 0], dtype=np.int32))
    runner.call([a, b, c])

    np.testing.assert_array_equal(np.asarray(c), np.asarray(a) + np.asarray(b))


class MlirKernelRunnerTest(absltest.TestCase):

  def test_mlir_kernel_runner(self):
    ir = """
      #indexing_map = #xla.indexing_map<"()[s0, s1] -> (s0, s1), domain: s0 in [0, 1023], s1 in [0, 31]">
      module attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>} {
        func.func
        @sum(%input_buffer: tensor<1024x32xf32>,
                          %output_buffer: tensor<1xf32>) -> tensor<1xf32>
                            attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
          // Initial sum set to 0.
          %sum_0 = arith.constant 0.0 : f32
          // iter_args binds initial values to the loop's region arguments.
          %sum = xla.loop ()[%i, %j] -> (%r0, %r1)
              in #indexing_map iter_args(%sum_iter = %sum_0) -> (f32) {
            %t = tensor.extract %input_buffer[%i, %j] : tensor<1024x32xf32>
            %sum_next = arith.addf %sum_iter, %t : f32
            // Yield current iteration sum to next iteration %sum_iter or to %sum
            // if final iteration.
            xla.yield %sum_next : f32
          }

          // Ideally it would be be possible to do this in the kernel region,
          // but currently our lowering results in xla_cpu.store being removed
          // before the tensors are lowered which then results in the insert
          // being removed as tensors have value scemantics and are treated as
          // pure.
          %zero_index = arith.constant 0 : index
          %inserted = tensor.insert %sum into %output_buffer[%zero_index] : tensor<1xf32>

          return %inserted : tensor<1xf32>
        }
      }
    """
    mlir_emitter = cpu_testlib.MlirKernelEmitter(ir, "sum_kernel", (1, 1, 1))

    kernel_definition = mlir_emitter.emit_kernel_definition()

    runner = cpu_testlib.KernelRunner.create(
        kernel_definition,
        cpu_testlib.JitCompiler(base_testlib.HloModuleConfig()),
    )
    input_tensor = create_literal(np.ones([1024, 32], dtype=np.float32))
    output_sum = create_literal(np.zeros([1], dtype=np.float32))
    runner.call([input_tensor, output_sum])

    np.testing.assert_array_equal(
        np.asarray(output_sum).item(), np.asarray(input_tensor).sum()
    )


if __name__ == "__main__":
  absltest.main()
