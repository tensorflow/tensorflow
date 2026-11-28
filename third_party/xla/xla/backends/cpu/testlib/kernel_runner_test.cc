/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/cpu/testlib/kernel_runner.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/jit_compiler.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/testlib/llvm_ir_kernel_emitter.h"
#include "xla/backends/cpu/testlib/mlir_kernel_emitter.h"
#include "xla/codegen/device_spec.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/codegen/testlib/kernel_runner.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/work_group.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

using ::testing::Eq;

TEST(KernelRunnerTest, Add) {
  static constexpr absl::string_view kLlvmAddI32 = R"(
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
  )";

  constexpr int64_t kNumElements = 8;
  constexpr size_t kArgSizeBytes = kNumElements * sizeof(int32_t);
  LlvmTestKernelEmitter::KernelArg read_arg{kArgSizeBytes,
                                            BufferUse::MemoryAccess ::kRead};
  LlvmTestKernelEmitter::KernelArg write_arg{kArgSizeBytes,
                                             BufferUse::MemoryAccess::kWrite};
  LlvmTestKernelEmitter emitter(kLlvmAddI32, "LlvmAddI32",
                                NumWorkGroups{kNumElements},
                                {read_arg, read_arg, write_arg});

  TF_ASSERT_OK_AND_ASSIGN(KernelDefinition<LlvmKernelSource> kernel_definition,
                          emitter.EmitKernelDefinition());
  TF_ASSERT_OK_AND_ASSIGN(JitCompiler compiler,
                          KernelRunner::CreateJitCompiler(HloModuleConfig()));

  TF_ASSERT_OK_AND_ASSIGN(
      KernelRunner runner,
      KernelRunner::Create(std::move(kernel_definition), std::move(compiler)));

  std::minstd_rand0 engine;
  Shape shape = ShapeUtil::MakeShape(S32, {kNumElements});
  TF_ASSERT_OK_AND_ASSIGN(
      Literal in_arg1,
      LiteralUtil::CreateRandomLiteral<S32>(shape, &engine, 10, 10));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal in_arg2,
      LiteralUtil::CreateRandomLiteral<S32>(shape, &engine, 15, 100));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal out_arg,
      LiteralUtil::CreateRandomLiteral<S32>(shape, &engine, 0, 1));

  absl::Status status =
      runner.Call({KernelRunnerUtil::CreateArgument(in_arg1),
                   KernelRunnerUtil::CreateArgument(in_arg2),
                   KernelRunnerUtil::CreateArgument(out_arg)});
  EXPECT_TRUE(status.ok());

  std::vector<int32_t> expected_result;
  expected_result.reserve(kNumElements);
  for (int64_t idx = 0; idx < kNumElements; ++idx) {
    expected_result.push_back(in_arg1.data<int32_t>()[idx] +
                              in_arg2.data<int32_t>()[idx]);
  }

  ASSERT_THAT(out_arg.data<int32_t>(), Eq(expected_result));
}

TEST(KernelRunnerTest, MlirAtomic128C128StoreAndAdd) {
  static constexpr absl::string_view kMlirAtomicC128 = R"(
    module {
      func.func @mlir_atomic_c128(
          %buf: tensor<2xcomplex<f64>> {llvm.align = 16 : index, xla.slice_index = 0})
          -> tensor<2xcomplex<f64>>
          attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %store_val = complex.constant [9.0 : f64, -4.0 : f64] : complex<f64>
        %add_val = complex.constant [3.5 : f64, 7.25 : f64] : complex<f64>
        %s = xla.atomic_store %store_val into %buf[%c0] : tensor<2xcomplex<f64>>
        %r = xla.atomic_rmw %s[%c1] : tensor<2xcomplex<f64>> {
          ^bb0(%cur: complex<f64>):
            %sum = complex.add %cur, %add_val : complex<f64>
            xla.yield %sum : complex<f64>
        }
        return %r : tensor<2xcomplex<f64>>
      }
    }
  )";

  ASSERT_OK_AND_ASSIGN(JitCompiler compiler,
                       KernelRunner::CreateJitCompiler(HloModuleConfig()));
  CpuDeviceSpec host_spec =
      CpuDeviceSpecFromTargetMachine(*compiler.target_machine());
  if (!host_spec.has_atomic_cmpxchg_128) {
    GTEST_SKIP() << "Host CPU does not support native 128 bit atomic CAS.";
  }

  MlirTestKernelEmitter::KernelArg rw_arg{2 * sizeof(complex128),
                                          BufferUse::MemoryAccess::kWrite};
  MlirTestKernelEmitter emitter(kMlirAtomicC128, "mlir_atomic_c128",
                                NumWorkGroups{1}, {rw_arg});
  ASSERT_OK_AND_ASSIGN(KernelDefinition<MlirKernelSource> kernel_definition,
                       emitter.EmitKernelDefinition());
  ASSERT_OK_AND_ASSIGN(
      KernelRunner runner,
      KernelRunner::Create(std::move(kernel_definition), std::move(compiler)));

  Literal buf = LiteralUtil::CreateR1<complex128>(
      {complex128(1.0, 2.0), complex128(10.0, -2.25)});
  ASSERT_OK(runner.Call({KernelRunnerUtil::CreateArgument(buf)}));
  EXPECT_EQ(buf.Get<complex128>({0}), complex128(9.0, -4.0));
  EXPECT_EQ(buf.Get<complex128>({1}), complex128(13.5, 5.0));
}

}  // namespace xla::cpu
