/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/rng_get_and_update_state.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/tests/gpu_pjrt_codegen_test.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/sycl/oneapi_compute_capability.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;

class RngGetAndUpdateStateTest : public GpuPjRtCodegenTest {
 protected:
  RngGetAndUpdateStateTest() : mlir_context_(CreateMlirContext()) {
    mlir_context_->appendDialectRegistry(
        MlirKernelEmitter::GetDialectRegistry());
    mlir_context_->loadAllAvailableDialects();
  }

  std::unique_ptr<mlir::MLIRContext> mlir_context_;
};

TEST_F(RngGetAndUpdateStateTest, CreateAndLowerMlirModule) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule RngStateMlir
    ENTRY main {
      ROOT rng = u64[2]{0} rng-get-and-update-state(), delta=131072
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloRngGetAndUpdateStateInstruction* instr =
      Cast<HloRngGetAndUpdateStateInstruction>(
          module->entry_computation()->root_instruction());

  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
      CreateRngGetAndUpdateStateMlirModule(
          *mlir_context_, instr, /*buffer_assignment=*/nullptr, "rng"));

  std::string mlir_str;
  llvm::raw_string_ostream mlir_stream(mlir_str);
  mlir_stream << *mlir_module;

  EXPECT_THAT(RunFileCheck(mlir_str, R"(
    CHECK-NOT: i128
    CHECK: llvm.mlir.global private @rng_state(dense<[117515157, 0]> : vector<2xi64>) {addr_space = 1 : i32, alignment = 16 : i64} : vector<2xi64>
    CHECK: func.func @rng(
    CHECK-SAME: xla.backend_kind = #xla.backend_kind<gpu>
    CHECK:   %[[PTR:.*]] = llvm.mlir.addressof @rng_state : !llvm.ptr<1>
    CHECK:   %[[OLD:.*]] = llvm.load %[[PTR]] <alignment = 16> : !llvm.ptr<1> -> vector<2xi64>
    CHECK:   %[[LO:.*]] = vector.extract %[[OLD]][0] : i64 from vector<2xi64>
    CHECK:   %[[HI:.*]] = vector.extract %[[OLD]][1] : i64 from vector<2xi64>
    CHECK:   %[[NEW_LO:.*]], %[[CARRY_I1:.*]] = arith.addui_extended %[[LO]], %[[DELTA:.*]] : i64, i1
    CHECK:   %[[CARRY:.*]] = arith.extui %[[CARRY_I1]] : i1 to i64
    CHECK:   %[[NEW_HI:.*]] = arith.addi %[[HI]], %[[CARRY]] : i64
    CHECK:   %[[NEW:.*]] = vector.from_elements %[[NEW_LO]], %[[NEW_HI]] : vector<2xi64>
    CHECK:   llvm.store %[[NEW]], %[[PTR]] <alignment = 16> : vector<2xi64>, !llvm.ptr<1>
    CHECK:   vector.transfer_write %[[OLD]], %arg0[%{{.*}}] {in_bounds = [true]} : vector<2xi64>, tensor<2xi64>
  )"),
              IsOkAndHolds(true));

  auto check_lowered_llvm =
      [&](const stream_executor::GpuComputeCapability& cc) {
        ASSERT_OK_AND_ASSIGN(
            mlir::OwningOpRef<mlir::ModuleOp> module_copy,
            CreateRngGetAndUpdateStateMlirModule(
                *mlir_context_, instr, /*buffer_assignment=*/nullptr, "rng"));
        stream_executor::DeviceDescription device;
        device.set_gpu_compute_capability(cc);
        MlirKernelSource mlir_source(nullptr, std::move(module_copy));
        ASSERT_OK_AND_ASSIGN(
            LlvmKernelSource llvm_source,
            CompileMlirToLlvm(device, *module, "rng", /*unroll_factor=*/0,
                              *mlir_context_, std::move(mlir_source)));
        EXPECT_THAT(RunFileCheck(llvm_source.ToString(), R"(
      CHECK-NOT: i128
      CHECK: @rng_state = private addrspace(1) global <2 x i64> <i64 117515157, i64 0>, align 16
      CHECK: define void @rng(ptr noalias %[[OUT:.*]])
      CHECK:   %[[OLD:.*]] = load <2 x i64>, ptr addrspace(1) @rng_state, align 16
      CHECK:   %[[LO:.*]] = extractelement <2 x i64> %[[OLD]], i64 0
      CHECK:   %[[HI:.*]] = extractelement <2 x i64> %[[OLD]], i64 1
      CHECK:   %[[ADD_RES:.*]] = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %[[LO]], i64 131072)
      CHECK:   %[[NEW_LO:.*]] = extractvalue { i64, i1 } %[[ADD_RES]], 0
      CHECK:   %[[CARRY_I1:.*]] = extractvalue { i64, i1 } %[[ADD_RES]], 1
      CHECK:   %[[CARRY:.*]] = zext i1 %[[CARRY_I1]] to i64
      CHECK:   add i64 %[[HI]], %[[CARRY]]
      CHECK:   store <2 x i64> %{{.*}}, ptr addrspace(1) @rng_state, align 16
      CHECK:   store <2 x i64> %[[OLD]], ptr %{{.*}}, align 8
    )"),
                    IsOkAndHolds(true));
      };

  check_lowered_llvm(stream_executor::CudaComputeCapability::Ampere());
  check_lowered_llvm(stream_executor::GpuComputeCapability(
      stream_executor::OneAPIComputeCapability{}));
}

TEST_F(RngGetAndUpdateStateTest, ExecuteReturnsInitialStatePerKernel) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule RngStateAdvance
    ENTRY main {
      first = u64[2]{0} rng-get-and-update-state(), delta=131072
      second = u64[2]{0} rng-get-and-update-state(), delta=65536
      ROOT tuple = (u64[2]{0}, u64[2]{0}) tuple(first, second)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {}));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 2);

  // Each RngGetAndUpdateState op compiles into its own kernel module with a
  // private @rng_state initialized to 0x7012395.
  EXPECT_EQ(outputs[0].Get<uint64_t>({0}), 0x7012395ULL);
  EXPECT_EQ(outputs[0].Get<uint64_t>({1}), 0ULL);
  EXPECT_EQ(outputs[1].Get<uint64_t>({0}), 0x7012395ULL);
  EXPECT_EQ(outputs[1].Get<uint64_t>({1}), 0ULL);
}

TEST_F(RngGetAndUpdateStateTest, ExecuteLoopAdvancesState) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule RngStateLoop
    cond {
      state = (s32[], u64[2]{0}, u64[2]{0}) parameter(0)
      idx = s32[] get-tuple-element(state), index=0
      limit = s32[] constant(2)
      ROOT lt = pred[] compare(idx, limit), direction=LT
    }
    body {
      state = (s32[], u64[2]{0}, u64[2]{0}) parameter(0)
      idx = s32[] get-tuple-element(state), index=0
      prev = u64[2]{0} get-tuple-element(state), index=2
      curr = u64[2]{0} rng-get-and-update-state(), delta=131072
      one = s32[] constant(1)
      next_idx = s32[] add(idx, one)
      ROOT out = (s32[], u64[2]{0}, u64[2]{0}) tuple(next_idx, prev, curr)
    }
    ENTRY main {
      zero = s32[] constant(0)
      init = u64[2]{0} constant({0, 0})
      state = (s32[], u64[2]{0}, u64[2]{0}) tuple(zero, init, init)
      ROOT loop = (s32[], u64[2]{0}, u64[2]{0}) while(state),
        condition=cond, body=body
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {}));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 3);
  EXPECT_EQ(outputs[1].Get<uint64_t>({0}), 0x7012395ULL);
  EXPECT_EQ(outputs[1].Get<uint64_t>({1}), 0ULL);
  EXPECT_EQ(outputs[2].Get<uint64_t>({0}), 0x7012395ULL + 131072ULL);
  EXPECT_EQ(outputs[2].Get<uint64_t>({1}), 0ULL);
}

TEST_F(RngGetAndUpdateStateTest, ExecuteLoopCarryIntoHighWord) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule RngStateCarry
    cond {
      state = (s32[], u64[2]{0}) parameter(0)
      idx = s32[] get-tuple-element(state), index=0
      limit = s32[] constant(4)
      ROOT lt = pred[] compare(idx, limit), direction=LT
    }
    body {
      state = (s32[], u64[2]{0}) parameter(0)
      idx = s32[] get-tuple-element(state), index=0
      curr = u64[2]{0} rng-get-and-update-state(), delta=8070450532247928832
      one = s32[] constant(1)
      next_idx = s32[] add(idx, one)
      ROOT out = (s32[], u64[2]{0}) tuple(next_idx, curr)
    }
    ENTRY main {
      zero = s32[] constant(0)
      init = u64[2]{0} constant({0, 0})
      state = (s32[], u64[2]{0}) tuple(zero, init)
      ROOT loop = (s32[], u64[2]{0}) while(state),
        condition=cond, body=body
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {}));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 2);
  // 0x7012395 + 3 * 0x7000000000000000 = 0x1_5000000007012395.
  EXPECT_EQ(outputs[1].Get<uint64_t>({0}), 0x5000000007012395ULL);
  EXPECT_EQ(outputs[1].Get<uint64_t>({1}), 1ULL);
}

}  // namespace
}  // namespace xla::gpu
