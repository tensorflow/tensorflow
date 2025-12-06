/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/cpu/parallel_fusion_emitter.h"

#include <cstdint>
#include <functional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/platform/threadpool_interface.h"

namespace xla::cpu {
namespace {

using ::absl_testing::IsOk;
using ::testing::_;
using ::testing::Not;

class ParallelFusionEmitterTest : public HloHardwareIndependentTestBase {
 protected:
  // Binds the mock hooks to the fusion compiler hooks and sets the number of
  // expected calls.
  FusionCompiler::CompilationHooks CreateMockHooks(int64_t num_calls);

 private:
  testing::MockFunction<void(mlir::ModuleOp)> pre_optimization_mock_;
  testing::MockFunction<void(mlir::ModuleOp)> post_optimization_mock_;
  testing::MockFunction<void(mlir::ModuleOp)> post_lowering_mock_;
};

FusionCompiler::Options CreateDefaultOptions() {
  FusionCompiler::Options options;
  options.vector_width = 128;
  options.verification_level = 1;
  options.fast_min_max = false;
  options.fast_math_flags = llvm::FastMathFlags::getFast();

  return options;
}

FusionCompiler::CompilationHooks ParallelFusionEmitterTest::CreateMockHooks(
    int64_t num_calls) {
  auto create_mock_function =
      [num_calls](testing::MockFunction<void(mlir::ModuleOp)>& mock_function) {
        EXPECT_CALL(mock_function, Call(_)).Times(num_calls);
        return mock_function.AsStdFunction();
      };

  FusionCompiler::CompilationHooks hooks;
  hooks.pre_optimization = create_mock_function(pre_optimization_mock_);
  hooks.post_optimization = create_mock_function(post_optimization_mock_);
  hooks.post_lowering = create_mock_function(post_lowering_mock_);

  return hooks;
}

class BlockingThreadPool final : public tsl::thread::ThreadPoolInterface {
 public:
  void Schedule(std::function<void()> fn) override { fn(); }
  int NumThreads() const override { return 1; }
  int CurrentThreadId() const override { return 0; }
};

TEST_F(ParallelFusionEmitterTest, HappyPathSingleFusion) {
  constexpr absl::string_view expected_name = "root_fusion";
  constexpr absl::string_view trivial_fusion = R"(
    add1 {
      p = f32[] parameter(0)
      c = f32[] constant(1)
      ROOT a = f32[] add(p, c)
    }

    ENTRY main {
      p = f32[] parameter(0)
      ROOT root_fusion = f32[] fusion(p), kind=kLoop, calls=add1
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(trivial_fusion));
  HloComputation* computation = hlo_module->entry_computation();
  HloFusionInstruction* fusion =
      Cast<HloFusionInstruction>(computation->root_instruction());

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_pool", 4);

  xla::cpu::ParallelFusionEmitter fussion_emitter(
      thread_pool, CreateDefaultOptions(), CreateMockHooks(1), nullptr, false,
      false);

  TF_ASSERT_OK_AND_ASSIGN(auto kernel_spec, fussion_emitter.AddFusion(fusion));
  EXPECT_EQ(kernel_spec.name(), expected_name);

  TF_ASSERT_OK_AND_ASSIGN(auto kernels, fussion_emitter.ConsumeKernels());
  ASSERT_EQ(kernels.size(), 1);
  KernelDefinition<LlvmKernelSource>& lowered_kernel = kernels[0];
  EXPECT_EQ(lowered_kernel.spec().name(), expected_name);
  auto source = std::move(lowered_kernel).TakeSource();

  llvm::orc::ThreadSafeModule thread_safe_llvm_module =
      std::move(source).thread_safe_module();
  llvm::Module* llvm_module = thread_safe_llvm_module.getModuleUnlocked();
  EXPECT_NE(llvm_module->getFunction(expected_name), nullptr);
  TF_ASSERT_OK_AND_ASSIGN(
      bool passed,
      RunFileCheck(llvm_ir::DumpToString(*llvm_module), "CHECK: fadd fast"));
  EXPECT_TRUE(passed);
}

TEST_F(ParallelFusionEmitterTest, FusionsAreSorted) {
  constexpr absl::string_view trivial_fusion = R"(
    fusion_computation_0 {
      p = s32[] parameter(0)
      c = s32[] constant(1)
      ROOT a = s32[] add(p, c)
    }

    fusion_computation_1 {
      p = s32[] parameter(0)
      c = s32[] constant(1)
      ROOT a = s32[] add(p, c)
    }

    ENTRY main {
      p = s32[] parameter(0)
      fusion_0 = s32[] fusion(p), kind=kLoop, calls=fusion_computation_0
      fusion_1 = s32[] fusion(p), kind=kLoop, calls=fusion_computation_1
      ROOT result_tuple = (s32[], s32[]) tuple(fusion_0, fusion_1)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(trivial_fusion));
  HloComputation* computation = hlo_module->entry_computation();
  HloInstruction* root_tuple = computation->root_instruction();
  const auto* fusion_0 = Cast<HloFusionInstruction>(root_tuple->operand(0));
  const auto* fusion_1 = Cast<HloFusionInstruction>(root_tuple->operand(1));

  BlockingThreadPool blocking_thread_pool;
  tsl::thread::ThreadPool thread_pool(&blocking_thread_pool);

  xla::cpu::ParallelFusionEmitter fussion_emitter(
      thread_pool, CreateDefaultOptions(), CreateMockHooks(2),
      /*buffer_assignment=*/nullptr, /*use_unique_c_name=*/false,
      /*enable_tiled_emitter=*/false);

  // Add the fusions in reverse order.
  TF_ASSERT_OK_AND_ASSIGN(auto kernel_spec_1,
                          fussion_emitter.AddFusion(fusion_1));
  EXPECT_EQ(kernel_spec_1.name(), "fusion_1");

  TF_ASSERT_OK_AND_ASSIGN(auto kernel_spec_0,
                          fussion_emitter.AddFusion(fusion_0));
  EXPECT_EQ(kernel_spec_0.name(), "fusion_0");

  TF_ASSERT_OK_AND_ASSIGN(auto kernels, fussion_emitter.ConsumeKernels());
  ASSERT_EQ(kernels.size(), 2);
  EXPECT_EQ(kernels[0].spec().name(), "fusion_0");
  EXPECT_EQ(kernels[1].spec().name(), "fusion_1");
}

// Check that error condition from emitting is propagated.
TEST_F(ParallelFusionEmitterTest, Error) {
  // Dot instruction with algorithm dot_bf16_bf16_bf16 is not supported by the
  // loop emitter.
  constexpr absl::string_view trivial_fusion = R"(
    dot_fusion {
      lhs = bf16[10] parameter(0)
      rhs = bf16[10] parameter(1)
      ROOT dot = bf16[] dot(lhs, rhs),
        lhs_contracting_dims={0}, rhs_contracting_dims={0},
        algorithm=dot_bf16_bf16_bf16
    }

    ENTRY main {
      lhs = bf16[10] parameter(0)
      rhs = bf16[10] parameter(1)
      ROOT result = bf16[] fusion(lhs, rhs), kind=kLoop, calls=dot_fusion
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(trivial_fusion));
  HloComputation* computation = hlo_module->entry_computation();
  HloFusionInstruction* fusion =
      Cast<HloFusionInstruction>(computation->root_instruction());

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_pool", 4);
  xla::cpu::ParallelFusionEmitter fussion_emitter(
      thread_pool, CreateDefaultOptions(), CreateMockHooks(0), nullptr, false,
      false);

  EXPECT_THAT(fussion_emitter.AddFusion(fusion), Not(IsOk()));
}

}  // namespace
}  // namespace xla::cpu
