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

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/codegen/llvm_kernel_definition.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::cpu {
namespace {

using ::absl_testing::IsOk;
using ::testing::Not;
using ParallelFusionEmitterTest = HloHardwareIndependentTestBase;

FusionCompiler::Options CreateDefaultOptions() {
  FusionCompiler::Options options;
  options.vector_width = 128;
  options.verification_level = 1;
  options.fast_min_max = false;

  return options;
}

TEST_F(ParallelFusionEmitterTest, HappyPathSingleFusion) {
  constexpr absl::string_view expected_name = "root_fusion";
  constexpr absl::string_view trivial_fusion = R"(
    add1 {
      p = s32[] parameter(0)
      c = s32[] constant(1)
      ROOT a = s32[] add(p, c)
    }

    ENTRY main {
      p = s32[] parameter(0)
      ROOT root_fusion = s32[] fusion(p), kind=kLoop, calls=add1
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(trivial_fusion));
  HloComputation* computation = hlo_module->entry_computation();
  HloFusionInstruction* fusion =
      Cast<HloFusionInstruction>(computation->root_instruction());

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_pool", 4);
  xla::cpu::ParallelFusionEmitter fussion_emitter(
      thread_pool, CreateDefaultOptions(), nullptr, false);

  TF_ASSERT_OK_AND_ASSIGN(auto kernel_spec, fussion_emitter.AddFusion(fusion));
  EXPECT_EQ(kernel_spec.name(), expected_name);

  TF_ASSERT_OK_AND_ASSIGN(auto kernels, fussion_emitter.ConsumeKernels());
  ASSERT_EQ(kernels.size(), 1);
  LlvmKernelDefinition& lowered_kernel = kernels[0];
  auto [spec, source] = std::move(lowered_kernel).ReleaseStorage();
  EXPECT_EQ(spec.name(), expected_name);

  llvm::orc::ThreadSafeModule llvm_module =
      std::move(source).thread_safe_module();
  EXPECT_NE(llvm_module.getModuleUnlocked()->getFunction(expected_name),
            nullptr);
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
      thread_pool, CreateDefaultOptions(), nullptr, false);

  EXPECT_THAT(fussion_emitter.AddFusion(fusion), Not(IsOk()));
}

}  // namespace
}  // namespace xla::cpu
