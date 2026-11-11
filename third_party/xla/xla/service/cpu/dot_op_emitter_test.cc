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

#include "xla/service/cpu/dot_op_emitter.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/cpu/target_machine_features_stub.h"
#include "xla/service/hlo_module_config.h"
#include "xla/xla.pb.h"

namespace xla::cpu {
namespace {

using DotOpEmitterTest = HloHardwareIndependentTestBase;

TEST_F(DotOpEmitterTest, LargeM1DotStrategyDependsOnYnnDotEnabled) {
  constexpr absl::string_view kHloText = R"hlo(
    HloModule large_m1_dot

    ENTRY main {
      lhs = f32[1,1024]{1,0} parameter(0)
      rhs = f32[1024,1024]{1,0} parameter(1)
      ROOT dot = f32[1,1024]{1,0} dot(lhs, rhs), lhs_contracting_dims={1},
                                                 rhs_contracting_dims={0}
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHloText));
  const HloInstruction* dot = FindInstruction(module.get(), "dot");
  ASSERT_NE(dot, nullptr);
  TargetMachineFeaturesStub target_machine_features(
      [](int64_t size) { return MinAlign(); });

  HloModuleConfig& config = module->mutable_config();
  config.mutable_debug_options().clear_xla_cpu_experimental_ynn_fusion_type();
  config.mutable_debug_options().add_xla_cpu_experimental_ynn_fusion_type(
      DebugOptions::LIBRARY_FUSION_TYPE_INDIVIDUAL_DOT);
  EXPECT_EQ(GetDotImplementationStrategy(config, *dot, target_machine_features,
                                         /*allow_runtime_calls=*/true),
            DotImplementationStrategy::kEigen);

  config.mutable_debug_options().clear_xla_cpu_experimental_ynn_fusion_type();
  EXPECT_EQ(GetDotImplementationStrategy(config, *dot, target_machine_features,
                                         /*allow_runtime_calls=*/true),
            DotImplementationStrategy::kTiledLlvmIrGemv);

  config.mutable_debug_options().add_xla_cpu_experimental_ynn_fusion_type(
      DebugOptions::LIBRARY_FUSION_TYPE_INDIVIDUAL_DOT);
  (*config.mutable_debug_options()
        .mutable_xla_backend_extra_options())["xla_is_host_offload"] = "true";
  EXPECT_EQ(GetDotImplementationStrategy(config, *dot, target_machine_features,
                                         /*allow_runtime_calls=*/true),
            DotImplementationStrategy::kTiledLlvmIrGemv);
}

TEST_F(DotOpEmitterTest, LargeM1UnsupportedYnnTypeUsesTiledGemv) {
  constexpr absl::string_view kHloText = R"hlo(
    HloModule large_m1_s32_dot

    ENTRY main {
      lhs = s32[1,1024]{1,0} parameter(0)
      rhs = s32[1024,1024]{1,0} parameter(1)
      ROOT dot = s32[1,1024]{1,0} dot(lhs, rhs), lhs_contracting_dims={1},
                                                 rhs_contracting_dims={0}
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHloText));
  const HloInstruction* dot = FindInstruction(module.get(), "dot");
  ASSERT_NE(dot, nullptr);
  TargetMachineFeaturesStub target_machine_features(
      [](int64_t size) { return MinAlign(); });

  HloModuleConfig ynn_enabled_config = module->config();
  ynn_enabled_config.mutable_debug_options()
      .clear_xla_cpu_experimental_ynn_fusion_type();
  ynn_enabled_config.mutable_debug_options()
      .add_xla_cpu_experimental_ynn_fusion_type(
          DebugOptions::LIBRARY_FUSION_TYPE_INDIVIDUAL_DOT);
  EXPECT_EQ(GetDotImplementationStrategy(ynn_enabled_config, *dot,
                                         target_machine_features,
                                         /*allow_runtime_calls=*/true),
            DotImplementationStrategy::kTiledLlvmIrGemv);
}

}  // namespace
}  // namespace xla::cpu
