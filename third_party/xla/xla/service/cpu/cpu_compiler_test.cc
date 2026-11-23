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

#include "xla/service/cpu/cpu_compiler.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/lib/monitoring/collected_metrics.h"
#include "xla/tsl/lib/monitoring/collection_registry.h"
#include "tsl/platform/platform.h"

namespace xla {
namespace cpu {
namespace {

using CpuCompilerTest = HloTestBase;

using CpuCompilerBufferAssignmentTest = HloHardwareIndependentTestBase;

TEST_F(CpuCompilerBufferAssignmentTest,
       OneDnnFusionDoesNotOverwritePostOpInput) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"hlo(
    HloModule test, is_scheduled=true

    fused {
      lhs = f32[17,23]{1,0} parameter(0)
      rhs = f32[23,19]{1,0} parameter(1)
      zeros = f32[17,19]{1,0} parameter(2)
      product = f32[17,19]{1,0} dot(lhs, rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT result = f32[17,19]{1,0} maximum(product, zeros)
    }

    ENTRY main {
      lhs = f32[17,23]{1,0} parameter(0)
      rhs = f32[23,19]{1,0} parameter(1)
      zero = f32[] constant(0)
      zeros = f32[17,19]{1,0} broadcast(zero), dimensions={}
      ROOT result = f32[17,19]{1,0} fusion(lhs, rhs, zeros),
        kind=kCustom, calls=fused,
        backend_config={"fusion_config":{"kind":"__onednn_fusion"}}
    }
  )hlo"));

  CpuCompiler compiler;
  ASSERT_OK_AND_ASSIGN(auto assignment,
                       compiler.CreateBufferAssignment(*module));
  HloInstruction* result = module->entry_computation()->root_instruction();
  EXPECT_FALSE(assignment->SharesTopLevelSlice(result, result->operand(2)));
}

TEST_F(CpuCompilerBufferAssignmentTest, LoopFusionCanStillReuseOperand) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"hlo(
    HloModule test, is_scheduled=true

    fused {
      input = f32[17,19]{1,0} parameter(0)
      ROOT result = f32[17,19]{1,0} negate(input)
    }

    ENTRY main {
      input = f32[17,19]{1,0} parameter(0)
      temporary = f32[17,19]{1,0} exponential(input)
      ROOT result = f32[17,19]{1,0} fusion(temporary), kind=kLoop, calls=fused
    }
  )hlo"));

  CpuCompiler compiler;
  ASSERT_OK_AND_ASSIGN(auto assignment,
                       compiler.CreateBufferAssignment(*module));
  HloInstruction* result = module->entry_computation()->root_instruction();
  EXPECT_TRUE(assignment->SharesTopLevelSlice(result, result->operand(0)));
}

TEST_F(CpuCompilerBufferAssignmentTest, OneDnnFusionCopiesAliasedPostOpInput) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"hlo(
    HloModule test, input_output_alias={{}: (2, {}, may-alias)}

    fused {
      lhs = f32[17,23]{1,0} parameter(0)
      rhs = f32[23,19]{1,0} parameter(1)
      zeros = f32[17,19]{1,0} parameter(2)
      product = f32[17,19]{1,0} dot(lhs, rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT result = f32[17,19]{1,0} maximum(product, zeros)
    }

    ENTRY main {
      lhs = f32[17,23]{1,0} parameter(0)
      rhs = f32[23,19]{1,0} parameter(1)
      zeros = f32[17,19]{1,0} parameter(2)
      ROOT result = f32[17,19]{1,0} fusion(lhs, rhs, zeros),
        kind=kCustom, calls=fused,
        backend_config={"fusion_config":{"kind":"__onednn_fusion"}}
    }
  )hlo"));

  CpuCompiler compiler;
  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       compiler.RunHloPasses(std::move(module), nullptr, {}));
  ASSERT_OK_AND_ASSIGN(auto schedule,
                       compiler.CreateHloSchedule(*optimized_module));
  ASSERT_OK(optimized_module->set_schedule(std::move(schedule)));
  ASSERT_OK_AND_ASSIGN(auto assignment,
                       compiler.CreateBufferAssignment(*optimized_module));
  HloInstruction* result =
      optimized_module->entry_computation()->root_instruction();
  HloInstruction* fusion = nullptr;
  for (HloInstruction* instruction :
       optimized_module->entry_computation()->instructions()) {
    if (instruction->IsCustomFusion()) {
      ASSERT_EQ(fusion, nullptr);
      fusion = instruction;
    }
  }
  ASSERT_NE(fusion, nullptr) << optimized_module->ToString();
  EXPECT_TRUE(result->opcode() == HloOpcode::kCopy ||
              fusion->operand(2)->opcode() == HloOpcode::kCopy);
  EXPECT_FALSE(assignment->SharesTopLevelSlice(fusion, fusion->operand(2)));
  // A copy separates the library buffers while preserving the caller's alias.
  EXPECT_TRUE(assignment->SharesTopLevelSlice(
      result, optimized_module->entry_computation()->parameter_instruction(2)));
}

struct CustomFusionReuseCase {
  const char* name;
  const char* backend_config;
  bool can_reuse;
};

class CpuCompilerCustomFusionBufferAssignmentTest
    : public HloHardwareIndependentTestBase,
      public ::testing::WithParamInterface<CustomFusionReuseCase> {};

TEST_P(CpuCompilerCustomFusionBufferAssignmentTest, ElementwiseOperandReuse) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(absl::StrReplaceAll(
                           R"hlo(
    HloModule test, is_scheduled=true

    fused {
      input = f32[17,19]{1,0} parameter(0)
      ROOT result = f32[17,19]{1,0} negate(input)
    }

    ENTRY main {
      input = f32[17,19]{1,0} parameter(0)
      temporary = f32[17,19]{1,0} exponential(input)
      ROOT result = f32[17,19]{1,0} fusion(temporary),
        kind=kCustom, calls=fused, backend_config=$config
    }
  )hlo",
                           {{"$config", GetParam().backend_config}})));

  CpuCompiler compiler;
  ASSERT_OK_AND_ASSIGN(auto assignment,
                       compiler.CreateBufferAssignment(*module));
  HloInstruction* result = module->entry_computation()->root_instruction();
  EXPECT_EQ(assignment->SharesTopLevelSlice(result, result->operand(0)),
            GetParam().can_reuse);
}

INSTANTIATE_TEST_SUITE_P(
    CustomFusions, CpuCompilerCustomFusionBufferAssignmentTest,
    ::testing::Values(
        CustomFusionReuseCase{
            "OneDnn", R"({"fusion_config":{"kind":"__onednn_fusion"}})", false},
        CustomFusionReuseCase{
            "Unknown", R"({"fusion_config":{"kind":"__unknown_fusion"}})",
            false},
        CustomFusionReuseCase{"MissingConfig", "{}", false},
        CustomFusionReuseCase{"InvalidConfig", R"({"fusion_config":42})",
                              false},
        CustomFusionReuseCase{
            "Ynn", R"({"fusion_config":{"kind":"__ynn_fusion"}})", true}),
    [](const ::testing::TestParamInfo<CustomFusionReuseCase>& info) {
      return info.param.name;
    });

constexpr absl::string_view kCpuCompilerStacktraceMetricName =
    "/xla/service/cpu/compiler_stacktrace_count";

TEST_F(CpuCompilerTest, RecordsStreamzStackTrace) {
  if (tsl::kIsOpenSource) {
    GTEST_SKIP() << "Streamz is not supported in OSS.";
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
    HloModule test
    ENTRY main {
      p = f32[10]{0} parameter(0)
      ROOT neg = f32[10]{0} negate(p)
    }
  )"));
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/true));

  tsl::monitoring::CollectionRegistry::CollectMetricsOptions options;
  std::unique_ptr<tsl::monitoring::CollectedMetrics> metrics =
      tsl::monitoring::CollectionRegistry::Default()->CollectMetrics(options);

  const auto it = metrics->point_set_map.find(
      std::string(kCpuCompilerStacktraceMetricName));
  ASSERT_TRUE(it != metrics->point_set_map.end());

  // Since Streamz is recorded every call, we expect at least one point.
  // All other callers may increment the counter as well.
  EXPECT_GT(it->second->points.size(), 0);
}

TEST_F(CpuCompilerTest, CompilationWithLargeConstants) {
  absl::string_view module_string = R"(
HloModule module

ENTRY main {
  a = f32[1000,1000]{1,0} parameter(0)
  b = f32[1000,1000]{1,0} constant({...})
  a_plus_b = f32[1000,1000]{1,0} add(a, b)
  c = f32[1000,1000]{1,0} constant({...})
  ROOT result = f32[1000,1000]{1,0} add(a_plus_b, c)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(module_string));

  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/true));
}

TEST_F(CpuCompilerTest, NoLibraryCallsInHostOffload) {
  absl::string_view hlo = R"(
HloModule module
ENTRY main {
  a = f32[128,128]{1,0} parameter(0)
  b = f32[128,128]{1,0} parameter(1)
  ROOT dot = f32[128,128]{1,0} dot(a, b), lhs_contracting_dims={1},
                                          rhs_contracting_dims={0}
}
)";
  HloModuleConfig config;
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.add_xla_cpu_experimental_ynn_fusion_type(
      xla::DebugOptions::LIBRARY_FUSION_TYPE_INDIVIDUAL_DOT);
  debug_options.set_xla_cpu_experimental_onednn_custom_call(false);
  debug_options.mutable_xla_backend_extra_options()->insert(
      {"xla_is_host_offload", "true"});
  config.set_debug_options(debug_options);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                       GetOptimizedModule(hlo, config));

  for (auto* computation : optimized_module->computations()) {
    for (auto* instruction : computation->instructions()) {
      // A single dot is only in a custom fusion when it is sent to libraries.
      if (instruction->opcode() == HloOpcode::kFusion) {
        EXPECT_NE(instruction->fusion_kind(),
                  HloInstruction::FusionKind::kCustom);
      }
    }
  }
}

TEST_F(CpuCompilerTest, PermutationSortConvertedToScatter) {
  absl::string_view hlo = R"(
    HloModule test

    compare {
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, order=TOTAL
    }

    compare2 {
      p.0.lhs = s32[] parameter(0)
      p.0.rhs = s32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
    }

    ENTRY sort_computation {
      keys = f32[4,8]{1,0} parameter(0)
      values = s32[4,8]{1,0} iota(), iota_dimension=1
      sort = (f32[4,8]{1,0}, s32[4,8]{1,0}) sort(keys, values), dimensions={1}, to_apply=compare
      gte = s32[4,8]{1,0} get-tuple-element(sort), index=1
      ROOT sort2 = (s32[4,8]{1,0}, s32[4,8]{1,0}) sort(gte, values), dimensions={1}, to_apply=compare2
    }
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                       GetOptimizedModule(hlo, config));

  int64_t sort_count = 0;
  for (auto* computation : optimized_module->computations()) {
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kSort) {
        sort_count++;
      }
    }
  }
  EXPECT_EQ(sort_count, 1);

  EXPECT_TRUE(Run(hlo, /*run_hlo_passes=*/true));
}

TEST_F(CpuCompilerTest, CompilesElementalOpsWithFusionWrapperPassDisabled) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
    HloModule test
    ENTRY main {
      p0 = f32[8]{0} parameter(0)
      p1 = f32[8]{0} parameter(1)
      ROOT add = f32[8]{0} add(p0, p1)
    }
  )"));
  module->mutable_config().mutable_debug_options().add_xla_disable_hlo_passes(
      "fusion-wrapper");

  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/true));
}

}  // namespace
}  // namespace cpu
}  // namespace xla
