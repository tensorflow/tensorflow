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

#include "xla/service/memory_space_assignment/simulator.h"

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/memory_space_assignment/allocation.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using memory_space_assignment::CostAnalysis;
using memory_space_assignment::CostAnalysisOptions;
using memory_space_assignment::RuntimeSimulator;

constexpr int64_t kPointerSize = 8;

int64_t ShapeSize(const Shape& shape) {
  return ShapeUtil::ByteSizeOf(shape, kPointerSize);
}

class MemorySpaceAssignmentSimulatorTest : public HloTestBase {
 protected:
  absl::Status Initialize(const HloModule* module) {
    HloCostAnalysis::Options tpu_device_options;
    tpu_device_options.shape_size = ShapeSize;
    // Assume 1 FLOP per second for testing.
    tpu_device_options.set_flops_per_second(1);
    hlo_cost_analysis_ = std::make_unique<HloCostAnalysis>(tpu_device_options);
    TF_RETURN_IF_ERROR(
        module->entry_computation()->Accept(hlo_cost_analysis_.get()));
    hlo_cost_analysis_costs_ =
        std::make_unique<memory_space_assignment::HloCostAnalysisCosts>(
            *hlo_cost_analysis_);
    CostAnalysisOptions _options;
    TF_ASSIGN_OR_RETURN(
        cost_analysis_,
        CostAnalysis::Create(*hlo_cost_analysis_costs_, _options, *module));
    runtime_simulator_ =
        std::make_unique<xla::memory_space_assignment::RuntimeSimulator>(
            cost_analysis_.get());
    return absl::OkStatus();
  }
  std::unique_ptr<HloCostAnalysis> hlo_cost_analysis_;
  std::unique_ptr<memory_space_assignment::HloCostAnalysisCosts>
      hlo_cost_analysis_costs_;
  std::unique_ptr<CostAnalysis> cost_analysis_;
  std::unique_ptr<RuntimeSimulator> runtime_simulator_;
};

TEST_F(MemorySpaceAssignmentSimulatorTest, SingleLayerNestedLoop) {
  absl::string_view hlo_string =
      R"(HloModule module, is_scheduled=true

      %body {
        %constant.1 = s32[] constant(1)
        %param = (s32[]) parameter(0)
        %count = s32[] get-tuple-element(%param), index=0
        %increment = s32[] add(s32[] %count, s32[] %constant.1)
        ROOT %loop_result = (s32[]) tuple(%increment)
      }

      %condition {
        %param = (s32[]) parameter(0)
        %constant.42 = s32[] constant(42)
        %condition_input = s32[] get-tuple-element(%param), index=0
        ROOT %greater = pred[] compare(s32[] %constant.42, s32[] %condition_input), direction=GT
      }

      ENTRY Entry {
        %dummy_input = s32[] parameter(0)
        %constant.0 = s32[] constant(0)
        ROOT %while = (s32[]) while(tuple(%constant.0)), condition=%condition, body=%body
      }

    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(Initialize(module.get()));

  TF_ASSERT_OK_AND_ASSIGN(auto alias_analysis,
                          HloAliasAnalysis::Run(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_live_range,
                          HloLiveRange::Run(module->schedule(), *alias_analysis,
                                            module->entry_computation()));

  // Since the HLO does not contain memory access, pass an empty allocation
  // sequence for test.
  memory_space_assignment::AllocationSequence allocations;
  // The while loop has 42 iterations, and each iteration has 2 FLOP (for
  // %increment and %greater). Thus, the total FLOPs are 84 FLOPs.
  float expected_elapsed_time = 84;
  EXPECT_EQ(runtime_simulator_->ComputeEstimatedElapsedTime(*hlo_live_range,
                                                            allocations),
            expected_elapsed_time);
}

}  // namespace
}  // namespace xla
