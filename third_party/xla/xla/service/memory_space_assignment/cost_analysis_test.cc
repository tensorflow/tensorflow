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

#include "xla/service/memory_space_assignment/cost_analysis.h"

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/hlo_cost_analysis.h"
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

constexpr int64_t kPointerSize = 8;

int64_t ShapeSize(const Shape& shape) {
  return ShapeUtil::ByteSizeOf(shape, kPointerSize);
}

class MemorySpaceAssignmentCostAnalysisTest : public HloTestBase {
 protected:
  Status Initialize(const HloModule* module,
                    float pipeline_overhead_window_size_mib = 0.0) {
    HloCostAnalysis::Options options;
    options_.alternate_mem_bandwidth_bytes_per_second = 128;
    options_.async_copy_bandwidth_bytes_per_second = 32;
    options_.pipeline_overhead_window_size_mib =
        pipeline_overhead_window_size_mib;
    options.shape_size = ShapeSize;
    options.set_flops_per_second(8);
    options.set_bytes_per_second(32);
    options.set_transcendentals_per_second(16);
    hlo_cost_analysis_ = std::make_unique<HloCostAnalysis>(options);
    TF_RETURN_IF_ERROR(
        module->entry_computation()->Accept(hlo_cost_analysis_.get()));
    TF_ASSIGN_OR_RETURN(
        cost_analysis_,
        CostAnalysis::Create(*hlo_cost_analysis_, options_, *module));
    return OkStatus();
  }

  CostAnalysisOptions options_;
  std::unique_ptr<HloCostAnalysis> hlo_cost_analysis_;
  std::unique_ptr<CostAnalysis> cost_analysis_;
};

TEST_F(MemorySpaceAssignmentCostAnalysisTest, NoPipelineOverhead) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY Entry {
    param0 = f32[2,4] parameter(0)
    param1 = f32[2,4] parameter(1)
    ROOT add = f32[2,4] add(param0, param1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(Initialize(module.get()));

  const HloInstruction* add = module->entry_computation()->root_instruction();
  const float expected_compute_elapsed =
      /*num_flops=*/8 / /*flops_per_second=*/8.0;
  LOG(INFO) << "Expected compute elapsed = " << expected_compute_elapsed;
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedDueToCompute(*add),
            expected_compute_elapsed);
  float expected_memory_elapsed =
      /*bytes_accessed=*/(3 * 4 * 8) / /*bytes_per_second=*/32.0;
  LOG(INFO) << "Expected memory elapsed = " << expected_memory_elapsed;
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedDueToMemory(*add),
            expected_memory_elapsed);

  // This HLO is memory-bound.
  EXPECT_EQ(cost_analysis_->GetInstructionElapsed(*add),
            expected_memory_elapsed);
  EXPECT_EQ(
      cost_analysis_->GetInstructionElapsedInAlternateMemory(*add, {}, {}),
      expected_memory_elapsed);

  // Put operand 0 in alternate memory. Still memory bound.
  expected_memory_elapsed =
      (/*bytes_accessed=*/(2 * 4 * 8) / /*bytes_per_second=*/32.0) +
      (/*bytes_accessed=*/(4 * 8) / /*bytes_per_second=*/128.0);
  LOG(INFO) << "Expected memory elapsed = " << expected_memory_elapsed;
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedDueToMemory(*add, {{0, {}}}),
            expected_memory_elapsed);
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedInAlternateMemory(
                *add, {{0, {}}}, {}),
            expected_memory_elapsed);

  // Put operand 0 and output in alternate memory. Still memory bound.
  expected_memory_elapsed =
      (/*bytes_accessed=*/(4 * 8) / /*bytes_per_second=*/32.0) +
      (/*bytes_accessed=*/(2 * 4 * 8) / /*bytes_per_second=*/128.0);
  LOG(INFO) << "Expected memory elapsed = " << expected_memory_elapsed;
  EXPECT_EQ(
      cost_analysis_->GetInstructionElapsedDueToMemory(*add, {{0, {}}}, {{}}),
      expected_memory_elapsed);
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedInAlternateMemory(
                *add, {{0, {}}}, {{}}),
            expected_memory_elapsed);

  // Put everything in alternate memory. We're now compute bound.
  expected_memory_elapsed =
      /*bytes_accessed=*/(3 * 4 * 8) / /*bytes_per_second=*/128.0;
  LOG(INFO) << "Expected memory elapsed = " << expected_memory_elapsed;
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedDueToMemory(
                *add, {{0, {}}, {1, {}}}, {{}}),
            expected_memory_elapsed);
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedInAlternateMemory(
                *add, {{0, {}}, {1, {}}}, {{}}),
            expected_compute_elapsed);
}

TEST_F(MemorySpaceAssignmentCostAnalysisTest, PipelineOverhead) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY Entry {
    param0 = f32[2,4] parameter(0)
    param1 = f32[2,4] parameter(1)
    ROOT add = f32[2,4] add(param0, param1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  // Set the window size 64B.
  TF_ASSERT_OK(
      Initialize(module.get(),
                 /*pipeline_overhead_window_size_mib=*/(64.0 / 1024 / 1024)));

  const HloInstruction* add = module->entry_computation()->root_instruction();
  const float expected_compute_elapsed =
      /*num_flops=*/8 / /*flops_per_second=*/8.0;
  LOG(INFO) << "Expected compute elapsed = " << expected_compute_elapsed;
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedDueToCompute(*add),
            expected_compute_elapsed);
  float expected_memory_elapsed =
      /*bytes_accessed=*/(3 * 4 * 8) / /*bytes_per_second=*/32.0;
  LOG(INFO) << "Expected memory elapsed = " << expected_memory_elapsed;
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedDueToMemory(*add),
            expected_memory_elapsed);

  float expected_overhead = expected_compute_elapsed * 2 / 3;
  LOG(INFO) << "Expected overhead = " << expected_overhead;
  EXPECT_EQ(cost_analysis_->GetDefaultMemoryAccessOverhead(*add),
            expected_overhead);
  // This HLO is memory-bound.
  EXPECT_EQ(cost_analysis_->GetInstructionElapsed(*add),
            expected_memory_elapsed + expected_overhead);
  EXPECT_EQ(
      cost_analysis_->GetInstructionElapsedInAlternateMemory(*add, {}, {}),
      expected_memory_elapsed + expected_overhead);

  // Put operand 0 in alternate memory. Still memory bound.
  expected_memory_elapsed =
      (/*bytes_accessed=*/(2 * 4 * 8) / /*bytes_per_second=*/32.0) +
      (/*bytes_accessed=*/(4 * 8) / /*bytes_per_second=*/128.0);
  LOG(INFO) << "Expected memory elapsed = " << expected_memory_elapsed;
  EXPECT_EQ(cost_analysis_->GetDefaultMemoryAccessOverhead(*add, {{0, {}}}),
            expected_overhead);
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedDueToMemory(*add, {{0, {}}}),
            expected_memory_elapsed);
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedInAlternateMemory(
                *add, {{0, {}}}, {}),
            expected_memory_elapsed + expected_overhead);

  // Put operand 0 and output in alternate memory. Still memory bound.
  expected_memory_elapsed =
      (/*bytes_accessed=*/(4 * 8) / /*bytes_per_second=*/32.0) +
      (/*bytes_accessed=*/(2 * 4 * 8) / /*bytes_per_second=*/128.0);
  LOG(INFO) << "Expected memory elapsed = " << expected_memory_elapsed;
  expected_overhead = expected_compute_elapsed / 3;
  LOG(INFO) << "Expected overhead = " << expected_overhead;
  EXPECT_EQ(
      cost_analysis_->GetDefaultMemoryAccessOverhead(*add, {{0, {}}}, {{}}),
      expected_overhead);
  EXPECT_EQ(
      cost_analysis_->GetInstructionElapsedDueToMemory(*add, {{0, {}}}, {{}}),
      expected_memory_elapsed);
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedInAlternateMemory(
                *add, {{0, {}}}, {{}}),
            expected_memory_elapsed + expected_overhead);

  // Put everything in alternate memory. We're now compute bound.
  expected_memory_elapsed =
      /*bytes_accessed=*/(3 * 4 * 8) / /*bytes_per_second=*/128.0;
  LOG(INFO) << "Expected memory elapsed = " << expected_memory_elapsed;
  expected_overhead = 0;
  LOG(INFO) << "Expected overhead = " << expected_overhead;
  EXPECT_EQ(cost_analysis_->GetDefaultMemoryAccessOverhead(
                *add, {{0, {}}, {1, {}}}, {{}}),
            expected_overhead);
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedDueToMemory(
                *add, {{0, {}}, {1, {}}}, {{}}),
            expected_memory_elapsed);
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedInAlternateMemory(
                *add, {{0, {}}, {1, {}}}, {{}}),
            expected_compute_elapsed);
}

}  // namespace
}  // namespace xla
