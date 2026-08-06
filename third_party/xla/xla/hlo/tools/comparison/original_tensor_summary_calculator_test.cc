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

#include "xla/hlo/tools/comparison/original_tensor_summary_calculator.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/tools/comparison/comparison_service.pb.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/hlo/tools/comparison/tensor_summary_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/test.h"

namespace xla::numerics::comparison {
namespace {

using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::Pointee;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;
using ::tsl::testing::IsOk;

using FloatSummary = ::xla::comparison::FloatSummary;
using FloatBlockSummary = ::xla::comparison::FloatBlockSummary;
using DimSplitSpec = ::xla::comparison::DimSplitSpec;
using OriginalTensorInfo = OriginalTensorSummaryCalculator::OriginalTensorInfo;
using TensorTransformation = tensor_transformation::TensorTransformation;
using Reshape = tensor_transformation::Reshape;
using Unshard = tensor_transformation::Unshard;
using ShardTensorSummary = OriginalTensorSummaryCalculator::ShardTensorSummary;

struct CallbackResult {
  AbsoluteScopedTensorKey original_tensor_key;
  std::shared_ptr<const TensorTransformation> pending_transformation;
  OriginalTensorSummary original_tensor_summary;
};

// Helper to create a FloatSummary with multiple blocks.
FloatSummary CreateSummary(absl::Span<const float> values,
                           absl::Span<const DimSplitSpec> split_spec) {
  int64_t expected_num_blocks;
  if (split_spec.empty()) {
    expected_num_blocks = values.size();
    CHECK(expected_num_blocks <= 1)
        << "Scalar summary can have at most 1 block.";
  } else {
    expected_num_blocks = 1;
    for (const auto& spec : split_spec) {
      expected_num_blocks *= spec.block_count;
    }
  }

  CHECK_EQ(values.size(), expected_num_blocks)
      << "Number of values does not match the expected number of blocks "
         "according to the split spec.";

  std::vector<FloatBlockSummary> block_summaries;
  std::vector<int64_t> current_indices(split_spec.size(), 0);

  for (int i = 0; i < values.size(); ++i) {
    float val = values[i];
    block_summaries.push_back({/*block_indices=*/current_indices,
                               /*min=*/val,
                               /*max=*/val,
                               /*mean=*/val,
                               /*stddev=*/0,
                               /*count=*/1});

    if (!split_spec.empty()) {
      // Increment current_indices for the next block
      for (int j = split_spec.size() - 1; j >= 0; --j) {
        current_indices[j]++;
        if (current_indices[j] < split_spec[j].block_count) {
          break;
        }
        current_indices[j] = 0;
      }
    }
  }

  return FloatSummary{
      /*block_summaries=*/std::move(block_summaries),
      /*split_spec=*/
      std::vector<DimSplitSpec>(split_spec.begin(), split_spec.end()),
  };
}

class OriginalTensorSummaryCalculatorCreateTest
    : public ::xla::HloHardwareIndependentTestBase {
 protected:
  void SetUp() override {
    HloHardwareIndependentTestBase::SetUp();
    results_.clear();
  }

  std::vector<CallbackResult> results_;
  OriginalTensorSummaryCallback callback_ =
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const TensorTransformation> pending,
          const OriginalTensorSummary& summary) {
        results_.push_back({key, std::move(pending), summary});
        return absl::OkStatus();
      };
};

TEST_F(OriginalTensorSummaryCalculatorCreateTest, BasicUnshard) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule jit_demo, entry_computation_layout={(s32[2,8]{1,0}, s32[2,8]{1,0})->s32[2,8]{1,0}}, origin_recovery_table={
  {"add.3"} : {"add.3__ovp0"},
  "
    ENTRY %recovery_computation (param: s32[2,8]) -> s32[8,8] {
      %param = s32[2,8]{1,0} parameter(0), sharding={devices=[4,1]<=[4]}
      ROOT %all-gather = s32[8,8]{1,0} all-gather(%param), dimensions={0}, sharding={replicated}
    }
  "
}

ENTRY %main.6_spmd (param: s32[2,8], param.1: s32[2,8]) -> s32[2,8] {
  %param.1 = s32[2,8]{1,0} parameter(1), sharding={devices=[4,1]<=[4]}
  %param = s32[2,8]{1,0} parameter(0), sharding={devices=[4,1]<=[4]}
  ROOT %add.1 = s32[2,8]{1,0} add(%param, %param.1), origin={{"add.3__ovp0"}}
}
)hlo";
  constexpr absl::string_view original_hlo_string = R"hlo(
HloModule original_module
ENTRY %main(p0: s32[8,8], p1: s32[8,8]) -> s32[8,8] {
  %p0 = s32[8,8]{1,0} parameter(0)
  %p1 = s32[8,8]{1,0} parameter(1)
  ROOT %add.3 = s32[8,8]{1,0} add(%p0, %p1)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto original_module,
                       ParseAndReturnVerifiedModule(original_hlo_string));
  ASSERT_OK_AND_ASSIGN(
      auto calculator_with_metrics,
      OriginalTensorSummaryCalculator::Create(
          optimized_module.get(), original_module.get(), std::move(callback_)));
  auto& calculator = calculator_with_metrics.first;
  EXPECT_EQ(calculator->DumpHloDerivedData(),
            R"(Optimized Tensor Dimensions:
  add.1: [2, 8]

Call Map:

Original Tensor by Optimized Tensor Key:
  add.1:
    add.3 via Unshard{dimensions=[8, 8], sharding={devices=[4,1]<=[4]}, continuation=nullptr}
)");
  const auto& creation_metrics = calculator_with_metrics.second;
  EXPECT_EQ(creation_metrics.optimized_module_tensor_count, 3);
  EXPECT_EQ(creation_metrics.optimized_module_tensor_with_original_array_count,
            1);
  EXPECT_EQ(creation_metrics.original_module_recoverable_tensor_count, 1);
  EXPECT_THAT(creation_metrics.recoverable_tensor_keys,
              UnorderedElementsAre(TensorKey::Create("add.3")));

  AbsoluteScopedTensorKey opt_key{
      /*tensor_key=*/TensorKey{/*instruction_name=*/"add.1"}};
  // Shard shape is [2,8]. Split dim 0 of shard into 2 blocks.
  std::vector<DimSplitSpec> shard_split_spec = {
      {/*dim_index=*/0, /*block_count=*/2}};
  for (int i = 0; i < 4; ++i) {
    std::vector<float> values = {static_cast<float>(i),
                                 static_cast<float>(i) + 0.1f};
    ASSERT_THAT(calculator->ProcessShardSummary(
                    opt_key, {static_cast<int64_t>(i),
                              CreateSummary(values, shard_split_spec)}),
                IsOk());
  }
  ASSERT_THAT(results_, SizeIs(1));
  EXPECT_THAT(results_[0].original_tensor_key.tensor_key.instruction_name,
              Eq("add.3"));
  EXPECT_THAT(results_[0].pending_transformation, Eq(nullptr));
  EXPECT_THAT(results_[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{8, 8}));
  const auto& summary = results_[0].original_tensor_summary.summaries[0];
  EXPECT_THAT(
      summary.split_spec,
      Eq(std::vector<DimSplitSpec>{{/*dim_index=*/0, /*block_count=*/8}}));
  ASSERT_THAT(summary.block_summaries, SizeIs(8));
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(summary.block_summaries[i * 2].min, Eq(static_cast<float>(i)));
    EXPECT_THAT(summary.block_summaries[i * 2 + 1].min,
                Eq(static_cast<float>(i) + 0.1f));
  }
  auto processing_metrics = calculator->GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count, 4);
  EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count, 4);
  EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics.completed_original_tensor_count, 1);
  EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 0);
}

TEST_F(OriginalTensorSummaryCalculatorCreateTest,
       UnshardWithComplexRecoveryModule) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule jit_demo, origin_recovery_table={
  {"add.3"} : {"add.3__ovp0"},
  "
    ENTRY %recovery_computation (param: s32[1,1]) -> s32[2,4] {
      %param = s32[1,1]{1,0} parameter(0), sharding={devices=[2,4]<=[8]}
      %reshape = s32[1,1,1]{2,1,0} reshape(%param)
      %all-gather = s32[8,1,1]{2,1,0} all-gather(%reshape), dimensions={0}
      %reshape.1 = s32[2,4,1,1]{3,2,1,0} reshape(%all-gather)
      %transpose = s32[2,1,4,1]{3,1,2,0} transpose(%reshape.1), dimensions={0,2,1,3}
      ROOT %reshape.2 = s32[2,4]{1,0} reshape(%transpose)
    }
  "
}

ENTRY %main.0_spmd (param: s32[1,1]) -> s32[1,1] {
  %param = s32[1,1]{1,0} parameter(0), sharding={devices=[2,4]<=[8]}
  ROOT %add.1 = s32[1,1]{1,0} add(%param, %param), origin={{"add.3__ovp0"}}
}
)hlo";

  constexpr absl::string_view original_hlo_string = R"hlo(
HloModule original_module
ENTRY %main(p0: s32[2,4]) -> s32[2,4] {
  %p0 = s32[2,4]{1,0} parameter(0)
  ROOT %add.3 = s32[2,4]{1,0} add(%p0, %p0)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto original_module,
                       ParseAndReturnVerifiedModule(original_hlo_string));
  ASSERT_OK_AND_ASSIGN(
      auto calculator_with_metrics,
      OriginalTensorSummaryCalculator::Create(
          optimized_module.get(), original_module.get(), std::move(callback_)));
  auto& calculator = calculator_with_metrics.first;
  EXPECT_EQ(calculator->DumpHloDerivedData(),
            R"(Optimized Tensor Dimensions:
  add.1: [1, 1]

Call Map:

Original Tensor by Optimized Tensor Key:
  add.1:
    add.3 via Unshard{dimensions=[2, 4], sharding={devices=[2,4]<=[8]}, continuation=nullptr}
)");
}

TEST_F(OriginalTensorSummaryCalculatorCreateTest, CallMap) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule call_map_test, entry_computation_layout={(s32[1]{0})->s32[1]{0}}

%called_computation (p: s32[1]) -> s32[1] {
  %p = s32[1]{0} parameter(0)
  ROOT %add = s32[1]{0} add(%p, %p), origin={{"inner_scope/add"}}
}

ENTRY %main (param: s32[1]) -> s32[1] {
  %param = s32[1]{0} parameter(0)
  ROOT %call = s32[1]{0} call(%param), to_apply=%called_computation, origin={{"outer_scope/call"}}
}
)hlo";
  constexpr absl::string_view original_hlo_string = R"hlo(
HloModule call_map_test_orig, entry_computation_layout={(s32[1]{0})->s32[1]{0}}

%inner_scope_computation (p: s32[1]) -> s32[1] {
  %p = s32[1]{0} parameter(0)
  ROOT %add = s32[1]{0} add(%p, %p)
}

%call_computation (p: s32[1]) -> s32[1] {
  %p = s32[1]{0} parameter(0)
  ROOT %inner_scope = s32[1]{0} call(%p), to_apply=%inner_scope_computation
}

%outer_scope_computation (p: s32[1]) -> s32[1] {
  %p = s32[1]{0} parameter(0)
  ROOT %call = s32[1]{0} call(%p), to_apply=%call_computation
}

ENTRY %main (param: s32[1]) -> s32[1] {
  %param = s32[1]{0} parameter(0)
  ROOT %outer_scope = s32[1]{0} call(%param), to_apply=%outer_scope_computation
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto original_module,
                       ParseAndReturnVerifiedModule(original_hlo_string));
  ASSERT_OK_AND_ASSIGN(
      auto calculator_with_metrics,
      OriginalTensorSummaryCalculator::Create(
          optimized_module.get(), original_module.get(), std::move(callback_)));
  auto& calculator = calculator_with_metrics.first;
  EXPECT_EQ(calculator->DumpHloDerivedData(),
            R"(Optimized Tensor Dimensions:
  add: [1]
  call: [1]

Call Map:
  call: [outer_scope/call]

Original Tensor by Optimized Tensor Key:
  add:
    inner_scope/add via no transformation
  call:
    outer_scope/call via no transformation
)");

  const auto& creation_metrics = calculator_with_metrics.second;
  EXPECT_EQ(creation_metrics.optimized_module_tensor_count, 4);
  EXPECT_EQ(creation_metrics.optimized_module_tensor_with_original_array_count,
            2);
  EXPECT_EQ(creation_metrics.optimized_module_call_like_instr_count, 1);
  EXPECT_EQ(creation_metrics
                .optimized_module_call_like_instr_with_original_value_count,
            1);
  EXPECT_EQ(creation_metrics.original_module_recoverable_tensor_count, 2);
  EXPECT_THAT(creation_metrics.recoverable_tensor_keys,
              UnorderedElementsAre(TensorKey::Create("add"),
                                   TensorKey::Create("call")));

  AbsoluteScopedTensorKey opt_key{
      /*scope_instructions=*/{ScopeInstruction::Create("call")},
      /*tensor_key=*/TensorKey{/*instruction_name=*/"add"},
  };
  std::vector<DimSplitSpec> shard_split_spec = {};
  ASSERT_THAT(calculator->ProcessShardSummary(
                  opt_key, {0, CreateSummary({1.0f}, shard_split_spec)}),
              IsOk());
  ASSERT_THAT(results_, SizeIs(1));
  EXPECT_THAT(results_[0].original_tensor_key.scope_instructions,
              Eq(std::vector<ScopeInstruction>{
                  ScopeInstruction::Create("outer_scope"),
                  ScopeInstruction::Create("call"),
                  ScopeInstruction::Create("inner_scope")}));
  EXPECT_THAT(results_[0].original_tensor_key.tensor_key.instruction_name,
              Eq("add"));
  EXPECT_THAT(results_[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{1}));
  auto processing_metrics = calculator->GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics.completed_original_tensor_count, 1);
  EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 0);
}

TEST_F(OriginalTensorSummaryCalculatorCreateTest, CallMapReconciliation) {
  constexpr absl::string_view optimized_hlo = R"hlo(
HloModule optimized_module
%opt_comp(p: s32[]) -> s32[] {
  p0 = s32[] parameter(0)
  ROOT add_opt = s32[] add(p0, p0), origin={{"add"}}
}
ENTRY %main (p: s32[]) -> s32[] {
  p1 = s32[] parameter(0)
  ROOT call_opt = s32[] call(p1), to_apply=%opt_comp
}
)hlo";
  constexpr absl::string_view original_hlo = R"hlo(
HloModule original_module
%add_comp(p: s32[]) -> s32[] {
  p0 = s32[] parameter(0)
  ROOT add = s32[] add(p0, p0)
}
%call2_comp(p: s32[]) -> s32[] {
  p1 = s32[] parameter(0)
  ROOT call2 = s32[] call(p1), to_apply=%add_comp
}
ENTRY %main (p: s32[]) -> s32[] {
  p2 = s32[] parameter(0)
  ROOT call1 = s32[] call(p2), to_apply=%call2_comp
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       ParseAndReturnVerifiedModule(optimized_hlo));
  ASSERT_OK_AND_ASSIGN(auto original_module,
                       ParseAndReturnVerifiedModule(original_hlo));
  ASSERT_OK_AND_ASSIGN(
      auto calculator_with_metrics,
      OriginalTensorSummaryCalculator::Create(
          optimized_module.get(), original_module.get(), std::move(callback_)));
  auto& calculator = calculator_with_metrics.first;
  EXPECT_EQ(calculator->DumpHloDerivedData(),
            R"(Optimized Tensor Dimensions:
  add_opt: []

Call Map:
  call_opt: [call1/call2]

Original Tensor by Optimized Tensor Key:
  add_opt:
    add via no transformation
)");
}

TEST_F(OriginalTensorSummaryCalculatorCreateTest,
       CallMapReconciliationFailure) {
  constexpr absl::string_view optimized_hlo = R"hlo(
HloModule optimized_module

%add_comp(p: s32[]) -> s32[] {
  %p_add = s32[] parameter(0)
  ROOT %add = s32[] add(%p_add, %p_add), origin={{"orig_add"}}
}

%call_comp(p: s32[]) -> s32[] {
  %p_call = s32[] parameter(0)
  ROOT %call_opt = s32[] call(%p_call), to_apply=%add_comp
}

ENTRY %main(p: s32[]) -> s32[] {
  %p_main = s32[] parameter(0)
  %caller1 = s32[] call(%p_main), to_apply=%call_comp
  ROOT %caller2 = s32[] call(%p_main), to_apply=%call_comp
}
)hlo";
  constexpr absl::string_view original_hlo = R"hlo(
HloModule original_module

%orig_add_comp(p: s32[]) -> s32[] {
  %p_add = s32[] parameter(0)
  ROOT %orig_add = s32[] add(%p_add, %p_add)
}

%orig_call_comp(p: s32[]) -> s32[] {
  %p_call = s32[] parameter(0)
  ROOT %orig_call = s32[] call(%p_call), to_apply=%orig_add_comp
}

ENTRY %orig_main(p: s32[]) -> s32[] {
  %p_main = s32[] parameter(0)
  ROOT %entry_call = s32[] call(%p_main), to_apply=%orig_call_comp
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       ParseAndReturnVerifiedModule(optimized_hlo));
  ASSERT_OK_AND_ASSIGN(auto original_module,
                       ParseAndReturnVerifiedModule(original_hlo));
  ASSERT_OK_AND_ASSIGN(
      auto calculator_with_metrics,
      OriginalTensorSummaryCalculator::Create(
          optimized_module.get(), original_module.get(), std::move(callback_)));
  auto& calculator = calculator_with_metrics.first;
  EXPECT_EQ(calculator->DumpHloDerivedData(),
            R"(Optimized Tensor Dimensions:
  add: []

Call Map:
  caller1: [entry_call]
  caller2: [entry_call]

Original Tensor by Optimized Tensor Key:
  add:
    orig_add via no transformation
)");
}

TEST_F(OriginalTensorSummaryCalculatorCreateTest, CallMapGuessMultipleCallers) {
  constexpr absl::string_view optimized_hlo = R"hlo(
HloModule optimized_module
%add_opt_comp(p: s32[]) -> s32[] {
  p0 = s32[] parameter(0)
  ROOT add_opt = s32[] add(p0, p0), origin={{"add"}}
}
%call2_opt_comp(p:s32[]) -> s32[] {
  p_sub = s32[] parameter(0)
  ROOT call2_opt = s32[] call(p_sub), to_apply=%add_opt_comp
}
ENTRY %main (p: s32[]) -> s32[] {
  p1 = s32[] parameter(0)
  ROOT call1_opt = s32[] call(p1), to_apply=%call2_opt_comp
}
)hlo";
  constexpr absl::string_view original_hlo = R"hlo(
HloModule original_module
%add_comp(p: s32[]) -> s32[] {
  p0 = s32[] parameter(0)
  ROOT add = s32[] add(p0, p0)
}
%call2_comp(p: s32[]) -> s32[] {
  p1 = s32[] parameter(0)
  ROOT call2 = s32[] call(p1), to_apply=%add_comp
}
ENTRY %main (p: s32[]) -> s32[] {
  p2 = s32[] parameter(0)
  ROOT call1 = s32[] call(p2), to_apply=%call2_comp
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       ParseAndReturnVerifiedModule(optimized_hlo));
  ASSERT_OK_AND_ASSIGN(auto original_module,
                       ParseAndReturnVerifiedModule(original_hlo));
  ASSERT_OK_AND_ASSIGN(
      auto calculator_with_metrics,
      OriginalTensorSummaryCalculator::Create(
          optimized_module.get(), original_module.get(), std::move(callback_)));
  auto& calculator = calculator_with_metrics.first;
  EXPECT_EQ(calculator->DumpHloDerivedData(),
            R"(Optimized Tensor Dimensions:
  add_opt: []

Call Map:
  call1_opt: [call1]
  call2_opt: [call2]

Original Tensor by Optimized Tensor Key:
  add_opt:
    add via no transformation
)");
}

TEST_F(OriginalTensorSummaryCalculatorCreateTest,
       ScopedInstructionWithIterationIndex) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule iteration_index_test

ENTRY %main (p: s32[]) -> s32[] {
  %p = s32[] parameter(0)
  ROOT %add = s32[] add(%p, %p), origin={{"while.1#6/while.2#*/add"}}
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto calculator_with_metrics,
                       OriginalTensorSummaryCalculator::Create(
                           module.get(), module.get(), std::move(callback_)));
  auto& calculator = calculator_with_metrics.first;
  EXPECT_EQ(calculator->DumpHloDerivedData(),
            R"(Optimized Tensor Dimensions:
  add: []

Call Map:

Original Tensor by Optimized Tensor Key:
  add:
    while.1#6/while.2#*/add via no transformation
)");

  const auto& creation_metrics = calculator_with_metrics.second;
  EXPECT_EQ(creation_metrics.optimized_module_tensor_count, 2);
  EXPECT_EQ(creation_metrics.optimized_module_tensor_with_original_array_count,
            1);
  EXPECT_EQ(creation_metrics.original_module_recoverable_tensor_count, 1);
  EXPECT_THAT(creation_metrics.recoverable_tensor_keys,
              UnorderedElementsAre(TensorKey::Create("add")));

  AbsoluteScopedTensorKey opt_key{
      /*tensor_key=*/TensorKey{/*instruction_name=*/"add"}};
  std::vector<DimSplitSpec> shard_split_spec = {};
  ASSERT_THAT(calculator->ProcessShardSummary(
                  opt_key, {0, CreateSummary({1.0f}, shard_split_spec)}),
              IsOk());
  ASSERT_THAT(results_, SizeIs(1));
  EXPECT_THAT(results_[0].original_tensor_key.scope_instructions,
              Eq(std::vector<ScopeInstruction>{
                  ScopeInstruction::Create("while.1", 6),
                  ScopeInstruction::Create("while.2", -1)}));
  EXPECT_THAT(results_[0].original_tensor_key.tensor_key.instruction_name,
              Eq("add"));
  EXPECT_THAT(results_[0].original_tensor_summary.dimensions, IsEmpty());
  auto processing_metrics = calculator->GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics.completed_original_tensor_count, 1);
  EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 0);
}

TEST_F(OriginalTensorSummaryCalculatorCreateTest,
       BuildTransformationChainWithScopedGoal) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule scoped_key_test, entry_computation_layout={(s32[4,8]{1,0})->s32[4,8]{1,0}}, origin_recovery_table={
  {"call.1/orig"} : {"orig__ovp0"},
  "
    ENTRY %reshape_computation (param: s32[4,8]) -> s32[8,4] {
      %p1 = s32[4,8]{1,0} parameter(0)
      ROOT %reshape = s32[8,4]{1,0} reshape(%p1)
    }
  "
}
ENTRY %main (p: s32[4,8]) -> s32[4,8] {
  %p = s32[4,8]{1,0} parameter(0)
  ROOT %opt = s32[4,8]{1,0} add(%p, %p), origin={{"orig__ovp0"}}
}
)hlo";
  constexpr absl::string_view original_hlo_string = R"hlo(
HloModule orig_mod
ENTRY %main(p:s32[8,4]) -> s32[8,4] {
  %p = s32[8,4]{1,0} parameter(0)
  ROOT %orig = s32[8,4]{1,0} add(%p, %p)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto original_module,
                       ParseAndReturnVerifiedModule(original_hlo_string));
  ASSERT_OK_AND_ASSIGN(
      auto calculator_with_metrics,
      OriginalTensorSummaryCalculator::Create(
          optimized_module.get(), original_module.get(), std::move(callback_)));
  auto& calculator = calculator_with_metrics.first;
  EXPECT_EQ(calculator->DumpHloDerivedData(),
            R"(Optimized Tensor Dimensions:
  opt: [4, 8]

Call Map:

Original Tensor by Optimized Tensor Key:
  opt:
    call.1/orig via Reshape{dimensions=[8, 4], continuation=nullptr}
)");

  AbsoluteScopedTensorKey opt_key{
      /*tensor_key=*/TensorKey{/*instruction_name=*/"opt"}};
  std::vector<DimSplitSpec> shard_split_spec = {};
  ASSERT_THAT(calculator->ProcessShardSummary(
                  opt_key, {0, CreateSummary({1.0f}, shard_split_spec)}),
              IsOk());
  ASSERT_THAT(results_, SizeIs(1));
  EXPECT_THAT(
      results_[0].original_tensor_key.scope_instructions,
      Eq(std::vector<ScopeInstruction>{ScopeInstruction::Create("call.1")}));
  EXPECT_THAT(results_[0].original_tensor_key.tensor_key.instruction_name,
              Eq("orig"));
  auto expected_reshape = std::make_shared<const TensorTransformation>(
      Reshape{/*output_dimensions=*/{8, 4}});
  EXPECT_THAT(results_[0].pending_transformation, Pointee(*expected_reshape));
  EXPECT_THAT(results_[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{4, 8}));
}

TEST_F(OriginalTensorSummaryCalculatorCreateTest,
       BuildTransformationChainWithIdentity) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule identity_test, origin_recovery_table={
  {"orig"} : {"orig__ovp0"}
}
ENTRY %main (p: s32[2,8]) -> s32[2,8] {
  %p = s32[2,8]{1,0} parameter(0)
  ROOT %opt = s32[2,8]{1,0} add(%p, %p), origin={{"orig__ovp0"}}
}
)hlo";
  constexpr absl::string_view original_hlo_string = R"hlo(
HloModule identity_test_orig
ENTRY %main(p:s32[2,8]) -> s32[2,8] {
  %p = s32[2,8]{1,0} parameter(0)
  ROOT %orig = s32[2,8]{1,0} add(%p, %p)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto original_module,
                       ParseAndReturnVerifiedModule(original_hlo_string));
  ASSERT_OK_AND_ASSIGN(
      auto calculator_with_metrics,
      OriginalTensorSummaryCalculator::Create(
          optimized_module.get(), original_module.get(), std::move(callback_)));
  auto& calculator = calculator_with_metrics.first;
  EXPECT_EQ(calculator->DumpHloDerivedData(),
            R"(Optimized Tensor Dimensions:
  opt: [2, 8]

Call Map:

Original Tensor by Optimized Tensor Key:
  opt:
    orig via no transformation
)");

  AbsoluteScopedTensorKey opt_key{
      /*tensor_key=*/TensorKey{/*instruction_name=*/"opt"}};
  std::vector<DimSplitSpec> shard_split_spec = {};
  ASSERT_THAT(calculator->ProcessShardSummary(
                  opt_key, {0, CreateSummary({1.0f}, shard_split_spec)}),
              IsOk());
  ASSERT_THAT(results_, SizeIs(1));
  EXPECT_THAT(results_[0].original_tensor_key.tensor_key.instruction_name,
              Eq("orig"));
  EXPECT_THAT(results_[0].pending_transformation, Eq(nullptr));
  EXPECT_THAT(results_[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{2, 8}));
}

TEST_F(OriginalTensorSummaryCalculatorCreateTest,
       BuildTransformationChainWithChainedIdentity) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule chained_identity_reshape, entry_computation_layout={(s32[4,8]{1,0})->s32[4,8]{1,0}}, origin_recovery_table={
  {"orig"} : {"orig__ovp1"},
  "
    ENTRY %reshape_computation (param: s32[4,8]) -> s32[8,4] {
      %p1 = s32[4,8]{1,0} parameter(0)
      ROOT %reshape = s32[8,4]{1,0} reshape(%p1)
    }
  "
  {"orig__ovp1"} : {"orig__ovp0"}
}
ENTRY %main (p: s32[4,8]) -> s32[4,8] {
  %p = s32[4,8]{1,0} parameter(0)
  ROOT %opt = s32[4,8]{1,0} add(%p, %p), origin={{"orig__ovp0"}}
}
)hlo";
  constexpr absl::string_view original_hlo_string = R"hlo(
HloModule chained_identity_reshape_orig
ENTRY %main(p:s32[8,4]) -> s32[8,4] {
  %p = s32[8,4]{1,0} parameter(0)
  ROOT %orig = s32[8,4]{1,0} add(%p, %p)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto original_module,
                       ParseAndReturnVerifiedModule(original_hlo_string));
  ASSERT_OK_AND_ASSIGN(
      auto calculator_with_metrics,
      OriginalTensorSummaryCalculator::Create(
          optimized_module.get(), original_module.get(), std::move(callback_)));
  auto& calculator = calculator_with_metrics.first;
  EXPECT_EQ(calculator->DumpHloDerivedData(),
            R"(Optimized Tensor Dimensions:
  opt: [4, 8]

Call Map:

Original Tensor by Optimized Tensor Key:
  opt:
    orig via Reshape{dimensions=[8, 4], continuation=nullptr}
)");

  AbsoluteScopedTensorKey opt_key{
      /*tensor_key=*/TensorKey{/*instruction_name=*/"opt"}};
  std::vector<DimSplitSpec> shard_split_spec = {};
  ASSERT_THAT(calculator->ProcessShardSummary(
                  opt_key, {0, CreateSummary({1.0f}, shard_split_spec)}),
              IsOk());
  ASSERT_THAT(results_, SizeIs(1));
  EXPECT_THAT(results_[0].original_tensor_key.tensor_key.instruction_name,
              Eq("orig"));
  auto expected_reshape = std::make_shared<const TensorTransformation>(
      Reshape{/*continuation=*/nullptr, /*output_dimensions=*/{8, 4}});
  EXPECT_THAT(results_[0].pending_transformation, Pointee(*expected_reshape));
  EXPECT_THAT(results_[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{4, 8}));
}

TEST_F(OriginalTensorSummaryCalculatorCreateTest, CyclicRecoveryTable) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule cyclic_recovery, origin_recovery_table={
  {"a__ovp0"} : {"a__ovp1"}
  {"a__ovp1"} : {"a__ovp0"}
}
ENTRY %main (p: s32[1]) -> s32[1] {
  %p = s32[1]{0} parameter(0)
  ROOT %a = s32[1]{0} add(%p, %p), origin={{"a__ovp0"}}
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto calculator_with_metrics,
                       OriginalTensorSummaryCalculator::Create(
                           module.get(), module.get(), std::move(callback_)));
  auto& calculator = calculator_with_metrics.first;
  EXPECT_EQ(calculator->DumpHloDerivedData(),
            R"(Optimized Tensor Dimensions:
  a: [1]

Call Map:

Original Tensor by Optimized Tensor Key:
)");
}

TEST_F(OriginalTensorSummaryCalculatorCreateTest, ShardingWithReplication) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule rep_sharding, entry_computation_layout={(s32[2,8]{1,0})->s32[2,8]{1,0}}, origin_recovery_table={
  {"orig"} : {"orig__ovp0"},
  "
    ENTRY %recovery_computation (param: s32[2,8]) -> s32[2,16] {
      %param = s32[2,8]{1,0} parameter(0), sharding={devices=[1,2]<=[2]}
      ROOT %all-gather = s32[2,16]{1,0} all-gather(%param), dimensions={1}, sharding={replicated}
    }
  "
}
ENTRY %main (p: s32[2,8]) -> s32[2,8] {
  %p = s32[2,8]{1,0} parameter(0), sharding={devices=[1,2]<=[2]}
  ROOT %add = s32[2,8]{1,0} add(%p, %p), origin={{"orig__ovp0"}}
}
)hlo";
  constexpr absl::string_view original_hlo_string = R"hlo(
HloModule rep_sharding_orig
ENTRY %main(p:s32[2,16]) -> s32[2,16] {
  %p = s32[2,16]{1,0} parameter(0)
  ROOT %orig = s32[2,16]{1,0} add(%p, %p)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto original_module,
                       ParseAndReturnVerifiedModule(original_hlo_string));
  ASSERT_OK_AND_ASSIGN(
      auto calculator_with_metrics,
      OriginalTensorSummaryCalculator::Create(
          optimized_module.get(), original_module.get(), std::move(callback_)));
  auto& calculator = calculator_with_metrics.first;
  EXPECT_EQ(calculator->DumpHloDerivedData(),
            R"(Optimized Tensor Dimensions:
  add: [2, 8]

Call Map:

Original Tensor by Optimized Tensor Key:
  add:
    orig via Unshard{dimensions=[2, 16], sharding={devices=[1,2]<=[2]}, continuation=nullptr}
)");
  const auto& creation_metrics = calculator_with_metrics.second;
  EXPECT_EQ(creation_metrics.optimized_module_tensor_count, 2);
  EXPECT_EQ(creation_metrics.optimized_module_tensor_with_original_array_count,
            1);
  EXPECT_EQ(creation_metrics.original_module_recoverable_tensor_count, 1);
  EXPECT_THAT(creation_metrics.recoverable_tensor_keys,
              UnorderedElementsAre(TensorKey::Create("orig")));

  AbsoluteScopedTensorKey opt_key{
      /*tensor_key=*/TensorKey{/*instruction_name=*/"add"}};
  std::vector<DimSplitSpec> shard_split_spec = {
      {/*dim_index=*/0, /*block_count=*/2}};
  ASSERT_THAT(calculator->ProcessShardSummary(
                  opt_key, {0, CreateSummary({1.0f, 2.0f}, shard_split_spec)}),
              IsOk());
  ASSERT_THAT(calculator->ProcessShardSummary(
                  opt_key, {1, CreateSummary({3.0f, 4.0f}, shard_split_spec)}),
              IsOk());

  ASSERT_THAT(results_, SizeIs(1));
  EXPECT_THAT(results_[0].original_tensor_key.tensor_key.instruction_name,
              Eq("orig"));
  EXPECT_THAT(results_[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{2, 16}));
  const auto& summary = results_[0].original_tensor_summary.summaries[0];

  // The two shards are concatenated along the second dimension, and the split
  // spec is extended to cover the full dimension.
  std::vector<DimSplitSpec> combined_split_spec = {
      {/*dim_index=*/0, /*block_count=*/2},
      {/*dim_index=*/1, /*block_count=*/2}};
  EXPECT_THAT(summary.split_spec, Eq(combined_split_spec));
  ASSERT_THAT(summary.block_summaries, SizeIs(4));
  EXPECT_THAT(summary.block_summaries[0].min, Eq(1.0f));
  EXPECT_THAT(summary.block_summaries[0].max, Eq(1.0f));
  EXPECT_THAT(summary.block_summaries[1].min, Eq(3.0f));
  EXPECT_THAT(summary.block_summaries[1].max, Eq(3.0f));
  EXPECT_THAT(summary.block_summaries[2].min, Eq(2.0f));
  EXPECT_THAT(summary.block_summaries[2].max, Eq(2.0f));
  EXPECT_THAT(summary.block_summaries[3].min, Eq(4.0f));
  EXPECT_THAT(summary.block_summaries[3].max, Eq(4.0f));
  auto processing_metrics = calculator->GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count, 2);
  EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count, 2);
  EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics.completed_original_tensor_count, 1);
  EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 0);
}

TEST_F(OriginalTensorSummaryCalculatorCreateTest, ManualSharding) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule manual_sharding, entry_computation_layout={(f32[2,4]{1,0})->f32[2,4]{1,0}}, origin_recovery_table={
  {"orig"} : {"orig__ovp0"},
  "
    ENTRY %recovery_computation (param: f32[2,4]) -> f32[4,4] {
      %param = f32[2,4]{1,0} parameter(0), sharding={devices=[2,1,2,2]<=[2,2,2]T(1,0,2) last_tile_dims={manual,replicated}}
      ROOT %all-gather = f32[4,4]{1,0} all-gather(%param), dimensions={0}, sharding={replicated}
    }
  "
}

ENTRY %main (p: f32[2,4]) -> f32[2,4] {
  %p = f32[2,4]{1,0} parameter(0), sharding={devices=[2,1,2,2]<=[2,2,2]T(1,0,2) last_tile_dims={manual,replicated}}
  ROOT %add = f32[2,4]{1,0} add(%p, %p), origin={{"orig__ovp0"}}
}
)hlo";
  constexpr absl::string_view original_hlo_string = R"hlo(
HloModule manual_sharding_orig
ENTRY %main(p:f32[4,4]) -> f32[4,4] {
  %p = f32[4,4]{1,0} parameter(0)
  ROOT %orig = f32[4,4]{1,0} add(%p, %p)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto original_module,
                       ParseAndReturnVerifiedModule(original_hlo_string));
  ASSERT_OK_AND_ASSIGN(
      auto calculator_with_metrics,
      OriginalTensorSummaryCalculator::Create(
          optimized_module.get(), original_module.get(), std::move(callback_)));
  auto& calculator = calculator_with_metrics.first;
  EXPECT_EQ(calculator->DumpHloDerivedData(),
            R"(Optimized Tensor Dimensions:
  add: [2, 4]

Call Map:

Original Tensor by Optimized Tensor Key:
  add:
    orig via Unshard{dimensions=[4, 4], sharding={devices=[2,1,2,2]<=[2,2,2]T(1,0,2) last_tile_dims={manual, replicated}}, continuation=nullptr}
)");
  const auto& creation_metrics = calculator_with_metrics.second;
  EXPECT_EQ(creation_metrics.optimized_module_tensor_count, 2);
  EXPECT_EQ(creation_metrics.optimized_module_tensor_with_original_array_count,
            1);
  EXPECT_EQ(creation_metrics.original_module_recoverable_tensor_count, 1);
  EXPECT_THAT(creation_metrics.recoverable_tensor_keys,
              UnorderedElementsAre(TensorKey::Create("orig")));

  AbsoluteScopedTensorKey opt_key{
      /*tensor_key=*/TensorKey{/*instruction_name=*/"add"}};
  // Devices 0,1,2,3,4,5,6,7 correspond to assignment 0,1,4,5,2,3,6,7.
  // Sharding has replication dim 3 and manual dim 2.
  // Shards with replication index 0 are on devices where tile_index[3]=0:
  // 0->[0,0,0,0], 2->[1,0,0,0], 4->[0,0,1,0], 6->[1,0,1,0].
  // These are devices 0, 2, 4, 6.
  ASSERT_THAT(
      calculator->ProcessShardSummary(opt_key, {0, CreateSummary({1.0f}, {})}),
      IsOk());
  ASSERT_THAT(
      calculator->ProcessShardSummary(opt_key, {2, CreateSummary({2.0f}, {})}),
      IsOk());
  ASSERT_THAT(
      calculator->ProcessShardSummary(opt_key, {4, CreateSummary({3.0f}, {})}),
      IsOk());
  ASSERT_THAT(
      calculator->ProcessShardSummary(opt_key, {6, CreateSummary({4.0f}, {})}),
      IsOk());

  ASSERT_THAT(results_, SizeIs(1));
  EXPECT_THAT(results_[0].original_tensor_key.tensor_key.instruction_name,
              Eq("orig"));
  EXPECT_THAT(results_[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{4, 4}));
  // One summary per manual group. There are 2 manual groups.
  ASSERT_THAT(results_[0].original_tensor_summary.summaries, SizeIs(2));

  // Manual group 0 contains data from devices 0 and 2.
  const auto& summary0 = results_[0].original_tensor_summary.summaries[0];
  EXPECT_THAT(
      summary0.split_spec,
      Eq(std::vector<DimSplitSpec>{{/*dim_index=*/0, /*block_count=*/2}}));
  ASSERT_THAT(summary0.block_summaries, SizeIs(2));
  EXPECT_THAT(summary0.block_summaries[0].min, Eq(1.0f));
  EXPECT_THAT(summary0.block_summaries[1].min, Eq(2.0f));

  // Manual group 1 contains data from devices 4 and 6.
  const auto& summary1 = results_[0].original_tensor_summary.summaries[1];
  EXPECT_THAT(
      summary1.split_spec,
      Eq(std::vector<DimSplitSpec>{{/*dim_index=*/0, /*block_count=*/2}}));
  ASSERT_THAT(summary1.block_summaries, SizeIs(2));
  EXPECT_THAT(summary1.block_summaries[0].min, Eq(3.0f));
  EXPECT_THAT(summary1.block_summaries[1].min, Eq(4.0f));

  auto processing_metrics = calculator->GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count, 4);
  EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count, 4);
  EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics.completed_original_tensor_count, 1);
  EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 0);
}

TEST_F(OriginalTensorSummaryCalculatorCreateTest, ChainedUnshardAndReshape) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule chained_unshard_reshape, entry_computation_layout={(s32[2,8]{1,0})->s32[2,8]{1,0}}, origin_recovery_table={
  {"orig"} : {"orig__ovp1"},
  "
    ENTRY %reshape_computation (param: s32[4,8]) -> s32[8,4] {
      %p1 = s32[4,8]{1,0} parameter(0)
      ROOT %reshape = s32[8,4]{1,0} reshape(%p1)
    }
  "
  {"orig__ovp1"} : {"orig__ovp0"},
  "
    ENTRY %unshard_computation (param: s32[2,8]) -> s32[4,8] {
      %p2 = s32[2,8]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}
      ROOT %all-gather = s32[4,8]{1,0} all-gather(%p2), dimensions={0}, sharding={replicated}
    }
  "
}
ENTRY %main (p: s32[2,8]) -> s32[2,8] {
  %p = s32[2,8]{1,0} parameter(0)
  ROOT %opt = s32[2,8]{1,0} add(%p, %p), origin={{"orig__ovp0"}}
}
)hlo";
  constexpr absl::string_view original_hlo_string = R"hlo(
HloModule chained_orig
ENTRY %main(p:s32[8,4]) -> s32[8,4] {
  %p = s32[8,4]{1,0} parameter(0)
  ROOT %orig = s32[8,4]{1,0} add(%p, %p)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto original_module,
                       ParseAndReturnVerifiedModule(original_hlo_string));
  ASSERT_OK_AND_ASSIGN(
      auto calculator_with_metrics,
      OriginalTensorSummaryCalculator::Create(
          optimized_module.get(), original_module.get(), std::move(callback_)));
  auto& calculator = calculator_with_metrics.first;
  EXPECT_EQ(calculator->DumpHloDerivedData(), R"(Optimized Tensor Dimensions:
  opt: [2, 8]

Call Map:

Original Tensor by Optimized Tensor Key:
  opt:
    orig via Unshard{dimensions=[4, 8], sharding={devices=[2,1]<=[2]}, continuation=Reshape{dimensions=[8, 4], continuation=nullptr}}
)");
  const auto& creation_metrics = calculator_with_metrics.second;
  EXPECT_EQ(creation_metrics.optimized_module_tensor_count, 2);
  EXPECT_EQ(creation_metrics.optimized_module_tensor_with_original_array_count,
            1);
  EXPECT_EQ(creation_metrics.original_module_recoverable_tensor_count, 1);
  EXPECT_THAT(creation_metrics.recoverable_tensor_keys,
              UnorderedElementsAre(TensorKey::Create("orig")));

  AbsoluteScopedTensorKey opt_key{
      /*tensor_key=*/TensorKey{/*instruction_name=*/"opt"}};
  std::vector<DimSplitSpec> shard_split_spec = {};
  ASSERT_THAT(calculator->ProcessShardSummary(
                  opt_key, {0, CreateSummary({1.0f}, shard_split_spec)}),
              IsOk());
  ASSERT_THAT(calculator->ProcessShardSummary(
                  opt_key, {1, CreateSummary({2.0f}, shard_split_spec)}),
              IsOk());

  ASSERT_THAT(results_, SizeIs(1));
  EXPECT_THAT(results_[0].original_tensor_key.tensor_key.instruction_name,
              Eq("orig"));
  // We have unshard from [2,8] sharded to [4,8] replicated, then reshape to
  // [8,4].
  // The unshard is applied, and reshape is pending.
  auto expected_reshape = std::make_shared<const TensorTransformation>(
      Reshape{/*continuation=*/nullptr, /*output_dimensions=*/{8, 4}});
  EXPECT_THAT(results_[0].pending_transformation, Pointee(*expected_reshape));
  EXPECT_THAT(results_[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{4, 8}));
  const auto& summary = results_[0].original_tensor_summary.summaries[0];
  EXPECT_THAT(summary.split_spec, SizeIs(1));
  ASSERT_THAT(summary.block_summaries, SizeIs(2));
  EXPECT_THAT(summary.block_summaries[0].min, Eq(1.0f));
  EXPECT_THAT(summary.block_summaries[0].max, Eq(1.0f));
  EXPECT_THAT(summary.block_summaries[1].min, Eq(2.0f));
  EXPECT_THAT(summary.block_summaries[1].max, Eq(2.0f));
  auto processing_metrics = calculator->GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count, 2);
  EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count, 2);
  EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics.completed_original_tensor_count, 1);
  EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 0);
}

TEST_F(OriginalTensorSummaryCalculatorCreateTest,
       WhileLoopMappingByBodySucceedsWithOptimizedFusion) {
  constexpr absl::string_view optimized_hlo = R"hlo(
HloModule optimized_module

%fusion_comp (p: (s32[])) -> s32[] {
  %p = (s32[]) parameter(0)
  %gep0 = s32[] get-tuple-element(%p), index=0
  ROOT %add_opt = s32[] add(%gep0, %gep0), origin={{"add.243"}}
}

%opt_cond (p: (s32[])) -> pred[] {
  %p = (s32[]) parameter(0)
  ROOT %c = pred[] constant(true)
}

%opt_body (p: (s32[])) -> (s32[]) {
  %p = (s32[]) parameter(0)
  %fusion = s32[] fusion(%p), kind=kLoop, calls=%fusion_comp
  ROOT %tuple = (s32[]) tuple(%fusion)
}

ENTRY %main (p: (s32[])) -> (s32[]) {
  %p = (s32[]) parameter(0)
  ROOT %while.678 = (s32[]) while(%p), condition=%opt_cond, body=%opt_body
}
)hlo";

  constexpr absl::string_view original_hlo = R"hlo(
HloModule original_module

%cond (p: (s32[])) -> pred[] {
  %p = (s32[]) parameter(0)
  ROOT %c = pred[] constant(true)
}

%body (p: (s32[])) -> (s32[]) {
  %p = (s32[]) parameter(0)
  %gep0 = s32[] get-tuple-element(%p), index=0
  %add.243 = s32[] add(%gep0, %gep0)
  ROOT %tuple = (s32[]) tuple(%add.243)
}

ENTRY %main (p: (s32[])) -> (s32[]) {
  %p = (s32[]) parameter(0)
  ROOT %while.13 = (s32[]) while(%p), condition=%cond, body=%body
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       ParseAndReturnVerifiedModule(optimized_hlo));
  ASSERT_OK_AND_ASSIGN(auto original_module,
                       ParseAndReturnVerifiedModule(original_hlo));
  ASSERT_OK_AND_ASSIGN(
      auto calculator_with_metrics,
      OriginalTensorSummaryCalculator::Create(
          optimized_module.get(), original_module.get(), std::move(callback_)));
  auto& calculator = calculator_with_metrics.first;

  // The mapping for while.678 is successfully recovered by traversing through
  // the fusion.
  EXPECT_EQ(calculator->DumpHloDerivedData(),
            R"(Optimized Tensor Dimensions:
  add_opt: []

Call Map:
  while.678: [while.13]

Original Tensor by Optimized Tensor Key:
  add_opt:
    add.243 via no transformation
)");
}

TEST_F(OriginalTensorSummaryCalculatorCreateTest, NestedWhileLoopsMapping) {
  constexpr absl::string_view optimized_hlo = R"hlo(
HloModule optimized_module

%opt_cond (p: (s32[])) -> pred[] {
  %p = (s32[]) parameter(0)
  ROOT %c = pred[] constant(true)
}

%opt_inner_body (p: (s32[])) -> (s32[]) {
  %p = (s32[]) parameter(0)
  %gep0 = s32[] get-tuple-element(%p), index=0
  %add_opt = s32[] add(%gep0, %gep0), origin={{"add.243"}}
  ROOT %tuple = (s32[]) tuple(%add_opt)
}

%opt_outer_body (p: (s32[])) -> (s32[]) {
  %p = (s32[]) parameter(0)
  %gep0 = s32[] get-tuple-element(%p), index=0
  %inner_p = (s32[]) tuple(%gep0)
  ROOT %inner_while = (s32[]) while(%inner_p), condition=%opt_cond, body=%opt_inner_body
}

ENTRY %main (p: (s32[])) -> (s32[]) {
  %p = (s32[]) parameter(0)
  ROOT %outer_while = (s32[]) while(%p), condition=%opt_cond, body=%opt_outer_body
}
)hlo";

  constexpr absl::string_view original_hlo = R"hlo(
HloModule original_module

%cond (p: (s32[])) -> pred[] {
  %p = (s32[]) parameter(0)
  ROOT %c = pred[] constant(true)
}

%inner_body (p: (s32[])) -> (s32[]) {
  %p = (s32[]) parameter(0)
  %gep0 = s32[] get-tuple-element(%p), index=0
  %add.243 = s32[] add(%gep0, %gep0)
  ROOT %tuple = (s32[]) tuple(%add.243)
}

%outer_body (p: (s32[])) -> (s32[]) {
  %p = (s32[]) parameter(0)
  %gep0 = s32[] get-tuple-element(%p), index=0
  %inner_p = (s32[]) tuple(%gep0)
  ROOT %inner_while = (s32[]) while(%inner_p), condition=%cond, body=%inner_body
}

ENTRY %main (p: (s32[])) -> (s32[]) {
  %p = (s32[]) parameter(0)
  ROOT %outer_while = (s32[]) while(%p), condition=%cond, body=%outer_body
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       ParseAndReturnVerifiedModule(optimized_hlo));
  ASSERT_OK_AND_ASSIGN(auto original_module,
                       ParseAndReturnVerifiedModule(original_hlo));
  ASSERT_OK_AND_ASSIGN(
      auto calculator_with_metrics,
      OriginalTensorSummaryCalculator::Create(
          optimized_module.get(), original_module.get(), std::move(callback_)));
  auto& calculator = calculator_with_metrics.first;

  EXPECT_EQ(calculator->DumpHloDerivedData(),
            R"(Optimized Tensor Dimensions:
  add_opt: []

Call Map:
  inner_while: [inner_while]
  outer_while: [outer_while]

Original Tensor by Optimized Tensor Key:
  add_opt:
    add.243 via no transformation
)");
}

TEST_F(OriginalTensorSummaryCalculatorCreateTest, MultipleOriginalTensors) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule multi_orig, entry_computation_layout={(s32[2,8]{1,0})->s32[2,8]{1,0}}, origin_recovery_table={
  {"orig1"} : {"orig2"},
  "
    ENTRY %recovery1 (param: s32[4,8]) -> s32[2,2,8] {
      %p = s32[4,8]{1,0} parameter(0)
      ROOT %reshape = s32[2,2,8]{2,1,0} reshape(%p)
    }
  "
  {"orig2"} : {"opt__ovp1"},
  "
    ENTRY %recovery2 (param: s32[4,8]) -> s32[8,4] {
      %p = s32[4,8]{1,0} parameter(0)
      ROOT %reshape = s32[8,4]{1,0} reshape(%p)
    }
  "
  {"opt__ovp1"} : {"opt__ovp0"},
  "
    ENTRY %recovery3 (param: s32[2,8]) -> s32[4,8] {
      %p = s32[2,8]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}
      ROOT %all-gather = s32[4,8]{1,0} all-gather(%p), dimensions={0}, sharding={replicated}
    }
  "
}
ENTRY %main (p: s32[2,8]) -> s32[2,8] {
  %p = s32[2,8]{1,0} parameter(0)
  ROOT %opt = s32[2,8]{1,0} add(%p, %p), origin={{"opt__ovp0"}}
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto calculator_with_metrics,
                       OriginalTensorSummaryCalculator::Create(
                           module.get(), module.get(), std::move(callback_)));
  auto& calculator = calculator_with_metrics.first;
  EXPECT_EQ(calculator->DumpHloDerivedData(), R"(Optimized Tensor Dimensions:
  opt: [2, 8]

Call Map:

Original Tensor by Optimized Tensor Key:
  opt:
    orig2 via Unshard{dimensions=[4, 8], sharding={devices=[2,1]<=[2]}, continuation=Reshape{dimensions=[8, 4], continuation=nullptr}}
    orig1 via Unshard{dimensions=[4, 8], sharding={devices=[2,1]<=[2]}, continuation=Reshape{dimensions=[8, 4], continuation=Reshape{dimensions=[2, 2, 8], continuation=nullptr}}}
)");
  const auto& creation_metrics = calculator_with_metrics.second;
  EXPECT_EQ(creation_metrics.optimized_module_tensor_count, 2);
  EXPECT_EQ(creation_metrics.optimized_module_tensor_with_original_array_count,
            1);
  EXPECT_EQ(creation_metrics.original_module_recoverable_tensor_count, 2);
  EXPECT_THAT(creation_metrics.recoverable_tensor_keys,
              UnorderedElementsAre(TensorKey::Create("orig1"),
                                   TensorKey::Create("orig2")));

  AbsoluteScopedTensorKey opt_key{
      /*tensor_key=*/TensorKey{/*instruction_name=*/"opt"}};
  std::vector<DimSplitSpec> shard_split_spec = {};
  ASSERT_THAT(calculator->ProcessShardSummary(
                  opt_key, {0, CreateSummary({1.0f}, shard_split_spec)}),
              IsOk());
  ASSERT_THAT(calculator->ProcessShardSummary(
                  opt_key, {1, CreateSummary({2.0f}, shard_split_spec)}),
              IsOk());

  ASSERT_THAT(results_, SizeIs(2));
  std::sort(results_.begin(), results_.end(), [](const auto& a, const auto& b) {
    return a.original_tensor_key.tensor_key.instruction_name <
           b.original_tensor_key.tensor_key.instruction_name;
  });
  EXPECT_THAT(results_[0].original_tensor_key.tensor_key.instruction_name,
              Eq("orig1"));
  EXPECT_THAT(results_[1].original_tensor_key.tensor_key.instruction_name,
              Eq("orig2"));
  // For orig1: unshard to [4,8] then reshape to [8,4] then reshape to [2,2,8]
  auto expected_reshape_orig1_outer =
      std::make_shared<const TensorTransformation>(Reshape{
          /*continuation=*/std::make_shared<const TensorTransformation>(Reshape{
              /*continuation=*/nullptr, /*output_dimensions=*/{2, 2, 8}}),
          /*output_dimensions=*/{8, 4}});
  EXPECT_THAT(results_[0].pending_transformation,
              Pointee(*expected_reshape_orig1_outer));
  EXPECT_THAT(results_[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{4, 8}));

  // For orig2: unshard to [4,8] then reshape to [8,4]
  auto expected_reshape_orig2 = std::make_shared<const TensorTransformation>(
      Reshape{/*continuation=*/nullptr, /*output_dimensions=*/{8, 4}});
  EXPECT_THAT(results_[1].pending_transformation,
              Pointee(*expected_reshape_orig2));
  EXPECT_THAT(results_[1].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{4, 8}));
  auto processing_metrics = calculator->GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count, 2);
  EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count, 4);
  EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics.completed_original_tensor_count, 2);
  EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 0);
}

TEST(OriginalTensorSummaryCalculatorTest, NoTransformation) {
  const TensorKey opt_tensor_key{/*instruction_name=*/"opt"};
  absl::flat_hash_map<TensorKey, std::vector<int64_t>> opt_dims = {
      {opt_tensor_key, {2, 2}}};
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map;
  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      orig_map = {{opt_tensor_key,
                   {{/*original_scoped_tensor_key=*/ScopedTensorKey{
                         /*tensor_key=*/TensorKey{/*instruction_name=*/
                                                  "orig"}},
                     /*tensor_transformation=*/nullptr}}}};

  std::vector<CallbackResult> results;
  OriginalTensorSummaryCalculator calculator(
      std::make_shared<
          const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>(
          std::move(opt_dims)),
      std::make_shared<const absl::flat_hash_map<
          std::string, std::vector<ScopeInstruction>>>(std::move(call_map)),
      std::make_shared<const absl::flat_hash_map<
          TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>(
          std::move(orig_map)),
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const TensorTransformation> pending,
          const OriginalTensorSummary& summary) {
        results.push_back({key, std::move(pending), summary});
        return absl::OkStatus();
      });

  AbsoluteScopedTensorKey opt_key{/*tensor_key=*/opt_tensor_key};
  std::vector<DimSplitSpec> split_spec = {{/*dim_index=*/0, /*block_count=*/2}};
  FloatSummary shard_summary = CreateSummary({1.0f, 2.0f}, split_spec);
  ShardTensorSummary shard = {0, shard_summary};

  ASSERT_THAT(calculator.ProcessShardSummary(opt_key, shard), IsOk());
  ASSERT_THAT(results, SizeIs(1));
  EXPECT_THAT(results[0].original_tensor_key.tensor_key.instruction_name,
              Eq("orig"));
  EXPECT_THAT(results[0].pending_transformation, Eq(nullptr));
  EXPECT_THAT(results[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{2, 2}));

  const auto& summary = results[0].original_tensor_summary.summaries[0];
  EXPECT_THAT(summary.split_spec, Eq(split_spec));
  ASSERT_THAT(summary.block_summaries, SizeIs(2));
  EXPECT_THAT(summary.block_summaries[0].min, Eq(1.0f));
  EXPECT_THAT(summary.block_summaries[0].block_indices,
              Eq(std::vector<int64_t>{0}));
  EXPECT_THAT(summary.block_summaries[1].min, Eq(2.0f));
  EXPECT_THAT(summary.block_summaries[1].block_indices,
              Eq(std::vector<int64_t>{1}));
  auto processing_metrics = calculator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics.completed_original_tensor_count, 1);
  EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 0);
}

TEST(OriginalTensorSummaryCalculatorTest, UnshardReplicated) {
  const TensorKey opt_tensor_key{/*instruction_name=*/"opt"};
  absl::flat_hash_map<TensorKey, std::vector<int64_t>> opt_dims = {
      {opt_tensor_key, {2, 2}}};
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map;
  auto unshard = std::make_shared<TensorTransformation>(
      Unshard{/*continuation=*/nullptr,
              /*original_dimensions=*/{2, 2},
              /*sharding=*/HloSharding::Replicate()});
  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      orig_map = {{opt_tensor_key,
                   {{/*original_scoped_tensor_key=*/ScopedTensorKey{
                         /*tensor_key=*/TensorKey{/*instruction_name=*/
                                                  "orig"}},
                     /*tensor_transformation=*/unshard}}}};

  std::vector<CallbackResult> results;
  OriginalTensorSummaryCalculator calculator(
      std::make_shared<
          const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>(
          std::move(opt_dims)),
      std::make_shared<const absl::flat_hash_map<
          std::string, std::vector<ScopeInstruction>>>(std::move(call_map)),
      std::make_shared<const absl::flat_hash_map<
          TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>(
          std::move(orig_map)),
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const TensorTransformation> pending,
          const OriginalTensorSummary& summary) {
        results.push_back({key, std::move(pending), summary});
        return absl::OkStatus();
      });

  AbsoluteScopedTensorKey opt_key{/*tensor_key=*/opt_tensor_key};
  std::vector<DimSplitSpec> split_spec = {{/*dim_index=*/1, /*block_count=*/2}};
  FloatSummary shard_summary = CreateSummary({2.0f, 3.0f}, split_spec);
  ShardTensorSummary shard = {0, shard_summary};

  ASSERT_THAT(calculator.ProcessShardSummary(opt_key, shard), IsOk());
  ASSERT_THAT(results, SizeIs(1));
  EXPECT_THAT(results[0].original_tensor_key.tensor_key.instruction_name,
              Eq("orig"));
  EXPECT_THAT(results[0].pending_transformation, Eq(nullptr));
  EXPECT_THAT(results[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{2, 2}));

  const auto& summary = results[0].original_tensor_summary.summaries[0];
  EXPECT_THAT(summary.split_spec, Eq(split_spec));
  ASSERT_THAT(summary.block_summaries, SizeIs(2));
  EXPECT_THAT(summary.block_summaries[0].min, Eq(2.0f));
  EXPECT_THAT(summary.block_summaries[0].block_indices,
              Eq(std::vector<int64_t>{0}));
  EXPECT_THAT(summary.block_summaries[1].min, Eq(3.0f));
  EXPECT_THAT(summary.block_summaries[1].block_indices,
              Eq(std::vector<int64_t>{1}));
  auto processing_metrics = calculator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics.completed_original_tensor_count, 1);
  EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 0);
}

TEST(OriginalTensorSummaryCalculatorTest, UnshardTiled) {
  const TensorKey opt_tensor_key{/*instruction_name=*/"opt"};
  absl::flat_hash_map<TensorKey, std::vector<int64_t>> opt_dims = {
      {opt_tensor_key, {2, 1}}};  // Shard shape
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map;
  Array<int64_t> tile_assignment({2, 1});
  tile_assignment(0, 0) = 0;
  tile_assignment(1, 0) = 1;
  auto unshard = std::make_shared<TensorTransformation>(
      Unshard{/*continuation=*/nullptr,
              /*original_dimensions=*/{4, 1},  // Original shape
              /*sharding=*/HloSharding::Tile(tile_assignment)});
  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      orig_map = {{opt_tensor_key,
                   {{/*original_scoped_tensor_key=*/ScopedTensorKey{
                         /*tensor_key=*/TensorKey{/*instruction_name=*/
                                                  "orig"}},
                     /*tensor_transformation=*/unshard}}}};

  std::vector<CallbackResult> results;
  OriginalTensorSummaryCalculator calculator(
      std::make_shared<
          const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>(
          std::move(opt_dims)),
      std::make_shared<const absl::flat_hash_map<
          std::string, std::vector<ScopeInstruction>>>(std::move(call_map)),
      std::make_shared<const absl::flat_hash_map<
          TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>(
          std::move(orig_map)),
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const TensorTransformation> pending,
          const OriginalTensorSummary& summary) {
        results.push_back({key, std::move(pending), summary});
        return absl::OkStatus();
      });

  AbsoluteScopedTensorKey opt_key{/*tensor_key=*/opt_tensor_key};
  std::vector<DimSplitSpec> shard_split_spec = {
      {/*dim_index=*/0, /*block_count=*/2}};
  ShardTensorSummary shard0 = {0,
                               CreateSummary({1.0f, 1.1f}, shard_split_spec)};
  ShardTensorSummary shard1 = {1,
                               CreateSummary({2.0f, 2.1f}, shard_split_spec)};

  ASSERT_THAT(calculator.ProcessShardSummary(opt_key, shard0), IsOk());
  EXPECT_THAT(results, IsEmpty());  // Waiting for shard 1
  auto processing_metrics0 = calculator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics0.received_optimized_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics0.processed_original_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics0.completed_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics0.completed_original_tensor_count, 0);
  EXPECT_EQ(processing_metrics0.incomplete_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics0.incomplete_original_tensor_count, 1);

  ASSERT_THAT(calculator.ProcessShardSummary(opt_key, shard1), IsOk());
  ASSERT_THAT(results, SizeIs(1));
  EXPECT_THAT(results[0].original_tensor_key.tensor_key.instruction_name,
              Eq("orig"));
  EXPECT_THAT(results[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{4, 1}));
  const auto& summary = results[0].original_tensor_summary.summaries[0];
  EXPECT_THAT(
      summary.split_spec,
      Eq(std::vector<DimSplitSpec>{{/*dim_index=*/0, /*block_count=*/4}}));
  ASSERT_THAT(summary.block_summaries, SizeIs(4));
  EXPECT_THAT(summary.block_summaries[0].block_indices,
              Eq(std::vector<int64_t>{0}));
  EXPECT_THAT(summary.block_summaries[0].min, Eq(1.0f));
  EXPECT_THAT(summary.block_summaries[1].block_indices,
              Eq(std::vector<int64_t>{1}));
  EXPECT_THAT(summary.block_summaries[1].min, Eq(1.1f));
  EXPECT_THAT(summary.block_summaries[2].block_indices,
              Eq(std::vector<int64_t>{2}));
  EXPECT_THAT(summary.block_summaries[2].min, Eq(2.0f));
  EXPECT_THAT(summary.block_summaries[3].block_indices,
              Eq(std::vector<int64_t>{3}));
  EXPECT_THAT(summary.block_summaries[3].min, Eq(2.1f));
  auto processing_metrics1 = calculator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics1.received_optimized_tensor_shard_count, 2);
  EXPECT_EQ(processing_metrics1.processed_original_tensor_shard_count, 2);
  EXPECT_EQ(processing_metrics1.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics1.completed_original_tensor_count, 1);
  EXPECT_EQ(processing_metrics1.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics1.incomplete_original_tensor_count, 0);
}

TEST(OriginalTensorSummaryCalculatorTest, ReshapeOnly) {
  const TensorKey opt_tensor_key{/*instruction_name=*/"opt"};
  absl::flat_hash_map<TensorKey, std::vector<int64_t>> opt_dims = {
      {opt_tensor_key, {4}}};
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map;
  auto reshape = std::make_shared<TensorTransformation>(
      Reshape{/*continuation=*/nullptr, /*output_dimensions=*/{2, 2}});
  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      orig_map = {{opt_tensor_key,
                   {{/*original_scoped_tensor_key=*/ScopedTensorKey{
                         /*tensor_key=*/TensorKey{/*instruction_name=*/
                                                  "orig"}},
                     /*tensor_transformation=*/reshape}}}};

  std::vector<CallbackResult> results;
  OriginalTensorSummaryCalculator calculator(
      std::make_shared<
          const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>(
          std::move(opt_dims)),
      std::make_shared<const absl::flat_hash_map<
          std::string, std::vector<ScopeInstruction>>>(std::move(call_map)),
      std::make_shared<const absl::flat_hash_map<
          TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>(
          std::move(orig_map)),
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const TensorTransformation> pending,
          const OriginalTensorSummary& summary) {
        results.push_back({key, std::move(pending), summary});
        return absl::OkStatus();
      });

  AbsoluteScopedTensorKey opt_key{/*tensor_key=*/opt_tensor_key};
  std::vector<DimSplitSpec> split_spec = {{/*dim_index=*/0, /*block_count=*/2}};
  ShardTensorSummary shard = {0, CreateSummary({3.0f, 3.1f}, split_spec)};

  ASSERT_THAT(calculator.ProcessShardSummary(opt_key, shard), IsOk());
  ASSERT_THAT(results, SizeIs(1));
  EXPECT_THAT(results[0].original_tensor_key.tensor_key.instruction_name,
              Eq("orig"));
  EXPECT_THAT(results[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{4}));
  // With the new logic, reshape is not applied if there is no unshard.
  EXPECT_THAT(results[0].pending_transformation, Eq(reshape));
  const auto& summary = results[0].original_tensor_summary.summaries[0];
  EXPECT_THAT(summary.split_spec, Eq(split_spec));
  ASSERT_THAT(summary.block_summaries, SizeIs(2));
  EXPECT_THAT(summary.block_summaries[0].min, Eq(3.0f));
  EXPECT_THAT(summary.block_summaries[1].min, Eq(3.1f));
  auto processing_metrics = calculator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics.completed_original_tensor_count, 1);
  EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 0);
}

TEST(OriginalTensorSummaryCalculatorTest, UnshardThenReshape) {
  const TensorKey opt_tensor_key{/*instruction_name=*/"opt"};
  absl::flat_hash_map<TensorKey, std::vector<int64_t>> opt_dims = {
      {opt_tensor_key, {4}}};
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map;
  auto reshape = std::make_shared<TensorTransformation>(
      Reshape{/*continuation=*/nullptr, /*output_dimensions=*/{2, 2}});
  auto unshard = std::make_shared<TensorTransformation>(
      Unshard{/*continuation=*/reshape,
              /*original_dimensions=*/{4},
              /*sharding=*/HloSharding::Replicate()});
  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      orig_map = {{opt_tensor_key,
                   {{/*original_scoped_tensor_key=*/ScopedTensorKey{
                         /*tensor_key=*/TensorKey{/*instruction_name=*/
                                                  "orig"}},
                     /*tensor_transformation=*/unshard}}}};

  std::vector<CallbackResult> results;
  OriginalTensorSummaryCalculator calculator(
      std::make_shared<
          const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>(
          std::move(opt_dims)),
      std::make_shared<const absl::flat_hash_map<
          std::string, std::vector<ScopeInstruction>>>(std::move(call_map)),
      std::make_shared<const absl::flat_hash_map<
          TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>(
          std::move(orig_map)),
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const TensorTransformation> pending,
          const OriginalTensorSummary& summary) {
        results.push_back({key, std::move(pending), summary});
        return absl::OkStatus();
      });

  AbsoluteScopedTensorKey opt_key{/*tensor_key=*/opt_tensor_key};
  std::vector<DimSplitSpec> split_spec = {{/*dim_index=*/0, /*block_count=*/2}};
  ShardTensorSummary shard = {0, CreateSummary({4.0f, 4.1f}, split_spec)};

  ASSERT_THAT(calculator.ProcessShardSummary(opt_key, shard), IsOk());
  ASSERT_THAT(results, SizeIs(1));
  EXPECT_THAT(results[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{4}));
  // The reshape after unshard should be pending.
  EXPECT_THAT(results[0].pending_transformation, Eq(reshape));
  const auto& summary = results[0].original_tensor_summary.summaries[0];
  EXPECT_THAT(summary.split_spec, Eq(split_spec));
  ASSERT_THAT(summary.block_summaries, SizeIs(2));
  EXPECT_THAT(summary.block_summaries[0].min, Eq(4.0f));
  EXPECT_THAT(summary.block_summaries[1].min, Eq(4.1f));
  auto processing_metrics = calculator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics.completed_original_tensor_count, 1);
  EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 0);
}

TEST(OriginalTensorSummaryCalculatorTest, ReshapeThenUnshard) {
  const TensorKey opt_tensor_key{/*instruction_name=*/"opt"};
  absl::flat_hash_map<TensorKey, std::vector<int64_t>> opt_dims = {
      {opt_tensor_key, {4}}};
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map;
  auto unshard = std::make_shared<TensorTransformation>(
      Unshard{/*continuation=*/nullptr,
              /*original_dimensions=*/{2, 2},
              /*sharding=*/HloSharding::Replicate()});
  auto reshape = std::make_shared<TensorTransformation>(
      Reshape{/*continuation=*/unshard, /*output_dimensions=*/{2, 2}});
  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      orig_map = {{opt_tensor_key,
                   {{/*original_scoped_tensor_key=*/ScopedTensorKey{
                         /*tensor_key=*/TensorKey{/*instruction_name=*/
                                                  "orig"}},
                     /*tensor_transformation=*/reshape}}}};

  std::vector<CallbackResult> results;
  OriginalTensorSummaryCalculator calculator(
      std::make_shared<
          const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>(
          std::move(opt_dims)),
      std::make_shared<const absl::flat_hash_map<
          std::string, std::vector<ScopeInstruction>>>(std::move(call_map)),
      std::make_shared<const absl::flat_hash_map<
          TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>(
          std::move(orig_map)),
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const TensorTransformation> pending,
          const OriginalTensorSummary& summary) {
        results.push_back({key, std::move(pending), summary});
        return absl::OkStatus();
      });

  AbsoluteScopedTensorKey opt_key{/*tensor_key=*/opt_tensor_key};
  std::vector<DimSplitSpec> split_spec = {{/*dim_index=*/0, /*block_count=*/2}};
  ShardTensorSummary shard = {0, CreateSummary({4.0f, 4.1f}, split_spec)};

  ASSERT_THAT(calculator.ProcessShardSummary(opt_key, shard), IsOk());
  ASSERT_THAT(results, SizeIs(1));
  // Reshape is before unshard, so it's applied. Unshard is applied too.
  // The continuation of unshard is null, so pending is null.
  EXPECT_THAT(results[0].pending_transformation, Eq(nullptr));
  EXPECT_THAT(results[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{2, 2}));
  // The reshape from {4} to {2,2} should merge the split on dim 0.
  const auto& summary = results[0].original_tensor_summary.summaries[0];
  EXPECT_THAT(summary.split_spec, IsEmpty());
  ASSERT_THAT(summary.block_summaries, SizeIs(1));
  EXPECT_THAT(summary.block_summaries[0].min, Eq(4.0f));
  EXPECT_THAT(summary.block_summaries[0].max, Eq(4.1f));
  EXPECT_THAT(summary.block_summaries[0].count, Eq(2.0f));
  auto processing_metrics = calculator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics.completed_original_tensor_count, 1);
  EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 0);
}

TEST(OriginalTensorSummaryCalculatorTest, ReshapeThenTiledUnshard) {
  const TensorKey opt_tensor_key{/*instruction_name=*/"opt"};
  // Optimized shape (shard): [4]
  absl::flat_hash_map<TensorKey, std::vector<int64_t>> opt_dims = {
      {opt_tensor_key, {4}}};
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map;

  // Unshard: [4, 4] -> [8, 4] via tiling dim 0 by 2.
  Array<int64_t> tile_assignment({2, 1});
  tile_assignment(0, 0) = 0;
  tile_assignment(1, 0) = 1;
  auto unshard = std::make_shared<TensorTransformation>(
      Unshard{/*continuation=*/nullptr,
              /*original_dimensions=*/{8, 4},
              /*sharding=*/HloSharding::Tile(tile_assignment)});

  // Reshape: [4] -> [4, 4]
  auto reshape = std::make_shared<TensorTransformation>(
      Reshape{/*continuation=*/unshard, /*output_dimensions=*/{4, 4}});

  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      orig_map = {{opt_tensor_key,
                   {{/*original_scoped_tensor_key=*/ScopedTensorKey{
                         /*tensor_key=*/TensorKey{/*instruction_name=*/
                                                  "orig"}},
                     /*tensor_transformation=*/reshape}}}};

  std::vector<CallbackResult> results;
  OriginalTensorSummaryCalculator calculator(
      std::make_shared<
          const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>(
          std::move(opt_dims)),
      std::make_shared<const absl::flat_hash_map<
          std::string, std::vector<ScopeInstruction>>>(std::move(call_map)),
      std::make_shared<const absl::flat_hash_map<
          TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>(
          std::move(orig_map)),
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const TensorTransformation> pending,
          const OriginalTensorSummary& summary) {
        results.push_back({key, std::move(pending), summary});
        return absl::OkStatus();
      });

  AbsoluteScopedTensorKey opt_key{/*tensor_key=*/opt_tensor_key};
  std::vector<DimSplitSpec> shard_split_spec = {};

  // Process shard 0.
  ASSERT_THAT(calculator.ProcessShardSummary(
                  opt_key, {0, CreateSummary({1.0f}, shard_split_spec)}),
              IsOk());

  // Should NOT be emitted yet because we are waiting for shard 1 (hidden
  // Unshard).
  EXPECT_THAT(results, IsEmpty());

  // Process shard 1.
  ASSERT_THAT(calculator.ProcessShardSummary(
                  opt_key, {1, CreateSummary({2.0f}, shard_split_spec)}),
              IsOk());

  ASSERT_THAT(results, SizeIs(1));
  const auto& summary = results[0].original_tensor_summary.summaries[0];
  // Dim 0 sharded by 2.
  EXPECT_THAT(
      summary.split_spec,
      Eq(std::vector<DimSplitSpec>{{/*dim_index=*/0, /*block_count=*/2}}));
  ASSERT_THAT(summary.block_summaries, SizeIs(2));
  EXPECT_THAT(summary.block_summaries[0].min, Eq(1.0f));
  EXPECT_THAT(summary.block_summaries[1].min, Eq(2.0f));
}

TEST(OriginalTensorSummaryCalculatorTest, ReshapeThenUnshardMergeAllSplits) {
  const TensorKey opt_tensor_key{/*instruction_name=*/"opt"};
  absl::flat_hash_map<TensorKey, std::vector<int64_t>> opt_dims = {
      {opt_tensor_key, {2, 6}}};
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map;
  auto unshard = std::make_shared<TensorTransformation>(
      Unshard{/*continuation=*/nullptr,
              /*original_dimensions=*/{12},
              /*sharding=*/HloSharding::Replicate()});
  auto reshape = std::make_shared<TensorTransformation>(
      Reshape{/*continuation=*/unshard, /*output_dimensions=*/{12}});
  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      orig_map = {{opt_tensor_key,
                   {{/*original_scoped_tensor_key=*/ScopedTensorKey{
                         /*tensor_key=*/TensorKey{/*instruction_name=*/
                                                  "orig"}},
                     /*tensor_transformation=*/reshape}}}};

  std::vector<CallbackResult> results;
  OriginalTensorSummaryCalculator calculator(
      std::make_shared<
          const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>(
          std::move(opt_dims)),
      std::make_shared<const absl::flat_hash_map<
          std::string, std::vector<ScopeInstruction>>>(std::move(call_map)),
      std::make_shared<const absl::flat_hash_map<
          TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>(
          std::move(orig_map)),
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const TensorTransformation> pending,
          const OriginalTensorSummary& summary) {
        results.push_back({key, std::move(pending), summary});
        return absl::OkStatus();
      });

  AbsoluteScopedTensorKey opt_key{/*tensor_key=*/opt_tensor_key};
  std::vector<DimSplitSpec> split_spec = {{/*dim_index=*/0, /*block_count=*/2},
                                          {/*dim_index=*/1, /*block_count=*/3}};
  ShardTensorSummary shard = {
      0, CreateSummary({1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f}, split_spec)};

  ASSERT_THAT(calculator.ProcessShardSummary(opt_key, shard), IsOk());
  ASSERT_THAT(results, SizeIs(1));
  EXPECT_THAT(results[0].pending_transformation, Eq(nullptr));
  EXPECT_THAT(results[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{12}));
  const auto& summary = results[0].original_tensor_summary.summaries[0];
  EXPECT_THAT(summary.split_spec, IsEmpty());
  ASSERT_THAT(summary.block_summaries, SizeIs(1));
  EXPECT_THAT(summary.block_summaries[0].min, Eq(1.0f));
  EXPECT_THAT(summary.block_summaries[0].max, Eq(1.5f));
  EXPECT_THAT(summary.block_summaries[0].count, Eq(6.0f));
  EXPECT_NEAR(summary.block_summaries[0].mean, 1.25f, 1e-6);
  auto processing_metrics = calculator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics.completed_original_tensor_count, 1);
  EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 0);
}

TEST(OriginalTensorSummaryCalculatorTest,
     ReshapeThenUnshardPreservePrefixSuffixSplits) {
  const TensorKey opt_tensor_key{/*instruction_name=*/"opt"};
  absl::flat_hash_map<TensorKey, std::vector<int64_t>> opt_dims = {
      {opt_tensor_key, {2, 6, 4}}};
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map;
  auto unshard = std::make_shared<TensorTransformation>(
      Unshard{/*continuation=*/nullptr,
              /*original_dimensions=*/{2, 2, 3, 4},
              /*sharding=*/HloSharding::Replicate()});
  auto reshape = std::make_shared<TensorTransformation>(
      Reshape{/*continuation=*/unshard, /*output_dimensions=*/{2, 2, 3, 4}});
  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      orig_map = {{opt_tensor_key,
                   {{/*original_scoped_tensor_key=*/ScopedTensorKey{
                         /*tensor_key=*/TensorKey{/*instruction_name=*/
                                                  "orig"}},
                     /*tensor_transformation=*/reshape}}}};

  std::vector<CallbackResult> results;
  OriginalTensorSummaryCalculator calculator(
      std::make_shared<
          const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>(
          std::move(opt_dims)),
      std::make_shared<const absl::flat_hash_map<
          std::string, std::vector<ScopeInstruction>>>(std::move(call_map)),
      std::make_shared<const absl::flat_hash_map<
          TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>(
          std::move(orig_map)),
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const TensorTransformation> pending,
          const OriginalTensorSummary& summary) {
        results.push_back({key, std::move(pending), summary});
        return absl::OkStatus();
      });

  AbsoluteScopedTensorKey opt_key{/*tensor_key=*/opt_tensor_key};
  std::vector<DimSplitSpec> split_spec = {{/*dim_index=*/0, /*block_count=*/2},
                                          {/*dim_index=*/1, /*block_count=*/2},
                                          {/*dim_index=*/2, /*block_count=*/2}};
  ShardTensorSummary shard = {
      0, CreateSummary({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
                       split_spec)};

  ASSERT_THAT(calculator.ProcessShardSummary(opt_key, shard), IsOk());
  ASSERT_THAT(results, SizeIs(1));
  EXPECT_THAT(results[0].pending_transformation, Eq(nullptr));
  EXPECT_THAT(results[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{2, 2, 3, 4}));
  const auto& summary = results[0].original_tensor_summary.summaries[0];
  EXPECT_THAT(summary.split_spec, Eq(std::vector<DimSplitSpec>{
                                      {/*dim_index=*/0, /*block_count=*/2},
                                      {/*dim_index=*/3, /*block_count=*/2}}));
  ASSERT_THAT(summary.block_summaries, SizeIs(4));
  // block_indices: {0,0} from original blocks {0,0,0} and {0,1,0}
  EXPECT_THAT(summary.block_summaries[0].block_indices,
              Eq(std::vector<int64_t>{0, 0}));
  EXPECT_THAT(summary.block_summaries[0].min, Eq(1.0f));
  EXPECT_THAT(summary.block_summaries[0].max, Eq(3.0f));
  EXPECT_NEAR(summary.block_summaries[0].mean, 2.0f, 1e-6);
  // block_indices: {0,1} from original blocks {0,0,1} and {0,1,1}
  EXPECT_THAT(summary.block_summaries[1].block_indices,
              Eq(std::vector<int64_t>{0, 1}));
  EXPECT_THAT(summary.block_summaries[1].min, Eq(2.0f));
  EXPECT_THAT(summary.block_summaries[1].max, Eq(4.0f));
  EXPECT_NEAR(summary.block_summaries[1].mean, 3.0f, 1e-6);
  // block_indices: {1,0} from original blocks {1,0,0} and {1,1,0}
  EXPECT_THAT(summary.block_summaries[2].block_indices,
              Eq(std::vector<int64_t>{1, 0}));
  EXPECT_THAT(summary.block_summaries[2].min, Eq(5.0f));
  EXPECT_THAT(summary.block_summaries[2].max, Eq(7.0f));
  EXPECT_NEAR(summary.block_summaries[2].mean, 6.0f, 1e-6);
  // block_indices: {1,1} from original blocks {1,0,1} and {1,1,1}
  EXPECT_THAT(summary.block_summaries[3].block_indices,
              Eq(std::vector<int64_t>{1, 1}));
  EXPECT_THAT(summary.block_summaries[3].min, Eq(6.0f));
  EXPECT_THAT(summary.block_summaries[3].max, Eq(8.0f));
  EXPECT_NEAR(summary.block_summaries[3].mean, 7.0f, 1e-6);
  auto processing_metrics = calculator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics.completed_original_tensor_count, 1);
  EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 0);
}

TEST(OriginalTensorSummaryCalculatorTest, ReshapeThenUnshardPreserveAllSplits) {
  const TensorKey opt_tensor_key{/*instruction_name=*/"opt"};
  absl::flat_hash_map<TensorKey, std::vector<int64_t>> opt_dims = {
      {opt_tensor_key, {2, 6}}};
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map;
  auto unshard = std::make_shared<TensorTransformation>(
      Unshard{/*continuation=*/nullptr,
              /*original_dimensions=*/{1, 2, 6},
              /*sharding=*/HloSharding::Replicate()});
  auto reshape = std::make_shared<TensorTransformation>(
      Reshape{/*continuation=*/unshard, /*output_dimensions=*/{1, 2, 6}});
  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      orig_map = {{opt_tensor_key,
                   {{/*original_scoped_tensor_key=*/ScopedTensorKey{
                         /*tensor_key=*/TensorKey{/*instruction_name=*/
                                                  "orig"}},
                     /*tensor_transformation=*/reshape}}}};

  std::vector<CallbackResult> results;
  OriginalTensorSummaryCalculator calculator(
      std::make_shared<
          const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>(
          std::move(opt_dims)),
      std::make_shared<const absl::flat_hash_map<
          std::string, std::vector<ScopeInstruction>>>(std::move(call_map)),
      std::make_shared<const absl::flat_hash_map<
          TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>(
          std::move(orig_map)),
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const TensorTransformation> pending,
          const OriginalTensorSummary& summary) {
        results.push_back({key, std::move(pending), summary});
        return absl::OkStatus();
      });

  AbsoluteScopedTensorKey opt_key{/*tensor_key=*/opt_tensor_key};
  std::vector<DimSplitSpec> split_spec = {{/*dim_index=*/0, /*block_count=*/2},
                                          {/*dim_index=*/1, /*block_count=*/3}};
  ShardTensorSummary shard = {
      0, CreateSummary({1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f}, split_spec)};

  ASSERT_THAT(calculator.ProcessShardSummary(opt_key, shard), IsOk());
  ASSERT_THAT(results, SizeIs(1));
  EXPECT_THAT(results[0].pending_transformation, Eq(nullptr));
  EXPECT_THAT(results[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{1, 2, 6}));
  const auto& summary = results[0].original_tensor_summary.summaries[0];
  EXPECT_THAT(summary.split_spec, Eq(std::vector<DimSplitSpec>{
                                      {/*dim_index=*/1, /*block_count=*/2},
                                      {/*dim_index=*/2, /*block_count=*/3}}));
  ASSERT_THAT(summary.block_summaries, SizeIs(6));
  EXPECT_THAT(summary.block_summaries[0].block_indices,
              Eq(std::vector<int64_t>{0, 0}));
  EXPECT_THAT(summary.block_summaries[0].min, Eq(1.0f));
  EXPECT_THAT(summary.block_summaries[1].block_indices,
              Eq(std::vector<int64_t>{0, 1}));
  EXPECT_THAT(summary.block_summaries[1].min, Eq(1.1f));
  EXPECT_THAT(summary.block_summaries[2].block_indices,
              Eq(std::vector<int64_t>{0, 2}));
  EXPECT_THAT(summary.block_summaries[2].min, Eq(1.2f));
  EXPECT_THAT(summary.block_summaries[3].block_indices,
              Eq(std::vector<int64_t>{1, 0}));
  EXPECT_THAT(summary.block_summaries[3].min, Eq(2.0f));
  EXPECT_THAT(summary.block_summaries[4].block_indices,
              Eq(std::vector<int64_t>{1, 1}));
  EXPECT_THAT(summary.block_summaries[4].min, Eq(2.1f));
  EXPECT_THAT(summary.block_summaries[5].block_indices,
              Eq(std::vector<int64_t>{1, 2}));
  EXPECT_THAT(summary.block_summaries[5].min, Eq(2.2f));
  auto processing_metrics = calculator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics.completed_original_tensor_count, 1);
  EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 0);
}

TEST(OriginalTensorSummaryCalculatorTest, UnshardWithReplicatedDimSplit) {
  const TensorKey opt_tensor_key{/*instruction_name=*/"opt"};
  absl::flat_hash_map<TensorKey, std::vector<int64_t>> opt_dims = {
      {opt_tensor_key, {2, 2, 5}}};  // Shard shape
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map;
  Array<int64_t> tile_assignment({1, 1, 2});
  tile_assignment(0, 0, 0) = 0;
  tile_assignment(0, 0, 1) = 1;
  auto unshard = std::make_shared<TensorTransformation>(
      Unshard{/*continuation=*/nullptr,
              /*original_dimensions=*/{2, 2, 10},  // Original shape
              /*sharding=*/HloSharding::Tile(tile_assignment)});
  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      orig_map = {{opt_tensor_key,
                   {{/*original_scoped_tensor_key=*/ScopedTensorKey{
                         /*tensor_key=*/TensorKey{/*instruction_name=*/
                                                  "orig"}},
                     /*tensor_transformation=*/unshard}}}};

  std::vector<CallbackResult> results;
  OriginalTensorSummaryCalculator calculator(
      std::make_shared<
          const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>(
          std::move(opt_dims)),
      std::make_shared<const absl::flat_hash_map<
          std::string, std::vector<ScopeInstruction>>>(std::move(call_map)),
      std::make_shared<const absl::flat_hash_map<
          TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>(
          std::move(orig_map)),
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const TensorTransformation> pending,
          const OriginalTensorSummary& summary) {
        results.push_back({key, std::move(pending), summary});
        return absl::OkStatus();
      });

  AbsoluteScopedTensorKey opt_key{/*tensor_key=*/opt_tensor_key};
  std::vector<DimSplitSpec> shard_split_spec = {
      {/*dim_index=*/0, /*block_count=*/2},
      {/*dim_index=*/1, /*block_count=*/2}};
  ShardTensorSummary shard0 = {
      0, CreateSummary({1.0f, 1.1f, 1.2f, 1.3f}, shard_split_spec)};
  ShardTensorSummary shard1 = {
      1, CreateSummary({2.0f, 2.1f, 2.2f, 2.3f}, shard_split_spec)};

  ASSERT_THAT(calculator.ProcessShardSummary(opt_key, shard0), IsOk());
  EXPECT_THAT(results, IsEmpty());  // Waiting for shard 1
  auto processing_metrics0 = calculator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics0.received_optimized_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics0.processed_original_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics0.completed_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics0.completed_original_tensor_count, 0);
  EXPECT_EQ(processing_metrics0.incomplete_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics0.incomplete_original_tensor_count, 1);

  ASSERT_THAT(calculator.ProcessShardSummary(opt_key, shard1), IsOk());
  ASSERT_THAT(results, SizeIs(1));
  EXPECT_THAT(results[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{2, 2, 10}));
  const auto& summary = results[0].original_tensor_summary.summaries[0];
  std::vector<DimSplitSpec> combined_split_spec = {
      {/*dim_index=*/0, /*block_count=*/2},
      {/*dim_index=*/1, /*block_count=*/2},
      {/*dim_index=*/2, /*block_count=*/2}};
  EXPECT_THAT(summary.split_spec, Eq(combined_split_spec));
  ASSERT_THAT(summary.block_summaries, SizeIs(8));

  // block {0,0,0}
  EXPECT_THAT(summary.block_summaries[0].block_indices,
              Eq(std::vector<int64_t>{0, 0, 0}));
  EXPECT_THAT(summary.block_summaries[0].min, Eq(1.0f));
  EXPECT_THAT(summary.block_summaries[0].max, Eq(1.0f));
  EXPECT_THAT(summary.block_summaries[0].mean, Eq(1.0f));
  EXPECT_THAT(summary.block_summaries[0].count, Eq(1));

  // block {0,1,1}
  EXPECT_THAT(summary.block_summaries[3].block_indices,
              Eq(std::vector<int64_t>{0, 1, 1}));
  EXPECT_THAT(summary.block_summaries[3].min, Eq(2.1f));
  EXPECT_THAT(summary.block_summaries[3].max, Eq(2.1f));
  EXPECT_THAT(summary.block_summaries[3].mean, Eq(2.1f));
  EXPECT_THAT(summary.block_summaries[3].count, Eq(1));

  // block {1,0,1}
  EXPECT_THAT(summary.block_summaries[5].block_indices,
              Eq(std::vector<int64_t>{1, 0, 1}));
  EXPECT_THAT(summary.block_summaries[5].min, Eq(2.2f));
  EXPECT_THAT(summary.block_summaries[5].max, Eq(2.2f));
  EXPECT_THAT(summary.block_summaries[5].mean, Eq(2.2f));
  EXPECT_THAT(summary.block_summaries[5].count, Eq(1));

  // block {1,1,1}
  EXPECT_THAT(summary.block_summaries[7].block_indices,
              Eq(std::vector<int64_t>{1, 1, 1}));
  EXPECT_THAT(summary.block_summaries[7].min, Eq(2.3f));
  EXPECT_THAT(summary.block_summaries[7].max, Eq(2.3f));
  EXPECT_THAT(summary.block_summaries[7].mean, Eq(2.3f));
  EXPECT_THAT(summary.block_summaries[7].count, Eq(1));

  auto processing_metrics1 = calculator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics1.received_optimized_tensor_shard_count, 2);
  EXPECT_EQ(processing_metrics1.processed_original_tensor_shard_count, 2);
  EXPECT_EQ(processing_metrics1.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics1.completed_original_tensor_count, 1);
  EXPECT_EQ(processing_metrics1.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics1.incomplete_original_tensor_count, 0);
}

TEST(OriginalTensorSummaryCalculatorTest, ShardTensorSummaryToDebugString) {
  ShardTensorSummary shard_summary;
  shard_summary.logical_shard_id = 123;
  shard_summary.summary.split_spec = {{/*dim_index=*/0, /*block_count=*/2},
                                      {/*dim_index=*/1, /*block_count=*/3}};
  shard_summary.summary.block_summaries = {{/*block_indices=*/{0, 0},
                                            /*min=*/1.0f,
                                            /*max=*/2.0f,
                                            /*mean=*/1.5f,
                                            /*stddev=*/0.5f,
                                            /*count=*/10},
                                           {/*block_indices=*/{1, 2},
                                            /*min=*/3.0f,
                                            /*max=*/4.0f,
                                            /*mean=*/3.5f,
                                            /*stddev=*/0.5f,
                                            /*count=*/20}};
  EXPECT_EQ(shard_summary.ToDebugString(),
            "ShardTensorSummary{\n"
            "  logical_shard_id: 123\n"
            "  summary:\n"
            "    split_spec:\n"
            "      {dim_index: 0, block_count: 2}\n"
            "      {dim_index: 1, block_count: 3}\n"
            "    block_summaries:\n"
            "      {block_indices: [0, 0], min: 1, max: 2, mean: 1.5, "
            "stddev: 0.5, count: 10}\n"
            "      {block_indices: [1, 2], min: 3, max: 4, mean: 3.5, "
            "stddev: 0.5, count: 20}\n"
            "}\n");
}

TEST(OriginalTensorSummaryCalculatorTest, MultipleOriginalTensors) {
  const TensorKey opt_tensor_key{/*instruction_name=*/"opt"};
  absl::flat_hash_map<TensorKey, std::vector<int64_t>> opt_dims = {
      {opt_tensor_key, {2, 2}}};
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map;
  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      orig_map = {{opt_tensor_key,
                   {{/*original_scoped_tensor_key=*/ScopedTensorKey{
                         /*tensor_key=*/TensorKey{/*instruction_name=*/
                                                  "orig1"}},
                     /*tensor_transformation=*/nullptr},
                    {/*original_scoped_tensor_key=*/ScopedTensorKey{
                         /*tensor_key=*/TensorKey{/*instruction_name=*/
                                                  "orig2"}},
                     /*tensor_transformation=*/nullptr}}}};

  std::vector<CallbackResult> results;
  OriginalTensorSummaryCalculator calculator(
      std::make_shared<
          const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>(
          std::move(opt_dims)),
      std::make_shared<const absl::flat_hash_map<
          std::string, std::vector<ScopeInstruction>>>(std::move(call_map)),
      std::make_shared<const absl::flat_hash_map<
          TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>(
          std::move(orig_map)),
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const TensorTransformation> pending,
          const OriginalTensorSummary& summary) {
        results.push_back({key, std::move(pending), summary});
        return absl::OkStatus();
      });

  AbsoluteScopedTensorKey opt_key{/*tensor_key=*/opt_tensor_key};
  std::vector<DimSplitSpec> split_spec = {{/*dim_index=*/0, /*block_count=*/1}};
  ShardTensorSummary shard = {0, CreateSummary({5.0f}, split_spec)};

  ASSERT_THAT(calculator.ProcessShardSummary(opt_key, shard), IsOk());
  ASSERT_THAT(results, SizeIs(2));
  EXPECT_THAT(results[0].original_tensor_key.tensor_key.instruction_name,
              Eq("orig1"));
  EXPECT_THAT(results[0].original_tensor_summary.summaries[0].split_spec,
              Eq(split_spec));
  EXPECT_THAT(
      results[0].original_tensor_summary.summaries[0].block_summaries[0].min,
      Eq(5.0f));
  EXPECT_THAT(results[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{2, 2}));
  EXPECT_THAT(results[1].original_tensor_key.tensor_key.instruction_name,
              Eq("orig2"));
  EXPECT_THAT(results[1].original_tensor_summary.summaries[0].split_spec,
              Eq(split_spec));
  EXPECT_THAT(
      results[1].original_tensor_summary.summaries[0].block_summaries[0].min,
      Eq(5.0f));
  EXPECT_THAT(results[1].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{2, 2}));
  auto processing_metrics = calculator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count, 2);
  EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics.completed_original_tensor_count, 2);
  EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 0);
}

TEST(OriginalTensorSummaryCalculatorTest,
     MultipleOptimizedTensorsToOneOriginal) {
  const TensorKey opt_tensor_key1{/*instruction_name=*/"opt1"};
  const TensorKey opt_tensor_key2{/*instruction_name=*/"opt2"};
  const RelativeScopedTensorKey orig_tensor_key{
      /*tensor_key=*/TensorKey{/*instruction_name=*/"orig"}};
  absl::flat_hash_map<TensorKey, std::vector<int64_t>> opt_dims = {
      {opt_tensor_key1, {2, 2}}, {opt_tensor_key2, {2, 2}}};
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map;
  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      orig_map = {{opt_tensor_key1,
                   {{/*original_scoped_tensor_key=*/orig_tensor_key,
                     /*tensor_transformation=*/nullptr}}},
                  {opt_tensor_key2,
                   {{/*original_scoped_tensor_key=*/orig_tensor_key,
                     /*tensor_transformation=*/nullptr}}}};

  std::vector<CallbackResult> results;
  OriginalTensorSummaryCalculator calculator(
      std::make_shared<
          const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>(
          std::move(opt_dims)),
      std::make_shared<const absl::flat_hash_map<
          std::string, std::vector<ScopeInstruction>>>(std::move(call_map)),
      std::make_shared<const absl::flat_hash_map<
          TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>(
          std::move(orig_map)),
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const TensorTransformation> pending,
          const OriginalTensorSummary& summary) {
        results.push_back({key, std::move(pending), summary});
        return absl::OkStatus();
      });

  AbsoluteScopedTensorKey opt_key1{/*tensor_key=*/opt_tensor_key1};
  std::vector<DimSplitSpec> split_spec1 = {};
  ShardTensorSummary shard1 = {0, CreateSummary({1.0f}, split_spec1)};

  ASSERT_THAT(calculator.ProcessShardSummary(opt_key1, shard1), IsOk());
  ASSERT_THAT(results, SizeIs(1));
  EXPECT_THAT(results[0].original_tensor_key.tensor_key.instruction_name,
              Eq("orig"));
  EXPECT_THAT(
      results[0].original_tensor_summary.summaries[0].block_summaries[0].min,
      Eq(1.0f));

  AbsoluteScopedTensorKey opt_key2{/*tensor_key=*/opt_tensor_key2};
  std::vector<DimSplitSpec> split_spec2 = {};
  ShardTensorSummary shard2 = {0, CreateSummary({2.0f}, split_spec2)};

  ASSERT_THAT(calculator.ProcessShardSummary(opt_key2, shard2), IsOk());
  ASSERT_THAT(results, SizeIs(1));  // result size should still be 1.

  auto processing_metrics = calculator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count, 2);
  EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count, 2);
  EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 2);
  EXPECT_EQ(processing_metrics.completed_original_tensor_count, 1);
  EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 0);
}

TEST(OriginalTensorSummaryCalculatorTest, ConstructsOriginalKeyWithCallMap) {
  const TensorKey opt_tensor_key{/*instruction_name=*/"opt_instr"};
  absl::flat_hash_map<TensorKey, std::vector<int64_t>> opt_dims = {
      {opt_tensor_key, {2, 2}}};

  // call_map: call2_inlined -> [call1, call2]
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map = {
      {"call2_inlined",
       {ScopeInstruction::Create("call1"), ScopeInstruction::Create("call2")}}};

  // original_tensor_by_optimized_tensor_key_: opt_instr -> orig_instr
  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      orig_map = {{opt_tensor_key,
                   {{/*original_scoped_tensor_key=*/ScopedTensorKey{
                         /*tensor_key=*/TensorKey{/*instruction_name=*/
                                                  "orig_instr"}},
                     /*tensor_transformation=*/nullptr}}}};

  std::vector<CallbackResult> results;
  OriginalTensorSummaryCalculator calculator(
      std::make_shared<
          const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>(
          std::move(opt_dims)),
      std::make_shared<const absl::flat_hash_map<
          std::string, std::vector<ScopeInstruction>>>(std::move(call_map)),
      std::make_shared<const absl::flat_hash_map<
          TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>(
          std::move(orig_map)),
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const TensorTransformation> pending,
          const OriginalTensorSummary& summary) {
        results.push_back({key, std::move(pending), summary});
        return absl::OkStatus();
      });

  // Optimized tensor has scope [call2_inlined]
  AbsoluteScopedTensorKey opt_key{
      /*scope_instructions=*/{ScopeInstruction::Create("call2_inlined")},
      /*tensor_key=*/opt_tensor_key,
  };

  std::vector<DimSplitSpec> split_spec = {{/*dim_index=*/0, /*block_count=*/2}};
  FloatSummary shard_summary = CreateSummary({1.0f, 2.0f}, split_spec);
  ShardTensorSummary shard = {0, shard_summary};

  ASSERT_THAT(calculator.ProcessShardSummary(opt_key, shard), IsOk());
  ASSERT_THAT(results, SizeIs(1));

  // The resulting original tensor key should have scope [call1, call2]
  EXPECT_THAT(results[0].original_tensor_key.tensor_key.instruction_name,
              Eq("orig_instr"));
  EXPECT_THAT(
      results[0].original_tensor_key.scope_instructions,
      Eq(std::vector<ScopeInstruction>{ScopeInstruction::Create("call1"),
                                       ScopeInstruction::Create("call2")}));
  EXPECT_THAT(results[0].pending_transformation, Eq(nullptr));
  EXPECT_THAT(results[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{2, 2}));

  const auto& summary = results[0].original_tensor_summary.summaries[0];
  EXPECT_THAT(summary.split_spec, Eq(split_spec));
  auto processing_metrics = calculator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics.completed_original_tensor_count, 1);
  EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 0);
}

TEST(OriginalTensorSummaryCalculatorTest,
     ConstructsOriginalKeyWithCallMapAndUnmappedScopes) {
  const TensorKey opt_tensor_key{/*instruction_name=*/"opt_instr"};
  absl::flat_hash_map<TensorKey, std::vector<int64_t>> opt_dims = {
      {opt_tensor_key, {2, 2}}};

  // call_map: call2_inlined -> [call1, call2]
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map = {
      {"call2_inlined",
       {ScopeInstruction::Create("call1"), ScopeInstruction::Create("call2")}}};

  // original_tensor_by_optimized_tensor_key_: opt_instr -> orig_instr
  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      orig_map = {{opt_tensor_key,
                   {{/*original_scoped_tensor_key=*/ScopedTensorKey{
                         /*tensor_key=*/TensorKey{/*instruction_name=*/
                                                  "orig_instr"}},
                     /*tensor_transformation=*/nullptr}}}};

  std::vector<CallbackResult> results;
  OriginalTensorSummaryCalculator calculator(
      std::make_shared<
          const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>(
          std::move(opt_dims)),
      std::make_shared<const absl::flat_hash_map<
          std::string, std::vector<ScopeInstruction>>>(std::move(call_map)),
      std::make_shared<const absl::flat_hash_map<
          TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>(
          std::move(orig_map)),
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const TensorTransformation> pending,
          const OriginalTensorSummary& summary) {
        results.push_back({key, std::move(pending), summary});
        return absl::OkStatus();
      });

  // Optimized tensor has scope [call2_inlined, call3]
  AbsoluteScopedTensorKey opt_key{
      /*scope_instructions=*/{ScopeInstruction::Create("call2_inlined"),
                              ScopeInstruction::Create("call3")},
      /*tensor_key=*/opt_tensor_key,
  };

  std::vector<DimSplitSpec> split_spec = {{/*dim_index=*/0, /*block_count=*/2}};
  FloatSummary shard_summary = CreateSummary({1.0f, 2.0f}, split_spec);
  ShardTensorSummary shard = {0, shard_summary};

  ASSERT_THAT(calculator.ProcessShardSummary(opt_key, shard), IsOk());
  ASSERT_THAT(results, SizeIs(1));

  // The resulting original tensor key should have scope [call1, call2,
  // call3?]
  EXPECT_THAT(results[0].original_tensor_key.tensor_key.instruction_name,
              Eq("orig_instr"));
  EXPECT_THAT(
      results[0].original_tensor_key.scope_instructions,
      Eq(std::vector<ScopeInstruction>{ScopeInstruction::Create("call1"),
                                       ScopeInstruction::Create("call2"),
                                       ScopeInstruction::Create("call3?")}));
  EXPECT_THAT(results[0].pending_transformation, Eq(nullptr));
  EXPECT_THAT(results[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{2, 2}));

  const auto& summary = results[0].original_tensor_summary.summaries[0];
  EXPECT_THAT(summary.split_spec, Eq(split_spec));
  auto processing_metrics = calculator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics.completed_original_tensor_count, 1);
  EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 0);
}

TEST(OriginalTensorSummaryCalculatorTest,
     ConstructsOriginalKeyWithCallMapAndRelativeScopes) {
  const TensorKey opt_tensor_key{/*instruction_name=*/"opt_instr"};
  absl::flat_hash_map<TensorKey, std::vector<int64_t>> opt_dims = {
      {opt_tensor_key, {2, 2}}};

  // call_map: call2_inlined -> [call1]
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map = {
      {"call2_inlined", {ScopeInstruction::Create("call1")}}};

  // original_tensor_by_optimized_tensor_key_: opt_instr -> orig_instr with
  // relative scope [call2]
  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      orig_map = {
          {opt_tensor_key,
           {{/*original_scoped_tensor_key=*/
             {/*scope_instructions=*/{ScopeInstruction::Create("call2")},
              /*tensor_key=*/TensorKey{/*instruction_name=*/"orig_instr"}},
             /*tensor_transformation=*/nullptr}}}};

  std::vector<CallbackResult> results;
  OriginalTensorSummaryCalculator calculator(
      std::make_shared<
          const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>(
          std::move(opt_dims)),
      std::make_shared<const absl::flat_hash_map<
          std::string, std::vector<ScopeInstruction>>>(std::move(call_map)),
      std::make_shared<const absl::flat_hash_map<
          TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>(
          std::move(orig_map)),
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const TensorTransformation> pending,
          const OriginalTensorSummary& summary) {
        results.push_back({key, std::move(pending), summary});
        return absl::OkStatus();
      });

  // Optimized tensor has scope [call2_inlined]
  AbsoluteScopedTensorKey opt_key{
      /*scope_instructions=*/{ScopeInstruction::Create("call2_inlined")},
      /*tensor_key=*/opt_tensor_key,
  };

  std::vector<DimSplitSpec> split_spec = {{/*dim_index=*/0, /*block_count=*/2}};
  FloatSummary shard_summary = CreateSummary({1.0f, 2.0f}, split_spec);
  ShardTensorSummary shard = {0, shard_summary};

  ASSERT_THAT(calculator.ProcessShardSummary(opt_key, shard), IsOk());
  ASSERT_THAT(results, SizeIs(1));

  // The resulting original tensor key should have scope [call1, call2]
  EXPECT_THAT(results[0].original_tensor_key.tensor_key.instruction_name,
              Eq("orig_instr"));
  EXPECT_THAT(
      results[0].original_tensor_key.scope_instructions,
      Eq(std::vector<ScopeInstruction>{ScopeInstruction::Create("call1"),
                                       ScopeInstruction::Create("call2")}));
  EXPECT_THAT(results[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{2, 2}));
  auto processing_metrics = calculator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics.completed_original_tensor_count, 1);
  EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 0);
}

TEST(OriginalTensorSummaryCalculatorTest,
     ConstructsOriginalKeyWithCallMapAndIterationIndex) {
  const TensorKey opt_tensor_key{/*instruction_name=*/"opt_instr"};
  absl::flat_hash_map<TensorKey, std::vector<int64_t>> opt_dims = {
      {opt_tensor_key, {2, 2}}};

  // call_map: while_loop_unrolled_1 -> [call1, while_loop]
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map = {
      {"while_loop_unrolled_1",
       {ScopeInstruction::Create("call1"),
        ScopeInstruction::Create("while_loop", 1)}}};

  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      orig_map = {{opt_tensor_key,
                   {{/*original_scoped_tensor_key=*/ScopedTensorKey{
                         /*tensor_key=*/TensorKey{/*instruction_name=*/
                                                  "orig_instr"}},
                     /*tensor_transformation=*/nullptr}}}};

  std::vector<CallbackResult> results;
  OriginalTensorSummaryCalculator calculator(
      std::make_shared<
          const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>(
          std::move(opt_dims)),
      std::make_shared<const absl::flat_hash_map<
          std::string, std::vector<ScopeInstruction>>>(std::move(call_map)),
      std::make_shared<const absl::flat_hash_map<
          TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>(
          std::move(orig_map)),
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const TensorTransformation> pending,
          const OriginalTensorSummary& summary) {
        results.push_back({key, std::move(pending), summary});
        return absl::OkStatus();
      });

  // Optimized tensor has scope [while_loop_unrolled_1] with iteration index 1
  AbsoluteScopedTensorKey opt_key{
      /*scope_instructions=*/{
          ScopeInstruction::Create("while_loop_unrolled_1", 0)},
      /*tensor_key=*/opt_tensor_key,
  };

  std::vector<DimSplitSpec> split_spec = {{/*dim_index=*/0, /*block_count=*/2}};
  FloatSummary shard_summary = CreateSummary({1.0f, 2.0f}, split_spec);
  ShardTensorSummary shard = {0, shard_summary};

  ASSERT_THAT(calculator.ProcessShardSummary(opt_key, shard), IsOk());
  ASSERT_THAT(results, SizeIs(1));

  // The resulting original tensor key should have scope [call1, {while_loop,
  // 1}]
  EXPECT_THAT(results[0].original_tensor_key.tensor_key.instruction_name,
              Eq("orig_instr"));
  EXPECT_THAT(results[0].original_tensor_key.scope_instructions,
              Eq(std::vector<ScopeInstruction>{
                  ScopeInstruction::Create("call1"),
                  ScopeInstruction::Create("while_loop", 1)}));
  EXPECT_THAT(results[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{2, 2}));
  auto processing_metrics = calculator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count, 1);
  EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics.completed_original_tensor_count, 1);
  EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 0);
}

TEST(OriginalTensorSummaryCalculatorTest, ComplexTiledUnshard) {
  const TensorKey opt_tensor_key{/*instruction_name=*/"opt"};
  // Optimized tensor shape per shard: [2, 3]
  absl::flat_hash_map<TensorKey, std::vector<int64_t>> opt_dims = {
      {opt_tensor_key, {2, 3}}};
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map;
  // Original tensor shape: [4, 6]
  // Sharding: devices=[2, 2]
  // dim 0 is sharded 2 ways, dim 1 is sharded 2 ways.
  Array<int64_t> tile_assignment({2, 2});
  tile_assignment(0, 0) = 0;
  tile_assignment(0, 1) = 1;
  tile_assignment(1, 0) = 2;
  tile_assignment(1, 1) = 3;
  auto unshard = std::make_shared<TensorTransformation>(
      Unshard{/*continuation=*/nullptr,
              /*original_dimensions=*/{4, 6},
              /*sharding=*/HloSharding::Tile(tile_assignment)});
  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      orig_map = {{opt_tensor_key,
                   {{/*original_scoped_tensor_key=*/ScopedTensorKey{
                         /*tensor_key=*/TensorKey{/*instruction_name=*/
                                                  "orig"}},
                     /*tensor_transformation=*/unshard}}}};

  std::vector<CallbackResult> results;
  OriginalTensorSummaryCalculator calculator(
      std::make_shared<
          const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>(
          std::move(opt_dims)),
      std::make_shared<const absl::flat_hash_map<
          std::string, std::vector<ScopeInstruction>>>(std::move(call_map)),
      std::make_shared<const absl::flat_hash_map<
          TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>(
          std::move(orig_map)),
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const TensorTransformation> pending,
          const OriginalTensorSummary& summary) {
        results.push_back({key, std::move(pending), summary});
        return absl::OkStatus();
      });

  AbsoluteScopedTensorKey opt_key{/*tensor_key=*/opt_tensor_key};
  // Shard 0: {id=0, val=1.0}
  // Shard 1: {id=1, val=2.0}
  // Shard 2: {id=2, val=3.0}
  // Shard 3: {id=3, val=4.0}
  std::vector<DimSplitSpec> shard_split_spec = {
      {/*dim_index=*/0, /*block_count=*/2},
      {/*dim_index=*/1, /*block_count=*/3}};
  for (int i = 0; i < 4; ++i) {
    std::vector<float> values;
    values.reserve(6);
    for (int j = 0; j < 6; ++j) {
      values.push_back(static_cast<float>(i + 1) +
                       static_cast<float>(j) * 0.1f);
    }
    ASSERT_THAT(calculator.ProcessShardSummary(
                    opt_key, {static_cast<int64_t>(i),
                              CreateSummary(values, shard_split_spec)}),
                IsOk());
    if (i < 3) {
      EXPECT_THAT(results, IsEmpty());
      auto processing_metrics = calculator.GetProcessingMetrics();
      EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count,
                i + 1);
      EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count,
                i + 1);
      EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 0);
      EXPECT_EQ(processing_metrics.completed_original_tensor_count, 0);
      EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 1);
      EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 1);
    }
  }

  ASSERT_THAT(results, SizeIs(1));
  EXPECT_THAT(results[0].original_tensor_summary.dimensions,
              Eq(std::vector<int64_t>{4, 6}));
  const auto& summary = results[0].original_tensor_summary.summaries[0];
  EXPECT_THAT(summary.split_spec, Eq(std::vector<DimSplitSpec>{
                                      {/*dim_index=*/0, /*block_count=*/4},
                                      {/*dim_index=*/1, /*block_count=*/6}}));
  ASSERT_THAT(summary.block_summaries, SizeIs(24));
  // from shard 0 (val = 1.0 + 0.1*j)
  EXPECT_THAT(summary.block_summaries[0].block_indices,
              Eq(std::vector<int64_t>{0, 0}));
  EXPECT_THAT(summary.block_summaries[0].min, Eq(1.0f));
  // from shard 1 (val = 2.0 + 0.1*j)
  EXPECT_THAT(summary.block_summaries[3].block_indices,
              Eq(std::vector<int64_t>{0, 3}));
  EXPECT_THAT(summary.block_summaries[3].min, Eq(2.0f));
  // from shard 2 (val = 3.0 + 0.1*j)
  EXPECT_THAT(summary.block_summaries[12].block_indices,
              Eq(std::vector<int64_t>{2, 0}));
  EXPECT_THAT(summary.block_summaries[12].min, Eq(3.0f));
  // from shard 3 (val = 4.0 + 0.1*j)
  EXPECT_THAT(summary.block_summaries[15].block_indices,
              Eq(std::vector<int64_t>{2, 3}));
  EXPECT_THAT(summary.block_summaries[15].min, Eq(4.0f));
  // Last block
  EXPECT_THAT(summary.block_summaries[23].block_indices,
              Eq(std::vector<int64_t>{3, 5}));
  EXPECT_THAT(summary.block_summaries[23].min, Eq(4.5f));
  auto processing_metrics = calculator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_optimized_tensor_shard_count, 4);
  EXPECT_EQ(processing_metrics.processed_original_tensor_shard_count, 4);
  EXPECT_EQ(processing_metrics.completed_optimized_tensor_count, 1);
  EXPECT_EQ(processing_metrics.completed_original_tensor_count, 1);
  EXPECT_EQ(processing_metrics.incomplete_optimized_tensor_count, 0);
  EXPECT_EQ(processing_metrics.incomplete_original_tensor_count, 0);
}

TEST(OriginalTensorSummaryCalculatorTest, PropagateIterationIndexToVariable) {
  const TensorKey opt_tensor_key{/*instruction_name=*/"opt_instr"};
  absl::flat_hash_map<TensorKey, std::vector<int64_t>> opt_dims = {
      {opt_tensor_key, {2, 2}}};

  // call_map: while_opt -> [while_orig] where while_orig has iteration_index =
  // -2, which means this is a placeholding variable.
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map = {
      {"while_opt", {ScopeInstruction::Create("while_orig", -2)}}};

  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      orig_map = {{opt_tensor_key,
                   {{/*original_scoped_tensor_key=*/ScopedTensorKey{
                         /*tensor_key=*/TensorKey{/*instruction_name=*/
                                                  "orig_instr"}},
                     /*tensor_transformation=*/nullptr}}}};

  std::vector<CallbackResult> results;
  OriginalTensorSummaryCalculator calculator(
      std::make_shared<
          const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>(
          std::move(opt_dims)),
      std::make_shared<const absl::flat_hash_map<
          std::string, std::vector<ScopeInstruction>>>(std::move(call_map)),
      std::make_shared<const absl::flat_hash_map<
          TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>(
          std::move(orig_map)),
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const TensorTransformation> pending,
          const OriginalTensorSummary& summary) {
        results.push_back({key, std::move(pending), summary});
        return absl::OkStatus();
      });

  // Optimized tensor has scope [while_opt] with iteration index 5
  AbsoluteScopedTensorKey opt_key{
      /*scope_instructions=*/{ScopeInstruction::Create("while_opt", 5)},
      /*tensor_key=*/opt_tensor_key,
  };

  std::vector<DimSplitSpec> split_spec = {};
  FloatSummary shard_summary = CreateSummary({1.0f}, split_spec);
  ShardTensorSummary shard = {0, shard_summary};

  ASSERT_THAT(calculator.ProcessShardSummary(opt_key, shard), IsOk());
  ASSERT_THAT(results, SizeIs(1));

  // The resulting original tensor key should have scope [while_orig] with
  // iteration index 5.
  EXPECT_THAT(results[0].original_tensor_key.tensor_key.instruction_name,
              Eq("orig_instr"));
  EXPECT_THAT(results[0].original_tensor_key.scope_instructions,
              Eq(std::vector<ScopeInstruction>{
                  ScopeInstruction::Create("while_orig", 5)}));
}

TEST(OriginalTensorSummaryCalculatorTest, PropagateIterationIndexToWhileLoop) {
  const TensorKey opt_tensor_key{/*instruction_name=*/"opt_instr"};
  absl::flat_hash_map<TensorKey, std::vector<int64_t>> opt_dims = {
      {opt_tensor_key, {2, 2}}};

  // call_map: while_opt -> [scope1, while.orig]
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map = {
      {"while_opt",
       {ScopeInstruction::Create("scope1"),
        ScopeInstruction::Create("while.orig", 0)}}};

  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      orig_map = {{opt_tensor_key,
                   {{/*original_scoped_tensor_key=*/ScopedTensorKey{
                         /*tensor_key=*/TensorKey{/*instruction_name=*/
                                                  "orig_instr"}},
                     /*tensor_transformation=*/nullptr}}}};

  std::vector<CallbackResult> results;
  OriginalTensorSummaryCalculator calculator(
      std::make_shared<
          const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>(
          std::move(opt_dims)),
      std::make_shared<const absl::flat_hash_map<
          std::string, std::vector<ScopeInstruction>>>(std::move(call_map)),
      std::make_shared<const absl::flat_hash_map<
          TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>(
          std::move(orig_map)),
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const TensorTransformation> pending,
          const OriginalTensorSummary& summary) {
        results.push_back({key, std::move(pending), summary});
        return absl::OkStatus();
      });

  // Optimized tensor has scope [while_opt] with iteration index 5
  AbsoluteScopedTensorKey opt_key{
      /*scope_instructions=*/{ScopeInstruction::Create("while_opt", 5)},
      /*tensor_key=*/opt_tensor_key,
  };

  std::vector<DimSplitSpec> split_spec = {};
  FloatSummary shard_summary = CreateSummary({1.0f}, split_spec);
  ShardTensorSummary shard = {0, shard_summary};

  ASSERT_THAT(calculator.ProcessShardSummary(opt_key, shard), IsOk());
  ASSERT_THAT(results, SizeIs(1));

  // The resulting original tensor key should have scope [scope1, while.orig]
  // with while.orig having iteration index 5.
  EXPECT_THAT(results[0].original_tensor_key.tensor_key.instruction_name,
              Eq("orig_instr"));
  EXPECT_THAT(results[0].original_tensor_key.scope_instructions,
              Eq(std::vector<ScopeInstruction>{
                  ScopeInstruction::Create("scope1"),
                  ScopeInstruction::Create("while.orig", 5)}));
}

TEST(OriginalTensorSummaryCalculatorTest, PropagateIterationIndexToLastScope) {
  const TensorKey opt_tensor_key{/*instruction_name=*/"opt_instr"};
  absl::flat_hash_map<TensorKey, std::vector<int64_t>> opt_dims = {
      {opt_tensor_key, {2, 2}}};

  // call_map: while_opt -> [scope1, scope2]
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map = {
      {"while_opt",
       {ScopeInstruction::Create("scope1"),
        ScopeInstruction::Create("scope2")}}};

  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      orig_map = {{opt_tensor_key,
                   {{/*original_scoped_tensor_key=*/ScopedTensorKey{
                         /*tensor_key=*/TensorKey{/*instruction_name=*/
                                                  "orig_instr"}},
                     /*tensor_transformation=*/nullptr}}}};

  std::vector<CallbackResult> results;
  OriginalTensorSummaryCalculator calculator(
      std::make_shared<
          const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>(
          std::move(opt_dims)),
      std::make_shared<const absl::flat_hash_map<
          std::string, std::vector<ScopeInstruction>>>(std::move(call_map)),
      std::make_shared<const absl::flat_hash_map<
          TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>(
          std::move(orig_map)),
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const TensorTransformation> pending,
          const OriginalTensorSummary& summary) {
        results.push_back({key, std::move(pending), summary});
        return absl::OkStatus();
      });

  // Optimized tensor has scope [while_opt] with iteration index 5
  AbsoluteScopedTensorKey opt_key{
      /*scope_instructions=*/{ScopeInstruction::Create("while_opt", 5)},
      /*tensor_key=*/opt_tensor_key,
  };

  std::vector<DimSplitSpec> split_spec = {};
  FloatSummary shard_summary = CreateSummary({1.0f}, split_spec);
  ShardTensorSummary shard = {0, shard_summary};

  ASSERT_THAT(calculator.ProcessShardSummary(opt_key, shard), IsOk());
  ASSERT_THAT(results, SizeIs(1));

  // The resulting original tensor key should have scope [scope1, scope2] with
  // scope2 having iteration index 5.
  EXPECT_THAT(results[0].original_tensor_key.tensor_key.instruction_name,
              Eq("orig_instr"));
  EXPECT_THAT(
      results[0].original_tensor_key.scope_instructions,
      Eq(std::vector<ScopeInstruction>{ScopeInstruction::Create("scope1"),
                                       ScopeInstruction::Create("scope2", 5)}));
}

TEST_F(OriginalTensorSummaryCalculatorCreateTest,
       CallMapSingleCallerByHeuristics) {
  constexpr absl::string_view optimized_hlo = R"hlo(
HloModule optimized_module

%called_computation (p: s32[]) -> s32[] {
  %p = s32[] parameter(0)
  ROOT %add = s32[] add(%p, %p), origin={{"orig_add"}}
}

ENTRY %main (param: s32[]) -> s32[] {
  %param = s32[] parameter(0)
  ROOT %call = s32[] call(%param), to_apply=%called_computation
}
)hlo";
  constexpr absl::string_view original_hlo = R"hlo(
HloModule original_module

%orig_called_computation (p: s32[]) -> s32[] {
  %p = s32[] parameter(0)
  ROOT %orig_add = s32[] add(%p, %p)
}

ENTRY %orig_main (p: s32[]) -> s32[] {
  %p = s32[] parameter(0)
  ROOT %orig_call = s32[] call(%p), to_apply=%orig_called_computation
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       ParseAndReturnVerifiedModule(optimized_hlo));
  ASSERT_OK_AND_ASSIGN(auto original_module,
                       ParseAndReturnVerifiedModule(original_hlo));
  ASSERT_OK_AND_ASSIGN(
      auto calculator_with_metrics,
      OriginalTensorSummaryCalculator::Create(
          optimized_module.get(), original_module.get(), std::move(callback_)));
  auto& calculator = calculator_with_metrics.first;
  EXPECT_EQ(calculator->DumpHloDerivedData(),
            R"(Optimized Tensor Dimensions:
  add: []

Call Map:
  call: [orig_call]

Original Tensor by Optimized Tensor Key:
  add:
    orig_add via no transformation
)");
}

TEST_F(OriginalTensorSummaryCalculatorCreateTest,
       CallMapMultipleCallersByHeuristics) {
  constexpr absl::string_view optimized_hlo = R"hlo(
HloModule optimized_module

%called_computation (p: s32[]) -> s32[] {
  %p = s32[] parameter(0)
  ROOT %add = s32[] add(%p, %p), origin={{"orig_add"}}
}

ENTRY %main (param: s32[]) -> s32[] {
  %param = s32[] parameter(0)
  ROOT %call = s32[] call(%param), to_apply=%called_computation
}
)hlo";
  constexpr absl::string_view original_hlo = R"hlo(
HloModule original_module

%orig_called_computation (p: s32[]) -> s32[] {
  %p = s32[] parameter(0)
  ROOT %orig_add = s32[] add(%p, %p)
}

ENTRY %orig_main (p: s32[]) -> s32[] {
  %p = s32[] parameter(0)
  %orig_call1 = s32[] call(%p), to_apply=%orig_called_computation
  ROOT %orig_call2 = s32[] call(%p), to_apply=%orig_called_computation
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       ParseAndReturnVerifiedModule(optimized_hlo));
  ASSERT_OK_AND_ASSIGN(auto original_module,
                       ParseAndReturnVerifiedModule(original_hlo));
  ASSERT_OK_AND_ASSIGN(
      auto calculator_with_metrics,
      OriginalTensorSummaryCalculator::Create(
          optimized_module.get(), original_module.get(), std::move(callback_)));
  auto& calculator = calculator_with_metrics.first;
  // Heuristic should not trigger because there are multiple callers to
  // %orig_called_computation.
  EXPECT_EQ(calculator->DumpHloDerivedData(),
            R"(Optimized Tensor Dimensions:
  add: []

Call Map:

Original Tensor by Optimized Tensor Key:
  add:
    orig_add via no transformation
)");
}

TEST_F(OriginalTensorSummaryCalculatorCreateTest, CallMapNoCallerByHeuristics) {
  constexpr absl::string_view optimized_hlo = R"hlo(
HloModule optimized_module

%called_computation (p: s32[]) -> s32[] {
  %p = s32[] parameter(0)
  ROOT %add = s32[] add(%p, %p), origin={{"orig_add"}}
}

ENTRY %main (param: s32[]) -> s32[] {
  %param = s32[] parameter(0)
  ROOT %call = s32[] call(%param), to_apply=%called_computation
}
)hlo";
  constexpr absl::string_view original_hlo = R"hlo(
HloModule original_module

%orig_called_computation (p: s32[]) -> s32[] {
  %p = s32[] parameter(0)
  ROOT %orig_add = s32[] add(%p, %p)
}

ENTRY %orig_main (p: s32[]) -> s32[] {
  %p = s32[] parameter(0)
  ROOT %p_ret = s32[] add(%p, %p)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       ParseAndReturnVerifiedModule(optimized_hlo));
  ASSERT_OK_AND_ASSIGN(auto original_module,
                       ParseAndReturnVerifiedModule(original_hlo));
  ASSERT_OK_AND_ASSIGN(
      auto calculator_with_metrics,
      OriginalTensorSummaryCalculator::Create(
          optimized_module.get(), original_module.get(), std::move(callback_)));
  auto& calculator = calculator_with_metrics.first;
  // Heuristic should not trigger because there are no callers to
  // %orig_called_computation.
  EXPECT_EQ(calculator->DumpHloDerivedData(),
            R"(Optimized Tensor Dimensions:
  add: []

Call Map:

Original Tensor by Optimized Tensor Key:
  add:
    orig_add via no transformation
)");
}

TEST_F(OriginalTensorSummaryCalculatorCreateTest,
       CallMapNoOriginalInstructionInCalledComputationByHeuristics) {
  constexpr absl::string_view optimized_hlo = R"hlo(
HloModule optimized_module

%called_computation (p: s32[]) -> s32[] {
  %p = s32[] parameter(0)
  ROOT %add = s32[] add(%p, %p)
}

ENTRY %main (param: s32[]) -> s32[] {
  %param = s32[] parameter(0)
  ROOT %call = s32[] call(%param), to_apply=%called_computation
}
)hlo";
  constexpr absl::string_view original_hlo = R"hlo(
HloModule original_module

%orig_called_computation (p: s32[]) -> s32[] {
  %p = s32[] parameter(0)
  ROOT %orig_add = s32[] add(%p, %p)
}

ENTRY %orig_main (p: s32[]) -> s32[] {
  %p = s32[] parameter(0)
  ROOT %orig_call = s32[] call(%p), to_apply=%orig_called_computation
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       ParseAndReturnVerifiedModule(optimized_hlo));
  ASSERT_OK_AND_ASSIGN(auto original_module,
                       ParseAndReturnVerifiedModule(original_hlo));
  ASSERT_OK_AND_ASSIGN(
      auto calculator_with_metrics,
      OriginalTensorSummaryCalculator::Create(
          optimized_module.get(), original_module.get(), std::move(callback_)));
  auto& calculator = calculator_with_metrics.first;
  // Heuristic should not trigger because %add has no origin.
  EXPECT_EQ(calculator->DumpHloDerivedData(),
            R"(Optimized Tensor Dimensions:

Call Map:

Original Tensor by Optimized Tensor Key:
)");
}

TEST_F(OriginalTensorSummaryCalculatorCreateTest,
       CallMapWithNestedCallByHeuristics) {
  constexpr absl::string_view optimized_hlo = R"hlo(
HloModule optimized_module

%inner_called_computation(p: s32[]) -> s32[] {
  %p0 = s32[] parameter(0)
  ROOT %inner_add = s32[] add(%p0, %p0), origin={{"orig_inner_add"}}
}

%called_computation (p: s32[]) -> s32[] {
  %p1 = s32[] parameter(0)
  ROOT %inner_call = s32[] call(%p1), to_apply=%inner_called_computation
}

ENTRY %main (param: s32[]) -> s32[] {
  %param = s32[] parameter(0)
  ROOT %call = s32[] call(%param), to_apply=%called_computation
}
)hlo";
  constexpr absl::string_view original_hlo = R"hlo(
HloModule original_module

%orig_inner_called_computation(p: s32[]) -> s32[] {
  %p0 = s32[] parameter(0)
  ROOT %orig_inner_add = s32[] add(%p0, %p0)
}

%orig_called_computation (p: s32[]) -> s32[] {
  %p1 = s32[] parameter(0)
  ROOT %orig_inner_call = s32[] call(%p1), to_apply=%orig_inner_called_computation
}

ENTRY %orig_main (p: s32[]) -> s32[] {
  %p = s32[] parameter(0)
  ROOT %orig_call = s32[] call(%p), to_apply=%orig_called_computation
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       ParseAndReturnVerifiedModule(optimized_hlo));
  ASSERT_OK_AND_ASSIGN(auto original_module,
                       ParseAndReturnVerifiedModule(original_hlo));
  ASSERT_OK_AND_ASSIGN(
      auto calculator_with_metrics,
      OriginalTensorSummaryCalculator::Create(
          optimized_module.get(), original_module.get(), std::move(callback_)));
  auto& calculator = calculator_with_metrics.first;
  EXPECT_EQ(calculator->DumpHloDerivedData(),
            R"(Optimized Tensor Dimensions:
  inner_add: []

Call Map:
  call: [orig_call]
  inner_call: [orig_inner_call]

Original Tensor by Optimized Tensor Key:
  inner_add:
    orig_inner_add via no transformation
)");
}

}  // namespace
}  // namespace xla::numerics::comparison
