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

#include "xla/hlo/utils/hlo_original_value_reconstructor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/utils/hlo_original_value_analysis.h"
#include "xla/hlo/utils/hlo_original_value_analyzer_utils.h"
#include "xla/hlo/utils/hlo_sharding_reconstruction_util.h"
#include "xla/literal.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

class HloOriginalValueReconstructorTest
    : public HloHardwareIndependentTestBase {};

TEST_F(HloOriginalValueReconstructorTest, ChainedRecoveryWithUnshard) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule chained_module, entry_computation_layout={(f32[4,8]{1,0})->f32[4,8]{1,0}}, origin_recovery_table={
  {"original"} : {"__ovp_1"},
  "
    ENTRY %recovery_1 (p: f32[2,16]) -> f32[4,8] {
      %p = f32[2,16]{1,0} parameter(0)
      ROOT %reshape = f32[4,8]{1,0} reshape(%p)
    }
  "
  {"__ovp_1"} : {"__ovp_2"},
  "
    ENTRY %recovery_2 (p: f32[1,16]) -> f32[2,16] {
      %p = f32[1,16]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}
      ROOT %ag = f32[2,16]{1,0} all-gather(%p), dimensions={0}
    }
  "
  {"__ovp_2"} : {"__ovp_3"},
  "
    ENTRY %recovery_3 (p: f32[2,8]) -> f32[1,16] {
      %p = f32[2,8]{1,0} parameter(0)
      ROOT %reshape = f32[1,16]{1,0} reshape(%p)
    }
  "
  },
  debug_attributes={
    {"original"}:({callback_id=123,partitioned=false})
  }
ENTRY %main (p: f32[4,8]) -> f32[4,8] {
  %p = f32[4,8]{1,0} parameter(0)
  ROOT %opt = f32[4,8]{1,0} add(%p, %p), origin={{"__ovp_3"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));

  Literal recovered_literal;
  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t manual_shard_id) {
        if (recovered_data != nullptr) {
          recovered_literal = recovered_data->Clone();
        }
      };

  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));
  HloOriginalValueReconstructor reconstructor(analysis_shared, callback);

  AbsoluteScopedTensorKey opt_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("opt"), {});

  // Shard 0: [2, 8] filled with 1.0
  Literal literal0(ShapeUtil::MakeShape(F32, {2, 8}));
  literal0.PopulateWithValue(1.0f);
  ShardTensor shard0 = {.logical_shard_id = 0,
                        .data = std::make_shared<Literal>(std::move(literal0))};

  // Shard 1: [2, 8] filled with 2.0
  Literal literal1(ShapeUtil::MakeShape(F32, {2, 8}));
  literal1.PopulateWithValue(2.0f);
  ShardTensor shard1 = {.logical_shard_id = 1,
                        .data = std::make_shared<Literal>(std::move(literal1))};

  ASSERT_OK(reconstructor.ProcessShardTensor(opt_key, std::move(shard0)));
  ASSERT_OK(reconstructor.ProcessShardTensor(opt_key, std::move(shard1)));

  // Recovery chain:
  // 1. recovery_3 applied to shards: f32[2,8] -> f32[1,16]
  //    shard0 becomes all 1.0 [1,16]
  //    shard1 becomes all 2.0 [1,16]
  // 2. recovery_2 (unshard) combines them into f32[2,16]
  //    [[1,1...], [2,2...]]
  // 3. recovery_1 applied to full literal: f32[2,16] -> f32[4,8]
  //    first two rows of output are from shard0 (all 1.0),
  //    last two rows of output are from shard1 (all 2.0).

  EXPECT_THAT(recovered_literal.shape().dimensions(), ElementsAre(4, 8));
  EXPECT_EQ(recovered_literal.Get<float>({0, 0}), 1.0f);
  EXPECT_EQ(recovered_literal.Get<float>({1, 7}), 1.0f);
  EXPECT_EQ(recovered_literal.Get<float>({2, 0}), 2.0f);
  EXPECT_EQ(recovered_literal.Get<float>({3, 7}), 2.0f);
}

TEST_F(HloOriginalValueReconstructorTest, PartitionedMiddleUnshardDropped) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule chained_module, entry_computation_layout={(f32[4,8]{1,0})->f32[4,8]{1,0}}, origin_recovery_table={
  {"original"} : {"__ovp_1"},
  "
    ENTRY %recovery_1 (p: f32[2,16]) -> f32[4,8] {
      %p = f32[2,16]{1,0} parameter(0)
      ROOT %reshape = f32[4,8]{1,0} reshape(%p)
    }
  "
  {"__ovp_1"} : {"__ovp_2"},
  "
    ENTRY %recovery_2 (p: f32[1,16]) -> f32[2,16] {
      %p = f32[1,16]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}
      ROOT %ag = f32[2,16]{1,0} all-gather(%p), dimensions={0}
    }
  "
  {"__ovp_2"} : {"__ovp_3"},
  "
    ENTRY %recovery_3 (p: f32[2,8]) -> f32[1,16] {
      %p = f32[2,8]{1,0} parameter(0)
      ROOT %reshape = f32[1,16]{1,0} reshape(%p)
    }
  "
  },
  debug_attributes={
    {"original"}:({callback_id=123,partitioned=true})
  }
ENTRY %main (p: f32[4,8]) -> f32[4,8] {
  %p = f32[4,8]{1,0} parameter(0)
  ROOT %opt = f32[4,8]{1,0} add(%p, %p), origin={{"__ovp_3"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));

  int call_count = 0;
  std::vector<std::optional<Literal>> received_literals;
  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t partition_id) {
        call_count++;
        if (recovered_data != nullptr) {
          received_literals.push_back(recovered_data->Clone());
        } else {
          received_literals.push_back(std::nullopt);
        }
      };

  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));
  HloOriginalValueReconstructor reconstructor(analysis_shared, callback);

  AbsoluteScopedTensorKey opt_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("opt"), {});

  Literal literal0(ShapeUtil::MakeShape(F32, {2, 8}));
  literal0.PopulateWithValue(1.0f);
  ShardTensor shard0 = {.logical_shard_id = 0,
                        .data = std::make_shared<Literal>(std::move(literal0))};

  Literal literal1(ShapeUtil::MakeShape(F32, {2, 8}));
  literal1.PopulateWithValue(2.0f);
  ShardTensor shard1 = {.logical_shard_id = 1,
                        .data = std::make_shared<Literal>(std::move(literal1))};

  ASSERT_OK(reconstructor.ProcessShardTensor(opt_key, std::move(shard0)));
  ASSERT_OK(reconstructor.ProcessShardTensor(opt_key, std::move(shard1)));

  EXPECT_EQ(call_count, 2);
  EXPECT_FALSE(received_literals[0].has_value());
  EXPECT_FALSE(received_literals[1].has_value());
}

TEST_F(HloOriginalValueReconstructorTest, PartitionedLastUnshardReportShards) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, origin_recovery_table={
  {"original"} : {"__ovp_1"},
  "
    ENTRY %recovery_1 (p: f32[1,16]) -> f32[2,16] {
      %p = f32[1,16]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}
      ROOT %ag = f32[2,16]{1,0} all-gather(%p), dimensions={0}
    }
  "
  {"__ovp_1"} : {"__ovp_2"},
  "
    ENTRY %recovery_2 (p: f32[2,8]) -> f32[1,16] {
      %p = f32[2,8]{1,0} parameter(0)
      ROOT %reshape = f32[1,16]{1,0} reshape(%p)
    }
  "
  },
  debug_attributes={
    {"original"}:({callback_id=123,partitioned=true})
  }
ENTRY %main (p: f32[4,8]) -> f32[4,8] {
  %p = f32[4,8]{1,0} parameter(0)
  ROOT %opt = f32[4,8]{1,0} add(%p, %p), origin={{"__ovp_2"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));

  int call_count = 0;
  std::vector<Literal> recovered_shards;
  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t manual_shard_id) {
        call_count++;
        if (recovered_data != nullptr) {
          recovered_shards.push_back(recovered_data->Clone());
        }
      };

  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));
  HloOriginalValueReconstructor reconstructor(analysis_shared, callback);

  AbsoluteScopedTensorKey opt_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("opt"), {});

  Literal literal0(ShapeUtil::MakeShape(F32, {2, 8}));
  literal0.PopulateWithValue(1.0f);
  ShardTensor shard0 = {.logical_shard_id = 0,
                        .data = std::make_shared<Literal>(std::move(literal0))};

  Literal literal1(ShapeUtil::MakeShape(F32, {2, 8}));
  literal1.PopulateWithValue(2.0f);
  ShardTensor shard1 = {.logical_shard_id = 1,
                        .data = std::make_shared<Literal>(std::move(literal1))};

  ASSERT_OK(reconstructor.ProcessShardTensor(opt_key, std::move(shard0)));
  ASSERT_OK(reconstructor.ProcessShardTensor(opt_key, std::move(shard1)));

  EXPECT_EQ(call_count, 2);
  EXPECT_THAT(recovered_shards[0].shape().dimensions(), ElementsAre(1, 16));
  EXPECT_THAT(recovered_shards[1].shape().dimensions(), ElementsAre(1, 16));
}

TEST_F(HloOriginalValueReconstructorTest, PartitionedNoUnshardReportShards) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, origin_recovery_table={
  {"original"} : {"__ovp_1"},
  "
    ENTRY %recovery_1 (p: f32[2,8]) -> f32[2,8] {
      %p = f32[2,8]{1,0} parameter(0)
      ROOT %neg = f32[2,8]{1,0} negate(%p)
    }
  "
  },
  debug_attributes={
    {"original"}:({callback_id=123,partitioned=true})
  }
ENTRY %main (p: f32[4,8]) -> f32[4,8] {
  %p = f32[4,8]{1,0} parameter(0)
  ROOT %opt = f32[4,8]{1,0} add(%p, %p), sharding={devices=[2,1]<=[2]}, origin={{"__ovp_1"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  module->mutable_config().set_num_partitions(2);

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));

  int call_count = 0;
  std::vector<Literal> recovered_shards;
  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t manual_shard_id) {
        call_count++;
        if (recovered_data != nullptr) {
          recovered_shards.push_back(recovered_data->Clone());
        }
      };

  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));
  HloOriginalValueReconstructor reconstructor(analysis_shared, callback);

  AbsoluteScopedTensorKey opt_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("opt"), {});

  Literal literal0(ShapeUtil::MakeShape(F32, {2, 8}));
  literal0.PopulateWithValue(1.0f);
  ShardTensor shard0 = {.logical_shard_id = 0,
                        .data = std::make_shared<Literal>(std::move(literal0))};

  Literal literal1(ShapeUtil::MakeShape(F32, {2, 8}));
  literal1.PopulateWithValue(2.0f);
  ShardTensor shard1 = {.logical_shard_id = 1,
                        .data = std::make_shared<Literal>(std::move(literal1))};

  ASSERT_OK(reconstructor.ProcessShardTensor(opt_key, std::move(shard0)));
  ASSERT_OK(reconstructor.ProcessShardTensor(opt_key, std::move(shard1)));

  EXPECT_EQ(call_count, 2);
  EXPECT_THAT(recovered_shards[0].shape().dimensions(), ElementsAre(2, 8));
  EXPECT_EQ(recovered_shards[0].Get<float>({0, 0}), -1.0f);
}

TEST_F(HloOriginalValueReconstructorTest, MixedPartitionedAttributes) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, origin_recovery_table={
  {"original"} : {"__ovp_1"},
  "
    ENTRY %recovery_1 (p: f32[1,16]) -> f32[2,16] {
      %p = f32[1,16]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}
      ROOT %ag = f32[2,16]{1,0} all-gather(%p), dimensions={0}
    }
  "
  {"__ovp_1"} : {"__ovp_2"},
  "
    ENTRY %recovery_2 (p: f32[2,8]) -> f32[1,16] {
      %p = f32[2,8]{1,0} parameter(0)
      ROOT %reshape = f32[1,16]{1,0} reshape(%p)
    }
  "
  },
  debug_attributes={
    {"original"}:({callback_id=1,partitioned=false},{callback_id=2,partitioned=true})
  }
ENTRY %main (p: f32[4,8]) -> f32[4,8] {
  %p = f32[4,8]{1,0} parameter(0)
  ROOT %opt = f32[4,8]{1,0} add(%p, %p), origin={{"__ovp_2"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));

  int call_count_false = 0;
  int call_count_true = 0;
  std::vector<Literal> recovered_shards_false;
  std::vector<Literal> recovered_shards_true;
  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t manual_shard_id) {
        for (const auto& attr : debug_attributes) {
          if (attr.partitioned) {
            call_count_true++;
            if (recovered_data != nullptr) {
              recovered_shards_true.push_back(recovered_data->Clone());
            }
          } else {
            call_count_false++;
            if (recovered_data != nullptr) {
              recovered_shards_false.push_back(recovered_data->Clone());
            }
          }
        }
      };

  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));
  HloOriginalValueReconstructor reconstructor(analysis_shared, callback);

  AbsoluteScopedTensorKey opt_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("opt"), {});

  Literal literal0(ShapeUtil::MakeShape(F32, {2, 8}));
  literal0.PopulateWithValue(1.0f);
  ShardTensor shard0 = {.logical_shard_id = 0,
                        .data = std::make_shared<Literal>(std::move(literal0))};

  Literal literal1(ShapeUtil::MakeShape(F32, {2, 8}));
  literal1.PopulateWithValue(2.0f);
  ShardTensor shard1 = {.logical_shard_id = 1,
                        .data = std::make_shared<Literal>(std::move(literal1))};

  ASSERT_OK(reconstructor.ProcessShardTensor(opt_key, std::move(shard0)));
  ASSERT_OK(reconstructor.ProcessShardTensor(opt_key, std::move(shard1)));

  // partitioned=false should be called once (full unshard).
  EXPECT_EQ(call_count_false, 1);
  // partitioned=true should be called twice (once per shard).
  EXPECT_EQ(call_count_true, 2);

  // Assert the content of the recovered literals.
  // For partitioned=false, we expect one full tensor of shape f32[2,16].
  ASSERT_EQ(recovered_shards_false.size(), 1);
  const Literal& full_literal = recovered_shards_false[0];
  EXPECT_THAT(full_literal.shape().dimensions(), ElementsAre(2, 16));
  // The first row should be from shard0 (1.0f).
  EXPECT_EQ(full_literal.Get<float>({0, 0}), 1.0f);
  EXPECT_EQ(full_literal.Get<float>({0, 15}), 1.0f);
  // The second row should be from shard1 (2.0f).
  EXPECT_EQ(full_literal.Get<float>({1, 0}), 2.0f);
  EXPECT_EQ(full_literal.Get<float>({1, 15}), 2.0f);

  // For partitioned=true, we expect two shards of shape f32[1,16].
  ASSERT_EQ(recovered_shards_true.size(), 2);
  // Shard 0 recovery.
  EXPECT_THAT(recovered_shards_true[0].shape().dimensions(),
              ElementsAre(1, 16));
  EXPECT_EQ(recovered_shards_true[0].Get<float>({0, 0}), 1.0f);
  // Shard 1 recovery.
  EXPECT_THAT(recovered_shards_true[1].shape().dimensions(),
              ElementsAre(1, 16));
  EXPECT_EQ(recovered_shards_true[1].Get<float>({0, 0}), 2.0f);
}

TEST_F(HloOriginalValueReconstructorTest, MissingShardsNoCallback) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, origin_recovery_table={
  {"original"} : {"__ovp_1"},
  "
    ENTRY %recovery_1 (p: f32[1,16]) -> f32[2,16] {
      %p = f32[1,16]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}
      ROOT %ag = f32[2,16]{1,0} all-gather(%p), dimensions={0}
    }
  "
  },
  debug_attributes={
    {"original"}:({callback_id=123,partitioned=false})
  }
ENTRY %main (p: f32[4,8]) -> f32[4,8] {
  %p = f32[4,8]{1,0} parameter(0)
  ROOT %opt = f32[4,8]{1,0} add(%p, %p), origin={{"__ovp_1"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));

  int call_count = 0;
  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t manual_shard_id) { call_count++; };

  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));
  HloOriginalValueReconstructor reconstructor(analysis_shared, callback);

  AbsoluteScopedTensorKey opt_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("opt"), {});

  Literal literal0(ShapeUtil::MakeShape(F32, {1, 16}));
  literal0.PopulateWithValue(1.0f);
  ShardTensor shard0 = {.logical_shard_id = 0,
                        .data = std::make_shared<Literal>(std::move(literal0))};

  ASSERT_OK(reconstructor.ProcessShardTensor(opt_key, std::move(shard0)));

  EXPECT_EQ(call_count, 0);
}

TEST_F(HloOriginalValueReconstructorTest, EvaluationErrorPropagated) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, origin_recovery_table={
  {"original"} : {"__ovp_1"},
  "
    ENTRY %recovery_1 (p: f32[2,8]) -> f32[2,8] {
      %p = f32[2,8]{1,0} parameter(0)
      ROOT %custom = f32[2,8]{1,0} custom-call(%p), custom_call_target=\"NonExistentCall\"
    }
  "
  },
  debug_attributes={
    {"original"}:({callback_id=123,partitioned=false})
  }
ENTRY %main (p: f32[4,8]) -> f32[4,8] {
  %p = f32[4,8]{1,0} parameter(0)
  ROOT %opt = f32[4,8]{1,0} add(%p, %p), origin={{"__ovp_1"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));

  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t manual_shard_id) {};

  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));
  HloOriginalValueReconstructor reconstructor(analysis_shared, callback);

  AbsoluteScopedTensorKey opt_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("opt"), {});

  Literal literal0(ShapeUtil::MakeShape(F32, {2, 8}));
  literal0.PopulateWithValue(1.0f);
  ShardTensor shard0 = {.logical_shard_id = 0,
                        .data = std::make_shared<Literal>(std::move(literal0))};

  EXPECT_FALSE(
      reconstructor.ProcessShardTensor(opt_key, std::move(shard0)).ok());
}

TEST_F(HloOriginalValueReconstructorTest, MultiControllerAbort) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule chained_module, entry_computation_layout={(f32[4,8]{1,0})->f32[4,8]{1,0}}, origin_recovery_table={
  {"original"} : {"__ovp_1"},
  "
    ENTRY %recovery_1 (p: f32[2,16]) -> f32[4,8] {
      %p = f32[2,16]{1,0} parameter(0)
      ROOT %reshape = f32[4,8]{1,0} reshape(%p)
    }
  "
  {"__ovp_1"} : {"__ovp_2"},
  "
    ENTRY %recovery_2 (p: f32[1,16]) -> f32[2,16] {
      %p = f32[1,16]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}
      ROOT %ag = f32[2,16]{1,0} all-gather(%p), dimensions={0}
    }
  "
  {"__ovp_2"} : {"__ovp_3"},
  "
    ENTRY %recovery_3 (p: f32[2,8]) -> f32[1,16] {
      %p = f32[2,8]{1,0} parameter(0)
      ROOT %reshape = f32[1,16]{1,0} reshape(%p)
    }
  "
  },
  debug_attributes={
    {"original"}:({callback_id=123,partitioned=false})
  }
ENTRY %main (p: f32[4,8]) -> f32[4,8] {
  %p = f32[4,8]{1,0} parameter(0)
  ROOT %opt = f32[4,8]{1,0} add(%p, %p), origin={{"__ovp_3"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));

  bool callback_called = false;
  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t manual_shard_id) { callback_called = true; };

  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  absl::flat_hash_set<int64_t> addressable_devices = {0};
  DeviceAssignment device_assignment(1, 2);
  device_assignment(0, 0) = 0;
  device_assignment(0, 1) = 1;

  auto logical_device_is_addressable = [&](int64_t logical_device_id) -> bool {
    int64_t physical_dev_id = device_assignment(0, logical_device_id);
    return addressable_devices.contains(physical_dev_id);
  };

  HloOriginalValueReconstructor reconstructor(
      analysis_shared, callback, logical_device_is_addressable, module.get());

  Literal shard0(ShapeUtil::MakeShape(F32, {2, 8}));
  shard0.PopulateWithValue<float>(42.0f);
  AbsoluteScopedTensorKey opt_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("opt"), {});

  ShardTensor shard_tensor{
      .logical_shard_id = 0,
      .data = std::make_shared<Literal>(std::move(shard0))};

  ASSERT_OK(reconstructor.ProcessShardTensor(opt_key, std::move(shard_tensor)));

  EXPECT_FALSE(callback_called);
}

}  // namespace
}  // namespace xla
