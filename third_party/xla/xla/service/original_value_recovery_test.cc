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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/utils/hlo_original_value_analysis.h"
#include "xla/hlo/utils/hlo_original_value_analyzer_utils.h"
#include "xla/hlo/utils/hlo_original_value_reconstructor.h"
#include "xla/hlo/utils/hlo_sharding_reconstruction_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/call_inliner.h"
#include "xla/service/spmd/spmd_partitioner.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

// Tests original value recovery table generation, chaining, and reconstruction
// across XLA compiler passes:
// - SpmdPartitioner: 1D/2D sharding, transposed device assignment, uneven
//   tiling with slice recovery, partial replication (last_tile_dim_replicate),
//   manual subgroups (manual_axes), multi-leaf tuple sharding, and mixed
//   sharded/replicated tuple elements with non-empty ShapeIndex.
// - AlgebraicSimplifier & Chained Passes: reshape + broadcast folding,
//   AlgebraicSimplifier followed by SpmdPartitioner (partitioned=false and
//   partitioned=true), and CallInliner followed by AlgebraicSimplifier
//   preserving call hierarchy scopes.
// - HloOriginalValueReconstructor: partitioned=true per-shard callbacks, mixed
//   partitioned=false and partitioned=true debug attributes, and multi-
//   controller unaddressable device early abort.
class OriginalValueRecoveryTest : public HloHardwareIndependentTestBase {
 protected:
  // Helper to locate the optimized tensor key in the analysis that corresponds
  // to an original instruction name (optionally scoped) and shape index.
  AbsoluteScopedTensorKey FindOptimizedTensorKey(
      const HloOriginalValueAnalysis& analysis, absl::string_view original_name,
      const ShapeIndex& original_shape_index = {}) {
    RelativeScopedTensorKey rel_key = RelativeScopedTensorKey::FromString(
        original_name, original_shape_index);
    auto it =
        analysis.original_to_optimized_tensor_map().find(rel_key.tensor_key);
    if (it == analysis.original_to_optimized_tensor_map().end()) {
      return {};
    }
    for (const auto& [opt_key, info] : it->second) {
      if (info->original_scoped_tensor_key == rel_key) {
        return AbsoluteScopedTensorKey::Create(opt_key, {});
      }
    }
    return {};
  }
};

TEST_F(OriginalValueRecoveryTest, SpmdPartitioning1DSharding) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={(f32[4,8]{1,0})->f32[4,8]{1,0}}, num_partitions=2

ENTRY entry (p: f32[4,8]) -> f32[4,8] {
  %p = f32[4,8]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}
  ROOT %neg = f32[4,8]{1,0} negate(%p), sharding={devices=[2,1]<=[2]}, origin={{"neg_origin"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(module->original_value_recovery_table().empty());

  spmd::SpmdPartitionerOptions options;
  options.allow_module_signature_change = true;
  spmd::SpmdPartitioner partitioner(2, 1, options);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(partitioner, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(module->original_value_recovery_table().empty());

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  std::optional<Literal> recovered_literal;
  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t manual_shard_id) {
        if (original_tensor.instruction_name == "neg_origin" &&
            recovered_data != nullptr) {
          recovered_literal = recovered_data->Clone();
        }
      };

  HloOriginalValueReconstructor reconstructor(analysis_shared, callback,
                                              std::nullopt, module.get());

  AbsoluteScopedTensorKey opt_key =
      FindOptimizedTensorKey(*analysis_shared, "neg_origin");
  ASSERT_FALSE(opt_key.tensor_key.instruction_name.empty());

  // Shard 0: [2, 8] filled with 1.0f
  ASSERT_OK_AND_ASSIGN(Literal literal0,
                       Literal::Make(ShapeUtil::MakeShape(F32, {2, 8})));
  literal0.PopulateWithValue(1.0f);
  ShardTensor shard0 = {
      /*logical_shard_id=*/0,
      /*data=*/std::make_shared<Literal>(std::move(literal0))};

  // Shard 1: [2, 8] filled with 2.0f
  ASSERT_OK_AND_ASSIGN(Literal literal1,
                       Literal::Make(ShapeUtil::MakeShape(F32, {2, 8})));
  literal1.PopulateWithValue(2.0f);
  ShardTensor shard1 = {
      /*logical_shard_id=*/1,
      /*data=*/std::make_shared<Literal>(std::move(literal1))};

  ASSERT_OK(reconstructor.ProcessShardTensor(opt_key, std::move(shard0)));
  EXPECT_FALSE(recovered_literal.has_value());

  ASSERT_OK(reconstructor.ProcessShardTensor(opt_key, std::move(shard1)));
  ASSERT_TRUE(recovered_literal.has_value());

  EXPECT_THAT(recovered_literal->shape().dimensions(), ElementsAre(4, 8));
  EXPECT_EQ(recovered_literal->Get<float>({0, 0}), 1.0f);
  EXPECT_EQ(recovered_literal->Get<float>({1, 7}), 1.0f);
  EXPECT_EQ(recovered_literal->Get<float>({2, 0}), 2.0f);
  EXPECT_EQ(recovered_literal->Get<float>({3, 7}), 2.0f);
}

TEST_F(OriginalValueRecoveryTest, SpmdPartitioning2DSharding) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={(f32[4,6]{1,0}, f32[4,6]{1,0})->f32[4,6]{1,0}}, num_partitions=4

ENTRY entry (p0: f32[4,6], p1: f32[4,6]) -> f32[4,6] {
  %p0 = f32[4,6]{1,0} parameter(0), sharding={devices=[2,2]<=[4]}
  %p1 = f32[4,6]{1,0} parameter(1), sharding={devices=[2,2]<=[4]}
  ROOT %add = f32[4,6]{1,0} add(%p0, %p1), sharding={devices=[2,2]<=[4]}, origin={{"add_origin"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(module->original_value_recovery_table().empty());

  spmd::SpmdPartitionerOptions options;
  options.allow_module_signature_change = true;
  spmd::SpmdPartitioner partitioner(4, 1, options);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(partitioner, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(module->original_value_recovery_table().empty());

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  std::optional<Literal> recovered_literal;
  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t manual_shard_id) {
        if (original_tensor.instruction_name == "add_origin" &&
            recovered_data != nullptr) {
          recovered_literal = recovered_data->Clone();
        }
      };

  HloOriginalValueReconstructor reconstructor(analysis_shared, callback,
                                              std::nullopt, module.get());

  AbsoluteScopedTensorKey opt_key =
      FindOptimizedTensorKey(*analysis_shared, "add_origin");
  ASSERT_FALSE(opt_key.tensor_key.instruction_name.empty());

  // Shard shape is [2, 3].
  // Device grid [2, 2] <= [4]:
  // logical_shard_id 0 -> tile (0, 0): top-left
  // logical_shard_id 1 -> tile (0, 1): top-right
  // logical_shard_id 2 -> tile (1, 0): bottom-left
  // logical_shard_id 3 -> tile (1, 1): bottom-right
  for (int64_t shard_id = 0; shard_id < 4; ++shard_id) {
    ASSERT_OK_AND_ASSIGN(Literal shard_data,
                         Literal::Make(ShapeUtil::MakeShape(F32, {2, 3})));
    shard_data.PopulateWithValue(static_cast<float>(shard_id + 1));
    ShardTensor shard = {shard_id,
                         std::make_shared<Literal>(std::move(shard_data))};
    ASSERT_OK(reconstructor.ProcessShardTensor(opt_key, std::move(shard)));
  }

  ASSERT_TRUE(recovered_literal.has_value());
  EXPECT_THAT(recovered_literal->shape().dimensions(), ElementsAre(4, 6));

  // Tile (0, 0) from shard 0 (1.0f)
  EXPECT_EQ(recovered_literal->Get<float>({0, 0}), 1.0f);
  EXPECT_EQ(recovered_literal->Get<float>({1, 2}), 1.0f);
  // Tile (0, 1) from shard 1 (2.0f)
  EXPECT_EQ(recovered_literal->Get<float>({0, 3}), 2.0f);
  EXPECT_EQ(recovered_literal->Get<float>({1, 5}), 2.0f);
  // Tile (1, 0) from shard 2 (3.0f)
  EXPECT_EQ(recovered_literal->Get<float>({2, 0}), 3.0f);
  EXPECT_EQ(recovered_literal->Get<float>({3, 2}), 3.0f);
  // Tile (1, 1) from shard 3 (4.0f)
  EXPECT_EQ(recovered_literal->Get<float>({2, 3}), 4.0f);
  EXPECT_EQ(recovered_literal->Get<float>({3, 5}), 4.0f);
}

TEST_F(OriginalValueRecoveryTest, SpmdPartitioningTransposedSharding) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={(f32[4,6]{1,0}, f32[4,6]{1,0})->f32[4,6]{1,0}}, num_partitions=4

ENTRY entry (p0: f32[4,6], p1: f32[4,6]) -> f32[4,6] {
  %p0 = f32[4,6]{1,0} parameter(0), sharding={devices=[2,2]<=[2,2]T(1,0)}
  %p1 = f32[4,6]{1,0} parameter(1), sharding={devices=[2,2]<=[2,2]T(1,0)}
  ROOT %add = f32[4,6]{1,0} add(%p0, %p1), sharding={devices=[2,2]<=[2,2]T(1,0)}, origin={{"transposed_origin"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(module->original_value_recovery_table().empty());

  spmd::SpmdPartitionerOptions options;
  options.allow_module_signature_change = true;
  spmd::SpmdPartitioner partitioner(4, 1, options);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(partitioner, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(module->original_value_recovery_table().empty());

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  std::optional<Literal> recovered_literal;
  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t manual_shard_id) {
        if (original_tensor.instruction_name == "transposed_origin" &&
            recovered_data != nullptr) {
          recovered_literal = recovered_data->Clone();
        }
      };

  HloOriginalValueReconstructor reconstructor(analysis_shared, callback,
                                              std::nullopt, module.get());

  AbsoluteScopedTensorKey opt_key =
      FindOptimizedTensorKey(*analysis_shared, "transposed_origin");
  ASSERT_FALSE(opt_key.tensor_key.instruction_name.empty());

  // Shard shape is [2, 3].
  // Device assignment is transposed T(1,0):
  // logical_shard_id 0 -> tile (0, 0)
  // logical_shard_id 1 -> tile (1, 0)
  // logical_shard_id 2 -> tile (0, 1)
  // logical_shard_id 3 -> tile (1, 1)
  for (int64_t shard_id = 0; shard_id < 4; ++shard_id) {
    ASSERT_OK_AND_ASSIGN(Literal shard_data,
                         Literal::Make(ShapeUtil::MakeShape(F32, {2, 3})));
    shard_data.PopulateWithValue(static_cast<float>(shard_id + 1));
    ShardTensor shard = {shard_id,
                         std::make_shared<Literal>(std::move(shard_data))};
    ASSERT_OK(reconstructor.ProcessShardTensor(opt_key, std::move(shard)));
  }

  ASSERT_TRUE(recovered_literal.has_value());
  EXPECT_THAT(recovered_literal->shape().dimensions(), ElementsAre(4, 6));

  // Tile (0, 0) from shard 0 (1.0f)
  EXPECT_EQ(recovered_literal->Get<float>({0, 0}), 1.0f);
  EXPECT_EQ(recovered_literal->Get<float>({1, 2}), 1.0f);
  // Tile (1, 0) from shard 1 (2.0f)
  EXPECT_EQ(recovered_literal->Get<float>({2, 0}), 2.0f);
  EXPECT_EQ(recovered_literal->Get<float>({3, 2}), 2.0f);
  // Tile (0, 1) from shard 2 (3.0f)
  EXPECT_EQ(recovered_literal->Get<float>({0, 3}), 3.0f);
  EXPECT_EQ(recovered_literal->Get<float>({1, 5}), 3.0f);
  // Tile (1, 1) from shard 3 (4.0f)
  EXPECT_EQ(recovered_literal->Get<float>({2, 3}), 4.0f);
  EXPECT_EQ(recovered_literal->Get<float>({3, 5}), 4.0f);
}

TEST_F(OriginalValueRecoveryTest, SpmdPartitioningUnevenTilingWithSlice) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={(f32[3,4]{1,0})->f32[3,4]{1,0}}, num_partitions=2

ENTRY entry (p: f32[3,4]) -> f32[3,4] {
  %p = f32[3,4]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}
  ROOT %neg = f32[3,4]{1,0} negate(%p), sharding={devices=[2,1]<=[2]}, origin={{"uneven_origin"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(module->original_value_recovery_table().empty());

  spmd::SpmdPartitionerOptions options;
  options.allow_module_signature_change = true;
  spmd::SpmdPartitioner partitioner(2, 1, options);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(partitioner, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(module->original_value_recovery_table().empty());

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  std::optional<Literal> recovered_literal;
  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t manual_shard_id) {
        if (original_tensor.instruction_name == "uneven_origin" &&
            recovered_data != nullptr) {
          recovered_literal = recovered_data->Clone();
        }
      };

  HloOriginalValueReconstructor reconstructor(analysis_shared, callback,
                                              std::nullopt, module.get());

  AbsoluteScopedTensorKey opt_key =
      FindOptimizedTensorKey(*analysis_shared, "uneven_origin");
  ASSERT_FALSE(opt_key.tensor_key.instruction_name.empty());

  // Shard shape is ceil(3/2) = 2, so shard shape is [2, 4].
  ASSERT_OK_AND_ASSIGN(Literal literal0,
                       Literal::Make(ShapeUtil::MakeShape(F32, {2, 4})));
  literal0.PopulateWithValue(5.0f);
  ShardTensor shard0 = {
      /*logical_shard_id=*/0,
      /*data=*/std::make_shared<Literal>(std::move(literal0))};

  ASSERT_OK_AND_ASSIGN(Literal literal1,
                       Literal::Make(ShapeUtil::MakeShape(F32, {2, 4})));
  literal1.PopulateWithValue(9.0f);
  ShardTensor shard1 = {
      /*logical_shard_id=*/1,
      /*data=*/std::make_shared<Literal>(std::move(literal1))};

  ASSERT_OK(reconstructor.ProcessShardTensor(opt_key, std::move(shard0)));
  ASSERT_OK(reconstructor.ProcessShardTensor(opt_key, std::move(shard1)));

  ASSERT_TRUE(recovered_literal.has_value());
  // The recovered literal should be sliced back to original shape [3, 4]
  EXPECT_THAT(recovered_literal->shape().dimensions(), ElementsAre(3, 4));
  EXPECT_EQ(recovered_literal->Get<float>({0, 0}), 5.0f);
  EXPECT_EQ(recovered_literal->Get<float>({1, 3}), 5.0f);
  EXPECT_EQ(recovered_literal->Get<float>({2, 0}), 9.0f);
  EXPECT_EQ(recovered_literal->Get<float>({2, 3}), 9.0f);
}

TEST_F(OriginalValueRecoveryTest, SpmdPartitioningTupleSharding) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={(f32[4,4]{1,0}, f32[4,8]{1,0})->(f32[4,4]{1,0}, f32[4,8]{1,0})}, num_partitions=2

ENTRY entry (x: f32[4,4], y: f32[4,8]) -> (f32[4,4], f32[4,8]) {
  %x = f32[4,4]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}
  %y = f32[4,8]{1,0} parameter(1), sharding={devices=[2,1]<=[2]}
  ROOT %tuple = (f32[4,4]{1,0}, f32[4,8]{1,0}) tuple(%x, %y),
    sharding={{devices=[2,1]<=[2]}, {devices=[2,1]<=[2]}},
    origin={({"tuple_x"}, {"tuple_y"})}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(module->original_value_recovery_table().empty());

  spmd::SpmdPartitionerOptions options;
  options.allow_module_signature_change = true;
  spmd::SpmdPartitioner partitioner(2, 1, options);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(partitioner, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(module->original_value_recovery_table().empty());

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  std::optional<Literal> recovered_x;
  std::optional<Literal> recovered_y;
  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t manual_shard_id) {
        if (recovered_data != nullptr) {
          if (original_tensor.instruction_name == "tuple_x") {
            recovered_x = recovered_data->Clone();
          } else if (original_tensor.instruction_name == "tuple_y") {
            recovered_y = recovered_data->Clone();
          }
        }
      };

  HloOriginalValueReconstructor reconstructor(analysis_shared, callback,
                                              std::nullopt, module.get());

  AbsoluteScopedTensorKey opt_key_x =
      FindOptimizedTensorKey(*analysis_shared, "tuple_x");
  ASSERT_FALSE(opt_key_x.tensor_key.instruction_name.empty());

  AbsoluteScopedTensorKey opt_key_y =
      FindOptimizedTensorKey(*analysis_shared, "tuple_y");
  ASSERT_FALSE(opt_key_y.tensor_key.instruction_name.empty());

  // Shards for tuple_x: shape [2, 4]
  ASSERT_OK_AND_ASSIGN(Literal x_shard0,
                       Literal::Make(ShapeUtil::MakeShape(F32, {2, 4})));
  x_shard0.PopulateWithValue(10.0f);
  ASSERT_OK_AND_ASSIGN(Literal x_shard1,
                       Literal::Make(ShapeUtil::MakeShape(F32, {2, 4})));
  x_shard1.PopulateWithValue(20.0f);

  ASSERT_OK(reconstructor.ProcessShardTensor(
      opt_key_x, {0, std::make_shared<Literal>(std::move(x_shard0))}));
  ASSERT_OK(reconstructor.ProcessShardTensor(
      opt_key_x, {1, std::make_shared<Literal>(std::move(x_shard1))}));

  ASSERT_TRUE(recovered_x.has_value());
  EXPECT_THAT(recovered_x->shape().dimensions(), ElementsAre(4, 4));
  EXPECT_EQ(recovered_x->Get<float>({0, 0}), 10.0f);
  EXPECT_EQ(recovered_x->Get<float>({2, 0}), 20.0f);

  // Shards for tuple_y: shape [2, 8]
  ASSERT_OK_AND_ASSIGN(Literal y_shard0,
                       Literal::Make(ShapeUtil::MakeShape(F32, {2, 8})));
  y_shard0.PopulateWithValue(30.0f);
  ASSERT_OK_AND_ASSIGN(Literal y_shard1,
                       Literal::Make(ShapeUtil::MakeShape(F32, {2, 8})));
  y_shard1.PopulateWithValue(40.0f);

  ASSERT_OK(reconstructor.ProcessShardTensor(
      opt_key_y, {0, std::make_shared<Literal>(std::move(y_shard0))}));
  ASSERT_OK(reconstructor.ProcessShardTensor(
      opt_key_y, {1, std::make_shared<Literal>(std::move(y_shard1))}));

  ASSERT_TRUE(recovered_y.has_value());
  EXPECT_THAT(recovered_y->shape().dimensions(), ElementsAre(4, 8));
  EXPECT_EQ(recovered_y->Get<float>({0, 0}), 30.0f);
  EXPECT_EQ(recovered_y->Get<float>({2, 0}), 40.0f);
}

TEST_F(OriginalValueRecoveryTest, SpmdPartitioningTupleWithReplicatedElement) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={(f32[2]{0}, f32[4,4]{1,0})->(f32[2]{0}, f32[4,4]{1,0})}, num_partitions=2

ENTRY entry (x: f32[2], y: f32[4,4]) -> (f32[2], f32[4,4]) {
  %x = f32[2]{0} parameter(0), sharding={replicated}
  %y = f32[4,4]{1,0} parameter(1), sharding={devices=[2,1]<=[2]}
  ROOT %tuple = (f32[2]{0}, f32[4,4]{1,0}) tuple(%x, %y),
    sharding={{replicated}, {devices=[2,1]<=[2]}},
    origin={({"tuple_origin" {0}}, {"tuple_origin" {1}})}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(module->original_value_recovery_table().empty());

  spmd::SpmdPartitionerOptions options;
  options.allow_module_signature_change = true;
  spmd::SpmdPartitioner partitioner(2, 1, options);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(partitioner, module.get()));
  EXPECT_TRUE(changed);
  // Only the sharded element (index {1}) requires a recovery computation;
  // the replicated element (index {0}) is directly propagated.
  EXPECT_EQ(module->original_value_recovery_table().size(), 1);

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  std::optional<Literal> recovered_x;
  std::optional<Literal> recovered_y;
  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t manual_shard_id) {
        if (original_tensor.instruction_name == "tuple_origin" &&
            recovered_data != nullptr) {
          if (original_tensor.shape_index == ShapeIndex({0})) {
            recovered_x = recovered_data->Clone();
          } else if (original_tensor.shape_index == ShapeIndex({1})) {
            recovered_y = recovered_data->Clone();
          }
        }
      };

  HloOriginalValueReconstructor reconstructor(analysis_shared, callback,
                                              std::nullopt, module.get());

  // Reconstruct the replicated tuple element {0} from a single shard.
  AbsoluteScopedTensorKey opt_key_x =
      FindOptimizedTensorKey(*analysis_shared, "tuple_origin", {0});
  ASSERT_FALSE(opt_key_x.tensor_key.instruction_name.empty());

  Literal x_shard0 = LiteralUtil::CreateR1<float>({3.0f, 6.0f});
  ASSERT_OK(reconstructor.ProcessShardTensor(
      opt_key_x, {0, std::make_shared<Literal>(std::move(x_shard0))}));
  ASSERT_TRUE(recovered_x.has_value());
  EXPECT_THAT(recovered_x->shape().dimensions(), ElementsAre(2));
  EXPECT_EQ(recovered_x->Get<float>({0}), 3.0f);
  EXPECT_EQ(recovered_x->Get<float>({1}), 6.0f);

  // Reconstruct the sharded tuple element {1} from both shards.
  AbsoluteScopedTensorKey opt_key_y =
      FindOptimizedTensorKey(*analysis_shared, "tuple_origin", {1});
  ASSERT_FALSE(opt_key_y.tensor_key.instruction_name.empty());

  ASSERT_OK_AND_ASSIGN(Literal y_shard0,
                       Literal::Make(ShapeUtil::MakeShape(F32, {2, 4})));
  y_shard0.PopulateWithValue(7.0f);
  ASSERT_OK_AND_ASSIGN(Literal y_shard1,
                       Literal::Make(ShapeUtil::MakeShape(F32, {2, 4})));
  y_shard1.PopulateWithValue(14.0f);

  ASSERT_OK(reconstructor.ProcessShardTensor(
      opt_key_y, {0, std::make_shared<Literal>(std::move(y_shard0))}));
  ASSERT_OK(reconstructor.ProcessShardTensor(
      opt_key_y, {1, std::make_shared<Literal>(std::move(y_shard1))}));

  ASSERT_TRUE(recovered_y.has_value());
  EXPECT_THAT(recovered_y->shape().dimensions(), ElementsAre(4, 4));
  EXPECT_EQ(recovered_y->Get<float>({0, 0}), 7.0f);
  EXPECT_EQ(recovered_y->Get<float>({2, 0}), 14.0f);
}

TEST_F(OriginalValueRecoveryTest, SpmdPartitioningPartiallyReplicatedSharding) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={(f32[4,6]{1,0})->f32[4,6]{1,0}}, num_partitions=4

ENTRY entry (p: f32[4,6]) -> f32[4,6] {
  %p = f32[4,6]{1,0} parameter(0), sharding={devices=[2,1,2]<=[4] last_tile_dim_replicate}
  ROOT %neg = f32[4,6]{1,0} negate(%p), sharding={devices=[2,1,2]<=[4] last_tile_dim_replicate}, origin={{"partial_rep_origin"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(module->original_value_recovery_table().empty());

  spmd::SpmdPartitionerOptions options;
  options.allow_module_signature_change = true;
  spmd::SpmdPartitioner partitioner(4, 1, options);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(partitioner, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(module->original_value_recovery_table().empty());

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  std::optional<Literal> recovered_literal;
  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t manual_shard_id) {
        if (original_tensor.instruction_name == "partial_rep_origin" &&
            recovered_data != nullptr) {
          recovered_literal = recovered_data->Clone();
        }
      };

  HloOriginalValueReconstructor reconstructor(analysis_shared, callback,
                                              std::nullopt, module.get());

  AbsoluteScopedTensorKey opt_key =
      FindOptimizedTensorKey(*analysis_shared, "partial_rep_origin");
  ASSERT_FALSE(opt_key.tensor_key.instruction_name.empty());

  // Devices {0, 1} hold tile 0 (11.0f), devices {2, 3} hold tile 1 (22.0f).
  for (int64_t shard_id = 0; shard_id < 4; ++shard_id) {
    ASSERT_OK_AND_ASSIGN(Literal shard_data,
                         Literal::Make(ShapeUtil::MakeShape(F32, {2, 6})));
    shard_data.PopulateWithValue(shard_id < 2 ? 11.0f : 22.0f);
    ASSERT_OK(reconstructor.ProcessShardTensor(
        opt_key, {shard_id, std::make_shared<Literal>(std::move(shard_data))}));
  }

  ASSERT_TRUE(recovered_literal.has_value());
  EXPECT_THAT(recovered_literal->shape().dimensions(), ElementsAre(4, 6));
  EXPECT_EQ(recovered_literal->Get<float>({0, 0}), 11.0f);
  EXPECT_EQ(recovered_literal->Get<float>({1, 5}), 11.0f);
  EXPECT_EQ(recovered_literal->Get<float>({2, 0}), 22.0f);
  EXPECT_EQ(recovered_literal->Get<float>({3, 5}), 22.0f);
}

TEST_F(OriginalValueRecoveryTest, SpmdPartitioningManualSubgroupSharding) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={(f32[8,6]{1,0})->f32[8,6]{1,0}}, num_partitions=4

ENTRY entry (p: f32[8,6]) -> f32[8,6] {
  %p = f32[8,6]{1,0} parameter(0), sharding={mesh['x'=2,'y'=2], [{'y'}, {}], manual={'x'}}
  ROOT %neg = f32[8,6]{1,0} negate(%p), sharding={mesh['x'=2,'y'=2], [{'y'}, {}], manual={'x'}}, origin={{"manual_subgroup_origin"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(module->original_value_recovery_table().empty());

  spmd::SpmdPartitionerOptions options;
  options.allow_module_signature_change = true;
  spmd::SpmdPartitioner partitioner(4, 1, options);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(partitioner, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(module->original_value_recovery_table().empty());

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  absl::flat_hash_map<int64_t, Literal> recovered_by_manual_id;
  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t manual_shard_id) {
        if (original_tensor.instruction_name == "manual_subgroup_origin" &&
            recovered_data != nullptr) {
          recovered_by_manual_id[manual_shard_id] = recovered_data->Clone();
        }
      };

  HloOriginalValueReconstructor reconstructor(analysis_shared, callback,
                                              std::nullopt, module.get());

  AbsoluteScopedTensorKey opt_key =
      FindOptimizedTensorKey(*analysis_shared, "manual_subgroup_origin");
  ASSERT_FALSE(opt_key.tensor_key.instruction_name.empty());

  // Per-device shard shape is [2, 6] (8 divided by 2 tiled * 2 manual = 2).
  // mesh['x'=2,'y'=2], [{'y'}, {}], manual={'x'}:
  //   manual_id 0 (x=0): device 0 (y=0, tile 0) and device 1 (y=1, tile 1)
  //   manual_id 1 (x=1): device 2 (y=0, tile 0) and device 3 (y=1, tile 1)
  for (int64_t shard_id = 0; shard_id < 4; ++shard_id) {
    ASSERT_OK_AND_ASSIGN(Literal shard_data,
                         Literal::Make(ShapeUtil::MakeShape(F32, {2, 6})));
    shard_data.PopulateWithValue(static_cast<float>((shard_id + 1) * 10));
    ASSERT_OK(reconstructor.ProcessShardTensor(
        opt_key, {shard_id, std::make_shared<Literal>(std::move(shard_data))}));
  }

  ASSERT_EQ(recovered_by_manual_id.size(), 2);
  ASSERT_TRUE(recovered_by_manual_id.contains(0));
  ASSERT_TRUE(recovered_by_manual_id.contains(1));

  // Manual group 0 combines device 0 (10.0f) and device 1 (20.0f) -> [4, 6]
  EXPECT_THAT(recovered_by_manual_id.at(0).shape().dimensions(),
              ElementsAre(4, 6));
  EXPECT_EQ(recovered_by_manual_id.at(0).Get<float>({0, 0}), 10.0f);
  EXPECT_EQ(recovered_by_manual_id.at(0).Get<float>({1, 5}), 10.0f);
  EXPECT_EQ(recovered_by_manual_id.at(0).Get<float>({2, 0}), 20.0f);
  EXPECT_EQ(recovered_by_manual_id.at(0).Get<float>({3, 5}), 20.0f);

  // Manual group 1 combines device 2 (30.0f) and device 3 (40.0f) -> [4, 6]
  EXPECT_THAT(recovered_by_manual_id.at(1).shape().dimensions(),
              ElementsAre(4, 6));
  EXPECT_EQ(recovered_by_manual_id.at(1).Get<float>({0, 0}), 30.0f);
  EXPECT_EQ(recovered_by_manual_id.at(1).Get<float>({1, 5}), 30.0f);
  EXPECT_EQ(recovered_by_manual_id.at(1).Get<float>({2, 0}), 40.0f);
  EXPECT_EQ(recovered_by_manual_id.at(1).Get<float>({3, 5}), 40.0f);
}

TEST_F(OriginalValueRecoveryTest,
       AlgebraicSimplifierReshapeAndBroadcastMerged) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module

ENTRY entry (param0: f32[5]) -> f32[1,2,3,5,1] {
  %param0 = f32[5]{0} parameter(0)
  %reshape = f32[1,5,1]{2,1,0} reshape(%param0), origin={{"reshape"}}
  ROOT %broadcast = f32[1,2,3,5,1]{4,3,2,1,0} broadcast(%reshape), dimensions={0,3,4}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(module->original_value_recovery_table().empty());

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(simplifier, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(module->original_value_recovery_table().empty());

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  std::optional<Literal> recovered_reshape;
  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t manual_shard_id) {
        if (original_tensor.instruction_name == "reshape" &&
            recovered_data != nullptr) {
          recovered_reshape = recovered_data->Clone();
        }
      };

  HloOriginalValueReconstructor reconstructor(analysis_shared, callback,
                                              std::nullopt, module.get());

  AbsoluteScopedTensorKey opt_key =
      FindOptimizedTensorKey(*analysis_shared, "reshape");
  ASSERT_FALSE(opt_key.tensor_key.instruction_name.empty());

  Literal literal =
      LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  ShardTensor shard = {/*logical_shard_id=*/0,
                       /*data=*/std::make_shared<Literal>(std::move(literal))};

  ASSERT_OK(reconstructor.ProcessShardTensor(opt_key, std::move(shard)));

  ASSERT_TRUE(recovered_reshape.has_value());
  EXPECT_THAT(recovered_reshape->shape().dimensions(), ElementsAre(1, 5, 1));
  EXPECT_EQ(recovered_reshape->Get<float>({0, 0, 0}), 1.0f);
  EXPECT_EQ(recovered_reshape->Get<float>({0, 1, 0}), 2.0f);
  EXPECT_EQ(recovered_reshape->Get<float>({0, 2, 0}), 3.0f);
  EXPECT_EQ(recovered_reshape->Get<float>({0, 3, 0}), 4.0f);
  EXPECT_EQ(recovered_reshape->Get<float>({0, 4, 0}), 5.0f);
}

TEST_F(OriginalValueRecoveryTest,
       ChainedPassesAlgebraicSimplifierThenSpmdPartitioner) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={(f32[4]{0})->f32[1,2,3,4,1]{4,3,2,1,0}}, num_partitions=2

ENTRY entry (param0: f32[4]) -> f32[1,2,3,4,1] {
  %param0 = f32[4]{0} parameter(0), sharding={devices=[2]<=[2]}
  %reshape = f32[1,4,1]{2,1,0} reshape(%param0), origin={{"reshape"}}
  ROOT %broadcast = f32[1,2,3,4,1]{4,3,2,1,0} broadcast(%reshape), dimensions={0,3,4}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(module->original_value_recovery_table().empty());

  // Pass 1: AlgebraicSimplifier merges reshape and broadcast.
  AlgebraicSimplifierOptions simplifier_options;
  AlgebraicSimplifier simplifier(simplifier_options);
  ASSERT_OK_AND_ASSIGN(bool simplifier_changed,
                       RunHloPass(simplifier, module.get()));
  EXPECT_TRUE(simplifier_changed);
  EXPECT_FALSE(module->original_value_recovery_table().empty());

  // Pass 2: SpmdPartitioner partitions the remaining instructions.
  spmd::SpmdPartitionerOptions spmd_options;
  spmd_options.allow_module_signature_change = true;
  spmd::SpmdPartitioner partitioner(2, 1, spmd_options);
  ASSERT_OK_AND_ASSIGN(bool spmd_changed,
                       RunHloPass(partitioner, module.get()));
  EXPECT_TRUE(spmd_changed);

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  std::optional<Literal> recovered_reshape;
  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t manual_shard_id) {
        if (original_tensor.instruction_name == "reshape" &&
            recovered_data != nullptr) {
          recovered_reshape = recovered_data->Clone();
        }
      };

  HloOriginalValueReconstructor reconstructor(analysis_shared, callback,
                                              std::nullopt, module.get());

  AbsoluteScopedTensorKey opt_key =
      FindOptimizedTensorKey(*analysis_shared, "reshape");
  ASSERT_FALSE(opt_key.tensor_key.instruction_name.empty());

  // Each shard has shape [2].
  // Shard 0: {1.0f, 2.0f}
  Literal shard0_lit = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  // Shard 1: {3.0f, 4.0f}
  Literal shard1_lit = LiteralUtil::CreateR1<float>({3.0f, 4.0f});

  ASSERT_OK(reconstructor.ProcessShardTensor(
      opt_key, {0, std::make_shared<Literal>(std::move(shard0_lit))}));
  ASSERT_OK(reconstructor.ProcessShardTensor(
      opt_key, {1, std::make_shared<Literal>(std::move(shard1_lit))}));

  ASSERT_TRUE(recovered_reshape.has_value());
  EXPECT_THAT(recovered_reshape->shape().dimensions(), ElementsAre(1, 4, 1));
  EXPECT_EQ(recovered_reshape->Get<float>({0, 0, 0}), 1.0f);
  EXPECT_EQ(recovered_reshape->Get<float>({0, 1, 0}), 2.0f);
  EXPECT_EQ(recovered_reshape->Get<float>({0, 2, 0}), 3.0f);
  EXPECT_EQ(recovered_reshape->Get<float>({0, 3, 0}), 4.0f);
}

TEST_F(OriginalValueRecoveryTest,
       ChainedPassesPartitionedMiddleUnshardDropped) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={(f32[4]{0})->f32[1,2,3,4,1]{4,3,2,1,0}}, num_partitions=2,
debug_attributes={
  {"reshape"}:({callback_id=123,partitioned=true})
}

ENTRY entry (param0: f32[4]) -> f32[1,2,3,4,1] {
  %param0 = f32[4]{0} parameter(0), sharding={devices=[2]<=[2]}
  %reshape = f32[1,4,1]{2,1,0} reshape(%param0), origin={{"reshape"}}
  ROOT %broadcast = f32[1,2,3,4,1]{4,3,2,1,0} broadcast(%reshape), dimensions={0,3,4}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(module->original_value_recovery_table().empty());

  AlgebraicSimplifierOptions simplifier_options;
  AlgebraicSimplifier simplifier(simplifier_options);
  ASSERT_OK_AND_ASSIGN(bool simplifier_changed,
                       RunHloPass(simplifier, module.get()));
  EXPECT_TRUE(simplifier_changed);

  spmd::SpmdPartitionerOptions spmd_options;
  spmd_options.allow_module_signature_change = true;
  spmd::SpmdPartitioner partitioner(2, 1, spmd_options);
  ASSERT_OK_AND_ASSIGN(bool spmd_changed,
                       RunHloPass(partitioner, module.get()));
  EXPECT_TRUE(spmd_changed);

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  int call_count = 0;
  std::vector<std::optional<Literal>> received_literals;
  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t manual_shard_id) {
        if (original_tensor.instruction_name == "reshape") {
          ++call_count;
          if (recovered_data != nullptr) {
            received_literals.push_back(recovered_data->Clone());
          } else {
            received_literals.push_back(std::nullopt);
          }
        }
      };

  HloOriginalValueReconstructor reconstructor(analysis_shared, callback,
                                              std::nullopt, module.get());

  AbsoluteScopedTensorKey opt_key =
      FindOptimizedTensorKey(*analysis_shared, "reshape");
  ASSERT_FALSE(opt_key.tensor_key.instruction_name.empty());

  Literal shard0_lit = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  Literal shard1_lit = LiteralUtil::CreateR1<float>({3.0f, 4.0f});

  ASSERT_OK(reconstructor.ProcessShardTensor(
      opt_key, {0, std::make_shared<Literal>(std::move(shard0_lit))}));
  ASSERT_OK(reconstructor.ProcessShardTensor(
      opt_key, {1, std::make_shared<Literal>(std::move(shard1_lit))}));

  // Because the unshard recovery module precedes the full-shape reshape module,
  // per-shard recovery (partitioned=true) cannot be evaluated and reports
  // nullptr for each shard.
  ASSERT_EQ(call_count, 2);
  EXPECT_FALSE(received_literals[0].has_value());
  EXPECT_FALSE(received_literals[1].has_value());
}

TEST_F(OriginalValueRecoveryTest, PartitionedDebugAttributeReportsShards) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={(f32[4,8]{1,0})->f32[4,8]{1,0}}, num_partitions=2,
debug_attributes={
  {"neg_origin"}:({callback_id=42,partitioned=true})
}

ENTRY entry (p: f32[4,8]) -> f32[4,8] {
  %p = f32[4,8]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}
  ROOT %neg = f32[4,8]{1,0} negate(%p), sharding={devices=[2,1]<=[2]}, origin={{"neg_origin"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(module->original_value_recovery_table().empty());

  spmd::SpmdPartitionerOptions options;
  options.allow_module_signature_change = true;
  spmd::SpmdPartitioner partitioner(2, 1, options);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(partitioner, module.get()));
  EXPECT_TRUE(changed);

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  std::vector<Literal> received_shards;
  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t manual_shard_id) {
        if (original_tensor.instruction_name == "neg_origin" &&
            recovered_data != nullptr) {
          received_shards.push_back(recovered_data->Clone());
        }
      };

  HloOriginalValueReconstructor reconstructor(analysis_shared, callback,
                                              std::nullopt, module.get());

  AbsoluteScopedTensorKey opt_key =
      FindOptimizedTensorKey(*analysis_shared, "neg_origin");
  ASSERT_FALSE(opt_key.tensor_key.instruction_name.empty());

  ASSERT_OK_AND_ASSIGN(Literal literal0,
                       Literal::Make(ShapeUtil::MakeShape(F32, {2, 8})));
  literal0.PopulateWithValue(1.0f);
  ASSERT_OK(reconstructor.ProcessShardTensor(
      opt_key, {0, std::make_shared<Literal>(std::move(literal0))}));

  ASSERT_OK_AND_ASSIGN(Literal literal1,
                       Literal::Make(ShapeUtil::MakeShape(F32, {2, 8})));
  literal1.PopulateWithValue(2.0f);
  ASSERT_OK(reconstructor.ProcessShardTensor(
      opt_key, {1, std::make_shared<Literal>(std::move(literal1))}));

  // With partitioned=true, the callback is invoked for each shard immediately,
  // skipping the final unshard!
  ASSERT_EQ(received_shards.size(), 2);
  EXPECT_THAT(received_shards[0].shape().dimensions(), ElementsAre(2, 8));
  EXPECT_EQ(received_shards[0].Get<float>({0, 0}), 1.0f);
  EXPECT_THAT(received_shards[1].shape().dimensions(), ElementsAre(2, 8));
  EXPECT_EQ(received_shards[1].Get<float>({0, 0}), 2.0f);
}

TEST_F(OriginalValueRecoveryTest, MixedPartitionedDebugAttributes) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={(f32[4,8]{1,0})->f32[4,8]{1,0}}, num_partitions=2,
debug_attributes={
  {"neg_origin"}:({callback_id=1,partitioned=false},{callback_id=2,partitioned=true})
}

ENTRY entry (p: f32[4,8]) -> f32[4,8] {
  %p = f32[4,8]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}
  ROOT %neg = f32[4,8]{1,0} negate(%p), sharding={devices=[2,1]<=[2]}, origin={{"neg_origin"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(module->original_value_recovery_table().empty());

  spmd::SpmdPartitionerOptions options;
  options.allow_module_signature_change = true;
  spmd::SpmdPartitioner partitioner(2, 1, options);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(partitioner, module.get()));
  EXPECT_TRUE(changed);

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  std::vector<Literal> recovered_unsharded;
  std::vector<Literal> recovered_partitioned;
  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t manual_shard_id) {
        if (original_tensor.instruction_name != "neg_origin" ||
            recovered_data == nullptr) {
          return;
        }
        for (const auto& attr : debug_attributes) {
          if (attr.partitioned) {
            recovered_partitioned.push_back(recovered_data->Clone());
          } else {
            recovered_unsharded.push_back(recovered_data->Clone());
          }
        }
      };

  HloOriginalValueReconstructor reconstructor(analysis_shared, callback,
                                              std::nullopt, module.get());

  AbsoluteScopedTensorKey opt_key =
      FindOptimizedTensorKey(*analysis_shared, "neg_origin");
  ASSERT_FALSE(opt_key.tensor_key.instruction_name.empty());

  ASSERT_OK_AND_ASSIGN(Literal literal0,
                       Literal::Make(ShapeUtil::MakeShape(F32, {2, 8})));
  literal0.PopulateWithValue(1.0f);
  ASSERT_OK(reconstructor.ProcessShardTensor(
      opt_key, {0, std::make_shared<Literal>(std::move(literal0))}));

  ASSERT_OK_AND_ASSIGN(Literal literal1,
                       Literal::Make(ShapeUtil::MakeShape(F32, {2, 8})));
  literal1.PopulateWithValue(2.0f);
  ASSERT_OK(reconstructor.ProcessShardTensor(
      opt_key, {1, std::make_shared<Literal>(std::move(literal1))}));

  ASSERT_EQ(recovered_partitioned.size(), 2);
  EXPECT_THAT(recovered_partitioned[0].shape().dimensions(), ElementsAre(2, 8));
  EXPECT_EQ(recovered_partitioned[0].Get<float>({0, 0}), 1.0f);
  EXPECT_THAT(recovered_partitioned[1].shape().dimensions(), ElementsAre(2, 8));
  EXPECT_EQ(recovered_partitioned[1].Get<float>({0, 0}), 2.0f);

  ASSERT_EQ(recovered_unsharded.size(), 1);
  EXPECT_THAT(recovered_unsharded[0].shape().dimensions(), ElementsAre(4, 8));
  EXPECT_EQ(recovered_unsharded[0].Get<float>({0, 0}), 1.0f);
  EXPECT_EQ(recovered_unsharded[0].Get<float>({2, 0}), 2.0f);
}

TEST_F(OriginalValueRecoveryTest,
       MultiControllerUnaddressableDeviceAbortsRecovery) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={(f32[4,8]{1,0})->f32[4,8]{1,0}}, num_partitions=2

ENTRY entry (p: f32[4,8]) -> f32[4,8] {
  %p = f32[4,8]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}
  ROOT %neg = f32[4,8]{1,0} negate(%p), sharding={devices=[2,1]<=[2]}, origin={{"neg_origin"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(module->original_value_recovery_table().empty());

  spmd::SpmdPartitionerOptions options;
  options.allow_module_signature_change = true;
  spmd::SpmdPartitioner partitioner(2, 1, options);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(partitioner, module.get()));
  EXPECT_TRUE(changed);

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  bool callback_called = false;
  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t manual_shard_id) { callback_called = true; };

  // Only logical device 0 is addressable on this controller; device 1 is not.
  auto logical_device_is_addressable = [](int64_t logical_device_id) -> bool {
    return logical_device_id == 0;
  };

  HloOriginalValueReconstructor reconstructor(
      analysis_shared, callback, logical_device_is_addressable, module.get());

  AbsoluteScopedTensorKey opt_key =
      FindOptimizedTensorKey(*analysis_shared, "neg_origin");
  ASSERT_FALSE(opt_key.tensor_key.instruction_name.empty());

  ASSERT_OK_AND_ASSIGN(Literal literal0,
                       Literal::Make(ShapeUtil::MakeShape(F32, {2, 8})));
  literal0.PopulateWithValue(1.0f);
  ASSERT_OK(reconstructor.ProcessShardTensor(
      opt_key, {0, std::make_shared<Literal>(std::move(literal0))}));

  EXPECT_FALSE(callback_called);
}

TEST_F(OriginalValueRecoveryTest,
       CallInlinerThenAlgebraicSimplifierPreservesScope) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module

callee (x: f32[5]) -> f32[1,2,3,5,1] {
  %x = f32[5]{0} parameter(0)
  %reshape = f32[1,5,1]{2,1,0} reshape(%x), origin={{"reshape"}}
  ROOT %broadcast = f32[1,2,3,5,1]{4,3,2,1,0} broadcast(%reshape), dimensions={0,3,4}
}

ENTRY entry (param0: f32[5]) -> f32[1,2,3,5,1] {
  %param0 = f32[5]{0} parameter(0)
  ROOT %call.1 = f32[1,2,3,5,1]{4,3,2,1,0} call(%param0), to_apply=callee, origin={{"call.1"},["call.1"]}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(module->original_value_recovery_table().empty());

  // Pass 1: CallInliner inlines callee and prefixes origin with "call.1/".
  CallInliner inliner(/*single_call_site=*/false);
  ASSERT_OK_AND_ASSIGN(bool inliner_changed, RunHloPass(inliner, module.get()));
  EXPECT_TRUE(inliner_changed);

  // Pass 2: AlgebraicSimplifier merges reshape and broadcast and records the
  // recovery computation for {"call.1/reshape"}.
  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  ASSERT_OK_AND_ASSIGN(bool simplifier_changed,
                       RunHloPass(simplifier, module.get()));
  EXPECT_TRUE(simplifier_changed);
  EXPECT_FALSE(module->original_value_recovery_table().empty());

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  std::optional<AbsoluteScopedTensorKey> recovered_key;
  std::optional<Literal> recovered_reshape;
  auto callback =
      [&](const AbsoluteScopedTensorKey& original_tensor_key,
          const OriginalArray& original_tensor,
          std::shared_ptr<Literal> recovered_data,
          const std::vector<HloModule::DebugAttributes>& debug_attributes,
          int64_t manual_shard_id) {
        if (original_tensor_key.tensor_key.instruction_name == "reshape" &&
            recovered_data != nullptr) {
          recovered_key = original_tensor_key;
          recovered_reshape = recovered_data->Clone();
        }
      };

  HloOriginalValueReconstructor reconstructor(analysis_shared, callback,
                                              std::nullopt, module.get());

  AbsoluteScopedTensorKey opt_key =
      FindOptimizedTensorKey(*analysis_shared, "call.1/reshape");
  ASSERT_FALSE(opt_key.tensor_key.instruction_name.empty());

  Literal literal =
      LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  ASSERT_OK(reconstructor.ProcessShardTensor(
      opt_key, {0, std::make_shared<Literal>(std::move(literal))}));

  ASSERT_TRUE(recovered_key.has_value());
  EXPECT_EQ(recovered_key->tensor_key.instruction_name, "reshape");
  EXPECT_THAT(recovered_key->scope_instructions,
              ElementsAre(ScopeInstruction::Create("call.1")));

  ASSERT_TRUE(recovered_reshape.has_value());
  EXPECT_THAT(recovered_reshape->shape().dimensions(), ElementsAre(1, 5, 1));
  EXPECT_EQ(recovered_reshape->Get<float>({0, 0, 0}), 1.0f);
  EXPECT_EQ(recovered_reshape->Get<float>({0, 4, 0}), 5.0f);
}

}  // namespace
}  // namespace xla
