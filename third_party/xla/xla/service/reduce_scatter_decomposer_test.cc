/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/service/reduce_scatter_decomposer.h"

#include <optional>
#include <string>
#include <utility>

#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal_util.h"
#include "xla/service/collective_ops_utils.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

class ReduceScatterDecomposerTest : public HloHardwareIndependentTestBase {
 public:
  enum class PassAction {
    kNoChange,
    kTrivialGroups,
    kTableLookup,
  };
  void RunPass(
      absl::string_view hlo_module, PassAction action,
      CollectiveOpGroupMode mode = CollectiveOpGroupMode::kCrossReplica,
      int64_t shard_size = 0, int64_t shard_dimension = 0,
      int64_t replica_count = 2,
      std::function<bool(const HloInstruction *)> should_decompose =
          [](const HloInstruction *) { return true; },
      std::optional<std::pair<std::string, std::string>> attribute =
          std::nullopt) {
    const int64_t partition_count = 2;
    TF_ASSERT_OK_AND_ASSIGN(
        auto module, ParseAndReturnVerifiedModule(hlo_module, replica_count,
                                                  partition_count));
    TF_ASSERT_OK_AND_ASSIGN(
        bool changed,
        ReduceScatterDecomposer(/*update_layout=*/nullptr,
                                /*should_decompose=*/should_decompose)
            .Run(module.get()));
    if (action == PassAction::kNoChange) {
      ASSERT_FALSE(changed);
      return;
    }
    ASSERT_TRUE(changed);

    Literal multiplier = LiteralUtil::CreateR0<uint32_t>(shard_size);
    ::testing::Matcher<const ::xla::HloInstruction *> id_matcher = [&]() {
      switch (mode) {
        case CollectiveOpGroupMode::kCrossPartition:
          return op::PartitionId();
        case CollectiveOpGroupMode::kCrossReplica:
          return op::ReplicaId();
        case CollectiveOpGroupMode::kCrossReplicaAndPartition:
          return op::ReplicaId();
        case CollectiveOpGroupMode::kFlattenedID: {
          return op::Add(
              op::Multiply(op::ReplicaId(),
                           op::Constant(LiteralUtil::CreateR0<uint32_t>(
                               partition_count))),
              op::PartitionId());
        }
      }
    }();
    auto root = module->entry_computation()->root_instruction();
    const Shape &shape = root->shape();

    ::testing::Matcher<const ::xla::HloInstruction *> slice_index = id_matcher;
    if (action == PassAction::kTableLookup) {
      slice_index = op::Reshape(op::DynamicSlice(op::Constant(), id_matcher));
    }
    if (mode == CollectiveOpGroupMode::kCrossReplicaAndPartition) {
      slice_index = op::Add(
          op::Multiply(
              slice_index,
              op::Constant(LiteralUtil::CreateR0<uint32_t>(partition_count))),
          op::PartitionId());
    }
    auto zero_matcher = op::Constant(LiteralUtil::Zero(U32));

    std::vector<::testing::Matcher<const ::xla::HloInstruction *>> ds_operands(
        shape.dimensions().size() + 1, zero_matcher);
    if (attribute.has_value()) {
      ds_operands[0] =
          AllOf(op::AllReduce(op::Parameter(0)),
                op::FrontendAttribute(attribute->first, attribute->second));
    } else {
      ds_operands[0] = op::AllReduce(op::Parameter(0));
    }
    ds_operands[shard_dimension + 1] =
        op::Multiply(slice_index, op::Constant(std::move(multiplier)));
    EXPECT_THAT(root, op::DynamicSlice(ds_operands));
  }
};

TEST_F(ReduceScatterDecomposerTest, TrivialReplicaID) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  ROOT rs = f32[4] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={0}, to_apply=sum
}
)";
  RunPass(hlo_string, PassAction::kTrivialGroups,
          CollectiveOpGroupMode::kCrossReplica,
          /*shard_size=*/4);
}

TEST_F(ReduceScatterDecomposerTest, TableLookupReplicaId) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  ROOT rs = f32[4] reduce-scatter(p0), replica_groups={{1, 0}}, dimensions={0}, to_apply=sum
}
)";
  RunPass(hlo_string, PassAction::kTableLookup,
          CollectiveOpGroupMode::kCrossReplica,
          /*shard_size=*/4);
}

TEST_F(ReduceScatterDecomposerTest, TrivialCrossReplicaAndPartition) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[4, 8] parameter(0)
  // Tn this mode, the participants are the given replicas across all partitions.
  // Here, we have 2 replicas and 2 partitions, so 4 total shards.
  ROOT rs = f32[4, 2] reduce-scatter(p0), replica_groups={{0, 1}}, channel_id=1, dimensions={1}, to_apply=sum
}
)";
  RunPass(hlo_string, PassAction::kTrivialGroups,
          CollectiveOpGroupMode::kCrossReplicaAndPartition,
          /*shard_size=*/2, /*shard_dimension=*/1);
}

TEST_F(ReduceScatterDecomposerTest,
       TrivialCrossReplicaAndPartition_SingleReplica) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[4, 8] parameter(0)
  // Tn this mode, the participants are the given replicas across all partitions.
  // Here, we have 1 replicas and 2 partitions, so 2 total shards.
  ROOT rs = f32[4, 4] reduce-scatter(p0), frontend_attributes={_scheduling_group_id="1"}, replica_groups={{0}}, channel_id=1, dimensions={1}, to_apply=sum
}
)";
  // kCrossPartition here indicates that replica_index * num_partitions +
  // partition_id will be simplified by the pass to just partition_id
  RunPass(
      hlo_string, PassAction::kTrivialGroups,
      CollectiveOpGroupMode::kCrossPartition,
      /*shard_size=*/4, /*shard_dimension=*/1, /*replica_count=*/1,
      /*should_decompose =*/[](const HloInstruction *) { return true; },
      std::make_pair("_scheduling_group_id", "1"));
}

TEST_F(ReduceScatterDecomposerTest, TableLookupFlattenedId) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[4, 8] parameter(0)
  ROOT rs = f32[4, 2] reduce-scatter(p0), replica_groups={{1,0, 3, 2}}, channel_id=1, dimensions={1}, to_apply=sum, use_global_device_ids=true
}
)";
  RunPass(hlo_string, PassAction::kTableLookup,
          CollectiveOpGroupMode::kFlattenedID,
          /*shard_size=*/2, /*shard_dimension=*/1);
}

TEST_F(ReduceScatterDecomposerTest, NoChange) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[4, 8] parameter(0)
  ROOT rs = (f32[4, 2], f32[4,2]) reduce-scatter(p0, p0), replica_groups={{1,0, 3, 2}}, channel_id=1, dimensions={1}, to_apply=sum, use_global_device_ids=true
}
)";
  RunPass(hlo_string, PassAction::kNoChange);
}

TEST_F(ReduceScatterDecomposerTest, NoChangeWithShouldDecompose) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[4, 8] parameter(0)
  ROOT rs = f32[4, 4] reduce-scatter(p0), replica_groups={{0,1}, {2,3}}, channel_id=1, dimensions={1}, to_apply=sum, use_global_device_ids=true
}
)";
  RunPass(hlo_string, PassAction::kNoChange,
          CollectiveOpGroupMode::kCrossReplica,
          /*shard_size=*/0, /*shard_dimension=*/0,
          /*replica_count=*/2, [](const HloInstruction *) { return false; });
}

}  // namespace
}  // namespace xla
