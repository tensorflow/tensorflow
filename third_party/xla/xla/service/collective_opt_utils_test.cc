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

#include "xla/service/collective_opt_utils.h"

#include <cstdint>
#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

class CheckUniformReplicaGroupsTest : public HloHardwareIndependentTestBase {};
TEST_F(CheckUniformReplicaGroupsTest, CheckUniformReplicaGroupsUniform) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      param = f32[16,10] parameter(0)
      ROOT ag = f32[16,10] all-gather(param), dimensions={0},
          replica_groups={{0,1},{2,3}}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  const HloAllGatherInstruction* ag = Cast<HloAllGatherInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_TRUE(CheckUniformReplicaGroups(ag));
}

TEST_F(CheckUniformReplicaGroupsTest, CheckUniformReplicaGroupsNonUniform) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      param = f32[16,10] parameter(0)
      ROOT ag = f32[16,10] all-gather(param), dimensions={0},
          replica_groups={{0,1},{2}}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  const HloAllGatherInstruction* ag = Cast<HloAllGatherInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_FALSE(CheckUniformReplicaGroups(ag));
}

TEST_F(CheckUniformReplicaGroupsTest, CheckUniformReplicaGroupsSingleGroup) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      param = f32[16,10] parameter(0)
      ROOT ag = f32[16,10] all-gather(param), dimensions={0},
        replica_groups={{0,1,2,3}}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  const HloAllGatherInstruction* ag = Cast<HloAllGatherInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_TRUE(CheckUniformReplicaGroups(ag));
}

class ExtractSplitDimSpecTest : public HloHardwareIndependentTestBase {};

TEST_F(ExtractSplitDimSpecTest, SingleDim) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      param = f32[16,10] parameter(0)
      zero = s32[] constant(0)
      ds_index = s32[] parameter(1)
      ROOT ds = f32[8,10] dynamic-slice(param, ds_index, zero),
        dynamic_slice_sizes={8,10}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* ds = module->entry_computation()->root_instruction();
  ASSERT_THAT(ds, testing::NotNull());

  std::optional<SplitDimSpec> spec =
      ExtractSplitDimSpec(*ds, /*allow_multiple_split_dims=*/false);

  ASSERT_TRUE(spec.has_value());
  EXPECT_EQ(spec->split_dim, 0);
  EXPECT_THAT(spec->split_dims, ElementsAre(0));
}

TEST_F(ExtractSplitDimSpecTest, MultipleDimOffsets) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      param = f32[16,20] parameter(0)
      zero = s32[] constant(0)
      ds_index1 = s32[] parameter(1)
      ROOT ds = f32[8,10] dynamic-slice(param, zero, ds_index1),
        dynamic_slice_sizes={8,10}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* ds = module->entry_computation()->root_instruction();
  ASSERT_THAT(ds, testing::NotNull());

  std::optional<SplitDimSpec> spec =
      ExtractSplitDimSpec(*ds, /*allow_multiple_split_dims=*/false);

  ASSERT_TRUE(spec.has_value());
  EXPECT_EQ(spec->split_dim, 1);
  EXPECT_THAT(spec->split_dims, ElementsAre(1));
}

TEST_F(ExtractSplitDimSpecTest, MultipleSplitDimsAllowed) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      param = f32[16,20,30] parameter(0)
      zero = s32[] constant(0)
      ds_index1 = s32[] parameter(1)
      ds_index2 = s32[] parameter(2)
      ROOT ds = f32[16,10,15] dynamic-slice(param, zero, ds_index1, ds_index2),
        dynamic_slice_sizes={16,10,15}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* ds = module->entry_computation()->root_instruction();
  ASSERT_THAT(ds, testing::NotNull());

  std::optional<SplitDimSpec> spec =
      ExtractSplitDimSpec(*ds, /*allow_multiple_split_dims=*/true);

  ASSERT_TRUE(spec.has_value());
  EXPECT_EQ(spec->split_dim, 1);
  EXPECT_THAT(spec->split_dims, ElementsAre(1, 2));
}

TEST_F(ExtractSplitDimSpecTest, MultipleSplitDimsNonConsecutive) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      param = f32[16,20,30] parameter(0)
      ds_index0 = s32[] parameter(1)
      zero = s32[] constant(0)
      ds_index2 = s32[] parameter(2)
      ROOT ds = f32[8,20,15] dynamic-slice(param, ds_index0, zero, ds_index2),
        dynamic_slice_sizes={8,20,15}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* ds = module->entry_computation()->root_instruction();
  ASSERT_THAT(ds, testing::NotNull());

  std::optional<SplitDimSpec> spec =
      ExtractSplitDimSpec(*ds, /*allow_multiple_split_dims=*/true);

  EXPECT_FALSE(spec.has_value());
}

TEST_F(ExtractSplitDimSpecTest, MultipleSplitDimsNotAllowed) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      param = f32[16,20,30] parameter(0)
      zero = s32[] constant(0)
      ds_index1 = s32[] parameter(1)
      ds_index2 = s32[] parameter(2)
      ROOT ds = f32[16,10,15] dynamic-slice(param, zero, ds_index1, ds_index2),
        dynamic_slice_sizes={16,10,15}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* ds = module->entry_computation()->root_instruction();
  ASSERT_THAT(ds, testing::NotNull());

  std::optional<SplitDimSpec> spec =
      ExtractSplitDimSpec(*ds, /*allow_multiple_split_dims=*/false);

  EXPECT_FALSE(spec.has_value());
}

TEST_F(ExtractSplitDimSpecTest, NoSplitDimFound) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      param = f32[16,10] parameter(0)
      zero = s32[] constant(0)
      ROOT ds = f32[16,10] dynamic-slice(param, zero, zero),
        dynamic_slice_sizes={16,10}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* ds = module->entry_computation()->root_instruction();
  ASSERT_THAT(ds, testing::NotNull());

  std::optional<SplitDimSpec> spec =
      ExtractSplitDimSpec(*ds, /*allow_multiple_split_dims=*/false);

  ASSERT_TRUE(spec.has_value());
  EXPECT_EQ(spec->split_dim, -1);
  EXPECT_TRUE(spec->split_dims.empty());
}

class FindUniqueDynamicSliceUserFromCollectiveTest
    : public HloHardwareIndependentTestBase {};

TEST_F(FindUniqueDynamicSliceUserFromCollectiveTest, CaptureDsAllGatger) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      param = f32[16,10] parameter(0)
      ag = f32[32,10] all-gather(param), dimensions={0}, replica_groups={{0,1}}
      zero = s32[] constant(0)
      replica_id = u32[] replica-id()
      slice_index = s32[] convert(replica_id)
      ROOT ds = f32[16,10] dynamic-slice(ag, slice_index, zero),
        dynamic_slice_sizes={16,10}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloAllGatherInstruction* ag = Cast<HloAllGatherInstruction>(
      module->entry_computation()->GetInstructionWithName("ag"));
  ASSERT_THAT(ag, testing::NotNull());

  std::optional<CollectiveUsers> result =
      FindUniqueDynamicSliceUserFromCollective(
          ag, /*allow_multiple_users=*/false,
          /*allow_intervening_reshape=*/false,
          /*allow_intervening_bitcast=*/false);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->dynamic_slice->opcode(), HloOpcode::kDynamicSlice);
  EXPECT_EQ(result->dynamic_slice->name(), "ds");
  EXPECT_EQ(result->bitcast, nullptr);
  EXPECT_EQ(result->reshape, nullptr);
}

TEST_F(FindUniqueDynamicSliceUserFromCollectiveTest,
       CaptureDsReshapeAllGather) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      param = f32[16,10] parameter(0)
      ag = f32[32,10] all-gather(param), dimensions={0}, replica_groups={{0,1}}
      reshape = f32[320] reshape(ag)
      zero = s32[] constant(0)
      replica_id = u32[] replica-id()
      slice_index = s32[] convert(replica_id)
      ROOT ds = f32[160] dynamic-slice(reshape, slice_index),
        dynamic_slice_sizes={160}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloAllGatherInstruction* ag = Cast<HloAllGatherInstruction>(
      module->entry_computation()->GetInstructionWithName("ag"));
  ASSERT_THAT(ag, testing::NotNull());
  std::optional<CollectiveUsers> result =
      FindUniqueDynamicSliceUserFromCollective(
          ag, /*allow_multiple_users=*/false,
          /*allow_intervening_reshape=*/true,
          /*allow_intervening_bitcast=*/false);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->dynamic_slice->opcode(), HloOpcode::kDynamicSlice);
  EXPECT_EQ(result->dynamic_slice->name(), "ds");
  EXPECT_NE(result->reshape, nullptr);
  EXPECT_EQ(result->reshape->name(), "reshape");
  EXPECT_EQ(result->bitcast, nullptr);
}

TEST_F(FindUniqueDynamicSliceUserFromCollectiveTest, WithBitcast) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      param = f32[16,10] parameter(0)
      ag = f32[32,10] all-gather(param), dimensions={0}, replica_groups={{0,1}}
      bitcast = f32[32,10] bitcast(ag)
      zero = s32[] constant(0)
      replica_id = u32[] replica-id()
      slice_index = s32[] convert(replica_id)
      ROOT ds = f32[16,10] dynamic-slice(bitcast, slice_index, zero),
        dynamic_slice_sizes={16,10}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloAllGatherInstruction* ag = Cast<HloAllGatherInstruction>(
      module->entry_computation()->GetInstructionWithName("ag"));
  ASSERT_THAT(ag, testing::NotNull());

  std::optional<CollectiveUsers> result =
      FindUniqueDynamicSliceUserFromCollective(
          ag, /*allow_multiple_users=*/false,
          /*allow_intervening_reshape=*/false,
          /*allow_intervening_bitcast=*/true);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->dynamic_slice->opcode(), HloOpcode::kDynamicSlice);
  EXPECT_EQ(result->dynamic_slice->name(), "ds");
  EXPECT_NE(result->bitcast, nullptr);
  EXPECT_EQ(result->bitcast->name(), "bitcast");
  EXPECT_EQ(result->reshape, nullptr);
}

TEST_F(FindUniqueDynamicSliceUserFromCollectiveTest, WithReshapeAndBitcast) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      param = f32[16,10] parameter(0)
      ag = f32[32,10] all-gather(param), dimensions={0}, replica_groups={{0,1}}
      reshape = f32[320] reshape(ag)
      bitcast = f32[320] bitcast(reshape)
      replica_id = u32[] replica-id()
      slice_index = s32[] convert(replica_id)
      ROOT ds = f32[160] dynamic-slice(bitcast, slice_index),
        dynamic_slice_sizes={160}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloAllGatherInstruction* ag = Cast<HloAllGatherInstruction>(
      module->entry_computation()->GetInstructionWithName("ag"));
  ASSERT_THAT(ag, testing::NotNull());

  std::optional<CollectiveUsers> result =
      FindUniqueDynamicSliceUserFromCollective(
          ag, /*allow_multiple_users=*/false,
          /*allow_intervening_reshape=*/true,
          /*allow_intervening_bitcast=*/true);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->dynamic_slice->opcode(), HloOpcode::kDynamicSlice);
  EXPECT_EQ(result->dynamic_slice->name(), "ds");
  EXPECT_NE(result->reshape, nullptr);
  EXPECT_EQ(result->reshape->name(), "reshape");
  EXPECT_NE(result->bitcast, nullptr);
  EXPECT_EQ(result->bitcast->name(), "bitcast");
}

TEST_F(FindUniqueDynamicSliceUserFromCollectiveTest, MultipleUsersWithReshape) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      param = f32[16,10] parameter(0)
      ag = f32[32,10] all-gather(param), dimensions={0}, replica_groups={{0,1}}
      neg = f32[32,10] negate(ag)
      reshape = f32[320] reshape(ag)
      bitcast = f32[320] bitcast(reshape)
      replica_id = u32[] replica-id()
      slice_index = s32[] convert(replica_id)
      ROOT ds = f32[160] dynamic-slice(bitcast, slice_index),
        dynamic_slice_sizes={160}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloAllGatherInstruction* ag = Cast<HloAllGatherInstruction>(
      module->entry_computation()->GetInstructionWithName("ag"));
  ASSERT_THAT(ag, testing::NotNull());

  std::optional<CollectiveUsers> result =
      FindUniqueDynamicSliceUserFromCollective(
          ag, /*allow_multiple_users=*/true,
          /*allow_intervening_reshape=*/true,
          /*allow_intervening_bitcast=*/true);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->dynamic_slice->opcode(), HloOpcode::kDynamicSlice);
  EXPECT_EQ(result->dynamic_slice->name(), "ds");
  EXPECT_NE(result->reshape, nullptr);
  EXPECT_EQ(result->reshape->name(), "reshape");
  EXPECT_NE(result->bitcast, nullptr);
  EXPECT_EQ(result->bitcast->name(), "bitcast");
}

TEST_F(FindUniqueDynamicSliceUserFromCollectiveTest, NoDynamicSlice) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      param = f32[16,10] parameter(0)
      ag = f32[32,10] all-gather(param), dimensions={0}, replica_groups={{0,1}}
      ROOT neg = f32[32,10] negate(ag)
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloAllGatherInstruction* ag = Cast<HloAllGatherInstruction>(
      module->entry_computation()->GetInstructionWithName("ag"));
  ASSERT_THAT(ag, testing::NotNull());

  std::optional<CollectiveUsers> result =
      FindUniqueDynamicSliceUserFromCollective(
          ag, /*allow_multiple_users=*/false,
          /*allow_intervening_reshape=*/true,
          /*allow_intervening_bitcast=*/true);
  EXPECT_FALSE(result.has_value());
}

TEST_F(FindUniqueDynamicSliceUserFromCollectiveTest,
       InterveningOpWithMultipleUsers) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      param = f32[16,10] parameter(0)
      ag = f32[32,10] all-gather(param), dimensions={0}, replica_groups={{0,1}}
      reshape = f32[320] reshape(ag)
      neg = f32[320] negate(reshape)
      replica_id = u32[] replica-id()
      slice_index = s32[] convert(replica_id)
      ROOT ds = f32[160] dynamic-slice(reshape, slice_index),
        dynamic_slice_sizes={160}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloAllGatherInstruction* ag = Cast<HloAllGatherInstruction>(
      module->entry_computation()->GetInstructionWithName("ag"));
  ASSERT_THAT(ag, testing::NotNull());

  std::optional<CollectiveUsers> result =
      FindUniqueDynamicSliceUserFromCollective(
          ag, /*allow_multiple_users=*/true,
          /*allow_intervening_reshape=*/true,
          /*allow_intervening_bitcast=*/true);
  EXPECT_FALSE(result.has_value());
}

using GetIndicesSpecForDynamicSliceWithMultiplyTest =
    HloHardwareIndependentTestBase;

TEST_F(GetIndicesSpecForDynamicSliceWithMultiplyTest, SingleRgMatch) {
  constexpr absl::string_view hlo_string = R"(
    HloModule module
    ENTRY main {
      param = f32[1,1,1024] parameter(0)
      ag = f32[4,1,1024] all-gather(param), replica_groups={{0,1,2,3}},
        dimensions={0}
      const = s32[4]{0} constant({0, 6, 4, 2})
      pid = u32[] partition-id()
      ds_offset = s32[] dynamic-slice(const, pid), dynamic_slice_sizes={1}
      ROOT ds = f32[1,1,1024] dynamic-slice(ag, ds_offset, s32[] constant(0),
        s32[] constant(0)), dynamic_slice_sizes={1,1,1024}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* ag =
      module->entry_computation()->GetInstructionWithName("ag");
  ASSERT_NE(ag, nullptr);
  HloInstruction* ds_offset =
      module->entry_computation()->GetInstructionWithName("ds_offset");
  ASSERT_NE(ds_offset, nullptr);
  auto map_id = [](const HloInstruction* hlo, int64_t id) {
    return (hlo->opcode() == HloOpcode::kPartitionId) ? id : -1;
  };
  std::optional<PartitionOffsetSpec> indices_spec =
      GetIndicesSpecForDynamicSliceWithMultiply(
          Cast<HloAllGatherInstruction>(ag), ds_offset, map_id,
          /*split_dim_size=*/1);
  ASSERT_TRUE(indices_spec.has_value());
  EXPECT_EQ(indices_spec.value().per_replica_group_offsets.size(), 1);
  EXPECT_THAT(
      indices_spec.value().per_replica_group_offsets[0],
      UnorderedElementsAre(Pair(0, 0), Pair(6, 1), Pair(4, 2), Pair(2, 3)));
}

TEST_F(GetIndicesSpecForDynamicSliceWithMultiplyTest, SingleRgWithReshape) {
  constexpr absl::string_view hlo_string = R"(
    HloModule module
    ENTRY main {
      param = f32[1,1,1024] parameter(0)
      ag = f32[4,1,1024] all-gather(param), replica_groups={{0,1,2,3}},
        dimensions={0}
      const = s32[4]{0} constant({0, 6, 4, 2})
      pid = u32[] partition-id()
      ds_offset = s32[] dynamic-slice(const, pid), dynamic_slice_sizes={1}
      ds_offset_reshape = s32[] reshape(ds_offset)
      ROOT ds = f32[1,1,1024] dynamic-slice(ag, ds_offset, s32[] constant(0),
        s32[] constant(0)), dynamic_slice_sizes={1,1,1024}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* ag =
      module->entry_computation()->GetInstructionWithName("ag");
  ASSERT_NE(ag, nullptr);
  HloInstruction* ds_offset =
      module->entry_computation()->GetInstructionWithName("ds_offset");
  ASSERT_NE(ds_offset, nullptr);
  auto map_id = [](const HloInstruction* hlo, int64_t id) {
    return (hlo->opcode() == HloOpcode::kPartitionId) ? id : -1;
  };
  std::optional<PartitionOffsetSpec> indices_spec =
      GetIndicesSpecForDynamicSliceWithMultiply(
          Cast<HloAllGatherInstruction>(ag), ds_offset, map_id,
          /*split_dim_size=*/1);
  ASSERT_TRUE(indices_spec.has_value());
  EXPECT_EQ(indices_spec.value().per_replica_group_offsets.size(), 1);
  EXPECT_THAT(
      indices_spec.value().per_replica_group_offsets[0],
      UnorderedElementsAre(Pair(0, 0), Pair(6, 1), Pair(4, 2), Pair(2, 3)));
}

TEST_F(GetIndicesSpecForDynamicSliceWithMultiplyTest, MultiRgsUnordered) {
  constexpr absl::string_view hlo_string = R"(
    HloModule module
    ENTRY main {
      param = f32[1,1,1024] parameter(0)
      ag = f32[8,1,1024] all-gather(param),
        replica_groups={{0,1,2,3},{4,5,6,7}}, dimensions={0}
      const = s32[8]{0} constant({0, 6, 4, 2, 0, 4, 6, 2})
      pid = u32[] partition-id()
      ds_offset = s32[] dynamic-slice(const, pid), dynamic_slice_sizes={1}
      ds_offset_reshape = s32[] reshape(ds_offset)
      ROOT ds = f32[1,1,1024] dynamic-slice(ag, ds_offset,
        s32[] constant(0), s32[] constant(0)), dynamic_slice_sizes={1,1,1024}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* ag =
      module->entry_computation()->GetInstructionWithName("ag");
  HloInstruction* ds_offset =
      module->entry_computation()->GetInstructionWithName("ds_offset");
  auto map_id = [](const HloInstruction* hlo, int64_t id) {
    return (hlo->opcode() == HloOpcode::kPartitionId) ? id : -1;
  };
  std::optional<PartitionOffsetSpec> indices_spec =
      GetIndicesSpecForDynamicSliceWithMultiply(
          Cast<HloAllGatherInstruction>(ag), ds_offset, map_id,
          /*split_dim_size=*/1);
  ASSERT_TRUE(indices_spec.has_value());
  EXPECT_EQ(indices_spec.value().per_replica_group_offsets.size(), 2);
  EXPECT_THAT(
      indices_spec.value().per_replica_group_offsets[0],
      UnorderedElementsAre(Pair(0, 0), Pair(6, 1), Pair(4, 2), Pair(2, 3)));
  EXPECT_THAT(
      indices_spec.value().per_replica_group_offsets[1],
      UnorderedElementsAre(Pair(0, 4), Pair(6, 6), Pair(4, 5), Pair(2, 7)));
}

TEST_F(GetIndicesSpecForDynamicSliceWithMultiplyTest, SingleRg8Replicas) {
  constexpr absl::string_view hlo_string = R"(
    HloModule module
    ENTRY entry {
      p = f32[32,8,128] parameter(0)
      ag = f32[256,8,128] all-gather(p), replica_groups={{0,1,2,3,4,5,6,7}},
        dimensions={0}, channel_id=1, use_global_device_ids=true
      pid = u32[] partition-id()
      permuted_idx_list = s32[8]{0} constant({224,192,160,128,96,64,32,0})
      offset = s32[1] dynamic-slice(permuted_idx_list, pid),
        dynamic_slice_sizes={1}
      offset_reshape = s32[] reshape(offset)
      zero = s32[] constant(0)
      ROOT ds = f32[32,8,128] dynamic-slice(ag, offset_reshape, zero, zero),
        dynamic_slice_sizes={32,8,128}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* ag =
      module->entry_computation()->GetInstructionWithName("ag");
  HloInstruction* ds_offset =
      module->entry_computation()->GetInstructionWithName("offset_reshape");
  auto map_id = [](const HloInstruction* hlo, int64_t id) {
    return (hlo->opcode() == HloOpcode::kPartitionId) ? id : -1;
  };
  std::optional<PartitionOffsetSpec> indices_spec =
      GetIndicesSpecForDynamicSliceWithMultiply(
          Cast<HloAllGatherInstruction>(ag), ds_offset, map_id,
          /*split_dim_size=*/32);
  ASSERT_TRUE(indices_spec.has_value());
  EXPECT_EQ(indices_spec.value().per_replica_group_offsets.size(), 1);
  EXPECT_THAT(indices_spec.value().per_replica_group_offsets[0],
              UnorderedElementsAre(Pair(0, 7), Pair(32, 6), Pair(64, 5),
                                   Pair(96, 4), Pair(128, 3), Pair(160, 2),
                                   Pair(192, 1), Pair(224, 0)));
}

TEST_F(GetIndicesSpecForDynamicSliceWithMultiplyTest, SingleRgMultiply) {
  constexpr absl::string_view hlo_string = R"(
    HloModule module
    ENTRY main {
      param = f32[2,1,1024] parameter(0)
      ag = f32[8,1,1024] all-gather(param), replica_groups={{0,1,2,3}},
        dimensions={0}
      const = s32[4]{0} constant({3, 2, 1, 0})
      pid = u32[] partition-id()
      ds_offset = s32[1]{0} dynamic-slice(const, pid), dynamic_slice_sizes={1}
      constant.2 = s32[1]{0} constant({2})
      ds_offset_mul = s32[1]{0} multiply(ds_offset, constant.2)
      reshape = s32[] reshape(ds_offset_mul)
      ROOT ds = f32[2,1,1024] dynamic-slice(ag, reshape, s32[] constant(0),
        s32[] constant(0)), dynamic_slice_sizes={2,1,1024}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* ag =
      module->entry_computation()->GetInstructionWithName("ag");
  ASSERT_NE(ag, nullptr);
  HloInstruction* ds_offset =
      module->entry_computation()->GetInstructionWithName("reshape");
  ASSERT_NE(ds_offset, nullptr);
  auto map_id = [](const HloInstruction* hlo, int64_t id) {
    return (hlo->opcode() == HloOpcode::kPartitionId) ? id : -1;
  };
  std::optional<PartitionOffsetSpec> indices_spec =
      GetIndicesSpecForDynamicSliceWithMultiply(
          Cast<HloAllGatherInstruction>(ag), ds_offset, map_id,
          /*split_dim_size=*/2);
  ASSERT_TRUE(indices_spec.has_value());
  EXPECT_EQ(indices_spec.value().per_replica_group_offsets.size(), 1);
  EXPECT_THAT(
      indices_spec.value().per_replica_group_offsets[0],
      UnorderedElementsAre(Pair(6, 0), Pair(4, 1), Pair(2, 2), Pair(0, 3)));
}

TEST_F(GetIndicesSpecForDynamicSliceWithMultiplyTest, UnmatchedNotMultiply) {
  constexpr absl::string_view hlo_string = R"(
    HloModule module
    ENTRY main {
      param = f32[2,1,1024] parameter(0)
      ag = f32[8,1,1024] all-gather(param), replica_groups={{0,1,2,3}},
        dimensions={0}
      const = s32[4]{0} constant({3, 2, 1, 0})
      pid = u32[] partition-id()
      ds_offset = s32[1]{0} dynamic-slice(const, pid), dynamic_slice_sizes={1}
      constant.2 = s32[1]{0} constant({2})
      ds_offset_add = s32[1]{0} add(ds_offset, constant.2)
      reshape = s32[] reshape(ds_offset_add)
      ROOT ds = f32[2,1,1024] dynamic-slice(ag, reshape, s32[] constant(0),
        s32[] constant(0)), dynamic_slice_sizes={2,1,1024}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* ag =
      module->entry_computation()->GetInstructionWithName("ag");
  ASSERT_NE(ag, nullptr);
  HloInstruction* ds_offset =
      module->entry_computation()->GetInstructionWithName("reshape");
  ASSERT_NE(ds_offset, nullptr);
  auto map_id = [](const HloInstruction* hlo, int64_t id) {
    return (hlo->opcode() == HloOpcode::kPartitionId) ? id : -1;
  };
  std::optional<PartitionOffsetSpec> indices_spec =
      GetIndicesSpecForDynamicSliceWithMultiply(
          Cast<HloAllGatherInstruction>(ag), ds_offset, map_id,
          /*split_dim_size=*/2);
  ASSERT_FALSE(indices_spec.has_value());
}

TEST_F(GetIndicesSpecForDynamicSliceWithMultiplyTest, MultiRgsWithMultiply) {
  constexpr absl::string_view hlo_string = R"(
    HloModule module
    ENTRY main {
      param = f32[4,1,1024] parameter(0)
      ag = f32[16,1,1024] all-gather(param),
        replica_groups={{0,1,2,3},{4,5,6,7}}, dimensions={0}
      const = s32[8]{0} constant({2, 3, 1, 0, 3, 2, 1, 0})
      pid = u32[] partition-id()
      ds_offset = s32[] dynamic-slice(const, pid), dynamic_slice_sizes={1}
      constant.4 = s32[1]{0} constant({4})
      ds_offset_mul = s32[1]{0} multiply(ds_offset, constant.4)
      reshape = s32[] reshape(ds_offset_mul)
      ROOT ds = f32[4,1,1024] dynamic-slice(ag, reshape,
        s32[] constant(0), s32[] constant(0)), dynamic_slice_sizes={4,1,1024}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* ag =
      module->entry_computation()->GetInstructionWithName("ag");
  HloInstruction* ds_offset =
      module->entry_computation()->GetInstructionWithName("reshape");
  auto map_id = [](const HloInstruction* hlo, int64_t id) {
    return (hlo->opcode() == HloOpcode::kPartitionId) ? id : -1;
  };
  std::optional<PartitionOffsetSpec> indices_spec =
      GetIndicesSpecForDynamicSliceWithMultiply(
          Cast<HloAllGatherInstruction>(ag), ds_offset, map_id,
          /*split_dim_size=*/4);
  ASSERT_TRUE(indices_spec.has_value());
  EXPECT_EQ(indices_spec.value().per_replica_group_offsets.size(), 2);
  EXPECT_THAT(
      indices_spec.value().per_replica_group_offsets[0],
      UnorderedElementsAre(Pair(8, 0), Pair(12, 1), Pair(4, 2), Pair(0, 3)));
  EXPECT_THAT(
      indices_spec.value().per_replica_group_offsets[1],
      UnorderedElementsAre(Pair(12, 4), Pair(8, 5), Pair(4, 6), Pair(0, 7)));
}

TEST_F(GetIndicesSpecForDynamicSliceWithMultiplyTest,
       UnmatchedNotMultiplyArray) {
  constexpr absl::string_view hlo_string = R"(
    HloModule module
    ENTRY main {
      param = f32[2,1,1024] parameter(0)
      ag = f32[8,1,1024] all-gather(param), replica_groups={{0,1,2,3}},
        dimensions={0}
      const = s32[4]{0} constant({3, 2, 1, 0})
      pid = u32[] partition-id()
      ds_offset = s32[1]{0} dynamic-slice(const, pid), dynamic_slice_sizes={1}
      constant.2 = s32[1]{0} constant({2})
      constant.3 = s32[1]{0} constant({10})
      ds_offset_add = s32[1]{0} multiply(constant.3, constant.2)
      reshape = s32[] reshape(ds_offset_add)
      ROOT ds = f32[2,1,1024] dynamic-slice(ag, reshape, s32[] constant(0),
        s32[] constant(0)), dynamic_slice_sizes={2,1,1024}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* ag =
      module->entry_computation()->GetInstructionWithName("ag");
  ASSERT_NE(ag, nullptr);
  HloInstruction* ds_offset =
      module->entry_computation()->GetInstructionWithName("reshape");
  ASSERT_NE(ds_offset, nullptr);
  auto map_id = [](const HloInstruction* hlo, int64_t id) {
    return (hlo->opcode() == HloOpcode::kPartitionId) ? id : -1;
  };
  std::optional<PartitionOffsetSpec> indices_spec =
      GetIndicesSpecForDynamicSliceWithMultiply(
          Cast<HloAllGatherInstruction>(ag), ds_offset, map_id,
          /*split_dim_size=*/2);
  ASSERT_FALSE(indices_spec.has_value());
}

TEST_F(GetIndicesSpecForDynamicSliceWithMultiplyTest,
       UnmatchedNotMultiplyNonShardSize) {
  constexpr absl::string_view hlo_string = R"(
    HloModule module
    ENTRY main {
      param = f32[2,1,1024] parameter(0)
      ag = f32[8,1,1024] all-gather(param), replica_groups={{0,1,2,3}},
        dimensions={0}
      const = s32[4]{0} constant({3, 2, 1, 0})
      pid = u32[] partition-id()
      ds_offset = s32[1]{0} dynamic-slice(const, pid), dynamic_slice_sizes={1}
      constant.2 = s32[1]{0} constant({3})
      ds_offset_add = s32[1]{0} multiply(ds_offset, constant.2)
      reshape = s32[] reshape(ds_offset_add)
      ROOT ds = f32[2,1,1024] dynamic-slice(ag, reshape, s32[] constant(0),
        s32[] constant(0)), dynamic_slice_sizes={2,1,1024}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* ag =
      module->entry_computation()->GetInstructionWithName("ag");
  ASSERT_NE(ag, nullptr);
  HloInstruction* ds_offset =
      module->entry_computation()->GetInstructionWithName("reshape");
  ASSERT_NE(ds_offset, nullptr);
  auto map_id = [](const HloInstruction* hlo, int64_t id) {
    return (hlo->opcode() == HloOpcode::kPartitionId) ? id : -1;
  };
  std::optional<PartitionOffsetSpec> indices_spec =
      GetIndicesSpecForDynamicSliceWithMultiply(
          Cast<HloAllGatherInstruction>(ag), ds_offset, map_id,
          /*split_dim_size=*/2);
  ASSERT_FALSE(indices_spec.has_value());
}

class MatchPermutedSliceAndPartitionOffsetTest
    : public HloHardwareIndependentTestBase {};

TEST_F(MatchPermutedSliceAndPartitionOffsetTest,
       MatchPermutedSliceAndPartitionOffsetSingleReplicaGroup) {
  constexpr absl::string_view hlo_string = R"(
    HloModule module
    ENTRY entry {
      p = f32[32,8,128] parameter(0)
      ag = f32[256,8,128] all-gather(p), replica_groups={{0,1,2,3,4,5,6,7}},
        dimensions={0}, channel_id=1, use_global_device_ids=true
      pid = u32[] partition-id()
      permuted_idx_list = s32[8]{0} constant({224,192,160,128,96, 64,32,0})
      offset = s32[1] dynamic-slice(permuted_idx_list, pid),
        dynamic_slice_sizes={1}
      offset_reshape = s32[] reshape(offset)
      zero = s32[] constant(0)
      ROOT ds = f32[32,8,128] dynamic-slice(ag, offset_reshape, zero, zero),
        dynamic_slice_sizes={32,8,128}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  const HloInstruction* ag =
      module->entry_computation()->GetInstructionWithName("ag");
  const HloAllGatherInstruction* ag_instr = Cast<HloAllGatherInstruction>(ag);

  std::optional<AllGatherDynamicSliceMatchSpec> spec =
      MatchPermutedSliceAndPartitionOffset(
          ag_instr, 8, 1, HloPredicateIsOp<HloOpcode::kPartitionId>,
          /*allow_multiple_users=*/false);

  ASSERT_TRUE(spec.has_value());
  EXPECT_THAT(
      spec->permutation_pairs,
      UnorderedElementsAre(Pair(0, 7), Pair(1, 6), Pair(2, 5), Pair(3, 4),
                           Pair(4, 3), Pair(5, 2), Pair(6, 1), Pair(7, 0)));
}

TEST_F(MatchPermutedSliceAndPartitionOffsetTest,
       MatchPermutedSliceAndPartitionOffsetMultipleReplicaGroups) {
  constexpr absl::string_view hlo_string = R"(
    HloModule module
    ENTRY entry {
      p = f32[1,4,32] parameter(0)
      ag = f32[1,16,32] all-gather(p), replica_groups={{0,1,2,3},{4,5,6,7}},
        dimensions={1}, channel_id=1, use_global_device_ids=true
      pid = u32[] partition-id()
      permuted_idx_list = s32[8]{0} constant({12,8,4,0,8,4,12,0})
      offset = s32[1] dynamic-slice(permuted_idx_list, pid),
        dynamic_slice_sizes={1}
      offset_reshape = s32[] reshape(offset)
      zero = s32[] constant(0)
      ROOT ds = f32[1,4,32] dynamic-slice(ag, zero, offset_reshape, zero),
        dynamic_slice_sizes={1,4,32}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  const HloInstruction* ag =
      module->entry_computation()->GetInstructionWithName("ag");
  const HloAllGatherInstruction* ag_instr = Cast<HloAllGatherInstruction>(ag);

  std::optional<AllGatherDynamicSliceMatchSpec> spec =
      MatchPermutedSliceAndPartitionOffset(
          ag_instr, 8, 1, HloPredicateIsOp<HloOpcode::kPartitionId>,
          /*allow_multiple_users=*/false);

  ASSERT_TRUE(spec.has_value());
  EXPECT_THAT(
      spec->permutation_pairs,
      UnorderedElementsAre(Pair(0, 3), Pair(1, 2), Pair(2, 1), Pair(3, 0),
                           Pair(4, 7), Pair(5, 5), Pair(6, 4), Pair(7, 6)));
}
using MatchDsPadAllGatherTest = HloHardwareIndependentTestBase;

TEST_F(MatchDsPadAllGatherTest, MatchDsPadAllGather) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {
      param = f64[1,2,40]{2,1,0} parameter(0)
      zero = f64[] constant(0)
      zero_idx = s32[] constant(0)
      const.24 = s32[1]{0} constant({24})
      all-gather = f64[1,8,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups={{0,1,2,3,4,5,6,7}},
        dimensions={1}, use_global_device_ids=true
      pad = f64[1,96,40]{2,1,0} pad(all-gather, zero), padding=0_0x0_88x0_0
      const.list = s32[8]{0} constant({0, 1, 2, 3, 0, 1, 2, 3})
      partition-id = u32[] partition-id()
      ds1 = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      multiply.8 = s32[1]{0} multiply(ds1, const.24)
      reshape.15 = s32[] reshape(multiply.8)
      ROOT ds = f64[1,24,40]{2,1,0} dynamic-slice(pad, zero_idx,
        reshape.15, zero_idx), dynamic_slice_sizes={1,24,40}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->name(), "ds");
  HloInstruction* pad_hlo = nullptr;
  HloInstruction* ag_hlo = nullptr;
  EXPECT_TRUE(MatchDsPadAllGather(root, &pad_hlo, &ag_hlo));
  ASSERT_NE(pad_hlo, nullptr);
  EXPECT_EQ(pad_hlo->name(), "pad");
  ASSERT_NE(ag_hlo, nullptr);
  EXPECT_EQ(ag_hlo->name(), "all-gather");
}

TEST_F(MatchDsPadAllGatherTest, MatchDsPadAllGatherNoPad) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {
      param = f64[1,2,40]{2,1,0} parameter(0)
      zero_idx = s32[] constant(0)
      const.24 = s32[1]{0} constant({24})
      all-gather = f64[1,8,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={1},
        use_global_device_ids=true
      const.list = s32[8]{0} constant({0, 1, 2, 3, 0, 1, 2, 3})
      partition-id = u32[] partition-id()
      ds1 = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      multiply.8 = s32[1]{0} multiply(ds1, const.24)
      reshape.15 = s32[] reshape(multiply.8)
      ROOT ds = f64[1,2,40]{2,1,0} dynamic-slice(all-gather,
        zero_idx, reshape.15, zero_idx), dynamic_slice_sizes={1,2,40}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->name(), "ds");
  HloInstruction* pad_hlo = nullptr;
  HloInstruction* ag_hlo = nullptr;
  EXPECT_FALSE(MatchDsPadAllGather(root, &pad_hlo, &ag_hlo));
}

TEST_F(MatchDsPadAllGatherTest, MatchDsPadAllGatherNoAllGather) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {
      param = f64[1,8,40]{2,1,0} parameter(0)
      zero = f64[] constant(0)
      zero_idx = s32[] constant(0)
      const.24 = s32[1]{0} constant({24})
      pad = f64[1,96,40]{2,1,0} pad(param, zero), padding=0_0x0_88x0_0
      const.list = s32[8]{0} constant({0, 1, 2, 3, 0, 1, 2, 3})
      partition-id = u32[] partition-id()
      ds1 = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      multiply.8 = s32[1]{0} multiply(ds1, const.24)
      reshape.15 = s32[] reshape(multiply.8)
      ROOT ds = f64[1,24,40]{2,1,0} dynamic-slice(pad, zero_idx,
        reshape.15, zero_idx), dynamic_slice_sizes={1,24,40}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->name(), "ds");
  HloInstruction* pad_hlo = nullptr;
  HloInstruction* ag_hlo = nullptr;
  EXPECT_FALSE(MatchDsPadAllGather(root, &pad_hlo, &ag_hlo));
}

}  // namespace
}  // namespace xla
