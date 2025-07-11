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

class CheckUniformReplicaGroupsTest : public HloHardwareIndependentTestBase {};
TEST_F(CheckUniformReplicaGroupsTest, CheckUniformReplicaGroupsUniform) {
  absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      param = f32[16,10] parameter(0)
      ROOT ag = f32[16,10] all-gather(param), dimensions={0},
          replica_groups={{0,1},{2,3}}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  const auto* ag = Cast<HloAllGatherInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_TRUE(CheckUniformReplicaGroups(ag));
}

TEST_F(CheckUniformReplicaGroupsTest, CheckUniformReplicaGroupsNonUniform) {
  absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      param = f32[16,10] parameter(0)
      ROOT ag = f32[16,10] all-gather(param), dimensions={0},
          replica_groups={{0,1},{2}}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  const auto* ag = Cast<HloAllGatherInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_FALSE(CheckUniformReplicaGroups(ag));
}

TEST_F(CheckUniformReplicaGroupsTest, CheckUniformReplicaGroupsSingleGroup) {
  absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      param = f32[16,10] parameter(0)
      ROOT ag = f32[16,10] all-gather(param), dimensions={0},
        replica_groups={{0,1,2,3}}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  const auto* ag = Cast<HloAllGatherInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_TRUE(CheckUniformReplicaGroups(ag));
}

class ExtractSplitDimSpecTest : public HloHardwareIndependentTestBase {};

TEST_F(ExtractSplitDimSpecTest, SingleDim) {
  absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      param = f32[16,10] parameter(0)
      zero = s32[] constant(0)
      ds_index = s32[] parameter(1)
      ROOT ds = f32[8,10] dynamic-slice(param, ds_index, zero),
        dynamic_slice_sizes={8,10}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* ds = module->entry_computation()->root_instruction();
  ASSERT_THAT(ds, testing::NotNull());

  std::optional<SplitDimSpec> spec =
      ExtractSplitDimSpec(*ds, /*allow_multiple_split_dims=*/false);

  ASSERT_TRUE(spec.has_value());
  EXPECT_EQ(spec->split_dim, 0);
  EXPECT_THAT(spec->split_dims, testing::ElementsAre(0));
}

TEST_F(ExtractSplitDimSpecTest, MultipleDimOffsets) {
  absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      param = f32[16,20] parameter(0)
      zero = s32[] constant(0)
      ds_index1 = s32[] parameter(1)
      ROOT ds = f32[8,10] dynamic-slice(param, zero, ds_index1),
        dynamic_slice_sizes={8,10}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* ds = module->entry_computation()->root_instruction();
  ASSERT_THAT(ds, testing::NotNull());

  std::optional<SplitDimSpec> spec =
      ExtractSplitDimSpec(*ds, /*allow_multiple_split_dims=*/false);

  ASSERT_TRUE(spec.has_value());
  EXPECT_EQ(spec->split_dim, 1);
  EXPECT_THAT(spec->split_dims, testing::ElementsAre(1));
}

TEST_F(ExtractSplitDimSpecTest, MultipleSplitDimsAllowed) {
  absl::string_view hlo_string = R"(
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* ds = module->entry_computation()->root_instruction();
  ASSERT_THAT(ds, testing::NotNull());

  std::optional<SplitDimSpec> spec =
      ExtractSplitDimSpec(*ds, /*allow_multiple_split_dims=*/true);

  ASSERT_TRUE(spec.has_value());
  EXPECT_EQ(spec->split_dim, 1);
  EXPECT_THAT(spec->split_dims, testing::ElementsAre(1, 2));
}

TEST_F(ExtractSplitDimSpecTest, MultipleSplitDimsNonConsecutive) {
  absl::string_view hlo_string = R"(
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* ds = module->entry_computation()->root_instruction();
  ASSERT_THAT(ds, testing::NotNull());

  std::optional<SplitDimSpec> spec =
      ExtractSplitDimSpec(*ds, /*allow_multiple_split_dims=*/true);

  EXPECT_FALSE(spec.has_value());
}

TEST_F(ExtractSplitDimSpecTest, MultipleSplitDimsNotAllowed) {
  absl::string_view hlo_string = R"(
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* ds = module->entry_computation()->root_instruction();
  ASSERT_THAT(ds, testing::NotNull());

  std::optional<SplitDimSpec> spec =
      ExtractSplitDimSpec(*ds, /*allow_multiple_split_dims=*/false);

  EXPECT_FALSE(spec.has_value());
}

TEST_F(ExtractSplitDimSpecTest, NoSplitDimFound) {
  absl::string_view hlo_string = R"(
HloModule test
ENTRY main {
  param = f32[16,10] parameter(0)
  zero = s32[] constant(0)
  ROOT ds = f32[16,10] dynamic-slice(param, zero, zero),
    dynamic_slice_sizes={16,10}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* ds = module->entry_computation()->root_instruction();
  ASSERT_THAT(ds, testing::NotNull());

  std::optional<SplitDimSpec> spec =
      ExtractSplitDimSpec(*ds, /*allow_multiple_split_dims=*/false);

  ASSERT_TRUE(spec.has_value());
  EXPECT_EQ(spec->split_dim, -1);
  EXPECT_TRUE(spec->split_dims.empty());
}

class ExtractDynamicSliceFromCollectiveUserTest
    : public HloHardwareIndependentTestBase {};

TEST_F(ExtractDynamicSliceFromCollectiveUserTest, Basic) {
  absl::string_view hlo_string = R"(
HloModule test
ENTRY main {
  param = f32[16,10] parameter(0)
  ag = f32[32,10] all-gather(param), dimensions={0}, replica_groups={{0,1}}, metadata={op_name="ag"}
  zero = s32[] constant(0)
  replica_id = u32[] replica-id()
  slice_index = s32[] convert(replica_id)
  ROOT ds = f32[16,10] dynamic-slice(ag, slice_index, zero), dynamic_slice_sizes={16,10}, metadata={op_name="ds"}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  auto* ag = Cast<HloAllGatherInstruction>(
      module->entry_computation()->GetInstructionWithName("ag"));
  ASSERT_THAT(ag, testing::NotNull());

  HloInstruction* ds_user = nullptr;
  HloInstruction* bitcast = nullptr;
  HloInstruction* reshape = nullptr;
  ExtractDynamicSliceFromCollectiveUser(ag, /*allow_multiple_users=*/false,
                                        /*allow_intervening_reshape=*/false,
                                        /*allow_intervening_bitcast=*/false,
                                        &ds_user, &bitcast, &reshape);
  ASSERT_NE(ds_user, nullptr);
  EXPECT_EQ(ds_user->opcode(), HloOpcode::kDynamicSlice);
  EXPECT_EQ(ds_user->name(), "ds");
  EXPECT_EQ(bitcast, nullptr);
  EXPECT_EQ(reshape, nullptr);
}

TEST_F(ExtractDynamicSliceFromCollectiveUserTest, WithReshape) {
  absl::string_view hlo_string = R"(
HloModule test
ENTRY main {
  param = f32[16,10] parameter(0)
  ag = f32[32,10] all-gather(param), dimensions={0}, replica_groups={{0,1}}, metadata={op_name="ag"}
  reshape = f32[320] reshape(ag), metadata={op_name="reshape"}
  zero = s32[] constant(0)
  replica_id = u32[] replica-id()
  slice_index = s32[] convert(replica_id)
  ROOT ds = f32[160] dynamic-slice(reshape, slice_index), dynamic_slice_sizes={160}, metadata={op_name="ds"}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  auto* ag = Cast<HloAllGatherInstruction>(
      module->entry_computation()->GetInstructionWithName("ag"));
  ASSERT_THAT(ag, testing::NotNull());
  HloInstruction* ds_user = nullptr;
  HloInstruction* bitcast = nullptr;
  HloInstruction* reshape_instr = nullptr;
  ExtractDynamicSliceFromCollectiveUser(ag, /*allow_multiple_users=*/false,
                                        /*allow_intervening_reshape=*/true,
                                        /*allow_intervening_bitcast=*/false,
                                        &ds_user, &bitcast, &reshape_instr);
  ASSERT_NE(ds_user, nullptr);
  EXPECT_EQ(ds_user->opcode(), HloOpcode::kDynamicSlice);
  EXPECT_EQ(ds_user->name(), "ds");
  EXPECT_NE(reshape_instr, nullptr);
  EXPECT_EQ(reshape_instr->name(), "reshape");
  EXPECT_EQ(bitcast, nullptr);
}

TEST_F(ExtractDynamicSliceFromCollectiveUserTest, WithBitcast) {
  absl::string_view hlo_string = R"(
HloModule test
ENTRY main {
  param = f32[16,10] parameter(0)
  ag = f32[32,10] all-gather(param), dimensions={0}, replica_groups={{0,1}}, metadata={op_name="ag"}
  bitcast = f32[32,10] bitcast(ag), metadata={op_name="bitcast"}
  zero = s32[] constant(0)
  replica_id = u32[] replica-id()
  slice_index = s32[] convert(replica_id)
  ROOT ds = f32[16,10] dynamic-slice(bitcast, slice_index, zero), dynamic_slice_sizes={16,10}, metadata={op_name="ds"}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  auto* ag = Cast<HloAllGatherInstruction>(
      module->entry_computation()->GetInstructionWithName("ag"));
  ASSERT_THAT(ag, testing::NotNull());

  HloInstruction* ds_user = nullptr;
  HloInstruction* bitcast_instr = nullptr;
  HloInstruction* reshape = nullptr;
  ExtractDynamicSliceFromCollectiveUser(ag, /*allow_multiple_users=*/false,
                                        /*allow_intervening_reshape=*/false,
                                        /*allow_intervening_bitcast=*/true,
                                        &ds_user, &bitcast_instr, &reshape);
  ASSERT_NE(ds_user, nullptr);
  EXPECT_EQ(ds_user->opcode(), HloOpcode::kDynamicSlice);
  EXPECT_EQ(ds_user->name(), "ds");
  EXPECT_NE(bitcast_instr, nullptr);
  EXPECT_EQ(bitcast_instr->name(), "bitcast");
  EXPECT_EQ(reshape, nullptr);
}

TEST_F(ExtractDynamicSliceFromCollectiveUserTest, WithReshapeAndBitcast) {
  absl::string_view hlo_string = R"(
HloModule test
ENTRY main {
  param = f32[16,10] parameter(0)
  ag = f32[32,10] all-gather(param), dimensions={0}, replica_groups={{0,1}}, metadata={op_name="ag"}
  reshape = f32[320] reshape(ag), metadata={op_name="reshape"}
  bitcast = f32[320] bitcast(reshape), metadata={op_name="bitcast"}
  replica_id = u32[] replica-id()
  slice_index = s32[] convert(replica_id)
  ROOT ds = f32[160] dynamic-slice(bitcast, slice_index), dynamic_slice_sizes={160}, metadata={op_name="ds"}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  auto* ag = Cast<HloAllGatherInstruction>(
      module->entry_computation()->GetInstructionWithName("ag"));
  ASSERT_THAT(ag, testing::NotNull());

  HloInstruction* ds_user = nullptr;
  HloInstruction* bitcast_instr = nullptr;
  HloInstruction* reshape_instr = nullptr;
  ExtractDynamicSliceFromCollectiveUser(ag, /*allow_multiple_users=*/false,
                                        /*allow_intervening_reshape=*/true,
                                        /*allow_intervening_bitcast=*/true,
                                        &ds_user, &bitcast_instr,
                                        &reshape_instr);
  ASSERT_NE(ds_user, nullptr);
  EXPECT_EQ(ds_user->opcode(), HloOpcode::kDynamicSlice);
  EXPECT_EQ(ds_user->name(), "ds");
  EXPECT_NE(reshape_instr, nullptr);
  EXPECT_EQ(reshape_instr->name(), "reshape");
  EXPECT_NE(bitcast_instr, nullptr);
  EXPECT_EQ(bitcast_instr->name(), "bitcast");
}

TEST_F(ExtractDynamicSliceFromCollectiveUserTest, MultipleUsersWithReshape) {
  absl::string_view hlo_string = R"(
HloModule test
ENTRY main {
  param = f32[16,10] parameter(0)
  ag = f32[32,10] all-gather(param), dimensions={0}, replica_groups={{0,1}}, metadata={op_name="ag"}
  neg = f32[32,10] negate(ag)
  reshape = f32[320] reshape(ag), metadata={op_name="reshape"}
  bitcast = f32[320] bitcast(reshape), metadata={op_name="bitcast"}
  replica_id = u32[] replica-id()
  slice_index = s32[] convert(replica_id)
  ROOT ds = f32[160] dynamic-slice(bitcast, slice_index), dynamic_slice_sizes={160}, metadata={op_name="ds"}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  auto* ag = Cast<HloAllGatherInstruction>(
      module->entry_computation()->GetInstructionWithName("ag"));
  ASSERT_THAT(ag, testing::NotNull());

  HloInstruction* ds_user = nullptr;
  HloInstruction* bitcast_instr = nullptr;
  HloInstruction* reshape_instr = nullptr;
  ExtractDynamicSliceFromCollectiveUser(ag, /*allow_multiple_users=*/true,
                                        /*allow_intervening_reshape=*/true,
                                        /*allow_intervening_bitcast=*/true,
                                        &ds_user, &bitcast_instr,
                                        &reshape_instr);
  ASSERT_NE(ds_user, nullptr);
  EXPECT_EQ(ds_user->opcode(), HloOpcode::kDynamicSlice);
  EXPECT_EQ(ds_user->name(), "ds");
  EXPECT_NE(reshape_instr, nullptr);
  EXPECT_EQ(reshape_instr->name(), "reshape");
  EXPECT_NE(bitcast_instr, nullptr);
  EXPECT_EQ(bitcast_instr->name(), "bitcast");
}

TEST_F(ExtractDynamicSliceFromCollectiveUserTest, NoDynamicSlice) {
  absl::string_view hlo_string = R"(
HloModule test
ENTRY main {
  param = f32[16,10] parameter(0)
  ag = f32[32,10] all-gather(param), dimensions={0}, replica_groups={{0,1}}, metadata={op_name="ag"}
  ROOT neg = f32[32,10] negate(ag)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  auto* ag = Cast<HloAllGatherInstruction>(
      module->entry_computation()->GetInstructionWithName("ag"));
  ASSERT_THAT(ag, testing::NotNull());

  HloInstruction* ds_user = nullptr;
  HloInstruction* bitcast = nullptr;
  HloInstruction* reshape = nullptr;
  ExtractDynamicSliceFromCollectiveUser(ag, /*allow_multiple_users=*/false,
                                        /*allow_intervening_reshape=*/true,
                                        /*allow_intervening_bitcast=*/true,
                                        &ds_user, &bitcast, &reshape);
  EXPECT_EQ(ds_user, nullptr);
}

TEST_F(ExtractDynamicSliceFromCollectiveUserTest,
       InterveningOpWithMultipleUsers) {
  absl::string_view hlo_string = R"(
HloModule test
ENTRY main {
  param = f32[16,10] parameter(0)
  ag = f32[32,10] all-gather(param), dimensions={0}, replica_groups={{0,1}}, metadata={op_name="ag"}
  reshape = f32[320] reshape(ag), metadata={op_name="reshape"}
  neg = f32[320] negate(reshape)
  replica_id = u32[] replica-id()
  slice_index = s32[] convert(replica_id)
  ROOT ds = f32[160] dynamic-slice(reshape, slice_index), dynamic_slice_sizes={160}, metadata={op_name="ds"}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  auto* ag = Cast<HloAllGatherInstruction>(
      module->entry_computation()->GetInstructionWithName("ag"));
  ASSERT_THAT(ag, testing::NotNull());

  HloInstruction* ds_user = nullptr;
  HloInstruction* bitcast = nullptr;
  HloInstruction* reshape = nullptr;
  ExtractDynamicSliceFromCollectiveUser(ag, /*allow_multiple_users=*/false,
                                        /*allow_intervening_reshape=*/true,
                                        /*allow_intervening_bitcast=*/true,
                                        &ds_user, &bitcast, &reshape);
  EXPECT_EQ(ds_user, nullptr);
}

using GetIndicesSpecForDynamicSliceTest = HloHardwareIndependentTestBase;

TEST_F(GetIndicesSpecForDynamicSliceTest, GetIndicesSpecForDynamicSliceTest) {
  absl::string_view hlo_string = R"(
HloModule module
ENTRY main {
  param = f32[1,1,1024] parameter(0)
  ag = f32[4,1,1024] all-gather(param), replica_groups={{0,1,2,3}}, dimensions={0}
  const = s32[4]{0} constant({0, 6, 4, 2})
  pid = u32[] partition-id()
  ds_offset = s32[] dynamic-slice(const, pid), dynamic_slice_sizes={1}
  ROOT ds = f32[1,1,1024] dynamic-slice(ag, ds_offset, s32[] constant(0), s32[] constant(0)), dynamic_slice_sizes={1,1,1024}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  auto* ag = module->entry_computation()->GetInstructionWithName("ag");
  auto* ds_offset =
      module->entry_computation()->GetInstructionWithName("ds_offset");
  auto map_id = [](const HloInstruction* hlo, int64_t id) {
    return (hlo->opcode() == HloOpcode::kPartitionId) ? id : -1;
  };
  auto indices_spec = GetIndicesSpecForDynamicSlice(
      Cast<HloAllGatherInstruction>(ag), ds_offset, map_id);
  ASSERT_TRUE(indices_spec.has_value());
  EXPECT_THAT(*indices_spec, ::testing::UnorderedElementsAre(
                                 ::testing::Pair(0, 0), ::testing::Pair(6, 1),
                                 ::testing::Pair(4, 2), ::testing::Pair(2, 3)));
}

TEST_F(GetIndicesSpecForDynamicSliceTest,
       GetIndicesSpecForDynamicSliceTestReshape) {
  absl::string_view hlo_string = R"(
HloModule module
ENTRY main {
  param = f32[1,1,1024] parameter(0)
  ag = f32[4,1,1024] all-gather(param), replica_groups={{0,1,2,3}}, dimensions={0}
  const = s32[4]{0} constant({0, 6, 4, 2})
  pid = u32[] partition-id()
  ds_offset = s32[] dynamic-slice(const, pid), dynamic_slice_sizes={1}
  ds_offset_reshape = s32[] reshape(ds_offset)
  ROOT ds = f32[1,1,1024] dynamic-slice(ag, ds_offset, s32[] constant(0), s32[] constant(0)), dynamic_slice_sizes={1,1,1024}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  auto* ag = module->entry_computation()->GetInstructionWithName("ag");
  auto* ds_offset =
      module->entry_computation()->GetInstructionWithName("ds_offset");
  auto map_id = [](const HloInstruction* hlo, int64_t id) {
    return (hlo->opcode() == HloOpcode::kPartitionId) ? id : -1;
  };
  auto indices_spec = GetIndicesSpecForDynamicSlice(
      Cast<HloAllGatherInstruction>(ag), ds_offset, map_id);
  ASSERT_TRUE(indices_spec.has_value());
  EXPECT_THAT(*indices_spec, ::testing::UnorderedElementsAre(
                                 ::testing::Pair(0, 0), ::testing::Pair(6, 1),
                                 ::testing::Pair(4, 2), ::testing::Pair(2, 3)));
}

TEST_F(GetIndicesSpecForDynamicSliceTest, GetAllGatherShardOffsetTest) {
  absl::string_view hlo_string = R"(
HloModule module
ENTRY main {
  param = f32[1,1,1024] parameter(0)
  ag = f32[4,1,1024] all-gather(param), replica_groups={{0,1,2,3}}, dimensions={0}
  const = s32[4]{0} constant({0, 6, 4, 2})
  pid = u32[] partition-id()
  ds_offset = s32[] dynamic-slice(const, pid), dynamic_slice_sizes={1}
  ds_offset_reshape = s32[] reshape(ds_offset)
  ROOT ds = f32[1,1,1024] dynamic-slice(ag, ds_offset, s32[] constant(0), s32[] constant(0)), dynamic_slice_sizes={1,1,1024}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  auto* ag = module->entry_computation()->GetInstructionWithName("ag");
  auto* ds_offset =
      module->entry_computation()->GetInstructionWithName("ds_offset");
  auto map_id = [](const HloInstruction* hlo, int64_t id) {
    return (hlo->opcode() == HloOpcode::kPartitionId) ? id : -1;
  };
  auto indices_spec = GetIndicesSpecForDynamicSlice(
      Cast<HloAllGatherInstruction>(ag), ds_offset, map_id);
  ASSERT_TRUE(indices_spec.has_value());
  EXPECT_THAT(*indices_spec, ::testing::UnorderedElementsAre(
                                 ::testing::Pair(0, 0), ::testing::Pair(6, 1),
                                 ::testing::Pair(4, 2), ::testing::Pair(2, 3)));
}

}  // namespace
}  // namespace xla
