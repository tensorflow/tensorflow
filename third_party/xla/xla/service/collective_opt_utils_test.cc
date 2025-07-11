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

}  // namespace
}  // namespace xla
