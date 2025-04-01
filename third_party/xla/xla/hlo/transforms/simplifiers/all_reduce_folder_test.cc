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

#include "xla/hlo/transforms/simplifiers/all_reduce_folder.h"

#include <cstddef>
#include <initializer_list>
#include <memory>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

namespace matcher = xla::testing::opcode_matchers;
using ::testing::HasSubstr;

class AllReduceFolderTest : public HloHardwareIndependentTestBase {};

const char *k2AllReduce = R"(
    HloModule m

    sum {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT add.2 = f32[] add(a, b)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      ar0 = f32[8] all-reduce(p0), replica_groups=$group_0, to_apply=sum
      ROOT ar1 = f32[8] all-reduce(ar0), replica_groups=$group_1, to_apply=sum
    }
  )";

size_t AllReduceCount(HloModule *module) {
  return absl::c_count_if(module->entry_computation()->instructions(),
                          HloPredicateIsOp<HloOpcode::kAllReduce>);
}

void ExpectOneAllReduce(HloModule *module,
                        absl::string_view target_replica_groups) {
  EXPECT_EQ(AllReduceCount(module), 1);
  HloInstruction *root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, matcher::AllReduce(matcher::Parameter(0)));
  EXPECT_THAT(root->ToString(), HasSubstr(target_replica_groups));
}

TEST_F(AllReduceFolderTest, Simple) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, RunAndCheckHloRewrite(k2AllReduce, AllReduceFolder(), true,
                                         {{"$group_0", "{{0,1},{2,3}}"},
                                          {"$group_1", "{{0,2},{1,3}}"}}));
  ExpectOneAllReduce(module.get(), "replica_groups={{0,1,2,3}}");
}

// Same as Simple, but groups for the 2 all-reduce's are swapped.
TEST_F(AllReduceFolderTest, SimpleSwap) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, RunAndCheckHloRewrite(k2AllReduce, AllReduceFolder(), true,
                                         {{"$group_1", "{{0,1},{2,3}}"},
                                          {"$group_0", "{{0,2},{1,3}}"}}));
  ExpectOneAllReduce(module.get(), "replica_groups={{0,1,2,3}}");
}

TEST_F(AllReduceFolderTest, BothEmptyReplicaGroups_NotTransformed) {
  TF_ASSERT_OK(RunAndCheckHloRewrite(k2AllReduce, AllReduceFolder(), false,
                                     {{"$group_0", "{}"}, {"$group_1", "{}"}}));
}

TEST_F(AllReduceFolderTest, EmptyReplicaGroups_NotTransformed) {
  TF_ASSERT_OK(RunAndCheckHloRewrite(
      k2AllReduce, AllReduceFolder(), false,
      {{"$group_0", "{}"}, {"$group_1", "{{0,2},{1,3}}"}}));
}

TEST_F(AllReduceFolderTest, MismatchOtherProperties0_NotTransformed) {
  absl::string_view hlo_string = R"(
    HloModule m

    sum {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT add.2 = f32[] add(a, b)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      ar0 = f32[8] all-reduce(p0), replica_groups={{0,1},{2,3}}, channel_id=1, to_apply=sum
      ROOT ar1 = f32[8] all-reduce(ar0), replica_groups={{0,2},{1,3}}, to_apply=sum
    }
    )";
  TF_ASSERT_OK(RunAndCheckHloRewrite(hlo_string, AllReduceFolder(), false));
}

TEST_F(AllReduceFolderTest, MismatchOtherProperties1_NotTransformed) {
  absl::string_view hlo_string = R"(
    HloModule m

    sum {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT add.2 = f32[] add(a, b)
    }

    mul {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT mul = f32[] multiply(a, b)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      ar0 = f32[8] all-reduce(p0), replica_groups={{0,1},{2,3}}, to_apply=sum
      ROOT ar1 = f32[8] all-reduce(ar0), replica_groups={{0,2},{1,3}}, to_apply=mul
    }
    )";
  TF_ASSERT_OK(RunAndCheckHloRewrite(hlo_string, AllReduceFolder(), false));
}

TEST_F(AllReduceFolderTest, NotFoldable_NotTransformed) {
  absl::string_view hlo_string = R"(
    HloModule m

    sum {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT add.2 = f32[] add(a, b)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      ar0 = f32[8] all-reduce(p0), replica_groups={{0,1},{2,3}}, to_apply=sum
      ROOT ar1 = f32[8] all-reduce(ar0), replica_groups={{0,1},{2,3}}, to_apply=sum
    }
    )";
  TF_ASSERT_OK(RunAndCheckHloRewrite(hlo_string, AllReduceFolder(), false));
}

TEST_F(AllReduceFolderTest, Foldable0) {
  absl::string_view hlo_string = R"(
    HloModule m

    sum {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT add.2 = f32[] add(a, b)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      ar0 = f32[8] all-reduce(p0), replica_groups={{0,4},{1,5},{2,3},{6,7}}, to_apply=sum
      ROOT ar1 = f32[8] all-reduce(ar0), replica_groups={{0,5},{4,1},{2,7},{3,6}}, to_apply=sum
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunAndCheckHloRewrite(hlo_string, AllReduceFolder()));
  ExpectOneAllReduce(module.get(), "replica_groups={{0,1,4,5},{2,3,6,7}}");
}

// Verify that a chain of foldable all-reduce's folds in a single pass
// invocation.
TEST_F(AllReduceFolderTest, FoldableChain) {
  absl::string_view hlo_string = R"(
    HloModule m

    sum {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT add.2 = f32[] add(a, b)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      ar0 = f32[8] all-reduce(p0), replica_groups={{0,1},{2,3},{4,5},{6,7}}, to_apply=sum
      ar1 = f32[8] all-reduce(ar0), replica_groups={{0,2},{1,3},{4,6},{5,7}}, to_apply=sum
      ROOT ar2 = f32[8] all-reduce(ar1), replica_groups={{0,4},{1,5},{2,6},{3,7}}, to_apply=sum
    }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunAndCheckHloRewrite(hlo_string, AllReduceFolder()));
  ExpectOneAllReduce(module.get(), "replica_groups={{0,1,2,3,4,5,6,7}}");
}

}  // namespace
}  // namespace xla
