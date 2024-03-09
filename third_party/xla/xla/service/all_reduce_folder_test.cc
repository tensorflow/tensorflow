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

#include "xla/service/all_reduce_folder.h"

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace m = xla::testing::opcode_matchers;
using ::testing::HasSubstr;

class AllReduceFolderTest : public HloTestBase {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> RunPass(
      absl::string_view hlo_module, bool expect_change) {
    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo_module));
    auto changed = AllReduceFolder().Run(module.get());
    if (!changed.ok()) {
      return changed.status();
    }
    EXPECT_EQ(changed.value(), expect_change);
    return absl::StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }

  size_t AllReduceCount(std::unique_ptr<HloModule> &module) {
    return absl::c_count_if(module->entry_computation()->instructions(),
                            HloPredicateIsOp<HloOpcode::kAllReduce>);
  }
};

TEST_F(AllReduceFolderTest, Simple) {
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
  ROOT ar1 = f32[8] all-reduce(ar0), replica_groups={{0,2},{1,3}}, to_apply=sum
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_EQ(AllReduceCount(module), 1);
  HloInstruction *root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, m::AllReduce(m::Parameter(0)));
  EXPECT_THAT(root->ToString(), HasSubstr("replica_groups={{0,1,2,3}}"));
}

// Same as Simple, but groups for the 2 all-reduce's are swapped.
TEST_F(AllReduceFolderTest, SimpleSwap) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  ar0 = f32[8] all-reduce(p0), replica_groups={{0,2},{1,3}}, to_apply=sum
  ROOT ar1 = f32[8] all-reduce(ar0), replica_groups={{0,1},{2,3}}, to_apply=sum
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_EQ(AllReduceCount(module), 1);
  HloInstruction *root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, m::AllReduce(m::Parameter(0)));
  EXPECT_THAT(root->ToString(), HasSubstr("replica_groups={{0,1,2,3}}"));
}

TEST_F(AllReduceFolderTest, EmptyReplicaGroups) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  ROOT ar1 = f32[8] all-reduce(ar0), replica_groups={{0,2},{1,3}}, to_apply=sum
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(AllReduceFolderTest, MismatchOtherProperties0) {
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(AllReduceFolderTest, MismatchOtherProperties1) {
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(AllReduceFolderTest, NotFoldable) {
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
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
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_EQ(AllReduceCount(module), 1);
  HloInstruction *root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, m::AllReduce(m::Parameter(0)));
  EXPECT_THAT(root->ToString(),
              HasSubstr("replica_groups={{0,1,4,5},{2,3,6,7}}"));
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
                          RunPass(hlo_string, /*expect_change=*/true));
  std::cerr << module->ToString();
  EXPECT_EQ(AllReduceCount(module), 1);
  HloInstruction *root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, m::AllReduce(m::Parameter(0)));
  EXPECT_THAT(root->ToString(),
              HasSubstr("replica_groups={{0,1,2,3,4,5,6,7}}"));
}

}  // namespace
}  // namespace xla
