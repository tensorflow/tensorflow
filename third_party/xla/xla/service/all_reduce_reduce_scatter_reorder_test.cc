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

#include "xla/service/all_reduce_reduce_scatter_reorder.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class AllReduceReduceScatterReorderTest
    : public HloHardwareIndependentTestBase {
 public:
  AllReduceReduceScatterReorder pass_;
};

TEST_F(AllReduceReduceScatterReorderTest, KeepingReplicaGroups) {
  absl::string_view hlo_text = R"(
  sum {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    p0 = f32[8] parameter(0)
    ar = f32[8] all-reduce(p0), replica_groups={{0,1}, {2,3}}, to_apply=sum
    ROOT rs = f32[4] reduce-scatter(ar), dimensions={0}, replica_groups={{0,2}, {1,3}}, to_apply=sum
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass_, module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* param = module->entry_computation()->parameter_instruction(0);
  HloInstruction* new_rs = param->users().front();
  HloInstruction* new_ar = new_rs->users().front();

  EXPECT_EQ(new_rs->replica_groups().size(), 2);
  EXPECT_THAT(new_rs->replica_groups()[0].replica_ids(),
              ::testing::ElementsAre(0, 2));
  EXPECT_THAT(new_rs->replica_groups()[1].replica_ids(),
              ::testing::ElementsAre(1, 3));

  EXPECT_EQ(new_ar->replica_groups().size(), 2);
  EXPECT_THAT(new_ar->replica_groups()[0].replica_ids(),
              ::testing::ElementsAre(0, 1));
  EXPECT_THAT(new_ar->replica_groups()[1].replica_ids(),
              ::testing::ElementsAre(2, 3));

  EXPECT_EQ(new_rs->opcode(), HloOpcode::kReduceScatter);
  EXPECT_EQ(new_ar->opcode(), HloOpcode::kAllReduce);

  EXPECT_EQ(new_rs->shape().dimensions(0), 4);
  EXPECT_EQ(new_ar->shape().dimensions(0), 4);

  EXPECT_EQ(new_ar, module->entry_computation()->root_instruction());
}

TEST_F(AllReduceReduceScatterReorderTest, AllReduceMultipleUsers) {
  absl::string_view hlo_text = R"(
  sum {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    p0 = f32[8] parameter(0)
    ar = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
    rs = f32[4] reduce-scatter(ar), dimensions={0}, replica_groups={}, to_apply=sum
    ROOT tuple = (f32[8], f32[4]) tuple(ar, rs)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass_, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(AllReduceReduceScatterReorderTest, DifferentReductionFunctions) {
  absl::string_view hlo_text = R"(
  sum {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  product {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT product = f32[] multiply(a, b)
  }

  ENTRY main {
    p0 = f32[8] parameter(0)
    ar = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
    ROOT rs = f32[4] reduce-scatter(ar), dimensions={0}, replica_groups={}, to_apply=product
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass_, module.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace xla
