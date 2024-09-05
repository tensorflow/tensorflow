/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/all_reduce_promotion.h"

#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace {
namespace m = ::xla::match;

class AllReducePromotionTest : public HloTestBase {
 public:
  AllReducePromotion pass_{{{U16, U32}, {S16, S32}}};
};

TEST_F(AllReducePromotionTest, SimplePromotionAllReduce) {
  absl::string_view hlo_text = R"(
  HloModule test

  sum {
    a = u16[] parameter(0)
    b = u16[] parameter(1)
    ROOT add.2 = u16[] add(a, b)
  }

  ENTRY test_computation {
    id32 = u32[] replica-id()
    id = u16[] convert(id32)
    id2 = u16[2] broadcast(id), dimensions={}
    a0 = u16[2] constant({10, 15})
    a1 = u16[2] add(id2, a0)
    ROOT cp = u16[2] all-reduce(a1), replica_groups={}, to_apply=sum
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass_, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Convert(m::AllReduce(m::Convert().WithShape(U32, {2}))
                                .WithShape(U32, {2}))
                     .WithShape(U16, {2})));
}

TEST_F(AllReducePromotionTest, SimplePromotionReduceScatter) {
  absl::string_view hlo_text = R"(
  HloModule test

  sum {
    a = u16[] parameter(0)
    b = u16[] parameter(1)
    ROOT add.2 = u16[] add(a, b)
  }

  ENTRY test_computation {
    id32 = u32[] replica-id()
    id = u16[] convert(id32)
    id2 = u16[2] broadcast(id), dimensions={}
    a0 = u16[2] constant({10, 15})
    a1 = u16[2] add(id2, a0)
    ROOT cp = u16[1] reduce-scatter(a1), dimensions={0}, replica_groups={}, to_apply=sum
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass_, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Convert(m::ReduceScatter(m::Convert().WithShape(U32, {2}))
                                .WithShape(U32, {1}))
                     .WithShape(U16, {1})));
}

}  // namespace
}  // namespace xla
