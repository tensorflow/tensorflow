/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/sharding_remover.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

using ShardingRemoverTest = HloTestBase;

TEST_F(ShardingRemoverTest, RemoveSharding) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
 %parameter.3379 = f32[1,1]{1,0} parameter(0)
 %custom-call.3380 = f32[1,1]{1,0} custom-call(f32[1,1]{1,0} %parameter.3379),
   custom_call_target="Sharding", sharding={replicated}
 ROOT %reshape.6032 = f32[] reshape(f32[1,1]{1,0} %custom-call.3380)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, ShardingRemover().Run(module.get()));
  EXPECT_TRUE(changed);
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Reshape(op::Parameter()));
}

TEST_F(ShardingRemoverTest, RemoveSPMDShardingToFullShape) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
 %parameter.3379 = f32[1,1]{1,0} parameter(0)
 %custom-call.3380 = f32[1,1]{1,0} custom-call(f32[1,1]{1,0} %parameter.3379),
   custom_call_target="SPMDShardToFullShape", sharding={replicated}
 ROOT %reshape.6032 = f32[] reshape(f32[1,1]{1,0} %custom-call.3380)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, ShardingRemover().Run(module.get()));
  EXPECT_TRUE(changed);
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Reshape(op::Parameter()));
}

TEST_F(ShardingRemoverTest, RemoveSPMDFullToShardShape) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
 %parameter.3379 = f32[1,1]{1,0} parameter(0)
 %custom-call.3380 = f32[1,1]{1,0} custom-call(f32[1,1]{1,0} %parameter.3379),
   custom_call_target="SPMDFullToShardShape", sharding={replicated}
 ROOT %reshape.6032 = f32[] reshape(f32[1,1]{1,0} %custom-call.3380)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, ShardingRemover().Run(module.get()));
  EXPECT_TRUE(changed);
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Reshape(op::Parameter()));
}

TEST_F(ShardingRemoverTest, NoChangeForOtherCustomCall) {
  const char* const hlo_string = R"(
HloModule cluster_2013453984438090939__.47

ENTRY %cluster_2013453984438090939__.47
  (arg_tuple.1: ()) -> (bf16[2,2000], s32[2,2000]) {
  %arg_tuple.1 = bf16[2,209664] parameter(0)
  %custom-call = (bf16[2,2000]{1,0}, s32[2,2000]{1,0})
    custom-call(bf16[2,209664]{1,0} %arg_tuple.1), custom_call_target="TopK"
  %get-tuple-element = bf16[2,2000]{1,0}
    get-tuple-element((bf16[2,2000]{1,0}, s32[2,2000]{1,0}) %custom-call),
    index=0
  %get-tuple-element.1 = s32[2,2000]{1,0} get-tuple-element((bf16[2,2000]{1,0},
    s32[2,2000]{1,0}) %custom-call), index=1, sharding={replicated}
  ROOT %tuple.46 = (bf16[2,2000]{1,0}, s32[2,2000]{1,0})
    tuple(bf16[2,2000]{1,0} %get-tuple-element, s32[2,2000]{1,0}
    %get-tuple-element.1),
    metadata={op_name="XLA_Retvals"}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, ShardingRemover().Run(module.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace xla
