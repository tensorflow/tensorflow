/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/collective_transformation_reorderer.h"

#include <gmock/gmock.h>
#include "tensorflow/compiler/xla/hlo/utils/hlo_matchers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

class CollectiveTransformationReordererTest : public HloTestBase {
 public:
  StatusOr<bool> RunCollectiveTransformationReorderer(HloModule* module) {
    CollectiveTransformationReorder reorderer;
    return reorderer.Run(module, {});
  }
};

TEST_F(CollectiveTransformationReordererTest,
       ReshapeWithinShardAfterAllGatherDim) {
  absl::string_view hlo_string = R"(
  HloModule module
  ENTRY entry {
    param = bf16[8,4,1024] parameter(0)
    all-gather = bf16[8,32,1024] all-gather(param), dimensions={1}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1
    ROOT reshape = bf16[8,32,8,128] reshape(all-gather)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunCollectiveTransformationReorderer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::AllGather(op::Reshape(op::Parameter())));
  HloInstruction* all_gather = module->entry_computation()->root_instruction();
  EXPECT_THAT(all_gather->dimensions(), ::testing::ElementsAre(1));
}

TEST_F(CollectiveTransformationReordererTest,
       ReshapeWithinShardBeforeAllGatherDim) {
  absl::string_view hlo_string = R"(
  HloModule module
  ENTRY entry {
    param = bf16[8,32,8,4,1024] parameter(0)
    all-gather = bf16[8,32,8,32,1024] all-gather(param), dimensions={3}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1
    ROOT reshape = bf16[2048,32,1024] reshape(all-gather)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunCollectiveTransformationReorderer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::AllGather(op::Reshape(op::Parameter())));
  HloInstruction* all_gather = module->entry_computation()->root_instruction();
  EXPECT_THAT(all_gather->dimensions(), ::testing::ElementsAre(1));
}

TEST_F(CollectiveTransformationReordererTest,
       ReshapeWithinShardBeforeAndAfterAllGatherDim) {
  absl::string_view hlo_string = R"(
  HloModule module
  ENTRY entry {
    param = bf16[8,32,8,4,1024] parameter(0)
    all-gather = bf16[8,32,8,32,1024] all-gather(param), dimensions={3}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1
    ROOT reshape = bf16[2048,32,8,128] reshape(all-gather)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunCollectiveTransformationReorderer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::AllGather(op::Reshape(op::Parameter())));
  HloInstruction* all_gather = module->entry_computation()->root_instruction();
  EXPECT_THAT(all_gather->dimensions(), ::testing::ElementsAre(1));
}

TEST_F(CollectiveTransformationReordererTest, ReshapeAcrossShards) {
  absl::string_view hlo_string = R"(
  HloModule module
  ENTRY entry {
    param = bf16[8,1,8,128] parameter(0)
    all-gather = bf16[8,8,8,128] all-gather(param), dimensions={1}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1
    ROOT reshape = bf16[64,8,128] reshape(all-gather)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunCollectiveTransformationReorderer(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectiveTransformationReordererTest, MergeAllGatherDimensionWithNext) {
  absl::string_view hlo_string = R"(
  HloModule module
  ENTRY entry {
    param = bf16[8,8,16,16] parameter(0)
    all-gather = bf16[64,8,16,16] all-gather(param), dimensions={0}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1
    ROOT reshape = bf16[512,16,16] reshape(all-gather)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunCollectiveTransformationReorderer(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectiveTransformationReordererTest,
       MergeAllGatherDimensionWithPrevious) {
  absl::string_view hlo_string = R"(
  HloModule module
  ENTRY entry {
    param = bf16[8,8,16,16] parameter(0)
    all-gather = bf16[8,64,16,16] all-gather(param), dimensions={1}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1
    ROOT reshape = bf16[512,16,16] reshape(all-gather)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunCollectiveTransformationReorderer(module.get()));
  EXPECT_FALSE(changed);
}

}  // namespace

}  // namespace xla
