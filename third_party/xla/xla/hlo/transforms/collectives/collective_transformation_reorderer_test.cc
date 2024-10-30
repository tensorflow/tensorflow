/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/hlo/transforms/collectives/collective_transformation_reorderer.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
namespace op = xla::testing::opcode_matchers;

class CollectiveTransformationReordererTest
    : public HloHardwareIndependentTestBase {
 public:
  absl::StatusOr<bool> RunCollectiveTransformationReorderer(HloModule* module) {
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
  EXPECT_THAT(all_gather->dimensions(), ElementsAre(1));
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
  EXPECT_THAT(all_gather->dimensions(), ElementsAre(1));
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
  EXPECT_THAT(all_gather->dimensions(), ElementsAre(1));
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

TEST_F(CollectiveTransformationReordererTest, AllReduceSingleReshape) {
  absl::string_view hlo_string = R"(
  HloModule module

  add {
    a = bf16[] parameter(0)
    b = bf16[] parameter(1)
    ROOT s = bf16[] add(a, b)
  }

  ENTRY entry {
    param = bf16[16384,6144] parameter(0)
    reshape = bf16[1,16384,6144] reshape(param)
    all-reduce = bf16[1,16384,6144] all-reduce(reshape), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=add
    constant = s32[] constant(0)
    ROOT dynamic-slice = bf16[1,16384,384] dynamic-slice(all-reduce, constant, constant, constant), dynamic_slice_sizes={1,16384,384}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunCollectiveTransformationReorderer(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/false,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::DynamicSlice(op::Reshape(op::AllReduce(op::Parameter())),
                               op::Constant(), op::Constant(), op::Constant()));
}

TEST_F(CollectiveTransformationReordererTest, AllReduceTwoReshapes) {
  absl::string_view hlo_string = R"(
  HloModule module

  add {
    a = bf16[] parameter(0)
    b = bf16[] parameter(1)
    ROOT s = bf16[] add(a, b)
  }

  ENTRY entry {
    param = bf16[16384,3072,2] parameter(0)
    reshape.1 = bf16[16384,6144] reshape(param)
    reshape.2 = bf16[1,16384,6144] reshape(reshape.1)
    all-reduce = bf16[1,16384,6144] all-reduce(reshape.2), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=add
    constant = s32[] constant(0)
    ROOT dynamic-slice = bf16[1,16384,384] dynamic-slice(all-reduce, constant, constant, constant), dynamic_slice_sizes={1,16384,384}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunCollectiveTransformationReorderer(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/false,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::DynamicSlice(op::Reshape(op::Reshape(op::AllReduce(op::Parameter()))),
                       op::Constant(), op::Constant(), op::Constant()));
}

TEST_F(CollectiveTransformationReordererTest, AllReduceReshapeWithTwoUsers) {
  absl::string_view hlo_string = R"(
  HloModule module

  add {
    a = bf16[] parameter(0)
    b = bf16[] parameter(1)
    ROOT s = bf16[] add(a, b)
  }

  ENTRY entry {
    param = bf16[16384,6144] parameter(0)
    reshape = bf16[1,16384,6144] reshape(param)
    all-reduce = bf16[1,16384,6144] all-reduce(reshape), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=add
    constant = s32[] constant(0)
    dynamic-slice = bf16[1,16384,384] dynamic-slice(all-reduce, constant, constant, constant), dynamic_slice_sizes={1,16384,384}
    copy = bf16[1,16384,6144] copy(reshape)
    ROOT tuple = (bf16[1,16384,6144], bf16[1,16384,384]) tuple(copy, dynamic-slice)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunCollectiveTransformationReorderer(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectiveTransformationReordererTest, AllReduceWithTwoUsersReshape) {
  absl::string_view hlo_string = R"(
  HloModule module

  add {
    a = bf16[] parameter(0)
    b = bf16[] parameter(1)
    ROOT s = bf16[] add(a, b)
  }

  ENTRY entry {
    param = bf16[16384,6144] parameter(0)
    reshape = bf16[1,16384,6144] reshape(param)
    all-reduce = bf16[1,16384,6144] all-reduce(reshape), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=add
    constant = s32[] constant(0)
    dynamic-slice = bf16[1,16384,384] dynamic-slice(all-reduce, constant, constant, constant), dynamic_slice_sizes={1,16384,384}
    copy = bf16[1,16384,6144] copy(all-reduce)
    ROOT tuple = (bf16[1,16384,6144], bf16[1,16384,384]) tuple(copy, dynamic-slice)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunCollectiveTransformationReorderer(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectiveTransformationReordererTest, AllReduceConstrainLayout) {
  absl::string_view hlo_string = R"(
  HloModule module

  add {
    a = bf16[] parameter(0)
    b = bf16[] parameter(1)
    ROOT s = bf16[] add(a, b)
  }

  ENTRY entry {
    param = bf16[16384,6144] parameter(0)
    reshape = bf16[1,16384,6144] reshape(param)
    all-reduce = bf16[1,16384,6144] all-reduce(reshape), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}}, constrain_layout=true, to_apply=add
    constant = s32[] constant(0)
    ROOT dynamic-slice = bf16[1,16384,384] dynamic-slice(all-reduce, constant, constant, constant), dynamic_slice_sizes={1,16384,384}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunCollectiveTransformationReorderer(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectiveTransformationReordererTest,
       AllReduceWithFrontendAttributesSingleReshape) {
  static constexpr absl::string_view kHloString = R"(
  HloModule module

  add {
    a = bf16[] parameter(0)
    b = bf16[] parameter(1)
    ROOT s = bf16[] add(a, b)
  }

  ENTRY entry {
    param = bf16[16384,6144] parameter(0)
    reshape = bf16[1,16384,6144] reshape(param)
    all-reduce = bf16[1,16384,6144] all-reduce(reshape), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=add
    constant = s32[] constant(0)
    ROOT dynamic-slice = bf16[1,16384,384] dynamic-slice(all-reduce, constant, constant, constant), dynamic_slice_sizes={1,16384,384}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  // Set a front-end attribute on the all-reduce.
  FrontendAttributes frontend_attributes;
  (*frontend_attributes.mutable_map())["_xla_tpu_quantize_allreduce"] = "true";
  HloInstruction* all_reduce =
      FindInstruction(module.get(), HloOpcode::kAllReduce);
  all_reduce->set_frontend_attributes(frontend_attributes);

  // Run the transformation.
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunCollectiveTransformationReorderer(module.get()));

  // Check the results.
  EXPECT_TRUE(changed);
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/false,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::DynamicSlice(op::Reshape(op::AllReduce(op::Parameter())),
                               op::Constant(), op::Constant(), op::Constant()));

  // Check that the front-end attribute is still on the all-reduce.
  const HloInstruction* all_reduce_after_transformation =
      FindInstruction(module.get(), HloOpcode::kAllReduce);
  const FrontendAttributes& front_end_attributes_after_transformation =
      all_reduce_after_transformation->frontend_attributes();
  EXPECT_EQ(front_end_attributes_after_transformation.SerializeAsString(),
            frontend_attributes.SerializeAsString());
}

TEST_F(CollectiveTransformationReordererTest,
       AllGatherWithFrontEndAttributesWithReshape) {
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

  // Set a front-end attribute on the all-gather.
  FrontendAttributes frontend_attributes;
  (*frontend_attributes.mutable_map())["_xla_collective_matmul"] = "lhs_ag";
  HloInstruction* all_gather =
      FindInstruction(module.get(), HloOpcode::kAllGather);
  all_gather->set_frontend_attributes(frontend_attributes);

  // Run the transformation.
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunCollectiveTransformationReorderer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::AllGather(op::Reshape(op::Parameter())));

  const HloInstruction* all_gather_after_transformation =
      module->entry_computation()->root_instruction();
  EXPECT_EQ(all_gather_after_transformation->opcode(), HloOpcode::kAllGather);
  EXPECT_THAT(all_gather_after_transformation->dimensions(), ElementsAre(1));

  // Check that the front-end attribute is still on the all-gather.
  const FrontendAttributes& front_end_attributes_after_transformation =
      all_gather_after_transformation->frontend_attributes();
  EXPECT_EQ(front_end_attributes_after_transformation.SerializeAsString(),
            frontend_attributes.SerializeAsString());
}

}  // namespace

}  // namespace xla
