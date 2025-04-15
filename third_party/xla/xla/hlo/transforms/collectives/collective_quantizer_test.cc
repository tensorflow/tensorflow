/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/transforms/collectives/collective_quantizer.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

class CollectiveQuantizerTest : public HloHardwareIndependentTestBase {
 public:
  absl::StatusOr<bool> RunCollectiveQuantizer(HloModule* module) {
    CollectiveQuantizer collective_quantizer;
    return collective_quantizer.Run(module, {});
  }
};

TEST_F(CollectiveQuantizerTest, AllGatherConvert) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  ENTRY entry {
    param = bf16[8,4,8,128] parameter(0)
    all-gather = bf16[8,32,8,128] all-gather(param), dimensions={1}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1, use_global_device_ids=true
    ROOT convert = f8e4m3fn[8,32,8,128] convert(all-gather)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::AllGather(op::Convert(op::Parameter())));
  HloInstruction* all_gather = module->entry_computation()->root_instruction();
  EXPECT_THAT(all_gather->shape().element_type(), F8E4M3FN);
}

TEST_F(CollectiveQuantizerTest, AllGatherConvertUnary) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  ENTRY entry {
    param = bf16[8,4,8,128] parameter(0)
    all-gather = bf16[8,32,8,128] all-gather(param), dimensions={1}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1, use_global_device_ids=true
    reshape = bf16[8,32,1024] reshape(all-gather)
    slice = bf16[8,32,512] slice(reshape), slice={[0:8], [0:32], [256:768]}
    ROOT convert = f8e4m3fn[8,32,512] convert(slice)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Slice(op::Reshape(op::AllGather(op::Convert(op::Parameter())))));
  HloInstruction* all_gather = module->entry_computation()->root_instruction();
  EXPECT_THAT(all_gather->shape().element_type(), F8E4M3FN);
}

TEST_F(CollectiveQuantizerTest, AllGatherQuantize) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  ENTRY entry {
    param = bf16[8,4,8,128] parameter(0)
    all-gather = bf16[8,32,8,128] all-gather(param), dimensions={1}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1, use_global_device_ids=true
    scale = bf16[] parameter(1), sharding={replicated}
    scale_bcast = bf16[8,32,8,128] broadcast(scale), dimensions={}
    divide = bf16[8,32,8,128] divide(all-gather, scale_bcast)
    clamp_lower = bf16[] constant(-448.0)
    clamp_lower_bcast = bf16[8,32,8,128] broadcast(clamp_lower), dimensions={}
    clamp_upper = bf16[] constant(448.0)
    clamp_upper_bcast = bf16[8,32,8,128] broadcast(clamp_upper), dimensions={}
    clamp = bf16[8,32,8,128] clamp(clamp_lower_bcast, divide, clamp_upper_bcast)
    ROOT convert = f8e4m3fn[8,32,8,128] convert(clamp)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::AllGather(op::Convert(op::Clamp(
                  op::Broadcast(), op::Divide(op::Parameter(), op::Broadcast()),
                  op::Broadcast()))));
  HloInstruction* all_gather = module->entry_computation()->root_instruction();
  EXPECT_THAT(all_gather->shape().element_type(), F8E4M3FN);
}

TEST_F(CollectiveQuantizerTest, AllToAllQuantize) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  ENTRY entry {
    param = bf16[8,32,8,128] parameter(0)
    all-to-all = bf16[8,32,8,128] all-to-all(param), dimensions={1}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1
    scale = bf16[] parameter(1), sharding={replicated}
    scale_bcast = bf16[8,32,8,128] broadcast(scale), dimensions={}
    divide = bf16[8,32,8,128] divide(all-to-all, scale_bcast)
    clamp_lower = bf16[] constant(-448.0)
    clamp_lower_bcast = bf16[8,32,8,128] broadcast(clamp_lower), dimensions={}
    clamp_upper = bf16[] constant(448.0)
    clamp_upper_bcast = bf16[8,32,8,128] broadcast(clamp_upper), dimensions={}
    clamp = bf16[8,32,8,128] clamp(clamp_lower_bcast, divide, clamp_upper_bcast)
    ROOT convert = f8e4m3fn[8,32,8,128] convert(clamp)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::AllToAll(op::Convert(op::Clamp(
                  op::Broadcast(), op::Divide(op::Parameter(), op::Broadcast()),
                  op::Broadcast()))));
  HloInstruction* all_to_all = module->entry_computation()->root_instruction();
  EXPECT_THAT(all_to_all->shape().element_type(), F8E4M3FN);
}

TEST_F(CollectiveQuantizerTest, CollectiveBroadcastQuantize) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  ENTRY entry {
    param = bf16[8,32,8,128] parameter(0)
    collective-broadcast = bf16[8,32,8,128] collective-broadcast(param), replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1
    scale = bf16[] parameter(1), sharding={replicated}
    scale_bcast = bf16[8,32,8,128] broadcast(scale), dimensions={}
    divide = bf16[8,32,8,128] divide(collective-broadcast, scale_bcast)
    clamp_lower = bf16[] constant(-448.0)
    clamp_lower_bcast = bf16[8,32,8,128] broadcast(clamp_lower), dimensions={}
    clamp_upper = bf16[] constant(448.0)
    clamp_upper_bcast = bf16[8,32,8,128] broadcast(clamp_upper), dimensions={}
    clamp = bf16[8,32,8,128] clamp(clamp_lower_bcast, divide, clamp_upper_bcast)
    ROOT convert = f8e4m3fn[8,32,8,128] convert(clamp)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CollectiveBroadcast(op::Convert(op::Clamp(
                  op::Broadcast(), op::Divide(op::Parameter(), op::Broadcast()),
                  op::Broadcast()))));
  HloInstruction* collective_broadcast =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(collective_broadcast->shape().element_type(), F8E4M3FN);
}

TEST_F(CollectiveQuantizerTest, CollectivePermuteQuantize) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  ENTRY entry {
    param = bf16[8,32,8,128] parameter(0)
    collective-permute = bf16[8,32,8,128] collective-permute(param), source_target_pairs={{0,1},{2,3},{4,5},{6,7}}, channel_id=1
    scale = bf16[] parameter(1), sharding={replicated}
    scale_bcast = bf16[8,32,8,128] broadcast(scale), dimensions={}
    divide = bf16[8,32,8,128] divide(collective-permute, scale_bcast)
    clamp_lower = bf16[] constant(-448.0)
    clamp_lower_bcast = bf16[8,32,8,128] broadcast(clamp_lower), dimensions={}
    clamp_upper = bf16[] constant(448.0)
    clamp_upper_bcast = bf16[8,32,8,128] broadcast(clamp_upper), dimensions={}
    clamp = bf16[8,32,8,128] clamp(clamp_lower_bcast, divide, clamp_upper_bcast)
    ROOT convert = f8e4m3fn[8,32,8,128] convert(clamp)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CollectivePermute(op::Convert(op::Clamp(
                  op::Broadcast(), op::Divide(op::Parameter(), op::Broadcast()),
                  op::Broadcast()))));
  HloInstruction* collective_permute =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(collective_permute->shape().element_type(), F8E4M3FN);
}

TEST_F(CollectiveQuantizerTest, AllGatherQuantizeUnary) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  ENTRY entry {
    param = bf16[8,4,8,128] parameter(0)
    all-gather = bf16[8,32,8,128] all-gather(param), dimensions={1}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1, use_global_device_ids=true
    reshape = bf16[8,32,1024] reshape(all-gather)
    slice = bf16[8,32,512] slice(reshape), slice={[0:8], [0:32], [256:768]}
    scale = bf16[] parameter(1), sharding={replicated}
    scale_bcast = bf16[8,32,512] broadcast(scale), dimensions={}
    divide = bf16[8,32,512] divide(slice, scale_bcast)
    clamp_lower = bf16[] constant(-448.0)
    clamp_lower_bcast = bf16[8,32,512] broadcast(clamp_lower), dimensions={}
    clamp_upper = bf16[] constant(448.0)
    clamp_upper_bcast = bf16[8,32,512] broadcast(clamp_upper), dimensions={}
    clamp = bf16[8,32,512] clamp(clamp_lower_bcast, divide, clamp_upper_bcast)
    ROOT convert = f8e4m3fn[8,32,512] convert(clamp)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Slice(op::Reshape(op::AllGather(op::Convert(op::Clamp(
                  op::Broadcast(), op::Divide(op::Parameter(), op::Broadcast()),
                  op::Broadcast()))))));
  HloInstruction* slice = module->entry_computation()->root_instruction();
  EXPECT_THAT(slice->shape().element_type(), F8E4M3FN);
}

TEST_F(CollectiveQuantizerTest, AllGatherQuantizeMultiUser) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  ENTRY entry {
    param = bf16[8,4,8,128] parameter(0)
    all-gather = bf16[8,32,8,128] all-gather(param), dimensions={1}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1, use_global_device_ids=true
    scale = bf16[] parameter(1), sharding={replicated}
    scale_bcast = bf16[8,32,8,128] broadcast(scale), dimensions={}
    divide = bf16[8,32,8,128] divide(all-gather, scale_bcast)
    clamp_lower = bf16[] constant(-448.0)
    clamp_lower_bcast = bf16[8,32,8,128] broadcast(clamp_lower), dimensions={}
    clamp_upper = bf16[] constant(448.0)
    clamp_upper_bcast = bf16[8,32,8,128] broadcast(clamp_upper), dimensions={}
    clamp = bf16[8,32,8,128] clamp(clamp_lower_bcast, divide, clamp_upper_bcast)
    add = bf16[8,32,8,128] add(divide, clamp)
    ROOT convert = f8e4m3fn[8,32,8,128] convert(add)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectiveQuantizerTest, AllGatherQuantizeNonReplicatedScale) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  ENTRY entry {
    param = bf16[8,4,8,128] parameter(0)
    all-gather = bf16[8,32,8,128] all-gather(param), dimensions={1}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1, use_global_device_ids=true
    scale = bf16[] parameter(1)
    scale_bcast = bf16[8,32,8,128] broadcast(scale), dimensions={}
    divide = bf16[8,32,8,128] divide(all-gather, scale_bcast)
    clamp_lower = bf16[] constant(-448.0)
    clamp_lower_bcast = bf16[8,32,8,128] broadcast(clamp_lower), dimensions={}
    clamp_upper = bf16[] constant(448.0)
    clamp_upper_bcast = bf16[8,32,8,128] broadcast(clamp_upper), dimensions={}
    clamp = bf16[8,32,8,128] clamp(clamp_lower_bcast, divide, clamp_upper_bcast)
    ROOT convert = f8e4m3fn[8,32,8,128] convert(clamp)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectiveQuantizerTest, AllGatherQuantizePartialReplication) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  max {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT max = f32[] maximum(a, b)
  }

  ENTRY entry {
    param = bf16[8,4,8,128] parameter(0)
    all-gather = bf16[8,16,8,128] all-gather(param), dimensions={1}, replica_groups={{0,1,2,3},{4,5,6,7}}, channel_id=1, use_global_device_ids=true
    scale = bf16[1] parameter(1), sharding={devices=[8]<=[8]}
    scalar_scale = bf16[] reshape(scale)
    all_reduced_scale = bf16[] all-reduce(scalar_scale), to_apply=max, replica_groups={{0,1,2,3},{4,5,6,7}}, channel_id=2, use_global_device_ids=true
    scale_bcast = bf16[8,16,8,128] broadcast(all_reduced_scale), dimensions={}
    divide = bf16[8,16,8,128] divide(all-gather, scale_bcast)
    clamp_lower = bf16[] constant(-448.0)
    clamp_lower_bcast = bf16[8,16,8,128] broadcast(clamp_lower), dimensions={}
    clamp_upper = bf16[] constant(448.0)
    clamp_upper_bcast = bf16[8,16,8,128] broadcast(clamp_upper), dimensions={}
    clamp = bf16[8,16,8,128] clamp(clamp_lower_bcast, divide, clamp_upper_bcast)
    ROOT convert = f8e4m3fn[8,16,8,128] convert(clamp)
    }
  )";

  HloModuleConfig config;
  config.set_num_partitions(8);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::AllGather(op::Convert(op::Clamp(
                  op::Broadcast(), op::Divide(op::Parameter(), op::Broadcast()),
                  op::Broadcast()))));
  HloInstruction* all_gather = module->entry_computation()->root_instruction();
  EXPECT_THAT(all_gather->shape().element_type(), F8E4M3FN);
}

TEST_F(CollectiveQuantizerTest, AllToAllQuantizePartialReplication) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  max {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT max = f32[] maximum(a, b)
  }

  ENTRY entry {
    param = bf16[8,32,8,128] parameter(0)
    all-to-all = bf16[8,32,8,128] all-to-all(param), dimensions={1}, replica_groups={{0,1,2,3},{4,5,6,7}}, channel_id=1
    scale = bf16[1] parameter(1), sharding={devices=[8]<=[8]}
    scalar_scale = bf16[] reshape(scale)
    all_reduced_scale = bf16[] all-reduce(scalar_scale), to_apply=max, replica_groups={{0,1,2,3},{4,5,6,7}}, channel_id=2, use_global_device_ids=true
    scale_bcast = bf16[8,32,8,128] broadcast(all_reduced_scale), dimensions={}
    divide = bf16[8,32,8,128] divide(all-to-all, scale_bcast)
    clamp_lower = bf16[] constant(-448.0)
    clamp_lower_bcast = bf16[8,32,8,128] broadcast(clamp_lower), dimensions={}
    clamp_upper = bf16[] constant(448.0)
    clamp_upper_bcast = bf16[8,32,8,128] broadcast(clamp_upper), dimensions={}
    clamp = bf16[8,32,8,128] clamp(clamp_lower_bcast, divide, clamp_upper_bcast)
    ROOT convert = f8e4m3fn[8,32,8,128] convert(clamp)
    }
  )";

  HloModuleConfig config;
  config.set_num_partitions(8);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::AllToAll(op::Convert(op::Clamp(
                  op::Broadcast(), op::Divide(op::Parameter(), op::Broadcast()),
                  op::Broadcast()))));
  HloInstruction* all_to_all = module->entry_computation()->root_instruction();
  EXPECT_THAT(all_to_all->shape().element_type(), F8E4M3FN);
}

TEST_F(CollectiveQuantizerTest,
       AllToAllQuantizePartialReplicationSeparateComputation) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  max {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT max = f32[] maximum(a, b)
  }

  all_reduce {
    scale = bf16[] parameter(0)
    ROOT all_reduced_scale = bf16[] all-reduce(scale), to_apply=max, replica_groups={{0,1,2,3},{4,5,6,7}}, channel_id=2, use_global_device_ids=true
  }

  ENTRY entry {
    param = bf16[8,32,8,128] parameter(0)
    all-to-all = bf16[8,32,8,128] all-to-all(param), dimensions={1}, replica_groups={{0,1,2,3},{4,5,6,7}}, channel_id=1
    scale = bf16[1] parameter(1), sharding={devices=[8]<=[8]}
    scalar_scale = bf16[] reshape(scale)
    all_reduced_scale = bf16[] call(scalar_scale), to_apply=all_reduce
    scale_bcast = bf16[8,32,8,128] broadcast(all_reduced_scale), dimensions={}
    divide = bf16[8,32,8,128] divide(all-to-all, scale_bcast)
    clamp_lower = bf16[] constant(-448.0)
    clamp_lower_bcast = bf16[8,32,8,128] broadcast(clamp_lower), dimensions={}
    clamp_upper = bf16[] constant(448.0)
    clamp_upper_bcast = bf16[8,32,8,128] broadcast(clamp_upper), dimensions={}
    clamp = bf16[8,32,8,128] clamp(clamp_lower_bcast, divide, clamp_upper_bcast)
    ROOT convert = f8e4m3fn[8,32,8,128] convert(clamp)
    }
  )";

  HloModuleConfig config;
  config.set_num_partitions(8);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::AllToAll(op::Convert(op::Clamp(
                  op::Broadcast(), op::Divide(op::Parameter(), op::Broadcast()),
                  op::Broadcast()))));
  HloInstruction* all_to_all = module->entry_computation()->root_instruction();
  EXPECT_THAT(all_to_all->shape().element_type(), F8E4M3FN);
}

// Expecting no change as the all-reduced scale is not replicated identically to
// the all-gather.
TEST_F(CollectiveQuantizerTest,
       AllGatherQuantizePartialReplicationGroupMismatch) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  max {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT max = f32[] maximum(a, b)
  }

  ENTRY entry {
    param = bf16[8,4,8,128] parameter(0)
    all-gather = bf16[8,32,8,128] all-gather(param), dimensions={1}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1, use_global_device_ids=true
    scale = bf16[1] parameter(1), sharding={devices=[8]<=[8]}
    scalar_scale = bf16[] reshape(scale)
    all_reduced_scale = bf16[] all-reduce(scalar_scale), to_apply=max, replica_groups={{0,1,2,3},{4,5,6,7}}, channel_id=2, use_global_device_ids=true
    scale_bcast = bf16[8,32,8,128] broadcast(all_reduced_scale), dimensions={}
    divide = bf16[8,32,8,128] divide(all-gather, scale_bcast)
    clamp_lower = bf16[] constant(-448.0)
    clamp_lower_bcast = bf16[8,32,8,128] broadcast(clamp_lower), dimensions={}
    clamp_upper = bf16[] constant(448.0)
    clamp_upper_bcast = bf16[8,32,8,128] broadcast(clamp_upper), dimensions={}
    clamp = bf16[8,32,8,128] clamp(clamp_lower_bcast, divide, clamp_upper_bcast)
    ROOT convert = f8e4m3fn[8,32,8,128] convert(clamp)
    }
  )";

  HloModuleConfig config;
  config.set_num_partitions(8);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_FALSE(changed);
}

// Expecting no change as the all-reduced scale is not replicated identically to
// the all-gather.
TEST_F(CollectiveQuantizerTest,
       AllToAllQuantizePartialReplicationGroupMismatchSeparateComputation) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  max {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT max = f32[] maximum(a, b)
  }

  all_reduce {
    scale = bf16[] parameter(0)
    ROOT all_reduced_scale = bf16[] all-reduce(scale), to_apply=max, replica_groups={{0,1,2,3},{4,5,6,7}}, channel_id=2, use_global_device_ids=true
  }

  ENTRY entry {
    param = bf16[8,32,8,128] parameter(0)
    all-to-all = bf16[8,32,8,128] all-to-all(param), dimensions={1}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1
    scale = bf16[1] parameter(1), sharding={devices=[8]<=[8]}
    scalar_scale = bf16[] reshape(scale)
    all_reduced_scale = bf16[] call(scalar_scale), to_apply=all_reduce
    scale_bcast = bf16[8,32,8,128] broadcast(all_reduced_scale), dimensions={}
    divide = bf16[8,32,8,128] divide(all-to-all, scale_bcast)
    clamp_lower = bf16[] constant(-448.0)
    clamp_lower_bcast = bf16[8,32,8,128] broadcast(clamp_lower), dimensions={}
    clamp_upper = bf16[] constant(448.0)
    clamp_upper_bcast = bf16[8,32,8,128] broadcast(clamp_upper), dimensions={}
    clamp = bf16[8,32,8,128] clamp(clamp_lower_bcast, divide, clamp_upper_bcast)
    ROOT convert = f8e4m3fn[8,32,8,128] convert(clamp)
    }
  )";

  HloModuleConfig config;
  config.set_num_partitions(8);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectiveQuantizerTest, ConvertAllGather) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  ENTRY entry {
    param = f8e4m3fn[8,4,8,128] parameter(0)
    convert = bf16[8,4,8,128] convert(param)
    ROOT all-gather = bf16[8,32,8,128] all-gather(convert), dimensions={1}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1, use_global_device_ids=true
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Convert(op::AllGather(op::Parameter())));
  const HloInstruction* all_gather =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_THAT(all_gather->shape().element_type(), F8E4M3FN);
}

TEST_F(CollectiveQuantizerTest, ConvertAllGatherUnary) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  ENTRY entry {
    param = f8e4m3fn[8,4,8,128] parameter(0)
    convert = bf16[8,4,8,128] convert(param)
    reshape = bf16[8,4,1024] reshape(convert)
    slice = bf16[8,4,512] slice(reshape), slice={[0:8], [0:4], [256:768]}
    ROOT all-gather = bf16[8,32,512] all-gather(slice), dimensions={1}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1, use_global_device_ids=true
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Convert(op::AllGather(op::Slice(op::Reshape(op::Parameter())))));
  const HloInstruction* all_gather =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_THAT(all_gather->shape().element_type(), F8E4M3FN);
}

TEST_F(CollectiveQuantizerTest, DequantizeAllGather) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  ENTRY entry {
    param = f8e4m3fn[8,4,8,128] parameter(0)
    convert = bf16[8,4,8,128] convert(param)
    scale = bf16[] parameter(1), sharding={replicated}
    scale_bcast = bf16[8,4,8,128] broadcast(scale), dimensions={}
    multiply = bf16[8,4,8,128] multiply(convert, scale_bcast)
    ROOT all-gather = bf16[8,32,8,128] all-gather(multiply), dimensions={1}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1, use_global_device_ids=true
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Multiply(op::Convert(op::AllGather(op::Parameter())),
                           op::Broadcast()));
  const HloInstruction* all_gather =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  EXPECT_THAT(all_gather->shape().element_type(), F8E4M3FN);
}

TEST_F(CollectiveQuantizerTest, DequantizeAllToAll) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  ENTRY entry {
    param = f8e4m3fn[8,32,8,128] parameter(0)
    convert = bf16[8,32,8,128] convert(param)
    scale = bf16[] parameter(1), sharding={replicated}
    scale_bcast = bf16[8,32,8,128] broadcast(scale), dimensions={}
    multiply = bf16[8,32,8,128] multiply(convert, scale_bcast)
    ROOT all-to-all = bf16[8,32,8,128] all-to-all(multiply), dimensions={1}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Multiply(op::Convert(op::AllToAll(op::Parameter())),
                           op::Broadcast()));
  const HloInstruction* all_to_all =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  EXPECT_THAT(all_to_all->shape().element_type(), F8E4M3FN);
}

TEST_F(CollectiveQuantizerTest, DequantizeCollectiveBroadcast) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  ENTRY entry {
    param = f8e4m3fn[8,32,8,128] parameter(0)
    convert = bf16[8,32,8,128] convert(param)
    scale = bf16[] parameter(1), sharding={replicated}
    scale_bcast = bf16[8,32,8,128] broadcast(scale), dimensions={}
    multiply = bf16[8,32,8,128] multiply(convert, scale_bcast)
    ROOT collective-broadcast = bf16[8,32,8,128] collective-broadcast(multiply), replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Multiply(op::Convert(op::CollectiveBroadcast(op::Parameter())),
                   op::Broadcast()));
  const HloInstruction* collective_broadcast =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  EXPECT_THAT(collective_broadcast->shape().element_type(), F8E4M3FN);
}

TEST_F(CollectiveQuantizerTest, DequantizeCollectivePermute) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  ENTRY entry {
    param = f8e4m3fn[8,32,8,128] parameter(0)
    convert = bf16[8,32,8,128] convert(param)
    scale = bf16[] parameter(1), sharding={replicated}
    scale_bcast = bf16[8,32,8,128] broadcast(scale), dimensions={}
    multiply = bf16[8,32,8,128] multiply(convert, scale_bcast)
    ROOT collective-permute = bf16[8,32,8,128] collective-permute(multiply), source_target_pairs={{0,1},{2,3},{4,5},{6,7}}, channel_id=1
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Multiply(op::Convert(op::CollectivePermute(op::Parameter())),
                           op::Broadcast()));
  const HloInstruction* collective_permute =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  EXPECT_THAT(collective_permute->shape().element_type(), F8E4M3FN);
}

TEST_F(CollectiveQuantizerTest, DequantizeAllGatherUnary) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  ENTRY entry {
    param = f8e4m3fn[8,4,8,128] parameter(0)
    convert = bf16[8,4,8,128] convert(param)
    scale = bf16[] parameter(1), sharding={replicated}
    scale_bcast = bf16[8,4,8,128] broadcast(scale), dimensions={}
    multiply = bf16[8,4,8,128] multiply(convert, scale_bcast)
    reshape = bf16[8,4,1024] reshape(multiply)
    slice = bf16[8,4,512] slice(reshape), slice={[0:8], [0:4], [256:768]}
    ROOT all-gather = bf16[8,32,512] all-gather(slice), dimensions={1}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1, use_global_device_ids=true
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Multiply(
          op::Convert(op::AllGather(op::Slice(op::Reshape(op::Parameter())))),
          op::Broadcast()));
  HloInstruction* all_gather = module->entry_computation()
                                   ->root_instruction()
                                   ->mutable_operand(0)
                                   ->mutable_operand(0);
  EXPECT_THAT(all_gather->shape().element_type(), F8E4M3FN);
}

TEST_F(CollectiveQuantizerTest, DequantizeAllGatherPartialReplication) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  max {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT max = f32[] maximum(a, b)
  }

  ENTRY entry {
    param = f8e4m3fn[8,4,8,128] parameter(0)
    convert = bf16[8,4,8,128] convert(param)
    scale = bf16[1] parameter(1), sharding={devices=[8]<=[8]}
    scalar_scale = bf16[] reshape(scale)
    all_reduced_scale = bf16[] all-reduce(scalar_scale), to_apply=max, replica_groups={{0,1,2,3},{4,5,6,7}}, channel_id=1, use_global_device_ids=true
    scale_bcast = bf16[8,4,8,128] broadcast(all_reduced_scale), dimensions={}
    multiply = bf16[8,4,8,128] multiply(convert, scale_bcast)
    ROOT all-gather = bf16[8,16,8,128] all-gather(multiply), dimensions={1}, replica_groups={{0,1,2,3},{4,5,6,7}}, channel_id=2, use_global_device_ids=true
    }
  )";

  HloModuleConfig config;
  config.set_num_partitions(8);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Multiply(op::Convert(op::AllGather(op::Parameter())),
                           op::Broadcast()));
  const HloInstruction* all_gather =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  EXPECT_THAT(all_gather->shape().element_type(), F8E4M3FN);
}

TEST_F(CollectiveQuantizerTest, DequantizeAllToAllPartialReplication) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  max {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT max = f32[] maximum(a, b)
  }

  ENTRY entry {
    param = f8e4m3fn[8,32,8,128] parameter(0)
    convert = bf16[8,32,8,128] convert(param)
    scale = bf16[1] parameter(1), sharding={devices=[8]<=[8]}
    scalar_scale = bf16[] reshape(scale)
    all_reduced_scale = bf16[] all-reduce(scalar_scale), to_apply=max, replica_groups={{0,1,2,3},{4,5,6,7}}, channel_id=1, use_global_device_ids=true
    scale_bcast = bf16[8,32,8,128] broadcast(all_reduced_scale), dimensions={}
    multiply = bf16[8,32,8,128] multiply(convert, scale_bcast)
    ROOT all-to-all = bf16[8,32,8,128] all-to-all(multiply), dimensions={1}, replica_groups={{0,1,2,3},{4,5,6,7}}, channel_id=2
    }
  )";

  HloModuleConfig config;
  config.set_num_partitions(8);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Multiply(op::Convert(op::AllToAll(op::Parameter())),
                           op::Broadcast()));
  const HloInstruction* all_gather =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  EXPECT_THAT(all_gather->shape().element_type(), F8E4M3FN);
}

TEST_F(CollectiveQuantizerTest,
       DequantizeAllToAllPartialReplicationSeparateComputation) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  max {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT max = f32[] maximum(a, b)
  }

  all_reduce {
    scale = bf16[] parameter(0)
    ROOT all_reduced_scale = bf16[] all-reduce(scale), to_apply=max, replica_groups={{0,1,2,3},{4,5,6,7}}, channel_id=1, use_global_device_ids=true
  }

  ENTRY entry {
    param = f8e4m3fn[8,32,8,128] parameter(0)
    convert = bf16[8,32,8,128] convert(param)
    scale = bf16[1] parameter(1), sharding={devices=[8]<=[8]}
    scalar_scale = bf16[] reshape(scale)
    all_reduced_scale = bf16[] call(scalar_scale), to_apply=all_reduce
    scale_bcast = bf16[8,32,8,128] broadcast(all_reduced_scale), dimensions={}
    multiply = bf16[8,32,8,128] multiply(convert, scale_bcast)
    ROOT all-to-all = bf16[8,32,8,128] all-to-all(multiply), dimensions={1}, replica_groups={{0,1,2,3},{4,5,6,7}}, channel_id=2
    }
  )";

  HloModuleConfig config;
  config.set_num_partitions(8);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Multiply(op::Convert(op::AllToAll(op::Parameter())),
                           op::Broadcast()));
  const HloInstruction* all_gather =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  EXPECT_THAT(all_gather->shape().element_type(), F8E4M3FN);
}

// Expecting no change as the all-reduced scale is not replicated identically to
// the all-gather.
TEST_F(CollectiveQuantizerTest,
       DequantizeAllGatherPartialReplicationGroupMismatch) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  max {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT max = f32[] maximum(a, b)
  }

  ENTRY entry {
    param = f8e4m3fn[8,4,8,128] parameter(0)
    convert = bf16[8,4,8,128] convert(param)
    scale = bf16[1] parameter(1), sharding={devices=[8]<=[8]}
    scalar_scale = bf16[] reshape(scale)
    all_reduced_scale = bf16[] all-reduce(scalar_scale), to_apply=max, replica_groups={{0,1,2,3},{4,5,6,7}}, channel_id=1, use_global_device_ids=true
    scale_bcast = bf16[8,4,8,128] broadcast(all_reduced_scale), dimensions={}
    multiply = bf16[8,4,8,128] multiply(convert, scale_bcast)
    ROOT all-gather = bf16[8,32,8,128] all-gather(multiply), dimensions={1}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=2, use_global_device_ids=true
    }
  )";

  HloModuleConfig config;
  config.set_num_partitions(8);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_FALSE(changed);
}

// Expecting no change as the all-reduced scale is not replicated identically to
// the all-gather.
TEST_F(CollectiveQuantizerTest,
       DequantizeAllToAllPartialReplicationGroupMismatchSeparateComputation) {
  absl::string_view hlo_string = R"(
  HloModule module, num_partitions=8
  max {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT max = f32[] maximum(a, b)
  }

  all_reduce {
    scale = bf16[] parameter(0)
    ROOT all_reduced_scale = bf16[] all-reduce(scale), to_apply=max, replica_groups={{0,1,2,3},{4,5,6,7}}, channel_id=1, use_global_device_ids=true
  }

  ENTRY entry {
    param = f8e4m3fn[8,32,8,128] parameter(0)
    convert = bf16[8,32,8,128] convert(param)
    scale = bf16[1] parameter(1), sharding={devices=[8]<=[8]}
    scalar_scale = bf16[] reshape(scale)
    all_reduced_scale = bf16[] call(scalar_scale), to_apply=all_reduce
    scale_bcast = bf16[8,32,8,128] broadcast(all_reduced_scale), dimensions={}
    multiply = bf16[8,32,8,128] multiply(convert, scale_bcast)
    ROOT all-to-all = bf16[8,32,8,128] all-to-all(multiply), dimensions={1}, replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=2
    }
  )";

  HloModuleConfig config;
  config.set_num_partitions(8);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunCollectiveQuantizer(module.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace xla
