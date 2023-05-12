/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner.h"

#include <memory>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_matchers.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_sharding_util.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace spmd {
namespace {

using ::testing::_;
using ::testing::AllOf;
namespace op = xla::testing::opcode_matchers;

class SpmdPartitioningTest : public HloTestBase {
 public:
  StatusOr<std::unique_ptr<HloModule>> PartitionComputation(
      absl::string_view hlo_module, int64_t num_devices,
      bool conv_halo_exchange_always_on_lhs = true,
      bool choose_faster_windowed_einsum = false,
      bool unroll_windowed_einsum = false,
      bool bidirectional_windowed_einsum = false,
      int64_t threshold_for_windowed_einsum_mib = -1) {
    // Some tests (BackpropFilter convs) set this flag false to test two
    // different paths of the implementation.
    SpmdPartitionerOptions options;
    options.conv_halo_exchange_always_on_lhs = conv_halo_exchange_always_on_lhs;
    options.allow_module_signature_change = true;
    options.choose_faster_windowed_einsum_over_mem =
        choose_faster_windowed_einsum;
    options.unroll_windowed_einsum = unroll_windowed_einsum;
    options.bidirectional_windowed_einsum = bidirectional_windowed_einsum;
    if (threshold_for_windowed_einsum_mib >= 0) {
      options.threshold_for_windowed_einsum_mib =
          threshold_for_windowed_einsum_mib;
    }
    auto collective_ops_creator =
        GetDefaultCollectiveOpsCreator(num_devices, /*num_replicas=*/1);
    // Do not use all-gather for pattern-matching purpose, as the partitioner
    // might create reshape/transposes around it.
    collective_ops_creator.create_cross_partition_all_gather = nullptr;

    HloModuleConfig config = GetModuleConfigForTest();
    config.set_use_spmd_partitioning(true);
    config.set_num_partitions(num_devices);
    TF_ASSIGN_OR_RETURN(auto module,
                        ParseAndReturnVerifiedModule(hlo_module, config));
    HloPassPipeline pass("spmd-partitioning");
    pass.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                              /*allow_mixed_precision=*/false);
    pass.AddPass<SpmdPartitioner>(num_devices, /*num_replicas=*/1, options,
                                  collective_ops_creator);
    pass.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                              /*allow_mixed_precision=*/false);
    TF_RETURN_IF_ERROR(pass.Run(module.get()).status());

    VerifyNoShardingOnCollectives(module.get());
    return StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }

  void VerifyNoShardingOnCollectives(HloModule* module) {
    for (const HloComputation* c : module->computations()) {
      for (const HloInstruction* inst : c->instructions()) {
        if (!absl::c_linear_search(
                std::vector<HloOpcode>{
                    HloOpcode::kAllToAll, HloOpcode::kAllReduce,
                    HloOpcode::kAllGather, HloOpcode::kCollectivePermute,
                    HloOpcode::kReduceScatter},
                inst->opcode())) {
          continue;
        }
        EXPECT_FALSE(inst->has_sharding());
      }
    }
  }
};

TEST_F(SpmdPartitioningTest, SingleDeviceToReplicated) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %constant = s32[2,3]{1,0} constant({{1,1,1},{1,1,1}}),
    sharding={maximal device=0}
  ROOT %copy = s32[2,3]{1,0} copy(%constant), sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Copy(op::AllReduce(
                              op::Select(op::Broadcast(op::Compare()),
                                         op::Constant(), op::Broadcast()))),
                          op::Shape("s32[2,3]")));
}

TEST_F(SpmdPartitioningTest, SingleDeviceCustomCall) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %constant = s32[2,3]{1,0} constant({{1,1,1},{1,1,1}}),
    sharding={maximal device=0}
  %cc = s32[2,3] custom-call(%constant), custom_call_target="SomeCustomCall",
    sharding={maximal device=0}
  ROOT %copy = s32[2,3]{1,0} copy(%cc), sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* custom_call = FindInstruction(module.get(), "cc.1");
  EXPECT_NE(custom_call, nullptr);
  EXPECT_NE(custom_call->parent(), module->entry_computation());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Copy(op::AllReduce(
                              op::Select(op::Broadcast(op::Compare()),
                                         op::Conditional(), op::Broadcast()))),
                          op::Shape("s32[2,3]")));
}

TEST_F(SpmdPartitioningTest, SingleDeviceToSingleDevice) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %constant = s32[2,3]{1,0} constant({{1,1,1},{1,1,1}}),
    sharding={maximal device=0}
  ROOT %copy = s32[2,3]{1,0} copy(%constant), sharding={maximal device=1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  HloInstruction* root = module->entry_computation()->root_instruction();
  VLOG(1) << module->ToString();
  EXPECT_THAT(root, op::Copy(AllOf(op::Copy(op::AllReduce(op::Select(
                                       op::Broadcast(op::Compare()),
                                       op::Constant(), op::Broadcast()))),
                                   op::Shape("s32[2,3]"))));
}

TEST_F(SpmdPartitioningTest, SingleDeviceToTiled) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %constant = s32[2,3]{1,0} constant({{1,1,1},{1,1,1}}),
    sharding={maximal device=0}
  ROOT %copy = s32[2,3]{1,0} copy(%constant),
    sharding={devices=[2,1]1,0}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      AllOf(
          op::Copy(op::DynamicSlice(
              op::AllReduce(op::Select(
                  op::Broadcast(op::Compare(op::PartitionId(), op::Constant())),
                  op::Constant(), op::Broadcast())),
              op::Reshape(op::DynamicSlice(op::Constant(), op::PartitionId())),
              op::Constant())),
          op::Shape("s32[1,3]")));
}

TEST_F(SpmdPartitioningTest, TiledToReplicated) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %constant = s32[2,3]{1,0} constant({{1,1,1},{1,1,1}}),
    sharding={devices=[2,1]0,1}
  ROOT %copy = s32[2,3]{1,0} copy(%constant), sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      op::Copy(op::AllReduce(AllOf(
          op::DynamicUpdateSlice(
              op::Broadcast(), AllOf(op::Constant(), op::Shape("s32[1,3]")),
              op::Reshape(op::DynamicSlice(op::Constant(), op::PartitionId())),
              op::Constant()),
          op::Shape("s32[2,3]")))));
}

TEST_F(SpmdPartitioningTest, TiledToSingleDevice) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %constant = s32[2,3]{1,0} constant({{1,1,1},{1,1,1}}),
    sharding={devices=[2,1]0,1}
  ROOT %copy = s32[2,3]{1,0} copy(%constant), sharding={maximal device=0}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      op::Copy(op::Copy(op::AllReduce(AllOf(
          op::DynamicUpdateSlice(
              op::Broadcast(), AllOf(op::Constant(), op::Shape("s32[1,3]")),
              op::Reshape(op::DynamicSlice(op::Constant(), op::PartitionId())),
              op::Constant()),
          op::Shape("s32[2,3]"))))));
}

TEST_F(SpmdPartitioningTest, TiledToTiledEven) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param= s32[8,2]{1,0} parameter(0), sharding={devices=[2,1]0,1}
  ROOT %copy = s32[8,2]{1,0} copy(%param), sharding={devices=[1,2]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      AllOf(op::Copy(op::Reshape(op::Transpose(op::AllToAll(AllOf(
                op::Reshape(op::Parameter()), op::Shape("s32[4,2,1]")))))),
            op::Shape("s32[8,1]")));
}

TEST_F(SpmdPartitioningTest, TiledToTiledUneven) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param= f32[7,31,128]{2,1,0} parameter(0), sharding={devices=[1,2,1]0,1}
  ROOT %copy = f32[7,31,128]{2,1,0} copy(%param), sharding={devices=[2,1,1]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      AllOf(op::Copy(op::Slice(op::Reshape(AllOf(op::Transpose(op::AllToAll(
          op::Reshape(AllOf(op::Pad(), op::Shape("f32[8,16,128]")))))))))));
}

TEST_F(SpmdPartitioningTest, GetTupleElementSwapDevice) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param.0 = (f32[2,3]{1,0}, u32[]) parameter(0),
    sharding={{maximal device=1}, {maximal device=1}}
  %gte.0 = f32[2,3]{1,0} get-tuple-element(%param.0), index=0,
    sharding={maximal device=0}
  %gte.1 = u32[] get-tuple-element(%param.0), index=1,
    sharding={maximal device=0}
  ROOT %tuple = (f32[2,3]{1,0}, u32[]) tuple(%gte.0, %gte.1),
    sharding={{maximal device=0},{maximal device=0}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_THAT(root, op::Tuple());

  EXPECT_THAT(root->operand(0),
              op::Copy(op::AllReduce(op::Select(
                  op::Broadcast(op::Compare(op::PartitionId(), op::Constant())),
                  op::GetTupleElement(op::Parameter()), op::Broadcast()))));
  EXPECT_THAT(root->operand(1),
              op::Copy(op::AllReduce(op::Select(
                  op::Broadcast(op::Compare(op::PartitionId(), op::Constant())),
                  op::GetTupleElement(op::Parameter()), op::Broadcast()))));
}

TEST_F(SpmdPartitioningTest, GetTupleElementTiled) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  param.0 = (f32[2,3]{1,0}, u32[2,3]{1,0}) parameter(0),
    sharding={{replicated}, {replicated}}
  gte.0 = f32[2,3]{1,0} get-tuple-element(param.0), index=0,
    sharding={devices=[2,1]0,1}
  gte.1 = u32[2,3]{1,0} get-tuple-element(param.0), index=1,
    sharding={devices=[2,1]0,1}
  ROOT %tuple = (f32[2,3]{1,0}, u32[2,3]{1,0}) tuple(gte.0, gte.1),
    sharding={{devices=[2,1]0,1},{devices=[2,1]0,1}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_THAT(root, op::Tuple());

  auto offset =
      op::Reshape(op::DynamicSlice(op::Constant(), op::PartitionId()));

  EXPECT_THAT(root->operand(0),
              op::DynamicSlice(op::GetTupleElement(op::Parameter()), offset,
                               op::Constant()));
  EXPECT_THAT(root->operand(1),
              op::DynamicSlice(op::GetTupleElement(op::Parameter()), offset,
                               op::Constant()));
}

TEST_F(SpmdPartitioningTest, TiledInfeed) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  token0 = token[] after-all(), sharding={maximal device=0}
  infeed = (f32[8,2]{1,0}, token[]) infeed(token0),
    sharding={{devices=[2,1]0,1}, {maximal device=0}}
  ROOT infeed.data = f32[8,2]{1,0} get-tuple-element(infeed), index=0,
    sharding={maximal device=0}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      op::Copy(op::AllReduce(op::DynamicUpdateSlice(
          op::Broadcast(),
          op::GetTupleElement(
              AllOf(op::Infeed(), op::Shape("(f32[4,2]{1,0}, token[])"))),
          op::Reshape(op::DynamicSlice(op::Constant(), op::PartitionId())),
          op::Constant()))));
}

TEST_F(SpmdPartitioningTest, UnevenTiledInfeed) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  token0 = token[] after-all(), sharding={maximal device=0}
  infeed = (f32[9,2]{1,0}, token[]) infeed(token0),
    sharding={{devices=[2,1]0,1}, {maximal device=0}}
  ROOT infeed.data = f32[9,2]{1,0} get-tuple-element(infeed), index=0,
    sharding={devices=[2,1]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, AllOf(op::Shape("f32[5,2]"), op::GetTupleElement(op::Conditional(
                                             op::Convert(op::PartitionId()),
                                             op::AfterAll(), op::AfterAll()))));
  EXPECT_THAT(
      root->operand(0)->called_computations()[0]->root_instruction(),
      AllOf(op::Shape("(f32[5,2], token[])"), op::Infeed(op::Parameter())));
  auto second_infeed =
      AllOf(op::Shape("(f32[4,2], token[])"), op::Infeed(op::Parameter()));
  EXPECT_THAT(root->operand(0)->called_computations()[1]->root_instruction(),
              AllOf(op::Shape("(f32[5,2], token[])"),
                    op::Tuple(op::Pad(op::GetTupleElement(second_infeed),
                                      op::Constant()),
                              op::GetTupleElement(second_infeed))));
}

TEST_F(SpmdPartitioningTest, UnevenTiledTupleInfeed) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  token0 = token[] after-all(), sharding={maximal device=0}
  infeed = ((f32[9,2]{1,0}, f32[2]{0}), token[]) infeed(token0),
    sharding={{devices=[2,1]0,1}, {replicated}, {maximal device=0}}
  ROOT infeed.data = (f32[9,2]{1,0}, f32[2]{0}) get-tuple-element(infeed),
    index=0, sharding={{devices=[2,1]0,1}, {replicated}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("(f32[5,2], f32[2])"),
                          op::GetTupleElement(op::Conditional(
                              op::Convert(op::PartitionId()), op::AfterAll(),
                              op::AfterAll()))));
  EXPECT_THAT(root->operand(0)->called_computations()[0]->root_instruction(),
              AllOf(op::Shape("((f32[5,2], f32[2]), token[])"),
                    op::Infeed(op::Parameter())));
  auto second_infeed = AllOf(op::Shape("((f32[4,2], f32[2]), token[])"),
                             op::Infeed(op::Parameter()));
  EXPECT_THAT(
      root->operand(0)->called_computations()[1]->root_instruction(),
      AllOf(op::Shape("((f32[5,2], f32[2]), token[])"),
            op::Tuple(op::Tuple(op::Pad(op::GetTupleElement(
                                            op::GetTupleElement(second_infeed)),
                                        op::Constant()),
                                op::GetTupleElement(
                                    op::GetTupleElement(second_infeed))),
                      op::GetTupleElement(second_infeed))));
}

TEST_F(SpmdPartitioningTest, MixedTupleInfeed) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  token0 = token[] after-all(), sharding={maximal device=0}
  infeed = ((f32[9,2]{1,0}, f32[2]{0}), token[]) infeed(token0),
    sharding={{maximal device=0}, {maximal device=1}, {maximal device=0}}
  ROOT infeed.data = (f32[9,2]{1,0}, f32[2]{0}) get-tuple-element(infeed),
    index=0, sharding={{maximal device=0}, {maximal device=1}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("(f32[9,2], f32[2])"),
                          op::GetTupleElement(op::Conditional(
                              op::Convert(op::PartitionId()), op::AfterAll(),
                              op::AfterAll()))));
  auto first_infeed = AllOf(op::Shape("((f32[9,2], ()), token[])"),
                            op::Infeed(op::Parameter()));
  EXPECT_THAT(root->operand(0)->called_computations()[0]->root_instruction(),
              AllOf(op::Shape("((f32[9,2], f32[2]), token[])"),
                    op::Tuple(op::Tuple(op::GetTupleElement(
                                            op::GetTupleElement(first_infeed)),
                                        op::Broadcast(op::Constant())),
                              op::GetTupleElement(first_infeed))));
  auto second_infeed =
      AllOf(op::Shape("(((), f32[2]), token[])"), op::Infeed(op::Parameter()));
  EXPECT_THAT(root->operand(0)->called_computations()[1]->root_instruction(),
              AllOf(op::Shape("((f32[9,2], f32[2]), token[])"),
                    op::Tuple(op::Tuple(op::Broadcast(op::Constant()),
                                        op::GetTupleElement(op::GetTupleElement(
                                            second_infeed))),
                              op::GetTupleElement(second_infeed))));
}

TEST_F(SpmdPartitioningTest, TiledToReplicatedReduce) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  constant = f32[3,3]{1,0} constant({{1,1,1},{1,1,1},{1,1,1}}),
    sharding={devices=[2,1]0,1}
  constant.1 = f32[] constant(0), sharding={replicated}
  ROOT reduce = f32[] reduce(constant, constant.1), dimensions={0,1},
    to_apply=sum, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      op::AllReduce(op::Reduce(
          op::Select(
              op::Compare(op::Add(op::Iota(), op::Broadcast(op::Reshape())),
                          op::Broadcast(op::Constant())),
              AllOf(op::Shape("f32[2,3]{1,0}"),
                    op::DynamicSlice(op::Pad(op::Constant(), op::Constant()),
                                     op::Reshape(), op::Constant())),
              op::Broadcast(op::Constant())),
          op::Constant())));
}

TEST_F(SpmdPartitioningTest, TiledElementwise) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  constant = f32[3,3]{1,0} constant({{1,1,1},{1,1,1},{1,1,1}}),
    sharding={devices=[2,1]0,1}
  constant.1 = f32[3,3]{1,0} constant({{2,2,2},{2,2,2},{2,2,2}}),
    sharding={replicated}
  multiply = f32[3,3]{1,0} multiply(constant, constant.1),
    sharding={devices=[2,1]0,1}
  ROOT add = f32[3,3]{1,0} add(multiply, constant.1),
    sharding={devices=[2,1]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      AllOf(
          op::Shape("f32[2,3]{1,0}"),
          op::Add(op::Multiply(
                      op::DynamicSlice(op::Pad(op::Constant(), op::Constant()),
                                       op::Reshape(), op::Constant()),
                      op::DynamicSlice(op::Pad(op::Constant(), op::Constant()),
                                       op::Reshape(), op::Constant())),
                  op::DynamicSlice(op::Pad(op::Constant(), op::Constant()),
                                   op::Reshape(), op::Constant()))));
}

TEST_F(SpmdPartitioningTest, TiledAllReduce) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  parameter = f32[3,3]{1,0} parameter(0), sharding={devices=[2,1]0,1}
  ROOT all-reduce = f32[3,3]{1,0} all-reduce(parameter), to_apply=sum,
    replica_groups={}, sharding={devices=[2,1]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, AllOf(op::Shape("f32[2,3]{1,0}"), op::AllReduce(op::Parameter(0))));
}

TEST_F(SpmdPartitioningTest, BroadcastOnlyNewDimsSharded) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  constant = f32[4,3]{1,0} constant({{1,1,1},{1,1,1},{1,1,1},{1,1,1}}),
    sharding={replicated}
  ROOT broadcast = f32[3,4,3]{2,1,0} broadcast(constant), dimensions={1,2},
    sharding={devices=[2,1,1]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("f32[2,4,3]{2,1,0}"),
                          op::Broadcast(op::Constant())));
}

TEST_F(SpmdPartitioningTest, BroadcastOnlyOldDimsSharded) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  constant = f32[4,3]{1,0} constant({{1,1,1},{1,1,1},{1,1,1},{1,1,1}}),
    sharding={replicated}
  ROOT broadcast = f32[4,4,3]{2,1,0} broadcast(constant), dimensions={1,2},
    sharding={devices=[1,2,1]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("f32[4,2,3]{2,1,0}"),
                          op::Broadcast(op::DynamicSlice(
                              op::Constant(), op::Reshape(), op::Constant()))));
}

TEST_F(SpmdPartitioningTest, BroadcastBothOldAndNewDimsSharded) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  constant = f32[4,3]{1,0} constant({{1,1,1},{1,1,1},{1,1,1},{1,1,1}}),
    sharding={replicated}
  ROOT broadcast = f32[4,4,3]{2,1,0} broadcast(constant), dimensions={1,2},
    sharding={devices=[2,2,1]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      AllOf(op::Shape("f32[2,2,3]{2,1,0}"),
            op::Broadcast(AllOf(op::Shape("f32[2,3]{1,0}"),
                                op::DynamicSlice(op::Constant(), op::Reshape(),
                                                 op::Constant())))));
}

TEST_F(SpmdPartitioningTest,
       BroadcastBothOldAndNewDimsShardedPartiallySharded) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  param = f32[4,3] parameter(0),
    sharding={devices=[1,2,4]0,1,4,5,2,3,6,7 last_tile_dim_replicate}
  ROOT broadcast = f32[4,4,3] broadcast(param), dimensions={1,2},
    sharding={devices=[2,1,2,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      AllOf(op::Shape("f32[2,4,2]"),
            op::Broadcast(AllOf(op::Shape("f32[4,2]"), op::Parameter(0)))));
}

TEST_F(SpmdPartitioningTest,
       ConvWithParallelDimAndNonParallelSpatialDimPartitioned) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,12,12,24,32] parameter(0)
  %lhs.copy = f32[32,12,12,24,32] copy(%lhs),
    sharding={devices=[2,2,1,1,1]0,1,2,3}
  %rhs = f32[32,6,6,16,32] parameter(1)
  %rhs.copy = f32[32,6,6,16,32] copy(%rhs),
    sharding={devices=[2,2,1,1,1]0,1,2,3}
  ROOT %conv = f32[32,7,7,24,16] convolution(%lhs.copy, %rhs.copy),
    dim_labels=012bf_012oi->012bf,
    window={size=32x6x6 stride=31x1x1 lhs_dilate=32x1x1},
    sharding={devices=[2,2,1,1,1]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(op::Copy(op::DynamicSlice(
                             op::Parameter(), op::Reshape(), op::Reshape(),
                             op::Constant(), op::Constant(), op::Constant())),
                         op::Shape("f32[16,6,12,24,32]"));
  const auto rhs = AllOf(op::Copy(op::DynamicSlice(
                             op::Parameter(), op::Reshape(), op::Reshape(),
                             op::Constant(), op::Constant(), op::Constant())),
                         op::Shape("f32[16,3,6,16,32]"));
  auto resharded_rhs =
      AllOf(op::Shape("f32[16,6,6,16,32]"),
            op::AllReduce(op::DynamicUpdateSlice(
                op::Broadcast(), rhs, op::Constant(), op::Reshape(),
                op::Constant(), op::Constant(), op::Constant())));

  auto left_halo = AllOf(op::CollectivePermute(op::Slice(lhs)),
                         op::Shape("f32[16,2,12,24,32]"));
  auto right_halo = AllOf(op::CollectivePermute(op::Slice(lhs)),
                          op::Shape("f32[16,3,12,24,32]"));
  EXPECT_THAT(
      root,
      AllOf(op::Convolution(
                op::Select(op::Compare(),
                           op::DynamicSlice(
                               op::Concatenate(left_halo, lhs, right_halo),
                               op::Constant(), op::Add(), op::Constant(),
                               op::Constant(), op::Constant()),
                           op::Broadcast()),
                resharded_rhs),
            op::Shape("f32[16,4,7,24,16]")));
}

TEST_F(SpmdPartitioningTest, BroadcastPropagateTiledSharding) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  constant = f32[4,3]{1,0} constant({{1,1,1},{1,4,1},{1,3,1},{1,2,1}}),
    sharding={devices=[2,1]0,1}
  ROOT broadcast = f32[4,4,3]{2,1,0} broadcast(constant), dimensions={1,2},
    sharding={devices=[1,2,1]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("f32[4,2,3]{2,1,0}"),
                          op::Broadcast(op::DynamicSlice(
                              op::Constant(), op::Reshape(), op::Constant()))));
}

TEST_F(SpmdPartitioningTest, OutfeedSingleDevice) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  token.0 = token[] after-all()
  data = f32[1024]{0} parameter(0), sharding={maximal device=0}
  outfeed = token[] outfeed(data, token.0), sharding={maximal device=0}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("token[]"),
                          op::Conditional(
                              op::Compare(op::PartitionId(), op::Constant()),
                              op::Tuple(op::Parameter(0), op::AfterAll()),
                              op::Tuple(op::Parameter(0), op::AfterAll()))));

  HloInstruction* root_b0 = root->branch_computation(0)->root_instruction();
  EXPECT_THAT(root_b0,
              AllOf(op::Shape("token[]"),
                    op::Outfeed(op::GetTupleElement(op::Parameter(), 0),
                                op::GetTupleElement(op::Parameter(), 1))));

  HloInstruction* root_b1 = root->branch_computation(1)->root_instruction();
  EXPECT_THAT(root_b1, AllOf(op::Shape("token[]"), op::AfterAll()));
}

TEST_F(SpmdPartitioningTest, OutfeedEvenlyTiled) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  token.0 = token[] after-all()
  data = f32[1024]{0} parameter(0), sharding={devices=[2]0,1}
  ROOT outfeed = token[] outfeed(data, token.0), sharding={devices=[2]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("token[]"),
                          op::Outfeed(op::Parameter(), op::AfterAll())));
}

TEST_F(SpmdPartitioningTest, OutfeedTupleEvenlyTiled) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  token.0 = token[] after-all()
  data = (f32[1024,2]{1,0}, f32[2]{0}) parameter(0), sharding={{devices=[2,1]0,1},
    {devices=[2]0,1}}
  ROOT outfeed = token[] outfeed(data, token.0),
    outfeed_shape=(f32[1024,2]{0,1}, f32[2]{0}), sharding={{devices=[2,1]0,1},
    {devices=[2]0,1}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("token[]"),
                          op::Outfeed(op::Parameter(), op::AfterAll())));
  auto expected_layout0 = LayoutUtil::MakeLayout({0, 1});
  auto expected_layout1 = LayoutUtil::MakeLayout({0});
  EXPECT_TRUE(LayoutUtil::Equal(root->outfeed_shape().tuple_shapes(0).layout(),
                                expected_layout0));
  EXPECT_TRUE(LayoutUtil::Equal(root->outfeed_shape().tuple_shapes(1).layout(),
                                expected_layout1));
}

TEST_F(SpmdPartitioningTest, OutfeedReplicated) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  token.0 = token[] after-all()
  data = (f32[1024,2]{1,0}, f32[2]{0}) parameter(0), sharding={{devices=[2,1]0,1},
    {replicated}}
  ROOT outfeed = token[] outfeed(data, token.0), sharding={{devices=[2,1]0,1},
    {replicated}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("token[]"),
                          op::Outfeed(op::Parameter(), op::AfterAll())));
}

TEST_F(SpmdPartitioningTest, OutfeedUnevenlyTiled) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  token.0 = token[] after-all()
  data = (f32[1023,2]{1,0}, f32[3]{0}) parameter(0), sharding={{devices=[2,1]0,1},
    {devices=[2]0,1}}
  outfeed = token[] outfeed(data, token.0),
    outfeed_shape=(f32[1023,2]{0,1}, f32[3]{0}), sharding={{devices=[2,1]0,1},
    {devices=[2]0,1}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, AllOf(op::Shape("token[]"),
                  op::Conditional(op::Convert(),
                                  op::Tuple(op::Parameter(), op::AfterAll()),
                                  op::Tuple(op::Parameter(), op::AfterAll()))));

  auto first_outfeed =
      AllOf(op::Shape("(f32[512,2], f32[2])"), op::GetTupleElement());
  EXPECT_THAT(root->called_computations()[0]->root_instruction(),
              AllOf(op::Shape("token[]"),
                    op::Outfeed(first_outfeed, op::GetTupleElement())));

  auto second_outfeed = AllOf(op::Shape("(f32[511,2], f32[1])"), op::Tuple());
  EXPECT_THAT(root->called_computations()[1]->root_instruction(),
              AllOf(op::Shape("token[]"),
                    op::Outfeed(second_outfeed, op::GetTupleElement())));

  auto expected_layout0 = LayoutUtil::MakeLayout({0, 1});
  auto expected_layout1 = LayoutUtil::MakeLayout({0});
  auto first_outfeed_instr = root->called_computations()[0]->root_instruction();
  auto second_outfeed_instr =
      root->called_computations()[1]->root_instruction();
  EXPECT_TRUE(LayoutUtil::Equal(
      first_outfeed_instr->outfeed_shape().tuple_shapes(0).layout(),
      expected_layout0));
  EXPECT_TRUE(LayoutUtil::Equal(
      first_outfeed_instr->outfeed_shape().tuple_shapes(1).layout(),
      expected_layout1));
  EXPECT_TRUE(LayoutUtil::Equal(
      second_outfeed_instr->outfeed_shape().tuple_shapes(0).layout(),
      expected_layout0));
  EXPECT_TRUE(LayoutUtil::Equal(
      second_outfeed_instr->outfeed_shape().tuple_shapes(1).layout(),
      expected_layout1));
}

TEST_F(SpmdPartitioningTest, ReduceWindowReplicatedInput) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  constant = f32[6,2]{1,0} constant({{1,1},{1,4},{2,1},{3,1},{1,2},{2,2}}),
    sharding={replicated}
  constant.1 = f32[] constant(0), sharding={replicated}
  ROOT reduce-window = f32[3,2]{1,0} reduce-window(constant, constant.1),
    window={size=3x1 stride=2x1 pad=1_0x0_0}, to_apply=sum,
    sharding={devices=[2,1]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      AllOf(op::Shape("f32[2,2]{1,0}"),
            op::ReduceWindow(
                op::DynamicSlice(AllOf(op::Shape("f32[9,2]{1,0}"),
                                       op::Pad(op::Constant(), op::Constant())),
                                 op::Multiply(op::Reshape(), op::Constant()),
                                 op::Constant()),
                op::Constant())));
}

TEST_F(SpmdPartitioningTest, ReduceWindowTiledNegativeLeftHalo) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  constant = f32[6,2]{1,0} constant({{1,1},{1,4},{2,1},{3,1},{1,2},{2,2}}),
    sharding={devices=[2,1]0,1}
  constant.1 = f32[] constant(0), sharding={replicated}
  ROOT %reduce-window = f32[3,2]{1,0} reduce-window(%constant, %constant.1),
    window={size=3x1 stride=2x1 pad=0_1x0_0}, to_apply=sum,
    sharding={devices=[2,1]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();

  auto sharded_input =
      op::DynamicSlice(op::Constant(), op::Reshape(), op::Constant());
  auto right_halo = AllOf(op::Shape("f32[2,2]{1,0}"),
                          op::CollectivePermute(op::Slice(sharded_input)));
  auto pre_masking = op::DynamicSlice(
      AllOf(
          op::Shape("f32[6,2]{1,0}"),
          op::Pad(op::Concatenate(sharded_input, right_halo), op::Constant())),
      op::Reshape(), op::Constant());
  auto index_in_padded = op::Add(
      op::Iota(), op::Broadcast(op::Multiply(op::Reshape(), op::Constant())));
  auto masked =
      op::Select(op::Compare(index_in_padded, op::Broadcast(op::Constant())),
                 pre_masking, op::Broadcast(op::Constant()));
  EXPECT_THAT(root, AllOf(op::Shape("f32[2,2]{1,0}"),
                          op::ReduceWindow(masked, op::Constant())));
}

TEST_F(SpmdPartitioningTest, ReduceWindowTiledOneSideHaloBeyondNeighbor) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  param = f32[9,2] parameter(0), sharding={devices=[5,1]0,1,2,3,4}
  constant.1 = f32[] constant(0), sharding={replicated}
  ROOT reduce-window = f32[5,2]{1,0} reduce-window(param, constant.1),
    window={size=4x1 stride=2x1 pad=3_0x0_0}, to_apply=sum,
    sharding={devices=[5,1]0,1,2,3,4}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/5));
  VLOG(1) << module->ToString();
  auto halo0 = AllOf(op::Shape("f32[1,2]"),
                     op::CollectivePermute(op::Slice(op::Parameter(0))));
  auto halo1 =
      AllOf(op::Shape("f32[2,2]"), op::CollectivePermute(op::Parameter(0)));
  auto pre_mask =
      AllOf(op::Shape("f32[4,2]"),
            op::Concatenate(halo0, halo1, op::Slice(op::Parameter(0))));
  auto masked =
      op::Select(op::Compare(op::Add(op::Iota(), op::Broadcast(op::Multiply())),
                             op::Broadcast(op::Constant())),
                 pre_mask, op::Broadcast(op::Constant()));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("f32[1,2]{1,0}"),
                          op::ReduceWindow(masked, op::Constant())));
}

TEST_F(SpmdPartitioningTest, ReduceWindowTiledOneSideUnequalHalo) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  constant = f32[9,2]{1,0} constant(
    {{1,1},{1,4},{2,1},{3,1},{1,2},{2,2},{4,1},{1,2},{2,1}}),
    sharding={devices=[3,1]0,1,2}
  constant.1 = f32[] constant(0), sharding={replicated}
  ROOT reduce-window = f32[5,2]{1,0} reduce-window(constant, constant.1),
    window={size=3x1 stride=2x1 pad=1_1x0_0}, to_apply=sum,
    sharding={devices=[3,1]0,1,2}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/3));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();

  auto sharded_input =
      op::DynamicSlice(op::Constant(), op::Reshape(), op::Constant());
  auto right_halo = AllOf(op::Shape("f32[2,2]{1,0}"),
                          op::CollectivePermute(op::Slice(sharded_input)));
  auto pre_masking = op::DynamicSlice(
      AllOf(
          op::Shape("f32[7,2]{1,0}"),
          op::Pad(op::Concatenate(sharded_input, right_halo), op::Constant())),
      op::Reshape(), op::Constant());
  auto index_in_padded = op::Add(
      op::Iota(), op::Broadcast(op::Multiply(op::Reshape(), op::Constant())));
  auto masked = op::Select(
      op::And(op::Compare(index_in_padded, op::Broadcast(op::Constant())),
              op::Compare(index_in_padded, op::Broadcast(op::Constant()))),
      pre_masking, op::Broadcast(op::Constant()));
  EXPECT_THAT(root, AllOf(op::Shape("f32[2,2]{1,0}"),
                          op::ReduceWindow(masked, op::Constant())));
}

TEST_F(SpmdPartitioningTest, ReduceWindowTiledTwoSideHalo) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  constant = f32[4,2]{1,0} constant({{1,1},{1,4},{2,1},{3,1}}),
    sharding={devices=[2,1]0,1}
  constant.1 = f32[] constant(0), sharding={replicated}
  ROOT reduce-window = f32[2,2]{1,0} reduce-window(constant, constant.1),
    window={size=5x1 stride=3x1 pad=2_2x0_0}, to_apply=sum,
    sharding={devices=[2,1]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();

  auto sharded_input =
      op::DynamicSlice(op::Constant(), op::Reshape(), op::Constant());
  auto left_halo = AllOf(op::Shape("f32[1,2]{1,0}"),
                         op::CollectivePermute(op::Slice(sharded_input)));
  auto right_halo = AllOf(op::Shape("f32[1,2]{1,0}"),
                          op::CollectivePermute(op::Slice(sharded_input)));
  auto pre_masking = AllOf(
      op::Shape("f32[5,2]{1,0}"),
      op::DynamicSlice(
          AllOf(op::Shape("f32[6,2]{1,0}"),
                op::Pad(op::Concatenate(left_halo, sharded_input, right_halo),
                        op::Constant())),
          op::Reshape(), op::Constant()));
  auto index_in_padded = op::Add(
      op::Iota(), op::Broadcast(op::Multiply(op::Reshape(), op::Constant())));
  auto masked = op::Select(
      op::And(op::Compare(index_in_padded, op::Broadcast(op::Constant())),
              op::Compare(index_in_padded, op::Broadcast(op::Constant()))),
      pre_masking, op::Broadcast(op::Constant()));
  EXPECT_THAT(root, AllOf(op::Shape("f32[1,2]{1,0}"),
                          op::ReduceWindow(masked, op::Constant())));
}

TEST_F(SpmdPartitioningTest, ReduceWindowTiled2D) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  token0 = token[] after-all(), sharding={maximal device=0}
  infeed = (f32[4,4,2,2]{3,2,1,0}, token[]) infeed(token0),
    sharding={{devices=[2,2,1,1]0,1,2,3}, {maximal device=0}}
  infeed.data = f32[4,4,2,2]{3,2,1,0} get-tuple-element(infeed), index=0,
    sharding={devices=[2,2,1,1]0,1,2,3}
  constant = f32[] constant(0), sharding={replicated}
  ROOT reduce-window = f32[2,2,2,2]{3,2,1,0} reduce-window(infeed.data, constant),
    window={size=5x5x1x1 stride=3x3x1x1 pad=2_2x2_2x0_0x0_0}, to_apply=sum,
    sharding={devices=[2,2,1,1]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();

  auto sharded_input = AllOf(op::Shape("f32[2,2,2,2]{3,2,1,0}"),
                             op::GetTupleElement(op::Infeed()));
  auto dim0_left_halo = AllOf(op::Shape("f32[1,2,2,2]{3,2,1,0}"),
                              op::CollectivePermute(op::Slice(sharded_input)));
  auto dim0_right_halo = AllOf(op::Shape("f32[1,2,2,2]{3,2,1,0}"),
                               op::CollectivePermute(op::Slice(sharded_input)));
  auto dim0_pre_masking = op::DynamicSlice(
      AllOf(op::Shape("f32[6,2,2,2]{3,2,1,0}"),
            op::Pad(
                op::Concatenate(dim0_left_halo, sharded_input, dim0_right_halo),
                op::Constant())),
      op::Reshape(), op::Constant(), op::Constant(), op::Constant());
  auto dim0_index_in_padded = op::Add(
      op::Iota(), op::Broadcast(op::Multiply(op::Reshape(), op::Constant())));
  auto dim0_masked = op::Select(
      op::And(op::Compare(dim0_index_in_padded, op::Broadcast(op::Constant())),
              op::Compare(dim0_index_in_padded, op::Broadcast(op::Constant()))),
      dim0_pre_masking, op::Broadcast(op::Constant()));
  auto dim0_resharded = AllOf(op::Shape("f32[5,2,2,2]{3,2,1,0}"), dim0_masked);
  auto dim1_left_halo = AllOf(op::Shape("f32[5,1,2,2]{3,2,1,0}"),
                              op::CollectivePermute(op::Slice(dim0_resharded)));
  auto dim1_right_halo =
      AllOf(op::Shape("f32[5,1,2,2]{3,2,1,0}"),
            op::CollectivePermute(op::Slice(dim0_resharded)));
  auto dim1_pre_masking = op::DynamicSlice(
      AllOf(op::Shape("f32[5,6,2,2]{3,2,1,0}"),
            op::Pad(op::Concatenate(dim1_left_halo, dim0_resharded,
                                    dim1_right_halo),
                    op::Constant())),
      op::Constant(), op::Reshape(), op::Constant(), op::Constant());
  auto dim1_index_in_padded = op::Add(
      op::Iota(), op::Broadcast(op::Multiply(op::Reshape(), op::Constant())));
  auto dim1_masked = op::Select(
      op::And(op::Compare(dim1_index_in_padded, op::Broadcast(op::Constant())),
              op::Compare(dim1_index_in_padded, op::Broadcast(op::Constant()))),
      dim1_pre_masking, op::Broadcast(op::Constant()));
  auto dim1_resharded = AllOf(op::Shape("f32[5,5,2,2]{3,2,1,0}"), dim1_masked);
  EXPECT_THAT(root, AllOf(op::Shape("f32[1,1,2,2]{3,2,1,0}"),
                          op::ReduceWindow(dim1_resharded, op::Constant())));
}

TEST_F(SpmdPartitioningTest, ConvolutionLhsTiledRhsReplicated) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[128,224,224,3] parameter(0)
  %lhs.copy = f32[128,224,224,3] copy(f32[128,224,224,3] %lhs),
    sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[7,7,3,64] parameter(1)
  %rhs.copy = f32[7,7,3,64] copy(f32[7,7,3,64] %rhs),
    sharding={replicated}
  ROOT %conv = f32[128,112,112,64] convolution(
    f32[128,224,224,3] %lhs.copy,
    f32[7,7,3,64] %rhs.copy),
    window={size=7x7 stride=2x2 pad=3_3x3_3},
    dim_labels=b01f_01io->b01f,
    sharding={devices=[1,2,1,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,112,224,3]"));
  const auto rhs = AllOf(op::Copy(op::Parameter()), op::Shape("f32[7,7,3,64]"));

  auto left_halo = AllOf(op::CollectivePermute(op::Slice(lhs)),
                         op::Shape("f32[128,3,224,3]"));
  auto right_halo = AllOf(op::CollectivePermute(op::Slice(lhs)),
                          op::Shape("f32[128,2,224,3]"));
  EXPECT_THAT(root,
              AllOf(op::Convolution(
                        op::Select(op::And(),
                                   op::Concatenate(left_halo, lhs, right_halo),
                                   op::Broadcast()),
                        rhs),
                    op::Shape("f32[128,56,112,64]")));
}

TEST_F(SpmdPartitioningTest, ConvolutionLhsTiledRhsReplicatedNeedReshard) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[128,224,224,3] parameter(0)
  %lhs.copy = f32[128,224,224,3] copy(f32[128,224,224,3] %lhs),
    sharding={devices=[2,1,1,1]0,1}
  %rhs = f32[7,7,3,64] parameter(1)
  %rhs.copy = f32[7,7,3,64] copy(f32[7,7,3,64] %rhs),
    sharding={replicated}
  ROOT %conv = f32[128,112,112,64] convolution(
    f32[128,224,224,3] %lhs.copy,
    f32[7,7,3,64] %rhs.copy),
    window={size=7x7 stride=2x2 pad=3_3x3_3},
    dim_labels=b01f_01io->b01f,
    sharding={devices=[1,2,1,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Reshape(), op::Constant(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[64,224,224,3]"));
  auto all_to_all =
      AllOf(op::AllToAll(op::Reshape(lhs)), op::Shape("f32[64,2,112,224,3]"));
  auto reshard_lhs = AllOf(op::Reshape(op::Transpose(all_to_all)),
                           op::Shape("f32[128,112,224,3]"));

  const auto rhs = AllOf(op::Copy(op::Parameter()), op::Shape("f32[7,7,3,64]"));

  auto left_halo = AllOf(op::CollectivePermute(op::Slice(reshard_lhs)),
                         op::Shape("f32[128,3,224,3]"));
  auto right_halo = AllOf(op::CollectivePermute(op::Slice(reshard_lhs)),
                          op::Shape("f32[128,2,224,3]"));
  EXPECT_THAT(
      root,
      AllOf(op::Convolution(
                op::Select(op::And(),
                           op::Concatenate(left_halo, reshard_lhs, right_halo),
                           op::Broadcast()),
                rhs),
            op::Shape("f32[128,56,112,64]")));
}

TEST_F(SpmdPartitioningTest, ConvolutionLhsTiledRhsReplicatedReordered) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[224,224,3,128] parameter(0)
  %lhs.copy = f32[224,224,3,128] copy(%lhs), sharding={devices=[2,1,1,1]0,1}
  %rhs = f32[7,7,3,64] parameter(1)
  %rhs.copy = f32[7,7,3,64] copy(%rhs), sharding={replicated}
  ROOT %conv = f32[128,112,112,64] convolution(%lhs.copy, %rhs.copy),
    window={size=7x7 stride=2x2 pad=3_3x3_3},
    dim_labels=01fb_01io->b01f,
    sharding={devices=[1,2,1,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Reshape(), op::Constant(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[112,224,3,128]"));
  const auto rhs = AllOf(op::Copy(op::Parameter()), op::Shape("f32[7,7,3,64]"));

  auto left_halo = AllOf(op::CollectivePermute(op::Slice(lhs)),
                         op::Shape("f32[3,224,3,128]"));
  auto right_halo = AllOf(op::CollectivePermute(op::Slice(lhs)),
                          op::Shape("f32[2,224,3,128]"));
  EXPECT_THAT(root,
              AllOf(op::Convolution(
                        op::Select(op::And(),
                                   op::Concatenate(left_halo, lhs, right_halo),
                                   op::Broadcast()),
                        rhs),
                    op::Shape("f32[128,56,112,64]")));
}

// (stride * per_shard_window_count) % dilation == 0
TEST_F(SpmdPartitioningTest,
       ConvolutionBaseDilationSameStartPatternLhsTiledRhsReplicated) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[128,7,7,512] parameter(0)
  %lhs.copy = f32[128,7,7,512] copy(%lhs),
    sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[3,3,512,512] parameter(1)
  %rhs.copy = f32[3,3,512,512] copy(%rhs),
    sharding={replicated}
  ROOT %conv = f32[128,4,4,512] convolution(%lhs.copy, %rhs.copy),
    window={size=3x3 stride=4x4 pad=1_1x1_1 lhs_dilate=2x2 rhs_reversal=1x1},
    dim_labels=b01f_01io->b01f,
    sharding={devices=[1,2,1,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  // There is no halo exchange, and because the last element in the shard is not
  // needed (stride == 4), the LHS will be just a slice.
  auto sliced_lhs =
      AllOf(op::Slice(op::Copy(op::DynamicSlice(
                op::Pad(op::Parameter(), op::Constant()), op::Constant(),
                op::Reshape(), op::Constant(), op::Constant()))),
            op::Shape("f32[128,3,7,512]"));
  const auto rhs =
      AllOf(op::Copy(op::Parameter()), op::Shape("f32[3,3,512,512]"));
  EXPECT_THAT(root, AllOf(op::Convolution(sliced_lhs, rhs),
                          op::Shape("f32[128,2,4,512]")));
  EXPECT_EQ(root->window().dimensions(0).padding_low(), 1);
  EXPECT_EQ(root->window().dimensions(0).padding_high(), 1);
}

// (stride * per_shard_window_count) % dilation != 0 but stride == 1
TEST_F(SpmdPartitioningTest,
       ConvolutionBaseDilationStride1LhsTiledRhsReplicated) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[128,7,7,512] parameter(0)
  %lhs.copy = f32[128,7,7,512] copy(%lhs),
    sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[3,3,512,512] parameter(1)
  %rhs.copy = f32[3,3,512,512] copy(%rhs),
    sharding={replicated}
  ROOT %conv = f32[128,14,14,512] convolution(%lhs.copy, %rhs.copy),
    window={size=3x3 pad=1_2x1_2 lhs_dilate=2x2 rhs_reversal=1x1},
    dim_labels=b01f_01io->b01f,
    sharding={devices=[1,2,1,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(), op::Constant()),
                                      op::Constant(), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[128,4,7,512]"));
  const auto rhs =
      AllOf(op::Copy(op::Parameter()), op::Shape("f32[3,3,512,512]"));

  auto left_halo = AllOf(op::CollectivePermute(op::Slice(lhs)),
                         op::Shape("f32[128,1,7,512]"));
  auto start_window = op::Multiply(op::Reshape(), op::Constant());
  auto start_input_element = op::Divide(start_window, op::Constant());
  auto dynamic_offset_for_padded_concat = op::Subtract(
      op::Constant(), op::Subtract(op::Multiply(op::Reshape(), op::Constant()),
                                   start_input_element));
  auto pre_masking =
      AllOf(op::Shape("f32[128,5,7,512]"),
            op::DynamicSlice(
                AllOf(op::Shape("f32[128,6,7,512]"),
                      op::Pad(op::Concatenate(left_halo, lhs), op::Constant())),
                op::Constant(), dynamic_offset_for_padded_concat,
                op::Constant(), op::Constant()));
  auto masked = op::Select(
      op::Compare(op::Add(op::Iota(), op::Broadcast(start_input_element)),
                  op::Broadcast(op::Constant())),
      pre_masking, op::Broadcast(op::Constant()));
  auto dynamic_offset_on_output = op::Subtract(
      start_window, op::Multiply(start_input_element, op::Constant()));
  EXPECT_THAT(root,
              AllOf(op::DynamicSlice(AllOf(op::Convolution(masked, rhs),
                                           op::Shape("f32[128,8,14,512]")),
                                     op::Constant(), dynamic_offset_on_output,
                                     op::Constant(), op::Constant()),
                    op::Shape("f32[128,7,14,512]")));
  EXPECT_EQ(root->operand(0)->window().dimensions(0).padding_low(), 1);
  EXPECT_EQ(root->operand(0)->window().dimensions(0).padding_high(), 0);
}

TEST_F(SpmdPartitioningTest, SelectAndScatterNoOverlap) {
  absl::string_view hlo_string = R"(
HloModule module

ge {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT compare = pred[] compare(a, b), direction=GE
}

sum {
  c = f32[] parameter(0)
  d = f32[] parameter(1)
  ROOT add = f32[] add(c, d)
}

ENTRY entry {
  %param = f32[11,4]{1,0} parameter(0)
  %param.copy = f32[11,4] copy(%param),
    sharding={devices=[4,1]0,1,2,3}
  constant = f32[4,2]{1,0} constant({{1,2},{3,4},{1,0},{2,8}}),
    sharding={devices=[4,1]0,1,2,3}
  constant.1 = f32[] constant(0), sharding={replicated}
  ROOT select-and-scatter = f32[11,4]{1,0} select-and-scatter(param.copy,
    constant, constant.1), window={size=3x2 stride=3x2 pad=0_1x0_0},
    select=ge, scatter=sum, sharding={devices=[4,1]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  auto source =
      AllOf(op::Shape("f32[1,2]{1,0}"),
            op::DynamicSlice(op::Constant(), op::Reshape(), op::Constant()));
  auto masked_data = AllOf(
      op::Shape("f32[3,4]{1,0}"),
      op::Select(
          op::Compare(op::Add(op::Iota(), op::Broadcast(op::Multiply(
                                              op::Reshape(), op::Constant()))),
                      op::Broadcast(op::Constant())),
          op::Copy(op::DynamicSlice(op::Pad(op::Parameter(), op::Constant()),
                                    op::Reshape(), op::Constant())),
          op::Broadcast(op::Constant())));

  EXPECT_THAT(root,
              AllOf(op::SelectAndScatter(masked_data, source, op::Constant()),
                    op::Shape("f32[3,4]{1,0}")));
  EXPECT_EQ(root->window().dimensions(0).padding_low(), 0);
  EXPECT_EQ(root->window().dimensions(0).padding_high(), 0);
}

TEST_F(SpmdPartitioningTest, SelectAndScatterNoOverlapReshard) {
  absl::string_view hlo_string = R"(
HloModule module

ge {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT compare = pred[] compare(a, b), direction=GE
}

sum {
  c = f32[] parameter(0)
  d = f32[] parameter(1)
  ROOT add = f32[] add(c, d)
}

ENTRY entry {
  %param = f32[11,4]{1,0} parameter(0)
  %param.copy = f32[11,4] copy(%param),
    sharding={devices=[1,4]0,1,2,3}
  constant = f32[4,2]{1,0} constant({{1,2},{3,4},{1,0},{2,8}}),
    sharding={devices=[4,1]0,1,2,3}
  constant.1 = f32[] constant(0), sharding={replicated}
  ROOT select-and-scatter = f32[11,4]{1,0} select-and-scatter(param.copy,
    constant, constant.1), window={size=3x2 stride=3x2 pad=0_1x0_0},
    select=ge, scatter=sum, sharding={devices=[4,1]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  auto source =
      AllOf(op::Shape("f32[1,2]{1,0}"),
            op::DynamicSlice(op::Constant(), op::Reshape(), op::Constant()));
  auto operand = AllOf(op::Copy(op::DynamicSlice(
                           op::Parameter(0), op::Constant(), op::Reshape())),
                       op::Shape("f32[11,1]"));
  auto reshard_operand = op::Reshape(op::Transpose(
      op::AllToAll(op::Reshape(op::Pad(operand, op::Constant())))));
  auto masked_data = AllOf(
      op::Shape("f32[3,4]{1,0}"),
      op::Select(
          op::Compare(op::Add(op::Iota(), op::Broadcast(op::Multiply(
                                              op::Reshape(), op::Constant()))),
                      op::Broadcast(op::Constant())),
          reshard_operand, op::Broadcast(op::Constant())));

  EXPECT_THAT(root,
              AllOf(op::SelectAndScatter(masked_data, source, op::Constant()),
                    op::Shape("f32[3,4]{1,0}")));
  EXPECT_EQ(root->window().dimensions(0).padding_low(), 0);
  EXPECT_EQ(root->window().dimensions(0).padding_high(), 0);
}

TEST_F(SpmdPartitioningTest, SelectAndScatterWithOverlap) {
  absl::string_view hlo_string = R"(
HloModule module

ge {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT compare = pred[] compare(a, b), direction=GE
}

sum {
  c = f32[] parameter(0)
  d = f32[] parameter(1)
  ROOT add = f32[] add(c, d)
}

ENTRY entry {
  %param = f32[11,4]{1,0} parameter(0)
  %param.copy = f32[11,4] copy(%param),
    sharding={devices=[4,1]0,1,2,3}
  constant = f32[6,2]{1,0} constant({{1,2},{3,4},{1,0},{2,8},{6,6},{1,9}}),
    sharding={devices=[4,1]0,1,2,3}
  constant.1 = f32[] constant(0), sharding={replicated}
  ROOT select-and-scatter = f32[11,4]{1,0} select-and-scatter(param.copy,
    constant, constant.1), window={size=3x2 stride=2x2 pad=1_1x0_0},
    select=ge, scatter=sum, sharding={devices=[4,1]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();

  auto source_shard =
      AllOf(op::Shape("f32[2,2]{1,0}"),
            op::DynamicSlice(op::Pad(), op::Reshape(), op::Constant()));
  // Max halo size is the same as the shard size, so slice is not needed.
  auto source_left_halo = op::CollectivePermute(source_shard);
  auto required_source_shard_start =
      op::Divide(op::Multiply(op::Reshape(), op::Constant()), op::Constant());
  auto source_with_halo = op::DynamicSlice(
      AllOf(op::Shape("f32[5,2]{1,0}"),
            op::Pad(op::Concatenate(source_left_halo, source_shard),
                    op::Constant())),
      op::Subtract(op::Constant(),
                   op::Subtract(op::Multiply(op::Reshape(), op::Constant()),
                                required_source_shard_start)),
      op::Constant());
  auto masked_source_with_halo = AllOf(
      AllOf(op::Shape("f32[3,2]{1,0}")),
      op::Select(
          op::Compare(
              op::Add(op::Iota(), op::Broadcast(required_source_shard_start)),
              op::Broadcast(op::Constant())),
          source_with_halo, op::Broadcast(op::Constant())));

  auto data_shard =
      AllOf(op::Shape("f32[3,4]{1,0}"),
            op::Copy(op::DynamicSlice(op::Pad(op::Parameter(), op::Constant()),
                                      op::Reshape(), op::Constant())));
  auto data_left_halo = AllOf(op::Shape("f32[2,4]{1,0}"),
                              op::CollectivePermute(op::Slice(data_shard)));
  auto data_right_halo = AllOf(op::Shape("f32[2,4]{1,0}"),
                               op::CollectivePermute(op::Slice(data_shard)));
  auto required_data_start_on_padded =
      op::Multiply(required_source_shard_start, op::Constant());
  auto left_halo_size = op::Subtract(
      op::Add(op::Multiply(op::Reshape(), op::Constant()), op::Constant()),
      required_data_start_on_padded);
  auto data_with_halo =
      AllOf(op::Shape("f32[7,4]{1,0}"),
            op::DynamicSlice(
                AllOf(op::Shape("f32[8,4]{1,0}"),
                      op::Pad(op::Concatenate(data_left_halo, data_shard,
                                              data_right_halo),
                              op::Constant())),
                op::Subtract(op::Constant(), left_halo_size), op::Constant()));
  auto index_on_padded =
      op::Add(op::Iota(), op::Broadcast(required_data_start_on_padded));
  auto masked_data_with_halo = op::Select(
      op::And(op::Compare(index_on_padded, op::Broadcast(op::Constant())),
              op::Compare(index_on_padded, op::Broadcast(op::Constant()))),
      data_with_halo, op::Broadcast(op::Constant()));

  EXPECT_THAT(
      root, AllOf(op::DynamicSlice(op::SelectAndScatter(masked_data_with_halo,
                                                        masked_source_with_halo,
                                                        op::Constant()),
                                   left_halo_size, op::Constant()),
                  op::Shape("f32[3,4]{1,0}")));
  EXPECT_EQ(root->operand(0)->window().dimensions(0).padding_low(), 0);
  EXPECT_EQ(root->operand(0)->window().dimensions(0).padding_high(), 0);
}

TEST_F(SpmdPartitioningTest, ConvolutionLhsTiledRhsTiled) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[128,56,56,64] parameter(0)
  %lhs.copy = f32[128,56,56,64] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[128,56,56,256] parameter(1)
  %rhs.copy = f32[128,56,56,256] copy(%rhs), sharding={devices=[1,2,1,1]0,1}
  ROOT %conv = f32[1,1,64,256] convolution(%lhs.copy, %rhs.copy),
    window={size=56x56}, dim_labels=f01b_i01o->01bf, sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,28,56,64]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,28,56,256]"));

  EXPECT_THAT(root, AllOf(op::AllReduce(op::Convolution(lhs, rhs)),
                          op::Shape("f32[1,1,64,256]")));
}

TEST_F(SpmdPartitioningTest, ConvolutionLhsTiledRhsTiledWindowReversal) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[5,128,64] parameter(0), sharding={devices=[2,1,1]0,1}
  %rhs = f32[5,128,256] parameter(1), sharding={devices=[2,1,1]1,0}
  ROOT %conv = f32[1,64,256] convolution(%lhs, %rhs),
    window={size=5 rhs_reversal=1}, dim_labels=0fb_0io->0bf,
    sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto lhs_masked =
      AllOf(op::Shape("f32[3,128,64]"), op::Select(_, op::Parameter(0), _));
  const auto rhs_left_padded =
      op::Concatenate(op::CollectivePermute(op::Slice(op::Parameter(1))),
                      op::Slice(op::Parameter(1)));
  const auto rhs_masked =
      AllOf(op::Shape("f32[3,128,256]"), op::Select(_, rhs_left_padded, _));

  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              AllOf(op::AllReduce(op::Convolution(lhs_masked, rhs_masked)),
                    op::Shape("f32[1,64,256]")));
}

TEST_F(SpmdPartitioningTest, DotLhsTiledRhsTiledWithReshard) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[128,56,56,64] parameter(0)
  %lhs.copy = f32[128,56,56,64] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[128,56,56,256] parameter(1)
  %rhs.copy = f32[128,56,56,256] copy(%rhs), sharding={devices=[2,1,1,1]0,1}
  ROOT %conv = f32[1,1,64,256] convolution(%lhs.copy, %rhs.copy),
    window={size=56x56}, dim_labels=f01b_i01o->01bf, sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,28,56,64]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Reshape(), op::Constant(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[64,56,56,256]"));
  auto all_to_all =
      AllOf(op::AllToAll(op::Reshape(lhs)), op::Shape("f32[2,64,28,56,64]"));
  auto reshard = AllOf(op::Reshape(op::Transpose(all_to_all)));

  EXPECT_THAT(root, AllOf(op::AllReduce(op::Convolution(reshard, rhs)),
                          op::Shape("f32[1,1,64,256]")));
}

TEST_F(SpmdPartitioningTest, ConvolutionLhsTiledRhsTiledWithReshard) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[128,56,56,512] parameter(0)
  %lhs.copy = f32[128,56,56,512] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[128,28,28,64] parameter(1)
  %rhs.copy = f32[128,28,28,64] copy(%rhs), sharding={devices=[2,1,1,1]0,1}
  ROOT %conv = f32[1,1,512,64] convolution(%lhs.copy, %rhs.copy),
    window={size=28x28 pad=0_-1x0_-1 rhs_dilate=2x2},
    dim_labels=f01b_i01o->01bf, sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,28,56,512]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Reshape(), op::Constant(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[64,28,28,64]"));
  auto all_to_all =
      AllOf(op::AllToAll(op::Reshape(rhs)), op::Shape("f32[64,2,14,28,64]"));
  auto reshard = op::Reshape(op::Transpose(all_to_all));

  EXPECT_THAT(root,
              AllOf(op::AllReduce(op::Convolution(op::Slice(lhs), reshard)),
                    op::Shape("f32[1,1,512,64]")));
}

TEST_F(SpmdPartitioningTest,
       ConvolutionLhsTiledRhsTiled_UnevenDilatedRHSPartitioned) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[8,28,28,8] parameter(0)
  %lhs.copy = f32[8,28,28,8] copy(%lhs), sharding={devices=[1,4,1,1]0,1,2,3}
  %rhs = f32[8,14,14,64] parameter(1)
  %rhs.copy = f32[8,14,14,64] copy(%rhs), sharding={devices=[1,4,1,1]0,1,2,3}
  ROOT %conv = f32[1,1,8,64] convolution(%lhs.copy, %rhs.copy),
    window={size=14x14 pad=0_-1x0_-1 rhs_dilate=2x2},
    dim_labels=f01b_i01o->01bf, sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[8,7,28,8]"));
  const auto rhs = AllOf(op::Pad(op::Parameter(), op::Constant()),
                         op::Shape("f32[8,16,14,64]"));
  auto selected_rhs = AllOf(
      op::Select(op::Compare(),
                 op::Copy(op::DynamicSlice(rhs, op::Constant(), op::Reshape(),
                                           op::Constant(), op::Constant())),
                 op::Broadcast()),
      op::Shape("f32[8,4,14,64]"));
  auto right_halo =
      AllOf(op::CollectivePermute(op::Slice(lhs)), op::Shape("f32[8,2,28,8]"));
  auto selected_lhs =
      AllOf(op::DynamicSlice(
                op::Pad(op::Concatenate(lhs, right_halo), op::Constant()),
                op::Constant(), op::Reshape(), op::Constant(), op::Constant()),
            op::Shape("f32[8,7,28,8]"));
  EXPECT_THAT(root,
              AllOf(op::AllReduce(op::Convolution(selected_lhs, selected_rhs)),
                    op::Shape("f32[1,1,8,64]")));
}

TEST_F(SpmdPartitioningTest, ConvolutionLhsTiledRhsTiledWithPadding) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,28,28,128] parameter(0)
  %lhs.copy = f32[32,28,28,128] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[32,28,28,64] parameter(1)
  %rhs.copy = f32[32,28,28,64] copy(%rhs), sharding={devices=[1,2,1,1]0,1}
  ROOT %conv = f32[3,3,128,64] convolution(%lhs.copy, %rhs.copy),
    window={size=28x28 pad=1_1x1_1}, dim_labels=f01b_i01o->01bf, sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/2,
                           /*conv_halo_exchange_always_on_lhs=*/false));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[32,14,28,128]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[32,14,28,64]"));

  auto left_halo = AllOf(op::CollectivePermute(op::Slice(rhs)),
                         op::Shape("f32[32,1,28,64]"));
  auto right_halo = AllOf(op::CollectivePermute(op::Slice(rhs)),
                          op::Shape("f32[32,1,28,64]"));
  EXPECT_THAT(root,
              AllOf(op::AllReduce(op::Convolution(
                        lhs, AllOf(op::Concatenate(left_halo, rhs, right_halo),
                                   op::Shape("f32[32,16,28,64]")))),
                    op::Shape("f32[3,3,128,64]")));
}

TEST_F(SpmdPartitioningTest, ConvolutionLhsTiledRhsTiledWindowDilate) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[128,224,224,3] parameter(0)
  %lhs.copy = f32[128,224,224,3] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[128,112,112,64] parameter(1)
  %rhs.copy = f32[128,112,112,64] copy(%rhs), sharding={devices=[1,2,1,1]0,1}
  ROOT %conv = f32[7,7,3,64] convolution(%lhs.copy, %rhs.copy),
    window={size=112x112 pad=3_2x3_2 rhs_dilate=2x2}, dim_labels=f01b_i01o->01bf, sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/2,
                           /*conv_halo_exchange_always_on_lhs=*/false));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,112,224,3]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,56,112,64]"));

  auto left_halo = AllOf(op::CollectivePermute(op::Slice(rhs)),
                         op::Shape("f32[128,2,112,64]"));
  auto right_halo = AllOf(op::CollectivePermute(op::Slice(rhs)),
                          op::Shape("f32[128,2,112,64]"));
  EXPECT_THAT(root,
              AllOf(op::AllReduce(op::Convolution(
                        lhs, AllOf(op::Concatenate(left_halo, rhs, right_halo),
                                   op::Shape("f32[128,60,112,64]")))),
                    op::Shape("f32[7,7,3,64]")));
}

TEST_F(SpmdPartitioningTest,
       ConvolutionLhsTiledRhsTiledWindowDilateNegativeRhsPadding) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[128,56,56,256] parameter(0)
  %lhs.copy = f32[128,56,56,256] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[128,28,28,512] parameter(1)
  %rhs.copy = f32[128,28,28,512] copy(%rhs), sharding={devices=[1,2,1,1]0,1}
  ROOT %conv = f32[1,1,256,512] convolution(%lhs.copy, %rhs.copy),
    window={size=28x28 pad=0_-1x0_-1 rhs_dilate=2x2}, dim_labels=f01b_i01o->01bf, sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/2,
                           /*conv_halo_exchange_always_on_lhs=*/false));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,28,56,256]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,14,28,512]"));

  EXPECT_THAT(root, AllOf(op::AllReduce(op::Convolution(lhs, rhs)),
                          op::Shape("f32[1,1,256,512]")));
}

TEST_F(SpmdPartitioningTest, ConvolutionLhsTiledRhsTiledWindowDilateUneven) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[128,14,14,512] parameter(0)
  %lhs.copy = f32[128,14,14,512] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[128,7,7,512] parameter(1)
  %rhs.copy = f32[128,7,7,512] copy(%rhs), sharding={devices=[1,2,1,1]0,1}
  ROOT %conv = f32[3,3,512,512] convolution(%lhs.copy, %rhs.copy),
    window={size=7x7 pad=1_0x1_0 rhs_dilate=2x2}, dim_labels=f01b_i01o->01bf, sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/2,
                           /*conv_halo_exchange_always_on_lhs=*/false));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,7,14,512]"));
  const auto rhs = AllOf(
      op::Select(op::Compare(),
                 op::Copy(op::DynamicSlice(
                     op::Pad(op::Parameter(), op::Constant()), op::Constant(),
                     op::Reshape(), op::Constant(), op::Constant())),
                 op::Broadcast()),
      op::Shape("f32[128,4,7,512]"));

  auto left_halo = AllOf(op::CollectivePermute(op::Slice(rhs)),
                         op::Shape("f32[128,1,7,512]"));
  EXPECT_THAT(root,
              AllOf(op::AllReduce(op::Convolution(
                        AllOf(op::DynamicSlice(op::Pad(lhs, op::Constant()),
                                               op::Constant(), op::Subtract(),
                                               op::Constant(), op::Constant()),
                              op::Shape("f32[128,10,14,512]")),
                        AllOf(op::Concatenate(left_halo, rhs),
                              op::Shape("f32[128,5,7,512]")))),
                    op::Shape("f32[3,3,512,512]")));
}

TEST_F(SpmdPartitioningTest, ConvolutionLhsTiledRhsTiledWithPadding_HaloOnLhs) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,28,28,128] parameter(0)
  %lhs.copy = f32[32,28,28,128] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[32,28,28,64] parameter(1)
  %rhs.copy = f32[32,28,28,64] copy(%rhs), sharding={devices=[1,2,1,1]0,1}
  ROOT %conv = f32[3,3,128,64] convolution(%lhs.copy, %rhs.copy),
    window={size=28x28 pad=1_1x1_1}, dim_labels=f01b_i01o->01bf, sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[32,14,28,128]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[32,14,28,64]"));

  auto left_halo = AllOf(op::CollectivePermute(op::Slice(lhs)),
                         op::Shape("f32[32,1,28,128]"));
  auto right_halo = AllOf(op::CollectivePermute(op::Slice(lhs)),
                          op::Shape("f32[32,1,28,128]"));
  EXPECT_THAT(root, AllOf(op::AllReduce(op::Convolution(
                              AllOf(op::Concatenate(left_halo, lhs, right_halo),
                                    op::Shape("f32[32,16,28,128]")),
                              rhs)),
                          op::Shape("f32[3,3,128,64]")));
}

TEST_F(SpmdPartitioningTest,
       ConvolutionLhsTiledRhsTiledWindowDilate_HaloOnLhs) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[128,224,224,3] parameter(0)
  %lhs.copy = f32[128,224,224,3] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[128,112,112,64] parameter(1)
  %rhs.copy = f32[128,112,112,64] copy(%rhs), sharding={devices=[1,2,1,1]0,1}
  ROOT %conv = f32[7,7,3,64] convolution(%lhs.copy, %rhs.copy),
    window={size=112x112 pad=3_2x3_2 rhs_dilate=2x2}, dim_labels=f01b_i01o->01bf, sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,112,224,3]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,56,112,64]"));

  auto left_halo = AllOf(op::CollectivePermute(op::Slice(lhs)),
                         op::Shape("f32[128,3,224,3]"));
  auto right_halo = AllOf(op::CollectivePermute(op::Slice(lhs)),
                          op::Shape("f32[128,2,224,3]"));
  EXPECT_THAT(root, AllOf(op::AllReduce(op::Convolution(
                              AllOf(op::Concatenate(left_halo, lhs, right_halo),
                                    op::Shape("f32[128,117,224,3]")),
                              rhs)),
                          op::Shape("f32[7,7,3,64]")));
}

TEST_F(SpmdPartitioningTest,
       ConvolutionLhsTiledRhsTiledWindowDilateNegativeRhsPadding_HaloOnLhs) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[128,56,56,256] parameter(0)
  %lhs.copy = f32[128,56,56,256] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[128,28,28,512] parameter(1)
  %rhs.copy = f32[128,28,28,512] copy(%rhs), sharding={devices=[1,2,1,1]0,1}
  ROOT %conv = f32[1,1,256,512] convolution(%lhs.copy, %rhs.copy),
    window={size=28x28 pad=0_-1x0_-1 rhs_dilate=2x2}, dim_labels=f01b_i01o->01bf, sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,28,56,256]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,14,28,512]"));

  EXPECT_THAT(root, AllOf(op::AllReduce(op::Convolution(op::Slice(lhs), rhs)),
                          op::Shape("f32[1,1,256,512]")));
}

TEST_F(SpmdPartitioningTest,
       ConvolutionLhsTiledRhsTiledWindowDilateUneven_HaloOnLhs) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[128,14,14,512] parameter(0)
  %lhs.copy = f32[128,14,14,512] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[128,7,7,512] parameter(1)
  %rhs.copy = f32[128,7,7,512] copy(%rhs), sharding={devices=[1,2,1,1]0,1}
  ROOT %conv = f32[3,3,512,512] convolution(%lhs.copy, %rhs.copy),
    window={size=7x7 pad=1_0x1_0 rhs_dilate=2x2}, dim_labels=f01b_i01o->01bf, sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,7,14,512]"));
  const auto rhs = AllOf(
      op::Select(op::Compare(),
                 op::Copy(op::DynamicSlice(
                     op::Pad(op::Parameter(), op::Constant()), op::Constant(),
                     op::Reshape(), op::Constant(), op::Constant())),
                 op::Broadcast()),
      op::Shape("f32[128,4,7,512]"));

  auto right_halo = AllOf(op::CollectivePermute(op::Slice(lhs)),
                          op::Shape("f32[128,1,14,512]"));
  EXPECT_THAT(
      root, AllOf(op::AllReduce(op::Convolution(
                      AllOf(op::DynamicSlice(
                                AllOf(op::Pad(op::Concatenate(lhs, right_halo),
                                              op::Constant()),
                                      op::Shape("f32[128,10,14,512]")),
                                op::Constant(), op::Reshape(), op::Constant(),
                                op::Constant()),
                            op::Shape("f32[128,9,14,512]")),
                      rhs)),
                  op::Shape("f32[3,3,512,512]")));
}

TEST_F(SpmdPartitioningTest, ConcatenateAlongNonPartitionedDimension) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[14,257] parameter(0)
  %param0.copy = f32[14,257] copy(%param0), sharding={devices=[2,1]0,1}
  %param1 = f32[14,116] parameter(1)
  %param1.copy = f32[14,116] copy(%param1), sharding={devices=[2,1]0,1}
  ROOT %concatenate = f32[14,373] concatenate(%param0.copy, %param1.copy),
    dimensions={1}, sharding={devices=[2,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(op::Copy(op::DynamicSlice(op::Parameter(), op::Reshape(),
                                                op::Constant())),
                      op::Shape("f32[7,257]"));
  auto param1 = AllOf(op::Copy(op::DynamicSlice(op::Parameter(), op::Reshape(),
                                                op::Constant())),
                      op::Shape("f32[7,116]"));
  EXPECT_THAT(root,
              AllOf(op::Concatenate(param0, param1), op::Shape("f32[7,373]")));
}

TEST_F(SpmdPartitioningTest, ConcatenateAlongPartitionedDimension) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[14,257] parameter(0)
  %param0.copy = f32[14,257] copy(%param0), sharding={devices=[1,2]0,1}
  %param1 = f32[14,116] parameter(1)
  %param1.copy = f32[14,116] copy(%param1), sharding={devices=[1,2]0,1}
  ROOT %concatenate = f32[14,373] concatenate(%param0.copy, %param1.copy),
    dimensions={1}, sharding={devices=[1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(), op::Constant()),
                                      op::Constant(), op::Reshape())),
            op::Shape("f32[14,129]"));
  auto param1 = AllOf(op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(),
                                                op::Reshape())),
                      op::Shape("f32[14,58]"));
  EXPECT_THAT(root, AllOf(op::DynamicSlice(
                              AllOf(op::AllReduce(op::DynamicUpdateSlice(
                                        op::DynamicUpdateSlice(
                                            op::Broadcast(), param0,
                                            op::Constant(), op::Multiply()),
                                        param1, op::Constant(), op::Add())),
                                    op::Shape("f32[14,374]")),
                              op::Constant(), op::Multiply()),
                          op::Shape("f32[14,187]")));
}

TEST_F(SpmdPartitioningTest, ConcatenateAlongBothDimensions) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[14,257] parameter(0), sharding={devices=[2,2]0,1,2,3}
  %param1 = f32[14,116] parameter(1), sharding={devices=[2,2]0,1,2,3}
  ROOT %concatenate = f32[14,373] concatenate(%param0, %param1),
    dimensions={1}, sharding={devices=[2,2]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(op::Parameter(0), op::Shape("f32[7,129]"));
  auto param1 = AllOf(op::Parameter(1), op::Shape("f32[7,58]"));
  EXPECT_THAT(root, AllOf(op::DynamicSlice(
                              AllOf(op::AllReduce(op::DynamicUpdateSlice(
                                        op::DynamicUpdateSlice(
                                            op::Broadcast(), param0,
                                            op::Constant(), op::Multiply()),
                                        param1, op::Constant(), op::Add())),
                                    op::Shape("f32[7,374]")),
                              op::Constant(), op::Multiply()),
                          op::Shape("f32[7,187]")));
}

TEST_F(SpmdPartitioningTest, PadAlongNonPartitionedDimension) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[128,14,257] parameter(0), sharding={devices=[1,1,2]0,1}
  %const = f32[] constant(0)
  ROOT %pad = f32[128,17,257] pad(%param0, %const), padding=0_0x1_2x0_0,
    sharding={devices=[1,1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(op::Parameter(), op::Shape("f32[128,14,129]"));
  EXPECT_THAT(root, AllOf(op::Pad(param0, op::Constant()),
                          op::Shape("f32[128,17,129]")));
}

TEST_F(SpmdPartitioningTest, PadAlongNonPartitionedDimensionReshard) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[128,14,257] parameter(0), sharding={replicated}
  %const = f32[] constant(0)
  ROOT %pad = f32[128,17,257] pad(%param0, %const), padding=0_0x1_2x0_0,
    sharding={devices=[1,1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(op::Parameter(), op::Shape("f32[128,14,257]"));
  auto operand = op::DynamicSlice(op::Pad(param0, _), op::Constant(),
                                  op::Constant(), op::Multiply());
  EXPECT_THAT(root, AllOf(op::Pad(operand, op::Constant()),
                          op::Shape("f32[128,17,129]")));
}

TEST_F(SpmdPartitioningTest, PadAlongPartitionedDimension) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[14,257] parameter(0), sharding={devices=[1,2]0,1}
  %const = f32[] constant(0)
  ROOT %pad = f32[14,259] pad(%param0, %const), padding=0_0x0_2,
    sharding={devices=[1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(op::Parameter(), op::Shape("f32[14,129]"));
  auto after_halo_exchange =
      AllOf(op::Shape("f32[14,130]"),
            op::Concatenate(param0, op::CollectivePermute(op::Slice(param0))));
  auto pad = AllOf(op::Shape("f32[14,131]"),
                   op::Pad(after_halo_exchange, op::Constant()));
  EXPECT_THAT(root, op::Select(_, op::DynamicSlice(pad, op::Constant(), _), _));
}

TEST_F(SpmdPartitioningTest, PadAlongPartitionedDimensionWithInteriorPadding) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[7] parameter(0), sharding={devices=[2]0,1}
  %param1 = f32[] parameter(1), sharding={replicated}
  ROOT %pad = f32[22] pad(%param0, %param1), padding=2_1_2,
    sharding={devices=[2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();

  auto param0 = AllOf(op::Parameter(), op::Shape("f32[4]"));
  auto after_halo_exchange = AllOf(
      op::Shape("f32[4]"),
      op::DynamicSlice(
          AllOf(op::Shape("f32[5]"),
                op::Pad(AllOf(op::Shape("f32[4]"),
                              op::Concatenate(
                                  op::CollectivePermute(op::Slice(param0)),
                                  op::Slice(param0))),
                        op::Parameter(1))),
          _));
  auto pad = op::Pad(after_halo_exchange, op::Parameter(1));
  EXPECT_THAT(root, op::DynamicSlice(pad, _));
}

TEST_F(SpmdPartitioningTest, PartialReplicatePad) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[11,7] parameter(0),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}
  %param1 = f32[] parameter(1), sharding={replicated}
  ROOT %pad = f32[27,22] pad(%param0, %param1), padding=2_4_1x2_1_2,
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();

  auto param0 = AllOf(op::Parameter(), op::Shape("f32[11,4]"));
  auto after_halo_exchange = AllOf(
      op::Shape("f32[11,4]"),
      op::DynamicSlice(
          AllOf(op::Shape("f32[11,5]"),
                op::Pad(AllOf(op::Shape("f32[11,4]"),
                              op::Concatenate(
                                  op::CollectivePermute(op::Slice(param0)),
                                  op::Slice(param0))),
                        op::Parameter(1))),
          op::Constant(), _));
  auto pad = op::Pad(after_halo_exchange, op::Parameter(1));
  EXPECT_THAT(root, AllOf(op::DynamicSlice(pad, op::Constant(), _),
                          op::Shape("f32[27,11]")));
}

TEST_F(SpmdPartitioningTest, SliceAlongNonPartitionedDimension) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[128,14,257] parameter(0)
  %param0.copy = f32[128,14,257] copy(%param0), sharding={devices=[1,1,2]0,1}
  ROOT %slice = f32[128,11,257] slice(%param0.copy),
    slice={[0:128:1], [2:13:1], [0:257:1]}, sharding={devices=[1,1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(
      op::Copy(op::DynamicSlice(op::Pad(op::Parameter(), op::Constant()),
                                op::Constant(), op::Constant(), op::Reshape())),
      op::Shape("f32[128,14,129]"));
  EXPECT_THAT(root, AllOf(op::Slice(param0), op::Shape("f32[128,11,129]")));
}

TEST_F(SpmdPartitioningTest, SliceAlongPartitionedDimension) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[128,14,257] parameter(0), sharding={devices=[1,1,2]0,1}
  ROOT %slice = f32[63,14,251] slice(%param0),
    slice={[2:128:2], [0:14:1], [5:256:1]}, sharding={devices=[1,1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(op::Parameter(0), op::Shape("f32[128,14,129]"));
  EXPECT_THAT(
      root,
      AllOf(op::Slice(AllOf(
                op::DynamicSlice(
                    AllOf(op::Concatenate(
                              op::Slice(param0),
                              AllOf(op::CollectivePermute(op::Slice(param0)),
                                    op::Shape("f32[128,14,2]"))),
                          op::Shape("f32[128,14,129]")),
                    op::Constant(), op::Constant(), op::Add()),
                op::Shape("f32[128,14,126]"))),
            op::Shape("f32[63,14,126]")));
}

TEST_F(SpmdPartitioningTest, SliceAlongPartitionedDimension2) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[4] parameter(0), sharding={devices=[4]0,1,2,3}
  ROOT %slice = f32[1] slice(%param0),
    slice={[3:4]}, sharding={devices=[4]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(op::Parameter(0), op::Shape("f32[1]"));
  EXPECT_THAT(root, AllOf(op::Copy(op::CollectivePermute(param0)),
                          op::Shape("f32[1]")));
}

TEST_F(SpmdPartitioningTest, MergedPadThenSliceShiftRight) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[4] parameter(0), sharding={devices=[4]0,1,2,3}
  %init = f32[] constant(2.0)
  %pad = f32[5] pad(%param0, %init), padding=1_0, sharding={devices=[4]0,1,2,3}
  %copy = f32[5] copy(%pad), sharding={devices=[4]0,1,2,3}
  %copy.1 = f32[5] copy(%copy), sharding={devices=[4]0,1,2,3}
  ROOT %slice = f32[4] slice(%copy.1), slice={[0:4]}, sharding={devices=[4]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(op::Parameter(0), op::Shape("f32[1]"));
  EXPECT_THAT(root, AllOf(op::Select(_, op::CollectivePermute(param0), _),
                          op::Shape("f32[1]")));
}

// Same as above except that it uses zero padding, so there is no need for
// masking.
TEST_F(SpmdPartitioningTest, MergedPadThenSliceShiftRightNoMasking) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[4] parameter(0), sharding={devices=[4]0,1,2,3}
  %init = f32[] constant(0)
  %pad = f32[5] pad(%param0, %init), padding=1_0, sharding={devices=[4]0,1,2,3}
  %copy = f32[5] copy(%pad), sharding={devices=[4]0,1,2,3}
  %copy.1 = f32[5] copy(%copy), sharding={devices=[4]0,1,2,3}
  ROOT %slice = f32[4] slice(%copy.1), slice={[0:4]}, sharding={devices=[4]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(op::Parameter(0), op::Shape("f32[1]"));
  EXPECT_THAT(root, AllOf(op::CollectivePermute(param0), op::Shape("f32[1]")));
}

TEST_F(SpmdPartitioningTest, MergedSliceThenConcatRotateRight) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[12] parameter(0), sharding={devices=[4]0,1,2,3}
  %slice0 = f32[2] slice(%param0), slice={[10:12]}, sharding={devices=[4]0,1,2,3}
  %slice1 = f32[10] slice(%param0), slice={[0:10]}, sharding={devices=[4]0,1,2,3}
  ROOT %concat = f32[12] concatenate(%slice0, %slice1), dimensions={0},
    sharding={devices=[4]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(op::Parameter(0), op::Shape("f32[3]"));
  auto rotate = op::Concatenate(op::CollectivePermute(op::Slice(param0)),
                                op::Slice(param0));
  EXPECT_THAT(root, AllOf(rotate, op::Shape("f32[3]")));
}

TEST_F(SpmdPartitioningTest,
       MergedSliceThenConcatRotateRightWithAlignedPadding) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[6] parameter(0), sharding={devices=[4]0,1,2,3}
  %slice0 = f32[2] slice(%param0), slice={[4:6]}, sharding={devices=[4]0,1,2,3}
  %slice1 = f32[4] slice(%param0), slice={[0:4]}, sharding={devices=[4]0,1,2,3}
  ROOT %concat = f32[6] concatenate(%slice0, %slice1), dimensions={0},
    sharding={devices=[4]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(op::Parameter(0), op::Shape("f32[2]"));
  EXPECT_THAT(root, op::CollectivePermute(param0));
}

TEST_F(SpmdPartitioningTest,
       MergedSliceThenConcatRotateRightWithUnalignedPadding) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[10] parameter(0), sharding={devices=[4]0,1,2,3}
  %slice0 = f32[6] slice(%param0), slice={[4:10]}, sharding={devices=[4]0,1,2,3}
  %slice1 = f32[4] slice(%param0), slice={[0:4]}, sharding={devices=[4]0,1,2,3}
  ROOT %concat = f32[10] concatenate(%slice0, %slice1), dimensions={0},
    sharding={devices=[4]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(op::Parameter(0), op::Shape("f32[3]"));
  auto rotate0 = op::CollectivePermute(param0);
  auto rotate1 = op::Concatenate(op::CollectivePermute(op::Slice(param0)),
                                 op::CollectivePermute(op::Slice(param0)));
  EXPECT_THAT(root,
              AllOf(op::Select(_, rotate1, rotate0), op::Shape("f32[3]")));
}

TEST_F(SpmdPartitioningTest,
       PartialReplicateSliceAlongNonPartitionedDimension) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[128,14,257] parameter(0), sharding={devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}
  ROOT %slice = f32[128,11,257] slice(%param0),
    slice={[0:128:1], [2:13:1], [0:257:1]}, sharding={devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(op::Parameter(), op::Shape("f32[128,14,129]"));
  EXPECT_THAT(root, AllOf(op::Slice(param0), op::Shape("f32[128,11,129]")));
}

TEST_F(SpmdPartitioningTest, PartialReplicateSliceAlongPartitionedDimension) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[128,14,257] parameter(0), sharding={devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}
  ROOT %slice = f32[63,14,251] slice(%param0),
    slice={[2:128:2], [0:14:1], [5:256:1]}, sharding={devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(op::Parameter(), op::Shape("f32[128,14,129]"));
  EXPECT_THAT(
      root,
      AllOf(
          op::Slice(AllOf(
              op::DynamicSlice(
                  AllOf(op::Concatenate(
                            op::Slice(param0),
                            AllOf(op::CollectivePermute(op::Slice(param0)),
                                  op::Shape("f32[128,14,2]"))),
                        op::Shape("f32[128,14,129]")),
                  op::Constant(), op::Constant(),
                  op::Add(op::Multiply(op::Reshape(op::DynamicSlice(
                                           op::Constant(), op::PartitionId())),
                                       op::Constant()),
                          op::Constant())),
              op::Shape("f32[128,14,126]"))),
          op::Shape("f32[63,14,126]")));
}

TEST_F(SpmdPartitioningTest, DeviceMaximalTupleSort) {
  absl::string_view hlo_string = R"(
HloModule module

ge {
  p.0 = f32[] parameter(0)
  p.1 = f32[] parameter(1)
  p.2 = s32[] parameter(2)
  p.3 = s32[] parameter(3)
  ROOT compare = pred[] compare(p.0, p.1), direction=GT
}

ENTRY %main {
  %p.0 = f32[3]{0} parameter(0), sharding={maximal device=0}
  %iota = s32[3]{0} iota(), iota_dimension=0, sharding={maximal device=0}
  ROOT %sort = (f32[3]{0}, s32[3]{0}) sort(p.0, iota), dimensions={0},
    to_apply=ge, sharding={{maximal device=0}, {maximal device=0}}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Sort(op::Parameter(0), op::Iota()),
                          op::Shape("(f32[3], s32[3])")));
}

TEST_F(SpmdPartitioningTest, SortAlongNonPartitionedDimension) {
  absl::string_view hlo_string = R"(
HloModule module

ge {
  p.0.lhs.1247 = f32[]{:T(256)} parameter(0), sharding={replicated}
  bitcast-convert = s32[]{:T(256)} bitcast-convert(p.0.lhs.1247), sharding={replicated}
  constant = s32[]{:T(256)} constant(0), sharding={replicated}
  compare = pred[]{:T(256)} compare(bitcast-convert, constant), direction=LT, sharding={replicated}
  constant.1 = u32[]{:T(256)} constant(2147483647), sharding={replicated}
  bitcast-convert.1 = u32[]{:T(256)} bitcast-convert(p.0.lhs.1247), sharding={replicated}
  subtract = u32[]{:T(256)} subtract(constant.1, bitcast-convert.1), sharding={replicated}
  bitcast-convert.2 = s32[]{:T(256)} bitcast-convert(subtract), sharding={replicated}
  select = s32[]{:T(256)} select(compare, bitcast-convert.2, bitcast-convert), sharding={replicated}
  p.0.rhs.1248 = f32[]{:T(256)} parameter(1), sharding={replicated}
  bitcast-convert.3 = s32[]{:T(256)} bitcast-convert(p.0.rhs.1248), sharding={replicated}
  compare.1 = pred[]{:T(256)} compare(bitcast-convert.3, constant), direction=LT, sharding={replicated}
  bitcast-convert.4 = u32[]{:T(256)} bitcast-convert(p.0.rhs.1248), sharding={replicated}
  subtract.1 = u32[]{:T(256)} subtract(constant.1, bitcast-convert.4), sharding={replicated}
  bitcast-convert.5 = s32[]{:T(256)} bitcast-convert(subtract.1), sharding={replicated}
  select.1 = s32[]{:T(256)} select(compare.1, bitcast-convert.5, bitcast-convert.3), sharding={replicated}
  compare.2 = pred[]{:T(256)} compare(select, select.1), direction=GT, sharding={replicated}
  compare.258 = pred[]{:T(256)} compare(select.1, select), direction=GT, sharding={replicated}
  compare.259 = pred[]{:T(256)} compare(compare.2, compare.258), direction=EQ, sharding={replicated}
  p.1.lhs.1249 = s32[]{:T(256)} parameter(2), sharding={replicated}
  p.1.rhs.1250 = s32[]{:T(256)} parameter(3), sharding={replicated}
  compare.260 = pred[]{:T(256)} compare(p.1.lhs.1249, p.1.rhs.1250), direction=LT, sharding={replicated}
  ROOT select.86 = pred[]{:T(256)} select(compare.259, compare.260, compare.2), sharding={replicated}
}

ENTRY entry {
  %param0 = f32[128,14,257] parameter(0)
  %param0.copy = f32[128,14,257] copy(%param0), sharding={devices=[1,2,1]0,1}
  %param1 = s32[128,14,257] parameter(1)
  %param1.copy = s32[128,14,257] copy(%param1), sharding={devices=[1,2,1]0,1}
  ROOT %sort.6 = (f32[128,14,257]{2,1,0:T(8,128)}, s32[128,14,257]{2,1,0:T(8,128)})
    sort(%param0.copy, %param1.copy), dimensions={2}, is_stable=true,
    to_apply=%ge, sharding={{devices=[1,2,1]0,1},{devices=[1,2,1]0,1}}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 =
      AllOf(op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(),
                                      op::Reshape(), op::Constant())),
            op::Shape("f32[128,7,257]"));
  auto param1 =
      AllOf(op::Copy(op::DynamicSlice(op::Parameter(1), op::Constant(),
                                      op::Reshape(), op::Constant())),
            op::Shape("s32[128,7,257]"));
  EXPECT_THAT(root, AllOf(op::Sort(param0, param1),
                          op::Shape("(f32[128,7,257], s32[128,7,257])")));
}

TEST_F(SpmdPartitioningTest, PartitionCustomCall) {
  absl::string_view hlo_string = R"(
HloModule cluster_2013453984438090939__.47

ENTRY %cluster_2013453984438090939__.47
  (arg_tuple.1: ()) -> (bf16[2,2000], s32[2,2000]) {
  %arg_tuple.1 = bf16[2,209664] parameter(0)
  %copy.arg_tuple.1 = bf16[2,209664] copy(%arg_tuple.1), sharding={devices=[1,2]0,1}
  %custom-call = (bf16[2,2000]{1,0}, s32[2,2000]{1,0})
    custom-call(bf16[2,209664]{1,0} %copy.arg_tuple.1), custom_call_target="TopK"
  %get-tuple-element = bf16[2,2000]{1,0}
    get-tuple-element((bf16[2,2000]{1,0}, s32[2,2000]{1,0}) %custom-call),
    index=0, sharding={replicated}
  %get-tuple-element.1 = s32[2,2000]{1,0} get-tuple-element((bf16[2,2000]{1,0},
    s32[2,2000]{1,0}) %custom-call), index=1, sharding={replicated}
  ROOT %tuple.46 = (bf16[2,2000]{1,0}, s32[2,2000]{1,0})
    tuple(bf16[2,2000]{1,0} %get-tuple-element, s32[2,2000]{1,0}
    %get-tuple-element.1), sharding={{replicated}, {replicated}},
    metadata={op_name="XLA_Retvals"}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  auto custom_call = FindInstruction(module.get(), "custom-call.1");
  EXPECT_EQ(custom_call->operand(0)->shape().dimensions(1), 104832);
  auto sort = FindInstruction(module.get(), "sort");
  EXPECT_EQ(sort->operand(0)->shape().dimensions(1), 4000);
  EXPECT_EQ(sort->operand(1)->shape().dimensions(1), 4000);
}

TEST_F(SpmdPartitioningTest, PartitionCustomCall_BatchPartitionedDims) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[8,32128] parameter(0)
  %copy.0 = f32[8,32128] copy(%param0),
    sharding={devices=[8,1]0,1,2,3,4,5,6,7}
  %custom-call = (f32[8,2]{1,0}, s32[8,2]{1,0})
    custom-call(%copy.0), custom_call_target="TopK"
  %get-tuple-element = f32[8,2]{1,0}
    get-tuple-element((f32[8,2]{1,0}, s32[8,2]{1,0}) %custom-call), index=0,
    sharding={devices=[8,1]0,1,2,3,4,5,6,7}
  %get-tuple-element.1 = s32[8,2]{1,0}
    get-tuple-element((f32[8,2]{1,0}, s32[8,2]{1,0}) %custom-call), index=1,
    sharding={devices=[8,1]0,1,2,3,4,5,6,7}
  ROOT %tuple = (f32[8,2]{1,0}, s32[8,2]{1,0})
    tuple(%get-tuple-element, %get-tuple-element.1),
    sharding={{replicated}, {replicated}}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  LOG(ERROR) << module->ToString();
  auto custom_call = FindInstruction(module.get(), "custom-call.1");
  EXPECT_EQ(custom_call->operand(0)->shape().dimensions(1), 32128);
  auto sort = FindInstruction(module.get(), "sort");
  EXPECT_EQ(sort->operand(0)->shape().dimensions(0), 1);
  EXPECT_EQ(sort->operand(0)->shape().dimensions(1), 2);
  EXPECT_EQ(sort->operand(1)->shape().dimensions(0), 1);
  EXPECT_EQ(sort->operand(1)->shape().dimensions(1), 2);
}

TEST_F(SpmdPartitioningTest, PartitionCustomCall_TwoPartitionedDims) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[8,32128] parameter(0)
  %copy.0 = f32[8,32128] copy(%param0),
    sharding={devices=[4,2]0,1,2,3,4,5,6,7}
  %custom-call = (f32[8,2]{1,0}, s32[8,2]{1,0})
    custom-call(%copy.0), custom_call_target="TopK"
  %get-tuple-element = f32[8,2]{1,0}
    get-tuple-element((f32[8,2]{1,0}, s32[8,2]{1,0}) %custom-call), index=0,
    sharding={devices=[4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %get-tuple-element.1 = s32[8,2]{1,0}
    get-tuple-element((f32[8,2]{1,0}, s32[8,2]{1,0}) %custom-call), index=1,
    sharding={devices=[4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %tuple = (f32[8,2]{1,0}, s32[8,2]{1,0})
    tuple(%get-tuple-element, %get-tuple-element.1),
    sharding={{replicated}, {replicated}}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  auto custom_call = FindInstruction(module.get(), "custom-call.1");
  EXPECT_EQ(custom_call->operand(0)->shape().dimensions(1), 16064);
  auto sort = FindInstruction(module.get(), "sort");
  EXPECT_EQ(sort->operand(0)->shape().dimensions(0), 2);
  EXPECT_EQ(sort->operand(0)->shape().dimensions(1), 4);
  EXPECT_EQ(sort->operand(1)->shape().dimensions(0), 2);
  EXPECT_EQ(sort->operand(1)->shape().dimensions(1), 4);
}

TEST_F(SpmdPartitioningTest, PartitionSortInTopK) {
  absl::string_view hlo_string = R"(
HloModule module

%compare-greater-than.8 (p.0.lhs.9: bf16[], p.0.rhs.10: bf16[], p.1.lhs.11:
   s32[], p.1.rhs.12: s32[]) -> pred[] {
  %p.1.lhs.11 = s32[] parameter(2)
  %p.1.rhs.12 = s32[] parameter(3)
  %p.0.lhs.9 = bf16[] parameter(0)
  %convert.13 = f32[] convert(bf16[] %p.0.lhs.9)
  %bitcast-convert.16 = s32[] bitcast-convert(f32[] %convert.13)
  %constant.20 = s32[] constant(0)
  %compare.21 = pred[] compare(s32[] %bitcast-convert.16, s32[] %constant.20),
    direction=LT
  %constant.15 = u32[] constant(2147483647)
  %bitcast-convert.17 = u32[] bitcast-convert(f32[] %convert.13)
  %subtract.18 = u32[] subtract(u32[] %constant.15, u32[] %bitcast-convert.17)
  %bitcast-convert.19 = s32[] bitcast-convert(u32[] %subtract.18)
  %select.22 = s32[] select(pred[] %compare.21, s32[] %bitcast-convert.19, s32[]
    %bitcast-convert.16)
  %p.0.rhs.10 = bf16[] parameter(1)
  %convert.14 = f32[] convert(bf16[] %p.0.rhs.10)
  %bitcast-convert.24 = s32[] bitcast-convert(f32[] %convert.14)
  %constant.28 = s32[] constant(0)
  %compare.29 = pred[] compare(s32[] %bitcast-convert.24, s32[] %constant.28),
    direction=LT
  %constant.23 = u32[] constant(2147483647)
  %bitcast-convert.25 = u32[] bitcast-convert(f32[] %convert.14)
  %subtract.26 = u32[] subtract(u32[] %constant.23, u32[] %bitcast-convert.25)
  %bitcast-convert.27 = s32[] bitcast-convert(u32[] %subtract.26)
  %select.30 = s32[] select(pred[] %compare.29, s32[] %bitcast-convert.27, s32[]
    %bitcast-convert.24)
  ROOT %compare.31 = pred[] compare(s32[] %select.22, s32[] %select.30),
    direction=GT
}

ENTRY entry
  (arg_tuple.1: ()) -> (bf16[2,2000], s32[2,2000]) {
  %arg_tuple.1 = bf16[2,209664] parameter(0)
  %copy.arg_tuple.1 = bf16[2,209664] copy(%arg_tuple.1), sharding={devices=[1,2]0,1}
  %iota.7 = s32[2,209664] iota(), iota_dimension=1,
    metadata={op_type="TopKV2" op_name="TopKV2"}
  %sort.32 = (bf16[2,209664], s32[2,209664])
    sort(bf16[2,209664] %copy.arg_tuple.1, s32[2,209664] %iota.7),
    dimensions={1}, is_stable=true, to_apply=%compare-greater-than.8,
    metadata={op_type="TopKV2" op_name="TopKV2"}
  %get-tuple-element.33 = bf16[2,209664]
    get-tuple-element((bf16[2,209664], s32[2,209664]) %sort.32),
    index=0, metadata={op_type="TopKV2" op_name="TopKV2"}
  %slice.34 = bf16[2,2000] slice(bf16[2,209664]
    %get-tuple-element.33), slice={[0:2], [0:2000]},
    metadata={op_type="TopKV2" op_name="TopKV2"}
  %get-tuple-element.35 = s32[2,209664]
    get-tuple-element((bf16[2,209664], s32[2,209664]) %sort.32),
    index=1, metadata={op_type="TopKV2" op_name="TopKV2"}
  %slice.36 = s32[2,2000] slice(s32[2,209664]
    %get-tuple-element.35), slice={[0:2], [0:2000]},
    metadata={op_type="TopKV2" op_name="TopKV2"}
  ROOT %tuple.46 = (bf16[2,2000], s32[2,2000])
    tuple(bf16[2,2000] %slice.34, s32[2,2000]
    %slice.36), sharding={{replicated}, {replicated}},
    metadata={op_name="XLA_Retvals"}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  auto sort = FindInstruction(module.get(), "sort.0");
  EXPECT_EQ(sort->operand(0)->shape().dimensions(1), 104832);
  EXPECT_EQ(sort->operand(1)->shape().dimensions(1), 104832);
  auto final_sort = FindInstruction(module.get(), "sort.1");
  EXPECT_EQ(final_sort->operand(0)->shape().dimensions(1), 4000);
  EXPECT_EQ(final_sort->operand(1)->shape().dimensions(1), 4000);
}

TEST_F(SpmdPartitioningTest, PartitionSortInTopKWhenComparisonWithSelect) {
  absl::string_view hlo_string = R"(
HloModule module

%compare-greater-than.8 (p.0.lhs.2566: bf16[],
  p.0.rhs.2567: bf16[], p.1.lhs.2586: s32[], p.1.rhs.2587: s32[]) -> pred[] {
  %p.0.lhs.2566 = bf16[] parameter(0)
  %convert.164 = f32[] convert(bf16[] %p.0.lhs.2566)
  %bitcast-convert.48 = s32[] bitcast-convert(f32[] %convert.164)
  %constant.285 = s32[] constant(0)
  %compare.84 = pred[] compare(s32[] %bitcast-convert.48, s32[] %constant.285),
    direction=LT
  %constant.286 = u32[] constant(2147483647)
  %bitcast-convert.49 = u32[] bitcast-convert(f32[] %convert.164)
  %subtract.84 = u32[] subtract(u32[] %constant.286, u32[] %bitcast-convert.49)
  %bitcast-convert.50 = s32[] bitcast-convert(u32[] %subtract.84)
  %select.40 = s32[] select(pred[] %compare.84, s32[] %bitcast-convert.50,
    s32[] %bitcast-convert.48)
  %p.0.rhs.2567 = bf16[] parameter(1)
  %convert.165 = f32[] convert(bf16[] %p.0.rhs.2567)
  %bitcast-convert.51 = s32[] bitcast-convert(f32[] %convert.165)
  %compare.85 = pred[] compare(s32[] %bitcast-convert.51, s32[] %constant.285),
    direction=LT
  %bitcast-convert.52 = u32[] bitcast-convert(f32[] %convert.165)
  %subtract.85 = u32[] subtract(u32[] %constant.286, u32[] %bitcast-convert.52)
  %bitcast-convert.53 = s32[] bitcast-convert(u32[] %subtract.85)
  %select.41 = s32[] select(pred[] %compare.85, s32[] %bitcast-convert.53,
    s32[] %bitcast-convert.51)
  %compare.86 = pred[] compare(s32[] %select.40, s32[] %select.41), direction=GT
  %compare.1645 = pred[] compare(s32[] %select.41, s32[] %select.40), direction=GT
  %compare.1646 = pred[] compare(pred[] %compare.86, pred[] %compare.1645),
    direction=EQ
  %p.1.lhs.2586 = s32[] parameter(2)
  %p.1.rhs.2587 = s32[] parameter(3)
  %compare.1647 = pred[] compare(s32[] %p.1.lhs.2586, s32[] %p.1.rhs.2587),
    direction=LT
  ROOT %select.1054 = pred[] select(pred[] %compare.1646, pred[] %compare.1647,
    pred[] %compare.86)
}

ENTRY entry
  (arg_tuple.1: ()) -> (bf16[2,2000], s32[2,2000]) {
  %arg_tuple.1 = bf16[2,209664] parameter(0)
  %copy.arg_tuple.1 = bf16[2,209664] copy(%arg_tuple.1), sharding={devices=[1,2]0,1}
  %iota.7 = s32[2,209664] iota(), iota_dimension=1,
    metadata={op_type="TopKV2" op_name="TopKV2"}
  %sort.32 = (bf16[2,209664], s32[2,209664])
    sort(bf16[2,209664] %copy.arg_tuple.1, s32[2,209664] %iota.7),
    dimensions={1}, is_stable=true, to_apply=%compare-greater-than.8,
    metadata={op_type="TopKV2" op_name="TopKV2"}
  %get-tuple-element.33 = bf16[2,209664]
    get-tuple-element((bf16[2,209664], s32[2,209664]) %sort.32),
    index=0, metadata={op_type="TopKV2" op_name="TopKV2"}
  %slice.34 = bf16[2,2000] slice(bf16[2,209664]
    %get-tuple-element.33), slice={[0:2], [0:2000]},
    metadata={op_type="TopKV2" op_name="TopKV2"}
  %get-tuple-element.35 = s32[2,209664]
    get-tuple-element((bf16[2,209664], s32[2,209664]) %sort.32),
    index=1, metadata={op_type="TopKV2" op_name="TopKV2"}
  %slice.36 = s32[2,2000] slice(s32[2,209664]
    %get-tuple-element.35), slice={[0:2], [0:2000]},
    metadata={op_type="TopKV2" op_name="TopKV2"}
  ROOT %tuple.46 = (bf16[2,2000], s32[2,2000])
    tuple(bf16[2,2000] %slice.34, s32[2,2000]
    %slice.36), sharding={{replicated}, {replicated}},
    metadata={op_name="XLA_Retvals"}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  auto sort = FindInstruction(module.get(), "sort.0");
  EXPECT_EQ(sort->operand(0)->shape().dimensions(1), 104832);
  EXPECT_EQ(sort->operand(1)->shape().dimensions(1), 104832);
  auto final_sort = FindInstruction(module.get(), "sort.1");
  EXPECT_EQ(final_sort->operand(0)->shape().dimensions(1), 4000);
  EXPECT_EQ(final_sort->operand(1)->shape().dimensions(1), 4000);
}

TEST_F(SpmdPartitioningTest, NoPartitionSortInTopKWhenSecondOperandIsNotIota) {
  absl::string_view hlo_string = R"(
HloModule module

%compare-greater-than.8 (p.0.lhs.2566: bf16[],
  p.0.rhs.2567: bf16[], p.1.lhs.2586: s32[], p.1.rhs.2587: s32[]) -> pred[] {
  %p.0.lhs.2566 = bf16[] parameter(0)
  %convert.164 = f32[] convert(bf16[] %p.0.lhs.2566)
  %bitcast-convert.48 = s32[] bitcast-convert(f32[] %convert.164)
  %constant.285 = s32[] constant(0)
  %compare.84 = pred[] compare(s32[] %bitcast-convert.48, s32[] %constant.285),
    direction=LT
  %constant.286 = u32[] constant(2147483647)
  %bitcast-convert.49 = u32[] bitcast-convert(f32[] %convert.164)
  %subtract.84 = u32[] subtract(u32[] %constant.286, u32[] %bitcast-convert.49)
  %bitcast-convert.50 = s32[] bitcast-convert(u32[] %subtract.84)
  %select.40 = s32[] select(pred[] %compare.84, s32[] %bitcast-convert.50,
    s32[] %bitcast-convert.48)
  %p.0.rhs.2567 = bf16[] parameter(1)
  %convert.165 = f32[] convert(bf16[] %p.0.rhs.2567)
  %bitcast-convert.51 = s32[] bitcast-convert(f32[] %convert.165)
  %compare.85 = pred[] compare(s32[] %bitcast-convert.51, s32[] %constant.285),
    direction=LT
  %bitcast-convert.52 = u32[] bitcast-convert(f32[] %convert.165)
  %subtract.85 = u32[] subtract(u32[] %constant.286, u32[] %bitcast-convert.52)
  %bitcast-convert.53 = s32[] bitcast-convert(u32[] %subtract.85)
  %select.41 = s32[] select(pred[] %compare.85, s32[] %bitcast-convert.53,
    s32[] %bitcast-convert.51)
  %compare.86 = pred[] compare(s32[] %select.40, s32[] %select.41), direction=GT
  %compare.1645 = pred[] compare(s32[] %select.41, s32[] %select.40), direction=GT
  %compare.1646 = pred[] compare(pred[] %compare.86, pred[] %compare.1645),
    direction=EQ
  %p.1.lhs.2586 = s32[] parameter(2)
  %p.1.rhs.2587 = s32[] parameter(3)
  %compare.1647 = pred[] compare(s32[] %p.1.lhs.2586, s32[] %p.1.rhs.2587),
    direction=LT
  ROOT %select.1054 = pred[] select(pred[] %compare.1646, pred[] %compare.1647,
    pred[] %compare.86)
}

ENTRY entry {
  %arg_tuple.1 = bf16[2,209664] parameter(0)
  %arg_tuple.2 = s32[2,209664] parameter(1)
  %copy.arg_tuple.1 = bf16[2,209664] copy(%arg_tuple.1), sharding={devices=[1,2]0,1}
  %sort.32 = (bf16[2,209664], s32[2,209664])
    sort(bf16[2,209664] %copy.arg_tuple.1, s32[2,209664] %arg_tuple.2),
    dimensions={1}, is_stable=true, to_apply=%compare-greater-than.8,
    metadata={op_type="TopKV2" op_name="TopKV2"}
  %get-tuple-element.33 = bf16[2,209664]
    get-tuple-element((bf16[2,209664], s32[2,209664]) %sort.32),
    index=0, metadata={op_type="TopKV2" op_name="TopKV2"}
  %slice.34 = bf16[2,2000] slice(bf16[2,209664]
    %get-tuple-element.33), slice={[0:2], [0:2000]},
    metadata={op_type="TopKV2" op_name="TopKV2"}
  %get-tuple-element.35 = s32[2,209664]
    get-tuple-element((bf16[2,209664], s32[2,209664]) %sort.32),
    index=1, metadata={op_type="TopKV2" op_name="TopKV2"}
  %slice.36 = s32[2,2000] slice(s32[2,209664]
    %get-tuple-element.35), slice={[0:2], [0:2000]},
    metadata={op_type="TopKV2" op_name="TopKV2"}
  ROOT %tuple.46 = (bf16[2,2000], s32[2,2000])
    tuple(bf16[2,2000] %slice.34, s32[2,2000]
    %slice.36), sharding={{replicated}, {replicated}},
    metadata={op_name="XLA_Retvals"}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  auto sort = FindInstruction(module.get(), "sort.0");
  // The shape of the operands changed from [2,209664] to [1,209664] due to
  // moving the sharding from the sort dim (dim 1) to dim 0. This optimization
  // was implemented for b/258523376.
  EXPECT_EQ(sort->operand(0)->shape().dimensions(1), 209664);
  EXPECT_EQ(sort->operand(1)->shape().dimensions(1), 209664);
}

TEST_F(SpmdPartitioningTest, NoPartitionSortInTopKWhenNoPartitionInSortDim) {
  absl::string_view hlo_string = R"(
HloModule module

%compare-greater-than.8 (p.0.lhs.2566: bf16[],
  p.0.rhs.2567: bf16[], p.1.lhs.2586: s32[], p.1.rhs.2587: s32[]) -> pred[] {
  %p.0.lhs.2566 = bf16[] parameter(0)
  %convert.164 = f32[] convert(bf16[] %p.0.lhs.2566)
  %bitcast-convert.48 = s32[] bitcast-convert(f32[] %convert.164)
  %constant.285 = s32[] constant(0)
  %compare.84 = pred[] compare(s32[] %bitcast-convert.48, s32[] %constant.285),
    direction=LT
  %constant.286 = u32[] constant(2147483647)
  %bitcast-convert.49 = u32[] bitcast-convert(f32[] %convert.164)
  %subtract.84 = u32[] subtract(u32[] %constant.286, u32[] %bitcast-convert.49)
  %bitcast-convert.50 = s32[] bitcast-convert(u32[] %subtract.84)
  %select.40 = s32[] select(pred[] %compare.84, s32[] %bitcast-convert.50,
    s32[] %bitcast-convert.48)
  %p.0.rhs.2567 = bf16[] parameter(1)
  %convert.165 = f32[] convert(bf16[] %p.0.rhs.2567)
  %bitcast-convert.51 = s32[] bitcast-convert(f32[] %convert.165)
  %compare.85 = pred[] compare(s32[] %bitcast-convert.51, s32[] %constant.285),
    direction=LT
  %bitcast-convert.52 = u32[] bitcast-convert(f32[] %convert.165)
  %subtract.85 = u32[] subtract(u32[] %constant.286, u32[] %bitcast-convert.52)
  %bitcast-convert.53 = s32[] bitcast-convert(u32[] %subtract.85)
  %select.41 = s32[] select(pred[] %compare.85, s32[] %bitcast-convert.53,
    s32[] %bitcast-convert.51)
  %compare.86 = pred[] compare(s32[] %select.40, s32[] %select.41), direction=GT
  %compare.1645 = pred[] compare(s32[] %select.41, s32[] %select.40), direction=GT
  %compare.1646 = pred[] compare(pred[] %compare.86, pred[] %compare.1645),
    direction=EQ
  %p.1.lhs.2586 = s32[] parameter(2)
  %p.1.rhs.2587 = s32[] parameter(3)
  %compare.1647 = pred[] compare(s32[] %p.1.lhs.2586, s32[] %p.1.rhs.2587),
    direction=LT
  ROOT %select.1054 = pred[] select(pred[] %compare.1646, pred[] %compare.1647,
    pred[] %compare.86)
}

ENTRY entry
  (arg_tuple.1: ()) -> (bf16[2,2000], s32[2,2000]) {
  %arg_tuple.1 = bf16[2,209664] parameter(0)
  %copy.arg_tuple.1 = bf16[2,209664] copy(%arg_tuple.1), sharding={devices=[2,1]0,1}
  %iota.7 = s32[2,209664] iota(), iota_dimension=1,
    metadata={op_type="TopKV2" op_name="TopKV2"}
  %sort.32 = (bf16[2,209664], s32[2,209664])
    sort(bf16[2,209664] %copy.arg_tuple.1, s32[2,209664] %iota.7),
    dimensions={1}, is_stable=true, to_apply=%compare-greater-than.8,
    metadata={op_type="TopKV2" op_name="TopKV2"}
  %get-tuple-element.33 = bf16[2,209664]
    get-tuple-element((bf16[2,209664], s32[2,209664]) %sort.32),
    index=0, metadata={op_type="TopKV2" op_name="TopKV2"}
  %slice.34 = bf16[2,2000] slice(bf16[2,209664]
    %get-tuple-element.33), slice={[0:2], [0:2000]},
    metadata={op_type="TopKV2" op_name="TopKV2"}
  %get-tuple-element.35 = s32[2,209664]
    get-tuple-element((bf16[2,209664], s32[2,209664]) %sort.32),
    index=1, metadata={op_type="TopKV2" op_name="TopKV2"}
  %slice.36 = s32[2,2000] slice(s32[2,209664]
    %get-tuple-element.35), slice={[0:2], [0:2000]},
    metadata={op_type="TopKV2" op_name="TopKV2"}
  ROOT %tuple.46 = (bf16[2,2000], s32[2,2000])
    tuple(bf16[2,2000] %slice.34, s32[2,2000]
    %slice.36), sharding={{replicated}, {replicated}},
    metadata={op_name="XLA_Retvals"}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  auto sort = FindInstruction(module.get(), "sort.0");
  EXPECT_EQ(sort->operand(0)->shape().dimensions(1), 209664);
  EXPECT_EQ(sort->operand(1)->shape().dimensions(1), 209664);
}

TEST_F(SpmdPartitioningTest, NoPartitionSortInTopKWhenSliceInOtherDim) {
  absl::string_view hlo_string = R"(
HloModule module

%compare-greater-than.8 (p.0.lhs.2566: bf16[],
  p.0.rhs.2567: bf16[], p.1.lhs.2586: s32[], p.1.rhs.2587: s32[]) -> pred[] {
  %p.0.lhs.2566 = bf16[] parameter(0)
  %convert.164 = f32[] convert(bf16[] %p.0.lhs.2566)
  %bitcast-convert.48 = s32[] bitcast-convert(f32[] %convert.164)
  %constant.285 = s32[] constant(0)
  %compare.84 = pred[] compare(s32[] %bitcast-convert.48, s32[] %constant.285),
    direction=LT
  %constant.286 = u32[] constant(2147483647)
  %bitcast-convert.49 = u32[] bitcast-convert(f32[] %convert.164)
  %subtract.84 = u32[] subtract(u32[] %constant.286, u32[] %bitcast-convert.49)
  %bitcast-convert.50 = s32[] bitcast-convert(u32[] %subtract.84)
  %select.40 = s32[] select(pred[] %compare.84, s32[] %bitcast-convert.50,
    s32[] %bitcast-convert.48)
  %p.0.rhs.2567 = bf16[] parameter(1)
  %convert.165 = f32[] convert(bf16[] %p.0.rhs.2567)
  %bitcast-convert.51 = s32[] bitcast-convert(f32[] %convert.165)
  %compare.85 = pred[] compare(s32[] %bitcast-convert.51, s32[] %constant.285),
    direction=LT
  %bitcast-convert.52 = u32[] bitcast-convert(f32[] %convert.165)
  %subtract.85 = u32[] subtract(u32[] %constant.286, u32[] %bitcast-convert.52)
  %bitcast-convert.53 = s32[] bitcast-convert(u32[] %subtract.85)
  %select.41 = s32[] select(pred[] %compare.85, s32[] %bitcast-convert.53,
    s32[] %bitcast-convert.51)
  %compare.86 = pred[] compare(s32[] %select.40, s32[] %select.41), direction=GT
  %compare.1645 = pred[] compare(s32[] %select.41, s32[] %select.40), direction=GT
  %compare.1646 = pred[] compare(pred[] %compare.86, pred[] %compare.1645),
    direction=EQ
  %p.1.lhs.2586 = s32[] parameter(2)
  %p.1.rhs.2587 = s32[] parameter(3)
  %compare.1647 = pred[] compare(s32[] %p.1.lhs.2586, s32[] %p.1.rhs.2587),
    direction=LT
  ROOT %select.1054 = pred[] select(pred[] %compare.1646, pred[] %compare.1647,
    pred[] %compare.86)
}

ENTRY entry {
  %arg_tuple.1 = bf16[2,209664] parameter(0)
  %copy.arg_tuple.1 = bf16[2,209664] copy(%arg_tuple.1), sharding={devices=[1,2]0,1}
  %iota.7 = s32[2,209664] iota(), iota_dimension=1,
    metadata={op_type="TopKV2" op_name="TopKV2"}
  %sort.32 = (bf16[2,209664], s32[2,209664])
    sort(bf16[2,209664] %copy.arg_tuple.1, s32[2,209664] %iota.7),
    dimensions={1}, is_stable=true, to_apply=%compare-greater-than.8,
    metadata={op_type="TopKV2" op_name="TopKV2"}
  %get-tuple-element.33 = bf16[2,209664]
    get-tuple-element((bf16[2,209664], s32[2,209664]) %sort.32),
    index=0, metadata={op_type="TopKV2" op_name="TopKV2"}
  %slice.34 = bf16[1,209664] slice(bf16[2,209664]
    %get-tuple-element.33), slice={[0:1], [0:209664]},
    metadata={op_type="TopKV2" op_name="TopKV2"}
  %get-tuple-element.35 = s32[2,209664]
    get-tuple-element((bf16[2,209664], s32[2,209664]) %sort.32),
    index=1, metadata={op_type="TopKV2" op_name="TopKV2"}
  %slice.36 = s32[1,209664] slice(s32[2,209664]
    %get-tuple-element.35), slice={[0:1], [0:209664]},
    metadata={op_type="TopKV2" op_name="TopKV2"}
  ROOT %tuple.46 = (bf16[1,209664], s32[1,209664])
    tuple(bf16[1,209664] %slice.34, s32[1,209664]
    %slice.36), sharding={{replicated}, {replicated}},
    metadata={op_name="XLA_Retvals"}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  auto sort = FindInstruction(module.get(), "sort.0");
  EXPECT_EQ(sort->operand(0)->shape().dimensions(1), 209664);
  EXPECT_EQ(sort->operand(1)->shape().dimensions(1), 209664);
}

TEST_F(SpmdPartitioningTest, SortShardedOnSortDim_SlowSortBug) {
  // Test with the sort in b/258523376 (same comparator, shapes, and sharding)
  absl::string_view hlo_string = R"(
HloModule module, entry_computation_layout={(f32[32768,65536]{1,0})->(f32[32768,65536]{1,0}, s32[32768,65536]{1,0})}

region_174.7326 {
  Arg_0.7327 = f32[] parameter(0), sharding={replicated}
  compare.7339 = pred[] compare(Arg_0.7327, Arg_0.7327), direction=NE, sharding={replicated}
  constant.7332 = s32[] constant(2143289344), sharding={replicated}
  constant.7334 = f32[] constant(0), sharding={replicated}
  compare.7337 = pred[] compare(Arg_0.7327, constant.7334), direction=EQ, sharding={replicated}
  constant.7333 = s32[] constant(0), sharding={replicated}
  bitcast-convert.7335 = s32[] bitcast-convert(Arg_0.7327), sharding={replicated}
  select.7338 = s32[] select(compare.7337, constant.7333, bitcast-convert.7335), sharding={replicated}
  select.7340 = s32[] select(compare.7339, constant.7332, select.7338), sharding={replicated}
  constant.1127 = s32[] constant(0), sharding={replicated}
  compare.7343 = pred[] compare(select.7340, constant.1127), direction=LT, sharding={replicated}
  constant.7331 = u32[] constant(2147483647), sharding={replicated}
  bitcast-convert.7336 = u32[] bitcast-convert(Arg_0.7327), sharding={replicated}
  subtract.7341 = u32[] subtract(constant.7331, bitcast-convert.7336), sharding={replicated}
  bitcast-convert.7342 = s32[] bitcast-convert(subtract.7341), sharding={replicated}
  select.7344 = s32[] select(compare.7343, bitcast-convert.7342, select.7340), sharding={replicated}
  Arg_1.7328 = f32[] parameter(1), sharding={replicated}
  compare.7349 = pred[] compare(Arg_1.7328, Arg_1.7328), direction=NE, sharding={replicated}
  constant.1125 = s32[] constant(2143289344), sharding={replicated}
  constant.1126 = f32[] constant(0), sharding={replicated}
  compare.7347 = pred[] compare(Arg_1.7328, constant.1126), direction=EQ, sharding={replicated}
  constant.1128 = s32[] constant(0), sharding={replicated}
  bitcast-convert.7345 = s32[] bitcast-convert(Arg_1.7328), sharding={replicated}
  select.7348 = s32[] select(compare.7347, constant.1128, bitcast-convert.7345), sharding={replicated}
  select.7350 = s32[] select(compare.7349, constant.1125, select.7348), sharding={replicated}
  constant.1129 = s32[] constant(0), sharding={replicated}
  compare.7353 = pred[] compare(select.7350, constant.1129), direction=LT, sharding={replicated}
  constant.1130 = u32[] constant(2147483647), sharding={replicated}
  bitcast-convert.7346 = u32[] bitcast-convert(Arg_1.7328), sharding={replicated}
  subtract.7351 = u32[] subtract(constant.1130, bitcast-convert.7346), sharding={replicated}
  bitcast-convert.7352 = s32[] bitcast-convert(subtract.7351), sharding={replicated}
  select.7354 = s32[] select(compare.7353, bitcast-convert.7352, select.7350), sharding={replicated}
  compare.7355 = pred[] compare(select.7344, select.7354), direction=LT, sharding={replicated}
  compare.24 = pred[] compare(select.7354, select.7344), direction=LT, sharding={replicated}
  compare.25 = pred[] compare(compare.7355, compare.24), direction=EQ, sharding={replicated}
  Arg_2.7329 = s32[] parameter(2), sharding={replicated}
  Arg_3.7330 = s32[] parameter(3), sharding={replicated}
  compare.26 = pred[] compare(Arg_2.7329, Arg_3.7330), direction=LT, sharding={replicated}
  ROOT select.21 = pred[] select(compare.25, compare.26, compare.7355), sharding={replicated}
}

ENTRY entry {
  param.0 = f32[32768,65536]{1,0} parameter(0)
  negate.7325 = f32[32768,65536]{1,0} negate(param.0), sharding={devices=[1,64]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63}
  iota.30 = s32[32768,65536]{1,0} iota(), iota_dimension=1, sharding={devices=[1,64]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63}
  ROOT sort.0 = (f32[32768,65536]{1,0}, s32[32768,65536]{1,0}) sort(negate.7325, iota.30), dimensions={1}, is_stable=true, to_apply=region_174.7326, sharding={{devices=[1,64]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63}, {devices=[1,64]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63}}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/64));
  VLOG(1) << module->ToString();
  auto sort = FindInstruction(module.get(), "sort.1");
  for (auto operand : sort->operands()) {
    EXPECT_EQ(operand->shape().dimensions(0), 512);
    EXPECT_EQ(operand->shape().dimensions(1), 65536);
  }
}

TEST_F(SpmdPartitioningTest, SortShardedOnSortDim_OneOperand) {
  absl::string_view hlo_string = R"(
HloModule module, entry_computation_layout={(f32[1024,1024]{1,0})->f32[1024,1024]{1,0}}

compare {
  p.0.lhs = f32[] parameter(0), sharding={replicated}
  p.0.rhs = f32[] parameter(1), sharding={replicated}
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, sharding={replicated}
}

ENTRY entry {
  param.0 = f32[1024,1024]{1,0} parameter(0)
  negate.0 = f32[1024,1024]{1,0} negate(param.0), sharding={devices=[1,8]0,1,2,3,4,5,6,7}
  ROOT sort.0 = f32[1024,1024]{1,0} sort(negate.0), dimensions={1}, is_stable=true, to_apply=compare, sharding={devices=[1,8]0,1,2,3,4,5,6,7}
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  auto sort = FindInstruction(module.get(), "sort.1");
  for (auto operand : sort->operands()) {
    EXPECT_EQ(operand->shape().dimensions(0), 128);
    EXPECT_EQ(operand->shape().dimensions(1), 1024);
  }
}

TEST_F(SpmdPartitioningTest, SortShardedOnSortDim_TwoOperands) {
  absl::string_view hlo_string = R"(
HloModule module, entry_computation_layout={(f32[1024,1024]{1,0})->(f32[1024,1024]{1,0},s32[1024,1024]{1,0})}

compare {
  p.0.lhs = f32[] parameter(0), sharding={replicated}
  p.0.rhs = f32[] parameter(1), sharding={replicated}
  p.1.lhs = s32[] parameter(2), sharding={replicated}
  p.1.rhs = s32[] parameter(3), sharding={replicated}
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, sharding={replicated}
}

ENTRY entry {
  param.0 = f32[1024,1024]{1,0} parameter(0)
  negate.0 = f32[1024,1024]{1,0} negate(param.0), sharding={devices=[1,8]0,1,2,3,4,5,6,7}
  iota.0 = s32[1024,1024]{1,0} iota(), iota_dimension=1, sharding={devices=[1,8]0,1,2,3,4,5,6,7}
  ROOT sort.0 = (f32[1024,1024]{1,0}, s32[1024,1024]{1,0}) sort(negate.0, iota.0), dimensions={1}, is_stable=true, to_apply=compare, sharding={{devices=[1,8]0,1,2,3,4,5,6,7},{devices=[1,8]0,1,2,3,4,5,6,7}}
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  auto sort = FindInstruction(module.get(), "sort.1");
  for (auto operand : sort->operands()) {
    EXPECT_EQ(operand->shape().dimensions(0), 128);
    EXPECT_EQ(operand->shape().dimensions(1), 1024);
  }
}

TEST_F(SpmdPartitioningTest, SortShardedOnSortDim_ThreeOperands) {
  absl::string_view hlo_string = R"(
HloModule module, entry_computation_layout={(f32[1024,1024]{1,0})->(f32[1024,1024]{1,0},s32[1024,1024]{1,0},s32[1024,1024]{1,0})}

compare {
  p.0.lhs = f32[] parameter(0), sharding={replicated}
  p.0.rhs = f32[] parameter(1), sharding={replicated}
  p.1.lhs = s32[] parameter(2), sharding={replicated}
  p.1.rhs = s32[] parameter(3), sharding={replicated}
  p.2.lhs = s32[] parameter(4), sharding={replicated}
  p.2.rhs = s32[] parameter(5), sharding={replicated}
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, sharding={replicated}
}

ENTRY entry {
  param.0 = f32[1024,1024]{1,0} parameter(0)
  negate.0 = f32[1024,1024]{1,0} negate(param.0), sharding={devices=[1,8]0,1,2,3,4,5,6,7}
  iota.0 = s32[1024,1024]{1,0} iota(), iota_dimension=0, sharding={devices=[1,8]0,1,2,3,4,5,6,7}
  iota.1 = s32[1024,1024]{1,0} iota(), iota_dimension=1, sharding={devices=[1,8]0,1,2,3,4,5,6,7}
  ROOT sort.0 = (f32[1024,1024]{1,0}, s32[1024,1024]{1,0}, s32[1024,1024]{1,0}) sort(negate.0, iota.0, iota.1), dimensions={1}, is_stable=true, to_apply=compare, sharding={{devices=[1,8]0,1,2,3,4,5,6,7},{devices=[1,8]0,1,2,3,4,5,6,7},{devices=[1,8]0,1,2,3,4,5,6,7}}
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  auto sort = FindInstruction(module.get(), "sort.1");
  for (auto operand : sort->operands()) {
    EXPECT_EQ(operand->shape().dimensions(0), 128);
    EXPECT_EQ(operand->shape().dimensions(1), 1024);
  }
}

TEST_F(SpmdPartitioningTest, SortShardedOnSortDim_RankOne) {
  absl::string_view hlo_string = R"(
HloModule module, entry_computation_layout={(f32[1024]{0})->(f32[1024]{0},s32[1024]{0})}

compare {
  p.0.lhs = f32[] parameter(0), sharding={replicated}
  p.0.rhs = f32[] parameter(1), sharding={replicated}
  p.1.lhs = s32[] parameter(2), sharding={replicated}
  p.1.rhs = s32[] parameter(3), sharding={replicated}
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, sharding={replicated}
}

ENTRY entry {
  param.0 = f32[1024]{0} parameter(0)
  negate.0 = f32[1024]{0} negate(param.0), sharding={devices=[8]0,1,2,3,4,5,6,7}
  iota.0 = s32[1024]{0} iota(), iota_dimension=0
  ROOT sort.0 = (f32[1024]{0}, s32[1024]{0}) sort(negate.0, iota.0), dimensions={0}, is_stable=true, to_apply=compare
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  auto sort = FindInstruction(module.get(), "sort.1");
  for (auto operand : sort->operands()) {
    EXPECT_EQ(operand->shape().dimensions(0), 1024);
  }
}

TEST_F(SpmdPartitioningTest, SortShardedOnSortDim_TwoFreeDivisibleDims) {
  absl::string_view hlo_string = R"(
HloModule module, entry_computation_layout={(f32[8,1024,1024]{2,1,0})->(f32[8,1024,1024]{2,1,0},s32[8,1024,1024]{2,1,0})}

compare {
  p.0.lhs = f32[] parameter(0), sharding={replicated}
  p.0.rhs = f32[] parameter(1), sharding={replicated}
  p.1.lhs = s32[] parameter(2), sharding={replicated}
  p.1.rhs = s32[] parameter(3), sharding={replicated}
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, sharding={replicated}
}

ENTRY entry {
  param.0 = f32[8,1024,1024]{2,1,0} parameter(0)
  negate.0 = f32[8,1024,1024]{2,1,0} negate(param.0), sharding={devices=[1,1,8]0,1,2,3,4,5,6,7}
  iota.0 = s32[8,1024,1024]{2,1,0} iota(), iota_dimension=2, sharding={devices=[1,1,8]0,1,2,3,4,5,6,7}
  ROOT sort.0 = (f32[8,1024,1024]{2,1,0}, s32[8,1024,1024]{2,1,0}) sort(negate.0, iota.0), dimensions={2}, is_stable=true, to_apply=compare, sharding={{devices=[1,1,8]0,1,2,3,4,5,6,7},{devices=[1,1,8]0,1,2,3,4,5,6,7}}
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  auto sort = FindInstruction(module.get(), "sort.1");
  for (auto operand : sort->operands()) {
    EXPECT_EQ(operand->shape().dimensions(0), 1);
    EXPECT_EQ(operand->shape().dimensions(1), 1024);
    EXPECT_EQ(operand->shape().dimensions(2), 1024);
  }
}

TEST_F(SpmdPartitioningTest, SortShardedOnSortDim_OneFreeDivisibleDim) {
  absl::string_view hlo_string = R"(
HloModule module, entry_computation_layout={(f32[7,1024,1024]{2,1,0})->(f32[7,1024,1024]{2,1,0},s32[7,1024,1024]{2,1,0})}

compare {
  p.0.lhs = f32[] parameter(0), sharding={replicated}
  p.0.rhs = f32[] parameter(1), sharding={replicated}
  p.1.lhs = s32[] parameter(2), sharding={replicated}
  p.1.rhs = s32[] parameter(3), sharding={replicated}
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, sharding={replicated}
}

ENTRY entry {
  param.0 = f32[7,1024,1024]{2,1,0} parameter(0)
  negate.0 = f32[7,1024,1024]{2,1,0} negate(param.0), sharding={devices=[1,1,8]0,1,2,3,4,5,6,7}
  iota.0 = s32[7,1024,1024]{2,1,0} iota(), iota_dimension=2, sharding={devices=[1,1,8]0,1,2,3,4,5,6,7}
  ROOT sort.0 = (f32[7,1024,1024]{2,1,0}, s32[7,1024,1024]{2,1,0}) sort(negate.0, iota.0), dimensions={2}, is_stable=true, to_apply=compare, sharding={{devices=[1,1,8]0,1,2,3,4,5,6,7},{devices=[1,1,8]0,1,2,3,4,5,6,7}}
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  auto sort = FindInstruction(module.get(), "sort.1");
  for (auto operand : sort->operands()) {
    EXPECT_EQ(operand->shape().dimensions(0), 7);
    EXPECT_EQ(operand->shape().dimensions(1), 128);
    EXPECT_EQ(operand->shape().dimensions(2), 1024);
  }
}

TEST_F(SpmdPartitioningTest, SortShardedOnSortDim_OneFreeNondivisibleDim) {
  absl::string_view hlo_string = R"(
HloModule module, entry_computation_layout={(f32[7,1024,1024]{2,1,0})->(f32[7,1024,1024]{2,1,0},s32[7,1024,1024]{2,1,0})}

compare {
  p.0.lhs = f32[] parameter(0), sharding={replicated}
  p.0.rhs = f32[] parameter(1), sharding={replicated}
  p.1.lhs = s32[] parameter(2), sharding={replicated}
  p.1.rhs = s32[] parameter(3), sharding={replicated}
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, sharding={replicated}
}

ENTRY entry {
  param.0 = f32[7,1024,1024]{2,1,0} parameter(0)
  negate.0 = f32[7,1024,1024]{2,1,0} negate(param.0), sharding={devices=[1,2,4]0,1,2,3,4,5,6,7}
  iota.0 = s32[7,1024,1024]{2,1,0} iota(), iota_dimension=2, sharding={devices=[1,2,4]0,1,2,3,4,5,6,7}
  ROOT sort.0 = (f32[7,1024,1024]{2,1,0}, s32[7,1024,1024]{2,1,0}) sort(negate.0, iota.0), dimensions={2}, is_stable=true, to_apply=compare, sharding={{devices=[1,2,4]0,1,2,3,4,5,6,7},{devices=[1,2,4]0,1,2,3,4,5,6,7}}
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  auto sort = FindInstruction(module.get(), "sort.1");
  for (auto operand : sort->operands()) {
    EXPECT_EQ(operand->shape().dimensions(0), 2);
    EXPECT_EQ(operand->shape().dimensions(1), 512);
    EXPECT_EQ(operand->shape().dimensions(2), 1024);
  }
}

TEST_F(SpmdPartitioningTest, SortShardedOnSortDim_LastTileDimReplicate) {
  absl::string_view hlo_string = R"(
HloModule module, entry_computation_layout={(f32[1024,1024]{1,0})->f32[1024,1024]{1,0}}

compare {
  p.0.lhs = f32[] parameter(0), sharding={replicated}
  p.0.rhs = f32[] parameter(1), sharding={replicated}
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, sharding={replicated}
}

ENTRY entry {
  param.0 = f32[1024,1024]{1,0} parameter(0)
  negate.0 = f32[1024,1024]{1,0} negate(param.0), sharding={devices=[1,2,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT sort.0 = f32[1024,1024]{1,0} sort(negate.0), dimensions={1}, is_stable=true, to_apply=compare, sharding={devices=[1,2,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  auto sort = FindInstruction(module.get(), "sort.1");
  for (auto operand : sort->operands()) {
    EXPECT_EQ(operand->shape().dimensions(0), 512);
    EXPECT_EQ(operand->shape().dimensions(1), 1024);
  }
}

TEST_F(SpmdPartitioningTest, ShardableTranspose) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[16,38,38,4] parameter(0)
  %param0.copy = f32[16,38,38,4] copy(%param0), sharding={devices=[1,2,1,1]0,1}
  ROOT %transpose = f32[16,4,38,38] transpose(%param0.copy),
    dimensions={0,3,1,2}, sharding={devices=[1,1,2,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[16,19,38,4]"));
  EXPECT_THAT(root, AllOf(op::Transpose(param0), op::Shape("f32[16,4,19,38]")));
}

TEST_F(SpmdPartitioningTest, MultiDimensionShardedTranspose) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[16,38,38,4] parameter(0)
  %param0.copy = f32[16,38,38,4] copy(%param0),
    sharding={devices=[4,2,1,1]0,1,2,3,4,5,6,7}
  ROOT %transpose = f32[38,4,16,38] transpose(%param0.copy),
    dimensions={1,3,0,2}, sharding={devices=[2,1,4,1]0,2,4,6,1,3,5,7}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Reshape(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[4,19,38,4]"));
  EXPECT_THAT(root, AllOf(op::Transpose(param0), op::Shape("f32[19,4,4,38]")));
}

TEST_F(SpmdPartitioningTest, NonShardableTranspose) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[16,38,38,4] parameter(0)
  %param0.copy = f32[16,38,38,4] copy(%param0), sharding={devices=[1,2,1,1]0,1}
  ROOT %transpose = f32[16,4,38,38] transpose(%param0.copy),
    dimensions={0,3,1,2}, sharding={devices=[1,2,1,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto resahrd = AllOf(op::Reshape(op::Transpose(op::Reshape(op::AllToAll()))),
                       op::Shape("f32[16,38,38,2]"));
  EXPECT_THAT(root, AllOf(op::Transpose(), op::Shape("f32[16,2,38,38]")));
}

TEST_F(SpmdPartitioningTest, PartialReplicateShardableTranspose) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[16,38,38,4] parameter(0)
  %param0.copy = f32[16,38,38,4] copy(%param0),
    sharding={devices=[1,2,1,1,2]0,1,2,3 last_tile_dim_replicate}
  ROOT %transpose = f32[16,4,38,38] transpose(%param0.copy),
    dimensions={0,3,1,2},
    sharding={devices=[1,1,2,1,2]0,1,2,3 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[16,19,38,4]"));
  EXPECT_THAT(root, AllOf(op::Transpose(param0), op::Shape("f32[16,4,19,38]")));
}

TEST_F(SpmdPartitioningTest, PartialReplicateNonShardableTranspose) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[16,38,38,4] parameter(0)
  %param0.copy = f32[16,38,38,4] copy(%param0),
    sharding={devices=[1,2,1,1,2]0,1,2,3 last_tile_dim_replicate}
  ROOT %transpose = f32[16,4,38,38] transpose(%param0.copy),
    dimensions={0,3,1,2},
    sharding={devices=[1,2,1,1,2]0,1,2,3 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto resahrd = AllOf(op::Reshape(op::Transpose(op::Reshape(op::AllToAll()))),
                       op::Shape("f32[16,38,38,2]"));
  EXPECT_THAT(root, AllOf(op::Transpose(), op::Shape("f32[16,2,38,38]")));
}

TEST_F(SpmdPartitioningTest, PartialReplicateMultiDimensionShardedTranspose) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[16,38,38,4] parameter(0)
  %param0.copy = f32[16,38,38,4] copy(%param0),
    sharding={devices=[2,2,1,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %transpose = f32[38,4,16,38] transpose(%param0.copy),
    dimensions={1,3,0,2},
    sharding={devices=[2,1,2,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Reshape(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[8,19,38,4]"));
  EXPECT_THAT(root, AllOf(op::Transpose(param0), op::Shape("f32[19,4,8,38]")));
}

TEST_F(SpmdPartitioningTest, ShardableReshape) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[38,38,324] parameter(0)
  %param0.copy = f32[38,38,324] copy(%param0), sharding={devices=[2,1,1]0,1}
  ROOT %reshape = f32[38,38,4,81] reshape(%param0.copy),
    sharding={devices=[2,1,1,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 =
      AllOf(op::Copy(op::DynamicSlice(op::Parameter(), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[19,38,324]"));
  EXPECT_THAT(root, AllOf(op::Reshape(param0), op::Shape("f32[19,38,4,81]")));
}

TEST_F(SpmdPartitioningTest, ReshapeWithReshard) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[38,38,324] parameter(0), sharding={devices=[2,1,1]0,1}
  ROOT %reshape = f32[38,38,4,81] reshape(%param0),
    sharding={devices=[1,2,1,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto input_reshard =
      op::Reshape(op::Transpose(op::AllToAll(op::Reshape(op::Parameter(0)))));
  EXPECT_THAT(root,
              AllOf(op::Reshape(input_reshard), op::Shape("f32[38,19,4,81]")));
}

TEST_F(SpmdPartitioningTest, ReshapeWithReshard2) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[38,38,324] parameter(0), sharding={devices=[2,1,1]0,1}
  ROOT %reshape = f32[38,38,2,162] reshape(%param0),
    sharding={devices=[1,1,1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto local_reshape =
      AllOf(op::Reshape(op::Parameter(0)), op::Shape("f32[19,38,2,162]"));
  EXPECT_THAT(root, AllOf(op::Shape("f32[38,38,2,81]"),
                          op::Reshape(op::Transpose(
                              op::AllToAll(op::Reshape(local_reshape))))));
}

TEST_F(SpmdPartitioningTest, PartialReplicateShardableReshape) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[38,38,324] parameter(0)
  %param0.copy = f32[38,38,324] copy(%param0),
    sharding={devices=[2,1,1,2]0,1,2,3 last_tile_dim_replicate}
  ROOT %reshape = f32[38,38,4,81] reshape(%param0.copy),
    sharding={devices=[2,1,1,1,2]0,1,2,3 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 =
      AllOf(op::Copy(op::DynamicSlice(op::Parameter(), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[19,38,324]"));
  EXPECT_THAT(root, AllOf(op::Reshape(param0), op::Shape("f32[19,38,4,81]")));
}

TEST_F(SpmdPartitioningTest, ReshapeMergeDimsWithHaloExchange) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %input = s32[2,3,7,10] parameter(0), sharding={devices=[1,1,2,1]0,1}
  ROOT %reshape = s32[3,2,1,14,5] reshape(%input),
    sharding={devices=[1,1,1,2,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  auto reshape =
      AllOf(op::Reshape(op::Parameter(0)), op::Shape("s32[3,2,1,8,5]"));
  auto halo = op::CollectivePermute(op::Slice(reshape));
  auto exchanged = op::DynamicSlice(op::Concatenate(halo, op::Slice(reshape)),
                                    _, _, _, _, _);
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(exchanged, op::Shape("s32[3,2,1,7,5]")));
}

TEST_F(SpmdPartitioningTest, PartialReplicateReshapeMergeDimsWithHaloExchange) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %input = s32[2,3,7,10] parameter(0),
    sharding={devices=[1,1,2,1,2]0,1,2,3 last_tile_dim_replicate}
  ROOT %reshape = s32[3,2,1,14,5] reshape(%input),
    sharding={devices=[1,1,1,2,1,2]0,1,2,3 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  auto reshape =
      AllOf(op::Reshape(op::Parameter(0)), op::Shape("s32[3,2,1,8,5]"));
  auto halo = op::CollectivePermute(op::Slice(reshape));
  auto exchanged = op::DynamicSlice(op::Concatenate(halo, op::Slice(reshape)),
                                    _, _, _, _, _);
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(exchanged, op::Shape("s32[3,2,1,7,5]")));
}

TEST_F(SpmdPartitioningTest, TileToPartialReplicateHaloExchangeWithPadding) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[2,123]{1,0} parameter(0), sharding={devices=[8,1]0,1,2,3,4,5,6,7}
  ROOT %reshape = f32[2,1,123]{2,1,0} reshape(%input),
    sharding={devices=[2,1,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  auto reshape = AllOf(op::Reshape(op::AllReduce(op::Select(
                           _, op::CollectivePermute(op::Parameter()), _))),
                       op::Shape("f32[1,1,123]"));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, reshape);
}

// Produces an invalid module after transformation.
TEST_F(SpmdPartitioningTest, InceptionV3_4_way_ReduceWindowDilated) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  %param0 = f32[128,5,5,768] parameter(0)
  %param0.copy = f32[128,5,5,768] copy(%param0),
    sharding={devices=[1,4,1,1]0,1,2,3}
  %constant.1 = f32[] constant(0), sharding={replicated}
  ROOT %rw = f32[128,17,17,768] reduce-window(%param0.copy, %constant.1),
    window={size=1x5x5x1 pad=0_0x4_4x4_4x0_0 lhs_dilate=1x3x3x1},
    to_apply=sum, sharding={devices=[1,4,1,1]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  auto input_shard = op::Copy(op::DynamicSlice(
      op::Pad(op::Parameter(0), op::Constant()), op::Constant(), op::Reshape(),
      op::Constant(), op::Constant()));
  auto id_mul4_add1 =
      op::Add(op::Multiply(op::Reshape(), op::Constant()), op::Constant());
  auto id_mul5 = op::Multiply(op::Reshape(), op::Constant());
  auto id_mul5_add1_div3 =
      op::Divide(op::Add(id_mul5, op::Constant()), op::Constant());
  auto before_masking = AllOf(
      op::Shape("f32[128,3,5,768]"),
      op::DynamicSlice(
          AllOf(
              op::Shape("f32[128,4,5,768]"),
              op::Concatenate(op::CollectivePermute(input_shard), input_shard)),
          op::Constant(),
          op::Subtract(op::Constant(),
                       op::Subtract(id_mul4_add1, id_mul5_add1_div3)),
          op::Constant(), op::Constant()));
  auto masked = op::Select(
      op::And(op::Compare(op::Add(op::Iota(), op::Broadcast(id_mul5_add1_div3)),
                          op::Broadcast(op::Constant())),
              op::Compare(op::Add(op::Iota(), op::Broadcast(id_mul5_add1_div3)),
                          op::Broadcast(op::Constant()))),
      before_masking, op::Broadcast(op::Constant()));
  auto rw = AllOf(op::Shape("f32[128,7,17,768]"),
                  op::ReduceWindow(masked, op::Constant()));
  auto final_slice_index = op::Subtract(
      id_mul5,
      op::Add(op::Multiply(id_mul5_add1_div3, op::Constant()), op::Constant()));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              AllOf(op::Shape("f32[128,5,17,768]"),
                    op::DynamicSlice(rw, op::Constant(), final_slice_index,
                                     op::Constant(), op::Constant())));
}

TEST_F(SpmdPartitioningTest, TiledToTiledReduce) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  %param0 = f32[4,32,32,128] parameter(0)
  %param0.copy = f32[4,32,32,128] copy(%param0),
    sharding={devices=[1,1,1,2]0,1}
  %constant.1 = f32[] constant(0), sharding={replicated}
  %reduce = f32[128] reduce(%param0.copy, %constant.1), dimensions={0,1,2},
    to_apply=%sum, sharding={devices=[2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[4,32,32,64]"));

  EXPECT_THAT(root,
              AllOf(op::Reduce(param0, op::Constant()), op::Shape("f32[64]")));
}

TEST_F(SpmdPartitioningTest, PartialTiledToPartialTiledReduce) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  %param0 = f32[4,4] parameter(0),
    sharding={devices=[2,2,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %constant.1 = f32[] constant(0), sharding={replicated}
  ROOT %reduce = f32[4] reduce(%param0, %constant.1), dimensions={0},
    to_apply=%sum,
    sharding={devices=[2,4]0,1,4,5,2,3,6,7 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              AllOf(op::AllReduce(op::Reduce(op::Parameter(0), op::Constant())),
                    op::Shape("f32[2]")));
}

TEST_F(SpmdPartitioningTest, DeviceMaximalTupleReduce) {
  absl::string_view hlo_string = R"(
HloModule module

%minmax_func {
  %lhs_value = f32[] parameter(0)
  %rhs_value = f32[] parameter(2)
  %compare.2 = pred[] compare(%lhs_value, %rhs_value), direction=GT
  %select.4 = f32[] select(%compare.2, %lhs_value, %rhs_value)
  %lhs_index = s32[] parameter(1)
  %rhs_index = s32[] parameter(3)
  %select.5 = s32[] select(%compare.2, %lhs_index, %rhs_index)
  ROOT %tuple.2 = (f32[], s32[]) tuple(%select.4, %select.5)
}

ENTRY %main {
  %param0 = f32[28,10] parameter(0), sharding={maximal device=0}
  %param1 = s32[28,10] parameter(1), sharding={maximal device=0}
  %init0 = f32[] parameter(2), sharding={maximal device=0}
  %init1 = s32[] parameter(3), sharding={maximal device=0}
  ROOT %reduce = (f32[28], s32[28]) reduce(%param0, %param1, %init0, %init1),
    dimensions={1}, to_apply=%minmax_func,
    sharding={{maximal device=0}, {maximal device=0}}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Reduce(op::Parameter(0), op::Parameter(1),
                                     op::Parameter(2), op::Parameter(3)),
                          op::Shape("(f32[28], s32[28])")));
}

TEST_F(SpmdPartitioningTest, TiledToTiledTupleReduce) {
  absl::string_view hlo_string = R"(
HloModule module

%minmax_func {
  %lhs_value = f32[] parameter(0)
  %rhs_value = f32[] parameter(2)
  %compare.2 = pred[] compare(%lhs_value, %rhs_value), direction=GT
  %select.4 = f32[] select(%compare.2, %lhs_value, %rhs_value)
  %lhs_index = s32[] parameter(1)
  %rhs_index = s32[] parameter(3)
  %select.5 = s32[] select(%compare.2, %lhs_index, %rhs_index)
  ROOT %tuple.2 = (f32[], s32[]) tuple(%select.4, %select.5)
}

ENTRY %main {
  %param0 = f32[28,10] parameter(0), sharding={devices=[2,1]0,1}
  %param1 = s32[28,10] parameter(1), sharding={devices=[2,1]0,1}
  %init0 = f32[] parameter(2)
  %init1 = s32[] parameter(3)
  ROOT %reduce = (f32[28], s32[28]) reduce(%param0, %param1, %init0, %init1),
    dimensions={1}, to_apply=%minmax_func,
    sharding={{devices=[2]0,1}, {devices=[2]0,1}}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Reduce(op::Parameter(0), op::Parameter(1),
                                     op::Parameter(2), op::Parameter(3)),
                          op::Shape("(f32[14], s32[14])")));
}

TEST_F(SpmdPartitioningTest, TiledToPartiallyTiledTupleReduce) {
  absl::string_view hlo_string = R"(
HloModule module

%minmax_func {
  %lhs_value = f32[] parameter(0)
  %rhs_value = f32[] parameter(2)
  %compare.2 = pred[] compare(%lhs_value, %rhs_value), direction=GT
  %select.4 = f32[] select(%compare.2, %lhs_value, %rhs_value)
  %lhs_index = s32[] parameter(1)
  %rhs_index = s32[] parameter(3)
  %select.5 = s32[] select(%compare.2, %lhs_index, %rhs_index)
  ROOT %tuple.2 = (f32[], s32[]) tuple(%select.4, %select.5)
}

ENTRY %main {
  %param0 = f32[28,12] parameter(0), sharding={devices=[2,4]0,1,2,3,4,5,6,7}
  %param1 = s32[28,12] parameter(1), sharding={devices=[2,4]0,1,2,3,4,5,6,7}
  %init0 = f32[] parameter(2)
  %init1 = s32[] parameter(3)
  ROOT %reduce = (f32[28], s32[28]) reduce(%param0, %param1, %init0, %init1),
    dimensions={1}, to_apply=%minmax_func,
    sharding={{devices=[2,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate},
              {devices=[2,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();

  const auto lhs = AllOf(op::Shape("f32[14,3]"), op::Parameter(0));
  const auto rhs = AllOf(op::Shape("s32[14,3]"), op::Parameter(1));
  auto local_reduce =
      AllOf(op::Reduce(lhs, rhs, op::Parameter(2), op::Parameter(3)),
            op::Shape("(f32[14], s32[14])"));
  auto reshape_l = AllOf(op::Reshape(op::GetTupleElement(local_reduce)),
                         op::Shape("f32[14,1]"));
  auto reshape_r = AllOf(op::Reshape(op::GetTupleElement(local_reduce)),
                         op::Shape("s32[14,1]"));
  auto broadcast_l =
      AllOf(op::AllReduce(op::DynamicUpdateSlice(_, reshape_l, _, _)),
            op::Shape("f32[14,4]"));
  auto broadcast_r =
      AllOf(op::AllReduce(op::DynamicUpdateSlice(_, reshape_r, _, _)),
            op::Shape("s32[14,4]"));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Reduce(broadcast_l, broadcast_r, op::Parameter(2),
                                     op::Parameter(3)),
                          op::Shape("(f32[14], s32[14])")));
}

TEST_F(SpmdPartitioningTest, TupleReduceSubgroupManual) {
  absl::string_view hlo_string = R"(
HloModule module

%minmax_func {
  %lhs_value = f32[] parameter(0)
  %rhs_value = f32[] parameter(2)
  %compare.2 = pred[] compare(%lhs_value, %rhs_value), direction=GT
  %select.4 = f32[] select(%compare.2, %lhs_value, %rhs_value)
  %lhs_index = s32[] parameter(1)
  %rhs_index = s32[] parameter(3)
  %select.5 = s32[] select(%compare.2, %lhs_index, %rhs_index)
  ROOT %tuple.2 = (f32[], s32[]) tuple(%select.4, %select.5)
}

ENTRY %main {
  %param0 = f32[28,12] parameter(0),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dims={manual}}
  %param1 = s32[28,12] parameter(1),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dims={manual}}
  %init0 = f32[] parameter(2),
    sharding={devices=[2,2]0,1,2,3 last_tile_dims={replicated,manual}}
  %init1 = s32[] parameter(3),
    sharding={devices=[2,2]0,1,2,3 last_tile_dims={replicated,manual}}
  ROOT %reduce = (f32[28], s32[28]) reduce(%param0, %param1, %init0, %init1),
    dimensions={1}, to_apply=%minmax_func,
    sharding={{devices=[1,2,2]0,1,2,3 last_tile_dims={replicated,manual}},
              {devices=[1,2,2]0,1,2,3 last_tile_dims={replicated,manual}}}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto lhs = AllOf(op::Shape("f32[28,6]"), op::Parameter(0));
  const auto rhs = AllOf(op::Shape("s32[28,6]"), op::Parameter(1));
  auto local_reduce =
      AllOf(op::Reduce(lhs, rhs, op::Parameter(2), op::Parameter(3)),
            op::Shape("(f32[28], s32[28])"));
  auto reshape_l = AllOf(op::Reshape(op::GetTupleElement(local_reduce)),
                         op::Shape("f32[28,1]"));
  auto reshape_r = AllOf(op::Reshape(op::GetTupleElement(local_reduce)),
                         op::Shape("s32[28,1]"));
  auto broadcast_l =
      AllOf(op::AllReduce(op::DynamicUpdateSlice(_, reshape_l, _, _)),
            op::Shape("f32[28,2]"));
  auto broadcast_r =
      AllOf(op::AllReduce(op::DynamicUpdateSlice(_, reshape_r, _, _)),
            op::Shape("s32[28,2]"));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Reduce(broadcast_l, broadcast_r, op::Parameter(2),
                                     op::Parameter(3)),
                          op::Shape("(f32[28], s32[28])")));
}

TEST_F(SpmdPartitioningTest, TiledToTiledReduceOutputReshard) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  %param0 = f32[4,32,32,128] parameter(0)
  %param0.copy = f32[4,32,32,128] copy(%param0),
    sharding={devices=[1,2,1,1]0,1}
  %constant.1 = f32[] constant(0), sharding={replicated}
  %reduce = f32[128] reduce(%param0.copy, %constant.1), dimensions={0,1,2},
    to_apply=%sum, sharding={devices=[2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[4,16,32,128]"));

  EXPECT_THAT(root,
              AllOf(op::DynamicSlice(
                        AllOf(op::AllReduce(op::Reduce(param0, op::Constant())),
                              op::Shape("f32[128]")),
                        op::Reshape()),
                    op::Shape("f32[64]")));
}

TEST_F(SpmdPartitioningTest, IotaAlongNonTileDimension) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  ROOT %iota = s32[16,80,91] iota(), iota_dimension=1,
    sharding={devices=[1,1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Iota(), op::Shape("s32[16,80,46]")));
}

TEST_F(SpmdPartitioningTest, IotaAlongTileDimension) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  ROOT %iota = s32[16,80,91] iota(), iota_dimension=2,
    sharding={devices=[1,1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Add(op::Iota(), op::Broadcast()),
                          op::Shape("s32[16,80,46]")));
}

TEST_F(SpmdPartitioningTest, U32IotaAlongTileDimension) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  ROOT %iota = u32[16,80,91] iota(), iota_dimension=2,
    sharding={devices=[1,1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Add(op::Iota(), op::Broadcast()),
                          op::Shape("u32[16,80,46]")));
}

TEST_F(SpmdPartitioningTest, Conditional) {
  absl::string_view hlo_string = R"(
HloModule module

Negate {
  x = f32[4,5] parameter(0), sharding={devices=[2,1]0,1}
  ROOT negate = f32[4,5] negate(x), sharding={devices=[2,1]0,1}
}

Identity {
  y = f32[4,5] parameter(0), sharding={devices=[2,1]0,1}
  ROOT copy = f32[4,5] copy(y), sharding={devices=[2,1]0,1}
}

ENTRY entry {
  %param.0 = pred[] parameter(0)
  %param.0.copy = pred[] copy(%param.0), sharding={maximal device=0}
  %param.1 = f32[4,5] parameter(1)
  %param.1.copy = f32[4,5] copy(%param.1), sharding={replicated}
  %param.2 = f32[4,5] parameter(2)
  %param.2.copy = f32[4,5] copy(%param.2), sharding={devices=[2,1]0,1}
  ROOT cond = f32[4,5] conditional(%param.0.copy, %param.1.copy, %param.2.copy),
    true_computation=Negate, false_computation=Identity,
    sharding={devices=[2,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  auto param0 = AllOf(op::Copy(op::Copy(op::Parameter()), op::Shape("pred[]")));
  auto param1 = AllOf(op::Copy(op::Parameter()), op::Shape("f32[4,5]"));
  auto param2 = AllOf(op::Copy(op::DynamicSlice(op::Parameter(), op::Reshape(),
                                                op::Constant())),
                      op::Shape("f32[2,5]"));

  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Conditional(op::AllReduce(), param1, param2),
                          op::Shape("f32[2,5]")));

  auto then_branch_root = root->branch_computation(0)->root_instruction();
  EXPECT_THAT(then_branch_root,
              AllOf(op::Negate(op::DynamicSlice(op::Parameter(), op::Reshape(),
                                                op::Constant())),
                    op::Shape("f32[2,5]")));

  auto else_branch_root = root->branch_computation(1)->root_instruction();
  EXPECT_THAT(else_branch_root,
              AllOf(op::Copy(op::Parameter()), op::Shape("f32[2,5]")));
}

TEST_F(SpmdPartitioningTest, ConditionalManual) {
  absl::string_view hlo_string = R"(
HloModule module

Negate {
  x = f32[4,5] parameter(0), sharding={manual}
  ROOT negate = f32[4,5] negate(x), sharding={manual}
}

Identity {
  y = f32[4,5] parameter(0), sharding={manual}
  ROOT copy = f32[4,5] copy(y), sharding={manual}
}

ENTRY entry {
  %param.0 = pred[] parameter(0), sharding={manual}
  %param.1 = f32[4,5] parameter(1), sharding={manual}
  %param.2 = f32[4,5] parameter(2), sharding={manual}
  ROOT cond = f32[4,5] conditional(%param.0, %param.1, %param.2),
    true_computation=Negate, false_computation=Identity, sharding={manual}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  auto param0 = AllOf(op::Parameter(0), op::Shape("pred[]"));
  auto param1 = AllOf(op::Parameter(1), op::Shape("f32[4,5]"));
  auto param2 = AllOf(op::Parameter(2), op::Shape("f32[4,5]"));

  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Conditional(param0, param1, param2),
                          op::Shape("f32[4,5]")));
}

TEST_F(SpmdPartitioningTest, WhileManual) {
  absl::string_view hlo_string = R"(
HloModule module

LoopCond {
  x = s32[] parameter(0), sharding={manual}
  const = s32[] constant(5), sharding={manual}
  ROOT lt = pred[] compare(x, const), direction=LT, sharding={manual}
}

Inc {
  x = s32[] parameter(0), sharding={manual}
  const = s32[] constant(1), sharding={manual}
  ROOT add = s32[] add(x, const), sharding={manual}
}

ENTRY entry {
  zero = s32[] parameter(0), sharding={manual}
  ROOT while = s32[] while(zero), body=Inc, condition=LoopCond,
    sharding={manual}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  auto zero = AllOf(op::Parameter(0), op::Shape("s32[]"));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::While(zero), op::Shape("s32[]")));
}

TEST_F(SpmdPartitioningTest, SelectAndScatter_RetinaNet) {
  absl::string_view hlo_string = R"(
HloModule module

ge {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT compare = pred[] compare(a, b), direction=GE
}

sum {
  c = f32[] parameter(0)
  d = f32[] parameter(1)
  ROOT add = f32[] add(c, d)
}

ENTRY entry {
  %param.0 = f32[32,128,384,64] parameter(0)
  %param.0.copy = f32[32,128,384,64] copy(%param.0),
    sharding={devices=[1,8,1,1]0,1,2,3,4,5,6,7}
  %param.1 = f32[32,64,192,64] parameter(1)
  %param.1.copy = f32[32,64,192,64] copy(%param.1),
    sharding={devices=[1,8,1,1]0,1,2,3,4,5,6,7}
  constant.1 = f32[] constant(0), sharding={replicated}
  ROOT select-and-scatter = f32[32,128,384,64] select-and-scatter(param.0.copy,
    %param.1.copy, constant.1), window={size=1x1x1x1 stride=1x2x2x1},
    select=ge, scatter=sum, sharding={devices=[1,8,1,1]0,1,2,3,4,5,6,7}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto source = AllOf(
      op::Shape("f32[32,8,192,64]"),
      op::Copy(op::DynamicSlice(op::Parameter(1), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())));
  auto data = AllOf(
      op::Shape("f32[32,16,384,64]"),
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())));

  EXPECT_THAT(root, op::SelectAndScatter(data, source, op::Constant()));
  EXPECT_EQ(root->window().dimensions(0).padding_low(), 0);
  EXPECT_EQ(root->window().dimensions(0).padding_high(), 0);
}

TEST_F(SpmdPartitioningTest, TiledDot) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[128,64] parameter(0)
  %lhs.copy = f32[128,64] copy(%lhs), sharding={devices=[1,2]0,1}
  %rhs = f32[64,256] parameter(1)
  %rhs.copy = f32[64,256] copy(%rhs), sharding={devices=[2,1]0,1}
  ROOT %conv = f32[128,256] convolution(%lhs.copy, %rhs.copy),
    dim_labels=bf_io->bf, sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/2,
                           /*conv_halo_exchange_always_on_lhs=*/false));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(op::Copy(op::DynamicSlice(
                             op::Parameter(), op::Constant(), op::Reshape())),
                         op::Shape("f32[128,32]"));
  const auto rhs = AllOf(op::Copy(op::DynamicSlice(
                             op::Parameter(), op::Reshape(), op::Constant())),
                         op::Shape("f32[32,256]"));
  EXPECT_THAT(root, AllOf(op::AllReduce(op::Convolution(lhs, rhs)),
                          op::Shape("f32[128,256]")));
}

TEST_F(SpmdPartitioningTest, TiledDotOutputTiled) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[128,64] parameter(0)
  %lhs.copy = f32[128,64] copy(%lhs), sharding={devices=[1,2]0,1}
  %rhs = f32[64,256] parameter(1)
  %rhs.copy = f32[64,256] copy(%rhs), sharding={devices=[2,1]0,1}
  ROOT %conv = f32[128,256] convolution(%lhs.copy, %rhs.copy),
    dim_labels=bf_io->bf, sharding={devices=[1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(op::Copy(op::DynamicSlice(
                             op::Parameter(), op::Constant(), op::Reshape())),
                         op::Shape("f32[128,32]"));
  const auto rhs = AllOf(op::Copy(op::DynamicSlice(
                             op::Parameter(), op::Reshape(), op::Constant())),
                         op::Shape("f32[32,256]"));
  EXPECT_THAT(root, AllOf(op::DynamicSlice(
                              AllOf(op::AllReduce(op::Convolution(lhs, rhs)),
                                    op::Shape("f32[128,256]")),
                              op::Constant(), op::Reshape()),
                          op::Shape("f32[128,128]")));
}

TEST_F(SpmdPartitioningTest, BatchPartitionedConvolution) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[128,256,256] parameter(0)
  %lhs.copy = f32[128,256,256] copy(%lhs), sharding={devices=[1,2,1]0,1}
  %rhs = f32[256,8,1] parameter(1)
  %rhs.copy = f32[256,8,1] copy(%rhs), sharding={replicated}
  ROOT %conv = f32[128,256,8] convolution(%lhs.copy, %rhs.copy),
    window={size=1}, dim_labels=0bf_io0->0bf, sharding={devices=[1,2,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs =
      AllOf(op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(),
                                      op::Reshape(), op::Constant())),
            op::Shape("f32[128,128,256]"));
  const auto rhs = AllOf(op::Copy(op::Parameter(1)), op::Shape("f32[256,8,1]"));
  EXPECT_THAT(root,
              AllOf(op::Convolution(lhs, rhs), op::Shape("f32[128,128,8]")));
}

TEST_F(SpmdPartitioningTest, DotOutputFeaturePartitioned) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[24,64] parameter(0)
  %lhs.copy = f32[24,64] copy(%lhs), sharding={replicated}
  %rhs = f32[39296,64] parameter(1)
  %rhs.copy = f32[39296,64] copy(%rhs), sharding={devices=[2,1]0,1}
  ROOT %dot = f32[24,39296] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={1}, rhs_contracting_dims={1},
    sharding={devices=[1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(op::Copy(op::Parameter(0)), op::Shape("f32[24,64]"));
  const auto rhs = AllOf(op::Copy(op::DynamicSlice(
                             op::Parameter(1), op::Reshape(), op::Constant())),
                         op::Shape("f32[19648,64]"));
  EXPECT_THAT(root, AllOf(op::Dot(lhs, rhs), op::Shape("f32[24,19648]")));
}

TEST_F(SpmdPartitioningTest, WindowedEinsumTwoContractingDimsLhsReshard) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %p0 = f32[2048,2,3264]{2,1,0} parameter(0), sharding={devices=[1,1,2]0,1}
  %p1 = f32[2,3264,2176]{2,1,0} parameter(1), sharding={devices=[2,1,1]0,1}
  ROOT %dot.224 = f32[2048,2176]{1,0} dot(f32[2048,2,3264]{2,1,0} %p0, f32[2,3264,2176]{2,1,0} %p1), lhs_contracting_dims={1,2}, rhs_contracting_dims={0,1}, sharding={devices=[1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/2,
                           /*conv_halo_exchange_always_on_lhs=*/true,
                           /*choose_faster_windowed_einsum=*/false,
                           /*unroll_windowed_einsum=*/false,
                           /*bidirectional_windowed_einsum=*/false,
                           /*threshold_for_windowed_einsum_mib=*/0));
  VLOG(1) << module->ToString();

  // Check while op.
  const auto arg0 = AllOf(
      op::Reshape(op::Transpose(op::AllToAll(op::Reshape(op::Parameter(0))))),
      op::Shape("f32[2048,1,3264]"));
  const auto arg1 = AllOf(op::Parameter(1), op::Shape("f32[1,3264,2176]"));

  const auto while_op =
      AllOf(op::While(op::Tuple(arg0, arg1, op::Broadcast(), op::Broadcast(),
                                op::Constant())),
            op::Shape("(f32[2048,1,3264]{2,1,0}, f32[1,3264,2176]{2,1,0},"
                      " f32[2048,1088]{1,0}, f32[2048,1088]{1,0}, u32[])"));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, AllOf(op::GetTupleElement(while_op), op::Shape("f32[2048,1088]")));

  // Check while op body.
  const auto while_loop = root->operand(0);
  const auto next_i =
      op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant());
  auto lhs = AllOf(op::GetTupleElement(op::Parameter(0)),
                   op::Shape("f32[2048,1,3264]"));
  auto rhs = AllOf(op::DynamicSlice(), op::Shape("f32[1,3264,1088]"));
  auto dot_op = op::Dot(lhs, rhs);
  auto add_op = op::Add(op::GetTupleElement(op::Parameter(0)), dot_op);
  auto cond_op =
      op::Conditional(op::Compare(next_i, op::Constant()), add_op, add_op);
  EXPECT_THAT(while_loop->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Parameter(0)),
                        op::GetTupleElement(op::Parameter(0)), cond_op,
                        op::GetTupleElement(op::Parameter(0)), next_i));
}

TEST_F(SpmdPartitioningTest, WindowedEinsumTwoContractingDimsRhsReshard) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %p0 = f32[4096,2,3264]{2,1,0} parameter(0), sharding={devices=[1,1,2]0,1}
  %p1 = f32[2,3264,2176]{2,1,0} parameter(1), sharding={devices=[2,1,1]0,1}
  ROOT %dot.224 = f32[4096,2176]{1,0} dot(f32[4096,2,3264]{2,1,0} %p0, f32[2,3264,2176]{2,1,0} %p1), lhs_contracting_dims={1,2}, rhs_contracting_dims={0,1}, sharding={devices=[1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/2,
                           /*conv_halo_exchange_always_on_lhs=*/true,
                           /*choose_faster_windowed_einsum=*/false,
                           /*unroll_windowed_einsum=*/false,
                           /*bidirectional_windowed_einsum=*/false,
                           /*threshold_for_windowed_einsum_mib=*/0));
  VLOG(1) << module->ToString();

  // Check while op.
  const auto arg0 = AllOf(op::Parameter(0), op::Shape("f32[4096,2,1632]"));
  const auto arg1 = AllOf(
      op::Reshape(op::Transpose(op::AllToAll(op::Reshape(op::Parameter(1))))),
      op::Shape("f32[2,1632,2176]"));

  const auto while_op =
      AllOf(op::While(op::Tuple(arg0, arg1, op::Broadcast(), op::Broadcast(),
                                op::Constant())),
            op::Shape("(f32[4096,2,1632]{2,1,0}, f32[2,1632,2176]{2,1,0},"
                      " f32[4096,1088]{1,0}, f32[4096,1088]{1,0}, u32[])"));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, AllOf(op::GetTupleElement(while_op), op::Shape("f32[4096,1088]")));

  // Check while op body.
  const auto while_loop = root->operand(0);
  const auto next_i =
      op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant());
  auto lhs = AllOf(op::GetTupleElement(op::Parameter(0)),
                   op::Shape("f32[4096,2,1632]"));
  auto rhs = AllOf(op::DynamicSlice(), op::Shape("f32[2,1632,1088]"));
  auto dot_op = op::Dot(lhs, rhs);
  auto add_op = op::Add(op::GetTupleElement(op::Parameter(0)), dot_op);
  auto cond_op =
      op::Conditional(op::Compare(next_i, op::Constant()), add_op, add_op);
  EXPECT_THAT(while_loop->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Parameter(0)),
                        op::GetTupleElement(op::Parameter(0)), cond_op,
                        op::GetTupleElement(op::Parameter(0)), next_i));
}

TEST_F(SpmdPartitioningTest, ChooseWindowedEinsumOverIncreasedMemUsageOption) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %p0 = bf16[512,4,512]{2,1,0} parameter(0), sharding={devices=[16,1,4]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63}
  %p1 = bf16[512,4,512]{2,1,0} parameter(1), sharding={devices=[16,1,4]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63}
  %multiply.611 = bf16[512,4,512]{2,1,0} multiply(bf16[512,4,512]{2,1,0} %p0, bf16[512,4,512]{2,1,0} %p1), sharding={devices=[16,1,4]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63}

  %p2 = bf16[1,2048,768]{2,1,0} parameter(2), sharding={devices=[1,4,16]0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61,2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63}
  %reshape.1074 = bf16[4,512,768]{2,1,0} reshape(bf16[1,2048,768]{2,1,0} %p2), sharding={devices=[4,1,16]0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61,2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63}

  ROOT %dot.128 = bf16[512,768]{1,0} dot(bf16[512,4,512]{2,1,0} %multiply.611, bf16[4,512,768]{2,1,0} %reshape.1074), lhs_contracting_dims={1,2}, rhs_contracting_dims={0,1}, sharding={devices=[16,4]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/64,
                           /*conv_halo_exchange_always_on_lhs=*/true,
                           /*choose_faster_windowed_einsum=*/true,
                           /*unroll_windowed_einsum=*/false,
                           /*bidirectional_windowed_einsum=*/false,
                           /*threshold_for_windowed_einsum_mib=*/0));
  VLOG(1) << module->ToString();

  // Check while op.
  const auto arg0 = AllOf(op::Reshape(), op::Shape("bf16[32,1,512]{2,1,0}"));
  const auto arg1 = AllOf(op::AllReduce(), op::Shape("bf16[1,512,768]{2,1,0}"));

  const auto while_op =
      AllOf(op::While(op::Tuple(arg0, arg1, op::Broadcast(), op::Broadcast(),
                                op::Constant())),
            op::Shape("(bf16[32,1,512]{2,1,0}, bf16[1,512,768]{2,1,0},"
                      " bf16[32,192]{1,0}, bf16[32,192]{1,0}, u32[])"));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::GetTupleElement(while_op),
                          op::Shape("bf16[32,192]{1,0}")));

  // Check while op body.
  const auto while_loop = root->operand(0);
  const auto next_i =
      op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant());
  auto lhs = AllOf(op::GetTupleElement(op::Parameter(0)),
                   op::Shape("bf16[32,1,512]{2,1,0}"));
  auto rhs = AllOf(op::DynamicSlice(), op::Shape("bf16[1,512,192]{2,1,0}"));
  auto dot_op = op::Dot(lhs, rhs);
  auto add_op = op::Add(op::GetTupleElement(op::Parameter(0)), dot_op);
  auto cond_op =
      op::Conditional(op::Compare(next_i, op::Constant()), add_op, add_op);
  EXPECT_THAT(while_loop->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Parameter(0)),
                        op::GetTupleElement(op::Parameter(0)), cond_op,
                        op::GetTupleElement(op::Parameter(0)), next_i));
}

TEST_F(SpmdPartitioningTest, DotPartialDeviceOrder) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[16,256,4096] parameter(0), sharding={devices=[1,1,2,2]1,3,0,2 last_tile_dim_replicate}
  %rhs = f32[4096,2048] parameter(1), sharding={devices=[2,2]3,1,2,0}
  ROOT %dot = f32[16,256,2048] dot(%lhs, %rhs),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={2}, rhs_contracting_dims={0},
    sharding={devices=[1,1,2,2]2,3,0,1 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(op::Parameter(0), op::Shape("f32[16,256,2048]"));
  const auto rhs = AllOf(op::Parameter(1), op::Shape("f32[2048,1024]"));
  EXPECT_THAT(root, AllOf(op::AllReduce(op::Dot(lhs, rhs)),
                          op::Shape("f32[16,256,1024]")));
}

TEST_F(SpmdPartitioningTest, EinsumBatchPartitioned) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64] parameter(0)
  %lhs.copy = f32[32,24,64] copy(%lhs), sharding={devices=[2,1,1]0,1}
  %rhs = f32[32,39296,64] parameter(1)
  %rhs.copy = f32[32,39296,64] copy(%rhs), sharding={devices=[2,1,1]0,1}
  ROOT %dot = f32[32,24,39296] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2},
    sharding={devices=[2,1,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs =
      AllOf(op::Copy(op::DynamicSlice(op::Parameter(0), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[16,24,64]"));
  const auto rhs =
      AllOf(op::Copy(op::DynamicSlice(op::Parameter(1), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[16,39296,64]"));
  EXPECT_THAT(root, AllOf(op::Dot(lhs, rhs), op::Shape("f32[16,24,39296]")));
}

TEST_F(SpmdPartitioningTest, EinsumLHSandOutputBatchPartitioned) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64] parameter(0)
  %lhs.copy = f32[32,24,64] copy(%lhs), sharding={devices=[2,1,1]0,1}
  %rhs = f32[32,39296,64] parameter(1)
  %rhs.copy = f32[32,39296,64] copy(%rhs), sharding={replicated}
  ROOT %dot = f32[32,24,39296] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2},
    sharding={devices=[2,1,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs =
      AllOf(op::Copy(op::DynamicSlice(op::Parameter(0), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[16,24,64]"));
  const auto rhs =
      AllOf(op::Copy(op::Parameter(1)), op::Shape("f32[32,39296,64]"));
  EXPECT_THAT(root, AllOf(op::Dot(lhs, op::DynamicSlice(rhs, op::Reshape(),
                                                        op::Constant(),
                                                        op::Constant())),
                          op::Shape("f32[16,24,39296]")));
}

TEST_F(SpmdPartitioningTest, EinsumRHSandOutputBatchPartitioned) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64] parameter(0)
  %lhs.copy = f32[32,24,64] copy(%lhs), sharding={devices=[1,2,1]0,1}
  %rhs = f32[32,39296,64] parameter(1)
  %rhs.copy = f32[32,39296,64] copy(%rhs), sharding={devices=[2,1,1]0,1}
  ROOT %dot = f32[32,24,39296] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2},
    sharding={devices=[2,1,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs =
      AllOf(op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(),
                                      op::Reshape(), op::Constant())),
            op::Shape("f32[32,12,64]"));
  const auto rhs =
      AllOf(op::Copy(op::DynamicSlice(op::Parameter(1), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[16,39296,64]"));
  const auto lhs_reshard =
      op::Reshape(op::Transpose(op::AllToAll(op::Reshape(lhs))));
  EXPECT_THAT(root,
              AllOf(op::Dot(lhs_reshard, rhs), op::Shape("f32[16,24,39296]")));
}

TEST_F(SpmdPartitioningTest, EinsumOutputBatchPartitioned) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64] parameter(0)
  %lhs.copy = f32[32,24,64] copy(%lhs), sharding={replicated}
  %rhs = f32[32,39296,64] parameter(1)
  %rhs.copy = f32[32,39296,64] copy(%rhs), sharding={replicated}
  ROOT %dot = f32[32,24,39296] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2},
    sharding={devices=[2,1,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs_slice =
      AllOf(op::DynamicSlice(op::Copy(op::Parameter(0)), op::Reshape(),
                             op::Constant(), op::Constant()),
            op::Shape("f32[16,24,64]"));
  const auto rhs_slice =
      AllOf(op::DynamicSlice(op::Copy(op::Parameter(1)), op::Reshape(),
                             op::Constant(), op::Constant()),
            op::Shape("f32[16,39296,64]"));
  EXPECT_THAT(root, AllOf(op::Dot(lhs_slice, rhs_slice),
                          op::Shape("f32[16,24,39296]")));
}

TEST_F(SpmdPartitioningTest, EinsumContractingDimsPartitioned) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,128] parameter(0)
  %lhs.copy = f32[32,24,64,128] copy(%lhs), sharding={devices=[1,1,2,2]0,1,2,3}
  %rhs = f32[32,39296,64,128] parameter(1)
  %rhs.copy = f32[32,39296,64,128] copy(%rhs), sharding={devices=[1,1,2,2]0,1,2,3}
  ROOT %dot = f32[32,24,39296] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(),
                                op::Constant(), op::Reshape(), op::Reshape())),
      op::Shape("f32[32,24,32,64]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(1), op::Constant(),
                                op::Constant(), op::Reshape(), op::Reshape())),
      op::Shape("f32[32,39296,32,64]"));
  EXPECT_THAT(root, AllOf(op::AllReduce(op::AllReduce(op::Dot(lhs, rhs))),
                          op::Shape("f32[32,24,39296]")));
}

TEST_F(SpmdPartitioningTest,
       EinsumContractingDimsPartitionedResultPartiallySliced) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,64] parameter(0), sharding={devices=[1,4]0,1,2,3}
  %rhs = f32[64,128] parameter(1), sharding={devices=[4,1]0,1,2,3}
  ROOT %dot = f32[32,128] dot(%lhs, %rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    sharding={devices=[2,1,2]0,2,1,3 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(op::Parameter(0), op::Shape("f32[32,16]"));
  const auto rhs = AllOf(op::Parameter(1), op::Shape("f32[16,128]"));
  EXPECT_THAT(root, AllOf(op::AllReduce(op::DynamicSlice(
                              op::AllReduce(op::Dot(lhs, rhs)), _, _)),
                          op::Shape("f32[16,128]")));
}

TEST_F(SpmdPartitioningTest, EinsumLHSNonContractingDimsPartitioned) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,128] parameter(0)
  %lhs.copy = f32[32,24,64,128] copy(%lhs), sharding={devices=[1,2,1,2]0,1,2,3}
  %rhs = f32[32,39296,64] parameter(1)
  %rhs.copy = f32[32,39296,64] copy(%rhs), sharding={replicated}
  ROOT %dot = f32[32,24,128,39296] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2},
    sharding={devices=[1,2,2,1]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(), op::Reshape(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[32,12,64,64]"));
  const auto rhs =
      AllOf(op::Copy(op::Parameter(1)), op::Shape("f32[32,39296,64]"));
  EXPECT_THAT(root, AllOf(op::Dot(lhs, rhs), op::Shape("f32[32,12,64,39296]")));
}

TEST_F(SpmdPartitioningTest, EinsumRHSNonContractingDimsPartitioned) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64] parameter(0)
  %lhs.copy = f32[32,24,64] copy(%lhs), sharding={replicated}
  %rhs = f32[32,39296,64,128] parameter(1)
  %rhs.copy = f32[32,39296,64,128] copy(%rhs), sharding={devices=[1,2,1,2]0,1,2,3}
  ROOT %dot = f32[32,24,39296,128] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2},
    sharding={devices=[1,1,2,2]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs =
      AllOf(op::Copy(op::Parameter(0)), op::Shape("f32[32,24,64]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(1), op::Constant(), op::Reshape(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[32,19648,64,64]"));
  EXPECT_THAT(root, AllOf(op::Dot(lhs, rhs), op::Shape("f32[32,24,19648,64]")));
}

TEST_F(SpmdPartitioningTest, EinsumOutputLHSNonContractingDimPartitioned) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,128] parameter(0)
  %lhs.copy = f32[32,24,64,128] copy(%lhs), sharding={replicated}
  %rhs = f32[32,39296,64,128] parameter(1)
  %rhs.copy = f32[32,39296,64,128] copy(%rhs), sharding={replicated}
  ROOT %dot = f32[32,24,39296] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,2,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs =
      AllOf(op::Copy(op::Parameter(0)), op::Shape("f32[32,24,64,128]"));
  const auto rhs =
      AllOf(op::Copy(op::Parameter(1)), op::Shape("f32[32,39296,64,128]"));
  EXPECT_THAT(
      root,
      AllOf(op::Dot(AllOf(op::DynamicSlice(lhs, op::Constant(), op::Reshape(),
                                           op::Constant(), op::Constant()),
                          op::Shape("f32[32,12,64,128]")),
                    rhs),
            op::Shape("f32[32,12,39296]")));
}

TEST_F(SpmdPartitioningTest, EinsumOutputRHSNonContractingDimPartitioned) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,128] parameter(0)
  %lhs.copy = f32[32,24,64,128] copy(%lhs), sharding={replicated}
  %rhs = f32[32,39296,64,128] parameter(1)
  %rhs.copy = f32[32,39296,64,128] copy(%rhs), sharding={replicated}
  ROOT %dot = f32[32,24,39296] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs =
      AllOf(op::Copy(op::Parameter(0)), op::Shape("f32[32,24,64,128]"));
  const auto rhs =
      AllOf(op::Copy(op::Parameter(1)), op::Shape("f32[32,39296,64,128]"));
  EXPECT_THAT(root,
              AllOf(op::Dot(lhs, AllOf(op::DynamicSlice(
                                           rhs, op::Constant(), op::Reshape(),
                                           op::Constant(), op::Constant()),
                                       op::Shape("f32[32,19648,64,128]"))),
                    op::Shape("f32[32,24,19648]")));
}

TEST_F(SpmdPartitioningTest,
       EinsumRHSWindowedInContractingOutNonContractingPartitioned) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[320,25,64,128] parameter(0)
  %lhs.copy = f32[320,25,64,128] copy(%lhs), sharding={devices=[1,1,4,1]0,1,2,3}
  %rhs = f32[320,39296,64,128] parameter(1)
  %rhs.copy = f32[320,39296,64,128] copy(%rhs),
    sharding={devices=[1,1,4,1]0,1,2,3}
  ROOT %dot = f32[320,25,39296] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,4,1]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(),
                                op::Constant(), op::Reshape(), op::Constant())),
      op::Shape("f32[320,25,16,128]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(1), op::Constant(),
                                op::Constant(), op::Reshape(), op::Constant())),
      op::Shape("f32[320,39296,16,128]"));
  EXPECT_THAT(
      root,
      AllOf(op::GetTupleElement(op::While(op::Tuple(
                lhs, rhs, op::Broadcast(), op::Broadcast(), op::Constant()))),
            op::Shape("f32[320,7,39296]")));

  const auto while_loop = root->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  const auto next_i =
      op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant());
  auto ds =
      AllOf(op::DynamicSlice(
                op::Pad(op::GetTupleElement(op::Parameter(0)), op::Constant()),
                op::Constant(), op::Reshape(), op::Constant(), op::Constant()),
            op::Shape("f32[320,7,16,128]"));
  auto partial_output =
      AllOf(op::Add(op::GetTupleElement(op::Parameter(0)),
                    op::Dot(ds, op::GetTupleElement(op::Parameter(0)))),
            op::Shape("f32[320,7,39296]"));
  auto window = op::Conditional(op::Compare(next_i, op::Constant()),
                                partial_output, partial_output);
  EXPECT_THAT(while_loop->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Parameter(0)),
                        op::GetTupleElement(op::Parameter(0)), window,
                        op::GetTupleElement(op::Parameter(0)), next_i));

  // Check the conditional that contains the collective permute.
  auto cp_conditional =
      while_loop->while_body()->root_instruction()->operand(2);
  EXPECT_THAT(cp_conditional->true_computation()->root_instruction(),
              op::CollectivePermute(op::Parameter(0)));
  EXPECT_THAT(cp_conditional->false_computation()->root_instruction(),
              op::Parameter(0));
}

TEST_F(SpmdPartitioningTest,
       UnrolledEinsumRHSWindowedInContractingOutNonContractingPartitioned) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[320,25,64,128] parameter(0)
  %lhs.copy = f32[320,25,64,128] copy(%lhs), sharding={devices=[1,1,4,1]0,1,2,3}
  %rhs = f32[320,39296,64,128] parameter(1)
  %rhs.copy = f32[320,39296,64,128] copy(%rhs),
    sharding={devices=[1,1,4,1]0,1,2,3}
  ROOT %dot = f32[320,25,39296] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,4,1]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/4,
                           /*conv_halo_exchange_always_on_lhs =*/true,
                           /*choose_faster_windowed_einsum =*/false,
                           /*unroll_windowed_einsum =*/true));
  VLOG(1) << module->ToString();

  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(),
                                op::Constant(), op::Reshape(), op::Constant())),
      op::Shape("f32[320,25,16,128]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(1), op::Constant(),
                                op::Constant(), op::Reshape(), op::Constant())),
      op::Shape("f32[320,39296,16,128]"));
  const auto while_op = AllOf(
      op::While(op::Tuple(lhs, rhs, op::Broadcast(), op::Broadcast(),
                          op::Constant())),
      op::Shape("(f32[320,25,16,128], f32[320,39296,16,128], f32[320,7,39296],"
                " f32[320,7,39296], u32[])"));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, AllOf(op::Add(op::CollectivePermute(op::GetTupleElement(while_op)),
                          op::GetTupleElement(while_op)),
                  op::Shape("f32[320,7,39296]")));

  const auto while_loop = root->operand(1)->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  const auto next_i =
      op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant());
  auto ds =
      AllOf(op::DynamicSlice(
                op::Pad(op::GetTupleElement(op::Parameter(0)), op::Constant()),
                op::Constant(), op::Reshape(), op::Constant(), op::Constant()),
            op::Shape("f32[320,7,16,128]"));
  auto partial_output = AllOf(
      op::Add(op::CollectivePermute(op::GetTupleElement(op::Parameter(0))),
              op::Dot(ds, op::GetTupleElement(op::Parameter(0)))),
      op::Shape("f32[320,7,39296]"));
  auto partial_output2 =
      AllOf(op::CollectivePermute(
                op::Add(op::GetTupleElement(op::Parameter(0)),
                        op::Dot(ds, op::GetTupleElement(op::Parameter(0))))),
            op::Shape("f32[320,7,39296]"));

  EXPECT_THAT(while_loop->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Parameter(0)),
                        op::GetTupleElement(op::Parameter(0)), partial_output,
                        partial_output2, next_i));
}

TEST_F(
    SpmdPartitioningTest,
    BidirectionalEinsumRHSWindowedInContractingOutNonContractingPartitioned) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[320,25,64,128] parameter(0)
  %lhs.copy = f32[320,25,64,128] copy(%lhs), sharding={devices=[1,1,4,1]0,1,2,3}
  %rhs = f32[320,39296,64,128] parameter(1)
  %rhs.copy = f32[320,39296,64,128] copy(%rhs),
    sharding={devices=[1,1,4,1]0,1,2,3}
  ROOT %dot = f32[320,25,39296] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,4,1]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/4,
                           /*conv_halo_exchange_always_on_lhs =*/true,
                           /*choose_faster_windowed_einsum =*/false,
                           /*unroll_windowed_einsum =*/false,
                           /*bidirectional_windowed_einsum =*/true));
  VLOG(1) << module->ToString();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(),
                                op::Constant(), op::Reshape(), op::Constant())),
      op::Shape("f32[320,25,16,128]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(1), op::Constant(),
                                op::Constant(), op::Reshape(), op::Constant())),
      op::Shape("f32[320,39296,16,128]"));
  const auto while_op = AllOf(
      op::While(op::Tuple(lhs, rhs, op::Broadcast(), op::Broadcast(),
                          op::Constant())),
      op::Shape("(f32[320,25,16,128], f32[320,39296,16,128], f32[320,7,39296],"
                " f32[320,7,39296], u32[])"));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, AllOf(op::Add(op::GetTupleElement(while_op),
                          op::CollectivePermute(op::GetTupleElement(while_op))),
                  op::Shape("f32[320,7,39296]")));

  const auto while_loop = root->operand(0)->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  const auto next_i =
      op::Add(op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant()),
              op::Constant());
  const auto partial_dot_pattern =
      AllOf(op::Reshape(op::Slice(
                op::Dot(op::Maximum(), op::GetTupleElement(op::Parameter(0))))),
            op::Shape("f32[320,7,39296]"));
  const auto partial_output_pattern = AllOf(
      op::Add(op::CollectivePermute(op::Add(
                  op::CollectivePermute(op::GetTupleElement(op::Parameter(0))),
                  partial_dot_pattern)),
              partial_dot_pattern),
      op::Shape("f32[320,7,39296]"));

  EXPECT_THAT(
      while_loop->while_body()->root_instruction(),
      op::Tuple(op::GetTupleElement(op::Parameter(0)),
                op::GetTupleElement(op::Parameter(0)), partial_output_pattern,
                partial_output_pattern, next_i));
}

TEST_F(SpmdPartitioningTest,
       EinsumRHSWindowedInContractingOutNonContractingFromBroadcast) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %constant.1 = f32[] constant(2)
  %broadcast = f32[32,25,64,128] broadcast(%constant.1), dimensions={},
    sharding={devices=[1,1,4,1]0,1,2,3}
  %add = f32[32,25,64,128] add(%broadcast, %broadcast),
    sharding={devices=[1,1,4,1]0,1,2,3}
  %rhs = f32[32,39296,64,128] parameter(0)
  %rhs.copy = f32[32,39296,64,128] copy(%rhs),
    sharding={devices=[1,1,4,1]0,1,2,3}
  ROOT %dot = f32[32,25,39296] dot(%add, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,4,1]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, PartitionComputation(hlo_string,
                                                            /*num_devices=*/4));
  VLOG(1) << module->ToString();
  // Involves loop code motion, skips pattern matching.
}

TEST_F(SpmdPartitioningTest,
       EinsumLHSWindowedInContractingOutNonContractingPartitioned) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[16,1024,16384] parameter(0)
  %lhs.copy = f32[16,1024,16384] copy(%lhs),
    sharding={devices=[2,1,4]0,1,2,3,4,5,6,7}
  %rhs = f32[16384,67,128] parameter(1)
  %rhs.copy = f32[16384,67,128] copy(%rhs),
    sharding={devices=[4,1,1,2]0,4,1,5,2,6,3,7 last_tile_dim_replicate}
  ROOT %dot = f32[16,1024,67,128] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={2}, rhs_contracting_dims={0},
    sharding={devices=[2,1,4,1]0,1,2,3,4,5,6,7}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs =
      AllOf(op::Copy(op::DynamicSlice(op::Parameter(0), op::Reshape(),
                                      op::Constant(), op::Reshape())),
            op::Shape("f32[8,1024,4096]"));
  const auto rhs =
      AllOf(op::Copy(op::DynamicSlice(op::Parameter(1), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[4096,67,128]"));
  EXPECT_THAT(
      root,
      AllOf(op::GetTupleElement(op::While(op::Tuple(
                lhs, rhs, op::Broadcast(), op::Broadcast(), op::Constant()))),
            op::Shape("f32[8,1024,17,128]")));

  const auto while_loop = root->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  const auto next_i =
      op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant());
  auto ds =
      AllOf(op::DynamicSlice(
                op::Pad(op::GetTupleElement(op::Parameter(0)), op::Constant()),
                op::Constant(), op::Reshape(), op::Constant()),
            op::Shape("f32[4096,17,128]"));
  auto partial_output =
      AllOf(op::Add(op::GetTupleElement(op::Parameter(0)),
                    op::Dot(op::GetTupleElement(op::Parameter(0)), ds)),
            op::Shape("f32[8,1024,17,128]"));
  auto window = op::Conditional(op::Compare(next_i, op::Constant()),
                                partial_output, partial_output);
  EXPECT_THAT(while_loop->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Parameter(0)),
                        op::GetTupleElement(op::Parameter(0)), window,
                        op::GetTupleElement(op::Parameter(0)), next_i));

  // Check the conditional that contains the collective permute.
  auto cp_conditional =
      while_loop->while_body()->root_instruction()->operand(2);
  EXPECT_THAT(cp_conditional->true_computation()->root_instruction(),
              op::CollectivePermute(op::Parameter(0)));
  EXPECT_THAT(cp_conditional->false_computation()->root_instruction(),
              op::Parameter(0));
}

TEST_F(SpmdPartitioningTest,
       UnrollEinsumLHSWindowedInContractingOutNonContractingPartitioned) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[16,1024,16384] parameter(0)
  %lhs.copy = f32[16,1024,16384] copy(%lhs),
    sharding={devices=[2,1,4]0,1,2,3,4,5,6,7}
  %rhs = f32[16384,67,128] parameter(1)
  %rhs.copy = f32[16384,67,128] copy(%rhs),
    sharding={devices=[4,1,1,2]0,4,1,5,2,6,3,7 last_tile_dim_replicate}
  ROOT %dot = f32[16,1024,67,128] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={2}, rhs_contracting_dims={0},
    sharding={devices=[2,1,4,1]0,1,2,3,4,5,6,7}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/8,
                           /*conv_halo_exchange_always_on_lhs =*/true,
                           /*choose_faster_windowed_einsum =*/false,
                           /*unroll_windowed_einsum =*/true));
  VLOG(1) << module->ToString();

  const auto lhs =
      AllOf(op::Copy(op::DynamicSlice(op::Parameter(0), op::Reshape(),
                                      op::Constant(), op::Reshape())),
            op::Shape("f32[8,1024,4096]"));
  const auto rhs =
      AllOf(op::Copy(op::DynamicSlice(op::Parameter(1), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[4096,67,128]"));
  const auto while_op =
      AllOf(op::While(op::Tuple(lhs, rhs, op::Broadcast(), op::Broadcast(),
                                op::Constant())),
            op::Shape("(f32[8,1024,4096], f32[4096,67,128], f32[8,1024,17,128],"
                      " f32[8,1024,17,128], u32[])"));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, AllOf(op::Add(op::CollectivePermute(op::GetTupleElement(while_op)),
                          op::GetTupleElement(while_op)),
                  op::Shape("f32[8,1024,17,128]")));

  const auto while_loop = root->operand(1)->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  const auto next_i =
      op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant());
  auto ds =
      AllOf(op::DynamicSlice(
                op::Pad(op::GetTupleElement(op::Parameter(0)), op::Constant()),
                op::Constant(), op::Reshape(), op::Constant()),
            op::Shape("f32[4096,17,128]"));
  auto partial_output = AllOf(
      op::Add(op::CollectivePermute(op::GetTupleElement(op::Parameter(0))),
              op::Dot(op::GetTupleElement(op::Parameter(0)), ds)),
      op::Shape("f32[8,1024,17,128]"));
  auto partial_output2 =
      AllOf(op::CollectivePermute(
                op::Add(op::GetTupleElement(op::Parameter(0)),
                        op::Dot(op::GetTupleElement(op::Parameter(0)), ds))),
            op::Shape("f32[8,1024,17,128]"));

  EXPECT_THAT(while_loop->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Parameter(0)),
                        op::GetTupleElement(op::Parameter(0)), partial_output,
                        partial_output2, next_i));
}

TEST_F(
    SpmdPartitioningTest,
    BidirectionalEinsumLHSWindowedInContractingOutNonContractingPartitioned) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[16,1024,16384] parameter(0)
  %lhs.copy = f32[16,1024,16384] copy(%lhs),
    sharding={devices=[2,1,4]0,1,2,3,4,5,6,7}
  %rhs = f32[16384,67,128] parameter(1)
  %rhs.copy = f32[16384,67,128] copy(%rhs),
    sharding={devices=[4,1,1,2]0,4,1,5,2,6,3,7 last_tile_dim_replicate}
  ROOT %dot = f32[16,1024,67,128] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={2}, rhs_contracting_dims={0},
    sharding={devices=[2,1,4,1]0,1,2,3,4,5,6,7}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/8,
                           /*conv_halo_exchange_always_on_lhs =*/true,
                           /*choose_faster_windowed_einsum =*/false,
                           /*unroll_windowed_einsum =*/false,
                           /*bidirectional_windowed_einsum =*/true));
  VLOG(1) << module->ToString();

  const auto lhs =
      AllOf(op::Copy(op::DynamicSlice(op::Parameter(0), op::Reshape(),
                                      op::Constant(), op::Reshape())),
            op::Shape("f32[8,1024,4096]"));
  const auto rhs =
      AllOf(op::Copy(op::DynamicSlice(op::Parameter(1), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[4096,67,128]"));
  const auto while_op =
      AllOf(op::While(op::Tuple(lhs, rhs, op::Broadcast(), op::Broadcast(),
                                op::Constant())),
            op::Shape("(f32[8,1024,4096], f32[4096,67,128], f32[8,1024,17,128],"
                      " f32[8,1024,17,128], u32[])"));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, AllOf(op::Add(op::GetTupleElement(while_op),
                          op::CollectivePermute(op::GetTupleElement(while_op))),
                  op::Shape("f32[8,1024,17,128]")));

  const auto while_loop = root->operand(0)->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  const auto next_i =
      op::Add(op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant()),
              op::Constant());
  const auto partial_dot_pattern =
      AllOf(op::Reshape(op::Slice(
                op::Dot(op::GetTupleElement(op::Parameter(0)), op::Maximum()))),
            op::Shape("f32[8,1024,17,128]"));
  const auto partial_output_pattern = AllOf(
      op::Add(op::CollectivePermute(op::Add(
                  op::CollectivePermute(op::GetTupleElement(op::Parameter(0))),
                  partial_dot_pattern)),
              partial_dot_pattern),
      op::Shape("f32[8,1024,17,128]"));

  EXPECT_THAT(
      while_loop->while_body()->root_instruction(),
      op::Tuple(op::GetTupleElement(op::Parameter(0)),
                op::GetTupleElement(op::Parameter(0)), partial_output_pattern,
                partial_output_pattern, next_i));
}

TEST_F(SpmdPartitioningTest,
       EinsumLHSWindowedInContractingOutNonContractingPartitioned2) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[16,1024,16384] parameter(0)
  %lhs.copy = f32[16,1024,16384] copy(%lhs),
    sharding={devices=[2,1,4]0,1,2,3,4,5,6,7}
  %rhs = f32[16384,2,33,128] parameter(1)
  %rhs.copy = f32[16384,2,33,128] copy(%rhs),
    sharding={devices=[4,1,1,1,2]0,4,1,5,2,6,3,7 last_tile_dim_replicate}
  ROOT %dot = f32[16,1024,2,33,128] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={2}, rhs_contracting_dims={0},
    sharding={devices=[2,1,2,2,1]0,1,2,3,4,5,6,7}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs =
      AllOf(op::Copy(op::DynamicSlice(op::Parameter(0), op::Reshape(),
                                      op::Constant(), op::Reshape())),
            op::Shape("f32[8,1024,4096]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(1), op::Reshape(), op::Constant(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[4096,2,33,128]"));
  EXPECT_THAT(
      root,
      AllOf(op::GetTupleElement(op::While(op::Tuple(
                lhs, rhs, op::Broadcast(), op::Broadcast(), op::Constant()))),
            op::Shape("f32[8,1024,1,17,128]")));

  const auto while_loop = root->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  const auto next_i =
      op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant());
  auto ds =
      AllOf(op::DynamicSlice(
                op::Pad(op::GetTupleElement(op::Parameter(0)), op::Constant()),
                op::Constant(), op::Reshape(), op::Reshape(), op::Constant()),
            op::Shape("f32[4096,1,17,128]"));
  auto partial_output =
      AllOf(op::Add(op::GetTupleElement(op::Parameter(0)),
                    op::Dot(op::GetTupleElement(op::Parameter(0)), ds)),
            op::Shape("f32[8,1024,1,17,128]"));
  auto window = op::Conditional(op::Compare(next_i, op::Constant()),
                                partial_output, partial_output);
  EXPECT_THAT(while_loop->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Parameter(0)),
                        op::GetTupleElement(op::Parameter(0)), window,
                        op::GetTupleElement(op::Parameter(0)), next_i));

  // Check the conditional that contains the collective permute.
  auto cp_conditional =
      while_loop->while_body()->root_instruction()->operand(2);
  EXPECT_THAT(cp_conditional->true_computation()->root_instruction(),
              op::CollectivePermute(op::Parameter(0)));
  EXPECT_THAT(cp_conditional->false_computation()->root_instruction(),
              op::Parameter(0));
}

TEST_F(SpmdPartitioningTest, EinsumRHSWindowedNonContractingNoDoubleAG) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,128] parameter(0)
  %lhs.copy = f32[32,24,64,128] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %lhs2 = f32[32,24,64,128] parameter(2)
  %lhs2.copy = f32[32,24,64,128] copy(%lhs2), sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[32,39295,64,128] parameter(1)
  %rhs.copy = f32[32,39295,64,128] copy(%rhs), sharding={devices=[1,2,1,1]0,1}
  %dot = f32[32,24,39295] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,2,1]0,1}
  %dot2 = f32[32,24,39295] dot(%lhs2.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,2,1]0,1}
  ROOT %t = tuple(%dot, %dot2)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, PartitionComputation(hlo_string,
                                                            /*num_devices=*/2));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Tuple(op::AllReduce(op::DynamicUpdateSlice(
                                  _, op::Dot(_, op::Slice(_)), _, _, _)),
                              op::AllReduce(op::DynamicUpdateSlice(
                                  _, op::Dot(_, op::Slice(_)), _, _, _))));
}

TEST_F(SpmdPartitioningTest, EinsumRHSWindowedNonContractingNoSharedSharding) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,128] parameter(0)
  %lhs.copy = f32[32,24,64,128] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %lhs2 = f32[32,24,64,128] parameter(2)
  %lhs2.copy = f32[32,24,64,128] copy(%lhs2), sharding={devices=[1,1,2,1]0,1}
  %rhs = f32[32,39295,64,128] parameter(1)
  %rhs.copy = f32[32,39295,64,128] copy(%rhs), sharding={devices=[1,2,1,1]0,1}
  %dot = f32[32,24,39295] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,2,1]0,1}
  %dot2 = f32[32,24,39295] dot(%lhs2.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[2,1,1]0,1}
  ROOT %t = tuple(%dot, %dot2)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, PartitionComputation(hlo_string,
                                                            /*num_devices=*/2));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      op::Tuple(op::AllReduce(op::DynamicUpdateSlice(
                    _, op::Slice(op::GetTupleElement(op::While(_))), _, _, _)),
                op::AllReduce(op::DynamicUpdateSlice(
                    _, op::Dot(_, op::Slice(_)), _, _, _))));
}

TEST_F(SpmdPartitioningTest,
       UnrollEinsumRHSWindowedNonContractingNoSharedSharding) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,128] parameter(0)
  %lhs.copy = f32[32,24,64,128] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %lhs2 = f32[32,24,64,128] parameter(2)
  %lhs2.copy = f32[32,24,64,128] copy(%lhs2), sharding={devices=[1,1,2,1]0,1}
  %rhs = f32[32,39295,64,128] parameter(1)
  %rhs.copy = f32[32,39295,64,128] copy(%rhs), sharding={devices=[1,2,1,1]0,1}
  %dot = f32[32,24,39295] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,2,1]0,1}
  %dot2 = f32[32,24,39295] dot(%lhs2.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[2,1,1]0,1}
  ROOT %t = tuple(%dot, %dot2)
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/2,
                           /*conv_halo_exchange_always_on_lhs =*/true,
                           /*choose_faster_windowed_einsum =*/false,
                           /*unroll_windowed_einsum =*/true));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      op::Tuple(op::AllReduce(op::DynamicUpdateSlice(
                    _, op::Slice(op::GetTupleElement(op::While(_))), _, _, _)),
                op::AllReduce(op::DynamicUpdateSlice(
                    _, op::Dot(_, op::Slice(_)), _, _, _))));

  // Tuple<-AllReduce<-DynamicUpdateSlice<-Slice<-GetTupleElement<-While
  const auto while_loop =
      root->operand(0)->operand(0)->operand(1)->operand(0)->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  const auto next_i =
      op::Add(op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant()),
              op::Constant());
  auto intermediate_output = AllOf(
      op::DynamicUpdateSlice(op::GetTupleElement(op::Parameter(0)),
                             op::Dot(op::GetTupleElement(op::Parameter(0)),
                                     op::GetTupleElement(op::Parameter(0))),
                             op::Constant(), op::Constant(), op::Reshape()),
      op::Shape("f32[32,12,39296]"));
  auto output = AllOf(
      op::DynamicUpdateSlice(
          intermediate_output,
          op::Dot(op::GetTupleElement(op::Parameter(0)),
                  op::CollectivePermute(op::GetTupleElement(op::Parameter(0)))),
          op::Constant(), op::Constant(), op::Reshape()),
      op::Shape("f32[32,12,39296]"));

  EXPECT_THAT(while_loop->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Parameter(0)),
                        op::CollectivePermute(op::CollectivePermute(
                            op::GetTupleElement(op::Parameter(0)))),
                        output, op::GetTupleElement(op::Parameter(0)), next_i));
}

TEST_F(SpmdPartitioningTest,
       BidirectionalEinsumRHSWindowedNonContractingNoSharedSharding) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,128] parameter(0)
  %lhs.copy = f32[32,24,64,128] copy(%lhs), sharding={devices=[1,4,1,1]0,1,2,3}
  %lhs2 = f32[32,24,64,128] parameter(2)
  %lhs2.copy = f32[32,24,64,128] copy(%lhs2), sharding={devices=[1,1,4,1]0,1,2,3}
  %rhs = f32[32,39295,64,128] parameter(1)
  %rhs.copy = f32[32,39295,64,128] copy(%rhs), sharding={devices=[1,4,1,1]0,1,2,3}
  %dot = f32[32,24,39295] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,4,1]0,1,2,3}
  %dot2 = f32[32,24,39295] dot(%lhs2.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[4,1,1]0,1,2,3}
  ROOT %t = tuple(%dot, %dot2)
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/4,
                           /*conv_halo_exchange_always_on_lhs =*/true,
                           /*choose_faster_windowed_einsum =*/false,
                           /*unroll_windowed_einsum =*/false,
                           /*bidirectional_windowed_einsum =*/true));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      op::Tuple(op::AllReduce(op::DynamicUpdateSlice(
                    _, op::Slice(op::GetTupleElement(op::While(_))), _, _, _)),
                op::AllReduce(op::DynamicUpdateSlice(
                    _, op::Dot(_, op::Slice(_)), _, _, _))));

  // Tuple<-AllReduce<-DynamicUpdateSlice<-Slice<-GetTupleElement<-While
  const auto while_loop =
      root->operand(0)->operand(0)->operand(1)->operand(0)->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  const auto next_i =
      op::Add(op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant()),
              op::Constant());
  const auto partial_dot_pattern =
      AllOf(op::Reshape(op::Slice(op::Dot(op::GetTupleElement(op::Parameter(0)),
                                          op::Concatenate()))),
            op::Shape("f32[32,6,9824]"));
  auto intermediate_output1 =
      AllOf(op::DynamicUpdateSlice(op::GetTupleElement(op::Parameter(0)),
                                   partial_dot_pattern, op::Constant(),
                                   op::Constant(), op::Reshape()),
            op::Shape("f32[32,6,39296]"));
  auto intermediate_output2 = AllOf(
      op::DynamicUpdateSlice(intermediate_output1, partial_dot_pattern,
                             op::Constant(), op::Constant(), op::Reshape()),
      op::Shape("f32[32,6,39296]"));
  auto intermediate_output3 = AllOf(
      op::DynamicUpdateSlice(intermediate_output2, partial_dot_pattern,
                             op::Constant(), op::Constant(), op::Reshape()),
      op::Shape("f32[32,6,39296]"));
  auto partial_output = AllOf(
      op::DynamicUpdateSlice(intermediate_output3, partial_dot_pattern,
                             op::Constant(), op::Constant(), op::Reshape()),
      op::Shape("f32[32,6,39296]"));

  EXPECT_THAT(while_loop->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Parameter(0)),
                        op::CollectivePermute(op::CollectivePermute(
                            op::GetTupleElement(op::Parameter(0)))),
                        partial_output,
                        op::CollectivePermute(op::CollectivePermute(
                            op::GetTupleElement(op::Parameter(0)))),
                        next_i));
}

TEST_F(SpmdPartitioningTest, EinsumRHSWindowedNonContracting) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,128] parameter(0)
  %lhs.copy = f32[32,24,64,128] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[32,39295,64,128] parameter(1)
  %rhs.copy = f32[32,39295,64,128] copy(%rhs), sharding={devices=[1,2,1,1]0,1}
  ROOT %dot = f32[32,24,39295] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,2,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, PartitionComputation(hlo_string,
                                                            /*num_devices=*/2));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[32,12,64,128]"));
  const auto rhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(1), op::Constant()),
                                      op::Constant(), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[32,19648,64,128]"));
  EXPECT_THAT(root,
              AllOf(op::Slice(AllOf(op::GetTupleElement(op::While(op::Tuple(
                                        lhs, rhs, op::Broadcast(),
                                        op::Broadcast(), op::Constant()))),
                                    op::Shape("f32[32,12,39296]"))),
                    op::Shape("f32[32,12,39295]")));
  const auto while_loop = root->operand(0)->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  const auto next_i =
      op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant());
  auto window = op::Conditional(op::Compare(next_i, op::Constant()),
                                op::GetTupleElement(op::Parameter(0)),
                                op::GetTupleElement(op::Parameter(0)));
  auto partial_output = op::Dot(op::GetTupleElement(op::Parameter(0)),
                                op::GetTupleElement(op::Parameter(0)));
  EXPECT_THAT(
      while_loop->while_body()->root_instruction(),
      op::Tuple(op::GetTupleElement(op::Parameter(0)), window,
                op::DynamicUpdateSlice(op::GetTupleElement(op::Parameter(0)),
                                       partial_output, op::Constant(),
                                       op::Constant(), op::Reshape()),
                op::GetTupleElement(op::Parameter(0)), next_i));

  // Check the conditional that contains the collective permute.
  auto cp_conditional =
      while_loop->while_body()->root_instruction()->operand(1);
  EXPECT_THAT(cp_conditional->true_computation()->root_instruction(),
              op::CollectivePermute(op::Parameter(0)));
  EXPECT_THAT(cp_conditional->false_computation()->root_instruction(),
              op::Parameter(0));
}

TEST_F(SpmdPartitioningTest, UnrollEinsumRHSWindowedNonContracting) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,128] parameter(0)
  %lhs.copy = f32[32,24,64,128] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[32,39295,64,128] parameter(1)
  %rhs.copy = f32[32,39295,64,128] copy(%rhs), sharding={devices=[1,2,1,1]0,1}
  ROOT %dot = f32[32,24,39295] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,2,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/2,
                           /*conv_halo_exchange_always_on_lhs =*/true,
                           /*choose_faster_windowed_einsum =*/false,
                           /*unroll_windowed_einsum =*/true));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[32,12,64,128]"));
  const auto rhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(1), op::Constant()),
                                      op::Constant(), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[32,19648,64,128]"));
  EXPECT_THAT(root,
              AllOf(op::Slice(AllOf(op::GetTupleElement(op::While(op::Tuple(
                                        lhs, rhs, op::Broadcast(),
                                        op::Broadcast(), op::Constant()))),
                                    op::Shape("f32[32,12,39296]"))),
                    op::Shape("f32[32,12,39295]")));

  const auto while_loop = root->operand(0)->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  const auto next_i =
      op::Add(op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant()),
              op::Constant());
  auto intermediate_output = AllOf(
      op::DynamicUpdateSlice(op::GetTupleElement(op::Parameter(0)),
                             op::Dot(op::GetTupleElement(op::Parameter(0)),
                                     op::GetTupleElement(op::Parameter(0))),
                             op::Constant(), op::Constant(), op::Reshape()),
      op::Shape("f32[32,12,39296]"));
  auto output = AllOf(
      op::DynamicUpdateSlice(
          intermediate_output,
          op::Dot(op::GetTupleElement(op::Parameter(0)),
                  op::CollectivePermute(op::GetTupleElement(op::Parameter(0)))),
          op::Constant(), op::Constant(), op::Reshape()),
      op::Shape("f32[32,12,39296]"));

  EXPECT_THAT(while_loop->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Parameter(0)),
                        op::CollectivePermute(op::CollectivePermute(
                            op::GetTupleElement(op::Parameter(0)))),
                        output, op::GetTupleElement(op::Parameter(0)), next_i));
}

TEST_F(SpmdPartitioningTest, BidirectionalEinsumRHSWindowedNonContracting) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,128] parameter(0)
  %lhs.copy = f32[32,24,64,128] copy(%lhs), sharding={devices=[1,4,1,1]0,1,2,3}
  %rhs = f32[32,39295,64,128] parameter(1)
  %rhs.copy = f32[32,39295,64,128] copy(%rhs), sharding={devices=[1,4,1,1]0,1,2,3}
  ROOT %dot = f32[32,24,39295] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,4,1]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/4,
                           /*conv_halo_exchange_always_on_lhs =*/true,
                           /*choose_faster_windowed_einsum =*/false,
                           /*unroll_windowed_einsum =*/false,
                           /*bidirectional_windowed_einsum =*/true));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[32,6,64,128]"));
  const auto rhs =
      AllOf(op::Reshape(op::Copy(op::DynamicSlice(
                op::Pad(op::Parameter(1), op::Constant()), op::Constant(),
                op::Reshape(), op::Constant(), op::Constant()))),
            op::Shape("f32[32,1,9824,64,128]"));
  EXPECT_THAT(
      root,
      AllOf(op::Slice(AllOf(op::GetTupleElement(op::While(op::Tuple(
                                lhs, rhs, op::Broadcast(),
                                op::CollectivePermute(rhs), op::Constant()))),
                            op::Shape("f32[32,6,39296]"))),
            op::Shape("f32[32,6,39295]")));

  const auto while_loop = root->operand(0)->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  const auto next_i =
      op::Add(op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant()),
              op::Constant());
  const auto partial_dot_pattern =
      AllOf(op::Reshape(op::Slice(op::Dot(op::GetTupleElement(op::Parameter(0)),
                                          op::Concatenate()))),
            op::Shape("f32[32,6,9824]"));
  auto intermediate_output1 =
      AllOf(op::DynamicUpdateSlice(op::GetTupleElement(op::Parameter(0)),
                                   partial_dot_pattern, op::Constant(),
                                   op::Constant(), op::Reshape()),
            op::Shape("f32[32,6,39296]"));
  auto intermediate_output2 = AllOf(
      op::DynamicUpdateSlice(intermediate_output1, partial_dot_pattern,
                             op::Constant(), op::Constant(), op::Reshape()),
      op::Shape("f32[32,6,39296]"));
  auto intermediate_output3 = AllOf(
      op::DynamicUpdateSlice(intermediate_output2, partial_dot_pattern,
                             op::Constant(), op::Constant(), op::Reshape()),
      op::Shape("f32[32,6,39296]"));
  auto partial_output = AllOf(
      op::DynamicUpdateSlice(intermediate_output3, partial_dot_pattern,
                             op::Constant(), op::Constant(), op::Reshape()),
      op::Shape("f32[32,6,39296]"));

  EXPECT_THAT(while_loop->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Parameter(0)),
                        op::CollectivePermute(op::CollectivePermute(
                            op::GetTupleElement(op::Parameter(0)))),
                        partial_output,
                        op::CollectivePermute(op::CollectivePermute(
                            op::GetTupleElement(op::Parameter(0)))),
                        next_i));
}

TEST_F(SpmdPartitioningTest, EinsumRHSWindowedContracting) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,63,128] parameter(0)
  %lhs.copy = f32[32,24,63,128] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[32,39296,63,128] parameter(1)
  %rhs.copy = f32[32,39296,63,128] copy(%rhs), sharding={devices=[1,1,2,1]0,1}
  ROOT %dot = f32[32,24,39296] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,2,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, PartitionComputation(hlo_string,
                                                            /*num_devices=*/2));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[32,12,63,128]"));
  const auto rhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(1), op::Constant()),
                                      op::Constant(), op::Constant(),
                                      op::Reshape(), op::Constant())),
            op::Shape("f32[32,39296,32,128]"));
  auto masked_rhs =
      op::Select(op::Compare(), rhs, op::Broadcast(op::Constant()));
  EXPECT_THAT(root, AllOf(op::GetTupleElement(op::While(
                              op::Tuple(lhs, masked_rhs, op::Broadcast(),
                                        op::Broadcast(), op::Constant()))),
                          op::Shape("f32[32,12,39296]")));
  const auto while_loop = root->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  const auto next_i =
      op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant());
  auto window = op::Conditional(op::Compare(next_i, op::Constant()),
                                op::GetTupleElement(op::Parameter(0)),
                                op::GetTupleElement(op::Parameter(0)));
  auto partial_output = op::Dot(
      op::DynamicSlice(
          op::Pad(op::GetTupleElement(op::Parameter(0)), op::Constant()),
          op::Constant(), op::Constant(), op::Reshape(), op::Constant()),
      op::GetTupleElement(op::Parameter(0)));
  EXPECT_THAT(
      while_loop->while_body()->root_instruction(),
      op::Tuple(op::GetTupleElement(op::Parameter(0)), window,
                op::Add(op::GetTupleElement(op::Parameter(0)), partial_output),
                op::GetTupleElement(op::Parameter(0)), next_i));

  // Check the conditional that contains the collective permute.
  auto cp_conditional =
      while_loop->while_body()->root_instruction()->operand(1);
  EXPECT_THAT(cp_conditional->true_computation()->root_instruction(),
              op::CollectivePermute(op::Parameter(0)));
  EXPECT_THAT(cp_conditional->false_computation()->root_instruction(),
              op::Parameter(0));
}

TEST_F(SpmdPartitioningTest, UnrollEinsumRHSWindowedContracting) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,63,128] parameter(0)
  %lhs.copy = f32[32,24,63,128] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[32,39296,63,128] parameter(1)
  %rhs.copy = f32[32,39296,63,128] copy(%rhs), sharding={devices=[1,1,2,1]0,1}
  ROOT %dot = f32[32,24,39296] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,2,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/2,
                           /*conv_halo_exchange_always_on_lhs =*/true,
                           /*choose_faster_windowed_einsum =*/false,
                           /*unroll_windowed_einsum =*/true));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[32,12,63,128]"));
  const auto rhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(1), op::Constant()),
                                      op::Constant(), op::Constant(),
                                      op::Reshape(), op::Constant())),
            op::Shape("f32[32,39296,32,128]"));
  auto masked_rhs =
      op::Select(op::Compare(), rhs, op::Broadcast(op::Constant()));
  EXPECT_THAT(root, AllOf(op::GetTupleElement(op::While(
                              op::Tuple(lhs, masked_rhs, op::Broadcast(),
                                        op::Broadcast(), op::Constant()))),
                          op::Shape("f32[32,12,39296]")));
  const auto while_loop = root->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  const auto next_i =
      op::Add(op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant()),
              op::Constant());
  auto intermediate_output =
      AllOf(op::Add(op::GetTupleElement(op::Parameter(0)),
                    op::Dot(op::DynamicSlice(
                                op::Pad(op::GetTupleElement(op::Parameter(0)),
                                        op::Constant()),
                                op::Constant(), op::Constant(), op::Reshape(),
                                op::Constant()),
                            op::GetTupleElement(op::Parameter(0)))),
            op::Shape("f32[32,12,39296]"));
  auto output = AllOf(
      op::Add(
          intermediate_output,
          op::Dot(
              op::DynamicSlice(op::Pad(op::GetTupleElement(op::Parameter(0)),
                                       op::Constant()),
                               op::Constant(), op::Constant(), op::Reshape(),
                               op::Constant()),
              op::CollectivePermute(op::GetTupleElement(op::Parameter(0))))),
      op::Shape("f32[32,12,39296]"));

  EXPECT_THAT(while_loop->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Parameter(0)),
                        op::CollectivePermute(op::CollectivePermute(
                            op::GetTupleElement(op::Parameter(0)))),
                        output, op::GetTupleElement(op::Parameter(0)), next_i));
}

TEST_F(SpmdPartitioningTest, BidirectionalEinsumRHSWindowedContracting) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,63,128] parameter(0)
  %lhs.copy = f32[32,24,63,128] copy(%lhs), sharding={devices=[1,4,1,1]0,1,2,3}
  %rhs = f32[32,39296,63,128] parameter(1)
  %rhs.copy = f32[32,39296,63,128] copy(%rhs), sharding={devices=[1,1,4,1]0,1,2,3}
  ROOT %dot = f32[32,24,39296] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,4,1]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/4,
                           /*conv_halo_exchange_always_on_lhs =*/true,
                           /*choose_faster_windowed_einsum =*/false,
                           /*unroll_windowed_einsum =*/false,
                           /*bidirectional_windowed_einsum =*/true));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[32,6,63,128]"));
  const auto rhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(1), op::Constant()),
                                      op::Constant(), op::Constant(),
                                      op::Reshape(), op::Constant())),
            op::Shape("f32[32,39296,16,128]"));
  auto masked_rhs = op::Reshape(
      op::Select(op::Compare(), rhs, op::Broadcast(op::Constant())));
  EXPECT_THAT(root,
              AllOf(op::GetTupleElement(op::While(op::Tuple(
                        lhs, masked_rhs, op::Broadcast(),
                        op::CollectivePermute(masked_rhs), op::Constant()))),
                    op::Shape("f32[32,6,39296]")));
  const auto while_loop = root->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  const auto next_i =
      op::Add(op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant()),
              op::Constant());
  auto partial_output =
      AllOf(op::Add(op::Add(op::GetTupleElement(op::Parameter(0)),
                            op::Dot(op::Maximum(), op::Concatenate())),
                    op::Dot(op::Maximum(), op::Concatenate())),
            op::Shape("f32[32,6,39296]"));

  EXPECT_THAT(while_loop->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Parameter(0)),
                        op::CollectivePermute(op::CollectivePermute(
                            op::GetTupleElement(op::Parameter(0)))),
                        partial_output,
                        op::CollectivePermute(op::CollectivePermute(
                            op::GetTupleElement(op::Parameter(0)))),
                        next_i));
}

TEST_F(SpmdPartitioningTest,
       EinsumWindowedNonContractingDimensionsNoCodeMotionWithDependentNodes) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  %lhs = f32[32,24,64,128] parameter(0)
  %lhs.copy = f32[32,24,64,128] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[32,39295,64,128] parameter(1)
  %rhs.copy = f32[32,39295,64,128] copy(%rhs), sharding={devices=[1,2,1,1]0,1}
  %dot = f32[32,24,39295] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,2,1]0,1}
  %constant = f32[] constant(0)
  %constant.1 = f32[] constant(2)
  %constant.2 = f32[] constant(4)
  %broadcast = f32[32,24,39295] broadcast(%constant.1), dimensions={},
    sharding={devices=[1,2,1]0,1}
  %multiply = f32[32,24,39295] multiply(%dot, %broadcast),
    sharding={devices=[1,2,1]0,1}
  %reduce = f32[32,24] reduce(%multiply, %constant), dimensions={2},
    to_apply=sum, sharding={devices=[1,2]0,1}
  %all-reduce = f32[32,24] all-reduce(%reduce),
    to_apply=sum, sharding={devices=[1,2]0,1}
  %broadcast.1 = f32[32,24,39295] broadcast(%all-reduce), dimensions={0,1},
    sharding={devices=[1,2,1]0,1}
  %subtract = f32[32,24,39295] subtract(%multiply, %broadcast.1),
    sharding={devices=[1,2,1]0,1}
  ROOT %reduce.1 = f32[32,24] reduce(%subtract, %constant.2), dimensions={2},
    to_apply=sum, sharding={devices=[1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[32,12,64,128]"));
  const auto rhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(1), op::Constant()),
                                      op::Constant(), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[32,19648,64,128]"));
  const auto while_output =
      AllOf(op::Slice(op::GetTupleElement(op::While(op::Tuple(
                lhs, rhs, op::Broadcast(), op::Broadcast(), op::Constant())))),
            op::Shape("f32[32,12,39295]"));
  // All the multiples, subtracts and reduces should remain in the spmd entry
  // computation.
  const auto multiply =
      AllOf(op::Multiply(while_output, op::Broadcast(op::Constant())),
            op::Shape("f32[32,12,39295]"));
  EXPECT_THAT(
      root,
      AllOf(op::Reduce(
                op::Subtract(multiply, op::Broadcast(op::AllReduce(op::Reduce(
                                           multiply, op::Constant())))),
                op::Constant()),
            op::Shape("f32[32,12]")));

  const auto while_loop =
      root->operand(0)->operand(0)->operand(0)->operand(0)->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body. There is not be any multple, subtract, reduce, etc.
  // that has been moved into the loop body.
  const auto next_i =
      op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant());
  auto output = op::DynamicUpdateSlice(
      op::GetTupleElement(op::Parameter(0)),
      op::Dot(op::GetTupleElement(op::Parameter(0)),
              op::GetTupleElement(op::Parameter(0))),
      op::Constant(), op::Constant(), op::Reshape(op::DynamicSlice()));

  EXPECT_THAT(while_loop->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Parameter(0)),
                        op::Conditional(op::Compare(next_i, op::Constant()),
                                        op::GetTupleElement(op::Parameter(0)),
                                        op::GetTupleElement(op::Parameter(0))),
                        output, op::GetTupleElement(op::Parameter(0)), next_i));
}

TEST_F(SpmdPartitioningTest, EinsumRHSWindowedNonContractingReduce1) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  %lhs = f32[32,24,64,128] parameter(0)
  %lhs.copy = f32[32,24,64,128] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[32,39295,64,128] parameter(1)
  %rhs.copy = f32[32,39295,64,128] copy(%rhs), sharding={devices=[1,2,1,1]0,1}
  %dot = f32[32,24,39295] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,2,1]0,1}
  %constant = f32[] constant(0)
  %constant.1 = f32[] constant(2)
  %broadcast = f32[32,24,39295] broadcast(%constant.1), dimensions={},
    sharding={devices=[1,2,1]0,1}
  %multiply = f32[32,24,39295] multiply(%dot, %broadcast),
    sharding={devices=[1,2,1]0,1}
  ROOT %reduce = f32[32,24] reduce(%multiply, %constant), dimensions={2},
    to_apply=sum, sharding={devices=[1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[32,12,64,128]"));
  const auto rhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(1), op::Constant()),
                                      op::Constant(), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[32,19648,64,128]"));
  auto input_subtuple =
      op::Tuple(op::Constant(), op::Constant(), op::Broadcast(op::Constant()));
  EXPECT_THAT(
      root,
      AllOf(op::GetTupleElement(op::GetTupleElement(op::While(op::Tuple(
                lhs, rhs, input_subtuple, op::Broadcast(), op::Constant())))),
            op::Shape("f32[32,12]")));

  const auto while_loop = root->operand(0)->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  const auto next_i =
      op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant());
  auto output_tuple = op::Tuple(
      op::GetTupleElement(op::GetTupleElement(op::Parameter(0))),
      op::GetTupleElement(op::GetTupleElement(op::Parameter(0))),
      op::Add(op::Reduce(
                  op::Select(op::Compare(),
                             op::Multiply(
                                 op::Dot(op::GetTupleElement(op::Parameter(0)),
                                         op::GetTupleElement(op::Parameter(0))),
                                 op::DynamicSlice()),
                             op::Broadcast()),
                  op::GetTupleElement(op::GetTupleElement(op::Parameter(0)))),
              op::DynamicSlice(
                  op::GetTupleElement(op::GetTupleElement(op::Parameter(0))),
                  op::Constant(), op::Constant())));

  EXPECT_THAT(
      while_loop->while_body()->root_instruction(),
      op::Tuple(op::GetTupleElement(op::Parameter(0)),
                op::Conditional(op::Compare(next_i, op::Constant()),
                                op::GetTupleElement(op::Parameter(0)),
                                op::GetTupleElement(op::Parameter(0))),
                output_tuple, op::GetTupleElement(op::Parameter(0)), next_i));
}

TEST_F(SpmdPartitioningTest, UnrollEinsumRHSWindowedNonContractingReduce1) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  %lhs = f32[32,24,64,128] parameter(0)
  %lhs.copy = f32[32,24,64,128] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[32,39295,64,128] parameter(1)
  %rhs.copy = f32[32,39295,64,128] copy(%rhs), sharding={devices=[1,2,1,1]0,1}
  %dot = f32[32,24,39295] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,2,1]0,1}
  %constant = f32[] constant(0)
  %constant.1 = f32[] constant(2)
  %broadcast = f32[32,24,39295] broadcast(%constant.1), dimensions={},
    sharding={devices=[1,2,1]0,1}
  %multiply = f32[32,24,39295] multiply(%dot, %broadcast),
  sharding={devices=[1,2,1]0,1}
  ROOT %reduce = f32[32,24] reduce(%multiply, %constant), dimensions={2},
    to_apply=sum, sharding={devices=[1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/2,
                           /*conv_halo_exchange_always_on_lhs =*/true,
                           /*choose_faster_windowed_einsum =*/false,
                           /*unroll_windowed_einsum =*/true));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[32,12,64,128]"));
  const auto rhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(1), op::Constant()),
                                      op::Constant(), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[32,19648,64,128]"));
  auto input_subtuple =
      op::Tuple(op::Constant(), op::Constant(), op::Broadcast(op::Constant()));
  EXPECT_THAT(
      root,
      AllOf(op::GetTupleElement(op::GetTupleElement(op::While(op::Tuple(
                lhs, rhs, input_subtuple, op::Broadcast(), op::Constant())))),
            op::Shape("f32[32,12]")));

  const auto while_loop = root->operand(0)->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  const auto next_i =
      op::Add(op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant()),
              op::Constant());
  auto intermediate_output = AllOf(
      op::Add(
          op::Reduce(
              op::Select(op::Compare(),
                         op::Multiply(
                             op::Dot(op::GetTupleElement(op::Parameter(0)),
                                     op::CollectivePermute(op::GetTupleElement(
                                         op::Parameter(0)))),
                             op::DynamicSlice()),
                         op::Broadcast()),
              op::GetTupleElement(op::GetTupleElement(op::Parameter(0)))),
          op::DynamicSlice(
              op::GetTupleElement(op::GetTupleElement(op::Parameter(0))),
              op::Constant(), op::Constant())),
      op::Shape("f32[32,12]"));
  auto output_tuple = op::Tuple(
      op::GetTupleElement(op::GetTupleElement(op::Parameter(0))),
      op::GetTupleElement(op::GetTupleElement(op::Parameter(0))),
      op::Add(op::Reduce(
                  op::Select(op::Compare(),
                             op::Multiply(
                                 op::Dot(op::GetTupleElement(op::Parameter(0)),
                                         op::GetTupleElement(op::Parameter(0))),
                                 op::DynamicSlice()),
                             op::Broadcast()),
                  op::GetTupleElement(op::GetTupleElement(op::Parameter(0)))),
              op::DynamicSlice(intermediate_output, op::Constant(),
                               op::Constant())));

  EXPECT_THAT(
      while_loop->while_body()->root_instruction(),
      op::Tuple(op::GetTupleElement(op::Parameter(0)),
                op::CollectivePermute(op::CollectivePermute(
                    op::GetTupleElement(op::Parameter(0)))),
                output_tuple, op::GetTupleElement(op::Parameter(0)), next_i));
}

TEST_F(SpmdPartitioningTest,
       BidirectionalEinsumRHSWindowedNonContractingReduce1) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  %lhs = f32[32,24,64,128] parameter(0)
  %lhs.copy = f32[32,24,64,128] copy(%lhs), sharding={devices=[1,4,1,1]0,1,2,3}
  %rhs = f32[32,39295,64,128] parameter(1)
  %rhs.copy = f32[32,39295,64,128] copy(%rhs), sharding={devices=[1,4,1,1]0,1,2,3}
  %dot = f32[32,24,39295] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,4,1]0,1,2,3}
  %constant = f32[] constant(0)
  %constant.1 = f32[] constant(2)
  %broadcast = f32[32,24,39295] broadcast(%constant.1), dimensions={},
    sharding={devices=[1,4,1]0,1,2,3}
  %multiply = f32[32,24,39295] multiply(%dot, %broadcast),
  sharding={devices=[1,4,1]0,1,2,3}
  ROOT %reduce = f32[32,24] reduce(%multiply, %constant), dimensions={2},
    to_apply=sum, sharding={devices=[1,4]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/4,
                           /*conv_halo_exchange_always_on_lhs =*/true,
                           /*choose_faster_windowed_einsum =*/false,
                           /*unroll_windowed_einsum =*/false,
                           /*bidirectional_windowed_einsum =*/true));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[32,6,64,128]"));
  const auto rhs =
      AllOf(op::Reshape(op::Copy(op::DynamicSlice(
                op::Pad(op::Parameter(1), op::Constant()), op::Constant(),
                op::Reshape(), op::Constant(), op::Constant()))),
            op::Shape("f32[32,1,9824,64,128]"));
  auto input_subtuple =
      op::Tuple(op::Constant(), op::Constant(), op::Broadcast(op::Constant()));
  EXPECT_THAT(root,
              AllOf(op::GetTupleElement(op::GetTupleElement(op::While(
                        op::Tuple(lhs, rhs, input_subtuple,
                                  op::CollectivePermute(), op::Constant())))),
                    op::Shape("f32[32,6]")));

  const auto while_loop = root->operand(0)->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  const auto next_i =
      op::Add(op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant()),
              op::Constant());
  auto partial_reduce_pattern = AllOf(
      op::Reduce(
          op::Select(op::Compare(),
                     op::Multiply(op::Reshape(op::Slice(op::Dot(
                                      op::GetTupleElement(op::Parameter(0)),
                                      op::Concatenate()))),
                                  op::DynamicSlice()),
                     op::Broadcast()),
          op::GetTupleElement(op::GetTupleElement(op::Parameter(0)))),
      op::Shape("f32[32,6]"));
  auto intermediate_output1 = AllOf(
      op::Add(partial_reduce_pattern,
              op::DynamicSlice(
                  op::GetTupleElement(op::GetTupleElement(op::Parameter(0))),
                  op::Constant(), op::Constant())),
      op::Shape("f32[32,6]"));
  auto intermediate_output2 =
      AllOf(op::Add(partial_reduce_pattern,
                    op::DynamicSlice(intermediate_output1, op::Constant(),
                                     op::Constant())),
            op::Shape("f32[32,6]"));
  auto intermediate_output3 =
      AllOf(op::Add(partial_reduce_pattern,
                    op::DynamicSlice(intermediate_output2, op::Constant(),
                                     op::Constant())),
            op::Shape("f32[32,6]"));
  auto output_tuple =
      op::Tuple(op::GetTupleElement(op::GetTupleElement(op::Parameter(0))),
                op::GetTupleElement(op::GetTupleElement(op::Parameter(0))),
                op::Add(partial_reduce_pattern,
                        op::DynamicSlice(intermediate_output3, op::Constant(),
                                         op::Constant())));

  EXPECT_THAT(while_loop->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Parameter(0)),
                        op::CollectivePermute(op::CollectivePermute(
                            op::GetTupleElement(op::Parameter(0)))),
                        output_tuple,
                        op::CollectivePermute(op::CollectivePermute(
                            op::GetTupleElement(op::Parameter(0)))),
                        next_i));
}

TEST_F(SpmdPartitioningTest, EinsumRHSWindowedNonContractingReduce2) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  %lhs = f32[32,24,64,128] parameter(0)
  %lhs.copy = f32[32,24,64,128] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[32,39295,64,128] parameter(1)
  %rhs.copy = f32[32,39295,64,128] copy(%rhs), sharding={devices=[1,2,1,1]0,1}
  %dot = f32[32,24,39295] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,2,1]0,1}
  %constant = f32[] constant(0)
  %constant.1 = f32[] constant(2)
  %broadcast = f32[32,24,39295] broadcast(%constant.1), dimensions={},
    sharding={devices=[1,2,1]0,1}
  %multiply = f32[32,24,39295] multiply(%dot, %broadcast),
    sharding={devices=[1,2,1]0,1}
  ROOT %reduce = f32[32,39295] reduce(%multiply, %constant), dimensions={1},
    to_apply=sum, sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, PartitionComputation(hlo_string,
                                                            /*num_devices=*/2));
  VLOG(1) << module->ToString();
  // Involves loop code motion, skips pattern matching.
}

TEST_F(SpmdPartitioningTest, UnrollEinsumRHSWindowedNonContractingReduce2) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  %lhs = f32[32,24,64,128] parameter(0)
  %lhs.copy = f32[32,24,64,128] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[32,39295,64,128] parameter(1)
  %rhs.copy = f32[32,39295,64,128] copy(%rhs), sharding={devices=[1,2,1,1]0,1}
  %dot = f32[32,24,39295] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,2,1]0,1}
  %constant = f32[] constant(0)
  %constant.1 = f32[] constant(2)
  %broadcast = f32[32,24,39295] broadcast(%constant.1), dimensions={},
    sharding={devices=[1,2,1]0,1}
  %multiply = f32[32,24,39295] multiply(%dot, %broadcast),
    sharding={devices=[1,2,1]0,1}
  ROOT %reduce = f32[32,39295] reduce(%multiply, %constant), dimensions={1},
    to_apply=sum, sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/2,
                           /*conv_halo_exchange_always_on_lhs =*/true,
                           /*choose_faster_windowed_einsum =*/false,
                           /*unroll_windowed_einsum =*/true));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[32,12,64,128]"));
  const auto rhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(1), op::Constant()),
                                      op::Constant(), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[32,19648,64,128]"));
  auto input_subtuple =
      op::Tuple(op::Constant(), op::Constant(), op::Broadcast(op::Constant()));
  EXPECT_THAT(
      root,
      AllOf(op::AllReduce(op::Slice(op::GetTupleElement(op::GetTupleElement(
                op::While(op::Tuple(lhs, rhs, input_subtuple, op::Broadcast(),
                                    op::Constant())))))),
            op::Shape("f32[32,39295]")));

  // AllReduce<-Slice<-GetTupleElement<-GetTupleElement<-While
  const auto while_loop = root->operand(0)->operand(0)->operand(0)->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  const auto next_i =
      op::Add(op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant()),
              op::Constant());
  auto intermediate_output = AllOf(
      op::DynamicUpdateSlice(
          op::GetTupleElement(op::GetTupleElement(op::Parameter(0))),
          op::Reduce(
              op::Multiply(op::Dot(op::GetTupleElement(op::Parameter(0)),
                                   op::CollectivePermute(
                                       op::GetTupleElement(op::Parameter(0)))),
                           op::DynamicSlice()),
              op::GetTupleElement(op::GetTupleElement(op::Parameter(0)))),
          op::Constant(), op::Reshape()),
      op::Shape("f32[32,39296]"));
  auto output_tuple = op::Tuple(
      op::GetTupleElement(op::GetTupleElement(op::Parameter(0))),
      op::GetTupleElement(op::GetTupleElement(op::Parameter(0))),
      op::DynamicUpdateSlice(
          intermediate_output,
          op::Reduce(
              op::Multiply(op::Dot(op::GetTupleElement(op::Parameter(0)),
                                   op::GetTupleElement(op::Parameter(0))),
                           op::DynamicSlice()),
              op::GetTupleElement(op::GetTupleElement(op::Parameter(0)))),
          op::Constant(), op::Reshape()));

  EXPECT_THAT(
      while_loop->while_body()->root_instruction(),
      op::Tuple(op::GetTupleElement(op::Parameter(0)),
                op::CollectivePermute(op::CollectivePermute(
                    op::GetTupleElement(op::Parameter(0)))),
                output_tuple, op::GetTupleElement(op::Parameter(0)), next_i));
}

TEST_F(SpmdPartitioningTest,
       BidirectionalEinsumRHSWindowedNonContractingReduce2) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  %lhs = f32[32,24,64,128] parameter(0)
  %lhs.copy = f32[32,24,64,128] copy(%lhs), sharding={devices=[1,4,1,1]0,1,2,3}
  %rhs = f32[32,39295,64,128] parameter(1)
  %rhs.copy = f32[32,39295,64,128] copy(%rhs), sharding={devices=[1,4,1,1]0,1,2,3}
  %dot = f32[32,24,39295] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,4,1]0,1,2,3}
  %constant = f32[] constant(0)
  %constant.1 = f32[] constant(2)
  %broadcast = f32[32,24,39295] broadcast(%constant.1), dimensions={},
    sharding={devices=[1,4,1]0,1,2,3}
  %multiply = f32[32,24,39295] multiply(%dot, %broadcast),
    sharding={devices=[1,4,1]0,1,2,3}
  ROOT %reduce = f32[32,39295] reduce(%multiply, %constant), dimensions={1},
    to_apply=sum, sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/4,
                           /*conv_halo_exchange_always_on_lhs =*/true,
                           /*choose_faster_windowed_einsum =*/false,
                           /*unroll_windowed_einsum =*/false,
                           /*bidirectional_windowed_einsum =*/true));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[32,6,64,128]"));
  const auto rhs =
      AllOf(op::Reshape(op::Copy(op::DynamicSlice(
                op::Pad(op::Parameter(1), op::Constant()), op::Constant(),
                op::Reshape(), op::Constant(), op::Constant()))),
            op::Shape("f32[32,1,9824,64,128]"));
  auto input_subtuple =
      op::Tuple(op::Constant(), op::Constant(), op::Broadcast(op::Constant()));
  EXPECT_THAT(
      root, AllOf(op::AllReduce(op::Slice(op::GetTupleElement(
                      op::GetTupleElement(op::While(op::Tuple(
                          lhs, rhs, input_subtuple, op::CollectivePermute(rhs),
                          op::Constant())))))),
                  op::Shape("f32[32,39295]")));

  // AllReduce<-Slice<-GetTupleElement<-GetTupleElement<-While
  const auto while_loop = root->operand(0)->operand(0)->operand(0)->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  const auto next_i =
      op::Add(op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant()),
              op::Constant());
  auto partial_reduce_pattern = AllOf(
      op::Reduce(op::Multiply(op::Reshape(op::Slice(
                                  op::Dot(op::GetTupleElement(op::Parameter(0)),
                                          op::Concatenate()))),
                              op::DynamicSlice(op::Broadcast(), op::Constant(),
                                               op::Constant(), op::Reshape())),
                 op::GetTupleElement(op::GetTupleElement(op::Parameter(0)))),
      op::Shape("f32[32,9824]"));
  auto intermediate_output1 =
      AllOf(op::DynamicUpdateSlice(
                op::GetTupleElement(op::GetTupleElement(op::Parameter(0))),
                partial_reduce_pattern, op::Constant(), op::Reshape()),
            op::Shape("f32[32,39296]"));
  auto intermediate_output2 =
      AllOf(op::DynamicUpdateSlice(intermediate_output1, partial_reduce_pattern,
                                   op::Constant(), op::Reshape()),
            op::Shape("f32[32,39296]"));
  auto intermediate_output3 =
      AllOf(op::DynamicUpdateSlice(intermediate_output2, partial_reduce_pattern,
                                   op::Constant(), op::Reshape()),
            op::Shape("f32[32,39296]"));
  auto output_tuple = op::Tuple(
      op::GetTupleElement(op::GetTupleElement(op::Parameter(0))),
      op::GetTupleElement(op::GetTupleElement(op::Parameter(0))),
      op::DynamicUpdateSlice(intermediate_output3, partial_reduce_pattern,
                             op::Constant(), op::Reshape()));

  EXPECT_THAT(while_loop->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Parameter(0)),
                        op::CollectivePermute(op::CollectivePermute(
                            op::GetTupleElement(op::Parameter(0)))),
                        output_tuple,
                        op::CollectivePermute(op::CollectivePermute(
                            op::GetTupleElement(op::Parameter(0)))),
                        next_i));
}

TEST_F(SpmdPartitioningTest, EinsumRHSWindowedContractingFromBroadcast) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %rhs = f32[32,39296,63,128] parameter(0)
  %rhs.copy = f32[32,39296,63,128] copy(%rhs), sharding={devices=[1,1,2,1]0,1}
  %constant.1 = f32[] constant(2)
  %broadcast = f32[32,24,63,128] broadcast(%constant.1), dimensions={},
    sharding={devices=[1,2,1,1]0,1}
  %add = f32[32,24,63,128] add(%broadcast, %broadcast),
    sharding={devices=[1,2,1,1]0,1}
  ROOT %dot = f32[32,24,39296] dot(%add, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,2,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, PartitionComputation(hlo_string,
                                                            /*num_devices=*/2));
  VLOG(1) << module->ToString();
  // Involves loop code motion, skips pattern matching.
}

TEST_F(SpmdPartitioningTest, UnrollEinsumRHSWindowedContractingFromBroadcast) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %rhs = f32[32,39296,63,128] parameter(0)
  %rhs.copy = f32[32,39296,63,128] copy(%rhs), sharding={devices=[1,1,2,1]0,1}
  %constant.1 = f32[] constant(2)
  %broadcast = f32[32,24,63,128] broadcast(%constant.1), dimensions={},
    sharding={devices=[1,2,1,1]0,1}
  %add = f32[32,24,63,128] add(%broadcast, %broadcast),
    sharding={devices=[1,2,1,1]0,1}
  ROOT %dot = f32[32,24,39296] dot(%add, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,2,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/2,
                           /*conv_halo_exchange_always_on_lhs =*/true,
                           /*choose_faster_windowed_einsum =*/false,
                           /*unroll_windowed_einsum =*/true));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = op::Tuple(op::Constant());
  const auto rhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(0), op::Constant()),
                                      op::Constant(), op::Constant(),
                                      op::Reshape(), op::Constant())),
            op::Shape("f32[32,39296,32,128]"));
  auto masked_rhs =
      op::Select(op::Compare(), rhs, op::Broadcast(op::Constant()));
  EXPECT_THAT(root, AllOf(op::GetTupleElement(op::While(
                              op::Tuple(lhs, masked_rhs, op::Broadcast(),
                                        op::Broadcast(), op::Constant()))),
                          op::Shape("f32[32,12,39296]")));

  const auto while_loop = root->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  const auto next_i =
      op::Add(op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant()),
              op::Constant());
  auto padded_broadcast_sum = op::Pad(
      op::Add(op::Broadcast(
                  op::GetTupleElement(op::GetTupleElement(op::Parameter(0)))),
              op::Broadcast(
                  op::GetTupleElement(op::GetTupleElement(op::Parameter(0))))),
      op::Constant());
  auto intermediate_output =
      AllOf(op::Add(op::GetTupleElement(op::Parameter(0)),
                    op::Dot(op::DynamicSlice(padded_broadcast_sum,
                                             op::Constant(), op::Constant(),
                                             op::Reshape(), op::Constant()),
                            op::GetTupleElement(op::Parameter(0)))),
            op::Shape("f32[32,12,39296]"));
  auto output = AllOf(
      op::Add(
          intermediate_output,
          op::Dot(
              op::DynamicSlice(padded_broadcast_sum, op::Constant(),
                               op::Constant(), op::Reshape(), op::Constant()),
              op::CollectivePermute(op::GetTupleElement(op::Parameter(0))))),
      op::Shape("f32[32,12,39296]"));

  EXPECT_THAT(while_loop->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Parameter(0)),
                        op::CollectivePermute(op::CollectivePermute(
                            op::GetTupleElement(op::Parameter(0)))),
                        output, op::GetTupleElement(op::Parameter(0)), next_i));
}

TEST_F(SpmdPartitioningTest,
       BidirectionalEinsumRHSWindowedContractingFromBroadcast) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %rhs = f32[32,39296,63,128] parameter(0)
  %rhs.copy = f32[32,39296,63,128] copy(%rhs), sharding={devices=[1,1,4,1]0,1,2,3}
  %constant.1 = f32[] constant(2)
  %broadcast = f32[32,24,63,128] broadcast(%constant.1), dimensions={},
    sharding={devices=[1,4,1,1]0,1,2,3}
  %add = f32[32,24,63,128] add(%broadcast, %broadcast),
    sharding={devices=[1,4,1,1]0,1,2,3}
  ROOT %dot = f32[32,24,39296] dot(%add, %rhs.copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
    sharding={devices=[1,4,1]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/4,
                           /*conv_halo_exchange_always_on_lhs =*/true,
                           /*choose_faster_windowed_einsum =*/false,
                           /*unroll_windowed_einsum =*/false,
                           /*bidirectional_windowed_einsum =*/true));
  VLOG(1) << module->ToString();

  auto input_subtuple =
      op::Tuple(op::Constant(), op::Constant(), op::Broadcast(op::Constant()));

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = op::Tuple(op::Constant());
  const auto rhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(0), op::Constant()),
                                      op::Constant(), op::Constant(),
                                      op::Reshape(), op::Constant())),
            op::Shape("f32[32,39296,16,128]"));
  auto masked_rhs = op::Reshape(
      op::Select(op::Compare(), rhs, op::Broadcast(op::Constant())));
  EXPECT_THAT(root,
              AllOf(op::GetTupleElement(op::While(op::Tuple(
                        lhs, masked_rhs, op::Broadcast(),
                        op::CollectivePermute(masked_rhs), op::Constant()))),
                    op::Shape("f32[32,6,39296]")));

  const auto while_loop = root->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  const auto next_i =
      op::Add(op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant()),
              op::Constant());
  auto output =
      AllOf(op::Add(op::Add(op::GetTupleElement(op::Parameter(0)),
                            op::Dot(op::Maximum(), op::Concatenate())),
                    op::Dot(op::Maximum(), op::Concatenate())),
            op::Shape("f32[32,6,39296]"));

  EXPECT_THAT(while_loop->while_body()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Parameter(0)),
                        op::CollectivePermute(op::CollectivePermute(
                            op::GetTupleElement(op::Parameter(0)))),
                        output,
                        op::CollectivePermute(op::CollectivePermute(
                            op::GetTupleElement(op::Parameter(0)))),
                        next_i));
}

TEST_F(SpmdPartitioningTest, EinsumNonContractingDimPartitionOnTwoDims) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = bf16[8,1024,2,1536] parameter(0)
  %lhs.copy = bf16[8,1024,2,1536] copy(lhs),
    sharding={devices=[4,1,2,1]0,1,2,3,4,5,6,7}
  %rhs = bf16[2,1536,512,1] parameter(1)
  %rhs.copy = bf16[2,1536,512,1] copy(rhs),
    sharding={devices=[2,1,2,1,2]0,4,2,6,1,5,3,7 last_tile_dim_replicate}
  ROOT %convolution = bf16[8,1024,512,1] convolution(lhs.copy, rhs.copy),
    window={size=1x2}, dim_labels=0b1f_1io0->0bf1,
    sharding={devices=[4,1,2,1]0,1,2,3,4,5,6,7}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Reshape(), op::Constant(),
                                op::Reshape(), op::Constant())),
      op::Shape("bf16[2,1024,1,1536]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(1), op::Reshape(), op::Constant(),
                                op::Reshape(), op::Constant())),
      op::Shape("bf16[1,1536,256,1]"));

  const auto partial_replicate_rhs =
      AllOf(op::AllReduce(op::DynamicUpdateSlice(
                op::Broadcast(), rhs, op::Constant(), op::Constant(),
                op::Reshape(), op::Constant())),
            op::Shape("bf16[1,1536,512,1]"));
  EXPECT_THAT(
      root,
      AllOf(op::DynamicSlice(
                op::AllReduce(op::Convolution(lhs, partial_replicate_rhs)),
                op::Constant(), op::Constant(), op::Reshape(), op::Constant()),
            op::Shape("bf16[2,1024,256,1]")));
}

TEST_F(SpmdPartitioningTest, EinsumNonContractingDimPartitionOnTwoDims2) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = bf16[8,1024,2,1536] parameter(0)
  %lhs.copy = bf16[8,1024,2,1536] copy(lhs),
    sharding={devices=[4,1,2,1]0,1,2,3,4,5,6,7}
  %rhs = bf16[2,1536,512,1] parameter(1)
  %rhs.copy = bf16[2,1536,512,1] copy(rhs),
    sharding={devices=[2,1,2,1,2]0,2,4,6,1,3,5,7 last_tile_dim_replicate}
  ROOT %convolution = bf16[8,1024,512,1] convolution(lhs.copy, rhs.copy),
    window={size=1x2}, dim_labels=0b1f_1io0->0bf1,
    sharding={devices=[4,1,2,1]0,1,2,3,4,5,6,7}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Reshape(), op::Constant(),
                                op::Reshape(), op::Constant())),
      op::Shape("bf16[2,1024,1,1536]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(1), op::Reshape(), op::Constant(),
                                op::Reshape(), op::Constant())),
      op::Shape("bf16[1,1536,256,1]"));

  const auto partial_replicate_rhs =
      AllOf(op::AllReduce(op::DynamicUpdateSlice(
                op::Broadcast(), rhs, op::Constant(), op::Constant(),
                op::Reshape(), op::Constant())),
            op::Shape("bf16[1,1536,512,1]"));
  EXPECT_THAT(
      root,
      AllOf(op::DynamicSlice(
                op::AllReduce(op::Convolution(lhs, partial_replicate_rhs)),
                op::Constant(), op::Constant(), op::Reshape(), op::Constant()),
            op::Shape("bf16[2,1024,256,1]")));
}

TEST_F(SpmdPartitioningTest, ReplicatedRng) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = s32[] parameter(0)
  %lhs.copy = s32[] copy(%lhs), sharding={replicated}
  %rhs = s32[] parameter(1)
  %rhs.copy = s32[] copy(%rhs), sharding={replicated}
  ROOT %rng = s32[4]{0} rng(%lhs.copy, %rhs.copy),
      distribution=rng_uniform, sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(op::Copy(op::Parameter(0)), op::Shape("s32[]"));
  const auto rhs = AllOf(op::Copy(op::Parameter(1)), op::Shape("s32[]"));
  EXPECT_THAT(
      root,
      AllOf(op::AllReduce(op::Select(
                op::Broadcast(op::Compare(op::PartitionId(), op::Constant())),
                op::Rng(), op::Broadcast(op::Constant()))),
            op::Shape("s32[4]")));
}

TEST_F(SpmdPartitioningTest, ManualRng) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = s32[] parameter(0), sharding={manual}
  %rhs = s32[] parameter(1), sharding={manual}
  ROOT %rng = s32[4]{0} rng(%lhs, %rhs),
      distribution=rng_uniform, sharding={manual}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Rng(op::Parameter(0), op::Parameter(1)),
                          op::Shape("s32[4]")));
}

TEST_F(SpmdPartitioningTest, PartitionedRng) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = s32[] parameter(0)
  %lhs.copy = s32[] copy(%lhs), sharding={replicated}
  %rhs = s32[] parameter(1)
  %rhs.copy = s32[] copy(%rhs), sharding={maximal device=1}
  ROOT %rng = s32[4]{0} rng(%lhs.copy, %rhs.copy),
      distribution=rng_uniform, sharding={devices=[2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(op::Copy(op::Parameter(0)), op::Shape("s32[]"));
  const auto rhs =
      AllOf(op::Copy(op::Copy(op::Parameter(1))), op::Shape("s32[]"));
  EXPECT_THAT(root, AllOf(op::Rng(lhs, op::AllReduce(op::Select(
                                           op::Broadcast(op::Compare()), rhs,
                                           op::Broadcast(op::Constant())))),
                          op::Shape("s32[2]")));
}

TEST_F(SpmdPartitioningTest, PartialReplicatedRng) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = s32[] parameter(0), sharding={replicated}
  %rhs = s32[] parameter(1), sharding={replicated}
  ROOT %rng = s32[8]{0} rng(%lhs, %rhs),
      distribution=rng_uniform,
      sharding={devices=[2,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(op::Parameter(0), op::Shape("s32[]"));
  const auto rhs = AllOf(op::Parameter(1), op::Shape("s32[]"));
  auto partition_id =
      AllOf(op::Reshape(op::DynamicSlice(op::Constant(), op::PartitionId())),
            op::Shape("u32[]"));
  EXPECT_THAT(
      root, AllOf(op::AllReduce(op::Select(
                      op::Broadcast(op::Compare(partition_id, op::Constant())),
                      op::Rng(lhs, rhs), op::Broadcast(op::Constant()))),
                  op::Shape("s32[4]")));
}

TEST_F(SpmdPartitioningTest, ManualPartitionId) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  ROOT %lhs = s32[] partition-id(), sharding={manual}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::PartitionId());
}

TEST_F(SpmdPartitioningTest, DynamicSliceAlongNonPartitionedDimension) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %input = s32[128,64] parameter(0), sharding={devices=[2,1]0,1}
  %index = s32[] parameter(1)
  %trivial_index = s32[] parameter(2)
  ROOT %dynamic-slice = s32[128,2] dynamic-slice(%input, %trivial_index, %index),
    dynamic_slice_sizes={128,2}, sharding={devices=[2,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto input = AllOf(op::Parameter(0), op::Shape("s32[64,64]"));
  EXPECT_THAT(root,
              AllOf(op::DynamicSlice(input, op::Constant(), op::Parameter(1)),
                    op::Shape("s32[64,2]")));
}

TEST_F(SpmdPartitioningTest, DynamicUpdateSliceAlongNonPartitionedDimension) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %input = s32[128,64] parameter(0), sharding={devices=[2,1]0,1}
  %index = s32[] parameter(1)
  %update = s32[128,2] parameter(2)
  %trivial_index = s32[] parameter(3)
  %update.copy = s32[128,2] copy(%update), sharding={devices=[2,1]0,1}
  ROOT %dynamic-update-slice = s32[128,64]
    dynamic-update-slice(%input, %update.copy, %trivial_index, %index),
    sharding={devices=[2,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto input = AllOf(op::Parameter(0), op::Shape("s32[64,64]"));
  auto update = AllOf(op::Copy(op::DynamicSlice(op::Parameter(2), op::Reshape(),
                                                op::Constant())),
                      op::Shape("s32[64,2]"));
  EXPECT_THAT(root, AllOf(op::DynamicUpdateSlice(input, update, op::Constant(),
                                                 op::Parameter(1)),
                          op::Shape("s32[64,64]")));
}

TEST_F(SpmdPartitioningTest, DynamicUpdateSliceAlongPartitionedDimension) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %input = s32[128,64] parameter(0), sharding={devices=[1,2]0,1}
  %index = s32[] parameter(1)
  %constant = s32[] constant(60)
  %update = s32[128,2] parameter(2), sharding={devices=[1,2]0,1}
  ROOT %dynamic-update-slice = s32[128,64]
    dynamic-update-slice(%input, %update, %index, %constant),
    sharding={devices=[1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto input = AllOf(op::Parameter(0), op::Shape("s32[128,32]"));
  auto update = AllOf(
      op::AllReduce(op::DynamicUpdateSlice(op::Broadcast(), op::Parameter(2),
                                           op::Constant(), op::Reshape())),
      op::Shape("s32[128,2]"));

  EXPECT_THAT(root,
              AllOf(op::Select(op::Broadcast(),
                               op::DynamicUpdateSlice(
                                   input, update, op::Constant(), op::Select()),
                               input),
                    op::Shape("s32[128,32]")));
}

TEST_F(SpmdPartitioningTest, DynamicUpdateSliceAlongPartitionedDimension2) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %input = s32[8,790,2] parameter(0),
    sharding={devices=[8,1,1]0,1,2,3,4,5,6,7}
  %index = s32[] parameter(1)
  %constant = s32[] constant(0)
  %update = s32[1,790,2] parameter(2),
    sharding={devices=[8,1,1]0,1,2,3,4,5,6,7}
  ROOT %dynamic-update-slice = s32[8,790,2]
    dynamic-update-slice(%input, %update, %index, %constant, %constant),
    sharding={devices=[8,1,1]0,1,2,3,4,5,6,7}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto input = AllOf(op::Parameter(0), op::Shape("s32[1,790,2]"));
  auto update = AllOf(op::AllReduce(op::Select(
                          op::Broadcast(), op::Parameter(2), op::Broadcast())),
                      op::Shape("s32[1,790,2]"));
  EXPECT_THAT(
      root,
      AllOf(op::Select(op::Broadcast(),
                       op::DynamicUpdateSlice(input, update, op::Select(),
                                              op::Constant(), op::Constant()),
                       input),
            op::Shape("s32[1,790,2]")));
}

TEST_F(SpmdPartitioningTest, DynamicUpdateSlicePartitionSliceAndNonSliceDims) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %input = s32[128,64] parameter(0)
  %input.copy = s32[128,64] copy(%input), sharding={devices=[2,2]0,1,2,3}
  %constant.0 = s32[] constant(0)
  %constant.1 = s32[] constant(60)
  %update = s32[128,2] parameter(1)
  %update.copy = s32[128,2] copy(%update), sharding={devices=[2,2]0,1,2,3}
  ROOT %dynamic-update-slice = s32[128,64]
    dynamic-update-slice(%input.copy, %update.copy, %constant.0, %constant.1),
    sharding={devices=[2,2]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto input = AllOf(op::Copy(op::DynamicSlice(op::Parameter(0), op::Reshape(),
                                               op::Reshape())),
                     op::Shape("s32[64,32]"));
  auto update = AllOf(op::AllReduce(op::DynamicUpdateSlice(
                          op::Broadcast(),
                          op::Copy(op::DynamicSlice(
                              op::Parameter(1), op::Reshape(), op::Reshape())),
                          op::Constant(), op::Reshape())),
                      op::Shape("s32[64,2]"));

  EXPECT_THAT(root,
              AllOf(op::Select(op::Broadcast(),
                               op::DynamicUpdateSlice(
                                   input, update, op::Constant(), op::Select()),
                               input),
                    op::Shape("s32[64,32]")));
}

TEST_F(SpmdPartitioningTest, UnpartitionedGather) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[2,9] parameter(0), sharding={replicated}
  %indices = s32[3] parameter(1), sharding={replicated}
  ROOT %gather = f32[3,9] gather(%input, %indices), offset_dims={1},
    collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1,
    slice_sizes={1,9}, sharding={devices=[1,2]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      AllOf(
          op::DynamicSlice(
              op::Pad(op::Gather(op::Parameter(0), op::Parameter(1)), _), _, _),
          op::Shape("f32[3,5]")));
}

TEST_F(SpmdPartitioningTest, PassthroughGather) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[2,9] parameter(0), sharding={devices=[1,2]0,1}
  %indices = s32[3] parameter(1), sharding={replicated}
  ROOT %gather = f32[3,9] gather(%input, %indices), offset_dims={1},
    collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1,
    slice_sizes={1,9}, sharding={devices=[1,2]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Gather(op::Parameter(0), op::Parameter(1)),
                          op::Shape("f32[3,5]")));
}

TEST_F(SpmdPartitioningTest, PassthroughGather_PartialReplicate) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[2,9] parameter(0),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}
  %indices = s32[3] parameter(1), sharding={replicated}
  ROOT %gather = f32[3,9] gather(%input, %indices), offset_dims={1},
    collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1,
    slice_sizes={1,9}, sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Gather(op::Parameter(0), op::Parameter(1)),
                          op::Shape("f32[3,5]")));
}

TEST_F(SpmdPartitioningTest, IndexPassthroughGather) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[2,9,8] parameter(0), sharding={replicated}
  %indices = s32[4,2,4] parameter(1), sharding={devices=[2,1,2]0,1,2,3}
  ROOT %gather = f32[8,4,4] gather(%input, %indices), offset_dims={0},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=1,
    slice_sizes={1,1,8}, sharding={devices=[1,2,2]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Gather(op::Parameter(0), op::Parameter(1)),
                          op::Shape("f32[8,2,2]")));
}

TEST_F(SpmdPartitioningTest, IndexPassthroughGather_PartialReplicate) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[2,9,8] parameter(0), sharding={replicated}
  %indices = s32[4,2,4] parameter(1),
    sharding={devices=[2,1,2,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %gather = f32[8,4,4] gather(%input, %indices), offset_dims={0},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=1,
    slice_sizes={1,1,8},
    sharding={devices=[1,2,2,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Gather(op::Parameter(0), op::Parameter(1)),
                          op::Shape("f32[8,2,2]")));
}

TEST_F(SpmdPartitioningTest, IndexAndOperandPassthroughGather) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[7,12] parameter(0),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}
  %indices = s32[16,2] parameter(1),
    sharding={devices=[2,1,2]0,2,1,3 last_tile_dim_replicate}
  ROOT %gather = f32[16,1,12] gather(%input, %indices),
    offset_dims={1,2}, collapsed_slice_dims={}, start_index_map={0,1},
    index_vector_dim=1, slice_sizes={1,12},
    sharding={devices=[2,1,2]0,2,1,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Gather(op::Parameter(0), op::Parameter(1)),
                          op::Shape("f32[8,1,6]")));
}

TEST_F(SpmdPartitioningTest, IndexPassthroughGatherPartitionedIndexVectorDim) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[2,9,8] parameter(0), sharding={replicated}
  %indices = s32[4,2,4] parameter(1), sharding={devices=[2,2,2]0,1,2,3,4,5,6,7}
  ROOT %gather = f32[8,4,4] gather(%input, %indices), offset_dims={0},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=1,
    slice_sizes={1,1,8},
    sharding={devices=[1,2,2,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  auto root = module->entry_computation()->root_instruction();
  auto operand = AllOf(op::Shape("f32[2,9,8]"), op::Parameter(0));
  auto indices = AllOf(op::Shape("s32[2,2,2]"), op::AllReduce());
  auto gather = AllOf(op::Shape("f32[8,2,2]"), op::Gather(operand, indices));
  VLOG(1) << module->ToString();
  EXPECT_THAT(root, op::CollectivePermute(gather));
}

TEST_F(SpmdPartitioningTest, GatherPartitionedOnTrivialSliceDims) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[17,9] parameter(0), sharding={devices=[2,1]0,1}
  %indices = s32[2,3] parameter(1), sharding={replicated}
  ROOT %gather = f32[2,3,9] gather(%input, %indices), offset_dims={2},
    collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=2,
    slice_sizes={1,9}, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  auto offset =
      op::Reshape(op::DynamicSlice(op::Constant(), op::PartitionId()));
  auto min = AllOf(op::Broadcast(offset), op::Shape("s32[2,3]"));
  auto max = AllOf(op::Broadcast(op::Add(offset, op::Constant())),
                   op::Shape("s32[2,3]"));
  auto clamp = op::Clamp(min, op::Parameter(1), max);
  auto gather = op::Gather(op::Parameter(0), op::Subtract(clamp, min));
  auto mask =
      op::Or(op::Lt(op::Parameter(1), min), op::Gt(op::Parameter(1), max));
  auto masked =
      op::Select(op::Broadcast(mask), op::Broadcast(op::Constant()), gather);
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::AllReduce(masked), op::Shape("f32[2,3,9]")));
}

TEST_F(SpmdPartitioningTest,
       GatherPartitionedOnTrivialSliceDims_PartialReplicate) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[17,9] parameter(0),
    sharding={devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}
  %indices = s32[2,3] parameter(1), sharding={replicated}
  ROOT %gather = f32[2,3,9] gather(%input, %indices), offset_dims={2},
    collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=2,
    slice_sizes={1,9}, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  auto offset =
      op::Reshape(op::DynamicSlice(op::Constant(), op::PartitionId()));
  auto min = AllOf(op::Broadcast(offset), op::Shape("s32[2,3]"));
  auto max = AllOf(op::Broadcast(op::Add(offset, op::Constant())),
                   op::Shape("s32[2,3]"));
  auto clamp = op::Clamp(min, op::Parameter(1), max);
  auto gather = op::Gather(op::Parameter(0), op::Subtract(clamp, min));
  auto mask =
      op::Or(op::Lt(op::Parameter(1), min), op::Gt(op::Parameter(1), max));
  auto masked =
      op::Select(op::Broadcast(mask), op::Broadcast(op::Constant()), gather);
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::AllReduce(masked), op::Shape("f32[2,3,9]")));
}

TEST_F(SpmdPartitioningTest, UnpartitionedScatter) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %input = f32[2,9] parameter(0), sharding={replicated}
  %indices = s32[3] parameter(1), sharding={replicated}
  %updates = f32[3,9] parameter(2), sharding={replicated}
  ROOT %scatter = f32[2,9] scatter(%input, %indices, %updates),
      to_apply=add,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1, sharding={devices=[1,2]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              AllOf(op::DynamicSlice(
                        op::Pad(op::Scatter(op::Parameter(0), op::Parameter(1),
                                            op::Parameter(2)),
                                _),
                        _, _),
                    op::Shape("f32[2,5]")));
}

TEST_F(SpmdPartitioningTest, PassthroughScatter) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %input = f32[2,9] parameter(0), sharding={devices=[1,2]0,1}
  %indices = s32[3] parameter(1), sharding={replicated}
  %updates = f32[3,9] parameter(2), sharding={devices=[1,2]0,1}
  ROOT %scatter = f32[2,9] scatter(%input, %indices, %updates),
      to_apply=add,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1, sharding={devices=[1,2]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Scatter(op::Parameter(0), op::Parameter(1),
                                      op::Parameter(2)),
                          op::Shape("f32[2,5]")));
}

TEST_F(SpmdPartitioningTest, PassthroughScatterVariadic) {
  absl::string_view hlo_string = R"(
HloModule module

add_min_max {
  lhs0 = f32[] parameter(0)
  lhs1 = f32[] parameter(1)
  rhs0 = f32[] parameter(2)
  rhs1 = f32[] parameter(3)
  min = minimum(rhs0, rhs1)
  max = maximum(rhs0, rhs1)
  min_sum = add(lhs0, min)
  max_sum = add(lhs1, max)
  ROOT tuple = tuple(min_sum, max_sum)
}

ENTRY entry {
  %input0 = f32[2,9] parameter(0), sharding={devices=[1,2]0,1}
  %input1 = f32[2,9] parameter(1), sharding={devices=[1,2]0,1}
  %indices = s32[3] parameter(2), sharding={replicated}
  %updates0 = f32[3,9] parameter(3), sharding={devices=[1,2]0,1}
  %updates1 = f32[3,9] parameter(4), sharding={devices=[1,2]0,1}
  ROOT %scatter = (f32[2,9], f32[2,9])
    scatter(%input0, %input1, %indices, %updates0, %updates1),
      to_apply=add_min_max, update_window_dims={1}, inserted_window_dims={0},
      scatter_dims_to_operand_dims={0}, index_vector_dim=1,
      sharding={{devices=[1,2]0,1},{devices=[1,2]0,1}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Scatter(op::Parameter(0), op::Parameter(1),
                                      op::Parameter(2), op::Parameter(3),
                                      op::Parameter(4)),
                          op::Shape("(f32[2,5], f32[2,5])")));
}

TEST_F(SpmdPartitioningTest, PassthroughScatter_PartialReplicate) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %input = f32[2,9] parameter(0),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}
  %indices = s32[3] parameter(1), sharding={replicated}
  %updates = f32[3,9] parameter(2),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}
  ROOT %scatter = f32[2,9] scatter(%input, %indices, %updates),
      to_apply=add,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1,
      sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Scatter(op::Parameter(0), op::Parameter(1),
                                      op::Parameter(2)),
                          op::Shape("f32[2,5]")));
}

TEST_F(SpmdPartitioningTest, PassthroughScatterVariadic_PartialReplicate) {
  absl::string_view hlo_string = R"(
HloModule module

add_min_max {
  lhs0 = f32[] parameter(0)
  lhs1 = f32[] parameter(1)
  rhs0 = f32[] parameter(2)
  rhs1 = f32[] parameter(3)
  min = minimum(rhs0, rhs1)
  max = maximum(rhs0, rhs1)
  min_sum = add(lhs0, min)
  max_sum = add(lhs1, max)
  ROOT tuple = tuple(min_sum, max_sum)
}

ENTRY entry {
  %input0 = f32[2,9] parameter(0),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}
  %input1 = f32[2,9] parameter(1),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}
  %indices = s32[3] parameter(2), sharding={replicated}
  %updates0 = f32[3,9] parameter(3),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}
  %updates1 = f32[3,9] parameter(4),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}
  ROOT %scatter = (f32[2,9], f32[2,9])
    scatter(%input0, %input1, %indices, %updates0, %updates1),
      to_apply=add_min_max, update_window_dims={1}, inserted_window_dims={0},
      scatter_dims_to_operand_dims={0}, index_vector_dim=1,
      sharding={{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate},
                {devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Scatter(op::Parameter(0), op::Parameter(1),
                                      op::Parameter(2), op::Parameter(3),
                                      op::Parameter(4)),
                          op::Shape("(f32[2,5], f32[2,5])")));
}

TEST_F(SpmdPartitioningTest, IndexPassthroughScatter) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %input = f32[2,9,8] parameter(0), sharding={replicated}
  %indices = s32[4,2,4] parameter(1), sharding={devices=[2,1,2]0,1,2,3}
  %updates = f32[4,4,8] parameter(2), sharding={devices=[2,2,1]0,1,2,3}
  ROOT %scatter = f32[2,9,8] scatter(%input, %indices, %updates),
      to_apply=add,
      update_window_dims={2},
      inserted_window_dims={0,1},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      AllOf(op::AllReduce(op::AllReduce(op::Scatter(
                op::Select(op::Broadcast(op::Convert(op::PartitionId())),
                           op::Broadcast(op::Constant()), op::Parameter(0)),
                op::Parameter(1), op::Parameter(2)))),
            op::Shape("f32[2,9,8]")));
}

TEST_F(SpmdPartitioningTest, IndexPassthroughScatter_PartialReplicate) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %input = f32[2,9,8] parameter(0), sharding={replicated}
  %indices = s32[4,2,4] parameter(1),
    sharding={devices=[2,1,2,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %updates = f32[4,4,8] parameter(2),
    sharding={devices=[2,2,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %scatter = f32[2,9,8] scatter(%input, %indices, %updates),
      to_apply=add,
      update_window_dims={2},
      inserted_window_dims={0,1},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      AllOf(op::AllReduce(op::AllReduce(op::Scatter(
                op::Select(op::Broadcast(op::Convert(op::Reshape())),
                           op::Broadcast(op::Constant()), op::Parameter(0)),
                op::Parameter(1), op::Parameter(2)))),
            op::Shape("f32[2,9,8]")));
}

TEST_F(SpmdPartitioningTest, IndexPassthroughScatterPartitionedIndexVectorDim) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %input = f32[2,9,8] parameter(0), sharding={replicated}
  %indices = s32[4,2,4] parameter(1), sharding={devices=[2,2,2]0,1,2,3,4,5,6,7}
  %updates = f32[4,4,8] parameter(2),
    sharding={devices=[2,2,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %scatter = f32[2,9,8] scatter(%input, %indices, %updates),
      to_apply=add,
      update_window_dims={2},
      inserted_window_dims={0,1},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  auto operand = AllOf(op::Shape("f32[2,9,8]"), op::Select());
  auto indices = AllOf(op::Shape("s32[2,2,2]"), op::AllReduce());
  auto update = AllOf(op::Shape("f32[2,2,8]"), op::CollectivePermute());
  auto scatter =
      AllOf(op::Shape("f32[2,9,8]"), op::Scatter(operand, indices, update));
  EXPECT_THAT(root, op::AllReduce(op::AllReduce(op::AllReduce(scatter))));
}

TEST_F(SpmdPartitioningTest, IndexPassthroughScatter_Min) {
  absl::string_view hlo_string = R"(
HloModule module

min (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT min = f32[] minimum(lhs, rhs)
}

ENTRY entry {
  %input = f32[2,9,8] parameter(0), sharding={replicated}
  %indices = s32[4,2,4] parameter(1), sharding={devices=[2,1,2]0,1,2,3}
  %updates = f32[4,4,8] parameter(2), sharding={devices=[2,2,1]0,1,2,3}
  ROOT %scatter = f32[2,9,8] scatter(%input, %indices, %updates),
      to_apply=min,
      update_window_dims={2},
      inserted_window_dims={0,1},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      AllOf(op::AllReduce(op::AllReduce(op::Scatter(
                op::Select(op::Broadcast(op::Convert(op::PartitionId())),
                           op::Broadcast(op::Constant()), op::Parameter(0)),
                op::Parameter(1), op::Parameter(2)))),
            op::Shape("f32[2,9,8]")));
}

TEST_F(SpmdPartitioningTest, ScatterPartitionedOnTrivialSliceDims) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %input = f32[17,9] parameter(0), sharding={devices=[2,1]0,1}
  %indices = s32[2,3] parameter(1), sharding={replicated}
  %updates = f32[2,3,9] parameter(2), sharding={replicated}
  ROOT %scatter = f32[17,9] scatter(%input, %indices, %updates),
      to_apply=add,
      update_window_dims={2},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=2, sharding={devices=[2,1]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  auto offset =
      op::Reshape(op::DynamicSlice(op::Constant(), op::PartitionId()));
  auto indices = op::Subtract(
      op::Parameter(1), AllOf(op::Broadcast(offset), op::Shape("s32[2,3]")));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              AllOf(op::Scatter(op::Parameter(0), indices, op::Parameter(2)),
                    op::Shape("f32[9,9]")));
}

TEST_F(SpmdPartitioningTest, ScatterPartitionedOnTrivialSliceDimsVariadic) {
  absl::string_view hlo_string = R"(
HloModule module

add_min_max {
  lhs0 = f32[] parameter(0)
  lhs1 = f32[] parameter(1)
  rhs0 = f32[] parameter(2)
  rhs1 = f32[] parameter(3)
  min = minimum(rhs0, rhs1)
  max = maximum(rhs0, rhs1)
  min_sum = add(lhs0, min)
  max_sum = add(lhs1, max)
  ROOT tuple = tuple(min_sum, max_sum)
}

ENTRY entry {
  %input0 = f32[17,9] parameter(0), sharding={devices=[2,1]0,1}
  %input1 = f32[17,9] parameter(1), sharding={devices=[2,1]0,1}
  %indices = s32[2,3] parameter(2), sharding={replicated}
  %updates0 = f32[2,3,9] parameter(3), sharding={replicated}
  %updates1 = f32[2,3,9] parameter(4), sharding={replicated}
  ROOT %scatter = (f32[17,9], f32[17,9])
    scatter(%input0, %input1, %indices, %updates0, %updates1),
      to_apply=add_min_max, update_window_dims={2}, inserted_window_dims={0},
      scatter_dims_to_operand_dims={0}, index_vector_dim=2,
      sharding={{devices=[2,1]0,1},{devices=[2,1]0,1}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  auto offset =
      op::Reshape(op::DynamicSlice(op::Constant(), op::PartitionId()));
  auto indices = op::Subtract(
      op::Parameter(2), AllOf(op::Broadcast(offset), op::Shape("s32[2,3]")));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              AllOf(op::Scatter(op::Parameter(0), op::Parameter(1), indices,
                                op::Parameter(3), op::Parameter(4)),
                    op::Shape("(f32[9,9], f32[9,9])")));
}

TEST_F(SpmdPartitioningTest,
       ScatterPartitionedOnTrivialSliceDims_PartialReplicate) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %input = f32[17,9] parameter(0),
    sharding={devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}
  %indices = s32[2,3] parameter(1), sharding={replicated}
  %updates = f32[2,3,9] parameter(2), sharding={replicated}
  ROOT %scatter = f32[17,9] scatter(%input, %indices, %updates),
      to_apply=add,
      update_window_dims={2},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=2,
      sharding={devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  auto offset =
      op::Reshape(op::DynamicSlice(op::Constant(), op::PartitionId()));
  auto indices = op::Subtract(
      op::Parameter(1), AllOf(op::Broadcast(offset), op::Shape("s32[2,3]")));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              AllOf(op::Scatter(op::Parameter(0), indices, op::Parameter(2)),
                    op::Shape("f32[9,9]")));
}

TEST_F(SpmdPartitioningTest,
       ScatterPartitionedOnTrivialSliceDimsVariadic_PartialReplicate) {
  absl::string_view hlo_string = R"(
HloModule module

add_min_max {
  lhs0 = f32[] parameter(0)
  lhs1 = f32[] parameter(1)
  rhs0 = f32[] parameter(2)
  rhs1 = f32[] parameter(3)
  min = minimum(rhs0, rhs1)
  max = maximum(rhs0, rhs1)
  min_sum = add(lhs0, min)
  max_sum = add(lhs1, max)
  ROOT tuple = tuple(min_sum, max_sum)
}

ENTRY entry {
  %input0 = f32[17,9] parameter(0),
    sharding={devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}
  %input1 = f32[17,9] parameter(1),
    sharding={devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}
  %indices = s32[2,3] parameter(2), sharding={replicated}
  %updates0 = f32[2,3,9] parameter(3), sharding={replicated}
  %updates1 = f32[2,3,9] parameter(4), sharding={replicated}
  ROOT %scatter = (f32[17,9], f32[17,9])
    scatter(%input0, %input1, %indices, %updates0, %updates1),
      to_apply=add_min_max, update_window_dims={2}, inserted_window_dims={0},
      scatter_dims_to_operand_dims={0}, index_vector_dim=2,
      sharding={{devices=[2,1,2]0,1,2,3 last_tile_dim_replicate},
                {devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  auto offset =
      op::Reshape(op::DynamicSlice(op::Constant(), op::PartitionId()));
  auto indices = op::Subtract(
      op::Parameter(2), AllOf(op::Broadcast(offset), op::Shape("s32[2,3]")));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              AllOf(op::Scatter(op::Parameter(0), op::Parameter(1), indices,
                                op::Parameter(3), op::Parameter(4)),
                    op::Shape("(f32[9,9], f32[9,9])")));
}

TEST_F(SpmdPartitioningTest, TiledReversePassthrough) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  constant = f32[3,3]{1,0} constant({{1,1,1},{1,1,1},{1,1,1}}),
    sharding={devices=[2,1]0,1}
  ROOT reverse = f32[3,3]{1,0} reverse(constant), dimensions={1},
    sharding={devices=[2,1]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("f32[2,3]{1,0}"),
                          op::Reverse(op::DynamicSlice(
                              op::Pad(op::Constant(), op::Constant()),
                              op::Reshape(), op::Constant()))));
}

TEST_F(SpmdPartitioningTest, TiledReversePassthroughViaReversedSharding) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  param = f32[4] parameter(0), sharding={devices=[2]0,1}
  ROOT reverse = f32[4] reverse(param), dimensions={0},
    sharding={devices=[2]1,0}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("f32[2]"), op::Reverse(op::Parameter(0))));
}

TEST_F(SpmdPartitioningTest, TiledReverseSwapShards) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  param = f32[4] parameter(0), sharding={devices=[2]0,1}
  ROOT reverse = f32[4] reverse(param), dimensions={0},
    sharding={devices=[2]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              AllOf(op::Shape("f32[2]"),
                    op::Reverse(op::CollectivePermute(op::Parameter(0)))));
}

TEST_F(SpmdPartitioningTest, TiledReverseHaloExchange) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  param = f32[3] parameter(0), sharding={devices=[2]0,1}
  ROOT reverse = f32[3] reverse(param), dimensions={0},
    sharding={devices=[2]1,0}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  auto halo_exchange_concat =
      op::Concatenate(AllOf(op::Shape("f32[1]"),
                            op::CollectivePermute(op::Slice(op::Parameter(0)))),
                      op::Slice(op::Parameter(0)));
  EXPECT_THAT(root,
              AllOf(op::Shape("f32[2]"), op::Reverse(halo_exchange_concat)));
}

TEST_F(SpmdPartitioningTest, MixWithManualPartitioning) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  param = (f32[8,2], f32[4,2]) parameter(0), sharding={{devices=[2,1]0,1},{manual}}
  param0 = f32[8,2] get-tuple-element(param), index=0, sharding={devices=[2,1]0,1}
  param1 = f32[4,2] get-tuple-element(param), index=1, sharding={manual}
  to_shard = f32[4,2] custom-call(param0), custom_call_target="SPMDFullToShardShape", sharding={manual}
  add = f32[4,2] add(to_shard, param1), sharding={manual}
  to_full = f32[8,2] custom-call(add), custom_call_target="SPMDShardToFullShape", sharding={devices=[2,1]0,1}
  mul = f32[8,2] multiply(to_full, param0), sharding={devices=[2,1]0,1}
  to_shard2 = f32[4,2] custom-call(mul), custom_call_target="SPMDFullToShardShape", sharding={manual}
  ROOT tuple = (f32[4,2]) tuple(to_shard2), sharding={{manual}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  auto p0 = op::GetTupleElement(op::Parameter(0));
  auto to_shard = op::Copy(p0);
  auto p1 = op::GetTupleElement(op::Parameter(0));
  auto mul = AllOf(op::Shape("f32[4,2]"),
                   op::Multiply(op::Copy(op::Add(to_shard, p1)), p0));
  EXPECT_THAT(root, op::Tuple(op::Copy(mul)));
}

TEST_F(SpmdPartitioningTest, SubgroupAllToAllReshard) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[8,8,8,8] parameter(0),
    sharding={devices=[2,2,1,2]0,1,2,3,4,5,6,7}
  ROOT %copy = f32[8,8,8,8] copy(%param0),
    sharding={devices=[1,2,2,2]0,1,4,5,2,3,6,7}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto reshape =
      AllOf(op::Shape("f32[4,4,2,4,4]"), op::Reshape(op::Parameter(0)));
  auto all_to_all = AllOf(op::Shape("f32[4,4,2,4,4]"), op::AllToAll(reshape));
  auto xpose = AllOf(op::Shape("f32[2,4,4,4,4]"), op::Transpose(all_to_all));
  EXPECT_THAT(root,
              op::Copy(AllOf(op::Reshape(xpose), op::Shape("f32[8,4,4,4]"))));
  EXPECT_EQ(root->operand(0)->operand(0)->operand(0)->replica_groups().size(),
            4);
}

TEST_F(SpmdPartitioningTest, SubgroupAllToAllReshard2) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[8,8] parameter(0),
    sharding={devices=[2,4]0,1,2,3,4,5,6,7}
  ROOT %copy = f32[8,8] copy(%param0),
    sharding={devices=[4,2]0,1,4,5,2,3,6,7}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto all_to_all = op::AllToAll(
      AllOf(op::Shape("f32[2,2,2]"), op::Reshape(op::Parameter(0))));
  auto reshape =
      AllOf(op::Shape("f32[2,4]"), op::Reshape(op::Transpose(all_to_all)));
  EXPECT_THAT(root, op::Copy(op::CollectivePermute(reshape)));
}

TEST_F(SpmdPartitioningTest, SubgroupAllToAllReshard3) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[8,8,8] parameter(0),
    sharding={devices=[2,4,1]0,1,2,3,4,5,6,7}
  ROOT %copy = f32[8,8,8] copy(%param0),
    sharding={devices=[1,2,4]0,1,4,5,2,3,6,7}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto all_to_all = op::AllToAll(
      AllOf(op::Shape("f32[4,2,4,2]"), op::Reshape(op::Parameter(0))));
  auto reshape =
      AllOf(op::Shape("f32[4,8,2]"), op::Reshape(op::Transpose(all_to_all)));
  auto all_to_all2 =
      op::AllToAll(AllOf(op::Shape("f32[4,2,4,2]"), op::Reshape(reshape)));
  auto reshape2 =
      AllOf(op::Shape("f32[8,4,2]"), op::Reshape(op::Transpose(all_to_all2)));
  EXPECT_THAT(root, op::Copy(op::CollectivePermute(reshape2)));
}

TEST_F(SpmdPartitioningTest, Dot2DPartitionedNonContractingAndContracting0) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[48,12] parameter(0), sharding={devices=[2,2]0,1,2,3}
  %rhs = f32[32,12] parameter(1), sharding={devices=[2,2]0,2,1,3}
  ROOT %dot = f32[48,32] dot(%lhs, %rhs),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={1}, rhs_contracting_dims={1},
    sharding={devices=[2,2]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto lhs = AllOf(op::Shape("f32[24,6]"), op::Parameter(0));
  auto partial_replicated_lhs =
      AllOf(op::Shape("f32[24,12]"),
            op::AllReduce(op::DynamicUpdateSlice(_, lhs, _, _)));
  const auto rhs = AllOf(op::Shape("f32[16,6]"), op::Parameter(1));
  auto partial_replicated_rhs =
      AllOf(op::Shape("f32[16,12]"),
            op::AllReduce(op::DynamicUpdateSlice(_, rhs, _, _)));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              AllOf(op::Dot(partial_replicated_lhs, partial_replicated_rhs),
                    op::Shape("f32[24,16]")));
}

TEST_F(SpmdPartitioningTest, Dot2DPartitionedNonContractingAndContracting1) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[48,100] parameter(0), sharding={devices=[2,2]0,1,2,3}
  %rhs = f32[32,100] parameter(1), sharding={devices=[2,2]0,1,2,3}
  ROOT %dot = f32[48,32] dot(%lhs, %rhs),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={1}, rhs_contracting_dims={1},
    sharding={devices=[2,2]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto lhs = AllOf(op::Shape("f32[24,50]"), op::Parameter(0));
  const auto rhs = AllOf(op::Shape("f32[16,50]"), op::Parameter(1));
  auto partial_replicated_rhs =
      AllOf(op::Shape("f32[32,50]"),
            op::AllReduce(op::DynamicUpdateSlice(_, rhs, _, _)));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, AllOf(op::Shape("f32[24,16]"),
                  op::DynamicSlice(
                      op::AllReduce(AllOf(op::Dot(lhs, partial_replicated_rhs),
                                          op::Shape("f32[24,32]"))),
                      _, _)));
}

TEST_F(SpmdPartitioningTest, Dot2DPartitionedNonContractingAndContracting2) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[48,100] parameter(0), sharding={replicated}
  %rhs = f32[32,100] parameter(1), sharding={devices=[2,2]0,1,2,3}
  ROOT %dot = f32[48,32] dot(%lhs, %rhs),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={1}, rhs_contracting_dims={1},
    sharding={devices=[2,2]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto lhs = AllOf(op::Shape("f32[48,100]"), op::Parameter(0));
  const auto lhs_slice =
      AllOf(op::Shape("f32[24,100]"), op::DynamicSlice(lhs, _, _));
  const auto rhs = AllOf(op::Shape("f32[16,50]"), op::Parameter(1));
  auto partial_replicated_rhs = AllOf(
      op::Shape("f32[16,100]"), op::AllReduce(op::DynamicUpdateSlice(
                                    _, op::CollectivePermute(rhs), _, _)));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("f32[24,16]"),
                          op::Dot(lhs_slice, partial_replicated_rhs)));
}

TEST_F(SpmdPartitioningTest, Dot2DPartitionedNoncontractingAndContracting3) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[23,24] parameter(0), sharding={devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}
  %rhs = f32[23,32] parameter(1), sharding={devices=[2,2]0,1,2,3}
  ROOT %dot = f32[24,32] dot(%lhs, %rhs),
    lhs_contracting_dims={0}, rhs_contracting_dims={0},
    sharding={devices=[2,2]1,0,3,2}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto lhs = AllOf(op::Shape("f32[12,24]"), op::Parameter(0));
  auto masked_lhs = op::Select(_, lhs, op::Broadcast(op::Constant()));
  const auto rhs = AllOf(op::Shape("f32[12,16]"), op::Parameter(1));
  auto masked_rhs = op::Select(_, rhs, op::Broadcast(op::Constant()));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              AllOf(op::Shape("f32[12,16]"),
                    op::DynamicSlice(
                        AllOf(op::Shape("f32[24,16]"),
                              op::AllReduce(op::Dot(masked_lhs, masked_rhs))),
                        _, _)));
}

TEST_F(SpmdPartitioningTest, Dot2DPartitionedBatchAndNonContracting) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[4,24,100] parameter(0), sharding={devices=[2,2,1]0,1,2,3}
  %rhs = f32[4,32,100] parameter(1), sharding={devices=[2,2,1]0,1,2,3}
  ROOT %dot = f32[4,24,32] dot(%lhs, %rhs),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2},
    sharding={devices=[2,2,1]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto lhs = AllOf(op::Shape("f32[2,12,100]"), op::Parameter(0));
  const auto rhs = AllOf(op::Shape("f32[2,16,100]"), op::Parameter(1));
  auto partial_replicated_rhs =
      AllOf(op::Shape("f32[2,32,100]"),
            op::AllReduce(op::DynamicUpdateSlice(_, rhs, _, _, _)));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("f32[2,12,32]"),
                          op::Dot(lhs, partial_replicated_rhs)));
}

TEST_F(SpmdPartitioningTest, Dot2DPartitionedBatchAndContracting) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[4,24,100] parameter(0), sharding={devices=[2,1,2]0,1,2,3}
  %rhs = f32[4,32,100] parameter(1), sharding={devices=[1,2,2]0,1,2,3}
  ROOT %dot = f32[4,24,32] dot(%lhs, %rhs),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2},
    sharding={devices=[2,2,1]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto lhs = AllOf(op::Shape("f32[2,24,50]"), op::Parameter(0));
  const auto rhs = AllOf(op::Shape("f32[4,16,50]"), op::Parameter(1));
  auto resharded_rhs =
      AllOf(op::Shape("f32[2,32,50]"),
            op::Reshape(op::Transpose(op::AllToAll(op::Reshape(rhs)))));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("f32[2,12,32]"),
                          op::DynamicSlice(
                              AllOf(op::Shape("f32[2,24,32]"),
                                    op::AllReduce(op::Dot(lhs, resharded_rhs))),
                              _, _, _)));
}

TEST_F(SpmdPartitioningTest, Dot2DPartitionedBatchAndContracting2) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[4,24,100] parameter(0), sharding={devices=[2,1,2]0,1,2,3}
  %rhs = f32[4,32,100] parameter(1), sharding={replicated}
  ROOT %dot = f32[4,24,32] dot(%lhs, %rhs),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2},
    sharding={devices=[2,2,1]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto lhs = AllOf(op::Shape("f32[2,24,50]"), op::Parameter(0));
  auto resharded_lhs =
      AllOf(op::Shape("f32[2,12,100]"),
            op::Reshape(op::Transpose(op::AllToAll(op::Reshape(lhs)))));
  const auto rhs = AllOf(op::Shape("f32[4,32,100]"), op::Parameter(1));
  const auto rhs_slice =
      AllOf(op::Shape("f32[2,32,100]"), op::DynamicSlice(rhs, _, _, _));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("f32[2,12,32]"),
                          op::Dot(resharded_lhs, rhs_slice)));
}

TEST_F(SpmdPartitioningTest,
       Dot2DPartitionedBatchNonContractingAndContracting) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[4,24,100] parameter(0), sharding={devices=[2,1,2]0,1,2,3}
  %rhs = f32[4,32,100] parameter(1), sharding={devices=[2,2,1]0,1,2,3}
  ROOT %dot = f32[4,24,32] dot(%lhs, %rhs),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2},
    sharding={devices=[2,1,2]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto lhs = AllOf(op::Shape("f32[2,24,50]"), op::Parameter(0));
  const auto rhs = AllOf(op::Shape("f32[2,16,100]"), op::Parameter(1));
  auto partial_replicated_lhs =
      AllOf(op::Shape("f32[2,24,100]"),
            op::AllReduce(op::DynamicUpdateSlice(_, lhs, _, _, _)));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("f32[2,24,16]"),
                          op::Dot(partial_replicated_lhs, rhs)));
}

TEST_F(SpmdPartitioningTest, Dot2DPartitionedBatchAndReshard) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[4,8,24,100] parameter(0), sharding={devices=[2,1,2,1]0,1,2,3}
  %rhs = f32[4,8,32,100] parameter(1), sharding={devices=[2,1,2,1]0,1,2,3}
  ROOT %dot = f32[4,8,24,32] dot(%lhs, %rhs),
    lhs_batch_dims={0,1}, rhs_batch_dims={0,1},
    lhs_contracting_dims={3}, rhs_contracting_dims={3},
    sharding={devices=[1,2,2,1]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto lhs = AllOf(op::Shape("f32[2,8,12,100]"), op::Parameter(0));
  const auto rhs = AllOf(op::Shape("f32[2,8,16,100]"), op::Parameter(1));
  auto partial_replicated_rhs =
      AllOf(op::Shape("f32[2,8,32,100]"),
            op::AllReduce(op::DynamicUpdateSlice(_, rhs, _, _, _, _)));
  auto dot =
      AllOf(op::Shape("f32[2,8,12,32]"), op::Dot(lhs, partial_replicated_rhs));
  auto reshape = AllOf(op::Shape("f32[2,2,4,12,32]"), op::Reshape(dot));
  auto all_to_all = AllOf(op::Shape("f32[2,2,4,12,32]"), op::AllToAll(reshape));
  auto xpose = AllOf(op::Shape("f32[2,2,4,12,32]"), op::Transpose(all_to_all));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("f32[4,4,12,32]"), op::Reshape(xpose)));
}

TEST_F(SpmdPartitioningTest, SimpleDotPartial) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[2,24,100] parameter(0),
    sharding={devices=[2,1,1,2]0,1,2,3 last_tile_dim_replicate}
  %rhs = f32[2,32,100] parameter(1),
    sharding={devices=[2,1,1,2]0,1,2,3 last_tile_dim_replicate}
  ROOT %dot = f32[2,24,32] dot(%lhs, %rhs),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2},
    sharding={devices=[2,1,1,2]0,1,2,3 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto lhs = AllOf(op::Shape("f32[1,24,100]"), op::Parameter(0));
  const auto rhs = AllOf(op::Shape("f32[1,32,100]"), op::Parameter(1));
  auto dot = AllOf(op::Shape("f32[1,24,32]"), op::Dot(lhs, rhs));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, dot);
}

TEST_F(SpmdPartitioningTest, DotPartialContracting) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[24,100] parameter(0),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}
  %rhs = f32[32,100] parameter(1),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}
  ROOT %dot = f32[24,32] dot(%lhs, %rhs),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={1}, rhs_contracting_dims={1},
    sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto lhs = AllOf(op::Shape("f32[24,50]"), op::Parameter(0));
  const auto rhs = AllOf(op::Shape("f32[32,50]"), op::Parameter(1));
  auto dot = AllOf(op::Shape("f32[24,32]"), op::Dot(lhs, rhs));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::AllReduce(dot));
}

TEST_F(SpmdPartitioningTest, DotPartialContracting2) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[24,100] parameter(0),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}
  %rhs = f32[32,100] parameter(1),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}
  ROOT %dot = f32[24,32] dot(%lhs, %rhs),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={1}, rhs_contracting_dims={1},
    sharding={devices=[2,1,2]0,2,1,3 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto lhs = AllOf(op::Shape("f32[24,50]"), op::Parameter(0));
  const auto rhs = AllOf(op::Shape("f32[32,50]"), op::Parameter(1));
  auto dot =
      AllOf(op::Shape("f32[12,32]"),
            op::Dot(AllOf(op::Shape("f32[12,50]"), op::DynamicSlice(lhs, _, _)),
                    rhs));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::AllReduce(dot));
}

TEST_F(SpmdPartitioningTest, DotPartialContracting3) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[24,100] parameter(0),
    sharding={devices=[1,2,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %rhs = f32[32,100] parameter(1),
    sharding={devices=[1,2,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %dot = f32[24,32] dot(%lhs, %rhs),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={1}, rhs_contracting_dims={1},
    sharding={devices=[1,2,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();

  const auto lhs = AllOf(op::Shape("f32[24,50]"), op::Parameter(0));
  const auto rhs =
      AllOf(op::Shape("f32[16,50]"), op::DynamicSlice(op::Parameter(1), _, _));
  auto dot = AllOf(op::Shape("f32[24,16]"), op::Dot(lhs, rhs));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::CollectivePermute(op::AllReduce(dot)));
}

TEST_F(SpmdPartitioningTest, DotBatchAndPartialContracting) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[4,24,100] parameter(0),
    sharding={devices=[2,2,2]0,1,2,3,4,5,6,7}
  %rhs = f32[4,32,100] parameter(1),
    sharding={devices=[2,1,2,2]0,2,1,3,4,6,5,7 last_tile_dim_replicate}
  ROOT %dot = f32[4,24,32] dot(%lhs, %rhs),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2},
    sharding={devices=[2,2,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();

  const auto lhs = AllOf(op::Shape("f32[2,12,50]"), op::Parameter(0));
  const auto rhs = AllOf(op::Shape("f32[2,32,50]"), op::Parameter(1));
  auto dot = AllOf(op::Shape("f32[2,12,32]"), op::Dot(lhs, rhs));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::AllReduce(dot));
}

TEST_F(SpmdPartitioningTest, DotPartialNonContracting) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[24,8,100] parameter(0),
    sharding={devices=[2,1,1,2]0,1,2,3 last_tile_dim_replicate}
  %rhs = f32[32,100] parameter(1), sharding={devices=[2,2]0,2,1,3}
  ROOT %dot = f32[24,8,32] dot(%lhs, %rhs),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={2}, rhs_contracting_dims={1},
    sharding={devices=[2,1,2]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto lhs = AllOf(op::Shape("f32[12,8,100]"), op::Parameter(0));
  const auto rhs = AllOf(op::Shape("f32[16,50]"), op::Parameter(1));
  auto partially_replicated_rhs =
      AllOf(op::Shape("f32[16,100]"),
            op::AllReduce(op::DynamicUpdateSlice(op::Broadcast(_), rhs, _, _)));
  auto dot =
      AllOf(op::Shape("f32[12,8,16]"), op::Dot(lhs, partially_replicated_rhs));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, dot);
}

TEST_F(SpmdPartitioningTest, DotPartialNonContractingPartialMatch) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[24,8,100] parameter(0), sharding={devices=[2,2,1]0,1,2,3}
  %rhs = f32[32,100] parameter(1),
    sharding={devices=[2,1,2]0,2,1,3 last_tile_dim_replicate}
  ROOT %dot = f32[24,8,32] dot(%lhs, %rhs),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={2}, rhs_contracting_dims={1},
    sharding={devices=[2,1,2]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto lhs = AllOf(op::Shape("f32[12,4,100]"), op::Parameter(0));
  const auto rhs = AllOf(op::Shape("f32[16,100]"), op::Parameter(1));
  auto partially_replicated_lhs = AllOf(
      op::Shape("f32[12,8,100]"),
      op::AllReduce(op::DynamicUpdateSlice(op::Broadcast(_), lhs, _, _, _)));
  auto dot =
      AllOf(op::Shape("f32[12,8,16]"), op::Dot(partially_replicated_lhs, rhs));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, dot);
}

TEST_F(SpmdPartitioningTest, DotPartialContractingPartialMatch) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[24,8,100] parameter(0), sharding={devices=[1,2,2]0,1,2,3}
  %rhs = f32[32,8,100] parameter(1),
    sharding={devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}
  ROOT %dot = f32[24,32] dot(%lhs, %rhs),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={1,2}, rhs_contracting_dims={1,2},
    sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto lhs = AllOf(op::Shape("f32[24,4,50]"), op::Parameter(0));
  const auto rhs = AllOf(op::Shape("f32[32,8,50]"), op::Parameter(1));
  auto dot = AllOf(op::Shape("f32[24,32]"),
                   op::Dot(lhs, AllOf(op::Shape("f32[32,4,50]"),
                                      op::DynamicSlice(rhs, _, _, _))));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::AllReduce(op::AllReduce(dot)));
}

TEST_F(SpmdPartitioningTest, DotNonContractingPartialMatchContractingMatch) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[24,8,100] parameter(0), sharding={devices=[2,1,2]0,1,2,3}
  %rhs = f32[100,50] parameter(1), sharding={devices=[2,2]0,2,1,3}
  ROOT %dot = f32[24,8,50] dot(%lhs, %rhs),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={2}, rhs_contracting_dims={0},
    sharding={devices=[2,2,1]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto lhs = AllOf(op::Shape("f32[12,8,50]"), op::Parameter(0));
  const auto rhs = AllOf(op::Shape("f32[50,25]"), op::Parameter(1));
  auto dot = AllOf(
      op::Shape("f32[12,8,50]"),
      op::Dot(lhs, AllOf(op::Shape("f32[50,50]"),
                         op::AllReduce(op::DynamicUpdateSlice(_, rhs, _, _)))));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("f32[12,4,50]"),
                          op::DynamicSlice(op::AllReduce(dot), _, _, _)))
      << module->ToString();
}

TEST_F(SpmdPartitioningTest, DotLHSMutiNonContractingRHSNotMatch) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[24,8,10] parameter(0), sharding={devices=[2,2,1]0,1,2,3}
  %rhs = f32[10,50] parameter(1),
    sharding={devices=[2,1,2]0,2,1,3 last_tile_dim_replicate}
  ROOT %dot = f32[24,8,50] dot(%lhs, %rhs),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={2}, rhs_contracting_dims={0},
    sharding={devices=[2,2,1]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto lhs = AllOf(op::Shape("f32[12,4,10]"), op::Parameter(0));
  const auto rhs = AllOf(op::Shape("f32[5,50]"), op::Parameter(1));
  auto dot = AllOf(
      op::Shape("f32[12,4,50]"),
      op::Dot(lhs, AllOf(op::Shape("f32[10,50]"),
                         op::AllReduce(op::DynamicUpdateSlice(_, rhs, _, _)))));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, dot) << module->ToString();
}

TEST_F(SpmdPartitioningTest, ElementwiseTest_SubgroupSharding_TileToReplicate) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  constant = f32[6,3]{1,0}
    constant({{1,3,7},{5,1,4},{1,2,8},{2,3,7},{5,2,4},{2,2,8}}),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dims={manual}}
  constant.1 = f32[6,3]{1,0}
    constant({{2,7,2},{2,9,2},{2,6,2},{3,7,2},{2,9,3},{2,3,2}}),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dims={manual}}
   multiply = f32[6,3]{1,0} multiply(constant, constant.1),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dims={manual}}
   ROOT add = f32[6,3]{1,0} add(multiply, constant.1),
    sharding={devices=[1,1,2,2]0,1,2,3 last_tile_dims={replicated, manual}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  auto multiply_lhs =
      AllOf(op::Shape("f32[6,2]"),
            op::DynamicSlice(op::Pad(op::Constant(), op::Constant()),
                             op::Constant(), op::Reshape()));
  auto multiply_rhs =
      AllOf(op::Shape("f32[6,2]"),
            op::DynamicSlice(op::Pad(op::Constant(), op::Constant()),
                             op::Constant(), op::Reshape()));
  auto multiply =
      AllOf(op::Shape("f32[6,2]"), op::Multiply(multiply_lhs, multiply_rhs));
  auto replicated_lhs =
      AllOf(op::Shape("f32[6,3]"),
            op::Slice(op::AllReduce(op::DynamicUpdateSlice(
                op::Broadcast(), multiply, op::Constant(), op::Reshape()))));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("f32[6,3]"),
                          op::Add(replicated_lhs, op::Constant())));
}

TEST_F(SpmdPartitioningTest, ElementwiseTest_SubgroupSharding_ReplicateToTile) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  constant = f32[6,3]{1,0}
    constant({{1,3,7},{5,1,4},{1,2,8},{2,3,7},{5,2,4},{2,2,8}}),
    sharding={devices=[1,1,2,2]0,1,2,3 last_tile_dims={replicated,manual}}
  constant.1 = f32[6,3]{1,0}
    constant({{2,7,2},{2,9,2},{2,6,2},{3,7,2},{2,9,3},{2,3,2}}),
    sharding={devices=[1,1,2,2]0,1,2,3 last_tile_dims={replicated,manual}}
   multiply = f32[6,3]{1,0} multiply(constant, constant.1),
    sharding={devices=[1,1,2,2]0,1,2,3 last_tile_dims={replicated,manual}}
   ROOT add = f32[6,3]{1,0} add(multiply, constant.1),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dims={manual}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  auto multiply = AllOf(op::Shape("f32[6,3]"),
                        op::Multiply(op::Constant(), op::Constant()));
  auto add_lhs = AllOf(op::Shape("f32[6,2]"),
                       op::DynamicSlice(op::Pad(multiply, op::Constant()),
                                        op::Constant(), op::Reshape()));
  auto add_rhs = AllOf(op::Shape("f32[6,2]"),
                       op::DynamicSlice(op::Pad(op::Constant(), op::Constant()),
                                        op::Constant(), op::Reshape()));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("f32[6,2]"), op::Add(add_lhs, add_rhs)));
}

TEST_F(SpmdPartitioningTest,
       ElementwiseTest_PartialReplicateToTiledHaloExchange) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  constant = f32[6,3]{1,0}
    constant({{1,3,7},{5,1,4},{1,2,8},{2,3,7},{5,2,4},{2,2,8}}),
    sharding={replicated}
  constant.1 = f32[6,3]{1,0}
    constant({{2,7,2},{2,9,2},{2,6,2},{3,7,2},{2,9,3},{2,3,2}}),
    sharding={replicated}
  multiply = f32[6,3]{1,0} multiply(constant, constant.1),
    sharding={devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}
  ROOT add = f32[6,3]{1,0} add(multiply, constant.1),
    sharding={devices=[4,1]0,1,2,3}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  auto partial_replicate_lhs =
      AllOf(op::Shape("f32[3,3]"),
            op::DynamicSlice(op::Constant(), op::Reshape(), op::Constant()));
  auto partial_replicate_rhs =
      AllOf(op::Shape("f32[3,3]"),
            op::DynamicSlice(op::Constant(), op::Reshape(), op::Constant()));
  auto multiply =
      AllOf(op::Shape("f32[3,3]"),
            op::Multiply(partial_replicate_lhs, partial_replicate_rhs));
  auto right_halo =
      AllOf(op::Shape("f32[1,3]"), op::CollectivePermute(op::Slice(multiply)));
  auto add_lhs = AllOf(
      op::Shape("f32[2,3]"),
      op::DynamicSlice(
          op::DynamicSlice(
              op::Pad(op::Concatenate(multiply, right_halo), op::Constant()),
              op::Reshape(), op::Constant()),
          op::Subtract(), op::Subtract()));
  auto add_rhs = AllOf(op::Shape("f32[2,3]"),
                       op::DynamicSlice(op::Pad(op::Constant(), op::Constant()),
                                        op::Reshape(), op::Constant()));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("f32[2,3]"), op::Add(add_lhs, add_rhs)));
}

TEST_F(SpmdPartitioningTest, TileToPartialReplicateReshard) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[8,8] parameter(0)
  %copy = f32[8,8] copy(%param0),
    sharding={devices=[2,2]0,1,2,3}
  ROOT %copy0 = f32[8,8] copy(%copy),
    sharding={devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  auto tiled = AllOf(op::Shape("f32[4,4]"),
                     op::Copy(op::DynamicSlice(op::Parameter(0), op::Reshape(),
                                               op::Reshape())));
  auto partially_replicated = AllOf(
      op::Shape("f32[4,8]"), op::Copy(op::AllReduce(op::DynamicUpdateSlice(
                                 op::Broadcast(_), tiled, _, _))));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, partially_replicated);
}

TEST_F(SpmdPartitioningTest, TileToPartialReplicateReshardUnevenPartition) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[8,8] parameter(0),
    sharding={devices=[2,3]0,1,2,3,4,5}
  ROOT %copy0 = f32[8,8] copy(%param0),
    sharding={devices=[1,2,3]0,1,2,3,4,5 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/6));
  VLOG(1) << module->ToString();
  auto tiled = AllOf(op::Shape("f32[4,3]"), op::Parameter(0));
  auto partially_replicated = AllOf(
      op::Shape("f32[8,4]"),
      op::Copy(op::Reshape(
          op::Transpose(op::AllToAll(op::Reshape(op::Slice(op::AllReduce(
              op::DynamicUpdateSlice(op::Broadcast(), tiled, _, _)))))))));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, partially_replicated);
}

TEST_F(SpmdPartitioningTest, PartialReplicateToTileReshardUnevenPartition) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[8,8] parameter(0),
    sharding={devices=[1,2,3]0,1,2,3,4,5 last_tile_dim_replicate}
  ROOT %copy0 = f32[8,8] copy(%param0),
    sharding={devices=[2,3]0,1,2,3,4,5}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/6));
  VLOG(1) << module->ToString();
  auto partial_replicated = AllOf(op::Shape("f32[8,4]"), op::Parameter(0));
  auto tiled = AllOf(
      op::Shape("f32[4,3]"),
      op::Copy(op::DynamicSlice(op::Pad(op::Reshape(op::Transpose(op::AllToAll(
                                            op::Reshape(partial_replicated)))),
                                        _),
                                _, _)));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, tiled);
}

TEST_F(SpmdPartitioningTest, PartialReplicateToTileReshard) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[8,8] parameter(0)
  %copy = f32[8,8] copy(%param0),
    sharding={devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}
  ROOT %copy0 = f32[8,8] copy(%copy),
    sharding={devices=[2,2]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  auto partially_replicated =
      AllOf(op::Shape("f32[4,8]"),
            op::Copy(op::DynamicSlice(op::Parameter(0), op::Reshape(),
                                      op::Constant())));
  auto tiled =
      AllOf(op::Shape("f32[4,4]"),
            op::Copy(op::DynamicSlice(partially_replicated, op::Subtract(),
                                      op::Subtract())));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, tiled);
}

TEST_F(SpmdPartitioningTest,
       PartialReplicateToPartialReplicateReshard_AllReduce) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[8,8] parameter(0)
  %copy = f32[8,8] copy(param0),
    sharding={devices=[2,2,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %copy0 = f32[8,8] copy(%copy),
    sharding={devices=[2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));

  VLOG(1) << module->ToString();
  auto partially_replicated_init =
      AllOf(op::Shape("f32[4,4]"),
            op::Copy(op::DynamicSlice(op::Parameter(0), op::Reshape(),
                                      op::Reshape())));
  auto partially_replicated =
      AllOf(op::Shape("f32[4,8]"),
            op::Copy(op::AllReduce(op::DynamicUpdateSlice(
                op::Broadcast(_), partially_replicated_init, _, _))));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, partially_replicated);
}

TEST_F(SpmdPartitioningTest,
       PartialReplicateToPartialReplicateReshard_DynamicSlice) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[8,8] parameter(0)
  %copy = f32[8,8] copy(%param0),
    sharding={devices=[2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %copy0 = f32[8,8] copy(%copy),
    sharding={devices=[2,2,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  auto partially_replicated =
      AllOf(op::Shape("f32[4,8]"),
            op::Copy(op::DynamicSlice(op::Parameter(0), op::Reshape(),
                                      op::Constant())));
  auto tiled =
      AllOf(op::Shape("f32[4,4]"),
            op::Copy(op::DynamicSlice(partially_replicated, op::Subtract(),
                                      op::Subtract())));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, tiled);
}

TEST_F(SpmdPartitioningTest,
       PartialReplicateToPartialReplicateReshardWithCollectivePermute) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[8,8] parameter(0)
  %copy = f32[8,8] copy(param0),
    sharding={devices=[2,2,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %copy0 = f32[8,8] copy(%copy),
    sharding={devices=[1,2,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));

  VLOG(1) << module->ToString();
  auto partially_replicated_init =
      AllOf(op::Shape("f32[4,4]"),
            op::CollectivePermute(op::Copy(op::DynamicSlice(
                op::Parameter(0), op::Reshape(), op::Reshape()))));
  auto partially_replicated =
      AllOf(op::Shape("f32[8,4]"),
            op::Copy(op::AllReduce(op::DynamicUpdateSlice(
                op::Broadcast(_), partially_replicated_init, _, _))));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, partially_replicated);
}

TEST_F(SpmdPartitioningTest,
       PartialReplicateToPartialReplicateReshardCollectivePermute1) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[8,8] parameter(0)
  %copy = f32[8,8] copy(%param0),
    sharding={devices=[1,2,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %copy0 = f32[8,8] copy(%copy),
    sharding={devices=[2,2,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  auto partially_replicated =
      AllOf(op::Shape("f32[8,4]"),
            op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(),
                                      op::Reshape())));
  auto tiled =
      AllOf(op::Shape("f32[4,4]"),
            op::Copy(op::CollectivePermute(op::DynamicSlice(
                partially_replicated, op::Subtract(), op::Subtract()))));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, tiled);
}

TEST_F(SpmdPartitioningTest,
       PartialReplicateToPartialReplicateReshardHaloExchange) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[6,3] parameter(0)
  %copy = f32[6,3] copy(param0),
    sharding={devices=[4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %copy0 = f32[6,3] copy(%copy),
    sharding={devices=[2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));

  VLOG(1) << module->ToString();
  auto partially_replicated_init =
      AllOf(op::Shape("f32[2,3]"),
            op::Copy(op::DynamicSlice(op::Pad(op::Parameter(0), op::Constant()),
                                      op::Reshape(), op::Constant())));
  auto slice =
      AllOf(op::Shape("f32[2,3]"),
            op::DynamicSlice(op::Concatenate(op::CollectivePermute(op::Slice(
                                                 partially_replicated_init)),
                                             partially_replicated_init),
                             _, _));
  auto partially_replicated =
      AllOf(op::Shape("f32[3,3]"),
            op::Copy(op::Slice(op::AllReduce(
                op::DynamicUpdateSlice(op::Broadcast(_), slice, _, _)))));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, partially_replicated);
}

TEST_F(SpmdPartitioningTest,
       PartialReplicateToPartialReplicateReshardHaloExchange1) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[6,3] parameter(0)
  %copy = f32[6,3] copy(param0),
    sharding={devices=[2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %copy0 = f32[6,3] copy(%copy),
    sharding={devices=[4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));

  VLOG(1) << module->ToString();
  auto partially_replicated_init =
      AllOf(op::Shape("f32[3,3]"),
            op::Copy(op::DynamicSlice(op::Parameter(0), op::Reshape(),
                                      op::Constant())));
  auto slice = AllOf(
      op::Shape("f32[4,3]"),
      op::DynamicSlice(op::Pad(op::Concatenate(partially_replicated_init,
                                               op::CollectivePermute(op::Slice(
                                                   partially_replicated_init))),
                               op::Constant()),
                       _, _));
  auto partially_replicated =
      AllOf(op::Shape("f32[2,3]"), op::Copy(op::DynamicSlice(slice, _, _)));
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, partially_replicated);
}

TEST_F(SpmdPartitioningTest, PartitionConvWithBathGroupCount) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[16,801,1,1024] parameter(0)
  %lhs.copy = f32[16,801,1,1024] copy(%lhs),
    sharding={devices=[1,1,1,2]0,1}
  %rhs = f32[16,801,1,1024] parameter(1)
  %rhs.copy = f32[16,801,1,1024] copy(%rhs),
    sharding={devices=[1,1,1,2]0,1}
  ROOT %conv = f32[5,1,1,1024] convolution(%lhs.copy, %rhs.copy),
    dim_labels=f01b_i01o->01bf,batch_group_count=1024,
    window={size=801x1 pad=2_2x0_0},
    sharding={devices=[1,1,1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));

  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[16,801,1,512]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[16,801,1,512]"));
  EXPECT_THAT(root,
              AllOf(op::Convolution(lhs, rhs), op::Shape("f32[5,1,1,512]")));
}

TEST_F(SpmdPartitioningTest, PartitionConvWithBathGroupCountRHSAlignWithLHS) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[16,801,1,1024] parameter(0)
  %lhs.copy = f32[16,801,1,1024] copy(%lhs),
    sharding={devices=[1,1,1,2]0,1}
  %rhs = f32[16,801,1,1024] parameter(1)
  %rhs.copy = f32[16,801,1,1024] copy(%rhs),
    sharding={devices=[1,2,1,1]0,1}
  ROOT %conv = f32[5,1,1,1024] convolution(%lhs.copy, %rhs.copy),
    dim_labels=f01b_i01o->01bf,batch_group_count=1024,
    window={size=801x1 pad=2_2x0_0},
    sharding={devices=[1,1,1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[16,801,1,512]"));
  const auto rhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(), op::Constant()),
                                      op::Constant(), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[16,401,1,1024]"));
  auto resharded_rhs = AllOf(
      op::Slice(op::Reshape(op::Transpose(op::AllToAll(op::Reshape(rhs))))),
      op::Shape("f32[16,801,1,512]"));
  EXPECT_THAT(root, AllOf(op::Convolution(lhs, resharded_rhs),
                          op::Shape("f32[5,1,1,512]")));
}

TEST_F(SpmdPartitioningTest, PartitionConvWithBathGroupCountLHSAlignWithRHS) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[16,801,1,1024] parameter(0)
  %lhs.copy = f32[16,801,1,1024] copy(%lhs),
    sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[16,801,1,1024] parameter(1)
  %rhs.copy = f32[16,801,1,1024] copy(%rhs),
    sharding={devices=[1,1,1,2]0,1}
  ROOT %conv = f32[5,1,1,1024] convolution(%lhs.copy, %rhs.copy),
    dim_labels=f01b_i01o->01bf,batch_group_count=1024,
    window={size=801x1 pad=2_2x0_0},
    sharding={devices=[1,1,1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[16,801,1,512]"));
  const auto lhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(), op::Constant()),
                                      op::Constant(), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[16,401,1,1024]"));
  auto resharded_lhs = AllOf(
      op::Slice(op::Reshape(op::Transpose(op::AllToAll(op::Reshape(lhs))))),
      op::Shape("f32[16,801,1,512]"));
  EXPECT_THAT(root, AllOf(op::Convolution(resharded_lhs, rhs),
                          op::Shape("f32[5,1,1,512]")));
}

TEST_F(SpmdPartitioningTest,
       PartitionConvWithBathGroupCountOutputAlignWithLHS) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[16,801,1,1024] parameter(0)
  %lhs.copy = f32[16,801,1,1024] copy(%lhs),
    sharding={devices=[1,1,1,2]0,1}
  %rhs = f32[16,801,1,1024] parameter(1)
  %rhs.copy = f32[16,801,1,1024] copy(%rhs),
    sharding={devices=[1,1,1,2]0,1}
  ROOT %conv = f32[5,1,1,1024] convolution(%lhs.copy, %rhs.copy),
    dim_labels=f01b_i01o->01bf,batch_group_count=1024,
    window={size=801x1 pad=2_2x0_0},
    sharding={devices=[2,1,1,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[16,801,1,512]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[16,801,1,512]"));
  auto conv = AllOf(op::Convolution(lhs, rhs), op::Shape("f32[5,1,1,512]"));
  EXPECT_THAT(root, AllOf(op::Reshape(op::Transpose(op::AllToAll(
                              op::Reshape(op::Pad(conv, op::Constant()))))),
                          op::Shape("f32[3,1,1,1024]")));
}

TEST_F(SpmdPartitioningTest,
       PartitionConvWithBathGroupCountOutputAlignWithRHS) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[16,801,1,1024] parameter(0)
  %lhs.copy = f32[16,801,1,1024] copy(%lhs),
    sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[16,801,1,1024] parameter(1)
  %rhs.copy = f32[16,801,1,1024] copy(%rhs),
    sharding={devices=[1,1,1,2]0,1}
  ROOT %conv = f32[5,1,1,1024] convolution(%lhs.copy, %rhs.copy),
    dim_labels=f01b_i01o->01bf,batch_group_count=1024,
    window={size=801x1 pad=2_2x0_0},
    sharding={devices=[2,1,1,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[16,801,1,512]"));
  const auto lhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(), op::Constant()),
                                      op::Constant(), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[16,401,1,1024]"));
  auto resharded_lhs = AllOf(
      op::Slice(op::Reshape(op::Transpose(op::AllToAll(op::Reshape(lhs))))),
      op::Shape("f32[16,801,1,512]"));
  auto conv =
      AllOf(op::Convolution(resharded_lhs, rhs), op::Shape("f32[5,1,1,512]"));
  EXPECT_THAT(root, AllOf(op::Reshape(op::Transpose(op::AllToAll(
                              op::Reshape(op::Pad(conv, op::Constant()))))),
                          op::Shape("f32[3,1,1,1024]")));
}

TEST_F(SpmdPartitioningTest, PartitionConvWithBathGroupAlignWithLHSPartial) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[4,275,64]{2,1,0} parameter(0)
  %multiply.5810 = f32[4,275,64]{2,1,0} copy(lhs), sharding={devices=[2,1,4]0,1,2,3,4,5,6,7}
  %rhs = f32[4,275,64]{2,1,0} parameter(1)
  %copy.25 = f32[4,275,64]{2,1,0} copy(rhs), sharding={devices=[4,1,2]0,1,2,3,4,5,6,7}
  ROOT %convolution.6144 = f32[5,1,64]{2,1,0} convolution(multiply.5810, copy.25), window={size=275 pad=2_2},
    dim_labels=f0b_i0o->0bf, batch_group_count=64,
    operand_precision={HIGH,HIGH}, sharding={devices=[1,4,1,2]0,4,1,5,2,6,3,7 last_tile_dim_replicate}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(op::Shape("f32[4,275,16]"));
  const auto rhs = AllOf(op::Shape("f32[4,275,16]"));
  auto conv = AllOf(op::Convolution(lhs, rhs), op::Shape("f32[5,1,16]"));
  EXPECT_THAT(root, AllOf(op::Reshape(op::Transpose(op::AllToAll(
                              op::Reshape(op::Pad(conv, op::Constant()))))),
                          op::Shape("f32[5,1,64]")));
}

TEST_F(SpmdPartitioningTest,
       PartitionConvWithBathGroupCountAlignWithRHSPartial) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[4,275,64]{2,1,0} parameter(0)
  %multiply.5810 = f32[4,275,64]{2,1,0} copy(lhs), sharding={devices=[4,1,2]0,1,2,3,4,5,6,7}
  %rhs = f32[4,275,64]{2,1,0} parameter(1)
  %copy.25 = f32[4,275,64]{2,1,0} copy(rhs), sharding={devices=[2,1,4]0,1,2,3,4,5,6,7}
  ROOT %convolution.6144 = f32[5,1,64]{2,1,0} convolution(multiply.5810, copy.25), window={size=275 pad=2_2},
    dim_labels=f0b_i0o->0bf, batch_group_count=64,
    operand_precision={HIGH,HIGH}, sharding={devices=[1,4,1,2]0,4,1,5,2,6,3,7 last_tile_dim_replicate}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(op::Shape("f32[4,275,16]"));
  const auto rhs = AllOf(op::Shape("f32[4,275,16]"));
  auto conv = AllOf(op::Convolution(lhs, rhs), op::Shape("f32[5,1,16]"));
  EXPECT_THAT(root, AllOf(op::Reshape(op::Transpose(op::AllToAll(
                              op::Reshape(op::Pad(conv, op::Constant()))))),
                          op::Shape("f32[5,1,64]")));
}

TEST_F(SpmdPartitioningTest,
       PartitionConvWithBathGroupCountAlignWithOutputPartial) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[4,275,64]{2,1,0} parameter(0)
  %multiply.5810 = f32[4,275,64]{2,1,0} copy(lhs), sharding={devices=[4,1,2]0,1,2,3,4,5,6,7}
  %rhs = f32[4,275,64]{2,1,0} parameter(1)
  %copy.25 = f32[4,275,64]{2,1,0} copy(rhs), sharding={devices=[4,1,2]0,1,2,3,4,5,6,7}
  ROOT %convolution.6144 = f32[5,1,64]{2,1,0} convolution(multiply.5810, copy.25), window={size=275 pad=2_2},
    dim_labels=f0b_i0o->0bf, batch_group_count=64,
    operand_precision={HIGH,HIGH}, sharding={devices=[1,1,4,2]0,4,1,5,2,6,3,7 last_tile_dim_replicate}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(op::Shape("f32[4,275,16]"));
  const auto rhs = AllOf(op::Shape("f32[4,275,16]"));
  EXPECT_THAT(root, AllOf(op::Convolution(lhs, rhs), op::Shape("f32[5,1,16]")));
}

TEST_F(SpmdPartitioningTest, PartitionConvWithFeatureGroupCount) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[16,801,1,1024] parameter(0)
  %lhs.copy = f32[16,801,1,1024] copy(%lhs),
    sharding={devices=[1,1,1,2]0,1}
  %rhs = f32[5,1,1,2048] parameter(1)
  %rhs.copy = f32[5,1,1,2048] copy(%rhs),
    sharding={devices=[1,1,1,2]0,1}
  ROOT %conv = f32[16,801,1,2048] convolution(%lhs.copy, %rhs.copy),
    dim_labels=b01f_01io->b01f,feature_group_count=1024,
    window={size=5x1 pad=2_2x0_0},
    sharding={devices=[1,1,1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[16,801,1,512]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[5,1,1,1024]"));
  EXPECT_THAT(
      root, AllOf(op::Convolution(lhs, rhs), op::Shape("f32[16,801,1,1024]")));
}

TEST_F(SpmdPartitioningTest, PartitionConvWithFeatureGroupCount2) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[64,3,1,3072] parameter(0)
  %lhs.copy = f32[64,3,1,3072] copy(%lhs),
    sharding={devices=[1,1,1,4,8]0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23,24,25
    ,26,27,28,29,30,31,8,9,10,11,12,13,14,15 last_tile_dim_replicate}
  %rhs = f32[3,1,1,3072] parameter(1)
  %rhs.copy = f32[3,1,1,3072] copy(%rhs),
    sharding={devices=[1,1,1,4,8]0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23,24,25
    ,26,27,28,29,30,31,8,9,10,11,12,13,14,15 last_tile_dim_replicate}
  ROOT %conv = f32[64,1,1,3072] convolution(%lhs.copy, %rhs.copy),
    dim_labels=b01f_01io->b01f,feature_group_count=3072,
    window={size=3x1},
    sharding={devices=[8,1,1,4]0,16,24,8,2,18,26,10,4,20,28,12,6,22,30,14,7,23,
    31,15,5,21,29,13,3,19,27,11,1,17,25,9}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/32));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs =
      AllOf(op::DynamicSlice(
                op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(),
                                          op::Constant(), op::Constant(),
                                          op::Reshape())),
                op::Reshape(), op::Constant(), op::Constant(), op::Constant()),
            op::Shape("f32[8,3,1,768]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[3,1,1,768]"));
  EXPECT_THAT(root,
              AllOf(op::Convolution(lhs, rhs), op::Shape("f32[8,1,1,768]")));
}

TEST_F(SpmdPartitioningTest,
       PartitionConvWithFeatureGroupCountAlignWithLHSPartial) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[4,275,16]{2,1,0} parameter(0)
  %multiply.5810 = f32[4,275,16]{2,1,0} copy(lhs), sharding={devices=[1,1,4,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %rhs = f32[1,275,16]{2,1,0} parameter(1)
  %copy.25 = f32[1,275,16]{2,1,0} copy(rhs), sharding={devices=[1,1,2,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %convolution.6144 = f32[5,4,16]{2,1,0} convolution(multiply.5810, copy.25), window={size=275 pad=2_2},
    dim_labels=b0f_i0o->0bf, feature_group_count=16,
    operand_precision={HIGH,HIGH}, sharding={devices=[1,1,2,4]0,4,1,5,2,6,3,7 last_tile_dim_replicate}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(op::Shape("f32[4,275,4]"));
  const auto rhs = AllOf(op::Shape("f32[1,275,4]"));
  auto conv = AllOf(op::Convolution(lhs, rhs), op::Shape("f32[5,4,4]"));
  EXPECT_THAT(root, AllOf(op::AllReduce(op::DynamicUpdateSlice(
                              _, op::CollectivePermute(conv), _, _, _)),
                          op::Shape("f32[5,4,8]")));
}

TEST_F(SpmdPartitioningTest,
       PartitionConvWithFeatureGroupCountAlignWithRHSPartial) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[4,275,16]{2,1,0} parameter(0)
  %multiply.5810 = f32[4,275,16]{2,1,0} copy(lhs), sharding={devices=[1,1,2,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %rhs = f32[1,275,16]{2,1,0} parameter(1)
  %copy.25 = f32[1,275,16]{2,1,0} copy(rhs), sharding={devices=[1,1,4,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %convolution.6144 = f32[5,4,16]{2,1,0} convolution(multiply.5810, copy.25), window={size=275 pad=2_2},
    dim_labels=b0f_i0o->0bf, feature_group_count=16,
    operand_precision={HIGH,HIGH}, sharding={devices=[1,1,2,4]0,4,1,5,2,6,3,7 last_tile_dim_replicate}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(op::Shape("f32[4,275,4]"));
  const auto rhs = AllOf(op::Shape("f32[1,275,4]"));
  auto conv = AllOf(op::Convolution(lhs, rhs), op::Shape("f32[5,4,4]"));
  EXPECT_THAT(root, AllOf(op::AllReduce(op::DynamicUpdateSlice(
                              _, op::CollectivePermute(conv), _, _, _)),
                          op::Shape("f32[5,4,8]")));
}

TEST_F(SpmdPartitioningTest,
       PartitionConvWithFeatureGroupCountAlignWithOutputPartial) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[4,275,16]{2,1,0} parameter(0)
  %multiply.5810 = f32[4,275,16]{2,1,0} copy(lhs), sharding={devices=[1,1,2,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %rhs = f32[1,275,16]{2,1,0} parameter(1)
  %copy.25 = f32[1,275,16]{2,1,0} copy(rhs), sharding={devices=[1,1,2,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %convolution.6144 = f32[5,4,16]{2,1,0} convolution(multiply.5810, copy.25), window={size=275 pad=2_2},
    dim_labels=b0f_i0o->0bf, feature_group_count=16,
    operand_precision={HIGH,HIGH}, sharding={devices=[1,1,4,2]0,4,1,5,2,6,3,7 last_tile_dim_replicate}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(op::Shape("f32[4,275,4]"));
  const auto rhs = AllOf(op::Shape("f32[1,275,4]"));
  EXPECT_THAT(root, AllOf(op::Convolution(lhs, rhs), op::Shape("f32[5,4,4]")));
}

TEST_F(SpmdPartitioningTest,
       PartitionConvWithFeatureGroupCountRHSAlignWithLHS) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[16,801,1,1024] parameter(0)
  %lhs.copy = f32[16,801,1,1024] copy(%lhs),
    sharding={devices=[1,1,1,2]0,1}
  %rhs = f32[5,1,1,1024] parameter(1)
  %rhs.copy = f32[5,1,1,1024] copy(%rhs),
    sharding={devices=[2,1,1,1]0,1}
  ROOT %conv = f32[16,801,1,1024] convolution(%lhs.copy, %rhs.copy),
    dim_labels=b01f_01io->b01f,feature_group_count=1024,
    window={size=5x1 pad=2_2x0_0},
    sharding={devices=[1,1,1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[16,801,1,512]"));
  const auto rhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(), op::Constant()),
                                      op::Reshape(), op::Constant(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[3,1,1,1024]"));
  auto resharded_rhs = AllOf(
      op::Slice(op::Reshape(op::Transpose(op::AllToAll(op::Reshape(rhs))))),
      op::Shape("f32[5,1,1,512]"));
  EXPECT_THAT(root, AllOf(op::Convolution(lhs, resharded_rhs),
                          op::Shape("f32[16,801,1,512]")));
}

TEST_F(SpmdPartitioningTest,
       PartitionConvWithFeatureGroupCountLHSAlignWithRHS) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[16,801,1,1024] parameter(0)
  %lhs.copy = f32[16,801,1,1024] copy(%lhs),
    sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[5,1,1,1024] parameter(1)
  %rhs.copy = f32[5,1,1,1024] copy(%rhs),
    sharding={devices=[1,1,1,2]0,1}
  ROOT %conv = f32[16,801,1,1024] convolution(%lhs.copy, %rhs.copy),
    dim_labels=b01f_01io->b01f,feature_group_count=1024,
    window={size=5x1 pad=2_2x0_0},
    sharding={devices=[1,1,1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(), op::Constant()),
                                      op::Constant(), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[16,401,1,1024]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[5,1,1,512]"));
  auto resharded_lhs = AllOf(
      op::Slice(op::Reshape(op::Transpose(op::AllToAll(op::Reshape(lhs))))),
      op::Shape("f32[16,801,1,512]"));
  EXPECT_THAT(root, AllOf(op::Convolution(resharded_lhs, rhs),
                          op::Shape("f32[16,801,1,512]")));
}

TEST_F(SpmdPartitioningTest,
       PartitionConvWithFeatureGroupCountAlignOuputWithLHS) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[16,801,1,1024] parameter(0)
  %lhs.copy = f32[16,801,1,1024] copy(%lhs),
    sharding={devices=[1,1,1,2]0,1}
  %rhs = f32[5,1,1,1024] parameter(1)
  %rhs.copy = f32[5,1,1,1024] copy(%rhs),
    sharding={devices=[1,1,1,2]0,1}
  ROOT %conv = f32[16,801,1,1024] convolution(%lhs.copy, %rhs.copy),
    dim_labels=b01f_01io->b01f,feature_group_count=1024,
    window={size=5x1 pad=2_2x0_0},
    sharding={devices=[2,1,1,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[16,801,1,512]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[5,1,1,512]"));
  auto conv = AllOf(op::Convolution(lhs, rhs), op::Shape("f32[16,801,1,512]"));
  EXPECT_THAT(root,
              AllOf(op::Reshape(op::Transpose(op::AllToAll(op::Reshape(conv)))),
                    op::Shape("f32[8,801,1,1024]")));
}

TEST_F(SpmdPartitioningTest,
       PartitionConvGroupOnFeatureGroupCount_RHSPartialReplicate) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[16,801,1,1024] parameter(0)
  %lhs.copy = f32[16,801,1,1024] copy(%lhs),
    sharding={devices=[1,2,1,2]0,1,2,3}
  %rhs = f32[5,1,1,1024] parameter(1)
  %rhs.copy = f32[5,1,1,1024] copy(%rhs),
    sharding={devices=[1,1,1,2,2]0,2,1,3 last_tile_dim_replicate}
  ROOT %conv = f32[16,801,1,1024] convolution(%lhs.copy, %rhs.copy),
    dim_labels=b01f_01io->b01f,feature_group_count=1024,
    window={size=5x1 pad=2_2x0_0},
    sharding={devices=[1,2,1,2]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(), op::Constant()),
                                      op::Constant(), op::Reshape(),
                                      op::Constant(), op::Reshape())),
            op::Shape("f32[16,401,1,512]"));
  auto left_halo = AllOf(op::Shape("f32[16,2, 1, 512]"),
                         op::CollectivePermute(op::Slice(lhs)));
  auto right_halo = AllOf(op::Shape("f32[16,2, 1, 512]"),
                          op::CollectivePermute(op::Slice(lhs)));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[5,1,1,512]"));
  EXPECT_THAT(
      root,
      AllOf(op::Convolution(
                op::Select(_, op::Concatenate(left_halo, lhs, right_halo), _),
                rhs),
            op::Shape("f32[16, 401, 1, 512]")));
}

TEST_F(SpmdPartitioningTest,
       PartitionConvGroupOnFeatureGroupCount_RHSAlignWithOutput) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[16,801,1,1024] parameter(0)
  %lhs.copy = f32[16,801,1,1024] copy(%lhs),
    sharding={devices=[1,2,1,2]0,1,2,3}
  %rhs = f32[5,1,1,1024] parameter(1), sharding={replicated}
  ROOT %conv = f32[16,801,1,1024] convolution(%lhs.copy, %rhs),
    dim_labels=b01f_01io->b01f,feature_group_count=1024,
    window={size=5x1 pad=2_2x0_0},
    sharding={devices=[1,2,1,2]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(), op::Constant()),
                                      op::Constant(), op::Reshape(),
                                      op::Constant(), op::Reshape())),
            op::Shape("f32[16,401,1,512]"));
  auto left_halo = AllOf(op::Shape("f32[16,2, 1, 512]"),
                         op::CollectivePermute(op::Slice(lhs)));
  auto right_halo = AllOf(op::Shape("f32[16,2, 1, 512]"),
                          op::CollectivePermute(op::Slice(lhs)));
  const auto rhs =
      AllOf(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                             op::Constant(), op::Reshape()),
            op::Shape("f32[5,1,1,512]"));
  EXPECT_THAT(
      root,
      AllOf(op::Convolution(
                op::Select(_, op::Concatenate(left_halo, lhs, right_halo), _),
                rhs),
            op::Shape("f32[16, 401, 1, 512]")));
}

TEST_F(SpmdPartitioningTest,
       PartitionConvGroupOnFeatureGroupCount_LHSAlignWithOutput) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[16,801,1,1024] parameter(0)
  %lhs.copy = f32[16,801,1,1024] copy(%lhs),
    sharding={devices=[2,1,1,1,2]0,1,2,3 last_tile_dim_replicate}
  %rhs = f32[5,1,1,1024] parameter(1)
  %rhs.copy = f32[5,1,1,1024] copy(%rhs),
    sharding={devices=[1,1,1,2,2]0,2,1,3 last_tile_dim_replicate}
  ROOT %conv = f32[16,801,1,1024] convolution(%lhs.copy, %rhs.copy),
    dim_labels=b01f_01io->b01f,feature_group_count=1024,
    window={size=5x1 pad=2_2x0_0},
    sharding={devices=[1,2,1,2]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Reshape(), op::Constant(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[8,801,1,1024]"));
  auto resharded_lhs =
      AllOf(op::Reshape(op::Transpose(op::AllToAll(op::Reshape(
                op::Pad(op::DynamicSlice(lhs, op::Subtract(), op::Subtract(),
                                         op::Subtract(), op::Subtract()),
                        op::Constant()))))),
            op::Shape("f32[16,401,1,512]"));
  auto left_halo = AllOf(op::Shape("f32[16,2, 1, 512]"),
                         op::CollectivePermute(op::Slice(resharded_lhs)));
  auto right_halo = AllOf(op::Shape("f32[16,2, 1, 512]"),
                          op::CollectivePermute(op::Slice(resharded_lhs)));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[5,1,1,512]"));
  EXPECT_THAT(
      root,
      AllOf(
          op::Convolution(
              op::Select(
                  _, op::Concatenate(left_halo, resharded_lhs, right_halo), _),
              rhs),
          op::Shape("f32[16, 401, 1, 512]")));
}

TEST_F(SpmdPartitioningTest, PartitionConvGroupOnBatchGroupCount) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[16,801,1,1024] parameter(0)
  %lhs.copy = f32[16,801,1,1024] copy(%lhs),
    sharding={devices=[1,2,1,2]0,1,2,3}
  %rhs = f32[16,801,1,1024] parameter(1)
  %rhs.copy = f32[16,801,1,1024] copy(%rhs),
    sharding={devices=[1,2,1,2]0,1,2,3}
  ROOT %conv = f32[5,1,1,1024] convolution(%lhs.copy, %rhs.copy),
    dim_labels=f01b_i01o->01bf,batch_group_count=1024,
    window={size=801x1 pad=2_2x0_0},
    sharding={devices=[1,1,1,2,2]0,1,2,3 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Select(_,
                 op::Copy(op::DynamicSlice(
                     op::Pad(op::Parameter(), op::Constant()), op::Constant(),
                     op::Reshape(), op::Constant(), op::Reshape())),
                 _),
      op::Shape("f32[16,401,1,512]"));
  auto left_halo = AllOf(op::Shape("f32[16,2, 1, 512]"),
                         op::CollectivePermute(op::Slice(lhs)));
  auto right_halo = AllOf(op::Shape("f32[16,2, 1, 512]"),
                          op::CollectivePermute(op::Slice(lhs)));
  const auto rhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(), op::Constant()),
                                      op::Constant(), op::Reshape(),
                                      op::Constant(), op::Reshape())),
            op::Shape("f32[16,401,1,512]"));
  auto conv = AllOf(op::Convolution(op::Concatenate(left_halo, lhs, right_halo),
                                    op::Select(_, rhs, _)),
                    op::Shape("f32[5,1,1,512]"));
  EXPECT_THAT(root, AllOf(op::CollectivePermute(op::AllReduce(conv)),
                          op::Shape("f32[5,1,1,512]")));
}

TEST_F(SpmdPartitioningTest,
       PartitionConvWithFeatureGroupCountAlignOuputWithRHS) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[16,801,1,1024] parameter(0)
  %lhs.copy = f32[16,801,1,1024] copy(%lhs),
    sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[5,1,1,1024] parameter(1)
  %rhs.copy = f32[5,1,1,1024] copy(%rhs),
    sharding={devices=[1,1,1,2]0,1}
  ROOT %conv = f32[16,801,1,1024] convolution(%lhs.copy, %rhs.copy),
    dim_labels=b01f_01io->b01f,feature_group_count=1024,
    window={size=5x1 pad=2_2x0_0},
    sharding={devices=[2,1,1,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(), op::Constant()),
                                      op::Constant(), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[16,401,1,1024]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[5,1,1,512]"));
  auto resharded_lhs = AllOf(
      op::Slice(op::Reshape(op::Transpose(op::AllToAll(op::Reshape(lhs))))),
      op::Shape("f32[16,801,1,512]"));
  auto conv = AllOf(op::Convolution(resharded_lhs, rhs),
                    op::Shape("f32[16,801,1,512]"));
  EXPECT_THAT(root,
              AllOf(op::Reshape(op::Transpose(op::AllToAll(op::Reshape(conv)))),
                    op::Shape("f32[8,801,1,1024]")));
}

TEST_F(SpmdPartitioningTest, PartitionConvWithFeatureGroupCountBackProp) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[16,801,1,1024] parameter(0)
  %lhs.copy = f32[16,801,1,1024] copy(%lhs),
    sharding={devices=[1,1,1,2]0,1}
  %rhs = f32[5,1,1024,1] parameter(1)
  %rhs.copy = f32[5,1,1024,1] copy(%rhs),
    sharding={devices=[1,1,2,1]0,1}
  ROOT %conv = f32[16,801,1,1024] convolution(%lhs.copy, %rhs.copy),
    dim_labels=b01f_01oi->b01f,feature_group_count=1024,
    window={size=5x1 pad=2_2x0_0 rhs_reversal=1x1},
    sharding={devices=[1,1,1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[16,801,1,512]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Reshape(), op::Constant())),
      op::Shape("f32[5,1,512,1]"));
  EXPECT_THAT(root,
              AllOf(op::Convolution(lhs, rhs), op::Shape("f32[16,801,1,512]")));
}

TEST_F(SpmdPartitioningTest, NoReshardOnBroadcastDims) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[2,3] parameter(0)
  %param1 = f32[2,3,20] parameter(1)
  %br0 = f32[20,2,20,3,20] broadcast(%param0), dimensions={1,3}, sharding={devices=[2,1,2,1,2]0,1,2,3,4,5,6,7}
  %br1 = f32[20,2,20,3,20] broadcast(%param1), dimensions={1,3,4}, sharding={devices=[2,1,2,1,2]0,1,2,3,4,5,6,7}
  %add = f32[20,2,20,3,20] add(%br0, %br1), sharding={devices=[2,1,2,1,2]0,1,2,3,4,5,6,7}
  %reshape = f32[10,4,10,6,20] reshape(%br0), sharding={devices=[2,1,2,1,2]0,1,2,3,4,5,6,7}
  %transpose = f32[2,3,20,20,20] transpose(%br0), dimensions={1,3,0,2,4}, sharding={devices=[1,1,2,2,2]0,1,2,3,4,5,6,7}
  %copy_add0 = f32[20,2,20,3,20] copy(%add), sharding={devices=[2,1,2,1,2]6,7,2,3,4,5,0,1}
  %copy_add1 = f32[20,2,20,3,20] copy(%add), sharding={devices=[2,1,2,1,2]7,6,3,2,5,4,0,1}
  %copy_reshape = f32[10,4,10,6,20] copy(%reshape), sharding={devices=[2,1,2,1,2]7,6,3,2,5,4,0,1}
  %copy_transpose = f32[2,3,20,20,20] copy(%transpose), sharding={devices=[1,1,2,2,2]7,6,3,2,5,4,0,1}
  ROOT %tuple = (f32[20,2,20,3,20], f32[20,2,20,3,20], f32[10,4,10,6,20], f32[2,3,20,20,20])
    tuple(%copy_add0, %copy_add1, %copy_reshape, %copy_transpose),
    sharding={{devices=[2,1,2,1,2]6,7,2,3,4,5,0,1},{devices=[2,1,2,1,2]7,6,3,2,5,4,0,1},{devices=[2,1,2,1,2]7,6,3,2,5,4,0,1},{devices=[1,1,2,2,2]7,6,3,2,5,4,0,1}}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  // Reshard on copy_add0 only happens on broadcast dims, can be skipped.
  auto copy_add0 =
      op::Copy(op::Copy(op::Add(op::Broadcast(_), op::Broadcast(_))));
  // Reshard on copy_add1 also happens on non-broadcast dims.
  auto copy_add1 = op::Copy(
      op::CollectivePermute(op::Add(op::Broadcast(_), op::Broadcast(_))));
  // Reshard on copy_reshape only happens on broadcast dims, can be skipped.
  auto copy_reshape = op::Copy(op::Copy(op::Reshape(op::Broadcast(_))));
  // Reshard on copy_transpose only happens on broadcast dims, can be skipped.
  auto copy_transpose = op::Copy(op::Copy(op::Transpose(op::Broadcast(_))));
  EXPECT_THAT(root,
              op::Tuple(copy_add0, copy_add1, copy_reshape, copy_transpose));
}

TEST_F(SpmdPartitioningTest,
       ConvolutionFilterIFOFPartitionedInputPartialReplicate) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[128,112,112,12] parameter(0)
  %lhs.copy = f32[128,112,112,12] copy(f32[128,112,112,12] %lhs),
    sharding={devices=[1,1,1,2,2]0,1,2,3 last_tile_dim_replicate}
  %rhs = f32[7,7,12,64] parameter(1)
  %rhs.copy = f32[7,7,12,64] copy(f32[7,7,12,64] %rhs),
    sharding={devices=[1,1,2,2]0,1,2,3}
  ROOT %conv = f32[128,56,56,64] convolution(
    f32[128,112,112,12] %lhs.copy,
    f32[7,7,12,64] %rhs.copy),
    window={size=7x7 stride=2x2 pad=3_3x3_3},
    dim_labels=b01f_01io->b01f,
    sharding={devices=[1,1,1,2,2]0,1,2,3 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[128,112,112,6]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Reshape(), op::Reshape())),
      op::Shape("f32[7,7,6,32]"));

  EXPECT_THAT(
      root,
      AllOf(op::CollectivePermute(op::AllReduce(op::Convolution(lhs, rhs))),
            op::Shape("f32[128,56,56,32]")));
}

TEST_F(SpmdPartitioningTest,
       ConvolutionInputKernelNonContractingDimPartialReplicate) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[128,56,56,256] parameter(0)
  %lhs.copy = f32[128,56,56,256] copy(%lhs),
  sharding={devices=[1,1,1,2,2]0,1,2,3 last_tile_dim_replicate}
  %rhs = f32[128,28,28,512] parameter(1)
  %rhs.copy = f32[128,28,28,512] copy(%rhs),
  sharding={devices=[1,1,1,2,2]0,1,2,3 last_tile_dim_replicate}
  ROOT %conv = f32[1,1,256,512] convolution(%lhs.copy, %rhs.copy),
    window={size=28x28 pad=0_-1x0_-1 rhs_dilate=2x2}, dim_labels=f01b_i01o->01bf,
    sharding={devices=[1,1,2,2]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[128,56,56,128]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[128,28,28,256]"));

  EXPECT_THAT(root, AllOf(op::Convolution(lhs, op::CollectivePermute(rhs)),
                          op::Shape("f32[1,1,128,256]")));
}

TEST_F(SpmdPartitioningTest,
       ConvolutionInputSpatialDimAndFeatureDimParttiioned) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[8,210,210,12] parameter(0)
  %lhs.copy = f32[8,210,210,12] copy(f32[8,210,210,12] %lhs),
    sharding={devices=[1,2,1,2]0,1,2,3}
  %rhs = f32[3,3,12,32] parameter(1)
  %rhs.copy = f32[3,3,12,32] copy(f32[3,3,12,32] %rhs),
    sharding={devices=[1,1,2,1,2]0,1,2,3 last_tile_dim_replicate}
  ROOT %conv = f32[8,210,210,32] convolution(
    f32[8,210,210,12] %lhs.copy,
    f32[3,3,12,32] %rhs.copy),
    window={size=3x3 pad=1_1x1_1},
    dim_labels=b01f_01io->b01f,
    sharding={devices=[1,2,1,1,2]0,1,2,3 last_tile_dim_replicate}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  const auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[8,105,210,6]"));
  auto left_halo =
      AllOf(op::CollectivePermute(op::Slice(lhs)), op::Shape("f32[8,1,210,6]"));
  auto right_halo =
      AllOf(op::CollectivePermute(op::Slice(lhs)), op::Shape("f32[8,1,210,6]"));
  auto exchanged_lhs = AllOf(
      op::Select(op::And(_, _), op::Concatenate(left_halo, lhs, right_halo),
                 op::Broadcast(_)),
      op::Shape("f32[8,107,210,6]"));
  const auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Reshape(), op::Constant())),
      op::Shape("f32[3,3,6,32]"));
  EXPECT_THAT(root, AllOf(op::AllReduce(op::Convolution(
                              exchanged_lhs, op::CollectivePermute(rhs))),
                          op::Shape("f32[8,105,210,32]")));
}

TEST_F(SpmdPartitioningTest, Fft3D) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  constant = c64[1,1,6]
    constant({{{(0,0),(1,1),(2,2),(3,3),(4,4),(5,5)}}}),
    sharding={devices=[1,1,2]0,1}
  ROOT fft = c64[1,1,6] fft(c64[1,1,6] constant), fft_type=FFT, fft_length={6},
    sharding={devices=[1,1,2]0,1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  auto input = AllOf(op::DynamicSlice(op::Constant(), op::Constant(),
                                      op::Constant(), op::Reshape()),
                     op::Shape("c64[1,1,3]"));
  auto padded_input =
      AllOf(op::DynamicSlice(
                op::Concatenate(input, op::CollectivePermute(op::Slice())),
                op::Constant(), op::Constant(), op::Reshape()),
            op::Shape("c64[1,1,4]"));

  auto shuffled_input =
      AllOf(op::Slice(op::AllToAll(op::Dot(padded_input, op::Convert()))),
            op::Shape("c64[1,1,3]"));

  auto local_fft = AllOf(op::Fft(shuffled_input), op::Shape("c64[1,1,3]"));

  EXPECT_THAT(root, AllOf(op::GetTupleElement(op::While(op::Tuple(
                              _, op::Multiply(local_fft, op::Exp()), _, _, _))),
                          op::Shape("c64[1,1,3]")));
}

TEST_F(SpmdPartitioningTest, DotInputsAreIdentical) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %parameter.1 = f32[4000,4000]{1,0} parameter(0),
    sharding={devices=[2,4]0,1,2,3,4,5,6,7}
  ROOT %convolution = f32[4000,4000]{1,0} convolution(
    f32[4000,4000]{1,0} %parameter.1, f32[4000,4000]{1,0} %parameter.1),
    dim_labels=bf_io->bf, sharding={devices=[2,4]0,1,2,3,4,5,6,7}
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  auto param = AllOf(op::Parameter(), op::Shape("f32[2000, 1000]"));
  auto resharded_lhs =
      AllOf(op::AllReduce(op::DynamicUpdateSlice(_, param, _, _)),
            op::Shape("f32[2000, 4000]"));
  auto resharded_rhs =
      AllOf(op::AllReduce(op::DynamicUpdateSlice(_, op::Copy(param), _, _)),
            op::Shape("f32[4000, 1000]"));
  EXPECT_THAT(root, AllOf(op::Convolution(resharded_lhs, resharded_rhs),
                          op::Shape("f32[2000, 1000]")));
}

TEST_F(SpmdPartitioningTest, ConstantSliceReshard) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %constant.785 = f32[1,8] constant({{0,1,2,3,4,5,6,7}}),
    sharding={devices=[1,8]0,1,2,3,4,5,6,7}
  %slice.62 = f32[1,1] slice(%constant.785), slice={[0:1], [0:1]},
    sharding={devices=[1,8]0,1,2,3,4,5,6,7}
  ROOT %reshape.779 = f32[] reshape(%slice.62), sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  const auto root = module->entry_computation()->root_instruction();
  VLOG(1) << module->ToString();
  auto slice = AllOf(op::Shape("f32[1,1]"),
                     op::Copy(op::DynamicSlice(op::Constant(), _, _)));
  EXPECT_THAT(root, op::Reshape(op::AllReduce(op::Select(_, slice, _))));
}

TEST_F(SpmdPartitioningTest, GatherParallelDimRedistributionOperand) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[4,2,1,1]0,1,2,3,4,5,6,7}
  %constant = s32[4] constant({0, 1, 2, 3}), sharding={replicated}
  %iota = s32[1,8,4]{2,1,0} broadcast(%constant), dimensions={2},
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  ROOT %gather.20 = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={1,0}, index_vector_dim=0,
    slice_sizes={1,1,2,2}, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  const auto root = module->entry_computation()->root_instruction();
  VLOG(1) << module->ToString();
  auto operand = AllOf(op::Shape("s32[1,4,2,2]"), op::Reshape());
  auto indices = AllOf(op::Shape("s32[2,1,4]"), op::Subtract());
  auto gather = AllOf(op::Shape("s32[1,4,2,2]"), op::Gather(operand, indices));
  EXPECT_THAT(root,
              op::AllReduce(op::DynamicUpdateSlice(_, gather, _, _, _, _)));
}

TEST_F(SpmdPartitioningTest, GatherParallelDimRedistributionIndices) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[8,1,1,1]0,1,2,3,4,5,6,7}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,4,2]0,1,2,3,4,5,6,7}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,4,2]0,1,2,3,4,5,6,7}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,4,2]0,1,2,3,4,5,6,7}
  ROOT %gather.20 = s32[8,4,2,2]{3,2,1,0} gather(s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  const auto root = module->entry_computation()->root_instruction();
  VLOG(1) << module->ToString();
  auto operand = AllOf(op::Shape("s32[2,2,2,2]"), op::Reshape());
  auto indices = AllOf(op::Shape("s32[2,2,2]"), op::Subtract());
  auto gather = AllOf(op::Shape("s32[2,2,2,2]"), op::Gather(operand, indices));
  EXPECT_THAT(root, op::AllReduce(op::AllReduce(
                        op::DynamicUpdateSlice(_, gather, _, _, _, _))));
}

TEST_F(SpmdPartitioningTest, GatherParallelDimReplicatedIndices) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[8,1,1,1]0,1,2,3,4,5,6,7}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={replicated}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={replicated}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={replicated}
  ROOT %gather.20 = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  const auto root = module->entry_computation()->root_instruction();
  VLOG(1) << module->ToString();
  auto operand = AllOf(op::Shape("s32[1,4,2,2]"), op::Parameter());
  auto indices = AllOf(op::Shape("s32[2,1,4]"), op::Subtract());
  auto gather = AllOf(op::Shape("s32[1,4,2,2]"), op::Gather(operand, indices));
  EXPECT_THAT(root,
              op::AllReduce(op::DynamicUpdateSlice(_, gather, _, _, _, _)));
}

TEST_F(SpmdPartitioningTest, GatherParallelDimReplicatedOperand) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0), sharding={replicated}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  ROOT %gather.20 = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  const auto root = module->entry_computation()->root_instruction();
  VLOG(1) << module->ToString();
  auto operand = AllOf(op::Shape("s32[1,4,2,2]"), op::DynamicSlice());
  auto indices = AllOf(op::Shape("s32[2,1,4]"), op::Subtract());
  auto gather = AllOf(op::Shape("s32[1,4,2,2]"), op::Gather(operand, indices));
  EXPECT_THAT(root,
              op::AllReduce(op::DynamicUpdateSlice(_, gather, _, _, _, _)));
}

TEST_F(SpmdPartitioningTest, GatherParallelDimPartialReplicatedIndices) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[8,1,1,1]0,1,2,3,4,5,6,7}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %gather.20 = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  const auto root = module->entry_computation()->root_instruction();
  VLOG(1) << module->ToString();
  auto operand = AllOf(op::Shape("s32[1,4,2,2]"), op::Parameter());
  auto indices = AllOf(op::Shape("s32[2,1,4]"), op::Subtract());
  auto gather = AllOf(op::Shape("s32[1,4,2,2]"), op::Gather(operand, indices));
  EXPECT_THAT(root,
              op::AllReduce(op::DynamicUpdateSlice(_, gather, _, _, _, _)));
}

TEST_F(SpmdPartitioningTest, GatherParallelDimPartialReplicatedOperand) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0), sharding={
    devices=[2,1,1,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  ROOT %gather.20 = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  const auto root = module->entry_computation()->root_instruction();
  VLOG(1) << module->ToString();
  auto operand = AllOf(op::Shape("s32[1,4,2,2]"), op::DynamicSlice());
  auto indices = AllOf(op::Shape("s32[2,1,4]"), op::Subtract());
  auto gather = AllOf(op::Shape("s32[1,4,2,2]"), op::Gather(operand, indices));
  EXPECT_THAT(root,
              op::AllReduce(op::DynamicUpdateSlice(_, gather, _, _, _, _)));
}

TEST_F(SpmdPartitioningTest, GatherParallelDimSwappedDimensions) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0), sharding={
    devices=[4,2,1,1]0,1,2,3,4,5,6,7}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,2,4]0,1,2,3,4,5,6,7}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,2,4]0,1,2,3,4,5,6,7}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,2,4]0,1,2,3,4,5,6,7}
  ROOT %gather.20 = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  const auto root = module->entry_computation()->root_instruction();
  VLOG(1) << module->ToString();
  auto operand = AllOf(op::Shape("s32[4,1,2,2]"), op::CollectivePermute());
  auto indices = AllOf(op::Shape("s32[2,4,1]"), op::Subtract());
  auto gather = AllOf(op::Shape("s32[4,1,2,2]"), op::Gather(operand, indices));
  EXPECT_THAT(root, op::AllReduce(op::AllReduce(
                        op::DynamicUpdateSlice(_, gather, _, _, _, _))));
}

TEST_F(SpmdPartitioningTest, GatherParallelDimFromOutsideWhilePositive) {
  absl::string_view hlo_string = R"(
HloModule module

cond {
  %parameters = (s32[8,4,2,2], s32[1,8,4], s32[]) parameter(0),
    sharding={{replicated}, {devices=[1,8,1]0,1,2,3,4,5,6,7}, {replicated}}
  %counter = s32[] get-tuple-element(parameters), index=2, sharding={replicated}
  %constant = s32[] constant(3), sharding={replicated}
  ROOT %lt = pred[] compare(counter, constant), direction=LT,
    sharding={replicated}
}

body {
  %parameters = (s32[8,4,2,2], s32[1,8,4], s32[]) parameter(0),
    sharding={{replicated}, {devices=[1,8,1]0,1,2,3,4,5,6,7}, {replicated}}
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} get-tuple-element(parameters), index=0,
    sharding={replicated}
  %iota = s32[1,8,4]{2,1,0} get-tuple-element(parameters), index=1,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %counter = s32[] get-tuple-element(parameters), index=2, sharding={replicated}
  %constant = s32[] constant(1), sharding={replicated}
  %updated_counter = s32[] add(counter, constant), sharding={replicated}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %gather.20 = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}, sharding={replicated}
  ROOT %tuple = (s32[8,4,2,2], s32[1,8,4], s32[])
    tuple(gather.20, iota, updated_counter),
    sharding={{replicated}, {devices=[1,8,1]0,1,2,3,4,5,6,7}, {replicated}}
}

ENTRY entry {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0), sharding={replicated}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %counter = s32[] constant(0), sharding={replicated}
  %tuple = (s32[8,4,2,2], s32[1,8,4], s32[]) tuple(parameter.0, iota, counter),
    sharding={{replicated}, {devices=[1,8,1]0,1,2,3,4,5,6,7}, {replicated}}
  ROOT while = (s32[8,4,2,2], s32[1,8,4], s32[]) while(tuple), body=body,
    condition=cond,
    sharding={{replicated}, {devices=[1,8,1]0,1,2,3,4,5,6,7}, {replicated}}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  const auto root = module->entry_computation()
                        ->root_instruction()
                        ->while_body()
                        ->root_instruction();
  VLOG(1) << module->ToString();
  auto operand = AllOf(op::Shape("s32[1,4,2,2]"), op::DynamicSlice());
  auto indices = AllOf(op::Shape("s32[2,1,4]"), op::Subtract());
  auto gather = AllOf(op::Shape("s32[1,4,2,2]"), op::Gather(operand, indices));
  EXPECT_THAT(
      root,
      op::Tuple(op::AllReduce(op::DynamicUpdateSlice(_, gather, _, _, _, _)), _,
                _));
}

TEST_F(SpmdPartitioningTest, GatherParallelDimFromOutsideWhileNegative) {
  absl::string_view hlo_string = R"(
HloModule module

cond {
  %parameters = (s32[8,4,2,2], s32[1,8,4], s32[]) parameter(0),
    sharding={{replicated}, {devices=[1,8,1]0,1,2,3,4,5,6,7}, {replicated}}
  %counter = s32[] get-tuple-element(parameters), index=2, sharding={replicated}
  %constant = s32[] constant(3), sharding={replicated}
  ROOT %lt = pred[] compare(counter, constant), direction=LT,
    sharding={replicated}
}

body {
  %parameters = (s32[8,4,2,2], s32[1,8,4], s32[]) parameter(0),
    sharding={{replicated}, {devices=[1,8,1]0,1,2,3,4,5,6,7}, {replicated}}
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} get-tuple-element(parameters), index=0,
    sharding={replicated}
  %iota = s32[1,8,4]{2,1,0} get-tuple-element(parameters), index=1,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %counter = s32[] get-tuple-element(parameters), index=2, sharding={replicated}
  %constant = s32[] constant(1), sharding={replicated}
  %updated_counter = s32[] add(counter, constant), sharding={replicated}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %gather.20 = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}, sharding={replicated}
  %iota.2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  ROOT %tuple = (s32[8,4,2,2], s32[1,8,4], s32[])
    tuple(gather.20, iota.2, updated_counter),
    sharding={{replicated}, {devices=[1,8,1]0,1,2,3,4,5,6,7}, {replicated}}
}

ENTRY entry {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0), sharding={replicated}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %counter = s32[] constant(0), sharding={replicated}
  %tuple = (s32[8,4,2,2], s32[1,8,4], s32[]) tuple(parameter.0, iota, counter),
    sharding={{replicated}, {devices=[1,8,1]0,1,2,3,4,5,6,7}, {replicated}}
  ROOT while = (s32[8,4,2,2], s32[1,8,4], s32[]) while(tuple), body=body,
    condition=cond,
    sharding={{replicated}, {devices=[1,8,1]0,1,2,3,4,5,6,7}, {replicated}}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  const auto root = module->entry_computation()
                        ->root_instruction()
                        ->while_body()
                        ->root_instruction();
  VLOG(1) << module->ToString();
  auto operand = AllOf(op::Shape("s32[8,4,2,2]"), op::GetTupleElement());
  auto indices = AllOf(op::Shape("s32[2,1,4]"), op::Concatenate());
  auto gather = AllOf(op::Shape("s32[1,4,2,2]"), op::Gather(operand, indices));
  EXPECT_THAT(
      root,
      op::Tuple(op::AllReduce(op::DynamicUpdateSlice(_, gather, _, _, _, _)), _,
                _));
}

TEST_F(SpmdPartitioningTest, ParallelDimFromOutsideConditionalPositive) {
  absl::string_view hlo_string = R"(
HloModule module

gather_comp {
  %parameters = (s32[8,4,2,2], s32[1,8,4]) parameter(0),
    sharding={{replicated}, {devices=[1,8,1]0,1,2,3,4,5,6,7}}
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} get-tuple-element(parameters), index=0,
    sharding={replicated}
  %iota = s32[1,8,4]{2,1,0} get-tuple-element(parameters), index=1,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %gather.20 = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather.20), sharding={replicated}
}

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

scatter_comp {
  %parameters = (s32[8,4,2,2], s32[1,8,4]) parameter(0),
    sharding={{replicated}, {devices=[1,8,1]0,1,2,3,4,5,6,7}}
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} get-tuple-element(parameters), index=0,
    sharding={replicated}
  %iota = s32[1,8,4]{2,1,0} get-tuple-element(parameters), index=1,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %constant = s32[] constant(0)
  %base = s32[8,4,2,2]{3,2,1,0} broadcast(constant), dimensions={},
    sharding={replicated}
  %scatter.20 = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %base,
    s32[2,8,4]{2,1,0} %concatenate.19,
    s32[8,4,2,2]{3,2,1,0} %parameter.0),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0
  ROOT copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter.20), sharding={replicated}
}

ENTRY entry {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0), sharding={replicated}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %counter = s32[] constant(0), sharding={replicated}
  %tuple = (s32[8,4,2,2], s32[1,8,4]) tuple(parameter.0, iota),
    sharding={{replicated}, {devices=[1,8,1]0,1,2,3,4,5,6,7}}
  %parameter.1 = pred[] parameter(1)
  ROOT conditional = s32[8,4,2,2] conditional(parameter.1, tuple, tuple),
    true_computation=gather_comp, false_computation=scatter_comp,
    sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  // Verify gather is partitioned properly.
  {
    const auto partitioned_gather = module->entry_computation()
                                        ->root_instruction()
                                        ->true_computation()
                                        ->root_instruction();
    auto operand = AllOf(op::Shape("s32[1,4,2,2]"), op::DynamicSlice());
    auto indices = AllOf(op::Shape("s32[2,1,4]"), op::Subtract());
    auto gather =
        AllOf(op::Shape("s32[1,4,2,2]"), op::Gather(operand, indices));
    EXPECT_THAT(
        partitioned_gather,
        op::Copy(op::AllReduce(op::DynamicUpdateSlice(_, gather, _, _, _, _))));
  }

  // Verify scatter is partitioned properly.
  {
    const auto partitioned_scatter = module->entry_computation()
                                         ->root_instruction()
                                         ->false_computation()
                                         ->root_instruction();
    auto operand = AllOf(op::Shape("s32[1,4,2,2]"), op::DynamicSlice());
    auto indices = AllOf(op::Shape("s32[2,1,4]"), op::Subtract());
    auto update = AllOf(op::Shape("s32[1,4,2,2]"), op::DynamicSlice());
    auto scatter =
        AllOf(op::Shape("s32[1,4,2,2]"), op::Scatter(operand, indices, update));
    EXPECT_THAT(partitioned_scatter,
                op::Copy(op::AllReduce(
                    op::DynamicUpdateSlice(_, scatter, _, _, _, _))));
  }
}

TEST_F(SpmdPartitioningTest, GatherParallelDimAndNonParallelDimPartitioned) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 = s32[1,8,4]{2,1,0} parameter(1)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0),
    sharding={devices=[2,2,1,1]0,1,2,3}
  %indices = s32[1,8,4]{2,1,0} copy(s32[1,8,4]{2,1,0} %arg.1),
    sharding={devices=[1,2,2]0,1,2,3}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,2,2]0,1,2,3}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %indices), dimensions={0},
    sharding={devices=[1,2,2]0,1,2,3}
  ROOT %gather.20 = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %concatenate.19), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}, sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  const auto root = module->entry_computation()->root_instruction();
  VLOG(1) << module->ToString();
  auto operand = AllOf(op::Shape("s32[4,4,2,2]"), op::AllReduce());
  auto indices = AllOf(op::Shape("s32[2,4,2]"), op::Subtract());
  auto gather = AllOf(op::Shape("s32[4,2,2,2]"), op::Gather(operand, indices));
  EXPECT_THAT(
      root, op::AllReduce(op::DynamicUpdateSlice(
                _, op::AllReduce(op::DynamicUpdateSlice(_, gather, _, _, _, _)),
                _, _, _, _)));
}

TEST_F(SpmdPartitioningTest, GatherMergedIndexParallelAndOperandPassthrough) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[2,2,2,1]0,1,2,3,4,5,6,7}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %gather.20 = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={1,0}, index_vector_dim=0,
    slice_sizes={1,1,2,2}, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  auto operand = AllOf(op::Shape("s32[2,4,1,2]"), op::Reshape());
  auto indices = AllOf(op::Shape("s32[2,2,4]"), op::Subtract());
  auto gather = AllOf(op::Shape("s32[2,4,1,2]"), op::Gather(operand, indices));
  EXPECT_THAT(root, op::AllReduce(op::AllReduce(
                        op::DynamicUpdateSlice(_, gather, _, _, _, _))));
}

TEST_F(SpmdPartitioningTest, GatherMergedIndexParallelAndTrivialSlicedOperand) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[4,2,1,1]0,1,2,3,4,5,6,7}
  %parameter.1 = s32[1,8,1]{2,1,0} parameter(1),
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %iota = s32[1,8,1]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %concatenate.19 = s32[2,8,1]{2,1,0} concatenate(
    s32[1,8,1]{2,1,0} %parameter.1, s32[1,8,1]{2,1,0} %iota), dimensions={0},
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %gather.20 = s32[8,1,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,1]{2,1,0} %concatenate.19), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={1,0}, index_vector_dim=0,
    slice_sizes={1,1,2,2}, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  const auto root = module->entry_computation()->root_instruction();
  auto operand = AllOf(op::Shape("s32[2,2,2,2]"), op::Parameter());
  auto indices = AllOf(op::Shape("s32[2,2,1]"), op::Subtract());
  auto gather = AllOf(op::Shape("s32[2,1,2,2]"), op::Gather(operand, indices));
  VLOG(1) << module->ToString();
  EXPECT_THAT(root,
              op::AllReduce(op::DynamicUpdateSlice(
                  _, op::AllReduce(op::Select(_, _, gather)), _, _, _, _)));
}

TEST_F(SpmdPartitioningTest, GatherMergedIndexParallelAndIndexPassthrough) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[4,1,1,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %parameter.1 = s32[1,8,4]{2,1,0} parameter(1),
    sharding={devices=[1,4,2]0,1,2,3,4,5,6,7}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,4,2]0,1,2,3,4,5,6,7}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(
    s32[1,8,4]{2,1,0} %parameter.1, s32[1,8,4]{2,1,0} %iota), dimensions={0},
    sharding={devices=[1,4,2]0,1,2,3,4,5,6,7}
  ROOT %gather.20 = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={1,0}, index_vector_dim=0,
    slice_sizes={1,1,2,2}, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  auto operand = AllOf(op::Shape("s32[2,4,2,2]"), op::Parameter());
  auto indices = AllOf(op::Shape("s32[2,2,2]"), op::Subtract());
  auto gather = AllOf(op::Shape("s32[2,2,2,2]"), op::Gather(operand, indices));
  EXPECT_THAT(
      root, op::AllReduce(op::DynamicUpdateSlice(
                _, op::AllReduce(op::DynamicUpdateSlice(_, gather, _, _, _, _)),
                _, _, _, _)));
}

TEST_F(SpmdPartitioningTest,
       GatherMergedOperandPassthroughAndTrivialSlicedOperand) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[2,2,2,1]0,1,2,3,4,5,6,7}
  %parameter.1 = s32[2,8,4]{2,1,0} parameter(1),
    sharding={replicated}
  ROOT %gather.20 = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %parameter.1), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  auto operand = AllOf(op::Shape("s32[4,2,1,2]"), op::Parameter());
  auto indices = AllOf(op::Shape("s32[2,8,4]"), op::Subtract());
  auto gather = AllOf(op::Shape("s32[8,4,1,2]"), op::Gather(operand, indices));
  EXPECT_THAT(
      root, op::AllReduce(op::DynamicUpdateSlice(
                _, op::AllReduce(op::AllReduce(op::Select(_, _, gather))), _, _,
                _, _)));
}

TEST_F(SpmdPartitioningTest,
       GatherMergedOperandPassthroughAndIndexPassthrough) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[1,1,2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %parameter.1 = s32[2,8,4]{2,1,0} parameter(1),
    sharding={devices=[1,2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %gather.20 = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %parameter.1), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  auto operand = AllOf(op::Shape("s32[8,4,1,2]"), op::Parameter());
  auto indices = AllOf(op::Shape("s32[2,4,4]"), op::CollectivePermute());
  auto gather = AllOf(op::Shape("s32[4,4,1,2]"), op::Gather(operand, indices));
  EXPECT_THAT(
      root, op::AllReduce(op::DynamicUpdateSlice(
                _, op::AllReduce(op::DynamicUpdateSlice(_, gather, _, _, _, _)),
                _, _, _, _)));
}

TEST_F(SpmdPartitioningTest,
       GatherMergedTrivialSlicedOperandAndIndexPassthrough) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[2,2,1,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %parameter.1 = s32[2,8,4]{2,1,0} parameter(1),
    sharding={devices=[1,2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %gather.20 = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %parameter.1), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  auto operand = AllOf(op::Shape("s32[4,2,2,2]"), op::Parameter());
  auto indices = AllOf(op::Shape("s32[2,4,4]"), op::Subtract());
  auto gather = AllOf(op::Shape("s32[4,4,2,2]"), op::Gather(operand, indices));
  EXPECT_THAT(
      root, op::AllReduce(op::DynamicUpdateSlice(
                _, op::AllReduce(op::AllReduce(op::Select(_, _, gather))), _, _,
                _, _)));
}

TEST_F(SpmdPartitioningTest, GatherTrivialSlicedOperandPartial) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY main.4 {
  %arg.0 = s64[8,2]{1,0} parameter(0), sharding={devices=[4,2]0,1,2,3,4,5,6,7}
  %arg.1 = s32[2]{0} parameter(1), sharding={replicated}
  ROOT gather = s64[2,1]{1,0} gather(arg.0, arg.1), offset_dims={0,1},
    collapsed_slice_dims={}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={2,1}, indices_are_sorted=true, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  auto operand = AllOf(op::Shape("s64[8,1]"), op::AllReduce());
  auto indices = AllOf(op::Shape("s32[2]"), op::Subtract());
  auto gather = AllOf(op::Shape("s64[2,1]"), op::Gather(operand, indices));
  EXPECT_THAT(root, op::AllReduce(op::Select(_, _, gather)));
}

TEST_F(SpmdPartitioningTest, GatherParallelIndexAndOperand) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[4,1,2,1]0,1,2,3,4,5,6,7}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %gather.20 = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={1,0}, index_vector_dim=0,
    slice_sizes={1,1,2,2}, sharding={devices=[4,1,2,1]0,1,2,3,4,5,6,7}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  auto operand = AllOf(op::Shape("s32[2,4,1,2]"), op::Parameter(0));
  auto indices = AllOf(op::Shape("s32[2,2,4]"), op::Subtract());
  auto gather = AllOf(op::Shape("s32[2,4,1,2]"), op::Gather(operand, indices));
  EXPECT_THAT(root, gather);
}

TEST_F(SpmdPartitioningTest, GatherReshardParallelIndexAndOperand) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[4,1,2,1]0,1,2,3,4,5,6,7}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %gather.20 = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={1,0}, index_vector_dim=0,
    slice_sizes={1,1,2,2}, sharding={devices=[4,1,2,1]1,0,3,2,4,5,6,7}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  auto operand = AllOf(op::Shape("s32[2,4,1,2]"), op::Parameter(0));
  auto indices = AllOf(op::Shape("s32[2,2,4]"), op::Subtract());
  auto gather = AllOf(op::Shape("s32[2,4,1,2]"), op::Gather(operand, indices));
  EXPECT_THAT(root, op::CollectivePermute(gather));
}

TEST_F(SpmdPartitioningTest, GatherParallelIndexAndOperandReshard) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[4,1,1,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %gather.20 = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={1,0}, index_vector_dim=0,
    slice_sizes={1,1,2,2}, sharding={devices=[4,1,2,1]0,1,2,3,4,5,6,7}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  auto operand = AllOf(op::Shape("s32[2,4,2,2]"), op::Parameter(0));
  auto indices = AllOf(op::Shape("s32[2,2,4]"), op::Subtract());
  auto gather = AllOf(op::Shape("s32[2,4,2,2]"), op::Gather(operand, indices));
  EXPECT_THAT(root, op::DynamicSlice(gather, _, _, _, _));
}

TEST_F(SpmdPartitioningTest, ScatterParallelDimRedistributionOperand) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[4,2,1,1]0,1,2,3,4,5,6,7}
  %constant = s32[4] constant({0, 1, 2, 3}), sharding={replicated}
  %iota = s32[1,8,4]{2,1,0} broadcast(%constant), dimensions={2},
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1),
    sharding={devices=[8,1,1,1]0,1,2,3,4,5,6,7}
  ROOT %scatter.20 = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19,
    s32[8,4,2,2]{3,2,1,0} %parameter.1),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={1,0},
    index_vector_dim=0, sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  const auto root = module->entry_computation()->root_instruction();
  VLOG(1) << module->ToString();
  auto operand = AllOf(op::Shape("s32[1,4,2,2]"), op::Reshape());
  auto indices = AllOf(op::Shape("s32[2,1,4]"), op::Subtract());
  auto update = AllOf(op::Shape("s32[1,4,2,2]"), op::Parameter());
  auto scatter =
      AllOf(op::Shape("s32[1,4,2,2]"), op::Scatter(operand, indices, update));
  EXPECT_THAT(root,
              op::AllReduce(op::DynamicUpdateSlice(_, scatter, _, _, _, _)));
}

TEST_F(SpmdPartitioningTest, ScatterParallelDimReplicatedIndices) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[8,1,1,1]0,1,2,3,4,5,6,7}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={replicated}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={replicated}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={replicated}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1),
    sharding={devices=[8,1,1,1]0,1,2,3,4,5,6,7}
  ROOT %scatter.20 = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19,
    s32[8,4,2,2]{3,2,1,0} %parameter.1),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, PartitionComputation(hlo_string,
                                                            /*num_devices=*/8));
  const auto root = module->entry_computation()->root_instruction();
  VLOG(1) << module->ToString();
  auto operand = AllOf(op::Shape("s32[1,4,2,2]"), op::Parameter());
  auto indices = AllOf(op::Shape("s32[2,1,4]"), op::Subtract());
  auto update = AllOf(op::Shape("s32[1,4,2,2]"), op::Parameter());
  auto scatter =
      AllOf(op::Shape("s32[1,4,2,2]"), op::Scatter(operand, indices, update));
  EXPECT_THAT(root,
              op::AllReduce(op::DynamicUpdateSlice(_, scatter, _, _, _, _)));
}

TEST_F(SpmdPartitioningTest, ScatterParallelDimReplicatedOperand) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0), sharding={replicated}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1),
    sharding={devices=[8,1,1,1]0,1,2,3,4,5,6,7}
  ROOT %scatter.20 = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19,
    s32[8,4,2,2]{3,2,1,0} %parameter.1),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, PartitionComputation(hlo_string,
                                                            /*num_devices=*/8));
  const auto root = module->entry_computation()->root_instruction();
  VLOG(1) << module->ToString();
  auto operand = AllOf(op::Shape("s32[1,4,2,2]"), op::DynamicSlice());
  auto indices = AllOf(op::Shape("s32[2,1,4]"), op::Subtract());
  auto update = AllOf(op::Shape("s32[1,4,2,2]"), op::Parameter());
  auto scatter =
      AllOf(op::Shape("s32[1,4,2,2]"), op::Scatter(operand, indices, update));
  EXPECT_THAT(root,
              op::AllReduce(op::DynamicUpdateSlice(_, scatter, _, _, _, _)));
}

TEST_F(SpmdPartitioningTest, ScatterParallelDimReplicatedUpdate) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[8,1,1,1]0,1,2,3,4,5,6,7}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1), sharding={replicated}
  ROOT %scatter.20 = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19,
    s32[8,4,2,2]{3,2,1,0} %parameter.1),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, PartitionComputation(hlo_string,
                                                            /*num_devices=*/8));
  const auto root = module->entry_computation()->root_instruction();
  VLOG(1) << module->ToString();
  auto operand = AllOf(op::Shape("s32[1,4,2,2]"), op::Parameter());
  auto indices = AllOf(op::Shape("s32[2,1,4]"), op::Subtract());
  auto update = AllOf(op::Shape("s32[1,4,2,2]"), op::DynamicSlice());
  auto scatter =
      AllOf(op::Shape("s32[1,4,2,2]"), op::Scatter(operand, indices, update));
  EXPECT_THAT(root,
              op::AllReduce(op::DynamicUpdateSlice(_, scatter, _, _, _, _)));
}

TEST_F(SpmdPartitioningTest, ScatterParallelDimPartialReplicatedIndices) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[8,1,1,1]0,1,2,3,4,5,6,7}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1),
    sharding={devices=[8,1,1,1]0,1,2,3,4,5,6,7}
  ROOT %scatter.20 = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19,
    s32[8,4,2,2]{3,2,1,0} %parameter.1),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, PartitionComputation(hlo_string,
                                                            /*num_devices=*/8));
  const auto root = module->entry_computation()->root_instruction();
  VLOG(1) << module->ToString();
  auto operand = AllOf(op::Shape("s32[1,4,2,2]"), op::Parameter());
  auto indices = AllOf(op::Shape("s32[2,1,4]"), op::Subtract());
  auto update = AllOf(op::Shape("s32[1,4,2,2]"), op::Parameter());
  auto scatter =
      AllOf(op::Shape("s32[1,4,2,2]"), op::Scatter(operand, indices, update));
  EXPECT_THAT(root,
              op::AllReduce(op::DynamicUpdateSlice(_, scatter, _, _, _, _)));
}

TEST_F(SpmdPartitioningTest, ScatterParallelDimPartialReplicatedOperand) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0), sharding={
    devices=[2,1,1,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1),
    sharding={devices=[8,1,1,1]0,1,2,3,4,5,6,7}
  ROOT %scatter.20 = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19,
    s32[8,4,2,2]{3,2,1,0} %parameter.1),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, PartitionComputation(hlo_string,
                                                            /*num_devices=*/8));
  const auto root = module->entry_computation()->root_instruction();
  VLOG(1) << module->ToString();
  auto operand = AllOf(op::Shape("s32[1,4,2,2]"), op::DynamicSlice());
  auto indices = AllOf(op::Shape("s32[2,1,4]"), op::Subtract());
  auto update = AllOf(op::Shape("s32[1,4,2,2]"), op::Parameter());
  auto scatter =
      AllOf(op::Shape("s32[1,4,2,2]"), op::Scatter(operand, indices, update));
  EXPECT_THAT(root,
              op::AllReduce(op::DynamicUpdateSlice(_, scatter, _, _, _, _)));
}

TEST_F(SpmdPartitioningTest, ScatterParallelDimPartialReplicatedUpdate) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[8,1,1,1]0,1,2,3,4,5,6,7}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1), sharding={
    devices=[2,1,1,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %scatter.20 = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19,
    s32[8,4,2,2]{3,2,1,0} %parameter.1),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, PartitionComputation(hlo_string,
                                                            /*num_devices=*/8));
  const auto root = module->entry_computation()->root_instruction();
  VLOG(1) << module->ToString();
  auto operand = AllOf(op::Shape("s32[1,4,2,2]"), op::Parameter());
  auto indices = AllOf(op::Shape("s32[2,1,4]"), op::Subtract());
  auto update = AllOf(op::Shape("s32[1,4,2,2]"), op::DynamicSlice());
  auto scatter =
      AllOf(op::Shape("s32[1,4,2,2]"), op::Scatter(operand, indices, update));
  EXPECT_THAT(root,
              op::AllReduce(op::DynamicUpdateSlice(_, scatter, _, _, _, _)));
}

TEST_F(SpmdPartitioningTest, ScatterParallelDimSwappedDimensions) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0), sharding={
    devices=[4,2,1,1]0,1,2,3,4,5,6,7}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,2,4]0,1,2,3,4,5,6,7}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,2,4]0,1,2,3,4,5,6,7}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,2,4]0,1,2,3,4,5,6,7}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1), sharding={
    devices=[4,2,1,1]0,1,2,3,4,5,6,7}
  ROOT %scatter.20 = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19,
    s32[8,4,2,2]{3,2,1,0} %parameter.1),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, PartitionComputation(hlo_string,
                                                            /*num_devices=*/8));
  const auto root = module->entry_computation()->root_instruction();
  VLOG(1) << module->ToString();
  auto operand = AllOf(op::Shape("s32[4,1,2,2]"), op::CollectivePermute());
  auto indices = AllOf(op::Shape("s32[2,4,1]"), op::Subtract());
  auto update = AllOf(op::Shape("s32[4,1,2,2]"), op::CollectivePermute());
  auto scatter =
      AllOf(op::Shape("s32[4,1,2,2]"), op::Scatter(operand, indices, update));
  EXPECT_THAT(root, op::AllReduce(op::AllReduce(
                        op::DynamicUpdateSlice(_, scatter, _, _, _, _))));
}

TEST_F(SpmdPartitioningTest, ScatterParallelDimFromOutsideWhilePositive) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

cond {
  %parameters = (s32[8,4,2,2], s32[1,8,4], s32[8,4,2,2], s32[]) parameter(0),
    sharding={{replicated}, {devices=[1,8,1]0,1,2,3,4,5,6,7}, {replicated}, {replicated}}
  %counter = s32[] get-tuple-element(parameters), index=3, sharding={replicated}
  %constant = s32[] constant(3), sharding={replicated}
  ROOT %lt = pred[] compare(counter, constant), direction=LT,
    sharding={replicated}
}

body {
  %parameters = (s32[8,4,2,2], s32[1,8,4], s32[8,4,2,2], s32[]) parameter(0),
    sharding={{replicated}, {devices=[1,8,1]0,1,2,3,4,5,6,7}, {replicated}, {replicated}}
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} get-tuple-element(parameters), index=0,
    sharding={replicated}
  %iota = s32[1,8,4]{2,1,0} get-tuple-element(parameters), index=1,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} get-tuple-element(parameters), index=2,
    sharding={replicated}
  %counter = s32[] get-tuple-element(parameters), index=3, sharding={replicated}
  %constant = s32[] constant(1), sharding={replicated}
  %updated_counter = s32[] add(counter, constant), sharding={replicated}
  %scatter.20 = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19,
    s32[8,4,2,2]{3,2,1,0} %parameter.1),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0, sharding={replicated}
  ROOT %tuple = (s32[8,4,2,2], s32[1,8,4], s32[8,4,2,2], s32[])
    tuple(scatter.20, iota, parameter.1, updated_counter),
    sharding={{replicated}, {devices=[1,8,1]0,1,2,3,4,5,6,7}, {replicated}, {replicated}}
}

ENTRY entry {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0), sharding={replicated}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,8,1]0,1,2,3,4,5,6,7}
  %counter = s32[] constant(0), sharding={replicated}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1), sharding={replicated}
  %tuple = (s32[8,4,2,2], s32[1,8,4], s32[8,4,2,2], s32[])
    tuple(parameter.0, iota, parameter.1, counter),
    sharding={{replicated}, {devices=[1,8,1]0,1,2,3,4,5,6,7}, {replicated}, {replicated}}
  ROOT while = (s32[8,4,2,2], s32[1,8,4], s32[8,4,2,2], s32[]) while(tuple), body=body,
    condition=cond,
    sharding={{replicated}, {devices=[1,8,1]0,1,2,3,4,5,6,7}, {replicated}, {replicated}}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  const auto root = module->entry_computation()
                        ->root_instruction()
                        ->while_body()
                        ->root_instruction();
  VLOG(1) << module->ToString();
  auto operand = AllOf(op::Shape("s32[1,4,2,2]"), op::DynamicSlice());
  auto indices = AllOf(op::Shape("s32[2,1,4]"), op::Subtract());
  auto update = AllOf(op::Shape("s32[1,4,2,2]"), op::DynamicSlice());
  auto scatter =
      AllOf(op::Shape("s32[1,4,2,2]"), op::Scatter(operand, indices, update));
  EXPECT_THAT(
      root,
      op::Tuple(op::AllReduce(op::DynamicUpdateSlice(_, scatter, _, _, _, _)),
                _, _, _));
}

TEST_F(SpmdPartitioningTest, ScatterParallelDimAndNonParallelDimPartitioned) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 = s32[1,8,4]{2,1,0} parameter(1)
  %arg.2 = s32[8,4,2,2]{3,2,1,0} parameter(2)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0),
    sharding={devices=[2,2,1,1]0,1,2,3}
  %update = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.2),
    sharding={devices=[2,2,1,1]0,1,2,3}
  %indices = s32[1,8,4]{2,1,0} copy(s32[1,8,4]{2,1,0} %arg.1),
    sharding={devices=[1,2,2]0,1,2,3}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,2,2]0,1,2,3}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %indices), dimensions={0},
    sharding={devices=[1,2,2]0,1,2,3}
  ROOT %scatter.20 = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %concatenate.19,
    s32[8,4,2,2]{3,2,1,0} %update),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0, sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  const auto root = module->entry_computation()->root_instruction();
  VLOG(1) << module->ToString();
  auto operand = AllOf(op::Shape("s32[4,4,2,2]"));
  auto indices = AllOf(op::Shape("s32[2,4,2]"));
  auto update = AllOf(op::Shape("s32[4,2,2,2]"));
  auto scatter =
      AllOf(op::Shape("s32[4,4,2,2]"), op::Scatter(operand, indices, update));
  EXPECT_THAT(root, op::AllReduce(op::AllReduce(op::DynamicUpdateSlice(
                        _, op::DynamicSlice(op::AllReduce(scatter), _, _, _, _),
                        _, _, _, _))));
}

TEST_F(SpmdPartitioningTest, ScatterMergedIndexParallelAndOperandPassthrough) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[2,2,2,1]0,1,2,3,4,5,6,7}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(
    s32[1,8,4]{2,1,0} %iota, s32[1,8,4]{2,1,0} %iota2), dimensions={0},
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1),
    sharding={replicated}
  ROOT %scatter.20 = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19,
    s32[8,4,2,2]{3,2,1,0} %parameter.1),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={1,0},
    index_vector_dim=0, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  auto operand = AllOf(op::Shape("s32[2,4,1,2]"), op::Reshape());
  auto indices = AllOf(op::Shape("s32[2,2,4]"), op::Subtract());
  auto update = AllOf(op::Shape("s32[2,4,1,2]"), op::DynamicSlice());
  auto scatter =
      AllOf(op::Shape("s32[2,4,1,2]"), op::Scatter(operand, indices, update));
  EXPECT_THAT(root, op::AllReduce(op::AllReduce(
                        op::DynamicUpdateSlice(_, scatter, _, _, _, _))));
}

TEST_F(SpmdPartitioningTest,
       ScatterMergedIndexParallelAndTrivialSlicedOperand) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[4,2,1,1]0,1,2,3,4,5,6,7}
  %parameter.1 = s32[1,8,4]{2,1,0} parameter(1),
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(
    s32[1,8,4]{2,1,0} %parameter.1, s32[1,8,4]{2,1,0} %iota), dimensions={0},
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %parameter.2 = s32[8,4,2,2]{3,2,1,0} parameter(2),
    sharding={replicated}
  ROOT %scatter.20 = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19,
    s32[8,4,2,2]{3,2,1,0} %parameter.2),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={1,0},
    index_vector_dim=0, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  const auto root = module->entry_computation()->root_instruction();
  auto operand = AllOf(op::Shape("s32[2,2,2,2]"), op::Parameter());
  auto indices = AllOf(op::Shape("s32[2,2,4]"), op::Subtract());
  auto update = AllOf(op::Shape("s32[2,4,2,2]"), op::DynamicSlice());
  auto scatter =
      AllOf(op::Shape("s32[2,2,2,2]"), op::Scatter(operand, indices, update));
  VLOG(1) << module->ToString();
  EXPECT_THAT(root, op::AllReduce(op::AllReduce(
                        op::DynamicUpdateSlice(_, scatter, _, _, _, _))));
}

TEST_F(SpmdPartitioningTest, ScatterMergedIndexParallelAndIndexPassthrough) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[4,1,1,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %parameter.1 = s32[1,8,4]{2,1,0} parameter(1),
    sharding={devices=[1,4,2]0,1,2,3,4,5,6,7}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,4,2]0,1,2,3,4,5,6,7}
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(
    s32[1,8,4]{2,1,0} %parameter.1, s32[1,8,4]{2,1,0} %iota), dimensions={0},
    sharding={devices=[1,4,2]0,1,2,3,4,5,6,7}
  %parameter.2 = s32[8,4,2,2]{3,2,1,0} parameter(2),
    sharding={replicated}
  ROOT %scatter.20 = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19,
    s32[8,4,2,2]{3,2,1,0} %parameter.2),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={1,0},
    index_vector_dim=0, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  auto operand = AllOf(op::Shape("s32[2,4,2,2]"), op::Select());
  auto indices = AllOf(op::Shape("s32[2,2,2]"), op::Subtract());
  auto update = AllOf(op::Shape("s32[2,2,2,2]"), op::DynamicSlice());
  auto scatter =
      AllOf(op::Shape("s32[2,4,2,2]"), op::Scatter(operand, indices, update));
  EXPECT_THAT(root, op::AllReduce(op::DynamicUpdateSlice(
                        _, op::AllReduce(scatter), _, _, _, _)));
}

TEST_F(SpmdPartitioningTest,
       ScatterMergedOperandPassthroughAndTrivialSlicedOperand) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[2,2,2,1]0,1,2,3,4,5,6,7}
  %parameter.1 = s32[2,8,4]{2,1,0} parameter(1),
    sharding={replicated}
  %parameter.2 = s32[8,4,2,2]{3,2,1,0} parameter(2),
    sharding={replicated}
  ROOT %scatter.20 = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %parameter.1,
    s32[8,4,2,2]{3,2,1,0} %parameter.2),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  auto operand = AllOf(op::Shape("s32[4,2,1,2]"), op::Parameter());
  auto indices = AllOf(op::Shape("s32[2,8,4]"), op::Subtract());
  auto update = AllOf(op::Shape("s32[8,4,1,2]"), op::DynamicSlice());
  auto scatter =
      AllOf(op::Shape("s32[4,2,1,2]"), op::Scatter(operand, indices, update));
  EXPECT_THAT(root, op::AllReduce(op::AllReduce(op::AllReduce(
                        op::DynamicUpdateSlice(_, scatter, _, _, _, _)))));
}

TEST_F(SpmdPartitioningTest,
       ScatterMergedOperandPassthroughAndIndexPassthrough) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[1,1,2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %parameter.1 = s32[2,8,4]{2,1,0} parameter(1),
    sharding={devices=[1,2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %parameter.2 = s32[8,4,2,2]{3,2,1,0} parameter(2),
    sharding={replicated}
  ROOT %scatter.20 = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %parameter.1,
    s32[8,4,2,2]{3,2,1,0} %parameter.2),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  auto operand = AllOf(op::Shape("s32[8,4,1,2]"), op::Select());
  auto indices = AllOf(op::Shape("s32[2,4,4]"), op::CollectivePermute());
  auto update = AllOf(op::Shape("s32[4,4,1,2]"), op::DynamicSlice());
  auto scatter =
      AllOf(op::Shape("s32[8,4,1,2]"), op::Scatter(operand, indices, update));
  EXPECT_THAT(root, op::AllReduce(op::DynamicUpdateSlice(
                        _, op::AllReduce(scatter), _, _, _, _)));
}

TEST_F(SpmdPartitioningTest,
       ScatterMergedTrivialSlicedOperandAndIndexPassthrough) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[2,2,1,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %parameter.1 = s32[2,8,4]{2,1,0} parameter(1),
    sharding={devices=[1,2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %parameter.2 = s32[8,4,2,2]{3,2,1,0} parameter(2),
    sharding={replicated}
  ROOT %scatter.20 = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %parameter.1,
    s32[8,4,2,2]{3,2,1,0} %parameter.2),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  auto operand = AllOf(op::Shape("s32[4,2,2,2]"), op::Select());
  auto indices = AllOf(op::Shape("s32[2,4,4]"), op::Subtract());
  auto update = AllOf(op::Shape("s32[4,4,2,2]"), op::DynamicSlice());
  auto scatter =
      AllOf(op::Shape("s32[4,2,2,2]"), op::Scatter(operand, indices, update));
  EXPECT_THAT(root, op::AllReduce(op::AllReduce(op::DynamicUpdateSlice(
                        _, op::AllReduce(scatter), _, _, _, _))));
}

TEST_F(SpmdPartitioningTest, ScatterTrivialSlicedOperandPartial) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s64[], rhs: s64[]) -> s64[] {
  lhs = s64[] parameter(0)
  rhs = s64[] parameter(1)
  ROOT sum = s64[] add(lhs, rhs)
}

ENTRY main.4 {
  %arg.0 = s64[8,2]{1,0} parameter(0), sharding={devices=[4,2]0,1,2,3,4,5,6,7}
  %arg.1 = s32[2]{0} parameter(1), sharding={replicated}
  %arg.2 = s64[2,1]{1,0} parameter(2), sharding={replicated}
  ROOT scatter = s64[8,2]{1,0} scatter(arg.0, arg.1, arg.2),
    to_apply=add,
    update_window_dims={0,1},
    inserted_window_dims={},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0, indices_are_sorted=true, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  auto operand = AllOf(op::Shape("s64[8,1]"), op::AllReduce());
  auto indices = AllOf(op::Shape("s32[2]"), op::Subtract());
  auto update = AllOf(op::Shape("s64[2,1]"), op::Parameter());
  auto scatter =
      AllOf(op::Shape("s64[8,1]"), op::Scatter(operand, indices, update));
  EXPECT_THAT(root, op::AllReduce(op::AllReduce(op::DynamicUpdateSlice(
                        _, op::DynamicSlice(scatter, _, _), _, _))));
}

TEST_F(SpmdPartitioningTest, SortTopKNonSortDimension) {
  absl::string_view hlo_string = R"(
HloModule module

%compare-greater-than.42077 (p.0.lhs.42078: f32[],
  p.0.rhs.42079: f32[], p.1.lhs.42080: s32[], p.1.rhs.42081: s32[]) -> pred[] {
  %p.0.lhs.42078 = f32[] parameter(0)
  %bitcast-convert.135 = s32[] bitcast-convert(f32[] %p.0.lhs.42078)
  %constant.45054 = s32[] constant(0)
  %compare.133 = pred[] compare(s32[] %bitcast-convert.135,
    s32[] %constant.45054), direction=LT
  %constant.45278 = u32[] constant(2147483647)
  %bitcast-convert.136 = u32[] bitcast-convert(f32[] %p.0.lhs.42078)
  %subtract.337 = u32[] subtract(u32[] %constant.45278,
    u32[] %bitcast-convert.136)
  %bitcast-convert.137 = s32[] bitcast-convert(u32[] %subtract.337)
  %select.282 = s32[] select(pred[] %compare.133, s32[] %bitcast-convert.137,
    s32[] %bitcast-convert.135)
  %p.0.rhs.42079 = f32[] parameter(1)
  %bitcast-convert.138 = s32[] bitcast-convert(f32[] %p.0.rhs.42079)
  %compare.134 = pred[] compare(s32[] %bitcast-convert.138,
    s32[] %constant.45054), direction=LT
  %bitcast-convert.139 = u32[] bitcast-convert(f32[] %p.0.rhs.42079)
  %subtract.338 = u32[] subtract(u32[] %constant.45278,
    u32[] %bitcast-convert.139)
  %bitcast-convert.140 = s32[] bitcast-convert(u32[] %subtract.338)
  %select.283 = s32[] select(pred[] %compare.134, s32[] %bitcast-convert.140,
    s32[] %bitcast-convert.138)
  %compare.135 = pred[] compare(s32[] %select.282,
    s32[] %select.283), direction=GT
  %compare.428 = pred[] compare(s32[] %select.283,
    s32[] %select.282), direction=GT
  %compare.429 = pred[] compare(pred[] %compare.135,
    pred[] %compare.428), direction=EQ
  %p.1.lhs.42080 = s32[] parameter(2)
  %p.1.rhs.42081 = s32[] parameter(3)
  %compare.430 = pred[] compare(s32[] %p.1.lhs.42080,
    s32[] %p.1.rhs.42081), direction=LT
  ROOT %select.579 = pred[] select(pred[] %compare.429,
    pred[] %compare.430, pred[] %compare.135)
}

ENTRY %module {
  %parameter.0 = f32[2,64,32128]{2,1,0} parameter(0),
     sharding={devices=[2,1,4]0,1,2,3,4,5,6,7}
  %iota = s32[2,64,32128]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[2,1,4]0,1,2,3,4,5,6,7}
  %sort.18 = (f32[2,64,32128]{2,1,0}, s32[2,64,32128]{2,1,0}) sort(
    f32[2,64,32128]{2,1,0} %parameter.0, s32[2,64,32128]{2,1,0} %iota),
    dimensions={2}, is_stable=true, to_apply=%compare-greater-than.42077,
    sharding={{devices=[2,1,4]0,1,2,3,4,5,6,7},
    {devices=[2,1,4]0,1,2,3,4,5,6,7}}
  output = f32[2,64,32128]{2,1,0} get-tuple-element(%sort.18), index=0,
    sharding={devices=[2,1,4]0,1,2,3,4,5,6,7}
  %slice.0 = f32[2,64,2]{2,1,0} slice(f32[2,64,32128]{2,1,0} output),
    slice={[0:2], [0:64], [0:2]},
    sharding={devices=[2,1,4]0,1,2,3,4,5,6,7}
  output2 = s32[2,64,32128]{2,1,0} get-tuple-element(%sort.18), index=1,
    sharding={replicated}
  %slice.1 = s32[2,64,2]{2,1,0} slice(s32[2,64,32128]{2,1,0} output2),
    slice={[0:2], [0:64], [0:2]},
    sharding={devices=[2,1,4]0,1,2,3,4,5,6,7}
  ROOT output.t = (f32[2,64,2]{2,1,0},
    s32[2,64,2]{2,1,0}) tuple(slice.0, slice.1),
    sharding={{replicated}, {replicated}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));

  const HloInstruction* sort = FindInstruction(module.get(), "sort.0");
  EXPECT_NE(sort, nullptr);
  // The subshape of the sort changed from [2,64,32128] to [1,16,32128] due to
  // moving the sharding from the sort dim (dim 2) to other dimensions. This
  // optimization was implemented for b/258523376.
  auto sort_match =
      AllOf(op::Shape("(f32[1,16,32128], s32[1,16,32128])"), op::Sort(_, _));
  EXPECT_THAT(sort, sort_match);
}

TEST_F(SpmdPartitioningTest, SortTopKPropagateBaseShape) {
  absl::string_view hlo_string = R"(
HloModule module

%compare-greater-than.42077 (p.0.lhs.42078: f32[],
  p.0.rhs.42079: f32[], p.1.lhs.42080: s32[], p.1.rhs.42081: s32[]) -> pred[] {
  %p.0.lhs.42078 = f32[] parameter(0)
  %bitcast-convert.135 = s32[] bitcast-convert(f32[] %p.0.lhs.42078)
  %constant.45054 = s32[] constant(0)
  %compare.133 = pred[] compare(s32[] %bitcast-convert.135,
    s32[] %constant.45054), direction=LT
  %constant.45278 = u32[] constant(2147483647)
  %bitcast-convert.136 = u32[] bitcast-convert(f32[] %p.0.lhs.42078)
  %subtract.337 = u32[] subtract(u32[] %constant.45278,
    u32[] %bitcast-convert.136)
  %bitcast-convert.137 = s32[] bitcast-convert(u32[] %subtract.337)
  %select.282 = s32[] select(pred[] %compare.133, s32[] %bitcast-convert.137,
    s32[] %bitcast-convert.135)
  %p.0.rhs.42079 = f32[] parameter(1)
  %bitcast-convert.138 = s32[] bitcast-convert(f32[] %p.0.rhs.42079)
  %compare.134 = pred[] compare(s32[] %bitcast-convert.138,
    s32[] %constant.45054), direction=LT
  %bitcast-convert.139 = u32[] bitcast-convert(f32[] %p.0.rhs.42079)
  %subtract.338 = u32[] subtract(u32[] %constant.45278,
    u32[] %bitcast-convert.139)
  %bitcast-convert.140 = s32[] bitcast-convert(u32[] %subtract.338)
  %select.283 = s32[] select(pred[] %compare.134, s32[] %bitcast-convert.140,
    s32[] %bitcast-convert.138)
  %compare.135 = pred[] compare(s32[] %select.282,
    s32[] %select.283), direction=GT
  %compare.428 = pred[] compare(s32[] %select.283,
    s32[] %select.282), direction=GT
  %compare.429 = pred[] compare(pred[] %compare.135,
    pred[] %compare.428), direction=EQ
  %p.1.lhs.42080 = s32[] parameter(2)
  %p.1.rhs.42081 = s32[] parameter(3)
  %compare.430 = pred[] compare(s32[] %p.1.lhs.42080,
    s32[] %p.1.rhs.42081), direction=LT
  ROOT %select.579 = pred[] select(pred[] %compare.429,
    pred[] %compare.430, pred[] %compare.135)
}

ENTRY %module {
  %parameter.0 = f32[2,64,32128]{2,1,0} parameter(0),
     sharding={devices=[1,1,8]0,1,2,3,4,5,6,7}
  %iota = s32[2,64,32128]{2,1,0} iota(), iota_dimension=2,
    sharding={devices=[1,1,8]0,1,2,3,4,5,6,7}
  %sort.18 = (f32[2,64,32128]{2,1,0}, s32[2,64,32128]{2,1,0}) sort(
    f32[2,64,32128]{2,1,0} %parameter.0, s32[2,64,32128]{2,1,0} %iota),
    dimensions={2}, is_stable=true, to_apply=%compare-greater-than.42077,
    sharding={{devices=[1,1,8]0,1,2,3,4,5,6,7},
    {devices=[1,1,8]0,1,2,3,4,5,6,7}}
  output = f32[2,64,32128]{2,1,0} get-tuple-element(%sort.18), index=0,
    sharding={devices=[1,1,8]0,1,2,3,4,5,6,7}
  %slice.0 = f32[2,64,2]{2,1,0} slice(f32[2,64,32128]{2,1,0} output),
    slice={[0:2], [0:64], [0:2]},
    sharding={devices=[1,1,8]0,1,2,3,4,5,6,7}
  output2 = s32[2,64,32128]{2,1,0} get-tuple-element(%sort.18), index=1,
    sharding={replicated}
  %slice.1 = s32[2,64,2]{2,1,0} slice(s32[2,64,32128]{2,1,0} output2),
    slice={[0:2], [0:64], [0:2]},
    sharding={devices=[1,1,8]0,1,2,3,4,5,6,7}
  ROOT output.t = (f32[2,64,2]{2,1,0},
    s32[2,64,2]{2,1,0}) tuple(slice.0, slice.1),
    sharding={{replicated}, {replicated}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  auto all_reduce_val =
      AllOf(op::Shape("f32[2,64,2]"),
            op::AllReduce(op::DynamicUpdateSlice(_, _, _, _, _)));
  auto all_reduce_idx =
      AllOf(op::Shape("s32[2,64,2]"),
            op::AllReduce(op::DynamicUpdateSlice(_, _, _, _, _)));
  auto tuple = AllOf(op::Shape("(f32[2,64,2], s32[2,64,2])"),
                     op::Tuple(all_reduce_val, all_reduce_idx));
  EXPECT_THAT(root, tuple);
}

TEST_F(SpmdPartitioningTest, GatherIndexOnlyCorrectReplacement) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = bf16[1,8,6,6]{3,2,1,0} parameter(0),
    sharding={replicated}
  %parameter.1 = s32[2,4]{1,0} parameter(1),
     sharding={devices=[2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %gather.100 = bf16[2,1,8,1,6]{4,3,2,1,0} gather(
    bf16[1,8,6,6]{3,2,1,0} %parameter.0, s32[2,4]{1,0} %parameter.1),
    offset_dims={1,2,3,4}, collapsed_slice_dims={}, start_index_map={0,1,2,3},
    index_vector_dim=1, slice_sizes={1,8,1,6},
    sharding={devices=[2,1,4,1,1]0,1,2,3,4,5,6,7}
  %constant.45590 = s32[] constant(0), sharding={replicated}
  %broadcast.54515 = s32[2,64,1,1]{3,2,1,0} broadcast(s32[] %constant.45590),
    dimensions={},
    sharding={devices=[2,1,1,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %reshape.4243 = bf16[2,8,6]{2,1,0} reshape(
    bf16[2,1,8,1,6]{4,3,2,1,0} %gather.100),
    sharding={devices=[2,4,1]0,1,2,3,4,5,6,7}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));

  const HloInstruction* root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(op::Shape("bf16[1,8,6,6]"), op::Parameter());
  auto param1 = AllOf(op::Shape("s32[1,4]"), op::Parameter());
  auto reshape = AllOf(
      op::Shape("bf16[1,2,6]"),
      op::Reshape(op::DynamicSlice(op::Gather(param0, param1), _, _, _, _, _)));
  EXPECT_THAT(root, reshape);
}

TEST_F(SpmdPartitioningTest, GatherRegressionTest1) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[1,4] parameter(0), sharding={devices=[1,8]0,1,2,3,4,5,6,7}
  %iota.10 = s32[4]{0} iota(), iota_dimension=0, sharding={devices=[8]0,1,2,3,4,5,6,7}
  ROOT %gather.44 = s32[1,4]{1,0} gather(%parameter.0, %iota.10),
    offset_dims={0}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=1,
    slice_sizes={1,1}, sharding={devices=[1,8]0,1,2,3,4,5,6,7}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));

  const HloInstruction* root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(op::Shape("s32[1,1]"), op::Parameter());
  EXPECT_THAT(root, op::Gather(param0, _));
}

TEST_F(SpmdPartitioningTest, WindowedEinsumPreferMemoryFootprint) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = bf16[128,1024,4,4,1152,1,1]{6,5,4,3,2,1,0} parameter(0),
    sharding={devices=[4,1,2,1,1,1,1]0,1,2,3,4,5,6,7}
  %parameter.1 = bf16[4,4,1152,4,176,256,1]{6,5,4,3,2,1,0} parameter(1),
    sharding={devices=[2,2,1,2,1,1,1]0,1,2,3,4,5,6,7}
  %convolution.3 = bf16[128,1024,4,176,256,1,1]{6,5,4,3,2,1,0}
    convolution(bf16[128,1024,4,4,1152,1,1]{6,5,4,3,2,1,0} %parameter.0,
    bf16[4,4,1152,4,176,256,1]{6,5,4,3,2,1,0} %parameter.1),
    window={size=1x4x176x4x4 pad=0_0x3_3x175_175x0_0x0_0
    rhs_reversal=0x1x1x0x0}, dim_labels=0b34f12_34i12o0->0b12f34,
    sharding={devices=[4,1,2,1,1,1,1]0,1,2,3,4,5,6,7}
  ROOT %reshape.3973 = bf16[128,1024,4,176,256]{4,3,2,1,0}
    reshape(bf16[128,1024,4,176,256,1,1]{6,5,4,3,2,1,0} %convolution.3),
    sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/8,
                           /*conv_halo_exchange_always_on_lhs =*/true,
                           /*choose_faster_windowed_einsum =*/false));
  const HloInstruction* while_inst = FindInstruction(module.get(), "while");
  EXPECT_NE(while_inst, nullptr);
  const HloComputation* cond_comp = while_inst->while_condition();
  const HloInstruction* root = cond_comp->root_instruction();
  EXPECT_THAT(root, op::Compare(_, op::Constant()));
  const HloConstantInstruction* iterations =
      Cast<HloConstantInstruction>(root->operand(1));
  EXPECT_TRUE(iterations->literal().GetFirstInteger());
  EXPECT_EQ(*iterations->literal().GetFirstInteger(), 4);
}

TEST_F(SpmdPartitioningTest, WindowedEinsumPreferNumberIterations) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = bf16[128,1024,4,4,1152,1,1]{6,5,4,3,2,1,0} parameter(0),
    sharding={devices=[4,1,2,1,1,1,1]0,1,2,3,4,5,6,7}
  %parameter.1 = bf16[4,4,1152,4,176,256,1]{6,5,4,3,2,1,0} parameter(1),
    sharding={devices=[2,2,1,2,1,1,1]0,1,2,3,4,5,6,7}
  %convolution.3 = bf16[128,1024,4,176,256,1,1]{6,5,4,3,2,1,0}
    convolution(bf16[128,1024,4,4,1152,1,1]{6,5,4,3,2,1,0} %parameter.0,
    bf16[4,4,1152,4,176,256,1]{6,5,4,3,2,1,0} %parameter.1),
    window={size=1x4x176x4x4 pad=0_0x3_3x175_175x0_0x0_0
    rhs_reversal=0x1x1x0x0}, dim_labels=0b34f12_34i12o0->0b12f34,
    sharding={devices=[4,1,2,1,1,1,1]0,1,2,3,4,5,6,7}
  ROOT %reshape.3973 = bf16[128,1024,4,176,256]{4,3,2,1,0}
    reshape(bf16[128,1024,4,176,256,1,1]{6,5,4,3,2,1,0} %convolution.3),
    sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/8,
                           /*conv_halo_exchange_always_on_lhs =*/true,
                           /*choose_faster_windowed_einsum =*/true));
  const HloInstruction* while_inst = FindInstruction(module.get(), "while");
  EXPECT_NE(while_inst, nullptr);
  const HloComputation* cond_comp = while_inst->while_condition();
  const HloInstruction* root = cond_comp->root_instruction();
  EXPECT_THAT(root, op::Compare(_, op::Constant()));
  const HloConstantInstruction* iterations =
      Cast<HloConstantInstruction>(root->operand(1));
  EXPECT_TRUE(iterations->literal().GetFirstInteger());
  EXPECT_EQ(*iterations->literal().GetFirstInteger(), 2);
}

TEST_F(SpmdPartitioningTest, WindowedEinsumPreferNumberIterations2) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = bf16[512,1024,16,36,256]{4,3,2,1,0} parameter(0)
  %lhs.copy = bf16[512,1024,16,36,256]{4,3,2,1,0} copy(%lhs),
  sharding={devices=[8,1,4,1,1]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,
            18,19,20,21,22,23,24,25,26,27,28,29,30,31}
  %rhs = bf16[512,1024,16,4,288]{4,3,2,1,0} parameter(1)
  %rhs.copy = bf16[512,1024,16,4,288]{4,3,2,1,0} copy(%rhs),
    sharding={devices=[8,1,4,1,1]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
              17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}
  %reshape.2556 = bf16[512,1024,16,4,288,1,1]{6,5,4,3,2,1,0} reshape(
    bf16[512,1024,16,4,288]{4,3,2,1,0} %rhs.copy), sharding={
      devices=[8,1,4,1,1,1,1]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
        20,21,22,23,24,25,26,27,28,29,30,31}
  %reshape.2570 = bf16[512,1024,16,36,256,1,1]{6,5,4,3,2,1,0}
    reshape(bf16[512,1024,16,36,256]{4,3,2,1,0} %lhs.copy), sharding={
    devices=[8,1,4,1,1,1,1]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
             20,21,22,23,24,25,26,27,28,29,30,31}
  %convolution.10 = bf16[16,36,256,16,4,288,1]{6,5,4,3,2,1,0}
    convolution(bf16[512,1024,16,36,256,1,1]{6,5,4,3,2,1,0} %reshape.2570,
    bf16[512,1024,16,4,288,1,1]{6,5,4,3,2,1,0} %reshape.2556),
    window={size=1x1x16x4x512 pad=0_0x0_0x15_15x3_3x0_0 rhs_reversal=0x0x1x1x0},
    dim_labels=4f01b23_4i23o01->01b23f4, sharding={devices=[4,1,1,4,2,1,1]0,4,8,
    12,16,20,24,28,1,5,9,13,17,21,25,29,2,6,10,14,18,22,26,30,3,7,11,15,19,23,
    27,31}
  ROOT %output = bf16[16,36,256,16,4,288,1]{6,5,4,3,2,1,0}
   copy(%convolution.10), sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/32,
                           /*conv_halo_exchange_always_on_lhs =*/true,
                           /*choose_faster_windowed_einsum =*/true));
  const HloInstruction* while_inst = FindInstruction(module.get(), "while");
  EXPECT_NE(while_inst, nullptr);
  const HloComputation* cond_comp = while_inst->while_condition();
  const HloInstruction* root = cond_comp->root_instruction();
  EXPECT_THAT(root, op::Compare(_, op::Constant()));
  const HloConstantInstruction* iterations =
      Cast<HloConstantInstruction>(root->operand(1));
  EXPECT_TRUE(iterations->literal().GetFirstInteger());
  EXPECT_EQ(*iterations->literal().GetFirstInteger(), 4);
}

TEST_F(SpmdPartitioningTest, WindowedEinsumPreferMemoryFootprint2) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = bf16[512,1024,16,36,256]{4,3,2,1,0} parameter(0)
  %lhs.copy = bf16[512,1024,16,36,256]{4,3,2,1,0} copy(%lhs),
  sharding={devices=[8,1,4,1,1]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,
            18,19,20,21,22,23,24,25,26,27,28,29,30,31}
  %rhs = bf16[512,1024,16,4,288]{4,3,2,1,0} parameter(1)
  %rhs.copy = bf16[512,1024,16,4,288]{4,3,2,1,0} copy(%rhs),
    sharding={devices=[8,1,4,1,1]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
              17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}
  %reshape.2556 = bf16[512,1024,16,4,288,1,1]{6,5,4,3,2,1,0} reshape(
    bf16[512,1024,16,4,288]{4,3,2,1,0} %rhs.copy), sharding={
      devices=[8,1,4,1,1,1,1]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
        20,21,22,23,24,25,26,27,28,29,30,31}
  %reshape.2570 = bf16[512,1024,16,36,256,1,1]{6,5,4,3,2,1,0}
    reshape(bf16[512,1024,16,36,256]{4,3,2,1,0} %lhs.copy), sharding={
    devices=[8,1,4,1,1,1,1]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
             20,21,22,23,24,25,26,27,28,29,30,31}
  %convolution.10 = bf16[16,36,256,16,4,288,1]{6,5,4,3,2,1,0}
    convolution(bf16[512,1024,16,36,256,1,1]{6,5,4,3,2,1,0} %reshape.2570,
    bf16[512,1024,16,4,288,1,1]{6,5,4,3,2,1,0} %reshape.2556),
    window={size=1x1x16x4x512 pad=0_0x0_0x15_15x3_3x0_0 rhs_reversal=0x0x1x1x0},
    dim_labels=4f01b23_4i23o01->01b23f4, sharding={devices=[4,1,1,4,2,1,1]0,4,8,
    12,16,20,24,28,1,5,9,13,17,21,25,29,2,6,10,14,18,22,26,30,3,7,11,15,19,23,
    27,31}
  ROOT %output = bf16[16,36,256,16,4,288,1]{6,5,4,3,2,1,0}
   copy(%convolution.10), sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/32,
                           /*conv_halo_exchange_always_on_lhs =*/true,
                           /*choose_faster_windowed_einsum =*/false));
  const HloInstruction* while_inst = FindInstruction(module.get(), "while");
  EXPECT_NE(while_inst, nullptr);
  const HloComputation* cond_comp = while_inst->while_condition();
  const HloInstruction* root = cond_comp->root_instruction();
  EXPECT_THAT(root, op::Compare(_, op::Constant()));
  const HloConstantInstruction* iterations =
      Cast<HloConstantInstruction>(root->operand(1));
  EXPECT_TRUE(iterations->literal().GetFirstInteger());
  EXPECT_EQ(*iterations->literal().GetFirstInteger(), 8);
}

TEST_F(SpmdPartitioningTest, ContractingPartitionDotOperandsSlicedWrong) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[8,2,15,4] parameter(0)
  %lhs.copy = f32[8,2,15,4] copy(%lhs),
    sharding={devices=[1,2,4,1]0,1,2,3,4,5,6,7}
  %rhs = f32[2,15,4] parameter(1)
  %rhs.copy = f32[2,15,4] copy(%rhs),
    sharding={devices=[2,4,1]0,1,2,3,4,5,6,7}
  %dot = f32[8,2,2] dot(%lhs.copy, %rhs.copy),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={2,3}, rhs_contracting_dims={1,2},
    operand_precision={HIGH,HIGH},
    sharding={devices=[2,2,2]0,1,2,3,4,5,6,7}
  ROOT %output = f32[8,2,2] copy(%dot), sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/8,
                           /*conv_halo_exchange_always_on_lhs =*/true,
                           /*choose_faster_windowed_einsum =*/true));

  const HloInstruction* dot_op = FindInstruction(module.get(), HloOpcode::kDot);
  auto op1 = op::Shape("f32[4,2,4,4]");
  auto op2 = op::Shape("f32[2,4,4]");
  EXPECT_THAT(dot_op, op::Dot(op1, op2));
}

TEST_F(SpmdPartitioningTest, PartitionDotGroupOnBatchContractingReshard) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,32,24,4096] parameter(0),
    sharding={devices=[2,1,1,2]0,1,2,3}
  %rhs = f32[32,4096,1024] parameter(1),
    sharding={devices=[2,2,1]0,1,2,3}
  ROOT %dot = f32[32,32,24,1024] dot(%lhs, %rhs),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={3}, rhs_contracting_dims={1},
    sharding={devices=[1,2,1,2]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_devices=*/4,
                           /*conv_halo_exchange_always_on_lhs=*/true,
                           /*choose_faster_windowed_einsum=*/true));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  auto dot = AllOf(op::Shape("f32[16,32,24,1024]"),
                   op::Dot(op::Parameter(0), op::Parameter(1)));
  auto reduce_scatter = AllOf(op::Shape("f32[16,32,24,512]"),
                              op::DynamicSlice(op::AllReduce(dot), _, _, _, _));
  EXPECT_THAT(root, AllOf(op::Reshape(op::Transpose(
                              op::AllToAll(op::Reshape(reduce_scatter)))),
                          op::Shape("f32[32,16,24,512]")));
}

TEST_F(SpmdPartitioningTest, PartitionPassthroughScatterCorrectOutputSharding) {
  absl::string_view hlo_string = R"(
HloModule module

%scatter_add (parameter.0: bf16[], parameter.1: bf16[]) -> bf16[] {
  %parameter.0 = bf16[] parameter(0)
  %parameter.1 = bf16[] parameter(1)
  ROOT %add = bf16[] add(bf16[] %parameter.0, bf16[] %parameter.1)
}

ENTRY entry {
  %operand = bf16[2,1024]{1,0} parameter(0),
    sharding={devices=[1,2]0,1}
  %indices = s32[8,512,1]{2,1,0} parameter(1),
    sharding={replicated}
  %updates = bf16[8,512,1024]{2,1,0} parameter(2),
    sharding={devices=[1,1,2]0,1}
  ROOT %scatter = bf16[2,1024]{1,0} scatter(bf16[2,1024]{1,0} %operand,
    s32[8,512,1]{2,1,0} %indices,
    bf16[8,512,1024]{2,1,0} %updates), update_window_dims={2},
    inserted_window_dims={0}, scatter_dims_to_operand_dims={0},
    index_vector_dim=2, to_apply=%scatter_add,
    sharding={devices=[1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  auto scatter = AllOf(op::Shape("bf16[2,512]"), op::Scatter(_, _, _));
  EXPECT_THAT(root, scatter);
}

bool IsTrivialCollectivePermute(HloInstruction* hlo) {
  if (hlo->opcode() != HloOpcode::kCollectivePermute) {
    return false;
  }
  if (hlo->source_target_pairs().empty()) {
    return true;
  }
  return absl::c_all_of(hlo->source_target_pairs(),
                        [](const std::pair<int64_t, int64_t>& pair) {
                          return pair.first == pair.second;
                        });
}

TEST_F(SpmdPartitioningTest, CollectivePermuteSimplifyIdentity) {
  absl::string_view hlo_string = R"(
HloModule test

ENTRY entry {
  %parameter.7 = f32[3,16] parameter(0), sharding={devices=[1,2]0,1}
  %constant.7 = f32[] constant(0)
  %pad.3 = f32[3,18] pad(f32[3,16] %parameter.7, f32[] %constant.7), padding=0_0x1_1, sharding={devices=[1,2]0,1}
  // Shift right by 16.
  %slice.8 = f32[3,16] slice(f32[3,18] %pad.3), slice={[0:3], [2:18]}, sharding={devices=[1,2]0,1}
  %slice.9 = f32[3,2] slice(f32[3,18] %pad.3), slice={[0:3], [0:2]}, sharding={devices=[1,2]0,1}
  ROOT %concatenate.6 = f32[3,18] concatenate(f32[3,16] %slice.8, f32[3,2] %slice.9), dimensions={1}, sharding={devices=[1,2]0,1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  // Check that the partitioned code does not have a "trivial" collective
  // permute (which would degenerate to a copy).
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* hlo : computation->instructions()) {
      EXPECT_FALSE(IsTrivialCollectivePermute(hlo)) << hlo->ToString();
    }
  }
}

TEST_F(SpmdPartitioningTest, CollectivePermuteSimplifyZero) {
  absl::string_view hlo_string = R"(
HloModule test

ENTRY entry {
  %parameter = f32[3,16,16,16,16,132]{5,4,3,2,1,0} parameter(0), sharding={devices=[1,2,1,1,1,1]0,1}
  %slice = f32[3,1,16,16,16,132]{5,4,3,2,1,0} slice(f32[3,16,16,16,16,132]{5,4,3,2,1,0} %parameter), slice={[0:3], [15:16], [0:16], [0:16], [0:16], [0:132]}, sharding={devices=[1,2,1,1,1,1]0,1}
  %c0 = f32[] constant(0)
  ROOT %pad = f32[3,18,16,16,16,132]{5,4,3,2,1,0} pad(f32[3,1,16,16,16,132]{5,4,3,2,1,0} %slice, f32[] %c0), padding=0_0x0_17x0_0x0_0x0_0x0_0, sharding={devices=[1,2,1,1,1,1]0,1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  // Check that the partitioned code does not have a collective permute with an
  // empty source_target_pair list.
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* hlo : computation->instructions()) {
      EXPECT_FALSE(IsTrivialCollectivePermute(hlo)) << hlo->ToString();
    }
  }
}

TEST_F(SpmdPartitioningTest, PadWithWrapPattern) {
  absl::string_view hlo_string = R"(
HloModule xla_computation_apply_fn__4.61

ENTRY %xla_computation_apply_fn__4.61 (parameter.7: f32[3,16,16,16,16,132]) -> f32[3,18,16,16,16,132] {
  %parameter.7 = f32[3,16,16,16,16,132]{5,4,3,2,1,0} parameter(0), sharding={devices=[1,2,1,1,1,1]0,1}
  %slice.2 = f32[3,1,16,16,16,132]{5,4,3,2,1,0} slice(f32[3,16,16,16,16,132]{5,4,3,2,1,0} %parameter.7), slice={[0:3], [15:16], [0:16], [0:16], [0:16], [0:132]}, sharding={devices=[1,2,1,1,1,1]0,1}
  %slice.3 = f32[3,1,16,16,16,132]{5,4,3,2,1,0} slice(f32[3,16,16,16,16,132]{5,4,3,2,1,0} %parameter.7), slice={[0:3], [0:1], [0:16], [0:16], [0:16], [0:132]}, sharding={devices=[1,2,1,1,1,1]0,1}
  ROOT %concatenate.3 = f32[3,18,16,16,16,132]{5,4,3,2,1,0} concatenate(f32[3,1,16,16,16,132]{5,4,3,2,1,0} %slice.2, f32[3,16,16,16,16,132]{5,4,3,2,1,0} %parameter.7, f32[3,1,16,16,16,132]{5,4,3,2,1,0} %slice.3), dimensions={1}, sharding={devices=[1,2,1,1,1,1]0,1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  // Check that the partitioned code does not have all-reduce and two
  // non-trivial collective permute instructions.
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* hlo : computation->instructions()) {
      EXPECT_FALSE(IsTrivialCollectivePermute(hlo)) << hlo->ToString();
      EXPECT_NE(hlo->opcode(), HloOpcode::kAllReduce) << hlo->ToString();
    }
  }
}

TEST_F(SpmdPartitioningTest, PadWrapWithNegatePattern) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %parameter.1 = f32[1,18] parameter(0), sharding={devices=[1,2]0,1}
  %slice.16 = f32[1,2] slice(f32[1,18] %parameter.1), slice={[0:1], [16:18]}, sharding={devices=[1,2]0,1}
  %negate.2 = f32[1,2] negate(f32[1,2] %slice.16), sharding={devices=[1,2]0,1}
  %slice.17 = f32[1,2] slice(f32[1,18] %parameter.1), slice={[0:1], [0:2]}, sharding={devices=[1,2]0,1}
  %negate.3 = f32[1,2] negate(f32[1,2] %slice.17), sharding={devices=[1,2]0,1}
  ROOT %concatenate.13 = f32[1,22] concatenate(f32[1,2] %negate.2, f32[1,18] %parameter.1, f32[1,2] %negate.3), dimensions={1}, sharding={devices=[1,2]0,1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  // Check that the partitioned code does not have all-reduce or trivial
  // collective permute
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* hlo : computation->instructions()) {
      EXPECT_FALSE(IsTrivialCollectivePermute(hlo)) << hlo->ToString();
      EXPECT_NE(hlo->opcode(), HloOpcode::kAllReduce) << hlo->ToString();
    }
  }
}

TEST_F(SpmdPartitioningTest, PadWrapWithMultipleModifiersPattern) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %parameter.1 = f32[1,18] parameter(0), sharding={devices=[1,2]0,1}
  %slice.16 = f32[1,2] slice(f32[1,18] %parameter.1), slice={[0:1], [16:18]}, sharding={devices=[1,2]0,1}
  %mod0.16 = f32[1,2] rsqrt(f32[1,2] %slice.16), sharding={devices=[1,2]0,1}
  %mod1.16 = f32[1,2] sine(f32[1,2] %mod0.16), sharding={devices=[1,2]0,1}
  %slice.17 = f32[1,2] slice(f32[1,18] %parameter.1), slice={[0:1], [0:2]}, sharding={devices=[1,2]0,1}
  %mod0.17 = f16[1,2] convert(f32[1,2] %slice.17), sharding={devices=[1,2]0,1}
  %mod1.17 = f16[1,2] cosine(f16[1,2] %mod0.17), sharding={devices=[1,2]0,1}
  %mod2.17 = f32[1,2] convert(f16[1,2] %mod1.17), sharding={devices=[1,2]0,1}
  ROOT %concatenate.13 = f32[1,22] concatenate(f32[1,2] %mod1.16, f32[1,18] %parameter.1, f32[1,2] %mod2.17), dimensions={1}, sharding={devices=[1,2]0,1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  // Check that the partitioned code does not have all-reduce or trivial
  // collective permute. Also make sure modifiers have the right dependencies.
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* hlo : computation->instructions()) {
      const HloOpcode op = hlo->opcode();
      EXPECT_FALSE(IsTrivialCollectivePermute(hlo)) << hlo->ToString();
      EXPECT_NE(op, HloOpcode::kAllReduce) << hlo->ToString();
      if (hlo->operand_count() != 1) {
        continue;
      }
      const PrimitiveType type = hlo->shape().element_type();
      const HloOpcode child_op = hlo->operand(0)->opcode();
      const PrimitiveType child_type = hlo->operand(0)->shape().element_type();

      if (op == HloOpcode::kSin) {
        EXPECT_EQ(child_op, HloOpcode::kRsqrt);
      } else if (op == HloOpcode::kConvert && type == F32) {
        EXPECT_EQ(child_op, HloOpcode::kCos);
        EXPECT_EQ(child_type, F16);
      } else if (op == HloOpcode::kCos) {
        EXPECT_EQ(child_op, HloOpcode::kConvert);
        EXPECT_EQ(child_type, F16);
      }
    }
  }
}

TEST_F(SpmdPartitioningTest, BroadcastAsReplicate) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[1,1] parameter(0), sharding={devices=[2,2]0,1,2,3}
  ROOT %copy = f32[1,1] copy(%param0), sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(op::Parameter(0), op::Shape("f32[1,1]"));
  EXPECT_THAT(root, AllOf(op::Copy(op::AllReduce(op::Select(_, param0, _))),
                          op::Shape("f32[1,1]")));
}

TEST_F(SpmdPartitioningTest, BroadcastAsReplicate2) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[1,2] parameter(0), sharding={devices=[2,2]0,1,2,3}
  ROOT %copy = f32[1,2] copy(%param0), sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(op::Parameter(0), op::Shape("f32[1,1]"));
  auto broadcast =
      AllOf(op::AllReduce(op::Select(_, param0, _)), op::Shape("f32[1,1]"));
  EXPECT_THAT(
      root,
      AllOf(op::Copy(op::AllReduce(op::DynamicUpdateSlice(_, broadcast, _, _))),
            op::Shape("f32[1,2]")));
}

TEST_F(SpmdPartitioningTest, BroadcastAsReplicate3) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[1,1] parameter(0),
    sharding={devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}
  ROOT %copy = f32[1,1] copy(%param0), sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  const auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(op::Parameter(0), op::Shape("f32[1,1]"));
  EXPECT_THAT(root, AllOf(op::Copy(op::AllReduce(op::Select(_, param0, _))),
                          op::Shape("f32[1,1]")));
}

TEST_F(SpmdPartitioningTest, TupleWithSubgroupManual) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  constant = f32[6,3]{1,0}
    constant({{1,3,7},{5,1,4},{1,2,8},{2,3,7},{5,2,4},{2,2,8}}),
    sharding={replicated}
  param = (f32[6,3]{1,0}, f32[]) parameter(0),
    sharding={{devices=[2,1,2]0,1,2,3 last_tile_dims={manual}},{replicated}}
  gte = f32[6,3]{1,0} get-tuple-element(param), index=0,
    sharding={devices=[2,1,2]0,1,2,3 last_tile_dims={manual}}
  ROOT tuple = (f32[6,3]{1,0}, f32[6,3]{1,0}) tuple(constant, gte),
    sharding={{replicated},{devices=[2,1,2]0,1,2,3 last_tile_dims={manual}}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              op::Tuple(op::Constant(), op::GetTupleElement(op::Parameter(0))));
}

TEST_F(SpmdPartitioningTest, SubgroupManualSharedOperand) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  constant = f32[] constant(1), sharding={replicated}
  broadcast = f32[2,2] broadcast(constant), dimensions={},
    sharding={devices=[2,1,2]0,1,2,3 last_tile_dims={manual}}
  ROOT add = f32[2,2] add(broadcast, broadcast),
    sharding={devices=[2,1,2]0,1,2,3 last_tile_dims={manual}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Add(op::Broadcast(op::Constant()),
                            op::Broadcast(op::Constant())));
}

TEST_F(SpmdPartitioningTest, SubgroupManualAllReduce) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  param = f32[2,2] parameter(0),
    sharding={devices=[2,1,2]0,2,1,3 last_tile_dims={manual}}
  ROOT all-reduce = f32[2,2]{1,0} all-reduce(param), to_apply=sum,
    replica_groups={{2,0},{1,3}}, use_global_device_ids=true, channel_id=1,
    sharding={devices=[2,1,2]0,2,1,3 last_tile_dims={manual}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              AllOf(op::AllReduce(op::Parameter(0)), op::Shape("f32[1,2]")));
  EXPECT_EQ(root->replica_groups().size(), 2);
}

TEST_F(SpmdPartitioningTest, SubgroupIllegalManualAllReduce) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  param = f32[2,2] parameter(0),
    sharding={devices=[2,1,2]0,2,1,3 last_tile_dims={manual}}
  ROOT all-reduce = f32[2,2]{1,0} all-reduce(param), to_apply=sum,
    replica_groups={{1,0},{2,3}}, use_global_device_ids=true, channel_id=1,
    sharding={devices=[2,1,2]0,2,1,3 last_tile_dims={manual}}
}
)";

  auto module_status = PartitionComputation(hlo_string, /*num_devices=*/4);
  EXPECT_FALSE(module_status.status().ok());
  EXPECT_THAT(module_status.status().ToString(),
              ::testing::HasSubstr("Manual all-reduce across devices that "
                                   "belong to different manual subgroups"));
}

TEST_F(SpmdPartitioningTest, SubgroupManualReduce) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY entry {
  constant = f32[] constant(0),
    sharding={devices=[2,2]0,1,2,3 last_tile_dims={manual,replicated}}
  param = f32[2,2] parameter(0),
    sharding={devices=[2,1,2]0,2,1,3 last_tile_dims={manual}}
  ROOT reduce = f32[2] reduce(param, constant), dimensions={0}, to_apply=sum,
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dims={manual,replicated}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              op::AllReduce(op::Reduce(op::Parameter(0), op::Constant())));
  EXPECT_EQ(root->replica_groups().size(), 2);
}

TEST_F(SpmdPartitioningTest, ScatterPreferUpdateIndexIfSmaller) {
  absl::string_view hlo_string = R"(
HloModule module

%scatter_add_reducer__33.191857 (parameter.191858: bf16[], parameter.191859: bf16[]) -> bf16[] {
  %parameter.191858 = bf16[] parameter(0)
  %parameter.191859 = bf16[] parameter(1)
  ROOT %add.4425 = bf16[] add(bf16[] %parameter.191858, bf16[] %parameter.191859)
}

ENTRY entry {
  p1 = s32[2048,1024,1]{2,1,0} parameter(0)
  p2 = bf16[2048,1024,2040]{2,1,0} parameter(1)
  %constant.8635 = bf16[] constant(0)
  %broadcast.21781 = bf16[50048,2040]{1,0} broadcast(bf16[] %constant.8635), dimensions={},
    sharding={devices=[1,2,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %select.1954 = s32[2048,1024,1]{2,1,0} copy(%p1), sharding={devices=[4,1,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %slice.1274 = bf16[2048,1024,2040]{2,1,0} copy(%p2),
  sharding={devices=[4,1,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %scatter.34 = bf16[50048,2040]{1,0} scatter(bf16[50048,2040]{1,0} %broadcast.21781,
    s32[2048,1024,1]{2,1,0} %select.1954, bf16[2048,1024,2040]{2,1,0} %slice.1274),
    update_window_dims={2}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0},
    index_vector_dim=2, to_apply=%scatter_add_reducer__33.191857,
    sharding={devices=[1,2,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT c = bf16[50048,2040]{1,0} copy(scatter.34),
    sharding={replicated}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, op::Copy(op::AllReduce(op::DynamicUpdateSlice(
                _,
                op::CollectivePermute(op::AllReduce(op::Scatter(
                    op::Shape("bf16[50048,1020]"), op::Shape("s32[512,1024,1]"),
                    op::Shape("bf16[512,1024,1020]")))),
                _, _))));
}

TEST_F(SpmdPartitioningTest, ScatterPreferTrivialIfSmallerThanIndices) {
  absl::string_view hlo_string = R"(
HloModule module

%scatter_add_reducer__33.191857 (parameter.191858: bf16[], parameter.191859: bf16[]) -> bf16[] {
  %parameter.191858 = bf16[] parameter(0)
  %parameter.191859 = bf16[] parameter(1)
  ROOT %add.4425 = bf16[] add(bf16[] %parameter.191858, bf16[] %parameter.191859)
}

ENTRY entry {
  p1 = s32[32,512,3]{2,1,0} parameter(0)
  p2 = bf16[32,512]{1,0} parameter(1)
  %constant.8635 = bf16[] constant(0)
  %broadcast.21781 = bf16[32,512,50001]{2,1,0} broadcast(bf16[] %constant.8635), dimensions={},
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %select.1954 = s32[32,512,3]{2,1,0} copy(%p1), sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %slice.1274 = bf16[32,512]{1,0} copy(%p2),
  sharding={devices=[1,4,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %scatter.34 = bf16[32,512,50001]{2,1,0} scatter(bf16[32,512,50001]{2,1,0} %broadcast.21781,
    s32[32,512,3]{2,1,0} %select.1954, bf16[32,512]{1,0} %slice.1274),
    update_window_dims={}, inserted_window_dims={0,1,2}, scatter_dims_to_operand_dims={0,1,2},
    index_vector_dim=2, to_apply=%scatter_add_reducer__33.191857,
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT c = bf16[32,512,50001]{2,1,0} copy(scatter.34),
    sharding={replicated}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      op::Copy(op::AllReduce(op::DynamicUpdateSlice(
          _,
          op::Scatter(op::Shape("bf16[32,128,50001]"),
                      op::Shape("s32[32,512,3]"), op::Shape("bf16[32,512]")),
          _, _, _))));
}

TEST_F(SpmdPartitioningTest, GatherOperandPassthroughIndexPassthrough) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[2,9] parameter(0), sharding={replicated}
  %indices = s32[7] parameter(1), sharding={replicated}
  %input.copy = f32[2,9] copy(%input), sharding={devices=[1,2,2]1,0,3,2 last_tile_dim_replicate}
  %indices.copy = s32[7] copy(%indices), sharding={devices=[2,2]1,2,3,0 last_tile_dim_replicate}
  %gather = f32[7,9] gather(%input.copy, %indices.copy), offset_dims={1},
    collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1,
    slice_sizes={1,9}, sharding={devices=[2,2]0,1,2,3}
  ROOT %copy = f32[7,9] copy(%gather), sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  const HloInstruction* gather = FindInstruction(module.get(), "gather.1");
  EXPECT_NE(gather, nullptr);
  EXPECT_THAT(gather,
              AllOf(op::Shape("f32[4,5]"),
                    op::Gather(op::Shape("f32[2,5]"), op::Shape("s32[4]"))));
}

TEST_F(SpmdPartitioningTest, GatherIndexPassthroughTrivialSlice) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[17,9] parameter(0)
  %indices = s32[2,3] parameter(1)
  %input.copy = f32[17,9] copy(%input), sharding={devices=[2,1,2]3,2,1,0 last_tile_dim_replicate}
  %indices.copy = s32[2,3] copy(%indices), sharding={devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}
  %gather = f32[2,3,9] gather(%input.copy, %indices.copy), offset_dims={2},
    collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=2,
    slice_sizes={1,9}, sharding={devices=[2,1,1,2]1,0,3,2 last_tile_dim_replicate}
  ROOT %copy = f32[2,3,9] copy(%gather), sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  const HloInstruction* gather = FindInstruction(module.get(), "gather.1");
  EXPECT_NE(gather, nullptr);
  EXPECT_THAT(gather,
              AllOf(op::Shape("f32[1,3,9]"),
                    op::Gather(op::Shape("f32[9,9]"), op::Shape("s32[1,3]"))));
}

TEST_F(SpmdPartitioningTest, GatherReplicatedCorrectOutput) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[64,2,250112] parameter(0), sharding={
    devices=[16,1,2]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
                    23,24,25,26,27,28,29,30,31}
  %indices = s32[10,1] parameter(1), sharding={replicated}
  %input.copy = f32[64,2,250112] copy(%input), sharding={
    devices=[16,1,2]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
                    23,24,25,26,27,28,29,30,31}
  %indices.copy = s32[10,1] copy(%indices), sharding={replicated}
  %gather = f32[64,2,10] gather(f32[64,2,250112] %input,
    s32[10,1]{1,0} %indices.copy), offset_dims={0,1}, collapsed_slice_dims={2},
    start_index_map={2}, index_vector_dim=1, slice_sizes={64,2,1},
    sharding={devices=[16,1,1,2]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,
                                19,20,21,22,23,24,25,26,27,28,29,30,
                                31 last_tile_dim_replicate}
  ROOT %copy = (f32[64,2,10]) tuple(gather), sharding={{devices=[16,1,1,2]0,1,2,
    3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,
    30,31 last_tile_dim_replicate}}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/32));
  VLOG(1) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Shape("(f32[4,2,10])"));
}

TEST_F(SpmdPartitioningTest, GatherTrivialRestoreSharding) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %input = bf16[250112,4096] parameter(0), sharding={replicated}
  %cpy.input = bf16[250112,4096] copy(%input), sharding={
    devices=[32,1]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
                    23,24,25,26,27,28,29,30,31}
  %indices = s32[64,1,1] parameter(1), sharding={replicated}
  %cpy.indices = s32[64,1,1] copy(%indices), sharding={replicated}
  %gather = bf16[64,1,4096] gather(bf16[250112,4096] %cpy.input, s32[64,1,1] %cpy.indices),
    offset_dims={2}, collapsed_slice_dims={0}, start_index_map={0},
    index_vector_dim=2, slice_sizes={1,4096}, sharding={replicated}
  ROOT %copy = bf16[64,1,4096] copy(gather), sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/32));
  VLOG(1) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Shape("bf16[64,1,4096]"));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Copy(op::AllReduce(op::Select(
                  _, _, op::Gather(op::Shape("bf16[7816,4096]"), _)))));
}

TEST_F(SpmdPartitioningTest, SliceTo1) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[512] parameter(0), sharding={devices=[4]0,1,2,3}
  ROOT slice.134 = f32[1] slice(input), slice={[0:1]},
    sharding={devices=[4]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              AllOf(op::Slice(op::Parameter()), op::Shape("f32[1]")));
}

TEST_F(SpmdPartitioningTest, SliceTo1_8Shards) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[4,4] parameter(0), sharding={devices=[4,2]0,1,2,3,4,5,6,7}
  ROOT %slice = f32[1,4] slice(%input), slice={[0:1], [0:4]},
    sharding={devices=[4,2]0,1,2,3,4,5,6,7}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              AllOf(op::Copy(op::Parameter()), op::Shape("f32[1,2]")));
}

TEST_F(SpmdPartitioningTest, SliceTo1PartialReplicate) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[16] parameter(0),
    sharding={devices=[2,2]0,1,2,3 last_tile_dim_replicate}
  ROOT slice.134 = f32[1] slice(input), slice={[0:1]},
    sharding={devices=[2,2]0,1,2,3 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              AllOf(op::Slice(op::Parameter()), op::Shape("f32[1]")));
}

TEST_F(SpmdPartitioningTest, SliceTo2) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[512] parameter(0), sharding={devices=[4]0,1,2,3}
  ROOT slice.134 = f32[2] slice(input), slice={[0:2]},
    sharding={devices=[4]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();
  auto slice1 = AllOf(op::Slice(op::Parameter()), op::Shape("f32[2]"));
  auto halo =
      op::CollectivePermute(AllOf(op::Slice(slice1), op::Shape("f32[1]")));
  auto slice_self = AllOf(op::Slice(slice1), op::Shape("f32[1]"));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Copy(AllOf(op::DynamicSlice(op::Concatenate(halo, slice_self), _),
                     op::Shape("f32[1]"))));
}

TEST_F(SpmdPartitioningTest, SliceToMiddle2) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[512] parameter(0), sharding={devices=[8]0,1,2,3,4,5,6,7}
  ROOT %slice = f32[2] slice(input), slice={[300:302]},
    sharding={devices=[8]0,1,2,3,4,5,6,7}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  auto slice = AllOf(op::Slice(op::Parameter()), op::Shape("f32[2]"));
  auto halo1 = AllOf(op::CollectivePermute(slice), op::Shape("f32[2]"));
  auto halo2 =
      AllOf(op::CollectivePermute(op::Slice(slice)), op::Shape("f32[1]"));
  VLOG(1) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Copy(AllOf(op::DynamicSlice(op::Concatenate(halo1, halo2), _),
                             op::Shape("f32[1]"))));
}

TEST_F(SpmdPartitioningTest, SliceToMiddle2PartiallyReplicated) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[512] parameter(0),
    sharding={devices=[8,2]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 last_tile_dim_replicate}
  ROOT %slice = f32[2] slice(input), slice={[300:302]},
    sharding={devices=[8,2]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/16));
  auto slice = AllOf(op::Slice(op::Parameter()), op::Shape("f32[2]"));
  auto halo1 = AllOf(op::CollectivePermute(slice), op::Shape("f32[2]"));
  auto halo2 =
      AllOf(op::CollectivePermute(op::Slice(slice)), op::Shape("f32[1]"));
  VLOG(1) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Copy(AllOf(op::DynamicSlice(op::Concatenate(halo1, halo2), _),
                             op::Shape("f32[1]"))));
}

TEST_F(SpmdPartitioningTest, PartialDusReplicate) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[3,2] parameter(0),
    sharding={devices=[8,2]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
  ROOT %copy = f32[3,2] copy(input), sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/16));
  VLOG(1) << module->ToString();
  auto dus =
      AllOf(op::Shape("f32[3,2]"),
            op::DynamicUpdateSlice(op::Broadcast(),
                                   op::Select(_, op::Parameter(0), _), _, _));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Copy(AllOf(op::AllReduce(op::AllReduce(dus)))));
}

TEST_F(SpmdPartitioningTest, GatherPassthrough) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  p = f32[16,64,768,768]{3,2,1,0} parameter(0), sharding={replicated}
  c = f32[16,64,768,768]{3,2,1,0} copy(p), sharding={devices=[1,4,1,1]0,1,2,3}
  constant.1669 = s32[] constant(0)
  iota.1012 = s32[6]{0} iota(), iota_dimension=0, sharding={replicated}
  constant.1748 = s32[] constant(128), sharding={replicated}
  broadcast.2642 = s32[6]{0} broadcast(constant.1748), dimensions={}, sharding={replicated}
  multiply.92 = s32[6]{0} multiply(iota.1012, broadcast.2642), sharding={replicated}
  broadcast.2643 = s32[2,6]{1,0} broadcast(multiply.92), dimensions={1}, sharding={replicated}
  transpose.542 = s32[6,2]{0,1} transpose(broadcast.2643), dimensions={1,0}, sharding={replicated}
  pad.19 = s32[6,4]{1,0} pad(transpose.542, constant.1669), padding=0_0x2_0, sharding={replicated}
  ROOT gather.1 = f32[16,64,6,128,128]{4,3,2,1,0} gather(c, pad.19), offset_dims={0,1,3,4}, collapsed_slice_dims={}, start_index_map={0,1,2,3}, index_vector_dim=1, slice_sizes={16,64,128,128}, sharding={devices=[1,4,1,1,1]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Gather(), op::Shape("f32[16,16,6,128,128]")));
}

TEST_F(SpmdPartitioningTest, ComplexReshardFromPartialReplicate) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %p = f32[4,15,4,16] parameter(0)
  %p.copy = f32[4,15,4,16] copy(p),
    sharding={devices=[1,1,1,2,4]0,2,4,6,1,3,5,7 last_tile_dim_replicate}
  %a = f32[4,15,4,16] add(p.copy, p.copy),
    sharding={devices=[1,1,1,2,4]0,2,4,6,1,3,5,7 last_tile_dim_replicate}
  ROOT %c2 = f32[4,15,4,16] copy(a), sharding={devices=[1,8,1,1]0,1,2,3,4,5,6,7}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Copy(op::Reshape(op::Reshape(op::Transpose(op::AllToAll(_))))));
}

TEST_F(SpmdPartitioningTest, ComplexReshardToPartialReplicate) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %p = f32[4,15,4,16] parameter(0)
  %p.copy = f32[4,15,4,16] copy(p),
    sharding={devices=[1,4,2,1]0,1,2,3,4,5,6,7}
  %a = f32[4,15,4,16] add(p.copy, p.copy),
    sharding={devices=[1,4,2,1]0,1,2,3,4,5,6,7}
  ROOT %c2 = f32[4,15,4,16] copy(a), sharding={devices=[1,1,1,2,4]0,2,4,6,1,3,5,7 last_tile_dim_replicate}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Copy(op::Reshape(op::Transpose(op::AllToAll(_)))));
}

TEST_F(SpmdPartitioningTest, ComplexReshardMoveMergeDimensionRight) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %p = f32[4,15,4,15] parameter(0)
  %p.copy = f32[4,15,4,15] copy(p),
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7}
  %a = f32[4,15,4,15] add(p.copy, p.copy),
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7}
  ROOT %c2 = f32[4,15,4,15] copy(a), sharding={devices=[1,1,1,8]0,2,4,6,1,3,5,7}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Copy(op::Reshape(
                  op::Slice(op::Reshape(op::Transpose(op::AllToAll(_)))))));
}

TEST_F(SpmdPartitioningTest, ComplexReshardMoveMergeDimensionLeft) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %p = f32[2,15,1,2] parameter(0)
  %p.copy = f32[2,15,1,2] copy(p),
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7}
  %a = f32[2,15,1,2] add(p.copy, p.copy),
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7}
  ROOT %c2 = f32[2,15,1,2] copy(a), sharding={devices=[1,8,1,1]0,1,2,3,4,5,6,7}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Copy(op::Reshape(op::Reshape(op::Transpose(op::AllToAll(_))))));
}

TEST_F(SpmdPartitioningTest, ComplexReshardMoveMergeDimensionLeftReorder) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %p = f32[4,15,4,16] parameter(0)
  %p.copy = f32[4,15,4,16] copy(p),
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7}
  %a = f32[4,15,4,16] add(p.copy, p.copy),
    sharding={devices=[1,4,1,2]0,1,2,3,4,5,6,7}
  ROOT %c2 = f32[4,15,4,16] copy(a), sharding={devices=[1,8,1,1]0,2,4,6,1,3,5,7}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  VLOG(1) << module->ToString();

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Copy(op::Reshape(op::CollectivePermute(
                  op::Reshape(op::Transpose(op::AllToAll(_)))))));
}

TEST_F(SpmdPartitioningTest, PaddedConvReshard) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %p = bf16[16,256,256,384]{3,2,1,0} parameter(0)
  %p2 = bf16[3,3,384,384]{3,2,1,0} parameter(1)
  %p.copy = bf16[16,256,256,384]{3,2,1,0} copy(%p), sharding={devices=[2,1,4,1]0,1,2,3,4,5,6,7}
  %p2.copy = bf16[3,3,384,384]{3,2,1,0} copy(%p2), sharding={replicated}
  ROOT %convolution.10115 = bf16[16,256,256,384]{3,2,1,0} convolution(%p.copy, %p2.copy), window={size=3x3 pad=128_128x128_128 rhs_dilate=128x128}, dim_labels=b01f_01io->b01f, sharding={devices=[2,1,4,1]0,1,2,3,4,5,6,7}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Convolution(
                  op::DynamicSlice(op::Pad(_, op::Constant()), _, _, _, _), _));
}

TEST_F(SpmdPartitioningTest, KeepPartitionedNonSlicedDimension) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %p = bf16[16,128,128,384]{3,2,1,0} parameter(0), sharding={replicated}
  %constant.1165 = s32[] constant(0), sharding={replicated}
  constant.1151 = s32[] constant(192), sharding={replicated}
  broadcast.1152 = s32[2]{0} broadcast(constant.1151), dimensions={}, sharding={replicated}
  slice.1576 = s32[1]{0} slice(broadcast.1152), slice={[0:1]}, sharding={replicated}
  reshape.1888 = s32[] reshape(slice.1576), sharding={replicated}
  slice.1546 = s32[1]{0} slice(broadcast.1152), slice={[1:2]}, sharding={replicated}
  reshape.1890 = s32[] reshape(slice.1546), sharding={replicated}
  constant.861 = bf16[] constant(0), sharding={replicated}
  broadcast.862 = bf16[16,512,512,384]{3,2,1,0} broadcast(constant.861), dimensions={}, sharding={devices=[2,2,1,1]0,1,2,3}
  %c = bf16[16,128,128,384]{3,2,1,0} copy(p), sharding={devices=[2,2,1,1]0,1,2,3}
  add.228 = bf16[16,128,128,384]{3,2,1,0} add(c, c), sharding={devices=[2,2,1,1]0,1,2,3}
  ROOT dynamic-update-slice.111 = bf16[16,512,512,384]{3,2,1,0} dynamic-update-slice(broadcast.862, add.228, constant.1165, reshape.1888, reshape.1890, /*index=5*/constant.1165), sharding={devices=[2,2,1,1]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));

  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::DynamicSlice(AllOf(op::DynamicUpdateSlice(),
                                     op::Shape("bf16[8,512,512,384]")),
                               _, _, _, _));
}

TEST_F(SpmdPartitioningTest,
       KeepPartitionedNonSlicedDimensionWithConstantIndices) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  p1 = bf16[16,192,192,384]{3,2,1,0} parameter(0), sharding={replicated}
  p2 = bf16[16,128,128,384]{3,2,1,0} parameter(1), sharding={replicated}
  c1 = bf16[16,192,192,384]{3,2,1,0} copy(p1), sharding={devices=[2,2,2,1]0,1,2,3,4,5,6,7}
  c2 = bf16[16,128,128,384]{3,2,1,0} copy(p2), sharding={devices=[2,2,2,1]0,1,2,3,4,5,6,7}
  constant.1163 = bf16[] constant(0), sharding={replicated}
  constant.1165 = s32[] constant(0), sharding={replicated}
  pad.179 = bf16[16,224,224,384]{3,2,1,0} pad(c1, constant.1163), padding=0_0x16_16x16_16x0_0, sharding={devices=[2,2,2,1]0,1,2,3,4,5,6,7}
  add.439 = bf16[16,128,128,384]{3,2,1,0} add(c2, c2), sharding={devices=[2,2,2,1]0,1,2,3,4,5,6,7}
  constant.1070 = s32[] constant(48), sharding={replicated}
  dynamic-update-slice.128 = bf16[16,224,224,384]{3,2,1,0} dynamic-update-slice(pad.179, add.439, constant.1165, constant.1070, constant.1070, /*index=5*/constant.1165), sharding={devices=[2,2,2,1]0,1,2,3,4,5,6,7}
  ROOT c = bf16[16,224,224,384]{3,2,1,0} copy(dynamic-update-slice.128), sharding={devices=[2,2,2,1]0,1,2,3,4,5,6,7}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));

  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Copy(op::DynamicSlice(
          AllOf(op::DynamicUpdateSlice(), op::Shape("bf16[8,224, 224,384]")), _,
          _, _, _)));
}

TEST_F(SpmdPartitioningTest, CustomCallManualSharding) {
  const char* const hlo_string = R"(
HloModule pjit_xmap_dummy.5

ENTRY %main.21 (Arg_0.1: f32[4,4,8], Arg_1.2: f32[4,8]) -> (f32[4,4,8], f32[4]) {
  %Arg_0.1 = f32[4,4,8]{2,1,0} parameter(0), sharding={devices=[4,1,1]0,1,2,3}
  %copy.3 = f32[4,4,8]{2,1,0} copy(f32[4,4,8]{2,1,0} %Arg_0.1), sharding={devices=[4,1,1]0,1,2,3}
  %custom-call.4 = f32[1,4,8]{2,1,0} custom-call(f32[4,4,8]{2,1,0} %copy.3), custom_call_target="SPMDFullToShardShape", sharding={manual}
  %reshape.7 = f32[4,8]{1,0} reshape(f32[1,4,8]{2,1,0} %custom-call.4), sharding={manual}
  %Arg_1.2 = f32[4,8]{1,0} parameter(1), sharding={replicated}
  %copy.2 = f32[4,8]{1,0} copy(f32[4,8]{1,0} %Arg_1.2), sharding={replicated}
  %custom-call.6 = f32[4,8]{1,0} custom-call(f32[4,8]{1,0} %copy.2), custom_call_target="SPMDFullToShardShape", sharding={manual}
  %custom-call.8 = (f32[4,8]{1,0}, f32[1]{0}) custom-call(f32[4,8]{1,0} %reshape.7, f32[4,8]{1,0} %custom-call.6), custom_call_target="dummy", operand_layout_constraints={f32[4,8]{1,0}, f32[4,8]{1,0}}, api_version=API_VERSION_STATUS_RETURNING, sharding={{manual}, {manual}}
  %get-tuple-element.9 = f32[4,8]{1,0} get-tuple-element((f32[4,8]{1,0}, f32[1]{0}) %custom-call.8), index=0, sharding={manual}
  %reshape.11 = f32[1,4,8]{2,1,0} reshape(f32[4,8]{1,0} %get-tuple-element.9), sharding={manual}
  %copy.1 = f32[1,4,8]{2,1,0} copy(f32[1,4,8]{2,1,0} %reshape.11), sharding={manual}
  %custom-call.14 = f32[4,4,8]{2,1,0} custom-call(f32[1,4,8]{2,1,0} %copy.1), custom_call_target="SPMDShardToFullShape", sharding={devices=[4,1,1]0,1,2,3}
  %reshape.18 = f32[4,4,8]{2,1,0} reshape(f32[4,4,8]{2,1,0} %custom-call.14), sharding={devices=[4,1,1]0,1,2,3}
  %get-tuple-element.10 = f32[1]{0} get-tuple-element((f32[4,8]{1,0}, f32[1]{0}) %custom-call.8), index=1, sharding={manual}
  %reshape.12 = f32[1,1]{1,0} reshape(f32[1]{0} %get-tuple-element.10), sharding={manual}
  %copy = f32[1,1]{1,0} copy(f32[1,1]{1,0} %reshape.12), sharding={manual}
  %custom-call.16 = f32[4,1]{1,0} custom-call(f32[1,1]{1,0} %copy), custom_call_target="SPMDShardToFullShape", sharding={devices=[4,1]0,1,2,3}
  %reshape.17 = f32[4]{0} reshape(f32[4,1]{1,0} %custom-call.16), sharding={devices=[4]0,1,2,3}
  %reshape.19 = f32[4]{0} reshape(f32[4]{0} %reshape.17), sharding={devices=[4]0,1,2,3}
  ROOT %tuple.20 = (f32[4,4,8]{2,1,0}, f32[4]{0}) tuple(f32[4,4,8]{2,1,0} %reshape.18, f32[4]{0} %reshape.19), sharding={{replicated}, {replicated}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));

  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::AllReduce(op::DynamicUpdateSlice(
                            _, op::Shape("f32[1,4,8]"), _, _, _)),
                        op::AllReduce(op::DynamicUpdateSlice(
                            _, op::Shape("f32[1]"), _))));
}

TEST_F(SpmdPartitioningTest, UnevenPadAllToAllReshard) {
  const char* const hlo_string = R"(
HloModule pjit_xmap_dummy.5

ENTRY %main.21 {
  %Arg_0.1 = f32[19,19]{1,0} parameter(0), sharding={devices=[4,2]0,1,2,3,4,5,6,7}
  add.3171 = f32[19,19]{1,0} add(Arg_0.1, Arg_0.1), sharding={devices=[4,2]0,1,2,3,4,5,6,7}
  transpose.3172 = f32[19,19]{0,1} transpose(add.3171), dimensions={1,0}, sharding={devices=[2,4]0,2,4,6,1,3,5,7}
  ROOT add.3173 = f32[19,19]{1,0} add(add.3171, transpose.3172), sharding={devices=[4,2]0,1,2,3,4,5,6,7}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));

  XLA_VLOG_LINES(1, module->ToString());
  int64_t collective_permute_count = 0;
  for (auto* i : module->entry_computation()->instructions()) {
    if (i->opcode() == HloOpcode::kCollectivePermute) {
      ++collective_permute_count;
    }
  }
  // Expected that the number of collective permutes is 1. Padding is the same
  // between the two dimension (1,1).
  EXPECT_EQ(collective_permute_count, 1);
}

TEST_F(SpmdPartitioningTest, UnevenPadAllToAllReshard2) {
  const char* const hlo_string = R"(
HloModule pjit_xmap_dummy.5

ENTRY %main.21 {
  %Arg_0.1 = f32[5,5]{1,0} parameter(0), sharding={devices=[4,2]0,1,2,3,4,5,6,7}
  add.3171 = f32[5,5]{1,0} add(Arg_0.1, Arg_0.1), sharding={devices=[4,2]0,1,2,3,4,5,6,7}
  transpose.3172 = f32[5,5]{0,1} transpose(add.3171), dimensions={1,0}, sharding={devices=[2,4]0,2,4,6,1,3,5,7}
  ROOT add.3173 = f32[5,5]{1,0} add(add.3171, transpose.3172), sharding={devices=[4,2]0,1,2,3,4,5,6,7}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));

  XLA_VLOG_LINES(1, module->ToString());
  int64_t collective_permute_count = 0;
  for (auto* i : module->entry_computation()->instructions()) {
    if (i->opcode() == HloOpcode::kCollectivePermute) {
      ++collective_permute_count;
    }
  }
  // Expected that the number of collective permutes is 3 for the correct
  // reshard.
  EXPECT_EQ(collective_permute_count, 3);
}

TEST_F(SpmdPartitioningTest, CustomCallShardingRegistration) {
  class BatchableCustomCallPartitioner : public CustomCallPartitioner {
   public:
    HloSharding PropagateUserSharding(
        const HloInstruction* instruction, const HloInstruction* user,
        const HloSharding& sharding) const override {
      return sharding;
    }
    std::optional<HloSharding> InferShardingFromOperands(
        const HloInstruction* instruction) const override {
      if (instruction->operand(0)->has_sharding()) {
        return instruction->operand(0)->sharding();
      }
      return std::nullopt;
    }
    bool IsCustomCallShardable(
        const HloInstruction* instruction) const override {
      return true;
    }
    xla::Status Partition(spmd::SpmdPartitioningVisitor* partitioner,
                          HloInstruction* hlo) const override {
      if (hlo->shape().rank() <= 2) {
        return partitioner->DefaultAction(hlo);
      }
      const int first_non_batch_dim = hlo->shape().rank() - 2;
      HloInstruction* operand = hlo->mutable_operand(0);
      HloSharding target_sharding =
          hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
              hlo->sharding(), {first_non_batch_dim, first_non_batch_dim + 1});
      spmd::PartitionedHlo operand_partitioned =
          partitioner->GetPartitionedHlo(operand).Reshard(target_sharding);
      HloCustomCallInstruction* custom_call =
          Cast<HloCustomCallInstruction>(hlo);
      Shape partitioned_shape_with_layout_constraint =
          operand_partitioned.hlo()->shape();
      (*partitioned_shape_with_layout_constraint.mutable_layout()) =
          custom_call->operand_shapes_with_layout()[0].layout();
      HloInstruction* partitioned_hlo = partitioner->builder()->AddInstruction(
          HloInstruction::CreateCustomCall(
              operand_partitioned.hlo()->shape(), {operand_partitioned.hlo()},
              "BatchableCustomCall",
              {partitioned_shape_with_layout_constraint}));
      partitioned_hlo->set_sharding(target_sharding);
      spmd::PartitionedHlo result_partitioned =
          spmd::PartitionedHlo(partitioned_hlo,
                               operand_partitioned.base_shape(),
                               operand_partitioned.state())
              .Reshard(hlo->sharding());
      partitioner->SetPartitionedHlo(hlo, result_partitioned);
      return OkStatus();
    }
  };
  RegisterCustomCallPartitioner(
      "BatchableCustomCall",
      std::make_unique<BatchableCustomCallPartitioner>());
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %p = f32[102,128,128]{2,1,0:T(8,128)} parameter(0), sharding={devices=[2,1,2]0,1,2,3}
  ROOT custom-call = f32[102,128,128]{2,1,0:T(8,128)} custom-call(p), custom_call_target="BatchableCustomCall", operand_layout_constraints={f32[102,128,128]{2,1,0}}, sharding={devices=[2,1,2]0,1,2,3}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/4));
  VLOG(1) << module->ToString();

  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::DynamicSlice(
                        AllOf(op::CustomCall(_), op::Shape("f32[51,128,128]")),
                        _, _, _));
}

TEST_F(SpmdPartitioningTest, ManualGetTupleElement) {
  const char* const hlo_string = R"(
HloModule pjit

orclone {
  lhs.1 = u32[] parameter(0)
  rhs.1 = u32[] parameter(2)
  or.2 = u32[] or(lhs.1, rhs.1)
  lhs.0 = u32[] parameter(1)
  rhs.0 = u32[] parameter(3)
  or.3 = u32[] or(lhs.0, rhs.0)
  ROOT tuple.4 = (u32[], u32[]) tuple(or.2, or.3)
}

ENTRY %main.21 {
  select.104 = u32[2,2]{1,0} parameter(0), sharding={manual}
  shift-left.5 = u32[2,2]{1,0} parameter(1), sharding={manual}
  constant.4183 = u32[] constant(0), sharding={manual}
  reduce.1 = (u32[2]{0}, u32[2]{0}) reduce(shift-left.5, select.104, constant.4183, constant.4183), dimensions={1}, sharding={{manual},{manual}}, to_apply=orclone
  ROOT get-tuple-element.13 = u32[2]{0} get-tuple-element(reduce.1), index=0, sharding={manual}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));

  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::GetTupleElement(op::Reduce(_, _, _, _)));
}

TEST_F(SpmdPartitioningTest, CombiningScatterPartitiong) {
  const char* const hlo_string = R"(
HloModule pjit

region_110.8267 {
  Arg_0.8268 = bf16[] parameter(0)
  Arg_1.8269 = bf16[] parameter(1)
  ROOT add.8270 = bf16[] add(Arg_0.8268, Arg_1.8269)
}

ENTRY %main.21 {
  broadcast.8659 = bf16[2,8,12288,192,64]{4,3,2,1,0} parameter(0), sharding={devices=[2,1,2,4,1]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
  reshape.9796 = bf16[2,1,12288,192,64]{4,3,2,1,0} parameter(1), sharding={devices=[2,1,2,4,1]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
  iota.50 = s32[2,1]{1,0} iota(), iota_dimension=0, sharding={devices=[2,1,8]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 last_tile_dim_replicate}
  constant.1585 = s32[] constant(0), sharding={replicated}
  broadcast.3764 = s32[2,1]{1,0} broadcast(constant.1585), dimensions={}, sharding={devices=[2,1,8]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 last_tile_dim_replicate}
  reshape_idx = s32[2,1]{1,0} parameter(2), sharding={devices=[2,1,8]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 last_tile_dim_replicate}
  concatenate.8907 = s32[2,5]{1,0} concatenate(iota.50, reshape_idx, broadcast.3764, broadcast.3764, broadcast.3764), dimensions={1}, sharding={devices=[2,1,8]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 last_tile_dim_replicate}
  scatter.9797 = bf16[2,8,12288,192,64]{4,3,2,1,0} scatter(broadcast.8659, concatenate.8907, reshape.9796), update_window_dims={1,2,3,4}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0,1,2,3,4}, index_vector_dim=1, indices_are_sorted=true, unique_indices=true, to_apply=region_110.8267, sharding={devices=[2,1,2,4,1]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
  ROOT c = bf16[2,8,12288,192,64]{4,3,2,1,0} copy(scatter.9797), sharding={devices=[2,1,2,4,1]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/16));

  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Copy(AllOf(op::Shape("bf16[1,8,6144,48,64]"), op::Scatter(_, _, _))));
  // Check that there is no communication added.
  EXPECT_EQ(FindInstruction(module.get(), HloOpcode::kAllReduce), nullptr);
}

TEST_F(SpmdPartitioningTest, MatchOutputAlignmentNonContractingDot) {
  const char* const hlo_string = R"(
HloModule pjit

ENTRY %main.21 {
  multiply.3535 = f32[4,4]{1,0} parameter(0), sharding={devices=[2,4,2]0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15 last_tile_dim_replicate}
  reshape.4221 = f32[4,4]{1,0} parameter(1), sharding={devices=[4,1,4]0,8,4,12,1,9,5,13,2,10,6,14,3,11,7,15 last_tile_dim_replicate}
  dot.11597 = f32[4,4]{1,0} dot(multiply.3535, reshape.4221), lhs_contracting_dims={1}, rhs_contracting_dims={0}, sharding={devices=[2,1,8]0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15 last_tile_dim_replicate}
  ROOT copy.1 = f32[4,4]{1,0} copy(dot.11597), sharding={devices=[2,1,8]0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15 last_tile_dim_replicate}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/16));
  EXPECT_EQ(FindInstruction(module.get(), HloOpcode::kCollectivePermute),
            nullptr);
}

TEST_F(SpmdPartitioningTest, ComplexReshardPartialMerging) {
  const char* const hlo_string = R"(
HloModule pjit

ENTRY %main.21 {
  multiply.3535 = f32[256,256,256]{2,1,0} parameter(0), sharding={devices=[2,1,2,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT copy.1 = f32[256,256,256]{2,1,0} copy(multiply.3535), sharding={devices=[1,2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));

  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_NE(FindInstruction(module.get(), HloOpcode::kAllToAll), nullptr);
}

TEST_F(SpmdPartitioningTest, PartialReshardingInfiniteLoops) {
  const char* const hlo_string = R"(
HloModule pjit

ENTRY %main.21 {
  multiply.3535 = f32[256,256,256]{2,1,0} parameter(0), sharding={devices=[4,1,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT copy.1 = f32[256,256,256]{2,1,0} copy(multiply.3535), sharding={devices=[2,2,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));

  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(SpmdPartitioningTest, GatherCostModelForUnmatchedSharding) {
  const char* const hlo_string = R"(
HloModule pjit

region_10.581.clone {
  Arg_0.53 = bf16[] parameter(0)
  Arg_1.53 = bf16[] parameter(1)
  ROOT add.1294 = bf16[] add(Arg_0.53, Arg_1.53)
}

ENTRY %main.21 {
  p0 = bf16[8192,128]{1,0} parameter(0), sharding={devices=[2,4,2]0,8,2,10,4,12,6,14,1,9,3,11,5,13,7,15 last_tile_dim_replicate}
  p1 = s32[16384,1]{1,0} parameter(1), sharding={devices=[8,1,2]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 last_tile_dim_replicate}
  gather.0 = bf16[16384,128]{1,0} gather(p0, p1), offset_dims={1}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1,128}, sharding={devices=[8,2]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
  constant.2467 = bf16[] constant(0)
  reduce.1749 = bf16[16384]{0} reduce(gather.0, constant.2467), dimensions={1}, to_apply=region_10.581.clone, sharding={devices=[8,2]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 last_tile_dim_replicate}
  ROOT copy.1 = bf16[16384]{0} copy(reduce.1749), sharding={devices=[8,2]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 last_tile_dim_replicate}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/16));

  XLA_VLOG_LINES(1, module->ToString());
  auto* gather = FindInstruction(module.get(), HloOpcode::kGather);
  EXPECT_NE(gather, nullptr);
  EXPECT_THAT(gather, op::Shape("bf16[2048,128]"));
}

TEST_F(SpmdPartitioningTest, ScatterCostModelForUnmatchedSharding) {
  const char* const hlo_string = R"(
HloModule pjit

region_335.4575 {
  Arg_0.4576 = bf16[] parameter(0)
  Arg_1.4577 = bf16[] parameter(1)
  ROOT add.4578 = bf16[] add(Arg_0.4576, Arg_1.4577)
}

ENTRY %main.21 {
  p0 = bf16[8192,128]{1,0} parameter(0), sharding={devices=[2,4,2]0,8,2,10,4,12,6,14,1,9,3,11,5,13,7,15 last_tile_dim_replicate}
  p1 = s32[32768,1]{1,0} parameter(1), sharding={devices=[8,1,2]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 last_tile_dim_replicate}
  p2 = bf16[32768,128]{1,0} parameter(2), sharding={devices=[8,2]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
  scatter.0 = bf16[8192,128]{1,0} scatter(p0, p1, p2), update_window_dims={1}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=region_335.4575, sharding={devices=[2,4,2]0,8,2,10,4,12,6,14,1,9,3,11,5,13,7,15 last_tile_dim_replicate}
  ROOT convert.427 = f32[8192,128]{1,0} convert(scatter.0), sharding={devices=[2,4,2]0,8,2,10,4,12,6,14,1,9,3,11,5,13,7,15 last_tile_dim_replicate}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/16));

  XLA_VLOG_LINES(1, module->ToString());
  auto* scatter = FindInstruction(module.get(), HloOpcode::kScatter);
  EXPECT_NE(scatter, nullptr);
  auto* updates = scatter->operand(2);
  EXPECT_THAT(updates, op::Shape("bf16[4096,128]"));
}

TEST_F(SpmdPartitioningTest, ComplexReshardUnmergeToRight) {
  const char* const hlo_string = R"(
HloModule Test

ENTRY main.4 {
  Arg_0.1 = f32[8,32]{1,0} parameter(0), sharding={devices=[8,1]0,2,4,6,1,3,5,7}
  tuple.2 = (f32[8,32]{1,0}) tuple(Arg_0.1), sharding={{devices=[2,4]0,2,4,6,1,3,5,7}}
  ROOT get-tuple-element.3 = f32[8,32]{1,0} get-tuple-element(tuple.2), index=0, sharding={devices=[2,4]0,2,4,6,1,3,5,7}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));

  XLA_VLOG_LINES(1, module->ToString());
  auto* allreduce = FindInstruction(module.get(), HloOpcode::kAllReduce);
  EXPECT_EQ(allreduce, nullptr);
  auto* alltoall = FindInstruction(module.get(), HloOpcode::kAllToAll);
  EXPECT_NE(alltoall, nullptr);
}

TEST_F(SpmdPartitioningTest, ComplexReshardUnmergeToLeft) {
  const char* const hlo_string = R"(
HloModule Test

ENTRY main.4 {
  Arg_0.1 = f32[8,32]{1,0} parameter(0), sharding={devices=[1,8]0,2,4,6,1,3,5,7}
  tuple.2 = (f32[8,32]{1,0}) tuple(Arg_0.1), sharding={{devices=[2,4]0,2,4,6,1,3,5,7}}
  ROOT get-tuple-element.3 = f32[8,32]{1,0} get-tuple-element(tuple.2), index=0, sharding={devices=[2,4]0,2,4,6,1,3,5,7}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));

  XLA_VLOG_LINES(1, module->ToString());
  auto* allreduce = FindInstruction(module.get(), HloOpcode::kAllReduce);
  EXPECT_EQ(allreduce, nullptr);
  auto* alltoall = FindInstruction(module.get(), HloOpcode::kAllToAll);
  EXPECT_NE(alltoall, nullptr);
}

TEST_F(SpmdPartitioningTest, NoComplexReshardUnmergeToLeft) {
  const char* const hlo_string = R"(
HloModule Test

ENTRY main.4 {
  Arg_0.1 = f32[8,33]{1,0} parameter(0), sharding={devices=[1,8]0,2,4,6,1,3,5,7}
  tuple.2 = (f32[8,33]{1,0}) tuple(Arg_0.1), sharding={{devices=[2,4]0,2,4,6,1,3,5,7}}
  ROOT get-tuple-element.3 = f32[8,33]{1,0} get-tuple-element(tuple.2), index=0, sharding={devices=[2,4]0,2,4,6,1,3,5,7}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));

  XLA_VLOG_LINES(1, module->ToString());
  auto* allreduce = FindInstruction(module.get(), HloOpcode::kAllReduce);
  EXPECT_NE(allreduce, nullptr);
  auto* alltoall = FindInstruction(module.get(), HloOpcode::kAllToAll);
  EXPECT_EQ(alltoall, nullptr);
}

TEST_F(SpmdPartitioningTest, ReshardCrash) {
  const char* const hlo_string = R"(
HloModule Test

ENTRY main.6 {
  Arg_0.1 = f32[8,32,4]{2,1,0} parameter(0), sharding={devices=[4,2,1]0,2,1,3,4,6,5,7}
  ROOT copy = copy(Arg_0.1), sharding={devices=[2,2,2]0,1,2,3,4,5,6,7}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/8));

  XLA_VLOG_LINES(1, module->ToString());
  auto* alltoall = FindInstruction(module.get(), HloOpcode::kAllToAll);
  EXPECT_NE(alltoall, nullptr);
}

}  // namespace
}  // namespace spmd
}  // namespace xla
