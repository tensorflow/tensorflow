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

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace spmd {
namespace {

using ::testing::_;
using ::testing::AllOf;
namespace op = xla::testing::opcode_matchers;

class SpmdPartitioningTest : public HloTestBase {
 public:
  StatusOr<std::unique_ptr<HloModule>> PartitionComputation(
      const char* hlo_module, int64 num_devices,
      bool conv_halo_exchange_always_on_lhs = true) {
    // Some tests (BackpropFilter convs) set this flag false to test two
    // different paths of the implementation.
    SpmdPartitionerOptions options;
    options.conv_halo_exchange_always_on_lhs = conv_halo_exchange_always_on_lhs;
    options.allow_module_signature_change = true;
    auto collective_ops_creator =
        GetDefaultCollectiveOpsCreator(num_devices, /*num_replicas=*/1);
    // Do not use all-gather for pattern-matching purpose, as the partitioner
    // might create reshape/transposes around it.
    collective_ops_creator.create_cross_partition_all_gather = nullptr;

    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(
                                         hlo_module, GetModuleConfigForTest()));
    HloPassPipeline pass("spmd-partitioning");
    pass.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                              /*allow_mixed_precision=*/false);
    pass.AddPass<SpmdPartitioner>(num_devices, /*num_replicas=*/1, options,
                                  collective_ops_creator);
    pass.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                              /*allow_mixed_precision=*/false);
    TF_RETURN_IF_ERROR(pass.Run(module.get()).status());
    return StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }
};

TEST_F(SpmdPartitioningTest, InvalidSharding) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  token0 = token[] after-all(), sharding={maximal device=0}
  infeed = (f32[8,2]{1,0}, token[]) infeed(token0),
    sharding={{devices=[2,1]0,1}, {maximal device=0}}
  ROOT infeed.data = f32[8,2]{1,0} get-tuple-element(infeed), index=0,
    sharding={maximal device=0}
})";
  auto module_status = PartitionComputation(hlo_string, /*num_devices=*/4);
  EXPECT_FALSE(module_status.status().ok());
  EXPECT_THAT(module_status.status().ToString(),
              ::testing::HasSubstr(
                  "only supports tile sharding that includes all partitions"));
}

TEST_F(SpmdPartitioningTest, SingleDeviceToReplicated) {
  const char* const hlo_string = R"(
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

TEST_F(SpmdPartitioningTest, SingleDeviceToSingleDevice) {
  const char* const hlo_string = R"(
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
  const char* const hlo_string = R"(
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
              op::Reshape(op::DynamicSlice(op::Constant(), op::PartitionId(),
                                           op::Constant())),
              op::Constant())),
          op::Shape("s32[1,3]")));
}

TEST_F(SpmdPartitioningTest, TiledToReplicated) {
  const char* const hlo_string = R"(
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
              op::Reshape(op::DynamicSlice(op::Constant(), op::PartitionId(),
                                           op::Constant())),
              op::Constant()),
          op::Shape("s32[2,3]")))));
}

TEST_F(SpmdPartitioningTest, TiledToSingleDevice) {
  const char* const hlo_string = R"(
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
              op::Reshape(op::DynamicSlice(op::Constant(), op::PartitionId(),
                                           op::Constant())),
              op::Constant()),
          op::Shape("s32[2,3]"))))));
}

TEST_F(SpmdPartitioningTest, TiledToTiledEven) {
  const char* const hlo_string = R"(
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
  const char* const hlo_string = R"(
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
  const char* const hlo_string = R"(
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
  const char* const hlo_string = R"(
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

  auto offset = op::Reshape(
      op::DynamicSlice(op::Constant(), op::PartitionId(), op::Constant()));

  EXPECT_THAT(root->operand(0),
              op::DynamicSlice(op::GetTupleElement(op::Parameter()), offset,
                               op::Constant()));
  EXPECT_THAT(root->operand(1),
              op::DynamicSlice(op::GetTupleElement(op::Parameter()), offset,
                               op::Constant()));
}

TEST_F(SpmdPartitioningTest, TiledInfeed) {
  const char* const hlo_string = R"(
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
      root, op::Copy(op::AllReduce(op::DynamicUpdateSlice(
                op::Broadcast(),
                op::GetTupleElement(
                    AllOf(op::Infeed(), op::Shape("(f32[4,2]{1,0}, token[])"))),
                op::Reshape(op::DynamicSlice(op::Constant(), op::PartitionId(),
                                             op::Constant())),
                op::Constant()))));
}

TEST_F(SpmdPartitioningTest, UnevenTiledInfeed) {
  const char* const hlo_string = R"(
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
  const char* const hlo_string = R"(
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

TEST_F(SpmdPartitioningTest, TiledToReplicatedReduce) {
  const char* const hlo_string = R"(
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
  const char* const hlo_string = R"(
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
  const char* const hlo_string = R"(
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
  const char* const hlo_string = R"(
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
  const char* const hlo_string = R"(
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
  const char* const hlo_string = R"(
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

TEST_F(SpmdPartitioningTest, BroadcastPropagateTiledSharding) {
  const char* const hlo_string = R"(
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
  const char* const hlo_string = R"(
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

TEST_F(SpmdPartitioningTest, ReduceWindowReplicatedInput) {
  const char* const hlo_string = R"(
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
  const char* const hlo_string = R"(
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
  const char* const hlo_string = R"(
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
            op::Slice(AllOf(op::Shape("f32[5,2]"),
                            op::Concatenate(halo0, halo1, op::Parameter(0)))));
  auto masked =
      op::Select(op::Compare(op::Add(op::Iota(), op::Broadcast(op::Multiply())),
                             op::Broadcast(op::Constant())),
                 pre_mask, op::Broadcast(op::Constant()));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Shape("f32[1,2]{1,0}"),
                          op::ReduceWindow(masked, op::Constant())));
}

TEST_F(SpmdPartitioningTest, ReduceWindowTiledOneSideUnequalHalo) {
  const char* const hlo_string = R"(
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
  const char* const hlo_string = R"(
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
  const char* const hlo_string = R"(
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
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,112,224,3]"));
  auto rhs = AllOf(op::Copy(op::Parameter()), op::Shape("f32[7,7,3,64]"));

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
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Reshape(), op::Constant(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[64,224,224,3]"));
  auto all_to_all =
      AllOf(op::AllToAll(op::Reshape(lhs)), op::Shape("f32[64,2,112,224,3]"));
  auto reshard_lhs = AllOf(op::Reshape(op::Transpose(all_to_all)),
                           op::Shape("f32[128,112,224,3]"));

  auto rhs = AllOf(op::Copy(op::Parameter()), op::Shape("f32[7,7,3,64]"));

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
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Reshape(), op::Constant(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[112,224,3,128]"));
  auto rhs = AllOf(op::Copy(op::Parameter()), op::Shape("f32[7,7,3,64]"));

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
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  // There is no halo exchange, and because the last element in the shard is not
  // needed (stride == 4), the LHS will be just a slice.
  auto sliced_lhs =
      AllOf(op::Slice(op::Copy(op::DynamicSlice(
                op::Pad(op::Parameter(), op::Constant()), op::Constant(),
                op::Reshape(), op::Constant(), op::Constant()))),
            op::Shape("f32[128,3,7,512]"));
  auto rhs = AllOf(op::Copy(op::Parameter()), op::Shape("f32[3,3,512,512]"));
  EXPECT_THAT(root, AllOf(op::Convolution(sliced_lhs, rhs),
                          op::Shape("f32[128,2,4,512]")));
  EXPECT_EQ(root->window().dimensions(0).padding_low(), 1);
  EXPECT_EQ(root->window().dimensions(0).padding_high(), 1);
}

// (stride * per_shard_window_count) % dilation != 0 but stride == 1
TEST_F(SpmdPartitioningTest,
       ConvolutionBaseDilationStride1LhsTiledRhsReplicated) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(op::Copy(op::DynamicSlice(
                       op::Pad(op::Parameter(), op::Constant()), op::Constant(),
                       op::Reshape(), op::Constant(), op::Constant())),
                   op::Shape("f32[128,4,7,512]"));
  auto rhs = AllOf(op::Copy(op::Parameter()), op::Shape("f32[3,3,512,512]"));

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
  const char* const hlo_string = R"(
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
  auto root = module->entry_computation()->root_instruction();
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
  const char* const hlo_string = R"(
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
  auto root = module->entry_computation()->root_instruction();
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
  const char* const hlo_string = R"(
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
  auto root = module->entry_computation()->root_instruction();

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
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,28,56,64]"));
  auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,28,56,256]"));

  EXPECT_THAT(root, AllOf(op::AllReduce(op::Convolution(lhs, rhs)),
                          op::Shape("f32[1,1,64,256]")));
}

TEST_F(SpmdPartitioningTest, ConvolutionLhsTiledRhsTiledWindowReversal) {
  const char* const hlo_string = R"(
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

  auto lhs_masked =
      AllOf(op::Shape("f32[3,128,64]"), op::Select(_, op::Parameter(0), _));
  auto rhs_left_padded = op::Slice(op::Concatenate(
      op::CollectivePermute(op::Slice(op::Parameter(1))), op::Parameter(1)));
  auto rhs_masked =
      AllOf(op::Shape("f32[3,128,256]"), op::Select(_, rhs_left_padded, _));

  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              AllOf(op::AllReduce(op::Convolution(lhs_masked, rhs_masked)),
                    op::Shape("f32[1,64,256]")));
}

TEST_F(SpmdPartitioningTest, DotLhsTiledRhsTiledWithReshard) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,28,56,64]"));
  auto rhs = AllOf(
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
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,28,56,512]"));
  auto rhs = AllOf(
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

TEST_F(SpmdPartitioningTest, ConvolutionLhsTiledRhsTiledWithPadding) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[32,14,28,128]"));
  auto rhs = AllOf(
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
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,112,224,3]"));
  auto rhs = AllOf(
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
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,28,56,256]"));
  auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,14,28,512]"));

  EXPECT_THAT(root, AllOf(op::AllReduce(op::Convolution(lhs, rhs)),
                          op::Shape("f32[1,1,256,512]")));
}

TEST_F(SpmdPartitioningTest, ConvolutionLhsTiledRhsTiledWindowDilateUneven) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,7,14,512]"));
  auto rhs = AllOf(
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
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[32,14,28,128]"));
  auto rhs = AllOf(
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
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,112,224,3]"));
  auto rhs = AllOf(
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
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,28,56,256]"));
  auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,14,28,512]"));

  EXPECT_THAT(root, AllOf(op::AllReduce(op::Convolution(op::Slice(lhs), rhs)),
                          op::Shape("f32[1,1,256,512]")));
}

TEST_F(SpmdPartitioningTest,
       ConvolutionLhsTiledRhsTiledWindowDilateUneven_HaloOnLhs) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[128,7,14,512]"));
  auto rhs = AllOf(
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
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
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
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
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

TEST_F(SpmdPartitioningTest, PadAlongNonPartitionedDimension) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[128,14,257] parameter(0)
  %param0.copy = f32[128,14,257] copy(%param0), sharding={devices=[1,1,2]0,1}
  %const = f32[] constant(0)
  ROOT %pad = f32[128,17,257] pad(%param0.copy, %const), padding=0_0x1_2x0_0,
    sharding={devices=[1,1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(
      op::Copy(op::DynamicSlice(op::Pad(op::Parameter(), op::Constant()),
                                op::Constant(), op::Constant(), op::Reshape())),
      op::Shape("f32[128,14,129]"));
  EXPECT_THAT(root, AllOf(op::Pad(param0, op::Constant()),
                          op::Shape("f32[128,17,129]")));
}

TEST_F(SpmdPartitioningTest, SliceAlongNonPartitionedDimension) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(
      op::Copy(op::DynamicSlice(op::Pad(op::Parameter(), op::Constant()),
                                op::Constant(), op::Constant(), op::Reshape())),
      op::Shape("f32[128,14,129]"));
  EXPECT_THAT(root, AllOf(op::Slice(param0), op::Shape("f32[128,11,129]")));
}

TEST_F(SpmdPartitioningTest, SliceAlongPartitionedDimension) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[128,14,257] parameter(0)
  %param0.copy = f32[128,14,257] copy(%param0), sharding={devices=[1,1,2]0,1}
  ROOT %slice = f32[63,14,251] slice(%param0.copy),
    slice={[2:128:2], [0:14:1], [5:256:1]}, sharding={devices=[1,1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(
      op::Copy(op::DynamicSlice(op::Pad(op::Parameter(), op::Constant()),
                                op::Constant(), op::Constant(), op::Reshape())),
      op::Shape("f32[128,14,129]"));
  EXPECT_THAT(
      root,
      AllOf(op::Slice(AllOf(
                op::DynamicSlice(
                    AllOf(op::Concatenate(
                              param0,
                              AllOf(op::CollectivePermute(op::Slice(param0)),
                                    op::Shape("f32[128,14,2]"))),
                          op::Shape("f32[128,14,131]")),
                    op::Constant(), op::Constant(), op::Add()),
                op::Shape("f32[128,14,126]"))),
            op::Shape("f32[63,14,126]")));
}

TEST_F(SpmdPartitioningTest, SortAlongNonPartitionedDimension) {
  const char* const hlo_string = R"(
HloModule module

ge {
  p.0.lhs.1247 = f32[]{:T(256)} parameter(0), sharding={replicated}
  bitcast-convert = s32[]{:T(256)} bitcast-convert(p.0.lhs.1247), sharding={replicated}
  constant = s32[]{:T(256)} constant(0), sharding={replicated}
  compare = pred[]{:T(256)E(32)} compare(bitcast-convert, constant), direction=LT, sharding={replicated}
  constant.1 = u32[]{:T(256)} constant(2147483647), sharding={replicated}
  bitcast-convert.1 = u32[]{:T(256)} bitcast-convert(p.0.lhs.1247), sharding={replicated}
  subtract = u32[]{:T(256)} subtract(constant.1, bitcast-convert.1), sharding={replicated}
  bitcast-convert.2 = s32[]{:T(256)} bitcast-convert(subtract), sharding={replicated}
  select = s32[]{:T(256)} select(compare, bitcast-convert.2, bitcast-convert), sharding={replicated}
  p.0.rhs.1248 = f32[]{:T(256)} parameter(1), sharding={replicated}
  bitcast-convert.3 = s32[]{:T(256)} bitcast-convert(p.0.rhs.1248), sharding={replicated}
  compare.1 = pred[]{:T(256)E(32)} compare(bitcast-convert.3, constant), direction=LT, sharding={replicated}
  bitcast-convert.4 = u32[]{:T(256)} bitcast-convert(p.0.rhs.1248), sharding={replicated}
  subtract.1 = u32[]{:T(256)} subtract(constant.1, bitcast-convert.4), sharding={replicated}
  bitcast-convert.5 = s32[]{:T(256)} bitcast-convert(subtract.1), sharding={replicated}
  select.1 = s32[]{:T(256)} select(compare.1, bitcast-convert.5, bitcast-convert.3), sharding={replicated}
  compare.2 = pred[]{:T(256)E(32)} compare(select, select.1), direction=GT, sharding={replicated}
  compare.258 = pred[]{:T(256)E(32)} compare(select.1, select), direction=GT, sharding={replicated}
  compare.259 = pred[]{:T(256)E(32)} compare(compare.2, compare.258), direction=EQ, sharding={replicated}
  p.1.lhs.1249 = s32[]{:T(256)} parameter(2), sharding={replicated}
  p.1.rhs.1250 = s32[]{:T(256)} parameter(3), sharding={replicated}
  compare.260 = pred[]{:T(256)E(32)} compare(p.1.lhs.1249, p.1.rhs.1250), direction=LT, sharding={replicated}
  ROOT select.86 = pred[]{:T(256)E(32)} select(compare.259, compare.260, compare.2), sharding={replicated}
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

  auto root = module->entry_computation()->root_instruction();
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
  const char* const hlo_string = R"(
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

TEST_F(SpmdPartitioningTest, ShardableTranspose) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[16,19,38,4]"));
  EXPECT_THAT(root, AllOf(op::Transpose(param0), op::Shape("f32[16,4,19,38]")));
}

TEST_F(SpmdPartitioningTest, MultiDimensionShardedTranspose) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Reshape(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[4,19,38,4]"));
  EXPECT_THAT(root, AllOf(op::Transpose(param0), op::Shape("f32[19,4,4,38]")));
}

TEST_F(SpmdPartitioningTest, NonShardableTranspose) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto resahrd = AllOf(op::Reshape(op::Transpose(op::Reshape(op::AllToAll()))),
                       op::Shape("f32[16,38,38,2]"));
  EXPECT_THAT(root, AllOf(op::Transpose(), op::Shape("f32[16,2,38,38]")));
}

TEST_F(SpmdPartitioningTest, ShardableReshape) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto param0 =
      AllOf(op::Copy(op::DynamicSlice(op::Parameter(), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[19,38,324]"));
  EXPECT_THAT(root, AllOf(op::Reshape(param0), op::Shape("f32[19,38,4,81]")));
}

TEST_F(SpmdPartitioningTest, NonShardableReshape) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %param0 = f32[38,38,324] parameter(0)
  %param0.copy = f32[38,38,324] copy(%param0), sharding={devices=[1,1,2]0,1}
  ROOT %transpose = f32[38,38,4,81] reshape(%param0.copy),
    sharding={devices=[1,1,1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      AllOf(op::DynamicSlice(
                AllOf(op::Pad(
                          AllOf(op::Reshape(AllOf(op::AllReduce(),
                                                  op::Shape("f32[38,38,324]"))),
                                op::Shape("f32[38,38,4,81]")),
                          op::Constant()),
                      op::Shape("f32[38,38,4,82]")),
                op::Constant(), op::Constant(), op::Constant(), op::Reshape()),
            op::Shape("f32[38,38,4,41]")));
}

TEST_F(SpmdPartitioningTest, ReshapeMergeDimsWithHaloExchange) {
  const char* const hlo_string = R"(
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
  auto exchanged =
      op::DynamicSlice(op::Concatenate(halo, reshape), _, _, _, _, _);
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(exchanged, op::Shape("s32[3,2,1,7,5]")));
}

// Produces an invalid module after transformation.
TEST_F(SpmdPartitioningTest, InceptionV3_4_way_ReduceWindowDilated) {
  const char* const hlo_string = R"(
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
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              AllOf(op::Shape("f32[128,5,17,768]"),
                    op::DynamicSlice(rw, op::Constant(), final_slice_index,
                                     op::Constant(), op::Constant())));
}

TEST_F(SpmdPartitioningTest, TiledToTiledReduce) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto param0 = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(), op::Constant(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[4,32,32,64]"));

  EXPECT_THAT(root,
              AllOf(op::Reduce(param0, op::Constant()), op::Shape("f32[64]")));
}

TEST_F(SpmdPartitioningTest, TiledToTiledTupleReduce) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Reduce(op::Parameter(0), op::Parameter(1),
                                     op::Parameter(2), op::Parameter(3)),
                          op::Shape("(f32[14], s32[14])")));
}

TEST_F(SpmdPartitioningTest, TiledToTiledReduceOutputReshard) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
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
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  ROOT %iota = s32[16,80,91] iota(), iota_dimension=1,
    sharding={devices=[1,1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Iota(), op::Shape("s32[16,80,46]")));
}

TEST_F(SpmdPartitioningTest, IotaAlongTileDimension) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  ROOT %iota = s32[16,80,91] iota(), iota_dimension=2,
    sharding={devices=[1,1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Add(op::Iota(), op::Broadcast()),
                          op::Shape("s32[16,80,46]")));
}

TEST_F(SpmdPartitioningTest, U32IotaAlongTileDimension) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  ROOT %iota = u32[16,80,91] iota(), iota_dimension=2,
    sharding={devices=[1,1,2]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Add(op::Iota(), op::Broadcast()),
                          op::Shape("u32[16,80,46]")));
}

TEST_F(SpmdPartitioningTest, Conditional) {
  const char* const hlo_string = R"(
HloModule module

Negate {
  x = f32[4,5] parameter(0), sharding={replicated}
  ROOT negate = f32[4,5] negate(x), sharding={replicated}
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

  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, AllOf(op::Conditional(op::AllReduce(), param1, param2),
                          op::Shape("f32[2,5]")));

  auto then_branch_root = root->branch_computation(0)->root_instruction();
  EXPECT_THAT(then_branch_root,
              AllOf(op::DynamicSlice(op::Negate(op::Parameter()), op::Reshape(),
                                     op::Constant()),
                    op::Shape("f32[2,5]")));

  auto else_branch_root = root->branch_computation(1)->root_instruction();
  EXPECT_THAT(else_branch_root,
              AllOf(op::Copy(op::Parameter()), op::Shape("f32[2,5]")));
}

TEST_F(SpmdPartitioningTest, SelectAndScatter_RetinaNet) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
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
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(),
                                             op::Reshape())),
                   op::Shape("f32[128,32]"));
  auto rhs = AllOf(op::Copy(op::DynamicSlice(op::Parameter(), op::Reshape(),
                                             op::Constant())),
                   op::Shape("f32[32,256]"));
  EXPECT_THAT(root, AllOf(op::AllReduce(op::Convolution(lhs, rhs)),
                          op::Shape("f32[128,256]")));
}

TEST_F(SpmdPartitioningTest, TiledDotOutputTiled) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(op::Copy(op::DynamicSlice(op::Parameter(), op::Constant(),
                                             op::Reshape())),
                   op::Shape("f32[128,32]"));
  auto rhs = AllOf(op::Copy(op::DynamicSlice(op::Parameter(), op::Reshape(),
                                             op::Constant())),
                   op::Shape("f32[32,256]"));
  EXPECT_THAT(root, AllOf(op::DynamicSlice(
                              AllOf(op::AllReduce(op::Convolution(lhs, rhs)),
                                    op::Shape("f32[128,256]")),
                              op::Constant(), op::Reshape()),
                          op::Shape("f32[128,128]")));
}

TEST_F(SpmdPartitioningTest, BatchPartitionedConvolution) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(),
                                             op::Reshape(), op::Constant())),
                   op::Shape("f32[128,128,256]"));
  auto rhs = AllOf(op::Copy(op::Parameter(1)), op::Shape("f32[256,8,1]"));
  EXPECT_THAT(root,
              AllOf(op::Convolution(lhs, rhs), op::Shape("f32[128,128,8]")));
}

TEST_F(SpmdPartitioningTest, DotOutputFeaturePartitioned) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(op::Copy(op::Parameter(0)), op::Shape("f32[24,64]"));
  auto rhs = AllOf(op::Copy(op::DynamicSlice(op::Parameter(1), op::Reshape(),
                                             op::Constant())),
                   op::Shape("f32[19648,64]"));
  EXPECT_THAT(root, AllOf(op::Dot(lhs, rhs), op::Shape("f32[24,19648]")));
}

TEST_F(SpmdPartitioningTest, EinsumBatchPartitioned) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(op::Copy(op::DynamicSlice(op::Parameter(0), op::Reshape(),
                                             op::Constant(), op::Constant())),
                   op::Shape("f32[16,24,64]"));
  auto rhs = AllOf(op::Copy(op::DynamicSlice(op::Parameter(1), op::Reshape(),
                                             op::Constant(), op::Constant())),
                   op::Shape("f32[16,39296,64]"));
  EXPECT_THAT(root, AllOf(op::Dot(lhs, rhs), op::Shape("f32[16,24,39296]")));
}

TEST_F(SpmdPartitioningTest, EinsumLHSandOutputBatchPartitioned) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(op::Copy(op::DynamicSlice(op::Parameter(0), op::Reshape(),
                                             op::Constant(), op::Constant())),
                   op::Shape("f32[16,24,64]"));
  auto rhs = AllOf(op::Copy(op::Parameter(1)), op::Shape("f32[32,39296,64]"));
  EXPECT_THAT(root, AllOf(op::Dot(lhs, op::DynamicSlice(rhs, op::Reshape(),
                                                        op::Constant(),
                                                        op::Constant())),
                          op::Shape("f32[16,24,39296]")));
}

TEST_F(SpmdPartitioningTest, EinsumRHSandOutputBatchPartitioned) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(),
                                             op::Reshape(), op::Constant())),
                   op::Shape("f32[32,12,64]"));
  auto rhs = AllOf(op::Copy(op::DynamicSlice(op::Parameter(1), op::Reshape(),
                                             op::Constant(), op::Constant())),
                   op::Shape("f32[16,39296,64]"));
  auto lhs_reshard = op::Reshape(op::Transpose(op::AllToAll(op::Reshape(lhs))));
  EXPECT_THAT(root,
              AllOf(op::Dot(lhs_reshard, rhs), op::Shape("f32[16,24,39296]")));
}

TEST_F(SpmdPartitioningTest, EinsumOutputBatchPartitioned) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs_slice =
      AllOf(op::DynamicSlice(op::Copy(op::Parameter(0)), op::Reshape(),
                             op::Constant(), op::Constant()),
            op::Shape("f32[16,24,64]"));
  auto rhs_slice =
      AllOf(op::DynamicSlice(op::Copy(op::Parameter(1)), op::Reshape(),
                             op::Constant(), op::Constant()),
            op::Shape("f32[16,39296,64]"));
  EXPECT_THAT(root, AllOf(op::Dot(lhs_slice, rhs_slice),
                          op::Shape("f32[16,24,39296]")));
}

TEST_F(SpmdPartitioningTest, EinsumContractingDimsPartitioned) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(),
                                op::Constant(), op::Reshape(), op::Reshape())),
      op::Shape("f32[32,24,32,64]"));
  auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(1), op::Constant(),
                                op::Constant(), op::Reshape(), op::Reshape())),
      op::Shape("f32[32,39296,32,64]"));
  EXPECT_THAT(root, AllOf(op::AllReduce(op::Dot(lhs, rhs)),
                          op::Shape("f32[32,24,39296]")));
}

TEST_F(SpmdPartitioningTest, EinsumLHSNonContractingDimsPartitioned) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(), op::Reshape(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[32,12,64,64]"));
  auto rhs = AllOf(op::Copy(op::Parameter(1)), op::Shape("f32[32,39296,64]"));
  EXPECT_THAT(root, AllOf(op::Dot(lhs, rhs), op::Shape("f32[32,12,64,39296]")));
}

TEST_F(SpmdPartitioningTest, EinsumRHSNonContractingDimsPartitioned) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(op::Copy(op::Parameter(0)), op::Shape("f32[32,24,64]"));
  auto rhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(1), op::Constant(), op::Reshape(),
                                op::Constant(), op::Reshape())),
      op::Shape("f32[32,19648,64,64]"));
  EXPECT_THAT(root, AllOf(op::Dot(lhs, rhs), op::Shape("f32[32,24,19648,64]")));
}

TEST_F(SpmdPartitioningTest, EinsumOutputLHSNonContractingDimPartitioned) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(op::Copy(op::Parameter(0)), op::Shape("f32[32,24,64,128]"));
  auto rhs =
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
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(op::Copy(op::Parameter(0)), op::Shape("f32[32,24,64,128]"));
  auto rhs =
      AllOf(op::Copy(op::Parameter(1)), op::Shape("f32[32,39296,64,128]"));
  EXPECT_THAT(root,
              AllOf(op::Dot(lhs, AllOf(op::DynamicSlice(
                                           rhs, op::Constant(), op::Reshape(),
                                           op::Constant(), op::Constant()),
                                       op::Shape("f32[32,19648,64,128]"))),
                    op::Shape("f32[32,24,19648]")));
}

TEST_F(SpmdPartitioningTest, EinsumRHSWindowedNonContracting) {
  const char* const hlo_string = R"(
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
  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[32,12,64,128]"));
  auto rhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(1), op::Constant()),
                                      op::Constant(), op::Reshape(),
                                      op::Constant(), op::Constant())),
            op::Shape("f32[32,19648,64,128]"));
  EXPECT_THAT(
      root,
      AllOf(op::Slice(AllOf(op::GetTupleElement(op::While(op::Tuple(
                                lhs, rhs, op::Broadcast(), op::Constant()))),
                            op::Shape("f32[32,12,39296]"))),
            op::Shape("f32[32,12,39295]")));
  auto while_loop = root->operand(0)->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  auto next_i = op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant());
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
                next_i));

  // Check the conditional that contains the collective permute.
  auto cp_conditional =
      while_loop->while_body()->root_instruction()->operand(1);
  EXPECT_THAT(cp_conditional->true_computation()->root_instruction(),
              op::CollectivePermute(op::Parameter(0)));
  EXPECT_THAT(cp_conditional->false_computation()->root_instruction(),
              op::Parameter(0));
}

TEST_F(SpmdPartitioningTest, EinsumRHSWindowedContracting) {
  const char* const hlo_string = R"(
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
  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(
      op::Copy(op::DynamicSlice(op::Parameter(0), op::Constant(), op::Reshape(),
                                op::Constant(), op::Constant())),
      op::Shape("f32[32,12,63,128]"));
  auto rhs =
      AllOf(op::Copy(op::DynamicSlice(op::Pad(op::Parameter(1), op::Constant()),
                                      op::Constant(), op::Constant(),
                                      op::Reshape(), op::Constant())),
            op::Shape("f32[32,39296,32,128]"));
  auto masked_rhs =
      op::Select(op::Compare(), rhs, op::Broadcast(op::Constant()));
  EXPECT_THAT(root,
              AllOf(op::GetTupleElement(op::While(op::Tuple(
                        lhs, masked_rhs, op::Broadcast(), op::Constant()))),
                    op::Shape("f32[32,12,39296]")));
  auto while_loop = root->operand(0);
  // Check loop condition.
  EXPECT_THAT(
      while_loop->while_condition()->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0)), op::Constant()));

  // Check loop body.
  auto next_i = op::Add(op::GetTupleElement(op::Parameter(0)), op::Constant());
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
                next_i));

  // Check the conditional that contains the collective permute.
  auto cp_conditional =
      while_loop->while_body()->root_instruction()->operand(1);
  EXPECT_THAT(cp_conditional->true_computation()->root_instruction(),
              op::CollectivePermute(op::Parameter(0)));
  EXPECT_THAT(cp_conditional->false_computation()->root_instruction(),
              op::Parameter(0));
}

TEST_F(SpmdPartitioningTest, EinsumRHSWindowedNonContractingReduce1) {
  const char* const hlo_string = R"(
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

  TF_ASSERT_OK_AND_ASSIGN(auto module, PartitionComputation(hlo_string,
                                                            /*num_devices=*/2));
  VLOG(1) << module->ToString();
  // Involves loop code motion, skips pattern matching.
}

TEST_F(SpmdPartitioningTest, EinsumRHSWindowedNonContractingReduce2) {
  const char* const hlo_string = R"(
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

TEST_F(SpmdPartitioningTest, EinsumRHSWindowedContractingFromBroadcast) {
  const char* const hlo_string = R"(
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

TEST_F(SpmdPartitioningTest, ReplicatedRng) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(op::Copy(op::Parameter(0)), op::Shape("s32[]"));
  auto rhs = AllOf(op::Copy(op::Parameter(1)), op::Shape("s32[]"));
  EXPECT_THAT(
      root,
      AllOf(op::AllReduce(op::Select(
                op::Broadcast(op::Compare(op::PartitionId(), op::Constant())),
                op::Rng(), op::Broadcast(op::Constant()))),
            op::Shape("s32[4]")));
}

TEST_F(SpmdPartitioningTest, PartitionedRng) {
  const char* const hlo_string = R"(
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

  auto root = module->entry_computation()->root_instruction();
  auto lhs = AllOf(op::Copy(op::Parameter(0)), op::Shape("s32[]"));
  auto rhs = AllOf(op::Copy(op::Copy(op::Parameter(1))), op::Shape("s32[]"));
  EXPECT_THAT(root, AllOf(op::Rng(lhs, op::AllReduce(op::Select(
                                           op::Broadcast(op::Compare()), rhs,
                                           op::Broadcast(op::Constant())))),
                          op::Shape("s32[2]")));
}

TEST_F(SpmdPartitioningTest, DynamicSliceAlongNonPartitionedDimension) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %input = s32[128,64] parameter(0)
  %input.copy = s32[128,64] copy(%input), sharding={devices=[2,1]0,1}
  %index = s32[] parameter(1)
  %constant = s32[] constant(0)
  ROOT %dynamic-slice = s32[128,2] dynamic-slice(%input.copy, %constant, %index),
    dynamic_slice_sizes={128,2}, sharding={devices=[2,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  auto root = module->entry_computation()->root_instruction();
  auto input = AllOf(op::Copy(op::DynamicSlice(op::Parameter(0), op::Reshape(),
                                               op::Constant())),
                     op::Shape("s32[64,64]"));
  EXPECT_THAT(root,
              AllOf(op::DynamicSlice(input, op::Constant(), op::Parameter(1)),
                    op::Shape("s32[64,2]")));
}

TEST_F(SpmdPartitioningTest, DynamicUpdateSliceAlongNonPartitionedDimension) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %input = s32[128,64] parameter(0)
  %input.copy = s32[128,64] copy(%input), sharding={devices=[2,1]0,1}
  %index = s32[] parameter(1)
  %constant = s32[] constant(0)
  %update = s32[128,2] parameter(2)
  %update.copy = s32[128,2] copy(%update), sharding={devices=[2,1]0,1}
  ROOT %dynamic-update-slice = s32[128,64]
    dynamic-update-slice(%input.copy, %update.copy, %constant, %index),
    sharding={devices=[2,1]0,1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();

  auto root = module->entry_computation()->root_instruction();
  auto input = AllOf(op::Copy(op::DynamicSlice(op::Parameter(0), op::Reshape(),
                                               op::Constant())),
                     op::Shape("s32[64,64]"));
  auto update = AllOf(op::Copy(op::DynamicSlice(op::Parameter(2), op::Reshape(),
                                                op::Constant())),
                      op::Shape("s32[64,2]"));
  EXPECT_THAT(root, AllOf(op::DynamicUpdateSlice(input, update, op::Constant(),
                                                 op::Parameter(1)),
                          op::Shape("s32[64,64]")));
}

TEST_F(SpmdPartitioningTest, PassthroughGather) {
  const char* const hlo_string = R"(
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

TEST_F(SpmdPartitioningTest, GatherPartitionedOnTrivialSliceDims) {
  const char* const hlo_string = R"(
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
  auto offset = op::Reshape(
      op::DynamicSlice(op::Constant(), op::PartitionId(), op::Constant()));
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

TEST_F(SpmdPartitioningTest, PassthroughScatter) {
  const char* const hlo_string = R"(
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

TEST_F(SpmdPartitioningTest, ScatterPartitionedOnTrivialSliceDims) {
  const char* const hlo_string = R"(
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
  auto offset = op::Reshape(
      op::DynamicSlice(op::Constant(), op::PartitionId(), op::Constant()));
  auto indices = op::Subtract(
      op::Parameter(1), AllOf(op::Broadcast(offset), op::Shape("s32[2,3]")));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              AllOf(op::Scatter(op::Parameter(0), indices, op::Parameter(2)),
                    op::Shape("f32[9,9]")));
}

TEST_F(SpmdPartitioningTest, TiledReversePassthrough) {
  const char* const hlo_string = R"(
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
  const char* const hlo_string = R"(
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
  const char* const hlo_string = R"(
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
  const char* const hlo_string = R"(
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
                      op::Parameter(0));
  auto after_halo_exchange = op::Slice(halo_exchange_concat);
  EXPECT_THAT(root,
              AllOf(op::Shape("f32[2]"), op::Reverse(after_halo_exchange)));
}

TEST_F(SpmdPartitioningTest, MixWithManualPartitioning) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  param = f32[8,2] parameter(0), sharding={devices=[2,1]0,1}
  to_shard = f32[4,2] custom-call(param), custom_call_target="SPMDFullToShardShape", sharding={replicated}
  add = f32[4,2] add(to_shard, to_shard), sharding={replicated}
  to_full = f32[8,2] custom-call(add), custom_call_target="SPMDShardToFullShape", sharding={devices=[2,1]0,1}
  ROOT mul = f32[8,2] multiply(to_full, param), sharding={devices=[2,1]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(hlo_string, /*num_devices=*/2));
  VLOG(1) << module->ToString();
  HloInstruction* root = module->entry_computation()->root_instruction();
  auto to_shard = op::Copy(op::Parameter(0));
  EXPECT_THAT(root, AllOf(op::Shape("f32[4,2]"),
                          op::Multiply(op::Copy(op::Add(to_shard, to_shard)),
                                       op::Parameter(0))));
}

}  // namespace
}  // namespace spmd
}  // namespace xla
