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

#include "xla/tools/hlo_control_flow_flattening.h"

#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/transforms/despecializer.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/spmd/spmd_partitioner.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

class HloControlFlowFlatteningTest : public HloTestBase {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> PartitionComputation(
      std::unique_ptr<VerifiedHloModule> hlo_module, int64_t num_devices = 2) {
    spmd::SpmdPartitionerOptions options;
    auto collective_ops_creator =
        spmd::GetDefaultCollectiveOpsCreator(num_devices, /*num_replicas=*/1);
    collective_ops_creator.create_cross_partition_all_gather = nullptr;

    HloModuleConfig config = GetModuleConfigForTest();
    config.set_use_spmd_partitioning(true);
    config.set_num_partitions(num_devices);
    HloPassPipeline pass("spmd-partitioning");
    pass.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                              /*allow_mixed_precision=*/false);
    pass.AddPass<spmd::SpmdPartitioner>(num_devices, /*num_replicas=*/1,
                                        options, collective_ops_creator);
    pass.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                              /*allow_mixed_precision=*/false);
    TF_RETURN_IF_ERROR(pass.Run(hlo_module.get()).status());
    return absl::StatusOr<std::unique_ptr<HloModule>>(std::move(hlo_module));
  }
};

constexpr int kDefaultMaxLoopCount = 1000;

TEST_F(HloControlFlowFlatteningTest, WhileRoot) {
  absl::string_view hlo_string = R"(
  HloModule While
  While.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add, multiply)
  }
  While.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(100)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY While {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=While.condition, body=While.body
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());

  auto root = module->entry_computation()->root_instruction();
  auto while_op = module->entry_computation()->GetInstructionWithName("while");
  EXPECT_THAT(root, op::Tuple(op::GetTupleElement(while_op, 0),
                              op::GetTupleElement(while_op, 1)));
  EXPECT_THAT(while_op,
              op::While(op::Tuple(op::GetTupleElement(), op::GetTupleElement(),
                                  op::Constant())));
  auto condition = while_op->while_condition();
  EXPECT_THAT(
      condition->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0), 2), op::Constant()));

  auto body = while_op->while_body();
  EXPECT_THAT(body->root_instruction(),
              op::Tuple(op::GetTupleElement(), op::GetTupleElement(),
                        op::Add(op::GetTupleElement(op::Parameter(0), 2),
                                op::Constant())));
}

TEST_F(HloControlFlowFlatteningTest, WhileConditionCallComputation) {
  absl::string_view hlo_string = R"(
  HloModule While
  While.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add, multiply)
  }
  While.condition.called {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] custom-call(), custom_call_target="AllocateBuffer", custom_call_has_side_effect=true
    less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
    ROOT tuple.2 = (pred[]) tuple(less-than)
  }
  While.condition {
    loop_var.3 = (s32[], s32[3]{0}) parameter(0)
    call = (pred[]) call(loop_var.3), to_apply=While.condition.called
    ROOT get-tuple-element.4 = pred[] get-tuple-element(call), index=0
  }
  ENTRY While {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=While.condition, body=While.body
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  XLA_VLOG_LINES(3, "Loaded HLO module: " + module->ToString());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());

  auto root = module->entry_computation()->root_instruction();
  auto while_op = module->entry_computation()->GetInstructionWithName("while");
  EXPECT_THAT(root, op::Tuple(op::GetTupleElement(while_op, 0),
                              op::GetTupleElement(while_op, 1)));
  EXPECT_THAT(while_op,
              op::While(op::Tuple(op::GetTupleElement(), op::GetTupleElement(),
                                  op::Constant())));
  auto condition = while_op->while_condition();
  EXPECT_THAT(
      condition->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0), 2), op::Constant()));

  auto body = while_op->while_body();
  EXPECT_THAT(body->root_instruction(),
              op::Tuple(op::GetTupleElement(), op::GetTupleElement(),
                        op::Add(op::GetTupleElement(op::Parameter(0), 2),
                                op::Constant())));
}

TEST_F(HloControlFlowFlatteningTest, WhileRootScheduled) {
  absl::string_view hlo_string = R"(
  HloModule While, is_scheduled=true
  While.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add, multiply)
  }
  While.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(100)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY While {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=While.condition, body=While.body
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());

  auto root = module->entry_computation()->root_instruction();
  auto while_op = module->entry_computation()->GetInstructionWithName("while");
  EXPECT_THAT(root, op::Tuple(op::GetTupleElement(while_op, 0),
                              op::GetTupleElement(while_op, 1)));
  EXPECT_THAT(while_op,
              op::While(op::Tuple(op::GetTupleElement(), op::GetTupleElement(),
                                  op::Constant())));
  auto condition = while_op->while_condition();
  EXPECT_THAT(
      condition->root_instruction(),
      op::Compare(op::GetTupleElement(op::Parameter(0), 2), op::Constant()));
}

TEST_F(HloControlFlowFlatteningTest, WhileUser) {
  absl::string_view hlo_string = R"(
  HloModule While
  While.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add, multiply)
  }
  While.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(100)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  FusedComputation {
    param = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.4 = s32[] get-tuple-element(param), index=0
    get-tuple-element.5 = s32[3]{0} get-tuple-element(param), index=1
    broadcast = s32[3]{0} broadcast(get-tuple-element.4), dimensions={}
    ROOT add = s32[3]{0} add(broadcast, get-tuple-element.5)
  }
  ENTRY While {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    while = (s32[], s32[3]{0}) while(tuple.1), condition=While.condition, body=While.body
    ROOT fusion = s32[3]{0} fusion(while), kind=kLoop, calls=FusedComputation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());

  auto fusion = module->entry_computation()->root_instruction();
  auto while_op = module->entry_computation()->GetInstructionWithName("while");
  EXPECT_THAT(fusion, op::Fusion(op::Tuple(op::GetTupleElement(while_op, 0),
                                           op::GetTupleElement(while_op, 1))));
}

TEST_F(HloControlFlowFlatteningTest, Infeed) {
  absl::string_view hlo_string = R"(
  HloModule Infeed
  ENTRY Infeed {
    after-all = token[] after-all()
    ROOT infeed.23 = ((bf16[3]{0}, s32[12,5]{0,1}), token[]) infeed(after-all)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  auto custom_call =
      module->entry_computation()->GetInstructionWithName("infeed.23");
  EXPECT_THAT(custom_call, op::CustomCall());
  auto tuple = module->entry_computation()->root_instruction();
  EXPECT_THAT(tuple, op::Tuple(custom_call, op::AfterAll()));
}

TEST_F(HloControlFlowFlatteningTest, InfeedPreserveLayout) {
  absl::string_view hlo_string = R"(
  HloModule Infeed
  ENTRY Infeed {
    after-all = token[] after-all()
    ROOT infeed = ((bf16[3]{0}, s32[12,5]{0,1:T(8,128)}), token[]) infeed(after-all)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Shape root_shape = module->entry_computation()->root_instruction()->shape();
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  auto tuple = module->entry_computation()->root_instruction();
  EXPECT_THAT(tuple, op::Tuple(op::CustomCall(), op::AfterAll()));
  EXPECT_EQ(tuple->shape(), root_shape);
}

TEST_F(HloControlFlowFlatteningTest, OutfeedCustomCallIsPartitionable) {
  absl::string_view hlo_string = R"(
  HloModule Outfeed
  ENTRY Outfeed {
    param = (bf16[3]{0}, s32[12,5]{0,1}) parameter(0)
    after-all = token[] after-all()
    ROOT outfeed.23 = token[] outfeed(param, after-all)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(HloControlFlowFlattening::Options{
      /*while_execution_count=*/3, /*max_outer_loop_count=*/3,
      /*max_loop_count=*/3, /*remove_infeed_outfeed=*/true});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  auto custom_call = module->entry_computation()->root_instruction();
  EXPECT_EQ(custom_call->name(), "outfeed.23");
  EXPECT_TRUE(custom_call->has_sharding());
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          PartitionComputation(std::move(module)));
}

TEST_F(HloControlFlowFlatteningTest, Outfeed) {
  absl::string_view hlo_string = R"(
  HloModule Outfeed
  ENTRY Outfeed {
    param = (bf16[3]{0}, s32[12,5]{0,1}) parameter(0)
    after-all = token[] after-all()
    ROOT outfeed.23 = token[] outfeed(param, after-all)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  auto custom_call = module->entry_computation()->root_instruction();
  EXPECT_EQ(custom_call->name(), "outfeed.23");
  EXPECT_THAT(custom_call, op::CustomCall(op::Parameter(0), op::AfterAll()));
}

TEST_F(HloControlFlowFlatteningTest, PredicatedConditional) {
  absl::string_view hlo_string = R"(

  HloModule pred_conditional, entry_computation_layout={()->f32[]}

  Negate {
    x = f32[] parameter(0)
    ROOT negate = f32[] negate(x)
  }

  Identity {
    y = f32[] parameter(0)
    ROOT copy = f32[] copy(y)
  }

  ENTRY Parameters1.v4 {
    constant = pred[] constant(true)
    constant.1 = f32[] constant(56)
    constant.2 = f32[] constant(12)
    ROOT conditional = f32[] conditional(constant, constant.1, constant.2), true_computation=Negate, false_computation=Identity
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(HloControlFlowFlattening::Options{
      /*while_execution_count=*/3, /*max_outer_loop_count=*/3,
      /*max_loop_count=*/3, /*remove_infeed_outfeed=*/true,
      /*flatten_while_loop=*/true, /*remove_comm=*/false,
      /*remove_host_transfer=*/false, /*remove_id=*/false,
      /*flatten_conditional=*/true, /*conditional_value=*/false});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  auto conditional = module->entry_computation()->root_instruction();
  EXPECT_EQ(conditional->name(), "conditional");

  auto new_predicate = conditional->operand(0);
  EXPECT_EQ(new_predicate->literal().Get<bool>({}), false);
  EXPECT_EQ(new_predicate->name(), "constant_flattened");
  EXPECT_TRUE(module->entry_computation()->GetInstructionWithName(
                  "custom-call") != nullptr);
}

TEST_F(HloControlFlowFlatteningTest, IndexedConditional) {
  absl::string_view hlo_string = R"(
  HloModule indexed_conditional, entry_computation_layout={()->f32[]}

  %Negate (x: f32[]) -> f32[] {
    %x = f32[] parameter(0)
    ROOT %negate = f32[] negate(f32[] %x)
  }

  %Identity (y: f32[]) -> f32[] {
    %y = f32[] parameter(0)
    ROOT %copy = f32[] copy(f32[] %y)
  }

  %Floor (z: f32[]) -> f32[] {
    %z = f32[] parameter(0)
    ROOT %floor = f32[] floor(f32[] %z)
  }

  ENTRY %Parameters1.v4 () -> f32[] {
    %constant = s32[] constant(1)
    %constant.1 = f32[] constant(56)
    %constant.2 = f32[] constant(12)
    %constant.3 = f32[] constant(13)
    ROOT %conditional = f32[] conditional(s32[] %constant, f32[] %constant.1, f32[] %constant.2, f32[] %constant.3), branch_computations={%Negate, %Identity, %Floor}
  }

  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(HloControlFlowFlattening::Options{
      /*while_execution_count=*/3, /*max_outer_loop_count=*/3,
      /*max_loop_count=*/3, /*remove_infeed_outfeed=*/true,
      /*flatten_while_loop=*/true, /*remove_comm=*/false,
      /*remove_host_transfer=*/false, /*remove_id=*/false,
      /*flatten_conditional=*/true, /*conditional_value=*/false});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  auto conditional = module->entry_computation()->root_instruction();
  EXPECT_EQ(conditional->name(), "conditional");

  auto new_index = conditional->operand(0);
  EXPECT_EQ(new_index->literal().Get<int32_t>({}), 2);
  EXPECT_EQ(new_index->name(), "constant_flattened");
  EXPECT_TRUE(module->entry_computation()->GetInstructionWithName(
                  "custom-call") != nullptr);
}

TEST_F(HloControlFlowFlatteningTest, AllReduce) {
  absl::string_view hlo_string = R"(
  HloModule AllReduce
  sum {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    ROOT add = f32[] add(p0, p1)
  }

  ENTRY AllReduce {
    param0 = f32[3]{0} parameter(0)
    param1 = f32[12,5]{0,1} parameter(1)
    ROOT all-reduce = (bf16[3]{0}, bf16[12,5]{0,1}) all-reduce(param0, param1), to_apply=sum, replica_groups={}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::Parameter(0), op::Parameter(1)));
  EXPECT_EQ(module->entry_computation()->root_instruction()->name(),
            "all-reduce");
}

TEST_F(HloControlFlowFlatteningTest, AllReduceStartAndDone) {
  absl::string_view hlo_string = R"(
  HloModule CRS

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  ENTRY CRS {
    input = f32[8]{0} parameter(0)
    crs = f32[8]{0} all-reduce-start(input), replica_groups={}, to_apply=add
    ROOT done = f32[8]{0} all-reduce-done(crs)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::CustomCall(op::Parameter(0))));
  EXPECT_EQ(module->entry_computation()->root_instruction()->name(), "done");
  EXPECT_EQ(module->entry_computation()->root_instruction()->operand(0)->name(),
            "crs");
}

TEST_F(HloControlFlowFlatteningTest, AllGather) {
  absl::string_view hlo_string = R"(
  HloModule AllGather

  ENTRY AllGather {
    input = f32[128,32]{0,1} parameter(0)
    ROOT ag = f32[128,128]{0,1} all-gather(input), replica_groups={}, dimensions={1}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::Parameter(0)));
  EXPECT_EQ(module->entry_computation()->root_instruction()->name(), "ag");
}

TEST_F(HloControlFlowFlatteningTest, AllToAll) {
  absl::string_view hlo_string = R"(
  HloModule AllToAll

  ENTRY AllToAll {
    input = f32[128,32]{0,1} parameter(0)
    ROOT a2a = (f32[128,32]{0,1}) all-to-all(input), replica_groups={}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::Parameter(0)));
  EXPECT_EQ(module->entry_computation()->root_instruction()->name(), "a2a");
}

TEST_F(HloControlFlowFlatteningTest, CollectivePermute) {
  absl::string_view hlo_string = R"(
  HloModule CollectivePermute

  ENTRY CollectivePermute {
    input = f32[128,32]{0,1} parameter(0)
    ROOT collective-permute = f32[128,32]{0,1} collective-permute(input), source_target_pairs={{0,1},{1,2},{2,3}}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::Parameter(0)));
  EXPECT_EQ(module->entry_computation()->root_instruction()->name(),
            "collective-permute");
}

TEST_F(HloControlFlowFlatteningTest, ReplicaIdSucceedsWithChange) {
  absl::string_view hlo_string = R"(
  HloModule ReplicaId

  ENTRY ReplicaId {
    ROOT replica-id.18600 = u32[]{:T(128)} replica-id()
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(HloControlFlowFlattening::Options{});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  EXPECT_THAT(module->entry_computation()->root_instruction(), op::Constant());
  EXPECT_EQ(module->entry_computation()->root_instruction()->name(),
            "replica-id.18600");
}

TEST_F(HloControlFlowFlatteningTest, RemoveReplicaIdButKeepAllReduce) {
  absl::string_view kHloText = R"(
  HloModule RemoveReplicaIdButKeepCollective

%sum (a: f32[], b: f32[]) -> f32[] {
    %a = f32[] parameter(0)
    %b = f32[] parameter(1)
    ROOT %add = f32[] add(f32[] a, f32[] b)
  }
  ENTRY ReplicaId {
    replica-id.1 = u32[]{:T(128)} replica-id()
    ROOT all-reduce.1 = u32[]{:T(128)} all-reduce(replica-id.1), to_apply=sum, replica_groups={}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  HloControlFlowFlattening flattening(HloControlFlowFlattening::Options{
      /*while_execution_count=*/1, /*max_outer_loop_count=*/1,
      /*max_loop_count=*/1, /*remove_infeed_outfeed=*/false,
      /*flatten_while_loop=*/false, /*remove_comm=*/false,
      /*remove_host_transfer=*/false, /*remove_id=*/true});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  EXPECT_THAT(module->entry_computation()->root_instruction(), op::AllReduce());
  EXPECT_THAT(module->entry_computation()->root_instruction()->operand(0),
              op::Constant());
}

TEST_F(HloControlFlowFlatteningTest, CollectivePermuteInPlaceUpdate) {
  absl::string_view hlo_string = R"(
  HloModule CollectivePermuteInPlaceUpdate

  ENTRY CollectivePermuteInPlaceUpdate {
    input = f32[128,32]{0,1} parameter(0)
    constant = f32[] constant(1)
    output = f32[128,128]{0,1} broadcast(constant), dimensions={}
    constant.1 = s32[] constant(0)
    tuple.1 = (s32[], s32[]) tuple(constant.1, constant.1)
    constant.2 = s32[] constant(64)
    tuple.2 = (s32[], s32[]) tuple(constant.1, constant.2)
    ROOT collective-permute = f32[128,128]{0,1} collective-permute(input, output, tuple.1, tuple.2), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{128,32}}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::Parameter(0), op::Broadcast(op::Constant()),
                             op::Tuple(op::Constant(), op::Constant()),
                             op::Tuple(op::Constant(), op::Constant())));
  EXPECT_EQ(module->entry_computation()->root_instruction()->name(),
            "collective-permute");
}

TEST_F(HloControlFlowFlatteningTest, CollectivePermuteStartAndDone) {
  absl::string_view hlo_string = R"(
  HloModule CollectivePermuteStartAndDone

  ENTRY CollectivePermuteStartAndDone {
    input = f32[128,32]{0,1} parameter(0)
    collective-permute-start.1 = (f32[128,32]{0,1}, f32[128,32]{0,1}, u32[], u32[]) collective-permute-start(input), source_target_pairs={{0,1},{1,2},{2,3}}
    ROOT collective-permute-done.1 = f32[128,32]{0,1} collective-permute-done(collective-permute-start.1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::CustomCall(op::Parameter(0))));
  EXPECT_EQ(module->entry_computation()->root_instruction()->name(),
            "collective-permute-done.1");
  EXPECT_EQ(module->entry_computation()->root_instruction()->operand(0)->name(),
            "collective-permute-start.1");
}

TEST_F(HloControlFlowFlatteningTest, Recv) {
  absl::string_view hlo_string = R"(
  HloModule Recv

  ENTRY %Recv () -> (f32[], token[]) {
    %token0 = token[] after-all()
    %recv = (f32[], u32[], token[]) recv(token[] %token0), channel_id=15
    ROOT %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv), channel_id=15
    %constant = f32[] constant(2.1)
    %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0), channel_id=16, control-predecessors={%recv}
    %send-done = token[] send-done((f32[], u32[], token[]) %send), channel_id=16
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ControlDepRemover control_remover;
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  TF_ASSERT_OK(control_remover.Run(module.get()).status());
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::CustomCall()));
  EXPECT_EQ(module->entry_computation()->root_instruction()->name(),
            "recv-done");
  EXPECT_EQ(module->entry_computation()->root_instruction()->operand(0)->name(),
            "recv");
}

TEST_F(HloControlFlowFlatteningTest, RecvHostTransfer) {
  absl::string_view hlo_string = R"(
  HloModule Recv

  ENTRY %Recv () -> (f32[], token[]) {
    %token0 = token[] after-all()
    %recv = (f32[], u32[], token[]) recv(token[] %token0), channel_id=15, is_host_transfer=true
    ROOT %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv), channel_id=15, is_host_transfer=true
    %constant = f32[] constant(2.1)
    %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0), channel_id=16, control-predecessors={%recv}
    %send-done = token[] send-done((f32[], u32[], token[]) %send), channel_id=16
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ControlDepRemover control_remover;
  HloControlFlowFlattening flattening(HloControlFlowFlattening::Options{
      /*while_execution_count=*/3, /*max_outer_loop_count=*/3,
      /*max_loop_count=*/3, /*remove_infeed_outfeed=*/true,
      /*flatten_while_loop=*/true, /*remove_comm=*/false,
      /*remove_host_transfer=*/true});
  TF_ASSERT_OK(control_remover.Run(module.get()).status());
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::CustomCall()));
  EXPECT_EQ(module->entry_computation()->root_instruction()->name(),
            "recv-done");
  EXPECT_EQ(module->entry_computation()->root_instruction()->operand(0)->name(),
            "recv");
}

TEST_F(HloControlFlowFlatteningTest, Send) {
  absl::string_view hlo_string = R"(
  HloModule Send

  ENTRY %Send () -> token[] {
    %token0 = token[] after-all()
    %recv = (f32[], u32[], token[]) recv(token[] %token0), channel_id=15
    %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv), channel_id=15
    %constant = f32[] constant(2.1)
    %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0), channel_id=16, control-predecessors={%recv}
    ROOT %send-done = token[] send-done((f32[], u32[], token[]) %send), channel_id=16
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ControlDepRemover control_remover;
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  TF_ASSERT_OK(control_remover.Run(module.get()).status());
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::CustomCall()));
  EXPECT_EQ(module->entry_computation()->root_instruction()->name(),
            "send-done");
  EXPECT_EQ(module->entry_computation()->root_instruction()->operand(0)->name(),
            "send");
}

TEST_F(HloControlFlowFlatteningTest, SendHostTransfer) {
  absl::string_view hlo_string = R"(
  HloModule Send

  ENTRY %Send () -> token[] {
    %token0 = token[] after-all()
    %recv = (f32[], u32[], token[]) recv(token[] %token0), channel_id=15
    %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv), channel_id=15
    %constant = f32[] constant(2.1)
    %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0), channel_id=16, is_host_transfer=true, control-predecessors={%recv}
    ROOT %send-done = token[] send-done((f32[], u32[], token[]) %send), channel_id=16, is_host_transfer=true
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ControlDepRemover control_remover;
  HloControlFlowFlattening flattening(HloControlFlowFlattening::Options{
      /*while_execution_count=*/3, /*max_outer_loop_count=*/3,
      /*max_loop_count=*/3, /*remove_infeed_outfeed=*/true,
      /*flatten_while_loop=*/true, /*remove_comm=*/false,
      /*remove_host_transfer=*/true});
  TF_ASSERT_OK(control_remover.Run(module.get()).status());
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::CustomCall()));
  EXPECT_EQ(module->entry_computation()->root_instruction()->name(),
            "send-done");
  EXPECT_EQ(module->entry_computation()->root_instruction()->operand(0)->name(),
            "send");
}

TEST_F(HloControlFlowFlatteningTest, AllGatherStartAndDone) {
  absl::string_view hlo_string = R"(
  HloModule AllGatherStartAndDone

  ENTRY AllGatherStartAndDone {
    %input = f32[8,256,256] parameter(0)
    %ag-start = (f32[8,256,256], f32[16,256,256]) all-gather-start(
      f32[8,256,256] %input), replica_groups={{0,1}}, dimensions={0},
      metadata={op_type="AllGather" op_name="ag0"}
    ROOT %ag-done = f32[16,256,256] all-gather-done(
      (f32[8,256,256], f32[16,256,256]) %ag-start),
      metadata={op_type="AllGather" op_name="ag0"}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(
      HloControlFlowFlattening::Options{/*while_execution_count=*/3});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::CustomCall(op::Parameter(0))));
  EXPECT_EQ(module->entry_computation()->root_instruction()->name(), "ag-done");
  EXPECT_EQ(module->entry_computation()->root_instruction()->operand(0)->name(),
            "ag-start");
}

TEST_F(HloControlFlowFlatteningTest, CollectiveFusion) {
  absl::string_view hlo_template = R"(
HloModule collective-fusion, is_scheduled=true

%sum (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] a, f32[] b)
}

%all-gather {
  %constant.3 = f32[] constant(0)
  %broadcast = f32[full_size,8,128]{2,1,0} broadcast(%constant.3), dimensions={}
  %input.0 = f32[4,8,128]{2,1,0} parameter(0)
  %input.1 = f32[4,8,128]{2,1,0} parameter(1)
  %replica-id.1 = u32[] replica-id()
  %constant.4 = u32[] constant(4)
  %multiply.1 = u32[] multiply(%replica-id.1, %constant.4)
  %constant.5 = u32[] constant(0)
  %constant.6 = u32[] constant(0)
  %dynamic-update-slice = f32[full_size,8,128]{2,1,0} dynamic-update-slice(%broadcast, %input.0, %multiply.1, %constant.5, %constant.6)
  %dynamic-update-slice.1 = f32[full_size,8,128]{2,1,0} dynamic-update-slice(%broadcast, %input.1, %multiply.1, %constant.5, %constant.6)
  %all-reduce = (f32[full_size,8,128]{2,1,0}, f32[full_size,8,128]{2,1,0}) all-reduce(%dynamic-update-slice,  %dynamic-update-slice.1), replica_groups={}, backend_config="{barrier_config:{barrier_type:3,id:0}}", to_apply=%sum
  %gte0 = f32[full_size,8,128]{2,1,0} get-tuple-element(%all-reduce), index=0
  %slice = f32[unpadded_size,8,128]{2,1,0} slice(%gte0), slice={[0:unpadded_size], [0:8], [0:128]}
  %bitcast = f32[unpadded_size,1,8,128]{3,2,1,0} bitcast(%slice)
  %gte1 = f32[full_size,8,128]{2,1,0} get-tuple-element(%all-reduce), index=1
  ROOT %tuple = (f32[unpadded_size,1,8,128]{3,2,1,0}, f32[full_size,8,128]{2,1,0}) tuple(%bitcast, %gte1)
}

ENTRY main {
  %add.1 = f32[4,8,128]{2,1,0} parameter(0)
  %add.2 = f32[4,8,128]{2,1,0} parameter(1)
  ROOT %fusion = (f32[unpadded_size,1,8,128]{3,2,1,0}, f32[full_size,8,128]{2,1,0}) fusion(%add.1, %add.2), kind=kCustom, calls=%all-gather
}
  )";
  auto hlo_string = absl::StrReplaceAll(
      hlo_template, {{"full_size", absl::StrCat(12288)},
                     {"unpadded_size", absl::StrCat(12285)}});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(IsCollective(module->entry_computation()->root_instruction()));

  HloControlFlowFlattening flattening({});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::Parameter(0), op::Parameter(1)));
  EXPECT_EQ(module->entry_computation()->root_instruction()->name(), "fusion");
}

TEST_F(HloControlFlowFlatteningTest, AsyncAllToAll) {
  absl::string_view hlo = R"(

  ENTRY main {
  param = f32[4,8,128]{2,1,0} parameter(0)
  all-to-all-start = ((f32[4,8,128]{2,1,0}), f32[4,8,128]{2,1,0}, u32[], u32[]) all-to-all-start(param), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={1}
  ROOT all-to-all-done = f32[4,8,128]{2,1,0} all-to-all-done(all-to-all-start)
  }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_TRUE(IsCollective(module->entry_computation()->root_instruction()));
  HloControlFlowFlattening flattening({});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::CustomCall(op::Parameter(0))));
}

void CheckWhileBound(HloInstruction* while_op, int expected_bound) {
  auto* cond = while_op->while_condition();
  ASSERT_NE(cond, nullptr);
  auto* hlo_bound = cond->root_instruction()->operand(1);
  EXPECT_TRUE(hlo_bound->IsConstant());
  if (hlo_bound->IsConstant()) {
    EXPECT_TRUE(hlo_bound->literal().IsAll(expected_bound));
  }
}

TEST_F(HloControlFlowFlatteningTest, MaxOuterLoopCount) {
  absl::string_view hlo_string = R"(
  HloModule NestedWhileComp

  InnerBody {
    constant.8 = pred[] constant(false)
    parameter.5 = (s32[], s32[]) parameter(0)
    get-tuple-element.6 = s32[] get-tuple-element(parameter.5), index=0
    constant.9 = s32[] constant(1)
    add.10 = s32[] add(get-tuple-element.6, constant.9)
    get-tuple-element.7 = s32[] get-tuple-element(parameter.5), index=1
    constant.11 = s32[] constant(1)
    add.12 = s32[] add(get-tuple-element.7, constant.11)
    ROOT tuple.13 = (s32[], s32[]) tuple(add.10, add.12)
  }

  InnerCond {
    parameter.15 = (s32[], s32[]) parameter(0)
    get-tuple-element.17 = s32[] get-tuple-element(parameter.15), index=1
    constant.18 = pred[] constant(false)
    get-tuple-element.16 = s32[] get-tuple-element(parameter.15), index=0
    inner_bound = s32[] constant(100)
    ROOT compare.20 = pred[] compare(get-tuple-element.16, inner_bound), direction=LT
  }

  OuterBody {
    constant.24 = pred[] constant(false)
    constant.25 = s32[] constant(0)
    parameter.22 = (s32[]) parameter(0)
    get-tuple-element.23 = s32[] get-tuple-element(parameter.22), index=0
    tuple.26 = (s32[], s32[]) tuple(constant.25, get-tuple-element.23)
    inner_while = (s32[], s32[]) while(tuple.26), condition=InnerCond, body=InnerBody
    get-tuple-element.28 = s32[] get-tuple-element(inner_while), index=0
    get-tuple-element.29 = s32[] get-tuple-element(inner_while), index=1
    tuple.30 = (s32[], s32[]) tuple(get-tuple-element.28, get-tuple-element.29)
    get-tuple-element.31 = s32[] get-tuple-element(tuple.30), index=0
    get-tuple-element.32 = s32[] get-tuple-element(tuple.30), index=1
    ROOT tuple.33 = (s32[]) tuple(get-tuple-element.32)
  }

  OuterCond {
    constant.37 = pred[] constant(false)
    parameter.35 = (s32[]) parameter(0)
    get-tuple-element.36 = s32[] get-tuple-element(parameter.35), index=0
    outer_bound = s32[] constant(1000)
    ROOT compare.39 = pred[] compare(get-tuple-element.36, outer_bound), direction=LT
  }

  ENTRY NestedWhileComp {
    constant.1 = pred[] constant(false)
    constant.2 = s32[] constant(0)
    tuple.3 = (s32[]) tuple(constant.2)
    outer_while = (s32[]) while(tuple.3), condition=OuterCond, body=OuterBody
    get-tuple-element.41 = s32[] get-tuple-element(outer_while), index=0
    tuple.42 = (s32[]) tuple(get-tuple-element.41)
    get-tuple-element.43 = s32[] get-tuple-element(tuple.42), index=0
    ROOT tuple.44 = (s32[]) tuple(get-tuple-element.43)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  constexpr int kWhileExecutionCount = 5;
  constexpr int kExistingInnerLoopCount = 100;
  constexpr int kMaxLoopCount = 10;
  HloControlFlowFlattening flattening(HloControlFlowFlattening::Options{
      /*while_execution_count=*/kWhileExecutionCount,
      /*max_outer_loop_count=*/kMaxLoopCount});
  EXPECT_TRUE(flattening.Run(module.get()).value());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();

  auto* outer_while =
      module->entry_computation()->GetInstructionWithName("outer_while");
  ASSERT_NE(outer_while, nullptr);
  // Checks that the outer while loop has changed its loop bound.
  CheckWhileBound(outer_while, kMaxLoopCount);
  auto* while_body = outer_while->while_body();
  ASSERT_NE(while_body, nullptr);

  auto* inner_while = while_body->GetInstructionWithName("inner_while");
  ASSERT_NE(inner_while, nullptr);
  // Checks that the inner loop bound has not changed.
  CheckWhileBound(inner_while, kExistingInnerLoopCount);
}

TEST_F(HloControlFlowFlatteningTest, MatchLtUseInferedLoopCount) {
  absl::string_view hlo_string = R"(
  HloModule While
  While.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add, multiply)
  }
  While.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(100)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY While {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=While.condition, body=While.body
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  EXPECT_EQ(GetLoopBound(*module->entry_computation()->root_instruction(), 123,
                         kDefaultMaxLoopCount),
            100);
}

TEST_F(HloControlFlowFlatteningTest, MatchGtUseInferedLoopCount) {
  absl::string_view hlo_string = R"(
  HloModule While
  While.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add, multiply)
  }
  While.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(50)
    ROOT greater-than = pred[] compare(constant.2, get-tuple-element.3), direction=GT
  }
  ENTRY While {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=While.condition, body=While.body
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  EXPECT_EQ(GetLoopBound(*module->entry_computation()->root_instruction(), 123,
                         kDefaultMaxLoopCount),
            50);
}

TEST_F(HloControlFlowFlatteningTest, NotMatchEqUseDefaultLoopCount) {
  absl::string_view hlo_string = R"(
  HloModule While
  While.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add, multiply)
  }
  While.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(100)
    ROOT equal = pred[] compare(get-tuple-element.3, constant.2), direction=EQ
  }
  ENTRY While {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=While.condition, body=While.body
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  EXPECT_EQ(GetLoopBound(*module->entry_computation()->root_instruction(), 123,
                         kDefaultMaxLoopCount),
            123);
}

}  // namespace
}  // namespace xla
