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

#include "tensorflow/compiler/xla/tools/hlo_control_flow_flattening.h"

#include "tensorflow/compiler/xla/service/despecializer.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

using HloControlFlowFlatteningTest = HloTestBase;

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
  HloControlFlowFlattening flattening(3);
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
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
  HloControlFlowFlattening flattening(3);
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
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
  HloControlFlowFlattening flattening(3);
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
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
    ROOT infeed = ((bf16[3]{0}, s32[12,5]{0,1}), token[]) infeed(after-all)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(3);
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  auto tuple = module->entry_computation()->root_instruction();
  EXPECT_THAT(tuple, op::Tuple(op::CustomCall(), op::AfterAll()));
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
  HloControlFlowFlattening flattening(3);
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  auto tuple = module->entry_computation()->root_instruction();
  EXPECT_THAT(tuple, op::Tuple(op::CustomCall(), op::AfterAll()));
  EXPECT_EQ(tuple->shape(), root_shape);
}

TEST_F(HloControlFlowFlatteningTest, Outfeed) {
  absl::string_view hlo_string = R"(
  HloModule Outfeed
  ENTRY Outfeed {
    param = (bf16[3]{0}, s32[12,5]{0,1}) parameter(0)
    after-all = token[] after-all()
    ROOT outfeed = token[] outfeed(param, after-all)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(3);
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  auto custom_call = module->entry_computation()->root_instruction();
  EXPECT_THAT(custom_call, op::CustomCall(op::Parameter(0), op::AfterAll()));
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
  HloControlFlowFlattening flattening(3);
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::Parameter(0), op::Parameter(1)));
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
    crs = (f32[8]{0}, f32[8]{0}) all-reduce-start(input), replica_groups={}, to_apply=add
    ROOT done = f32[8]{0} all-reduce-done(crs)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(3);
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::CustomCall(op::Parameter(0))));
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
  HloControlFlowFlattening flattening(3);
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::Parameter(0)));
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
  HloControlFlowFlattening flattening(3);
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::Parameter(0)));
}

TEST_F(HloControlFlowFlatteningTest, CollectivePermute) {
  absl::string_view hlo_string = R"(
  HloModule CollectivePermute

  ENTRY CollectivePermute {
    input = f32[128,32]{0,1} parameter(0)
    ROOT root = f32[128,32]{0,1} collective-permute(input), source_target_pairs={{0,1},{1,2},{2,3}}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(3);
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::Parameter(0)));
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
    ROOT root = f32[128,128]{0,1} collective-permute(input, output, tuple.1, tuple.2), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{128,32}}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloControlFlowFlattening flattening(3);
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::Parameter(0), op::Broadcast(op::Constant()),
                             op::Tuple(op::Constant(), op::Constant()),
                             op::Tuple(op::Constant(), op::Constant())));
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
  HloControlFlowFlattening flattening(3);
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::CustomCall(op::Parameter(0))));
}

TEST_F(HloControlFlowFlatteningTest, Recv) {
  absl::string_view hlo_string = R"(
  HloModule Recv

  ENTRY %Recv () -> (f32[], token[]) {
    %token0 = token[] after-all()
    %recv = (f32[], u32[], token[]) recv(token[] %token0), channel_id=15, sharding={maximal device=1}
    ROOT %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv), channel_id=15, sharding={maximal device=1}
    %constant = f32[] constant(2.1), sharding={maximal device=0}
    %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0), channel_id=16, sharding={maximal device=0}, control-predecessors={%recv}
    %send-done = token[] send-done((f32[], u32[], token[]) %send), channel_id=16, sharding={maximal device=0}  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ControlDepRemover control_remover;
  HloControlFlowFlattening flattening(3);
  TF_ASSERT_OK(control_remover.Run(module.get()).status());
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::CustomCall(), op::AfterAll()));
}

TEST_F(HloControlFlowFlatteningTest, Send) {
  absl::string_view hlo_string = R"(
  HloModule Send

  ENTRY %Send () -> token[] {
    %token0 = token[] after-all()
    %recv = (f32[], u32[], token[]) recv(token[] %token0), channel_id=15, sharding={maximal device=1}
    %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv), channel_id=15, sharding={maximal device=1}
    %constant = f32[] constant(2.1), sharding={maximal device=0}
    %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0), channel_id=16, sharding={maximal device=0}, control-predecessors={%recv}
    ROOT %send-done = token[] send-done((f32[], u32[], token[]) %send), channel_id=16, sharding={maximal device=0}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ControlDepRemover control_remover;
  HloControlFlowFlattening flattening(3);
  TF_ASSERT_OK(control_remover.Run(module.get()).status());
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::Constant(), op::AfterAll()));
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
  HloControlFlowFlattening flattening(3);
  EXPECT_TRUE(flattening.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/true)
                   .Run(module.get())
                   .status());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::CustomCall(op::CustomCall(op::Parameter(0))));
}

}  // namespace
}  // namespace xla
