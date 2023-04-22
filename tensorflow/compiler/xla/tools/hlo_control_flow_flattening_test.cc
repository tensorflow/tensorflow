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

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

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
  EXPECT_EQ(HloVerifier(/*layout_sensitive=*/true,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status(),
            Status::OK());

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
  EXPECT_EQ(HloVerifier(/*layout_sensitive=*/true,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status(),
            Status::OK());

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
  EXPECT_EQ(HloVerifier(/*layout_sensitive=*/true,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status(),
            Status::OK());

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
  EXPECT_EQ(HloVerifier(/*layout_sensitive=*/true,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status(),
            Status::OK());
  auto tuple = module->entry_computation()->root_instruction();
  EXPECT_THAT(tuple, op::Tuple(op::Tuple(op::Constant(), op::Constant()),
                               op::AfterAll()));
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
  EXPECT_EQ(HloVerifier(/*layout_sensitive=*/true,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status(),
            Status::OK());
  auto tuple = module->entry_computation()->root_instruction();
  EXPECT_THAT(tuple, op::Tuple(op::Tuple(op::Constant(), op::Constant()),
                               op::AfterAll()));
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
  EXPECT_EQ(HloVerifier(/*layout_sensitive=*/true,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status(),
            Status::OK());
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
  EXPECT_EQ(HloVerifier(/*layout_sensitive=*/true,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status(),
            Status::OK());
  LOG(INFO) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::Constant(), op::Constant()));
  HloInstruction* param0 =
      module->entry_computation()->parameter_instruction(0);
  EXPECT_EQ(param0->user_count(), 1);
  EXPECT_THAT(param0->users().at(0),
              op::CustomCall(op::Parameter(0), op::Parameter(1)));
}

}  // namespace
}  // namespace xla
