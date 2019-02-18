/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/root_token_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using RootTokenReplacerTest = HloTestBase;

TEST_F(RootTokenReplacerTest, TestTokenControlDependencyAdded) {
  auto builder = HloComputation::Builder(TestName());
  auto input1_literal = xla::LiteralUtil::CreateR1<float>({1.1f, 1.1f});
  auto shape = input1_literal.shape();

  auto i1 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "i1"));
  auto i2 = builder.AddInstruction(HloInstruction::CreateToken());
  auto computation = builder.Build();
  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(std::move(computation));
  auto original_root = hlo_module->entry_computation()->root_instruction();
  EXPECT_TRUE(
      hlo_module->entry_computation()->root_instruction()->shape().IsToken());

  RootTokenReplacer replacer;
  EXPECT_TRUE(replacer.Run(hlo_module.get()).ValueOrDie());

  const Shape& new_shape = hlo_module->result_shape();
  EXPECT_TRUE(ShapeUtil::IsEmptyTuple(new_shape));
  auto new_root = hlo_module->entry_computation()->root_instruction();
  auto control_predecessors = new_root->control_predecessors();
  EXPECT_EQ(control_predecessors.size(), 1);
  EXPECT_EQ(control_predecessors[0], original_root);
}

TEST_F(RootTokenReplacerTest, TestTokenInTupleWithScalar) {
  auto hlo_module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  const auto empty_tuple_shape = ShapeUtil::MakeNil();
  const auto token_shape = ShapeUtil::MakeTokenShape();
  const auto scalar_shape = ShapeUtil::MakeShape(S32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape, token_shape});

  auto i1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "i1"));
  auto i2 = builder.AddInstruction(HloInstruction::CreateToken());

  builder.AddInstruction(HloInstruction::CreateTuple({i1, i2}));
  auto computation = builder.Build();
  hlo_module->AddEntryComputation(std::move(computation));

  RootTokenReplacer replacer;
  EXPECT_TRUE(replacer.Run(hlo_module.get()).ValueOrDie());
  const Shape& new_shape = hlo_module->result_shape();
  EXPECT_TRUE(new_shape.IsTuple());
  const auto num_shapes = ShapeUtil::TupleElementCount(new_shape);
  EXPECT_EQ(num_shapes, 1);
  const auto& element_shape = ShapeUtil::GetTupleElementShape(new_shape, 0);
  EXPECT_TRUE(ShapeUtil::Equal(element_shape, scalar_shape));
}

TEST_F(RootTokenReplacerTest, TestAllTokensInTuple) {
  auto hlo_module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  const auto empty_tuple_shape = ShapeUtil::MakeNil();
  const auto token_shape = ShapeUtil::MakeTokenShape();
  Shape tuple_shape =
      ShapeUtil::MakeTupleShape({token_shape, token_shape, token_shape});

  auto i1 = builder.AddInstruction(HloInstruction::CreateToken());
  auto i2 = builder.AddInstruction(HloInstruction::CreateToken());
  auto i3 = builder.AddInstruction(HloInstruction::CreateToken());

  builder.AddInstruction(HloInstruction::CreateTuple({i1, i2, i3}));
  auto computation = builder.Build();
  hlo_module->AddEntryComputation(std::move(computation));

  RootTokenReplacer replacer;
  EXPECT_TRUE(replacer.Run(hlo_module.get()).ValueOrDie());
  const Shape& new_shape = hlo_module->result_shape();
  EXPECT_TRUE(new_shape.IsTuple());
  const auto num_shapes = ShapeUtil::TupleElementCount(new_shape);
  EXPECT_EQ(num_shapes, 0);
  EXPECT_TRUE(ShapeUtil::Equal(new_shape, empty_tuple_shape));
}

TEST_F(RootTokenReplacerTest, TestSingleTokenInTuple) {
  auto hlo_module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  const auto empty_tuple_shape = ShapeUtil::MakeNil();
  const auto token_shape = ShapeUtil::MakeTokenShape();
  Shape tuple_shape = ShapeUtil::MakeTupleShape({token_shape});

  auto i1 = builder.AddInstruction(HloInstruction::CreateToken());

  builder.AddInstruction(HloInstruction::CreateTuple({i1}));
  auto computation = builder.Build();
  hlo_module->AddEntryComputation(std::move(computation));

  RootTokenReplacer replacer;
  EXPECT_TRUE(replacer.Run(hlo_module.get()).ValueOrDie());
  const Shape& new_shape = hlo_module->result_shape();
  EXPECT_TRUE(new_shape.IsTuple());
  const auto num_shapes = ShapeUtil::TupleElementCount(new_shape);
  EXPECT_EQ(num_shapes, 0);
}

TEST_F(RootTokenReplacerTest, TestNestedTupleFailure) {
  auto hlo_module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  const auto empty_tuple_shape = ShapeUtil::MakeNil();
  const auto token_shape = ShapeUtil::MakeTokenShape();
  Shape inner_tuple_shape = ShapeUtil::MakeTupleShape({token_shape});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({inner_tuple_shape});

  auto i1 = builder.AddInstruction(HloInstruction::CreateToken());
  auto inner_tuple = builder.AddInstruction(HloInstruction::CreateTuple({i1}));
  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({inner_tuple}));

  auto computation = builder.Build();
  hlo_module->AddEntryComputation(std::move(computation));

  RootTokenReplacer replacer;
  auto status = replacer.Run(hlo_module.get());
  EXPECT_FALSE(status.ok());
}

TEST_F(RootTokenReplacerTest, TokenInWhileLoop) {
  auto hlo_module = CreateNewVerifiedModule();
  const auto empty_tuple_shape = ShapeUtil::MakeNil();
  const auto token_shape = ShapeUtil::MakeTokenShape();
  const auto scalar_shape = ShapeUtil::MakeShape(S32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape});

  HloComputation* comp_cond;
  {
    auto builder_cond = HloComputation::Builder(TestName());
    auto tuple = builder_cond.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "cond_tuple"));
    auto limit0 = builder_cond.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(10)));
    auto c1 = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(S32, {}), tuple, 0));
    auto lt1 = builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, c1, limit0));

    comp_cond = hlo_module->AddEmbeddedComputation(builder_cond.Build());
  }

  /* Create while body */
  HloComputation* comp_body;
  {
    auto builder_body = HloComputation::Builder(TestName());
    auto tuple = builder_body.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "body_tuple"));
    auto c = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape, tuple, 0));
    auto one = builder_body.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));
    auto new_c = builder_body.AddInstruction(
        HloInstruction::CreateBinary(c->shape(), HloOpcode::kAdd, c, one));
    auto token = builder_body.AddInstruction(HloInstruction::CreateToken());
    builder_body.AddInstruction(HloInstruction::CreateTuple({new_c, token}));
    comp_body = hlo_module->AddEmbeddedComputation(builder_body.Build());
  }

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto c = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));

  auto init = builder_main.AddInstruction(HloInstruction::CreateTuple({c}));
  auto main = builder_main.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, comp_cond, comp_body, init));
  hlo_module->AddEntryComputation(builder_main.Build());

  RootTokenReplacer replacer;
  replacer.Run(hlo_module.get()).ValueOrDie();

  const Shape& new_shape = hlo_module->result_shape();
  EXPECT_TRUE(new_shape.IsTuple());
  EXPECT_FALSE(ShapeUtil::IsNestedTuple(new_shape));
  const auto num_shapes = ShapeUtil::TupleElementCount(new_shape);
  EXPECT_EQ(num_shapes, 1);
  auto flattened_shapes = FlattenedXlaShape(new_shape);
  EXPECT_EQ(flattened_shapes.size(), 1);
  EXPECT_TRUE(ShapeUtil::Equal(scalar_shape, flattened_shapes[0]));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
