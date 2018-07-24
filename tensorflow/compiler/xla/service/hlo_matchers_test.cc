/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace op = xla::testing::opcode_matchers;
using ::testing::_;
using ::testing::Eq;

namespace xla {
namespace {

string DescribeHloMatcher(const ::testing::Matcher<const HloInstruction*>& m) {
  std::stringstream ss;
  m.DescribeTo(&ss);
  return ss.str();
}

template <typename M, typename T>
string Explain(const T& t, const M& m) {
  ::testing::StringMatchResultListener listener;
  EXPECT_THAT(t, ::testing::Not(m));  // For the error message.
  EXPECT_FALSE(m.MatchAndExplain(t, &listener));
  return listener.str();
}

TEST(HloMatchersTest, Test) {
  auto shape = ShapeUtil::MakeShape(F32, {1});
  auto param = HloInstruction::CreateParameter(0, shape, "param");
  auto mul = HloInstruction::CreateBinary(shape, HloOpcode::kMultiply,
                                          param.get(), param.get());
  auto add = HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param.get(),
                                          mul.get());

  EXPECT_THAT(add.get(), op::Add());
  EXPECT_THAT(add.get(), op::Add(op::Parameter(), op::Multiply()));
  EXPECT_THAT(add.get(),
              op::Add(op::Parameter(), op::Multiply(_, op::Parameter())));

  // Negative matches: check the explanation string.
  EXPECT_THAT(Explain(add.get(), op::Parameter()), Eq(""));
  EXPECT_THAT(Explain(add.get(), op::Add(op::Parameter())),
              Eq("has too many operands (got 2, want 1)"));
  EXPECT_THAT(
      Explain(add.get(), op::Add(op::Parameter(), op::Parameter())),
      Eq("\noperand 1:\n\t"
         "%multiply = f32[1]{0} multiply(f32[1]{0} %param, f32[1]{0} %param)\n"
         "doesn't match expected:\n\t"
         "parameter"));
  EXPECT_THAT(
      Explain(add.get(),
              op::Add(op::Parameter(), op::Multiply(op::Add(), op::Add()))),
      Eq("\noperand 1:\n\t"
         "%multiply = f32[1]{0} multiply(f32[1]{0} %param, f32[1]{0} %param)\n"
         "doesn't match expected:\n\t"
         "multiply(add, add), \n"
         "operand 0:\n\t"
         "%param = f32[1]{0} parameter(0)\n"
         "doesn't match expected:\n\t"
         "add"));
}

TEST(HloMatchersTest, CustomCallMatcher) {
  auto c1 =
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({1, 2, 3}));
  auto c2 =
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int32>({1, 2, 3}));
  auto call = HloInstruction::CreateCustomCall(
      ShapeUtil::MakeShape(F32, {1}), {c1.get(), c2.get()}, "foo_target");

  EXPECT_THAT(call.get(), op::CustomCall());
  EXPECT_THAT(call.get(), op::CustomCall(c1.get(), c2.get()));
  EXPECT_THAT(call.get(), op::CustomCall("foo_target"));
  EXPECT_THAT(call.get(), op::CustomCall("foo_target", c1.get(), c2.get()));
  EXPECT_THAT(call.get(), op::CustomCall(::testing::StartsWith("foo")));
  EXPECT_THAT(call.get(),
              op::CustomCall(::testing::Not(::testing::StartsWith("bar"))));

  // Wrong number of operands.
  EXPECT_THAT(call.get(), ::testing::Not(op::CustomCall(c1.get())));

  // Call target does not match.
  EXPECT_THAT(call.get(),
              ::testing::Not(op::CustomCall(::testing::StartsWith("bar"))));

  EXPECT_THAT(Explain(call.get(), op::CustomCall("bar")),
              R"(custom-call with call target that isn't equal to "bar")");
  EXPECT_THAT(DescribeHloMatcher(op::CustomCall("foo_target")),
              R"(custom-call with call target that is equal to "foo_target")");
}

TEST(HloMatchersTest, ShapeMatcher) {
  auto p0 = HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShapeWithLayout(F32, {5, 7}, {0, 1}), "param");

  EXPECT_THAT(p0.get(), op::Shape(ShapeUtil::MakeShape(F32, {5, 7})));
  EXPECT_THAT(p0.get(), op::Shape("f32[5,7]"));
  EXPECT_THAT(
      p0.get(),
      ::testing::Not(op::ShapeWithLayout(ShapeUtil::MakeShape(F32, {5, 7}))));
  EXPECT_THAT(p0.get(), ::testing::Not(op::ShapeWithLayout("f32[5,7]")));
  EXPECT_THAT(p0.get(),
              ::testing::Not(op::Shape(ShapeUtil::MakeShape(F32, {7, 5}))));
  EXPECT_THAT(p0.get(), ::testing::Not(op::Shape("f32[7,5]")));
  EXPECT_THAT(
      p0.get(),
      ::testing::Not(op::ShapeWithLayout(ShapeUtil::MakeShape(F32, {7, 5}))));
  EXPECT_THAT(p0.get(), ::testing::Not(op::ShapeWithLayout("f32[7,5]")));
  EXPECT_THAT(p0.get(),
              op::Shape(ShapeUtil::MakeShapeWithLayout(F32, {5, 7}, {0, 1})));
  EXPECT_THAT(p0.get(), op::Shape("f32[5,7]{0,1}"));
  EXPECT_THAT(p0.get(), op::ShapeWithLayout(ShapeUtil::MakeShapeWithLayout(
                            F32, {5, 7}, {0, 1})));
  EXPECT_THAT(p0.get(), op::ShapeWithLayout("f32[5,7]{0,1}"));
  EXPECT_THAT(p0.get(),
              ::testing::Not(op::ShapeWithLayout(
                  ShapeUtil::MakeShapeWithLayout(F32, {5, 7}, {1, 0}))));
  EXPECT_THAT(p0.get(), ::testing::Not(op::ShapeWithLayout("f32[5,7]{1,0}")));

  EXPECT_THAT(Explain(p0.get(), op::Shape(ShapeUtil::MakeShape(F32, {7, 5}))),
              "%param = f32[5,7]{0,1} parameter(0) has incorrect shape "
              "(expected: f32[7,5])");
  EXPECT_THAT(
      Explain(p0.get(), op::ShapeWithLayout(ShapeUtil::MakeShapeWithLayout(
                            F32, {7, 5}, {1, 0}))),
      "%param = f32[5,7]{0,1} parameter(0) has incorrect shape "
      "(expected: f32[7,5]{1,0})");
}

TEST(HloMatchersTest, ShardingMatcher) {
  auto p0 = HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {5}),
                                            "param.0");
  p0->clear_sharding();
  auto p1 = HloInstruction::CreateParameter(1, ShapeUtil::MakeShape(F32, {7}),
                                            "param.1");
  p1->set_sharding(HloSharding::AssignDevice(1));

  auto tuple_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {7}), ShapeUtil::MakeShape(S32, {9}),
       ShapeUtil::MakeShape(F32, {11})});
  auto p2 = HloInstruction::CreateParameter(1, tuple_shape, "param.2");
  Array<int64> assignment({2});
  assignment.SetValues({0, 1});
  auto sharding = HloSharding::Tuple(
      tuple_shape,
      {HloSharding::Tile(ShapeUtil::MakeShape(F32, {5}), assignment),
       HloSharding::AssignDevice(1), HloSharding::Replicate()});
  p2->set_sharding(sharding);

  EXPECT_THAT(p0.get(), op::NoSharding());
  EXPECT_THAT(p0.get(),
              ::testing::Not(op::Sharding(HloSharding::AssignDevice(1))));
  EXPECT_THAT(p1.get(), ::testing::Not(op::NoSharding()));
  EXPECT_THAT(p1.get(),
              ::testing::Not(op::Sharding(HloSharding::AssignDevice(0))));
  EXPECT_THAT(p1.get(), op::Sharding(HloSharding::AssignDevice(1)));

  EXPECT_THAT(
      p2.get(),
      op::Sharding(
          "{{f32[5] devices=[2]0,1}, {maximal device=1}, {replicated}}"));

  EXPECT_THAT(Explain(p0.get(), op::Sharding(HloSharding::AssignDevice(1))),
              "%param.0 = f32[5]{0} parameter(0) has no sharding (expected: "
              "{maximal device=1})");
  EXPECT_THAT(Explain(p1.get(), op::NoSharding()),
              "%param.1 = f32[7]{0} parameter(1), sharding={maximal device=1} "
              "expected to have no sharding.");
  EXPECT_THAT(Explain(p1.get(), op::Sharding(HloSharding::AssignDevice(0))),
              "%param.1 = f32[7]{0} parameter(1), sharding={maximal device=1} "
              "has incorrect sharding (expected: {maximal device=0})");
}

TEST(HloMatchersTest, DotMatcher) {
  string hlo_string = R"(
HloModule DotOperationFusion_TransposeFusion

ENTRY DotOperationFusion_TransposeFusion {
  arg0 = f32[1,256] parameter(0)
  arg1 = f32[256,1024] parameter(1)
  ROOT dot = f32[1,1024] dot(arg0, arg1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();

  EXPECT_THAT(root, op::Dot(op::Parameter(0), op::Parameter(1),
                            /*lhs_contracting_dim=*/1,
                            /*rhs_contracting_dim=*/0));

  EXPECT_THAT(
      Explain(root, op::Dot(op::Parameter(0), op::Parameter(1),
                            /*lhs_contracting_dim=*/0,
                            /*rhs_contracting_dim=*/0)),
      "%dot = f32[1,1024]{1,0} dot(f32[1,256]{1,0} %arg0, f32[256,1024]{1,0} "
      "%arg1), lhs_contracting_dims={1}, rhs_contracting_dims={0} has wrong "
      "lhs_contracting_dimensions (got {1} want {0})");

  EXPECT_THAT(
      Explain(root, op::Dot(op::Parameter(0), op::Parameter(1),
                            /*lhs_contracting_dim=*/1,
                            /*rhs_contracting_dim=*/1)),
      "%dot = f32[1,1024]{1,0} dot(f32[1,256]{1,0} %arg0, f32[256,1024]{1,0} "
      "%arg1), lhs_contracting_dims={1}, rhs_contracting_dims={0} has wrong "
      "rhs_contracting_dimensions (got {0} want {1})");
}

}  // namespace
}  // namespace xla
