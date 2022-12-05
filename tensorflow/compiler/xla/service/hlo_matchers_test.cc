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
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace op = xla::testing::opcode_matchers;
using ::testing::_;
using ::testing::Eq;

namespace xla {
namespace {

using HloMatchersTest = HloTestBase;

std::string DescribeHloMatcher(
    const ::testing::Matcher<const HloInstruction*>& m) {
  std::stringstream ss;
  m.DescribeTo(&ss);
  return ss.str();
}

template <typename M, typename T>
std::string Explain(const T& t, const M& m) {
  ::testing::StringMatchResultListener listener;
  EXPECT_THAT(t, ::testing::Not(m));  // For the error message.
  EXPECT_FALSE(m.MatchAndExplain(t, &listener));
  return listener.str();
}

TEST_F(HloMatchersTest, Test) {
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
  EXPECT_THAT(
      Explain(add.get(), op::Parameter()),
      Eq("(%add = f32[1]{0} add(f32[1]{0} %param, f32[1]{0} %multiply))"));
  EXPECT_THAT(
      Explain(add.get(), op::Add(op::Parameter())),
      Eq("(%add = f32[1]{0} add(f32[1]{0} %param, f32[1]{0} %multiply)) "
         "has too many operands (got 2, want 1)"));
  EXPECT_THAT(
      Explain(add.get(), op::Add(op::Parameter(), op::Parameter())),
      Eq("(%add = f32[1]{0} add(f32[1]{0} %param, f32[1]{0} %multiply))"
         "\noperand 1:\n\t"
         "%multiply = f32[1]{0} multiply(f32[1]{0} %param, f32[1]{0} %param)\n"
         "doesn't match expected:\n\t"
         "parameter"
         ", (%multiply = f32[1]{0} multiply(f32[1]{0} %param, f32[1]{0} "
         "%param))"));
  EXPECT_THAT(
      Explain(add.get(),
              op::Add(op::Parameter(), op::Multiply(op::Add(), op::Add()))),
      Eq("(%add = f32[1]{0} add(f32[1]{0} %param, f32[1]{0} %multiply))"
         "\noperand 1:\n\t"
         "%multiply = f32[1]{0} multiply(f32[1]{0} %param, f32[1]{0} %param)\n"
         "doesn't match expected:\n\t"
         "multiply(add, add)"
         ", (%multiply = f32[1]{0} multiply(f32[1]{0} %param, f32[1]{0} "
         "%param))\n"
         "operand 0:\n\t"
         "%param = f32[1]{0} parameter(0)\n"
         "doesn't match expected:\n\t"
         "add, (%param = f32[1]{0} parameter(0))"));
}

TEST_F(HloMatchersTest, CustomCallMatcher) {
  auto c1 =
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({1, 2, 3}));
  auto c2 =
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int32_t>({1, 2, 3}));
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
              "(%custom-call = f32[1]{0} custom-call(f32[3]{0} %constant, "
              "s32[3]{0} %constant), custom_call_target=\"foo_target\") "
              "custom-call with call target that isn't equal to \"bar\"");
  EXPECT_THAT(DescribeHloMatcher(op::CustomCall("foo_target")),
              R"(custom-call with call target that is equal to "foo_target")");
}

TEST_F(HloMatchersTest, ShapeMatcher) {
  auto p0 = HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShapeWithDenseLayout(F32, {5, 7}, {0, 1}), "param");

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
  EXPECT_THAT(p0.get(), op::Shape(ShapeUtil::MakeShapeWithDenseLayout(
                            F32, {5, 7}, {0, 1})));
  EXPECT_THAT(p0.get(), op::Shape("f32[5,7]{0,1}"));
  EXPECT_THAT(p0.get(), op::ShapeWithLayout(ShapeUtil::MakeShapeWithDenseLayout(
                            F32, {5, 7}, {0, 1})));
  EXPECT_THAT(p0.get(), op::ShapeWithLayout("f32[5,7]{0,1}"));
  EXPECT_THAT(p0.get(),
              ::testing::Not(op::ShapeWithLayout(
                  ShapeUtil::MakeShapeWithDenseLayout(F32, {5, 7}, {1, 0}))));
  EXPECT_THAT(p0.get(), ::testing::Not(op::ShapeWithLayout("f32[5,7]{1,0}")));

  EXPECT_THAT(Explain(p0.get(), op::Shape(ShapeUtil::MakeShape(F32, {7, 5}))),
              "%param = f32[5,7]{0,1} parameter(0) has incorrect shape "
              "(expected: f32[7,5])");
  EXPECT_THAT(
      Explain(p0.get(), op::ShapeWithLayout(ShapeUtil::MakeShapeWithDenseLayout(
                            F32, {7, 5}, {1, 0}))),
      "%param = f32[5,7]{0,1} parameter(0) has incorrect shape "
      "(expected: f32[7,5]{1,0})");
}

TEST_F(HloMatchersTest, ShardingMatcher) {
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
  Array<int64_t> assignment({2});
  assignment.SetValues({0, 1});
  auto sharding = HloSharding::Tuple(
      tuple_shape, {HloSharding::Tile(assignment), HloSharding::AssignDevice(1),
                    HloSharding::Replicate()});
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
      op::Sharding("{{devices=[2]0,1}, {maximal device=1}, {replicated}}"));

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

TEST_F(HloMatchersTest, DotMatcher) {
  std::string hlo_string = R"(
HloModule DotOperationFusion_TransposeFusion

ENTRY DotOperationFusion_TransposeFusion {
  arg0 = f32[1,256] parameter(0)
  arg1 = f32[256,1024] parameter(1)
  ROOT dot = f32[1,1024] dot(arg0, arg1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();

  EXPECT_THAT(root, op::Dot(op::Parameter(0), op::Parameter(1),
                            /*lhs_contracting_dim=*/1,
                            /*rhs_contracting_dim=*/0));

  EXPECT_THAT(
      Explain(root, op::Dot(op::Parameter(0), op::Parameter(1),
                            /*lhs_contracting_dim=*/0,
                            /*rhs_contracting_dim=*/0)),
      "(%dot = f32[1,1024]{1,0} dot(f32[1,256]{1,0} %arg0, f32[256,1024]{1,0} "
      "%arg1), lhs_contracting_dims={1}, rhs_contracting_dims={0}) has wrong "
      "lhs_contracting_dimensions (got {1} want {0})");

  EXPECT_THAT(
      Explain(root, op::Dot(op::Parameter(0), op::Parameter(1),
                            /*lhs_contracting_dim=*/1,
                            /*rhs_contracting_dim=*/1)),
      "(%dot = f32[1,1024]{1,0} dot(f32[1,256]{1,0} %arg0, f32[256,1024]{1,0} "
      "%arg1), lhs_contracting_dims={1}, rhs_contracting_dims={0}) has wrong "
      "rhs_contracting_dimensions (got {0} want {1})");
}

TEST_F(HloMatchersTest, ComparisonMatcher) {
  auto shape = ShapeUtil::MakeShape(F32, {1});
  auto p0 = HloInstruction::CreateParameter(0, shape, "param.0");
  auto p1 = HloInstruction::CreateParameter(1, shape, "param.1");
  auto eq = HloInstruction::CreateCompare(shape, p0.get(), p1.get(),
                                          ComparisonDirection::kEq);
  auto ne = HloInstruction::CreateCompare(shape, p0.get(), p1.get(),
                                          ComparisonDirection::kNe);
  auto add =
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p0.get(), p1.get());
  auto le = HloInstruction::CreateCompare(shape, p0.get(), add.get(),
                                          ComparisonDirection::kLe);

  EXPECT_THAT(eq.get(), op::Compare());
  EXPECT_THAT(eq.get(), op::Eq());
  EXPECT_THAT(ne.get(), op::Compare());
  EXPECT_THAT(ne.get(), op::Ne());
  EXPECT_THAT(le.get(),
              op::Compare(op::Parameter(0),
                          op::Add(op::Parameter(0), op::Parameter(1))));
  EXPECT_THAT(le.get(), op::Le(op::Parameter(0),
                               op::Add(op::Parameter(0), op::Parameter(1))));

  EXPECT_THAT(Explain(eq.get(), op::Add()),
              Eq("(%compare = f32[1]{0} compare(f32[1]{0} %param.0, "
                 "f32[1]{0} %param.1), direction=EQ)"));
  EXPECT_THAT(Explain(eq.get(), op::Ne()),
              Eq("(%compare = f32[1]{0} compare(f32[1]{0} %param.0, "
                 "f32[1]{0} %param.1), direction=EQ) "
                 "has wrong comparison direction (got EQ, want NE)"));
}

TEST_F(HloMatchersTest, AsyncCopyMatcher) {
  Shape shape_memspace1 = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {16}, /*minor_to_major=*/{0}, /*tiles=*/{},
      /*memory_space=*/1);
  Shape shape_memspace2 = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {16}, /*minor_to_major=*/{0}, /*tiles=*/{},
      /*memory_space=*/2);

  auto p0 = HloInstruction::CreateParameter(0, shape_memspace1, "p0");
  auto copy_start = HloInstruction::CreateCopyStart(
      ShapeUtil::MakeTupleShape(
          {shape_memspace2, shape_memspace1, ShapeUtil::MakeShape(U32, {})}),
      p0.get());
  auto copy_done = HloInstruction::CreateUnary(
      shape_memspace2, HloOpcode::kCopyDone, copy_start.get());

  EXPECT_THAT(copy_done.get(), op::AsyncCopy(2, 1, op::Parameter(0)));

  EXPECT_THAT(Explain(copy_start.get(), op::AsyncCopy(2, 1, op::Parameter(0))),
              Eq("(%copy-start = (f32[16]{0:S(2)}, f32[16]{0:S(1)}, u32[]) "
                 "copy-start(f32[16]{0:S(1)} %p0))"));
  EXPECT_THAT(Explain(copy_done.get(), op::AsyncCopy(3, 1, op::Parameter(0))),
              "(%copy-done = f32[16]{0:S(2)} copy-done((f32[16]{0:S(2)}, "
              "f32[16]{0:S(1)}, u32[]) "
              "%copy-start)) "
              "copies to memory space 2, expected 3");
  EXPECT_THAT(Explain(copy_done.get(), op::AsyncCopy(2, 3, op::Parameter(0))),
              "(%copy-done = f32[16]{0:S(2)} copy-done((f32[16]{0:S(2)}, "
              "f32[16]{0:S(1)}, u32[]) "
              "%copy-start)) "
              "is in the memory space 1, expected 3");
}

TEST_F(HloMatchersTest, ConstantMatcher) {
  std::string hlo_string = R"(
HloModule Constant

ENTRY main {
  ROOT x = u32[2] constant({1, 2})
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();

  EXPECT_THAT(root, op::Constant());
  EXPECT_THAT(root, op::Constant(LiteralUtil::CreateR1<uint32_t>({1, 2})));
  EXPECT_THAT(root, ::testing::Not(
                        op::Constant(LiteralUtil::CreateR1<uint32_t>({1, 1}))));

  EXPECT_THAT(Explain(root, op::Constant(LiteralUtil::CreateR0<uint32_t>(1))),
              "(%x = u32[2]{0} constant({1, 2})) has wrong value (got u32[2] "
              "{1, 2}, want u32[] 1)");
}

TEST_F(HloMatchersTest, ReplicaGroupsMatcher) {
  Shape shape = ShapeUtil::MakeShape(F32, {5, 7});
  std::unique_ptr<HloInstruction> p0 =
      HloInstruction::CreateParameter(0, shape, "param");

  std::vector<ReplicaGroup> replica_groups(2);
  replica_groups[0].add_replica_ids(0);
  replica_groups[0].add_replica_ids(2);
  replica_groups[1].add_replica_ids(1);
  replica_groups[1].add_replica_ids(3);
  std::unique_ptr<HloInstruction> all_to_all =
      HloInstruction::CreateAllToAll(shape, {p0.get()}, replica_groups,
                                     /*constrain_layout=*/false,
                                     /*channel_id=*/std::nullopt);

  EXPECT_THAT(Explain(p0.get(), op::ReplicaGroups({})),
              "%param = f32[5,7]{1,0} parameter(0) not a collective op");
  EXPECT_THAT(Explain(all_to_all.get(), op::ReplicaGroups({{0, 1}, {2, 3}})),
              "%all-to-all = f32[5,7]{1,0} all-to-all(f32[5,7]{1,0} %param), "
              "replica_groups={{0,2},{1,3}} has incorrect replica_groups "
              "(expected: {{0,1},{2,3}})");
  EXPECT_THAT(all_to_all.get(), op::ReplicaGroups({{0, 2}, {1, 3}}));
}

}  // namespace
}  // namespace xla
