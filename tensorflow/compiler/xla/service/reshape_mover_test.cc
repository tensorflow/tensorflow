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

#include "tensorflow/compiler/xla/service/reshape_mover.h"

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_verified_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {
using ReshapeMoverTest = HloVerifiedTestBase;

TEST_F(ReshapeMoverTest, ReshapesWithDifferentInputShapesNotMoved) {
  HloComputation::Builder builder(TestName());
  auto root_shape = ShapeUtil::MakeShape(F32, {8, 7});
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 8, 1, 7}), "param0"));
  auto param1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {1, 8, 7, 1}), "param1"));
  auto reshape0 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param0));
  auto reshape1 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param1));
  builder.AddInstruction(HloInstruction::CreateBinary(
      root_shape, HloOpcode::kAdd, reshape0, reshape1));

  auto computation = module().AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Reshape(param0), op::Reshape(param1)));

  EXPECT_FALSE(ReshapeMover().Run(&module()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Reshape(param0), op::Reshape(param1)));
}

// For a graph that looks like:
//
// +- reshape0 - rng0
// |
// +- const1
// |
// add
//
// where rng0 has a different shape than reshape0.
//
// Verifies that the reshape is not moved, since rng0 is trivially reshapable
// and therefore there is no nontrivial reshapes to move.
TEST_F(ReshapeMoverTest, 1ConstantAnd1ReshapesOnRngNotMoved) {
  HloComputation::Builder builder(TestName());
  auto root_shape = ShapeUtil::MakeShape(F32, {8, 7});
  auto rng0 = builder.AddInstruction(
      HloInstruction::CreateRng(ShapeUtil::MakeShape(F32, {1, 8, 1, 7, 1}),
                                RandomDistribution::RNG_UNIFORM, {}));
  auto reshape0 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, rng0));

  auto const1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(root_shape)));

  builder.AddInstruction(HloInstruction::CreateBinary(
      root_shape, HloOpcode::kAdd, reshape0, const1));

  auto computation = module().AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Reshape(rng0), const1));

  EXPECT_FALSE(ReshapeMover().Run(&module()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Reshape(rng0), const1));
}

TEST_F(ReshapeMoverTest, ScalarReshapesNotMoved) {
  HloComputation::Builder builder(TestName());
  auto root_shape = ShapeUtil::MakeShape(F32, {});
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 1, 1}), "param0"));
  auto param1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {1, 1, 1}), "param1"));
  auto reshape0 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param0));
  auto reshape1 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param1));
  builder.AddInstruction(HloInstruction::CreateBinary(
      root_shape, HloOpcode::kAdd, reshape0, reshape1));

  auto computation = module().AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Reshape(param0), op::Reshape(param1)));

  EXPECT_FALSE(ReshapeMover().Run(&module()).ValueOrDie());

  EXPECT_THAT(
      computation->root_instruction(),
      op::Add(op::Reshape(op::Parameter()), op::Reshape(op::Parameter())));
}

TEST_F(ReshapeMoverTest, EquivalentReshapesMoved) {
  HloComputation::Builder builder(TestName());
  auto root_shape = ShapeUtil::MakeShape(F32, {8, 7});
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 8, 1, 7}), "param0"));
  auto param1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {1, 8, 1, 7}), "param1"));
  auto reshape0 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param0));
  auto reshape1 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param1));
  builder.AddInstruction(HloInstruction::CreateBinary(
      root_shape, HloOpcode::kAdd, reshape0, reshape1));

  auto computation = module().AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Reshape(param0), op::Reshape(param1)));
  EXPECT_TRUE(ReshapeMover().Run(&module()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Reshape(op::Add(param0, param1)));
  EXPECT_EQ(root_shape.DebugString(),
            computation->root_instruction()->shape().DebugString());
}

// For a graph that looks like:
//
// +- reshape2 - param2
// |
// +- reshape1 - param1
// |
// +- constant0
// |
// select
//
// Verifies that the reshape1 and reshape2 sink past select:
//
// +- param2
// |
// +- param1
// |
// +- reshape3(constant0)
// |
// select
// |
// reshape4
TEST_F(ReshapeMoverTest, 1ConstantAnd2ReshapesMoved) {
  HloComputation::Builder builder(TestName());
  auto root_shape = ShapeUtil::MakeShape(F32, {2, 3});
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR2<bool>({{true, true, false}, {false, false, true}})));

  auto param1 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 3, 1, 2}), "param1"));
  auto reshape1 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param1));

  auto param2 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {1, 3, 1, 2}), "param2"));
  auto reshape2 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param2));

  builder.AddInstruction(HloInstruction::CreateTernary(
      root_shape, HloOpcode::kSelect, const0, reshape1, reshape2));

  auto computation = module().AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Select(const0, reshape1, reshape2));

  EXPECT_TRUE(ReshapeMover().Run(&module()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Reshape(op::Select(op::Reshape(const0), param1, param2)));

  EXPECT_EQ(root_shape.DebugString(),
            computation->root_instruction()->shape().DebugString());
}

// For a graph that looks like:
//
// +- reshape0 - param0
// |
// +- param1
// |
// add
//
// Verifies that the reshape0 does not sink below add, because param1 is not
// trivially reshapable nor is a Reshape/Transpose.
TEST_F(ReshapeMoverTest, 1ParameterAnd1ReshapeNotMoved) {
  HloComputation::Builder builder(TestName());
  auto root_shape = ShapeUtil::MakeShape(F32, {8, 7});
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 8, 1, 7}), "param0"));
  auto reshape0 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param0));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, root_shape, "param1"));
  builder.AddInstruction(HloInstruction::CreateBinary(
      root_shape, HloOpcode::kAdd, reshape0, param1));

  auto computation = module().AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Reshape(param0), param1));
  EXPECT_FALSE(ReshapeMover().Run(&module()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Reshape(param0), param1));
  EXPECT_EQ(root_shape.DebugString(),
            computation->root_instruction()->shape().DebugString());
}

// For a graph that looks like:
//
// +- pred
// |
// +- reshape0 - const0
// |
// +- reshape1 - const1
// |
// select
//
// Verifies that we don't unnecessarily sink reshapes, which are in fact
// trivial reshapes.
TEST_F(ReshapeMoverTest, 2TrivialConstantReshapeNotMoved) {
  HloComputation::Builder builder(TestName());
  auto root_shape = ShapeUtil::MakeShape(F32, {3, 2});
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR2<float>({{1, 2, 3}, {4, 5, 6}})));
  auto reshape0 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, const0));

  auto const1 = builder.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR2<float>({{1, 2, 3}, {4, 5, 6}})));
  auto reshape1 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, const1));

  auto pred = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(PRED, {3, 2}), "pred"));

  builder.AddInstruction(HloInstruction::CreateTernary(
      root_shape, HloOpcode::kSelect, pred, reshape0, reshape1));

  auto computation = module().AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Select(pred, op::Reshape(const0), op::Reshape(const1)));

  EXPECT_FALSE(ReshapeMover().Run(&module()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Select(pred, op::Reshape(const0), op::Reshape(const1)));
  EXPECT_EQ(root_shape.DebugString(),
            computation->root_instruction()->shape().DebugString());
}

// For a graph that looks like:
//
// +- reshape0 - param0
// |
// +- const1
// |
// add
//
// where there is only 1 non-trivial reshape (reshape0), we sink the reshape
// here for canonicalization benefit:
//
// +- param0
// |
// +- reshape1 - const1
// |
// add
// |
// reshape2
//
// (note that reshape1 here is trivial).
TEST_F(ReshapeMoverTest, 1NonTrivialReshapeMoved) {
  HloComputation::Builder builder(TestName());
  auto root_shape = ShapeUtil::MakeShape(F32, {2, 3});
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 3, 1, 2}), "param0"));
  auto const1 = builder.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR2<float>({{1, 2, 3}, {4, 5, 6}})));
  auto reshape0 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param0));
  builder.AddInstruction(HloInstruction::CreateBinary(
      root_shape, HloOpcode::kAdd, reshape0, const1));

  auto computation = module().AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Reshape(param0), const1));

  EXPECT_TRUE(ReshapeMover().Run(&module()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Reshape(op::Add(param0, op::Reshape(const1))));
  EXPECT_EQ(root_shape.DebugString(),
            computation->root_instruction()->shape().DebugString());
}

// For a graph that looks like:
//
// +- reshape0 - param0 (shape A)
// |
// +- reshape1 - const1 (shape B)
// |
// add
//
// There is 1 non-trivial reshape (reshape0). It's not clear whether reshape1
// should be trivial or not; conceptually it's trivial, but handling it would
// complicate the rest of our logic.
//
// For now we treat it as non-trivial, so we verify that we don't sink the
// reshapes in this case.
TEST_F(ReshapeMoverTest, 1NonTrivialReshapeWith1ReshapedConstNotMoved) {
  HloComputation::Builder builder(TestName());
  auto root_shape = ShapeUtil::MakeShape(F32, {1, 1, 3});
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 3}), "param0"));
  auto const1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR1<float>({9, 8, 7})));
  auto reshape0 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param0));
  auto reshape1 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, const1));

  builder.AddInstruction(HloInstruction::CreateBinary(
      root_shape, HloOpcode::kAdd, reshape0, reshape1));

  auto computation = module().AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Reshape(param0), op::Reshape(const1)));

  EXPECT_FALSE(ReshapeMover().Run(&module()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Reshape(param0), op::Reshape(const1)));
  EXPECT_EQ(root_shape.DebugString(),
            computation->root_instruction()->shape().DebugString());
}

TEST_F(ReshapeMoverTest, EquivalentReshapesMovedAcrossFusion) {
  HloComputation::Builder builder(TestName());
  auto root_shape = ShapeUtil::MakeShape(F32, {8, 7});
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 8, 1, 7}), "param0"));
  auto param1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {1, 8, 1, 7}), "param1"));
  auto reshape0 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param0));
  auto reshape1 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param1));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      root_shape, HloOpcode::kAdd, reshape0, reshape1));

  auto computation = module().AddEntryComputation(builder.Build());
  computation->CreateFusionInstruction({add},
                                       HloInstruction::FusionKind::kLoop);

  EXPECT_THAT(computation->root_instruction(),
              op::Fusion(op::Reshape(param0), op::Reshape(param1)));

  EXPECT_TRUE(ReshapeMover().Run(&module()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Reshape(op::Fusion(param0, param1)));
  EXPECT_EQ(root_shape.DebugString(),
            computation->root_instruction()->shape().DebugString());
}

TEST_F(ReshapeMoverTest, EquivalentReshapesMovedAcrossSelect) {
  HloComputation::Builder builder(TestName());
  auto root_shape = ShapeUtil::MakeShape(F32, {8, 7});
  auto pred_shape = ShapeUtil::MakeShape(PRED, {8, 7});
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 8, 1, 7}), "param0"));
  auto param1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {1, 8, 1, 7}), "param1"));
  auto pred = builder.AddInstruction(HloInstruction::CreateParameter(
      2, ShapeUtil::MakeShape(PRED, {1, 8, 1, 7}), "pred"));
  auto reshape0 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param0));
  auto reshape1 =
      builder.AddInstruction(HloInstruction::CreateReshape(root_shape, param1));
  auto reshape_pred =
      builder.AddInstruction(HloInstruction::CreateReshape(pred_shape, pred));
  builder.AddInstruction(HloInstruction::CreateTernary(
      root_shape, HloOpcode::kSelect, reshape_pred, reshape0, reshape1));

  auto computation = module().AddEntryComputation(builder.Build());

  EXPECT_THAT(
      computation->root_instruction(),
      op::Select(op::Reshape(pred), op::Reshape(param0), op::Reshape(param1)));

  EXPECT_TRUE(ReshapeMover().Run(&module()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Reshape(op::Select(pred, param0, param1)));
  EXPECT_EQ(root_shape.DebugString(),
            computation->root_instruction()->shape().DebugString());
}

TEST_F(ReshapeMoverTest, ScalarReshapeNotMovedAcrossSelect) {
  HloComputation::Builder builder(TestName());
  auto root_shape = ShapeUtil::MakeShape(F32, {});
  auto pred_shape = ShapeUtil::MakeShape(PRED, {});
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {}), "param0"));
  auto param1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {}), "param1"));
  auto pred = builder.AddInstruction(HloInstruction::CreateParameter(
      2, ShapeUtil::MakeShape(PRED, {1, 1, 1}), "pred"));
  auto reshape_pred =
      builder.AddInstruction(HloInstruction::CreateReshape(pred_shape, pred));
  auto select = builder.AddInstruction(HloInstruction::CreateTernary(
      root_shape, HloOpcode::kSelect, reshape_pred, param0, param1));

  auto computation = module().AddEntryComputation(builder.Build());
  EXPECT_THAT(computation->root_instruction(),
              op::Select(op::Reshape(pred), param0, param1));

  EXPECT_FALSE(ReshapeMover().Run(&module()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Select(op::Reshape(pred), param0, param1));
  EXPECT_EQ(select, computation->root_instruction());
}

// Tree looks like:
//
// param0 [1,128,1]
//  |
// reshape [128,1]          constant [128,1024]
//   \                         /
//     multiply w/implicit broadcast [128,1024]
//
// The reshape mover would like to sink the reshape below the multiply.
//
// Previously we would attempt to insert a reshape of the constant to [1,128,1]
// (which is unsound, because it has a different number of elements) as
// preparation for sinking the reshape.
//
// To eliminate the unsoundness, we outlaw reshape sinking when one of the
// operands is implicitly broadcast in the elementwise consumer.
//
// TODO(b/37799338) However, it would be possible in this case to do a more
// in-depth analysis to get reshape movement to occur:
//
// 1. Note that the broadcast dimension (logical dimension 1) in the operands
//    would map back to logical dimension 2 in the param0 node.
// 2. Match rank of the constant to the param0 node (by prepending a trivial 1
//    dimension).
// 3. Reshape to [128,1024] at the root.
//
// But this is not currently done.
TEST_F(ReshapeMoverTest, ImplicitlyBroadcastReshapeIsNotMovedBug37787999) {
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 128, 1}), "param0"));
  auto reshape = builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {128, 1}), param0));
  Array2D<float> a(128, 1024);
  auto literal = Literal::CreateR2FromArray2D<float>(a);
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));
  auto multiply = builder.AddInstruction(HloInstruction::CreateBinary(
      constant->shape(), HloOpcode::kMultiply, constant, reshape));

  auto computation = module().AddEntryComputation(builder.Build());
  EXPECT_THAT(computation->root_instruction(),
              op::Multiply(op::Constant(), op::Reshape(param0)));

  EXPECT_FALSE(ReshapeMover().Run(&module()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Multiply(op::Constant(), op::Reshape(param0)));
  EXPECT_EQ(multiply, computation->root_instruction());
}

// Tree looks like this:
//
// add1
// |
// +- reshape2 - param2
// |
// +- reshape3 - add0
//               |
//               + reshape0 - param0
//               |
//               + reshape1 - param1
//
// We expect reshape{0,1} AND reshape{2,3} to be lifted.
TEST_F(ReshapeMoverTest, MultiplePasses) {
  auto shape1 = ShapeUtil::MakeShape(F32, {1, 8, 1, 7});
  auto shape2 = ShapeUtil::MakeShape(F32, {8, 7, 1});
  auto shape3 = ShapeUtil::MakeShape(F32, {8, 7});
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape1, "param0"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape1, "param1"));
  auto param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, shape2, "param2"));
  auto reshape0 =
      builder.AddInstruction(HloInstruction::CreateReshape(shape2, param0));
  auto reshape1 =
      builder.AddInstruction(HloInstruction::CreateReshape(shape2, param1));
  auto add0 = builder.AddInstruction(HloInstruction::CreateBinary(
      shape2, HloOpcode::kAdd, reshape0, reshape1));
  auto reshape2 =
      builder.AddInstruction(HloInstruction::CreateReshape(shape3, param2));
  auto reshape3 =
      builder.AddInstruction(HloInstruction::CreateReshape(shape3, add0));
  builder.AddInstruction(HloInstruction::CreateBinary(shape3, HloOpcode::kAdd,
                                                      reshape2, reshape3));

  auto computation = module().AddEntryComputation(builder.Build());

  EXPECT_THAT(
      computation->root_instruction(),
      op::Add(op::Reshape(param2),
              op::Reshape(op::Add(op::Reshape(param0), op::Reshape(param1)))));

  EXPECT_TRUE(ReshapeMover().Run(&module()).ValueOrDie());

  EXPECT_THAT(
      computation->root_instruction(),
      op::Reshape(op::Add(param2, op::Reshape(op::Add(param0, param1)))));
}

TEST_F(ReshapeMoverTest, SinkTransposeAcrossBroadcastScalar) {
  const string hlo_string = R"(
    HloModule TransposeMulInversedTransposeModule
    ENTRY TransposeMulInversedTranspose {
      src0 = f32[20,8]{1,0} parameter(0)
      transpose0 = f32[8,20]{1,0} transpose(src0), dimensions={1,0}
      src1 = f32[] parameter(1)
      broadcast0 = f32[8,20]{1,0} broadcast(src1), dimensions={}
      ROOT multiply0 = f32[8,20]{1,0} multiply(transpose0, broadcast0)
    }
  )";

  ParseAndVerifyModule(hlo_string);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, ReshapeMover().Run(&module()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(module().entry_computation()->root_instruction(),
              op::Transpose(op::Multiply()));
}

TEST_F(ReshapeMoverTest, ReshapeWithUsersOutsideCandidatesNotSink) {
  const string hlo_string = R"(
    HloModule ReshapeWithUsersOutsideCandidates
    ENTRY ReshapeWithMultipleUsers {
      param0 = f32[20,8]{1,0} parameter(0)
      reshape0 = f32[8,20]{1,0} reshape(param0)
      param1 = f32[] parameter(1)
      broadcast0 = f32[8,20]{1,0} broadcast(param1), dimensions={}
      param2 = f32[20,8]{1,0} parameter(2)
      reshape1 = f32[8,20]{1,0} reshape(param2)
      param3 = f32[20,8]{1,0} parameter(3)
      reshape2 = f32[8,20]{1,0} reshape(param3)
      param4 = f32[8,20]{1,0} parameter(4)
      add0 = f32[8,20]{1,0} add(reshape0, broadcast0)
      add1 = f32[8,20]{1,0} add(reshape0, reshape1)
      add2 = f32[8,20]{1,0} add(reshape1, param4)
      ROOT tuple = (f32[8,20]{1,0},f32[8,20]{1,0},
        f32[8,20]{1,0}) tuple(add0, add1, add2)
    }
  )";

  ParseAndVerifyModule(hlo_string);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, ReshapeMover().Run(&module()));
  EXPECT_FALSE(changed);
}

TEST_F(ReshapeMoverTest, ReshapeNoUsersOutsideCandidatesSink1) {
  const string hlo_string = R"(
    HloModule ReshapeNoUsersOutsideCandidates1
    ENTRY ReshapeWithMultipleUsers1 {
      param0 = f32[20,8]{1,0} parameter(0)
      reshape0 = f32[8,20]{1,0} reshape(param0)
      param1 = f32[] parameter(1)
      broadcast0 = f32[8,20]{1,0} broadcast(param1), dimensions={}
      param2 = f32[20,8]{1,0} parameter(2)
      reshape1 = f32[8,20]{1,0} reshape(param2)
      param3 = f32[20,8]{1,0} parameter(3)
      reshape2 = f32[8,20]{1,0} reshape(param3)
      add0 = f32[8,20]{1,0} add(reshape0, broadcast0)
      add1 = f32[8,20]{1,0} add(reshape0, reshape1)
      add2 = f32[8,20]{1,0} add(reshape1, reshape2)
      ROOT tuple = (f32[8,20]{1,0},f32[8,20]{1,0},
        f32[8,20]{1,0}) tuple(add0, add1, add2)
    }
  )";

  ParseAndVerifyModule(hlo_string);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, ReshapeMover().Run(&module()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module().entry_computation()->root_instruction(),
              op::Tuple(op::Reshape(), op::Reshape(), op::Reshape()));
}

TEST_F(ReshapeMoverTest, ReshapeNoUsersOutsideCandidatesSink2) {
  const string hlo_string = R"(
    HloModule ReshapeNoUsersOutsideCandidates2
    ENTRY ReshapeWithMultipleUsers2 {
      param0 = f32[20,8]{1,0} parameter(0)
      reshape0 = f32[8,20]{1,0} reshape(param0)
      ROOT add0 = f32[8,20]{1,0} add(reshape0, reshape0)
    }
  )";

  ParseAndVerifyModule(hlo_string);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, ReshapeMover().Run(&module()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module().entry_computation()->root_instruction(),
              op::Reshape(op::Add()));
}

}  // namespace
}  // namespace xla
