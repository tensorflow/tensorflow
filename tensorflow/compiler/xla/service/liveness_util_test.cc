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

#include "tensorflow/compiler/xla/service/liveness_util.h"

#include <memory>

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

class PointsToAnalysisTestBase : public HloTestBase {
 protected:
  void BuildModule(std::unique_ptr<HloComputation> computation) {
    module_ = CreateNewModule();
    computation_ = module_->AddEntryComputation(std::move(computation));
  }

  void RunAnalysis() {
    CHECK_NOTNULL(module_.get());
    points_to_analysis_ =
        TuplePointsToAnalysis::Run(module_.get()).ConsumeValueOrDie();
    dataflow_analysis_ = HloDataflowAnalysis::Run(*module_).ConsumeValueOrDie();
  }

  void BuildModuleAndRunAnalysis(std::unique_ptr<HloComputation> computation) {
    BuildModule(std::move(computation));
    RunAnalysis();
  }

  std::unique_ptr<HloModule> module_;
  HloComputation* computation_ = nullptr;
  std::unique_ptr<TuplePointsToAnalysis> points_to_analysis_;
  std::unique_ptr<HloDataflowAnalysis> dataflow_analysis_;
};

class DoesNotUseOperandBufferTest : public PointsToAnalysisTestBase {};

TEST_F(DoesNotUseOperandBufferTest, GetTupleElement) {
  auto builder = HloComputation::Builder(TestName());

  Shape elem_shape = ShapeUtil::MakeShape(F32, {8});
  auto tuple = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeTupleShape({elem_shape, elem_shape}), "tuple"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(elem_shape, tuple, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(elem_shape, tuple, 1));
  builder.AddInstruction(
      HloInstruction::CreateBinary(elem_shape, HloOpcode::kAdd, gte0, gte1));

  BuildModuleAndRunAnalysis(builder.Build());

  // GetTupleElement instructions only access the top-level buffer of their
  // operand.
  EXPECT_TRUE(DoesNotUseOperandBuffer(tuple, {0}, gte0, *points_to_analysis_));
  EXPECT_TRUE(DoesNotUseOperandBuffer(tuple, {1}, gte1, *points_to_analysis_));
  EXPECT_FALSE(DoesNotUseOperandBuffer(tuple, {}, gte0, *points_to_analysis_));
  EXPECT_FALSE(DoesNotUseOperandBuffer(tuple, {}, gte1, *points_to_analysis_));

  EXPECT_TRUE(DoesNotUseOperandBuffer(tuple, {0}, gte0, *dataflow_analysis_));
  EXPECT_TRUE(DoesNotUseOperandBuffer(tuple, {1}, gte1, *dataflow_analysis_));
  EXPECT_FALSE(DoesNotUseOperandBuffer(tuple, {}, gte0, *dataflow_analysis_));
  EXPECT_FALSE(DoesNotUseOperandBuffer(tuple, {}, gte1, *dataflow_analysis_));
}

TEST_F(DoesNotUseOperandBufferTest, FusedDynamicUpdateSlice) {
  auto builder = HloComputation::Builder(TestName());

  Shape data_shape = ShapeUtil::MakeShape(F32, {8});
  auto tuple = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeTupleShape({data_shape, data_shape}), "tuple"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape, tuple, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape, tuple, 1));

  // Create a DynamicUpdateSlice instruction of tuple element 1.
  auto starts = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR1<int32>({2})));
  auto update = builder.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR1<float>({2.f, 2.f, 2.f})));
  auto dynamic_update_slice =
      builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
          data_shape, gte1, update, starts));
  builder.AddInstruction(
      HloInstruction::CreateTuple({gte0, dynamic_update_slice}));

  BuildModule(builder.Build());
  auto fusion = computation_->CreateFusionInstruction(
      {dynamic_update_slice, starts, update, gte1},
      HloInstruction::FusionKind::kLoop);
  RunAnalysis();

  // The fusion instruction never uses tuple element 0, but does use element 1.
  EXPECT_TRUE(
      DoesNotUseOperandBuffer(tuple, {0}, fusion, *points_to_analysis_));
  EXPECT_FALSE(
      DoesNotUseOperandBuffer(tuple, {1}, fusion, *points_to_analysis_));

  EXPECT_TRUE(DoesNotUseOperandBuffer(tuple, {0}, fusion, *dataflow_analysis_));
  EXPECT_FALSE(
      DoesNotUseOperandBuffer(tuple, {1}, fusion, *dataflow_analysis_));
}

class CanShareOperandBufferWithUserTest : public PointsToAnalysisTestBase {};

TEST_F(CanShareOperandBufferWithUserTest, ElementWiseSameShape) {
  auto builder = HloComputation::Builder(TestName());

  Shape shape = ShapeUtil::MakeShape(F32, {8});
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kExp, param));
  auto log = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kLog, exp));

  BuildModuleAndRunAnalysis(builder.Build());

  EXPECT_TRUE(
      CanShareOperandBufferWithUser(param, {}, exp, {}, *points_to_analysis_));
  EXPECT_TRUE(
      CanShareOperandBufferWithUser(exp, {}, log, {}, *points_to_analysis_));

  EXPECT_TRUE(
      CanShareOperandBufferWithUser(param, {}, exp, {}, *dataflow_analysis_));
  EXPECT_TRUE(
      CanShareOperandBufferWithUser(exp, {}, log, {}, *dataflow_analysis_));
}

TEST_F(CanShareOperandBufferWithUserTest, ElementWiseDifferentShape) {
  auto builder = HloComputation::Builder(TestName());

  Shape in_shape = ShapeUtil::MakeShape(F32, {8});
  Shape out_shape = ShapeUtil::MakeShape(PRED, {8});
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, in_shape, "param0"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, in_shape, "param1"));
  auto result = builder.AddInstruction(
      HloInstruction::CreateBinary(out_shape, HloOpcode::kEq, param0, param1));

  BuildModuleAndRunAnalysis(builder.Build());

  EXPECT_FALSE(CanShareOperandBufferWithUser(param0, {}, result, {},
                                             *points_to_analysis_));
  EXPECT_FALSE(CanShareOperandBufferWithUser(param1, {}, result, {},
                                             *points_to_analysis_));

  EXPECT_FALSE(CanShareOperandBufferWithUser(param0, {}, result, {},
                                             *dataflow_analysis_));
  EXPECT_FALSE(CanShareOperandBufferWithUser(param1, {}, result, {},
                                             *dataflow_analysis_));
}

TEST_F(CanShareOperandBufferWithUserTest, CopyShares) {
  auto builder = HloComputation::Builder(TestName());

  Shape shape = ShapeUtil::MakeShape(F32, {8});
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kExp, param));
  auto copy = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCopy, exp));

  BuildModuleAndRunAnalysis(builder.Build());

  EXPECT_TRUE(
      CanShareOperandBufferWithUser(param, {}, exp, {}, *points_to_analysis_));
  EXPECT_TRUE(
      CanShareOperandBufferWithUser(exp, {}, copy, {}, *points_to_analysis_));

  EXPECT_TRUE(
      CanShareOperandBufferWithUser(param, {}, exp, {}, *dataflow_analysis_));
  EXPECT_TRUE(
      CanShareOperandBufferWithUser(exp, {}, copy, {}, *dataflow_analysis_));
}

TEST_F(CanShareOperandBufferWithUserTest, FusedDynamicUpdateSlice) {
  auto builder = HloComputation::Builder(TestName());

  Shape data_shape = ShapeUtil::MakeShape(F32, {8});
  auto tuple = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeTupleShape({data_shape, data_shape}), "tuple"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape, tuple, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape, tuple, 1));

  // Create a DynamicUpdateSlice instruction of tuple element 1.
  auto starts = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR1<int32>({2})));
  auto update = builder.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR1<float>({2.f, 2.f, 2.f})));
  auto dynamic_update_slice =
      builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
          data_shape, gte1, update, starts));
  builder.AddInstruction(
      HloInstruction::CreateTuple({gte0, dynamic_update_slice}));

  BuildModule(builder.Build());
  auto fusion = computation_->CreateFusionInstruction(
      {dynamic_update_slice, starts, update, gte1},
      HloInstruction::FusionKind::kLoop);
  RunAnalysis();

  // The fusion instruction can share with tuple element 1.
  EXPECT_FALSE(CanShareOperandBufferWithUser(tuple, {0}, fusion, {},
                                             *points_to_analysis_));
  EXPECT_TRUE(CanShareOperandBufferWithUser(tuple, {1}, fusion, {},
                                            *points_to_analysis_));

  EXPECT_FALSE(CanShareOperandBufferWithUser(tuple, {0}, fusion, {},
                                             *dataflow_analysis_));
  EXPECT_TRUE(CanShareOperandBufferWithUser(tuple, {1}, fusion, {},
                                            *dataflow_analysis_));
}

TEST_F(CanShareOperandBufferWithUserTest, DynamicUpdateSliceCanShare) {
  auto builder = HloComputation::Builder(TestName());

  Shape data_shape = ShapeUtil::MakeShape(F32, {8});
  Shape update_shape = ShapeUtil::MakeShape(F32, {4});
  Shape starts_shape = ShapeUtil::MakeShape(S32, {1});
  auto data = builder.AddInstruction(
      HloInstruction::CreateParameter(0, data_shape, "data"));
  auto update = builder.AddInstruction(
      HloInstruction::CreateParameter(1, update_shape, "update"));
  auto starts = builder.AddInstruction(
      HloInstruction::CreateParameter(2, starts_shape, "starts"));
  auto dus = builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
      data_shape, data, update, starts));

  BuildModuleAndRunAnalysis(builder.Build());

  // The DynamicUpdateSlice instruction can share with the data operand, but not
  // with update or starts.
  EXPECT_TRUE(
      CanShareOperandBufferWithUser(data, {}, dus, {}, *points_to_analysis_));
  EXPECT_FALSE(
      CanShareOperandBufferWithUser(update, {}, dus, {}, *points_to_analysis_));
  EXPECT_FALSE(
      CanShareOperandBufferWithUser(starts, {}, dus, {}, *points_to_analysis_));

  EXPECT_TRUE(
      CanShareOperandBufferWithUser(data, {}, dus, {}, *dataflow_analysis_));
  EXPECT_FALSE(
      CanShareOperandBufferWithUser(update, {}, dus, {}, *dataflow_analysis_));
  EXPECT_FALSE(
      CanShareOperandBufferWithUser(starts, {}, dus, {}, *dataflow_analysis_));
}

TEST_F(CanShareOperandBufferWithUserTest, FusedDotAdd) {
  auto builder = HloComputation::Builder(TestName());
  Shape data_shape = ShapeUtil::MakeShape(F32, {2, 2});

  auto a = builder.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR2<float>({{1.0, 0.0}, {0.0, 1.0}})));
  auto b = builder.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR2<float>({{2.0, 2.0}, {2.0, 2.0}})));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(
      HloInstruction::CreateDot(data_shape, a, b, dot_dnums));

  auto one = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto add_operand = builder.AddInstruction(
      HloInstruction::CreateBroadcast(data_shape, one, {1}));

  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      data_shape, HloOpcode::kAdd, dot, add_operand));

  BuildModule(builder.Build());
  auto fusion = computation_->CreateFusionInstruction(
      {add, dot}, HloInstruction::FusionKind::kOutput);
  RunAnalysis();

  // Output fused dot add should be able to share buffer with 'add_operand'.
  EXPECT_TRUE(CanShareOperandBufferWithUser(add_operand, {}, fusion, {},
                                            *points_to_analysis_));

  EXPECT_TRUE(CanShareOperandBufferWithUser(add_operand, {}, fusion, {},
                                            *dataflow_analysis_));
}

TEST_F(CanShareOperandBufferWithUserTest, OutputFusionCantAliasOperandBuffer) {
  auto builder = HloComputation::Builder(TestName());
  Shape data_shape = ShapeUtil::MakeShape(F32, {2, 2});

  auto one = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto operand = builder.AddInstruction(
      HloInstruction::CreateBroadcast(data_shape, one, {1}));

  auto reverse = builder.AddInstruction(
      HloInstruction::CreateReverse(data_shape, operand, {0, 1}));

  auto two = builder.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR2<float>({{2.0, 2.0}, {2.0, 2.0}})));

  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(data_shape, HloOpcode::kAdd, reverse, two));

  BuildModule(builder.Build());
  auto fusion = computation_->CreateFusionInstruction(
      {add, two, reverse}, HloInstruction::FusionKind::kOutput);
  RunAnalysis();

  // Output fused operand->reverse->add cannot alias operand buffer 'operand'.
  EXPECT_FALSE(CanShareOperandBufferWithUser(operand, {}, fusion, {},
                                             *points_to_analysis_));

  EXPECT_FALSE(CanShareOperandBufferWithUser(operand, {}, fusion, {},
                                             *dataflow_analysis_));
}

TEST_F(CanShareOperandBufferWithUserTest, WhileCanShare) {
  Shape data_shape = ShapeUtil::MakeShape(F32, {8});

  auto make_cond = [this, &data_shape]() {
    auto builder = HloComputation::Builder(TestName() + ".Cond");
    auto data = builder.AddInstruction(
        HloInstruction::CreateParameter(0, data_shape, "data"));
    builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kEq, data, data));
    return builder.Build();
  };

  auto make_body = [this, &data_shape]() {
    auto builder = HloComputation::Builder(TestName() + ".Body");
    auto data = builder.AddInstruction(
        HloInstruction::CreateParameter(0, data_shape, "data"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(data_shape, HloOpcode::kAdd, data, data));
    return builder.Build();
  };

  module_ = CreateNewModule();
  HloComputation* cond_computation =
      module_->AddEmbeddedComputation(make_cond());
  HloComputation* body_computation =
      module_->AddEmbeddedComputation(make_body());

  auto builder = HloComputation::Builder(TestName());
  auto data = builder.AddInstruction(
      HloInstruction::CreateParameter(0, data_shape, "data"));
  auto whil = builder.AddInstruction(HloInstruction::CreateWhile(
      data_shape, cond_computation, body_computation, data));
  computation_ = module_->AddEntryComputation(builder.Build());

  RunAnalysis();

  // The While instruction can share with the data operand.
  EXPECT_TRUE(
      CanShareOperandBufferWithUser(data, {}, whil, {}, *points_to_analysis_));

  EXPECT_TRUE(
      CanShareOperandBufferWithUser(data, {}, whil, {}, *dataflow_analysis_));
}

// Tests that Call can alias operand buffer if the only use of the operand
// in the called computation is an elementwise instruction.
TEST_F(CanShareOperandBufferWithUserTest, CallToComputationWithFusionRoot) {
  Shape shape = ShapeUtil::MakeShape(F32, {8});
  // Build sub-computation with fusion root.
  auto sub_builder = HloComputation::Builder(TestName() + "_sub");
  auto sub_param = sub_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "sub_param"));
  auto one = sub_builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto ones = sub_builder.AddInstruction(
      HloInstruction::CreateBroadcast(shape, one, {1}));
  auto add = sub_builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, sub_param, ones));

  module_ = CreateNewModule();
  auto sub_computation = module_->AddEmbeddedComputation(sub_builder.Build());
  sub_computation->CreateFusionInstruction({add, ones},
                                           HloInstruction::FusionKind::kLoop);

  // Build entry-computation with kCall which calls 'sub_computation'.
  auto builder = HloComputation::Builder(TestName());

  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto reverse =
      builder.AddInstruction(HloInstruction::CreateReverse(shape, param, {0}));
  auto call = builder.AddInstruction(
      HloInstruction::CreateCall(shape, {reverse}, sub_computation));
  computation_ = module_->AddEntryComputation(builder.Build());

  RunAnalysis();

  EXPECT_TRUE(CanShareOperandBufferWithUser(reverse, {}, call, {},
                                            *points_to_analysis_));
  EXPECT_TRUE(CanShareOperandBufferWithUser(reverse, {}, call, {},
                                            *dataflow_analysis_));
}

}  // namespace
}  // namespace xla
