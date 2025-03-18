/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/dynamic_dimension_inference.h"

#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal.h"
#include "xla/service/hlo_runner.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test_benchmark.h"

namespace xla {
namespace {

class DynamicDimensionInferenceTest : public HloTestBase {
 protected:
  DynamicDimensionInferenceTest() : HloTestBase() {
    module_ = CreateNewVerifiedModule();
  }

  absl::Status RunInference(
      OpSupportsDynamismHandler op_supports_dynamism_handler = nullptr,
      DynamicDimensionInference::CustomCallInferenceHandler handler = nullptr,
      DynamicDimensionInference::ShapeCheckMode shape_check_mode =
          DynamicDimensionInference::ShapeCheckMode::kIgnore,
      const DynamicDimensionInference::AssertionGenerator& assertion_generator =
          nullptr) {
    TF_ASSIGN_OR_RETURN(DynamicDimensionInference inference,
                        DynamicDimensionInference::Run(
                            module_.get(), op_supports_dynamism_handler,
                            handler, shape_check_mode, assertion_generator));

    inference_ = std::make_unique<DynamicDimensionInference>(inference);
    return absl::OkStatus();
  }

  HloComputation* GetAdd() {
    auto embedded_builder = HloComputation::Builder("add");
    auto lhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
        0, ShapeUtil::MakeShape(F32, {}), "lhs"));
    auto rhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
        1, ShapeUtil::MakeShape(F32, {}), "rhs"));
    embedded_builder.AddInstruction(
        HloInstruction::CreateBinary(lhs->shape(), HloOpcode::kAdd, lhs, rhs));
    return module_->AddEmbeddedComputation(embedded_builder.Build());
  }

  HloComputation* GetAddTuple() {
    auto embedded_builder = HloComputation::Builder("add");
    auto lhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
        0, ShapeUtil::MakeShape(F32, {}), "lhs"));
    auto lhs_1 =
        embedded_builder.AddInstruction(HloInstruction::CreateParameter(
            1, ShapeUtil::MakeShape(F32, {}), "lhs.1"));
    auto rhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
        2, ShapeUtil::MakeShape(F32, {}), "rhs"));
    auto rhs_1 =
        embedded_builder.AddInstruction(HloInstruction::CreateParameter(
            3, ShapeUtil::MakeShape(F32, {}), "rhs.1"));
    auto add = embedded_builder.AddInstruction(
        HloInstruction::CreateBinary(lhs->shape(), HloOpcode::kAdd, lhs, rhs));
    auto add_1 = embedded_builder.AddInstruction(HloInstruction::CreateBinary(
        lhs->shape(), HloOpcode::kAdd, lhs_1, rhs_1));
    embedded_builder.AddInstruction(HloInstruction::CreateTuple({add, add_1}));
    return module_->AddEmbeddedComputation(embedded_builder.Build());
  }

  HloComputation* GetGe() {
    auto embedded_builder = HloComputation::Builder("ge");
    auto lhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
        0, ShapeUtil::MakeShape(F32, {}), "lhs"));
    auto rhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
        1, ShapeUtil::MakeShape(F32, {}), "rhs"));
    embedded_builder.AddInstruction(HloInstruction::CreateCompare(
        ShapeUtil::MakeShape(PRED, {}), lhs, rhs, ComparisonDirection::kGe));
    return module_->AddEmbeddedComputation(embedded_builder.Build());
  }

  std::unique_ptr<HloModule> module_;
  std::unique_ptr<DynamicDimensionInference> inference_;
  const Shape scalar_shape_ = ShapeUtil::MakeShape(S32, {});
};

TEST_F(DynamicDimensionInferenceTest, ParamTest) {
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
  auto dynamic_shape =
      ShapeUtil::MakeShape(F32, {1, 2, 2}, {false, true, false});

  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "param"));
  auto param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "param"));
  auto result = builder.AddInstruction(
      HloInstruction::CreateSetDimensionSize(dynamic_shape, param, param2, 1));

  module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());

  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(result, {}, 1), param2);
  EXPECT_EQ(inference_->GetDynamicSize(param, {}, 0), nullptr);
  EXPECT_EQ(inference_->GetDynamicSize(param2, {}, 0), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, ElementwiseTest) {
  // When data flows through elementwise, the dynamic dimension size keeps the
  // same.
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
  auto dynamic_shape =
      ShapeUtil::MakeShape(F32, {1, 2, 2}, {false, true, false});

  auto data_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "data_param"));
  auto size_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "size_param"));
  auto dynamic_param =
      builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
          dynamic_shape, data_param, size_param, 1));

  auto* negate = builder.AddInstruction(HloInstruction::CreateUnary(
      dynamic_shape, HloOpcode::kNegate, dynamic_param));

  module_->AddEntryComputation(builder.Build());

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(negate, {}, 1), size_param);
}

TEST_F(DynamicDimensionInferenceTest, ReduceTestI) {
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
  auto reduce_shape = ShapeUtil::MakeShape(F32, {2}, {true});
  auto dynamic_shape =
      ShapeUtil::MakeShape(F32, {1, 2, 2}, {false, true, false});

  auto data_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "data_param"));
  auto size_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "size_param"));
  auto dynamic_param =
      builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
          dynamic_shape, data_param, size_param, 1));

  auto negate = builder.AddInstruction(HloInstruction::CreateUnary(
      dynamic_shape, HloOpcode::kNegate, dynamic_param));

  auto init = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0)));

  auto reduce = builder.AddInstruction(HloInstruction::CreateReduce(
      reduce_shape, negate, init, {0, 2}, GetAdd()));

  module_->AddEntryComputation(builder.Build());

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(reduce, {}, 0), size_param);
}

TEST_F(DynamicDimensionInferenceTest, ReduceTestII) {
  // Same as ReduceTestI, but only reduce one dimension.
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
  auto reduce_shape = ShapeUtil::MakeShape(F32, {1, 2}, {false, true});
  auto dynamic_shape =
      ShapeUtil::MakeShape(F32, {1, 2, 2}, {false, false, true});

  auto data_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "data_param"));
  auto size_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "size_param"));
  auto dynamic_param =
      builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
          dynamic_shape, data_param, size_param, 2));

  auto negate = builder.AddInstruction(HloInstruction::CreateUnary(
      dynamic_shape, HloOpcode::kNegate, dynamic_param));

  auto init = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0)));

  auto reduce = builder.AddInstruction(
      HloInstruction::CreateReduce(reduce_shape, negate, init, {1}, GetAdd()));

  module_->AddEntryComputation(builder.Build());

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(reduce, {}, 1), size_param);
  EXPECT_EQ(inference_->GetDynamicSize(reduce, {}, 0), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, VariadicReduce) {
  // Handle variadic reduce where output is a tuple.
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
  auto reduce_shape = ShapeUtil::MakeShape(F32, {1, 2}, {false, true});
  auto dynamic_shape =
      ShapeUtil::MakeShape(F32, {1, 2, 2}, {false, false, true});

  auto data_param_1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "data_param"));
  auto data_param_2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, input_shape, "data_param.2"));
  auto size_param = builder.AddInstruction(
      HloInstruction::CreateParameter(2, scalar_shape_, "size_param"));
  auto data_param_dynamic_1 =
      builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
          dynamic_shape, data_param_1, size_param, 2));
  auto data_param_dynamic_2 =
      builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
          dynamic_shape, data_param_2, size_param, 2));

  auto dynamic_negate_1 = builder.AddInstruction(HloInstruction::CreateUnary(
      dynamic_shape, HloOpcode::kNegate, data_param_dynamic_1));

  auto dynamic_negate_2 = builder.AddInstruction(HloInstruction::CreateUnary(
      dynamic_shape, HloOpcode::kNegate, data_param_dynamic_2));

  auto init = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0)));

  auto reduce = builder.AddInstruction(HloInstruction::CreateReduce(
      ShapeUtil::MakeTupleShape({reduce_shape, reduce_shape}),
      {dynamic_negate_1, dynamic_negate_2}, {init, init}, {1}, GetAddTuple()));

  module_->AddEntryComputation(builder.Build());

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(reduce, {0}, 1), size_param);
  EXPECT_EQ(inference_->GetDynamicSize(reduce, {1}, 1), size_param);
  EXPECT_EQ(inference_->GetDynamicSize(reduce, {0}, 0), nullptr);
  EXPECT_EQ(inference_->GetDynamicSize(reduce, {1}, 0), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, DotTest) {
  auto builder = HloComputation::Builder(TestName());
  constexpr int xdim = 3;
  constexpr int ydim = 2;
  constexpr int zdim = 1;
  auto xy_shape = ShapeUtil::MakeShape(F32, {xdim, ydim});
  auto yz_shape = ShapeUtil::MakeShape(F32, {ydim, zdim});
  auto xy_dynamic_shape = ShapeUtil::MakeShape(F32, {xdim, ydim}, {true, true});
  auto yz_dynamic_shape =
      ShapeUtil::MakeShape(F32, {ydim, zdim}, {true, false});
  auto xz_dynamic_shape =
      ShapeUtil::MakeShape(F32, {xdim, zdim}, {true, false});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, xy_shape, "A"));
  auto* b_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, yz_shape, "B"));
  auto* size_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, scalar_shape_, "size_param"));

  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      ShapeUtil::MakeShape(F32, xy_shape.dimensions(), {true, false}), a_param,
      size_param, 0));
  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      xy_dynamic_shape, a_param, size_param, 1));
  b_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      yz_dynamic_shape, b_param, size_param, 0));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(
      HloInstruction::CreateDot(xz_dynamic_shape, a_param, b_param, dot_dnums,
                                HloTestBase::DefaultPrecisionConfig(2)));

  module_->AddEntryComputation(builder.Build());

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(dot, {}, 0), size_param);
  EXPECT_EQ(inference_->GetDynamicSize(dot, {}, 1), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, DotTestBatch) {
  auto builder = HloComputation::Builder(TestName());
  auto lhs_shape = ShapeUtil::MakeShape(F32, {4, 128, 2, 8});
  auto rhs_shape = ShapeUtil::MakeShape(F32, {4, 128, 2, 8});
  auto output_shape =
      ShapeUtil::MakeShape(F32, {4, 2, 128, 128}, {true, false, false, false});
  auto lhs_shape_dynamic =
      ShapeUtil::MakeShape(F32, {4, 128, 2, 8}, {true, false, false, false});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, lhs_shape, "A"));
  auto* b_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, rhs_shape, "B"));
  auto* size_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, scalar_shape_, "size_param"));

  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      lhs_shape_dynamic, a_param, size_param, 0));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(3);
  dot_dnums.add_rhs_contracting_dimensions(3);
  dot_dnums.add_lhs_batch_dimensions(0);
  dot_dnums.add_lhs_batch_dimensions(2);
  dot_dnums.add_rhs_batch_dimensions(0);
  dot_dnums.add_rhs_batch_dimensions(2);
  auto dot = builder.AddInstruction(
      HloInstruction::CreateDot(output_shape, a_param, b_param, dot_dnums,
                                HloTestBase::DefaultPrecisionConfig(2)));

  module_->AddEntryComputation(builder.Build());

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(dot, {}, 0), size_param);
  EXPECT_EQ(inference_->GetDynamicSize(dot, {}, 1), nullptr);
  EXPECT_EQ(inference_->GetDynamicSize(dot, {}, 2), nullptr);
  EXPECT_EQ(inference_->GetDynamicSize(dot, {}, 3), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, DotTestMultiContracting) {
  auto builder = HloComputation::Builder(TestName());
  auto lhs_shape = ShapeUtil::MakeShape(F32, {2, 2, 8, 64});
  auto rhs_shape = ShapeUtil::MakeShape(F32, {2, 2, 512});
  auto output_shape = ShapeUtil::MakeShape(F32, {8, 64, 512});
  auto lhs_shape_dynamic =
      ShapeUtil::MakeShape(F32, {2, 2, 8, 64}, {true, true, false, false});
  auto rhs_shape_dynamic =
      ShapeUtil::MakeShape(F32, {2, 2, 512}, {true, true, false});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, lhs_shape, "A"));
  auto* b_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, rhs_shape, "B"));
  auto* size_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, scalar_shape_, "size_param"));

  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      ShapeUtil::MakeShape(F32, lhs_shape.dimensions(),
                           {true, false, false, false}),
      a_param, size_param, 0));
  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      lhs_shape_dynamic, a_param, size_param, 1));
  b_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      ShapeUtil::MakeShape(F32, rhs_shape.dimensions(), {true, false, false}),
      b_param, size_param, 0));
  b_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      rhs_shape_dynamic, b_param, size_param, 1));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(1);
  auto dot = builder.AddInstruction(
      HloInstruction::CreateDot(output_shape, a_param, b_param, dot_dnums,
                                HloTestBase::DefaultPrecisionConfig(2)));

  module_->AddEntryComputation(builder.Build());

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  // Nothing is dynamic in the output.
  EXPECT_EQ(inference_->GetDynamicSize(dot, {}, 0), nullptr);
  EXPECT_EQ(inference_->GetDynamicSize(dot, {}, 1), nullptr);
  EXPECT_EQ(inference_->GetDynamicSize(dot, {}, 2), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, ConvolutionTest) {
  auto builder = HloComputation::Builder(TestName());
  constexpr int xdim = 3;
  constexpr int ydim = 2;
  constexpr int zdim = 1;
  auto xy_shape = ShapeUtil::MakeShape(F32, {xdim, ydim});
  auto yz_shape = ShapeUtil::MakeShape(F32, {ydim, zdim});
  auto xy_shape_dynamic = ShapeUtil::MakeShape(F32, {xdim, ydim}, {true, true});
  auto zx_shape_dynamic =
      ShapeUtil::MakeShape(F32, {zdim, xdim}, {false, true});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, xy_shape, "A"));
  auto* b_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, yz_shape, "B"));
  auto* size_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, scalar_shape_, "size_param"));

  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      ShapeUtil::MakeShape(F32, xy_shape.dimensions(), {true, false}), a_param,
      size_param, 0));
  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      xy_shape_dynamic, a_param, size_param, 1));

  auto dnums = XlaBuilder::CreateDefaultConvDimensionNumbers(0);

  dnums.set_kernel_input_feature_dimension(0);
  dnums.set_kernel_output_feature_dimension(1);
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(1);
  dnums.set_output_feature_dimension(0);

  Window window;

  auto* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      zx_shape_dynamic, a_param, b_param, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums,
      HloTestBase::DefaultPrecisionConfig(2)));

  module_->AddEntryComputation(builder.Build());

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(conv, {}, 1), size_param);
  EXPECT_EQ(inference_->GetDynamicSize(conv, {}, 0), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, TransposeTest) {
  // Test the ability to trace unmodified dimensions
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {1, 2, 3});
  auto output_shape = ShapeUtil::MakeShape(F32, {3, 2, 1}, {true, true, true});
  auto dynamic_shape = ShapeUtil::MakeShape(F32, {1, 2, 3}, {true, true, true});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, input_shape, "A"));
  auto* size_param_1 = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, scalar_shape_, "size_param"));
  auto* size_param_2 = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, scalar_shape_, "size_param"));
  auto* size_param_3 = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/3, scalar_shape_, "size_param"));

  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      ShapeUtil::MakeShape(F32, {1, 2, 3}, {true, false, false}), a_param,
      size_param_1, 0));
  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      ShapeUtil::MakeShape(F32, {1, 2, 3}, {true, true, false}), a_param,
      size_param_2, 1));
  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      dynamic_shape, a_param, size_param_3, 2));

  auto* transpose = builder.AddInstruction(
      HloInstruction::CreateTranspose(output_shape, a_param, {2, 1, 0}));

  module_->AddEntryComputation(builder.Build());

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(transpose, {}, 0), size_param_3);
  EXPECT_EQ(inference_->GetDynamicSize(transpose, {}, 1), size_param_2);
  EXPECT_EQ(inference_->GetDynamicSize(transpose, {}, 2), size_param_1);
}

TEST_F(DynamicDimensionInferenceTest, NonDescendingTransposeTest) {
  // Test the ability to trace unmodified dimensions
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {1, 2, 3});
  auto output_shape = ShapeUtil::MakeShape(F32, {3, 1, 2}, {true, true, true});
  auto dynamic_shape = ShapeUtil::MakeShape(F32, {1, 2, 3}, {true, true, true});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, input_shape, "A"));
  auto* size_param_1 = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, scalar_shape_, "size_param"));
  auto* size_param_2 = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, scalar_shape_, "size_param"));
  auto* size_param_3 = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/3, scalar_shape_, "size_param"));

  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      ShapeUtil::MakeShape(F32, {1, 2, 3}, {true, false, false}), a_param,
      size_param_1, 0));
  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      ShapeUtil::MakeShape(F32, {1, 2, 3}, {true, true, false}), a_param,
      size_param_2, 1));
  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      dynamic_shape, a_param, size_param_3, 2));

  auto* transpose = builder.AddInstruction(
      HloInstruction::CreateTranspose(output_shape, a_param, {2, 0, 1}));

  module_->AddEntryComputation(builder.Build());

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(transpose, {}, 0), size_param_3);
  EXPECT_EQ(inference_->GetDynamicSize(transpose, {}, 1), size_param_1);
  EXPECT_EQ(inference_->GetDynamicSize(transpose, {}, 2), size_param_2);
}

TEST_F(DynamicDimensionInferenceTest, ReshapeTest) {
  // Test the ability to trace unmodified reshape dimensions.
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {2, 3, 4, 5, 6});
  auto output_shape = ShapeUtil::MakeShape(
      F32, {6, 4, 1, 5, 2, 3}, {false, true, false, true, false, false});
  auto dynamic_shape = ShapeUtil::MakeShape(F32, {2, 3, 4, 5, 6},
                                            {false, false, true, true, false});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, input_shape, "A"));
  auto* size_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, scalar_shape_, "size_param"));

  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      ShapeUtil::MakeShape(F32, {2, 3, 4, 5, 6},
                           {false, false, true, false, false}),
      a_param, size_param, 2));
  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      dynamic_shape, a_param, size_param, 3));

  auto* reshape = builder.AddInstruction(
      HloInstruction::CreateReshape(output_shape, a_param));

  module_->AddEntryComputation(builder.Build());

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(reshape, {}, 0), nullptr);
  EXPECT_EQ(inference_->GetDynamicSize(reshape, {}, 1), size_param);
  EXPECT_EQ(inference_->GetDynamicSize(reshape, {}, 2), nullptr);
  EXPECT_EQ(inference_->GetDynamicSize(reshape, {}, 3), size_param);
  EXPECT_EQ(inference_->GetDynamicSize(reshape, {}, 4), nullptr);
  EXPECT_EQ(inference_->GetDynamicSize(reshape, {}, 5), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, ReshapeInferredDimensionTest) {
  // Test the ability to trace inferred dimension when output is bigger than
  // input.
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {4, 5});
  auto output_shape =
      ShapeUtil::MakeShape(F32, {1, 4, 5}, {true, false, false});
  auto dynamic_shape = ShapeUtil::MakeShape(F32, {4, 5}, {true, false});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, input_shape, "A"));
  auto* size_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, scalar_shape_, "size_param"));

  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      dynamic_shape, a_param, size_param, 0));

  auto* reshape = builder.AddInstruction(HloInstruction::CreateReshape(
      output_shape, a_param, /*inferred_dimension=*/0));

  module_->AddEntryComputation(builder.Build());

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_NE(inference_->GetDynamicSize(reshape, {}, 0), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, ReshapeTestMajorDimension) {
  // Test the ability to trace dimension combining.
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {32, 10, 4});
  auto output_shape = ShapeUtil::MakeShape(F32, {320, 4}, {true, false});
  auto dynamic_shape =
      ShapeUtil::MakeShape(F32, {32, 10, 4}, {true, false, false});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, input_shape, "A"));

  auto* size_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, scalar_shape_, "size_param"));

  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      dynamic_shape, a_param, size_param, 0));

  auto* reshape = builder.AddInstruction(
      HloInstruction::CreateReshape(output_shape, a_param));

  module_->AddEntryComputation(builder.Build());

  SCOPED_TRACE(module_->ToString());
  absl::Status status = RunInference();
  EXPECT_NE(inference_->GetDynamicSize(reshape, {}, 0), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, ReshapeIntoScalar) {
  // Test the ability to a reshape into scalar.
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {1});
  auto output_shape = ShapeUtil::MakeShape(F32, {});
  auto dynamic_shape = ShapeUtil::MakeShape(F32, {1}, {true});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, input_shape, "A"));

  auto* size_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, scalar_shape_, "size_param"));

  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      dynamic_shape, a_param, size_param, 0));

  builder.AddInstruction(HloInstruction::CreateReshape(output_shape, a_param));

  module_->AddEntryComputation(builder.Build());

  SCOPED_TRACE(module_->ToString());
  TF_CHECK_OK(RunInference());
}

TEST_F(DynamicDimensionInferenceTest, GatherTest) {
  const std::string hlo_text = R"(
HloModule TensorFlowGatherV2

ENTRY main {
  operand = s32[20,10]{1,0} parameter(0)
  indices = s32[32,20] parameter(1)
  dynamic_size = s32[] parameter(2)
  indices_dynamic = s32[<=32,20] set-dimension-size(indices, dynamic_size), dimensions={0}
  ROOT gather = s32[<=32,20,10]{2,1,0} gather(%operand, %indices_dynamic),
                 offset_dims={2},
                 collapsed_slice_dims={0},
                 start_index_map={0},
                 index_vector_dim=2,
                 slice_sizes={1,10}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_text));
  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(
                module_->entry_computation()->root_instruction(), {}, 0),
            module_->entry_computation()->parameter_instruction(2));
}

TEST_F(DynamicDimensionInferenceTest, BroadcastTest) {
  // Test the ability to trace broadcast dimension.
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {2});
  auto output_shape =
      ShapeUtil::MakeShape(F32, {3, 2, 4}, {false, true, false});
  auto dynamic_shape = ShapeUtil::MakeShape(F32, {2}, {true});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, input_shape, "A"));
  auto* size_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, scalar_shape_, "size_param"));

  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      dynamic_shape, a_param, size_param, 0));

  auto* broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(output_shape, a_param, {1}));

  module_->AddEntryComputation(builder.Build());

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(broadcast, {}, 0), nullptr);
  EXPECT_EQ(inference_->GetDynamicSize(broadcast, {}, 1), size_param);
  EXPECT_EQ(inference_->GetDynamicSize(broadcast, {}, 2), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, WhileTest) {
  // Test the ability to trace into while loops.
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {2, 4, 4});
  auto dynamic_shape =
      ShapeUtil::MakeShape(F32, {2, 4, 4}, {true, false, false});
  auto tuple_shape = ShapeUtil::MakeTupleShape({input_shape, input_shape});
  auto dynamic_tuple_shape =
      ShapeUtil::MakeTupleShape({dynamic_shape, dynamic_shape});

  // Body:
  //
  //   Param
  //   |  |
  // GTE1 GTE2
  //   |  |
  //    ADD
  auto body_builder = HloComputation::Builder("body");
  auto body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, dynamic_tuple_shape, "param"));
  auto gte_0 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(dynamic_shape, body_param, 0));
  auto gte_1 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(dynamic_shape, body_param, 1));
  auto add = body_builder.AddInstruction(HloInstruction::CreateBinary(
      dynamic_shape, HloOpcode::kAdd, gte_0, gte_1));
  body_builder.AddInstruction(HloInstruction::CreateTuple({add, add}));

  HloComputation* body = module_->AddEmbeddedComputation(body_builder.Build());

  auto cond_builder = HloComputation::Builder("condition");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, dynamic_tuple_shape, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  HloComputation* condition =
      module_->AddEmbeddedComputation(cond_builder.Build());

  // Entry:
  //
  //  Param
  //   |
  //  While
  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, tuple_shape, "A"));
  auto* size_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, scalar_shape_, "size_param"));
  auto* a_0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(input_shape, a_param, 0));
  a_0 = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      dynamic_shape, a_0, size_param, 0));
  auto* a_1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(input_shape, a_param, 0));
  a_1 = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      dynamic_shape, a_1, size_param, 0));
  a_param = builder.AddInstruction(HloInstruction::CreateTuple({a_0, a_1}));
  builder.AddInstruction(HloInstruction::CreateWhile(dynamic_tuple_shape,
                                                     condition, body, a_param));

  module_->AddEntryComputation(builder.Build());

  TF_ASSERT_OK(RunInference());
  HloInstruction* while_hlo = nullptr;
  // The while hlo has been replaced, find the new one.
  for (HloInstruction* inst : module_->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kWhile) {
      while_hlo = inst;
    }
  }
  ASSERT_NE(while_hlo, nullptr);
  // The original while shape has 2 parameters. With dynamic size, the tuple
  // should have 4 elements (We don't deduplicate the arguments).
  EXPECT_EQ(while_hlo->shape().tuple_shapes_size(), 4);
  HloInstruction* add_inst = nullptr;
  for (HloInstruction* inst : while_hlo->while_body()->instructions()) {
    if (inst->opcode() == HloOpcode::kAdd) {
      add_inst = inst;
    }
  }
  EXPECT_NE(add_inst, nullptr);
  EXPECT_NE(inference_->GetDynamicSize(add_inst, {}, 0), nullptr);
  EXPECT_NE(inference_->GetDynamicSize(
                module_->entry_computation()->root_instruction(), {0}, 0),
            nullptr);
  EXPECT_NE(inference_->GetDynamicSize(
                module_->entry_computation()->root_instruction(), {1}, 0),
            nullptr);
}

TEST_F(DynamicDimensionInferenceTest, ConditionalInputTest) {
  // Test the ability to trace into conditional loops.
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {2, 4, 4});
  auto dynamic_shape =
      ShapeUtil::MakeShape(F32, {2, 4, 4}, {true, false, false});
  auto output_shape = ShapeUtil::MakeShape(F32, {2, 2, 2});
  // In this test we set inputs to different branches to different shapes.
  auto tuple_shape_1 = ShapeUtil::MakeTupleShape({input_shape});
  auto tuple_shape_2 = ShapeUtil::MakeTupleShape({input_shape, input_shape});
  auto tuple_shape_3 =
      ShapeUtil::MakeTupleShape({input_shape, input_shape, input_shape});
  auto tuple_shape_2_dynamic =
      ShapeUtil::MakeTupleShape({dynamic_shape, dynamic_shape});
  auto tuple_shape_3_dynamic =
      ShapeUtil::MakeTupleShape({input_shape, dynamic_shape, dynamic_shape});

  // true branch:
  //
  //   Param
  //   |  |
  // GTE1 GTE2
  //   |  |
  // Tuple(ADD)
  auto true_builder = HloComputation::Builder("true");
  {
    auto true_param = true_builder.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape_2_dynamic, "param"));
    auto gte_0 = true_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(dynamic_shape, true_param, 0));
    auto gte_1 = true_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(dynamic_shape, true_param, 1));
    auto add = true_builder.AddInstruction(HloInstruction::CreateBinary(
        dynamic_shape, HloOpcode::kAdd, gte_0, gte_1));
    true_builder.AddInstruction(HloInstruction::CreateTuple({add}));
  }
  HloComputation* true_branch =
      module_->AddEmbeddedComputation(true_builder.Build());
  // false branch:
  //
  //      Param
  //  |     |    |
  // GTE1  GTE2 GTE3
  //        |     |
  //       Tuple(ADD)
  auto false_builder = HloComputation::Builder("false");
  {
    auto false_param = false_builder.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape_3_dynamic, "param"));
    auto gte_0 = false_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(dynamic_shape, false_param, 1));
    auto gte_1 = false_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(dynamic_shape, false_param, 2));
    auto add = false_builder.AddInstruction(HloInstruction::CreateBinary(
        dynamic_shape, HloOpcode::kAdd, gte_0, gte_1));
    false_builder.AddInstruction(HloInstruction::CreateTuple({add}));
  }
  HloComputation* false_branch =
      module_->AddEmbeddedComputation(false_builder.Build());

  // Entry:
  //
  //  Param(bool) Param2 (tuple_2) Param3(tuple_3)
  //   |            |                 |
  //   +---------Condition------------+
  auto* pred_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeScalarShape(PRED), "pred"));

  auto* tuple_2_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, tuple_shape_2, "tuple_2_param"));
  auto* tuple_3_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, tuple_shape_3, "tuple_3_param"));
  auto* size_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/3, scalar_shape_, "size_param"));

  auto* param_2_0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(input_shape, tuple_2_param, 0));
  param_2_0 = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      dynamic_shape, param_2_0, size_param, 0));
  auto* param_2_1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(input_shape, tuple_2_param, 1));
  param_2_1 = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      dynamic_shape, param_2_1, size_param, 0));
  tuple_2_param = builder.AddInstruction(
      HloInstruction::CreateTuple({param_2_0, param_2_1}));

  auto* param_3_0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(input_shape, tuple_3_param, 0));
  auto* param_3_1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(input_shape, tuple_3_param, 1));
  param_3_1 = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      dynamic_shape, param_3_1, size_param, 0));
  auto* param_3_2 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(input_shape, tuple_3_param, 2));
  param_3_2 = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      dynamic_shape, param_3_1, size_param, 0));
  tuple_3_param = builder.AddInstruction(
      HloInstruction::CreateTuple({param_3_0, param_3_1, param_3_2}));

  builder.AddInstruction(HloInstruction::CreateConditional(
      tuple_shape_1, pred_param, tuple_2_param, true_branch, tuple_3_param,
      false_branch));

  module_->AddEntryComputation(builder.Build());

  TF_ASSERT_OK(RunInference());

  HloInstruction* conditional_hlo = nullptr;
  // The conditional hlo has been replaced, find the new one.
  for (HloInstruction* inst : module_->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kConditional) {
      conditional_hlo = inst;
    }
  }
  ASSERT_NE(conditional_hlo, nullptr);
  // The original conditional shape has 1 parameters. With dynamic size passed
  // out from the computation, another element is added to the tuple.
  EXPECT_EQ(conditional_hlo->shape().tuple_shapes_size(), 2);
  HloInstruction* add_true_branch = nullptr;
  for (HloInstruction* inst :
       conditional_hlo->true_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kAdd) {
      add_true_branch = inst;
    }
  }
  EXPECT_NE(add_true_branch, nullptr);
  EXPECT_NE(inference_->GetDynamicSize(add_true_branch, {}, 0), nullptr);

  HloInstruction* add_false_branch = nullptr;
  for (HloInstruction* inst :
       conditional_hlo->false_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kAdd) {
      add_false_branch = inst;
    }
  }
  EXPECT_NE(add_false_branch, nullptr);
  EXPECT_NE(inference_->GetDynamicSize(add_false_branch, {}, 0), nullptr);

  EXPECT_NE(inference_->GetDynamicSize(conditional_hlo, {0}, 0), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, ReduceWindowBatchTest) {
  // Test the ability to trace reduce window batch dimensions.
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {2, 4, 4});
  auto output_shape =
      ShapeUtil::MakeShape(F32, {2, 2, 2}, {true, false, false});
  auto dynamic_shape =
      ShapeUtil::MakeShape(F32, {2, 4, 4}, {true, false, false});

  Window window;
  // First dimension is unchanged.
  WindowDimension* batch_dim = window.add_dimensions();
  batch_dim->set_size(1);
  batch_dim->set_stride(1);
  batch_dim->set_padding_low(0);
  batch_dim->set_padding_high(0);
  batch_dim->set_window_dilation(1);
  batch_dim->set_base_dilation(1);

  // Second and third dimension are reduced.
  for (int64_t i = 0; i < 2; ++i) {
    WindowDimension* dim = window.add_dimensions();
    dim->set_size(2);
    dim->set_stride(2);
    dim->set_padding_low(0);
    dim->set_padding_high(0);
    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
  }

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, input_shape, "A"));
  auto* size_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, scalar_shape_, "size_param"));

  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      dynamic_shape, a_param, size_param, 0));

  auto init = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0)));

  auto* reduce_window =
      builder.AddInstruction(HloInstruction::CreateReduceWindow(
          output_shape, a_param, init, window, GetAdd()));

  module_->AddEntryComputation(builder.Build());

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(reduce_window, {}, 0), size_param);
}

TEST_F(DynamicDimensionInferenceTest, SelectAndScatterTest) {
  // Test the ability to trace select and scatter batch dimensions.
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {2, 4, 4});
  auto source_shape = ShapeUtil::MakeShape(F32, {2, 2, 2});
  auto input_shape_dynamic =
      ShapeUtil::MakeShape(F32, {2, 4, 4}, {true, false, false});
  auto source_shape_dynamic =
      ShapeUtil::MakeShape(F32, {2, 2, 2}, {true, false, false});

  Window window;
  // First dimension is unchanged.
  WindowDimension* batch_dim = window.add_dimensions();
  batch_dim->set_size(1);
  batch_dim->set_stride(1);
  batch_dim->set_padding_low(0);
  batch_dim->set_padding_high(0);
  batch_dim->set_window_dilation(1);
  batch_dim->set_base_dilation(1);

  // Second and third dimension are reduced.
  for (int64_t i = 0; i < 2; ++i) {
    WindowDimension* dim = window.add_dimensions();
    dim->set_size(2);
    dim->set_stride(2);
    dim->set_padding_low(0);
    dim->set_padding_high(0);
    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
  }

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, input_shape, "A"));
  auto* size_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, scalar_shape_, "size_param"));
  auto* source = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, source_shape, "B"));

  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      input_shape_dynamic, a_param, size_param, 0));
  source = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      source_shape_dynamic, source, size_param, 0));

  auto init = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0)));

  auto* sns = builder.AddInstruction(HloInstruction::CreateSelectAndScatter(
      input_shape_dynamic, a_param, GetGe(), window, source, init, GetAdd()));

  module_->AddEntryComputation(builder.Build());

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(sns, {}, 0), size_param);
}

TEST_F(DynamicDimensionInferenceTest, ConcatTest) {
  // Concat two data params.
  auto builder = HloComputation::Builder(TestName());

  auto data_param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {5, 7}), "data_param_1"));
  auto data_param_2 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {5, 8}), "data_param_2"));
  auto size_param = builder.AddInstruction(
      HloInstruction::CreateParameter(2, scalar_shape_, "size_param"));

  data_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      ShapeUtil::MakeShape(F32, {5, 7}, {true, false}), data_param, size_param,
      0));
  data_param_2 = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      ShapeUtil::MakeShape(F32, {5, 8}, {true, false}), data_param_2,
      size_param, 0));

  auto* concat = builder.AddInstruction(HloInstruction::CreateConcatenate(
      ShapeUtil::MakeShape(F32, {5, 15}, {true, false}),
      {data_param, data_param_2}, 1));

  module_->AddEntryComputation(builder.Build());

  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(concat, {}, 0), size_param);
}

TEST_F(DynamicDimensionInferenceTest, SliceTest) {
  auto builder = HloComputation::Builder(TestName());

  auto dynamic_shape = ShapeUtil::MakeShape(F32, {5, 7}, {false, true});

  auto data_param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {5, 7}), "data_param"));
  auto size_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "size_param"));

  data_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      dynamic_shape, data_param, size_param, 1));

  auto* slice = builder.AddInstruction(HloInstruction::CreateSlice(
      dynamic_shape, data_param,
      /*start_indices=*/{0, 0},
      /*limit_indices=*/{5, 7}, /*strides=*/{1, 1}));

  module_->AddEntryComputation(builder.Build());

  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(slice, {}, 1), size_param);
}

TEST_F(DynamicDimensionInferenceTest, DynamicSliceTest) {
  auto builder = HloComputation::Builder(TestName());

  auto data_param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {5, 7}), "data_param"));
  auto size_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "size_param"));

  std::vector<HloInstruction*> params;
  for (int i = 0; i < 2; ++i) {
    params.push_back(builder.AddInstruction(HloInstruction::CreateParameter(
        i + 2, ShapeUtil::MakeShape(S32, {}), "slice_indices")));
  }

  data_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      ShapeUtil::MakeShape(F32, {5, 7}, {true, false}), data_param, size_param,
      0));

  auto* slice = builder.AddInstruction(HloInstruction::CreateDynamicSlice(
      ShapeUtil::MakeShape(F32, {5, 1}, {true, false}), data_param, params,
      /*slice_sizes=*/{5, 1}));

  module_->AddEntryComputation(builder.Build());

  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(slice, {}, 0), size_param);
}

TEST_F(DynamicDimensionInferenceTest, SortTest) {
  auto builder = HloComputation::Builder(TestName());

  auto dynamic_shape = ShapeUtil::MakeShape(F32, {5, 7}, {true, false});

  auto data_param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {5, 7}), "data_param"));
  auto size_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "size_param"));

  auto compare_builder = HloComputation::Builder("condition");
  compare_builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {}), "param1"));
  compare_builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {}), "param2"));
  compare_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  HloComputation* compare =
      module_->AddEmbeddedComputation(compare_builder.Build());

  data_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      dynamic_shape, data_param, size_param, 0));

  auto* sort = builder.AddInstruction(
      HloInstruction::CreateSort(dynamic_shape, 1, {data_param}, compare,
                                 /*is_stable=*/false));

  module_->AddEntryComputation(builder.Build());

  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(sort, {}, 0), size_param);
}

TEST_F(DynamicDimensionInferenceTest, MultiValueSortTest) {
  auto builder = HloComputation::Builder(TestName());

  auto shape = ShapeUtil::MakeShape(F32, {5, 7});
  auto dynamic_shape = ShapeUtil::MakeShape(F32, {5, 7}, {true, false});

  auto data_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "data_param"));
  auto size_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "size_param"));

  auto compare_builder = HloComputation::Builder("condition");
  compare_builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {}), "param1"));
  compare_builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {}), "param2"));
  compare_builder.AddInstruction(HloInstruction::CreateParameter(
      2, ShapeUtil::MakeShape(F32, {}), "param3"));
  compare_builder.AddInstruction(HloInstruction::CreateParameter(
      3, ShapeUtil::MakeShape(F32, {}), "param4"));
  compare_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  HloComputation* compare =
      module_->AddEmbeddedComputation(compare_builder.Build());

  data_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      dynamic_shape, data_param, size_param, 0));

  auto* sort = builder.AddInstruction(HloInstruction::CreateSort(
      ShapeUtil::MakeTupleShape({dynamic_shape, dynamic_shape}), 1,
      {data_param, data_param}, compare,
      /*is_stable=*/false));

  module_->AddEntryComputation(builder.Build());

  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(sort, {0}, 0), size_param);
  EXPECT_EQ(inference_->GetDynamicSize(sort, {1}, 0), size_param);
}

TEST_F(DynamicDimensionInferenceTest, DynamicSliceSingleElementTest) {
  // Slicing out a single element from a dynamic dimension terminates the
  // dynamic dimension.
  auto builder = HloComputation::Builder(TestName());

  auto data_param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {5, 7}), "data_param"));
  auto* size_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "size_param"));

  std::vector<HloInstruction*> params;
  for (int i = 0; i < 2; ++i) {
    params.push_back(builder.AddInstruction(HloInstruction::CreateParameter(
        i + 2, ShapeUtil::MakeShape(S32, {}), "slice_indices")));
  }

  data_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      ShapeUtil::MakeShape(F32, {5, 7}, {true, false}), data_param, size_param,
      0));

  auto* slice = builder.AddInstruction(HloInstruction::CreateDynamicSlice(
      ShapeUtil::MakeShape(F32, {1, 1}), data_param, params,
      /*slice_sizes=*/{1, 1}));

  module_->AddEntryComputation(builder.Build());

  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(slice, {}, 0), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, InfersCustomOp) {
  auto builder = HloComputation::Builder(TestName());

  auto data_param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {5, 7}), "data_param"));
  auto* size_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "size_param"));

  data_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      ShapeUtil::MakeShape(F32, {5, 7}, {true, false}), data_param, size_param,
      0));

  builder.AddInstruction(HloInstruction::CreateCustomCall(
      ShapeUtil::MakeShape(F32, {1, 1}), {data_param}, "MyCustomOp", ""));

  module_->AddEntryComputation(builder.Build());

  bool handler_called = false;
  auto handler = [&](HloInstruction* hlo,
                     DynamicDimensionInference* inference) {
    CHECK(inference != nullptr);
    CHECK(Cast<HloCustomCallInstruction>(hlo) != nullptr);
    handler_called = true;
    return hlo->IsCustomCall("MyCustomOp");
  };
  TF_ASSERT_OK(RunInference(/*op_supports_dynamism_handler=*/nullptr, handler));

  EXPECT_TRUE(handler_called);
}

TEST_F(DynamicDimensionInferenceTest, DynamicReshapeOp) {
  auto builder = HloComputation::Builder(TestName());
  auto input = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {9}), "data_input"));
  auto six = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(6)));
  // Creates an input of shape [<=9], dynamic size is 6.
  auto dynamic_input =
      builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
          ShapeUtil::MakeShape(F32, {9}, {true}), input, six, 0));
  auto dynamic_size = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(S32, {}), "size_param"));
  auto three = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(3)));

  // Reshape [<=9] into [3, <=3]

  auto dynamic_reshape =
      builder.AddInstruction(HloInstruction::CreateDynamicReshape(
          ShapeUtil::MakeShape(F32, {3, 3}, {false, true}), dynamic_input,
          {three, dynamic_size}));

  module_->AddEntryComputation(builder.Build());

  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(dynamic_reshape, {}, 0), nullptr);
  EXPECT_EQ(inference_->GetDynamicSize(dynamic_reshape, {}, 1), dynamic_size);
}

TEST_F(DynamicDimensionInferenceTest, ReshapeOpWithMultipleDynamicDimensions) {
  auto builder = HloComputation::Builder(TestName());
  auto input = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {9, 2}), "data_input"));
  auto six = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(6)));
  input = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      ShapeUtil::MakeShape(F32, {9, 2}, {true, false}), input, six, 0));
  auto one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(1)));
  input = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      ShapeUtil::MakeShape(F32, {9, 2}, {true, true}), input, one, 1));

  // Reshape [<=9, <=2] into [<=9, 1, <=2]

  auto dynamic_reshape = builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {9, 1, 2}, {true, false, true}), input));

  module_->AddEntryComputation(builder.Build());

  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(dynamic_reshape, {}, 0), six);
  EXPECT_EQ(inference_->GetDynamicSize(dynamic_reshape, {}, 1), nullptr);
  EXPECT_EQ(inference_->GetDynamicSize(dynamic_reshape, {}, 2), one);
}

TEST_F(DynamicDimensionInferenceTest, HandleMapInDynamicDimensionInference) {
  const char* module_str = R"(
HloModule test_module

%scatter-combiner.285 (p0.286: c128[], p1.287: c128[]) -> c128[] {
  %p0.286 = c128[] parameter(0)
  %p1.287 = c128[] parameter(1)
  ROOT %add.288 = c128[] add(c128[] %p0.286, c128[] %p1.287)
}

 %while_body  {
  %reshape.8 = s32[] parameter(4)
  %reshape.7 = c128[1]{0} parameter(3)
  %reduce = pred[] parameter(2)
  %concatenate = s32[1]{0} parameter(1)
  %slice.4 = s32[1]{0} slice(s32[1]{0} %concatenate), slice={[0 : 1]}
  %broadcast.7 = pred[1]{0} broadcast(pred[] %reduce), dimensions={}
  %param.1 = (s32[],c128[<=1]{0},s32[1]{0},c128[1]{0}) parameter(0)
  %get-tuple-element.2 = c128[<=1]{0} get-tuple-element((s32[],c128[<=1]{0},s32[1]{0},c128[1]{0}) %param.1), index=1
  %dynamic-slice.2 = c128[1]{0} dynamic-slice(c128[<=1]{0} %get-tuple-element.2,s32[] %reshape.8), dynamic_slice_sizes={1}
  %map = c128[1]{0} map(c128[1]{0} %dynamic-slice.2,c128[1]{0} %reshape.7), dimensions={0}, to_apply=%scatter-combiner.285
  %select = c128[1]{0} select(pred[1]{0} %broadcast.7,c128[1]{0} %map,c128[1]{0} %dynamic-slice.2)
  %reshape.9 = s32[] reshape(s32[1]{0} %slice.4)
  %dynamic-update-slice = c128[<=1]{0} dynamic-update-slice(c128[<=1]{0} %get-tuple-element.2,c128[1]{0} %select,s32[] %reshape.9)
})";
  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnUnverifiedModule(module_str));
  TF_ASSERT_OK(RunInference());
}

TEST_F(DynamicDimensionInferenceTest, RuntimeShapeCheck) {
  const char* hlo = R"(
HloModule module

ENTRY computation {
  a = f32[20,20] parameter(0)
  a_size_1 = s32[] parameter(1)
  a_size_2 = s32[] parameter(2)
  a_dynamic_1 = f32[<=20,20] set-dimension-size(a, a_size_1), dimensions={0}
  a_dynamic_2 = f32[<=20,<=20] set-dimension-size(a_dynamic_1, a_size_2), dimensions={1}
  b = f32[20,20] parameter(3)
  b_size_1 = s32[] parameter(4)
  b_size_2 = s32[] parameter(5)
  b_dynamic_1 = f32[<=20,20] set-dimension-size(b, b_size_1), dimensions={0}
  b_dynamic_2 = f32[<=20,<=20] set-dimension-size(b_dynamic_1, b_size_2), dimensions={1}
  ROOT f = add(a_dynamic_2, b_dynamic_2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo));

  TF_ASSERT_OK(RunInference(
      /*op_supports_dynamism_handler=*/nullptr,
      /*handler=*/nullptr, DynamicDimensionInference::ShapeCheckMode::kRuntime,
      /*assertion_generator=*/[&](HloInstruction* constraint) {
        constraint->parent()->AddInstruction(HloInstruction::CreateCustomCall(
            ShapeUtil::MakeTokenShape(), {constraint},
            /*custom_call_target=*/"__xla__assert",
            /*opaque=*/std::string{}, API_VERSION_STATUS_RETURNING));
      }));

  absl::StatusOr<bool> filecheck_result = RunFileCheck(
      module_->ToString(HloPrintOptions().set_print_operand_shape(true)),
      R"(
// CHECK: compare = pred[] compare(s32[] %a_size_1, s32[] %b_size_1), direction=EQ
// CHECK: compare.5 = pred[] compare(s32[] %a_size_2, s32[] %b_size_2), direction=EQ
// CHECK: and.2 = pred[] and(pred[] %compare, pred[] %compare.5)
// CHECK: custom-call(pred[] %and.2), custom_call_target="__xla__assert"
                   )");
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(*filecheck_result);
}

TEST_F(DynamicDimensionInferenceTest, NestedControlFlow) {
  // A module with heavily nested control flow that manipulates dynamic shapes.
  const char* hlo = R"(
HloModule tfcompile.377, entry_computation_layout={(s32[], f32[250]{0}, pred[], pred[], s32[], /*index=5*/pred[], s32[], pred[])->(f32[3]{0})}

cond_2_Sum-reduction.17 {
  x.18 = f32[] parameter(0)
  y.19 = f32[] parameter(1)
  ROOT add.20 = f32[] add(x.18, y.19)
}

cond_2_cond_true_214__.21 {
  arg_tuple.22 = () parameter(0)
  constant.23 = s32[] constant(1)
  reshape.24 = s32[] reshape(constant.23)
  ROOT tuple.25 = (s32[]) tuple(constant.23)
}

cond_2_cond_false_215__.26 {
  arg_tuple.27 = () parameter(0)
  constant.28 = s32[] constant(0)
  reshape.29 = s32[] reshape(constant.28)
  ROOT tuple.30 = (s32[]) tuple(constant.28)
}

cond_2_true_195__.31 {
  arg_tuple.32 = (s32[], f32[250]{0}) parameter(0)
  get-tuple-element.33 = s32[] get-tuple-element(arg_tuple.32), index=0
  constant.35 = s32[] constant(20)
  minimum.36 = s32[] minimum(get-tuple-element.33, constant.35)
  reshape.37 = s32[1]{0} reshape(minimum.36)
  concatenate.38 = s32[1]{0} concatenate(reshape.37), dimensions={0}
  slice.48 = s32[1]{0} slice(concatenate.38), slice={[0:1]}
  reshape.49 = s32[] reshape(reshape.37)
  constant.43 = s32[] constant(0)
  compare.50 = pred[] compare(minimum.36, constant.43), direction=LT
  constant.44 = s32[] constant(250)
  add.51 = s32[] add(constant.44, minimum.36)
  select.52 = s32[] select(compare.50, add.51, minimum.36)
  constant.45 = s32[1]{0} constant({0})
  slice.46 = s32[1]{0} slice(constant.45), slice={[0:1]}
  reshape.47 = s32[] reshape(slice.46)
  subtract.53 = s32[] subtract(select.52, reshape.47)
  maximum.54 = s32[] maximum(subtract.53, constant.43)
  convert.55 = s32[] convert(maximum.54)
  get-tuple-element.34 = f32[250]{0} get-tuple-element(arg_tuple.32), index=1
  constant.39 = f32[] constant(0)
  pad.40 = f32[500]{0} pad(get-tuple-element.34, constant.39), padding=0_250
  constant.41 = s32[] constant(500)
  set-dimension-size.42 = f32[500]{0} set-dimension-size(pad.40, constant.41), dimensions={0}
  dynamic-slice.56 = f32[250]{0} dynamic-slice(set-dimension-size.42, reshape.47), dynamic_slice_sizes={250}
  reshape.57 = f32[250]{0} reshape(dynamic-slice.56)
  set-dimension-size.58 = f32[<=250]{0} set-dimension-size(dynamic-slice.56, maximum.54), dimensions={0}
  constant.59 = f32[] constant(1)
  broadcast.60 = f32[250]{0} broadcast(constant.59), dimensions={}
  compare.61 = pred[<=250]{0} compare(set-dimension-size.58, broadcast.60), direction=GE
  convert.62 = f32[<=250]{0} convert(compare.61)
  convert.63 = f32[<=250]{0} convert(convert.62)
  constant.64 = f32[] constant(0)
  convert.65 = f32[] convert(constant.64)
  reduce.66 = f32[] reduce(convert.62, constant.64), dimensions={0}, to_apply=cond_2_Sum-reduction.17
  convert.67 = f32[] convert(reduce.66)
  reshape.73 = f32[] reshape(reduce.66)
  constant.68 = f32[] constant(6)
  compare.69 = pred[] compare(reduce.66, constant.68), direction=GE
  tuple.70 = () tuple()
  conditional.71 = (s32[]) conditional(compare.69, tuple.70, tuple.70), true_computation=cond_2_cond_true_214__.21, false_computation=cond_2_cond_false_215__.26
  get-tuple-element.72 = s32[] get-tuple-element(conditional.71), index=0
  reshape.74 = s32[] reshape(get-tuple-element.72)
  ROOT tuple.75 = (f32[], s32[]) tuple(reduce.66, get-tuple-element.72)
} // cond_2_true_195__.31

cond_2_false_196__.76 {
  arg_tuple.77 = (s32[], f32[250]{0}) parameter(0)
  constant.80 = f32[] constant(0)
  reshape.82 = f32[] reshape(constant.80)
  constant.81 = s32[] constant(0)
  reshape.83 = s32[] reshape(constant.81)
  ROOT tuple.84 = (f32[], s32[]) tuple(constant.80, constant.81)
} // cond_2_false_196__.76

cond_true_10__.85 {
  arg_tuple.86 = (pred[], pred[], pred[]) parameter(0)
  get-tuple-element.87 = pred[] get-tuple-element(arg_tuple.86), index=0
  reshape.90 = pred[] reshape(get-tuple-element.87)
  ROOT tuple.91 = (pred[]) tuple(get-tuple-element.87)
}

cond_cond_true_16__.92 {
  arg_tuple.93 = (pred[], pred[]) parameter(0)
  get-tuple-element.94 = pred[] get-tuple-element(arg_tuple.93), index=0
  reshape.96 = pred[] reshape(get-tuple-element.94)
  ROOT tuple.97 = (pred[]) tuple(get-tuple-element.94)
}

cond_cond_false_17__.98 {
  arg_tuple.99 = (pred[], pred[]) parameter(0)
  get-tuple-element.101 = pred[] get-tuple-element(arg_tuple.99), index=1
  reshape.102 = pred[] reshape(get-tuple-element.101)
  ROOT tuple.103 = (pred[]) tuple(get-tuple-element.101)
}

cond_false_11__.104 {
  arg_tuple.105 = (pred[], pred[], pred[]) parameter(0)
  get-tuple-element.107 = pred[] get-tuple-element(arg_tuple.105), index=1
  get-tuple-element.108 = pred[] get-tuple-element(arg_tuple.105), index=2
  tuple.109 = (pred[], pred[]) tuple(get-tuple-element.107, get-tuple-element.108)
  conditional.110 = (pred[]) conditional(get-tuple-element.107, tuple.109, tuple.109), true_computation=cond_cond_true_16__.92, false_computation=cond_cond_false_17__.98
  get-tuple-element.111 = pred[] get-tuple-element(conditional.110), index=0
  reshape.112 = pred[] reshape(get-tuple-element.111)
  ROOT tuple.113 = (pred[]) tuple(get-tuple-element.111)
} // cond_false_11__.104

cond_1_map_while_cond_true_82__.114 {
  arg_tuple.115 = (f32[]) parameter(0)
  constant.117 = f32[] constant(0)
  reshape.118 = f32[] reshape(constant.117)
  ROOT tuple.119 = (f32[]) tuple(constant.117)
}

cond_1_map_while_cond_cond_true_91__.120 {
  constant.123 = f32[] constant(0.1)
  arg_tuple.121 = (f32[]) parameter(0)
  get-tuple-element.122 = f32[] get-tuple-element(arg_tuple.121), index=0
  multiply.124 = f32[] multiply(constant.123, get-tuple-element.122)
  constant.125 = f32[] constant(0)
  add.126 = f32[] add(multiply.124, constant.125)
  constant.127 = f32[] constant(0.9)
  divide.128 = f32[] divide(add.126, constant.127)
  reshape.129 = f32[] reshape(divide.128)
  ROOT tuple.130 = (f32[]) tuple(divide.128)
} // cond_1_map_while_cond_cond_true_91__.120

cond_1_map_while_cond_cond_cond_true_106__.131 {
  constant.134 = f32[] constant(0.8)
  arg_tuple.132 = (f32[]) parameter(0)
  get-tuple-element.133 = f32[] get-tuple-element(arg_tuple.132), index=0
  multiply.135 = f32[] multiply(constant.134, get-tuple-element.133)
  constant.136 = f32[] constant(-0.711)
  add.137 = f32[] add(multiply.135, constant.136)
  constant.138 = f32[] constant(0.09)
  divide.139 = f32[] divide(add.137, constant.138)
  reshape.140 = f32[] reshape(divide.139)
  ROOT tuple.141 = (f32[]) tuple(divide.139)
} // cond_1_map_while_cond_cond_cond_true_106__.131

cond_1_map_while_cond_cond_cond_cond_true_121__.142 {
  constant.145 = f32[] constant(0.2)
  arg_tuple.143 = (f32[]) parameter(0)
  get-tuple-element.144 = f32[] get-tuple-element(arg_tuple.143), index=0
  multiply.146 = f32[] multiply(constant.145, get-tuple-element.144)
  constant.147 = f32[] constant(-0.18)
  add.148 = f32[] add(multiply.146, constant.147)
  constant.149 = f32[] constant(0.02)
  divide.150 = f32[] divide(add.148, constant.149)
  reshape.151 = f32[] reshape(divide.150)
  ROOT tuple.152 = (f32[]) tuple(divide.150)
} // cond_1_map_while_cond_cond_cond_cond_true_121__.142

cond_1_map_while_cond_cond_cond_cond_cond_true_136__.153 {
  constant.156 = f32[] constant(0.1)
  arg_tuple.154 = (f32[]) parameter(0)
  get-tuple-element.155 = f32[] get-tuple-element(arg_tuple.154), index=0
  multiply.157 = f32[] multiply(constant.156, get-tuple-element.155)
  constant.158 = f32[] constant(108.788)
  add.159 = f32[] add(multiply.157, constant.158)
  constant.160 = f32[] constant(98.99)
  divide.161 = f32[] divide(add.159, constant.160)
  reshape.162 = f32[] reshape(divide.161)
  ROOT tuple.163 = (f32[]) tuple(divide.161)
} // cond_1_map_while_cond_cond_cond_cond_cond_true_136__.153

cond_1_map_while_cond_cond_cond_cond_cond_false_137__.164 {
  arg_tuple.165 = (f32[]) parameter(0)
  constant.167 = f32[] constant(1.2)
  reshape.168 = f32[] reshape(constant.167)
  ROOT tuple.169 = (f32[]) tuple(constant.167)
}

cond_1_map_while_cond_cond_cond_cond_false_122__.170 {
  arg_tuple.171 = (f32[]) parameter(0)
  get-tuple-element.172 = f32[] get-tuple-element(arg_tuple.171), index=0
  constant.173 = f32[] constant(100)
  compare.174 = pred[] compare(get-tuple-element.172, constant.173), direction=LE
  tuple.175 = (f32[]) tuple(get-tuple-element.172)
  conditional.176 = (f32[]) conditional(compare.174, tuple.175, tuple.175), true_computation=cond_1_map_while_cond_cond_cond_cond_cond_true_136__.153, false_computation=cond_1_map_while_cond_cond_cond_cond_cond_false_137__.164
  get-tuple-element.177 = f32[] get-tuple-element(conditional.176), index=0
  reshape.178 = f32[] reshape(get-tuple-element.177)
  ROOT tuple.179 = (f32[]) tuple(get-tuple-element.177)
} // cond_1_map_while_cond_cond_cond_cond_false_122__.170

cond_1_map_while_cond_cond_cond_false_107__.180 {
  arg_tuple.181 = (f32[]) parameter(0)
  get-tuple-element.182 = f32[] get-tuple-element(arg_tuple.181), index=0
  constant.183 = f32[] constant(1.01)
  compare.184 = pred[] compare(get-tuple-element.182, constant.183), direction=LE
  tuple.185 = (f32[]) tuple(get-tuple-element.182)
  conditional.186 = (f32[]) conditional(compare.184, tuple.185, tuple.185), true_computation=cond_1_map_while_cond_cond_cond_cond_true_121__.142, false_computation=cond_1_map_while_cond_cond_cond_cond_false_122__.170
  get-tuple-element.187 = f32[] get-tuple-element(conditional.186), index=0
  reshape.188 = f32[] reshape(get-tuple-element.187)
  ROOT tuple.189 = (f32[]) tuple(get-tuple-element.187)
} // cond_1_map_while_cond_cond_cond_false_107__.180

cond_1_map_while_cond_cond_false_92__.190 {
  arg_tuple.191 = (f32[]) parameter(0)
  get-tuple-element.192 = f32[] get-tuple-element(arg_tuple.191), index=0
  constant.193 = f32[] constant(0.99)
  compare.194 = pred[] compare(get-tuple-element.192, constant.193), direction=LE
  tuple.195 = (f32[]) tuple(get-tuple-element.192)
  conditional.196 = (f32[]) conditional(compare.194, tuple.195, tuple.195), true_computation=cond_1_map_while_cond_cond_cond_true_106__.131, false_computation=cond_1_map_while_cond_cond_cond_false_107__.180
  get-tuple-element.197 = f32[] get-tuple-element(conditional.196), index=0
  reshape.198 = f32[] reshape(get-tuple-element.197)
  ROOT tuple.199 = (f32[]) tuple(get-tuple-element.197)
} // cond_1_map_while_cond_cond_false_92__.190

cond_1_map_while_cond_false_83__.200 {
  arg_tuple.201 = (f32[]) parameter(0)
  get-tuple-element.202 = f32[] get-tuple-element(arg_tuple.201), index=0
  constant.203 = f32[] constant(0.9)
  compare.204 = pred[] compare(get-tuple-element.202, constant.203), direction=LE
  tuple.205 = (f32[]) tuple(get-tuple-element.202)
  conditional.206 = (f32[]) conditional(compare.204, tuple.205, tuple.205), true_computation=cond_1_map_while_cond_cond_true_91__.120, false_computation=cond_1_map_while_cond_cond_false_92__.190
  get-tuple-element.207 = f32[] get-tuple-element(conditional.206), index=0
  reshape.208 = f32[] reshape(get-tuple-element.207)
  ROOT tuple.209 = (f32[]) tuple(get-tuple-element.207)
} // cond_1_map_while_cond_false_83__.200

cond_1_map_while_body_59__.210 {
  arg_tuple.211 = (s32[], s32[], s32[], (f32[<=250]{0}, s32[]), s32[], /*index=5*/(f32[<=250]{0}, s32[])) parameter(0)
  get-tuple-element.212 = s32[] get-tuple-element(arg_tuple.211), index=0
  constant.218 = s32[] constant(1)
  add.219 = s32[] add(get-tuple-element.212, constant.218)
  reshape.239 = s32[] reshape(add.219)
  get-tuple-element.213 = s32[] get-tuple-element(arg_tuple.211), index=1
  reshape.240 = s32[] reshape(get-tuple-element.213)
  get-tuple-element.214 = s32[] get-tuple-element(arg_tuple.211), index=2
  constant.220 = s32[] constant(1)
  add.221 = s32[] add(get-tuple-element.214, constant.220)
  reshape.241 = s32[] reshape(add.221)
  get-tuple-element.216 = s32[] get-tuple-element(arg_tuple.211), index=4
  reshape.242 = s32[] reshape(get-tuple-element.216)
  get-tuple-element.215 = (f32[<=250]{0}, s32[]) get-tuple-element(arg_tuple.211), index=3
  get-tuple-element.235 = f32[<=250]{0} get-tuple-element(get-tuple-element.215), index=0
  get-tuple-element.217 = (f32[<=250]{0}, s32[]) get-tuple-element(arg_tuple.211), index=5
  get-tuple-element.223 = f32[<=250]{0} get-tuple-element(get-tuple-element.217), index=0
  dynamic-slice.224 = f32[1]{0} dynamic-slice(get-tuple-element.223, get-tuple-element.214), dynamic_slice_sizes={1}
  reshape.225 = f32[] reshape(dynamic-slice.224)
  constant.226 = f32[] constant(0)
  compare.227 = pred[] compare(reshape.225, constant.226), direction=LE
  tuple.228 = (f32[]) tuple(reshape.225)
  conditional.229 = (f32[]) conditional(compare.227, tuple.228, tuple.228), true_computation=cond_1_map_while_cond_true_82__.114, false_computation=cond_1_map_while_cond_false_83__.200
  get-tuple-element.230 = f32[] get-tuple-element(conditional.229), index=0
  reshape.233 = f32[1]{0} reshape(get-tuple-element.230)
  dynamic-update-slice.236 = f32[<=250]{0} dynamic-update-slice(get-tuple-element.235, reshape.233, get-tuple-element.214)
  get-tuple-element.237 = s32[] get-tuple-element(get-tuple-element.215), index=1
  tuple.238 = (f32[<=250]{0}, s32[]) tuple(dynamic-update-slice.236, get-tuple-element.237)
  ROOT tuple.243 = (s32[], s32[], s32[], (f32[<=250]{0}, s32[]), s32[], /*index=5*/(f32[<=250]{0}, s32[])) tuple(add.219, get-tuple-element.213, add.221, tuple.238, get-tuple-element.216, /*index=5*/get-tuple-element.217)
} // cond_1_map_while_body_59__.210

cond_wrapper.257 {
  inputs.258 = (s32[], s32[], s32[], (f32[<=250]{0}, s32[]), s32[], /*index=5*/(f32[<=250]{0}, s32[])) parameter(0)
  get-tuple-element.0 = s32[] get-tuple-element(inputs.258), index=0
  get-tuple-element.1 = s32[] get-tuple-element(inputs.258), index=1
  compare.0 = pred[] compare(get-tuple-element.0, get-tuple-element.1), direction=LT
  get-tuple-element.2 = s32[] get-tuple-element(inputs.258), index=2
  get-tuple-element.3 = s32[] get-tuple-element(inputs.258), index=4
  compare.1 = pred[] compare(get-tuple-element.2, get-tuple-element.3), direction=LT
  and.0 = pred[] and(compare.0, compare.1)
  tuple.0 = (pred[]) tuple(and.0)
  ROOT get-tuple-element.260 = pred[] get-tuple-element(tuple.0), index=0
  reshape.0 = pred[] reshape(and.0)
} // cond_wrapper.257

cond_1_Sum-reduction.261 {
  x.262 = f32[] parameter(0)
  y.263 = f32[] parameter(1)
  ROOT add.264 = f32[] add(x.262, y.263)
}

cond_1_true_36__.265 {
  arg_tuple.266 = (s32[], f32[250]{0}) parameter(0)
  get-tuple-element.267 = s32[] get-tuple-element(arg_tuple.266), index=0
  reshape.269 = s32[1]{0} reshape(get-tuple-element.267)
  concatenate.270 = s32[1]{0} concatenate(reshape.269), dimensions={0}
  slice.280 = s32[1]{0} slice(concatenate.270), slice={[0:1]}
  reshape.281 = s32[] reshape(reshape.269)
  constant.275 = s32[] constant(0)
  compare.282 = pred[] compare(get-tuple-element.267, constant.275), direction=LT
  constant.276 = s32[] constant(250)
  add.283 = s32[] add(constant.276, get-tuple-element.267)
  select.284 = s32[] select(compare.282, add.283, get-tuple-element.267)
  constant.277 = s32[1]{0} constant({0})
  slice.278 = s32[1]{0} slice(constant.277), slice={[0:1]}
  reshape.279 = s32[] reshape(slice.278)
  subtract.285 = s32[] subtract(select.284, reshape.279)
  maximum.286 = s32[] maximum(subtract.285, constant.275)
  convert.287 = s32[] convert(maximum.286)
  get-tuple-element.268 = f32[250]{0} get-tuple-element(arg_tuple.266), index=1
  constant.271 = f32[] constant(0)
  pad.272 = f32[500]{0} pad(get-tuple-element.268, constant.271), padding=0_250
  constant.273 = s32[] constant(500)
  set-dimension-size.274 = f32[500]{0} set-dimension-size(pad.272, constant.273), dimensions={0}
  dynamic-slice.288 = f32[250]{0} dynamic-slice(set-dimension-size.274, reshape.279), dynamic_slice_sizes={250}
  reshape.289 = f32[250]{0} reshape(dynamic-slice.288)
  set-dimension-size.290 = f32[<=250]{0} set-dimension-size(dynamic-slice.288, maximum.286), dimensions={0}
  get-dimension-size.291 = s32[] get-dimension-size(set-dimension-size.290), dimensions={0}
  convert.292 = s32[] convert(get-dimension-size.291)
  broadcast.293 = s32[1]{0} broadcast(get-dimension-size.291), dimensions={}
  concatenate.294 = s32[1]{0} concatenate(broadcast.293), dimensions={0}
  slice.295 = s32[1]{0} slice(concatenate.294), slice={[0:1]}
  reshape.296 = s32[] reshape(broadcast.293)
  constant.309 = s32[] constant(0)
  constant.310 = s32[] constant(0)
  constant.312 = f32[] constant(0)
  broadcast.313 = f32[250]{0} broadcast(constant.312), dimensions={}
  constant.302 = s32[] constant(0)
  broadcast.303 = s32[250]{0} broadcast(constant.302), dimensions={}
  set-dimension-size.304 = s32[<=250]{0} set-dimension-size(broadcast.303, get-dimension-size.291), dimensions={0}
  get-dimension-size.311 = s32[] get-dimension-size(set-dimension-size.304), dimensions={0}
  set-dimension-size.314 = f32[<=250]{0} set-dimension-size(broadcast.313, get-dimension-size.311), dimensions={0}
  constant.315 = s32[] constant(0)
  tuple.316 = (f32[<=250]{0}, s32[]) tuple(set-dimension-size.314, constant.315)
  constant.305 = s32[] constant(250)
  tuple.306 = (f32[<=250]{0}, s32[]) tuple(set-dimension-size.290, constant.305)
  tuple.317 = (s32[], s32[], s32[], (f32[<=250]{0}, s32[]), s32[], /*index=5*/(f32[<=250]{0}, s32[])) tuple(constant.309, get-dimension-size.291, constant.310, tuple.316, get-dimension-size.291, /*index=5*/tuple.306)
  while.318 = (s32[], s32[], s32[], (f32[<=250]{0}, s32[]), s32[], /*index=5*/(f32[<=250]{0}, s32[])) while(tuple.317), condition=cond_wrapper.257, body=cond_1_map_while_body_59__.210
  get-tuple-element.319 = s32[] get-tuple-element(while.318), index=0
  get-tuple-element.320 = s32[] get-tuple-element(while.318), index=1
  get-tuple-element.321 = s32[] get-tuple-element(while.318), index=2
  get-tuple-element.322 = (f32[<=250]{0}, s32[]) get-tuple-element(while.318), index=3
  get-tuple-element.323 = s32[] get-tuple-element(while.318), index=4
  get-tuple-element.324 = (f32[<=250]{0}, s32[]) get-tuple-element(while.318), index=5
  tuple.325 = (s32[], s32[], s32[], (f32[<=250]{0}, s32[]), s32[], /*index=5*/(f32[<=250]{0}, s32[])) tuple(get-tuple-element.319, get-tuple-element.320, get-tuple-element.321, get-tuple-element.322, get-tuple-element.323, /*index=5*/get-tuple-element.324)
  get-tuple-element.329 = (f32[<=250]{0}, s32[]) get-tuple-element(tuple.325), index=3
  get-tuple-element.332 = f32[<=250]{0} get-tuple-element(get-tuple-element.329), index=0
  convert.333 = f32[<=250]{0} convert(get-tuple-element.332)
  constant.334 = f32[] constant(0)
  convert.335 = f32[] convert(constant.334)
  reduce.336 = f32[] reduce(get-tuple-element.332, constant.334), dimensions={0}, to_apply=cond_1_Sum-reduction.261
  convert.337 = f32[] convert(reduce.336)
  reshape.338 = f32[] reshape(reduce.336)
  ROOT tuple.339 = (f32[]) tuple(reduce.336)
} // cond_1_true_36__.265

cond_1_false_37__.340 {
  arg_tuple.341 = (s32[], f32[250]{0}) parameter(0)
  constant.344 = f32[] constant(0)
  reshape.345 = f32[] reshape(constant.344)
  ROOT tuple.346 = (f32[]) tuple(constant.344)
}

ENTRY tfcompile.377 {
  arg6.7 = s32[] parameter(6), parameter_replication={false}
  arg0.1 = s32[] parameter(0), parameter_replication={false}
  reshape.9 = s32[] reshape(arg0.1)
  arg1.2 = f32[250]{0} parameter(1), parameter_replication={false}
  reshape.10 = f32[250]{0} reshape(arg1.2)
  arg2.3 = pred[] parameter(2), parameter_replication={false}
  reshape.11 = pred[] reshape(arg2.3)
  arg3.4 = pred[] parameter(3), parameter_replication={false}
  reshape.12 = pred[] reshape(arg3.4)
  arg4.5 = s32[] parameter(4), parameter_replication={false}
  reshape.13 = s32[] reshape(arg4.5)
  arg5.6 = pred[] parameter(5), parameter_replication={false}
  reshape.14 = pred[] reshape(arg5.6)
  arg7.8 = pred[] parameter(7), parameter_replication={false}
  reshape.16 = pred[] reshape(arg7.8)
  tuple.1 = (s32[], f32[250]{0}) tuple(arg0.1, arg1.2)
  conditional.0 = (f32[], s32[]) conditional(arg2.3, tuple.1, tuple.1), true_computation=cond_2_true_195__.31, false_computation=cond_2_false_196__.76
  get-tuple-element.4 = f32[] get-tuple-element(conditional.0), index=0
  reshape.1 = f32[1]{0} reshape(get-tuple-element.4)
  get-tuple-element.5 = s32[] get-tuple-element(conditional.0), index=1
  convert.0 = f32[] convert(get-tuple-element.5)
  reshape.2 = f32[1]{0} reshape(convert.0)
  tuple.2 = (pred[], pred[], pred[]) tuple(arg3.4, arg5.6, arg7.8)
  conditional.1 = (pred[]) conditional(arg3.4, tuple.2, tuple.2), true_computation=cond_true_10__.85, false_computation=cond_false_11__.104
  get-tuple-element.6 = pred[] get-tuple-element(conditional.1), index=0
  tuple.3 = (s32[], f32[250]{0}) tuple(arg4.5, arg1.2)
  conditional.2 = (f32[]) conditional(get-tuple-element.6, tuple.3, tuple.3), true_computation=cond_1_true_36__.265, false_computation=cond_1_false_37__.340
  get-tuple-element.7 = f32[] get-tuple-element(conditional.2), index=0
  reshape.3 = f32[1]{0} reshape(get-tuple-element.7)
  concatenate.0 = f32[3]{0} concatenate(reshape.1, reshape.2, reshape.3), dimensions={0}
  tuple.4 = (f32[3]{0}) tuple(concatenate.0)
  get-tuple-element.374 = f32[3]{0} get-tuple-element(tuple.4), index=0
  reshape.375 = f32[3]{0} reshape(get-tuple-element.374)
  ROOT tuple.376 = (f32[3]{0}) tuple(get-tuple-element.374)
  reshape.4 = f32[3]{0} reshape(concatenate.0)
} // tfcompile.377
)";

  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo));

  TF_ASSERT_OK(RunInference());
}

}  // namespace
}  // namespace xla
