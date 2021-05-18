/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/dynamic_dimension_inference.h"

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

class DynamicDimensionInferenceTest : public HloTestBase {
 protected:
  DynamicDimensionInferenceTest() : HloTestBase() {
    module_ = CreateNewVerifiedModule();
  }

  Status RunInference(
      DynamicDimensionInference::CustomCallInferenceHandler handler = nullptr) {
    TF_ASSIGN_OR_RETURN(DynamicDimensionInference inference,
                        DynamicDimensionInference::Run(module_.get(), handler));

    inference_ = absl::make_unique<DynamicDimensionInference>(inference);
    return Status::OK();
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

  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "param"));
  auto param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "param"));

  module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());

  // Set up dynamic parameter binding.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 1}));

  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(param, {}, 1), param2);
  EXPECT_EQ(inference_->GetDynamicSize(param, {}, 0), nullptr);
  EXPECT_EQ(inference_->GetDynamicSize(param2, {}, 0), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, ParamTestTuple) {
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});

  auto param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeTupleShape({input_shape, scalar_shape_}), "param"));

  module_->AddEntryComputation(builder.Build());
  // Set up dynamic parameter binding.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{0, {1}},
      DynamicParameterBinding::DynamicDimension{0, {0}, 1}));

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_THAT(inference_->GetDynamicSize(param, {0}, 1),
              op::GetTupleElement(param, 1));

  EXPECT_EQ(inference_->GetDynamicSize(param, {0}, 0), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, GetTupleElement) {
  // When data flows through GTE, the dynamic dimension size keeps the
  // same, and the index has its front popped.
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});

  auto param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeTupleShape({input_shape, scalar_shape_}), "param"));

  auto gte = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(input_shape, param, 0));

  module_->AddEntryComputation(builder.Build());
  // Set up dynamic parameter binding.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{0, {1}},
      DynamicParameterBinding::DynamicDimension{0, {0}, 1}));

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_THAT(inference_->GetDynamicSize(param, {0}, 1),
              op::GetTupleElement(param, 1));

  EXPECT_THAT(inference_->GetDynamicSize(gte, {}, 1),
              op::GetTupleElement(param, 1));

  EXPECT_EQ(inference_->GetDynamicSize(param, {0}, 0), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, ElementwiseTest) {
  // When data flows through elementwise, the dynamic dimension size keeps the
  // same.
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});

  auto data_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "data_param"));
  auto size_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "size_param"));

  auto* negate = builder.AddInstruction(
      HloInstruction::CreateUnary(input_shape, HloOpcode::kNegate, data_param));

  module_->AddEntryComputation(builder.Build());
  // Set up dynamic parameter binding.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 1}));

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(negate, {}, 1), size_param);
}

TEST_F(DynamicDimensionInferenceTest, ReduceTestI) {
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
  auto reduce_shape = ShapeUtil::MakeShape(F32, {2});

  auto data_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "data_param"));
  auto size_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "size_param"));

  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(input_shape, HloOpcode::kNegate, data_param));

  auto init = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0)));

  auto reduce = builder.AddInstruction(HloInstruction::CreateReduce(
      reduce_shape, negate, init, {0, 2}, GetAdd()));

  module_->AddEntryComputation(builder.Build());

  // Set up dynamic parameter binding.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 1}));

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(reduce, {}, 0), size_param);
}

TEST_F(DynamicDimensionInferenceTest, ReduceTestII) {
  // Same as ReduceTestI, but only reduce one dimension.
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
  auto reduce_shape = ShapeUtil::MakeShape(F32, {1, 2});

  auto data_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "data_param"));
  auto size_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "size_param"));

  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(input_shape, HloOpcode::kNegate, data_param));

  auto init = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0)));

  auto reduce = builder.AddInstruction(
      HloInstruction::CreateReduce(reduce_shape, negate, init, {1}, GetAdd()));

  module_->AddEntryComputation(builder.Build());

  // Set up dynamic parameter binding.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 2}));

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(reduce, {}, 1), size_param);
  EXPECT_EQ(inference_->GetDynamicSize(reduce, {}, 0), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, VariadicReduce) {
  // Handle variadic reduce where output is a tuple.
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
  auto reduce_shape = ShapeUtil::MakeShape(F32, {1, 2});

  auto data_param_dynamic = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "data_param"));
  auto data_param_static = builder.AddInstruction(
      HloInstruction::CreateParameter(1, input_shape, "data_param.2"));
  auto size_param = builder.AddInstruction(
      HloInstruction::CreateParameter(2, scalar_shape_, "size_param"));

  // Set up dynamic parameter binding.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{2, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 2}));

  auto dynamic_negate = builder.AddInstruction(HloInstruction::CreateUnary(
      input_shape, HloOpcode::kNegate, data_param_dynamic));

  auto static_negate = builder.AddInstruction(HloInstruction::CreateUnary(
      input_shape, HloOpcode::kNegate, data_param_static));

  auto init = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0)));

  auto reduce = builder.AddInstruction(HloInstruction::CreateReduce(
      ShapeUtil::MakeTupleShape({reduce_shape, reduce_shape}),
      {dynamic_negate, static_negate}, {init, init}, {1}, GetAddTuple()));

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
  auto xz_shape = ShapeUtil::MakeShape(F32, {xdim, zdim});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, xy_shape, "A"));
  auto* b_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, yz_shape, "B"));
  auto* size_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, scalar_shape_, "size_param"));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(
      HloInstruction::CreateDot(xz_shape, a_param, b_param, dot_dnums,
                                HloTestBase::DefaultPrecisionConfig(2)));

  module_->AddEntryComputation(builder.Build());

  // Set up dynamic parameter binding for non-contracting dimension.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{2, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 0}));

  // Set up binding for contracting dimensions.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{2, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 1}));
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{2, {}},
      DynamicParameterBinding::DynamicDimension{1, {}, 0}));

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(dot, {}, 0), size_param);
  EXPECT_EQ(inference_->GetDynamicSize(dot, {}, 1), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, DotTestBatch) {
  auto builder = HloComputation::Builder(TestName());
  auto lhs_shape = ShapeUtil::MakeShape(F32, {4, 128, 2, 8});
  auto rhs_shape = ShapeUtil::MakeShape(F32, {4, 128, 2, 8});
  auto output_shape = ShapeUtil::MakeShape(F32, {4, 2, 128, 128});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, lhs_shape, "A"));
  auto* b_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, rhs_shape, "B"));
  auto* size_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, scalar_shape_, "size_param"));

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

  // Set up dynamic parameter binding for batch dimension.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{2, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 0}));

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

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, lhs_shape, "A"));
  auto* b_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, rhs_shape, "B"));
  builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, scalar_shape_, "size_param"));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(1);
  auto dot = builder.AddInstruction(
      HloInstruction::CreateDot(output_shape, a_param, b_param, dot_dnums,
                                HloTestBase::DefaultPrecisionConfig(2)));

  module_->AddEntryComputation(builder.Build());

  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{2, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 0}));

  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{2, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 1}));
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{2, {}},
      DynamicParameterBinding::DynamicDimension{1, {}, 0}));

  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{2, {}},
      DynamicParameterBinding::DynamicDimension{1, {}, 1}));

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
  auto zx_shape = ShapeUtil::MakeShape(F32, {zdim, xdim});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, xy_shape, "A"));
  auto* b_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, yz_shape, "B"));
  auto* size_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, scalar_shape_, "size_param"));

  auto dnums = XlaBuilder::CreateDefaultConvDimensionNumbers(0);

  dnums.set_kernel_input_feature_dimension(0);
  dnums.set_kernel_output_feature_dimension(1);
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(1);
  dnums.set_output_feature_dimension(0);

  Window window;

  auto* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      zx_shape, a_param, b_param, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums,
      HloTestBase::DefaultPrecisionConfig(2)));

  module_->AddEntryComputation(builder.Build());

  // Set up dynamic parameter binding for non-contracting dimension.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{2, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 0}));

  // Set up binding for contracting dimensions.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{2, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 1}));

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(conv, {}, 1), size_param);
  EXPECT_EQ(inference_->GetDynamicSize(conv, {}, 0), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, TransposeTest) {
  // Test the ability to trace unmodified dimensions
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {1, 2, 3});
  auto output_shape = ShapeUtil::MakeShape(F32, {3, 2, 1});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, input_shape, "A"));
  auto* size_param_1 = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, scalar_shape_, "size_param"));
  auto* size_param_2 = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, scalar_shape_, "size_param"));
  auto* size_param_3 = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/3, scalar_shape_, "size_param"));

  auto* transpose = builder.AddInstruction(
      HloInstruction::CreateTranspose(output_shape, a_param, {2, 1, 0}));

  module_->AddEntryComputation(builder.Build());

  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 0}));

  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{2, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 1}));

  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{3, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 2}));

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
  auto output_shape = ShapeUtil::MakeShape(F32, {3, 1, 2});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, input_shape, "A"));
  auto* size_param_1 = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, scalar_shape_, "size_param"));
  auto* size_param_2 = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, scalar_shape_, "size_param"));
  auto* size_param_3 = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/3, scalar_shape_, "size_param"));

  auto* transpose = builder.AddInstruction(
      HloInstruction::CreateTranspose(output_shape, a_param, {2, 0, 1}));

  module_->AddEntryComputation(builder.Build());

  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 0}));

  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{2, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 1}));

  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{3, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 2}));

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
  auto output_shape = ShapeUtil::MakeShape(F32, {6, 4, 1, 5, 2, 3});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, input_shape, "A"));
  auto* size_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, scalar_shape_, "size_param"));

  auto* reshape = builder.AddInstruction(
      HloInstruction::CreateReshape(output_shape, a_param));

  module_->AddEntryComputation(builder.Build());

  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 2}));

  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 3}));

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
  auto output_shape = ShapeUtil::MakeShape(F32, {1, 4, 5});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, input_shape, "A"));
  builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, scalar_shape_, "size_param"));

  auto* reshape = builder.AddInstruction(HloInstruction::CreateReshape(
      output_shape, a_param, /*inferred_dimension=*/0));

  module_->AddEntryComputation(builder.Build());

  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 0}));

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_NE(inference_->GetDynamicSize(reshape, {}, 0), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, ReshapeTestMajorDimension) {
  // Test the ability to trace dimension combining.
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {32, 10, 4});
  auto output_shape = ShapeUtil::MakeShape(F32, {320, 4});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, input_shape, "A"));

  builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, scalar_shape_, "size_param"));

  auto* reshape = builder.AddInstruction(
      HloInstruction::CreateReshape(output_shape, a_param));

  module_->AddEntryComputation(builder.Build());

  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 0}));

  SCOPED_TRACE(module_->ToString());
  Status status = RunInference();
  EXPECT_NE(inference_->GetDynamicSize(reshape, {}, 0), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, ReshapeIntoScalar) {
  // Test the ability to a reshape into scalar.
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {1});
  auto output_shape = ShapeUtil::MakeShape(F32, {});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, input_shape, "A"));

  builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, scalar_shape_, "size_param"));

  builder.AddInstruction(HloInstruction::CreateReshape(output_shape, a_param));

  module_->AddEntryComputation(builder.Build());

  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 0}));

  SCOPED_TRACE(module_->ToString());
  TF_CHECK_OK(RunInference());
}

TEST_F(DynamicDimensionInferenceTest, GatherTest) {
  const string hlo_text = R"(
HloModule TensorFlowGatherV2

ENTRY main {
  operand = s32[20,10]{1,0} parameter(0)
  indices = s32[32,20] parameter(1)
  dynamic_size = s32[] parameter(2)
  ROOT gather = s32[32,20,10]{2,1,0} gather(%operand, %indices),
                 offset_dims={2},
                 collapsed_slice_dims={0},
                 start_index_map={0},
                 index_vector_dim=2,
                 slice_sizes={1,10}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_text));
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{2, {}},
      DynamicParameterBinding::DynamicDimension{1, {}, 0}));
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
  auto output_shape = ShapeUtil::MakeShape(F32, {3, 2, 4});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, input_shape, "A"));
  auto* size_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, scalar_shape_, "size_param"));

  auto* broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(output_shape, a_param, {1}));

  module_->AddEntryComputation(builder.Build());

  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 0}));

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
  auto output_shape = ShapeUtil::MakeShape(F32, {2, 2, 2});
  auto tuple_shape = ShapeUtil::MakeTupleShape({input_shape, input_shape});

  // Body:
  //
  //   Param
  //   |  |
  // GTE1 GTE2
  //   |  |
  //    ADD
  auto body_builder = HloComputation::Builder("body");
  auto body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  auto gte_0 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(input_shape, body_param, 0));
  auto gte_1 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(input_shape, body_param, 1));
  auto add = body_builder.AddInstruction(
      HloInstruction::CreateBinary(input_shape, HloOpcode::kAdd, gte_0, gte_1));
  body_builder.AddInstruction(HloInstruction::CreateTuple({add, add}));

  HloComputation* body = module_->AddEmbeddedComputation(body_builder.Build());

  auto cond_builder = HloComputation::Builder("condition");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
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
  builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, scalar_shape_, "size_param"));
  builder.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, condition, body, a_param));

  module_->AddEntryComputation(builder.Build());

  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {0}, 0}));

  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {1}, 0}));

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
  // Test the ability to trace into contional loops.
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {2, 4, 4});
  auto output_shape = ShapeUtil::MakeShape(F32, {2, 2, 2});
  // In this test we set inputs to different branches to different shapes.
  auto tuple_shape_1 = ShapeUtil::MakeTupleShape({input_shape});
  auto tuple_shape_2 = ShapeUtil::MakeTupleShape({input_shape, input_shape});
  auto tuple_shape_3 =
      ShapeUtil::MakeTupleShape({input_shape, input_shape, input_shape});

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
        HloInstruction::CreateParameter(0, tuple_shape_2, "param"));
    auto gte_0 = true_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(input_shape, true_param, 0));
    auto gte_1 = true_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(input_shape, true_param, 1));
    auto add = true_builder.AddInstruction(HloInstruction::CreateBinary(
        input_shape, HloOpcode::kAdd, gte_0, gte_1));
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
        HloInstruction::CreateParameter(0, tuple_shape_3, "param"));
    auto gte_0 = false_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(input_shape, false_param, 1));
    auto gte_1 = false_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(input_shape, false_param, 2));
    auto add = false_builder.AddInstruction(HloInstruction::CreateBinary(
        input_shape, HloOpcode::kAdd, gte_0, gte_1));
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
  builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/3, scalar_shape_, "size_param"));
  builder.AddInstruction(HloInstruction::CreateConditional(
      tuple_shape_1, pred_param, tuple_2_param, true_branch, tuple_3_param,
      false_branch));

  module_->AddEntryComputation(builder.Build());

  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{3, {}},
      DynamicParameterBinding::DynamicDimension{1, {0}, 0}));
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{3, {}},
      DynamicParameterBinding::DynamicDimension{1, {1}, 0}));
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{3, {}},
      DynamicParameterBinding::DynamicDimension{2, {1}, 0}));
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{3, {}},
      DynamicParameterBinding::DynamicDimension{2, {2}, 0}));

  TF_ASSERT_OK(RunInference());

  HloInstruction* conditional_hlo = nullptr;
  // The while hlo has been replaced, find the new one.
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
  auto output_shape = ShapeUtil::MakeShape(F32, {2, 2, 2});

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
  for (int64 i = 0; i < 2; ++i) {
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

  auto init = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0)));

  auto* reduce_window =
      builder.AddInstruction(HloInstruction::CreateReduceWindow(
          output_shape, a_param, init, window, GetAdd()));

  module_->AddEntryComputation(builder.Build());

  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 0}));

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(reduce_window, {}, 0), size_param);
}

TEST_F(DynamicDimensionInferenceTest, SelectAndScatterTest) {
  // Test the ability to trace select and scatter batch dimensions.
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {2, 4, 4});
  auto source_shape = ShapeUtil::MakeShape(F32, {2, 2, 2});

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
  for (int64 i = 0; i < 2; ++i) {
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

  auto init = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0)));

  auto* sns = builder.AddInstruction(HloInstruction::CreateSelectAndScatter(
      input_shape, a_param, GetGe(), window, source, init, GetAdd()));

  module_->AddEntryComputation(builder.Build());

  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 0}));
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{2, {}, 0}));

  SCOPED_TRACE(module_->ToString());
  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(sns, {}, 0), size_param);
}

TEST_F(DynamicDimensionInferenceTest, ConcateTest) {
  // Concat two data params.
  auto builder = HloComputation::Builder(TestName());

  auto data_param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {5, 7}), "data_param_1"));
  auto data_param_2 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {5, 8}), "data_param_2"));
  auto size_param = builder.AddInstruction(
      HloInstruction::CreateParameter(2, scalar_shape_, "size_param"));

  auto* concat = builder.AddInstruction(HloInstruction::CreateConcatenate(
      ShapeUtil::MakeShape(F32, {5, 15}), {data_param, data_param_2}, 1));

  module_->AddEntryComputation(builder.Build());
  // Set up dynamic parameter binding.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{2, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 0}));

  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{2, {}},
      DynamicParameterBinding::DynamicDimension{1, {}, 0}));

  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(concat, {}, 0), size_param);
}

TEST_F(DynamicDimensionInferenceTest, SliceTest) {
  auto builder = HloComputation::Builder(TestName());

  auto data_param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {5, 7}), "data_param"));
  auto size_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "size_param"));

  auto* slice = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {5, 7}), data_param, /*start_indices=*/{0, 0},
      /*limit_indices=*/{5, 7}, /*strides=*/{1, 1}));

  module_->AddEntryComputation(builder.Build());
  // Set up dynamic parameter binding.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 1}));

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

  auto* slice = builder.AddInstruction(HloInstruction::CreateDynamicSlice(
      ShapeUtil::MakeShape(F32, {5, 1}), data_param, params,
      /*slice_sizes=*/{5, 1}));

  module_->AddEntryComputation(builder.Build());
  // Set up dynamic parameter binding.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 0}));

  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(slice, {}, 0), size_param);
}

TEST_F(DynamicDimensionInferenceTest, SortTest) {
  auto builder = HloComputation::Builder(TestName());

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

  auto* sort = builder.AddInstruction(HloInstruction::CreateSort(
      ShapeUtil::MakeShape(F32, {5, 7}), 1, {data_param}, compare,
      /*is_stable=*/false));

  module_->AddEntryComputation(builder.Build());
  // Set up dynamic parameter binding.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 0}));

  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(sort, {}, 0), size_param);
}

TEST_F(DynamicDimensionInferenceTest, MultiValueSortTest) {
  auto builder = HloComputation::Builder(TestName());

  auto shape = ShapeUtil::MakeShape(F32, {5, 7});

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

  auto* sort = builder.AddInstruction(
      HloInstruction::CreateSort(ShapeUtil::MakeTupleShape({shape, shape}), 1,
                                 {data_param, data_param}, compare,
                                 /*is_stable=*/false));

  module_->AddEntryComputation(builder.Build());
  // Set up dynamic parameter binding.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 0}));

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
  builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "size_param"));

  std::vector<HloInstruction*> params;
  for (int i = 0; i < 2; ++i) {
    params.push_back(builder.AddInstruction(HloInstruction::CreateParameter(
        i + 2, ShapeUtil::MakeShape(S32, {}), "slice_indices")));
  }

  auto* slice = builder.AddInstruction(HloInstruction::CreateDynamicSlice(
      ShapeUtil::MakeShape(F32, {1, 1}), data_param, params,
      /*slice_sizes=*/{1, 1}));

  module_->AddEntryComputation(builder.Build());
  // Set up dynamic parameter binding.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 0}));

  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(slice, {}, 0), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, InfersCustomOp) {
  auto builder = HloComputation::Builder(TestName());

  auto data_param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {5, 7}), "data_param"));
  builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "size_param"));

  builder.AddInstruction(HloInstruction::CreateCustomCall(
      ShapeUtil::MakeShape(F32, {1, 1}), {data_param}, "MyCustomOp", ""));

  module_->AddEntryComputation(builder.Build());

  // Set up dynamic parameter binding.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 0}));

  bool handler_called = false;
  auto handler = [&](HloInstruction* hlo,
                     DynamicDimensionInference* inference) {
    CHECK(inference != nullptr);
    CHECK(Cast<HloCustomCallInstruction>(hlo) != nullptr);
    handler_called = true;
    return Status::OK();
  };
  TF_ASSERT_OK(RunInference(handler));

  EXPECT_TRUE(handler_called);
}

TEST_F(DynamicDimensionInferenceTest, DynamicReshapeOp) {
  auto builder = HloComputation::Builder(TestName());
  auto input = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {9}), "data_input"));
  auto six = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(6)));
  // Creates an input of shape [<=9], dynamic size is 6.
  auto dynamic_input =
      builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
          ShapeUtil::MakeShape(F32, {9}, {true}), input, six, 0));
  auto dynamic_size = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(S32, {}), "size_param"));
  auto three = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(3)));

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
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(6)));
  input = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      ShapeUtil::MakeShape(F32, {9, 2}, {true, false}), input, six, 0));
  auto one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));
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

}  // namespace
}  // namespace xla
