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
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
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

  Status RunInference() {
    hlo_graph_dumper::MaybeDumpHloModule(*module_, "Before alias analysis");
    TF_ASSIGN_OR_RETURN(DynamicDimensionInference inference,
                        DynamicDimensionInference::Run(module_.get()));

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

  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(reduce, {}, 1), size_param);
  EXPECT_EQ(inference_->GetDynamicSize(reduce, {}, 0), nullptr);
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

  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(dot, {}, 0), size_param);
  EXPECT_EQ(inference_->GetDynamicSize(dot, {}, 1), nullptr);
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
      zx_shape, a_param, b_param, /*feature_group_count=*/1, window, dnums,
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

  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(transpose, {}, 0), size_param_3);
  EXPECT_EQ(inference_->GetDynamicSize(transpose, {}, 1), size_param_2);
  EXPECT_EQ(inference_->GetDynamicSize(transpose, {}, 2), size_param_1);
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

  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(reshape, {}, 0), nullptr);
  EXPECT_EQ(inference_->GetDynamicSize(reshape, {}, 1), size_param);
  EXPECT_EQ(inference_->GetDynamicSize(reshape, {}, 2), nullptr);
  EXPECT_EQ(inference_->GetDynamicSize(reshape, {}, 3), size_param);
  EXPECT_EQ(inference_->GetDynamicSize(reshape, {}, 4), nullptr);
  EXPECT_EQ(inference_->GetDynamicSize(reshape, {}, 5), nullptr);
}

TEST_F(DynamicDimensionInferenceTest, ReshapeTestUnimplemented) {
  // Test the ability to trace unmodified reshape dimensions.
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {2, 3, 4, 5, 6});
  auto output_shape = ShapeUtil::MakeShape(F32, {6, 4, 1, 5, 2, 3});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, input_shape, "A"));

  builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, scalar_shape_, "size_param"));

  builder.AddInstruction(HloInstruction::CreateReshape(output_shape, a_param));

  module_->AddEntryComputation(builder.Build());

  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 1}));

  Status status = RunInference();
  EXPECT_EQ(status.code(), tensorflow::error::UNIMPLEMENTED);
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

  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(broadcast, {}, 0), nullptr);
  EXPECT_EQ(inference_->GetDynamicSize(broadcast, {}, 1), size_param);
  EXPECT_EQ(inference_->GetDynamicSize(broadcast, {}, 2), nullptr);
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

  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(reduce_window, {}, 0), size_param);
}

TEST_F(DynamicDimensionInferenceTest, SelectAndScatterTest) {
  // Test the ability to trace select and scatter batch dimensions.
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

  TF_ASSERT_OK(RunInference());
  EXPECT_EQ(inference_->GetDynamicSize(reduce_window, {}, 0), size_param);
}

}  // namespace
}  // namespace xla
