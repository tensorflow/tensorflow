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

#include "tensorflow/compiler/xla/service/shape_inference.h"

#include <string>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::ContainsRegex;
using ::testing::HasSubstr;

class ShapeInferenceTest : public ::testing::Test {
 protected:
  // Some handy scalar shapes.
  const Shape s32_ = ShapeUtil::MakeShape(S32, {});
  const Shape f16_ = ShapeUtil::MakeShape(F16, {});
  const Shape f32_ = ShapeUtil::MakeShape(F32, {});
  const Shape f64_ = ShapeUtil::MakeShape(F64, {});
  const Shape pred_ = ShapeUtil::MakeShape(PRED, {});

  // Some handy vector and matrix shapes of F32 type.
  // Suffix: vector_length_, matrix_rows_cols_
  const Shape vector_32_ = ShapeUtil::MakeShape(F32, {32});
  const Shape vector_64_ = ShapeUtil::MakeShape(F32, {64});
  const Shape matrix_32_48_ = ShapeUtil::MakeShape(F32, {32, 48});
  const Shape matrix_32_64_ = ShapeUtil::MakeShape(F32, {32, 64});
  const Shape matrix_64_48_ = ShapeUtil::MakeShape(F32, {64, 48});

  // Some handy S32 arrays.
  const Shape s32matrix_64_64_ = ShapeUtil::MakeShape(S32, {64, 64});
};

// Subclass for testing InferReduceShape.
class ReduceShapeInferenceTest : public ShapeInferenceTest {
 protected:
  // Helper that runs reduce shape inference with the input 'arg' and given
  // dimensions to reduce, and checks the inferred shape is as expected. The
  // element type here is hard-coded to F32.
  void ExpectInferredReduceShape(const Shape& expected_inferred_shape,
                                 const Shape& arg,
                                 absl::Span<const int64> dimensions_to_reduce) {
    ProgramShape to_apply = ShapeUtil::MakeProgramShape({f32_, f32_}, f32_);
    auto inferred_status = ShapeInference::InferReduceShape(
        {&arg, &f32_}, dimensions_to_reduce, to_apply);
    EXPECT_IS_OK(inferred_status.status());
    EXPECT_TRUE(ShapeUtil::Equal(expected_inferred_shape,
                                 inferred_status.ValueOrDie()));
  }
};

// Subclass for testing InferSelectAndScatterShape.
class SelectAndScatterShapeInferenceTest : public ShapeInferenceTest {
 protected:
  SelectAndScatterShapeInferenceTest() {
    operand_shape_ = ShapeUtil::MakeShape(F32, {8, 16});
    source_shape_ = ShapeUtil::MakeShape(F32, {4, 8});
    WindowDimension dim;
    dim.set_size(2);
    dim.set_stride(2);
    dim.set_padding_low(0);
    dim.set_padding_high(0);
    dim.set_window_dilation(1);
    dim.set_base_dilation(1);
    *window_.add_dimensions() = dim;
    *window_.add_dimensions() = dim;
    init_value_shape_ = ShapeUtil::MakeShape(F32, {});
    select_program_shape_ = ShapeUtil::MakeProgramShape(
        {ShapeUtil::MakeShape(F32, {}), ShapeUtil::MakeShape(F32, {})}, pred_);
    scatter_program_shape_ = ShapeUtil::MakeProgramShape(
        {ShapeUtil::MakeShape(F32, {}), ShapeUtil::MakeShape(F32, {})}, f32_);
  }

  Shape operand_shape_;
  Shape source_shape_;
  Window window_;
  Shape init_value_shape_;
  ProgramShape select_program_shape_;
  ProgramShape scatter_program_shape_;
};

TEST_F(ShapeInferenceTest, UnaryNegateMatrix) {
  Shape matrix_shape = ShapeUtil::MakeShape(F32, {128, 64});
  auto inferred_status =
      ShapeInference::InferUnaryOpShape(HloOpcode::kNegate, matrix_shape);
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_shape, inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, SelectScalarPredBetweenTuples) {
  Shape tuple = ShapeUtil::MakeTupleShape({s32_, f32_});
  auto inferred_status = ShapeInference::InferTernaryOpShape(
      HloOpcode::kSelect, pred_, tuple, tuple);
  ASSERT_FALSE(inferred_status.ok());
  ASSERT_THAT(inferred_status.status().error_message(),
              HasSubstr("Expected array argument for select"));
}

TEST_F(ShapeInferenceTest, SelectScalarPredBetweenArrays) {
  auto inferred_status = ShapeInference::InferTernaryOpShape(
      HloOpcode::kSelect, pred_, matrix_64_48_, matrix_64_48_);
  ASSERT_FALSE(inferred_status.ok());
  ASSERT_THAT(
      inferred_status.status().error_message(),
      HasSubstr("Operands to select and predicate must be the same shape"));
}

TEST_F(ShapeInferenceTest, SelectArrayPredBetweenArrays) {
  auto predarray = ShapeUtil::MakeShape(PRED, {64, 48});
  auto inferred_status = ShapeInference::InferTernaryOpShape(
      HloOpcode::kSelect, predarray, matrix_64_48_, matrix_64_48_);
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, SelectBadShapes) {
  auto inferred_status_error1 = ShapeInference::InferTernaryOpShape(
      HloOpcode::kSelect, pred_, matrix_64_48_, matrix_32_64_);
  ASSERT_FALSE(inferred_status_error1.ok());
  ASSERT_THAT(inferred_status_error1.status().error_message(),
              HasSubstr("Operands to select must be the same shape"));

  auto inferred_status_error2 = ShapeInference::InferTernaryOpShape(
      HloOpcode::kSelect, s32_, matrix_64_48_, matrix_64_48_);
  ASSERT_FALSE(inferred_status_error2.ok());
  ASSERT_THAT(inferred_status_error2.status().error_message(),
              HasSubstr("pred operand must have PRED"));

  auto inferred_status_error3 = ShapeInference::InferTernaryOpShape(
      HloOpcode::kSelect, ShapeUtil::MakeShape(PRED, {64}), matrix_64_48_,
      matrix_64_48_);
  ASSERT_FALSE(inferred_status_error3.ok());
  ASSERT_THAT(
      inferred_status_error3.status().error_message(),
      HasSubstr("Operands to select and predicate must be the same shape"));

  // Tuples have a TUPLE element type and cannot be the pred of a select.
  auto inferred_status_error4 = ShapeInference::InferTernaryOpShape(
      HloOpcode::kSelect, ShapeUtil::MakeTupleShape({pred_, pred_}),
      ShapeUtil::MakeTupleShape({f32_, f32_}),
      ShapeUtil::MakeTupleShape({f32_, f32_}));
  ASSERT_FALSE(inferred_status_error4.ok());
  ASSERT_THAT(inferred_status_error4.status().error_message(),
              HasSubstr("Expected array argument for select pred"));
}

TEST_F(ShapeInferenceTest, ClampAllMatrix) {
  auto inferred_status = ShapeInference::InferTernaryOpShape(
      HloOpcode::kClamp, matrix_64_48_, matrix_64_48_, matrix_64_48_);
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, ClampAllScalar) {
  auto inferred_status =
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, f32_, f32_, f32_);
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(f32_, inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, ClampMinScalar) {
  auto inferred_status = ShapeInference::InferTernaryOpShape(
      HloOpcode::kClamp, f32_, matrix_64_48_, matrix_64_48_);
  ASSERT_FALSE(inferred_status.ok());
  ASSERT_THAT(inferred_status.status().error_message(),
              HasSubstr("Clamp with different shapes"));
}

TEST_F(ShapeInferenceTest, ClampMaxScalar) {
  auto inferred_status = ShapeInference::InferTernaryOpShape(
      HloOpcode::kClamp, matrix_64_48_, matrix_64_48_, f32_);
  ASSERT_FALSE(inferred_status.ok());
  ASSERT_THAT(inferred_status.status().error_message(),
              HasSubstr("Clamp with different shapes"));
}

TEST_F(ShapeInferenceTest, ClampOperandScalar) {
  auto inferred_status = ShapeInference::InferTernaryOpShape(
      HloOpcode::kClamp, matrix_64_48_, f32_, matrix_64_48_);
  ASSERT_FALSE(inferred_status.ok());
  ASSERT_THAT(inferred_status.status().error_message(),
              HasSubstr("Clamp with different shapes"));
}

TEST_F(ShapeInferenceTest, ClampMinMatrix) {
  auto inferred_status = ShapeInference::InferTernaryOpShape(
      HloOpcode::kClamp, matrix_64_48_, f32_, f32_);
  ASSERT_FALSE(inferred_status.ok());
  ASSERT_THAT(inferred_status.status().error_message(),
              HasSubstr("Clamp with different shapes"));
}

TEST_F(ShapeInferenceTest, ClampMaxMatrix) {
  auto inferred_status = ShapeInference::InferTernaryOpShape(
      HloOpcode::kClamp, f32_, f32_, matrix_64_48_);
  ASSERT_FALSE(inferred_status.ok());
  ASSERT_THAT(inferred_status.status().error_message(),
              HasSubstr("Clamp with different shapes"));
}

TEST_F(ShapeInferenceTest, ClampOperandMatrix) {
  auto inferred_status = ShapeInference::InferTernaryOpShape(
      HloOpcode::kClamp, f32_, matrix_64_48_, f32_);
  ASSERT_FALSE(inferred_status.ok());
  ASSERT_THAT(inferred_status.status().error_message(),
              HasSubstr("Clamp with different shapes"));
}

TEST_F(ShapeInferenceTest, ClampBadShapes) {
  // Type mismatch
  ASSERT_FALSE(
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, s32_, f32_, f32_)
          .ok());
  ASSERT_FALSE(
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, f32_, s32_, f32_)
          .ok());
  ASSERT_FALSE(
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, f32_, f32_, s32_)
          .ok());
  // Dimension mismatch
  ASSERT_FALSE(ShapeInference::InferTernaryOpShape(
                   HloOpcode::kClamp, vector_64_, vector_32_, vector_32_)
                   .ok());
  ASSERT_FALSE(ShapeInference::InferTernaryOpShape(
                   HloOpcode::kClamp, vector_32_, vector_64_, vector_32_)
                   .ok());
  ASSERT_FALSE(ShapeInference::InferTernaryOpShape(
                   HloOpcode::kClamp, vector_32_, vector_32_, vector_64_)
                   .ok());
  // Dimension mismatch, where one operand is a scalar
  ASSERT_FALSE(ShapeInference::InferTernaryOpShape(HloOpcode::kClamp,
                                                   vector_64_, vector_32_, f32_)
                   .ok());
  ASSERT_FALSE(ShapeInference::InferTernaryOpShape(HloOpcode::kClamp,
                                                   vector_64_, f32_, vector_32_)
                   .ok());
  ASSERT_FALSE(ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, f32_,
                                                   vector_64_, vector_32_)
                   .ok());
}

TEST_F(ShapeInferenceTest, Complex) {
  auto complex_shape = [&](const Shape& lhs, const Shape& rhs,
                           absl::Span<const int64> bcast) {
    return ShapeInference::InferBinaryOpShape(HloOpcode::kComplex, lhs, rhs,
                                              bcast);
  };
  // Inputs must be FP.
  ASSERT_FALSE(complex_shape(s32_, s32_, {}).ok());
  ASSERT_FALSE(complex_shape(pred_, pred_, {}).ok());
  // Component types must match.
  ASSERT_FALSE(complex_shape(f32_, f64_, {}).ok());
  // Only F32->C64 and F64->C128 supported.
  ASSERT_FALSE(complex_shape(f16_, f16_, {}).ok());
  // Validate correct uses.
  Shape c64_32 = ShapeUtil::MakeShape(C64, {32});
  TF_ASSERT_OK_AND_ASSIGN(Shape result, complex_shape(f32_, f32_, {}));
  ASSERT_TRUE(ShapeUtil::Equal(result, ShapeUtil::MakeShape(C64, {})));
  TF_ASSERT_OK_AND_ASSIGN(result, complex_shape(vector_32_, f32_, {}));
  ASSERT_TRUE(ShapeUtil::Equal(result, c64_32));
  TF_ASSERT_OK_AND_ASSIGN(result, complex_shape(f32_, vector_32_, {}));
  ASSERT_TRUE(ShapeUtil::Equal(result, c64_32));
  TF_ASSERT_OK_AND_ASSIGN(result, complex_shape(vector_32_, f32_, {}));
  ASSERT_TRUE(ShapeUtil::Equal(result, c64_32));

  Shape c64_32_64 = ShapeUtil::MakeShape(C64, {32, 64});
  TF_ASSERT_OK_AND_ASSIGN(result,
                          complex_shape(vector_64_, matrix_32_64_, {1}));
  ASSERT_TRUE(ShapeUtil::Equal(result, c64_32_64));
  TF_ASSERT_OK_AND_ASSIGN(result,
                          complex_shape(matrix_32_64_, vector_64_, {1}));
  ASSERT_TRUE(ShapeUtil::Equal(result, c64_32_64));
  TF_ASSERT_OK_AND_ASSIGN(result,
                          complex_shape(matrix_32_64_, matrix_32_64_, {}));
  ASSERT_TRUE(ShapeUtil::Equal(result, c64_32_64));
  TF_ASSERT_OK_AND_ASSIGN(result, complex_shape(matrix_32_64_, f32_, {}));
  ASSERT_TRUE(ShapeUtil::Equal(result, c64_32_64));

  TF_ASSERT_OK_AND_ASSIGN(result, complex_shape(f64_, f64_, {}));
  ASSERT_TRUE(ShapeUtil::Equal(result, ShapeUtil::MakeShape(C128, {})));
}

TEST_F(ShapeInferenceTest, VariadicOpTuplify) {
  StatusOr<Shape> result =
      ShapeInference::InferVariadicOpShape(HloOpcode::kTuple, {&s32_, &f32_});
  ASSERT_IS_OK(result.status());
  ASSERT_TRUE(ShapeUtil::Equal(result.ValueOrDie(),
                               ShapeUtil::MakeTupleShape({s32_, f32_})));
}

TEST_F(ShapeInferenceTest, ReduceWindowInHalf) {
  Shape matrix_shape = ShapeUtil::MakeShape(F32, {8, 8});
  Window window;
  WindowDimension dim;
  dim.set_size(2);
  dim.set_stride(2);
  dim.set_padding_low(0);
  dim.set_padding_high(0);
  dim.set_window_dilation(1);
  dim.set_base_dilation(1);
  *window.add_dimensions() = dim;
  *window.add_dimensions() = dim;
  Shape window_shape = ShapeUtil::MakeShape(F32, {2, 2});
  Shape init_value_shape = ShapeUtil::MakeShape(F32, {});
  Shape float_scalar = ShapeUtil::MakeShape(F32, {});
  ProgramShape to_apply = ShapeUtil::MakeProgramShape(
      {ShapeUtil::MakeShape(F32, {}), ShapeUtil::MakeShape(F32, {})}, f32_);
  auto inferred_status = ShapeInference::InferReduceWindowShape(
      matrix_shape, init_value_shape, window, to_apply);

  ASSERT_IS_OK(inferred_status.status());
  Shape inferred = inferred_status.ValueOrDie();
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {4, 4}), inferred));
}

TEST_F(SelectAndScatterShapeInferenceTest, SelectAndScatterProperShapes) {
  auto inferred_status_ok = ShapeInference::InferSelectAndScatterShape(
      operand_shape_, select_program_shape_, window_, source_shape_,
      init_value_shape_, scatter_program_shape_);
  ASSERT_IS_OK(inferred_status_ok.status());
  Shape inferred = inferred_status_ok.ValueOrDie();
  ASSERT_TRUE(ShapeUtil::Equal(operand_shape_, inferred));
}

TEST_F(SelectAndScatterShapeInferenceTest, SelectAndScatterWrongSourceShape) {
  Shape source_shape_fail = ShapeUtil::MakeShape(F32, {4, 6});
  auto inferred_status_fail = ShapeInference::InferSelectAndScatterShape(
      operand_shape_, select_program_shape_, window_, source_shape_fail,
      init_value_shape_, scatter_program_shape_);
  ASSERT_FALSE(inferred_status_fail.ok());
  ASSERT_THAT(inferred_status_fail.status().error_message(),
              HasSubstr("Source shape does not match"));
}

TEST_F(SelectAndScatterShapeInferenceTest, SelectAndScatterWrongSelectShape1) {
  ProgramShape select_program_shape_fail =
      ShapeUtil::MakeProgramShape({ShapeUtil::MakeShape(F32, {})}, pred_);
  auto inferred_status_fail = ShapeInference::InferSelectAndScatterShape(
      operand_shape_, select_program_shape_fail, window_, source_shape_,
      init_value_shape_, scatter_program_shape_);
  ASSERT_FALSE(inferred_status_fail.ok());
  ASSERT_THAT(inferred_status_fail.status().error_message(),
              HasSubstr("Select function must take 2 parameters"));
}

TEST_F(SelectAndScatterShapeInferenceTest, SelectAndScatterWrongSelectShape2) {
  ProgramShape select_program_shape_fail = ShapeUtil::MakeProgramShape(
      {ShapeUtil::MakeShape(F32, {}), ShapeUtil::MakeShape(F32, {})}, f32_);
  auto inferred_status_fail = ShapeInference::InferSelectAndScatterShape(
      operand_shape_, select_program_shape_fail, window_, source_shape_,
      init_value_shape_, scatter_program_shape_);
  ASSERT_FALSE(inferred_status_fail.ok());
  ASSERT_THAT(inferred_status_fail.status().error_message(),
              HasSubstr("Select function must have rank-0 PRED"));
}

TEST_F(SelectAndScatterShapeInferenceTest, SelectAndScatterWrongSelectShape3) {
  ProgramShape select_program_shape_fail = ShapeUtil::MakeProgramShape(
      {ShapeUtil::MakeShape(S32, {}), ShapeUtil::MakeShape(F32, {})}, pred_);
  auto inferred_status_fail = ShapeInference::InferSelectAndScatterShape(
      operand_shape_, select_program_shape_fail, window_, source_shape_,
      init_value_shape_, scatter_program_shape_);
  ASSERT_FALSE(inferred_status_fail.ok());
  ASSERT_THAT(inferred_status_fail.status().error_message(),
              HasSubstr("Select function's first parameter"));
}

TEST_F(SelectAndScatterShapeInferenceTest, SelectAndScatterWrongSelectShape4) {
  ProgramShape select_program_shape_fail = ShapeUtil::MakeProgramShape(
      {ShapeUtil::MakeShape(F32, {}), ShapeUtil::MakeShape(U32, {})}, pred_);
  auto inferred_status_fail = ShapeInference::InferSelectAndScatterShape(
      operand_shape_, select_program_shape_fail, window_, source_shape_,
      init_value_shape_, scatter_program_shape_);
  ASSERT_FALSE(inferred_status_fail.ok());
  ASSERT_THAT(inferred_status_fail.status().error_message(),
              HasSubstr("Select function's second parameter"));
}

TEST_F(ShapeInferenceTest, Convolve) {
  ConvolutionDimensionNumbers dnums;

  // Dimension order: batch, feature, x0, x1
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {10, 11, 3, 4});
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(0);
  dnums.set_input_feature_dimension(1);
  dnums.set_output_feature_dimension(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);
  dnums.add_input_spatial_dimensions(3);
  dnums.add_output_spatial_dimensions(3);

  // Dimension order: x1, batch, feature, x0
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {2, 12, 11, 3});
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(1);
  dnums.add_kernel_spatial_dimensions(3);
  dnums.add_kernel_spatial_dimensions(0);

  Window window;
  auto dim0 = window.add_dimensions();
  auto dim1 = window.add_dimensions();
  dim0->set_size(3);
  dim0->set_stride(2);
  dim0->set_padding_low(1);
  dim0->set_padding_high(1);
  dim0->set_window_dilation(1);
  dim0->set_base_dilation(1);
  dim1->set_size(2);
  dim1->set_stride(1);
  dim1->set_padding_low(0);
  dim1->set_padding_high(0);
  dim1->set_window_dilation(1);
  dim1->set_base_dilation(1);
  auto inferred_status = ShapeInference::InferConvolveShape(
      lhs_shape, rhs_shape, /*feature_group_count=*/1, /*batch_group_count=*/1,
      window, dnums);
  ASSERT_IS_OK(inferred_status.status());
  Shape inferred_shape = inferred_status.ValueOrDie();
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {10, 12, 2, 3}),
                               inferred_shape));
}

TEST_F(ShapeInferenceTest, ConvolveWithWindowDilation) {
  ConvolutionDimensionNumbers dnums;

  // Dimension order: batch, feature, x0, x1
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {10, 11, 103, 4});
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(0);
  dnums.set_input_feature_dimension(1);
  dnums.set_output_feature_dimension(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);
  dnums.add_input_spatial_dimensions(3);
  dnums.add_output_spatial_dimensions(3);

  // Dimension order: x1, batch, feature, x0
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {2, 12, 11, 3});
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(1);
  dnums.add_kernel_spatial_dimensions(3);
  dnums.add_kernel_spatial_dimensions(0);

  Window window;
  auto dim0 = window.add_dimensions();
  dim0->set_size(3);
  dim0->set_stride(3);
  dim0->set_padding_low(0);
  dim0->set_padding_high(0);
  dim0->set_window_dilation(6);
  dim0->set_base_dilation(1);

  auto dim1 = window.add_dimensions();
  dim1->set_size(2);
  dim1->set_stride(1);
  dim1->set_padding_low(2);
  dim1->set_padding_high(1);
  dim1->set_window_dilation(2);
  dim1->set_base_dilation(1);
  auto inferred_status = ShapeInference::InferConvolveShape(
      lhs_shape, rhs_shape, /*feature_group_count=*/1, /*batch_group_count=*/1,
      window, dnums);
  ASSERT_IS_OK(inferred_status.status());
  Shape inferred_shape = inferred_status.ValueOrDie();
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {10, 12, 31, 5}),
                               inferred_shape));
}

TEST_F(ShapeInferenceTest, ConvolveWithBaseDilation) {
  ConvolutionDimensionNumbers dnums;

  // Dimension order: batch, feature, x0, x1
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {10, 11, 3, 4});
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(0);
  dnums.set_input_feature_dimension(1);
  dnums.set_output_feature_dimension(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);
  dnums.add_input_spatial_dimensions(3);
  dnums.add_output_spatial_dimensions(3);

  // Dimension order: x1, batch, feature, x0
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {2, 12, 11, 4});
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(1);
  dnums.add_kernel_spatial_dimensions(3);
  dnums.add_kernel_spatial_dimensions(0);

  Window window;
  auto dim0 = window.add_dimensions();
  dim0->set_size(4);
  dim0->set_stride(3);
  dim0->set_padding_low(0);
  dim0->set_padding_high(0);
  dim0->set_window_dilation(1);
  dim0->set_base_dilation(6);

  auto dim1 = window.add_dimensions();
  dim1->set_size(2);
  dim1->set_stride(1);
  dim1->set_padding_low(2);
  dim1->set_padding_high(1);
  dim1->set_window_dilation(1);
  dim1->set_base_dilation(2);
  auto inferred_status = ShapeInference::InferConvolveShape(
      lhs_shape, rhs_shape, /*feature_group_count=*/1, /*batch_group_count=*/1,
      window, dnums);
  ASSERT_IS_OK(inferred_status.status());
  Shape inferred_shape = inferred_status.ValueOrDie();
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {10, 12, 4, 9}),
                               inferred_shape));
}

TEST_F(ShapeInferenceTest, ConvolveDimensionNumbersOverlapError) {
  // Dimension order for this test: batch, feature, x0, x1
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {10, 11, 3, 4});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {12, 11, 3, 2});

  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(3);
  dnums.set_output_batch_dimension(3);
  dnums.set_input_feature_dimension(2);
  dnums.set_output_feature_dimension(2);
  dnums.add_input_spatial_dimensions(0);
  dnums.add_output_spatial_dimensions(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.add_output_spatial_dimensions(1);
  dnums.set_kernel_input_feature_dimension(0);  // duplicated with kernel_x0
  dnums.set_kernel_output_feature_dimension(3);
  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);

  Window window;
  auto dim0 = window.add_dimensions();
  auto dim1 = window.add_dimensions();
  dim0->set_size(2);
  dim0->set_stride(1);
  dim0->set_padding_low(0);
  dim0->set_padding_high(0);
  dim1->set_size(3);
  dim1->set_stride(2);
  dim1->set_padding_low(1);
  dim1->set_padding_high(1);
  auto inferred_status = ShapeInference::InferConvolveShape(
      lhs_shape, rhs_shape, /*feature_group_count=*/1, /*batch_group_count=*/1,
      window, dnums);
  ASSERT_FALSE(inferred_status.ok());
  ASSERT_THAT(inferred_status.status().error_message(),
              HasSubstr("each dimension exactly once"));
}

TEST_F(ShapeInferenceTest, ConvolveBatchGroupCountUnequalOutputFeature) {
  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(0);
  dnums.set_input_feature_dimension(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_input_spatial_dimensions(3);
  dnums.set_kernel_input_feature_dimension(0);
  dnums.set_kernel_output_feature_dimension(1);
  dnums.add_kernel_spatial_dimensions(2);
  dnums.add_kernel_spatial_dimensions(3);
  dnums.set_output_batch_dimension(0);
  dnums.set_output_feature_dimension(1);
  dnums.add_output_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(3);
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {60, 38, 17, 13});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {38, 10, 4, 4});
  Window window;
  auto dim0 = window.add_dimensions();
  auto dim1 = window.add_dimensions();
  dim0->set_size(4);
  dim1->set_size(4);
  dim0->set_padding_low(0);
  dim0->set_padding_high(2);
  dim1->set_padding_low(2);
  dim1->set_padding_high(1);
  dim0->set_stride(1);
  dim1->set_stride(1);
  dim0->set_window_dilation(3);
  dim1->set_window_dilation(2);
  auto inferred_status = ShapeInference::InferConvolveShape(
      lhs_shape, rhs_shape, /*feature_group_count=*/1, /*batch_group_count=*/6,
      window, dnums);
  ASSERT_FALSE(inferred_status.ok());
  ASSERT_THAT(inferred_status.status().error_message(),
              HasSubstr("to be equal to batch group count"));
}

namespace fft {

static const char* unsupported_rank = "only supports ranks 1-3";
static const char* invalid_rank = "requires input of at least same rank";
static const char* requires_complex_input = "requires complex input type";
static const char* requires_f32_input = "requires F32 input type";
static const char* requires_c64_input = "requires C64 input type";
static const char* dimensions_match = "innermost dimensions match fft_length";
static const char* innermost_dimension_matches =
    "innermost dimension matches fft_length/2+1";

static void Pass(const Shape& shape, FftType type,
                 absl::Span<const int64> length, const Shape& expected_shape) {
  auto inferred_status = ShapeInference::InferFftShape(shape, type, length);
  ASSERT_IS_OK(inferred_status.status());
  Shape inferred_shape = inferred_status.ValueOrDie();
  ASSERT_TRUE(ShapeUtil::Equal(inferred_shape, expected_shape));
}

static void Fail(const Shape& shape, FftType type,
                 absl::Span<const int64> length, absl::string_view message) {
  auto inferred_status = ShapeInference::InferFftShape(shape, type, length);
  ASSERT_FALSE(inferred_status.ok());
  ASSERT_THAT(inferred_status.status().error_message(),
              HasSubstr(std::string(message)));
}

}  // namespace fft

TEST_F(ShapeInferenceTest, InferFftShapeTestFftRanks) {
  FftType type = FftType::FFT;
  Shape shape = ShapeUtil::MakeShape(C64, {16, 8});
  fft::Fail(shape, type, {}, fft::unsupported_rank);
  fft::Pass(shape, type, {8}, shape);
  fft::Pass(shape, type, {16, 8}, shape);
  fft::Fail(shape, type, {32, 16, 8}, fft::invalid_rank);
  fft::Fail(shape, type, {64, 32, 16, 8}, fft::unsupported_rank);
}

TEST_F(ShapeInferenceTest, InferFftShapeTestFftTypes) {
  FftType type = FftType::FFT;
  Shape shape_f32 = ShapeUtil::MakeShape(F32, {16, 8});
  Shape shape_c128 = ShapeUtil::MakeShape(C128, {16, 8});
  fft::Fail(shape_f32, type, {16, 8}, fft::requires_complex_input);
  fft::Fail(shape_c128, type, {16, 8}, fft::requires_complex_input);
}

TEST_F(ShapeInferenceTest, InferFftShapeTestIfftRanks) {
  FftType type = FftType::IFFT;
  Shape shape = ShapeUtil::MakeShape(C64, {16, 8});
  fft::Fail(shape, type, {}, fft::unsupported_rank);
  fft::Pass(shape, type, {8}, shape);
  fft::Pass(shape, type, {16, 8}, shape);
  fft::Fail(shape, type, {32, 16, 8}, fft::invalid_rank);
  fft::Fail(shape, type, {64, 32, 16, 8}, fft::unsupported_rank);
}

TEST_F(ShapeInferenceTest, InferFftShapeTestIfftTypes) {
  FftType type = FftType::IFFT;
  Shape shape_f32 = ShapeUtil::MakeShape(F32, {16, 8});
  Shape shape_c128 = ShapeUtil::MakeShape(C128, {16, 8});
  fft::Fail(shape_f32, type, {16, 8}, fft::requires_complex_input);
  fft::Fail(shape_c128, type, {16, 8}, fft::requires_complex_input);
}

TEST_F(ShapeInferenceTest, InferFftShapeTestRfftRanks) {
  FftType type = FftType::RFFT;
  Shape shape_in = ShapeUtil::MakeShape(F32, {16, 8});
  Shape shape_out = ShapeUtil::MakeShape(C64, {16, 5});
  fft::Fail(shape_in, type, {}, fft::unsupported_rank);
  fft::Pass(shape_in, type, {8}, shape_out);
  fft::Pass(shape_in, type, {16, 8}, shape_out);
  fft::Fail(shape_in, type, {32, 16, 8}, fft::invalid_rank);
  fft::Fail(shape_in, type, {64, 32, 16, 8}, fft::unsupported_rank);
}

TEST_F(ShapeInferenceTest, InferFftShapeTestRfftDimensions) {
  FftType type = FftType::RFFT;
  Shape shape = ShapeUtil::MakeShape(F32, {16, 8});
  fft::Fail(shape, type, {4}, fft::dimensions_match);
  fft::Fail(shape, type, {16, 4}, fft::dimensions_match);
  fft::Fail(shape, type, {8, 8}, fft::dimensions_match);
  fft::Fail(shape, type, {8, 16}, fft::dimensions_match);

  Shape zero_shape_in = ShapeUtil::MakeShape(F32, {16, 0});
  Shape zero_shape_out = ShapeUtil::MakeShape(C64, {16, 0});
  fft::Pass(zero_shape_in, type, {0}, zero_shape_out);
  fft::Pass(zero_shape_in, type, {16, 0}, zero_shape_out);

  Shape even_shape_in = ShapeUtil::MakeShape(F32, {16, 8});
  Shape odd_shape_in = ShapeUtil::MakeShape(F32, {16, 9});
  Shape shape_out = ShapeUtil::MakeShape(C64, {16, 5});
  fft::Pass(even_shape_in, type, {16, 8}, shape_out);
  fft::Pass(odd_shape_in, type, {16, 9}, shape_out);
}

TEST_F(ShapeInferenceTest, InferFftShapeTestRfftTypes) {
  FftType type = FftType::RFFT;
  Shape shape_c64 = ShapeUtil::MakeShape(C64, {16, 8});
  Shape shape_c128 = ShapeUtil::MakeShape(C128, {16, 8});
  fft::Fail(shape_c64, type, {16, 8}, fft::requires_f32_input);
  fft::Fail(shape_c128, type, {16, 8}, fft::requires_f32_input);
}

TEST_F(ShapeInferenceTest, InferFftShapeTestIrfftRanks) {
  FftType type = FftType::IRFFT;
  Shape shape_in = ShapeUtil::MakeShape(C64, {16, 5});
  Shape shape_out = ShapeUtil::MakeShape(F32, {16, 8});
  fft::Fail(shape_in, type, {}, fft::unsupported_rank);
  fft::Pass(shape_in, type, {8}, shape_out);
  fft::Pass(shape_in, type, {16, 8}, shape_out);
  fft::Fail(shape_in, type, {32, 16, 8}, fft::invalid_rank);
  fft::Fail(shape_in, type, {64, 32, 16, 8}, fft::unsupported_rank);
}

TEST_F(ShapeInferenceTest, InferFftShapeTestIrfftDimensions) {
  FftType type = FftType::IRFFT;
  Shape shape = ShapeUtil::MakeShape(C64, {16, 5});
  fft::Fail(shape, type, {5}, fft::innermost_dimension_matches);
  fft::Fail(shape, type, {16, 5}, fft::innermost_dimension_matches);
  fft::Fail(shape, type, {8, 8}, fft::dimensions_match);
  fft::Fail(shape, type, {8, 9}, fft::dimensions_match);

  Shape zero_shape_in = ShapeUtil::MakeShape(C64, {16, 0});
  Shape zero_shape_out = ShapeUtil::MakeShape(F32, {16, 0});
  fft::Pass(zero_shape_in, type, {0}, zero_shape_out);
  fft::Pass(zero_shape_in, type, {16, 0}, zero_shape_out);

  Shape even_shape_out = ShapeUtil::MakeShape(F32, {16, 8});
  Shape odd_shape_out = ShapeUtil::MakeShape(F32, {16, 9});
  fft::Pass(shape, type, {16, 8}, even_shape_out);
  fft::Pass(shape, type, {16, 9}, odd_shape_out);
}

TEST_F(ShapeInferenceTest, InferFftShapeTestIrfftTypes) {
  FftType type = FftType::IRFFT;
  Shape shape_f32 = ShapeUtil::MakeShape(F32, {16, 8});
  Shape shape_c128 = ShapeUtil::MakeShape(C128, {16, 8});
  fft::Fail(shape_f32, type, {16, 8}, fft::requires_c64_input);
  fft::Fail(shape_c128, type, {16, 8}, fft::requires_c64_input);
}

TEST_F(ShapeInferenceTest, MapThatChangesElementType) {
  Shape arg = ShapeUtil::MakeShape(F32, {20});
  ProgramShape to_apply = ShapeUtil::MakeProgramShape({f32_}, s32_);
  auto inferred_status = ShapeInference::InferMapShape({&arg}, to_apply, {0});
  EXPECT_IS_OK(inferred_status.status());
  Shape expected = ShapeUtil::MakeShape(S32, {20});
  EXPECT_TRUE(ShapeUtil::Equal(expected, inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, Map) {
  auto inferred_status_r1f32 = ShapeInference::InferMapShape(
      {&vector_32_, &vector_32_},
      ShapeUtil::MakeProgramShape({f32_, f32_}, f32_), {0});
  EXPECT_IS_OK(inferred_status_r1f32.status());
  EXPECT_TRUE(ShapeUtil::Equal(vector_32_, inferred_status_r1f32.ValueOrDie()));

  // It's OK to provide a single argument, as long as the applied arity matches
  // (this degenerates to a Map).
  auto inferred_status_r1f32_one = ShapeInference::InferMapShape(
      {&vector_32_}, ShapeUtil::MakeProgramShape({f32_}, f32_), {0});
  EXPECT_IS_OK(inferred_status_r1f32_one.status());
  EXPECT_TRUE(
      ShapeUtil::Equal(vector_32_, inferred_status_r1f32_one.ValueOrDie()));

  auto inferred_status_r2s32 = ShapeInference::InferMapShape(
      {&s32matrix_64_64_, &s32matrix_64_64_, &s32matrix_64_64_},
      ShapeUtil::MakeProgramShape({s32_, s32_, s32_}, s32_), {0, 1});
  EXPECT_IS_OK(inferred_status_r2s32.status());
  EXPECT_TRUE(
      ShapeUtil::Equal(s32matrix_64_64_, inferred_status_r2s32.ValueOrDie()));

  auto no_args_error = ShapeInference::InferMapShape(
      {}, ShapeUtil::MakeProgramShape({f32_, f32_}, f32_), {});
  ASSERT_FALSE(no_args_error.ok());
  ASSERT_THAT(no_args_error.status().error_message(),
              HasSubstr("expects at least one argument"));

  auto args_diff_shapes_error = ShapeInference::InferMapShape(
      {&vector_32_, &vector_64_},
      ShapeUtil::MakeProgramShape({f32_, f32_}, f32_), {0});
  ASSERT_FALSE(args_diff_shapes_error.ok());
  ASSERT_THAT(args_diff_shapes_error.status().error_message(),
              HasSubstr("requires all operands to have the same shape"));

  auto arity_error = ShapeInference::InferMapShape(
      {&vector_32_, &vector_32_}, ShapeUtil::MakeProgramShape({f32_}, f32_),
      {0});
  ASSERT_FALSE(arity_error.ok());
  ASSERT_THAT(arity_error.status().error_message(),
              HasSubstr("function arity must match"));

  auto output_shape_error = ShapeInference::InferMapShape(
      {&vector_32_, &vector_32_},
      ShapeUtil::MakeProgramShape({f32_, f32_}, vector_32_), {0});
  ASSERT_FALSE(output_shape_error.ok());
  ASSERT_THAT(output_shape_error.status().error_message(),
              HasSubstr("result has to be a scalar"));

  auto param_shape_error = ShapeInference::InferMapShape(
      {&vector_32_, &vector_32_},
      ShapeUtil::MakeProgramShape({vector_32_, f32_}, f32_), {0});
  ASSERT_FALSE(param_shape_error.ok());
  ASSERT_THAT(param_shape_error.status().error_message(),
              HasSubstr("parameter has to be a scalar"));

  auto param_element_type_error = ShapeInference::InferMapShape(
      {&vector_32_, &vector_32_},
      ShapeUtil::MakeProgramShape({f32_, s32_}, f32_), {0});
  ASSERT_FALSE(param_element_type_error.ok());
  ASSERT_THAT(param_element_type_error.status().error_message(),
              HasSubstr("parameter type has to match argument"));

  Shape arg = ShapeUtil::MakeShape(F32, {20});
  ProgramShape to_apply = ShapeUtil::MakeProgramShape({f32_}, f32_);
  auto inferred_status = ShapeInference::InferMapShape({&arg}, to_apply, {0});
  EXPECT_IS_OK(inferred_status.status());
  EXPECT_TRUE(ShapeUtil::Equal(arg, inferred_status.ValueOrDie()));

  auto inferred_status_error1 = ShapeInference::InferMapShape(
      {&arg}, ShapeUtil::MakeProgramShape({f32_, f32_}, f32_), {0});
  ASSERT_FALSE(inferred_status_error1.ok());
  ASSERT_THAT(inferred_status_error1.status().error_message(),
              HasSubstr("arity must match number of arguments"));

  auto inferred_status_error2 = ShapeInference::InferMapShape(
      {&arg}, ShapeUtil::MakeProgramShape({vector_32_}, f32_), {0});
  ASSERT_FALSE(inferred_status_error2.ok());
  ASSERT_THAT(inferred_status_error2.status().error_message(),
              HasSubstr("has to be a scalar"));

  auto inferred_status_error3 = ShapeInference::InferMapShape(
      {&arg}, ShapeUtil::MakeProgramShape({f32_}, vector_32_), {0});
  ASSERT_FALSE(inferred_status_error3.ok());
  ASSERT_THAT(inferred_status_error3.status().error_message(),
              HasSubstr("has to be a scalar"));

  auto inferred_status_error5 = ShapeInference::InferMapShape(
      {&arg}, ShapeUtil::MakeProgramShape({s32_}, s32_), {0});
  ASSERT_FALSE(inferred_status_error5.ok());
  ASSERT_THAT(inferred_status_error5.status().error_message(),
              HasSubstr("parameter type has to match argument"));
}

TEST_F(ReduceShapeInferenceTest, ReduceVectorToScalar) {
  ExpectInferredReduceShape(f32_, ShapeUtil::MakeShape(F32, {128}),
                            /*dimensions_to_reduce=*/{0});
}

TEST_F(ReduceShapeInferenceTest, ReduceCubeAmongFirstDimension) {
  ExpectInferredReduceShape(ShapeUtil::MakeShape(F32, {3, 4}),
                            ShapeUtil::MakeShape(F32, {2, 3, 4}),
                            /*dimensions_to_reduce=*/{0});
}

TEST_F(ReduceShapeInferenceTest, ReduceCubeAmongMiddleDimension) {
  ExpectInferredReduceShape(ShapeUtil::MakeShape(F32, {2, 4}),
                            ShapeUtil::MakeShape(F32, {2, 3, 4}),
                            /*dimensions_to_reduce=*/{1});
}

TEST_F(ReduceShapeInferenceTest, ReduceCubeAmongFirstTwoDimensions) {
  ExpectInferredReduceShape(ShapeUtil::MakeShape(F32, {4}),
                            ShapeUtil::MakeShape(F32, {2, 3, 4}),
                            /*dimensions_to_reduce=*/{0, 1});
}

TEST_F(ReduceShapeInferenceTest, ReduceCubeAmongLastTwoDimensions) {
  ExpectInferredReduceShape(ShapeUtil::MakeShape(F32, {2}),
                            ShapeUtil::MakeShape(F32, {2, 3, 4}),
                            /*dimensions_to_reduce=*/{1, 2});
}

TEST_F(ReduceShapeInferenceTest, ReduceCubeAmongFirstAndLastDimensions) {
  ExpectInferredReduceShape(ShapeUtil::MakeShape(F32, {3}),
                            ShapeUtil::MakeShape(F32, {2, 3, 4}),
                            /*dimensions_to_reduce=*/{0, 2});

  // Check that the order of dimensions_to_reduce doesn't matter.
  ExpectInferredReduceShape(ShapeUtil::MakeShape(F32, {3}),
                            ShapeUtil::MakeShape(F32, {2, 3, 4}),
                            /*dimensions_to_reduce=*/{2, 0});
}

TEST_F(ReduceShapeInferenceTest, ReduceCubeAmongAllDimensions) {
  ExpectInferredReduceShape(f32_, ShapeUtil::MakeShape(F32, {2, 3, 4}),
                            /*dimensions_to_reduce=*/{0, 1, 2});
}

TEST_F(ReduceShapeInferenceTest, ReduceMultiOutput) {
  Shape f32_arg_shape = ShapeUtil::MakeShape(F32, {5, 3});
  Shape s32_arg_shape = ShapeUtil::MakeShape(S32, {5, 3});
  ProgramShape to_apply = ShapeUtil::MakeProgramShape(
      {f32_, s32_, f32_, s32_}, ShapeUtil::MakeTupleShape({f32_, s32_}));
  auto inferred_status = ShapeInference::InferReduceShape(
      {&f32_arg_shape, &s32_arg_shape, &f32_, &s32_}, {0, 1}, to_apply);
  EXPECT_IS_OK(inferred_status.status());
  EXPECT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeTupleShape({f32_, s32_}),
                               inferred_status.ValueOrDie()));
}

TEST_F(ReduceShapeInferenceTest, ErrorMultiOutputBadReducerInput1) {
  Shape f32_arg_shape = ShapeUtil::MakeShape(F32, {5, 3});
  Shape s32_arg_shape = ShapeUtil::MakeShape(S32, {5, 3});
  ProgramShape to_apply =
      ShapeUtil::MakeProgramShape({f32_, s32_, f32_, s32_, f32_, s32_},
                                  ShapeUtil::MakeTupleShape({f32_, s32_}));
  auto inferred_status = ShapeInference::InferReduceShape(
      {&f32_arg_shape, &s32_arg_shape, &f32_, &s32_}, {0, 1}, to_apply);
  EXPECT_FALSE(inferred_status.ok());
  EXPECT_THAT(inferred_status.status().error_message(),
              HasSubstr("must take 4 parameters, but takes 6 parameter(s)"));
}

TEST_F(ReduceShapeInferenceTest, ErrorMultiOutputBadReducerInput2) {
  Shape f32_arg_shape = ShapeUtil::MakeShape(F32, {5, 3});
  Shape s32_arg_shape = ShapeUtil::MakeShape(S32, {5, 3});
  ProgramShape to_apply = ShapeUtil::MakeProgramShape(
      {s32_, s32_, f32_, s32_}, ShapeUtil::MakeTupleShape({f32_, s32_}));
  auto inferred_status = ShapeInference::InferReduceShape(
      {&f32_arg_shape, &s32_arg_shape, &f32_, &s32_}, {0, 1}, to_apply);
  EXPECT_FALSE(inferred_status.ok());
  EXPECT_THAT(
      inferred_status.status().error_message(),
      HasSubstr(
          "parameter shape differs from the result shape: s32[] vs f32[]"));
}

TEST_F(ReduceShapeInferenceTest, ErrorMultiOutputBadReducerInput3) {
  ProgramShape to_apply = ShapeUtil::MakeProgramShape(
      {s32_, s32_, f32_, s32_}, ShapeUtil::MakeTupleShape({f32_, s32_}));
  auto inferred_status = ShapeInference::InferReduceShape({}, {0, 1}, to_apply);
  EXPECT_FALSE(inferred_status.ok());
  EXPECT_THAT(inferred_status.status().error_message(),
              HasSubstr("must have at least 2 arguments, has 0"));
}

TEST_F(ReduceShapeInferenceTest, ErrorMultiOutputBadReducerOutput1) {
  Shape f32_arg_shape = ShapeUtil::MakeShape(F32, {5, 3});
  Shape s32_arg_shape = ShapeUtil::MakeShape(S32, {5, 3});
  ProgramShape to_apply =
      ShapeUtil::MakeProgramShape({f32_, s32_, f32_, s32_}, f32_);
  auto inferred_status = ShapeInference::InferReduceShape(
      {&f32_arg_shape, &s32_arg_shape, &f32_, &s32_}, {0, 1}, to_apply);
  EXPECT_FALSE(inferred_status.ok());
  EXPECT_THAT(
      inferred_status.status().error_message(),
      HasSubstr("must produce a tuple with 2 elements, but produces a scalar"));
}

TEST_F(ReduceShapeInferenceTest, ErrorMultiOutputBadReducerOutput2) {
  Shape f32_arg_shape = ShapeUtil::MakeShape(F32, {5, 3});
  Shape s32_arg_shape = ShapeUtil::MakeShape(S32, {5, 3});
  ProgramShape to_apply = ShapeUtil::MakeProgramShape(
      {f32_, s32_, f32_, s32_}, ShapeUtil::MakeTupleShape({f32_, s32_, s32_}));
  auto inferred_status = ShapeInference::InferReduceShape(
      {&f32_arg_shape, &s32_arg_shape, &f32_, &s32_}, {0, 1}, to_apply);
  EXPECT_FALSE(inferred_status.ok());
  EXPECT_THAT(
      inferred_status.status().error_message(),
      HasSubstr("must produce a tuple with 2 elements, but has 3 elements"));
}

TEST_F(ReduceShapeInferenceTest, ErrorMultiOutputBadReducerBoth) {
  Shape f32_arg_shape = ShapeUtil::MakeShape(F32, {5, 3});
  Shape s32_arg_shape = ShapeUtil::MakeShape(S32, {5, 3});
  ProgramShape to_apply = ShapeUtil::MakeProgramShape(
      {s32_, s32_, s32_, s32_}, ShapeUtil::MakeTupleShape({s32_, s32_}));
  auto inferred_status = ShapeInference::InferReduceShape(
      {&f32_arg_shape, &s32_arg_shape, &f32_, &s32_}, {0, 1}, to_apply);
  EXPECT_FALSE(inferred_status.ok());
  EXPECT_THAT(inferred_status.status().error_message(),
              HasSubstr("accumulator shape at index 0 differs from the "
                        "init_value shape: s32[] vs f32[]"));
}

TEST_F(ReduceShapeInferenceTest, ErrorOutOfBoundsDimension) {
  ProgramShape to_apply = ShapeUtil::MakeProgramShape({f32_, f32_}, f32_);
  Shape arg_shape = ShapeUtil::MakeShape(F32, {5, 3});
  auto inferred_status = ShapeInference::InferReduceShape(
      {&arg_shape, &f32_},
      /*dimensions_to_reduce=*/{3, 4}, to_apply);
  EXPECT_FALSE(inferred_status.ok());
  EXPECT_THAT(inferred_status.status().error_message(),
              HasSubstr("out-of-bounds dimension"));
}

TEST_F(ReduceShapeInferenceTest, ErrorToApplyArity) {
  ProgramShape to_apply = ShapeUtil::MakeProgramShape({f32_, f32_, f32_}, f32_);
  Shape arg_shape = ShapeUtil::MakeShape(F32, {5, 3});
  auto inferred_status =
      ShapeInference::InferReduceShape({&arg_shape, &f32_},
                                       /*dimensions_to_reduce=*/{0}, to_apply);
  EXPECT_FALSE(inferred_status.ok());
  EXPECT_THAT(inferred_status.status().error_message(),
              HasSubstr("take 2 parameters"));
}

TEST_F(ReduceShapeInferenceTest, ErrorElementTypeVsApplyType) {
  ProgramShape to_apply = ShapeUtil::MakeProgramShape({f32_, f32_}, s32_);
  Shape arg_shape = ShapeUtil::MakeShape(F32, {5, 3});
  auto inferred_status =
      ShapeInference::InferReduceShape({&arg_shape, &f32_},
                                       /*dimensions_to_reduce=*/{0}, to_apply);
  EXPECT_FALSE(inferred_status.ok());
  EXPECT_THAT(inferred_status.status().error_message(),
              HasSubstr("0-th parameter shape differs"));
}

TEST_F(ShapeInferenceTest, InferSliceShapeRank2) {
  Shape matrix_shape = ShapeUtil::MakeShape(F32, {128, 64});
  auto inferred_status =
      ShapeInference::InferSliceShape(matrix_shape, {32, 0}, {64, 64}, {1, 1});
  ASSERT_IS_OK(inferred_status.status());
  Shape inferred = inferred_status.ValueOrDie();
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {32, 64}), inferred));
}

TEST_F(ShapeInferenceTest, InferSliceShapeRank2WithStrides) {
  Shape matrix_shape = ShapeUtil::MakeShape(F32, {128, 64});
  auto inferred_status =
      ShapeInference::InferSliceShape(matrix_shape, {32, 0}, {64, 64}, {2, 4});
  ASSERT_IS_OK(inferred_status.status());
  Shape inferred = inferred_status.ValueOrDie();
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {16, 16}), inferred));
}

TEST_F(ShapeInferenceTest, InferSliceShapeRank2WithStridesNotIntegral) {
  Shape matrix_shape = ShapeUtil::MakeShape(F32, {128, 64});
  auto inferred_status =
      ShapeInference::InferSliceShape(matrix_shape, {15, 0}, {20, 13}, {2, 4});
  ASSERT_IS_OK(inferred_status.status());
  Shape inferred = inferred_status.ValueOrDie();
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {3, 4}), inferred));
}

TEST_F(ShapeInferenceTest, InferInvalidStride) {
  Shape matrix_shape = ShapeUtil::MakeShape(F32, {128, 64});
  auto inferred_status =
      ShapeInference::InferSliceShape(matrix_shape, {127, 0}, {129, 2}, {0, 1});
  ASSERT_FALSE(inferred_status.ok());
  ASSERT_EQ(tensorflow::error::INVALID_ARGUMENT,
            inferred_status.status().code());
}

TEST_F(ShapeInferenceTest, InferOobSliceShapeRank2) {
  Shape matrix_shape = ShapeUtil::MakeShape(F32, {128, 64});
  auto inferred_status =
      ShapeInference::InferSliceShape(matrix_shape, {127, 0}, {129, 2}, {1, 1});
  ASSERT_FALSE(inferred_status.ok());
  ASSERT_EQ(tensorflow::error::INVALID_ARGUMENT,
            inferred_status.status().code());
}

TEST_F(ShapeInferenceTest, InferSliceShapeRank1) {
  Shape vector_shape = ShapeUtil::MakeShape(F32, {17});
  auto inferred_status =
      ShapeInference::InferSliceShape(vector_shape, {2}, {4}, {1});
  ASSERT_TRUE(inferred_status.ok());
  Shape inferred = inferred_status.ValueOrDie();
  ASSERT_TRUE(ShapeUtil::Equal(inferred, ShapeUtil::MakeShape(F32, {2})));
}

TEST_F(ShapeInferenceTest, InferConstIndexShape) {
  Shape tuple_shape = ShapeUtil::MakeTupleShape({f32_, s32_});
  auto inferred0_status =
      ShapeInference::InferGetTupleElementShape(tuple_shape, 0);
  auto inferred1_status =
      ShapeInference::InferGetTupleElementShape(tuple_shape, 1);
  ASSERT_IS_OK(inferred0_status.status());
  ASSERT_IS_OK(inferred1_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(f32_, inferred0_status.ValueOrDie()));
  ASSERT_TRUE(ShapeUtil::Equal(s32_, inferred1_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, InferTupleElementShapeOutOfBound) {
  Shape tuple_shape = ShapeUtil::MakeTupleShape({f32_, s32_});
  auto inferredNegative_status =
      ShapeInference::InferGetTupleElementShape(tuple_shape, -1);
  auto inferred2_status =
      ShapeInference::InferGetTupleElementShape(tuple_shape, 2);
  ASSERT_FALSE(inferredNegative_status.ok());
  ASSERT_FALSE(inferred2_status.ok());
  EXPECT_THAT(inferredNegative_status.status().error_message(),
              HasSubstr("attempt to index out of tuple bounds"));
  EXPECT_THAT(inferred2_status.status().error_message(),
              HasSubstr("attempt to index out of tuple bounds"));
}

TEST_F(ShapeInferenceTest, InferPowShape) {
  auto ten_floats = ShapeUtil::MakeShape(F32, {10});
  auto inferred_status = ShapeInference::InferBinaryOpShape(
      HloOpcode::kPower, ten_floats, f32_, {});
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(ten_floats, inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, InferCompareShape) {
  auto ten_floats = ShapeUtil::MakeShape(F32, {10});
  auto inferred_status = ShapeInference::InferBinaryOpShape(
      HloOpcode::kCompare, ten_floats, f32_, {});
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(PRED, {10}),
                               inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, InferReshapeDegenerateCombine) {
  // [1, <=1]
  //   | reshape
  // [<=1]
  //
  // Both output dimension can be dynamic, use inferred_dimension to tie-break.
  auto operand = ShapeUtil::MakeShape(F32, {1, 1}, {false, true});
  auto status = ShapeInference::InferReshapeShape(operand, {1, 0}, {1},
                                                  /*inferred_dimension=*/-1);
  ASSERT_EQ(ShapeUtil::MakeShape(F32, {1}, {true}), status.ValueOrDie());
}

TEST_F(ShapeInferenceTest, InferReshapeSplit) {
  // [<=10]
  //   | reshape
  // [1, 10]
  //
  // Both output dimension can be dynamic, use inferred_dimension to tie-break.
  auto operand = ShapeUtil::MakeShape(F32, {10}, {true});
  auto status = ShapeInference::InferReshapeShape(operand, {0}, {1, 10},
                                                  /*inferred_dimension=*/0);
  ASSERT_EQ(ShapeUtil::MakeShape(F32, {1, 10}, {true, false}),
            status.ValueOrDie());
}

TEST_F(ShapeInferenceTest, InferReshapeCombine) {
  // [6, <=10]
  //   | reshape
  // [<=60]
  auto operand = ShapeUtil::MakeShape(F32, {6, 10}, {false, true});
  auto status = ShapeInference::InferReshapeShape(operand, {1, 0}, {60},
                                                  /*inferred_dimension=*/-11);
  ASSERT_EQ(ShapeUtil::MakeShape(F32, {60}, {true}), status.ValueOrDie());
}

TEST_F(ShapeInferenceTest, UnchangedDimension) {
  // [6, <=10]
  //   | reshape
  // [2, 3, <=10]
  auto operand = ShapeUtil::MakeShape(F32, {6, 10}, {false, true});
  auto status = ShapeInference::InferReshapeShape(operand, {1, 0}, {2, 3, 10},
                                                  /*inferred_dimension=*/-11);
  ASSERT_EQ(ShapeUtil::MakeShape(F32, {2, 3, 10}, {false, false, true}),
            status.ValueOrDie());
}

TEST_F(ShapeInferenceTest, BroadcastScalar) {
  for (auto element_type : {F32, U32, S8}) {
    const Shape scalar_shape = ShapeUtil::MakeShape(element_type, {});
    {  // no-op scalar broadcast
      auto status = ShapeInference::InferBroadcastShape(scalar_shape, {});
      ASSERT_IS_OK(status.status());
      ASSERT_TRUE(ShapeUtil::Equal(scalar_shape, status.ValueOrDie()));
    }
    const Shape oned_shape = ShapeUtil::MakeShape(element_type, {3});
    {  // scalar -> 1d broadcast
      auto status = ShapeInference::InferBroadcastShape(scalar_shape, {3});
      ASSERT_IS_OK(status.status());
      ASSERT_TRUE(ShapeUtil::Equal(oned_shape, status.ValueOrDie()));
    }
    {  // no-op 1d broadcast
      auto status = ShapeInference::InferBroadcastShape(oned_shape, {});
      ASSERT_IS_OK(status.status());
      ASSERT_TRUE(ShapeUtil::Equal(oned_shape, status.ValueOrDie()));
    }
    const Shape twod_shape = ShapeUtil::MakeShape(element_type, {2, 3});
    {  // scalar -> 2d broadcast
      auto status = ShapeInference::InferBroadcastShape(scalar_shape, {2, 3});
      ASSERT_IS_OK(status.status());
      ASSERT_TRUE(ShapeUtil::Equal(twod_shape, status.ValueOrDie()));
    }
    {  // 1d -> 2d broadcast
      auto status = ShapeInference::InferBroadcastShape(oned_shape, {2});
      ASSERT_IS_OK(status.status());
      ASSERT_TRUE(ShapeUtil::Equal(twod_shape, status.ValueOrDie()));
    }
  }
}

// scalar <dot> vector: ok
TEST_F(ShapeInferenceTest, ScalarDotVector) {
  DotDimensionNumbers dot_dnums;
  auto inferred_status =
      ShapeInference::InferDotOpShape(f32_, vector_32_, dot_dnums);
  EXPECT_TRUE(inferred_status.ok());
  EXPECT_EQ(inferred_status.ValueOrDie(), vector_32_);
}

// 3D <dot> 2D: error
TEST_F(ShapeInferenceTest, DotWithRankHigherThanTwo) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto inferred_status = ShapeInference::InferDotOpShape(
      ShapeUtil::MakeShape(F32, {32, 32, 32}), matrix_32_64_, dot_dnums);
  EXPECT_TRUE(inferred_status.ok());
  EXPECT_TRUE(ShapeUtil::Equal(inferred_status.ValueOrDie(),
                               ShapeUtil::MakeShape(F32, {32, 32, 64})));
}

// vector <dot> vector -> scalar
TEST_F(ShapeInferenceTest, VectorDotVector) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto inferred_status =
      ShapeInference::InferDotOpShape(vector_64_, vector_64_, dot_dnums);
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(f32_, inferred_status.ValueOrDie()));
  auto inferred_status_mismatch =
      ShapeInference::InferDotOpShape(vector_64_, vector_32_, dot_dnums);
  ASSERT_FALSE(inferred_status_mismatch.ok());
}

// matrix <dot> vector -> vector
TEST_F(ShapeInferenceTest, MatrixDotVector) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto inferred_status =
      ShapeInference::InferDotOpShape(matrix_32_64_, vector_64_, dot_dnums);
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(inferred_status.ValueOrDie(), vector_32_));
  auto inferred_status_mismatch =
      ShapeInference::InferDotOpShape(matrix_32_64_, vector_32_, dot_dnums);
  ASSERT_FALSE(inferred_status_mismatch.ok());
}

// vector <dot> matrix -> vector
TEST_F(ShapeInferenceTest, VectorDotMatrix) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto inferred_status =
      ShapeInference::InferDotOpShape(vector_32_, matrix_32_64_, dot_dnums);
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(inferred_status.ValueOrDie(), vector_64_));
  auto inferred_status_mismatch =
      ShapeInference::InferDotOpShape(vector_64_, matrix_32_64_, dot_dnums);
  ASSERT_FALSE(inferred_status_mismatch.ok());
}

// matrix <dot> matrix -> matrix
TEST_F(ShapeInferenceTest, MatrixDotMatrix) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto inferred_status_match =
      ShapeInference::InferDotOpShape(matrix_32_64_, matrix_64_48_, dot_dnums);
  ASSERT_IS_OK(inferred_status_match.status());
  ASSERT_TRUE(
      ShapeUtil::Equal(inferred_status_match.ValueOrDie(), matrix_32_48_))
      << "inferred: "
      << ShapeUtil::HumanString(inferred_status_match.ValueOrDie())
      << " expected: " << ShapeUtil::HumanString(matrix_64_48_);
  auto inferred_status_mismatch =
      ShapeInference::InferDotOpShape(matrix_32_64_, matrix_32_64_, dot_dnums);
  ASSERT_FALSE(inferred_status_mismatch.ok());
}

// BatchMatMul with two batch dimensions and one contracting dimension.
TEST_F(ShapeInferenceTest, DotGeneral) {
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {5, 2, 11, 3});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {5, 2, 3, 14});
  Shape output_shape = ShapeUtil::MakeShape(F32, {5, 2, 11, 14});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(3);
  dot_dnums.add_lhs_batch_dimensions(0);
  dot_dnums.add_lhs_batch_dimensions(1);

  dot_dnums.add_rhs_contracting_dimensions(2);
  dot_dnums.add_rhs_batch_dimensions(0);
  dot_dnums.add_rhs_batch_dimensions(1);

  auto inferred_status_match =
      ShapeInference::InferDotOpShape(lhs_shape, rhs_shape, dot_dnums);
  ASSERT_IS_OK(inferred_status_match.status());
  ASSERT_TRUE(
      ShapeUtil::Equal(inferred_status_match.ValueOrDie(), output_shape))
      << "inferred: "
      << ShapeUtil::HumanString(inferred_status_match.ValueOrDie())
      << " expected: " << ShapeUtil::HumanString(output_shape);
}

// BatchMatMul with two contracting dimensions fails.
TEST_F(ShapeInferenceTest, DotWithTwoContractingDimsFails) {
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {2, 11, 3, 2});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {2, 3, 14});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(2);
  dot_dnums.add_lhs_contracting_dimensions(3);
  dot_dnums.add_lhs_batch_dimensions(0);

  dot_dnums.add_rhs_contracting_dimensions(1);
  dot_dnums.add_rhs_batch_dimensions(0);

  auto inferred_status =
      ShapeInference::InferDotOpShape(lhs_shape, rhs_shape, dot_dnums);
  ASSERT_FALSE(inferred_status.ok());
  ASSERT_THAT(inferred_status.status().error_message(),
              HasSubstr("Must specify the same number of contracting "
                        "dimensions for lhs and rhs."));
}

TEST_F(ShapeInferenceTest, DotWithTwoContractingDimsPasses) {
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {2, 11, 3, 2});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {2, 3, 2, 14});
  Shape output_shape = ShapeUtil::MakeShape(F32, {2, 11, 14});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(2);
  dot_dnums.add_lhs_contracting_dimensions(3);
  dot_dnums.add_lhs_batch_dimensions(0);

  dot_dnums.add_rhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(2);
  dot_dnums.add_rhs_batch_dimensions(0);

  auto inferred_status =
      ShapeInference::InferDotOpShape(lhs_shape, rhs_shape, dot_dnums);
  EXPECT_TRUE(inferred_status.ok());
  EXPECT_TRUE(ShapeUtil::Equal(inferred_status.ValueOrDie(), output_shape));
}

// BatchMatMul with different batch dimension sizes fails.
TEST_F(ShapeInferenceTest, DotWithMisatchedBatchDimSizesFails) {
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {2, 11, 3});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {3, 3, 14});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(2);
  dot_dnums.add_lhs_batch_dimensions(0);

  dot_dnums.add_rhs_contracting_dimensions(1);
  dot_dnums.add_rhs_batch_dimensions(0);

  auto inferred_status =
      ShapeInference::InferDotOpShape(lhs_shape, rhs_shape, dot_dnums);
  ASSERT_FALSE(inferred_status.ok());
  ASSERT_THAT(inferred_status.status().error_message(),
              HasSubstr("Batch dimension sizes must match"));
}

// BatchMatMul with different batch dimension numbers passes
TEST_F(ShapeInferenceTest, DotWithMisatchedBatchDimNumbersPasses) {
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {2, 11, 3});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {3, 2, 14});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(2);
  dot_dnums.add_lhs_batch_dimensions(0);

  dot_dnums.add_rhs_contracting_dimensions(0);
  dot_dnums.add_rhs_batch_dimensions(1);

  auto inferred_status =
      ShapeInference::InferDotOpShape(lhs_shape, rhs_shape, dot_dnums);
  ASSERT_TRUE(inferred_status.ok());
  ASSERT_TRUE(ShapeUtil::Equal(inferred_status.ValueOrDie(),
                               ShapeUtil::MakeShape(F32, {2, 11, 14})));
}

// BatchMatMul with out-of-range dimension numbers fails.
TEST_F(ShapeInferenceTest, DotWithContractingDimNumberOutOfRange) {
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {2, 11, 3});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {2, 3, 14});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(3);
  dot_dnums.add_lhs_batch_dimensions(0);

  dot_dnums.add_rhs_contracting_dimensions(0);
  dot_dnums.add_rhs_batch_dimensions(1);

  auto inferred_status =
      ShapeInference::InferDotOpShape(lhs_shape, rhs_shape, dot_dnums);
  ASSERT_FALSE(inferred_status.ok());
  ASSERT_THAT(inferred_status.status().error_message(),
              HasSubstr("A dimension number is out of range"));
}

// BatchMatMul with non-unique dimension numbers fails.
TEST_F(ShapeInferenceTest, DotWithContractingNonUniqueDimNumber) {
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {2, 11, 3});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {2, 3, 14});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_lhs_batch_dimensions(0);

  dot_dnums.add_rhs_contracting_dimensions(0);
  dot_dnums.add_rhs_batch_dimensions(1);

  auto inferred_status =
      ShapeInference::InferDotOpShape(lhs_shape, rhs_shape, dot_dnums);
  ASSERT_FALSE(inferred_status.ok());
  ASSERT_THAT(inferred_status.status().error_message(),
              HasSubstr("A dimension number is not unique"));
}

TEST_F(ShapeInferenceTest, BinOpBroadcastMatrixVector) {
  // Test variations of broadcasting a vector for a binary add with a
  // matrix.
  const Shape mat = ShapeUtil::MakeShape(F32, {16, 8});
  const Shape vec8 = ShapeUtil::MakeShape(F32, {8});
  const Shape vec16 = ShapeUtil::MakeShape(F32, {16});

  auto inferred_status_match =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, mat, vec8, {1});
  ASSERT_IS_OK(inferred_status_match.status());
  ASSERT_TRUE(ShapeUtil::Equal(inferred_status_match.ValueOrDie(), mat));

  auto inferred_status_mismatch =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, mat, vec8, {0});
  ASSERT_FALSE(inferred_status_mismatch.ok());

  inferred_status_match =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, mat, vec16, {0});
  ASSERT_IS_OK(inferred_status_match.status());
  ASSERT_TRUE(ShapeUtil::Equal(inferred_status_match.ValueOrDie(), mat));

  inferred_status_mismatch =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, mat, vec16, {1});
  ASSERT_FALSE(inferred_status_mismatch.ok());
}

TEST_F(ShapeInferenceTest, BinOpBroadcastCubeMatrix) {
  // Test variations of broadcasting a matrix for a binary add with a cube.
  const Shape cube = ShapeUtil::MakeShape(F32, {16, 8, 4});
  const Shape matrix8_4 = ShapeUtil::MakeShape(F32, {8, 4});
  const Shape matrix16_4 = ShapeUtil::MakeShape(F32, {16, 4});
  const Shape matrix16_8 = ShapeUtil::MakeShape(F32, {16, 8});

  auto inferred_status_match = ShapeInference::InferBinaryOpShape(
      HloOpcode::kAdd, cube, matrix8_4, {1, 2});
  ASSERT_IS_OK(inferred_status_match.status());
  ASSERT_TRUE(ShapeUtil::Equal(inferred_status_match.ValueOrDie(), cube));

  inferred_status_match = ShapeInference::InferBinaryOpShape(
      HloOpcode::kAdd, cube, matrix16_4, {0, 2});
  ASSERT_IS_OK(inferred_status_match.status());
  ASSERT_TRUE(ShapeUtil::Equal(inferred_status_match.ValueOrDie(), cube));

  inferred_status_match = ShapeInference::InferBinaryOpShape(
      HloOpcode::kAdd, cube, matrix16_8, {0, 1});
  ASSERT_IS_OK(inferred_status_match.status());
  ASSERT_TRUE(ShapeUtil::Equal(inferred_status_match.ValueOrDie(), cube));
}

TEST_F(ShapeInferenceTest, BinOpBroadcastBadDimension) {
  // Test various errors with the broadcast argument.
  const Shape tensor = ShapeUtil::MakeShape(F32, {16, 8, 4});
  const Shape tensor8_8_8 = ShapeUtil::MakeShape(F32, {8, 8, 8});
  const Shape vec8 = ShapeUtil::MakeShape(F32, {8});
  const Shape matrix8_4 = ShapeUtil::MakeShape(F32, {8, 4});
  const Shape matrix8_8 = ShapeUtil::MakeShape(F32, {8, 8});

  // "magical" broadcast rejected
  auto inferred_status_error1 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor, vec8, {});
  ASSERT_FALSE(inferred_status_error1.ok());
  ASSERT_THAT(inferred_status_error1.status().error_message(),
              HasSubstr("Automatic"));

  // broadcast_dimension out of bounds for tensor's rank
  auto inferred_status_error2 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor, vec8, {3});
  ASSERT_FALSE(inferred_status_error2.ok());
  ASSERT_THAT(inferred_status_error2.status().error_message(),
              ContainsRegex("Broadcast dimension number .* too large"));

  // broadcast_dimension doesn't match corresponding dimension
  auto inferred_status_error3 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor, vec8, {0});
  ASSERT_FALSE(inferred_status_error3.ok());
  ASSERT_THAT(inferred_status_error3.status().error_message(),
              HasSubstr("Broadcast dimension 0 mismatch"));

  // broadcast_dimensions list too long
  auto inferred_status_error4 = ShapeInference::InferBinaryOpShape(
      HloOpcode::kAdd, tensor, matrix8_4, {0, 1, 2});
  ASSERT_FALSE(inferred_status_error4.ok());
  ASSERT_THAT(inferred_status_error4.status().error_message(),
              HasSubstr("broadcast_dimensions has to match"));

  // there's a dimension above the rank of the tensor
  auto inferred_status_error5 = ShapeInference::InferBinaryOpShape(
      HloOpcode::kAdd, tensor, matrix8_4, {3, 0});
  ASSERT_FALSE(inferred_status_error5.ok());
  ASSERT_THAT(inferred_status_error5.status().error_message(),
              ContainsRegex("dimension number .* too large"));

  // broadcasting dimensions don't match in this order
  auto inferred_status_error6 = ShapeInference::InferBinaryOpShape(
      HloOpcode::kAdd, tensor, matrix8_4, {2, 1});
  ASSERT_FALSE(inferred_status_error6.ok());
  ASSERT_THAT(inferred_status_error6.status().error_message(),
              HasSubstr("dimension 0 mismatch"));

  // The following two tests make sure that broadcasting dimensions are listed
  // in a proper (strictly increasing) order, even if the lower-rank array
  // matches the higher-rank array in many different ways.
  auto inferred_status_error7 = ShapeInference::InferBinaryOpShape(
      HloOpcode::kAdd, tensor8_8_8, matrix8_8, {0, 0});
  ASSERT_FALSE(inferred_status_error7.ok());
  ASSERT_THAT(inferred_status_error7.status().error_message(),
              HasSubstr("dimensions order is wrong"));

  auto inferred_status_error8 = ShapeInference::InferBinaryOpShape(
      HloOpcode::kAdd, tensor8_8_8, matrix8_8, {1, 0});
  ASSERT_FALSE(inferred_status_error8.ok());
  ASSERT_THAT(inferred_status_error8.status().error_message(),
              HasSubstr("dimensions order is wrong"));
}

// Tests for the while instruction with proper shapes.
TEST_F(ShapeInferenceTest, WhileWithCorrectShapes) {
  Shape result_shape = ShapeUtil::MakeTupleShape({s32_, vector_32_});
  ProgramShape cond = ShapeUtil::MakeProgramShape({result_shape}, pred_);
  ProgramShape body = ShapeUtil::MakeProgramShape({result_shape}, result_shape);
  auto inferred_status =
      ShapeInference::InferWhileShape(cond, body, result_shape);
  ASSERT_IS_OK(inferred_status.status());
  Shape inferred = inferred_status.ValueOrDie();
  ASSERT_TRUE(ShapeUtil::Equal(result_shape, inferred));
}

// Tests for the while instruction with wrong shapes.
TEST_F(ShapeInferenceTest, WhileWithBadShapes) {
  Shape result_shape = ShapeUtil::MakeTupleShape({s32_, vector_32_});
  ProgramShape cond = ShapeUtil::MakeProgramShape({result_shape}, pred_);
  ProgramShape body = ShapeUtil::MakeProgramShape({result_shape}, result_shape);

  auto bad_shape_1 = ShapeUtil::MakeProgramShape({s32_, result_shape}, pred_);
  auto inferred_status_error1 =
      ShapeInference::InferWhileShape(bad_shape_1, body, result_shape);
  ASSERT_FALSE(inferred_status_error1.ok());
  ASSERT_THAT(inferred_status_error1.status().error_message(),
              HasSubstr("Condition must take 1 arguments"));

  auto bad_shape_2 =
      ShapeUtil::MakeProgramShape({s32_, result_shape}, result_shape);
  auto inferred_status_error2 =
      ShapeInference::InferWhileShape(cond, bad_shape_2, result_shape);
  ASSERT_FALSE(inferred_status_error2.ok());
  ASSERT_THAT(inferred_status_error2.status().error_message(),
              HasSubstr("Body must take 1 arguments"));

  auto bad_shape_3 = ShapeUtil::MakeProgramShape({result_shape}, s32_);
  auto inferred_status_error3 =
      ShapeInference::InferWhileShape(bad_shape_3, body, result_shape);
  ASSERT_FALSE(inferred_status_error3.ok());
  ASSERT_THAT(inferred_status_error3.status().error_message(),
              HasSubstr("Condition must return a boolean"));

  auto bad_shape_4 = ShapeUtil::MakeProgramShape({result_shape}, vector_32_);
  auto inferred_status_error4 =
      ShapeInference::InferWhileShape(cond, bad_shape_4, result_shape);
  ASSERT_FALSE(inferred_status_error4.ok());
  ASSERT_THAT(inferred_status_error4.status().error_message(),
              HasSubstr("parameter of condition and body"));
}

// Tests for the concatenate instruction with proper shapes.
TEST_F(ShapeInferenceTest, ConcatenateWithCorrectShapes) {
  auto inferred_status_1 = ShapeInference::InferConcatOpShape(
      {&vector_32_, &vector_64_}, /*dimension=*/0);
  ASSERT_IS_OK(inferred_status_1.status());
  Shape inferred_1 = inferred_status_1.ValueOrDie();
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {96}), inferred_1));

  auto inferred_status_2 = ShapeInference::InferConcatOpShape(
      {&vector_32_, &vector_64_, &vector_32_}, /*dimension=*/0);
  ASSERT_IS_OK(inferred_status_2.status());
  Shape inferred_2 = inferred_status_2.ValueOrDie();
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {128}), inferred_2));

  auto inferred_status_3 = ShapeInference::InferConcatOpShape(
      {&matrix_32_48_, &matrix_32_64_, &matrix_32_48_}, /*dimension=*/1);
  ASSERT_IS_OK(inferred_status_3.status());
  Shape inferred_3 = inferred_status_3.ValueOrDie();
  ASSERT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {32, 160}), inferred_3));
}

// Tests for the concatenate instruction with wrong shapes.
TEST_F(ShapeInferenceTest, ConcatenateWithBadShapes) {
  auto inferred_status_error1 =
      ShapeInference::InferConcatOpShape({}, /*dimension=*/0);
  ASSERT_FALSE(inferred_status_error1.ok());
  ASSERT_THAT(inferred_status_error1.status().error_message(),
              HasSubstr("Concatenate expects at least one argument"));

  auto inferred_status_error2 =
      ShapeInference::InferConcatOpShape({&vector_32_}, /*dimension=*/-1);
  ASSERT_FALSE(inferred_status_error2.ok());
  ASSERT_THAT(inferred_status_error2.status().error_message(),
              HasSubstr("dimension out of bounds: -1"));

  auto inferred_status_error3 =
      ShapeInference::InferConcatOpShape({&vector_32_}, /*dimension=*/1);
  ASSERT_FALSE(inferred_status_error3.ok());
  ASSERT_THAT(inferred_status_error3.status().error_message(),
              HasSubstr("dimension out of bounds: 1"));

  Shape tuple = ShapeUtil::MakeTupleShape({vector_32_});
  auto inferred_status_error4 = ShapeInference::InferConcatOpShape(
      {&vector_32_, &tuple}, /*dimension=*/0);
  ASSERT_FALSE(inferred_status_error4.ok());
  ASSERT_THAT(
      inferred_status_error4.status().error_message(),
      HasSubstr("Expected array argument for operand of concatenation"));

  const Shape vector_s32 = ShapeUtil::MakeShape(S32, {32});
  auto inferred_status_error5 = ShapeInference::InferConcatOpShape(
      {&vector_32_, &vector_s32}, /*dimension=*/0);
  ASSERT_FALSE(inferred_status_error5.ok());
  ASSERT_THAT(inferred_status_error5.status().error_message(),
              HasSubstr("concatenate arrays with different element types"));

  auto inferred_status_error6 = ShapeInference::InferConcatOpShape(
      {&matrix_32_48_, &matrix_32_64_}, /*dimension=*/0);
  ASSERT_FALSE(inferred_status_error6.ok());
  ASSERT_THAT(inferred_status_error6.status().error_message(),
              HasSubstr("concatenate arrays that differ in "
                        "dimensions other than the one being "
                        "concatenated"));
}

TEST_F(ShapeInferenceTest, Pad) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {10, 25});
  Shape padding_value_shape = ShapeUtil::MakeShape(F32, {});
  // Padding for dimension 0: {low: 0, high: 2, interior: 3}
  // Padding for dimension 1: {low: 1, high: 5, interior: 0}
  PaddingConfig padding_config;
  auto dimension0 = padding_config.add_dimensions();
  dimension0->set_edge_padding_low(0);
  dimension0->set_edge_padding_high(2);
  dimension0->set_interior_padding(3);
  auto dimension1 = padding_config.add_dimensions();
  dimension1->set_edge_padding_low(1);
  dimension1->set_edge_padding_high(5);
  dimension1->set_interior_padding(0);

  auto inferred_status = ShapeInference::InferPadShape(
      input_shape, padding_value_shape, padding_config);
  ASSERT_IS_OK(inferred_status.status());
  Shape inferred_shape = inferred_status.ValueOrDie();
  ASSERT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {39, 31}), inferred_shape));

  dimension1->set_edge_padding_low(-20);
  dimension1->set_edge_padding_high(-10);
  auto negative_dimension_size = ShapeInference::InferPadShape(
      input_shape, padding_value_shape, padding_config);
  ASSERT_FALSE(negative_dimension_size.ok());
  ASSERT_THAT(negative_dimension_size.status().error_message(),
              HasSubstr("negative size for dimension 1"));
}

TEST_F(ShapeInferenceTest, Reverse) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {10, 25});

  auto inferred_status = ShapeInference::InferReverseShape(input_shape, {0, 1});
  ASSERT_IS_OK(inferred_status.status());
  Shape inferred_shape = inferred_status.ValueOrDie();
  ASSERT_TRUE(ShapeUtil::Equal(input_shape, inferred_shape));
}

TEST_F(ShapeInferenceTest, ReverseInvalidDimension) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {10, 25});

  auto inferred_status_error0 =
      ShapeInference::InferReverseShape(input_shape, {0, 2});
  ASSERT_FALSE(inferred_status_error0.ok());
  ASSERT_THAT(inferred_status_error0.status().error_message(),
              HasSubstr("out-of-bounds"));

  auto inferred_status_error1 =
      ShapeInference::InferReverseShape(input_shape, {0, -1});
  ASSERT_FALSE(inferred_status_error1.ok());
  ASSERT_THAT(inferred_status_error1.status().error_message(),
              HasSubstr("out-of-bounds"));

  auto inferred_status_error2 =
      ShapeInference::InferReverseShape(input_shape, {0, 0});
  ASSERT_FALSE(inferred_status_error2.ok());
  ASSERT_THAT(inferred_status_error2.status().error_message(),
              HasSubstr("duplicated"));

  Shape tuple_shape = ShapeUtil::MakeTupleShape({input_shape, input_shape});
  auto inferred_status_error3 =
      ShapeInference::InferReverseShape(tuple_shape, {0});
  ASSERT_FALSE(inferred_status_error3.ok());
  ASSERT_THAT(inferred_status_error3.status().error_message(),
              HasSubstr("Expected array argument"));
}

TEST_F(ShapeInferenceTest, Call) {
  auto inferred_status0 =
      ShapeInference::InferCallShape({}, ShapeUtil::MakeProgramShape({}, f32_));
  EXPECT_IS_OK(inferred_status0.status());
  EXPECT_TRUE(ShapeUtil::Equal(f32_, inferred_status0.ValueOrDie()));

  auto inferred_status1 = ShapeInference::InferCallShape(
      {&f32_, &s32_, &pred_, &vector_32_, &matrix_32_48_},
      ShapeUtil::MakeProgramShape(
          {f32_, s32_, pred_, vector_32_, matrix_32_48_}, s32matrix_64_64_));
  EXPECT_IS_OK(inferred_status1.status());
  EXPECT_TRUE(
      ShapeUtil::Equal(s32matrix_64_64_, inferred_status1.ValueOrDie()));

  auto inferred_status_error0 = ShapeInference::InferCallShape(
      {}, ShapeUtil::MakeProgramShape({f32_}, f32_));
  EXPECT_FALSE(inferred_status_error0.ok());
  EXPECT_THAT(inferred_status_error0.status().error_message(),
              HasSubstr("arity must match"));

  auto inferred_status_error1 = ShapeInference::InferCallShape(
      {&f32_}, ShapeUtil::MakeProgramShape({}, f32_));
  EXPECT_FALSE(inferred_status_error1.ok());
  EXPECT_THAT(inferred_status_error1.status().error_message(),
              HasSubstr("arity must match"));

  auto inferred_status_error2 = ShapeInference::InferCallShape(
      {&f32_}, ShapeUtil::MakeProgramShape({s32_}, f32_));
  EXPECT_FALSE(inferred_status_error2.ok());
  EXPECT_THAT(inferred_status_error2.status().error_message(),
              HasSubstr("parameter must match argument"));
}

TEST_F(ShapeInferenceTest, Transpose) {
  Shape a_shape = ShapeUtil::MakeShape(F32, {2, 3, 4, 5});
  auto inferred_shape_and_status =
      ShapeInference::InferTransposeShape(a_shape, {1, 2, 3, 0});
  EXPECT_IS_OK(inferred_shape_and_status);
  Shape inferred_shape = inferred_shape_and_status.ValueOrDie();
  EXPECT_TRUE(ShapeUtil::Compatible(inferred_shape,
                                    ShapeUtil::MakeShape(F32, {3, 4, 5, 2})));
}

TEST_F(ShapeInferenceTest, Rank1Transpose) {
  Shape a_shape = ShapeUtil::MakeShape(F32, {5});
  auto inferred_shape_and_status =
      ShapeInference::InferTransposeShape(a_shape, {0});
  EXPECT_IS_OK(inferred_shape_and_status);
  Shape inferred_shape = inferred_shape_and_status.ValueOrDie();
  EXPECT_TRUE(
      ShapeUtil::Compatible(inferred_shape, ShapeUtil::MakeShape(F32, {5})));
}

TEST_F(ShapeInferenceTest, ConditionalPred) {
  auto inferred_status0 = ShapeInference::InferConditionalShape(
      pred_,
      {ShapeUtil::MakeProgramShape({vector_32_}, f32_),
       ShapeUtil::MakeProgramShape({vector_64_}, f32_)},
      {vector_32_, vector_64_});
  EXPECT_IS_OK(inferred_status0.status());
  EXPECT_TRUE(ShapeUtil::Equal(f32_, inferred_status0.ValueOrDie()));

  auto inferred_status1 = ShapeInference::InferConditionalShape(
      pred_,
      {ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_64_),
       ShapeUtil::MakeProgramShape({vector_32_}, vector_64_)},
      {matrix_32_48_, vector_32_});
  EXPECT_IS_OK(inferred_status1.status());
  EXPECT_TRUE(ShapeUtil::Equal(vector_64_, inferred_status1.ValueOrDie()));

  auto tuple_f32_v32 = ShapeUtil::MakeTupleShape({f32_, vector_32_});
  auto inferred_status2 = ShapeInference::InferConditionalShape(
      pred_,
      {ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_32_),
       ShapeUtil::MakeProgramShape({tuple_f32_v32}, vector_32_)},
      {matrix_32_48_, tuple_f32_v32});
  EXPECT_IS_OK(inferred_status2.status());
  EXPECT_TRUE(ShapeUtil::Equal(vector_32_, inferred_status2.ValueOrDie()));

  auto inferred_status_error0 = ShapeInference::InferConditionalShape(
      f32_,
      {ShapeUtil::MakeProgramShape({vector_32_}, f32_),
       ShapeUtil::MakeProgramShape({vector_64_}, f32_)},
      {vector_32_, vector_64_});
  EXPECT_FALSE(inferred_status_error0.ok());
  EXPECT_THAT(inferred_status_error0.status().error_message(),
              HasSubstr("must be bool or int32"));

  auto inferred_status_error1 = ShapeInference::InferConditionalShape(
      pred_,
      {ShapeUtil::MakeProgramShape({f32_, vector_32_}, vector_32_),
       ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_32_)},
      {ShapeUtil::MakeTupleShape({f32_, vector_32_}), matrix_32_48_});
  EXPECT_FALSE(inferred_status_error1.ok());
  EXPECT_THAT(inferred_status_error1.status().error_message(),
              HasSubstr("branch computation 0 must take 1 argument"));

  auto inferred_status_error2 = ShapeInference::InferConditionalShape(
      pred_,
      {ShapeUtil::MakeProgramShape({vector_64_}, f32_),
       ShapeUtil::MakeProgramShape({vector_64_}, f32_)},
      {vector_32_, vector_64_});
  EXPECT_FALSE(inferred_status_error2.ok());
  EXPECT_THAT(inferred_status_error2.status().error_message(),
              HasSubstr("branch operand 0 must match the shape of the only "
                        "parameter of branch computation 0"));

  auto inferred_status_error3 = ShapeInference::InferConditionalShape(
      pred_,
      {ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_32_),
       ShapeUtil::MakeProgramShape({f32_, vector_32_}, vector_32_)},
      {matrix_32_48_, ShapeUtil::MakeTupleShape({f32_, vector_32_})});
  EXPECT_FALSE(inferred_status_error3.ok());
  EXPECT_THAT(inferred_status_error3.status().error_message(),
              HasSubstr("branch computation 1 must take 1 argument"));

  auto inferred_status_error4 = ShapeInference::InferConditionalShape(
      pred_,
      {ShapeUtil::MakeProgramShape({vector_32_}, f32_),
       ShapeUtil::MakeProgramShape({vector_32_}, f32_)},
      {vector_32_, vector_64_});
  EXPECT_FALSE(inferred_status_error4.ok());
  EXPECT_THAT(inferred_status_error4.status().error_message(),
              HasSubstr("branch operand 1 must match the shape of the only "
                        "parameter of branch computation 1"));

  auto inferred_status_error5 = ShapeInference::InferConditionalShape(
      pred_,
      {ShapeUtil::MakeProgramShape({vector_32_}, f32_),
       ShapeUtil::MakeProgramShape({vector_64_}, vector_32_)},
      {vector_32_, vector_64_});
  EXPECT_FALSE(inferred_status_error5.ok());
  EXPECT_THAT(inferred_status_error5.status().error_message(),
              HasSubstr("the result of branch 0 computation and branch 1 "
                        "computation must have the same shape"));
}

TEST_F(ShapeInferenceTest, ConditionalIndexed) {
  auto r0s32 = ShapeUtil::MakeShape(S32, {});
  auto inferred_status0 = ShapeInference::InferConditionalShape(
      r0s32,
      {ShapeUtil::MakeProgramShape({vector_32_}, f32_),
       ShapeUtil::MakeProgramShape({vector_64_}, f32_),
       ShapeUtil::MakeProgramShape({vector_64_}, f32_)},
      {vector_32_, vector_64_, vector_64_});
  EXPECT_IS_OK(inferred_status0.status());
  EXPECT_TRUE(ShapeUtil::Equal(f32_, inferred_status0.ValueOrDie()));

  auto inferred_status1 = ShapeInference::InferConditionalShape(
      r0s32,
      {ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_64_),
       ShapeUtil::MakeProgramShape({vector_32_}, vector_64_),
       ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_64_)},
      {matrix_32_48_, vector_32_, matrix_32_48_});
  EXPECT_IS_OK(inferred_status1.status());
  EXPECT_TRUE(ShapeUtil::Equal(vector_64_, inferred_status1.ValueOrDie()));

  auto tuple_f32_v32 = ShapeUtil::MakeTupleShape({f32_, vector_32_});
  auto inferred_status2 = ShapeInference::InferConditionalShape(
      r0s32, {ShapeUtil::MakeProgramShape({tuple_f32_v32}, vector_32_)},
      {tuple_f32_v32});
  EXPECT_IS_OK(inferred_status2.status());
  EXPECT_TRUE(ShapeUtil::Equal(vector_32_, inferred_status2.ValueOrDie()));

  auto inferred_status_error0 = ShapeInference::InferConditionalShape(
      pred_,
      {ShapeUtil::MakeProgramShape({vector_32_}, f32_),
       ShapeUtil::MakeProgramShape({vector_32_}, f32_),
       ShapeUtil::MakeProgramShape({vector_64_}, f32_)},
      {vector_32_, vector_32_, vector_64_});
  EXPECT_FALSE(inferred_status_error0.ok());
  EXPECT_THAT(inferred_status_error0.status().error_message(),
              HasSubstr("2 == branch_computations.size()"));

  auto inferred_status_error1 = ShapeInference::InferConditionalShape(
      r0s32,
      {ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_32_),
       ShapeUtil::MakeProgramShape({f32_, vector_32_}, vector_32_),
       ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_32_)},
      {matrix_32_48_, ShapeUtil::MakeTupleShape({f32_, vector_32_}),
       matrix_32_48_});
  EXPECT_FALSE(inferred_status_error1.ok());
  EXPECT_THAT(inferred_status_error1.status().error_message(),
              HasSubstr("branch computation 1 must take 1 argument"));

  auto inferred_status_error2 = ShapeInference::InferConditionalShape(
      r0s32,
      {ShapeUtil::MakeProgramShape({r0s32}, f32_),
       ShapeUtil::MakeProgramShape({vector_32_}, f32_),
       ShapeUtil::MakeProgramShape({vector_32_}, f32_)},
      {r0s32, vector_32_, vector_64_});
  EXPECT_FALSE(inferred_status_error2.ok());
  EXPECT_THAT(inferred_status_error2.status().error_message(),
              HasSubstr("branch operand 2 must match the shape of the only "
                        "parameter of branch computation 2"));

  auto inferred_status_error3 = ShapeInference::InferConditionalShape(
      r0s32,
      {ShapeUtil::MakeProgramShape({vector_32_}, f32_),
       ShapeUtil::MakeProgramShape({vector_32_}, f32_),
       ShapeUtil::MakeProgramShape({vector_32_}, f32_),
       ShapeUtil::MakeProgramShape({vector_64_}, vector_32_)},
      {vector_32_, vector_32_, vector_32_, vector_64_});
  EXPECT_FALSE(inferred_status_error3.ok());
  EXPECT_THAT(inferred_status_error3.status().error_message(),
              HasSubstr("the result of branch 0 computation and branch 3 "
                        "computation must have the same shape"));

  auto inferred_status_error4 =
      ShapeInference::InferConditionalShape(r0s32, {}, {});
  EXPECT_FALSE(inferred_status_error4.ok());
  EXPECT_THAT(inferred_status_error4.status().error_message(),
              HasSubstr("!branch_computations.empty()"));
}

TEST_F(ShapeInferenceTest, BadSlice) {
  auto arg = ShapeUtil::MakeShape(F32, {4});
  StatusOr<Shape> statusor =
      ShapeInference::InferSliceShape(arg, {0}, {5}, {1});
  ASSERT_FALSE(statusor.ok());

  LOG(INFO) << statusor.status();

  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("less than or equal to dimension size"))
      << statusor.status();
  EXPECT_THAT(statusor.status().error_message(), HasSubstr("argument shape"))
      << statusor.status();
}

TEST_F(ShapeInferenceTest, BadSort) {
  auto keys = ShapeUtil::MakeShape(F32, {4});
  auto values = ShapeUtil::MakeShape(F32, {5});
  StatusOr<Shape> statusor =
      ShapeInference::InferVariadicOpShape(HloOpcode::kSort, {&keys, &values});
  EXPECT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("dimensions must match"))
      << statusor.status();
}

TEST_F(ShapeInferenceTest, BadSortValuesMismatch) {
  auto keys = ShapeUtil::MakeShape(F32, {4});
  auto values_good = ShapeUtil::MakeShape(F32, {4});
  auto values_bad = ShapeUtil::MakeShape(F32, {5});
  StatusOr<Shape> statusor = ShapeInference::InferVariadicOpShape(
      HloOpcode::kSort, {&keys, &values_good, &values_bad});
  EXPECT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("dimensions must match"))
      << statusor.status();
}

TEST_F(ShapeInferenceTest, SortManyValues) {
  auto keys = ShapeUtil::MakeShape(F32, {4});
  auto values_s32 = ShapeUtil::MakeShape(S32, {4});
  auto values_u32 = ShapeUtil::MakeShape(U32, {4});
  StatusOr<Shape> statusor = ShapeInference::InferVariadicOpShape(
      HloOpcode::kSort, {&keys, &values_s32, &values_u32});
  EXPECT_IS_OK(statusor);
  Shape inferred_shape = statusor.ValueOrDie();
  EXPECT_TRUE(ShapeUtil::Compatible(
      inferred_shape,
      ShapeUtil::MakeTupleShape({keys, values_s32, values_u32})));
}

class ScatterGatherShapeInferenceTest : public ShapeInferenceTest {
 protected:
  const Shape s64_scalar_ = ShapeUtil::MakeShape(S64, {});
  const Shape s64_vector_5_ = ShapeUtil::MakeShape(S64, {5});
  const Shape s64_vector_32_ = ShapeUtil::MakeShape(S64, {32});
  const Shape s64_4d_tensor_10_9_8_7_1_ =
      ShapeUtil::MakeShape(S64, {10, 9, 8, 7, 1});
  const Shape s64_4d_tensor_10_9_8_7_5_ =
      ShapeUtil::MakeShape(S64, {10, 9, 8, 7, 5});
  const Shape s64_4d_tensor_5_10_9_7_6_ =
      ShapeUtil::MakeShape(S64, {5, 10, 9, 7, 6});
  const Shape s64_4d_tensor_10_9_5_7_6_ =
      ShapeUtil::MakeShape(S64, {10, 9, 5, 7, 6});
  const Shape f32_5d_tensor_50_49_48_47_46_ =
      ShapeUtil::MakeShape(F32, {50, 49, 48, 47, 46});
  const Shape tuple_shape_ = ShapeUtil::MakeTupleShape(
      {s64_4d_tensor_10_9_8_7_1_, s64_4d_tensor_10_9_8_7_1_});
  const ProgramShape to_apply_ =
      ShapeUtil::MakeProgramShape({f32_, f32_}, f32_);
};

// Shape inference tests for Gather.

TEST_F(ScatterGatherShapeInferenceTest, TensorFlowGather) {
  TF_ASSERT_OK_AND_ASSIGN(Shape gather_shape,
                          ShapeInference::InferGatherShape(
                              matrix_64_48_, s64_vector_32_,
                              HloGatherInstruction::MakeGatherDimNumbers(
                                  /*offset_dims=*/{0},
                                  /*collapsed_slice_dims=*/{1},
                                  /*start_index_map=*/{1},
                                  /*index_vector_dim=*/1),
                              /*slice_sizes=*/{64, 1}));
  EXPECT_TRUE(
      ShapeUtil::Equal(gather_shape, ShapeUtil::MakeShape(F32, {64, 32})))
      << ShapeUtil::HumanString(gather_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, TensorFlowGatherV2) {
  TF_ASSERT_OK_AND_ASSIGN(Shape gather_shape,
                          ShapeInference::InferGatherShape(
                              matrix_64_48_, s64_vector_32_,
                              HloGatherInstruction::MakeGatherDimNumbers(
                                  /*offset_dims=*/{1},
                                  /*collapsed_slice_dims=*/{0},
                                  /*start_index_map=*/{0},
                                  /*index_vector_dim=*/1),
                              /*slice_sizes=*/{1, 48}));
  EXPECT_TRUE(
      ShapeUtil::Equal(gather_shape, ShapeUtil::MakeShape(F32, {32, 48})))
      << ShapeUtil::HumanString(gather_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, TensorFlowGatherNd) {
  TF_ASSERT_OK_AND_ASSIGN(Shape gather_shape,
                          ShapeInference::InferGatherShape(
                              matrix_64_48_, s64_4d_tensor_10_9_8_7_1_,
                              HloGatherInstruction::MakeGatherDimNumbers(
                                  /*offset_dims=*/{4},
                                  /*collapsed_slice_dims=*/{0},
                                  /*start_index_map=*/{0},
                                  /*index_vector_dim=*/4),
                              /*slice_sizes=*/{1, 48}));
  EXPECT_TRUE(ShapeUtil::Equal(gather_shape,
                               ShapeUtil::MakeShape(F32, {10, 9, 8, 7, 48})))
      << ShapeUtil::HumanString(gather_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, TensorFlowBatchDynamicSlice) {
  TF_ASSERT_OK_AND_ASSIGN(
      Shape gather_shape,
      ShapeInference::InferGatherShape(
          f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
          HloGatherInstruction::MakeGatherDimNumbers(
              /*offset_dims=*/{4, 5, 6, 7, 8},
              /*collapsed_slice_dims=*/{},
              /*start_index_map=*/{0, 1, 2, 3, 4},
              /*index_vector_dim=*/4),
          /*slice_sizes=*/{30, 29, 28, 27, 26}));
  EXPECT_TRUE(ShapeUtil::Equal(
      gather_shape,
      ShapeUtil::MakeShape(F32, {10, 9, 8, 7, 30, 29, 28, 27, 26})))
      << ShapeUtil::HumanString(gather_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, DynamicGatherEntireDimension) {
  TF_ASSERT_OK_AND_ASSIGN(
      Shape gather_shape,
      ShapeInference::InferGatherShape(
          ShapeUtil::MakeShape(F32, {3, 2, 1}, {false, true, false}),
          ShapeUtil::MakeShape(S64, {}),
          HloGatherInstruction::MakeGatherDimNumbers(
              /*offset_dims=*/{0, 1},
              /*collapsed_slice_dims=*/{0},
              /*start_index_map=*/{0},
              /*index_vector_dim=*/0),
          /*slice_sizes=*/{1, 2, 1}));
  EXPECT_TRUE(ShapeUtil::Equal(
      gather_shape, ShapeUtil::MakeShape(F32, {2, 1}, {true, false})))
      << ShapeUtil::HumanString(gather_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, DynamicGatherCollapsedDimension) {
  TF_ASSERT_OK_AND_ASSIGN(
      Shape gather_shape,
      ShapeInference::InferGatherShape(
          ShapeUtil::MakeShape(F32, {3, 2, 1}, {true, false, false}),
          ShapeUtil::MakeShape(S64, {}),
          HloGatherInstruction::MakeGatherDimNumbers(
              /*offset_dims=*/{0, 1},
              /*collapsed_slice_dims=*/{0},
              /*start_index_map=*/{0},
              /*index_vector_dim=*/0),
          /*slice_sizes=*/{1, 2, 1}));
  EXPECT_TRUE(ShapeUtil::Equal(
      gather_shape, ShapeUtil::MakeShape(F32, {2, 1}, {false, false})))
      << ShapeUtil::HumanString(gather_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, DynamicIndices) {
  TF_ASSERT_OK_AND_ASSIGN(
      Shape gather_shape,
      ShapeInference::InferGatherShape(
          ShapeUtil::MakeShape(F32, {3, 2, 2}),
          ShapeUtil::MakeShape(S64, {3, 4, 2}, {false, true, false}),
          HloGatherInstruction::MakeGatherDimNumbers(
              /*offset_dims=*/{2, 3},
              /*collapsed_slice_dims=*/{0},
              /*start_index_map=*/{0, 1},
              /*index_vector_dim=*/2),
          /*slice_sizes=*/{1, 2, 2}));
  EXPECT_TRUE(ShapeUtil::Equal(
      gather_shape,
      ShapeUtil::MakeShape(F32, {3, 4, 2, 2}, {false, true, false, false})))
      << ShapeUtil::HumanString(gather_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, NonDefaultGatherIndicesLeafDim_A) {
  TF_ASSERT_OK_AND_ASSIGN(
      Shape gather_shape,
      ShapeInference::InferGatherShape(
          f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_5_7_6_,
          HloGatherInstruction::MakeGatherDimNumbers(
              /*offset_dims=*/{4, 5, 6, 7, 8},
              /*collapsed_slice_dims=*/{},
              /*start_index_map=*/{0, 1, 2, 3, 4},
              /*index_vector_dim=*/2),
          /*slice_sizes=*/{30, 29, 28, 27, 26}));

  EXPECT_TRUE(ShapeUtil::Equal(
      gather_shape,
      ShapeUtil::MakeShape(F32, {10, 9, 7, 6, 30, 29, 28, 27, 26})))
      << ShapeUtil::HumanString(gather_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, NonDefaultGatherIndicesLeafDim_B) {
  TF_ASSERT_OK_AND_ASSIGN(
      Shape gather_shape,
      ShapeInference::InferGatherShape(
          f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_5_10_9_7_6_,
          HloGatherInstruction::MakeGatherDimNumbers(
              /*offset_dims=*/{4, 5, 6, 7, 8},
              /*collapsed_slice_dims=*/{},
              /*start_index_map=*/{0, 1, 2, 3, 4},
              /*index_vector_dim=*/0),
          /*slice_sizes=*/{30, 29, 28, 27, 26}));

  EXPECT_TRUE(ShapeUtil::Equal(
      gather_shape,
      ShapeUtil::MakeShape(F32, {10, 9, 7, 6, 30, 29, 28, 27, 26})))
      << ShapeUtil::HumanString(gather_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, NoOutputGatherDims) {
  // This is equivalent to a dynamic slice.
  TF_ASSERT_OK_AND_ASSIGN(Shape gather_shape,
                          ShapeInference::InferGatherShape(
                              f32_5d_tensor_50_49_48_47_46_, s64_vector_5_,
                              HloGatherInstruction::MakeGatherDimNumbers(
                                  /*offset_dims=*/{0, 1, 2, 3, 4},
                                  /*collapsed_slice_dims=*/{},
                                  /*start_index_map=*/{0, 1, 2, 3, 4},
                                  /*index_vector_dim=*/0),
                              /*slice_sizes=*/{30, 29, 28, 27, 26}));

  EXPECT_TRUE(ShapeUtil::Equal(gather_shape,
                               ShapeUtil::MakeShape(F32, {30, 29, 28, 27, 26})))
      << ShapeUtil::HumanString(gather_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, ScalarGatherIndices) {
  // The gather indices "tensor" is a scalar S here that's used to slice out
  // [S,0,0,0,0]..[S,30,29,28,27] into a [30,29,28,27] shaped result.
  TF_ASSERT_OK_AND_ASSIGN(Shape gather_shape,
                          ShapeInference::InferGatherShape(
                              f32_5d_tensor_50_49_48_47_46_, s64_scalar_,
                              HloGatherInstruction::MakeGatherDimNumbers(
                                  /*offset_dims=*/{0, 1, 2, 3},
                                  /*collapsed_slice_dims=*/{0},
                                  /*start_index_map=*/{0},
                                  /*index_vector_dim=*/0),
                              /*slice_sizes=*/{1, 30, 29, 28, 27}));

  EXPECT_TRUE(ShapeUtil::Equal(gather_shape,
                               ShapeUtil::MakeShape(F32, {30, 29, 28, 27})))
      << ShapeUtil::HumanString(gather_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, TupleShapedTensorInput) {
  StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      tuple_shape_, s64_vector_32_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{0},
          /*collapsed_slice_dims=*/{1},
          /*start_index_map=*/{1},
          /*index_vector_dim=*/1),
      /*slice_sizes=*/{64, 1});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("Expected array argument for input"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest, TupleShapedGatherIndicesInput) {
  StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      s64_vector_32_, tuple_shape_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{0},
          /*collapsed_slice_dims=*/{1},
          /*start_index_map=*/{1},
          /*index_vector_dim=*/0),
      /*slice_sizes=*/{64, 1});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("Expected array argument for gather indices"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest, FloatingPointGatherIndicesInput) {
  StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      s64_vector_32_, vector_32_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{0},
          /*collapsed_slice_dims=*/{1},
          /*start_index_map=*/{1},
          /*index_vector_dim=*/0),
      /*slice_sizes=*/{64, 1});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("Gather indices parameter must be an integral tensor"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidGatherDimNumbers_NonAscendingWindowIndices) {
  StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 8, 7},
          /*collapsed_slice_dims=*/{},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 27, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().error_message(),
      HasSubstr("Output window dimensions in gather op must be ascending"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidGatherDimNumbers_RepeatedWindowIndices) {
  StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7, 7},
          /*collapsed_slice_dims=*/{},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 27, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().error_message(),
      HasSubstr("Output window dimensions in gather op must not repeat"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidGatherDimNumbers_WindowIndexOutOfBounds) {
  StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 99, 100, 101},
          /*collapsed_slice_dims=*/{},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 27, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("Offset dimension 2 in gather op is out of bounds"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidGatherDimNumbers_WindowIndexBarelyOutOfBounds) {
  StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7, 9},
          /*collapsed_slice_dims=*/{},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 27, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("Offset dimension 4 in gather op is out of bounds"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidGatherDimNumbers_MismatchingElidedWindowDims) {
  StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7, 8},
          /*collapsed_slice_dims=*/{4},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 27, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().error_message(),
      HasSubstr("All components of the offset index in a gather op must either "
                "be a offset dimension or explicitly collapsed"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidGatherDimNumbers_OutOfBoundsWindowToInputMapping) {
  StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7, 8},
          /*collapsed_slice_dims=*/{0, 1, 2, 3, 19},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 27, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("Invalid collapsed_slice_dims set in gather op; valid "
                        "range is [0, 5), got: 19"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidGatherDimNumbers_RepeatedWindowToInputMapping) {
  StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7, 8},
          /*collapsed_slice_dims=*/{0, 1, 2, 3, 3},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 27, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("Repeated dimensions not allowed in "
                        "collapsed_slice_dims in gather op"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidGatherDimNumbers_MismatchingGatherToInputMapping) {
  StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7, 8},
          /*collapsed_slice_dims=*/{},
          /*start_index_map=*/{0, 1, 2, 3},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 27, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("Gather op has 4 elements in start_index_map and "
                        "the bound of dimension index_vector_dim=4 of "
                        "start_indices is 5. These two numbers must be equal."))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidGatherDimNumbers_OutOfBoundsGatherToInputMapping) {
  StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7, 8},
          /*collapsed_slice_dims=*/{},
          /*start_index_map=*/{0, 1, 2, 3, 7},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 27, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("Invalid start_index_map; domain is [0, 5), got: 4->7"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidGatherDimNumbers_RepeatedGatherToInputMapping) {
  StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7, 8},
          /*collapsed_slice_dims=*/{},
          /*start_index_map=*/{0, 1, 2, 3, 3},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 27, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().error_message(),
      HasSubstr("Repeated dimensions are not allowed in start_index_map"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidGatherDimNumbers_NonAscendingElidedWindowDims) {
  StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7, 8},
          /*collapsed_slice_dims=*/{2, 1},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{1, 1, 28, 27, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("collapsed_slice_dims in gather op must be sorted"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidGatherDimNumbers_WindowBoundsTooLarge) {
  StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7},
          /*collapsed_slice_dims=*/{2},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 1, 300, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("Slice size at index 3 in gather op is out of range, "
                        "must be within [0, 48), got 300."))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidGatherDimNumbers_MismatchingNumberOfWindowBounds) {
  StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7, 8},
          /*collapsed_slice_dims=*/{},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().error_message(),
      HasSubstr("Gather op must have one slice size for every input dimension"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidGatherDimNumbers_WindowBoundsNot1ForElidedDim) {
  StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7},
          /*collapsed_slice_dims=*/{1},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 26, 20});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().error_message(),
      HasSubstr("Gather op can only collapse slice dims with bound 1 or 0, "
                "but bound is 29 for index 1 at position 0."))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest, OutOfBoundsGatherIndicesLeafDim) {
  StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_5_7_6_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7, 8},
          /*collapsed_slice_dims=*/{},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/32),
      /*slice_sizes=*/{30, 29, 28, 27, 26});

  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("Gather index leaf dimension must be within [0, "
                        "rank(start_indices) + 1)"))
      << statusor.status();
}

// Shape inference tests for Scatter.

TEST_F(ScatterGatherShapeInferenceTest, TfScatterWithFullUpdates) {
  TF_ASSERT_OK_AND_ASSIGN(Shape scatter_shape,
                          ShapeInference::InferScatterShape(
                              matrix_64_48_, s64_vector_32_,
                              ShapeUtil::MakeShape(F32, {64, 32}), to_apply_,
                              HloScatterInstruction::MakeScatterDimNumbers(
                                  /*update_window_dims=*/{0},
                                  /*inserted_window_dims=*/{1},
                                  /*scatter_dims_to_operand_dims=*/{1},
                                  /*index_vector_dim=*/1)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, matrix_64_48_))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, TfScatterWithFullUpdatesV2) {
  TF_ASSERT_OK_AND_ASSIGN(Shape scatter_shape,
                          ShapeInference::InferScatterShape(
                              matrix_64_48_, s64_vector_32_,
                              ShapeUtil::MakeShape(F32, {32, 48}), to_apply_,
                              HloScatterInstruction::MakeScatterDimNumbers(
                                  /*update_window_dims=*/{1},
                                  /*inserted_window_dims=*/{0},
                                  /*scatter_dims_to_operand_dims=*/{0},
                                  /*index_vector_dim=*/1)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, matrix_64_48_))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, TfScatterWithPartialUpdates) {
  TF_ASSERT_OK_AND_ASSIGN(Shape scatter_shape,
                          ShapeInference::InferScatterShape(
                              matrix_64_48_, s64_vector_32_,
                              ShapeUtil::MakeShape(F32, {10, 32}), to_apply_,
                              HloScatterInstruction::MakeScatterDimNumbers(
                                  /*update_window_dims=*/{0},
                                  /*inserted_window_dims=*/{1},
                                  /*scatter_dims_to_operand_dims=*/{1},
                                  /*index_vector_dim=*/1)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, matrix_64_48_))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, TfScatterWithPartialUpdatesV2) {
  TF_ASSERT_OK_AND_ASSIGN(Shape scatter_shape,
                          ShapeInference::InferScatterShape(
                              matrix_64_48_, s64_vector_32_,
                              ShapeUtil::MakeShape(F32, {32, 8}), to_apply_,
                              HloScatterInstruction::MakeScatterDimNumbers(
                                  /*update_window_dims=*/{1},
                                  /*inserted_window_dims=*/{0},
                                  /*scatter_dims_to_operand_dims=*/{0},
                                  /*index_vector_dim=*/1)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, matrix_64_48_))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, TfScatterWithUpdatesBiggerThanInput) {
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      matrix_64_48_, s64_vector_32_, ShapeUtil::MakeShape(F32, {65, 32}),
      to_apply_,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{0},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/1));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().error_message(),
      HasSubstr("Bounds of the window dimensions of updates must not exceed "
                "the bounds of the corresponding dimensions of operand."))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest, TfScatterWithUpdatesBiggerThanInputV2) {
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      matrix_64_48_, s64_vector_32_, ShapeUtil::MakeShape(F32, {32, 49}),
      to_apply_,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{1},
          /*inserted_window_dims=*/{0},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/1));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().error_message(),
      HasSubstr("Bounds of the window dimensions of updates must not exceed "
                "the bounds of the corresponding dimensions of operand."))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       TfScatterWithUpdatesNotMatchingIndices) {
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      matrix_64_48_, s64_vector_32_, ShapeUtil::MakeShape(F32, {64, 31}),
      to_apply_,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{0},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/1));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().error_message(),
      HasSubstr(
          "Bounds of the scatter dimensions of updates must be same as the "
          "bounds of the corresponding dimensions of scatter indices."))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       TfScatterWithUpdatesNotMatchingIndicesV2) {
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      matrix_64_48_, s64_vector_32_, ShapeUtil::MakeShape(F32, {31, 48}),
      to_apply_,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{1},
          /*inserted_window_dims=*/{0},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/1));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().error_message(),
      HasSubstr(
          "Bounds of the scatter dimensions of updates must be same as the "
          "bounds of the corresponding dimensions of scatter indices."))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest, TfScatterNdWithFullUpdates) {
  TF_ASSERT_OK_AND_ASSIGN(
      Shape scatter_shape,
      ShapeInference::InferScatterShape(
          matrix_64_48_, s64_4d_tensor_10_9_8_7_1_,
          ShapeUtil::MakeShape(F32, {10, 9, 8, 7, 48}), to_apply_,
          HloScatterInstruction::MakeScatterDimNumbers(
              /*update_window_dims=*/{4},
              /*inserted_window_dims=*/{0},
              /*scatter_dims_to_operand_dims=*/{0},
              /*index_vector_dim=*/4)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, matrix_64_48_))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, TfScatterNdWithFullUpdatesV2) {
  TF_ASSERT_OK_AND_ASSIGN(
      Shape scatter_shape,
      ShapeInference::InferScatterShape(
          matrix_64_48_, s64_4d_tensor_10_9_8_7_1_,
          ShapeUtil::MakeShape(F32, {10, 9, 8, 7, 64}), to_apply_,
          HloScatterInstruction::MakeScatterDimNumbers(
              /*update_window_dims=*/{4},
              /*inserted_window_dims=*/{1},
              /*scatter_dims_to_operand_dims=*/{0},
              /*index_vector_dim=*/4)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, matrix_64_48_))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, TfScatterNdWithPartialUpdates) {
  TF_ASSERT_OK_AND_ASSIGN(
      Shape scatter_shape,
      ShapeInference::InferScatterShape(
          matrix_64_48_, s64_4d_tensor_10_9_8_7_1_,
          ShapeUtil::MakeShape(F32, {10, 9, 8, 7, 10}), to_apply_,
          HloScatterInstruction::MakeScatterDimNumbers(
              /*update_window_dims=*/{4},
              /*inserted_window_dims=*/{0},
              /*scatter_dims_to_operand_dims=*/{0},
              /*index_vector_dim=*/4)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, matrix_64_48_))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, TfScatterNdWithPartialUpdatesV2) {
  TF_ASSERT_OK_AND_ASSIGN(
      Shape scatter_shape,
      ShapeInference::InferScatterShape(
          matrix_64_48_, s64_4d_tensor_10_9_8_7_1_,
          ShapeUtil::MakeShape(F32, {10, 9, 8, 7, 12}), to_apply_,
          HloScatterInstruction::MakeScatterDimNumbers(
              /*update_window_dims=*/{4},
              /*inserted_window_dims=*/{1},
              /*scatter_dims_to_operand_dims=*/{0},
              /*index_vector_dim=*/4)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, matrix_64_48_))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, TfScatterNdWithUpdatesBiggerThanInput) {
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      matrix_64_48_, s64_4d_tensor_10_9_8_7_1_,
      ShapeUtil::MakeShape(F32, {10, 9, 8, 7, 65}), to_apply_,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{0},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().error_message(),
      HasSubstr("Bounds of the window dimensions of updates must not exceed "
                "the bounds of the corresponding dimensions of operand."))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       TfScatterNdWithUpdatesNotMatchingIndices) {
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      matrix_64_48_, s64_4d_tensor_10_9_8_7_1_,
      ShapeUtil::MakeShape(F32, {9, 9, 8, 7, 64}), to_apply_,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{0},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().error_message(),
      HasSubstr(
          "Bounds of the scatter dimensions of updates must be same as the "
          "bounds of the corresponding dimensions of scatter indices."))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest, TfBatchDynamicUpdateSlice) {
  TF_ASSERT_OK_AND_ASSIGN(
      Shape scatter_shape,
      ShapeInference::InferScatterShape(
          f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
          ShapeUtil::MakeShape(F32, {10, 9, 8, 7, 30, 29, 28, 27, 26}),
          to_apply_,
          HloScatterInstruction::MakeScatterDimNumbers(
              /*update_window_dims=*/{4, 5, 6, 7, 8},
              /*inserted_window_dims=*/{},
              /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
              /*index_vector_dim=*/4)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, f32_5d_tensor_50_49_48_47_46_))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, NonDefaultScatterIndicesLeafDim) {
  TF_ASSERT_OK_AND_ASSIGN(
      Shape scatter_shape,
      ShapeInference::InferScatterShape(
          f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_5_7_6_,
          ShapeUtil::MakeShape(F32, {10, 9, 7, 6, 30, 29, 28, 27, 26}),
          to_apply_,
          HloScatterInstruction::MakeScatterDimNumbers(
              /*update_window_dims=*/{4, 5, 6, 7, 8},
              /*inserted_window_dims=*/{},
              /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
              /*index_vector_dim=*/2)));

  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, f32_5d_tensor_50_49_48_47_46_))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, NonDefaultScatterIndicesLeafDimV2) {
  TF_ASSERT_OK_AND_ASSIGN(
      Shape scatter_shape,
      ShapeInference::InferScatterShape(
          f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_5_10_9_7_6_,
          ShapeUtil::MakeShape(F32, {10, 9, 7, 6, 30, 29, 28, 27, 26}),
          to_apply_,
          HloScatterInstruction::MakeScatterDimNumbers(
              /*update_window_dims=*/{4, 5, 6, 7, 8},
              /*inserted_window_dims=*/{},
              /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
              /*index_vector_dim=*/0)));

  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, f32_5d_tensor_50_49_48_47_46_))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, NoUpdateScatterDims) {
  // This is equivalent to a dynamic update slice.
  TF_ASSERT_OK_AND_ASSIGN(
      Shape scatter_shape,
      ShapeInference::InferScatterShape(
          f32_5d_tensor_50_49_48_47_46_, s64_vector_5_,
          ShapeUtil::MakeShape(F32, {30, 29, 28, 27, 26}), to_apply_,
          HloScatterInstruction::MakeScatterDimNumbers(
              /*update_window_dims=*/{0, 1, 2, 3, 4},
              /*inserted_window_dims=*/{},
              /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
              /*index_vector_dim=*/0)));

  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, f32_5d_tensor_50_49_48_47_46_))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, ScalarScatterIndices) {
  // The scalar indices "tensor" is a scalar S here that's used to update a
  // [30,29,28,27] shaped tensor within the operand at position S.
  TF_ASSERT_OK_AND_ASSIGN(
      Shape scatter_shape,
      ShapeInference::InferScatterShape(
          f32_5d_tensor_50_49_48_47_46_, s64_scalar_,
          ShapeUtil::MakeShape(F32, {30, 29, 28, 27}), to_apply_,
          HloScatterInstruction::MakeScatterDimNumbers(
              /*update_window_dims=*/{0, 1, 2, 3},
              /*inserted_window_dims=*/{0},
              /*scatter_dims_to_operand_dims=*/{0},
              /*index_vector_dim=*/0)));

  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, f32_5d_tensor_50_49_48_47_46_))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_F(ScatterGatherShapeInferenceTest, ScatterWithTupleShapedTensorInput) {
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      tuple_shape_, s64_vector_32_, s64_vector_32_, to_apply_,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{0},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/1));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("Expected array argument for operand"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       ScatterWithTupleShapedScatterIndicesInput) {
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      s64_vector_32_, tuple_shape_, s64_vector_32_, to_apply_,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{0},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/0));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("Expected array argument for scatter indices"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest, ScatterWithTupleShapedUpdatesInput) {
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      s64_vector_32_, s64_vector_32_, tuple_shape_, to_apply_,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{0},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/0));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("Expected array argument for updates"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest, FloatingPointScatterIndicesInput) {
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      s64_vector_32_, vector_32_, s64_vector_32_, to_apply_,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{0},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/0));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("Scatter indices parameter must be an integral tensor"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest, OutOfBoundsScatterIndicesLeafDim) {
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      ShapeUtil::MakeShape(F32, {10, 9, 8, 7, 30, 29, 28}), to_apply_,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 2},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/10));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("Scatter index leaf dimension must be within [0, "
                        "rank(scatter_indices) + 1)"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest, InvalidUpdates) {
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      ShapeUtil::MakeShape(F32, {10, 9, 8, 7, 30, 29, 28, 50}), to_apply_,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 2},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("Updates tensor must be of rank 7; got 8."))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest, InvalidUpdateComputation) {
  const ProgramShape invalid_update_computation =
      ShapeUtil::MakeProgramShape({f32_}, f32_);
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      ShapeUtil::MakeShape(F32, {10, 9, 8, 7, 30, 29, 28}),
      invalid_update_computation,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 2},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().error_message(),
      HasSubstr("Reduction function must take 2 parameters, but takes 1"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidScatterDimNumbers_NonAscendingUpdateWindowDims) {
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      ShapeUtil::MakeShape(F32, {10, 9, 8, 7, 30, 29, 28, 27, 26}), to_apply_,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6, 8, 7},
          /*inserted_window_dims=*/{},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("update_window_dims in scatter op must be sorted"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidScatterDimNumbers_RepeatedUpdateWindowDims) {
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      ShapeUtil::MakeShape(F32, {10, 9, 8, 7, 30, 29, 28, 27, 26}), to_apply_,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6, 7, 7},
          /*inserted_window_dims=*/{},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("update_window_dims in scatter op must not repeat"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidScatterDimNumbers_OutOfBoundsUpdateWindowDims) {
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      ShapeUtil::MakeShape(F32, {10, 9, 8, 7, 30, 29, 28, 27, 26}), to_apply_,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6, 7, 9},
          /*inserted_window_dims=*/{},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("Invalid update_window_dims set in scatter op; valid "
                        "range is [0, 9)"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidScatterDimNumbers_NonAscendingInsertedWindowDims) {
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      ShapeUtil::MakeShape(F32, {10, 9, 8, 7, 30, 29, 28}), to_apply_,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{2, 1},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("inserted_window_dims in scatter op must be sorted"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidScatterDimNumbers_RepeatedInsertedWindowDims) {
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      ShapeUtil::MakeShape(F32, {10, 9, 8, 7, 30, 29, 28}), to_apply_,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 1},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("inserted_window_dims in scatter op must not repeat"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidScatterDimNumbers_OutOfBoundsInsertedWindowDims) {
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      ShapeUtil::MakeShape(F32, {10, 9, 8, 7, 30, 29, 28}), to_apply_,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 5},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("Invalid inserted_window_dims set in scatter op; valid "
                        "range is [0, 5)"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidScatterDimNumbers_MismatchingScatterDimsToOperandDims) {
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      ShapeUtil::MakeShape(F32, {10, 9, 8, 7, 30, 29, 28}), to_apply_,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 2},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().error_message(),
      HasSubstr("Scatter op has 4 elements in scatter_dims_to_operand_dims and "
                "the bound of dimension index_vector_dim=4 of scatter_indices "
                "is 5. These two numbers must be equal"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidScatterDimNumbers_OutOfBoundsScatterDimsToOperandDims) {
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      ShapeUtil::MakeShape(F32, {10, 9, 8, 7, 30, 29, 28}), to_apply_,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 2},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 10},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("Invalid scatter_dims_to_operand_dims mapping; domain "
                        "is [0, 5), got: 4->10"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidScatterDimNumbers_RepeatedValuesInScatterDimsToOperandDims) {
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      ShapeUtil::MakeShape(F32, {10, 9, 8, 7, 30, 29, 28}), to_apply_,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 2},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 2, 3},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().error_message(),
      HasSubstr(
          "Repeated dimensions not allowed in scatter_dims_to_operand_dims"))
      << statusor.status();
}

TEST_F(ScatterGatherShapeInferenceTest,
       InvalidScatterDimNumbers_InsufficientWindowDims) {
  StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      f32_5d_tensor_50_49_48_47_46_, s64_scalar_,
      ShapeUtil::MakeShape(F32, {30, 29, 28, 27}), to_apply_,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{0, 1, 2, 3},
          /*inserted_window_dims=*/{},
          /*scatter_dims_to_operand_dims=*/{0},
          /*index_vector_dim=*/0));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().error_message(),
      HasSubstr(
          "Scatter op has window of size 4; doesn't match operand of rank 5."))
      << statusor.status();
}

}  // namespace
}  // namespace xla
