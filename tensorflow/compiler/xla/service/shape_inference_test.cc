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

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace {

class ShapeInferenceTest : public ::testing::Test {
 protected:
  // Some handy scalar shapes.
  const Shape s32_ = ShapeUtil::MakeShape(S32, {});
  const Shape f32_ = ShapeUtil::MakeShape(F32, {});
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
  void ExpectInferredReduceShape(
      const Shape& expected_inferred_shape, const Shape& arg,
      tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce) {
    ProgramShape to_apply = ShapeUtil::MakeProgramShape({f32_, f32_}, f32_);
    auto inferred_status = ShapeInference::InferReduceShape(
        arg, f32_, dimensions_to_reduce, to_apply);
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
  auto inferred_status = ShapeInference::InferUnaryOpShape(
      UnaryOperation::UNOP_NEGATE, matrix_shape);
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_shape, inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, SelectScalarPredBetweenTuples) {
  Shape tuple = ShapeUtil::MakeTupleShape({s32_, f32_});
  auto inferred_status = ShapeInference::InferTernaryOpShape(
      TernaryOperation::TRIOP_SELECT, pred_, tuple, tuple);
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(tuple, inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, SelectScalarPredBetweenArrays) {
  auto inferred_status = ShapeInference::InferTernaryOpShape(
      TernaryOperation::TRIOP_SELECT, pred_, matrix_64_48_, matrix_64_48_);
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, SelectArrayPredBetweenArrays) {
  auto predarray = ShapeUtil::MakeShape(PRED, {64, 48});
  auto inferred_status = ShapeInference::InferTernaryOpShape(
      TernaryOperation::TRIOP_SELECT, predarray, matrix_64_48_, matrix_64_48_);
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, SelectBadShapes) {
  auto inferred_status_error1 = ShapeInference::InferTernaryOpShape(
      TernaryOperation::TRIOP_SELECT, pred_, matrix_64_48_, matrix_32_64_);
  ASSERT_FALSE(inferred_status_error1.ok());
  ASSERT_MATCH(
      inferred_status_error1.status().error_message(),
      testing::ContainsRegex("operands to select must be the same shape"));

  auto inferred_status_error2 = ShapeInference::InferTernaryOpShape(
      TernaryOperation::TRIOP_SELECT, s32_, matrix_64_48_, matrix_64_48_);
  ASSERT_FALSE(inferred_status_error2.ok());
  ASSERT_MATCH(inferred_status_error2.status().error_message(),
               testing::ContainsRegex("pred operand must have PRED"));

  auto inferred_status_error3 = ShapeInference::InferTernaryOpShape(
      TernaryOperation::TRIOP_SELECT, ShapeUtil::MakeShape(PRED, {64}),
      matrix_64_48_, matrix_64_48_);
  ASSERT_FALSE(inferred_status_error3.ok());
  ASSERT_MATCH(
      inferred_status_error3.status().error_message(),
      testing::ContainsRegex("with non-scalar predicate with dimensionality"));

  // Tuples have a TUPLE element type and cannot be the pred of a select.
  auto inferred_status_error4 = ShapeInference::InferTernaryOpShape(
      TernaryOperation::TRIOP_SELECT, ShapeUtil::MakeTupleShape({pred_, pred_}),
      ShapeUtil::MakeTupleShape({f32_, f32_}),
      ShapeUtil::MakeTupleShape({f32_, f32_}));
  ASSERT_FALSE(inferred_status_error4.ok());
  ASSERT_MATCH(
      inferred_status_error4.status().error_message(),
      testing::ContainsRegex("pred operand must have PRED element type"));
}

TEST_F(ShapeInferenceTest, ClampAllMatrix) {
  auto inferred_status = ShapeInference::InferTernaryOpShape(
      TernaryOperation::TRIOP_CLAMP, matrix_64_48_, matrix_64_48_,
      matrix_64_48_);
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, ClampAllScalar) {
  auto inferred_status = ShapeInference::InferTernaryOpShape(
      TernaryOperation::TRIOP_CLAMP, f32_, f32_, f32_);
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(f32_, inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, ClampMinScalar) {
  auto inferred_status = ShapeInference::InferTernaryOpShape(
      TernaryOperation::TRIOP_CLAMP, f32_, matrix_64_48_, matrix_64_48_);
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, ClampMaxScalar) {
  auto inferred_status = ShapeInference::InferTernaryOpShape(
      TernaryOperation::TRIOP_CLAMP, matrix_64_48_, matrix_64_48_, f32_);
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, ClampOperandScalar) {
  auto inferred_status = ShapeInference::InferTernaryOpShape(
      TernaryOperation::TRIOP_CLAMP, matrix_64_48_, f32_, matrix_64_48_);
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, ClampMinMatrix) {
  auto inferred_status = ShapeInference::InferTernaryOpShape(
      TernaryOperation::TRIOP_CLAMP, matrix_64_48_, f32_, f32_);
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, ClampMaxMatrix) {
  auto inferred_status = ShapeInference::InferTernaryOpShape(
      TernaryOperation::TRIOP_CLAMP, f32_, f32_, matrix_64_48_);
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, ClampOperandMatrix) {
  auto inferred_status = ShapeInference::InferTernaryOpShape(
      TernaryOperation::TRIOP_CLAMP, f32_, matrix_64_48_, f32_);
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, ClampBadShapes) {
  // Type mismatch
  ASSERT_FALSE(ShapeInference::InferTernaryOpShape(
                   TernaryOperation::TRIOP_CLAMP, s32_, f32_, f32_)
                   .ok());
  ASSERT_FALSE(ShapeInference::InferTernaryOpShape(
                   TernaryOperation::TRIOP_CLAMP, f32_, s32_, f32_)
                   .ok());
  ASSERT_FALSE(ShapeInference::InferTernaryOpShape(
                   TernaryOperation::TRIOP_CLAMP, f32_, f32_, s32_)
                   .ok());
  // Dimension mismatch
  ASSERT_FALSE(
      ShapeInference::InferTernaryOpShape(TernaryOperation::TRIOP_CLAMP,
                                          vector_64_, vector_32_, vector_32_)
          .ok());
  ASSERT_FALSE(
      ShapeInference::InferTernaryOpShape(TernaryOperation::TRIOP_CLAMP,
                                          vector_32_, vector_64_, vector_32_)
          .ok());
  ASSERT_FALSE(
      ShapeInference::InferTernaryOpShape(TernaryOperation::TRIOP_CLAMP,
                                          vector_32_, vector_32_, vector_64_)
          .ok());
  // Dimension mismatch, where one operand is a scalar
  ASSERT_FALSE(ShapeInference::InferTernaryOpShape(
                   TernaryOperation::TRIOP_CLAMP, vector_64_, vector_32_, f32_)
                   .ok());
  ASSERT_FALSE(ShapeInference::InferTernaryOpShape(
                   TernaryOperation::TRIOP_CLAMP, vector_64_, f32_, vector_32_)
                   .ok());
  ASSERT_FALSE(ShapeInference::InferTernaryOpShape(
                   TernaryOperation::TRIOP_CLAMP, f32_, vector_64_, vector_32_)
                   .ok());
}

TEST_F(ShapeInferenceTest, VariadicOpTuplify) {
  StatusOr<Shape> result = ShapeInference::InferVariadicOpShape(
      VariadicOperation::VAROP_TUPLE, {&s32_, &f32_});
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
  ASSERT_MATCH(inferred_status_fail.status().error_message(),
               testing::ContainsRegex("source shape does not match"));
}

TEST_F(SelectAndScatterShapeInferenceTest, SelectAndScatterWrongSelectShape1) {
  ProgramShape select_program_shape_fail =
      ShapeUtil::MakeProgramShape({ShapeUtil::MakeShape(F32, {})}, pred_);
  auto inferred_status_fail = ShapeInference::InferSelectAndScatterShape(
      operand_shape_, select_program_shape_fail, window_, source_shape_,
      init_value_shape_, scatter_program_shape_);
  ASSERT_FALSE(inferred_status_fail.ok());
  ASSERT_MATCH(
      inferred_status_fail.status().error_message(),
      testing::ContainsRegex("select function must take 2 parameters"));
}

TEST_F(SelectAndScatterShapeInferenceTest, SelectAndScatterWrongSelectShape2) {
  ProgramShape select_program_shape_fail = ShapeUtil::MakeProgramShape(
      {ShapeUtil::MakeShape(F32, {}), ShapeUtil::MakeShape(F32, {})}, f32_);
  auto inferred_status_fail = ShapeInference::InferSelectAndScatterShape(
      operand_shape_, select_program_shape_fail, window_, source_shape_,
      init_value_shape_, scatter_program_shape_);
  ASSERT_FALSE(inferred_status_fail.ok());
  ASSERT_MATCH(inferred_status_fail.status().error_message(),
               testing::ContainsRegex("select function must have rank-0 PRED"));
}

TEST_F(SelectAndScatterShapeInferenceTest, SelectAndScatterWrongSelectShape3) {
  ProgramShape select_program_shape_fail = ShapeUtil::MakeProgramShape(
      {ShapeUtil::MakeShape(S32, {}), ShapeUtil::MakeShape(F32, {})}, pred_);
  auto inferred_status_fail = ShapeInference::InferSelectAndScatterShape(
      operand_shape_, select_program_shape_fail, window_, source_shape_,
      init_value_shape_, scatter_program_shape_);
  ASSERT_FALSE(inferred_status_fail.ok());
  ASSERT_MATCH(inferred_status_fail.status().error_message(),
               testing::ContainsRegex("select function's first parameter"));
}

TEST_F(SelectAndScatterShapeInferenceTest, SelectAndScatterWrongSelectShape4) {
  ProgramShape select_program_shape_fail = ShapeUtil::MakeProgramShape(
      {ShapeUtil::MakeShape(F32, {}), ShapeUtil::MakeShape(U32, {})}, pred_);
  auto inferred_status_fail = ShapeInference::InferSelectAndScatterShape(
      operand_shape_, select_program_shape_fail, window_, source_shape_,
      init_value_shape_, scatter_program_shape_);
  ASSERT_FALSE(inferred_status_fail.ok());
  ASSERT_MATCH(inferred_status_fail.status().error_message(),
               testing::ContainsRegex("select function's second parameter"));
}

TEST_F(ShapeInferenceTest, Convolve) {
  ConvolutionDimensionNumbers dnums;

  // Dimension order: batch, feature, x0, x1
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {10, 11, 3, 4});
  dnums.set_batch_dimension(0);
  dnums.set_feature_dimension(1);
  dnums.add_spatial_dimensions(2);
  dnums.add_spatial_dimensions(3);

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
  auto inferred_status =
      ShapeInference::InferConvolveShape(lhs_shape, rhs_shape, window, dnums);
  ASSERT_IS_OK(inferred_status.status());
  Shape inferred_shape = inferred_status.ValueOrDie();
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {10, 12, 2, 3}),
                               inferred_shape));
}

TEST_F(ShapeInferenceTest, ConvolveWithWindowDilation) {
  ConvolutionDimensionNumbers dnums;

  // Dimension order: batch, feature, x0, x1
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {10, 11, 103, 4});
  dnums.set_batch_dimension(0);
  dnums.set_feature_dimension(1);
  dnums.add_spatial_dimensions(2);
  dnums.add_spatial_dimensions(3);

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
  auto inferred_status =
      ShapeInference::InferConvolveShape(lhs_shape, rhs_shape, window, dnums);
  ASSERT_IS_OK(inferred_status.status());
  Shape inferred_shape = inferred_status.ValueOrDie();
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {10, 12, 31, 5}),
                               inferred_shape));
}

TEST_F(ShapeInferenceTest, ConvolveWithBaseDilation) {
  ConvolutionDimensionNumbers dnums;

  // Dimension order: batch, feature, x0, x1
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {10, 11, 3, 4});
  dnums.set_batch_dimension(0);
  dnums.set_feature_dimension(1);
  dnums.add_spatial_dimensions(2);
  dnums.add_spatial_dimensions(3);

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
  auto inferred_status =
      ShapeInference::InferConvolveShape(lhs_shape, rhs_shape, window, dnums);
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
  dnums.set_batch_dimension(3);
  dnums.set_feature_dimension(2);
  dnums.add_spatial_dimensions(0);
  dnums.add_spatial_dimensions(1);
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
  auto inferred_status =
      ShapeInference::InferConvolveShape(lhs_shape, rhs_shape, window, dnums);
  ASSERT_FALSE(inferred_status.ok());
  ASSERT_MATCH(inferred_status.status().error_message(),
               testing::ContainsRegex("each dimension exactly once"));
}

TEST_F(ShapeInferenceTest, MapThatChangesElementType) {
  Shape arg = ShapeUtil::MakeShape(F32, {20});
  ProgramShape to_apply = ShapeUtil::MakeProgramShape({f32_}, s32_);
  auto inferred_status = ShapeInference::InferMapShape({&arg}, to_apply);
  EXPECT_IS_OK(inferred_status.status());
  Shape expected = ShapeUtil::MakeShape(S32, {20});
  EXPECT_TRUE(ShapeUtil::Equal(expected, inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, Map) {
  auto inferred_status_r1f32 = ShapeInference::InferMapShape(
      {&vector_32_, &vector_32_},
      ShapeUtil::MakeProgramShape({f32_, f32_}, f32_));
  EXPECT_IS_OK(inferred_status_r1f32.status());
  EXPECT_TRUE(ShapeUtil::Equal(vector_32_, inferred_status_r1f32.ValueOrDie()));

  // It's OK to provide a single argument, as long as the applied arity matches
  // (this degenerates to a Map).
  auto inferred_status_r1f32_one = ShapeInference::InferMapShape(
      {&vector_32_}, ShapeUtil::MakeProgramShape({f32_}, f32_));
  EXPECT_IS_OK(inferred_status_r1f32_one.status());
  EXPECT_TRUE(
      ShapeUtil::Equal(vector_32_, inferred_status_r1f32_one.ValueOrDie()));

  auto inferred_status_r2s32 = ShapeInference::InferMapShape(
      {&s32matrix_64_64_, &s32matrix_64_64_, &s32matrix_64_64_},
      ShapeUtil::MakeProgramShape({s32_, s32_, s32_}, s32_));
  EXPECT_IS_OK(inferred_status_r2s32.status());
  EXPECT_TRUE(
      ShapeUtil::Equal(s32matrix_64_64_, inferred_status_r2s32.ValueOrDie()));

  auto no_args_error = ShapeInference::InferMapShape(
      {}, ShapeUtil::MakeProgramShape({f32_, f32_}, f32_));
  ASSERT_FALSE(no_args_error.ok());
  ASSERT_MATCH(no_args_error.status().error_message(),
               testing::ContainsRegex("expects at least one argument"));

  auto args_diff_shapes_error = ShapeInference::InferMapShape(
      {&vector_32_, &vector_64_},
      ShapeUtil::MakeProgramShape({f32_, f32_}, f32_));
  ASSERT_FALSE(args_diff_shapes_error.ok());
  ASSERT_MATCH(
      args_diff_shapes_error.status().error_message(),
      testing::ContainsRegex("requires all operands to have the same shape"));

  auto arity_error = ShapeInference::InferMapShape(
      {&vector_32_, &vector_32_}, ShapeUtil::MakeProgramShape({f32_}, f32_));
  ASSERT_FALSE(arity_error.ok());
  ASSERT_MATCH(arity_error.status().error_message(),
               testing::ContainsRegex("function arity must match"));

  auto output_shape_error = ShapeInference::InferMapShape(
      {&vector_32_, &vector_32_},
      ShapeUtil::MakeProgramShape({f32_, f32_}, vector_32_));
  ASSERT_FALSE(output_shape_error.ok());
  ASSERT_MATCH(output_shape_error.status().error_message(),
               testing::ContainsRegex("result has to be a scalar"));

  auto param_shape_error = ShapeInference::InferMapShape(
      {&vector_32_, &vector_32_},
      ShapeUtil::MakeProgramShape({vector_32_, f32_}, f32_));
  ASSERT_FALSE(param_shape_error.ok());
  ASSERT_MATCH(param_shape_error.status().error_message(),
               testing::ContainsRegex("parameter has to be a scalar"));

  auto param_element_type_error = ShapeInference::InferMapShape(
      {&vector_32_, &vector_32_},
      ShapeUtil::MakeProgramShape({f32_, s32_}, f32_));
  ASSERT_FALSE(param_element_type_error.ok());
  ASSERT_MATCH(param_element_type_error.status().error_message(),
               testing::ContainsRegex("parameter type has to match argument"));

  Shape arg = ShapeUtil::MakeShape(F32, {20});
  ProgramShape to_apply = ShapeUtil::MakeProgramShape({f32_}, f32_);
  auto inferred_status = ShapeInference::InferMapShape({&arg}, to_apply);
  EXPECT_IS_OK(inferred_status.status());
  EXPECT_TRUE(ShapeUtil::Equal(arg, inferred_status.ValueOrDie()));

  auto inferred_status_error1 = ShapeInference::InferMapShape(
      {&arg}, ShapeUtil::MakeProgramShape({f32_, f32_}, f32_));
  ASSERT_FALSE(inferred_status_error1.ok());
  ASSERT_MATCH(inferred_status_error1.status().error_message(),
               testing::ContainsRegex("arity must match number of arguments"));

  auto inferred_status_error2 = ShapeInference::InferMapShape(
      {&arg}, ShapeUtil::MakeProgramShape({vector_32_}, f32_));
  ASSERT_FALSE(inferred_status_error2.ok());
  ASSERT_MATCH(inferred_status_error2.status().error_message(),
               testing::ContainsRegex("has to be a scalar"));

  auto inferred_status_error3 = ShapeInference::InferMapShape(
      {&arg}, ShapeUtil::MakeProgramShape({f32_}, vector_32_));
  ASSERT_FALSE(inferred_status_error3.ok());
  ASSERT_MATCH(inferred_status_error3.status().error_message(),
               testing::ContainsRegex("has to be a scalar"));

  auto inferred_status_error5 = ShapeInference::InferMapShape(
      {&arg}, ShapeUtil::MakeProgramShape({s32_}, s32_));
  ASSERT_FALSE(inferred_status_error5.ok());
  ASSERT_MATCH(inferred_status_error5.status().error_message(),
               testing::ContainsRegex("parameter type has to match argument"));
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

TEST_F(ReduceShapeInferenceTest, ErrorOutOfBoundsDimension) {
  ProgramShape to_apply = ShapeUtil::MakeProgramShape({f32_, f32_}, f32_);
  auto inferred_status = ShapeInference::InferReduceShape(
      ShapeUtil::MakeShape(F32, {5, 3}), f32_, /*dimensions_to_reduce=*/{3, 4},
      to_apply);
  EXPECT_FALSE(inferred_status.ok());
  EXPECT_MATCH(inferred_status.status().error_message(),
               testing::ContainsRegex("out-of-bounds dimension"));
}

TEST_F(ReduceShapeInferenceTest, ErrorToApplyArity) {
  ProgramShape to_apply = ShapeUtil::MakeProgramShape({f32_, f32_, f32_}, f32_);
  auto inferred_status =
      ShapeInference::InferReduceShape(ShapeUtil::MakeShape(F32, {5, 3}), f32_,
                                       /*dimensions_to_reduce=*/{0}, to_apply);
  EXPECT_FALSE(inferred_status.ok());
  EXPECT_MATCH(inferred_status.status().error_message(),
               testing::ContainsRegex("take 2 parameters"));
}

TEST_F(ReduceShapeInferenceTest, ErrorElementTypeVsApplyType) {
  ProgramShape to_apply = ShapeUtil::MakeProgramShape({f32_, f32_}, s32_);
  auto inferred_status =
      ShapeInference::InferReduceShape(ShapeUtil::MakeShape(F32, {5, 3}), f32_,
                                       /*dimensions_to_reduce=*/{0}, to_apply);
  EXPECT_FALSE(inferred_status.ok());
  EXPECT_MATCH(inferred_status.status().error_message(),
               testing::ContainsRegex("first parameter shape differs"));
}

TEST_F(ShapeInferenceTest, InferSliceShapeRank2) {
  Shape matrix_shape = ShapeUtil::MakeShape(F32, {128, 64});
  auto inferred_status =
      ShapeInference::InferSliceShape(matrix_shape, {32, 0}, {64, 64});
  ASSERT_IS_OK(inferred_status.status());
  Shape inferred = inferred_status.ValueOrDie();
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {32, 64}), inferred));
}

TEST_F(ShapeInferenceTest, InferOobSliceShapeRank2) {
  Shape matrix_shape = ShapeUtil::MakeShape(F32, {128, 64});
  auto inferred_status =
      ShapeInference::InferSliceShape(matrix_shape, {127, 0}, {129, 2});
  ASSERT_FALSE(inferred_status.ok());
  ASSERT_EQ(tensorflow::error::INVALID_ARGUMENT,
            inferred_status.status().code());
}

TEST_F(ShapeInferenceTest, InferSliceShapeRank1) {
  Shape vector_shape = ShapeUtil::MakeShape(F32, {17});
  auto inferred_status =
      ShapeInference::InferSliceShape(vector_shape, {2}, {4});
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

TEST_F(ShapeInferenceTest, InferPowShape) {
  auto ten_floats = ShapeUtil::MakeShape(F32, {10});
  auto inferred_status =
      ShapeInference::InferBinaryOpShape(BINOP_POW, ten_floats, f32_, {});
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(ten_floats, inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, InferCompareShapeEq) {
  auto ten_floats = ShapeUtil::MakeShape(F32, {10});
  auto inferred_status =
      ShapeInference::InferBinaryOpShape(BINOP_EQ, ten_floats, f32_, {});
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(PRED, {10}),
                               inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, InferCompareShapeGe) {
  auto ten_floats = ShapeUtil::MakeShape(F32, {10});
  auto inferred_status =
      ShapeInference::InferBinaryOpShape(BINOP_GE, ten_floats, f32_, {});
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(PRED, {10}),
                               inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, InferCompareShapeGt) {
  auto ten_floats = ShapeUtil::MakeShape(F32, {10});
  auto inferred_status =
      ShapeInference::InferBinaryOpShape(BINOP_GT, ten_floats, f32_, {});
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(PRED, {10}),
                               inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, InferCompareShapeLe) {
  auto ten_floats = ShapeUtil::MakeShape(F32, {10});
  auto inferred_status =
      ShapeInference::InferBinaryOpShape(BINOP_LE, ten_floats, f32_, {});
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(PRED, {10}),
                               inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, InferCompareShapeLt) {
  auto ten_floats = ShapeUtil::MakeShape(F32, {10});
  auto inferred_status =
      ShapeInference::InferBinaryOpShape(BINOP_LT, ten_floats, f32_, {});
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(PRED, {10}),
                               inferred_status.ValueOrDie()));
}

TEST_F(ShapeInferenceTest, InferCompareShapeNe) {
  auto ten_floats = ShapeUtil::MakeShape(F32, {10});
  auto inferred_status =
      ShapeInference::InferBinaryOpShape(BINOP_NE, ten_floats, f32_, {});
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(PRED, {10}),
                               inferred_status.ValueOrDie()));
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

// scalar <dot> vector: error
TEST_F(ShapeInferenceTest, ScalarDotVector) {
  auto inferred_status =
      ShapeInference::InferBinaryOpShape(BINOP_DOT, f32_, vector_32_, {});
  ASSERT_FALSE(inferred_status.ok());
  ASSERT_MATCH(inferred_status.status().error_message(),
               testing::ContainsRegex("dot only supports rank"));
}

// 3D <dot> 2D: error
TEST_F(ShapeInferenceTest, DotWithRankHigherThanTwo) {
  auto inferred_status = ShapeInference::InferBinaryOpShape(
      BINOP_DOT, ShapeUtil::MakeShape(F32, {32, 32, 32}), matrix_32_64_, {});
  ASSERT_FALSE(inferred_status.ok());
  ASSERT_MATCH(inferred_status.status().error_message(),
               testing::ContainsRegex("dot only supports rank"));
}

// vector <dot> vector -> scalar
TEST_F(ShapeInferenceTest, VectorDotVector) {
  auto inferred_status =
      ShapeInference::InferBinaryOpShape(BINOP_DOT, vector_64_, vector_64_, {});
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(f32_, inferred_status.ValueOrDie()));
  auto inferred_status_mismatch =
      ShapeInference::InferBinaryOpShape(BINOP_DOT, vector_64_, vector_32_, {});
  ASSERT_FALSE(inferred_status_mismatch.ok());
}

// matrix <dot> vector -> vector
TEST_F(ShapeInferenceTest, MatrixDotVector) {
  auto inferred_status = ShapeInference::InferBinaryOpShape(
      BinaryOperation::BINOP_DOT, matrix_32_64_, vector_64_, {});
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(inferred_status.ValueOrDie(), vector_32_));
  auto inferred_status_mismatch = ShapeInference::InferBinaryOpShape(
      BinaryOperation::BINOP_DOT, matrix_32_64_, vector_32_, {});
  ASSERT_FALSE(inferred_status_mismatch.ok());
}

// vector <dot> matrix -> vector
TEST_F(ShapeInferenceTest, VectorDotMatrix) {
  auto inferred_status = ShapeInference::InferBinaryOpShape(
      BinaryOperation::BINOP_DOT, vector_32_, matrix_32_64_, {});
  ASSERT_IS_OK(inferred_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(inferred_status.ValueOrDie(), vector_64_));
  auto inferred_status_mismatch = ShapeInference::InferBinaryOpShape(
      BinaryOperation::BINOP_DOT, vector_64_, matrix_32_64_, {});
  ASSERT_FALSE(inferred_status_mismatch.ok());
}

// matrix <dot> matrix -> matrix
TEST_F(ShapeInferenceTest, MatrixDotMatrix) {
  auto inferred_status_match = ShapeInference::InferBinaryOpShape(
      BinaryOperation::BINOP_DOT, matrix_32_64_, matrix_64_48_, {});
  ASSERT_IS_OK(inferred_status_match.status());
  ASSERT_TRUE(
      ShapeUtil::Equal(inferred_status_match.ValueOrDie(), matrix_32_48_))
      << "inferred: "
      << ShapeUtil::HumanString(inferred_status_match.ValueOrDie())
      << " expected: " << ShapeUtil::HumanString(matrix_64_48_);
  auto inferred_status_mismatch = ShapeInference::InferBinaryOpShape(
      BinaryOperation::BINOP_DOT, matrix_32_64_, matrix_32_64_, {});
  ASSERT_FALSE(inferred_status_mismatch.ok());
}

TEST_F(ShapeInferenceTest, BinOpBroadcastMatrixVector) {
  // Test variations of broadcasting a vector for a binary add with a
  // matrix.
  const Shape mat = ShapeUtil::MakeShape(F32, {16, 8});
  const Shape vec8 = ShapeUtil::MakeShape(F32, {8});
  const Shape vec16 = ShapeUtil::MakeShape(F32, {16});

  auto inferred_status_match = ShapeInference::InferBinaryOpShape(
      BinaryOperation::BINOP_ADD, mat, vec8, {1});
  ASSERT_IS_OK(inferred_status_match.status());
  ASSERT_TRUE(ShapeUtil::Equal(inferred_status_match.ValueOrDie(), mat));

  auto inferred_status_mismatch = ShapeInference::InferBinaryOpShape(
      BinaryOperation::BINOP_ADD, mat, vec8, {0});
  ASSERT_FALSE(inferred_status_mismatch.ok());

  inferred_status_match = ShapeInference::InferBinaryOpShape(
      BinaryOperation::BINOP_ADD, mat, vec16, {0});
  ASSERT_IS_OK(inferred_status_match.status());
  ASSERT_TRUE(ShapeUtil::Equal(inferred_status_match.ValueOrDie(), mat));

  inferred_status_mismatch = ShapeInference::InferBinaryOpShape(
      BinaryOperation::BINOP_ADD, mat, vec16, {1});
  ASSERT_FALSE(inferred_status_mismatch.ok());
}

TEST_F(ShapeInferenceTest, BinOpBroadcastCubeMatrix) {
  // Test variations of broadcasting a matrix for a binary add with a cube.
  const Shape cube = ShapeUtil::MakeShape(F32, {16, 8, 4});
  const Shape matrix8_4 = ShapeUtil::MakeShape(F32, {8, 4});
  const Shape matrix16_4 = ShapeUtil::MakeShape(F32, {16, 4});
  const Shape matrix16_8 = ShapeUtil::MakeShape(F32, {16, 8});

  auto inferred_status_match = ShapeInference::InferBinaryOpShape(
      BinaryOperation::BINOP_ADD, cube, matrix8_4, {1, 2});
  ASSERT_IS_OK(inferred_status_match.status());
  ASSERT_TRUE(ShapeUtil::Equal(inferred_status_match.ValueOrDie(), cube));

  inferred_status_match = ShapeInference::InferBinaryOpShape(
      BinaryOperation::BINOP_ADD, cube, matrix16_4, {0, 2});
  ASSERT_IS_OK(inferred_status_match.status());
  ASSERT_TRUE(ShapeUtil::Equal(inferred_status_match.ValueOrDie(), cube));

  inferred_status_match = ShapeInference::InferBinaryOpShape(
      BinaryOperation::BINOP_ADD, cube, matrix16_8, {0, 1});
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
  auto inferred_status_error1 = ShapeInference::InferBinaryOpShape(
      BinaryOperation::BINOP_ADD, tensor, vec8, {});
  ASSERT_FALSE(inferred_status_error1.ok());
  ASSERT_MATCH(inferred_status_error1.status().error_message(),
               testing::ContainsRegex("automatic"));

  // broadcast_dimension out of bounds for tensor's rank
  auto inferred_status_error2 = ShapeInference::InferBinaryOpShape(
      BinaryOperation::BINOP_ADD, tensor, vec8, {3});
  ASSERT_FALSE(inferred_status_error2.ok());
  ASSERT_MATCH(
      inferred_status_error2.status().error_message(),
      testing::ContainsRegex("broadcast dimension number .* too large"));

  // broadcast_dimension doesn't match corresponding dimension
  auto inferred_status_error3 = ShapeInference::InferBinaryOpShape(
      BinaryOperation::BINOP_ADD, tensor, vec8, {0});
  ASSERT_FALSE(inferred_status_error3.ok());
  ASSERT_MATCH(inferred_status_error3.status().error_message(),
               testing::ContainsRegex("broadcast dimension 0 mismatch"));

  // broadcast_dimensions list too long
  auto inferred_status_error4 = ShapeInference::InferBinaryOpShape(
      BinaryOperation::BINOP_ADD, tensor, matrix8_4, {0, 1, 2});
  ASSERT_FALSE(inferred_status_error4.ok());
  ASSERT_MATCH(
      inferred_status_error4.status().error_message(),
      testing::ContainsRegex("size of broadcast_dimensions has to match"));

  // there's a dimension above the rank of the tensor
  auto inferred_status_error5 = ShapeInference::InferBinaryOpShape(
      BinaryOperation::BINOP_ADD, tensor, matrix8_4, {3, 0});
  ASSERT_FALSE(inferred_status_error5.ok());
  ASSERT_MATCH(
      inferred_status_error5.status().error_message(),
      testing::ContainsRegex("broadcast dimension number .* too large"));

  // broadcasting dimensions don't match in this order
  auto inferred_status_error6 = ShapeInference::InferBinaryOpShape(
      BinaryOperation::BINOP_ADD, tensor, matrix8_4, {2, 1});
  ASSERT_FALSE(inferred_status_error6.ok());
  ASSERT_MATCH(inferred_status_error6.status().error_message(),
               testing::ContainsRegex("broadcast dimension 0 mismatch"));

  // The following two tests make sure that broadcasting dimensions are listed
  // in a proper (strictly increasing) order, even if the lower-rank array
  // matches the higher-rank array in many different ways.
  auto inferred_status_error7 = ShapeInference::InferBinaryOpShape(
      BinaryOperation::BINOP_ADD, tensor8_8_8, matrix8_8, {0, 0});
  ASSERT_FALSE(inferred_status_error7.ok());
  ASSERT_MATCH(inferred_status_error7.status().error_message(),
               testing::ContainsRegex("broadcast dimensions order is wrong"));

  auto inferred_status_error8 = ShapeInference::InferBinaryOpShape(
      BinaryOperation::BINOP_ADD, tensor8_8_8, matrix8_8, {1, 0});
  ASSERT_FALSE(inferred_status_error8.ok());
  ASSERT_MATCH(inferred_status_error8.status().error_message(),
               testing::ContainsRegex("broadcast dimensions order is wrong"));
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
  ASSERT_MATCH(inferred_status_error1.status().error_message(),
               testing::ContainsRegex("condition must take 1 arguments"));

  auto bad_shape_2 =
      ShapeUtil::MakeProgramShape({s32_, result_shape}, result_shape);
  auto inferred_status_error2 =
      ShapeInference::InferWhileShape(cond, bad_shape_2, result_shape);
  ASSERT_FALSE(inferred_status_error2.ok());
  ASSERT_MATCH(inferred_status_error2.status().error_message(),
               testing::ContainsRegex("body must take 1 arguments"));

  auto bad_shape_3 = ShapeUtil::MakeProgramShape({result_shape}, s32_);
  auto inferred_status_error3 =
      ShapeInference::InferWhileShape(bad_shape_3, body, result_shape);
  ASSERT_FALSE(inferred_status_error3.ok());
  ASSERT_MATCH(inferred_status_error3.status().error_message(),
               testing::ContainsRegex("condition must return a boolean"));

  auto bad_shape_4 = ShapeUtil::MakeProgramShape({result_shape}, vector_32_);
  auto inferred_status_error4 =
      ShapeInference::InferWhileShape(cond, bad_shape_4, result_shape);
  ASSERT_FALSE(inferred_status_error4.ok());
  ASSERT_MATCH(inferred_status_error4.status().error_message(),
               testing::ContainsRegex("parameter of condition and body"));
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
  ASSERT_MATCH(
      inferred_status_error1.status().error_message(),
      testing::ContainsRegex("Concatenate expects at least one argument"));

  auto inferred_status_error2 =
      ShapeInference::InferConcatOpShape({&vector_32_}, /*dimension=*/-1);
  ASSERT_FALSE(inferred_status_error2.ok());
  ASSERT_MATCH(inferred_status_error2.status().error_message(),
               testing::ContainsRegex(
                   "dimension to concatenate along out of bounds: -1"));

  auto inferred_status_error3 =
      ShapeInference::InferConcatOpShape({&vector_32_}, /*dimension=*/1);
  ASSERT_FALSE(inferred_status_error3.ok());
  ASSERT_MATCH(inferred_status_error3.status().error_message(),
               testing::ContainsRegex(
                   "dimension to concatenate along out of bounds: 1"));

  Shape tuple = ShapeUtil::MakeTupleShape({vector_32_});
  auto inferred_status_error4 = ShapeInference::InferConcatOpShape(
      {&vector_32_, &tuple}, /*dimension=*/0);
  ASSERT_FALSE(inferred_status_error4.ok());
  ASSERT_MATCH(
      inferred_status_error4.status().error_message(),
      testing::ContainsRegex(
          "Expected non-tuple argument for operand of concatenation."));

  const Shape vector_s32 = ShapeUtil::MakeShape(S32, {32});
  auto inferred_status_error5 = ShapeInference::InferConcatOpShape(
      {&vector_32_, &vector_s32}, /*dimension=*/0);
  ASSERT_FALSE(inferred_status_error5.ok());
  ASSERT_MATCH(inferred_status_error5.status().error_message(),
               testing::ContainsRegex(
                   "cannot concatenate arrays with different element types"));

  auto inferred_status_error6 = ShapeInference::InferConcatOpShape(
      {&matrix_32_48_, &matrix_32_64_}, /*dimension=*/0);
  ASSERT_FALSE(inferred_status_error6.ok());
  ASSERT_MATCH(
      inferred_status_error6.status().error_message(),
      testing::ContainsRegex("cannot concatenate arrays that differ in "
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
  ASSERT_MATCH(inferred_status_error0.status().error_message(),
               testing::ContainsRegex("out-of-bounds"));

  auto inferred_status_error1 =
      ShapeInference::InferReverseShape(input_shape, {0, -1});
  ASSERT_FALSE(inferred_status_error1.ok());
  ASSERT_MATCH(inferred_status_error1.status().error_message(),
               testing::ContainsRegex("out-of-bounds"));

  auto inferred_status_error2 =
      ShapeInference::InferReverseShape(input_shape, {0, 0});
  ASSERT_FALSE(inferred_status_error2.ok());
  ASSERT_MATCH(inferred_status_error2.status().error_message(),
               testing::ContainsRegex("duplicated"));

  Shape tuple_shape = ShapeUtil::MakeTupleShape({input_shape, input_shape});
  auto inferred_status_error3 =
      ShapeInference::InferReverseShape(tuple_shape, {0});
  ASSERT_FALSE(inferred_status_error3.ok());
  ASSERT_MATCH(inferred_status_error3.status().error_message(),
               testing::ContainsRegex("Expected non-tuple argument"));
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
  EXPECT_MATCH(inferred_status_error0.status().error_message(),
               testing::ContainsRegex("arity must match"));

  auto inferred_status_error1 = ShapeInference::InferCallShape(
      {&f32_}, ShapeUtil::MakeProgramShape({}, f32_));
  EXPECT_FALSE(inferred_status_error1.ok());
  EXPECT_MATCH(inferred_status_error1.status().error_message(),
               testing::ContainsRegex("arity must match"));

  auto inferred_status_error2 = ShapeInference::InferCallShape(
      {&f32_}, ShapeUtil::MakeProgramShape({s32_}, f32_));
  EXPECT_FALSE(inferred_status_error2.ok());
  EXPECT_MATCH(inferred_status_error2.status().error_message(),
               testing::ContainsRegex("parameter must match argument"));
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

}  // namespace
}  // namespace xla
