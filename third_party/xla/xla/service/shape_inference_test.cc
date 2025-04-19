/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/shape_inference.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/padding.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::ContainsRegex;
using ::testing::HasSubstr;

constexpr absl::string_view kBroadcastDimensionMismatchErrorMessage =
    "Broadcast dimension 0 mismatch";
constexpr absl::string_view kIncompatibleBinaryOpShapeErrorMessage =
    "Binary op with incompatible shapes";
std::array<const int64_t, 1> zero_array = {0};

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
  void ExpectInferredReduceShape(
      const Shape& expected_inferred_shape, const Shape& arg,
      absl::Span<const int64_t> dimensions_to_reduce) {
    ProgramShape to_apply = ShapeUtil::MakeProgramShape({f32_, f32_}, f32_);
    const absl::StatusOr<Shape> inferred_shape =
        ShapeInference::InferReduceShape({&arg, &f32_}, dimensions_to_reduce,
                                         to_apply);
    EXPECT_IS_OK(inferred_shape.status());
    EXPECT_TRUE(ShapeUtil::Equal(expected_inferred_shape, *inferred_shape));
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

struct BinaryOpTestCase {
  std::string lhs;
  std::string rhs;
  absl::Span<const int64_t> broadcast_dimensions;
  std::string expected;
  std::optional<absl::string_view> error_message;
};

// Subclass for testing unbounded dynamic logical ops
class UnboundedLogicalOpShapeInferenceTest
    : public ::testing::TestWithParam<BinaryOpTestCase> {};

// Subclass for testing unbounded dynamic binary ops
class UnboundedBinaryOpShapeInferenceTest
    : public ::testing::TestWithParam<BinaryOpTestCase> {};

// Subclass for testing unbounded dynamic compare op
class UnboundedCompareOpShapeInferenceTest
    : public ::testing::TestWithParam<BinaryOpTestCase> {};

// Subclass for testing unbounded dynamic complex op
class UnboundedComplexOpShapeInferenceTest
    : public ::testing::TestWithParam<BinaryOpTestCase> {};

// Subclass for testing unbounded dynamic concatenate op
class UnboundedConcatenateOpShapeInferenceTest
    : public ::testing::TestWithParam<std::vector<std::string>> {};

struct UnaryOpTestCase {
  std::string operand;
  std::string expected;
  HloOpcode opcode;
};

// Subclass for testing unbounded dynamic unary ops
class UnboundedUnaryOpShapeInferenceTest
    : public ::testing::TestWithParam<UnaryOpTestCase> {};

// Subclass for testing unbounded dynamic clamp op
class UnboundedClampOpShapeInferenceTest
    : public ::testing::TestWithParam<std::vector<std::string>> {};

// Subclass for testing unbounded dynamic select op
class UnboundedSelectOpShapeInferenceTest
    : public ::testing::TestWithParam<std::vector<std::string>> {};

TEST_F(ShapeInferenceTest, UnaryNegateMatrix) {
  const Shape matrix_shape = ShapeUtil::MakeShape(F32, {128, 64});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferUnaryOpShape(HloOpcode::kNegate, matrix_shape);
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_shape, *inferred_shape));
}

TEST_F(ShapeInferenceTest, SelectScalarPredBetweenTuples) {
  const Shape tuple = ShapeUtil::MakeTupleShape({s32_, f32_});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kSelect, pred_, tuple,
                                          tuple);
  ASSERT_FALSE(inferred_shape.ok());
  ASSERT_THAT(inferred_shape.status().message(),
              HasSubstr("Expected array argument for select"));
}

TEST_F(ShapeInferenceTest, SelectScalarPredBetweenArrays) {
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kSelect, pred_,
                                          matrix_64_48_, matrix_64_48_);
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, *inferred_shape));
}

TEST_F(ShapeInferenceTest, SelectArrayPredBetweenArrays) {
  const Shape predarray = ShapeUtil::MakeShape(PRED, {64, 48});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kSelect, predarray,
                                          matrix_64_48_, matrix_64_48_);
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, *inferred_shape));
}

TEST_F(ShapeInferenceTest, SelectBadShapes) {
  const absl::StatusOr<Shape> inferred_shape_error1 =
      ShapeInference::InferTernaryOpShape(HloOpcode::kSelect, pred_,
                                          matrix_64_48_, matrix_32_64_);
  ASSERT_FALSE(inferred_shape_error1.ok());
  ASSERT_THAT(inferred_shape_error1.status().message(),
              HasSubstr("Operands to select must be the same shape"));

  const absl::StatusOr<Shape> inferred_shape_error2 =
      ShapeInference::InferTernaryOpShape(HloOpcode::kSelect, s32_,
                                          matrix_64_48_, matrix_64_48_);
  ASSERT_FALSE(inferred_shape_error2.ok());
  ASSERT_THAT(inferred_shape_error2.status().message(),
              HasSubstr("pred operand must have PRED"));

  const absl::StatusOr<Shape> inferred_shape_error3 =
      ShapeInference::InferTernaryOpShape(HloOpcode::kSelect,
                                          ShapeUtil::MakeShape(PRED, {64}),
                                          matrix_64_48_, matrix_64_48_);
  ASSERT_FALSE(inferred_shape_error3.ok());
  ASSERT_THAT(
      inferred_shape_error3.status().message(),
      HasSubstr("Operands to select and predicate must be the same shape"));

  // Tuples have a TUPLE element type and cannot be the pred of a select.
  const absl::StatusOr<Shape> inferred_shape_error4 =
      ShapeInference::InferTernaryOpShape(
          HloOpcode::kSelect, ShapeUtil::MakeTupleShape({pred_, pred_}),
          ShapeUtil::MakeTupleShape({f32_, f32_}),
          ShapeUtil::MakeTupleShape({f32_, f32_}));
  ASSERT_FALSE(inferred_shape_error4.ok());
  ASSERT_THAT(inferred_shape_error4.status().message(),
              HasSubstr("Expected array argument for select pred"));
}

TEST_F(ShapeInferenceTest, SelectPreservesElementSize) {
  Shape pred_shape = ShapeUtil::MakeShape(PRED, {10});
  Shape int4_shape = ShapeUtil::MakeShape(S4, {10});
  int4_shape.mutable_layout()->set_element_size_in_bits(4);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kSelect, pred_shape,
                                          int4_shape, int4_shape);
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape, int4_shape));
}

TEST_F(ShapeInferenceTest, ClampAllMatrix) {
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, matrix_64_48_,
                                          matrix_64_48_, matrix_64_48_);
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, *inferred_shape));
}

TEST_F(ShapeInferenceTest, ClampAllScalar) {
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, f32_, f32_, f32_);
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(ShapeUtil::Equal(f32_, *inferred_shape));
}

TEST_F(ShapeInferenceTest, ClampMinScalar) {
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, f32_,
                                          matrix_64_48_, matrix_64_48_);
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, *inferred_shape));
}

TEST_F(ShapeInferenceTest, ClampMaxScalar) {
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, matrix_64_48_,
                                          matrix_64_48_, f32_);
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, *inferred_shape));
}

TEST_F(ShapeInferenceTest, ClampOperandScalar) {
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, matrix_64_48_,
                                          f32_, matrix_64_48_);
  ASSERT_FALSE(inferred_shape.ok());
  ASSERT_THAT(inferred_shape.status().message(),
              HasSubstr("Clamp with incompatible shapes"));
}

TEST_F(ShapeInferenceTest, ClampMinMatrix) {
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, matrix_64_48_,
                                          f32_, f32_);
  ASSERT_FALSE(inferred_shape.ok());
  ASSERT_THAT(inferred_shape.status().message(),
              HasSubstr("Clamp with incompatible shapes"));
}

TEST_F(ShapeInferenceTest, ClampMaxMatrix) {
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, f32_, f32_,
                                          matrix_64_48_);
  ASSERT_FALSE(inferred_shape.ok());
  ASSERT_THAT(inferred_shape.status().message(),
              HasSubstr("Clamp with incompatible shapes"));
}

TEST_F(ShapeInferenceTest, ClampOperandMatrix) {
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, f32_,
                                          matrix_64_48_, f32_);
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, *inferred_shape));
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

TEST_F(ShapeInferenceTest, Atan2FailsWithIntegerInput) {
  const Shape input = ShapeUtil::MakeScalarShape(S8);
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAtan2, input, input, {});
  EXPECT_THAT(
      inferred_shape.status(),
      tsl::testing::StatusIs(tsl::error::INVALID_ARGUMENT,
                             HasSubstr("Expected input element type to be "
                                       "floating or complex for atan2")));
}

TEST_F(ShapeInferenceTest, Complex) {
  const auto complex_shape = [&](const Shape& lhs, const Shape& rhs,
                                 absl::Span<const int64_t> bcast) {
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
  const Shape c64_32 = ShapeUtil::MakeShape(C64, {32});
  TF_ASSERT_OK_AND_ASSIGN(Shape result, complex_shape(f32_, f32_, {}));
  ASSERT_TRUE(ShapeUtil::Equal(result, ShapeUtil::MakeShape(C64, {})));
  TF_ASSERT_OK_AND_ASSIGN(result, complex_shape(vector_32_, f32_, {}));
  ASSERT_TRUE(ShapeUtil::Equal(result, c64_32));
  TF_ASSERT_OK_AND_ASSIGN(result, complex_shape(f32_, vector_32_, {}));
  ASSERT_TRUE(ShapeUtil::Equal(result, c64_32));
  TF_ASSERT_OK_AND_ASSIGN(result, complex_shape(vector_32_, f32_, {}));
  ASSERT_TRUE(ShapeUtil::Equal(result, c64_32));

  const Shape c64_32_64 = ShapeUtil::MakeShape(C64, {32, 64});
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

TEST_F(ShapeInferenceTest, ComplexCbrtIsNotSupported) {
  const Shape input = ShapeUtil::MakeScalarShape(C64);
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferUnaryOpShape(HloOpcode::kCbrt, input);
  EXPECT_THAT(
      inferred_shape.status(),
      tsl::testing::StatusIs(tsl::error::INVALID_ARGUMENT,
                             HasSubstr("Expected element type in shape to be "
                                       "floating for cbrt operation")));
}

TEST_F(ShapeInferenceTest, VariadicOpTuplify) {
  const absl::StatusOr<Shape> result =
      ShapeInference::InferVariadicOpShape(HloOpcode::kTuple, {&s32_, &f32_});
  ASSERT_IS_OK(result.status());
  ASSERT_TRUE(
      ShapeUtil::Equal(*result, ShapeUtil::MakeTupleShape({s32_, f32_})));
}

TEST_F(ShapeInferenceTest, ReduceWindowInHalf) {
  const Shape matrix_shape = ShapeUtil::MakeShape(F32, {8, 8});
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
  const Shape window_shape = ShapeUtil::MakeShape(F32, {2, 2});
  const Shape init_value_shape = ShapeUtil::MakeShape(F32, {});
  const Shape float_scalar = ShapeUtil::MakeShape(F32, {});
  ProgramShape to_apply = ShapeUtil::MakeProgramShape(
      {ShapeUtil::MakeShape(F32, {}), ShapeUtil::MakeShape(F32, {})}, f32_);
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferReduceWindowShape(matrix_shape, init_value_shape,
                                             window, to_apply);

  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {4, 4}), *inferred_shape));
}

TEST_F(SelectAndScatterShapeInferenceTest, SelectAndScatterProperShapes) {
  const absl::StatusOr<Shape> inferred_shape_ok =
      ShapeInference::InferSelectAndScatterShape(
          operand_shape_, select_program_shape_, window_, source_shape_,
          init_value_shape_, scatter_program_shape_);
  ASSERT_IS_OK(inferred_shape_ok.status());
  ASSERT_TRUE(ShapeUtil::Equal(operand_shape_, *inferred_shape_ok));
}

TEST_F(SelectAndScatterShapeInferenceTest, SelectAndScatterWrongSourceShape) {
  const Shape source_shape_fail = ShapeUtil::MakeShape(F32, {4, 6});
  const absl::StatusOr<Shape> inferred_shape_fail =
      ShapeInference::InferSelectAndScatterShape(
          operand_shape_, select_program_shape_, window_, source_shape_fail,
          init_value_shape_, scatter_program_shape_);
  ASSERT_FALSE(inferred_shape_fail.ok());
  ASSERT_THAT(inferred_shape_fail.status().message(),
              HasSubstr("Source shape does not match"));
}

TEST_F(SelectAndScatterShapeInferenceTest, SelectAndScatterWrongSelectShape1) {
  ProgramShape select_program_shape_fail =
      ShapeUtil::MakeProgramShape({ShapeUtil::MakeShape(F32, {})}, pred_);
  const absl::StatusOr<Shape> inferred_shape_fail =
      ShapeInference::InferSelectAndScatterShape(
          operand_shape_, select_program_shape_fail, window_, source_shape_,
          init_value_shape_, scatter_program_shape_);
  ASSERT_FALSE(inferred_shape_fail.ok());
  ASSERT_THAT(inferred_shape_fail.status().message(),
              HasSubstr("Select function must take 2 parameters"));
}

TEST_F(SelectAndScatterShapeInferenceTest, SelectAndScatterWrongSelectShape2) {
  ProgramShape select_program_shape_fail = ShapeUtil::MakeProgramShape(
      {ShapeUtil::MakeShape(F32, {}), ShapeUtil::MakeShape(F32, {})}, f32_);
  const absl::StatusOr<Shape> inferred_shape_fail =
      ShapeInference::InferSelectAndScatterShape(
          operand_shape_, select_program_shape_fail, window_, source_shape_,
          init_value_shape_, scatter_program_shape_);
  ASSERT_FALSE(inferred_shape_fail.ok());
  ASSERT_THAT(inferred_shape_fail.status().message(),
              HasSubstr("Select function must have rank-0 PRED"));
}

TEST_F(SelectAndScatterShapeInferenceTest, SelectAndScatterWrongSelectShape3) {
  ProgramShape select_program_shape_fail = ShapeUtil::MakeProgramShape(
      {ShapeUtil::MakeShape(S32, {}), ShapeUtil::MakeShape(F32, {})}, pred_);
  const absl::StatusOr<Shape> inferred_shape_fail =
      ShapeInference::InferSelectAndScatterShape(
          operand_shape_, select_program_shape_fail, window_, source_shape_,
          init_value_shape_, scatter_program_shape_);
  ASSERT_FALSE(inferred_shape_fail.ok());
  ASSERT_THAT(inferred_shape_fail.status().message(),
              HasSubstr("Select function's first parameter"));
}

TEST_F(SelectAndScatterShapeInferenceTest, SelectAndScatterWrongSelectShape4) {
  ProgramShape select_program_shape_fail = ShapeUtil::MakeProgramShape(
      {ShapeUtil::MakeShape(F32, {}), ShapeUtil::MakeShape(U32, {})}, pred_);
  const absl::StatusOr<Shape> inferred_shape_fail =
      ShapeInference::InferSelectAndScatterShape(
          operand_shape_, select_program_shape_fail, window_, source_shape_,
          init_value_shape_, scatter_program_shape_);
  ASSERT_FALSE(inferred_shape_fail.ok());
  ASSERT_THAT(inferred_shape_fail.status().message(),
              HasSubstr("Select function's second parameter"));
}

TEST_F(ShapeInferenceTest, AllGatherStart) {
  const Shape operand = ShapeUtil::MakeShape(F32, {1, 8, 4});
  const Shape expected_shape = ShapeUtil::MakeTupleShape(
      {operand, ShapeUtil::MakeShape(F32, {8, 8, 4})});

  const absl::StatusOr<Shape> inferred_ag_shape =
      ShapeInference::InferAllGatherStartShape(
          {&operand}, /*all_gather_dimension=*/0, /*shard_count=*/8);
  EXPECT_TRUE(inferred_ag_shape.ok());
  EXPECT_TRUE(ShapeUtil::Equal(*inferred_ag_shape, expected_shape));
}

TEST_F(ShapeInferenceTest, AllGatherStartMultiOperand) {
  const Shape operand0 = ShapeUtil::MakeShape(F32, {1, 8, 4});
  const Shape operand1 = ShapeUtil::MakeShape(BF16, {1, 5});
  const Shape expected_output0_shape = ShapeUtil::MakeShape(F32, {8, 8, 4});
  const Shape expected_output1_shape = ShapeUtil::MakeShape(BF16, {8, 5});
  const Shape expected_shape = ShapeUtil::MakeTupleShape(
      {/* tuple of all input shapes*/
       ShapeUtil::MakeTupleShape({operand0, operand1}),
       /* tuple of all output shapes*/
       ShapeUtil::MakeTupleShape(
           {expected_output0_shape, expected_output1_shape})});

  const absl::StatusOr<Shape> inferred_ag_shape =
      ShapeInference::InferAllGatherStartShape({&operand0, &operand1},
                                               /*all_gather_dimension=*/0,
                                               /*shard_count=*/8);
  EXPECT_TRUE(inferred_ag_shape.ok());
  EXPECT_TRUE(ShapeUtil::Equal(*inferred_ag_shape, expected_shape));
}

TEST_F(ShapeInferenceTest, AllGatherDone) {
  const Shape input_shape =
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {1, 8, 4}),
                                 ShapeUtil::MakeShape(F32, {8, 8, 4})});
  const Shape expected_shape = ShapeUtil::MakeShape(F32, {8, 8, 4});

  const absl::StatusOr<Shape> inferred_ag_done_shape =
      ShapeInference::InferAllGatherDoneShape(input_shape);
  EXPECT_TRUE(inferred_ag_done_shape.ok());
  EXPECT_TRUE(ShapeUtil::Equal(*inferred_ag_done_shape, expected_shape));
}

TEST_F(ShapeInferenceTest, AllGatherDoneMultiOperand) {
  const Shape operand0 = ShapeUtil::MakeShape(F32, {1, 8, 4});
  const Shape operand1 = ShapeUtil::MakeShape(BF16, {1, 5});
  const Shape expected_output0_shape = ShapeUtil::MakeShape(F32, {8, 8, 4});
  const Shape expected_output1_shape = ShapeUtil::MakeShape(BF16, {8, 5});
  const Shape input_shape = ShapeUtil::MakeTupleShape(
      {/* tuple of all input shapes*/
       ShapeUtil::MakeTupleShape({operand0, operand1}),
       /* tuple of all output shapes*/
       ShapeUtil::MakeTupleShape(
           {expected_output0_shape, expected_output1_shape})});

  const Shape expected_shape = ShapeUtil::MakeTupleShape(
      {expected_output0_shape, expected_output1_shape});

  const absl::StatusOr<Shape> inferred_ag_done_shape =
      ShapeInference::InferAllGatherDoneShape(input_shape);
  EXPECT_TRUE(inferred_ag_done_shape.ok());
  EXPECT_TRUE(ShapeUtil::Equal(*inferred_ag_done_shape, expected_shape));
}

TEST_F(ShapeInferenceTest, Convolve) {
  ConvolutionDimensionNumbers dnums;

  // Dimension order: batch, feature, x0, x1
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {10, 11, 3, 4});
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(0);
  dnums.set_input_feature_dimension(1);
  dnums.set_output_feature_dimension(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);
  dnums.add_input_spatial_dimensions(3);
  dnums.add_output_spatial_dimensions(3);

  // Dimension order: x1, batch, feature, x0
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {2, 12, 11, 3});
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(1);
  dnums.add_kernel_spatial_dimensions(3);
  dnums.add_kernel_spatial_dimensions(0);

  Window window;
  const auto dim0 = window.add_dimensions();
  const auto dim1 = window.add_dimensions();
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
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferConvolveShape(
          lhs_shape, rhs_shape, /*feature_group_count=*/1,
          /*batch_group_count=*/1, window, dnums,
          /*preferred_element_type=*/std::nullopt);
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {10, 12, 2, 3}),
                               *inferred_shape));
}

TEST_F(ShapeInferenceTest, ConvolveWithWindowDilation) {
  ConvolutionDimensionNumbers dnums;

  // Dimension order: batch, feature, x0, x1
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {10, 11, 103, 4});
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(0);
  dnums.set_input_feature_dimension(1);
  dnums.set_output_feature_dimension(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);
  dnums.add_input_spatial_dimensions(3);
  dnums.add_output_spatial_dimensions(3);

  // Dimension order: x1, batch, feature, x0
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {2, 12, 11, 3});
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(1);
  dnums.add_kernel_spatial_dimensions(3);
  dnums.add_kernel_spatial_dimensions(0);

  Window window;
  const auto dim0 = window.add_dimensions();
  dim0->set_size(3);
  dim0->set_stride(3);
  dim0->set_padding_low(0);
  dim0->set_padding_high(0);
  dim0->set_window_dilation(6);
  dim0->set_base_dilation(1);

  const auto dim1 = window.add_dimensions();
  dim1->set_size(2);
  dim1->set_stride(1);
  dim1->set_padding_low(2);
  dim1->set_padding_high(1);
  dim1->set_window_dilation(2);
  dim1->set_base_dilation(1);
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferConvolveShape(
          lhs_shape, rhs_shape, /*feature_group_count=*/1,
          /*batch_group_count=*/1, window, dnums,
          /*preferred_element_type=*/std::nullopt);
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {10, 12, 31, 5}),
                               *inferred_shape));
}

TEST_F(ShapeInferenceTest, ConvolveWithBaseDilation) {
  ConvolutionDimensionNumbers dnums;

  // Dimension order: batch, feature, x0, x1
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {10, 11, 3, 4});
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(0);
  dnums.set_input_feature_dimension(1);
  dnums.set_output_feature_dimension(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);
  dnums.add_input_spatial_dimensions(3);
  dnums.add_output_spatial_dimensions(3);

  // Dimension order: x1, batch, feature, x0
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {2, 12, 11, 4});
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(1);
  dnums.add_kernel_spatial_dimensions(3);
  dnums.add_kernel_spatial_dimensions(0);

  Window window;
  const auto dim0 = window.add_dimensions();
  dim0->set_size(4);
  dim0->set_stride(3);
  dim0->set_padding_low(0);
  dim0->set_padding_high(0);
  dim0->set_window_dilation(1);
  dim0->set_base_dilation(6);

  const auto dim1 = window.add_dimensions();
  dim1->set_size(2);
  dim1->set_stride(1);
  dim1->set_padding_low(2);
  dim1->set_padding_high(1);
  dim1->set_window_dilation(1);
  dim1->set_base_dilation(2);
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferConvolveShape(
          lhs_shape, rhs_shape, /*feature_group_count=*/1,
          /*batch_group_count=*/1, window, dnums,
          /*preferred_element_type=*/std::nullopt);
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {10, 12, 4, 9}),
                               *inferred_shape));
}

TEST_F(ShapeInferenceTest, ConvolveDimensionNumbersOverlapError) {
  // Dimension order for this test: batch, feature, x0, x1
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {10, 11, 3, 4});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {12, 11, 3, 2});

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
  const auto dim0 = window.add_dimensions();
  const auto dim1 = window.add_dimensions();
  dim0->set_size(2);
  dim0->set_stride(1);
  dim0->set_padding_low(0);
  dim0->set_padding_high(0);
  dim1->set_size(3);
  dim1->set_stride(2);
  dim1->set_padding_low(1);
  dim1->set_padding_high(1);
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferConvolveShape(
          lhs_shape, rhs_shape, /*feature_group_count=*/1,
          /*batch_group_count=*/1, window, dnums,
          /*preferred_element_type=*/std::nullopt);
  ASSERT_FALSE(inferred_shape.ok());
  ASSERT_THAT(inferred_shape.status().message(),
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
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {60, 38, 17, 13});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {38, 10, 4, 4});
  Window window;
  const auto dim0 = window.add_dimensions();
  const auto dim1 = window.add_dimensions();
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
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferConvolveShape(
          lhs_shape, rhs_shape, /*feature_group_count=*/1,
          /*batch_group_count=*/6, window, dnums,
          /*preferred_element_type=*/std::nullopt);
  ASSERT_FALSE(inferred_shape.ok());
  ASSERT_THAT(inferred_shape.status().message(),
              HasSubstr("to be a multiple of batch group count"));
}

struct ConvolveArgs {
  Shape lhs_shape;
  Shape rhs_shape;
  ConvolutionDimensionNumbers dnums;
  Window window;
};

ConvolveArgs MakeConvolveArgs(PrimitiveType lhs_type, PrimitiveType rhs_type) {
  ConvolveArgs args;
  ConvolutionDimensionNumbers& dnums = args.dnums;

  // Dimension order: batch, feature, x0, x1
  args.lhs_shape = ShapeUtil::MakeShape(lhs_type, {10, 11, 3, 4});
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(0);
  dnums.set_input_feature_dimension(1);
  dnums.set_output_feature_dimension(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);
  dnums.add_input_spatial_dimensions(3);
  dnums.add_output_spatial_dimensions(3);

  // Dimension order: x1, batch, feature, x0
  args.rhs_shape = ShapeUtil::MakeShape(rhs_type, {2, 12, 11, 3});
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(1);
  dnums.add_kernel_spatial_dimensions(3);
  dnums.add_kernel_spatial_dimensions(0);

  auto dim0 = args.window.add_dimensions();
  auto dim1 = args.window.add_dimensions();
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
  return args;
}

TEST_F(ShapeInferenceTest, ConvolveWithBF16_F16) {
  ConvolveArgs args = MakeConvolveArgs(BF16, F16);
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferConvolveShape(
          args.lhs_shape, args.rhs_shape, /*feature_group_count=*/1,
          /*batch_group_count=*/1, args.window, args.dnums,
          /*preferred_element_type=*/std::nullopt))
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(BF16, {10, 12, 2, 3}),
                               inferred_shape));
}

TEST_F(ShapeInferenceTest, ConvolveWithF16_BF16) {
  ConvolveArgs args = MakeConvolveArgs(F16, BF16);
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferConvolveShape(
          args.lhs_shape, args.rhs_shape, /*feature_group_count=*/1,
          /*batch_group_count=*/1, args.window, args.dnums,
          /*preferred_element_type=*/std::nullopt))
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(BF16, {10, 12, 2, 3}),
                               inferred_shape));
}

TEST_F(ShapeInferenceTest, ConvolveWithS32_U32) {
  ConvolveArgs args = MakeConvolveArgs(S32, U32);
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferConvolveShape(
          args.lhs_shape, args.rhs_shape, /*feature_group_count=*/1,
          /*batch_group_count=*/1, args.window, args.dnums,
          /*preferred_element_type=*/std::nullopt))
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(S32, {10, 12, 2, 3}),
                               inferred_shape));
}

TEST_F(ShapeInferenceTest, ConvolveWithU32_S32) {
  ConvolveArgs args = MakeConvolveArgs(U32, S32);
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferConvolveShape(
          args.lhs_shape, args.rhs_shape, /*feature_group_count=*/1,
          /*batch_group_count=*/1, args.window, args.dnums,
          /*preferred_element_type=*/std::nullopt))
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(S32, {10, 12, 2, 3}),
                               inferred_shape));
}

TEST_F(ShapeInferenceTest, ConvolveWithPreferredElementType) {
  ConvolveArgs args = MakeConvolveArgs(S8, S16);
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferConvolveShape(
          args.lhs_shape, args.rhs_shape, /*feature_group_count=*/1,
          /*batch_group_count=*/1, args.window, args.dnums,
          /*preferred_element_type=*/S16))
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(S16, {10, 12, 2, 3}),
                               inferred_shape));
}

TEST_F(ShapeInferenceTest, ConvolveWithPreferredElementTypeSameAsInferredType) {
  ConvolveArgs args = MakeConvolveArgs(S8, S16);
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferConvolveShape(
          args.lhs_shape, args.rhs_shape, /*feature_group_count=*/1,
          /*batch_group_count=*/1, args.window, args.dnums,
          /*preferred_element_type=*/S32))
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(S32, {10, 12, 2, 3}),
                               inferred_shape));
}

TEST_F(ShapeInferenceTest,
       FloatingPointConvolveWithNarrowerPreferredElementType) {
  ConvolveArgs args = MakeConvolveArgs(F32, F32);
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferConvolveShape(
          args.lhs_shape, args.rhs_shape, /*feature_group_count=*/1,
          /*batch_group_count=*/1, args.window, args.dnums,
          /*preferred_element_type=*/BF16))
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(BF16, {10, 12, 2, 3}),
                               inferred_shape));
}

TEST_F(ShapeInferenceTest,
       FloatingPointConvolveWithIntegralPreferredElementType) {
  ConvolveArgs args = MakeConvolveArgs(BF16, BF16);
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferConvolveShape(
          args.lhs_shape, args.rhs_shape, /*feature_group_count=*/1,
          /*batch_group_count=*/1, args.window, args.dnums,
          /*preferred_element_type=*/S32));
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(S32, {10, 12, 2, 3}),
                               inferred_shape));
}

TEST_F(ShapeInferenceTest,
       IntegralConvolveWithFloatingPointPreferredElementType) {
  ConvolveArgs args = MakeConvolveArgs(S8, S16);
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferConvolveShape(
          args.lhs_shape, args.rhs_shape, /*feature_group_count=*/1,
          /*batch_group_count=*/1, args.window, args.dnums,
          /*preferred_element_type=*/F32));
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {10, 12, 2, 3}),
                               inferred_shape));
}

TEST_F(ShapeInferenceTest,
       ConvolveWithPreferredElementTypeWithDifferentSignedness) {
  ConvolveArgs args = MakeConvolveArgs(S8, S16);
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferConvolveShape(
          args.lhs_shape, args.rhs_shape, /*feature_group_count=*/1,
          /*batch_group_count=*/1, args.window, args.dnums,
          /*preferred_element_type=*/U32));
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(U32, {10, 12, 2, 3}),
                               inferred_shape));
}

TEST_F(ShapeInferenceTest, ConvolveWithNarrowerPreferredElementType) {
  ConvolveArgs args = MakeConvolveArgs(S8, S16);
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferConvolveShape(
          args.lhs_shape, args.rhs_shape, /*feature_group_count=*/1,
          /*batch_group_count=*/1, args.window, args.dnums,
          /*preferred_element_type=*/S8));
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(S8, {10, 12, 2, 3}),
                               inferred_shape));
}

namespace fft {

static const char* unsupported_rank = "only supports ranks 1-3";
static const char* invalid_rank = "requires input of at least same rank";
static const char* requires_complex_input = "requires complex input type";
static const char* requires_f32_input = "requires F32 or F64 input type";
static const char* dimensions_match = "innermost dimensions match fft_length";
static const char* innermost_dimension_matches =
    "innermost dimension matches fft_length/2+1";

static void Pass(const Shape& shape, FftType type,
                 absl::Span<const int64_t> length,
                 const Shape& expected_shape) {
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferFftShape(shape, type, length);
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(ShapeUtil::Equal(expected_shape, *inferred_shape))
      << "\nshape: " << shape << "\ntype: " << type
      << "\nlength: " << absl::StrJoin(length, ",")
      << "\nexpected_shape: " << expected_shape
      << "\ninferred_shape: " << *inferred_shape;
}

static void Fail(const Shape& shape, FftType type,
                 absl::Span<const int64_t> length, absl::string_view message) {
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferFftShape(shape, type, length);
  ASSERT_FALSE(inferred_shape.ok());
  ASSERT_THAT(inferred_shape.status().message(),
              HasSubstr(std::string(message)));
}

}  // namespace fft

TEST_F(ShapeInferenceTest, InferFftShapeTestFftRanks) {
  FftType type = FftType::FFT;
  const Shape shape = ShapeUtil::MakeShape(C64, {16, 8});
  fft::Fail(shape, type, {}, fft::unsupported_rank);
  fft::Pass(shape, type, {8}, shape);
  fft::Pass(shape, type, {16, 8}, shape);
  fft::Fail(shape, type, {32, 16, 8}, fft::invalid_rank);
  fft::Fail(shape, type, {64, 32, 16, 8}, fft::unsupported_rank);
}

TEST_F(ShapeInferenceTest, InferFftShapeTestFftRanksBounded) {
  FftType type = FftType::FFT;
  const Shape shape = ShapeUtil::MakeShape(C64, {16, 8}, {false, true});
  fft::Fail(shape, type, {}, fft::unsupported_rank);
  fft::Pass(shape, type, {8}, shape);
  fft::Pass(shape, type, {16, 8}, shape);
  fft::Fail(shape, type, {32, 16, 8}, fft::invalid_rank);
  fft::Fail(shape, type, {64, 32, 16, 8}, fft::unsupported_rank);
}

TEST_F(ShapeInferenceTest, InferFftShapeTestFftTypes) {
  FftType type = FftType::FFT;
  const Shape shape_f32 = ShapeUtil::MakeShape(F32, {16, 8});
  const Shape shape_c128 = ShapeUtil::MakeShape(C128, {16, 8});
  fft::Fail(shape_f32, type, {16, 8}, fft::requires_complex_input);
  fft::Pass(shape_c128, type, {16, 8}, shape_c128);
}

TEST_F(ShapeInferenceTest, InferFftShapeTestIfftRanks) {
  FftType type = FftType::IFFT;
  const Shape shape = ShapeUtil::MakeShape(C64, {16, 8});
  fft::Fail(shape, type, {}, fft::unsupported_rank);
  fft::Pass(shape, type, {8}, shape);
  fft::Pass(shape, type, {16, 8}, shape);
  fft::Fail(shape, type, {32, 16, 8}, fft::invalid_rank);
  fft::Fail(shape, type, {64, 32, 16, 8}, fft::unsupported_rank);
}

TEST_F(ShapeInferenceTest, InferFftShapeTestIfftRanksBounded) {
  FftType type = FftType::IFFT;
  const Shape shape = ShapeUtil::MakeShape(C64, {16, 8}, {false, true});
  fft::Fail(shape, type, {}, fft::unsupported_rank);
  fft::Pass(shape, type, {8}, shape);
  fft::Pass(shape, type, {16, 8}, shape);
  fft::Fail(shape, type, {32, 16, 8}, fft::invalid_rank);
  fft::Fail(shape, type, {64, 32, 16, 8}, fft::unsupported_rank);
}

TEST_F(ShapeInferenceTest, InferFftShapeTestIfftTypes) {
  FftType type = FftType::IFFT;
  const Shape shape_f32 = ShapeUtil::MakeShape(F32, {16, 8});
  const Shape shape_c128 = ShapeUtil::MakeShape(C128, {16, 8});
  fft::Fail(shape_f32, type, {16, 8}, fft::requires_complex_input);
  fft::Pass(shape_c128, type, {16, 8}, shape_c128);
}

TEST_F(ShapeInferenceTest, InferFftShapeTestRfftRanks) {
  FftType type = FftType::RFFT;
  const Shape shape_in = ShapeUtil::MakeShape(F32, {16, 8});
  const Shape shape_out = ShapeUtil::MakeShape(C64, {16, 5});
  fft::Fail(shape_in, type, {}, fft::unsupported_rank);
  fft::Pass(shape_in, type, {8}, shape_out);
  fft::Pass(shape_in, type, {16, 8}, shape_out);
  fft::Fail(shape_in, type, {32, 16, 8}, fft::invalid_rank);
  fft::Fail(shape_in, type, {64, 32, 16, 8}, fft::unsupported_rank);
}

TEST_F(ShapeInferenceTest, InferFftShapeTestRfftDimensions) {
  FftType type = FftType::RFFT;
  const Shape shape = ShapeUtil::MakeShape(F32, {16, 8});
  fft::Fail(shape, type, {4}, fft::dimensions_match);
  fft::Fail(shape, type, {16, 4}, fft::dimensions_match);
  fft::Fail(shape, type, {8, 8}, fft::dimensions_match);
  fft::Fail(shape, type, {8, 16}, fft::dimensions_match);

  const Shape zero_shape_in = ShapeUtil::MakeShape(F32, {16, 0});
  const Shape zero_shape_out = ShapeUtil::MakeShape(C64, {16, 0});
  fft::Pass(zero_shape_in, type, {0}, zero_shape_out);
  fft::Pass(zero_shape_in, type, {16, 0}, zero_shape_out);

  const Shape even_shape_in = ShapeUtil::MakeShape(F32, {16, 8});
  const Shape odd_shape_in = ShapeUtil::MakeShape(F32, {16, 9});
  const Shape shape_out = ShapeUtil::MakeShape(C64, {16, 5});
  fft::Pass(even_shape_in, type, {16, 8}, shape_out);
  fft::Pass(odd_shape_in, type, {16, 9}, shape_out);

  const Shape bounded_shape_in =
      ShapeUtil::MakeShape(F32, {16, 8}, {false, true});
  const Shape bounded_shape_out =
      ShapeUtil::MakeShape(C64, {16, 5}, {false, true});
  fft::Pass(bounded_shape_in, type, {16, 8}, bounded_shape_out);
}

TEST_F(ShapeInferenceTest, InferFftShapeTestRfftTypes) {
  FftType type = FftType::RFFT;
  const Shape shape_c64 = ShapeUtil::MakeShape(C64, {16, 8});
  const Shape shape_c128 = ShapeUtil::MakeShape(C128, {16, 8});
  fft::Fail(shape_c64, type, {16, 8}, fft::requires_f32_input);
  fft::Fail(shape_c128, type, {16, 8}, fft::requires_f32_input);
}

TEST_F(ShapeInferenceTest, InferFftShapeTestIrfftRanks) {
  FftType type = FftType::IRFFT;
  const Shape shape_in = ShapeUtil::MakeShape(C64, {16, 5});
  const Shape shape_out = ShapeUtil::MakeShape(F32, {16, 8});
  fft::Fail(shape_in, type, {}, fft::unsupported_rank);
  fft::Pass(shape_in, type, {8}, shape_out);
  fft::Pass(shape_in, type, {16, 8}, shape_out);
  fft::Fail(shape_in, type, {32, 16, 8}, fft::invalid_rank);
  fft::Fail(shape_in, type, {64, 32, 16, 8}, fft::unsupported_rank);
}

TEST_F(ShapeInferenceTest, InferFftShapeTestIrfftDimensions) {
  FftType type = FftType::IRFFT;
  const Shape shape = ShapeUtil::MakeShape(C64, {16, 5});
  fft::Fail(shape, type, {5}, fft::innermost_dimension_matches);
  fft::Fail(shape, type, {16, 5}, fft::innermost_dimension_matches);
  fft::Fail(shape, type, {8, 8}, fft::dimensions_match);
  fft::Fail(shape, type, {8, 9}, fft::dimensions_match);

  const Shape zero_shape_in = ShapeUtil::MakeShape(C64, {16, 0});
  const Shape zero_shape_out = ShapeUtil::MakeShape(F32, {16, 0});
  fft::Pass(zero_shape_in, type, {0}, zero_shape_out);
  fft::Pass(zero_shape_in, type, {16, 0}, zero_shape_out);

  const Shape even_shape_out = ShapeUtil::MakeShape(F32, {16, 8});
  const Shape odd_shape_out = ShapeUtil::MakeShape(F32, {16, 9});
  fft::Pass(shape, type, {16, 8}, even_shape_out);
  fft::Pass(shape, type, {16, 9}, odd_shape_out);

  const Shape bounded_shape_in =
      ShapeUtil::MakeShape(C64, {16, 5}, {false, true});
  const Shape bounded_shape_out =
      ShapeUtil::MakeShape(F32, {16, 9}, {false, true});
  fft::Pass(bounded_shape_in, type, {16, 9}, bounded_shape_out);
}

TEST_F(ShapeInferenceTest, InferFftShapeTestIrfftTypes) {
  FftType type = FftType::IRFFT;
  const Shape shape_f32 = ShapeUtil::MakeShape(F32, {16, 8});
  const Shape shape_c128 = ShapeUtil::MakeShape(C128, {16, 5});
  const Shape shape_f64_out = ShapeUtil::MakeShape(F64, {16, 8});
  fft::Fail(shape_f32, type, {16, 8}, fft::requires_complex_input);
  fft::Pass(shape_c128, type, {16, 8}, shape_f64_out);
}

TEST_F(ShapeInferenceTest, MapThatChangesElementType) {
  const Shape arg = ShapeUtil::MakeShape(F32, {20});
  ProgramShape to_apply = ShapeUtil::MakeProgramShape({f32_}, s32_);
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferMapShape({&arg}, to_apply, {0});
  EXPECT_IS_OK(inferred_shape.status());
  const Shape expected = ShapeUtil::MakeShape(S32, {20});
  EXPECT_TRUE(ShapeUtil::Equal(expected, *inferred_shape));
}

TEST_F(ShapeInferenceTest, Map) {
  const absl::StatusOr<Shape> inferred_shape_r1f32 =
      ShapeInference::InferMapShape(
          {&vector_32_, &vector_32_},
          ShapeUtil::MakeProgramShape({f32_, f32_}, f32_), {0});
  EXPECT_IS_OK(inferred_shape_r1f32.status());
  EXPECT_TRUE(ShapeUtil::Equal(vector_32_, *inferred_shape_r1f32));

  // It's OK to provide a single argument, as long as the applied arity matches
  // (this degenerates to a Map).
  const absl::StatusOr<Shape> inferred_shape_r1f32_one =
      ShapeInference::InferMapShape(
          {&vector_32_}, ShapeUtil::MakeProgramShape({f32_}, f32_), {0});
  EXPECT_IS_OK(inferred_shape_r1f32_one.status());
  EXPECT_TRUE(ShapeUtil::Equal(vector_32_, *inferred_shape_r1f32_one));

  const absl::StatusOr<Shape> inferred_shape_r2s32 =
      ShapeInference::InferMapShape(
          {&s32matrix_64_64_, &s32matrix_64_64_, &s32matrix_64_64_},
          ShapeUtil::MakeProgramShape({s32_, s32_, s32_}, s32_), {0, 1});
  EXPECT_IS_OK(inferred_shape_r2s32.status());
  EXPECT_TRUE(ShapeUtil::Equal(s32matrix_64_64_, *inferred_shape_r2s32));

  const auto no_args_error = ShapeInference::InferMapShape(
      {}, ShapeUtil::MakeProgramShape({f32_, f32_}, f32_), {});
  ASSERT_FALSE(no_args_error.ok());
  ASSERT_THAT(no_args_error.status().message(),
              HasSubstr("expects at least one argument"));

  const auto args_diff_shapes_error = ShapeInference::InferMapShape(
      {&vector_32_, &vector_64_},
      ShapeUtil::MakeProgramShape({f32_, f32_}, f32_), {0});
  ASSERT_FALSE(args_diff_shapes_error.ok());
  ASSERT_THAT(args_diff_shapes_error.status().message(),
              HasSubstr("requires all operands to have the same shape"));

  const auto arity_error = ShapeInference::InferMapShape(
      {&vector_32_, &vector_32_}, ShapeUtil::MakeProgramShape({f32_}, f32_),
      {0});
  ASSERT_FALSE(arity_error.ok());
  ASSERT_THAT(arity_error.status().message(),
              HasSubstr("function arity must match"));

  const auto output_shape_error = ShapeInference::InferMapShape(
      {&vector_32_, &vector_32_},
      ShapeUtil::MakeProgramShape({f32_, f32_}, vector_32_), {0});
  ASSERT_FALSE(output_shape_error.ok());
  ASSERT_THAT(output_shape_error.status().message(),
              HasSubstr("result has to be a scalar"));

  const auto param_shape_error = ShapeInference::InferMapShape(
      {&vector_32_, &vector_32_},
      ShapeUtil::MakeProgramShape({vector_32_, f32_}, f32_), {0});
  ASSERT_FALSE(param_shape_error.ok());
  ASSERT_THAT(param_shape_error.status().message(),
              HasSubstr("parameter has to be a scalar"));

  const auto param_element_type_error = ShapeInference::InferMapShape(
      {&vector_32_, &vector_32_},
      ShapeUtil::MakeProgramShape({f32_, s32_}, f32_), {0});
  ASSERT_FALSE(param_element_type_error.ok());
  ASSERT_THAT(param_element_type_error.status().message(),
              HasSubstr("parameter type has to match argument"));

  const Shape arg = ShapeUtil::MakeShape(F32, {20});
  ProgramShape to_apply = ShapeUtil::MakeProgramShape({f32_}, f32_);
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferMapShape({&arg}, to_apply, {0});
  EXPECT_IS_OK(inferred_shape.status());
  EXPECT_TRUE(ShapeUtil::Equal(arg, *inferred_shape));

  const absl::StatusOr<Shape> inferred_shape_error1 =
      ShapeInference::InferMapShape(
          {&arg}, ShapeUtil::MakeProgramShape({f32_, f32_}, f32_), {0});
  ASSERT_FALSE(inferred_shape_error1.ok());
  ASSERT_THAT(inferred_shape_error1.status().message(),
              HasSubstr("arity must match number of arguments"));

  const absl::StatusOr<Shape> inferred_shape_error2 =
      ShapeInference::InferMapShape(
          {&arg}, ShapeUtil::MakeProgramShape({vector_32_}, f32_), {0});
  ASSERT_FALSE(inferred_shape_error2.ok());
  ASSERT_THAT(inferred_shape_error2.status().message(),
              HasSubstr("has to be a scalar"));

  const absl::StatusOr<Shape> inferred_shape_error3 =
      ShapeInference::InferMapShape(
          {&arg}, ShapeUtil::MakeProgramShape({f32_}, vector_32_), {0});
  ASSERT_FALSE(inferred_shape_error3.ok());
  ASSERT_THAT(inferred_shape_error3.status().message(),
              HasSubstr("has to be a scalar"));

  const absl::StatusOr<Shape> inferred_shape_error5 =
      ShapeInference::InferMapShape(
          {&arg}, ShapeUtil::MakeProgramShape({s32_}, s32_), {0});
  ASSERT_FALSE(inferred_shape_error5.ok());
  ASSERT_THAT(inferred_shape_error5.status().message(),
              HasSubstr("parameter type has to match argument"));
}

TEST_F(ShapeInferenceTest, MapWithDifferentInputTypes) {
  const Shape arg0 = ShapeUtil::MakeShape(F32, {20});
  const Shape arg1 = ShapeUtil::MakeShape(S32, {20});
  ProgramShape to_apply = ShapeUtil::MakeProgramShape({f32_, s32_}, s32_);
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferMapShape({&arg0, &arg1}, to_apply, {0});
  EXPECT_IS_OK(inferred_shape.status());
  const Shape expected = ShapeUtil::MakeShape(S32, {20});
  EXPECT_TRUE(ShapeUtil::Equal(expected, *inferred_shape));
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
  const Shape f32_arg_shape = ShapeUtil::MakeShape(F32, {5, 3});
  const Shape s32_arg_shape = ShapeUtil::MakeShape(S32, {5, 3});
  ProgramShape to_apply = ShapeUtil::MakeProgramShape(
      {f32_, s32_, f32_, s32_}, ShapeUtil::MakeTupleShape({f32_, s32_}));
  const absl::StatusOr<Shape> inferred_shape = ShapeInference::InferReduceShape(
      {&f32_arg_shape, &s32_arg_shape, &f32_, &s32_}, {0, 1}, to_apply);
  EXPECT_IS_OK(inferred_shape.status());
  EXPECT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeTupleShape({f32_, s32_}),
                               *inferred_shape));
}

TEST_F(ReduceShapeInferenceTest, ReduceWindowMultiOutput) {
  const Shape f32_arg_shape = ShapeUtil::MakeShape(F32, {5, 3, 1});
  const Shape s32_arg_shape = ShapeUtil::MakeShape(S32, {5, 3, 1});
  std::vector<const Shape*> args = {&f32_arg_shape, &s32_arg_shape};
  std::vector<const Shape*> inits = {&f32_, &s32_};
  ProgramShape to_apply = ShapeUtil::MakeProgramShape(
      {f32_, s32_, f32_, s32_}, ShapeUtil::MakeTupleShape({f32_, s32_}));
  std::vector<int64_t> window_dimensions = {1, 2, 4};
  std::vector<int64_t> window_strides = {1, 1, 1};
  std::vector<std::pair<int64_t, int64_t>> padding_values =
      MakePadding(f32_arg_shape.dimensions(), window_dimensions, window_strides,
                  Padding::kValid);
  TF_ASSERT_OK_AND_ASSIGN(
      const Window window,
      ShapeInference::InferWindowFromDimensions(
          window_dimensions, window_strides, padding_values, {}, {}));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferReduceWindowShape(
          absl::MakeSpan(args), absl::MakeSpan(inits), window, to_apply);
  VLOG(2) << inferred_shape->ToString() << "\n";
  EXPECT_IS_OK(inferred_shape.status());
  EXPECT_TRUE(ShapeUtil::Equal(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {5, 2, 0}),
                                 ShapeUtil::MakeShape(S32, {5, 2, 0})}),
      *inferred_shape));
}

TEST_F(ReduceShapeInferenceTest, ErrorMultiOutputBadReducerInput1) {
  const Shape f32_arg_shape = ShapeUtil::MakeShape(F32, {5, 3});
  const Shape s32_arg_shape = ShapeUtil::MakeShape(S32, {5, 3});
  ProgramShape to_apply =
      ShapeUtil::MakeProgramShape({f32_, s32_, f32_, s32_, f32_, s32_},
                                  ShapeUtil::MakeTupleShape({f32_, s32_}));
  const absl::StatusOr<Shape> inferred_shape = ShapeInference::InferReduceShape(
      {&f32_arg_shape, &s32_arg_shape, &f32_, &s32_}, {0, 1}, to_apply);
  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("must take 4 parameters, but takes 6 parameter(s)"));
}

TEST_F(ReduceShapeInferenceTest, ErrorMultiOutputBadReducerInput2) {
  const Shape f32_arg_shape = ShapeUtil::MakeShape(F32, {5, 3});
  const Shape s32_arg_shape = ShapeUtil::MakeShape(S32, {5, 3});
  ProgramShape to_apply = ShapeUtil::MakeProgramShape(
      {s32_, s32_, f32_, s32_}, ShapeUtil::MakeTupleShape({f32_, s32_}));
  const absl::StatusOr<Shape> inferred_shape = ShapeInference::InferReduceShape(
      {&f32_arg_shape, &s32_arg_shape, &f32_, &s32_}, {0, 1}, to_apply);
  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(
      inferred_shape.status().message(),
      HasSubstr(
          "parameter shape differs from the result shape: s32[] vs f32[]"));
}

TEST_F(ReduceShapeInferenceTest, ErrorMultiOutputBadReducerInput3) {
  ProgramShape to_apply = ShapeUtil::MakeProgramShape(
      {s32_, s32_, f32_, s32_}, ShapeUtil::MakeTupleShape({f32_, s32_}));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferReduceShape({}, {0, 1}, to_apply);
  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("must have at least 2 arguments, has 0"));
}

TEST_F(ReduceShapeInferenceTest, ErrorBadReduceWindowInput) {
  const Shape f32_arg_shape = ShapeUtil::MakeShape(F32, {5, 3, 1});
  const Shape s32_arg_shape = ShapeUtil::MakeShape(S32, {5, 3, 1});
  std::vector<const Shape*> args = {&f32_arg_shape, &s32_arg_shape};
  std::vector<const Shape*> inits = {&f32_, &s32_};
  ProgramShape to_apply = ShapeUtil::MakeProgramShape(
      {f32_, f32_, f32_, f32_}, ShapeUtil::MakeTupleShape({f32_, s32_}));
  std::vector<int64_t> window_dimensions = {1, 2, 4};
  std::vector<int64_t> window_strides = {1, 1, 1};
  std::vector<std::pair<int64_t, int64_t>> padding_values =
      MakePadding(f32_arg_shape.dimensions(), window_dimensions, window_strides,
                  Padding::kValid);
  TF_ASSERT_OK_AND_ASSIGN(
      const Window window,
      ShapeInference::InferWindowFromDimensions(
          window_dimensions, window_strides, padding_values, {}, {}));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferReduceWindowShape(
          absl::MakeSpan(args), absl::MakeSpan(inits), window, to_apply);
  EXPECT_FALSE(inferred_shape.status().ok());
  EXPECT_THAT(inferred_shape.status().message(), HasSubstr("f32[] vs s32[]"));
}

TEST_F(ReduceShapeInferenceTest, ErrorMultiOutputBadReducerOutput1) {
  const Shape f32_arg_shape = ShapeUtil::MakeShape(F32, {5, 3});
  const Shape s32_arg_shape = ShapeUtil::MakeShape(S32, {5, 3});
  ProgramShape to_apply =
      ShapeUtil::MakeProgramShape({f32_, s32_, f32_, s32_}, f32_);
  const absl::StatusOr<Shape> inferred_shape = ShapeInference::InferReduceShape(
      {&f32_arg_shape, &s32_arg_shape, &f32_, &s32_}, {0, 1}, to_apply);
  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(
      inferred_shape.status().message(),
      HasSubstr("must produce a tuple with 2 elements, but produces a scalar"));
}

TEST_F(ReduceShapeInferenceTest, ErrorMultiOutputBadReducerOutput2) {
  const Shape f32_arg_shape = ShapeUtil::MakeShape(F32, {5, 3});
  const Shape s32_arg_shape = ShapeUtil::MakeShape(S32, {5, 3});
  ProgramShape to_apply = ShapeUtil::MakeProgramShape(
      {f32_, s32_, f32_, s32_}, ShapeUtil::MakeTupleShape({f32_, s32_, s32_}));
  const absl::StatusOr<Shape> inferred_shape = ShapeInference::InferReduceShape(
      {&f32_arg_shape, &s32_arg_shape, &f32_, &s32_}, {0, 1}, to_apply);
  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(
      inferred_shape.status().message(),
      HasSubstr("must produce a tuple with 2 elements, but has 3 elements"));
}

TEST_F(ReduceShapeInferenceTest, ErrorMultiOutputBadReducerBoth) {
  const Shape f32_arg_shape = ShapeUtil::MakeShape(F32, {5, 3});
  const Shape s32_arg_shape = ShapeUtil::MakeShape(S32, {5, 3});
  ProgramShape to_apply = ShapeUtil::MakeProgramShape(
      {s32_, s32_, s32_, s32_}, ShapeUtil::MakeTupleShape({s32_, s32_}));
  const absl::StatusOr<Shape> inferred_shape = ShapeInference::InferReduceShape(
      {&f32_arg_shape, &s32_arg_shape, &f32_, &s32_}, {0, 1}, to_apply);
  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("accumulator shape at index 0 differs from the "
                        "init_value shape: s32[] vs f32[]"));
}

TEST_F(ReduceShapeInferenceTest, ErrorOutOfBoundsDimension) {
  ProgramShape to_apply = ShapeUtil::MakeProgramShape({f32_, f32_}, f32_);
  const Shape arg_shape = ShapeUtil::MakeShape(F32, {5, 3});
  const absl::StatusOr<Shape> inferred_shape = ShapeInference::InferReduceShape(
      {&arg_shape, &f32_},
      /*dimensions_to_reduce=*/{3, 4}, to_apply);
  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("out-of-bounds dimension"));
}

TEST_F(ReduceShapeInferenceTest, ErrorToApplyArity) {
  ProgramShape to_apply = ShapeUtil::MakeProgramShape({f32_, f32_, f32_}, f32_);
  const Shape arg_shape = ShapeUtil::MakeShape(F32, {5, 3});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferReduceShape({&arg_shape, &f32_},
                                       /*dimensions_to_reduce=*/{0}, to_apply);
  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("take 2 parameters"));
}

TEST_F(ReduceShapeInferenceTest, ErrorElementTypeVsApplyType) {
  ProgramShape to_apply = ShapeUtil::MakeProgramShape({f32_, f32_}, s32_);
  const Shape arg_shape = ShapeUtil::MakeShape(F32, {5, 3});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferReduceShape({&arg_shape, &f32_},
                                       /*dimensions_to_reduce=*/{0}, to_apply);
  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("0-th parameter shape differs"));
}

TEST_F(ReduceShapeInferenceTest, ReduceWithRepeatedReduceDimension) {
  ProgramShape to_apply = ShapeUtil::MakeProgramShape({f32_, f32_}, f32_);
  const Shape arg_shape = ShapeUtil::MakeShape(F32, {5, 3});
  const absl::StatusOr<Shape> inferred_shape = ShapeInference::InferReduceShape(
      {&arg_shape, &f32_},
      /*dimensions_to_reduce=*/{0, 0}, to_apply);
  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("Duplicate reduction dimension: 0"));
}

TEST_F(ShapeInferenceTest, InferSliceShapeRank2) {
  const Shape matrix_shape = ShapeUtil::MakeShape(F32, {128, 64});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferSliceShape(matrix_shape, {32, 0}, {64, 64}, {1, 1});
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {32, 64}), *inferred_shape));
}

TEST_F(ShapeInferenceTest, InferSliceWithDynamicDimensions) {
  const Shape matrix_shape = ShapeUtil::MakeShape(F32, {128, 64}, {true, true});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferSliceShape(matrix_shape, {32, 0}, {33, 64}, {1, 1});
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(ShapeUtil::Equal(
      ShapeUtil::MakeShape(F32, {1, 64}, {false, true}), *inferred_shape));
}

TEST_F(ShapeInferenceTest, InferSliceShapeRank2WithStrides) {
  const Shape matrix_shape = ShapeUtil::MakeShape(F32, {128, 64});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferSliceShape(matrix_shape, {32, 0}, {64, 64}, {2, 4});
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {16, 16}), *inferred_shape));
}

TEST_F(ShapeInferenceTest, InferSliceShapeRank2WithStridesNotIntegral) {
  const Shape matrix_shape = ShapeUtil::MakeShape(F32, {128, 64});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferSliceShape(matrix_shape, {15, 0}, {20, 13}, {2, 4});
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {3, 4}), *inferred_shape));
}

TEST_F(ShapeInferenceTest, InferInvalidStride) {
  const Shape matrix_shape = ShapeUtil::MakeShape(F32, {128, 64});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferSliceShape(matrix_shape, {127, 0}, {129, 2}, {0, 1});
  ASSERT_FALSE(inferred_shape.ok());
  ASSERT_EQ(tsl::error::INVALID_ARGUMENT, inferred_shape.status().code());
}

TEST_F(ShapeInferenceTest, InferOobSliceShapeRank2) {
  const Shape matrix_shape = ShapeUtil::MakeShape(F32, {128, 64});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferSliceShape(matrix_shape, {127, 0}, {129, 2}, {1, 1});
  ASSERT_FALSE(inferred_shape.ok());
  ASSERT_EQ(tsl::error::INVALID_ARGUMENT, inferred_shape.status().code());
}

TEST_F(ShapeInferenceTest, InferSliceShapeRank1) {
  const Shape vector_shape = ShapeUtil::MakeShape(F32, {17});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferSliceShape(vector_shape, {2}, {4}, {1});
  ASSERT_TRUE(inferred_shape.ok());
  ASSERT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {2}), *inferred_shape));
}

TEST_F(ShapeInferenceTest, InferConstIndexShape) {
  const Shape tuple_shape = ShapeUtil::MakeTupleShape({f32_, s32_});
  const absl::StatusOr<Shape> inferred0_status =
      ShapeInference::InferGetTupleElementShape(tuple_shape, 0);
  const absl::StatusOr<Shape> inferred1_status =
      ShapeInference::InferGetTupleElementShape(tuple_shape, 1);
  ASSERT_IS_OK(inferred0_status.status());
  ASSERT_IS_OK(inferred1_status.status());
  ASSERT_TRUE(ShapeUtil::Equal(f32_, *inferred0_status));
  ASSERT_TRUE(ShapeUtil::Equal(s32_, *inferred1_status));
}

TEST_F(ShapeInferenceTest, InferTupleElementShapeOutOfBound) {
  const Shape tuple_shape = ShapeUtil::MakeTupleShape({f32_, s32_});
  const absl::StatusOr<Shape> inferredNegative_status =
      ShapeInference::InferGetTupleElementShape(tuple_shape, -1);
  const absl::StatusOr<Shape> inferred2_status =
      ShapeInference::InferGetTupleElementShape(tuple_shape, 2);
  ASSERT_FALSE(inferredNegative_status.ok());
  ASSERT_FALSE(inferred2_status.ok());
  EXPECT_THAT(inferredNegative_status.status().message(),
              HasSubstr("attempt to index out of tuple bounds"));
  EXPECT_THAT(inferred2_status.status().message(),
              HasSubstr("attempt to index out of tuple bounds"));
}

TEST_F(ShapeInferenceTest, InferPowShape) {
  const Shape ten_floats = ShapeUtil::MakeShape(F32, {10});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kPower, ten_floats, f32_,
                                         {});
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(ShapeUtil::Equal(ten_floats, *inferred_shape));
}

TEST_F(ShapeInferenceTest, InferCompareShape) {
  const Shape ten_floats = ShapeUtil::MakeShape(F32, {10});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kCompare, ten_floats, f32_,
                                         {});
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(PRED, {10}), *inferred_shape));
}

TEST_F(ShapeInferenceTest, InferReshapeDegenerateCombine) {
  // [1, <=1]
  //   | reshape
  // [<=1]
  //
  // Both output dimension can be dynamic, use inferred_dimension to tie-break.
  const Shape operand = ShapeUtil::MakeShape(F32, {1, 1}, {false, true});
  const auto status =
      ShapeInference::InferReshapeShape(operand, {1},
                                        /*inferred_dimension=*/-1);
  ASSERT_EQ(ShapeUtil::MakeShape(F32, {1}, {true}), *status);
}

TEST_F(ShapeInferenceTest, InferReshapeSplit) {
  // [<=10]
  //   | reshape
  // [1, 10]
  //
  // Both output dimension can be dynamic, use inferred_dimension to tie-break.
  const Shape operand = ShapeUtil::MakeShape(F32, {10}, {true});
  const auto status =
      ShapeInference::InferReshapeShape(operand, {1, 10},
                                        /*inferred_dimension=*/0);
  ASSERT_EQ(ShapeUtil::MakeShape(F32, {1, 10}, {true, false}), *status);
}

TEST_F(ShapeInferenceTest, InferReshapeCombine) {
  // [6, <=10]
  //   | reshape
  // [<=60]
  const Shape operand = ShapeUtil::MakeShape(F32, {6, 10}, {false, true});
  const auto status =
      ShapeInference::InferReshapeShape(operand, {60},
                                        /*inferred_dimension=*/-11);
  ASSERT_EQ(ShapeUtil::MakeShape(F32, {60}, {true}), *status);
}

TEST_F(ShapeInferenceTest, UnchangedDimension) {
  // [6, <=10]
  //   | reshape
  // [2, 3, <=10]
  const Shape operand = ShapeUtil::MakeShape(F32, {6, 10}, {false, true});
  const auto status =
      ShapeInference::InferReshapeShape(operand, {2, 3, 10},
                                        /*inferred_dimension=*/-11);
  ASSERT_EQ(ShapeUtil::MakeShape(F32, {2, 3, 10}, {false, false, true}),
            *status);
}

TEST_F(ShapeInferenceTest, InferDynamicBroadcast) {
  // CHECK:
  // %broadcast = s32[15,<=15]{1,0} broadcast(s32[<=15]{0}), dimensions={1}

  const Shape operand_shape = ShapeUtil::MakeShape(F32, {15}, {true});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBroadcastShape(operand_shape, {15});
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_EQ(ShapeUtil::MakeShape(F32, {15, 15}, {false, true}),
            *inferred_shape);
}

TEST_F(ShapeInferenceTest, BroadcastScalar) {
  for (auto element_type : {F32, U32, S8}) {
    const Shape scalar_shape = ShapeUtil::MakeShape(element_type, {});
    {  // no-op scalar broadcast
      const auto status = ShapeInference::InferBroadcastShape(scalar_shape, {});
      ASSERT_IS_OK(status.status());
      ASSERT_TRUE(ShapeUtil::Equal(scalar_shape, *status));
    }
    const Shape oned_shape = ShapeUtil::MakeShape(element_type, {3});
    {  // scalar -> 1d broadcast
      const auto status =
          ShapeInference::InferBroadcastShape(scalar_shape, {3});
      ASSERT_IS_OK(status.status());
      ASSERT_TRUE(ShapeUtil::Equal(oned_shape, *status));
    }
    {  // no-op 1d broadcast
      const auto status = ShapeInference::InferBroadcastShape(oned_shape, {});
      ASSERT_IS_OK(status.status());
      ASSERT_TRUE(ShapeUtil::Equal(oned_shape, *status));
    }
    const Shape twod_shape = ShapeUtil::MakeShape(element_type, {2, 3});
    {  // scalar -> 2d broadcast
      const auto status =
          ShapeInference::InferBroadcastShape(scalar_shape, {2, 3});
      ASSERT_IS_OK(status.status());
      ASSERT_TRUE(ShapeUtil::Equal(twod_shape, *status));
    }
    {  // 1d -> 2d broadcast
      const auto status = ShapeInference::InferBroadcastShape(oned_shape, {2});
      ASSERT_IS_OK(status.status());
      ASSERT_TRUE(ShapeUtil::Equal(twod_shape, *status));
    }
  }
}

// scalar <dot> vector: ok
TEST_F(ShapeInferenceTest, ScalarDotVector) {
  DotDimensionNumbers dot_dnums;
  const absl::StatusOr<Shape> inferred_shape = ShapeInference::InferDotOpShape(
      f32_, vector_32_, dot_dnums, /*preferred_element_type=*/std::nullopt);
  EXPECT_TRUE(inferred_shape.ok());
  EXPECT_EQ(*inferred_shape, vector_32_);
}

// 3D <dot> 2D: error
TEST_F(ShapeInferenceTest, DotWithRankHigherThanTwo) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  const absl::StatusOr<Shape> inferred_shape = ShapeInference::InferDotOpShape(
      ShapeUtil::MakeShape(F32, {32, 32, 32}), matrix_32_64_, dot_dnums,
      /*preferred_element_type=*/std::nullopt);
  EXPECT_TRUE(inferred_shape.ok());
  EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape,
                               ShapeUtil::MakeShape(F32, {32, 32, 64})));
}

// vector <dot> vector -> scalar
TEST_F(ShapeInferenceTest, VectorDotVector) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(0);
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferDotOpShape(vector_64_, vector_64_, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(ShapeUtil::Equal(f32_, *inferred_shape));
  const absl::StatusOr<Shape> inferred_shape_mismatch =
      ShapeInference::InferDotOpShape(vector_64_, vector_32_, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_FALSE(inferred_shape_mismatch.ok());
}

// matrix <dot> vector -> vector
TEST_F(ShapeInferenceTest, MatrixDotVector) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferDotOpShape(matrix_32_64_, vector_64_, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape, vector_32_));
  const absl::StatusOr<Shape> inferred_shape_mismatch =
      ShapeInference::InferDotOpShape(matrix_32_64_, vector_32_, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_FALSE(inferred_shape_mismatch.ok());
}

// vector <dot> matrix -> vector
TEST_F(ShapeInferenceTest, VectorDotMatrix) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(0);
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferDotOpShape(vector_32_, matrix_32_64_, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape, vector_64_));
  const absl::StatusOr<Shape> inferred_shape_mismatch =
      ShapeInference::InferDotOpShape(vector_64_, matrix_32_64_, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_FALSE(inferred_shape_mismatch.ok());
}

// matrix <dot> matrix -> matrix
TEST_F(ShapeInferenceTest, MatrixDotMatrix) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  const absl::StatusOr<Shape> inferred_shape_match =
      ShapeInference::InferDotOpShape(matrix_32_64_, matrix_64_48_, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_IS_OK(inferred_shape_match.status());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, matrix_32_48_))
      << "inferred: " << ShapeUtil::HumanString(*inferred_shape_match)
      << " expected: " << ShapeUtil::HumanString(matrix_64_48_);
  const absl::StatusOr<Shape> inferred_shape_mismatch =
      ShapeInference::InferDotOpShape(matrix_32_64_, matrix_32_64_, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_FALSE(inferred_shape_mismatch.ok());
}

// BatchMatMul with two batch dimensions and one contracting dimension.
TEST_F(ShapeInferenceTest, DotGeneral) {
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {5, 2, 11, 3});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {5, 2, 3, 14});
  const Shape output_shape = ShapeUtil::MakeShape(F32, {5, 2, 11, 14});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(3);
  dot_dnums.add_lhs_batch_dimensions(0);
  dot_dnums.add_lhs_batch_dimensions(1);

  dot_dnums.add_rhs_contracting_dimensions(2);
  dot_dnums.add_rhs_batch_dimensions(0);
  dot_dnums.add_rhs_batch_dimensions(1);

  const absl::StatusOr<Shape> inferred_shape_match =
      ShapeInference::InferDotOpShape(lhs_shape, rhs_shape, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_IS_OK(inferred_shape_match.status());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, output_shape))
      << "inferred: " << ShapeUtil::HumanString(*inferred_shape_match)
      << " expected: " << ShapeUtil::HumanString(output_shape);
}

// BatchMatMul with two contracting dimensions fails.
TEST_F(ShapeInferenceTest, DotWithTwoContractingDimsFails) {
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {2, 11, 3, 2});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {2, 3, 14});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(2);
  dot_dnums.add_lhs_contracting_dimensions(3);
  dot_dnums.add_lhs_batch_dimensions(0);

  dot_dnums.add_rhs_contracting_dimensions(1);
  dot_dnums.add_rhs_batch_dimensions(0);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferDotOpShape(lhs_shape, rhs_shape, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_FALSE(inferred_shape.ok());
  ASSERT_THAT(inferred_shape.status().message(),
              HasSubstr("Must specify the same number of contracting "
                        "dimensions for lhs and rhs."));
}

TEST_F(ShapeInferenceTest, DotWithTwoContractingDimsPasses) {
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {2, 11, 3, 2});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {2, 3, 2, 14});
  const Shape output_shape = ShapeUtil::MakeShape(F32, {2, 11, 14});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(2);
  dot_dnums.add_lhs_contracting_dimensions(3);
  dot_dnums.add_lhs_batch_dimensions(0);

  dot_dnums.add_rhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(2);
  dot_dnums.add_rhs_batch_dimensions(0);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferDotOpShape(lhs_shape, rhs_shape, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  EXPECT_TRUE(inferred_shape.ok());
  EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, output_shape));
}

TEST_F(ShapeInferenceTest, ErrorSetDimensionSize) {
  const Shape arg_shape = ShapeUtil::MakeShape(F32, {5, 3});
  const Shape val_shape = ShapeUtil::MakeShape(S32, {1});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferSetDimensionSizeShape(arg_shape, val_shape,
                                                 /*dimension=*/0);

  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("value has to be S32 scalar"));
}

TEST_F(ShapeInferenceTest, ErrorSetDimensionSizeWrongType) {
  const Shape arg_shape = ShapeUtil::MakeShape(F32, {5, 3});
  const Shape val_shape = ShapeUtil::MakeShape(U32, {});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferSetDimensionSizeShape(arg_shape, val_shape,
                                                 /*dimension=*/0);

  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("value has to be S32 scalar"));
}

// BatchMatMul with different batch dimension sizes fails.
TEST_F(ShapeInferenceTest, DotWithMismatchedBatchDimSizesFails) {
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {2, 11, 3});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {3, 3, 14});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(2);
  dot_dnums.add_lhs_batch_dimensions(0);

  dot_dnums.add_rhs_contracting_dimensions(1);
  dot_dnums.add_rhs_batch_dimensions(0);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferDotOpShape(lhs_shape, rhs_shape, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_FALSE(inferred_shape.ok());
  ASSERT_THAT(inferred_shape.status().message(),
              HasSubstr("Batch dimension sizes are not compatible"));
}

// BatchMatMul with different batch dimension numbers passes
TEST_F(ShapeInferenceTest, DotWithMismatchedBatchDimNumbersPasses) {
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {2, 11, 3});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {3, 2, 14});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(2);
  dot_dnums.add_lhs_batch_dimensions(0);

  dot_dnums.add_rhs_contracting_dimensions(0);
  dot_dnums.add_rhs_batch_dimensions(1);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferDotOpShape(lhs_shape, rhs_shape, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_TRUE(inferred_shape.ok());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape,
                               ShapeUtil::MakeShape(F32, {2, 11, 14})));
}

// BatchMatMul with out-of-range dimension numbers fails.
TEST_F(ShapeInferenceTest, DotWithContractingDimNumberOutOfRange) {
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {2, 11, 3});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {2, 3, 14});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(3);
  dot_dnums.add_lhs_batch_dimensions(0);

  dot_dnums.add_rhs_contracting_dimensions(0);
  dot_dnums.add_rhs_batch_dimensions(1);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferDotOpShape(lhs_shape, rhs_shape, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_FALSE(inferred_shape.ok());
  ASSERT_THAT(inferred_shape.status().message(),
              HasSubstr("A dimension number is out of range"));
}

// BatchMatMul with non-unique dimension numbers fails.
TEST_F(ShapeInferenceTest, DotWithContractingNonUniqueDimNumber) {
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {2, 11, 3});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {2, 3, 14});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_lhs_batch_dimensions(0);

  dot_dnums.add_rhs_contracting_dimensions(0);
  dot_dnums.add_rhs_batch_dimensions(1);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferDotOpShape(lhs_shape, rhs_shape, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_FALSE(inferred_shape.ok());
  ASSERT_THAT(inferred_shape.status().message(),
              HasSubstr("A dimension number is not unique"));
}

TEST_F(ShapeInferenceTest, DotWithIntegralPreferredElementType) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferDotOpShape(
                              ShapeUtil::MakeShape(S8, {32, 32}),
                              ShapeUtil::MakeShape(S16, {32, 32}), dot_dnums,
                              /*preferred_element_type=*/S32));
  EXPECT_TRUE(
      ShapeUtil::Equal(inferred_shape, ShapeUtil::MakeShape(S32, {32, 32})));
}

TEST_F(ShapeInferenceTest, DotWithPreferredElementTypeSameAsInferredType) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferDotOpShape(
                              ShapeUtil::MakeShape(BF16, {32, 32}),
                              ShapeUtil::MakeShape(F32, {32, 32}), dot_dnums,
                              /*preferred_element_type=*/F32));
  EXPECT_TRUE(
      ShapeUtil::Equal(inferred_shape, ShapeUtil::MakeShape(F32, {32, 32})));
}

TEST_F(ShapeInferenceTest, FloatingPointDotWithNarrowerPreferredElementType) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferDotOpShape(
                              ShapeUtil::MakeShape(BF16, {32, 32}),
                              ShapeUtil::MakeShape(F32, {32, 32}), dot_dnums,
                              /*preferred_element_type=*/BF16));
  EXPECT_TRUE(
      ShapeUtil::Equal(inferred_shape, ShapeUtil::MakeShape(BF16, {32, 32})));
}

TEST_F(ShapeInferenceTest, FloatingPointDotWithIntegralPreferredElementType) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferDotOpShape(
                              ShapeUtil::MakeShape(BF16, {32, 32}),
                              ShapeUtil::MakeShape(BF16, {32, 32}), dot_dnums,
                              /*preferred_element_type=*/S32));
  EXPECT_TRUE(
      ShapeUtil::Equal(inferred_shape, ShapeUtil::MakeShape(S32, {32, 32})));
}

TEST_F(ShapeInferenceTest, IntegralDotWithFloatingPointPreferredElementType) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferDotOpShape(
                              ShapeUtil::MakeShape(S8, {32, 32}),
                              ShapeUtil::MakeShape(S16, {32, 32}), dot_dnums,
                              /*preferred_element_type=*/F32));
  EXPECT_TRUE(
      ShapeUtil::Equal(inferred_shape, ShapeUtil::MakeShape(F32, {32, 32})));
}

TEST_F(ShapeInferenceTest, DotWithPreferredElementTypeWithDifferentSignedness) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferDotOpShape(
                              ShapeUtil::MakeShape(S8, {32, 32}),
                              ShapeUtil::MakeShape(S16, {32, 32}), dot_dnums,
                              /*preferred_element_type=*/U32));
  EXPECT_TRUE(
      ShapeUtil::Equal(inferred_shape, ShapeUtil::MakeShape(U32, {32, 32})));
}

TEST_F(ShapeInferenceTest, DotWithNarrowerPreferredElementType) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferDotOpShape(
                              ShapeUtil::MakeShape(S8, {32, 32}),
                              ShapeUtil::MakeShape(S16, {32, 32}), dot_dnums,
                              /*preferred_element_type=*/S8));
  EXPECT_TRUE(
      ShapeUtil::Equal(inferred_shape, ShapeUtil::MakeShape(S8, {32, 32})));
}

TEST_F(ShapeInferenceTest, DotWithSparseLhs) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  SparsityDescriptor sparsity_descriptor;
  sparsity_descriptor.set_type(SparsityType::SPARSITY_STRUCTURED_N_M);
  sparsity_descriptor.set_n(2);
  sparsity_descriptor.set_m(4);
  sparsity_descriptor.set_index(0);
  sparsity_descriptor.set_dimension(1);

  std::vector<SparsityDescriptor> sparsity = {sparsity_descriptor};
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferDotOpShape(
          ShapeUtil::MakeShape(F32, {10, 16}),
          ShapeUtil::MakeShape(F32, {32, 20}), dot_dnums,
          /*preferred_element_type=*/std::nullopt, absl::MakeSpan(sparsity)));
  EXPECT_TRUE(
      ShapeUtil::Equal(inferred_shape, ShapeUtil::MakeShape(F32, {10, 20})));
}

TEST_F(ShapeInferenceTest, DotWithSparseRhs) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  SparsityDescriptor sparsity_descriptor;
  sparsity_descriptor.set_type(SparsityType::SPARSITY_STRUCTURED_N_M);
  sparsity_descriptor.set_n(2);
  sparsity_descriptor.set_m(4);
  sparsity_descriptor.set_index(1);
  sparsity_descriptor.set_dimension(0);

  std::vector<SparsityDescriptor> sparsity = {sparsity_descriptor};
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferDotOpShape(
          ShapeUtil::MakeShape(F32, {10, 32}),
          ShapeUtil::MakeShape(F32, {16, 20}), dot_dnums,
          /*preferred_element_type=*/std::nullopt, absl::MakeSpan(sparsity)));
  EXPECT_TRUE(
      ShapeUtil::Equal(inferred_shape, ShapeUtil::MakeShape(F32, {10, 20})));
}

TEST_F(ShapeInferenceTest, DotWithSparseBothOperands) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  SparsityDescriptor sparsity_lhs;
  sparsity_lhs.set_type(SparsityType::SPARSITY_STRUCTURED_N_M);
  sparsity_lhs.set_n(2);
  sparsity_lhs.set_m(4);
  sparsity_lhs.set_index(0);
  sparsity_lhs.set_dimension(1);
  SparsityDescriptor sparsity_rhs = sparsity_lhs;
  sparsity_rhs.set_index(1);
  sparsity_rhs.set_dimension(0);

  std::vector<SparsityDescriptor> sparsity = {sparsity_lhs, sparsity_rhs};
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferDotOpShape(
          ShapeUtil::MakeShape(F32, {10, 16}),
          ShapeUtil::MakeShape(F32, {16, 20}), dot_dnums,
          /*preferred_element_type=*/std::nullopt, absl::MakeSpan(sparsity)));
  EXPECT_TRUE(
      ShapeUtil::Equal(inferred_shape, ShapeUtil::MakeShape(F32, {10, 20})));
}

TEST_F(ShapeInferenceTest, DotWithIncorrectSparseDimensionSizeRatio) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  SparsityDescriptor sparsity_descriptor;
  sparsity_descriptor.set_type(SparsityType::SPARSITY_STRUCTURED_N_M);
  sparsity_descriptor.set_n(2);
  sparsity_descriptor.set_m(4);
  sparsity_descriptor.set_index(0);
  sparsity_descriptor.set_dimension(1);

  std::vector<SparsityDescriptor> sparsity = {sparsity_descriptor};
  const absl::StatusOr<Shape> inferred_shape = ShapeInference::InferDotOpShape(
      ShapeUtil::MakeShape(F32, {10, 32}), ShapeUtil::MakeShape(F32, {32, 20}),
      dot_dnums, /*preferred_element_type=*/std::nullopt,
      absl::MakeSpan(sparsity));
  ASSERT_FALSE(inferred_shape.ok());
  ASSERT_THAT(
      inferred_shape.status().message(),
      HasSubstr("Sparse dimension size ratio doesn't match the descriptor"));
}

TEST_F(ShapeInferenceTest, SparseDotMetadata) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_batch_dimensions(0);
  dot_dnums.add_lhs_contracting_dimensions(2);
  SparsityDescriptor sparsity_descriptor;
  sparsity_descriptor.set_type(SparsityType::SPARSITY_STRUCTURED_N_M);
  sparsity_descriptor.set_n(2);
  sparsity_descriptor.set_m(4);
  sparsity_descriptor.set_index(0);
  sparsity_descriptor.set_dimension(2);

  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferSparseDotMetadataShape(
                              ShapeUtil::MakeShape(F32, {5, 10, 16}), dot_dnums,
                              sparsity_descriptor));
  EXPECT_TRUE(
      ShapeUtil::Equal(inferred_shape, ShapeUtil::MakeShape(U16, {5, 10, 2})));
}

// <ragged-dot> mode 1 : [m,k], [g,k,n], [g] -> [m,n]
TEST_F(ShapeInferenceTest, RaggedDotRaggedNonContracting) {
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {11, 5});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {3, 5, 7});
  const Shape group_sizes_shape = ShapeUtil::MakeShape(U32, {3});
  const Shape output_shape = ShapeUtil::MakeShape(F32, {11, 7});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(1);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(0);
  ragged_dot_dnums.add_rhs_group_dimensions(0);

  const absl::StatusOr<Shape> inferred_shape_match =
      ShapeInference::InferRaggedDotOpShape(
          lhs_shape, rhs_shape, group_sizes_shape, ragged_dot_dnums,
          /*preferred_element_type=*/std::nullopt);
  ASSERT_IS_OK(inferred_shape_match.status());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, output_shape))
      << "inferred: " << ShapeUtil::HumanString(*inferred_shape_match)
      << " expected: " << ShapeUtil::HumanString(output_shape);
}

// <ragged-dot> mode 2 : [m,k], [k,n], [g] -> [g,m,n]
TEST_F(ShapeInferenceTest, RaggedDotRaggedContracting) {
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {11, 5});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {5, 7});
  const Shape group_sizes_shape = ShapeUtil::MakeShape(U32, {3});
  const Shape output_shape = ShapeUtil::MakeShape(F32, {3, 11, 7});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(1);

  const absl::StatusOr<Shape> inferred_shape_match =
      ShapeInference::InferRaggedDotOpShape(
          lhs_shape, rhs_shape, group_sizes_shape, ragged_dot_dnums,
          /*preferred_element_type=*/std::nullopt);
  ASSERT_IS_OK(inferred_shape_match.status());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, output_shape))
      << "inferred: " << ShapeUtil::HumanString(*inferred_shape_match)
      << " expected: " << ShapeUtil::HumanString(output_shape);
}

// <ragged-dot> mode 3 : [b,m,k], [b,k,n], [g] -> [b,m,n]
TEST_F(ShapeInferenceTest, RaggedDotRaggedBatch) {
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {3, 11, 5});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {3, 5, 7});
  const Shape group_sizes_shape = ShapeUtil::MakeShape(U32, {3});
  const Shape output_shape = ShapeUtil::MakeShape(F32, {3, 11, 7});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_batch_dimensions(0);
  dot_dnums.add_rhs_batch_dimensions(0);
  dot_dnums.add_lhs_contracting_dimensions(2);
  dot_dnums.add_rhs_contracting_dimensions(1);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(0);

  const absl::StatusOr<Shape> inferred_shape_match =
      ShapeInference::InferRaggedDotOpShape(
          lhs_shape, rhs_shape, group_sizes_shape, ragged_dot_dnums,
          /*preferred_element_type=*/std::nullopt);
  ASSERT_IS_OK(inferred_shape_match.status());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, output_shape))
      << "inferred: " << ShapeUtil::HumanString(*inferred_shape_match)
      << " expected: " << ShapeUtil::HumanString(output_shape);
}

// preferred_element_type should be respected
TEST_F(ShapeInferenceTest, RaggedDotRaggedContractingWithPreferredElementType) {
  const Shape lhs_shape = ShapeUtil::MakeShape(S8, {11, 5});
  const Shape rhs_shape = ShapeUtil::MakeShape(S8, {5, 7});
  const Shape group_sizes_shape = ShapeUtil::MakeShape(U32, {3});
  const Shape output_shape = ShapeUtil::MakeShape(S32, {3, 11, 7});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(1);

  const absl::StatusOr<Shape> inferred_shape_match =
      ShapeInference::InferRaggedDotOpShape(lhs_shape, rhs_shape,
                                            group_sizes_shape, ragged_dot_dnums,
                                            /*preferred_element_type=*/S32);
  ASSERT_IS_OK(inferred_shape_match.status());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, output_shape))
      << "inferred: " << ShapeUtil::HumanString(*inferred_shape_match)
      << " expected: " << ShapeUtil::HumanString(output_shape);
}

// ragged-dot contracting dim (k) must match between lhs and rhs
TEST_F(ShapeInferenceTest, RaggedDotRaggedNonContractingIncompatibleK) {
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {11, 5});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {3, 2, 7});
  const Shape group_sizes_shape = ShapeUtil::MakeShape(U32, {3});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(1);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(0);
  ragged_dot_dnums.add_rhs_group_dimensions(0);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferRaggedDotOpShape(
          lhs_shape, rhs_shape, group_sizes_shape, ragged_dot_dnums,
          /*preferred_element_type=*/std::nullopt);

  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("Contracting dimension sizes are not compatible"));
}

// ragged-dot contracting dim (k) must match between lhs and rhs
TEST_F(ShapeInferenceTest, RaggedDotRaggedContractingIncompatibleK) {
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {11, 5});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {2, 7});
  const Shape group_sizes_shape = ShapeUtil::MakeShape(U32, {3});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(1);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferRaggedDotOpShape(
          lhs_shape, rhs_shape, group_sizes_shape, ragged_dot_dnums,
          /*preferred_element_type=*/std::nullopt);

  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("Contracting dimension sizes are not compatible"));
}

// ragged-dot should have exactly one lhs ragged dimension
TEST_F(ShapeInferenceTest, RaggedDotIncorrectNumberOfLhsRaggedDimensions) {
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {11, 5});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {3, 5, 7});
  const Shape group_sizes_shape = ShapeUtil::MakeShape(U32, {3});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(1);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(0);
  ragged_dot_dnums.add_lhs_ragged_dimensions(1);
  ragged_dot_dnums.add_rhs_group_dimensions(0);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferRaggedDotOpShape(
          lhs_shape, rhs_shape, group_sizes_shape, ragged_dot_dnums,
          /*preferred_element_type=*/std::nullopt);

  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("There must be exactly one ragged dimension"));
}

// lhs_ragged_dimension should be in [0, rank(lhs)).
TEST_F(ShapeInferenceTest, RaggedDotLhsRaggedDimensionOutOfBounds) {
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {11, 5});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {3, 5, 7});
  const Shape group_sizes_shape = ShapeUtil::MakeShape(U32, {3});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(1);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(2);
  ragged_dot_dnums.add_rhs_group_dimensions(0);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferRaggedDotOpShape(
          lhs_shape, rhs_shape, group_sizes_shape, ragged_dot_dnums,
          /*preferred_element_type=*/std::nullopt);

  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("lhs ragged dimension 2 is out of range [0, 2)"));
}

TEST_F(ShapeInferenceTest, RaggedDotGroupSizesIncorrectRank) {
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {11, 5});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {3, 5, 7});
  const Shape group_sizes_shape = ShapeUtil::MakeShape(U32, {2, 3});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(1);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(0);
  ragged_dot_dnums.add_rhs_group_dimensions(0);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferRaggedDotOpShape(
          lhs_shape, rhs_shape, group_sizes_shape, ragged_dot_dnums,
          /*preferred_element_type=*/std::nullopt);

  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("expected group_sizes to have rank 1, got 2"));
}

// test the vectorized shape of group_sizes for ragged non-contracting dim
TEST_F(ShapeInferenceTest, RaggedDotMode1VectorizedGroupSizesIncorrectShape) {
  // [b, m1, m2, k]
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {19, 17, 11, 5});
  // [g, b, k, n]
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {3, 19, 5, 7});
  // when m2 is ragged, the correct group_sizes shape is [b, m1, g]
  const Shape group_sizes_shape = ShapeUtil::MakeShape(U32, {19, 11, 3});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_batch_dimensions(0);
  dot_dnums.add_lhs_contracting_dimensions(3);
  dot_dnums.add_rhs_batch_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(2);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(2);
  ragged_dot_dnums.add_rhs_group_dimensions(0);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferRaggedDotOpShape(
          lhs_shape, rhs_shape, group_sizes_shape, ragged_dot_dnums,
          /*preferred_element_type=*/std::nullopt);

  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("group_sizes is expected to have shape [19, 17, 3],"
                        " got [19, 11, 3]"));
}

// test the vectorized shape of group_sizes for ragged contracting dim
TEST_F(ShapeInferenceTest, RaggedDotMode2VectorizedGroupSizesIncorrectShape) {
  // [b, m, k1, k2]
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {19, 11, 17, 5});
  // [b, k1, k2, n]
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {19, 17, 5, 7});
  // when k2 is ragged, the correct group_sizes shape is [b, k1, g]
  const Shape group_sizes_shape = ShapeUtil::MakeShape(U32, {19, 11, 3});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_batch_dimensions(0);
  dot_dnums.add_lhs_contracting_dimensions(2);
  dot_dnums.add_lhs_contracting_dimensions(3);
  dot_dnums.add_rhs_batch_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(2);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(3);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferRaggedDotOpShape(
          lhs_shape, rhs_shape, group_sizes_shape, ragged_dot_dnums,
          /*preferred_element_type=*/std::nullopt);

  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("group_sizes is expected to have shape [19, 17, 3],"
                        " got [19, 11, 3]"));
}

// test the vectorized shape of group_sizes for ragged batch dim
TEST_F(ShapeInferenceTest, RaggedDotMode3VectorizedGroupSizesIncorrectShape) {
  // [b1, b2, m, k]
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {17, 19, 11, 5});
  // [b1, b2, k, n]
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {17, 19, 5, 7});
  // the correct shape is [b1, g]
  const Shape group_sizes_shape = ShapeUtil::MakeShape(U32, {19, 3});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_batch_dimensions(0);
  dot_dnums.add_lhs_batch_dimensions(1);
  dot_dnums.add_lhs_contracting_dimensions(3);
  dot_dnums.add_rhs_batch_dimensions(0);
  dot_dnums.add_rhs_batch_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(2);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(1);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferRaggedDotOpShape(
          lhs_shape, rhs_shape, group_sizes_shape, ragged_dot_dnums,
          /*preferred_element_type=*/std::nullopt);

  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("group_sizes is expected to have shape [17, 3],"
                        " got [19, 3]"));
}

// test the vectorized shape of group_sizes when some dimension-sizes repeat
TEST_F(ShapeInferenceTest, RaggedDotMode1VectorizedGroupSizesRepeatingDims) {
  // [b1, b2, m1, m2, k]
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {64, 64, 32, 32, 128});
  // [g, b1, b2, k, n]
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {3, 64, 64, 128, 32});
  // [b1, b2, m1, g]
  const Shape group_sizes_shape = ShapeUtil::MakeShape(U32, {64, 64, 32, 3});
  // [b1, b2, m1, m2, n]
  const Shape output_shape = ShapeUtil::MakeShape(F32, {64, 64, 32, 32, 32});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_batch_dimensions(0);
  dot_dnums.add_lhs_batch_dimensions(1);
  dot_dnums.add_lhs_contracting_dimensions(4);
  dot_dnums.add_rhs_batch_dimensions(1);
  dot_dnums.add_rhs_batch_dimensions(2);
  dot_dnums.add_rhs_contracting_dimensions(3);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(3);
  ragged_dot_dnums.add_rhs_group_dimensions(0);

  const absl::StatusOr<Shape> inferred_shape_match =
      ShapeInference::InferRaggedDotOpShape(
          lhs_shape, rhs_shape, group_sizes_shape, ragged_dot_dnums,
          /*preferred_element_type=*/std::nullopt);
  ASSERT_IS_OK(inferred_shape_match.status());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, output_shape))
      << "inferred: " << ShapeUtil::HumanString(*inferred_shape_match)
      << " expected: " << ShapeUtil::HumanString(output_shape);
}

// 1D group_sizes will get broadcasted to the full shape
TEST_F(ShapeInferenceTest, RaggedDotMode1BroadcastedGroupSizes) {
  // [b1, b2, m1, m2, k]
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {64, 64, 32, 32, 128});
  // [g, b1, b2, k, n]
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {3, 64, 64, 128, 32});
  // [g]
  const Shape group_sizes_shape = ShapeUtil::MakeShape(U32, {3});
  // [b1, b2, m1, m2, n]
  const Shape output_shape = ShapeUtil::MakeShape(F32, {64, 64, 32, 32, 32});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_batch_dimensions(0);
  dot_dnums.add_lhs_batch_dimensions(1);
  dot_dnums.add_lhs_contracting_dimensions(4);
  dot_dnums.add_rhs_batch_dimensions(1);
  dot_dnums.add_rhs_batch_dimensions(2);
  dot_dnums.add_rhs_contracting_dimensions(3);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(3);
  ragged_dot_dnums.add_rhs_group_dimensions(0);

  const absl::StatusOr<Shape> inferred_shape_match =
      ShapeInference::InferRaggedDotOpShape(
          lhs_shape, rhs_shape, group_sizes_shape, ragged_dot_dnums,
          /*preferred_element_type=*/std::nullopt);
  ASSERT_IS_OK(inferred_shape_match.status());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, output_shape))
      << "inferred: " << ShapeUtil::HumanString(*inferred_shape_match)
      << " expected: " << ShapeUtil::HumanString(output_shape);
}

// ragged-dot rhs group dim should not be a batch dim
TEST_F(ShapeInferenceTest, RaggedDotRhsGroupDimIsBatch) {
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {19, 11, 5});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {19, 5, 7});
  const Shape group_sizes_shape = ShapeUtil::MakeShape(U32, {19, 3});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_batch_dimensions(0);
  dot_dnums.add_rhs_batch_dimensions(0);
  dot_dnums.add_lhs_contracting_dimensions(2);
  dot_dnums.add_rhs_contracting_dimensions(1);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(1);
  ragged_dot_dnums.add_rhs_group_dimensions(0);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferRaggedDotOpShape(
          lhs_shape, rhs_shape, group_sizes_shape, ragged_dot_dnums,
          /*preferred_element_type=*/std::nullopt);

  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(
      inferred_shape.status().message(),
      HasSubstr(
          "rhs group dimension cannot be a batch or contracting dimension"));
}

// ragged-dot rhs group dim should not be a contracting dim
TEST_F(ShapeInferenceTest, RaggedDotRhsGroupDimIsContracting) {
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {11, 3});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {3, 3, 7});
  const Shape group_sizes_shape = ShapeUtil::MakeShape(U32, {3});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(1);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(0);
  ragged_dot_dnums.add_rhs_group_dimensions(1);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferRaggedDotOpShape(
          lhs_shape, rhs_shape, group_sizes_shape, ragged_dot_dnums,
          /*preferred_element_type=*/std::nullopt);

  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(
      inferred_shape.status().message(),
      HasSubstr(
          "rhs group dimension cannot be a batch or contracting dimension"));
}

// ragged-dot group_sizes must have as many elements as the rhs group dim
TEST_F(ShapeInferenceTest, RaggedDotGroupSizesIncorrectShape) {
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {11, 5});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {3, 5, 7});
  const Shape group_sizes_shape = ShapeUtil::MakeShape(U32, {2});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(1);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(0);
  ragged_dot_dnums.add_rhs_group_dimensions(0);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferRaggedDotOpShape(
          lhs_shape, rhs_shape, group_sizes_shape, ragged_dot_dnums,
          /*preferred_element_type=*/std::nullopt);

  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(
      inferred_shape.status().message(),
      HasSubstr("rhs group dimension is expected to have size=2, got 3"));
}

// ragged-dot should have zero rhs group dims for ragged batch
TEST_F(ShapeInferenceTest, RaggedDotRhsGroupDimsForRaggedBatch) {
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {2, 11, 5});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {3, 2, 5, 7});
  const Shape group_sizes_shape = ShapeUtil::MakeShape(U32, {3});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_batch_dimensions(0);
  dot_dnums.add_rhs_batch_dimensions(1);
  dot_dnums.add_lhs_contracting_dimensions(2);
  dot_dnums.add_rhs_contracting_dimensions(2);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(0);
  ragged_dot_dnums.add_rhs_group_dimensions(0);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferRaggedDotOpShape(
          lhs_shape, rhs_shape, group_sizes_shape, ragged_dot_dnums,
          /*preferred_element_type=*/std::nullopt);

  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("There must be zero group dimensions in the rhs"));
}

// ragged-dot should have zero rhs group dims for ragged contracting
TEST_F(ShapeInferenceTest, RaggedDotRhsGroupDimsForRaggedContracting) {
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {11, 5});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {3, 5, 7});
  const Shape group_sizes_shape = ShapeUtil::MakeShape(U32, {3});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(1);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(1);
  ragged_dot_dnums.add_rhs_group_dimensions(0);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferRaggedDotOpShape(
          lhs_shape, rhs_shape, group_sizes_shape, ragged_dot_dnums,
          /*preferred_element_type=*/std::nullopt);

  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("There must be zero group dimensions in the rhs"));
}

// ragged-dot should have exactly one rhs group dim for ragged non-contracting
TEST_F(ShapeInferenceTest, RaggedDotRhsGroupDimsForRaggedNonContracting) {
  const Shape lhs_shape = ShapeUtil::MakeShape(F32, {11, 5});
  const Shape rhs_shape = ShapeUtil::MakeShape(F32, {5, 7});
  const Shape group_sizes_shape = ShapeUtil::MakeShape(U32, {3});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(0);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferRaggedDotOpShape(
          lhs_shape, rhs_shape, group_sizes_shape, ragged_dot_dnums,
          /*preferred_element_type=*/std::nullopt);

  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(
      inferred_shape.status().message(),
      HasSubstr("There must be exactly one group dimension in the rhs"));
}

TEST_F(ShapeInferenceTest, BinOpBroadcastMatrixVector) {
  // Test variations of broadcasting a vector for a binary add with a
  // matrix.
  const Shape mat = ShapeUtil::MakeShape(F32, {16, 8});
  const Shape vec8 = ShapeUtil::MakeShape(F32, {8});
  const Shape vec16 = ShapeUtil::MakeShape(F32, {16});

  absl::StatusOr<Shape> inferred_shape_match =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, mat, vec8, {1});
  ASSERT_IS_OK(inferred_shape_match.status());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, mat));

  absl::StatusOr<Shape> inferred_shape_mismatch =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, mat, vec8, {0});
  ASSERT_FALSE(inferred_shape_mismatch.ok());

  inferred_shape_match =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, mat, vec16, {0});
  ASSERT_IS_OK(inferred_shape_match.status());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, mat));

  inferred_shape_mismatch =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, mat, vec16, {1});
  ASSERT_FALSE(inferred_shape_mismatch.ok());
}

TEST_F(ShapeInferenceTest, BinOpBroadcastCubeMatrix) {
  // Test variations of broadcasting a matrix for a binary add with a cube.
  const Shape cube = ShapeUtil::MakeShape(F32, {16, 8, 4});
  const Shape matrix8_4 = ShapeUtil::MakeShape(F32, {8, 4});
  const Shape matrix16_4 = ShapeUtil::MakeShape(F32, {16, 4});
  const Shape matrix16_8 = ShapeUtil::MakeShape(F32, {16, 8});

  absl::StatusOr<Shape> inferred_shape_match =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, cube, matrix8_4,
                                         {1, 2});
  ASSERT_IS_OK(inferred_shape_match.status());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, cube));

  inferred_shape_match = ShapeInference::InferBinaryOpShape(
      HloOpcode::kAdd, cube, matrix16_4, {0, 2});
  ASSERT_IS_OK(inferred_shape_match.status());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, cube));

  inferred_shape_match = ShapeInference::InferBinaryOpShape(
      HloOpcode::kAdd, cube, matrix16_8, {0, 1});
  ASSERT_IS_OK(inferred_shape_match.status());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, cube));
}

TEST_F(ShapeInferenceTest, BinOpBroadcastBadDimension) {
  // Test various errors with the broadcast argument.
  const Shape tensor = ShapeUtil::MakeShape(F32, {16, 8, 4});
  const Shape tensor8_8_8 = ShapeUtil::MakeShape(F32, {8, 8, 8});
  const Shape vec8 = ShapeUtil::MakeShape(F32, {8});
  const Shape matrix8_4 = ShapeUtil::MakeShape(F32, {8, 4});
  const Shape matrix8_8 = ShapeUtil::MakeShape(F32, {8, 8});

  // "magical" broadcast rejected
  const absl::StatusOr<Shape> inferred_shape_error1 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor, vec8, {});
  ASSERT_FALSE(inferred_shape_error1.ok());
  ASSERT_THAT(inferred_shape_error1.status().message(),
              HasSubstr("Shapes must be equal rank"));

  // broadcast_dimension out of bounds for tensor's rank
  const absl::StatusOr<Shape> inferred_shape_error2 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor, vec8, {3});
  ASSERT_FALSE(inferred_shape_error2.ok());
  ASSERT_THAT(inferred_shape_error2.status().message(),
              ContainsRegex("Broadcast dimension number .* too large"));

  // broadcast_dimension doesn't match corresponding dimension
  const absl::StatusOr<Shape> inferred_shape_error3 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor, vec8, {0});
  ASSERT_FALSE(inferred_shape_error3.ok());
  ASSERT_THAT(inferred_shape_error3.status().message(),
              HasSubstr("Broadcast dimension 0 mismatch"));

  // broadcast_dimensions list too long
  const absl::StatusOr<Shape> inferred_shape_error4 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor, matrix8_4,
                                         {0, 1, 2});
  ASSERT_FALSE(inferred_shape_error4.ok());
  ASSERT_THAT(inferred_shape_error4.status().message(),
              HasSubstr("broadcast_dimensions has to match"));

  // there's a dimension above the rank of the tensor
  const absl::StatusOr<Shape> inferred_shape_error5 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor, matrix8_4,
                                         {3, 0});
  ASSERT_FALSE(inferred_shape_error5.ok());
  ASSERT_THAT(inferred_shape_error5.status().message(),
              ContainsRegex("dimension number .* too large"));

  // broadcasting dimensions don't match in this order
  const absl::StatusOr<Shape> inferred_shape_error6 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor, matrix8_4,
                                         {2, 1});
  ASSERT_FALSE(inferred_shape_error6.ok());
  ASSERT_THAT(inferred_shape_error6.status().message(),
              HasSubstr("dimension 0 mismatch"));

  // The following two tests make sure that broadcasting dimensions are listed
  // in a proper (strictly increasing) order, even if the lower-rank array
  // matches the higher-rank array in many different ways.
  const absl::StatusOr<Shape> inferred_shape_error7 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor8_8_8,
                                         matrix8_8, {0, 0});
  ASSERT_FALSE(inferred_shape_error7.ok());
  ASSERT_THAT(inferred_shape_error7.status().message(),
              HasSubstr("dimensions order is wrong"));

  const absl::StatusOr<Shape> inferred_shape_error8 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor8_8_8,
                                         matrix8_8, {1, 0});
  ASSERT_FALSE(inferred_shape_error8.ok());
  ASSERT_THAT(inferred_shape_error8.status().message(),
              HasSubstr("dimensions order is wrong"));
}

// Tests for the while instruction with proper shapes.
TEST_F(ShapeInferenceTest, WhileWithCorrectShapes) {
  const Shape result_shape = ShapeUtil::MakeTupleShape({s32_, vector_32_});
  ProgramShape cond = ShapeUtil::MakeProgramShape({result_shape}, pred_);
  ProgramShape body = ShapeUtil::MakeProgramShape({result_shape}, result_shape);
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferWhileShape(cond, body, result_shape);
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(ShapeUtil::Equal(result_shape, *inferred_shape));
}

// Tests for the while instruction with wrong shapes.
TEST_F(ShapeInferenceTest, WhileWithBadShapes) {
  const Shape inferred_shape = ShapeUtil::MakeTupleShape({s32_, vector_32_});
  ProgramShape cond = ShapeUtil::MakeProgramShape({inferred_shape}, pred_);
  ProgramShape body =
      ShapeUtil::MakeProgramShape({inferred_shape}, inferred_shape);

  const auto bad_shape_1 =
      ShapeUtil::MakeProgramShape({s32_, inferred_shape}, pred_);
  const absl::StatusOr<Shape> inferred_shape_error1 =
      ShapeInference::InferWhileShape(bad_shape_1, body, inferred_shape);
  ASSERT_FALSE(inferred_shape_error1.ok());
  ASSERT_THAT(inferred_shape_error1.status().message(),
              HasSubstr("Condition must take 1 arguments"));

  const auto bad_shape_2 =
      ShapeUtil::MakeProgramShape({s32_, inferred_shape}, inferred_shape);
  const absl::StatusOr<Shape> inferred_shape_error2 =
      ShapeInference::InferWhileShape(cond, bad_shape_2, inferred_shape);
  ASSERT_FALSE(inferred_shape_error2.ok());
  ASSERT_THAT(inferred_shape_error2.status().message(),
              HasSubstr("Body must take 1 arguments"));

  const auto bad_shape_3 = ShapeUtil::MakeProgramShape({inferred_shape}, s32_);
  const absl::StatusOr<Shape> inferred_shape_error3 =
      ShapeInference::InferWhileShape(bad_shape_3, body, inferred_shape);
  ASSERT_FALSE(inferred_shape_error3.ok());
  ASSERT_THAT(inferred_shape_error3.status().message(),
              HasSubstr("Condition must return a boolean"));

  const auto bad_shape_4 =
      ShapeUtil::MakeProgramShape({inferred_shape}, vector_32_);
  const absl::StatusOr<Shape> inferred_shape_error4 =
      ShapeInference::InferWhileShape(cond, bad_shape_4, inferred_shape);
  ASSERT_FALSE(inferred_shape_error4.ok());
  ASSERT_THAT(inferred_shape_error4.status().message(),
              HasSubstr("parameter of condition and body"));
}

// Tests for the concatenate instruction with dynamic shapes.
TEST_F(ShapeInferenceTest, ConcatenateWithDynamicShapes) {
  const auto dynamic_shape_1 =
      ShapeUtil::MakeShape(F32, {32, 160, 10}, {true, false, false});
  const auto dynamic_shape_2 =
      ShapeUtil::MakeShape(F32, {32, 160, 10}, {false, true, false});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferConcatOpShape({&dynamic_shape_1, &dynamic_shape_2},
                                         /*dimension=*/0);
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(ShapeUtil::Equal(
      ShapeUtil::MakeShape(F32, {64, 160, 10}, {true, true, false}),
      *inferred_shape));
}

// Tests for the concatenate instruction with proper shapes.
TEST_F(ShapeInferenceTest, ConcatenateWithCorrectShapes) {
  const absl::StatusOr<Shape> inferred_shape_1 =
      ShapeInference::InferConcatOpShape({&vector_32_, &vector_64_},
                                         /*dimension=*/0);
  ASSERT_IS_OK(inferred_shape_1.status());
  ASSERT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {96}), *inferred_shape_1));

  const absl::StatusOr<Shape> inferred_shape_2 =
      ShapeInference::InferConcatOpShape(
          {&vector_32_, &vector_64_, &vector_32_}, /*dimension=*/0);
  ASSERT_IS_OK(inferred_shape_2.status());
  ASSERT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {128}), *inferred_shape_2));

  const absl::StatusOr<Shape> inferred_shape_3 =
      ShapeInference::InferConcatOpShape(
          {&matrix_32_48_, &matrix_32_64_, &matrix_32_48_}, /*dimension=*/1);
  ASSERT_IS_OK(inferred_shape_3.status());
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {32, 160}),
                               *inferred_shape_3));
}

// Tests for the concatenate instruction with wrong shapes.
TEST_F(ShapeInferenceTest, ConcatenateWithBadShapes) {
  const absl::StatusOr<Shape> inferred_shape_error1 =
      ShapeInference::InferConcatOpShape({}, /*dimension=*/0);
  ASSERT_FALSE(inferred_shape_error1.ok());
  ASSERT_THAT(inferred_shape_error1.status().message(),
              HasSubstr("Concatenate expects at least one argument"));

  const absl::StatusOr<Shape> inferred_shape_error2 =
      ShapeInference::InferConcatOpShape({&vector_32_}, /*dimension=*/-1);
  ASSERT_FALSE(inferred_shape_error2.ok());
  ASSERT_THAT(inferred_shape_error2.status().message(),
              HasSubstr("dimension out of bounds: -1"));

  const absl::StatusOr<Shape> inferred_shape_error3 =
      ShapeInference::InferConcatOpShape({&vector_32_}, /*dimension=*/1);
  ASSERT_FALSE(inferred_shape_error3.ok());
  ASSERT_THAT(inferred_shape_error3.status().message(),
              HasSubstr("dimension out of bounds: 1"));

  const Shape tuple = ShapeUtil::MakeTupleShape({vector_32_});
  const absl::StatusOr<Shape> inferred_shape_error4 =
      ShapeInference::InferConcatOpShape({&vector_32_, &tuple},
                                         /*dimension=*/0);
  ASSERT_FALSE(inferred_shape_error4.ok());
  ASSERT_THAT(
      inferred_shape_error4.status().message(),
      HasSubstr("Expected array argument for operand of concatenation"));

  const Shape vector_s32 = ShapeUtil::MakeShape(S32, {32});
  const absl::StatusOr<Shape> inferred_shape_error5 =
      ShapeInference::InferConcatOpShape({&vector_32_, &vector_s32},
                                         /*dimension=*/0);
  ASSERT_FALSE(inferred_shape_error5.ok());
  ASSERT_THAT(inferred_shape_error5.status().message(),
              HasSubstr("concatenate arrays with different element types"));

  const absl::StatusOr<Shape> inferred_shape_error6 =
      ShapeInference::InferConcatOpShape({&matrix_32_48_, &matrix_32_64_},
                                         /*dimension=*/0);
  ASSERT_FALSE(inferred_shape_error6.ok());
  ASSERT_THAT(inferred_shape_error6.status().message(),
              HasSubstr("concatenate arrays that differ in "
                        "dimensions other than the one being "
                        "concatenated"));
}

TEST_F(ShapeInferenceTest, Pad) {
  const Shape input_shape = ShapeUtil::MakeShape(F32, {10, 25});
  const Shape padding_value_shape = ShapeUtil::MakeShape(F32, {});
  // Padding for dimension 0: {low: 0, high: 2, interior: 3}
  // Padding for dimension 1: {low: 1, high: 5, interior: 0}
  PaddingConfig padding_config;
  const auto dimension0 = padding_config.add_dimensions();
  dimension0->set_edge_padding_low(0);
  dimension0->set_edge_padding_high(2);
  dimension0->set_interior_padding(3);
  const auto dimension1 = padding_config.add_dimensions();
  dimension1->set_edge_padding_low(1);
  dimension1->set_edge_padding_high(5);
  dimension1->set_interior_padding(0);

  const absl::StatusOr<Shape> inferred_shape = ShapeInference::InferPadShape(
      input_shape, padding_value_shape, padding_config);
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {39, 31}), *inferred_shape));

  dimension1->set_edge_padding_low(-20);
  dimension1->set_edge_padding_high(-10);
  const auto negative_dimension_size = ShapeInference::InferPadShape(
      input_shape, padding_value_shape, padding_config);
  ASSERT_FALSE(negative_dimension_size.ok());
  ASSERT_THAT(negative_dimension_size.status().message(),
              HasSubstr("negative size for dimension 1"));
}

TEST_F(ShapeInferenceTest, Reverse) {
  const Shape input_shape = ShapeUtil::MakeShape(F32, {10, 25});

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferReverseShape(input_shape, {0, 1});
  ASSERT_IS_OK(inferred_shape.status());
  ASSERT_TRUE(ShapeUtil::Equal(input_shape, *inferred_shape));
}

TEST_F(ShapeInferenceTest, ReverseInvalidDimension) {
  const Shape input_shape = ShapeUtil::MakeShape(F32, {10, 25});

  const absl::StatusOr<Shape> inferred_shape_error0 =
      ShapeInference::InferReverseShape(input_shape, {0, 2});
  ASSERT_FALSE(inferred_shape_error0.ok());
  ASSERT_THAT(inferred_shape_error0.status().message(),
              HasSubstr("out-of-bounds"));

  const absl::StatusOr<Shape> inferred_shape_error1 =
      ShapeInference::InferReverseShape(input_shape, {0, -1});
  ASSERT_FALSE(inferred_shape_error1.ok());
  ASSERT_THAT(inferred_shape_error1.status().message(),
              HasSubstr("out-of-bounds"));

  const absl::StatusOr<Shape> inferred_shape_error2 =
      ShapeInference::InferReverseShape(input_shape, {0, 0});
  ASSERT_FALSE(inferred_shape_error2.ok());
  ASSERT_THAT(inferred_shape_error2.status().message(),
              HasSubstr("duplicated"));

  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({input_shape, input_shape});
  const absl::StatusOr<Shape> inferred_shape_error3 =
      ShapeInference::InferReverseShape(tuple_shape, {0});
  ASSERT_FALSE(inferred_shape_error3.ok());
  ASSERT_THAT(inferred_shape_error3.status().message(),
              HasSubstr("Expected array argument"));
}

TEST_F(ShapeInferenceTest, Call) {
  const absl::StatusOr<Shape> inferred_shape0 =
      ShapeInference::InferCallShape({}, ShapeUtil::MakeProgramShape({}, f32_));
  EXPECT_IS_OK(inferred_shape0.status());
  EXPECT_TRUE(ShapeUtil::Equal(f32_, *inferred_shape0));

  const absl::StatusOr<Shape> inferred_shape1 = ShapeInference::InferCallShape(
      {&f32_, &s32_, &pred_, &vector_32_, &matrix_32_48_},
      ShapeUtil::MakeProgramShape(
          {f32_, s32_, pred_, vector_32_, matrix_32_48_}, s32matrix_64_64_));
  EXPECT_IS_OK(inferred_shape1.status());
  EXPECT_TRUE(ShapeUtil::Equal(s32matrix_64_64_, *inferred_shape1));

  const absl::StatusOr<Shape> inferred_shape_error0 =
      ShapeInference::InferCallShape({},
                                     ShapeUtil::MakeProgramShape({f32_}, f32_));
  EXPECT_FALSE(inferred_shape_error0.ok());
  EXPECT_THAT(inferred_shape_error0.status().message(),
              HasSubstr("arity must match"));

  const absl::StatusOr<Shape> inferred_shape_error1 =
      ShapeInference::InferCallShape({&f32_},
                                     ShapeUtil::MakeProgramShape({}, f32_));
  EXPECT_FALSE(inferred_shape_error1.ok());
  EXPECT_THAT(inferred_shape_error1.status().message(),
              HasSubstr("arity must match"));

  const absl::StatusOr<Shape> inferred_shape_error2 =
      ShapeInference::InferCallShape({&f32_},
                                     ShapeUtil::MakeProgramShape({s32_}, f32_));
  EXPECT_FALSE(inferred_shape_error2.ok());
  EXPECT_THAT(inferred_shape_error2.status().message(),
              HasSubstr("parameter must match argument"));
}

TEST_F(ShapeInferenceTest, Transpose) {
  const Shape a_shape = ShapeUtil::MakeShape(F32, {2, 3, 4, 5});
  const absl::StatusOr<Shape> inferred_shape_and_status =
      ShapeInference::InferTransposeShape(a_shape, {1, 2, 3, 0});
  EXPECT_IS_OK(inferred_shape_and_status);
  EXPECT_TRUE(ShapeUtil::Compatible(ShapeUtil::MakeShape(F32, {3, 4, 5, 2}),
                                    *inferred_shape_and_status));
}

TEST_F(ShapeInferenceTest, Rank1Transpose) {
  const Shape a_shape = ShapeUtil::MakeShape(F32, {5});
  const absl::StatusOr<Shape> inferred_shape_and_status =
      ShapeInference::InferTransposeShape(a_shape, {0});
  EXPECT_IS_OK(inferred_shape_and_status);
  EXPECT_TRUE(ShapeUtil::Compatible(ShapeUtil::MakeShape(F32, {5}),
                                    *inferred_shape_and_status));
}

TEST_F(ShapeInferenceTest, ConditionalPred) {
  const absl::StatusOr<Shape> inferred_shape0 =
      ShapeInference::InferConditionalShape(
          pred_,
          {ShapeUtil::MakeProgramShape({vector_32_}, f32_),
           ShapeUtil::MakeProgramShape({vector_64_}, f32_)},
          {vector_32_, vector_64_});
  EXPECT_IS_OK(inferred_shape0.status());
  EXPECT_TRUE(ShapeUtil::Equal(f32_, *inferred_shape0));

  const absl::StatusOr<Shape> inferred_shape1 =
      ShapeInference::InferConditionalShape(
          pred_,
          {ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_64_),
           ShapeUtil::MakeProgramShape({vector_32_}, vector_64_)},
          {matrix_32_48_, vector_32_});
  EXPECT_IS_OK(inferred_shape1.status());
  EXPECT_TRUE(ShapeUtil::Equal(vector_64_, *inferred_shape1));

  const auto tuple_f32_v32 = ShapeUtil::MakeTupleShape({f32_, vector_32_});
  const absl::StatusOr<Shape> inferred_shape2 =
      ShapeInference::InferConditionalShape(
          pred_,
          {ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_32_),
           ShapeUtil::MakeProgramShape({tuple_f32_v32}, vector_32_)},
          {matrix_32_48_, tuple_f32_v32});
  EXPECT_IS_OK(inferred_shape2.status());
  EXPECT_TRUE(ShapeUtil::Equal(vector_32_, *inferred_shape2));

  const absl::StatusOr<Shape> inferred_shape_error0 =
      ShapeInference::InferConditionalShape(
          f32_,
          {ShapeUtil::MakeProgramShape({vector_32_}, f32_),
           ShapeUtil::MakeProgramShape({vector_64_}, f32_)},
          {vector_32_, vector_64_});
  EXPECT_FALSE(inferred_shape_error0.ok());
  EXPECT_THAT(inferred_shape_error0.status().message(),
              HasSubstr("must be bool or int32_t"));

  const absl::StatusOr<Shape> inferred_shape_error1 =
      ShapeInference::InferConditionalShape(
          pred_,
          {ShapeUtil::MakeProgramShape({f32_, vector_32_}, vector_32_),
           ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_32_)},
          {ShapeUtil::MakeTupleShape({f32_, vector_32_}), matrix_32_48_});
  EXPECT_FALSE(inferred_shape_error1.ok());
  EXPECT_THAT(inferred_shape_error1.status().message(),
              HasSubstr("branch computation 0 must take 1 argument"));

  const absl::StatusOr<Shape> inferred_shape_error2 =
      ShapeInference::InferConditionalShape(
          pred_,
          {ShapeUtil::MakeProgramShape({vector_64_}, f32_),
           ShapeUtil::MakeProgramShape({vector_64_}, f32_)},
          {vector_32_, vector_64_});
  EXPECT_FALSE(inferred_shape_error2.ok());
  EXPECT_THAT(inferred_shape_error2.status().message(),
              HasSubstr("branch operand 0 must match the shape of the only "
                        "parameter of branch computation 0"));

  const absl::StatusOr<Shape> inferred_shape_error3 =
      ShapeInference::InferConditionalShape(
          pred_,
          {ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_32_),
           ShapeUtil::MakeProgramShape({f32_, vector_32_}, vector_32_)},
          {matrix_32_48_, ShapeUtil::MakeTupleShape({f32_, vector_32_})});
  EXPECT_FALSE(inferred_shape_error3.ok());
  EXPECT_THAT(inferred_shape_error3.status().message(),
              HasSubstr("branch computation 1 must take 1 argument"));

  const absl::StatusOr<Shape> inferred_shape_error4 =
      ShapeInference::InferConditionalShape(
          pred_,
          {ShapeUtil::MakeProgramShape({vector_32_}, f32_),
           ShapeUtil::MakeProgramShape({vector_32_}, f32_)},
          {vector_32_, vector_64_});
  EXPECT_FALSE(inferred_shape_error4.ok());
  EXPECT_THAT(inferred_shape_error4.status().message(),
              HasSubstr("branch operand 1 must match the shape of the only "
                        "parameter of branch computation 1"));

  const absl::StatusOr<Shape> inferred_shape_error5 =
      ShapeInference::InferConditionalShape(
          pred_,
          {ShapeUtil::MakeProgramShape({vector_32_}, f32_),
           ShapeUtil::MakeProgramShape({vector_64_}, vector_32_)},
          {vector_32_, vector_64_});
  EXPECT_FALSE(inferred_shape_error5.ok());
  EXPECT_THAT(inferred_shape_error5.status().message(),
              HasSubstr("the result of branch 0 computation and branch 1 "
                        "computation must have the same shape"));
}

TEST_F(ShapeInferenceTest, ConditionalIndexed) {
  const Shape r0s32 = ShapeUtil::MakeShape(S32, {});
  const absl::StatusOr<Shape> inferred_shape0 =
      ShapeInference::InferConditionalShape(
          r0s32,
          {ShapeUtil::MakeProgramShape({vector_32_}, f32_),
           ShapeUtil::MakeProgramShape({vector_64_}, f32_),
           ShapeUtil::MakeProgramShape({vector_64_}, f32_)},
          {vector_32_, vector_64_, vector_64_});
  EXPECT_IS_OK(inferred_shape0.status());
  EXPECT_TRUE(ShapeUtil::Equal(f32_, *inferred_shape0));

  const absl::StatusOr<Shape> inferred_shape1 =
      ShapeInference::InferConditionalShape(
          r0s32,
          {ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_64_),
           ShapeUtil::MakeProgramShape({vector_32_}, vector_64_),
           ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_64_)},
          {matrix_32_48_, vector_32_, matrix_32_48_});
  EXPECT_IS_OK(inferred_shape1.status());
  EXPECT_TRUE(ShapeUtil::Equal(vector_64_, *inferred_shape1));

  const auto tuple_f32_v32 = ShapeUtil::MakeTupleShape({f32_, vector_32_});
  const absl::StatusOr<Shape> inferred_shape2 =
      ShapeInference::InferConditionalShape(
          r0s32, {ShapeUtil::MakeProgramShape({tuple_f32_v32}, vector_32_)},
          {tuple_f32_v32});
  EXPECT_IS_OK(inferred_shape2.status());
  EXPECT_TRUE(ShapeUtil::Equal(vector_32_, *inferred_shape2));

  const absl::StatusOr<Shape> inferred_shape_error0 =
      ShapeInference::InferConditionalShape(
          pred_,
          {ShapeUtil::MakeProgramShape({vector_32_}, f32_),
           ShapeUtil::MakeProgramShape({vector_32_}, f32_),
           ShapeUtil::MakeProgramShape({vector_64_}, f32_)},
          {vector_32_, vector_32_, vector_64_});
  EXPECT_FALSE(inferred_shape_error0.ok());
  EXPECT_THAT(inferred_shape_error0.status().message(),
              HasSubstr("2 == branch_computations.size()"));

  const absl::StatusOr<Shape> inferred_shape_error1 =
      ShapeInference::InferConditionalShape(
          r0s32,
          {ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_32_),
           ShapeUtil::MakeProgramShape({f32_, vector_32_}, vector_32_),
           ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_32_)},
          {matrix_32_48_, ShapeUtil::MakeTupleShape({f32_, vector_32_}),
           matrix_32_48_});
  EXPECT_FALSE(inferred_shape_error1.ok());
  EXPECT_THAT(inferred_shape_error1.status().message(),
              HasSubstr("branch computation 1 must take 1 argument"));

  const absl::StatusOr<Shape> inferred_shape_error2 =
      ShapeInference::InferConditionalShape(
          r0s32,
          {ShapeUtil::MakeProgramShape({r0s32}, f32_),
           ShapeUtil::MakeProgramShape({vector_32_}, f32_),
           ShapeUtil::MakeProgramShape({vector_32_}, f32_)},
          {r0s32, vector_32_, vector_64_});
  EXPECT_FALSE(inferred_shape_error2.ok());
  EXPECT_THAT(inferred_shape_error2.status().message(),
              HasSubstr("branch operand 2 must match the shape of the only "
                        "parameter of branch computation 2"));

  const absl::StatusOr<Shape> inferred_shape_error3 =
      ShapeInference::InferConditionalShape(
          r0s32,
          {ShapeUtil::MakeProgramShape({vector_32_}, f32_),
           ShapeUtil::MakeProgramShape({vector_32_}, f32_),
           ShapeUtil::MakeProgramShape({vector_32_}, f32_),
           ShapeUtil::MakeProgramShape({vector_64_}, vector_32_)},
          {vector_32_, vector_32_, vector_32_, vector_64_});
  EXPECT_FALSE(inferred_shape_error3.ok());
  EXPECT_THAT(inferred_shape_error3.status().message(),
              HasSubstr("the result of branch 0 computation and branch 3 "
                        "computation must have the same shape"));

  const absl::StatusOr<Shape> inferred_shape_error4 =
      ShapeInference::InferConditionalShape(r0s32, {}, {});
  EXPECT_FALSE(inferred_shape_error4.ok());
  EXPECT_THAT(inferred_shape_error4.status().message(),
              HasSubstr("!branch_computations.empty()"));
}

TEST_F(ShapeInferenceTest, ConditionalDynamic) {
  const Shape r0s32 = ShapeUtil::MakeShape(S32, {});
  const Shape static_shape = ShapeUtil::MakeShape(S32, {4}, {false});
  const Shape dynamic_shape = ShapeUtil::MakeShape(S32, {4}, {true});
  const absl::StatusOr<Shape> inferred_shape0 =
      ShapeInference::InferConditionalShape(
          r0s32,
          {ShapeUtil::MakeProgramShape({vector_32_}, static_shape),
           ShapeUtil::MakeProgramShape({vector_64_}, dynamic_shape),
           ShapeUtil::MakeProgramShape({vector_64_}, dynamic_shape)},
          {vector_32_, vector_64_, vector_64_});
  EXPECT_IS_OK(inferred_shape0.status());
  EXPECT_TRUE(ShapeUtil::Equal(dynamic_shape, *inferred_shape0));

  const absl::StatusOr<Shape> inferred_shape1 =
      ShapeInference::InferConditionalShape(
          r0s32,
          {ShapeUtil::MakeProgramShape({vector_32_}, dynamic_shape),
           ShapeUtil::MakeProgramShape({vector_64_}, static_shape),
           ShapeUtil::MakeProgramShape({vector_64_}, dynamic_shape)},
          {vector_32_, vector_64_, vector_64_});
  EXPECT_IS_OK(inferred_shape1.status());
  EXPECT_TRUE(ShapeUtil::Equal(dynamic_shape, *inferred_shape1));
}

TEST_F(ShapeInferenceTest, BadSlice) {
  const Shape arg = ShapeUtil::MakeShape(F32, {4});
  const absl::StatusOr<Shape> statusor =
      ShapeInference::InferSliceShape(arg, {0}, {5}, {1});
  ASSERT_FALSE(statusor.ok());

  LOG(INFO) << statusor.status();

  EXPECT_THAT(statusor.status().message(),
              HasSubstr("less than or equal to dimension size"))
      << statusor.status();
  EXPECT_THAT(statusor.status().message(), HasSubstr("argument shape"))
      << statusor.status();
}

TEST_F(ShapeInferenceTest, BadSort) {
  const Shape keys = ShapeUtil::MakeShape(F32, {4});
  const Shape values = ShapeUtil::MakeShape(F32, {5});
  const absl::StatusOr<Shape> statusor =
      ShapeInference::InferVariadicOpShape(HloOpcode::kSort, {&keys, &values});
  EXPECT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(), HasSubstr("dimensions must match"))
      << statusor.status();
}

TEST_F(ShapeInferenceTest, BadSortValuesMismatch) {
  const Shape keys = ShapeUtil::MakeShape(F32, {4});
  const Shape values_good = ShapeUtil::MakeShape(F32, {4});
  const Shape values_bad = ShapeUtil::MakeShape(F32, {5});
  const absl::StatusOr<Shape> statusor = ShapeInference::InferVariadicOpShape(
      HloOpcode::kSort, {&keys, &values_good, &values_bad});
  EXPECT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(), HasSubstr("dimensions must match"))
      << statusor.status();
}

TEST_F(ShapeInferenceTest, SortManyValues) {
  const Shape keys = ShapeUtil::MakeShape(F32, {4});
  const Shape values_s32 = ShapeUtil::MakeShape(S32, {4});
  const Shape values_u32 = ShapeUtil::MakeShape(U32, {4});
  const absl::StatusOr<Shape> statusor = ShapeInference::InferVariadicOpShape(
      HloOpcode::kSort, {&keys, &values_s32, &values_u32});
  EXPECT_IS_OK(statusor);
  const Shape inferred_shape = *statusor;
  EXPECT_TRUE(ShapeUtil::Compatible(
      inferred_shape,
      ShapeUtil::MakeTupleShape({keys, values_s32, values_u32})));
}

TEST_F(ShapeInferenceTest, GoodTopK) {
  const Shape input = ShapeUtil::MakeShape(F32, {3, 4, 5});
  const absl::StatusOr<Shape> s =
      ShapeInference::InferTopKShape(input, /*k=*/2);
  ASSERT_IS_OK(s.status());
  ASSERT_TRUE(ShapeUtil::Equal(
      *s, ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {3, 4, 2}),
                                     ShapeUtil::MakeShape(S32, {3, 4, 2})})));
}

TEST_F(ShapeInferenceTest, FailTopKLargeK) {
  const Shape input = ShapeUtil::MakeShape(F32, {3, 4, 5});
  const absl::StatusOr<Shape> statusor =
      ShapeInference::InferTopKShape(input, /*k=*/10);
  EXPECT_FALSE(statusor.ok());
}

TEST_F(ShapeInferenceTest, InferStochasticConvertShape) {
  const Shape operand = ShapeUtil::MakeShape(F32, {4, 3});
  const Shape random = ShapeUtil::MakeShape(U32, {4, 3});
  const Shape expected_shape = ShapeUtil::MakeShape(S8, {4, 3});

  const absl::StatusOr<Shape> inferred_sr_shape =
      ShapeInference::InferStochasticConvertShape(operand, random, S8);
  EXPECT_TRUE(inferred_sr_shape.ok());
  EXPECT_TRUE(ShapeUtil::Equal(*inferred_sr_shape, expected_shape));
}

TEST_F(ShapeInferenceTest, InvalidStochasticConvert_MismatchRandomElementType) {
  const Shape operand = ShapeUtil::MakeShape(F32, {4, 3});
  const Shape random = ShapeUtil::MakeShape(U16, {4, 3});
  const Shape expected_shape = ShapeUtil::MakeShape(S8, {4, 3});

  const auto status_or =
      ShapeInference::InferStochasticConvertShape(operand, random, S8);
  ASSERT_FALSE(status_or.ok());
  EXPECT_THAT(
      status_or.status().message(),
      HasSubstr(
          "The random number is required to have same bits as the operand."));
}

TEST_F(ShapeInferenceTest,
       InvalidStochasticConvert_DisallowedRandomElementType) {
  const Shape operand = ShapeUtil::MakeShape(F32, {4, 3});
  const Shape random = ShapeUtil::MakeShape(S32, {4, 3});
  const Shape expected_shape = ShapeUtil::MakeShape(S8, {4, 3});

  const auto status_or =
      ShapeInference::InferStochasticConvertShape(operand, random, S8);
  ASSERT_FALSE(status_or.ok());
  EXPECT_THAT(
      status_or.status().message(),
      HasSubstr(
          "Random numbers for stochastic convert must be unsigned integers"));
}

class GatherShapeInferenceTest : public ShapeInferenceTest {
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
};

TEST_F(GatherShapeInferenceTest, TensorFlowGather) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape gather_shape,
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

TEST_F(GatherShapeInferenceTest, TensorFlowGatherV2) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape gather_shape,
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

TEST_F(GatherShapeInferenceTest, TensorFlowGatherBatchingDims) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape gather_shape,
                          ShapeInference::InferGatherShape(
                              ShapeUtil::MakeShape(F32, {100, 64, 5, 48}),
                              ShapeUtil::MakeShape(S64, {5, 100, 32}),
                              HloGatherInstruction::MakeGatherDimNumbers(
                                  /*offset_dims=*/{3},
                                  /*collapsed_slice_dims=*/{1},
                                  /*start_index_map=*/{1},
                                  /*index_vector_dim=*/3,
                                  /*operand_batching_dims=*/{0, 2},
                                  /*start_indices_batching_dims=*/{1, 0}),
                              /*slice_sizes=*/{1, 1, 1, 8}));
  EXPECT_TRUE(ShapeUtil::Equal(gather_shape,
                               ShapeUtil::MakeShape(F32, {5, 100, 32, 8})))
      << ShapeUtil::HumanString(gather_shape);
}

TEST_F(GatherShapeInferenceTest, TensorFlowGatherNd) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape gather_shape,
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

TEST_F(GatherShapeInferenceTest, TensorFlowBatchDynamicSlice) {
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape gather_shape,
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

TEST_F(GatherShapeInferenceTest, DynamicGatherEntireDimension) {
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape gather_shape,
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

TEST_F(GatherShapeInferenceTest, DynamicGatherCollapsedDimension) {
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape gather_shape,
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

TEST_F(GatherShapeInferenceTest, DynamicIndices) {
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape gather_shape,
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

TEST_F(GatherShapeInferenceTest, NonDefaultGatherIndicesLeafDim_A) {
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape gather_shape,
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

TEST_F(GatherShapeInferenceTest, NonDefaultGatherIndicesLeafDim_B) {
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape gather_shape,
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

TEST_F(GatherShapeInferenceTest, NoOutputGatherDims) {
  // This is equivalent to a dynamic slice.
  TF_ASSERT_OK_AND_ASSIGN(const Shape gather_shape,
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

TEST_F(GatherShapeInferenceTest, ScalarGatherIndices) {
  // The gather indices "tensor" is a scalar S here that's used to slice out
  // [S,0,0,0,0]..[S,30,29,28,27] into a [30,29,28,27] shaped result.
  TF_ASSERT_OK_AND_ASSIGN(const Shape gather_shape,
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

TEST_F(GatherShapeInferenceTest, TupleShapedTensorInput) {
  const absl::StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      tuple_shape_, s64_vector_32_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{0},
          /*collapsed_slice_dims=*/{1},
          /*start_index_map=*/{1},
          /*index_vector_dim=*/1),
      /*slice_sizes=*/{64, 1});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Expected array argument for input"))
      << statusor.status();
}

TEST_F(GatherShapeInferenceTest, TupleShapedGatherIndicesInput) {
  const absl::StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      s64_vector_32_, tuple_shape_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{0},
          /*collapsed_slice_dims=*/{1},
          /*start_index_map=*/{1},
          /*index_vector_dim=*/0),
      /*slice_sizes=*/{64, 1});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Expected array argument for gather indices"))
      << statusor.status();
}

TEST_F(GatherShapeInferenceTest, FloatingPointGatherIndicesInput) {
  const absl::StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      s64_vector_32_, vector_32_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{0},
          /*collapsed_slice_dims=*/{1},
          /*start_index_map=*/{1},
          /*index_vector_dim=*/0),
      /*slice_sizes=*/{64, 1});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Gather indices parameter must be an integral tensor"))
      << statusor.status();
}

TEST_F(GatherShapeInferenceTest,
       InvalidGatherDimNumbers_NonAscendingWindowIndices) {
  const absl::StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 8, 7},
          /*collapsed_slice_dims=*/{},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 27, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr("Output window dimensions in gather op must be ascending"))
      << statusor.status();
}

TEST_F(GatherShapeInferenceTest,
       InvalidGatherDimNumbers_RepeatedWindowIndices) {
  const absl::StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7, 7},
          /*collapsed_slice_dims=*/{},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 27, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr("Output window dimensions in gather op must not repeat"))
      << statusor.status();
}

TEST_F(GatherShapeInferenceTest,
       InvalidGatherDimNumbers_WindowIndexOutOfBounds) {
  const absl::StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 99, 100, 101},
          /*collapsed_slice_dims=*/{},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 27, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Offset dimension 2 in gather op is out of bounds"))
      << statusor.status();
}

TEST_F(GatherShapeInferenceTest,
       InvalidGatherDimNumbers_WindowIndexBarelyOutOfBounds) {
  const absl::StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7, 9},
          /*collapsed_slice_dims=*/{},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 27, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Offset dimension 4 in gather op is out of bounds"))
      << statusor.status();
}

TEST_F(GatherShapeInferenceTest,
       InvalidGatherDimNumbers_MismatchingElidedWindowDims) {
  const absl::StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7, 8},
          /*collapsed_slice_dims=*/{4},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 27, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr("All components of the offset index in a gather op must either "
                "be a offset dimension or explicitly collapsed"))
      << statusor.status();
}

TEST_F(GatherShapeInferenceTest,
       InvalidGatherDimNumbers_OutOfBoundsWindowToInputMapping) {
  const absl::StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7, 8},
          /*collapsed_slice_dims=*/{0, 1, 2, 3, 19},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 27, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Invalid collapsed_slice_dims set in gather op; valid "
                        "range is [0, 5), got: 19"))
      << statusor.status();
}

TEST_F(GatherShapeInferenceTest,
       InvalidGatherDimNumbers_RepeatedWindowToInputMapping) {
  const absl::StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7, 8},
          /*collapsed_slice_dims=*/{0, 1, 2, 3, 3},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 27, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Repeated dimensions not allowed in "
                        "collapsed_slice_dims in gather op"))
      << statusor.status();
}

TEST_F(GatherShapeInferenceTest,
       InvalidGatherDimNumbers_MismatchingGatherToInputMapping) {
  const absl::StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7, 8},
          /*collapsed_slice_dims=*/{},
          /*start_index_map=*/{0, 1, 2, 3},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 27, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Gather op has 4 elements in start_index_map and "
                        "the bound of dimension index_vector_dim=4 of "
                        "start_indices is 5. These two numbers must be equal."))
      << statusor.status();
}

TEST_F(GatherShapeInferenceTest,
       InvalidGatherDimNumbers_OutOfBoundsGatherToInputMapping) {
  const absl::StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7, 8},
          /*collapsed_slice_dims=*/{},
          /*start_index_map=*/{0, 1, 2, 3, 7},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 27, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Invalid start_index_map; domain is [0, 5), got: 4->7"))
      << statusor.status();
}

TEST_F(GatherShapeInferenceTest,
       InvalidGatherDimNumbers_RepeatedGatherToInputMapping) {
  const absl::StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7, 8},
          /*collapsed_slice_dims=*/{},
          /*start_index_map=*/{0, 1, 2, 3, 3},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 27, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr("Repeated dimensions are not allowed in start_index_map"))
      << statusor.status();
}

TEST_F(GatherShapeInferenceTest,
       InvalidGatherDimNumbers_NonAscendingElidedWindowDims) {
  const absl::StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7, 8},
          /*collapsed_slice_dims=*/{2, 1},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{1, 1, 28, 27, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("collapsed_slice_dims in gather op must be sorted"))
      << statusor.status();
}

TEST_F(GatherShapeInferenceTest, InvalidGatherDimNumbers_WindowBoundsTooLarge) {
  const absl::StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7},
          /*collapsed_slice_dims=*/{2},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 1, 300, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Slice size at index 3 in gather op is out of range, "
                        "must be within [0, 48), got 300."))
      << statusor.status();
}

TEST_F(GatherShapeInferenceTest,
       InvalidGatherDimNumbers_MismatchingNumberOfWindowBounds) {
  const absl::StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7, 8},
          /*collapsed_slice_dims=*/{},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 26});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr("Gather op must have one slice size for every input dimension"))
      << statusor.status();
}

TEST_F(GatherShapeInferenceTest,
       InvalidGatherDimNumbers_WindowBoundsNot1ForElidedDim) {
  const absl::StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_8_7_5_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7},
          /*collapsed_slice_dims=*/{1},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4),
      /*slice_sizes=*/{30, 29, 28, 26, 20});
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr("Gather op can only collapse slice dims with bound 1 or 0, "
                "but bound is 29 for index 1 at position 0."))
      << statusor.status();
}

TEST_F(GatherShapeInferenceTest, OutOfBoundsGatherIndicesLeafDim) {
  const absl::StatusOr<Shape> statusor = ShapeInference::InferGatherShape(
      f32_5d_tensor_50_49_48_47_46_, s64_4d_tensor_10_9_5_7_6_,
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{4, 5, 6, 7, 8},
          /*collapsed_slice_dims=*/{},
          /*start_index_map=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/32),
      /*slice_sizes=*/{30, 29, 28, 27, 26});

  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Gather index leaf dimension must be within [0, "
                        "rank(start_indices) + 1)"))
      << statusor.status();
}

class ScatterShapeInferenceTest
    : public ShapeInferenceTest,
      public ::testing::WithParamInterface<std::vector<PrimitiveType>> {
 protected:
  struct ScatterShapes {
    void Add(Shape shape) {
      shapes.push_back(std::move(shape));
      ptrs.push_back(&shapes.back());
    }
    std::vector<Shape> shapes;
    std::vector<const Shape*> ptrs;
  };
  static ScatterShapes CreateShapes(absl::Span<const int64_t> operand_dims,
                                    const Shape& scatter_indices_shape,
                                    absl::Span<const int64_t> update_dims,
                                    absl::Span<const PrimitiveType> types) {
    CHECK(!types.empty());
    size_t size = types.size() * 2 + 1;
    ScatterShapes shapes;
    shapes.shapes.reserve(size);
    shapes.ptrs.reserve(size);
    for (PrimitiveType type : types) {
      shapes.Add(ShapeUtil::MakeShape(type, operand_dims));
    }
    shapes.Add(scatter_indices_shape);
    for (PrimitiveType type : types) {
      shapes.Add(ShapeUtil::MakeShape(type, update_dims));
    }
    return shapes;
  }
  static Shape Collate(absl::Span<const int64_t> dims,
                       absl::Span<const PrimitiveType> types) {
    CHECK(!types.empty());
    if (types.size() == 1) {
      return ShapeUtil::MakeShape(types[0], dims);
    }
    std::vector<Shape> shapes;
    for (PrimitiveType type : types) {
      shapes.push_back(ShapeUtil::MakeShape(type, dims));
    }
    return ShapeUtil::MakeTupleShape(shapes);
  }
  static Shape scalar(PrimitiveType type) {
    return ShapeUtil::MakeShape(type, {});
  }
  static Shape s64_vector(int dim) { return ShapeUtil::MakeShape(S64, {dim}); }
  static Shape s64_tensor(absl::Span<const int64_t> dims) {
    return ShapeUtil::MakeShape(S64, dims);
  }
  static ProgramShape to_apply(absl::Span<const PrimitiveType> types) {
    CHECK(!types.empty());
    ProgramShape program_shape;
    Shape& result = *program_shape.mutable_result();
    result = ShapeUtil::MakeNil();
    result.mutable_tuple_shapes()->reserve(types.size());
    program_shape.mutable_parameters()->reserve(types.size() * 2);
    for (PrimitiveType type : types) {
      *program_shape.add_parameters() = scalar(type);
      *result.add_tuple_shapes() = scalar(type);
    }
    for (PrimitiveType type : types) {
      *program_shape.add_parameters() = scalar(type);
    }
    return program_shape;
  }
  std::vector<PrimitiveType> types() const { return GetParam(); }
};

TEST_P(ScatterShapeInferenceTest, TfScatterWithFullUpdates) {
  const auto shapes = CreateShapes({64, 48}, s64_vector(32), {64, 32}, types());
  TF_ASSERT_OK_AND_ASSIGN(const Shape scatter_shape,
                          ShapeInference::InferScatterShape(
                              shapes.ptrs, to_apply(types()),
                              HloScatterInstruction::MakeScatterDimNumbers(
                                  /*update_window_dims=*/{0},
                                  /*inserted_window_dims=*/{1},
                                  /*scatter_dims_to_operand_dims=*/{1},
                                  /*index_vector_dim=*/1)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, Collate({64, 48}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, TfScatterWithFullUpdatesV2) {
  const auto shapes = CreateShapes({64, 48}, s64_vector(32), {32, 48}, types());
  TF_ASSERT_OK_AND_ASSIGN(const Shape scatter_shape,
                          ShapeInference::InferScatterShape(
                              shapes.ptrs, to_apply(types()),
                              HloScatterInstruction::MakeScatterDimNumbers(
                                  /*update_window_dims=*/{1},
                                  /*inserted_window_dims=*/{0},
                                  /*scatter_dims_to_operand_dims=*/{0},
                                  /*index_vector_dim=*/1)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, Collate({64, 48}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, TfScatterWithPartialUpdates) {
  const auto shapes = CreateShapes({64, 48}, s64_vector(32), {10, 32}, types());
  TF_ASSERT_OK_AND_ASSIGN(const Shape scatter_shape,
                          ShapeInference::InferScatterShape(
                              shapes.ptrs, to_apply(types()),
                              HloScatterInstruction::MakeScatterDimNumbers(
                                  /*update_window_dims=*/{0},
                                  /*inserted_window_dims=*/{1},
                                  /*scatter_dims_to_operand_dims=*/{1},
                                  /*index_vector_dim=*/1)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, Collate({64, 48}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, TfScatterWithPartialUpdatesV2) {
  const auto shapes = CreateShapes({64, 48}, s64_vector(32), {32, 8}, types());
  TF_ASSERT_OK_AND_ASSIGN(const Shape scatter_shape,
                          ShapeInference::InferScatterShape(
                              shapes.ptrs, to_apply(types()),
                              HloScatterInstruction::MakeScatterDimNumbers(
                                  /*update_window_dims=*/{1},
                                  /*inserted_window_dims=*/{0},
                                  /*scatter_dims_to_operand_dims=*/{0},
                                  /*index_vector_dim=*/1)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, Collate({64, 48}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, TfScatterWithUpdatesBiggerThanInput) {
  const auto shapes = CreateShapes({64, 48}, s64_vector(32), {65, 32}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{0},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/1));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr("Bounds of the window dimensions of updates must not exceed "
                "the bounds of the corresponding dimensions of operand."))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, TfScatterWithUpdatesBiggerThanInputV2) {
  const auto shapes = CreateShapes({64, 48}, s64_vector(32), {32, 49}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{1},
          /*inserted_window_dims=*/{0},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/1));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr("Bounds of the window dimensions of updates must not exceed "
                "the bounds of the corresponding dimensions of operand."))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest,
       TfScatterBatchingDimsWithUpdatesBiggerThanInput) {
  const auto shapes = CreateShapes({100, 64, 48}, s64_tensor({100, 32}),
                                   {100, 65, 32}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{1},
          /*inserted_window_dims=*/{2},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/2,
          /*input_batching_dims=*/{0},
          /*scatter_indices_batching_dims=*/{0}));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr("Bounds of the window dimensions of updates must not exceed "
                "the bounds of the corresponding dimensions of operand."))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, TfScatterWithUpdatesNotMatchingIndices) {
  const auto shapes = CreateShapes({64, 48}, s64_vector(32), {64, 31}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{0},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/1));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr(
          "Bounds of the scatter dimensions of updates must be same as the "
          "bounds of the corresponding dimensions of scatter indices."))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, TfScatterWithUpdatesNotMatchingIndicesV2) {
  const auto shapes = CreateShapes({64, 48}, s64_vector(32), {31, 48}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{1},
          /*inserted_window_dims=*/{0},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/1));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr(
          "Bounds of the scatter dimensions of updates must be same as the "
          "bounds of the corresponding dimensions of scatter indices."))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, TfScatterNdWithFullUpdates) {
  const auto shapes = CreateShapes({64, 48}, s64_tensor({10, 9, 8, 7, 1}),
                                   {10, 9, 8, 7, 48}, types());
  TF_ASSERT_OK_AND_ASSIGN(const Shape scatter_shape,
                          ShapeInference::InferScatterShape(
                              shapes.ptrs, to_apply(types()),
                              HloScatterInstruction::MakeScatterDimNumbers(
                                  /*update_window_dims=*/{4},
                                  /*inserted_window_dims=*/{0},
                                  /*scatter_dims_to_operand_dims=*/{0},
                                  /*index_vector_dim=*/4)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, Collate({64, 48}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, TfScatterNdWithFullUpdatesV2) {
  const auto shapes = CreateShapes({64, 48}, s64_tensor({10, 9, 8, 7, 1}),
                                   {10, 9, 8, 7, 64}, types());
  TF_ASSERT_OK_AND_ASSIGN(const Shape scatter_shape,
                          ShapeInference::InferScatterShape(
                              shapes.ptrs, to_apply(types()),
                              HloScatterInstruction::MakeScatterDimNumbers(
                                  /*update_window_dims=*/{4},
                                  /*inserted_window_dims=*/{1},
                                  /*scatter_dims_to_operand_dims=*/{0},
                                  /*index_vector_dim=*/4)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, Collate({64, 48}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, TfScatterNdWithPartialUpdates) {
  const auto shapes = CreateShapes({64, 48}, s64_tensor({10, 9, 8, 7, 1}),
                                   {10, 9, 8, 7, 10}, types());
  TF_ASSERT_OK_AND_ASSIGN(const Shape scatter_shape,
                          ShapeInference::InferScatterShape(
                              shapes.ptrs, to_apply(types()),
                              HloScatterInstruction::MakeScatterDimNumbers(
                                  /*update_window_dims=*/{4},
                                  /*inserted_window_dims=*/{0},
                                  /*scatter_dims_to_operand_dims=*/{0},
                                  /*index_vector_dim=*/4)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, Collate({64, 48}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, TfScatterNdWithPartialUpdatesV2) {
  const auto shapes = CreateShapes({64, 48}, s64_tensor({10, 9, 8, 7, 1}),
                                   {10, 9, 8, 7, 12}, types());
  TF_ASSERT_OK_AND_ASSIGN(const Shape scatter_shape,
                          ShapeInference::InferScatterShape(
                              shapes.ptrs, to_apply(types()),
                              HloScatterInstruction::MakeScatterDimNumbers(
                                  /*update_window_dims=*/{4},
                                  /*inserted_window_dims=*/{1},
                                  /*scatter_dims_to_operand_dims=*/{0},
                                  /*index_vector_dim=*/4)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, Collate({64, 48}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, TfScatterNdWithUpdatesBiggerThanInput) {
  const auto shapes = CreateShapes({64, 48}, s64_tensor({10, 9, 8, 7, 1}),
                                   {10, 9, 8, 7, 65}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{0},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr("Bounds of the window dimensions of updates must not exceed "
                "the bounds of the corresponding dimensions of operand."))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, TfScatterNdWithUpdatesNotMatchingIndices) {
  const auto shapes = CreateShapes({64, 48}, s64_tensor({10, 9, 8, 7, 1}),
                                   {9, 9, 8, 7, 64}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{0},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr(
          "Bounds of the scatter dimensions of updates must be same as the "
          "bounds of the corresponding dimensions of scatter indices."))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, TfBatchDynamicUpdateSlice) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28, 27, 26}, types());
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape scatter_shape,
      ShapeInference::InferScatterShape(
          shapes.ptrs, to_apply(types()),
          HloScatterInstruction::MakeScatterDimNumbers(
              /*update_window_dims=*/{4, 5, 6, 7, 8},
              /*inserted_window_dims=*/{},
              /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
              /*index_vector_dim=*/4)));
  EXPECT_TRUE(
      ShapeUtil::Equal(scatter_shape, Collate({50, 49, 48, 47, 46}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, NonDefaultScatterIndicesLeafDim) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 5, 7, 6}),
                   {10, 9, 7, 6, 30, 29, 28, 27, 26}, types());
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape scatter_shape,
      ShapeInference::InferScatterShape(
          shapes.ptrs, to_apply(types()),
          HloScatterInstruction::MakeScatterDimNumbers(
              /*update_window_dims=*/{4, 5, 6, 7, 8},
              /*inserted_window_dims=*/{},
              /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
              /*index_vector_dim=*/2)));

  EXPECT_TRUE(
      ShapeUtil::Equal(scatter_shape, Collate({50, 49, 48, 47, 46}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, NonDefaultScatterIndicesLeafDimV2) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({5, 10, 9, 7, 6}),
                   {10, 9, 7, 6, 30, 29, 28, 27, 26}, types());
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape scatter_shape,
      ShapeInference::InferScatterShape(
          shapes.ptrs, to_apply(types()),
          HloScatterInstruction::MakeScatterDimNumbers(
              /*update_window_dims=*/{4, 5, 6, 7, 8},
              /*inserted_window_dims=*/{},
              /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
              /*index_vector_dim=*/0)));

  EXPECT_TRUE(
      ShapeUtil::Equal(scatter_shape, Collate({50, 49, 48, 47, 46}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, NoUpdateScatterDims) {
  const auto shapes = CreateShapes({50, 49, 48, 47, 46}, s64_vector(5),
                                   {30, 29, 28, 27, 26}, types());
  // This is equivalent to a dynamic update slice.
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape scatter_shape,
      ShapeInference::InferScatterShape(
          shapes.ptrs, to_apply(types()),
          HloScatterInstruction::MakeScatterDimNumbers(
              /*update_window_dims=*/{0, 1, 2, 3, 4},
              /*inserted_window_dims=*/{},
              /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
              /*index_vector_dim=*/0)));

  EXPECT_TRUE(
      ShapeUtil::Equal(scatter_shape, Collate({50, 49, 48, 47, 46}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, ScalarScatterIndices) {
  const auto shapes = CreateShapes({50, 49, 48, 47, 46}, scalar(S64),
                                   {30, 29, 28, 27}, types());
  // The scalar indices "tensor" is a scalar S here that's used to update a
  // [30,29,28,27] shaped tensor within the operand at position S.
  TF_ASSERT_OK_AND_ASSIGN(const Shape scatter_shape,
                          ShapeInference::InferScatterShape(
                              shapes.ptrs, to_apply(types()),
                              HloScatterInstruction::MakeScatterDimNumbers(
                                  /*update_window_dims=*/{0, 1, 2, 3},
                                  /*inserted_window_dims=*/{0},
                                  /*scatter_dims_to_operand_dims=*/{0},
                                  /*index_vector_dim=*/0)));

  EXPECT_TRUE(
      ShapeUtil::Equal(scatter_shape, Collate({50, 49, 48, 47, 46}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, ScatterWithTupleShapedTensorInput) {
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(S64, {10, 9, 8, 7, 1}),
                                 ShapeUtil::MakeShape(S64, {10, 9, 8, 7, 1})});
  const Shape s64_vector_32 = s64_vector(32);
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      {&tuple_shape, &s64_vector_32, &s64_vector_32}, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{0},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/1));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Expected array argument for operand"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, ScatterWithTupleShapedScatterIndicesInput) {
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(S64, {10, 9, 8, 7, 1}),
                                 ShapeUtil::MakeShape(S64, {10, 9, 8, 7, 1})});
  const Shape s64_vector_32 = s64_vector(32);
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      {&s64_vector_32, &tuple_shape, &s64_vector_32}, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{0},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/0));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Expected array argument for scatter indices"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, ScatterWithTupleShapedUpdatesInput) {
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(S64, {10, 9, 8, 7, 1}),
                                 ShapeUtil::MakeShape(S64, {10, 9, 8, 7, 1})});
  const Shape s64_vector_32 = s64_vector(32);
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      {&s64_vector_32, &s64_vector_32, &tuple_shape}, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{0},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/0));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Expected array argument for updates"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, FloatingPointScatterIndicesInput) {
  const Shape s64_vector_32 = s64_vector(32);
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      {&s64_vector_32, &vector_32_, &s64_vector_32}, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{0},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/0));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Scatter indices parameter must be an integral tensor"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, OutOfBoundsScatterIndicesLeafDim) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 2},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/10));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Scatter index leaf dimension must be within [0, "
                        "rank(scatter_indices) + 1)"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, InvalidUpdates) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28, 50}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 2},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Updates tensor must be of rank 7; got 8."))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, InvalidUpdateComputation) {
  const ProgramShape invalid_update_computation =
      ShapeUtil::MakeProgramShape({f32_}, f32_);
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, invalid_update_computation,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 2},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr(absl::Substitute(
                  "Reduction function must take $0 parameters, but takes 1",
                  2 * types().size())))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest,
       InvalidScatterDimNumbers_NonAscendingUpdateWindowDims) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28, 27, 26}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6, 8, 7},
          /*inserted_window_dims=*/{},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("update_window_dims in scatter op must be sorted"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest,
       InvalidScatterDimNumbers_RepeatedUpdateWindowDims) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28, 27, 26}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6, 7, 7},
          /*inserted_window_dims=*/{},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("update_window_dims in scatter op must not repeat"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest,
       InvalidScatterDimNumbers_OutOfBoundsUpdateWindowDims) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28, 27, 26}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6, 7, 9},
          /*inserted_window_dims=*/{},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Invalid update_window_dims set in scatter op; valid "
                        "range is [0, 9)"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest,
       InvalidScatterDimNumbers_NonAscendingInsertedWindowDims) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{2, 1},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("inserted_window_dims in scatter op must be sorted"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest,
       InvalidScatterDimNumbers_RepeatedInsertedWindowDims) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 1},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("inserted_window_dims in scatter op must not repeat"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest,
       InvalidScatterDimNumbers_OutOfBoundsInsertedWindowDims) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 5},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Invalid inserted_window_dims set in scatter op; valid "
                        "range is [0, 5)"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest,
       InvalidScatterDimNumbers_MismatchingScatterDimsToOperandDims) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 2},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr("Scatter op has 4 elements in scatter_dims_to_operand_dims and "
                "the bound of dimension index_vector_dim=4 of scatter_indices "
                "is 5. These two numbers must be equal"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest,
       InvalidScatterDimNumbers_OutOfBoundsScatterDimsToOperandDims) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 2},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 10},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Invalid scatter_dims_to_operand_dims mapping; domain "
                        "is [0, 5), got: 4->10"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest,
       InvalidScatterDimNumbers_RepeatedValuesInScatterDimsToOperandDims) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 2},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 2, 3},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr(
          "Repeated dimensions not allowed in scatter_dims_to_operand_dims"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest,
       InvalidScatterDimNumbers_InsufficientWindowDims) {
  const auto shapes = CreateShapes({50, 49, 48, 47, 46}, scalar(S64),
                                   {30, 29, 28, 27}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{0, 1, 2, 3},
          /*inserted_window_dims=*/{},
          /*scatter_dims_to_operand_dims=*/{0},
          /*index_vector_dim=*/0));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr(
          "Scatter op has window of size 4; doesn't match operand of rank 5."))
      << statusor.status();
}

struct ScatterTestName {
  std::string operator()(
      const ::testing::TestParamInfo<std::vector<PrimitiveType>>& info) const {
    return absl::StrJoin(info.param, "_", absl::StreamFormatter());
  }
};

INSTANTIATE_TEST_SUITE_P(All, ScatterShapeInferenceTest,
                         ::testing::Values(std::vector<PrimitiveType>{F32},
                                           std::vector<PrimitiveType>{F32,
                                                                      BF16}),
                         ScatterTestName());

TEST_P(UnboundedUnaryOpShapeInferenceTest, UnboundedUnaryOps) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape(GetParam().operand));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape(GetParam().expected));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred,
      ShapeInference::InferUnaryOpShape(GetParam().opcode, operand));
  EXPECT_TRUE(ShapeUtil::Equal(inferred, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedAdd) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_F(ShapeInferenceTest, UnboundedAllGather) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferAllGatherShape(
          {&operand}, /*all_gather_dimension=*/0, /*shard_count=*/2));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedAllReduce) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferAllReduceShape({&operand}));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedAllToAll) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferAllToAllShape(/*shape=*/operand,
                                         /*split_dimension=*/0,
                                         /*concat_dimension=*/0,
                                         /*split_count=*/3));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedAllToAllTupleUnsupported) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("(f32[?, 10], f32[?, 10])"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferAllToAllTupleShape(
          /*operand_shapes=*/{&operand, &operand});
  EXPECT_THAT(
      inferred_shape.status().message(),
      HasSubstr("AllToAllTuple does not support unbounded dynamic shapes"));
}

TEST_P(UnboundedLogicalOpShapeInferenceTest, UnboundedAnd) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAnd, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedAtan2) {
  TF_ASSERT_OK_AND_ASSIGN(Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAtan2, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(Shape expected, ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_F(ShapeInferenceTest, UnboundedBitcastConvert) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferBitcastConvertShape(operand, PrimitiveType::F16));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f16[?, 10, 2]"));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedBatchNormGrad) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, ?, 7]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape grad_operand, ParseShape("f32[?, ?, 7]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape scale, ParseShape("f32[5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape mean, ParseShape("f32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape variance, ParseShape("f32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape grad_scale, ParseShape("f32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape grad_offset, ParseShape("f32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape grad_output, ParseShape("f32[5, ?, 7]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferBatchNormGradShape(
                              operand, scale, mean, variance, grad_output, 1));
  const Shape expected_tuple_shape =
      ShapeUtil::MakeTupleShape({grad_operand, grad_scale, grad_offset});
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected_tuple_shape))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected_tuple_shape);
}

TEST_F(ShapeInferenceTest, UnboundedBatchNormInference) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, ?, 7]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape scale, ParseShape("f32[5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape offset, ParseShape("f32[5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape mean, ParseShape("f32[5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape variance, ParseShape("f32[5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferBatchNormInferenceShape(
                              operand, scale, offset, mean, variance, 1));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, ?, 7]"));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedBatchNormTraining) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, ?, 7]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output, ParseShape("f32[?, ?, 7]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape scale, ParseShape("f32[5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape offset, ParseShape("f32[5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape batch_mean, ParseShape("f32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape batch_var, ParseShape("f32[?]"));
  const Shape expected_tuple_shape =
      ShapeUtil::MakeTupleShape({output, batch_mean, batch_var});
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferBatchNormTrainingShape(operand, scale, offset, 1));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected_tuple_shape))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected_tuple_shape);
}

TEST_F(ShapeInferenceTest, UnboundedBroadcastUnsupportedOperand) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[<=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[1, <=2, ?]"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBroadcastShape(operand, /*broadcast_sizes=*/{1});
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("is_unbounded_dynamic"));
}

TEST_F(ShapeInferenceTest, UnboundedBroadcastUnsupportedBroadcastSize) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[<=2, 4]"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBroadcastShape(
          operand, /*broadcast_sizes=*/{Shape::kUnboundedSize});
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("Non-broadcast dimensions must not be dynamic."));
}

TEST_F(ShapeInferenceTest, UnboundedBroadcastInDim) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[<=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[<=2, 3, 4]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferBroadcastShape(operand, expected,
                                          /*broadcast_dimensions=*/{0, 2}));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedBroadcastInDimToBounded) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[<=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[<=2, 3, <=4]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferBroadcastShape(operand, expected,
                                          /*broadcast_dimensions=*/{0, 2}));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedBroadcastInDimUnsupportedOutput) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[<=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[<=2, 3, ?]"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBroadcastShape(operand, expected,
                                          /*broadcast_dimensions=*/{0, 2});
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("is_unbounded_dynamic"));
}

TEST_F(ShapeInferenceTest, UnboundedBroadcastInDimUnsupported) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[<=2, 4]"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBroadcastShape(
          operand, /*broadcast_sizes=*/{2, Shape::kUnboundedSize, 4});
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("Non-broadcast dimensions must not be dynamic."));
}

TEST_F(ShapeInferenceTest, UnboundedCall) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand0, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand1, ParseShape("f32[10, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferCallShape(
                              /*arg_shapes=*/{&operand0, &operand1},
                              /*to_apply=*/ShapeUtil::MakeProgramShape(
                                  {operand1, operand0}, operand0)));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedCholesky) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape a, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferCholeskyShape(a));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedClampOpShapeInferenceTest, UnboundedClamp) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam()[0]));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam()[1]));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape(GetParam()[2]));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, lhs, rhs, ehs);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape(GetParam()[3]));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    EXPECT_EQ(inferred_shape.status().message(), GetParam()[4]);
  }
}

TEST_F(ShapeInferenceTest, UnboundedClampWithTuple) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("(f32[2], f32[?])"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("(f32[?], f32[2])"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("(f32[2], f32[?])"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("(f32[?], f32[2])"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, lhs, rhs, ehs);
  EXPECT_THAT(
      inferred_shape.status().message(),
      HasSubstr(
          "Expected array argument for clamp min, but got (f32[2], f32[?])."));
}

TEST_F(ShapeInferenceTest, UnboundedCollectiveBroadcast) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferCollectiveBroadcastShape(
                              /*operand_shapes=*/{&operand}));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, CollectivePermute) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[8, 8]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[8, 8]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferCollectivePermuteShape(
          /*operand_shapes=*/{&operand}, /*inplace=*/false));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, CollectivePermuteStart) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[8, 8]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("(f32[8, 8], f32[8, 8])"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferCollectivePermuteStartShape(
          /*operand_shapes=*/{&operand}, {}, /*inplace=*/false));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, CombinedCollectivePermute) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand_0, ParseShape("f32[8, 8]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand_1, ParseShape("f32[16, 16]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("(f32[8, 8], f32[16, 16])"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferCollectivePermuteShape(
          /*operand_shapes=*/{&operand_0, &operand_1}, /*inplace=*/false));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, CombinedCollectivePermuteStart) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand_0, ParseShape("f32[8, 8]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand_1, ParseShape("f32[16, 16]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape expected,
      ParseShape("((f32[8, 8], f32[16, 16]), (f32[8, 8], f32[16, 16]))"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferCollectivePermuteStartShape(
          /*operand_shapes=*/{&operand_0, &operand_1}, {}, /*inplace=*/false));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, CombinedInplaceCollectivePermute) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand,
                          ParseShape("(f32[2,3], f32[2,3], u32[], u32[])"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[2,3]"));
  std::vector<const Shape*> operand_shapes;
  absl::c_transform(operand.tuple_shapes(), std::back_inserter(operand_shapes),
                    [](const Shape& shape) { return &shape; });
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferCollectivePermuteShape(
          /*operand_shapes=*/operand_shapes, /*inplace=*/true));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedCollectivePermute) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferCollectivePermuteShape(
          /*operand_shapes=*/{&operand}, /*inplace=*/false));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedCompareOpShapeInferenceTest, UnboundedCompare) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kCompare, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_P(UnboundedComplexOpShapeInferenceTest, UnboundedComplex) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape real, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape imag, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kComplex, real, imag,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_P(UnboundedConcatenateOpShapeInferenceTest, UnboundedConcatenate) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand1, ParseShape(GetParam()[0]));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand2, ParseShape(GetParam()[1]));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferConcatOpShape({&operand1, &operand2},
                                         /*dimension=*/0);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape(GetParam()[2]));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    EXPECT_EQ(inferred_shape.status().message(), GetParam()[3]);
  }
}

TEST_F(UnboundedConcatenateOpShapeInferenceTest,
       UnboundedConcatenateMismatchedDimensions) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand1, ParseShape("f32[2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand2, ParseShape("f32[2, 3]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand3, ParseShape("f32[2, 4]"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferConcatOpShape({&operand1, &operand2, &operand3},
                                         /*dimension=*/0);
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("Mismatched dimension sizes 3 and 4 in dimension 1"));
}

TEST_F(UnboundedConcatenateOpShapeInferenceTest,
       UnboundedConcatenateMismatchedBoundSizes) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand1, ParseShape("f32[2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand2, ParseShape("f32[2, <=3]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand3, ParseShape("f32[2, <=4]"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferConcatOpShape({&operand1, &operand2, &operand3},
                                         /*dimension=*/0);
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("Mismatched bound sizes 3 and 4 in dimension 1"));
}

TEST_F(ShapeInferenceTest, UnboundedConvert) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f64[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape result, ShapeInference::InferConvertShape(
                                                  operand, PrimitiveType::F64));
  EXPECT_TRUE(ShapeUtil::Equal(result, expected))
      << "inferred: " << ShapeUtil::HumanString(result)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedConvolution) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("f32[?, 2, ?, 128]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("f32[2, 2, <=128, 8]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 1, ?, 8]"));

  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.add_output_spatial_dimensions(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);
  dnums.set_input_feature_dimension(3);
  dnums.set_output_feature_dimension(3);
  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(3);

  TF_ASSERT_OK_AND_ASSIGN(
      const Window window,
      ShapeInference::InferWindowFromDimensions(
          /*window_dimensions=*/{2, 2}, /*window_strides=*/{1, 1},
          MakePadding(/*input_dimensions=*/{2, Shape::kUnboundedSize},
                      /*window_dimensions=*/{2, 2},
                      /*window_strides=*/{1, 1}, Padding::kValid),
          /*lhs_dilation=*/{}, /*rhs_dilation=*/{}));
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferConvolveShape(
                              lhs, rhs, /*feature_group_count=*/1,
                              /*batch_group_count=*/1, window, dnums,
                              /*preferred_element_type=*/std::nullopt));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedDiv) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kDivide, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_F(ShapeInferenceTest, UnboundedDot) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));

  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(1);
  dnums.add_rhs_contracting_dimensions(0);

  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferDotOpShape(lhs, rhs, dnums,
                                      /*preferred_element_type=*/std::nullopt));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedDotGeneral) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("f32[?, <=3, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("f32[2, 4, 5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, <=3, 5]"));

  DotDimensionNumbers dnums;
  dnums.add_lhs_batch_dimensions(0);
  dnums.add_rhs_batch_dimensions(0);
  dnums.add_lhs_contracting_dimensions(2);
  dnums.add_rhs_contracting_dimensions(1);

  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferDotOpShape(lhs, rhs, dnums,
                                      /*preferred_element_type=*/std::nullopt));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedDynamicSlice) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape start_index, ParseShape("s32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[2, 2]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferDynamicSliceShape(
          operand, /*start_index_shapes=*/{start_index, start_index},
          /*slice_sizes=*/{2, 2}, /*allow_scalar_indices=*/true));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedDynamicUpdateSlice) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape update, ParseShape("f32[?, 5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape start_index, ParseShape("s32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferDynamicUpdateSliceShape(
          operand, update, /*start_index_shapes=*/{start_index, start_index},
          /*allow_scalar_indices=*/true));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedFftWithFFT) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("c64[2, <=5, ?]"));
  const std::vector<int64_t> fft_length = {5, 10};
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("c64[2, <=5, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferFftShape(
                              operand, /*fft_type=*/FftType::FFT, fft_length));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedFftWithIFFT) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("c64[2, <=5, ?]"));
  const std::vector<int64_t> fft_length = {5, 10};
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("c64[2, <=5, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferFftShape(
                              operand, /*fft_type=*/FftType::IFFT, fft_length));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedFftWithRFFT) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f64[2, <=5, ?]"));
  const std::vector<int64_t> fft_length = {5, 10};
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("c128[2, <=5, 6]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferFftShape(
                              operand, /*fft_type=*/FftType::RFFT, fft_length));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedFftWithIRFFT) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("c128[2, <=5, ?]"));
  const std::vector<int64_t> fft_length = {5, 10};
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f64[2, <=5, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferFftShape(operand, /*fft_type=*/FftType::IRFFT,
                                    fft_length));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedGather) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[3, 4, 2]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape start_indices,
                          ParseShape("s32[?, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, ?, 2, 2]"));

  GatherDimensionNumbers dimension_numbers;
  dimension_numbers.add_offset_dims(2);
  dimension_numbers.add_offset_dims(3);
  dimension_numbers.add_collapsed_slice_dims(0);
  dimension_numbers.add_start_index_map(1);
  dimension_numbers.add_start_index_map(0);
  dimension_numbers.set_index_vector_dim(2);

  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferGatherShape(
                              operand, start_indices, dimension_numbers,
                              /*slice_sizes=*/{1, 2, 2}));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST(XlaBuilderTest, UnboundedGetTupleElement) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferGetTupleElementShape(
          ShapeUtil::MakeTupleShape({operand}), /*index=*/0));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedMap) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand0, ParseShape("f32[2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand1, ParseShape("f32[?, 3, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[2, ?, ?]"));

  const ProgramShape to_apply = ShapeUtil::MakeProgramShape({f32_, f32_}, f32_);

  TF_ASSERT_OK_AND_ASSIGN(
      const Shape result_shape,
      ShapeInference::InferMapShape(/*arg_shapes=*/{&operand0, &operand1},
                                    to_apply, /*dimensions=*/{0, 1, 2}));
  EXPECT_TRUE(ShapeUtil::Equal(result_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(result_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedMax) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kMaximum, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedMin) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_status =
      ShapeInference::InferBinaryOpShape(HloOpcode::kMinimum, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_status.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_status, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_status)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_status.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedMul) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kMultiply, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_P(UnboundedLogicalOpShapeInferenceTest, UnboundedOr) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kOr, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_F(ShapeInferenceTest, UnboundedPad) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape padding_value, ParseShape("f32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 21]"));

  PaddingConfig padding_config;
  for (int i = 0; i < 2; i++) {
    const auto dimension = padding_config.add_dimensions();
    dimension->set_edge_padding_low(1);
    dimension->set_edge_padding_high(1);
    dimension->set_interior_padding(1);
  }

  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferPadShape(operand, padding_value, padding_config));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedPow) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kPower, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_F(ShapeInferenceTest, UnboundedReduce) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape input0, ParseShape("f32[7, 5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape input1, ParseShape("f32[?, 5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape input2, ParseShape("f32[7, ?]"));

  ProgramShape to_apply = ShapeUtil::MakeProgramShape(
      {f32_, f32_, f32_, f32_, f32_, f32_},
      ShapeUtil::MakeTupleShape({f32_, f32_, f32_}));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferReduceShape(
          {&input0, &input1, &input2, &f32_, &f32_, &f32_}, {1}, to_apply));
  const Shape shape = ShapeUtil::MakeShape(F32, {7});
  const Shape expected = ShapeUtil::MakeTupleShape({shape, shape, shape});
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedReduceInvalidReduceDimension) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape input0, ParseShape("f32[7, 5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape input1, ParseShape("f32[?, 5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape input2, ParseShape("f32[5, ?]"));

  ProgramShape to_apply = ShapeUtil::MakeProgramShape(
      {f32_, f32_, f32_, f32_, f32_, f32_},
      ShapeUtil::MakeTupleShape({f32_, f32_, f32_}));
  const absl::StatusOr<Shape> inferred_shape = ShapeInference::InferReduceShape(
      {&input0, &input1, &input2, &f32_, &f32_, &f32_}, {1}, to_apply);
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("All reduced tensors must have compatible dimension"));
}

TEST_F(ShapeInferenceTest, UnboundedReducePrecision) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred,
      ShapeInference::InferReducePrecisionShape(operand, /*exponent_bits=*/2,
                                                /*mantissa_bits=*/2));
  ASSERT_TRUE(ShapeUtil::Equal(inferred, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedReduceScatter) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferReduceScatterShape(
                              /*operand_shapes=*/{&operand},
                              /*scatter_dimension=*/0, /*shard_count=*/2));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedReduceWindow) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape input, ParseShape("f32[?, 4, 8]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 3, 5]"));

  Window window;
  WindowDimension dim0, dim1, dim2;
  dim0.set_stride(1);
  dim0.set_padding_low(0);
  dim0.set_padding_high(0);
  dim0.set_window_dilation(1);
  dim0.set_base_dilation(1);
  dim1 = dim2 = dim0;
  dim0.set_size(1);
  dim1.set_size(2);
  dim2.set_size(4);
  *window.add_dimensions() = dim0;
  *window.add_dimensions() = dim1;
  *window.add_dimensions() = dim2;

  ProgramShape body = ShapeUtil::MakeProgramShape({f32_, f32_}, f32_);
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferReduceWindowShape(
                              input, /*init_value=*/f32_, window, body));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedRemainder) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_status =
      ShapeInference::InferBinaryOpShape(HloOpcode::kRemainder, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_status.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_status, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_status)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_status.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_F(ShapeInferenceTest, UnboundedReshape) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[2,3]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred,
      ShapeInference::InferReshapeShape(operand, /*dimensions=*/{2, 3}, -1));
  ASSERT_TRUE(ShapeUtil::Equal(inferred, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedReshapeUnsupportedOutputShape) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[6]"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferReshapeShape(
          operand,
          /*dimensions=*/{Shape::kUnboundedSize, Shape::kUnboundedSize}, -1);
  EXPECT_THAT(
      inferred_shape.status().message(),
      HasSubstr("Reshaping with unbounded result shape is not supported."));
}

TEST_F(ShapeInferenceTest, UnboundedReshapeUnsupportedMixOfDynamism) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, <=3]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[<=3]"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferReshapeShape(operand, /*dimensions=*/{3}, -1);
  ASSERT_THAT(inferred_shape.status().message(),
              HasSubstr("Reshape operand with bounded and unbounded dynamism "
                        "not supported."));
}

TEST_F(ShapeInferenceTest, UnboundedReverse) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferReverseShape(operand, /*dimensions=*/{0, 1}));
  ASSERT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedScatter) {
  TF_ASSERT_OK_AND_ASSIGN(Shape input, ParseShape("f32[?, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(Shape scatter_indices, ParseShape("s32[?, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(Shape updates, ParseShape("f32[?, ?, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(Shape expected, ParseShape("f32[?, ?, ?]"));

  const ProgramShape to_apply = ShapeUtil::MakeProgramShape({f32_, f32_}, f32_);

  ScatterDimensionNumbers dimension_numbers;
  dimension_numbers.add_update_window_dims(2);
  dimension_numbers.add_update_window_dims(3);
  dimension_numbers.add_inserted_window_dims(0);
  dimension_numbers.add_scatter_dims_to_operand_dims(1);
  dimension_numbers.add_scatter_dims_to_operand_dims(0);
  dimension_numbers.set_index_vector_dim(2);

  TF_ASSERT_OK_AND_ASSIGN(
      Shape result,
      ShapeInference::InferScatterShape({&input, &scatter_indices, &updates},
                                        to_apply, dimension_numbers));
  EXPECT_TRUE(ShapeUtil::Equal(result, expected))
      << "inferred: " << ShapeUtil::HumanString(result)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedSelectOpShapeInferenceTest, UnboundedSelect) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam()[0]));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam()[1]));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape(GetParam()[2]));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kSelect, lhs, rhs, ehs);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape(GetParam()[3]));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    EXPECT_EQ(inferred_shape.status().message(), GetParam()[4]);
  }
}

TEST_F(ShapeInferenceTest, UnboundedSelectWithTupleUnsupported) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("(pred[2], pred[?])"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("(f32[?], f32[2])"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("(f32[2], f32[?])"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("(f32[?], f32[2])"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kSelect, lhs, rhs, ehs);
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("Expected array argument for select pred, but got "
                        "(pred[2], pred[?])."));
}

TEST_F(ShapeInferenceTest, UnboundedSelectAndScatter) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape source, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape init_value, ParseShape("f32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));

  Window window;
  WindowDimension dim0;
  dim0.set_base_dilation(1);
  dim0.set_size(3);
  dim0.set_stride(2);
  dim0.set_padding_low(0);
  dim0.set_padding_high(1);
  dim0.set_window_dilation(1);

  WindowDimension dim1;
  dim1.set_base_dilation(1);
  dim1.set_size(1);
  dim1.set_stride(1);
  dim1.set_padding_low(0);
  dim1.set_padding_high(0);
  dim1.set_window_dilation(1);

  *window.add_dimensions() = dim0;
  *window.add_dimensions() = dim1;

  TF_ASSERT_OK_AND_ASSIGN(
      Shape result,
      ShapeInference::InferSelectAndScatterShape(
          operand,
          /*select_shape=*/ShapeUtil::MakeProgramShape({f32_, f32_}, pred_),
          window, source, init_value,
          /*scatter_shape=*/
          ShapeUtil::MakeProgramShape({f32_, f32_}, f32_)));

  EXPECT_TRUE(ShapeUtil::Equal(result, expected))
      << "inferred: " << ShapeUtil::HumanString(result)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedShiftLeft) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_status =
      ShapeInference::InferBinaryOpShape(HloOpcode::kShiftLeft, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_status.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_status, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_status)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_status.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedShiftRightArithmetic) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_status =
      ShapeInference::InferBinaryOpShape(HloOpcode::kShiftRightArithmetic, lhs,
                                         rhs, GetParam().broadcast_dimensions);
  if (inferred_status.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_status, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_status)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_status.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedShiftRightLogical) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_status =
      ShapeInference::InferBinaryOpShape(HloOpcode::kShiftRightLogical, lhs,
                                         rhs, GetParam().broadcast_dimensions);
  if (inferred_status.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_status, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_status)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_status.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_F(ShapeInferenceTest, UnboundedSlice) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[1, <=3, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[1, <=2, 3]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferSliceShape(operand, /*starts=*/{0, 1, 2},
                                      /*limits=*/{1, 3, 5},
                                      /*strides=*/{1, 1, 1}));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedSort) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferVariadicOpShape(HloOpcode::kSort, {&operand}));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedSub) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kSubtract, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_F(ShapeInferenceTest, UnboundedTranspose) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand,
                          ParseShape("f32[1, ?, 2, ?, <=2]{4,3,2,1,0}"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("f32[<=2, 1, ?, 2, ?]{0,2,3,4,1}"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferTransposeShape(
                              operand, /*dimensions=*/{4, 0, 3, 2, 1}));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedTransposeRank1) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferTransposeShape(operand, /*dimensions=*/{0}));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedTriangularSolve) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape a, ParseShape("f32[?, 3, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape b, ParseShape("f32[?, ?, 4]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, ?, 4]"));
  TriangularSolveOptions options;
  options.set_left_side(true);
  options.set_lower(true);
  options.set_unit_diagonal(false);
  options.set_transpose_a(TriangularSolveOptions::TRANSPOSE);
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape result_shape,
      ShapeInference::InferTriangularSolveShape(a, b, options));
  EXPECT_TRUE(ShapeUtil::Equal(result_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(result_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedTuple) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  const Shape expected = ShapeUtil::MakeTupleShape({operand});
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape result_shape,
      ShapeInference::InferVariadicOpShape(
          HloOpcode::kTuple, std::vector<const Shape*>({&operand})));
  EXPECT_TRUE(ShapeUtil::Equal(result_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(result_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedWhile) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape init, ParseShape("f32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape result_shape, ParseShape("f32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferWhileShape(
          /*condition=*/ShapeUtil::MakeProgramShape({result_shape}, pred_),
          /*body=*/ShapeUtil::MakeProgramShape({result_shape}, result_shape),
          /*init=*/init));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedLogicalOpShapeInferenceTest, UnboundedXor) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_status =
      ShapeInference::InferBinaryOpShape(HloOpcode::kXor, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_status.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_status, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_status)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_status.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

INSTANTIATE_TEST_SUITE_P(UnboundedDynamism,
                         UnboundedLogicalOpShapeInferenceTest,
                         ::testing::ValuesIn<BinaryOpTestCase>(
                             {// LHS | RHS | bdims | Res
                              // 1   | ?   | []    | ?
                              {"s32[1]", "s32[?]", {}, "s32[?]"},
                              // ?   | 1   | []    | ?
                              {"s32[?]", "s32[1]", {}, "s32[?]"},
                              // 2   | ?   | []    | 2
                              {"s32[2]", "s32[?]", {}, "s32[2]"},
                              // ?   | 2   | []    | 2
                              {"s32[?]", "s32[2]", {}, "s32[2]"},
                              // <=2 | ?   | []    | <=2
                              {"s32[<=2]", "s32[?]", {}, "s32[<=2]"},
                              // ?   | <=2 | []    | <=2
                              {"s32[?]", "s32[<=2]", {}, "s32[<=2]"},
                              // ?   | ?   | []    | ?
                              {"s32[?]", "s32[?]", {}, "s32[?]"},
                              // 1   | ?,3 | [0]   | ?,3
                              {"s32[1]", "s32[?,3]", zero_array, "s32[?,3]"},
                              // 2   | ?,3 | [0]   | err
                              {"s32[2]", "s32[?,3]", zero_array, "",
                               kBroadcastDimensionMismatchErrorMessage},
                              // ?,2 | ?,3 | []    | err
                              {"s32[?,2]",
                               "s32[?,3]",
                               {},
                               "",
                               kIncompatibleBinaryOpShapeErrorMessage}}));

INSTANTIATE_TEST_SUITE_P(UnboundedDynamism, UnboundedBinaryOpShapeInferenceTest,
                         ::testing::ValuesIn<BinaryOpTestCase>(
                             {// LHS | RHS | bdims | Res
                              // 1   | ?   | []    | ?
                              {"f32[1]", "f32[?]", {}, "f32[?]"},
                              // ?   | 1   | []    | ?
                              {"f32[?]", "f32[1]", {}, "f32[?]"},
                              // 2   | ?   | []    | 2
                              {"f32[2]", "f32[?]", {}, "f32[2]"},
                              // ?   | 2   | []    | 2
                              {"f32[?]", "f32[2]", {}, "f32[2]"},
                              // <=2 | ?   | []    | <=2
                              {"f32[<=2]", "f32[?]", {}, "f32[<=2]"},
                              // ?   | <=2 | []    | <=2
                              {"f32[?]", "f32[<=2]", {}, "f32[<=2]"},
                              // ?   | ?   | []    | ?
                              {"f32[?]", "f32[?]", {}, "f32[?]"},
                              // 1   | ?,3 | [0]   | ?,3
                              {"f32[1]", "f32[?,3]", zero_array, "f32[?,3]"},
                              // 2   | ?,3 | [0]   | err
                              {"f32[2]", "f32[?,3]", zero_array, "",
                               kBroadcastDimensionMismatchErrorMessage},
                              // ?,2 | ?,3 | []    | err
                              {"f32[?,2]",
                               "f32[?,3]",
                               {},
                               "",
                               kIncompatibleBinaryOpShapeErrorMessage}}));

INSTANTIATE_TEST_SUITE_P(UnboundedDynamism,
                         UnboundedCompareOpShapeInferenceTest,
                         ::testing::ValuesIn<BinaryOpTestCase>(
                             {// LHS | RHS | bdims | Res
                              // 1   | ?   | []    | ?
                              {"f32[1]", "f32[?]", {}, "pred[?]"},
                              // ?   | 1   | []    | ?
                              {"f32[?]", "f32[1]", {}, "pred[?]"},
                              // 2   | ?   | []    | 2
                              {"f32[2]", "f32[?]", {}, "pred[2]"},
                              // ?   | 2   | []    | 2
                              {"f32[?]", "f32[2]", {}, "pred[2]"},
                              // <=2 | ?   | []    | <=2
                              {"f32[<=2]", "f32[?]", {}, "pred[<=2]"},
                              // ?   | <=2 | []    | <=2
                              {"f32[?]", "f32[<=2]", {}, "pred[<=2]"},
                              // ?   | ?   | []    | ?
                              {"f32[?]", "f32[?]", {}, "pred[?]"},
                              // 1   | ?,3 | [0]   | ?,3
                              {"f32[1]", "f32[?,3]", zero_array, "pred[?,3]"},
                              // 2   | ?,3 | [0]   | err
                              {"f32[2]", "f32[?,3]", zero_array, "",
                               kBroadcastDimensionMismatchErrorMessage},
                              // ?,2 | ?,3 | []    | err
                              {"f32[?,2]",
                               "f32[?,3]",
                               {},
                               "",
                               kIncompatibleBinaryOpShapeErrorMessage}}));

INSTANTIATE_TEST_SUITE_P(UnboundedDynamism,
                         UnboundedComplexOpShapeInferenceTest,
                         ::testing::ValuesIn<BinaryOpTestCase>(
                             {// LHS | RHS | bdims | Res
                              // 1   | ?   | []    | ?
                              {"f32[1]", "f32[?]", {}, "c64[?]"},
                              // ?   | 1   | []    | ?
                              {"f32[?]", "f32[1]", {}, "c64[?]"},
                              // 2   | ?   | []    | 2
                              {"f32[2]", "f32[?]", {}, "c64[2]"},
                              // ?   | 2   | []    | 2
                              {"f32[?]", "f32[2]", {}, "c64[2]"},
                              // <=2 | ?   | []    | <=2
                              {"f32[<=2]", "f32[?]", {}, "c64[<=2]"},
                              // ?   | <=2 | []    | <=2
                              {"f32[?]", "f32[<=2]", {}, "c64[<=2]"},
                              // ?   | ?   | []    | ?
                              {"f32[?]", "f32[?]", {}, "c64[?]"},
                              // 1   | ?,3 | [0]   | ?,3
                              {"f32[1]", "f32[?,3]", zero_array, "c64[?,3]"},
                              // 2   | ?,3 | [0]   | err
                              {"f32[2]", "f32[?,3]", zero_array, "",
                               kBroadcastDimensionMismatchErrorMessage},
                              // ?,2 | ?,3 | []    | err
                              {"f32[?,2]",
                               "f32[?,3]",
                               {},
                               "",
                               kIncompatibleBinaryOpShapeErrorMessage}}));

INSTANTIATE_TEST_SUITE_P(
    UnboundedDynamism, UnboundedConcatenateOpShapeInferenceTest,
    ::testing::Values(
        // LHS shape | RHS shape   | Result shape (Concat dim is 0)
        // [X1, Y]   | [X2, Y]     | [X1+X2, Y]
        std::vector<std::string>({"f32[2, 3]", "f32[4, 3]", "f32[6, 3]", ""}),
        // [X, Y]    | [?, ?]      | [?, Y]
        std::vector<std::string>({"f32[2, 3]", "f32[?, ?]", "f32[?, 3]", ""}),
        // [X1, Y]   | [<=X2, <=Y] | [<=X1+X2, <=Y]
        std::vector<std::string>({"f32[4, 3]", "f32[<=2, <=3]", "f32[<=6, <=3]",
                                  ""}),
        // [?, ?]    | [?, ?]      | [?, ?]
        std::vector<std::string>({"f32[?, ?]", "f32[?, ?]", "f32[?, ?]", ""}),
        // [?, ?]    | [<=B1, <=B2]| [?, <=B2]
        std::vector<std::string>({"f32[?, ?]", "f32[<=2, <=3]", "f32[?, <=3]",
                                  ""}),
        // [<=B1, ?] | [<=B2, X]   | [<=B1+B2, X]
        std::vector<std::string>({"f32[<=2, ?]", "f32[<=4, 3]", "f32[<=6, 3]",
                                  ""}),
        // [X, <=B1] | [X, <=B2]   | Error, mismatched
        // bound sizes
        std::vector<std::string>(
            {"f32[2, <=3]", "f32[2, <=4]", "",
             "Cannot concatenate arrays that differ in dimensions other than "
             "the one being concatenated. Dimension 1 in both shapes must be "
             "equal (or compatible): f32[2,<=3] vs f32[2,<=4]."}),
        // [X, Y1]   | [X, Y2]     | Error, mismatched
        // dimension sizes
        std::vector<std::string>(
            {"f32[2, 3]", "f32[2, 4]", "",
             "Cannot concatenate arrays that differ in dimensions other than "
             "the one being concatenated. Dimension 1 in both shapes must be "
             "equal (or compatible): f32[2,3] vs f32[2,4]."})));

INSTANTIATE_TEST_SUITE_P(
    UnboundedDynamism, UnboundedClampOpShapeInferenceTest,
    ::testing::Values(
        // MIN shape | OPERAND shape | MAX shape  | Result
        // []        | [?]           | []         | [?]
        std::vector<std::string>({"f32[]", "f32[?]", "f32[]", "f32[?]", ""}),
        // []        | [?]           | [X]        | [?]
        std::vector<std::string>({"f32[]", "f32[?]", "f32[2]", "f32[?]", ""}),
        // []        | [?]           | [<=B]      | [?]
        std::vector<std::string>({"f32[]", "f32[?]", "f32[<=2]", "f32[?]", ""}),
        // [X]       | [?]           | [X]        | [?]
        std::vector<std::string>({"f32[2]", "f32[?]", "f32[2]", "f32[?]", ""}),
        // [?]       | [X]           | [X]        | [X]
        std::vector<std::string>({"f32[?]", "f32[2]", "f32[2]", "f32[2]", ""}),
        // [?]       | [<=B]         | [?]        | [<=B]
        std::vector<std::string>({"f32[?]", "f32[<=2]", "f32[?]", "f32[<=2]",
                                  ""}),
        // [<=B]     | [?]           | [<=B]      | [?]
        std::vector<std::string>({"f32[<=2]", "f32[?]", "f32[<=2]", "f32[?]",
                                  ""}),
        // [?]       | [?]           | [?]        | [?]
        std::vector<std::string>({"f32[?]", "f32[?]", "f32[?]", "f32[?]", ""}),
        // [?]       | []            | [?]        | error
        std::vector<std::string>(
            {"f32[?]", "f32[]", "f32[?]", "",
             "Clamp with incompatible shapes: f32[?], f32[], f32[?]."}),
        // A[]       | B[?]          | B[?]       | error
        std::vector<std::string>(
            {"s32[]", "f32[?]", "f32[?]", "",
             "Clamp with incompatible element types: s32[], f32[?], f32[?]."}),
        // [X]       | [<=B]         | [X]        | error
        std::vector<std::string>(
            {"f32[3]", "f32[<=2]", "f32[3]", "",
             "Clamp with incompatible shapes: f32[3], f32[<=2], f32[3]."}),
        // [X]       | [?]           | [Y]        | error
        std::vector<std::string>(
            {"f32[2]", "f32[?]", "f32[3]", "",
             "Clamp with incompatible shapes: f32[2], f32[?], f32[3]."})));

INSTANTIATE_TEST_SUITE_P(
    UnboundedDynamism, UnboundedSelectOpShapeInferenceTest,
    ::testing::Values(
        // PRED shape | ON_TRUE shape | ON_FALSE shape  | Result
        // []         | [?]           | [X]             | [X]
        std::vector<std::string>({"pred[]", "f32[?]", "f32[2]", "f32[2]", ""}),
        // []         | [?]           | [<=B]           | [<=B]
        std::vector<std::string>({"pred[]", "f32[?]", "f32[<=2]", "f32[<=2]",
                                  ""}),
        // [X]        | [?]           | [X]             | [X]
        std::vector<std::string>({"pred[2]", "f32[?]", "f32[2]", "f32[2]", ""}),
        // [?]        | [X]           | [X]             | [X]
        std::vector<std::string>({"pred[?]", "f32[2]", "f32[?]", "f32[2]", ""}),
        // [?]        | [<=B]         | [?]             | [<=B]
        std::vector<std::string>({"pred[?]", "f32[<=2]", "f32[?]", "f32[<=2]",
                                  ""}),
        // [<=B]      | [?]           | [<=B]           | [<=B]
        std::vector<std::string>({"pred[<=2]", "f32[?]", "f32[<=2]", "f32[<=2]",
                                  ""}),
        // [?]        | [?]           | [?]             | [?]
        std::vector<std::string>({"pred[?]", "f32[?]", "f32[?]", "f32[?]", ""}),
        // [X]        | A[X]          | B[X]            | error
        std::vector<std::string>({"pred[3]", "s32[3]", "f32[3]", "",
                                  "Operands to select must be the same shape; "
                                  "got s32[3] and f32[3]."}),
        // [X]        | [?]           | [<=B]           | error
        std::vector<std::string>(
            {"pred[3]", "f32[?]", "f32[<=2]", "",
             "Operands to select and predicate must be the same shape; got "
             "f32[?] and f32[<=2] and pred[3]."}),
        // [X]        | [<=B]         | [X]             | error
        std::vector<std::string>({"pred[3]", "f32[<=2]", "f32[3]", "",
                                  "Operands to select must be the same shape; "
                                  "got f32[<=2] and f32[3]."}),
        // [X]        | [?]           | [Y]             | error
        std::vector<std::string>(
            {"pred[2]", "f32[?]", "f32[3]", "f32[3]",
             "Operands to select and predicate must be the same shape; got "
             "f32[?] and f32[3] and pred[2]."}),
        // [?]        | []            | []              | error
        std::vector<std::string>(
            {"pred[?]", "f32[]", "f32[]", "",
             "Operands to select and predicate must be the same shape; got "
             "f32[] and f32[] and pred[?]."}),
        // []         | [?]           | []              | error
        std::vector<std::string>({"pred[]", "f32[?]", "f32[]", "",
                                  "Operands to select must be the same shape; "
                                  "got f32[?] and f32[]."})));

INSTANTIATE_TEST_SUITE_P(UnboundedDynamism, UnboundedUnaryOpShapeInferenceTest,
                         ::testing::ValuesIn<UnaryOpTestCase>(
                             {{"f32[?]", "f32[?]", HloOpcode::kAbs},
                              {"f32[?]", "f32[?]", HloOpcode::kCbrt},
                              {"f32[?]", "f32[?]", HloOpcode::kCeil},
                              {"u32[?]", "u32[?]", HloOpcode::kClz},
                              {"f32[?]", "f32[?]", HloOpcode::kCos},
                              {"f32[?]", "f32[?]", HloOpcode::kErf},
                              {"f32[?]", "f32[?]", HloOpcode::kExp},
                              {"f32[?]", "f32[?]", HloOpcode::kExpm1},
                              {"f32[?]", "f32[?]", HloOpcode::kFloor},
                              {"f32[?]", "f32[?]", HloOpcode::kImag},
                              {"f32[?]", "pred[?]", HloOpcode::kIsFinite},
                              {"f32[?]", "f32[?]", HloOpcode::kLog},
                              {"f32[?]", "f32[?]", HloOpcode::kLog1p},
                              {"f32[?]", "f32[?]", HloOpcode::kLogistic},
                              {"f32[?]", "f32[?]", HloOpcode::kNegate},
                              {"s32[?]", "s32[?]", HloOpcode::kNot},
                              {"u32[?]", "u32[?]", HloOpcode::kPopulationCount},
                              {"f32[?]", "f32[?]", HloOpcode::kReal},
                              {"f32[?]", "f32[?]", HloOpcode::kRoundNearestAfz},
                              {"f32[?]", "f32[?]",
                               HloOpcode::kRoundNearestEven},
                              {"f32[?]", "f32[?]", HloOpcode::kRsqrt},
                              {"f32[?]", "f32[?]", HloOpcode::kSign},
                              {"f32[?]", "f32[?]", HloOpcode::kSin},
                              {"f32[?]", "f32[?]", HloOpcode::kSqrt},
                              {"f32[?]", "f32[?]", HloOpcode::kTanh}}));

}  // namespace
}  // namespace xla
