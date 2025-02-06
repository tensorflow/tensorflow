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

#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/array2d.h"
#include "xla/array3d.h"
#include "xla/array4d.h"
#include "xla/client/local_client.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/testlib/test.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/reference_util.h"
#include "xla/shape_util.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/ml_dtypes.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

class ReshapeTest : public ::testing::WithParamInterface<PrimitiveType>,
                    public ClientLibraryTestBase {
 public:
  ReshapeTest() { set_float_type(GetParam()); }

  ErrorSpec zero_error_spec_{0.0};
};

// Collapses 2-dimensional pseudo-scalar (single-element array) to 1 dimension.
XLA_TEST_P(ReshapeTest, CollapseTrivial1x1) {
  XlaBuilder builder(TestName());
  Array2D<float> input_array(1, 1);
  input_array.Fill(1.0f);
  auto input_literal = LiteralUtil::CreateR2FromArray2D(input_array);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(
                      0, input_literal, "parameter", &builder, &parameter));
  Collapse(/*operand=*/parameter, /*dimensions=*/{0, 1});

  auto expected_literal = LiteralUtil::CreateR1<float>({1.0f});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, CollapseTrivialR1EmptyDims) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateR1<float>({1.0f});
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(
                      0, input_literal, "parameter", &builder, &parameter));
  Collapse(/*operand=*/parameter, /*dimensions=*/{});

  auto expected_literal = LiteralUtil::CreateR1<float>({1.0f});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, CollapseTrivialR1OnlyDim) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateR1<float>({1.0f});
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(
                      0, input_literal, "parameter", &builder, &parameter));
  Collapse(/*operand=*/parameter, /*dimensions=*/{0});

  auto expected_literal = LiteralUtil::CreateR1<float>({1.0f});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Collapses 2-dimensional pseudo-scalar (single-element array) to scalar.
XLA_TEST_P(ReshapeTest, SingleElementArrayToScalar) {
  XlaBuilder builder(TestName());
  Array2D<float> input_array(1, 1);
  input_array.Fill(1.0f);
  auto input_literal = LiteralUtil::CreateR2FromArray2D(input_array);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(
                      0, input_literal, "parameter", &builder, &parameter));
  auto reshape = Reshape(/*operand=*/parameter, /*dimensions=*/{0, 1},
                         /*new_sizes=*/{});
  auto new_shape = builder.GetShape(reshape).value();

  auto expected_literal = LiteralUtil::CreateR0<float>(1.0f);
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, ScalarToSingleElementArray) {
  XlaBuilder builder(TestName());

  Literal param0_literal = LiteralUtil::CreateR0<float>(1.0f);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, param0_literal, "param0",
                                                    &builder, &parameter));
  auto a = Neg(parameter);
  Reshape(/*operand=*/a, /*dimensions=*/{}, /*new_sizes=*/{1});

  auto expected_literal = LiteralUtil::CreateR1<float>({-1.0f});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, Trivial0x3) {
  XlaBuilder builder(TestName());
  Array2D<float> input_array(0, 3);
  auto input_literal = LiteralUtil::CreateR2FromArray2D(input_array);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Collapse(/*operand=*/parameter, /*dimensions=*/{0, 1});
  auto expected_literal = LiteralUtil::CreateR1<float>({});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, Trivial0x3WithParameter) {
  XlaBuilder builder(TestName());

  Literal param0_literal =
      LiteralUtil::CreateR2FromArray2D<float>(Array2D<float>(0, 3));
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, param0_literal, "param0",
                                                    &builder, &parameter));
  Collapse(/*operand=*/parameter, /*dimensions=*/{0, 1});
  auto expected_literal = LiteralUtil::CreateR1<float>({});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, Trivial3x0) {
  XlaBuilder builder(TestName());
  Array2D<float> input_array(3, 0);
  auto input_literal = LiteralUtil::CreateR2FromArray2D(input_array);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Collapse(/*operand=*/parameter, /*dimensions=*/{0, 1});
  auto expected_literal = LiteralUtil::CreateR1<float>({});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Collapses a 2-dimensional row vector to 1 dimension.
XLA_TEST_P(ReshapeTest, Trivial1x3) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateR2<float>({{1.0f, 2.0f, 3.0f}});
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Collapse(/*operand=*/parameter, /*dimensions=*/{0, 1});
  auto expected_literal = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Collapses a 2-dimensional column vector to 1 dimension.
XLA_TEST_P(ReshapeTest, Trivial3x1) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateR2<float>({{1.0f}, {2.0f}, {3.0f}});
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Collapse(/*operand=*/parameter, /*dimensions=*/{0, 1});
  auto expected_literal = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Splits an empty vector into an empty matrix.
XLA_TEST_P(ReshapeTest, R1ToR2_0_To_2x0) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateR1<float>({});
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{0},
          /*new_sizes=*/{2, 0});
  auto expected_literal = LiteralUtil::CreateR2<float>({{}, {}});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Splits a vector into a matrix.
XLA_TEST_P(ReshapeTest, R1ToR2_6_To_2x3) {
  XlaBuilder builder(TestName());
  auto input_literal =
      LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{0},
          /*new_sizes=*/{2, 3});
  auto expected_literal =
      LiteralUtil::CreateR2<float>({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Transposes a 2x0 array to a 0x2 array.
XLA_TEST_P(ReshapeTest, Reshape0x2To2x0) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateFromArray(Array2D<float>(0, 2));
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{0, 1},
          /*new_sizes=*/{2, 0});
  auto expected_literal = LiteralUtil::CreateR2<float>({{}, {}});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Transposes a 2-dimensional row vector to a column vector.
XLA_TEST_P(ReshapeTest, ReshapeRowToCol) {
  XlaBuilder builder(TestName());
  auto simple = MakeLinspaceArray2D(1.0f, 3.0f, 1, 3);
  auto input_literal = LiteralUtil::CreateFromArray(*simple);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{0, 1},
          /*new_sizes=*/{3, 1});

  auto expected = ReferenceUtil::TransposeArray2D(*simple);
  auto expected_literal = LiteralUtil::CreateFromArray(*expected);
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Transposes a 2-dimensional array.
XLA_TEST_P(ReshapeTest, TransposeAsReshape) {
  XlaBuilder builder(TestName());
  auto a4x3 = MakeLinspaceArray2D(1.0f, 12.0f, 4, 3);
  auto input_literal = LiteralUtil::CreateFromArray(*a4x3);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{1, 0},
          /*new_sizes=*/{3, 4});

  auto expected = ReferenceUtil::TransposeArray2D(*a4x3);
  auto expected_literal = LiteralUtil::CreateFromArray(*expected);
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Transposes a 0x4 array with XlaBuilder::Transpose.
XLA_TEST_P(ReshapeTest, Transpose0x4) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateFromArray(Array2D<float>(0, 4));
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Transpose(parameter, {1, 0});
  auto expected_literal = LiteralUtil::CreateR2<float>({{}, {}, {}, {}});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Transposes a 2-dimensional array with ComputationBuilder::Trans.
XLA_TEST_P(ReshapeTest, Transpose4x3) {
  XlaBuilder builder(TestName());
  auto a4x3 = MakeLinspaceArray2D(1.0f, 12.0f, 4, 3);
  auto input_literal = LiteralUtil::CreateFromArray(*a4x3);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Transpose(parameter, {1, 0});

  auto expected = ReferenceUtil::TransposeArray2D(*a4x3);
  auto expected_literal = LiteralUtil::CreateFromArray(*expected);
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Reshapes an empty 2-dimensional array with dimensions that are not just a
// rearrangement of the originals (split), but no reordering (no shuffle).
XLA_TEST_P(ReshapeTest, ReshapeSplitNoShuffleZeroElements) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateFromArray(Array2D<float>(6, 0));
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{0, 1},
          /*new_sizes=*/{2, 3, 0, 0});
  auto expected_literal =
      LiteralUtil::CreateFromArray(Array4D<float>(2, 3, 0, 0));
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, ReshapeR4ToR2ZeroElements) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateFromArray(Array4D<float>(2, 3, 4, 0));
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{0, 1, 2, 3},
          /*new_sizes=*/{24, 0});
  auto expected_literal = LiteralUtil::CreateFromArray(Array2D<float>(24, 0));
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Reshapes a 2-dimensional array with dimensions that are not just a
// rearrangement of the originals (split), but no reordering (no shuffle).
XLA_TEST_P(ReshapeTest, ReshapeSplitNoShuffle) {
  XlaBuilder builder(TestName());
  auto a4x3 = MakeLinspaceArray2D(1.0f, 12.0f, 4, 3);
  auto input_literal = LiteralUtil::CreateFromArray(*a4x3);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{0, 1},
          /*new_sizes=*/{2, 6});

  auto expected = MakeLinspaceArray2D(1.0f, 12.0f, 2, 6);
  auto expected_literal = LiteralUtil::CreateFromArray(*expected);
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, ReshapeSplitAndShuffleZeroElements) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateFromArray(Array2D<float>(0, 6));
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{1, 0},
          /*new_sizes=*/{3, 0});
  auto expected_literal = LiteralUtil::CreateFromArray(Array2D<float>(3, 0));
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Reshapes a 2-dimensional array with dimensions that are not just a
// rearrangement of the originals (split), and reorder the input (shuffle).
XLA_TEST_P(ReshapeTest, ReshapeSplitAndShuffle) {
  XlaBuilder builder(TestName());
  auto a4x3 = MakeLinspaceArray2D(1.0f, 12.0f, 4, 3);
  auto input_literal = LiteralUtil::CreateFromArray(*a4x3);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{1, 0},
          /*new_sizes=*/{2, 6});
  Array2D<float> expected({{1.0f, 4.0f, 7.0f, 10.0f, 2.0f, 5.0f},
                           {8.0f, 11.0f, 3.0f, 6.0f, 9.0f, 12.0f}});
  auto expected_literal = LiteralUtil::CreateFromArray(expected);
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// The following tests use the same input 3D array; they test the examples we
// show for the Reshape operation in the operation_semantics document.
// TODO(b/34503277): find a way to show this code in the documentation without
// duplication on the TF documentation server.
static Array3D<float> ArrayForDocR3Tests() {
  return Array3D<float>({{{10, 11, 12}, {15, 16, 17}},
                         {{20, 21, 22}, {25, 26, 27}},
                         {{30, 31, 32}, {35, 36, 37}},
                         {{40, 41, 42}, {45, 46, 47}}});
}

XLA_TEST_P(ReshapeTest, DocR3_R1_Collapse_012) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateFromArray(ArrayForDocR3Tests());
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{0, 1, 2},
          /*new_sizes=*/{24});
  auto expected_literal = LiteralUtil::CreateR1<float>(
      {10, 11, 12, 15, 16, 17, 20, 21, 22, 25, 26, 27,
       30, 31, 32, 35, 36, 37, 40, 41, 42, 45, 46, 47});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, DocR3_R2_Collapse_012_Refine_83) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateFromArray(ArrayForDocR3Tests());
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{0, 1, 2},
          /*new_sizes=*/{8, 3});
  auto expected_literal = LiteralUtil::CreateR2<float>({{10, 11, 12},
                                                        {15, 16, 17},
                                                        {20, 21, 22},
                                                        {25, 26, 27},
                                                        {30, 31, 32},
                                                        {35, 36, 37},
                                                        {40, 41, 42},
                                                        {45, 46, 47}});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, DocR3_R1_Collapse_120) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateFromArray(ArrayForDocR3Tests());
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{1, 2, 0},
          /*new_sizes=*/{24});
  auto expected_literal = LiteralUtil::CreateR1<float>(
      {10, 20, 30, 40, 11, 21, 31, 41, 12, 22, 32, 42,
       15, 25, 35, 45, 16, 26, 36, 46, 17, 27, 37, 47});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, DocR3_R2_Collapse_120_Refine_83) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateFromArray(ArrayForDocR3Tests());
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{1, 2, 0},
          /*new_sizes=*/{8, 3});
  auto expected_literal = LiteralUtil::CreateR2<float>({{10, 20, 30},
                                                        {40, 11, 21},
                                                        {31, 41, 12},
                                                        {22, 32, 42},
                                                        {15, 25, 35},
                                                        {45, 16, 26},
                                                        {36, 46, 17},
                                                        {27, 37, 47}});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, DocR3_R3_Collapse_120_Refine_262) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateFromArray(ArrayForDocR3Tests());
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{1, 2, 0},
          /*new_sizes=*/{2, 6, 2});
  auto expected_literal = LiteralUtil::CreateR3<float>(
      {{{10, 20}, {30, 40}, {11, 21}, {31, 41}, {12, 22}, {32, 42}},
       {{15, 25}, {35, 45}, {16, 26}, {36, 46}, {17, 27}, {37, 47}}});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Collapses the low dimensions of a 4D tensor to get a 2D matrix, without
// reordering dimensions (for NeuralNet::FullyConnected).
//
// First we create a tesseract raster-face like:
//
// 1 2 3
// 4 5 6
//
// First we collapse Y and X within the raster space yielding:
//
// 1 2 3 4 5 6
//
// Then we collapse Z be collapsed so we just end up with planes:
//
// 1 2 3 4 5 6 1 2 3 4 5 6
XLA_TEST_P(ReshapeTest, FullyConnectedCollapse) {
  XlaBuilder builder(TestName());
  Array4D<float> t2x2x2x3(2, 2, 2, 3);
  auto filler2x3 = MakeLinspaceArray2D(1.0f, 6.0f, 2, 3);
  t2x2x2x3.FillWithYX(*filler2x3);
  auto input_literal = LiteralUtil::CreateFromArray(t2x2x2x3);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Collapse(/*operand=*/parameter, /*dimensions=*/{1, 2, 3});
  auto expected_literal = LiteralUtil::CreateR2<float>(
      {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
       {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        6.0f}});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// As above, but uses reshape directly.
XLA_TEST_P(ReshapeTest, FullyConnectedCollapseDesugared) {
  XlaBuilder builder(TestName());
  Array4D<float> t(2, 1, 2, 2);
  t(0, 0, 0, 0) = 0;
  t(0, 0, 0, 1) = 1;
  t(0, 0, 1, 0) = 2;
  t(0, 0, 1, 1) = 3;
  t(1, 0, 0, 0) = 4;
  t(1, 0, 0, 1) = 5;
  t(1, 0, 1, 0) = 6;
  t(1, 0, 1, 1) = 7;
  auto input_literal = LiteralUtil::CreateFromArray(t);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{0, 1, 2, 3},
          /*new_sizes=*/{2, 4});

  auto expected_literal =
      LiteralUtil::CreateR2<float>({{0, 1, 2, 3}, {4, 5, 6, 7}});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Reshape various ranks to a scalar.
XLA_TEST_P(ReshapeTest, ToScalar) {
  for (int rank = 0; rank < 8; ++rank) {
    XlaBuilder b(TestName());
    std::vector<int64_t> ones(rank, 1);  // this is {1, ..., 1}.
    std::vector<int64_t> dimensions(rank);
    std::iota(dimensions.begin(), dimensions.end(), 0);
    Literal input_literal(ShapeUtil::MakeShape(F32, ones));
    std::vector<int64_t> zeros(rank, 0);  // this is {0, ..., 0}.
    input_literal.Set<float>(zeros, 83.0f);

    XlaOp parameter;
    TF_ASSERT_OK_AND_ASSIGN(
        auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                      &b, &parameter));
    Reshape(parameter, dimensions, {});

    auto expected_literal = LiteralUtil::CreateR0<float>(83.0f);
    ComputeAndCompareLiteral(&b, expected_literal, {input.get()},
                             zero_error_spec_);
  }
}

XLA_TEST_P(ReshapeTest, BadDimensions) {
  XlaBuilder b(TestName());
  auto input_literal = LiteralUtil::CreateR1<float>({1.0f});
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &b, &parameter));
  Reshape(parameter, {}, {});
  EXPECT_THAT(
      ExecuteToString(&b, {}),
      ::testing::HasSubstr("not a permutation of the operand dimensions"));
}

XLA_TEST_P(ReshapeTest, BadNewSizes) {
  XlaBuilder b(TestName());
  auto input_literal = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &b, &parameter));
  Reshape(parameter, {1}, {});
  EXPECT_THAT(ExecuteToString(&b, {}),
              ::testing::HasSubstr("mismatched element counts"));
}

XLA_TEST_P(ReshapeTest, R4Dim0MinorLayoutToR2Dim0MajorLayout) {
  XlaBuilder builder(TestName());
  // clang-format off
  auto input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      Array4D<float>{
    {
      {
        {0, 1},
        {2, 3},
      },
      {
        {100, 101},
        {102, 103},
      },
    },
    {
      {
        {222, 333},
        {444, 555},
      },
      {
        {666, 777},
        {888, 999},
      },
    },
  },
       LayoutUtil::MakeLayout({0, 1, 2, 3}));
  // clang-format on
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));

  Reshape(parameter, /*dimensions=*/{0, 1, 2, 3}, /*new_sizes=*/{2, 8});

  Array2D<float> expected_array({
      {0, 1, 2, 3, 100, 101, 102, 103},
      {222, 333, 444, 555, 666, 777, 888, 999},
  });

  XlaComputation computation = builder.Build().value();
  ExecutionOptions execution_options = execution_options_;
  *execution_options.mutable_shape_with_output_layout() =
      ShapeUtil::MakeShapeWithDenseLayout(FloatType(), {2, 8}, {1, 0})
          .ToProto();
  Literal actual =
      client_
          ->ExecuteAndTransfer(computation, {input.get()}, &execution_options)
          .value();
  Literal expected = LiteralUtil::CreateR2FromArray2D<float>(expected_array);
  if (FloatType() != F32) {
    expected = MaybeConvertLiteralToTestType(expected);
  }
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, actual));
}

XLA_TEST_P(ReshapeTest, R2ToR4_3x8_To_3x2x1x4) {
  XlaBuilder builder(TestName());
  Literal input_literal = LiteralUtil::CreateR2<float>({
      {0, 1, 2, 3, 4, 5, 6, 7},
      {100, 101, 102, 103, 104, 105, 106, 107},
      {200, 201, 202, 203, 204, 205, 206, 207},
  });
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{0, 1}, /*new_sizes=*/{3, 2, 1, 4});

  // clang-format off
  auto expected_literal = LiteralUtil::CreateR4<float>({
    {{{0, 1, 2, 3}},
     {{4, 5, 6, 7}}},
    {{{100, 101, 102, 103}},
     {{104, 105, 106, 107}}},
    {{{200, 201, 202, 203}},
     {{204, 205, 206, 207}}}
  });
  // clang-format on
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Tests R2->R4 reshape with the reshape dimensions {1, 0}.
XLA_TEST_P(ReshapeTest, R2ToR4_3x8_To_3x2x1x4_Dimensions_10) {
  XlaBuilder builder(TestName());
  Literal input_literal = LiteralUtil::CreateR2<float>({
      {0, 1, 2, 3, 4, 5, 6, 7},
      {100, 101, 102, 103, 104, 105, 106, 107},
      {200, 201, 202, 203, 204, 205, 206, 207},
  });
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{1, 0}, /*new_sizes=*/{3, 2, 1, 4});

  // clang-format off
  auto expected_literal = LiteralUtil::CreateR4<float>({
    {{{0, 100, 200, 1}},
     {{101, 201, 2, 102}}},
    {{{202, 3, 103, 203}},
     {{4, 104, 204, 5}}},
    {{{105, 205, 6, 106}},
     {{206, 7, 107, 207}}}
  });
  // clang-format on
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, R4ToR2_2x1x1x1_To_2x1) {
  XlaBuilder builder(TestName());
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  Array4D<float> input(2, 1, 1, 1);
  input.Each([&rng, &distribution](absl::Span<const int64_t> /* indices */,
                                   float* cell) { *cell = distribution(rng); });
  Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      input, LayoutUtil::MakeLayout({3, 2, 1, 0}));
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(auto input_data,
                          CreateParameterAndTransferLiteral(
                              0, input_literal, "input", &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{0, 1, 2, 3}, /*new_sizes=*/{2, 1});

  Literal expected = LiteralUtil::ReshapeSlice({2, 1}, {1, 0}, input_literal);
  ComputeAndCompareLiteral(&builder, expected, {input_data.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, R4ToR2_2x1x4x1_To_4x2) {
  XlaBuilder builder(TestName());
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  Array4D<float> input(2, 1, 4, 1);
  input.Each([&rng, &distribution](absl::Span<const int64_t> /* indices */,
                                   float* cell) { *cell = distribution(rng); });
  Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      input, LayoutUtil::MakeLayout({3, 2, 1, 0}));
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(auto input_data,
                          CreateParameterAndTransferLiteral(
                              0, input_literal, "input", &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{0, 1, 2, 3}, /*new_sizes=*/{4, 2});

  Literal expected = LiteralUtil::ReshapeSlice({4, 2}, {1, 0}, input_literal);
  ComputeAndCompareLiteral(&builder, expected, {input_data.get()},
                           zero_error_spec_);
}

// Tests R4->R2 reshape with the reshape dimensions {0, 2, 1, 3}.
XLA_TEST_P(ReshapeTest, R4ToR2_5x10x2x3_To_5x60_Dimensions_0213) {
  XlaBuilder builder(TestName());
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  Array4D<float> input(5, 10, 2, 3);
  input.Each([&rng, &distribution](absl::Span<const int64_t> /* indices */,
                                   float* cell) { *cell = distribution(rng); });
  Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      input, LayoutUtil::MakeLayout({3, 2, 1, 0}));
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(auto input_data,
                          CreateParameterAndTransferLiteral(
                              0, input_literal, "input", &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{0, 2, 1, 3},
          /*new_sizes=*/{5, 60});

  Array2D<float> expected_array(5, 60);
  input.Each([&](absl::Span<const int64_t> indices, float* cell) {
    expected_array(indices[0], indices[2] * 30 + indices[1] * 3 + indices[3]) =
        *cell;
  });
  auto expected = LiteralUtil::CreateR2FromArray2D(expected_array);
  ComputeAndCompareLiteral(&builder, expected, {input_data.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, NoopReshape) {
  XlaBuilder builder(TestName());
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  Array4D<float> input_array(2, 3, 5, 7);
  input_array.Each(
      [&rng, &distribution](absl::Span<const int64_t> /* indices */,
                            float* cell) { *cell = distribution(rng); });
  Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      input_array, LayoutUtil::MakeLayout({1, 2, 3, 0}));
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(auto input_data,
                          CreateParameterAndTransferLiteral(
                              0, input_literal, "input", &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{3, 0, 1, 2},
          /*new_sizes=*/{7, 2, 3, 5});
  XlaComputation computation = builder.Build().value();

  ExecutionOptions execution_options = execution_options_;
  *execution_options.mutable_shape_with_output_layout() =
      ShapeUtil::MakeShapeWithDenseLayout(FloatType(), {7, 2, 3, 5},
                                          {2, 3, 0, 1})
          .ToProto();
  Literal output_literal =
      client_
          ->ExecuteAndTransfer(computation, {input_data.get()},
                               &execution_options)
          .value();

  // Since the reshape is a no-op, verify that it does not change the underlying
  // data.
  switch (FloatType()) {
    case F32:
      EXPECT_EQ(input_literal.data<float>(), output_literal.data<float>());
      break;
    case BF16: {
      auto expected = MaybeConvertLiteralToTestType(input_literal);
      EXPECT_EQ(expected.data<bfloat16>(), output_literal.data<bfloat16>());
      break;
    }
    case F8E4M3FN: {
      auto expected = MaybeConvertLiteralToTestType(input_literal);
      EXPECT_EQ(expected.data<tsl::float8_e4m3fn>(),
                output_literal.data<tsl::float8_e4m3fn>());
      break;
    }
    case F8E5M2: {
      auto expected = MaybeConvertLiteralToTestType(input_literal);
      EXPECT_EQ(expected.data<tsl::float8_e5m2>(),
                output_literal.data<tsl::float8_e5m2>());
      break;
    }
    default:
      LOG(FATAL) << "Unsupported float type: " << FloatType();
  }
}

XLA_TEST_P(ReshapeTest, R4ToR4Reshape_Trivial) {
  XlaBuilder builder(TestName());
  auto literal_1x2x3x4 = LiteralUtil::CreateR4<float>(
      {{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
        {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}}});

  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, literal_1x2x3x4, "input",
                                                    &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{0, 1, 2, 3},
          /*new_sizes=*/{1, 2, 3, 4});

  ComputeAndCompareLiteral(&builder, literal_1x2x3x4, {input.get()});
}

XLA_TEST_P(ReshapeTest, R4ToR4Reshape) {
  auto literal_1x2x3x4 = LiteralUtil::CreateR4<float>(
      {{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
        {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}}});

  XlaBuilder builder(TestName());
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, literal_1x2x3x4, "input",
                                                    &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{1, 3, 2, 0},
          /*new_sizes=*/{2, 4, 3, 1});

  // clang-format off
  auto expected_2x4x3x1 = LiteralUtil::CreateR4<float>(
      {{{{1}, {5}, {9}},
        {{2}, {6}, {10}},
        {{3}, {7}, {11}},
        {{4}, {8}, {12}}},
       {{{13}, {17}, {21}},
        {{14}, {18}, {22}},
        {{15}, {19}, {23}},
        {{16}, {20}, {24}}}});
  // clang-format on

  ComputeAndCompareLiteral(&builder, expected_2x4x3x1, {input.get()});
}

XLA_TEST_P(ReshapeTest, R4TwoMinorTransposeSimple) {
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  std::vector<int64_t> bounds = {2, 2, 2, 2};
  std::vector<int64_t> new_bounds = {bounds[0], bounds[1], bounds[3],
                                     bounds[2]};
  Array4D<float> input(bounds[0], bounds[1], bounds[2], bounds[3]);
  input.Each([&rng, &distribution](absl::Span<const int64_t> /* indices */,
                                   float* cell) { *cell = distribution(rng); });
  Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      input, LayoutUtil::MakeLayout({3, 2, 1, 0}));
  XlaBuilder builder(TestName());
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(auto input_data,
                          CreateParameterAndTransferLiteral(
                              0, input_literal, "input", &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{0, 1, 3, 2},
          /*new_sizes=*/new_bounds);

  Literal expected =
      LiteralUtil::ReshapeSlice(new_bounds, {2, 3, 1, 0}, input_literal)
          .Relayout(LayoutUtil::MakeLayout({3, 2, 1, 0}));

  // Specify the requested output shape explicitly to ensure that this reshape
  // actually corresponds to a two minor transpose.
  ComputeAndCompareLiteral(&builder, expected, {input_data.get()},
                           zero_error_spec_, &expected.shape());
}

XLA_TEST_P(ReshapeTest, R4TwoMinorTransposeMajorFirstEffectiveR2) {
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  std::vector<int64_t> bounds = {1, 1, 250, 300};
  std::vector<int64_t> new_bounds = {bounds[0], bounds[1], bounds[3],
                                     bounds[2]};
  Array4D<float> input(bounds[0], bounds[1], bounds[2], bounds[3]);
  input.Each([&rng, &distribution](absl::Span<const int64_t> /* indices */,
                                   float* cell) { *cell = distribution(rng); });
  Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      input, LayoutUtil::MakeLayout({3, 2, 1, 0}));
  XlaBuilder builder(TestName());
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(auto input_data,
                          CreateParameterAndTransferLiteral(
                              0, input_literal, "input", &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{0, 1, 3, 2},
          /*new_sizes=*/new_bounds);

  Literal expected =
      LiteralUtil::ReshapeSlice(new_bounds, {2, 3, 1, 0}, input_literal)
          .Relayout(LayoutUtil::MakeLayout({3, 2, 1, 0}));

  // Specify the requested output shape explicitly to ensure that this reshape
  // actually corresponds to a two minor transpose.
  ComputeAndCompareLiteral(&builder, expected, {input_data.get()},
                           zero_error_spec_, &expected.shape());
}

XLA_TEST_P(ReshapeTest, R4TwoMinorTransposeMajorFirstMinorEffectiveR1) {
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  std::vector<int64_t> bounds = {5, 5, 1, 10};
  std::vector<int64_t> new_bounds = {bounds[0], bounds[1], bounds[3],
                                     bounds[2]};
  Array4D<float> input(bounds[0], bounds[1], bounds[2], bounds[3]);
  input.Each([&rng, &distribution](absl::Span<const int64_t> /* indices */,
                                   float* cell) { *cell = distribution(rng); });
  Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      input, LayoutUtil::MakeLayout({3, 2, 1, 0}));
  XlaBuilder builder(TestName());
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(auto input_data,
                          CreateParameterAndTransferLiteral(
                              0, input_literal, "input", &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{0, 1, 3, 2},
          /*new_sizes=*/new_bounds);

  Literal expected =
      LiteralUtil::ReshapeSlice(new_bounds, {2, 3, 1, 0}, input_literal)
          .Relayout(LayoutUtil::MakeLayout({3, 2, 1, 0}));

  // Specify the requested output shape explicitly to ensure that this reshape
  // actually corresponds to a two minor transpose.
  ComputeAndCompareLiteral(&builder, expected, {input_data.get()},
                           zero_error_spec_, &expected.shape());
}

XLA_TEST_P(ReshapeTest, R4TwoMinorTransposeMajorFirstMinorEffectiveR1InR2) {
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  // This happens in NN-Builder MNIST.
  std::vector<int64_t> bounds = {5, 5, 10, 1};
  std::vector<int64_t> new_bounds = {bounds[0], bounds[1], bounds[3],
                                     bounds[2]};
  Array4D<float> input(bounds[0], bounds[1], bounds[2], bounds[3]);
  input.Each([&rng, &distribution](absl::Span<const int64_t> /* indices */,
                                   float* cell) { *cell = distribution(rng); });
  Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      input, LayoutUtil::MakeLayout({3, 2, 1, 0}));
  XlaBuilder builder(TestName());
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(auto input_data,
                          CreateParameterAndTransferLiteral(
                              0, input_literal, "input", &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{0, 1, 3, 2},
          /*new_sizes=*/new_bounds);

  Literal expected =
      LiteralUtil::ReshapeSlice(new_bounds, {2, 3, 1, 0}, input_literal)
          .Relayout(LayoutUtil::MakeLayout({3, 2, 1, 0}));

  // Specify the requested output shape explicitly to ensure that this reshape
  // actually corresponds to a two minor transpose.
  ComputeAndCompareLiteral(&builder, expected, {input_data.get()},
                           zero_error_spec_, &expected.shape());
}

XLA_TEST_P(ReshapeTest, R4TwoMinorTransposeTrivialR2) {
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  std::vector<int64_t> bounds = {3, 3, 1, 3};
  std::vector<int64_t> new_bounds = {bounds[1], bounds[0], bounds[2],
                                     bounds[3]};
  Array4D<float> input(bounds[0], bounds[1], bounds[2], bounds[3]);
  input.Each([&rng, &distribution](absl::Span<const int64_t> /* indices */,
                                   float* cell) { *cell = distribution(rng); });
  Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      input, LayoutUtil::MakeLayout({0, 1, 2, 3}));
  XlaBuilder builder(TestName());
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(auto input_data,
                          CreateParameterAndTransferLiteral(
                              0, input_literal, "input", &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{1, 0, 2, 3},
          /*new_sizes=*/new_bounds);

  Literal expected =
      LiteralUtil::ReshapeSlice(new_bounds, {1, 0, 2, 3}, input_literal)
          .Relayout(input_literal.shape().layout());

  // Specify the requested output shape explicitly to ensure that this reshape
  // actually corresponds to a two minor transpose.
  ComputeAndCompareLiteral(&builder, expected, {input_data.get()},
                           zero_error_spec_, &expected.shape());
}

INSTANTIATE_TEST_CASE_P(ReshapeTestInstance, ReshapeTest,
                        ::testing::ValuesIn({F32, BF16, F8E5M2, F8E4M3FN}));

using ReshapeHloTest = HloTestBase;

TEST_F(ReshapeHloTest, NoHloPasses) {
  const std::string hlo_string = R"(
    HloModule Bug, is_scheduled=true

    ENTRY entry {
      %p0 = u32[1,35]{1,0} parameter(0)
      %reshape.4 = u32[35]{0} reshape(u32[1,35]{1,0} %p0)
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_string, ErrorSpec{0.01, 0.01}));
}

}  // namespace
}  // namespace xla
