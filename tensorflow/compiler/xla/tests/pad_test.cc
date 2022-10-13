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

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace {

#ifdef XLA_BACKEND_SUPPORTS_BFLOAT16
// Tests both F32 and BF16.
static std::array<bool, 2> use_bfloat16_params{false, true};
#else
// Only tests F32.
static std::array<bool, 1> use_bfloat16_params{false};
#endif

class PadTest : public ClientLibraryTestBase {
 protected:
  PadTest() {
    // Initializes the padding configuration used for R4 tests.
    // Pad only on the dimension 0 {low: 1, high: 0, interior: 2} and
    // dimension 1 {low: 0, high: 2, interior: 1}.
    auto dimension0 = r4_padding_on_dim0_dim1_.add_dimensions();
    dimension0->set_edge_padding_low(1);
    dimension0->set_edge_padding_high(0);
    dimension0->set_interior_padding(2);
    auto dimension1 = r4_padding_on_dim0_dim1_.add_dimensions();
    dimension1->set_edge_padding_low(0);
    dimension1->set_edge_padding_high(2);
    dimension1->set_interior_padding(1);
    auto dimension2 = r4_padding_on_dim0_dim1_.add_dimensions();
    dimension2->set_edge_padding_low(0);
    dimension2->set_edge_padding_high(0);
    dimension2->set_interior_padding(0);
    auto dimension3 = r4_padding_on_dim0_dim1_.add_dimensions();
    dimension3->set_edge_padding_low(0);
    dimension3->set_edge_padding_high(0);
    dimension3->set_interior_padding(0);
  }

  // Padding configuration for R4 that only pads dimension 0 and 1.
  PaddingConfig r4_padding_on_dim0_dim1_;
};

class PadTestFloat : public PadTest,
                     public ::testing::WithParamInterface<bool> {
 protected:
  PadTestFloat() { set_use_bfloat16(GetParam()); }

  ErrorSpec DefaultErrorSpec() const {
    if (use_bfloat16()) {
      return ErrorSpec(1e-3, 1e-3);
    } else {
      return ErrorSpec(1e-5, 1e-5);
    }
  }
};

// Tests a Pad() with a zero-element input and output.
XLA_TEST_P(PadTestFloat, Pad1DS0ToS0Array) {
  XlaBuilder b(TestName());
  // Set up the padding configuration {low: 0, high: 0, interior: 0}.
  PaddingConfig padding_config;
  auto dimension = padding_config.add_dimensions();
  dimension->set_edge_padding_low(0);
  dimension->set_edge_padding_high(0);
  dimension->set_interior_padding(0);

  Pad(AddParam(LiteralUtil::CreateR1<float>({}), &b),
      AddParam(LiteralUtil::CreateR0<float>(0.1), &b), padding_config);
  ComputeAndCompareR1<float>(&b, {}, {}, DefaultErrorSpec());
}

// Tests a Pad() with a zero-element input but a non-zero-element output.
XLA_TEST_P(PadTestFloat, Pad1DS0ToS5Array) {
  XlaBuilder b(TestName());
  // Set up the padding configuration {low: 3, high: 0, interior: 1}.
  PaddingConfig padding_config;
  auto dimension = padding_config.add_dimensions();
  dimension->set_edge_padding_low(1);
  dimension->set_edge_padding_high(4);
  dimension->set_interior_padding(7);

  Pad(AddParam(LiteralUtil::CreateR1<float>({}), &b),
      AddParam(LiteralUtil::CreateR0<float>(0.1), &b), padding_config);
  ComputeAndCompareR1<float>(&b, std::vector<float>(5, 0.1), {},
                             DefaultErrorSpec());
}

XLA_TEST_P(PadTestFloat, Pad1DS3Array) {
  XlaBuilder b(TestName());
  // Set up the padding configuration {low: 3, high: 0, interior: 1}.
  PaddingConfig padding_config;
  auto dimension = padding_config.add_dimensions();
  dimension->set_edge_padding_low(3);
  dimension->set_edge_padding_high(0);
  dimension->set_interior_padding(1);

  Pad(AddParam(LiteralUtil::CreateR1<float>({1, 2, 3}), &b),
      AddParam(LiteralUtil::CreateR0<float>(0.1), &b), padding_config);
  std::vector<float> expected({0.1, 0.1, 0.1, 1, 0.1, 2, 0.1, 3});
  ComputeAndCompareR1<float>(&b, expected, {}, DefaultErrorSpec());
}

XLA_TEST_P(PadTestFloat, Pad4D_2x0x3x2_FloatArray) {
  XlaBuilder b(TestName());
  Pad(AddParam(Array4D<float>(2, 0, 3, 2), &b),
      AddParam(LiteralUtil::CreateR0<float>(1.5), &b),
      r4_padding_on_dim0_dim1_);
  ComputeAndCompareR4<float>(&b, Array4D<float>(5, 2, 3, 2, 1.5f), {},
                             DefaultErrorSpec());
}

TEST_P(PadTestFloat, Pad4DFloat_1x1x3x2_Array) {
  XlaBuilder b(TestName());
  auto input = std::make_unique<Array4D<float>>(1, 1, 3, 2);
  Array2D<float> input_xy({
      {1.0f, 2.0f},  // row 0
      {3.0f, 4.0f},  // row 1
      {5.0f, 6.0f},  // row 2
  });
  input->FillWithYX(input_xy);

  Pad(AddParam(*input, &b), AddParam(LiteralUtil::CreateR0<float>(1.5), &b),
      r4_padding_on_dim0_dim1_);

  auto expected = std::make_unique<Array4D<float>>(2, 3, 3, 2);
  expected->Fill(1.5);
  (*expected)(1, 0, 0, 0) = 1.0f;
  (*expected)(1, 0, 0, 1) = 2.0f;
  (*expected)(1, 0, 1, 0) = 3.0f;
  (*expected)(1, 0, 1, 1) = 4.0f;
  (*expected)(1, 0, 2, 0) = 5.0f;
  (*expected)(1, 0, 2, 1) = 6.0f;
  ComputeAndCompareR4<float>(&b, *expected, {}, DefaultErrorSpec());
}

TEST_P(PadTestFloat, Pad4DFloatArrayWithInteriorPadding) {
  XlaBuilder b(TestName());

  const float pad_value = 1.5f;
  Array4D<float> input(3, 2, 1, 1, {1, 2, 3, 4, 5, 6});
  Pad(AddParam(input, &b),
      AddParam(LiteralUtil::CreateR0<float>(pad_value), &b),
      r4_padding_on_dim0_dim1_);

  auto expected = std::make_unique<Array4D<float>>(8, 5, 1, 1);
  expected->Fill(pad_value);
  (*expected)(1, 0, 0, 0) = 1.0f;
  (*expected)(1, 2, 0, 0) = 2.0f;
  (*expected)(4, 0, 0, 0) = 3.0f;
  (*expected)(4, 2, 0, 0) = 4.0f;
  (*expected)(7, 0, 0, 0) = 5.0f;
  (*expected)(7, 2, 0, 0) = 6.0f;
  ComputeAndCompareR4<float>(&b, *expected, {}, ErrorSpec(0.0001));
}

TEST_P(PadTestFloat, Pad4DFloatArrayMinorFirstSmall) {
  XlaBuilder b(TestName());

  PaddingConfig padding_config;
  auto dimension0 = padding_config.add_dimensions();
  dimension0->set_edge_padding_low(0);
  dimension0->set_edge_padding_high(0);
  dimension0->set_interior_padding(0);
  auto dimension1 = padding_config.add_dimensions();
  dimension1->set_edge_padding_low(0);
  dimension1->set_edge_padding_high(0);
  dimension1->set_interior_padding(0);
  auto dimension2 = padding_config.add_dimensions();
  dimension2->set_edge_padding_low(2);
  dimension2->set_edge_padding_high(1);
  dimension2->set_interior_padding(0);
  auto dimension3 = padding_config.add_dimensions();
  dimension3->set_edge_padding_low(2);
  dimension3->set_edge_padding_high(3);
  dimension3->set_interior_padding(0);

  const Layout layout = LayoutUtil::MakeLayout({0, 1, 2, 3});

  const float pad_value = -5.123f;
  Array4D<float> input_array(1, 1, 2, 3, {1, 2, 3, 4, 5, 6});
  auto input = LiteralUtil::CreateR4FromArray4D<float>(input_array);
  input = input.Relayout(layout);

  Pad(AddParam(input, &b),
      AddParam(LiteralUtil::CreateR0<float>(pad_value), &b), padding_config);

  Array4D<float> expected_array(1, 1, 5, 8);
  expected_array.Fill(pad_value);
  expected_array(0, 0, 2, 2) = 1.0f;
  expected_array(0, 0, 2, 3) = 2.0f;
  expected_array(0, 0, 2, 4) = 3.0f;
  expected_array(0, 0, 3, 2) = 4.0f;
  expected_array(0, 0, 3, 3) = 5.0f;
  expected_array(0, 0, 3, 4) = 6.0f;
  ComputeAndCompareR4<float>(&b, expected_array, {}, ErrorSpec(0.0001));
}

XLA_TEST_P(PadTestFloat, Pad4DFloatArrayMinorFirstNonTrivialMinorDimensions) {
  XlaBuilder b(TestName());

  PaddingConfig padding_config;
  auto dimension0 = padding_config.add_dimensions();
  dimension0->set_edge_padding_low(0);
  dimension0->set_edge_padding_high(0);
  dimension0->set_interior_padding(0);
  auto dimension1 = padding_config.add_dimensions();
  dimension1->set_edge_padding_low(0);
  dimension1->set_edge_padding_high(0);
  dimension1->set_interior_padding(0);
  auto dimension2 = padding_config.add_dimensions();
  dimension2->set_edge_padding_low(2);
  dimension2->set_edge_padding_high(2);
  dimension2->set_interior_padding(1);
  auto dimension3 = padding_config.add_dimensions();
  dimension3->set_edge_padding_low(2);
  dimension3->set_edge_padding_high(2);
  dimension3->set_interior_padding(0);

  const Layout layout = LayoutUtil::MakeLayout({0, 1, 2, 3});

  const float pad_value = -5.123f;
  Array4D<float> input_array(1, 25, 7, 7);
  input_array.Fill(pad_value);
  input_array(0, 0, 0, 0) = 1.0f;
  input_array(0, 24, 6, 6) = 2.0f;
  input_array(0, 17, 2, 5) = 3.0f;
  auto input = LiteralUtil::CreateR4FromArray4D<float>(input_array);
  input = input.Relayout(layout);

  Pad(AddParam(input, &b),
      AddParam(LiteralUtil::CreateR0<float>(pad_value), &b), padding_config);

  Array4D<float> expected_array(1, 25, 17, 11);
  expected_array.Fill(pad_value);
  expected_array(0, 0, 2, 2) = 1.0f;
  expected_array(0, 24, 14, 8) = 2.0f;
  expected_array(0, 17, 6, 7) = 3.0f;
  ComputeAndCompareR4<float>(&b, expected_array, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(PadTest, Pad4DU8Array) {
  XlaBuilder b(TestName());
  auto input = std::make_unique<Array4D<uint8_t>>(1, 1, 3, 2);
  Array2D<uint8_t> input_xy({
      {1, 2},  // row 0
      {3, 4},  // row 1
      {5, 6},  // row 2
  });
  input->FillWithYX(input_xy);

  Pad(AddParam(*input, &b), ConstantR0<uint8_t>(&b, 35),
      r4_padding_on_dim0_dim1_);

  auto expected = std::make_unique<Array4D<uint8_t>>(2, 3, 3, 2);
  expected->Fill(35);
  (*expected)(1, 0, 0, 0) = 1;
  (*expected)(1, 0, 0, 1) = 2;
  (*expected)(1, 0, 1, 0) = 3;
  (*expected)(1, 0, 1, 1) = 4;
  (*expected)(1, 0, 2, 0) = 5;
  (*expected)(1, 0, 2, 1) = 6;
  ComputeAndCompareR4<uint8_t>(&b, *expected, {});
}

XLA_TEST_F(PadTest, Pad4DPredArray) {
  XlaBuilder b(TestName());

  // Since bool is currently not well supported, use Broadcast operation to
  // create the operand for Pad.
  auto input = Broadcast(ConstantR0<bool>(&b, true), {1, 1, 3, 2});
  auto padded =
      Pad(input, ConstantR0<bool>(&b, false), r4_padding_on_dim0_dim1_);

  // For the same reason, use Select to convert boolean values to int32_t.
  auto zeros = std::make_unique<Array4D<int32_t>>(2, 3, 3, 2);
  auto ones = std::make_unique<Array4D<int32_t>>(2, 3, 3, 2);
  zeros->Fill(0);
  ones->Fill(1);
  Select(padded, AddParam(*ones, &b), AddParam(*zeros, &b));

  auto expected = std::make_unique<Array4D<int32_t>>(2, 3, 3, 2);
  expected->Fill(0);
  (*expected)(1, 0, 0, 0) = 1;
  (*expected)(1, 0, 0, 1) = 1;
  (*expected)(1, 0, 1, 0) = 1;
  (*expected)(1, 0, 1, 1) = 1;
  (*expected)(1, 0, 2, 0) = 1;
  (*expected)(1, 0, 2, 1) = 1;
  ComputeAndCompareR4<int32_t>(&b, *expected, {});
}

XLA_TEST_P(PadTestFloat, Large2DPad) {
  XlaBuilder b(TestName());

  auto ones = std::make_unique<Array2D<float>>(4, 4);
  ones->Fill(1.0f);
  auto input = AddParam(*ones, &b);
  PaddingConfig padding_config = MakeNoPaddingConfig(2);
  for (int dim : {0, 1}) {
    padding_config.mutable_dimensions(dim)->set_edge_padding_low(
        98 + 100 * (1 - dim));
    padding_config.mutable_dimensions(dim)->set_edge_padding_high(58 +
                                                                  100 * dim);
  }
  Pad(input, AddParam(LiteralUtil::CreateR0<float>(0.0f), &b), padding_config);

  auto expected = ReferenceUtil::PadArray2D(*ones, padding_config, 0.0f);
  ComputeAndCompareR2<float>(&b, *expected, {}, DefaultErrorSpec());
}

XLA_TEST_P(PadTestFloat, AllTypes2DPad) {
  XlaBuilder b(TestName());

  constexpr int64_t in_rows = 35;
  constexpr int64_t in_cols = 35;
  auto operand = std::make_unique<Array2D<float>>(in_rows, in_cols);
  operand->FillUnique(0.0f);
  auto input = AddParam(*operand, &b);

  PaddingConfig padding_config = MakeNoPaddingConfig(2);
  padding_config.mutable_dimensions(0)->set_edge_padding_low(7);
  padding_config.mutable_dimensions(0)->set_edge_padding_high(5);
  padding_config.mutable_dimensions(0)->set_interior_padding(3);
  padding_config.mutable_dimensions(1)->set_edge_padding_low(6);
  padding_config.mutable_dimensions(1)->set_edge_padding_high(4);
  padding_config.mutable_dimensions(1)->set_interior_padding(2);
  Pad(input, AddParam(LiteralUtil::CreateR0<float>(3.14f), &b), padding_config);

  auto expected = ReferenceUtil::PadArray2D(*operand, padding_config, 3.14f);
  ComputeAndCompareR2<float>(&b, *expected, {}, DefaultErrorSpec());
}

XLA_TEST_P(PadTestFloat, High2DPad) {
  XlaBuilder b(TestName());

  constexpr int64_t in_rows = 129;
  constexpr int64_t in_cols = 129;
  constexpr int64_t low_padding = 0;
  int64_t high_padding[2] = {5, 7};
  constexpr int64_t interior_padding = 0;
  auto operand = std::make_unique<Array2D<float>>(in_rows, in_cols);
  operand->FillUnique(1.0f);
  auto input = AddParam(*operand, &b);
  PaddingConfig padding_config = MakeNoPaddingConfig(2);
  for (int dim : {0, 1}) {
    padding_config.mutable_dimensions(dim)->set_edge_padding_low(low_padding);
    padding_config.mutable_dimensions(dim)->set_edge_padding_high(
        high_padding[dim]);
    padding_config.mutable_dimensions(dim)->set_interior_padding(
        interior_padding);
  }
  Pad(input, AddParam(LiteralUtil::CreateR0<float>(2.718f), &b),
      padding_config);

  auto expected = ReferenceUtil::PadArray2D(*operand, padding_config, 2.718f);

  ComputeAndCompareR2<float>(&b, *expected, {}, DefaultErrorSpec());
}

XLA_TEST_P(PadTestFloat, NegativePadding2D) {
  XlaBuilder b(TestName());

  constexpr int64_t in_rows = 129;
  constexpr int64_t in_cols = 129;
  int64_t low_padding[2] = {-1, -2};
  int64_t high_padding[2] = {-3, 4};
  constexpr int64_t interior_padding = 0;
  auto operand = std::make_unique<Array2D<float>>(in_rows, in_cols);
  operand->FillUnique(1.0f);
  auto input = AddParam(*operand, &b);
  PaddingConfig padding_config = MakeNoPaddingConfig(2);
  for (int dim : {0, 1}) {
    padding_config.mutable_dimensions(dim)->set_edge_padding_low(
        low_padding[dim]);
    padding_config.mutable_dimensions(dim)->set_edge_padding_high(
        high_padding[dim]);
    padding_config.mutable_dimensions(dim)->set_interior_padding(
        interior_padding);
  }
  Pad(input, AddParam(LiteralUtil::CreateR0<float>(2.718f), &b),
      padding_config);

  auto expected = ReferenceUtil::PadArray2D(*operand, padding_config, 2.718f);

  ComputeAndCompareR2<float>(&b, *expected, {}, DefaultErrorSpec());
}

XLA_TEST_P(PadTestFloat, NegativeAndInteriorPadding2D) {
  XlaBuilder b(TestName());

  constexpr int64_t in_rows = 8;
  constexpr int64_t in_cols = 11;
  int64_t low_padding[2] = {4, -1};
  int64_t high_padding[2] = {-2, -4};
  int64_t interior_padding[2] = {1, 2};
  auto operand = std::make_unique<Array2D<float>>(in_rows, in_cols);
  operand->FillUnique(1.0f);
  auto input = AddParam(*operand, &b);
  PaddingConfig padding_config = MakeNoPaddingConfig(2);
  for (int dim : {0, 1}) {
    padding_config.mutable_dimensions(dim)->set_edge_padding_low(
        low_padding[dim]);
    padding_config.mutable_dimensions(dim)->set_edge_padding_high(
        high_padding[dim]);
    padding_config.mutable_dimensions(dim)->set_interior_padding(
        interior_padding[dim]);
  }
  Pad(input, AddParam(LiteralUtil::CreateR0<float>(2.718f), &b),
      padding_config);

  auto expected = ReferenceUtil::PadArray2D(*operand, padding_config, 2.718f);

  ComputeAndCompareR2<float>(&b, *expected, {}, DefaultErrorSpec());
}

// Regression test for b/31827337.
XLA_TEST_P(PadTestFloat, ReducePad) {
  XlaBuilder b(TestName());
  auto ones = std::make_unique<Array4D<float>>(2, 2, 2, 2);
  ones->Fill(1.0);
  auto input = AddParam(*ones, &b);

  XlaComputation add = CreateScalarAddComputation(FloatType(), &b);
  auto reduce =
      Reduce(input, AddParam(LiteralUtil::CreateR0<float>(0.0), &b), add, {0});

  PaddingConfig padding_config = MakeNoPaddingConfig(3);
  padding_config.mutable_dimensions(0)->set_edge_padding_low(1);
  padding_config.mutable_dimensions(0)->set_edge_padding_high(1);
  Pad(reduce, AddParam(LiteralUtil::CreateR0<float>(0.0f), &b), padding_config);

  Array3D<float> expected({{{0.0, 0.0}, {0.0, 0.0}},
                           {{2.0, 2.0}, {2.0, 2.0}},
                           {{2.0, 2.0}, {2.0, 2.0}},
                           {{0.0, 0.0}, {0.0, 0.0}}});
  ComputeAndCompareR3<float>(&b, expected, {}, DefaultErrorSpec());
}

INSTANTIATE_TEST_CASE_P(PadTestFloatInstantiation, PadTestFloat,
                        ::testing::ValuesIn(use_bfloat16_params));

}  // namespace
}  // namespace xla
