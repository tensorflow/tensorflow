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

// Tests of 1D convolution with trivial kernels and no special variations (like
// strides and padding).

#include <cstdint>
#include <vector>

#include "Eigen/Core"
#include "xla/array3d.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/padding.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

#if XLA_TEST_BACKEND_GPU
// XLA:GPU sometimes uses FFT convolution which isn't as precise as spatial
// convolution. So relax the absolute error threshold.
constexpr ErrorSpec kErrorSpec(1e-2, 1e-3);
#else
constexpr ErrorSpec kErrorSpec(1e-4, 1e-3);
#endif

using ConvolutionTest = ClientLibraryTestRunnerMixin<
    HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>>;

#ifdef XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16
using TestTypes = ::testing::Types<float>;
#else
using TestTypes = ::testing::Types<float, Eigen::half>;
#endif

struct Convolve1DTestParam {
  int64_t input_feature;
  int64_t output_feature;
  int64_t batch;
  int64_t window_size;
  int64_t num_windows;
};

class Convolve1D1WindowTestBase
    : public ConvolutionTest,
      public ::testing::WithParamInterface<Convolve1DTestParam> {
 protected:
  template <typename T>
  void TestImpl() {
    XlaBuilder builder(TestName());
    int64_t input_feature = GetParam().input_feature;
    int64_t output_feature = GetParam().output_feature;
    int64_t batch = GetParam().batch;
    int64_t num_windows = GetParam().num_windows;
    int64_t window_size = GetParam().window_size;
    std::vector<int64_t> input_dims = {batch, window_size + num_windows - 1,
                                       input_feature};
    std::vector<int64_t> filter_dims = {window_size, input_feature,
                                        output_feature};
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>(input_dims);
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>(filter_dims);
    {
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");

      // Tensorflow dimension numbers for 1D convolution.
      ConvolutionDimensionNumbers dnums;
      dnums.set_input_batch_dimension(0);
      dnums.set_output_batch_dimension(0);
      dnums.add_input_spatial_dimensions(1);
      dnums.add_output_spatial_dimensions(1);
      dnums.set_input_feature_dimension(2);
      dnums.set_output_feature_dimension(2);
      dnums.add_kernel_spatial_dimensions(0);
      dnums.set_kernel_input_feature_dimension(1);
      dnums.set_kernel_output_feature_dimension(2);

      ConvWithGeneralDimensions(input, filter, {1}, Padding::kValid, dnums);
    }

    std::vector<T> input_elems(ShapeUtil::ElementsIn(input_shape),
                               static_cast<T>(1.0f));
    auto input_r1 = LiteralUtil::CreateR1<T>(input_elems);
    auto input_r3 = input_r1.Reshape(input_dims).value();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(1.0f));

    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r3 = filter_r1.Reshape(filter_dims).value();

    std::vector<T> expect_elems(batch * output_feature * num_windows,
                                static_cast<T>(window_size * input_feature));
    auto expected_r1 = LiteralUtil::CreateR1<T>(expect_elems);
    auto expected_r3 =
        expected_r1.Reshape({batch, num_windows, output_feature}).value();

    ComputeAndCompareLiteral(&builder, expected_r3, {&input_r3, &filter_r3},
                             kErrorSpec);
  }
};

class Convolve1D1WindowTestFloat : public Convolve1D1WindowTestBase {};

XLA_TEST_P(Convolve1D1WindowTestFloat, Convolve1D1Window) { TestImpl<float>(); }

INSTANTIATE_TEST_CASE_P(
    Convolve1D1WindowTest_Instantiation, Convolve1D1WindowTestFloat,
    ::testing::Values(Convolve1DTestParam{1, 1, 1, 1, 2},
                      Convolve1DTestParam{160, 1, 1, 5, 1},
                      Convolve1DTestParam{24, 1, 1, 20, 1},
                      Convolve1DTestParam{30, 1, 1, 20, 1},
                      Convolve1DTestParam{23, 1, 1, 20, 20},
                      Convolve1DTestParam{25, 1, 1, 20, 1},
                      Convolve1DTestParam{24, 1, 1, 10, 5},
                      Convolve1DTestParam{160, 1, 1, 10, 1},
                      Convolve1DTestParam{255, 1, 1, 3, 1},
                      Convolve1DTestParam{130, 1, 1, 1, 2},
                      Convolve1DTestParam{136, 1, 1, 1, 2},
                      Convolve1DTestParam{64, 1, 1, 1, 1},
                      Convolve1DTestParam{128, 1, 1, 1, 1},
                      Convolve1DTestParam{139, 1, 1, 128, 1},
                      Convolve1DTestParam{1, 10, 10, 1, 10},
                      Convolve1DTestParam{1, 10, 130, 1, 2},
                      Convolve1DTestParam{1, 10, 130, 1, 1},
                      Convolve1DTestParam{1, 64, 64, 1, 10},
                      Convolve1DTestParam{1, 65, 65, 1, 1},
                      Convolve1DTestParam{1, 128, 128, 1, 1},
                      Convolve1DTestParam{128, 128, 128, 128, 1},
                      Convolve1DTestParam{1, 128, 128, 1, 1},
                      Convolve1DTestParam{2, 2, 2, 2, 1},
                      Convolve1DTestParam{161, 1, 1, 10, 1},
                      Convolve1DTestParam{900, 1, 1, 10, 1},
                      Convolve1DTestParam{640, 3, 3, 128, 1})

);

#if (XLA_TEST_BACKEND_GPU || XLA_TEST_BACKEND_CPU)
class Convolve1D1WindowTestHalf : public Convolve1D1WindowTestBase {};

XLA_TEST_P(Convolve1D1WindowTestHalf, Convolve1D1Window) {
  TestImpl<Eigen::half>();
}

INSTANTIATE_TEST_CASE_P(
    Convolve1D1WindowTest_Instantiation, Convolve1D1WindowTestHalf,
    ::testing::Values(Convolve1DTestParam{1, 1, 1, 1, 2},
                      Convolve1DTestParam{160, 1, 1, 5, 1},
                      Convolve1DTestParam{24, 1, 1, 20, 1},
                      Convolve1DTestParam{30, 1, 1, 20, 1},
                      Convolve1DTestParam{23, 1, 1, 20, 20},
                      Convolve1DTestParam{25, 1, 1, 20, 1},
                      Convolve1DTestParam{24, 1, 1, 10, 5},
                      Convolve1DTestParam{160, 1, 1, 10, 1},
                      Convolve1DTestParam{255, 1, 1, 3, 1},
                      Convolve1DTestParam{130, 1, 1, 1, 3},
                      Convolve1DTestParam{64, 1, 1, 1, 1},
                      Convolve1DTestParam{128, 1, 1, 1, 1},
                      Convolve1DTestParam{139, 1, 1, 128, 1},
                      Convolve1DTestParam{640, 3, 3, 128, 1},
                      // Convolve1DTestParam{900, 1, 1, 10, 1}, b/195348220
                      Convolve1DTestParam{1, 10, 10, 1, 10},
                      Convolve1DTestParam{1, 10, 130, 1, 1},
                      Convolve1DTestParam{1, 10, 130, 1, 2},
                      Convolve1DTestParam{1, 64, 64, 1, 10},
                      Convolve1DTestParam{1, 65, 65, 1, 1},
                      Convolve1DTestParam{1, 128, 128, 1, 1},
                      Convolve1DTestParam{128, 128, 128, 128, 1},
                      Convolve1DTestParam{1, 128, 128, 1, 1},
                      Convolve1DTestParam{2, 2, 2, 2, 1},
                      Convolve1DTestParam{161, 1, 1, 10, 1})

);
#endif

TEST_F(ConvolutionTest, Convolve1D_1x2x5_1x2x2_Valid) {
  XlaBuilder builder(TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {1, 2, 5});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto filter = Parameter(&builder, 1, filter_shape, "filter");
    Conv(input, filter, {1}, Padding::kValid);
  }

  Array3D<float> input({{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}});
  Array3D<float> filter({{{10, 20}, {30, 40}}});

  Array3D<float> expected({{{510, 610, 710, 810}}});

  const Literal input_literal = LiteralUtil::CreateR3FromArray3D(input);
  const Literal filter_literal = LiteralUtil::CreateR3FromArray3D(filter);

  ComputeAndCompareR3<float>(&builder, expected,
                             {&input_literal, &filter_literal}, kErrorSpec);
}

template <typename T>
class Convolve1D_1x2x5_1x2x2_WithRHSDilation : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    {
      Shape input_shape = ShapeUtil::MakeShapeWithType<T>({1, 2, 5});
      Shape filter_shape = ShapeUtil::MakeShapeWithType<T>({1, 2, 2});
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");
      // Convolution dimensions are bf0_oi0->bf0.
      ConvGeneralDilated(
          input, filter, /*window_strides=*/{1}, /*padding=*/{{0, 0}},
          /*lhs_dilation=*/{1}, /*rhs_dilation=*/{2},
          /*dimension_numbers=*/builder.CreateDefaultConvDimensionNumbers(1));
    }

    Array3D<T> input(
        {{{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {6.0f, 7.0f, 8.0f, 9.0f, 10.0f}}});
    Array3D<T> filter({{{10.0f, 20.0f}, {30.0f, 40.0f}}});

    Array3D<T> expected({{{570.0f, 670.0f, 770.0f}}});

    const Literal input_literal = LiteralUtil::CreateR3FromArray3D(input);
    const Literal filter_literal = LiteralUtil::CreateR3FromArray3D(filter);

    ComputeAndCompareR3<T>(&builder, expected,
                           {&input_literal, &filter_literal}, kErrorSpec);
  }
};  // namespace

TYPED_TEST_CASE(Convolve1D_1x2x5_1x2x2_WithRHSDilation, TestTypes);
TYPED_TEST(Convolve1D_1x2x5_1x2x2_WithRHSDilation, Types) { this->RunTest(); }

// Basic test with LHS dilation (i.e. strided transposed convolution).
TEST_F(ConvolutionTest, Convolve1D_1x1x5_1x1x3_WithLHSDilation_FullPadding) {
  XlaBuilder builder(TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {1, 1, 5});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 1, 3});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto filter = Parameter(&builder, 1, filter_shape, "filter");
    // Convolution dimensions are bf0_oi0->bf0.
    ConvGeneralDilated(
        input, filter, /*window_strides=*/{1}, /*padding=*/{{0, 0}},
        /*lhs_dilation=*/{2}, /*rhs_dilation=*/{1},
        /*dimension_numbers=*/builder.CreateDefaultConvDimensionNumbers(1));
  }

  Array3D<float> input({{{1, 2, 3, 4, 5}}});
  Array3D<float> filter({{{10, 11, 12}}});

  Array3D<float> expected({{{34, 22, 56, 33, 78, 44, 100}}});

  const Literal input_literal = LiteralUtil::CreateR3FromArray3D(input);
  const Literal filter_literal = LiteralUtil::CreateR3FromArray3D(filter);

  ComputeAndCompareR3<float>(&builder, expected,
                             {&input_literal, &filter_literal}, kErrorSpec);
}

TEST_F(ConvolutionTest, Convolve1D_1x1x5_1x1x3_WithLHSDilation_NoPadding) {
  XlaBuilder builder(TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {1, 1, 5});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 1, 3});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto filter = Parameter(&builder, 1, filter_shape, "filter");
    // Convolution dimensions are bf0_oi0->bf0.
    ConvGeneralDilated(
        input, filter, /*window_strides=*/{1}, /*padding=*/{{2, 2}},
        /*lhs_dilation=*/{2}, /*rhs_dilation=*/{1},
        /*dimension_numbers=*/builder.CreateDefaultConvDimensionNumbers(1));
  }

  Array3D<float> input({{{1, 2, 3, 4, 5}}});
  Array3D<float> filter({{{10, 11, 12}}});
  Array3D<float> expected({{{12, 11, 34, 22, 56, 33, 78, 44, 100, 55, 50}}});

  const Literal input_literal = LiteralUtil::CreateR3FromArray3D(input);
  const Literal filter_literal = LiteralUtil::CreateR3FromArray3D(filter);

  ComputeAndCompareR3<float>(&builder, expected,
                             {&input_literal, &filter_literal}, kErrorSpec);
}

TEST_F(ConvolutionTest, Convolve1D_1x1x5_1x1x3_WithLHSDilation_HalfPadding) {
  XlaBuilder builder(TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {1, 1, 5});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 1, 3});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto filter = Parameter(&builder, 1, filter_shape, "filter");
    // Convolution dimensions are bf0_oi0->bf0.
    ConvGeneralDilated(
        input, filter, /*window_strides=*/{1}, /*padding=*/{{1, 1}},
        /*lhs_dilation=*/{2}, /*rhs_dilation=*/{1},
        /*dimension_numbers=*/builder.CreateDefaultConvDimensionNumbers(1));
  }

  Array3D<float> input({{{1, 2, 3, 4, 5}}});
  Array3D<float> filter({{{10, 11, 12}}});
  Array3D<float> expected({{{11, 34, 22, 56, 33, 78, 44, 100, 55}}});

  const Literal input_literal = LiteralUtil::CreateR3FromArray3D(input);
  const Literal filter_literal = LiteralUtil::CreateR3FromArray3D(filter);

  ComputeAndCompareR3<float>(&builder, expected,
                             {&input_literal, &filter_literal}, kErrorSpec);
}

// Test multiple output channels.
TEST_F(ConvolutionTest, Convolve1D_1x1x5_2x1x3_WithLHSDilation) {
  XlaBuilder builder(TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {1, 1, 5});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {2, 1, 3});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto filter = Parameter(&builder, 1, filter_shape, "filter");
    // Convolution dimensions are bf0_oi0->bf0.
    ConvGeneralDilated(
        input, filter, /*window_strides=*/{1}, /*padding=*/{{0, 0}},
        /*lhs_dilation=*/{2}, /*rhs_dilation=*/{1},
        /*dimension_numbers=*/builder.CreateDefaultConvDimensionNumbers(1));
  }

  Array3D<float> input({{{1, 2, 3, 4, 5}}});
  Array3D<float> filter({{{10, 11, 12}}, {{20, 22, 24}}});
  Array3D<float> expected(
      {{{34, 22, 56, 33, 78, 44, 100}, {68, 44, 112, 66, 156, 88, 200}}});

  const Literal input_literal = LiteralUtil::CreateR3FromArray3D(input);
  const Literal filter_literal = LiteralUtil::CreateR3FromArray3D(filter);

  ComputeAndCompareR3<float>(&builder, expected,
                             {&input_literal, &filter_literal}, kErrorSpec);
}

// Test multiple input channels.
TEST_F(ConvolutionTest, Convolve1D_1x2x5_1x2x3_WithLHSDilation) {
  XlaBuilder builder(TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {1, 2, 5});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 2, 3});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto filter = Parameter(&builder, 1, filter_shape, "filter");
    // Convolution dimensions are bf0_oi0->bf0.
    ConvGeneralDilated(
        input, filter, /*window_strides=*/{1}, /*padding=*/{{0, 0}},
        /*lhs_dilation=*/{2}, /*rhs_dilation=*/{1},
        /*dimension_numbers=*/builder.CreateDefaultConvDimensionNumbers(1));
  }

  Array3D<float> input({{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}});
  Array3D<float> filter({{{10, 20, 30}, {40, 50, 60}}});

  Array3D<float> expected({{{730, 390, 870, 460, 1010, 530, 1150}}});

  const Literal input_literal = LiteralUtil::CreateR3FromArray3D(input);
  const Literal filter_literal = LiteralUtil::CreateR3FromArray3D(filter);

  ComputeAndCompareR3<float>(&builder, expected,
                             {&input_literal, &filter_literal}, kErrorSpec);
}

// Batched version of the above test.
TEST_F(ConvolutionTest, Convolve1D_3x2x5_1x2x3_WithLHSDilation) {
  XlaBuilder builder(TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {3, 2, 5});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 2, 3});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto filter = Parameter(&builder, 1, filter_shape, "filter");
    // Convolution dimensions are bf0_oi0->bf0.
    ConvGeneralDilated(
        input, filter, /*window_strides=*/{1}, /*padding=*/{{0, 0}},
        /*lhs_dilation=*/{2}, /*rhs_dilation=*/{1},
        /*dimension_numbers=*/builder.CreateDefaultConvDimensionNumbers(1));
  }

  Array3D<float> input({{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                        {{10, 20, 30, 40, 50}, {60, 70, 80, 90, 100}},
                        {{2, 4, 6, 8, 10}, {12, 14, 16, 18, 20}}});

  Array3D<float> filter({{{10, 20, 30}, {40, 50, 60}}});

  Array3D<float> expected({{{730, 390, 870, 460, 1010, 530, 1150}},
                           {{7300, 3900, 8700, 4600, 10100, 5300, 11500}},
                           {{1460, 780, 1740, 920, 2020, 1060, 2300}}});

  const Literal input_literal = LiteralUtil::CreateR3FromArray3D(input);
  const Literal filter_literal = LiteralUtil::CreateR3FromArray3D(filter);

  ComputeAndCompareR3<float>(&builder, expected,
                             {&input_literal, &filter_literal}, kErrorSpec);
}

// Test all together: batched, multiple input and output channels.
TEST_F(ConvolutionTest, Convolve1D_3x2x5_2x2x3_WithLHSDilation) {
  XlaBuilder builder(TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {3, 2, 5});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {2, 2, 3});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto filter = Parameter(&builder, 1, filter_shape, "filter");
    // Convolution dimensions are bf0_oi0->bf0.
    ConvGeneralDilated(
        input, filter, /*window_strides=*/{1}, /*padding=*/{{0, 0}},
        /*lhs_dilation=*/{2}, /*rhs_dilation=*/{1},
        /*dimension_numbers=*/builder.CreateDefaultConvDimensionNumbers(1));
  }

  Array3D<float> input({{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                        {{10, 20, 30, 40, 50}, {60, 70, 80, 90, 100}},
                        {{2, 4, 6, 8, 10}, {12, 14, 16, 18, 20}}});

  Array3D<float> filter(
      {{{10, 20, 30}, {40, 50, 60}}, {{11, 22, 33}, {44, 55, 66}}});

  Array3D<float> expected({{{730, 390, 870, 460, 1010, 530, 1150},
                            {803, 429, 957, 506, 1111, 583, 1265}},
                           {{7300, 3900, 8700, 4600, 10100, 5300, 11500},
                            {8030, 4290, 9570, 5060, 11110, 5830, 12650}},
                           {{1460, 780, 1740, 920, 2020, 1060, 2300},
                            {1606, 858, 1914, 1012, 2222, 1166, 2530}}});

  const Literal input_literal = LiteralUtil::CreateR3FromArray3D(input);
  const Literal filter_literal = LiteralUtil::CreateR3FromArray3D(filter);

  ComputeAndCompareR3<float>(&builder, expected,
                             {&input_literal, &filter_literal}, kErrorSpec);
}

// Test LHS dilation (i.e. transposed convolution) and window strides at the
// same time. That's probably never used in practice, but since the generic
// algorithm covers it, we test it anyway with a simple case.
TEST_F(ConvolutionTest, Convolve1D_1x1x5_1x1x3_WithLHSDilationAndStrides) {
  XlaBuilder builder(TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {1, 1, 5});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 1, 3});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto filter = Parameter(&builder, 1, filter_shape, "filter");
    // Convolution dimensions are bf0_oi0->bf0.
    ConvGeneralDilated(
        input, filter, /*window_strides=*/{2}, /*padding=*/{{0, 0}},
        /*lhs_dilation=*/{2}, /*rhs_dilation=*/{1},
        /*dimension_numbers=*/builder.CreateDefaultConvDimensionNumbers(1));
  }

  Array3D<float> input({{{1, 2, 3, 4, 5}}});
  Array3D<float> filter({{{10, 11, 12}}});

  Array3D<float> expected({{{34, 56, 78, 100}}});

  const Literal input_literal = LiteralUtil::CreateR3FromArray3D(input);
  const Literal filter_literal = LiteralUtil::CreateR3FromArray3D(filter);

  ComputeAndCompareR3<float>(&builder, expected,
                             {&input_literal, &filter_literal}, kErrorSpec);
}

TEST_F(ConvolutionTest, Convolve1D_1x2x5_1x2x2_WithLHSAndRHSDilation) {
  XlaBuilder builder(TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {1, 2, 5});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto filter = Parameter(&builder, 1, filter_shape, "filter");
    // Convolution dimensions are bf0_oi0->bf0.
    ConvGeneralDilated(
        input, filter, /*window_strides=*/{1}, /*padding=*/{{0, 0}},
        /*lhs_dilation=*/{2}, /*rhs_dilation=*/{2},
        /*dimension_numbers=*/builder.CreateDefaultConvDimensionNumbers(1));
  }

  Array3D<float> input({{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}});
  Array3D<float> filter({{{10, 20}, {30, 40}}});

  Array3D<float> expected({{{510, 0, 610, 0, 710, 0, 810}}});

  const Literal input_literal = LiteralUtil::CreateR3FromArray3D(input);
  const Literal filter_literal = LiteralUtil::CreateR3FromArray3D(filter);

  ComputeAndCompareR3<float>(&builder, expected,
                             {&input_literal, &filter_literal}, kErrorSpec);
}

template <typename T>
class Convolve1D_1x2x5_1x2x2_WithPadding : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    {
      Shape input_shape = ShapeUtil::MakeShapeWithType<T>({1, 2, 5});
      Shape filter_shape = ShapeUtil::MakeShapeWithType<T>({1, 2, 2});
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");
      // Convolution dimensions are bf0_oi0->bf0.
      ConvGeneralDilated(
          input, filter, /*window_strides=*/{1}, /*padding=*/{{2, 2}},
          /*lhs_dilation=*/{1}, /*rhs_dilation=*/{1},
          /*dimension_numbers=*/builder.CreateDefaultConvDimensionNumbers(1));
    }

    Array3D<T> input(
        {{{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {6.0f, 7.0f, 8.0f, 9.0f, 10.0f}}});
    Array3D<T> filter({{{10.0f, 20.0f}, {30.0f, 40.0f}}});

    Array3D<T> expected(
        {{{0.0f, 260.0f, 510.0f, 610.0f, 710.0f, 810.0f, 350.0f, 0.0f}}});

    const Literal input_literal = LiteralUtil::CreateR3FromArray3D(input);
    const Literal filter_literal = LiteralUtil::CreateR3FromArray3D(filter);

    ComputeAndCompareR3<T>(&builder, expected,
                           {&input_literal, &filter_literal}, kErrorSpec);
  }
};

TYPED_TEST_CASE(Convolve1D_1x2x5_1x2x2_WithPadding, TestTypes);
TYPED_TEST(Convolve1D_1x2x5_1x2x2_WithPadding, Types) { this->RunTest(); }

}  // namespace
}  // namespace xla
