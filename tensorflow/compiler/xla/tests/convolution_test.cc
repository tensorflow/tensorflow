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

// Tests of convolution with trivial kernels and no special variations (like
// strides and padding).

#include <memory>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class ConvolutionTest : public ClientLibraryTestBase {
 protected:
#if XLA_TEST_BACKEND_GPU
  // XLA:GPU sometimes uses FFT convolution which isn't as precise as spatial
  // convolution. So relax the absolute error threshold.
  ErrorSpec error_spec_ = ErrorSpec(1e-2, 1e-4);
#else
  ErrorSpec error_spec_ = ErrorSpec(1e-4, 1e-4);
#endif
};

#ifdef XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16
using TestTypes = ::testing::Types<float>;
#else
using TestTypes = ::testing::Types<float, Eigen::half>;
#endif

template <typename T>
class ForwardPassConvolution_3x3x256_256_OutputZ_Iota : public ConvolutionTest {
 public:
  void RunTest() {
    const int kInputActivationSizeY = 3;
    const int kInputActivationSizeX = 3;
    const int kInputActivationSizeZ = 256;
    const int kKernelSizeX = 2;
    const int kKernelSizeY = 2;
    const int kOutputActivationSizeZ = 256;
    const int kMiniBatchSize = 4;
    auto alhs = absl::make_unique<Array4D<T>>(
        kMiniBatchSize, kInputActivationSizeZ, kInputActivationSizeY,
        kInputActivationSizeX);
    alhs->FillWithMultiples(static_cast<T>(1.0f));
    ASSERT_EQ(3, alhs->width());
    ASSERT_EQ(3, alhs->height());

    auto arhs = absl::make_unique<Array4D<T>>(kOutputActivationSizeZ,
                                              kInputActivationSizeZ,
                                              kKernelSizeY, kKernelSizeX);
    Array2D<T> rhs_raster({
        {1.0f, 0.0f},  // row 0
        {0.0f, 0.0f},  // row 1
    });
    arhs->FillWithYX(rhs_raster);
    ASSERT_EQ(2, arhs->width());
    ASSERT_EQ(2, arhs->height());

    XlaBuilder builder(TestName());
    auto lhs = ConstantR4FromArray4D<T>(&builder, *alhs);
    auto rhs = ConstantR4FromArray4D<T>(&builder, *arhs);
    PrecisionConfig precision;
    // The left hand side of the convolution is numbers between 0 and 2304 which
    // requires at least 11 mantissa bits and the DEFAULT precision config is
    // allowed to round to bfloat16 which only has 7 mantissa bits.
    precision.add_operand_precision(PrecisionConfig::HIGHEST);
    precision.add_operand_precision(PrecisionConfig::DEFAULT);
    Conv(lhs, rhs, {1, 1}, Padding::kValid, /*feature_group_count=*/1,
         /*batch_group_count=*/1, &precision);

    ComputeAndCompare(&builder, {}, error_spec_);
  }
};

TYPED_TEST_CASE(ForwardPassConvolution_3x3x256_256_OutputZ_Iota, TestTypes);
XLA_TYPED_TEST(ForwardPassConvolution_3x3x256_256_OutputZ_Iota, Types) {
  this->RunTest();
}

template <typename T>
class Convolve_1x1x1x2_1x1x1x2_Valid : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>({1, 1, 1, 2});
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>({1, 1, 1, 2});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto filter = Parameter(&builder, 1, filter_shape, "filter");
    Conv(input, filter, {1, 1}, Padding::kValid);

    Array4D<T> input_data(1, 1, 1, 2);
    input_data.FillWithYX(Array2D<T>({
        {1.0f, 2.0f},
    }));
    Array4D<T> filter_data(1, 1, 1, 2);
    filter_data.FillWithYX(Array2D<T>({
        {5.0f, 6.0f},
    }));

    ComputeAndCompare(&builder,
                      {LiteralUtil::CreateFromArray(input_data),
                       LiteralUtil::CreateFromArray(filter_data)},
                      error_spec_);
  }
};

TYPED_TEST_CASE(Convolve_1x1x1x2_1x1x1x2_Valid, TestTypes);
TYPED_TEST(Convolve_1x1x1x2_1x1x1x2_Valid, Types) { this->RunTest(); }

// Tests valid padding for 2D convolution in raster space.
template <typename T>
class Convolve_1x1x4x4_1x1x2x2_Valid : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>({1, 1, 4, 4});
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>({1, 1, 2, 2});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto filter = Parameter(&builder, 1, filter_shape, "filter");
    Conv(input, filter, {1, 1}, Padding::kValid);

    Array4D<T> input_data(1, 1, 4, 4);
    input_data.FillWithYX(Array2D<T>({
        {1.0f, 2.0f, 3.0f, 4.0f},
        {5.0f, 6.0f, 7.0f, 8.0f},
        {9.0f, 10.0f, 11.0f, 12.0f},
        {13.0f, 14.0f, 15.0f, 16.0f},
    }));
    Array4D<T> filter_data(1, 1, 2, 2);
    filter_data.FillWithYX(Array2D<T>({
        {5.0f, 6.0f},
        {7.0f, 8.0f},
    }));
    ComputeAndCompare(&builder,
                      {LiteralUtil::CreateFromArray(input_data),
                       LiteralUtil::CreateFromArray(filter_data)},
                      error_spec_);
  }
};

TYPED_TEST_CASE(Convolve_1x1x4x4_1x1x2x2_Valid, TestTypes);
TYPED_TEST(Convolve_1x1x4x4_1x1x2x2_Valid, Types) { this->RunTest(); }

// Tests same padding for 2D convolution in raster space.
template <typename T>
class Convolve_1x1x4x4_1x1x2x2_Same : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>({1, 1, 4, 4});
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>({1, 1, 2, 2});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto filter = Parameter(&builder, 1, filter_shape, "filter");
    Conv(input, filter, {1, 1}, Padding::kSame);

    Array4D<T> input_data(1, 1, 4, 4);
    input_data.FillWithYX(Array2D<T>({
        {1.0f, 2.0f, 3.0f, 4.0f},
        {5.0f, 6.0f, 7.0f, 8.0f},
        {9.0f, 10.0f, 11.0f, 12.0f},
        {13.0f, 14.0f, 15.0f, 16.0f},
    }));
    Array4D<T> filter_data(1, 1, 2, 2);
    filter_data.FillWithYX(Array2D<T>({
        {5.0f, 6.0f},
        {7.0f, 8.0f},
    }));

    ComputeAndCompare(&builder,
                      {LiteralUtil::CreateFromArray(input_data),
                       LiteralUtil::CreateFromArray(filter_data)},
                      error_spec_);
  }
};

TYPED_TEST_CASE(Convolve_1x1x4x4_1x1x2x2_Same, TestTypes);
TYPED_TEST(Convolve_1x1x4x4_1x1x2x2_Same, Types) { this->RunTest(); }

// Tests same padding for 2D convolution in raster space with an odd sized
// kernel.
template <typename T>
class Convolve_1x1x4x4_1x1x3x3_Same : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>({1, 1, 4, 4});
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>({1, 1, 3, 3});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto filter = Parameter(&builder, 1, filter_shape, "filter");
    Conv(input, filter, {1, 1}, Padding::kSame);

    Array4D<T> input_data(1, 1, 4, 4);
    input_data.FillWithYX(Array2D<T>({{1.0f, 2.0f, 3.0f, 4.0f},
                                      {5.0f, 6.0f, 7.0f, 8.0f},
                                      {9.0f, 10.0f, 11.0f, 12.0f},
                                      {13.0f, 14.0f, 15.0f, 16.0f}}));
    Array4D<T> filter_data(1, 1, 3, 3);
    filter_data.FillWithYX(Array2D<T>(
        {{5.0f, 6.0f, 7.0f}, {8.0f, 9.0f, 10.0f}, {11.0f, 12.0f, 13.0f}}));
    // clang-format on
    ComputeAndCompare(&builder,
                      {LiteralUtil::CreateFromArray(input_data),
                       LiteralUtil::CreateFromArray(filter_data)},
                      error_spec_);
  }
};

TYPED_TEST_CASE(Convolve_1x1x4x4_1x1x3x3_Same, TestTypes);
TYPED_TEST(Convolve_1x1x4x4_1x1x3x3_Same, Types) { this->RunTest(); }

XLA_TEST_F(ConvolutionTest, Convolve1D_1x2x5_1x2x2_Valid) {
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

  auto input_literal =
      client_->TransferToServer(LiteralUtil::CreateR3FromArray3D(input))
          .ConsumeValueOrDie();
  auto filter_literal =
      client_->TransferToServer(LiteralUtil::CreateR3FromArray3D(filter))
          .ConsumeValueOrDie();

  ComputeAndCompareR3<float>(&builder, expected,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
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
      // Convolution dimensions are bf0_oi0->bo0.
      ConvGeneralDilated(
          input, filter, /*window_strides=*/{1}, /*padding=*/{{0, 0}},
          /*lhs_dilation=*/{1}, /*rhs_dilation=*/{2},
          /*dimension_numbers=*/builder.CreateDefaultConvDimensionNumbers(1));
    }

    Array3D<T> input(
        {{{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {6.0f, 7.0f, 8.0f, 9.0f, 10.0f}}});
    Array3D<T> filter({{{10.0f, 20.0f}, {30.0f, 40.0f}}});

    Array3D<T> expected({{{570.0f, 670.0f, 770.0f}}});

    auto input_literal =
        client_->TransferToServer(LiteralUtil::CreateR3FromArray3D(input))
            .ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(LiteralUtil::CreateR3FromArray3D(filter))
            .ConsumeValueOrDie();

    ComputeAndCompareR3<T>(&builder, expected,
                           {input_literal.get(), filter_literal.get()},
                           error_spec_);
  }
};  // namespace

TYPED_TEST_CASE(Convolve1D_1x2x5_1x2x2_WithRHSDilation, TestTypes);
TYPED_TEST(Convolve1D_1x2x5_1x2x2_WithRHSDilation, Types) { this->RunTest(); }

XLA_TEST_F(ConvolutionTest, Convolve1D_1x2x5_1x2x2_WithLHSDilation) {
  XlaBuilder builder(TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {1, 2, 5});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto filter = Parameter(&builder, 1, filter_shape, "filter");
    // Convolution dimensions are bf0_oi0->bo0.
    ConvGeneralDilated(
        input, filter, /*window_strides=*/{1}, /*padding=*/{{0, 0}},
        /*lhs_dilation=*/{2}, /*rhs_dilation=*/{1},
        /*dimension_numbers=*/builder.CreateDefaultConvDimensionNumbers(1));
  }

  Array3D<float> input({{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}});
  Array3D<float> filter({{{10, 20}, {30, 40}}});

  Array3D<float> expected({{{190, 320, 230, 380, 270, 440, 310, 500}}});

  auto input_literal =
      client_->TransferToServer(LiteralUtil::CreateR3FromArray3D(input))
          .ConsumeValueOrDie();
  auto filter_literal =
      client_->TransferToServer(LiteralUtil::CreateR3FromArray3D(filter))
          .ConsumeValueOrDie();

  ComputeAndCompareR3<float>(&builder, expected,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
}

XLA_TEST_F(ConvolutionTest, Convolve1D_1x2x5_1x2x2_WithLHSAndRHSDilation) {
  XlaBuilder builder(TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {1, 2, 5});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto filter = Parameter(&builder, 1, filter_shape, "filter");
    // Convolution dimensions are bf0_oi0->bo0.
    ConvGeneralDilated(
        input, filter, /*window_strides=*/{1}, /*padding=*/{{0, 0}},
        /*lhs_dilation=*/{2}, /*rhs_dilation=*/{2},
        /*dimension_numbers=*/builder.CreateDefaultConvDimensionNumbers(1));
  }

  Array3D<float> input({{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}});
  Array3D<float> filter({{{10, 20}, {30, 40}}});

  Array3D<float> expected({{{510, 0, 610, 0, 710, 0, 810}}});

  auto input_literal =
      client_->TransferToServer(LiteralUtil::CreateR3FromArray3D(input))
          .ConsumeValueOrDie();
  auto filter_literal =
      client_->TransferToServer(LiteralUtil::CreateR3FromArray3D(filter))
          .ConsumeValueOrDie();

  ComputeAndCompareR3<float>(&builder, expected,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
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
      // Convolution dimensions are bf0_oi0->bo0.
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

    auto input_literal =
        client_->TransferToServer(LiteralUtil::CreateR3FromArray3D(input))
            .ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(LiteralUtil::CreateR3FromArray3D(filter))
            .ConsumeValueOrDie();

    ComputeAndCompareR3<T>(&builder, expected,
                           {input_literal.get(), filter_literal.get()},
                           error_spec_);
  }
};

TYPED_TEST_CASE(Convolve1D_1x2x5_1x2x2_WithPadding, TestTypes);
TYPED_TEST(Convolve1D_1x2x5_1x2x2_WithPadding, Types) { this->RunTest(); }

XLA_TEST_F(ConvolutionTest, Convolve3D_1x4x2x3x3_2x2x2x3x3_Valid) {
  XlaBuilder builder(TestName());
  std::vector<int64> input_dims = {1, 4, 2, 3, 3};
  std::vector<int64> filter_dims = {2, 2, 2, 3, 3};
  Shape input_shape = ShapeUtil::MakeShape(F32, input_dims);
  Shape filter_shape = ShapeUtil::MakeShape(F32, filter_dims);
  {
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto filter = Parameter(&builder, 1, filter_shape, "filter");

    // Tensorflow dimension numbers for 3D convolution.
    ConvolutionDimensionNumbers dnums;
    dnums.set_input_batch_dimension(0);
    dnums.set_output_batch_dimension(0);
    dnums.add_input_spatial_dimensions(1);
    dnums.add_output_spatial_dimensions(1);
    dnums.add_input_spatial_dimensions(2);
    dnums.add_output_spatial_dimensions(2);
    dnums.add_input_spatial_dimensions(3);
    dnums.add_output_spatial_dimensions(3);
    dnums.set_input_feature_dimension(4);
    dnums.set_output_feature_dimension(4);
    dnums.add_kernel_spatial_dimensions(0);
    dnums.add_kernel_spatial_dimensions(1);
    dnums.add_kernel_spatial_dimensions(2);
    dnums.set_kernel_input_feature_dimension(3);
    dnums.set_kernel_output_feature_dimension(4);

    ConvWithGeneralDimensions(input, filter, {1, 1, 1}, Padding::kValid, dnums);
  }

  std::vector<float> input_elems(ShapeUtil::ElementsIn(input_shape));
  iota(input_elems.begin(), input_elems.end(), 1.0f);
  auto input_r1 = LiteralUtil::CreateR1<float>(input_elems);
  auto input_r5 = input_r1.Reshape(input_dims).ConsumeValueOrDie();

  std::vector<float> filter_elems(ShapeUtil::ElementsIn(filter_shape));
  iota(filter_elems.begin(), filter_elems.end(), 1.0f);
  auto filter_r1 = LiteralUtil::CreateR1<float>(filter_elems);
  auto filter_r5 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();

  auto expected_r1 = LiteralUtil::CreateR1<float>(
      {19554, 19962, 20370, 22110, 22590, 23070, 34890, 35730, 36570, 37446,
       38358, 39270, 50226, 51498, 52770, 52782, 54126, 55470});
  auto expected_r5 = expected_r1.Reshape({1, 3, 1, 2, 3}).ConsumeValueOrDie();

  auto input_literal = client_->TransferToServer(input_r5).ConsumeValueOrDie();
  auto filter_literal =
      client_->TransferToServer(filter_r5).ConsumeValueOrDie();

  ComputeAndCompareLiteral(&builder, expected_r5,
                           {input_literal.get(), filter_literal.get()},
                           error_spec_);
}

// std::iota doesn't work when init_value has a type Eigen::half in some build
// servers. The error message is missing the operator ++.
template <typename T>
void iota_int_init_value(std::vector<T>& values, int init_value) {
  absl::c_for_each(values,
                   [&](T& value) { value = static_cast<T>(init_value++); });
}

template <typename T>
class Convolve2D_1x3x3x5_3x3x5x3_Valid : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64> input_dims = {1, 3, 3, 5};
    std::vector<int64> filter_dims = {3, 3, 5, 3};
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>(input_dims);
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>(filter_dims);
    {
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");

      // Tensorflow dimension numbers for 2D convolution.
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

      ConvWithGeneralDimensions(input, filter, {1, 1}, Padding::kValid, dnums);
    }

    std::vector<T> input_elems(ShapeUtil::ElementsIn(input_shape));
    iota_int_init_value(input_elems, 1);
    auto input_r1 = LiteralUtil::CreateR1<T>(input_elems);
    auto input_r4 = input_r1.Reshape(input_dims).ConsumeValueOrDie();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape));
    iota_int_init_value(filter_elems, 1);
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();

    auto expected_r1 = LiteralUtil::CreateR1<T>(
        {static_cast<T>(92115), static_cast<T>(93150), static_cast<T>(94185)});
    auto expected_r4 = expected_r1.Reshape({1, 1, 1, 3}).ConsumeValueOrDie();

    auto input_literal =
        client_->TransferToServer(input_r4).ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(filter_r4).ConsumeValueOrDie();

    ComputeAndCompareLiteral(&builder, expected_r4,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
  }
};

TYPED_TEST_CASE(Convolve2D_1x3x3x5_3x3x5x3_Valid, TestTypes);
TYPED_TEST(Convolve2D_1x3x3x5_3x3x5x3_Valid, Types) { this->RunTest(); }

template <typename T>
class Convolve2D_1x3x3x5_3x3x1x15_Depthwise_Valid : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64> input_dims = {1, 3, 3, 5};
    std::vector<int64> filter_dims = {3, 3, 1, 15};
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>(input_dims);
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>(filter_dims);
    {
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");

      // Tensorflow dimension numbers for 2D convolution.
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

      ConvWithGeneralDimensions(input, filter, {1, 1}, Padding::kValid, dnums,
                                /*feature_group_count=*/5);
    }

    std::vector<T> input_elems(ShapeUtil::ElementsIn(input_shape));
    iota_int_init_value(input_elems, 1);
    auto input_r1 = LiteralUtil::CreateR1<T>(input_elems);
    auto input_r4 = input_r1.Reshape(input_dims).ConsumeValueOrDie();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape));
    iota_int_init_value(filter_elems, 1);
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();

    auto expected_r1 = LiteralUtil::CreateR1<T>(
        {static_cast<T>(16029), static_cast<T>(16218), static_cast<T>(16407),
         static_cast<T>(17172), static_cast<T>(17370), static_cast<T>(17568),
         static_cast<T>(18369), static_cast<T>(18576), static_cast<T>(18783),
         static_cast<T>(19620), static_cast<T>(19836), static_cast<T>(20052),
         static_cast<T>(20925), static_cast<T>(21150), static_cast<T>(21375)});
    auto expected_r4 = expected_r1.Reshape({1, 1, 1, 15}).ConsumeValueOrDie();

    auto input_literal =
        client_->TransferToServer(input_r4).ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(filter_r4).ConsumeValueOrDie();

    ComputeAndCompareLiteral(&builder, expected_r4,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
  }
};

TYPED_TEST_CASE(Convolve2D_1x3x3x5_3x3x1x15_Depthwise_Valid, TestTypes);
TYPED_TEST(Convolve2D_1x3x3x5_3x3x1x15_Depthwise_Valid, Types) {
  this->RunTest();
}

template <typename T>
class Convolve2D_1x4x4x5_3x3x1x5_Depthwise_Valid : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64> input_dims = {1, 4, 4, 5};
    std::vector<int64> filter_dims = {3, 3, 1, 5};
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>(input_dims);
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>(filter_dims);
    {
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");

      // Tensorflow dimension numbers for 2D convolution.
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

      ConvWithGeneralDimensions(input, filter, {1, 1}, Padding::kValid, dnums,
                                /*feature_group_count=*/5);
    }

    std::vector<T> input_elems(ShapeUtil::ElementsIn(input_shape));
    iota_int_init_value(input_elems, 1);
    auto input_r1 = LiteralUtil::CreateR1<T>(input_elems);
    auto input_r4 = input_r1.Reshape(input_dims).ConsumeValueOrDie();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape));
    iota_int_init_value(filter_elems, 1);
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();

    auto expected_r1 = LiteralUtil::CreateR1<T>(
        {static_cast<T>(6864),  static_cast<T>(7296),  static_cast<T>(7746),
         static_cast<T>(8214),  static_cast<T>(8700),  static_cast<T>(7809),
         static_cast<T>(8286),  static_cast<T>(8781),  static_cast<T>(9294),
         static_cast<T>(9825),  static_cast<T>(10644), static_cast<T>(11256),
         static_cast<T>(11886), static_cast<T>(12534), static_cast<T>(13200),
         static_cast<T>(11589), static_cast<T>(12246), static_cast<T>(12921),
         static_cast<T>(13614), static_cast<T>(14325)});
    auto expected_r4 = expected_r1.Reshape({1, 2, 2, 5}).ConsumeValueOrDie();

    auto input_literal =
        client_->TransferToServer(input_r4).ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(filter_r4).ConsumeValueOrDie();

    ComputeAndCompareLiteral(&builder, expected_r4,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);

    auto filter_r = filter_r1.Reshape(filter_dims);
  }
};

TYPED_TEST_CASE(Convolve2D_1x4x4x5_3x3x1x5_Depthwise_Valid, TestTypes);
TYPED_TEST(Convolve2D_1x4x4x5_3x3x1x5_Depthwise_Valid, Types) {
  this->RunTest();
}

template <typename T>
class Convolve2D_1x4x4x512_3x3x1x512_Depthwise_Valid : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64> input_dims = {1, 4, 4, 512};
    std::vector<int64> filter_dims = {3, 3, 1, 512};
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>(input_dims);
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>(filter_dims);
    {
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");

      // Tensorflow dimension numbers for 2D convolution.
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

      ConvWithGeneralDimensions(input, filter, {1, 1}, Padding::kValid, dnums,
                                /*feature_group_count=*/512);
    }

    std::vector<T> input_elems(ShapeUtil::ElementsIn(input_shape),
                               static_cast<T>(1));
    auto input_r1 = LiteralUtil::CreateR1<T>(input_elems);
    auto input_r4 = input_r1.Reshape(input_dims).ConsumeValueOrDie();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(2));
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();

    std::vector<T> output_elems(2048, static_cast<T>(18));

    auto expected_r1 = LiteralUtil::CreateR1<T>(output_elems);
    auto expected_r4 = expected_r1.Reshape({1, 2, 2, 512}).ConsumeValueOrDie();

    auto input_literal =
        client_->TransferToServer(input_r4).ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(filter_r4).ConsumeValueOrDie();

    ComputeAndCompareLiteral(&builder, expected_r4,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
  }
};

TYPED_TEST_CASE(Convolve2D_1x4x4x512_3x3x1x512_Depthwise_Valid, TestTypes);
TYPED_TEST(Convolve2D_1x4x4x512_3x3x1x512_Depthwise_Valid, Types) {
  this->RunTest();
}

template <typename T>
class Convolve2D_1x4x4x512_3x3x1x512_Depthwise_Valid_Output_Batch_In_Lanes
    : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64> input_dims = {1, 4, 4, 512};
    std::vector<int64> filter_dims = {3, 3, 1, 512};
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>(input_dims);
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>(filter_dims);
    {
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");

      // Tensorflow dimension numbers for 2D convolution.
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

      ConvWithGeneralDimensions(input, filter, {1, 1}, Padding::kValid, dnums,
                                /*feature_group_count=*/512);
    }

    std::vector<T> input_elems(ShapeUtil::ElementsIn(input_shape),
                               static_cast<T>(1));
    auto input_r1 = LiteralUtil::CreateR1<T>(input_elems);
    auto input_r4 = input_r1.Reshape(input_dims).ConsumeValueOrDie();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(2));
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();

    std::vector<T> output_elems(2048, static_cast<T>(18));

    auto expected_r1 = LiteralUtil::CreateR1<T>(output_elems);
    auto expected_r4 = expected_r1.Reshape({1, 2, 2, 512}).ConsumeValueOrDie();
    auto expected_r4_relaid =
        expected_r4.Relayout(LayoutUtil::MakeLayout({0, 3, 2, 1}));

    auto input_literal =
        client_->TransferToServer(input_r4).ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(filter_r4).ConsumeValueOrDie();

    ComputeAndCompareLiteral(&builder, expected_r4_relaid,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_, &expected_r4_relaid.shape());
  }
};

TYPED_TEST_CASE(
    Convolve2D_1x4x4x512_3x3x1x512_Depthwise_Valid_Output_Batch_In_Lanes,
    TestTypes);
TYPED_TEST(Convolve2D_1x4x4x512_3x3x1x512_Depthwise_Valid_Output_Batch_In_Lanes,
           Types) {
  this->RunTest();
}

template <typename T>
class Convolve2D_256x4x4x512_3x3x1x512_Depthwise_Input_Batch_in_Lanes
    : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64> input_dims = {256, 4, 4, 512};
    std::vector<int64> filter_dims = {3, 3, 1, 512};
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>(input_dims);
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>(filter_dims);
    {
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");

      // Tensorflow dimension numbers for 2D convolution.
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

      ConvWithGeneralDimensions(input, filter, {1, 1}, Padding::kValid, dnums,
                                /*feature_group_count=*/512);
    }

    std::vector<T> input_elems(ShapeUtil::ElementsIn(input_shape),
                               static_cast<T>(1));
    auto input_r1 = LiteralUtil::CreateR1<T>(input_elems);
    auto input_r4 = input_r1.Reshape(input_dims).ConsumeValueOrDie();
    auto input_r4_relaid =
        input_r4.Relayout(LayoutUtil::MakeLayout({0, 3, 2, 1}));

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(2));
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();

    std::vector<T> output_elems(2048 * 256, static_cast<T>(18));

    auto expected_r1 = LiteralUtil::CreateR1<T>(output_elems);
    auto expected_r4 =
        expected_r1.Reshape({256, 2, 2, 512}).ConsumeValueOrDie();

    auto input_literal =
        client_->TransferToServer(input_r4_relaid).ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(filter_r4).ConsumeValueOrDie();

    ComputeAndCompareLiteral(&builder, expected_r4,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
  }
};

TYPED_TEST_CASE(Convolve2D_256x4x4x512_3x3x1x512_Depthwise_Input_Batch_in_Lanes,
                TestTypes);
TYPED_TEST(Convolve2D_256x4x4x512_3x3x1x512_Depthwise_Input_Batch_in_Lanes,
           Types) {
  this->RunTest();
}

template <typename T>
class Convolve2D_256x4x4x512_3x3x1x512_Depthwise_Both_Batch_in_Lanes
    : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64> input_dims = {256, 4, 4, 512};
    std::vector<int64> filter_dims = {3, 3, 1, 512};
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>(input_dims);
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>(filter_dims);
    {
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");

      // Tensorflow dimension numbers for 2D convolution.
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

      ConvWithGeneralDimensions(input, filter, {1, 1}, Padding::kValid, dnums,
                                /*feature_group_count=*/512);
    }

    std::vector<T> input_elems(ShapeUtil::ElementsIn(input_shape),
                               static_cast<T>(1));
    auto input_r1 = LiteralUtil::CreateR1<T>(input_elems);
    auto input_r4 = input_r1.Reshape(input_dims).ConsumeValueOrDie();
    auto input_r4_relaid =
        input_r4.Relayout(LayoutUtil::MakeLayout({0, 3, 2, 1}));

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(2));
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();

    std::vector<T> output_elems(2048 * 256, static_cast<T>(18));

    auto expected_r1 = LiteralUtil::CreateR1<T>(output_elems);
    auto expected_r4 =
        expected_r1.Reshape({256, 2, 2, 512}).ConsumeValueOrDie();
    auto expected_r4_relaid =
        expected_r4.Relayout(LayoutUtil::MakeLayout({0, 3, 2, 1}));

    auto input_literal =
        client_->TransferToServer(input_r4_relaid).ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(filter_r4).ConsumeValueOrDie();

    ComputeAndCompareLiteral(&builder, expected_r4_relaid,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_, &expected_r4_relaid.shape());
  }
};

TYPED_TEST_CASE(Convolve2D_256x4x4x512_3x3x1x512_Depthwise_Both_Batch_in_Lanes,
                TestTypes);
TYPED_TEST(Convolve2D_256x4x4x512_3x3x1x512_Depthwise_Both_Batch_in_Lanes,
           Types) {
  this->RunTest();
}

template <typename T>
class Convolve2D_1x4x4x5_3x3x1x5_Depthwise_Valid_Output_Batch_In_Lanes
    : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64> input_dims = {1, 4, 4, 5};
    std::vector<int64> filter_dims = {3, 3, 1, 5};
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>(input_dims);
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>(filter_dims);
    {
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");

      // Tensorflow dimension numbers for 2D convolution.
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

      ConvWithGeneralDimensions(input, filter, {1, 1}, Padding::kValid, dnums,
                                /*feature_group_count=*/5);
    }

    std::vector<T> input_elems(ShapeUtil::ElementsIn(input_shape));
    iota_int_init_value(input_elems, 1);
    auto input_r1 = LiteralUtil::CreateR1<T>(input_elems);
    auto input_r4 = input_r1.Reshape(input_dims).ConsumeValueOrDie();
    auto input_r4_relaid =
        input_r4.Relayout(LayoutUtil::MakeLayout({0, 3, 2, 1}));

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape));
    iota_int_init_value(filter_elems, 1);
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();

    auto expected_r1 = LiteralUtil::CreateR1<T>(
        {static_cast<T>(6864),  static_cast<T>(7296),  static_cast<T>(7746),
         static_cast<T>(8214),  static_cast<T>(8700),  static_cast<T>(7809),
         static_cast<T>(8286),  static_cast<T>(8781),  static_cast<T>(9294),
         static_cast<T>(9825),  static_cast<T>(10644), static_cast<T>(11256),
         static_cast<T>(11886), static_cast<T>(12534), static_cast<T>(13200),
         static_cast<T>(11589), static_cast<T>(12246), static_cast<T>(12921),
         static_cast<T>(13614), static_cast<T>(14325)});
    auto expected_r4 = expected_r1.Reshape({1, 2, 2, 5}).ConsumeValueOrDie();
    auto expected_r4_relaid =
        expected_r4.Relayout(LayoutUtil::MakeLayout({0, 3, 2, 1}));

    auto input_literal =
        client_->TransferToServer(input_r4_relaid).ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(filter_r4).ConsumeValueOrDie();

    ComputeAndCompareLiteral(&builder, expected_r4_relaid,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_, &expected_r4_relaid.shape());
  }
};

TYPED_TEST_CASE(
    Convolve2D_1x4x4x5_3x3x1x5_Depthwise_Valid_Output_Batch_In_Lanes,
    TestTypes);
TYPED_TEST(Convolve2D_1x4x4x5_3x3x1x5_Depthwise_Valid_Output_Batch_In_Lanes,
           Types) {
  this->RunTest();
}

template <typename T>
class Convolve2D_1x4x4x160_3x3x1x160_Depthwise_Valid : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64> input_dims = {1, 4, 4, 160};
    std::vector<int64> filter_dims = {3, 3, 1, 160};
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>(input_dims);
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>(filter_dims);
    {
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");

      // Tensorflow dimension numbers for 2D convolution.
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

      ConvWithGeneralDimensions(input, filter, {1, 1}, Padding::kValid, dnums,
                                /*feature_group_count=*/160);
    }

    std::vector<T> input_elems(ShapeUtil::ElementsIn(input_shape),
                               static_cast<T>(1));
    auto input_r1 = LiteralUtil::CreateR1<T>(input_elems);
    auto input_r4 = input_r1.Reshape(input_dims).ConsumeValueOrDie();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(2));
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();

    std::vector<T> output_elems(640, static_cast<T>(18));

    auto expected_r1 = LiteralUtil::CreateR1<T>(output_elems);
    auto expected_r4 = expected_r1.Reshape({1, 2, 2, 160}).ConsumeValueOrDie();

    auto input_literal =
        client_->TransferToServer(input_r4).ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(filter_r4).ConsumeValueOrDie();

    ComputeAndCompareLiteral(&builder, expected_r4,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
  }
};

TYPED_TEST_CASE(Convolve2D_1x4x4x160_3x3x1x160_Depthwise_Valid, TestTypes);
TYPED_TEST(Convolve2D_1x4x4x160_3x3x1x160_Depthwise_Valid, Types) {
  this->RunTest();
}

template <typename T>
class Convolve2D_1x4x4x160_3x3x1x160_Depthwise_Input_Batch_In_Lanes
    : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64> input_dims = {1, 4, 4, 160};
    std::vector<int64> filter_dims = {3, 3, 1, 160};
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>(input_dims);
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>(filter_dims);
    {
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");

      // Tensorflow dimension numbers for 2D convolution.
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

      ConvWithGeneralDimensions(input, filter, {1, 1}, Padding::kValid, dnums,
                                /*feature_group_count=*/160);
    }

    std::vector<T> input_elems(ShapeUtil::ElementsIn(input_shape),
                               static_cast<T>(1));
    auto input_r1 = LiteralUtil::CreateR1<T>(input_elems);
    auto input_r4 = input_r1.Reshape(input_dims).ConsumeValueOrDie();
    auto input_r4_relaid =
        input_r4.Relayout(LayoutUtil::MakeLayout({0, 3, 2, 1}));

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(2));
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();

    std::vector<T> output_elems(640, static_cast<T>(18));

    auto expected_r1 = LiteralUtil::CreateR1<T>(output_elems);
    auto expected_r4 = expected_r1.Reshape({1, 2, 2, 160}).ConsumeValueOrDie();
    auto expected_r4_relaid =
        expected_r4.Relayout(LayoutUtil::MakeLayout({3, 0, 2, 1}));

    auto input_literal =
        client_->TransferToServer(input_r4_relaid).ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(filter_r4).ConsumeValueOrDie();

    ComputeAndCompareLiteral(&builder, expected_r4_relaid,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_, &expected_r4_relaid.shape());
  }
};

TYPED_TEST_CASE(Convolve2D_1x4x4x160_3x3x1x160_Depthwise_Input_Batch_In_Lanes,
                TestTypes);
TYPED_TEST(Convolve2D_1x4x4x160_3x3x1x160_Depthwise_Input_Batch_In_Lanes,
           Types) {
  this->RunTest();
}

template <typename T>
class Convolve2D_1x4x4x160_3x3x1x160_Dephtwise_Both_Batch_In_Lanes
    : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64> input_dims = {1, 4, 4, 160};
    std::vector<int64> filter_dims = {3, 3, 1, 160};
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>(input_dims);
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>(filter_dims);
    {
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");

      // Tensorflow dimension numbers for 2D convolution.
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

      ConvWithGeneralDimensions(input, filter, {1, 1}, Padding::kValid, dnums,
                                /*feature_group_count=*/160);
    }

    std::vector<T> input_elems(ShapeUtil::ElementsIn(input_shape),
                               static_cast<T>(1));
    auto input_r1 = LiteralUtil::CreateR1<T>(input_elems);
    auto input_r4 = input_r1.Reshape(input_dims).ConsumeValueOrDie();
    auto input_r4_relaid =
        input_r4.Relayout(LayoutUtil::MakeLayout({0, 3, 2, 1}));

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(2));
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();

    std::vector<T> output_elems(640, static_cast<T>(18));

    auto expected_r1 = LiteralUtil::CreateR1<T>(output_elems);
    auto expected_r4 = expected_r1.Reshape({1, 2, 2, 160}).ConsumeValueOrDie();
    auto expected_r4_relaid =
        expected_r4.Relayout(LayoutUtil::MakeLayout({0, 3, 2, 1}));

    auto input_literal =
        client_->TransferToServer(input_r4_relaid).ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(filter_r4).ConsumeValueOrDie();

    ComputeAndCompareLiteral(&builder, expected_r4_relaid,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_, &expected_r4_relaid.shape());
  }
};

TYPED_TEST_CASE(Convolve2D_1x4x4x160_3x3x1x160_Dephtwise_Both_Batch_In_Lanes,
                TestTypes);
TYPED_TEST(Convolve2D_1x4x4x160_3x3x1x160_Dephtwise_Both_Batch_In_Lanes,
           Types) {
  this->RunTest();
}

template <typename T>
class Convolve2D_1x4x4x1024_3x3x1x1024_Depthwise_Valid
    : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64> input_dims = {1, 4, 4, 1024};
    std::vector<int64> filter_dims = {3, 3, 1, 1024};
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>(input_dims);
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>(filter_dims);
    {
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");

      // Tensorflow dimension numbers for 2D convolution.
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

      ConvWithGeneralDimensions(input, filter, {1, 1}, Padding::kValid, dnums,
                                /*feature_group_count=*/1024);
    }

    std::vector<T> input_elems(ShapeUtil::ElementsIn(input_shape),
                               static_cast<T>(1));
    auto input_r1 = LiteralUtil::CreateR1<T>(input_elems);
    auto input_r4 = input_r1.Reshape(input_dims).ConsumeValueOrDie();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(2));
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();

    std::vector<T> output_elems(4096, static_cast<T>(18));

    auto expected_r1 = LiteralUtil::CreateR1<T>(output_elems);
    auto expected_r4 = expected_r1.Reshape({1, 2, 2, 1024}).ConsumeValueOrDie();

    auto input_literal =
        client_->TransferToServer(input_r4).ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(filter_r4).ConsumeValueOrDie();

    ComputeAndCompareLiteral(&builder, expected_r4,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
  }
};

TYPED_TEST_CASE(Convolve2D_1x4x4x1024_3x3x1x1024_Depthwise_Valid, TestTypes);
TYPED_TEST(Convolve2D_1x4x4x1024_3x3x1x1024_Depthwise_Valid, Types) {
  this->RunTest();
}

template <typename T>
class Convolve2D_1x2x2x6_2x2x2x12_Grouped_Valid : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64> input_dims = {1, 2, 2, 6};
    std::vector<int64> filter_dims = {2, 2, 2, 12};
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>(input_dims);
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>(filter_dims);
    {
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");

      // Tensorflow dimension numbers for 2D convolution.
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

      ConvWithGeneralDimensions(input, filter, {1, 1}, Padding::kValid, dnums,
                                /*feature_group_count=*/3);
    }

    std::vector<T> input_elems(ShapeUtil::ElementsIn(input_shape));
    iota_int_init_value(input_elems, 1);
    auto input_r1 = LiteralUtil::CreateR1<T>(input_elems);
    auto input_r4 = input_r1.Reshape(input_dims).ConsumeValueOrDie();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape));
    iota_int_init_value(filter_elems, 1);
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();

    auto expected_r1 = LiteralUtil::CreateR1<T>(
        {static_cast<T>(5076), static_cast<T>(5160), static_cast<T>(5244),
         static_cast<T>(5328), static_cast<T>(6164), static_cast<T>(6264),
         static_cast<T>(6364), static_cast<T>(6464), static_cast<T>(7380),
         static_cast<T>(7496), static_cast<T>(7612), static_cast<T>(7728)});
    auto expected_r4 = expected_r1.Reshape({1, 1, 1, 12}).ConsumeValueOrDie();

    auto input_literal =
        client_->TransferToServer(input_r4).ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(filter_r4).ConsumeValueOrDie();

    ComputeAndCompareLiteral(&builder, expected_r4,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
  }
};

TYPED_TEST_CASE(Convolve2D_1x2x2x6_2x2x2x12_Grouped_Valid, TestTypes);
TYPED_TEST(Convolve2D_1x2x2x6_2x2x2x12_Grouped_Valid, Types) {
  this->RunTest();
}

template <typename T>
class Convolve2D_1x2x2x1024_2x2x128x512_Grouped_Valid : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64> input_dims = {1, 2, 2, 1024};
    std::vector<int64> filter_dims = {2, 2, 128, 512};
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>(input_dims);
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>(filter_dims);
    {
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");

      // Tensorflow dimension numbers for 2D convolution.
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

      ConvWithGeneralDimensions(input, filter, {1, 1}, Padding::kValid, dnums,
                                /*feature_group_count=*/8);
    }

    std::vector<T> input_elems(ShapeUtil::ElementsIn(input_shape),
                               static_cast<T>(1));

    auto input_r1 = LiteralUtil::CreateR1<T>(input_elems);
    auto input_r4 = input_r1.Reshape(input_dims).ConsumeValueOrDie();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(2));

    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();

    std::vector<T> output_elems(512, static_cast<T>(1024));
    auto expected_r1 = LiteralUtil::CreateR1<T>(output_elems);
    auto expected_r4 = expected_r1.Reshape({1, 1, 1, 512}).ConsumeValueOrDie();

    auto input_literal =
        client_->TransferToServer(input_r4).ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(filter_r4).ConsumeValueOrDie();

    ComputeAndCompareLiteral(&builder, expected_r4,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
  }
};

TYPED_TEST_CASE(Convolve2D_1x2x2x1024_2x2x128x512_Grouped_Valid, TestTypes);
TYPED_TEST(Convolve2D_1x2x2x1024_2x2x128x512_Grouped_Valid, Types) {
  this->RunTest();
}

template <typename T>
class Convolve2D_1x2x2x1024_2x2x128x8_Grouped_Valid : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64> input_dims = {1, 2, 2, 1024};
    std::vector<int64> filter_dims = {2, 2, 128, 8};
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>(input_dims);
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>(filter_dims);
    {
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");

      // Tensorflow dimension numbers for 2D convolution.
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

      ConvWithGeneralDimensions(input, filter, {1, 1}, Padding::kValid, dnums,
                                /*feature_group_count=*/8);
    }

    std::vector<T> input_elems(ShapeUtil::ElementsIn(input_shape),
                               static_cast<T>(1));

    auto input_r1 = LiteralUtil::CreateR1<T>(input_elems);
    auto input_r4 = input_r1.Reshape(input_dims).ConsumeValueOrDie();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(2));

    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();

    std::vector<T> output_elems(8, static_cast<T>(1024));
    auto expected_r1 = LiteralUtil::CreateR1<T>(output_elems);
    auto expected_r4 = expected_r1.Reshape({1, 1, 1, 8}).ConsumeValueOrDie();

    auto input_literal =
        client_->TransferToServer(input_r4).ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(filter_r4).ConsumeValueOrDie();

    ComputeAndCompareLiteral(&builder, expected_r4,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
  }
};

TYPED_TEST_CASE(Convolve2D_1x2x2x1024_2x2x128x8_Grouped_Valid, TestTypes);
TYPED_TEST(Convolve2D_1x2x2x1024_2x2x128x8_Grouped_Valid, Types) {
  this->RunTest();
}

template <typename T>
class Convolve2D_1x2x2x12_2x2x3x4_Grouped_Valid : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64> input_dims = {1, 2, 2, 12};
    std::vector<int64> filter_dims = {2, 2, 3, 4};
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>(input_dims);
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>(filter_dims);
    {
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");

      // Tensorflow dimension numbers for 2D convolution.
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

      ConvWithGeneralDimensions(input, filter, {1, 1}, Padding::kValid, dnums,
                                /*feature_group_count=*/4);
    }

    std::vector<T> input_elems(ShapeUtil::ElementsIn(input_shape));
    iota_int_init_value(input_elems, 1);
    auto input_r1 = LiteralUtil::CreateR1<T>(input_elems);
    auto input_r4 = input_r1.Reshape(input_dims).ConsumeValueOrDie();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape));
    iota_int_init_value(filter_elems, 1);
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();

    auto expected_r1 =
        LiteralUtil::CreateR1<T>({static_cast<T>(7712), static_cast<T>(8816),
                                  static_cast<T>(9992), static_cast<T>(11240)});
    auto expected_r4 = expected_r1.Reshape({1, 1, 1, 4}).ConsumeValueOrDie();

    auto input_literal =
        client_->TransferToServer(input_r4).ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(filter_r4).ConsumeValueOrDie();

    ComputeAndCompareLiteral(&builder, expected_r4,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
  }
};

TYPED_TEST_CASE(Convolve2D_1x2x2x12_2x2x3x4_Grouped_Valid, TestTypes);
TYPED_TEST(Convolve2D_1x2x2x12_2x2x3x4_Grouped_Valid, Types) {
  this->RunTest();
}

template <typename T>
class Convolve2D_1x2x2x12_2x2x3x4_Grouped_Valid_Filter_OF_In_Sublanes
    : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64> input_dims = {1, 2, 2, 12};
    std::vector<int64> filter_dims = {2, 2, 4, 3};
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>(input_dims);
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>(filter_dims);
    {
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");

      // Tensorflow dimension numbers for 2D convolution.
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
      dnums.set_kernel_input_feature_dimension(3);
      dnums.set_kernel_output_feature_dimension(2);

      ConvWithGeneralDimensions(input, filter, {1, 1}, Padding::kValid, dnums,
                                /*feature_group_count=*/4);
    }

    std::vector<T> input_elems(ShapeUtil::ElementsIn(input_shape));
    iota_int_init_value(input_elems, 1);
    auto input_r1 = LiteralUtil::CreateR1<T>(input_elems);
    auto input_r4 = input_r1.Reshape(input_dims).ConsumeValueOrDie();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape));
    iota_int_init_value(filter_elems, 1);
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();
    auto filter_r4_relaid =
        filter_r4.Relayout(LayoutUtil::MakeLayout({3, 2, 1, 0}));
    auto expected_r1 = LiteralUtil::CreateR1<T>(
        {static_cast<T>(6968), static_cast<T>(8516), static_cast<T>(10280),
         static_cast<T>(12260)});
    auto expected_r4 = expected_r1.Reshape({1, 1, 1, 4}).ConsumeValueOrDie();

    auto input_literal =
        client_->TransferToServer(input_r4).ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(filter_r4_relaid).ConsumeValueOrDie();

    ComputeAndCompareLiteral(&builder, expected_r4,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
  }
};

TYPED_TEST_CASE(Convolve2D_1x2x2x12_2x2x3x4_Grouped_Valid_Filter_OF_In_Sublanes,
                TestTypes);
TYPED_TEST(Convolve2D_1x2x2x12_2x2x3x4_Grouped_Valid_Filter_OF_In_Sublanes,
           Types) {
  this->RunTest();
}

template <typename T>
class Convolve2D_1x1x1x12_1x1x3x4_Grouped_Valid : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64> input_dims = {1, 1, 1, 12};
    std::vector<int64> filter_dims = {1, 1, 3, 4};
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>(input_dims);
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>(filter_dims);
    {
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");

      // Tensorflow dimension numbers for 2D convolution.
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

      ConvWithGeneralDimensions(input, filter, {1, 1}, Padding::kValid, dnums,
                                /*feature_group_count=*/4);
    }

    std::vector<T> input_elems(ShapeUtil::ElementsIn(input_shape));
    iota_int_init_value(input_elems, 1);
    auto input_r1 = LiteralUtil::CreateR1<T>(input_elems);
    auto input_r4 = input_r1.Reshape(input_dims).ConsumeValueOrDie();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape));
    iota_int_init_value(filter_elems, 1);
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();

    auto expected_r1 =
        LiteralUtil::CreateR1<T>({static_cast<T>(38), static_cast<T>(98),
                                  static_cast<T>(176), static_cast<T>(272)});
    auto expected_r4 = expected_r1.Reshape({1, 1, 1, 4}).ConsumeValueOrDie();

    auto input_literal =
        client_->TransferToServer(input_r4).ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(filter_r4).ConsumeValueOrDie();

    ComputeAndCompareLiteral(&builder, expected_r4,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
  }
};

TYPED_TEST_CASE(Convolve2D_1x1x1x12_1x1x3x4_Grouped_Valid, TestTypes);
TYPED_TEST(Convolve2D_1x1x1x12_1x1x3x4_Grouped_Valid, Types) {
  this->RunTest();
}

// Test fixture to run convolution tests with and without convolution
// canonicalization enabled.
class ConvolveWithAndWithoutCanonicalization
    : public ConvolutionTest,
      public ::testing::WithParamInterface<bool> {};

XLA_TEST_P(ConvolveWithAndWithoutCanonicalization,
           DISABLED_ON_GPU(Convolve2D_NoSpatialDims)) {
  if (GetParam()) {
    execution_options_.mutable_debug_options()->add_xla_disable_hlo_passes(
        "convolution-canonicalization");
  }
  XlaBuilder builder(TestName());
  Shape input_shape = ShapeUtil::MakeShape(F32, {4, 29});
  Shape filter_shape = ShapeUtil::MakeShape(F32, {4, 10});

  auto input = Parameter(&builder, 0, input_shape, "input");
  auto filter = Parameter(&builder, 1, filter_shape, "filter");

  ConvolutionDimensionNumbers dnums;
  dnums.set_input_feature_dimension(0);
  dnums.set_input_batch_dimension(1);
  dnums.set_kernel_input_feature_dimension(0);
  dnums.set_kernel_output_feature_dimension(1);
  dnums.set_output_batch_dimension(0);
  dnums.set_output_feature_dimension(1);
  ConvWithGeneralDimensions(input, filter, {}, Padding::kValid, dnums);

  Array2D<float> param0(4, 29);
  param0.FillUnique();

  Array2D<float> param1(4, 10);
  param1.FillUnique();

  Array2D<float> expected_result(29, 10);
  expected_result.Fill(0);

  ComputeAndCompare(&builder,
                    {LiteralUtil::CreateFromArray(param0),
                     LiteralUtil::CreateFromArray(param1)},
                    error_spec_);
}

INSTANTIATE_TEST_CASE_P(ConvolveWithAndWithoutCanonicalization_Instantiation,
                        ConvolveWithAndWithoutCanonicalization,
                        ::testing::Values(true, false));

struct Convolve1DTestParam {
  int64 input_feature;
  int64 output_feature;
  int64 batch;
  int64 window_size;
  int64 num_windows;
};

class Convolve1D1WindowTestBase
    : public ConvolutionTest,
      public ::testing::WithParamInterface<Convolve1DTestParam> {
 protected:
  template <typename T>
  void TestImpl() {
    XlaBuilder builder(TestName());
    int64 input_feature = GetParam().input_feature;
    int64 output_feature = GetParam().output_feature;
    int64 batch = GetParam().batch;
    int64 num_windows = GetParam().num_windows;
    int64 window_size = GetParam().window_size;
    std::vector<int64> input_dims = {batch, window_size + num_windows - 1,
                                     input_feature};
    std::vector<int64> filter_dims = {window_size, input_feature,
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
    auto input_r3 = input_r1.Reshape(input_dims).ConsumeValueOrDie();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(1.0f));

    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r3 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();

    std::vector<T> expect_elems(batch * output_feature * num_windows,
                                static_cast<T>(window_size * input_feature));
    auto expected_r1 = LiteralUtil::CreateR1<T>(expect_elems);
    auto expected_r3 = expected_r1.Reshape({batch, num_windows, output_feature})
                           .ConsumeValueOrDie();

    auto input_literal =
        client_->TransferToServer(input_r3).ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(filter_r3).ConsumeValueOrDie();
    ComputeAndCompareLiteral(&builder, expected_r3,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
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
                      Convolve1DTestParam{130, 1, 1, 1, 3},
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
// TODO(b/72566306): The following five tests failed on CPU with unreasonable
// relative errors.  Last ran on 2018-02-22.
#if XLA_TEST_BACKEND_GPU
                      Convolve1DTestParam{139, 1, 1, 128, 1},
                      Convolve1DTestParam{640, 3, 3, 128, 1},
                      Convolve1DTestParam{900, 1, 1, 10, 1},
                      Convolve1DTestParam{1, 10, 10, 1, 10},
                      Convolve1DTestParam{1, 10, 130, 1, 1},
#endif
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

XLA_TEST_F(ConvolutionTest, Convolve_bf16_1x1x1x2_1x1x1x2_Valid) {
  XlaBuilder builder(TestName());
  Shape input_shape = ShapeUtil::MakeShape(BF16, {1, 1, 1, 2});
  Shape filter_shape = ShapeUtil::MakeShape(BF16, {1, 1, 1, 2});
  auto input = Parameter(&builder, 0, input_shape, "input");
  auto filter = Parameter(&builder, 1, filter_shape, "filter");
  Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<bfloat16> input_data(1, 1, 1, 2);
  input_data.FillWithYX(Array2D<bfloat16>({
      {bfloat16(1), bfloat16(2)},
  }));
  Array4D<bfloat16> filter_data(1, 1, 1, 2);
  filter_data.FillWithYX(Array2D<bfloat16>({
      {bfloat16(5), bfloat16(6)},
  }));

  ComputeAndCompare(&builder,
                    {LiteralUtil::CreateFromArray(input_data),
                     LiteralUtil::CreateFromArray(filter_data)},
                    error_spec_);
}

// Check that GPU convs still work if the CudnnAlgorithmPicker pass is disabled.
// (We run this test on all platforms, because, what the heck.)
XLA_TEST_F(ConvolutionTest, NoCudnnAlgorithmPicker) {
  execution_options_.mutable_debug_options()->add_xla_disable_hlo_passes(
      "cudnn-conv-algorithm-picker");

  XlaBuilder builder(TestName());
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 1, 1, 2});
  Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 1, 1, 2});
  auto input = Parameter(&builder, 0, input_shape, "input");
  auto filter = Parameter(&builder, 1, filter_shape, "filter");
  Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> input_data(1, 1, 1, 2);
  input_data.FillIota(0);
  Array4D<float> filter_data(1, 1, 1, 2);
  filter_data.FillIota(10);

  ComputeAndCompare(&builder, {LiteralUtil::CreateFromArray(input_data),
                               LiteralUtil::CreateFromArray(filter_data)});
}

XLA_TEST_F(ConvolutionTest, ConvolveF32BackwardInputGroupedConvolution) {
  XlaBuilder builder(TestName());
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 64, 100, 100});
  Array4D<float> input_data(1, 64, 100, 100);
  input_data.FillRandom(/*value=*/0.023, 0.001, /*seed=*/45321);
  Shape filter_shape = ShapeUtil::MakeShape(F32, {7, 7, 1, 64});
  Array4D<float> filter_data(7, 7, 1, 64);
  input_data.FillRandom(/*value=*/0.023, 0.001, /*seed=*/45320);
  auto input = Parameter(&builder, 0, input_shape, "input");
  auto filter = ConstantR4FromArray4D(&builder, filter_data);

  // Specify bf01_01io->bf01 as dimension numbers.
  ConvolutionDimensionNumbers dnums;
  // Input
  dnums.set_input_feature_dimension(1);
  dnums.set_input_batch_dimension(0);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_input_spatial_dimensions(3);
  // Kernel
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(3);
  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);
  // Output
  dnums.set_output_batch_dimension(0);
  dnums.set_output_feature_dimension(1);
  dnums.add_output_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(3);
  ConvGeneral(input, filter, /*window_strides=*/{1, 1},
              /*padding=*/{{3, 3}, {3, 3}}, /*dimension_numbers=*/dnums,
              /*feature_group_count=*/64);

  ComputeAndCompare(&builder, {LiteralUtil::CreateFromArray(input_data)},
                    error_spec_);
}

class ConvolutionHloTest : public HloTestBase {};

XLA_TEST_F(ConvolutionHloTest, ConvolveF64Forward) {
  constexpr char kHlo[] = R"(
HloModule TestModule

ENTRY Test {
  %arg0 = f64[3,56,56,16] parameter(0)
  %arg1 = f64[3,3,3,64] parameter(1)
  ROOT %conv = f64[54,54,16,64] convolution(%arg0, %arg1), window={size=3x3}, dim_labels=f01b_i01o->01bf
})";
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{0.001}));
}

XLA_TEST_F(ConvolutionHloTest, ConvolveF32ForwardReversed) {
  constexpr char kHlo[] = R"(
HloModule TestModule

ENTRY Test {
  %arg0 = f32[3,56,56,16] parameter(0)
  %arg1 = f32[3,3,3,32] parameter(1)
  ROOT %conv = f32[54,54,16,32] convolution(%arg0, %arg1), window={size=3x3 rhs_reversal=1x1}, dim_labels=f01b_i01o->01bf
})";
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{0.001}));
}

XLA_TEST_F(ConvolutionHloTest, ConvolveF64BackwardFilter) {
  constexpr char kHlo[] = R"(
HloModule TestModule

ENTRY Test {
  %arg0 = f64[2,5,8,1] parameter(0)
  %arg1 = f64[2,5,8,2] parameter(1)
  ROOT %conv = f64[4,4,1,2] convolution(%arg0, %arg1), window={size=5x8 pad=1_2x1_2}, dim_labels=f01b_i01o->01bf
})";
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{0.001}));
}

XLA_TEST_F(ConvolutionHloTest, ConvolveF64BackwardInput) {
  constexpr char kHlo[] = R"(
HloModule TestModule

ENTRY Test {
  %output = f64[4,5,16,16] parameter(0)
  %kernel = f64[5,3,7,7] parameter(1)
  %reverse = f64[5,3,7,7] reverse(f64[5,3,7,7] %kernel), dimensions={2,3}
  ROOT %convolution = f64[4,3,16,16] convolution(%output, %reverse), window={size=7x7 pad=3_3x3_3}, dim_labels=bf01_io01->bf01
})";
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{0.001}));
}

}  // namespace
}  // namespace xla
