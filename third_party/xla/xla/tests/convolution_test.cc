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

// Tests of 2+D convolution with trivial kernels and no special variations (like
// strides and padding).

#include <cstdint>
#include <numeric>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "xla/tests/xla_test_backend_predicates.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "Eigen/Core"
#include "xla/array2d.h"
#include "xla/array4d.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/padding.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/types.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class ConvolutionTest : public ClientLibraryTestRunnerMixin<
                            HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>> {
 public:
  // Returns true if the test is running on ROCm.
  bool IsRocm() {
    return test_runner().HasProperty(HloRunnerPropertyTag::kUsingGpuRocm);
  }

 protected:
#if XLA_TEST_BACKEND_GPU
  // XLA:GPU sometimes uses FFT convolution which isn't as precise as spatial
  // convolution. So relax the absolute error threshold.
  ErrorSpec error_spec_ = ErrorSpec(1e-2, 1e-3);
#else
  ErrorSpec error_spec_ = ErrorSpec(1e-4, 1e-3);
#endif
};

using TestTypes = ::testing::Types<
// TODO(b/183565702): Support integer convs on GPU.
#if !XLA_TEST_BACKEND_GPU
    int32_t,
#endif
#ifndef XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16
    Eigen::half,
#endif
    float>;

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
    auto alhs = std::make_unique<Array4D<T>>(
        kMiniBatchSize, kInputActivationSizeZ, kInputActivationSizeY,
        kInputActivationSizeX);
    alhs->FillWithMultiples(static_cast<T>(static_cast<T>(1.0f)));
    ASSERT_EQ(3, alhs->width());
    ASSERT_EQ(3, alhs->height());

    auto arhs = std::make_unique<Array4D<T>>(kOutputActivationSizeZ,
                                             kInputActivationSizeZ,
                                             kKernelSizeY, kKernelSizeX);
    Array2D<T> rhs_raster({
        {static_cast<T>(1.0f), static_cast<T>(0.0f)},  // row 0
        {static_cast<T>(0.0f), static_cast<T>(0.0f)},  // row 1
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
        {static_cast<T>(1.0f), static_cast<T>(2.0f)},
    }));
    Literal input_data_literal = LiteralUtil::CreateFromArray(input_data);
    Array4D<T> filter_data(1, 1, 1, 2);
    filter_data.FillWithYX(Array2D<T>({
        {static_cast<T>(5.0f), static_cast<T>(6.0f)},
    }));
    Literal filter_data_literal = LiteralUtil::CreateFromArray(filter_data);

    ComputeAndCompare(&builder, {&input_data_literal, &filter_data_literal},
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
        {static_cast<T>(1.0f), static_cast<T>(2.0f), static_cast<T>(3.0f),
         static_cast<T>(4.0f)},
        {static_cast<T>(5.0f), static_cast<T>(6.0f), static_cast<T>(7.0f),
         static_cast<T>(8.0f)},
        {static_cast<T>(9.0f), static_cast<T>(10.0f), static_cast<T>(11.0f),
         static_cast<T>(12.0f)},
        {static_cast<T>(13.0f), static_cast<T>(14.0f), static_cast<T>(15.0f),
         static_cast<T>(16.0f)},
    }));
    Literal input_data_literal = LiteralUtil::CreateFromArray(input_data);
    Array4D<T> filter_data(1, 1, 2, 2);
    filter_data.FillWithYX(Array2D<T>({
        {static_cast<T>(5.0f), static_cast<T>(6.0f)},
        {static_cast<T>(7.0f), static_cast<T>(8.0f)},
    }));
    Literal filter_data_literal = LiteralUtil::CreateFromArray(filter_data);
    ComputeAndCompare(&builder, {&input_data_literal, &filter_data_literal},
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
        {static_cast<T>(1.0f), static_cast<T>(2.0f), static_cast<T>(3.0f),
         static_cast<T>(4.0f)},
        {static_cast<T>(5.0f), static_cast<T>(6.0f), static_cast<T>(7.0f),
         static_cast<T>(8.0f)},
        {static_cast<T>(9.0f), static_cast<T>(10.0f), static_cast<T>(11.0f),
         static_cast<T>(12.0f)},
        {static_cast<T>(13.0f), static_cast<T>(14.0f), static_cast<T>(15.0f),
         static_cast<T>(16.0f)},
    }));
    Literal input_data_literal = LiteralUtil::CreateFromArray(input_data);
    Array4D<T> filter_data(1, 1, 2, 2);
    filter_data.FillWithYX(Array2D<T>({
        {static_cast<T>(5.0f), static_cast<T>(6.0f)},
        {static_cast<T>(7.0f), static_cast<T>(8.0f)},
    }));
    Literal filter_data_literal = LiteralUtil::CreateFromArray(filter_data);

    ComputeAndCompare(&builder, {&input_data_literal, &filter_data_literal},
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
    input_data.FillWithYX(
        Array2D<T>({{static_cast<T>(1.0f), static_cast<T>(2.0f),
                     static_cast<T>(3.0f), static_cast<T>(4.0f)},
                    {static_cast<T>(5.0f), static_cast<T>(6.0f),
                     static_cast<T>(7.0f), static_cast<T>(8.0f)},
                    {static_cast<T>(9.0f), static_cast<T>(10.0f),
                     static_cast<T>(11.0f), static_cast<T>(12.0f)},
                    {static_cast<T>(13.0f), static_cast<T>(14.0f),
                     static_cast<T>(15.0f), static_cast<T>(16.0f)}}));
    Literal input_data_literal = LiteralUtil::CreateFromArray(input_data);
    Array4D<T> filter_data(1, 1, 3, 3);
    filter_data.FillWithYX(Array2D<T>(
        {{static_cast<T>(5.0f), static_cast<T>(6.0f), static_cast<T>(7.0f)},
         {static_cast<T>(8.0f), static_cast<T>(9.0f), static_cast<T>(10.0f)},
         {static_cast<T>(11.0f), static_cast<T>(12.0f),
          static_cast<T>(13.0f)}}));
    Literal filter_data_literal = LiteralUtil::CreateFromArray(filter_data);
    // clang-format on
    ComputeAndCompare(&builder, {&input_data_literal, &filter_data_literal},
                      error_spec_);
  }
};

TYPED_TEST_CASE(Convolve_1x1x4x4_1x1x3x3_Same, TestTypes);
TYPED_TEST(Convolve_1x1x4x4_1x1x3x3_Same, Types) { this->RunTest(); }

XLA_TEST_F(ConvolutionTest, Convolve3D_1x4x2x3x3_2x2x2x3x3_Valid) {
  XlaBuilder builder(TestName());
  std::vector<int64_t> input_dims = {1, 4, 2, 3, 3};
  std::vector<int64_t> filter_dims = {2, 2, 2, 3, 3};
  Shape input_shape = ShapeUtil::MakeValidatedShape(F32, input_dims).value();
  Shape filter_shape = ShapeUtil::MakeValidatedShape(F32, filter_dims).value();
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
  auto input_r5 = input_r1.Reshape(input_dims).value();

  std::vector<float> filter_elems(ShapeUtil::ElementsIn(filter_shape));
  iota(filter_elems.begin(), filter_elems.end(), 1.0f);
  auto filter_r1 = LiteralUtil::CreateR1<float>(filter_elems);
  auto filter_r5 = filter_r1.Reshape(filter_dims).value();

  auto expected_r1 = LiteralUtil::CreateR1<float>(
      {19554, 19962, 20370, 22110, 22590, 23070, 34890, 35730, 36570, 37446,
       38358, 39270, 50226, 51498, 52770, 52782, 54126, 55470});
  auto expected_r5 = expected_r1.Reshape({1, 3, 1, 2, 3}).value();

  ComputeAndCompareLiteral(&builder, expected_r5, {&input_r5, &filter_r5},
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
    std::vector<int64_t> input_dims = {1, 3, 3, 5};
    std::vector<int64_t> filter_dims = {3, 3, 5, 3};
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
    auto input_r4 = input_r1.Reshape(input_dims).value();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape));
    iota_int_init_value(filter_elems, 1);
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).value();

    auto expected_r1 = LiteralUtil::CreateR1<T>(
        {static_cast<T>(92115), static_cast<T>(93150), static_cast<T>(94185)});
    auto expected_r4 = expected_r1.Reshape({1, 1, 1, 3}).value();

    ComputeAndCompareLiteral(&builder, expected_r4, {&input_r4, &filter_r4},
                             error_spec_);
  }
};

TYPED_TEST_CASE(Convolve2D_1x3x3x5_3x3x5x3_Valid, TestTypes);
TYPED_TEST(Convolve2D_1x3x3x5_3x3x5x3_Valid, Types) { this->RunTest(); }

// Test same padding for 2D convolution with kernel of such size, that every
// single pad value is different (low and high, in x and y dimension).
// Intention of this test is to verify that padding is implemented correctly.
template <typename T>
class Convolve2D_1x6x6x1_6x2x1x1_Same : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64_t> input_dims = {1, 6, 6, 1};
    std::vector<int64_t> filter_dims = {6, 2, 1, 1};
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>(input_dims);
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>(filter_dims);
    {
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");

      // Tensorflow dimension numbers for 2D convolution.
      ConvolutionDimensionNumbers dnums;
      dnums.set_input_batch_dimension(0);
      dnums.add_input_spatial_dimensions(1);
      dnums.add_input_spatial_dimensions(2);
      dnums.set_input_feature_dimension(3);
      dnums.add_kernel_spatial_dimensions(0);
      dnums.add_kernel_spatial_dimensions(1);
      dnums.set_kernel_input_feature_dimension(2);
      dnums.set_kernel_output_feature_dimension(3);
      dnums.set_output_batch_dimension(0);
      dnums.add_output_spatial_dimensions(1);
      dnums.add_output_spatial_dimensions(2);
      dnums.set_output_feature_dimension(3);

      ConvWithGeneralDimensions(input, filter, {1, 1}, Padding::kSame, dnums);
    }

    std::vector<T> input_elems(ShapeUtil::ElementsIn(input_shape));
    iota_int_init_value(input_elems, 1);
    auto input_r1 = LiteralUtil::CreateR1<T>(input_elems);
    auto input_r4 = input_r1.Reshape(input_dims).value();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape));
    iota_int_init_value(filter_elems, 1);
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).value();

    auto expected_r1 = LiteralUtil::CreateR1<T>(
        {static_cast<T>(836),  static_cast<T>(904),  static_cast<T>(972),
         static_cast<T>(1040), static_cast<T>(1108), static_cast<T>(540),
         static_cast<T>(1255), static_cast<T>(1330), static_cast<T>(1405),
         static_cast<T>(1480), static_cast<T>(1555), static_cast<T>(750),
         static_cast<T>(1710), static_cast<T>(1788), static_cast<T>(1866),
         static_cast<T>(1944), static_cast<T>(2022), static_cast<T>(966),
         static_cast<T>(1315), static_cast<T>(1370), static_cast<T>(1425),
         static_cast<T>(1480), static_cast<T>(1535), static_cast<T>(720),
         static_cast<T>(932),  static_cast<T>(968),  static_cast<T>(1004),
         static_cast<T>(1040), static_cast<T>(1076), static_cast<T>(492),
         static_cast<T>(585),  static_cast<T>(606),  static_cast<T>(627),
         static_cast<T>(648),  static_cast<T>(669),  static_cast<T>(294)});

    auto expected_r4 = expected_r1.Reshape({1, 6, 6, 1}).value();

    ComputeAndCompareLiteral(&builder, expected_r4, {&input_r4, &filter_r4},
                             error_spec_);
  }
};

TYPED_TEST_CASE(Convolve2D_1x6x6x1_6x2x1x1_Same, TestTypes);
TYPED_TEST(Convolve2D_1x6x6x1_6x2x1x1_Same, Types) { this->RunTest(); }

template <typename T>
class Convolve1D_1x3x5_3x5x3_Valid : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64_t> input_dims = {1, 3, 5};
    std::vector<int64_t> filter_dims = {3, 5, 3};
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>(input_dims);
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>(filter_dims);
    {
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");

      // Tensorflow dimension numbers for 1D convolution.
      // Layout as supported by Eigen convolution.
      ConvolutionDimensionNumbers dnums;
      dnums.set_input_batch_dimension(0);
      dnums.add_input_spatial_dimensions(1);
      dnums.set_input_feature_dimension(2);
      dnums.add_kernel_spatial_dimensions(0);
      dnums.set_kernel_input_feature_dimension(1);
      dnums.set_kernel_output_feature_dimension(2);
      dnums.set_output_batch_dimension(0);
      dnums.add_output_spatial_dimensions(1);
      dnums.set_output_feature_dimension(2);

      ConvWithGeneralDimensions(input, filter, {1}, Padding::kValid, dnums);
    }

    std::vector<T> input_elems(ShapeUtil::ElementsIn(input_shape));
    iota_int_init_value(input_elems, 1);
    auto input_r1 = LiteralUtil::CreateR1<T>(input_elems);
    auto input_r3 = input_r1.Reshape(input_dims).value();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape));
    iota_int_init_value(filter_elems, 1);
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r3 = filter_r1.Reshape(filter_dims).value();

    auto expected_r1 = LiteralUtil::CreateR1<T>(
        {static_cast<T>(3480), static_cast<T>(3600), static_cast<T>(3720)});
    auto expected_r3 = expected_r1.Reshape({1, 1, 3}).value();

    ComputeAndCompareLiteral(&builder, expected_r3, {&input_r3, &filter_r3},
                             error_spec_);
  }
};

TYPED_TEST_CASE(Convolve1D_1x3x5_3x5x3_Valid, TestTypes);
TYPED_TEST(Convolve1D_1x3x5_3x5x3_Valid, Types) { this->RunTest(); }

template <typename T>
class Convolve2D_1x3x3x5_3x3x1x15_Depthwise_Valid : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64_t> input_dims = {1, 3, 3, 5};
    std::vector<int64_t> filter_dims = {3, 3, 1, 15};
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
    auto input_r4 = input_r1.Reshape(input_dims).value();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape));
    iota_int_init_value(filter_elems, 1);
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).value();

    auto expected_r1 = LiteralUtil::CreateR1<T>(
        {static_cast<T>(16029), static_cast<T>(16218), static_cast<T>(16407),
         static_cast<T>(17172), static_cast<T>(17370), static_cast<T>(17568),
         static_cast<T>(18369), static_cast<T>(18576), static_cast<T>(18783),
         static_cast<T>(19620), static_cast<T>(19836), static_cast<T>(20052),
         static_cast<T>(20925), static_cast<T>(21150), static_cast<T>(21375)});
    auto expected_r4 = expected_r1.Reshape({1, 1, 1, 15}).value();

    ComputeAndCompareLiteral(&builder, expected_r4, {&input_r4, &filter_r4},
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
    std::vector<int64_t> input_dims = {1, 4, 4, 5};
    std::vector<int64_t> filter_dims = {3, 3, 1, 5};
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
    auto input_r4 = input_r1.Reshape(input_dims).value();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape));
    iota_int_init_value(filter_elems, 1);
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).value();

    auto expected_r1 = LiteralUtil::CreateR1<T>(
        {static_cast<T>(6864),  static_cast<T>(7296),  static_cast<T>(7746),
         static_cast<T>(8214),  static_cast<T>(8700),  static_cast<T>(7809),
         static_cast<T>(8286),  static_cast<T>(8781),  static_cast<T>(9294),
         static_cast<T>(9825),  static_cast<T>(10644), static_cast<T>(11256),
         static_cast<T>(11886), static_cast<T>(12534), static_cast<T>(13200),
         static_cast<T>(11589), static_cast<T>(12246), static_cast<T>(12921),
         static_cast<T>(13614), static_cast<T>(14325)});
    auto expected_r4 = expected_r1.Reshape({1, 2, 2, 5}).value();

    ComputeAndCompareLiteral(&builder, expected_r4, {&input_r4, &filter_r4},
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
    std::vector<int64_t> input_dims = {1, 4, 4, 512};
    std::vector<int64_t> filter_dims = {3, 3, 1, 512};
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
    auto input_r4 = input_r1.Reshape(input_dims).value();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(2));
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).value();

    std::vector<T> output_elems(2048, static_cast<T>(18));

    auto expected_r1 = LiteralUtil::CreateR1<T>(output_elems);
    auto expected_r4 = expected_r1.Reshape({1, 2, 2, 512}).value();

    ComputeAndCompareLiteral(&builder, expected_r4, {&input_r4, &filter_r4},
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
    std::vector<int64_t> input_dims = {1, 4, 4, 512};
    std::vector<int64_t> filter_dims = {3, 3, 1, 512};
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
    auto input_r4 = input_r1.Reshape(input_dims).value();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(2));
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).value();

    std::vector<T> output_elems(2048, static_cast<T>(18));

    auto expected_r1 = LiteralUtil::CreateR1<T>(output_elems);
    auto expected_r4 = expected_r1.Reshape({1, 2, 2, 512}).value();
    auto expected_r4_relaid =
        expected_r4.Relayout(LayoutUtil::MakeLayout({0, 3, 2, 1}));

    ComputeAndCompareLiteral(&builder, expected_r4_relaid,
                             {&input_r4, &filter_r4}, error_spec_);
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
    std::vector<int64_t> input_dims = {256, 4, 4, 512};
    std::vector<int64_t> filter_dims = {3, 3, 1, 512};
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
    auto input_r4 = input_r1.Reshape(input_dims).value();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(2));
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).value();

    std::vector<T> output_elems(2048 * 256, static_cast<T>(18));

    auto expected_r1 = LiteralUtil::CreateR1<T>(output_elems);
    auto expected_r4 = expected_r1.Reshape({256, 2, 2, 512}).value();

    ComputeAndCompareLiteral(&builder, expected_r4, {&input_r4, &filter_r4},
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
    std::vector<int64_t> input_dims = {256, 4, 4, 512};
    std::vector<int64_t> filter_dims = {3, 3, 1, 512};
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
    auto input_r4 = input_r1.Reshape(input_dims).value();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(2));
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).value();

    std::vector<T> output_elems(2048 * 256, static_cast<T>(18));

    auto expected_r1 = LiteralUtil::CreateR1<T>(output_elems);
    auto expected_r4 = expected_r1.Reshape({256, 2, 2, 512}).value();
    auto expected_r4_relaid =
        expected_r4.Relayout(LayoutUtil::MakeLayout({0, 3, 2, 1}));

    ComputeAndCompareLiteral(&builder, expected_r4_relaid,
                             {&input_r4, &filter_r4}, error_spec_);
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
    std::vector<int64_t> input_dims = {1, 4, 4, 5};
    std::vector<int64_t> filter_dims = {3, 3, 1, 5};
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
    auto input_r4 = input_r1.Reshape(input_dims).value();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape));
    iota_int_init_value(filter_elems, 1);
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).value();

    auto expected_r1 = LiteralUtil::CreateR1<T>(
        {static_cast<T>(6864),  static_cast<T>(7296),  static_cast<T>(7746),
         static_cast<T>(8214),  static_cast<T>(8700),  static_cast<T>(7809),
         static_cast<T>(8286),  static_cast<T>(8781),  static_cast<T>(9294),
         static_cast<T>(9825),  static_cast<T>(10644), static_cast<T>(11256),
         static_cast<T>(11886), static_cast<T>(12534), static_cast<T>(13200),
         static_cast<T>(11589), static_cast<T>(12246), static_cast<T>(12921),
         static_cast<T>(13614), static_cast<T>(14325)});
    auto expected_r4 = expected_r1.Reshape({1, 2, 2, 5}).value();
    auto expected_r4_relaid =
        expected_r4.Relayout(LayoutUtil::MakeLayout({0, 3, 2, 1}));

    ComputeAndCompareLiteral(&builder, expected_r4_relaid,
                             {&input_r4, &filter_r4}, error_spec_);
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
    std::vector<int64_t> input_dims = {1, 4, 4, 160};
    std::vector<int64_t> filter_dims = {3, 3, 1, 160};
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
    auto input_r4 = input_r1.Reshape(input_dims).value();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(2));
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).value();

    std::vector<T> output_elems(640, static_cast<T>(18));

    auto expected_r1 = LiteralUtil::CreateR1<T>(output_elems);
    auto expected_r4 = expected_r1.Reshape({1, 2, 2, 160}).value();

    ComputeAndCompareLiteral(&builder, expected_r4, {&input_r4, &filter_r4},
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
    std::vector<int64_t> input_dims = {1, 4, 4, 160};
    std::vector<int64_t> filter_dims = {3, 3, 1, 160};
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
    auto input_r4 = input_r1.Reshape(input_dims).value();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(2));
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).value();

    std::vector<T> output_elems(640, static_cast<T>(18));

    auto expected_r1 = LiteralUtil::CreateR1<T>(output_elems);
    auto expected_r4 = expected_r1.Reshape({1, 2, 2, 160}).value();
    auto expected_r4_relaid =
        expected_r4.Relayout(LayoutUtil::MakeLayout({3, 0, 2, 1}));

    ComputeAndCompareLiteral(&builder, expected_r4_relaid,
                             {&input_r4, &filter_r4}, error_spec_);
  }
};

TYPED_TEST_CASE(Convolve2D_1x4x4x160_3x3x1x160_Depthwise_Input_Batch_In_Lanes,
                TestTypes);
TYPED_TEST(Convolve2D_1x4x4x160_3x3x1x160_Depthwise_Input_Batch_In_Lanes,
           Types) {
  this->RunTest();
}

template <typename T>
class Convolve2D_1x4x4x160_3x3x1x160_Depthwise_Both_Batch_In_Lanes
    : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64_t> input_dims = {1, 4, 4, 160};
    std::vector<int64_t> filter_dims = {3, 3, 1, 160};
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
    auto input_r4 = input_r1.Reshape(input_dims).value();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(2));
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).value();

    std::vector<T> output_elems(640, static_cast<T>(18));

    auto expected_r1 = LiteralUtil::CreateR1<T>(output_elems);
    auto expected_r4 = expected_r1.Reshape({1, 2, 2, 160}).value();
    auto expected_r4_relaid =
        expected_r4.Relayout(LayoutUtil::MakeLayout({0, 3, 2, 1}));

    ComputeAndCompareLiteral(&builder, expected_r4_relaid,
                             {&input_r4, &filter_r4}, error_spec_);
  }
};

TYPED_TEST_CASE(Convolve2D_1x4x4x160_3x3x1x160_Depthwise_Both_Batch_In_Lanes,
                TestTypes);
TYPED_TEST(Convolve2D_1x4x4x160_3x3x1x160_Depthwise_Both_Batch_In_Lanes,
           Types) {
  this->RunTest();
}

template <typename T>
class Convolve2D_1x4x4x1024_3x3x1x1024_Depthwise_Valid
    : public ConvolutionTest {
 public:
  void RunTest() {
    XlaBuilder builder(TestName());
    std::vector<int64_t> input_dims = {1, 4, 4, 1024};
    std::vector<int64_t> filter_dims = {3, 3, 1, 1024};
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
    auto input_r4 = input_r1.Reshape(input_dims).value();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(2));
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).value();

    std::vector<T> output_elems(4096, static_cast<T>(18));

    auto expected_r1 = LiteralUtil::CreateR1<T>(output_elems);
    auto expected_r4 = expected_r1.Reshape({1, 2, 2, 1024}).value();

    ComputeAndCompareLiteral(&builder, expected_r4, {&input_r4, &filter_r4},
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
    std::vector<int64_t> input_dims = {1, 2, 2, 6};
    std::vector<int64_t> filter_dims = {2, 2, 2, 12};
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
    auto input_r4 = input_r1.Reshape(input_dims).value();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape));
    iota_int_init_value(filter_elems, 1);
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).value();

    auto expected_r1 = LiteralUtil::CreateR1<T>(
        {static_cast<T>(5076), static_cast<T>(5160), static_cast<T>(5244),
         static_cast<T>(5328), static_cast<T>(6164), static_cast<T>(6264),
         static_cast<T>(6364), static_cast<T>(6464), static_cast<T>(7380),
         static_cast<T>(7496), static_cast<T>(7612), static_cast<T>(7728)});
    auto expected_r4 = expected_r1.Reshape({1, 1, 1, 12}).value();

    ComputeAndCompareLiteral(&builder, expected_r4, {&input_r4, &filter_r4},
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
    std::vector<int64_t> input_dims = {1, 2, 2, 1024};
    std::vector<int64_t> filter_dims = {2, 2, 128, 512};
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
    auto input_r4 = input_r1.Reshape(input_dims).value();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(2));

    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).value();

    std::vector<T> output_elems(512, static_cast<T>(1024));
    auto expected_r1 = LiteralUtil::CreateR1<T>(output_elems);
    auto expected_r4 = expected_r1.Reshape({1, 1, 1, 512}).value();

    ComputeAndCompareLiteral(&builder, expected_r4, {&input_r4, &filter_r4},
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
    std::vector<int64_t> input_dims = {1, 2, 2, 1024};
    std::vector<int64_t> filter_dims = {2, 2, 128, 8};
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
    auto input_r4 = input_r1.Reshape(input_dims).value();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(2));

    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).value();

    std::vector<T> output_elems(8, static_cast<T>(1024));
    auto expected_r1 = LiteralUtil::CreateR1<T>(output_elems);
    auto expected_r4 = expected_r1.Reshape({1, 1, 1, 8}).value();

    ComputeAndCompareLiteral(&builder, expected_r4, {&input_r4, &filter_r4},
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
    std::vector<int64_t> input_dims = {1, 2, 2, 12};
    std::vector<int64_t> filter_dims = {2, 2, 3, 4};
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
    auto input_r4 = input_r1.Reshape(input_dims).value();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape));
    iota_int_init_value(filter_elems, 1);
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).value();

    auto expected_r1 =
        LiteralUtil::CreateR1<T>({static_cast<T>(7712), static_cast<T>(8816),
                                  static_cast<T>(9992), static_cast<T>(11240)});
    auto expected_r4 = expected_r1.Reshape({1, 1, 1, 4}).value();

    ComputeAndCompareLiteral(&builder, expected_r4, {&input_r4, &filter_r4},
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
    std::vector<int64_t> input_dims = {1, 2, 2, 12};
    std::vector<int64_t> filter_dims = {2, 2, 4, 3};
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
    auto input_r4 = input_r1.Reshape(input_dims).value();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape));
    iota_int_init_value(filter_elems, 1);
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).value();
    auto filter_r4_relaid =
        filter_r4.Relayout(LayoutUtil::MakeLayout({3, 2, 1, 0}));
    auto expected_r1 = LiteralUtil::CreateR1<T>(
        {static_cast<T>(6968), static_cast<T>(8516), static_cast<T>(10280),
         static_cast<T>(12260)});
    auto expected_r4 = expected_r1.Reshape({1, 1, 1, 4}).value();

    ComputeAndCompareLiteral(&builder, expected_r4,
                             {&input_r4, &filter_r4_relaid}, error_spec_);
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
    std::vector<int64_t> input_dims = {1, 1, 1, 12};
    std::vector<int64_t> filter_dims = {1, 1, 3, 4};
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
    auto input_r4 = input_r1.Reshape(input_dims).value();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape));
    iota_int_init_value(filter_elems, 1);
    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r4 = filter_r1.Reshape(filter_dims).value();

    auto expected_r1 =
        LiteralUtil::CreateR1<T>({static_cast<T>(38), static_cast<T>(98),
                                  static_cast<T>(176), static_cast<T>(272)});
    auto expected_r4 = expected_r1.Reshape({1, 1, 1, 4}).value();

    ComputeAndCompareLiteral(&builder, expected_r4, {&input_r4, &filter_r4},
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

XLA_TEST_P(ConvolveWithAndWithoutCanonicalization, Convolve2D_NoSpatialDims) {
  if (GetParam()) {
    mutable_debug_options()->add_xla_disable_hlo_passes(
        "convolution-canonicalization");
  }
  XlaBuilder builder(TestName());
  Shape input_shape = ShapeUtil::MakeValidatedShape(F32, {4, 29}).value();
  Shape filter_shape = ShapeUtil::MakeValidatedShape(F32, {4, 10}).value();

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
  Literal param0_literal = LiteralUtil::CreateFromArray(param0);

  Array2D<float> param1(4, 10);
  param1.FillUnique();
  Literal param1_literal = LiteralUtil::CreateFromArray(param1);

  Array2D<float> expected_result(29, 10);
  expected_result.Fill(0);

  ComputeAndCompare(&builder, {&param0_literal, &param1_literal}, error_spec_);
}

INSTANTIATE_TEST_CASE_P(ConvolveWithAndWithoutCanonicalization_Instantiation,
                        ConvolveWithAndWithoutCanonicalization,
                        ::testing::Values(true, false));

XLA_TEST_F(ConvolutionTest, Convolve_bf16_1x1x1x2_1x1x1x2_Valid) {
  XlaBuilder builder(TestName());
  Shape input_shape = ShapeUtil::MakeValidatedShape(BF16, {1, 1, 1, 2}).value();
  Shape filter_shape =
      ShapeUtil::MakeValidatedShape(BF16, {1, 1, 1, 2}).value();
  auto input = Parameter(&builder, 0, input_shape, "input");
  auto filter = Parameter(&builder, 1, filter_shape, "filter");
  Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<bfloat16> input_data(1, 1, 1, 2);
  input_data.FillWithYX(Array2D<bfloat16>({
      {bfloat16(1), bfloat16(2)},
  }));
  Literal input_data_literal = LiteralUtil::CreateFromArray(input_data);
  Array4D<bfloat16> filter_data(1, 1, 1, 2);
  filter_data.FillWithYX(Array2D<bfloat16>({
      {bfloat16(5), bfloat16(6)},
  }));
  Literal filter_data_literal = LiteralUtil::CreateFromArray(filter_data);
  ComputeAndCompare(&builder, {&input_data_literal, &filter_data_literal},
                    error_spec_);
}

// Check that GPU convs still work if the CudnnAlgorithmPicker pass is disabled.
// (We run this test on all platforms, because, what the heck.)
XLA_TEST_F(ConvolutionTest, NoCudnnAlgorithmPicker) {
  if (IsRocm()) {
    GTEST_SKIP();
  }
  mutable_debug_options()->add_xla_disable_hlo_passes(
      "gpu-conv-algorithm-picker");

  XlaBuilder builder(TestName());
  Shape input_shape = ShapeUtil::MakeValidatedShape(F32, {1, 1, 1, 2}).value();
  Shape filter_shape = ShapeUtil::MakeValidatedShape(F32, {1, 1, 1, 2}).value();
  auto input = Parameter(&builder, 0, input_shape, "input");
  auto filter = Parameter(&builder, 1, filter_shape, "filter");
  Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> input_data(1, 1, 1, 2);
  input_data.FillIota(0);
  Literal input_data_literal = LiteralUtil::CreateFromArray(input_data);
  Array4D<float> filter_data(1, 1, 1, 2);
  filter_data.FillIota(10);
  Literal filter_data_literal = LiteralUtil::CreateFromArray(filter_data);

  ComputeAndCompare(&builder, {&input_data_literal, &filter_data_literal});
}

XLA_TEST_F(ConvolutionTest, ConvolveF32BackwardInputGroupedConvolution) {
  XlaBuilder builder(TestName());
  Shape input_shape =
      ShapeUtil::MakeValidatedShape(F32, {1, 64, 100, 100}).value();
  Array4D<float> input_data(1, 64, 100, 100);
  input_data.FillRandom(/*stddev=*/0.023, 0.001, /*seed=*/45321);
  Literal input_data_literal = LiteralUtil::CreateFromArray(input_data);
  Shape filter_shape =
      ShapeUtil::MakeValidatedShape(F32, {7, 7, 1, 64}).value();
  Array4D<float> filter_data(7, 7, 1, 64);
  filter_data.FillRandom(/*stddev=*/0.023, 0.001, /*seed=*/45320);
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

  ComputeAndCompare(&builder, {&input_data_literal}, error_spec_);
}

class ConvolutionHloTest
    : public HloPjRtInterpreterReferenceMixin<HloPjRtTestBase> {
 public:
  // Returns true if the test is running on ROCm.
  bool IsRocm() {
    return test_runner().HasProperty(HloRunnerPropertyTag::kUsingGpuRocm);
  }
};

XLA_TEST_F(ConvolutionHloTest, DISABLED_ON_TPU(ConvolveF64Forward)) {
  if (IsRocm()) {
    GTEST_SKIP() << "double datatype is not yet supported in ROCm";
  }
  constexpr char kHlo[] = R"(
HloModule TestModule

ENTRY Test {
  %arg0 = f64[3,56,56,16] parameter(0)
  %arg1 = f64[3,3,3,64] parameter(1)
  ROOT %conv = f64[54,54,16,64] convolution(%arg0, %arg1), window={size=3x3}, dim_labels=f01b_i01o->01bf
})";
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{0.001}));
}

XLA_TEST_F(ConvolutionHloTest, ConvolveC64Forward) {
  if (test::DeviceIs(test::kGpu)) {
    GTEST_SKIP();
  }
  constexpr char kHlo[] = R"(
HloModule TestModule

ENTRY Test {
  %arg0 = c64[3,56,56,16] parameter(0)
  %arg1 = c64[3,3,3,64] parameter(1)
  ROOT %conv = c64[54,54,16,64] convolution(%arg0, %arg1), window={size=3x3}, dim_labels=f01b_i01o->01bf
})";
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{0.01, 0.01}));
}

XLA_TEST_F(ConvolutionHloTest, ConvolveF32ForwardReversed) {
  if (IsRocm()) {
    GTEST_SKIP() << "Not supported on ROCm";
  }

  constexpr char kHlo[] = R"(
HloModule TestModule

ENTRY Test {
  %arg0 = f32[3,56,56,16] parameter(0)
  %arg1 = f32[3,3,3,32] parameter(1)
  ROOT %conv = f32[54,54,16,32] convolution(%arg0, %arg1), window={size=3x3 rhs_reversal=1x1}, dim_labels=f01b_i01o->01bf
})";
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{0.001}));
}

XLA_TEST_F(ConvolutionHloTest, DISABLED_ON_TPU(ConvolveF64BackwardFilter)) {
  if (IsRocm()) {
    GTEST_SKIP() << "double datatype is not yet supported in ROCm";
  }
  constexpr char kHlo[] = R"(
HloModule TestModule

ENTRY Test {
  %arg0 = f64[2,5,8,1] parameter(0)
  %arg1 = f64[2,5,8,2] parameter(1)
  ROOT %conv = f64[4,4,1,2] convolution(%arg0, %arg1), window={size=5x8 pad=1_2x1_2}, dim_labels=f01b_i01o->01bf
})";
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{0.001}));
}

XLA_TEST_F(ConvolutionHloTest, DISABLED_ON_TPU(ConvolveF64BackwardInput)) {
  if (IsRocm()) {
    GTEST_SKIP() << "double datatype is not yet supported in ROCm";
  }
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

XLA_TEST_F(ConvolutionHloTest, ConvolveBackwardInput) {
  constexpr char kHlo[] = R"(
HloModule TestModule

ENTRY Test {
  %output = f32[3,3,64,64] parameter(0)
  %kernel = f32[672,7,7,64] parameter(1)
  %reverse = f32[672,7,7,64]{3,2,1,0} reverse(f32[672,7,7,64]{3,2,1,0} %kernel), dimensions={1,2}
  ROOT %convolution = f32[672,9,9,64]{3,2,1,0} convolution(f32[3,3,64,64]{3,2,1,0} %output, f32[672,7,7,64]{3,2,1,0} %reverse), window={size=7x7 pad=6_6x6_6}, dim_labels=01bf_o01i->f01b
})";
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{0.01, 0.01}));
}

XLA_TEST_F(ConvolutionHloTest, SwappedOperandConvolve) {
  constexpr char kHlo[] = R"(
HloModule TestModule

ENTRY Test {
  %lhs = f32[3,3,7,7] parameter(0)
  %rhs = f32[5,11,11,7] parameter(1)
  ROOT %convolution = f32[5,21,2,7] convolution(lhs, rhs),
     window={size=11x11 pad=3_25x3_6},
     dim_labels=01bf_o01i->f01b
})";
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{0.01, 0.01}));
}

XLA_TEST_F(ConvolutionHloTest, SwappedOperandConvolveWithStride) {
  constexpr char kHlo[] = R"(
HloModule TestModule

ENTRY Test {
  %lhs = f32[3,3,7,7] parameter(0)
  %rhs = f32[5,11,11,7] parameter(1)
  ROOT %convolution = f32[5,11,2,7] convolution(lhs, rhs),
     window={size=11x11 pad=3_26x3_6 stride=2x1},
     dim_labels=01bf_o01i->f01b
})";
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{0.01, 0.01}));
}
XLA_TEST_F(ConvolutionHloTest, SwappedOperandConvolve2) {
  constexpr char kHlo[] = R"(
HloModule TestModule

ENTRY Test {
  %lhs = f32[3,3,7,7] parameter(0)
  %rhs = f32[5,11,11,7] parameter(1)
  ROOT %convolution = f32[5,11,4,7] convolution(lhs, rhs),
     window={size=11x11 pad=3_25x3_6 lhs_dilate=1x2 rhs_dilate=2x1},
     dim_labels=01bf_o01i->f01b
})";
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{0.01, 0.01}));
}

XLA_TEST_F(ConvolutionHloTest, TestConv0D) {
  constexpr char kHlo[] = R"(
HloModule TestModule

ENTRY TestComputation {
  %parameter.1 = f32[10,5]{1,0} parameter(0)
  %parameter.2 = f32[5,7]{1,0} parameter(1)
  ROOT %convolution.3 = f32[10,7]{1,0} convolution(f32[10,5]{1,0} %parameter.1, f32[5,7]{1,0} %parameter.2), dim_labels=bf_io->bf
})";
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{0.01, 0.01}));
}

XLA_TEST_F(ConvolutionHloTest, TestConv2DF16) {
  std::string kHlo = R"(
HloModule TestModule

ENTRY TestComputation {
  %p0 = f16[8,5,5,1] parameter(0)
  %p1 = f16[3,3,1,32] parameter(1)
  ROOT %conv = f16[8,5,5,32] convolution(p0, p1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
})";

  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{0.01, 0.01}));
}

XLA_TEST_F(ConvolutionHloTest, TestFusedConv2D) {
  std::string kHlo = R"(
HloModule TestModule

ENTRY TestComputation {
  %p0 = f32[8,5,5,1] parameter(0)
  %p1 = f32[3,3,1,32] parameter(1)
  %conv = f32[8,5,5,32] convolution(p0, p1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  %bias = f32[32] parameter(2)
  %broadcasted_bias = f32[8,5,5,32] broadcast(%bias), dimensions={3}
  %add = f32[8,5,5,32] add(%conv, %broadcasted_bias)
)";

  std::string kHloNoPad = R"(
HloModule TestModule

ENTRY TestComputation {
  %p0 = f32[8,7,7,1] parameter(0)
  %p1 = f32[3,3,1,32] parameter(1)
  %conv = f32[8,5,5,32] convolution(p0, p1), window={size=3x3 pad=0_0x0_0}, dim_labels=b01f_01io->b01f
  %bias = f32[32] parameter(2)
  %broadcasted_bias = f32[8,5,5,32] broadcast(%bias), dimensions={3}
  %add = f32[8,5,5,32] add(%conv, %broadcasted_bias)
)";

  std::string kHloRELU = R"(

  %zero = f32[] constant(0)
  %zeros = f32[8,5,5,32] broadcast(%zero), dimensions={}
  ROOT relu = f32[8,5,5,32] maximum(%zeros, %add)
})";

  std::string kHloTANH = R"(
  ROOT result = f32[8,5,5,32] tanh(%add)
})";

  std::string kHloELU = R"(
  %zero = f32[] constant(0)
  %zeros = f32[8,5,5,32] broadcast(%zero), dimensions={}
  %one = f32[] constant(1)
  %ones = f32[8,5,5,32] broadcast(%one), dimensions={}
  %exp = f32[8,5,5,32] exponential(%add)
  %expm1 = f32[8,5,5,32] subtract(%exp, %ones)
  %sgn = pred[8,5,5,32] compare(%add, %zeros), direction=GT
  ROOT elu = f32[8,5,5,32] select(%sgn, %add, %expm1)
})";

  EXPECT_TRUE(RunAndCompare(kHlo + kHloRELU, ErrorSpec{0.01, 0.01}));
  EXPECT_TRUE(RunAndCompare(kHlo + kHloTANH, ErrorSpec{0.01, 0.01}));
  EXPECT_TRUE(RunAndCompare(kHlo + kHloELU, ErrorSpec{0.01, 0.01}));
  EXPECT_TRUE(
      RunAndCompare(absl::StrReplaceAll(kHlo + kHloRELU, {{"f32", "f16"}}),
                    ErrorSpec{0.03, 0.03}));
  EXPECT_TRUE(
      RunAndCompare(absl::StrReplaceAll(kHloNoPad + kHloRELU, {{"f32", "f16"}}),
                    ErrorSpec{0.03, 0.03}));
}

XLA_TEST_F(ConvolutionHloTest, TestFusedConv3D) {
  constexpr char kHlo[] = R"(
HloModule TestModule

ENTRY TestComputation {
  %p0 = f32[8,4,5,5,1] parameter(0)
  %p1 = f32[3,3,3,1,32] parameter(1)
  %conv = f32[8,4,5,5,32] convolution(p0, p1), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f
  %bias = f32[32] parameter(2)
  %broadcasted_bias = f32[8,4,5,5,32] broadcast(%bias), dimensions={4}
  %add = f32[8,4,5,5,32] add(%conv, %broadcasted_bias)
  %zero = f32[] constant(0)
  %zeros = f32[8,4,5,5,32] broadcast(%zero), dimensions={}
  ROOT relu = f32[8,4,5,5,32] maximum(%zeros, %add)
})";
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{0.01, 0.01}));
}

XLA_TEST_F(ConvolutionHloTest, TestBooleanInput) {
  constexpr char kHlo[] = R"(
HloModule TestModule

ENTRY TestComputation {
  constant.1 = pred[] constant(true)
  broadcast.2 = pred[3,3,3]{2,1,0} broadcast(constant.1), dimensions={}
  convolution.3 = pred[3,3,3]{2,1,0} convolution(broadcast.2, broadcast.2), window={size=3 pad=1_1}, dim_labels=bf0_oi0->bf0
  ROOT tuple.4 = (pred[3,3,3]{2,1,0}) tuple(convolution.3)
})";
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{0.01, 0.01}));
}

enum class PaddingMode {
  kFull,
  kHalf,  // also called 'same' padding
  kNo,    // also called 'valid' padding
};

// Convolution with LHS dilation, i.e. strided transposed convolution. We use
// a custom convolution algorithm for this case, so we need to test all cases
// (batch, input channels, output channels, etc.)
// Parameters are: batch size, input channels, output channels, padding mode,
// and whether to use asymmetric shapes (i.e. x != y)
class Transposed2DConvHloTest
    : public ConvolutionHloTest,
      public ::testing::WithParamInterface<
          std::tuple<int, int, int, PaddingMode, bool>> {
 public:
  Transposed2DConvHloTest()
      : batch_(std::get<0>(GetParam())),
        input_channels_(std::get<1>(GetParam())),
        output_channels_(std::get<2>(GetParam())),
        padding_mode_(std::get<3>(GetParam())),
        asymmetric_shapes_(std::get<4>(GetParam())),
        input_x_(5),
        input_y_(asymmetric_shapes_ ? input_x_ + 1 : input_x_),
        kernel_x_(3),
        kernel_y_(asymmetric_shapes_ ? kernel_x_ + 1 : kernel_x_),
        lhs_dilation_x_(2),
        lhs_dilation_y_(asymmetric_shapes_ ? lhs_dilation_x_ + 1
                                           : lhs_dilation_x_) {}

 public:
  int GetPaddingValue(int kernel_size, bool low) {
    switch (padding_mode_) {
      case PaddingMode::kFull:
        return 0;
      case PaddingMode::kHalf:
        if (low) {
          // Padding on the low side (i.e. before the first element in given
          // dimension)
          return kernel_size / 2;
        } else {
          // Padding on the high side (i.e. after the last element in given
          // dimension)
          return (kernel_size - 1) / 2;
        }
      case PaddingMode::kNo:
        return kernel_size - 1;
    }
  }

  auto GetWindow() {
    Window window;

    auto add_dim = [&](int size, int lhs_dilation) {
      auto dim = window.add_dimensions();
      dim->set_size(size);
      dim->set_stride(1);
      dim->set_padding_low(GetPaddingValue(size, /*low=*/true));
      dim->set_padding_high(GetPaddingValue(size, /*low=*/false));
      dim->set_window_dilation(1);
      dim->set_base_dilation(lhs_dilation);
    };

    add_dim(kernel_x_, lhs_dilation_x_);
    add_dim(kernel_y_, lhs_dilation_y_);

    return window;
  }

 public:
  int batch_;
  int input_channels_;
  int output_channels_;
  PaddingMode padding_mode_;
  bool asymmetric_shapes_;
  int input_x_;
  int input_y_;
  int kernel_x_;
  int kernel_y_;
  int lhs_dilation_x_;
  int lhs_dilation_y_;
};

XLA_TEST_P(Transposed2DConvHloTest, Simple) {
  const auto input_shape =
      ShapeUtil::MakeValidatedShape(
          F32, {batch_, input_channels_, input_x_, input_y_})
          .value();
  const auto kernel_shape =
      ShapeUtil::MakeValidatedShape(
          F32, {output_channels_, input_channels_, kernel_x_, kernel_y_})
          .value();

  const auto window = GetWindow();

  // clang-format off
  const std::string hlo = absl::StrCat(R"(
    HloModule TestModule

    ENTRY TestComputation {
      input.1 = )", input_shape.ToString(), R"( parameter(0)
      filter.2 = )", kernel_shape.ToString(), R"( parameter(1)
      ROOT conv.3 = convolution(input.1, filter.2),
        window={)", window_util::ToString(window), R"(},
        dim_labels=bf01_oi01->bf01
    }
  )");
  // clang-format on

  EXPECT_TRUE(RunAndCompare(hlo, ErrorSpec{0.01, 0.01}));
}

INSTANTIATE_TEST_SUITE_P(
    Transposed2DConvHloTest, Transposed2DConvHloTest,
    ::testing::Combine(::testing::Values(1, 2),  // Batch size
                       ::testing::Values(1, 3),  // Input channels
                       ::testing::Values(1, 5),  // Output channels
                       ::testing::Values(PaddingMode::kFull, PaddingMode::kNo,
                                         PaddingMode::kHalf),  // Padding mode
                       ::testing::Bool()  // Asymmetric shapes
                       ));

}  // namespace
}  // namespace xla
