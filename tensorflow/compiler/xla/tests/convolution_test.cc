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

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class ConvolutionTest : public ClientLibraryTestBase {
 protected:
#if XLA_TEST_BACKEND_GPU
  // XLA:GPU sometimes uses FFT convolution which isn't as precise as spatial
  // convolution. So relax the absolute error threshold.
  ErrorSpec error_spec_ = ErrorSpec(1e-2);
#else
  ErrorSpec error_spec_ = ErrorSpec(1e-4);
#endif
};

XLA_TEST_F(ConvolutionTest, ForwardPassConvolution_3x3x256_256_OutputZ_Iota) {
  const int kInputActivationSizeY = 3;
  const int kInputActivationSizeX = 3;
  const int kInputActivationSizeZ = 256;
  const int kKernelSizeX = 2;
  const int kKernelSizeY = 2;
  const int kOutputActivationSizeZ = 256;
  const int kMiniBatchSize = 4;
  auto alhs =
      MakeUnique<Array4D<float>>(kMiniBatchSize, kInputActivationSizeZ,
                                 kInputActivationSizeY, kInputActivationSizeX);
  alhs->FillWithMultiples(1.0f);
  ASSERT_EQ(3, alhs->width());
  ASSERT_EQ(3, alhs->height());

  auto arhs =
      MakeUnique<Array4D<float>>(kOutputActivationSizeZ, kInputActivationSizeZ,
                                 kKernelSizeY, kKernelSizeX);
  Array2D<float> rhs_raster({
      {1.0f, 0.0f},  // row 0
      {0.0f, 0.0f},  // row 1
  });
  arhs->FillWithYX(rhs_raster);
  ASSERT_EQ(2, arhs->width());
  ASSERT_EQ(2, arhs->height());

  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR4FromArray4D<float>(*alhs);
  auto rhs = builder.ConstantR4FromArray4D<float>(*arhs);
  auto conv = builder.Conv(lhs, rhs, {1, 1}, Padding::kValid);

  ComputeAndCompare(&builder, conv, {}, error_spec_);
}

TEST_F(ConvolutionTest, Convolve_1x1x1x2_1x1x1x2_Valid) {
  ComputationBuilder builder(client_, TestName());
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 1, 1, 2});
  Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 1, 1, 2});
  auto input = builder.Parameter(0, input_shape, "input");
  auto filter = builder.Parameter(1, filter_shape, "filter");
  auto conv = builder.Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> input_data(1, 1, 1, 2);
  input_data.FillWithYX(Array2D<float>({
      {1, 2},
  }));
  Array4D<float> filter_data(1, 1, 1, 2);
  filter_data.FillWithYX(Array2D<float>({
      {5, 6},
  }));

  ComputeAndCompare(&builder, conv,
                    {*Literal::CreateFromArray(input_data),
                     *Literal::CreateFromArray(filter_data)},
                    error_spec_);
}

// Tests valid padding for 2D convolution in raster space.
TEST_F(ConvolutionTest, Convolve_1x1x4x4_1x1x2x2_Valid) {
  ComputationBuilder builder(client_, TestName());
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 1, 4, 4});
  Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 1, 2, 2});
  auto input = builder.Parameter(0, input_shape, "input");
  auto filter = builder.Parameter(1, filter_shape, "filter");
  auto conv = builder.Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> input_data(1, 1, 4, 4);
  // clang-format off
  input_data.FillWithYX(Array2D<float>({
    {1,  2,  3,  4 },
    {5,  6,  7,  8 },
    {9,  10, 11, 12},
    {13, 14, 15, 16},
  }));
  // clang-format on
  Array4D<float> filter_data(1, 1, 2, 2);
  // clang-format off
  filter_data.FillWithYX(Array2D<float>({
    {5, 6},
    {7, 8},
  }));
  // clang-format on
  ComputeAndCompare(&builder, conv,
                    {*Literal::CreateFromArray(input_data),
                     *Literal::CreateFromArray(filter_data)},
                    error_spec_);
}

// Tests same padding for 2D convolution in raster space.
TEST_F(ConvolutionTest, Convolve_1x1x4x4_1x1x2x2_Same) {
  ComputationBuilder builder(client_, TestName());
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 1, 4, 4});
  Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 1, 2, 2});
  auto input = builder.Parameter(0, input_shape, "input");
  auto filter = builder.Parameter(1, filter_shape, "filter");
  auto conv = builder.Conv(input, filter, {1, 1}, Padding::kSame);

  Array4D<float> input_data(1, 1, 4, 4);
  // clang-format off
  input_data.FillWithYX(Array2D<float>({
    {1,  2,  3,  4 },
    {5,  6,  7,  8 },
    {9,  10, 11, 12},
    {13, 14, 15, 16},
  }));
  // clang-format on
  Array4D<float> filter_data(1, 1, 2, 2);
  // clang-format off
  filter_data.FillWithYX(Array2D<float>({
    {5, 6},
    {7, 8},
  }));
  // clang-format on
  ComputeAndCompare(&builder, conv,
                    {*Literal::CreateFromArray(input_data),
                     *Literal::CreateFromArray(filter_data)},
                    error_spec_);
}

// Tests same padding for 2D convolution in raster space with an odd sized
// kernel.
TEST_F(ConvolutionTest, Convolve_1x1x4x4_1x1x3x3_Same) {
  ComputationBuilder builder(client_, TestName());
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 1, 4, 4});
  Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 1, 3, 3});
  auto input = builder.Parameter(0, input_shape, "input");
  auto filter = builder.Parameter(1, filter_shape, "filter");
  auto conv = builder.Conv(input, filter, {1, 1}, Padding::kSame);

  Array4D<float> input_data(1, 1, 4, 4);
  // clang-format off
  input_data.FillWithYX(Array2D<float>({
    {1,  2,  3,  4 },
    {5,  6,  7,  8 },
    {9,  10, 11, 12},
    {13, 14, 15, 16},
  }));
  // clang-format on
  Array4D<float> filter_data(1, 1, 3, 3);
  // clang-format off
  filter_data.FillWithYX(Array2D<float>({
    { 5,  6,  7},
    { 8,  9, 10},
    {11, 12, 13},
  }));
  // clang-format on
  ComputeAndCompare(&builder, conv,
                    {*Literal::CreateFromArray(input_data),
                     *Literal::CreateFromArray(filter_data)},
                    error_spec_);
}

XLA_TEST_F(ConvolutionTest, Convolve1D_1x2x5_1x2x2_Valid) {
  ComputationBuilder builder(client_, TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {1, 2, 5});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
    auto input = builder.Parameter(0, input_shape, "input");
    auto filter = builder.Parameter(1, filter_shape, "filter");
    builder.Conv(input, filter, {1}, Padding::kValid);
  }

  Array3D<float> input({{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}});
  Array3D<float> filter({{{10, 20}, {30, 40}}});

  Array3D<float> expected({{{510, 610, 710, 810}}});

  auto input_literal =
      client_->TransferToServer(*Literal::CreateR3FromArray3D(input))
          .ConsumeValueOrDie();
  auto filter_literal =
      client_->TransferToServer(*Literal::CreateR3FromArray3D(filter))
          .ConsumeValueOrDie();

  ComputeAndCompareR3<float>(&builder, expected,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
}

XLA_TEST_F(ConvolutionTest, Convolve1D_1x2x5_1x2x2_WithRHSDilation) {
  ComputationBuilder builder(client_, TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {1, 2, 5});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
    auto input = builder.Parameter(0, input_shape, "input");
    auto filter = builder.Parameter(1, filter_shape, "filter");
    // Convolution dimensions are bf0_oi0->bo0.
    builder.ConvGeneralDilated(
        input, filter, /*window_strides=*/{1}, /*padding=*/{{0, 0}},
        /*lhs_dilation=*/{1}, /*rhs_dilation=*/{2},
        /*dimension_numbers=*/builder.CreateDefaultConvDimensionNumbers(1));
  }

  Array3D<float> input({{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}});
  Array3D<float> filter({{{10, 20}, {30, 40}}});

  Array3D<float> expected({{{570, 670, 770}}});

  auto input_literal =
      client_->TransferToServer(*Literal::CreateR3FromArray3D(input))
          .ConsumeValueOrDie();
  auto filter_literal =
      client_->TransferToServer(*Literal::CreateR3FromArray3D(filter))
          .ConsumeValueOrDie();

  ComputeAndCompareR3<float>(&builder, expected,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
}

XLA_TEST_F(ConvolutionTest, Convolve1D_1x2x5_1x2x2_WithLHSDilation) {
  ComputationBuilder builder(client_, TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {1, 2, 5});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
    auto input = builder.Parameter(0, input_shape, "input");
    auto filter = builder.Parameter(1, filter_shape, "filter");
    // Convolution dimensions are bf0_oi0->bo0.
    builder.ConvGeneralDilated(
        input, filter, /*window_strides=*/{1}, /*padding=*/{{0, 0}},
        /*lhs_dilation=*/{2}, /*rhs_dilation=*/{1},
        /*dimension_numbers=*/builder.CreateDefaultConvDimensionNumbers(1));
  }

  Array3D<float> input({{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}});
  Array3D<float> filter({{{10, 20}, {30, 40}}});

  Array3D<float> expected({{{190, 320, 230, 380, 270, 440, 310, 500}}});

  auto input_literal =
      client_->TransferToServer(*Literal::CreateR3FromArray3D(input))
          .ConsumeValueOrDie();
  auto filter_literal =
      client_->TransferToServer(*Literal::CreateR3FromArray3D(filter))
          .ConsumeValueOrDie();

  ComputeAndCompareR3<float>(&builder, expected,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
}

XLA_TEST_F(ConvolutionTest, Convolve1D_1x2x5_1x2x2_WithLHSAndRHSDilation) {
  ComputationBuilder builder(client_, TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {1, 2, 5});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
    auto input = builder.Parameter(0, input_shape, "input");
    auto filter = builder.Parameter(1, filter_shape, "filter");
    // Convolution dimensions are bf0_oi0->bo0.
    builder.ConvGeneralDilated(
        input, filter, /*window_strides=*/{1}, /*padding=*/{{0, 0}},
        /*lhs_dilation=*/{2}, /*rhs_dilation=*/{2},
        /*dimension_numbers=*/builder.CreateDefaultConvDimensionNumbers(1));
  }

  Array3D<float> input({{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}});
  Array3D<float> filter({{{10, 20}, {30, 40}}});

  Array3D<float> expected({{{510, 0, 610, 0, 710, 0, 810}}});

  auto input_literal =
      client_->TransferToServer(*Literal::CreateR3FromArray3D(input))
          .ConsumeValueOrDie();
  auto filter_literal =
      client_->TransferToServer(*Literal::CreateR3FromArray3D(filter))
          .ConsumeValueOrDie();

  ComputeAndCompareR3<float>(&builder, expected,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
}

XLA_TEST_F(ConvolutionTest, Convolve1D_1x2x5_1x2x2_WithPadding) {
  ComputationBuilder builder(client_, TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {1, 2, 5});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
    auto input = builder.Parameter(0, input_shape, "input");
    auto filter = builder.Parameter(1, filter_shape, "filter");
    // Convolution dimensions are bf0_oi0->bo0.
    builder.ConvGeneralDilated(
        input, filter, /*window_strides=*/{1}, /*padding=*/{{2, 2}},
        /*lhs_dilation=*/{1}, /*rhs_dilation=*/{1},
        /*dimension_numbers=*/builder.CreateDefaultConvDimensionNumbers(1));
  }

  Array3D<float> input({{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}});
  Array3D<float> filter({{{10, 20}, {30, 40}}});

  Array3D<float> expected({{{0, 260, 510, 610, 710, 810, 350, 0}}});

  auto input_literal =
      client_->TransferToServer(*Literal::CreateR3FromArray3D(input))
          .ConsumeValueOrDie();
  auto filter_literal =
      client_->TransferToServer(*Literal::CreateR3FromArray3D(filter))
          .ConsumeValueOrDie();

  ComputeAndCompareR3<float>(&builder, expected,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
}

XLA_TEST_F(ConvolutionTest, Convolve3D_1x4x2x3x3_2x2x2x3x3_Valid) {
  ComputationBuilder builder(client_, TestName());
  std::vector<int64> input_dims = {1, 4, 2, 3, 3};
  std::vector<int64> filter_dims = {2, 2, 2, 3, 3};
  Shape input_shape = ShapeUtil::MakeShape(F32, input_dims);
  Shape filter_shape = ShapeUtil::MakeShape(F32, filter_dims);
  {
    auto input = builder.Parameter(0, input_shape, "input");
    auto filter = builder.Parameter(1, filter_shape, "filter");

    // Tensorflow dimension numbers for 3D convolution.
    ConvolutionDimensionNumbers dnums;
    dnums.set_input_batch_dimension(0);
    dnums.set_output_batch_dimension(0);
    dnums.add_spatial_dimensions(1);
    dnums.add_spatial_dimensions(2);
    dnums.add_spatial_dimensions(3);
    dnums.set_input_feature_dimension(4);
    dnums.set_output_feature_dimension(4);
    dnums.add_kernel_spatial_dimensions(0);
    dnums.add_kernel_spatial_dimensions(1);
    dnums.add_kernel_spatial_dimensions(2);
    dnums.set_kernel_input_feature_dimension(3);
    dnums.set_kernel_output_feature_dimension(4);

    builder.ConvWithGeneralDimensions(input, filter, {1, 1, 1}, Padding::kValid,
                                      dnums);
  }

  std::vector<float> input_elems(ShapeUtil::ElementsIn(input_shape));
  std::iota(input_elems.begin(), input_elems.end(), 1.0f);
  auto input_r1 = Literal::CreateR1<float>(input_elems);
  auto input_r5 = input_r1->Reshape(input_dims).ConsumeValueOrDie();

  std::vector<float> filter_elems(ShapeUtil::ElementsIn(filter_shape));
  std::iota(filter_elems.begin(), filter_elems.end(), 1.0f);
  auto filter_r1 = Literal::CreateR1<float>(filter_elems);
  auto filter_r5 = filter_r1->Reshape(filter_dims).ConsumeValueOrDie();

  auto expected_r1 = Literal::CreateR1<float>(
      {19554, 19962, 20370, 22110, 22590, 23070, 34890, 35730, 36570, 37446,
       38358, 39270, 50226, 51498, 52770, 52782, 54126, 55470});
  auto expected_r5 = expected_r1->Reshape({1, 3, 1, 2, 3}).ConsumeValueOrDie();

  auto input_literal = client_->TransferToServer(*input_r5).ConsumeValueOrDie();
  auto filter_literal =
      client_->TransferToServer(*filter_r5).ConsumeValueOrDie();

  ComputeAndCompareLiteral(&builder, *expected_r5,
                           {input_literal.get(), filter_literal.get()},
                           error_spec_);
}

XLA_TEST_F(ConvolutionTest, Convolve2D_1x3x3x5_3x3x5x5_Valid) {
  ComputationBuilder builder(client_, TestName());
  std::vector<int64> input_dims = {1, 3, 3, 5};
  std::vector<int64> filter_dims = {3, 3, 5, 3};
  Shape input_shape = ShapeUtil::MakeShape(F32, input_dims);
  Shape filter_shape = ShapeUtil::MakeShape(F32, filter_dims);
  {
    auto input = builder.Parameter(0, input_shape, "input");
    auto filter = builder.Parameter(1, filter_shape, "filter");

    // Tensorflow dimension numbers for 2D convolution.
    ConvolutionDimensionNumbers dnums;
    dnums.set_input_batch_dimension(0);
    dnums.set_output_batch_dimension(0);
    dnums.add_spatial_dimensions(1);
    dnums.add_spatial_dimensions(2);
    dnums.set_input_feature_dimension(3);
    dnums.set_output_feature_dimension(3);
    dnums.add_kernel_spatial_dimensions(0);
    dnums.add_kernel_spatial_dimensions(1);
    dnums.set_kernel_input_feature_dimension(2);
    dnums.set_kernel_output_feature_dimension(3);

    builder.ConvWithGeneralDimensions(input, filter, {1, 1}, Padding::kValid,
                                      dnums);
  }

  std::vector<float> input_elems(ShapeUtil::ElementsIn(input_shape));
  std::iota(input_elems.begin(), input_elems.end(), 1.0f);
  auto input_r1 = Literal::CreateR1<float>(input_elems);
  auto input_r4 = input_r1->Reshape(input_dims).ConsumeValueOrDie();

  std::vector<float> filter_elems(ShapeUtil::ElementsIn(filter_shape));
  std::iota(filter_elems.begin(), filter_elems.end(), 1.0f);
  auto filter_r1 = Literal::CreateR1<float>(filter_elems);
  auto filter_r4 = filter_r1->Reshape(filter_dims).ConsumeValueOrDie();

  auto expected_r1 = Literal::CreateR1<float>({92115, 93150, 94185});
  auto expected_r4 = expected_r1->Reshape({1, 1, 1, 3}).ConsumeValueOrDie();

  auto input_literal = client_->TransferToServer(*input_r4).ConsumeValueOrDie();
  auto filter_literal =
      client_->TransferToServer(*filter_r4).ConsumeValueOrDie();

  ComputeAndCompareLiteral(&builder, *expected_r4,
                           {input_literal.get(), filter_literal.get()},
                           error_spec_);
}

struct Convolve1DTestParam {
  int64 input_feature;
  int64 output_feature;
  int64 batch;
  int64 window_size;
  int64 num_windows;
};

class Convolve1D1WindowTest
    : public ConvolutionTest,
      public ::testing::WithParamInterface<Convolve1DTestParam> {};

XLA_TEST_P(Convolve1D1WindowTest, Convolve1D1Window) {
  ComputationBuilder builder(client_, TestName());
  int64 input_feature = GetParam().input_feature;
  int64 output_feature = GetParam().output_feature;
  int64 batch = GetParam().batch;
  int64 num_windows = GetParam().num_windows;
  int64 window_size = GetParam().window_size;
  std::vector<int64> input_dims = {batch, window_size + num_windows - 1,
                                   input_feature};
  std::vector<int64> filter_dims = {window_size, input_feature, output_feature};
  Shape input_shape = ShapeUtil::MakeShape(F32, input_dims);
  Shape filter_shape = ShapeUtil::MakeShape(F32, filter_dims);
  {
    auto input = builder.Parameter(0, input_shape, "input");
    auto filter = builder.Parameter(1, filter_shape, "filter");

    // Tensorflow dimension numbers for 1D convolution.
    ConvolutionDimensionNumbers dnums;
    dnums.set_input_batch_dimension(0);
    dnums.set_output_batch_dimension(0);
    dnums.add_spatial_dimensions(1);
    dnums.set_input_feature_dimension(2);
    dnums.set_output_feature_dimension(2);
    dnums.add_kernel_spatial_dimensions(0);
    dnums.set_kernel_input_feature_dimension(1);
    dnums.set_kernel_output_feature_dimension(2);

    builder.ConvWithGeneralDimensions(input, filter, {1}, Padding::kValid,
                                      dnums);
  }

  std::vector<float> input_elems(ShapeUtil::ElementsIn(input_shape), 1.0);
  auto input_r1 = Literal::CreateR1<float>(input_elems);
  auto input_r3 = input_r1->Reshape(input_dims).ConsumeValueOrDie();

  std::vector<float> filter_elems(ShapeUtil::ElementsIn(filter_shape), 1.0);

  auto filter_r1 = Literal::CreateR1<float>(filter_elems);
  auto filter_r3 = filter_r1->Reshape(filter_dims).ConsumeValueOrDie();

  std::vector<float> expect_elems(batch * output_feature * num_windows,
                                  window_size * input_feature);
  auto expected_r1 = Literal::CreateR1<float>(expect_elems);
  auto expected_r3 = expected_r1->Reshape({batch, num_windows, output_feature})
                         .ConsumeValueOrDie();

  auto input_literal = client_->TransferToServer(*input_r3).ConsumeValueOrDie();
  auto filter_literal =
      client_->TransferToServer(*filter_r3).ConsumeValueOrDie();
  ComputeAndCompareLiteral(&builder, *expected_r3,
                           {input_literal.get(), filter_literal.get()},
                           error_spec_);
}

INSTANTIATE_TEST_CASE_P(
    Convolve1D1WindowTest_Instantiation, Convolve1D1WindowTest,
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

}  // namespace
}  // namespace xla
