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
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
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
  ErrorSpec error_spec_ = ErrorSpec(1e-3);
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
  builder.Conv(lhs, rhs, {1, 1}, Padding::kValid);

  std::unique_ptr<Array4D<float>> aexpected =
      ReferenceUtil::ConvArray4D(*alhs, *arhs, {1, 1}, Padding::kValid);

  ComputeAndCompareR4<float>(&builder, *aexpected, {}, error_spec_);
}

TEST_F(ConvolutionTest, Convolve_1x1x1x2_1x1x1x2_Valid) {
  ComputationBuilder builder(client_, TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {1, 1, 1, 2});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 1, 1, 2});
    auto input = builder.Parameter(0, input_shape, "input");
    auto filter = builder.Parameter(1, filter_shape, "filter");
    builder.Conv(input, filter, {1, 1}, Padding::kValid);
  }

  Array4D<float> input(1, 1, 1, 2);
  input.FillWithYX(Array2D<float>({
      {1, 2},
  }));
  Array4D<float> filter(1, 1, 1, 2);
  filter.FillWithYX(Array2D<float>({
      {5, 6},
  }));

  std::unique_ptr<Array4D<float>> aexpected =
      ReferenceUtil::ConvArray4D(input, filter, {1, 1}, Padding::kValid);

  auto input_literal =
      client_->TransferToServer(*LiteralUtil::CreateR4FromArray4D(input))
          .ConsumeValueOrDie();
  auto filter_literal =
      client_->TransferToServer(*LiteralUtil::CreateR4FromArray4D(filter))
          .ConsumeValueOrDie();

  ComputeAndCompareR4<float>(&builder, *aexpected,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
}

// Tests valid padding for 2D convolution in raster space.
TEST_F(ConvolutionTest, Convolve_1x1x4x4_1x1x2x2_Valid) {
  ComputationBuilder builder(client_, TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {1, 1, 4, 4});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 1, 2, 2});
    auto input = builder.Parameter(0, input_shape, "input");
    auto filter = builder.Parameter(1, filter_shape, "filter");
    builder.Conv(input, filter, {1, 1}, Padding::kValid);
  }

  Array4D<float> input(1, 1, 4, 4);
  // clang-format off
  input.FillWithYX(Array2D<float>({
    {1,  2,  3,  4 },
    {5,  6,  7,  8 },
    {9,  10, 11, 12},
    {13, 14, 15, 16},
  }));
  // clang-format on
  Array4D<float> filter(1, 1, 2, 2);
  // clang-format off
  filter.FillWithYX(Array2D<float>({
    {5, 6},
    {7, 8},
  }));
  // clang-format on

  std::unique_ptr<Array4D<float>> aexpected =
      ReferenceUtil::ConvArray4D(input, filter, {1, 1}, Padding::kValid);

  auto input_literal =
      client_->TransferToServer(*LiteralUtil::CreateR4FromArray4D(input))
          .ConsumeValueOrDie();
  auto filter_literal =
      client_->TransferToServer(*LiteralUtil::CreateR4FromArray4D(filter))
          .ConsumeValueOrDie();

  ComputeAndCompareR4<float>(&builder, *aexpected,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
}

// Tests same padding for 2D convolution in raster space.
TEST_F(ConvolutionTest, Convolve_1x1x4x4_1x1x2x2_Same) {
  ComputationBuilder builder(client_, TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {1, 1, 4, 4});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 1, 2, 2});
    auto input = builder.Parameter(0, input_shape, "input");
    auto filter = builder.Parameter(1, filter_shape, "filter");
    builder.Conv(input, filter, {1, 1}, Padding::kSame);
  }

  Array4D<float> input(1, 1, 4, 4);
  // clang-format off
  input.FillWithYX(Array2D<float>({
    {1,  2,  3,  4 },
    {5,  6,  7,  8 },
    {9,  10, 11, 12},
    {13, 14, 15, 16},
  }));
  // clang-format on
  Array4D<float> filter(1, 1, 2, 2);
  // clang-format off
  filter.FillWithYX(Array2D<float>({
    {5, 6},
    {7, 8},
  }));
  // clang-format on

  std::unique_ptr<Array4D<float>> aexpected =
      ReferenceUtil::ConvArray4D(input, filter, {1, 1}, Padding::kSame);

  auto input_literal =
      client_->TransferToServer(*LiteralUtil::CreateR4FromArray4D(input))
          .ConsumeValueOrDie();
  auto filter_literal =
      client_->TransferToServer(*LiteralUtil::CreateR4FromArray4D(filter))
          .ConsumeValueOrDie();

  ComputeAndCompareR4<float>(&builder, *aexpected,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
}

// Tests same padding for 2D convolution in raster space with an odd sized
// kernel.
TEST_F(ConvolutionTest, Convolve_1x1x4x4_1x1x3x3_Same) {
  ComputationBuilder builder(client_, TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {1, 1, 4, 4});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 1, 3, 3});
    auto input = builder.Parameter(0, input_shape, "input");
    auto filter = builder.Parameter(1, filter_shape, "filter");
    builder.Conv(input, filter, {1, 1}, Padding::kSame);
  }

  Array4D<float> input(1, 1, 4, 4);
  // clang-format off
  input.FillWithYX(Array2D<float>({
    {1,  2,  3,  4 },
    {5,  6,  7,  8 },
    {9,  10, 11, 12},
    {13, 14, 15, 16},
  }));
  // clang-format on
  Array4D<float> filter(1, 1, 3, 3);
  // clang-format off
  filter.FillWithYX(Array2D<float>({
    { 5,  6,  7},
    { 8,  9, 10},
    {11, 12, 13},
  }));
  // clang-format on

  std::unique_ptr<Array4D<float>> aexpected =
      ReferenceUtil::ConvArray4D(input, filter, {1, 1}, Padding::kSame);

  auto input_literal =
      client_->TransferToServer(*LiteralUtil::CreateR4FromArray4D(input))
          .ConsumeValueOrDie();
  auto filter_literal =
      client_->TransferToServer(*LiteralUtil::CreateR4FromArray4D(filter))
          .ConsumeValueOrDie();

  ComputeAndCompareR4<float>(&builder, *aexpected,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
}

// TODO(b/32873825): implement 1D convolution on GPU.
XLA_TEST_F(ConvolutionTest, DISABLED_ON_GPU(Convolve1D_1x2x5_1x2x2_Valid)) {
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
      client_->TransferToServer(*LiteralUtil::CreateR3FromArray3D(input))
          .ConsumeValueOrDie();
  auto filter_literal =
      client_->TransferToServer(*LiteralUtil::CreateR3FromArray3D(filter))
          .ConsumeValueOrDie();

  ComputeAndCompareR3<float>(&builder, expected,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
}

// TODO(b/32873825): implement 3D convolution on GPU.
XLA_TEST_F(ConvolutionTest,
           DISABLED_ON_GPU(Convolve3D_1x4x2x3x3_2x2x2x3x3_Valid)) {
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
    dnums.set_batch_dimension(0);
    dnums.add_spatial_dimensions(1);
    dnums.add_spatial_dimensions(2);
    dnums.add_spatial_dimensions(3);
    dnums.set_feature_dimension(4);
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
  auto input_r1 = LiteralUtil::CreateR1<float>(input_elems);
  auto input_r5 =
      LiteralUtil::Reshape(*input_r1, input_dims).ConsumeValueOrDie();

  std::vector<float> filter_elems(ShapeUtil::ElementsIn(filter_shape));
  std::iota(filter_elems.begin(), filter_elems.end(), 1.0f);
  auto filter_r1 = LiteralUtil::CreateR1<float>(filter_elems);
  auto filter_r5 =
      LiteralUtil::Reshape(*filter_r1, filter_dims).ConsumeValueOrDie();

  auto expected_r1 = LiteralUtil::CreateR1<float>(
      {19554, 19962, 20370, 22110, 22590, 23070, 34890, 35730, 36570, 37446,
       38358, 39270, 50226, 51498, 52770, 52782, 54126, 55470});
  auto expected_r5 =
      LiteralUtil::Reshape(*expected_r1, {1, 3, 1, 2, 3}).ConsumeValueOrDie();

  auto input_literal = client_->TransferToServer(*input_r5).ConsumeValueOrDie();
  auto filter_literal =
      client_->TransferToServer(*filter_r5).ConsumeValueOrDie();

  ComputeAndCompareLiteral(&builder, *expected_r5,
                           {input_literal.get(), filter_literal.get()},
                           error_spec_);
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendCpuCompilerFlags(&flag_list);
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }
  testing::InitGoogleTest(&argc, argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return 2;
  }
  return RUN_ALL_TESTS();
}
