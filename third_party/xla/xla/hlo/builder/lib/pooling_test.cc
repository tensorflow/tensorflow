/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/hlo/builder/lib/pooling.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/padding.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/test_macros.h"

namespace xla {
namespace {

TensorFormat MakeNCHWFormat(int num_spatial_dims) {
  absl::InlinedVector<int64_t, 4> spatial_dimensions;
  for (int i = 0; i < num_spatial_dims; ++i) {
    spatial_dimensions.push_back(i + 2);
  }
  return TensorFormat(/*batch_dimension=*/0, /*feature_dimension=*/1,
                      /*spatial_dimensions=*/spatial_dimensions);
}

std::vector<std::pair<int64_t, int64_t>> MakeGeneralPadding(
    XlaOp input, absl::Span<const int64_t> kernel_size,
    absl::Span<const int64_t> stride, Padding padding,
    const xla::TensorFormat& data_format) {
  XlaBuilder* b = input.builder();
  Shape operand_shape = b->GetShape(input).value();
  std::vector<int64_t> input_size(operand_shape.dimensions().begin(),
                                  operand_shape.dimensions().end());
  return MakeSpatialPadding(input_size, kernel_size, stride, padding,
                            data_format);
}

// Add singleton batch and feature dimensions to spatial dimensions, according
// to 'data_format' specification.
std::vector<int64_t> ExpandWithBatchAndFeatureDimensions(
    absl::Span<const int64_t> spatial_dim_sizes,
    const xla::TensorFormat& data_format) {
  const int num_spatial_dims = spatial_dim_sizes.size();
  std::vector<int64_t> tensor_sizes(num_spatial_dims + 2, 1);
  for (int i = 0; i < num_spatial_dims; ++i) {
    int dim = data_format.spatial_dimension(i);
    tensor_sizes[dim] = spatial_dim_sizes[i];
  }
  return tensor_sizes;
}

class PoolingTest : public ClientLibraryTestBase {
 public:
  ErrorSpec error_spec_{0.0001};
};

XLA_TEST_F(PoolingTest, MaxPool2D) {
  XlaBuilder builder(TestName());

  XlaOp input = ConstantR4FromArray4D<float>(
      &builder, {{{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  auto stride = kernel_size;
  MaxPool(input, kernel_size, stride, Padding::kValid, data_format);

  ComputeAndCompareR4<float>(&builder, {{{{5, 4}}}}, {}, error_spec_);
}

XLA_TEST_F(PoolingTest, MaxPool2DWithPadding) {
  XlaBuilder builder(TestName());

  XlaOp input = ConstantR4FromArray4D<float>(
      &builder, {{{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  auto stride = kernel_size;
  MaxPool(input, kernel_size, stride, Padding::kSame, data_format);

  ComputeAndCompareR4<float>(&builder, {{{{5, 4, 5}}}}, {}, error_spec_);
}

XLA_TEST_F(PoolingTest, MaxPool2DWithPaddingAndStride) {
  XlaBuilder builder(TestName());

  XlaOp input = ConstantR4FromArray4D<float>(
      &builder, {{{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  auto stride = ExpandWithBatchAndFeatureDimensions({1, 1}, data_format);
  MaxPool(input, kernel_size, stride, Padding::kSame, data_format);

  ComputeAndCompareR4<float>(&builder, {{{{5, 4, 4, 5, 5}, {5, 4, 3, 2, 1}}}},
                             {}, error_spec_);
}

XLA_TEST_F(PoolingTest, AvgPool2D) {
  XlaBuilder builder(TestName());

  XlaOp input = ConstantR4FromArray4D<float>(
      &builder, {{{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  auto stride = kernel_size;
  auto padding = MakeGeneralPadding(input, kernel_size, stride, Padding::kValid,
                                    data_format);
  AvgPool(input, kernel_size, stride, padding, data_format,
          /*counts_include_padding=*/true);

  ComputeAndCompareR4<float>(&builder, {{{{3, 3}}}}, {}, error_spec_);
}

XLA_TEST_F(PoolingTest, AvgPool2DWithPadding) {
  XlaBuilder builder(TestName());

  XlaOp input = ConstantR4FromArray4D<float>(
      &builder, {{{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  auto stride = kernel_size;
  auto padding = MakeGeneralPadding(input, kernel_size, stride, Padding::kSame,
                                    data_format);
  AvgPool(input, kernel_size, stride, padding, data_format,
          /*counts_include_padding=*/false);

  ComputeAndCompareR4<float>(&builder, {{{{3, 3, 3}}}}, {}, error_spec_);
}

XLA_TEST_F(PoolingTest, AvgPool2DWithPaddingAndStride) {
  XlaBuilder builder(TestName());

  XlaOp input = ConstantR4FromArray4D<float>(
      &builder, {{{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  auto stride = ExpandWithBatchAndFeatureDimensions({1, 1}, data_format);
  auto padding = MakeGeneralPadding(input, kernel_size, stride, Padding::kSame,
                                    data_format);
  AvgPool(input, kernel_size, stride, padding, data_format,
          /*counts_include_padding=*/false);

  ComputeAndCompareR4<float>(&builder,
                             {{{{3, 3, 3, 3, 3}, {4.5, 3.5, 2.5, 1.5, 1}}}}, {},
                             error_spec_);
}

XLA_TEST_F(PoolingTest, AvgPool2DWithGeneralPaddingCountNotIncludePadding) {
  XlaBuilder builder(TestName());

  XlaOp input = ConstantR4FromArray4D<float>(
      &builder, {{{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({3, 3}, data_format);
  auto stride = kernel_size;
  AvgPool(input, kernel_size, stride, {{1, 1}, {2, 1}}, data_format,
          /*counts_include_padding=*/false);

  ComputeAndCompareR4<float>(&builder, {{{{3, 3}}}}, {}, error_spec_);
}

XLA_TEST_F(PoolingTest,
           AvgPool2DWithGeneralPaddingCountNotIncludePaddingAndStride) {
  XlaBuilder builder(TestName());

  XlaOp input = ConstantR4FromArray4D<float>(
      &builder, {{{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({3, 3}, data_format);
  auto stride = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  AvgPool(input, kernel_size, stride, {{2, 1}, {1, 1}}, data_format,
          /*counts_include_padding=*/false);

  ComputeAndCompareR4<float>(&builder, {{{{1.5, 3, 4.5}, {3, 3, 3}}}}, {},
                             error_spec_);
}

XLA_TEST_F(PoolingTest, AvgPool2DGradNoPadding) {
  XlaBuilder builder(TestName());
  for (bool counts_include_padding : {false, true}) {
    XlaOp out_backprop = ConstantR4FromArray4D<float>(&builder, {{{{1.}}}});
    auto data_format = MakeNCHWFormat(2);
    auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
    auto stride = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
    AvgPoolGrad(out_backprop, {1, 1, 3, 3}, kernel_size, stride,
                {{0, 0}, {0, 0}}, MakeNCHWFormat(2),
                /*counts_include_padding=*/counts_include_padding);
    // Without padding, counts_include_padding makes no difference.
    ComputeAndCompareR4<float>(
        &builder, {{{{0.25, 0.25, 0.}, {0.25, 0.25, 0.}, {0., 0., 0.}}}}, {},
        error_spec_);
  }
}

XLA_TEST_F(PoolingTest, AvgPool2DGradNoPaddingWithStride) {
  XlaBuilder builder(TestName());
  for (bool counts_include_padding : {false, true}) {
    XlaOp out_backprop =
        ConstantR4FromArray4D<float>(&builder, {{{{1., 1.}, {1., 1.}}}});
    auto data_format = MakeNCHWFormat(2);
    auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
    auto stride = ExpandWithBatchAndFeatureDimensions({1, 1}, data_format);
    AvgPoolGrad(out_backprop, {1, 1, 3, 3}, kernel_size, stride,
                {{0, 0}, {0, 0}}, MakeNCHWFormat(2),
                /*counts_include_padding=*/counts_include_padding);
    // Without padding, counts_include_padding makes no difference.
    ComputeAndCompareR4<float>(
        &builder, {{{{0.25, 0.5, 0.25}, {0.5, 1., 0.5}, {0.25, 0.5, 0.25}}}},
        {}, error_spec_);
  }
}

XLA_TEST_F(PoolingTest, AvgPool2DGradWithPadding) {
  XlaBuilder builder(TestName());

  XlaOp out_backprop =
      ConstantR4FromArray4D<float>(&builder, {{{{1., 1.}, {1., 1.}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  auto stride = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  AvgPoolGrad(out_backprop, {1, 1, 3, 3}, kernel_size, stride, {{1, 1}, {1, 1}},
              MakeNCHWFormat(2),
              /*counts_include_padding=*/true);
  ComputeAndCompareR4<float>(
      &builder,
      {{{{0.25, 0.25, 0.25}, {0.25, 0.25, 0.25}, {0.25, 0.25, 0.25}}}}, {},
      error_spec_);
}

XLA_TEST_F(PoolingTest, AvgPool2DGradWithPaddingCountNotIncludePadding) {
  XlaBuilder builder(TestName());

  XlaOp out_backprop =
      ConstantR4FromArray4D<float>(&builder, {{{{1., 1.}, {1., 1.}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  auto stride = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  AvgPoolGrad(out_backprop, {1, 1, 3, 3}, kernel_size, stride, {{1, 1}, {1, 1}},
              MakeNCHWFormat(2), false);
  ComputeAndCompareR4<float>(
      &builder, {{{{1., 0.5, 0.5}, {0.5, 0.25, 0.25}, {0.5, 0.25, 0.25}}}}, {},
      error_spec_);
}

XLA_TEST_F(PoolingTest, AvgPool2DGradWithPaddingCountWithStride) {
  XlaBuilder builder(TestName());

  XlaOp out_backprop =
      ConstantR4FromArray4D<float>(&builder, {{{{1., 1., 1., 1.},
                                                {1., 1., 1., 1.},
                                                {1., 1., 1., 1.},
                                                {1., 1., 1., 1.}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  auto stride = ExpandWithBatchAndFeatureDimensions({1, 1}, data_format);
  AvgPoolGrad(out_backprop, {1, 1, 3, 3}, kernel_size, stride, {{1, 1}, {1, 1}},
              MakeNCHWFormat(2), true);
  ComputeAndCompareR4<float>(&builder,
                             {{{{1., 1., 1.}, {1., 1., 1.}, {1., 1., 1.}}}}, {},
                             error_spec_);
}

XLA_TEST_F(PoolingTest,
           AvgPool2DGradWithPaddingCountWithStrideNotIncludePadding) {
  XlaBuilder builder(TestName());

  XlaOp out_backprop =
      ConstantR4FromArray4D<float>(&builder, {{{{1., 1., 1., 1.},
                                                {1., 1., 1., 1.},
                                                {1., 1., 1., 1.},
                                                {1., 1., 1., 1.}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  auto stride = ExpandWithBatchAndFeatureDimensions({1, 1}, data_format);
  AvgPoolGrad(out_backprop, {1, 1, 3, 3}, kernel_size, stride, {{1, 1}, {1, 1}},
              MakeNCHWFormat(2), false);
  ComputeAndCompareR4<float>(
      &builder, {{{{2.25, 1.5, 2.25}, {1.5, 1., 1.5}, {2.25, 1.5, 2.25}}}}, {},
      error_spec_);
}

}  // namespace
}  // namespace xla
