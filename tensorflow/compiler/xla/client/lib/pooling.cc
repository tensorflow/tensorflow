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

#include "tensorflow/compiler/xla/client/lib/pooling.h"
#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"

namespace xla {

namespace {

// Common computation shared between AvgPool and AvgPoolGrad. Divide each
// element of an image by the count of elements that contributed to that
// element during pooling.
XlaOp AvgPoolDivideByCountWithGeneralPadding(
    XlaOp sums, PrimitiveType dtype,
    tensorflow::gtl::ArraySlice<int64> input_shape,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> spatial_padding,
    tensorflow::gtl::ArraySlice<int64> ksize,
    tensorflow::gtl::ArraySlice<int64> stride,
    const TensorFormat& data_format) {
  // The padding shouldn't be included in the counts. We use another
  // ReduceWindow to find the right counts.
  const int num_spatial_dims = spatial_padding.size();

  std::vector<int64> input_dim_sizes(num_spatial_dims);
  std::vector<int64> window_dims(num_spatial_dims);
  std::vector<int64> window_ksize(num_spatial_dims);
  std::vector<int64> window_stride(num_spatial_dims);
  CHECK_EQ(data_format.num_spatial_dims(), num_spatial_dims)
      << "Invalid number of spatial dimentions in data format specification";
  for (int i = 0; i < num_spatial_dims; ++i) {
    int dim = data_format.spatial_dimension(i);
    input_dim_sizes[i] = input_shape[dim];
    window_dims[i] = dim;
    window_ksize[i] = ksize[dim];
    window_stride[i] = stride[dim];
  }

  XlaBuilder* b = sums.builder();
  // Build a matrix of all 1s, with the same width/height as the input.
  auto ones = Broadcast(One(b, dtype), input_dim_sizes);
  PaddingConfig padding_config;
  for (int i = 0; i < num_spatial_dims; ++i) {
    auto dims = padding_config.add_dimensions();
    dims->set_edge_padding_low(spatial_padding[i].first);
    dims->set_edge_padding_high(spatial_padding[i].second);
  }
  auto zero = Zero(b, dtype);
  auto padded_ones = Pad(ones, zero, padding_config);

  // Perform a ReduceWindow with the same window size, strides, and padding
  // to count the number of contributions to each result element.
  auto counts =
      ReduceWindow(padded_ones, zero, CreateScalarAddComputation(dtype, b),
                   window_ksize, window_stride, Padding::kValid);

  return Div(sums, counts, window_dims);
}

// Sums all elements in the window specified by 'kernel_size' and 'stride'.
XlaOp ComputeSums(XlaOp operand, XlaOp init_value,
                  tensorflow::gtl::ArraySlice<int64> kernel_size,
                  tensorflow::gtl::ArraySlice<int64> stride,
                  const TensorFormat& data_format) {
  XlaBuilder* b = operand.builder();
  return b->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape operand_shape, b->GetShape(operand));
    TF_ASSIGN_OR_RETURN(Shape init_shape, b->GetShape(init_value));
    PrimitiveType accumulation_type = init_shape.element_type();
    auto add_computation = CreateScalarAddComputation(accumulation_type, b);
    return ReduceWindow(operand, init_value, add_computation, kernel_size,
                        stride, Padding::kValid);
  });
}

// Creates a padding configuration out of spatial padding values.
PaddingConfig MakeSpatialPaddingConfig(
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> spatial_padding,
    tensorflow::gtl::ArraySlice<int64> kernel_size,
    tensorflow::gtl::ArraySlice<int64> stride,
    const TensorFormat& data_format) {
  const int num_spatial_dims = kernel_size.size() - 2;
  PaddingConfig padding_config;
  for (int i = 0; i < 2 + num_spatial_dims; ++i) {
    padding_config.add_dimensions();
  }
  CHECK_EQ(data_format.num_spatial_dims(), num_spatial_dims)
      << "Invalid number of spatial dimentions in data format specification";
  for (int i = 0; i < num_spatial_dims; ++i) {
    int dim = data_format.spatial_dimension(i);
    auto padding_dimension = padding_config.mutable_dimensions(dim);
    padding_dimension->set_edge_padding_low(spatial_padding[i].first);
    padding_dimension->set_edge_padding_high(spatial_padding[i].second);
  }
  return padding_config;
}

}  // namespace

XlaOp MaxPool(XlaOp operand, tensorflow::gtl::ArraySlice<int64> kernel_size,
              tensorflow::gtl::ArraySlice<int64> stride, Padding padding,
              const TensorFormat& data_format) {
  XlaBuilder* b = operand.builder();
  return b->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape operand_shape, b->GetShape(operand));
    PrimitiveType dtype = operand_shape.element_type();
    auto max_computation = CreateScalarMaxComputation(dtype, b);
    auto init_value = MinValue(b, dtype);
    return ReduceWindow(operand, init_value, max_computation, kernel_size,
                        stride, padding);
  });
}

XlaOp AvgPool(XlaOp operand, tensorflow::gtl::ArraySlice<int64> kernel_size,
              tensorflow::gtl::ArraySlice<int64> stride,
              tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
              const TensorFormat& data_format,
              const bool counts_include_padding) {
  XlaBuilder* b = operand.builder();
  return b->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape operand_shape, b->GetShape(operand));
    PrimitiveType dtype = operand_shape.element_type();
    auto init_value = Zero(b, dtype);
    std::vector<int64> input_size(operand_shape.dimensions().begin(),
                                  operand_shape.dimensions().end());
    auto padding_config =
        MakeSpatialPaddingConfig(padding, kernel_size, stride, data_format);
    auto padded_operand = Pad(operand, Zero(b, dtype), padding_config);
    auto pooled = ComputeSums(padded_operand, init_value, kernel_size, stride,
                              data_format);
    if (counts_include_padding) {
      // If counts include padding, all windows have the same number of elements
      // contributing to each average. Divide by the window size everywhere to
      // get the average.
      int64 window_size =
          std::accumulate(kernel_size.begin(), kernel_size.end(), 1,
                          [](int64 x, int64 y) { return x * y; });

      auto divisor = ConstantR0WithType(b, dtype, window_size);
      return pooled / divisor;
    } else {
      return AvgPoolDivideByCountWithGeneralPadding(
          pooled, dtype, input_size, padding, kernel_size, stride, data_format);
    }
  });
}

std::vector<std::pair<int64, int64>> MakeSpatialPadding(
    tensorflow::gtl::ArraySlice<int64> input_size,
    tensorflow::gtl::ArraySlice<int64> kernel_size,
    tensorflow::gtl::ArraySlice<int64> stride, Padding padding,
    const TensorFormat& data_format) {
  const int num_spatial_dims = kernel_size.size() - 2;
  std::vector<int64> input_spatial_dimensions;
  std::vector<int64> kernel_size_spatial_dimensions;
  std::vector<int64> stride_spatial_dimensions;
  CHECK_EQ(data_format.num_spatial_dims(), num_spatial_dims)
      << "Invalid number of spatial dimentions in data format specification";
  for (int i = 0; i < num_spatial_dims; ++i) {
    int dim = data_format.spatial_dimension(i);
    input_spatial_dimensions.push_back(input_size[dim]);
    kernel_size_spatial_dimensions.push_back(kernel_size[dim]);
    stride_spatial_dimensions.push_back(stride[dim]);
  }
  return MakePadding(input_spatial_dimensions, kernel_size_spatial_dimensions,
                     stride_spatial_dimensions, padding);
}

}  // namespace xla
