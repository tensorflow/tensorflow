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
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/conv_grad_size_util.h"

namespace xla {

namespace {

// Common computation shared between AvgPool and AvgPoolGrad. Divide each
// element of an image by the count of elements that contributed to that
// element during pooling.
XlaOp AvgPoolDivideByCountWithGeneralPadding(
    XlaOp sums, PrimitiveType dtype, absl::Span<const int64> input_shape,
    absl::Span<const std::pair<int64, int64>> spatial_padding,
    absl::Span<const int64> ksize, absl::Span<const int64> stride,
    const TensorFormat& data_format) {
  // The padding shouldn't be included in the counts. We use another
  // ReduceWindow to find the right counts.
  const int num_spatial_dims = spatial_padding.size();

  std::vector<int64> input_dim_sizes(num_spatial_dims);
  std::vector<int64> window_dims(num_spatial_dims);
  std::vector<int64> window_ksize(num_spatial_dims);
  std::vector<int64> window_stride(num_spatial_dims);
  CHECK_EQ(data_format.num_spatial_dims(), num_spatial_dims)
      << "Invalid number of spatial dimensions in data format specification";
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
                  absl::Span<const int64> kernel_size,
                  absl::Span<const int64> stride,
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
    absl::Span<const std::pair<int64, int64>> spatial_padding,
    int num_spatial_dims, absl::Span<const int64> stride,
    const TensorFormat& data_format) {
  PaddingConfig padding_config;
  padding_config.mutable_dimensions()->Reserve(2 + num_spatial_dims);
  for (int i = 0; i < 2 + num_spatial_dims; ++i) {
    padding_config.add_dimensions();
  }
  CHECK_EQ(data_format.num_spatial_dims(), num_spatial_dims)
      << "Invalid number of spatial dimensions in data format specification";
  for (int i = 0; i < num_spatial_dims; ++i) {
    int dim = data_format.spatial_dimension(i);
    auto padding_dimension = padding_config.mutable_dimensions(dim);
    padding_dimension->set_edge_padding_low(spatial_padding[i].first);
    padding_dimension->set_edge_padding_high(spatial_padding[i].second);
  }
  return padding_config;
}

XlaOp AvgPoolDivideByCount(XlaOp pooled, absl::Span<const int64> input_size,
                           absl::Span<const int64> window_dimensions,
                           absl::Span<const int64> window_strides,
                           absl::Span<const std::pair<int64, int64>> padding,
                           PrimitiveType dtype, const TensorFormat& data_format,
                           bool counts_include_padding) {
  if (counts_include_padding) {
    // If counts include padding, all windows have the same number of elements
    // contributing to each average. Divide by the window size everywhere to get
    // the average.
    int64 window_size =
        std::accumulate(window_dimensions.begin(), window_dimensions.end(), 1,
                        [](int64 a, int64 b) { return a * b; });
    auto divisor = ConstantR0WithType(pooled.builder(), dtype, window_size);

    return pooled / divisor;
  } else {
    return AvgPoolDivideByCountWithGeneralPadding(pooled, dtype, input_size,
                                                  padding, window_dimensions,
                                                  window_strides, data_format);
  }
}

}  // namespace

XlaOp MaxPool(XlaOp operand, absl::Span<const int64> kernel_size,
              absl::Span<const int64> stride, Padding padding,
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

XlaOp AvgPool(XlaOp operand, absl::Span<const int64> kernel_size,
              absl::Span<const int64> stride,
              absl::Span<const std::pair<int64, int64>> padding,
              const TensorFormat& data_format,
              const bool counts_include_padding) {
  XlaBuilder* b = operand.builder();
  return b->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape operand_shape, b->GetShape(operand));
    PrimitiveType dtype = operand_shape.element_type();
    auto init_value = Zero(b, dtype);
    std::vector<int64> input_size(operand_shape.dimensions().begin(),
                                  operand_shape.dimensions().end());
    const int num_dims = kernel_size.size();
    const int num_spatial_dims = num_dims - 2;
    auto padding_config = MakeSpatialPaddingConfig(padding, num_spatial_dims,
                                                   stride, data_format);
    auto padded_operand = Pad(operand, Zero(b, dtype), padding_config);
    auto pooled = ComputeSums(padded_operand, init_value, kernel_size, stride,
                              data_format);
    return AvgPoolDivideByCount(pooled, input_size, kernel_size, stride,
                                padding, dtype, data_format,
                                counts_include_padding);
  });
}

std::vector<std::pair<int64, int64>> MakeSpatialPadding(
    absl::Span<const int64> input_size, absl::Span<const int64> kernel_size,
    absl::Span<const int64> stride, Padding padding,
    const TensorFormat& data_format) {
  const int num_spatial_dims = kernel_size.size() - 2;
  std::vector<int64> input_spatial_dimensions;
  std::vector<int64> kernel_size_spatial_dimensions;
  std::vector<int64> stride_spatial_dimensions;
  CHECK_EQ(data_format.num_spatial_dims(), num_spatial_dims)
      << "Invalid number of spatial dimensions in data format specification";
  for (int i = 0; i < num_spatial_dims; ++i) {
    int dim = data_format.spatial_dimension(i);
    input_spatial_dimensions.push_back(input_size[dim]);
    kernel_size_spatial_dimensions.push_back(kernel_size[dim]);
    stride_spatial_dimensions.push_back(stride[dim]);
  }
  return MakePadding(input_spatial_dimensions, kernel_size_spatial_dimensions,
                     stride_spatial_dimensions, padding);
}

XlaOp AvgPoolGrad(XlaOp out_backprop, absl::Span<const int64> gradients_size,
                  absl::Span<const int64> kernel_size,
                  absl::Span<const int64> stride,
                  absl::Span<const std::pair<int64, int64>> spatial_padding,
                  const TensorFormat& data_format,
                  const bool counts_include_padding) {
  XlaBuilder* b = out_backprop.builder();
  return b->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    const int num_dims = kernel_size.size();
    const int num_gradients = gradients_size.size();
    if (num_gradients != num_dims) {
      return tensorflow::errors::InvalidArgument("gradients must be ", num_dims,
                                                 "-dimensional");
    }

    TF_ASSIGN_OR_RETURN(Shape out_backprop_xla_shape,
                        b->GetShape(out_backprop));
    const int backprop_xla_num_dims =
        out_backprop_xla_shape.dimensions().size();
    if (backprop_xla_num_dims != num_dims) {
      return tensorflow::errors::InvalidArgument("out_backprop must be ",
                                                 num_dims, "-dimensional");
    }

    // We can think of average-pooling as:
    // * a convolution with a kernel consisting entirely of 1s, where the
    //   input feature and output feature are equal, and 0s everywhere else.
    // * followed by dividing by the counts.
    //
    // This then gives us an algorithm to build the gradient:
    // * divide out_backprop by the counts, followed by
    // * Conv2DBackpropInput specialized for that kernel, which simplifies to
    //   a Pad and a ReduceWindow.
    //
    // For an explanation of backpropagation for convolution, see the comments
    // in third_party/tensorflow/core/kernels/conv_grad_ops.h

    // TF filter shape is [ H, W, ..., inC, outC ]

    // The input gradients are computed by a convolution of the output gradients
    // and the filter, with some appropriate padding. See the comment at the top
    // of conv_grad_ops.h for details.
    PrimitiveType dtype = out_backprop_xla_shape.element_type();
    auto out_backprop_div = AvgPoolDivideByCount(
        out_backprop, gradients_size, kernel_size, stride, spatial_padding,
        dtype, data_format, counts_include_padding);

    // Pad the gradients in the spatial dimensions. We use the same padding
    // as Conv2DBackpropInput.
    PaddingConfig padding_config = MakeNoPaddingConfig(num_dims);
    std::vector<int64> padded_gradients_size(gradients_size.begin(),
                                             gradients_size.end());
    // First, pad the output gradients the same way as the input. The additional
    // padding will be removed as a last step before returning the input
    // gradients.
    const int num_spatial_dims = num_dims - 2;
    for (int i = 0; i < num_spatial_dims; ++i) {
      int dim = data_format.spatial_dimension(i);
      padded_gradients_size[dim] +=
          (spatial_padding[i].first + spatial_padding[i].second);
    }
    for (int i = 0; i < num_spatial_dims; ++i) {
      int dim = data_format.spatial_dimension(i);
      TF_ASSIGN_OR_RETURN(
          SpatialDimensionOutputSizeAndPadding conv_backprop_spatial_dim,
          ConvGradExtractAndVerifyDimension(
              /*input_size=*/padded_gradients_size[dim],
              /*filter_size=*/kernel_size[dim],
              /*output_size=*/out_backprop_xla_shape.dimensions(dim),
              /*dilation=*/1,
              /*stride=*/stride[dim], /*padding=*/Padding::kValid));
      auto* padding = padding_config.mutable_dimensions(dim);
      padding->set_edge_padding_low(conv_backprop_spatial_dim.pad_before);
      padding->set_edge_padding_high(conv_backprop_spatial_dim.pad_after);
      padding->set_interior_padding(stride[dim] - 1);
    }

    auto zero = Zero(b, dtype);
    auto padded_gradients = Pad(out_backprop_div, zero, padding_config);

    // in_backprop = padded_gradients <conv> ones
    std::vector<int64> ones(num_dims, 1LL);
    auto in_backprop =
        ReduceWindow(padded_gradients, Zero(b, dtype),
                     CreateScalarAddComputation(dtype, b), kernel_size,
                     /*window_strides=*/ones, Padding::kValid);
    // The input padding doesn't contribute to the gradient, remove it.
    std::vector<std::pair<int64, int64>> neg_spatial_padding;
    neg_spatial_padding.reserve(spatial_padding.size());
    for (const std::pair<int64, int64>& spatial_padding_dim : spatial_padding) {
      neg_spatial_padding.emplace_back(-spatial_padding_dim.first,
                                       -spatial_padding_dim.second);
    }
    auto remove_padding_config = MakeSpatialPaddingConfig(
        neg_spatial_padding, num_spatial_dims, stride, data_format);
    return Pad(in_backprop, zero, remove_padding_config);
  });
}

}  // namespace xla
