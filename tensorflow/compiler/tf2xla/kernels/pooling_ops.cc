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

// XLA specific pooling ops.

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/conv_grad_ops.h"
#include "tensorflow/core/kernels/pooling_ops_common.h"

namespace tensorflow {
namespace {

// Superclass of pooling ops.
class PoolingOp : public XlaOpKernel {
 public:
  PoolingOp(OpKernelConstruction* ctx, int num_spatial_dims)
      : XlaOpKernel(ctx), num_spatial_dims_(num_spatial_dims) {
    if (ctx->num_inputs() == 1) {
      std::vector<int32> ksize_int;
      std::vector<int32> stride_int;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("ksize", &ksize_int));
      OP_REQUIRES(ctx, ksize_int.size() == num_dims(),
                  errors::InvalidArgument("Sliding window ksize field must "
                                          "specify ",
                                          num_dims(), " dimensions"));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &stride_int));
      OP_REQUIRES(ctx, stride_int.size() == num_dims(),
                  errors::InvalidArgument("Sliding window stride field must "
                                          "specify ",
                                          num_dims(), " dimensions"));
      for (int i = 0; i < num_dims(); ++i) {
        ksize_.push_back(ksize_int[i]);
        stride_.push_back(stride_int[i]);
      }
    }
    Padding padding;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding));
    padding_ = (padding == VALID) ? xla::Padding::kValid : xla::Padding::kSame;
  }

  int num_dims() const { return num_spatial_dims_ + 2; }

  // Method that builds an initial value to use in reductions.
  virtual xla::ComputationDataHandle InitValue(xla::ComputationBuilder* b,
                                               DataType data_type) = 0;

  // The reduction operation to apply to each window.
  virtual const xla::Computation* Reduction(XlaOpKernelContext* ctx,
                                            DataType dtype) = 0;

  // A post-processing operation to apply on the outputs of the ReduceWindow.
  virtual xla::ComputationDataHandle PostProcessOutput(
      XlaOpKernelContext* ctx, const xla::ComputationDataHandle& output,
      DataType dtype, const TensorShape& input_shape) = 0;

  void Compile(XlaOpKernelContext* ctx) override {
    xla::ComputationDataHandle input = ctx->Input(0);
    const TensorShape input_shape = ctx->InputShape(0);

    std::vector<int64> ksize = ksize_;
    std::vector<int64> stride = stride_;
    if (ctx->num_inputs() != 1) {
      const TensorShape ksize_shape = ctx->InputShape(1);
      // Validate input sizes.
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(ksize_shape),
                  errors::InvalidArgument("ksize must be a vector, not shape ",
                                          ksize_shape.DebugString()));
      OP_REQUIRES(ctx, ksize_shape.num_elements() == num_dims(),
                  errors::InvalidArgument("Sliding window ksize field must "
                                          "specify ",
                                          num_dims(), " dimensions"));
      ksize.clear();
      OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(1, &ksize));

      const TensorShape stride_shape = ctx->InputShape(2);
      // Validate input sizes.
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(stride_shape),
                  errors::InvalidArgument("stride must be a vector, not shape ",
                                          stride_shape.DebugString()));
      OP_REQUIRES(ctx, stride_shape.num_elements() == num_dims(),
                  errors::InvalidArgument("Sliding window stride field must "
                                          "specify ",
                                          num_dims(), " dimensions"));
      stride.clear();
      OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(2, &stride));
    }
    OP_REQUIRES(ctx, input_shape.dims() == num_dims(),
                errors::InvalidArgument("Input to ", type_string(),
                                        " operator must have ", num_dims(),
                                        " dimensions"));

    const DataType type = input_type(0);
    xla::ComputationDataHandle pooled = ctx->builder()->ReduceWindow(
        input, InitValue(ctx->builder(), type), *Reduction(ctx, type), ksize,
        stride, padding_);
    ctx->SetOutput(0, PostProcessOutput(ctx, pooled, type, input_shape));
  }

 protected:
  const int num_spatial_dims_;
  std::vector<int64> ksize_;
  std::vector<int64> stride_;
  xla::Padding padding_;
  TensorFormat data_format_ = FORMAT_NHWC;
};

class MaxPoolOp : public PoolingOp {
 public:
  MaxPoolOp(OpKernelConstruction* ctx, int num_spatial_dims)
      : PoolingOp(ctx, /*num_spatial_dims=*/num_spatial_dims) {}

  xla::ComputationDataHandle InitValue(xla::ComputationBuilder* b,
                                       DataType data_type) override {
    return XlaHelpers::MinValue(b, data_type);
  }

  const xla::Computation* Reduction(XlaOpKernelContext* ctx,
                                    DataType dtype) override {
    return ctx->GetOrCreateMax(dtype);
  }

  xla::ComputationDataHandle PostProcessOutput(
      XlaOpKernelContext* ctx, const xla::ComputationDataHandle& output,
      DataType dtype, const TensorShape& input_shape) override {
    return output;
  }
};

class MaxPool2DOp : public MaxPoolOp {
 public:
  explicit MaxPool2DOp(OpKernelConstruction* ctx)
      : MaxPoolOp(ctx, /*num_spatial_dims=*/2) {
    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(ctx, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));
  }
};
REGISTER_XLA_OP(Name("MaxPool"), MaxPool2DOp);
REGISTER_XLA_OP(Name("MaxPoolV2")
                    .CompileTimeConstInput("ksize")
                    .CompileTimeConstInput("strides"),
                MaxPool2DOp);

class MaxPool3DOp : public MaxPoolOp {
 public:
  explicit MaxPool3DOp(OpKernelConstruction* ctx)
      : MaxPoolOp(ctx, /*num_spatial_dims=*/3) {}
};
REGISTER_XLA_OP(Name("MaxPool3D"), MaxPool3DOp);

// Common computation shared between AvgPool and AvgPoolGrad. Divide each
// element of an image by the count of elements that contributed to that
// element during pooling.
static xla::ComputationDataHandle AvgPoolDivideByCount(
    XlaOpKernelContext* ctx, const xla::ComputationDataHandle& output,
    DataType dtype, const TensorShape& input_shape, xla::Padding padding,
    const std::vector<int64>& ksize, const std::vector<int64>& stride,
    int num_spatial_dims, TensorFormat data_format) {
  if (padding == xla::Padding::kValid) {
    // In VALID padding, all windows have the same number of elements
    // contributing to each average. Divide by the window size everywhere to
    // get the average.
    int64 window_size = std::accumulate(ksize.begin(), ksize.end(), 1,
                                        [](int64 a, int64 b) { return a * b; });

    auto divisor =
        XlaHelpers::IntegerLiteral(ctx->builder(), dtype, window_size);
    return ctx->builder()->Div(output, divisor);
  } else {
    // For SAME padding, the padding shouldn't be included in the
    // counts. We use another ReduceWindow to find the right counts.

    // TODO(phawkins): use a less brute-force way to compute this. Only
    // the boundary regions will have interesting values here.

    std::vector<int64> input_dim_sizes(num_spatial_dims);
    std::vector<int64> window_dims(num_spatial_dims);
    std::vector<int64> window_ksize(num_spatial_dims);
    std::vector<int64> window_stride(num_spatial_dims);
    for (int i = 0; i < num_spatial_dims; ++i) {
      int dim = GetTensorSpatialDimIndex(num_spatial_dims + 2, data_format, i);
      input_dim_sizes[i] = input_shape.dim_size(dim);
      window_dims[i] = dim;
      window_ksize[i] = ksize[dim];
      window_stride[i] = stride[dim];
    }

    // Build a matrix of all 1s, with the same width/height as the input.
    auto ones = ctx->builder()->Broadcast(
        XlaHelpers::One(ctx->builder(), dtype), input_dim_sizes);

    // Perform a ReduceWindow with the same window size, strides, and padding
    // to count the number of contributions to each result element.
    auto counts = ctx->builder()->ReduceWindow(
        ones, XlaHelpers::Zero(ctx->builder(), dtype),
        *ctx->GetOrCreateAdd(dtype), window_ksize, window_stride,
        xla::Padding::kSame);

    return ctx->builder()->Div(output, counts, window_dims);
  }
}

class AvgPoolOp : public PoolingOp {
 public:
  AvgPoolOp(OpKernelConstruction* ctx, int num_spatial_dims)
      : PoolingOp(ctx, num_spatial_dims) {}

  xla::ComputationDataHandle InitValue(xla::ComputationBuilder* b,
                                       DataType data_type) override {
    return XlaHelpers::Zero(b, data_type);
  }

  const xla::Computation* Reduction(XlaOpKernelContext* ctx,
                                    DataType dtype) override {
    return ctx->GetOrCreateAdd(dtype);
  }

  xla::ComputationDataHandle PostProcessOutput(
      XlaOpKernelContext* ctx, const xla::ComputationDataHandle& output,
      DataType dtype, const TensorShape& input_shape) override {
    return AvgPoolDivideByCount(ctx, output, dtype, input_shape, padding_,
                                ksize_, stride_, num_spatial_dims_,
                                data_format_);
  }
};

class AvgPool2DOp : public AvgPoolOp {
 public:
  explicit AvgPool2DOp(OpKernelConstruction* ctx)
      : AvgPoolOp(ctx, /*num_spatial_dims=*/2) {
    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(ctx, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));
  }
};
REGISTER_XLA_OP(Name("AvgPool"), AvgPool2DOp);

class AvgPool3DOp : public AvgPoolOp {
 public:
  explicit AvgPool3DOp(OpKernelConstruction* ctx)
      : AvgPoolOp(ctx, /*num_spatial_dims=*/3) {}
};
REGISTER_XLA_OP(Name("AvgPool3D"), AvgPool3DOp);

// The operation to compute MaxPool gradients.
// It takes three inputs:
//   - The original input tensor
//   - The original output tensor
//   - Backprop tensor for output
// It produces one output: backprop tensor for input.
class MaxPoolGradOp : public XlaOpKernel {
 public:
  MaxPoolGradOp(OpKernelConstruction* ctx, int num_spatial_dims)
      : XlaOpKernel(ctx), num_spatial_dims_(num_spatial_dims) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ksize", &ksize_));
    OP_REQUIRES(ctx, ksize_.size() == num_dims(),
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify ",
                                        num_dims(), " dimensions"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &stride_));
    OP_REQUIRES(ctx, stride_.size() == num_dims(),
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify ",
                                        num_dims(), " dimensions"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
  }

  int num_dims() const { return num_spatial_dims_ + 2; }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape tensor_in_shape = ctx->InputShape(0);
    const TensorShape tensor_out_shape = ctx->InputShape(1);
    const TensorShape out_backprop_shape = ctx->InputShape(2);

    // For maxpooling, tensor_in should have num_dims() dimensions.
    OP_REQUIRES(ctx, tensor_in_shape.dims() == num_dims(),
                errors::InvalidArgument("tensor_in must be ", num_dims(),
                                        "-dimensional"));
    OP_REQUIRES(ctx, tensor_out_shape.dims() == num_dims(),
                errors::InvalidArgument("tensor_out must be ", num_dims(),
                                        "-dimensional"));
    // For maxpooling, out_backprop should have num_dims() dimensions.
    OP_REQUIRES(ctx, out_backprop_shape.dims() == num_dims(),
                errors::InvalidArgument("out_backprop must be ", num_dims(),
                                        "-dimensional"));

    // TODO(phawkins): The XLA version doesn't need tensor_out. Investigate
    // whether this is a good time/space tradeoff.
    auto input = ctx->Input(0);
    auto out_backprop = ctx->Input(2);

    xla::Padding xla_padding =
        (padding_ == VALID) ? xla::Padding::kValid : xla::Padding::kSame;

    xla::PrimitiveType element_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(input_type(2), &element_type));
    xla::ComputationDataHandle init_value =
        XlaHelpers::Zero(ctx->builder(), input_type(2));
    auto select = CreateScalarGeComputation(element_type, ctx->builder());
    auto scatter = CreateScalarAddComputation(element_type, ctx->builder());
    xla::ComputationDataHandle gradients = ctx->builder()->SelectAndScatter(
        input, select, ksize_, stride_, xla_padding, out_backprop, init_value,
        scatter);

    ctx->SetOutput(0, gradients);
  }

 protected:
  const int num_spatial_dims_;
  std::vector<int64> ksize_;
  std::vector<int64> stride_;
  Padding padding_;
  TensorFormat data_format_ = FORMAT_NHWC;
};

class MaxPool2DGradOp : public MaxPoolGradOp {
 public:
  explicit MaxPool2DGradOp(OpKernelConstruction* ctx)
      : MaxPoolGradOp(ctx, /*num_spatial_dims=*/2) {
    string data_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
    OP_REQUIRES(ctx, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
  }
};
REGISTER_XLA_OP(Name("MaxPoolGrad"), MaxPool2DGradOp);

class MaxPool3DGradOp : public MaxPoolGradOp {
 public:
  explicit MaxPool3DGradOp(OpKernelConstruction* ctx)
      : MaxPoolGradOp(ctx, /*num_spatial_dims=*/3) {}
};
REGISTER_XLA_OP(Name("MaxPool3DGrad"), MaxPool3DGradOp);

// Average-pooling gradient
class AvgPoolGradOp : public XlaOpKernel {
 public:
  AvgPoolGradOp(OpKernelConstruction* ctx, int num_spatial_dims)
      : XlaOpKernel(ctx), num_spatial_dims_(num_spatial_dims) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ksize", &ksize_));
    OP_REQUIRES(ctx, ksize_.size() == num_dims(),
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify ",
                                        num_dims(), " dimensions"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &stride_));
    OP_REQUIRES(ctx, stride_.size() == num_dims(),
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify ",
                                        num_dims(), " dimensions"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
    OP_REQUIRES(ctx, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  int num_dims() const { return num_spatial_dims_ + 2; }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape gradients_shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &gradients_shape));

    const TensorShape out_backprop_shape = ctx->InputShape(1);

    // For avgpooling, tensor_in_shape should have num_dims() dimensions.
    OP_REQUIRES(ctx, gradients_shape.dims() == num_dims(),
                errors::InvalidArgument("orig_input_shape must be ", num_dims(),
                                        "-dimensional"));

    // For avgpooling, out_backprop should have num_dims() dimensions.
    OP_REQUIRES(ctx, out_backprop_shape.dims() == num_dims(),
                errors::InvalidArgument("out_backprop must be ", num_dims(),
                                        "-dimensional"));

    int depth_dim = GetTensorFeatureDimIndex(num_dims(), data_format_);
    int64 depth = out_backprop_shape.dim_size(depth_dim);

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
    std::vector<int64> filter_dims(num_dims());
    for (int i = 0; i < num_spatial_dims_; ++i) {
      int dim = GetTensorSpatialDimIndex(num_dims(), data_format_, i);
      filter_dims[i] = ksize_[dim];
    }
    filter_dims[num_dims() - 2] = depth;
    filter_dims[num_dims() - 1] = depth;
    TensorShape filter_shape(filter_dims);

    // Reuse the logic from Conv2DBackpropInput to compute padding.
    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(
        ctx, ConvBackpropComputeDimensions(
                 type_string(), /*num_spatial_dims=*/num_spatial_dims_,
                 gradients_shape, filter_shape, out_backprop_shape, stride_,
                 padding_, data_format_, &dims));

    auto out_backprop = ctx->Input(1);

    // The input gradients are computed by a convolution of the output
    // gradients
    // and the filter, with some appropriate padding. See the comment at
    // the top of conv_grad_ops.h for details.
    DataType dtype = input_type(1);

    xla::Padding xla_padding =
        (padding_ == VALID) ? xla::Padding::kValid : xla::Padding::kSame;

    // Divide the out_backprop values by the counts for each spatial position.
    std::vector<int64> stride_int64s(stride_.begin(), stride_.end());
    auto out_backprop_div = AvgPoolDivideByCount(
        ctx, out_backprop, dtype, gradients_shape, xla_padding, ksize_,
        stride_int64s, num_spatial_dims_, data_format_);

    // Pad the gradients in the spatial dimensions. We use the same padding
    // as Conv2DBackpropInput.
    xla::PaddingConfig padding_config = xla::MakeNoPaddingConfig(num_dims());
    for (int i = 0; i < num_spatial_dims_; ++i) {
      int dim = GetTensorSpatialDimIndex(num_dims(), data_format_, i);
      auto* padding = padding_config.mutable_dimensions(dim);
      padding->set_edge_padding_low(dims.spatial_dims[i].pad_before);
      padding->set_edge_padding_high(dims.spatial_dims[i].pad_after);
      padding->set_interior_padding(dims.spatial_dims[i].stride - 1);
    }

    auto zero = XlaHelpers::Zero(ctx->builder(), dtype);
    auto padded_gradients =
        ctx->builder()->Pad(out_backprop_div, zero, padding_config);

    // in_backprop = padded_gradients <conv> ones
    std::vector<int64> ones(num_dims(), 1LL);
    xla::ComputationDataHandle in_backprop = ctx->builder()->ReduceWindow(
        padded_gradients, zero, *ctx->GetOrCreateAdd(dtype), ksize_,
        /* window_strides=*/ones, xla::Padding::kValid);

    ctx->SetOutput(0, in_backprop);
  }

 protected:
  const int num_spatial_dims_;
  std::vector<int64> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_ = FORMAT_NHWC;
};

class AvgPool2DGradOp : public AvgPoolGradOp {
 public:
  explicit AvgPool2DGradOp(OpKernelConstruction* ctx)
      : AvgPoolGradOp(ctx, /*num_spatial_dims=*/2) {
    string data_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
    OP_REQUIRES(ctx, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
  }
};
REGISTER_XLA_OP(Name("AvgPoolGrad").CompileTimeConstInput("orig_input_shape"),
                AvgPool2DGradOp);

class AvgPool3DGradOp : public AvgPoolGradOp {
 public:
  explicit AvgPool3DGradOp(OpKernelConstruction* ctx)
      : AvgPoolGradOp(ctx, /*num_spatial_dims=*/3) {}
};
REGISTER_XLA_OP(Name("AvgPool3DGrad").CompileTimeConstInput("orig_input_shape"),
                AvgPool3DGradOp);

}  // anonymous namespace
}  // namespace tensorflow
