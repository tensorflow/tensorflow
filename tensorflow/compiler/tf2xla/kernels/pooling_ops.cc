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

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/pooling.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/pooling_ops_common.h"

namespace tensorflow {
namespace {

// Superclass of pooling ops.
class PoolingOp : public XlaOpKernel {
 public:
  PoolingOp(OpKernelConstruction* ctx, int num_spatial_dims,
            const DataType reduction_type)
      : XlaOpKernel(ctx),
        num_spatial_dims_(num_spatial_dims),
        reduction_type_(reduction_type) {
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

    OP_REQUIRES_OK(
        ctx, DataTypeToPrimitiveType(reduction_type_, &xla_reduction_type_));
  }

  int num_dims() const { return num_spatial_dims_ + 2; }

 protected:
  xla::StatusOr<std::vector<int64>> GetKernelSize(XlaOpKernelContext* ctx) {
    if (ctx->num_inputs() == 1) {
      return ksize_;
    }
    const TensorShape ksize_shape = ctx->InputShape(1);
    // Validate input sizes.
    if (!TensorShapeUtils::IsVector(ksize_shape)) {
      return errors::InvalidArgument("ksize must be a vector, not shape ",
                                     ksize_shape.DebugString());
    }
    if (ksize_shape.num_elements() != num_dims()) {
      return errors::InvalidArgument(
          "Sliding window ksize field must "
          "specify ",
          num_dims(), " dimensions");
    }
    std::vector<int64> ksize;
    auto status = ctx->ConstantInputAsIntVector(1, &ksize);
    if (!status.ok()) {
      return status;
    }
    return ksize;
  }

  xla::StatusOr<std::vector<int64>> GetStride(XlaOpKernelContext* ctx) {
    if (ctx->num_inputs() == 1) {
      return stride_;
    }
    const TensorShape stride_shape = ctx->InputShape(2);
    // Validate input sizes.
    if (!TensorShapeUtils::IsVector(stride_shape)) {
      return errors::InvalidArgument("stride must be a vector, not shape ",
                                     stride_shape.DebugString());
    }
    if (stride_shape.num_elements() != num_dims()) {
      return errors::InvalidArgument(
          "Sliding window stride field must "
          "specify ",
          num_dims(), " dimensions");
    }
    std::vector<int64> stride;
    auto status = ctx->ConstantInputAsIntVector(2, &stride);
    if (!status.ok()) {
      return status;
    }
    return stride;
  }

 protected:
  const int num_spatial_dims_;
  std::vector<int64> ksize_;
  std::vector<int64> stride_;
  xla::Padding padding_;
  TensorFormat data_format_ = FORMAT_NHWC;
  DataType reduction_type_;
  xla::PrimitiveType xla_reduction_type_;
};

// Converts the tensor data format to the one required by the XLA pooling
// library.
xla::TensorFormat XlaTensorFormat(tensorflow::TensorFormat data_format,
                                  int num_spatial_dims) {
  int num_dims = num_spatial_dims + 2;
  int batch_dimension = GetTensorBatchDimIndex(num_dims, data_format);
  int feature_dimension = GetTensorFeatureDimIndex(num_dims, data_format);
  absl::InlinedVector<int64, 4> spatial_dimensions(num_spatial_dims);
  for (int spatial_dim = 0; spatial_dim < num_spatial_dims; ++spatial_dim) {
    spatial_dimensions[spatial_dim] =
        GetTensorSpatialDimIndex(num_dims, data_format, spatial_dim);
  }
  return xla::TensorFormat(/*batch_dimension=*/batch_dimension,
                           /*feature_dimension=*/feature_dimension,
                           /*spatial_dimensions=*/spatial_dimensions);
}

class MaxPoolOp : public PoolingOp {
 public:
  MaxPoolOp(OpKernelConstruction* ctx, int num_spatial_dims)
      : PoolingOp(ctx, /*num_spatial_dims=*/num_spatial_dims,
                  /*reduction_type=*/ctx->input_type(0)) {
    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(ctx, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    auto ksize_or_error = GetKernelSize(ctx);
    OP_REQUIRES_OK(ctx, ksize_or_error.status());
    std::vector<int64> ksize = ksize_or_error.ValueOrDie();

    auto stride_or_error = GetStride(ctx);
    OP_REQUIRES_OK(ctx, stride_or_error.status());
    std::vector<int64> stride = stride_or_error.ValueOrDie();

    const TensorShape input_shape = ctx->InputShape(0);
    OP_REQUIRES(ctx, input_shape.dims() == num_dims(),
                errors::InvalidArgument("Input to ", type_string(),
                                        " operator must have ", num_dims(),
                                        " dimensions"));

    auto pooling =
        xla::MaxPool(ctx->Input(0), ksize, stride, padding_,
                     XlaTensorFormat(data_format_, input_shape.dims() - 2));
    ctx->SetOutput(0, pooling);
  }
};

class MaxPool2DOp : public MaxPoolOp {
 public:
  explicit MaxPool2DOp(OpKernelConstruction* ctx)
      : MaxPoolOp(ctx, /*num_spatial_dims=*/2) {}
};
REGISTER_XLA_OP(Name("MaxPool"), MaxPool2DOp);
REGISTER_XLA_OP(Name("MaxPoolV2")
                    .CompileTimeConstantInput("ksize")
                    .CompileTimeConstantInput("strides"),
                MaxPool2DOp);

class MaxPool3DOp : public MaxPoolOp {
 public:
  explicit MaxPool3DOp(OpKernelConstruction* ctx)
      : MaxPoolOp(ctx, /*num_spatial_dims=*/3) {}
};
REGISTER_XLA_OP(Name("MaxPool3D"), MaxPool3DOp);

class AvgPoolOp : public PoolingOp {
 public:
  AvgPoolOp(OpKernelConstruction* ctx, int num_spatial_dims)
      : PoolingOp(ctx, /*num_spatial_dims=*/num_spatial_dims,
                  /*reduction_type=*/
                  XlaHelpers::SumAccumulationType(ctx->input_type(0))) {
    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(ctx, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    auto ksize_or_error = GetKernelSize(ctx);
    OP_REQUIRES_OK(ctx, ksize_or_error.status());
    std::vector<int64> ksize = ksize_or_error.ValueOrDie();

    auto stride_or_error = GetStride(ctx);
    OP_REQUIRES_OK(ctx, stride_or_error.status());
    std::vector<int64> stride = stride_or_error.ValueOrDie();

    const TensorShape input_shape = ctx->InputShape(0);
    OP_REQUIRES(ctx, input_shape.dims() == num_dims(),
                errors::InvalidArgument("Input to ", type_string(),
                                        " operator must have ", num_dims(),
                                        " dimensions"));

    auto xla_data_format =
        XlaTensorFormat(data_format_, input_shape.dims() - 2);
    auto spatial_padding = MakeSpatialPadding(
        input_shape.dim_sizes(), ksize, stride, padding_, xla_data_format);

    // Convert the input to the reduction type.
    auto converted_input =
        ConvertElementType(ctx->Input(0), xla_reduction_type_);
    auto pooling =
        xla::AvgPool(converted_input, ksize, stride, spatial_padding,
                     xla_data_format, padding_ == xla::Padding::kValid);
    // Convert the pooling result back to the input type before returning it.
    ctx->SetOutput(0, ConvertElementType(pooling, ctx->input_xla_type(0)));
  }
};

class AvgPool2DOp : public AvgPoolOp {
 public:
  explicit AvgPool2DOp(OpKernelConstruction* ctx)
      : AvgPoolOp(ctx, /*num_spatial_dims=*/2) {}
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
    if (ctx->num_inputs() == 3) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("ksize", &ksize_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &stride_));
    }
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
  }

  int num_dims() const { return num_spatial_dims_ + 2; }

  void Compile(XlaOpKernelContext* ctx) override {
    if (ctx->num_inputs() != 3) {
      OP_REQUIRES(
          ctx, ctx->num_inputs() == 5,
          errors::InvalidArgument("Must supply ksize and stride arguments."));
      const TensorShape ksize_shape = ctx->InputShape(3);
      // Validate input sizes.
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(ksize_shape),
                  errors::InvalidArgument("ksize must be a vector, not shape ",
                                          ksize_shape.DebugString()));
      OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(3, &ksize_));

      const TensorShape stride_shape = ctx->InputShape(4);
      // Validate input sizes.
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(stride_shape),
                  errors::InvalidArgument("stride must be a vector, not shape ",
                                          stride_shape.DebugString()));
      OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(4, &stride_));
    }

    OP_REQUIRES(ctx, ksize_.size() == num_dims(),
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify ",
                                        num_dims(), " dimensions"));
    OP_REQUIRES(ctx, stride_.size() == num_dims(),
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify ",
                                        num_dims(), " dimensions"));

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

    // Create a MaxPool operation to check the expected resulting shape, and
    // then throw away the operation because we don't actually need it here.
    TensorShape expected_out_shape;
    auto pooling =
        xla::MaxPool(ctx->Input(0), ksize_, stride_, xla_padding,
                     XlaTensorFormat(data_format_, tensor_in_shape.dims() - 2));
    auto status_or_shape = pooling.builder()->GetShape(pooling);
    OP_REQUIRES_OK(ctx, status_or_shape.status());
    OP_REQUIRES_OK(ctx, XLAShapeToTensorShape(status_or_shape.ValueOrDie(),
                                              &expected_out_shape));
    OP_REQUIRES(ctx, expected_out_shape == out_backprop_shape,
                errors::Unimplemented("The output dimensions do not match the "
                                      "other input values."));

    xla::PrimitiveType element_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(input_type(2), &element_type));
    xla::XlaOp init_value = XlaHelpers::Zero(ctx->builder(), input_type(2));
    auto select = CreateScalarGeComputation(element_type, ctx->builder());
    auto scatter = CreateScalarAddComputation(element_type, ctx->builder());
    xla::XlaOp gradients =
        xla::SelectAndScatter(input, select, ksize_, stride_, xla_padding,
                              out_backprop, init_value, scatter);

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
REGISTER_XLA_OP(Name("MaxPoolGradV2")
                    .CompileTimeConstantInput("ksize")
                    .CompileTimeConstantInput("strides"),
                MaxPool2DGradOp);

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

    string data_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
    OP_REQUIRES(ctx, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
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

    auto out_backprop = ctx->Input(1);
    std::vector<int64> stride_int64s(stride_.begin(), stride_.end());
    xla::Padding xla_padding =
        (padding_ == VALID) ? xla::Padding::kValid : xla::Padding::kSame;
    xla::PrimitiveType xla_reduction_type;
    auto reduction_type = XlaHelpers::SumAccumulationType(ctx->input_type(1));
    OP_REQUIRES_OK(
        ctx, DataTypeToPrimitiveType(reduction_type, &xla_reduction_type));
    auto converted_out_backprop =
        xla::ConvertElementType(out_backprop, xla_reduction_type);
    auto xla_data_format =
        XlaTensorFormat(data_format_, gradients_shape.dims() - 2);
    auto padding_values =
        MakeSpatialPadding(gradients_shape.dim_sizes(), ksize_, stride_int64s,
                           xla_padding, xla_data_format);
    auto in_backprop =
        xla::AvgPoolGrad(converted_out_backprop, gradients_shape.dim_sizes(),
                         ksize_, stride_int64s, padding_values, xla_data_format,
                         /*counts_include_padding=*/padding_ == VALID);
    // Convert the pooling result back to the input type before returning it.
    xla::PrimitiveType xla_out_backprop_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(ctx->input_type(1),
                                                &xla_out_backprop_type));
    ctx->SetOutput(0,
                   xla::ConvertElementType(in_backprop, xla_out_backprop_type));
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
      : AvgPoolGradOp(ctx, /*num_spatial_dims=*/2) {}
};
REGISTER_XLA_OP(
    Name("AvgPoolGrad").CompileTimeConstantInput("orig_input_shape"),
    AvgPool2DGradOp);

class AvgPool3DGradOp : public AvgPoolGradOp {
 public:
  explicit AvgPool3DGradOp(OpKernelConstruction* ctx)
      : AvgPoolGradOp(ctx, /*num_spatial_dims=*/3) {}
};
REGISTER_XLA_OP(
    Name("AvgPool3DGrad").CompileTimeConstantInput("orig_input_shape"),
    AvgPool3DGradOp);

class MaxPoolGradGradOp : public XlaOpKernel {
 public:
  MaxPoolGradGradOp(OpKernelConstruction* ctx, int num_spatial_dims)
      : XlaOpKernel(ctx), num_spatial_dims_(num_spatial_dims) {
    if (ctx->num_inputs() == 3) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("ksize", &ksize_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &stride_));
    }
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
  }

  int num_dims() const { return num_spatial_dims_ + 2; }

  void Compile(XlaOpKernelContext* ctx) override {
    if (ctx->num_inputs() != 3) {
      OP_REQUIRES(
          ctx, ctx->num_inputs() == 5,
          errors::InvalidArgument("Must supply ksize and stride arguments."));
      const TensorShape ksize_shape = ctx->InputShape(3);
      // Validate input sizes.
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(ksize_shape),
                  errors::InvalidArgument("ksize must be a vector, not shape ",
                                          ksize_shape.DebugString()));
      OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(3, &ksize_));

      const TensorShape stride_shape = ctx->InputShape(4);
      // Validate input sizes.
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(stride_shape),
                  errors::InvalidArgument("stride must be a vector, not shape ",
                                          stride_shape.DebugString()));
      OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(4, &stride_));
    }

    OP_REQUIRES(ctx, ksize_.size() == num_dims(),
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify ",
                                        num_dims(), " dimensions"));
    OP_REQUIRES(ctx, stride_.size() == num_dims(),
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify ",
                                        num_dims(), " dimensions"));

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

    // What we want to compute:
    // Given y = MaxPool(x), and xs_grad = MaxPoolGrad(x, y, ys_grad)
    // MaxPoolGradGrad computes {ys_grad}_grad given x, y, and {xs_grad}_grad.
    //
    // In the regular TF op, this amounts to selecting for each window the
    // incoming backprop value from xs_grad_grad that corresponds to the maximal
    // value in the corresponding window of x.
    //
    // TODO(b/73062247): What we really want is a ReduceWindow with different
    // arrays for index selection vs return value selection--a select-to-gather.
    //
    // Here, we implement a bitwise hack: we use the hi 16 bits of input for
    // separate max pooling alongside each of the hi and lo 16 bits of
    // out_backprop packed into 16 lo bits, which we then glue back together at
    // the end to get a full 32 bits of gradient.
    //
    // This could select the wrong backprop value for two x values that are
    // equally maximal up to the first 16 bits, in which case we are taking the
    // latter.
    //
    // Note that in principle we could use 32 separate maxpools to recover each
    // of 32 bits of the gradient while preserving 31 bits of input for the max
    // pooling criteria; here, we just truncate to the first 16 bits of input.

    auto input = ctx->Input(0);
    auto out_backprop = ctx->Input(2);

    auto b = ctx->builder();

    auto sixteen = xla::ConstantR0<uint32>(b, 16);
    // in (f32) -> round to 7 mantissa bits (bf16)-> 16-high-bit u32.
    //
    // NOTE: Use a ReducePrecision operation instead of a cast to BF16 and back
    // to F32 since the XLA compiler may ignore narrowing casts to floating
    // point types if the debug option xla_allow_excess_precision is set.
    auto in_hi = xla::BitcastConvertType(
        xla::ReducePrecision(input, /*exponent_bits=*/8, /*mantissa_bits=*/7),
        xla::U32);
    auto bp_int = xla::BitcastConvertType(out_backprop, xla::U32);
    auto bp_hi = xla::ShiftRightLogical(bp_int, sixteen);
    auto bp_lo =
        xla::ShiftRightLogical(xla::ShiftLeft(bp_int, sixteen), sixteen);
    auto in_hi_bp_hi = xla::Add(in_hi, bp_hi);  // Want an unsigned add.
    auto in_hi_bp_lo = xla::Add(in_hi, bp_lo);  // Want an unsigned add.

    auto init_value = xla::MinValue(b, xla::F32);
    // We will reduce by taking the maximal value up to 16 bits (ignoring the lo
    // 16 bits of packed-in hi/lo backprop value).
    auto rb = b->CreateSubBuilder("GreaterOrEqOf_ByFirst16Bits");
    {
      // F32 parameters to satisfy lowering type restriction for reduce opcode.
      const xla::Shape scalar = xla::ShapeUtil::MakeShape(xla::F32, {});
      auto lhs = xla::Parameter(rb.get(), 0, scalar, "lhs");
      auto rhs = xla::Parameter(rb.get(), 1, scalar, "rhs");
      auto sixteen = xla::ConstantR0<int32>(rb.get(), 16);
      auto lhs_criteria =
          xla::ShiftLeft(xla::ShiftRightLogical(
                             xla::BitcastConvertType(lhs, xla::S32), sixteen),
                         sixteen);
      auto rhs_criteria =
          xla::ShiftLeft(xla::ShiftRightLogical(
                             xla::BitcastConvertType(rhs, xla::S32), sixteen),
                         sixteen);
      // Must use a F32 comparison, because S32 would not work for negatives.
      xla::Select(xla::Ge(xla::BitcastConvertType(lhs_criteria, xla::F32),
                          xla::BitcastConvertType(rhs_criteria, xla::F32)),
                  lhs, rhs);
    }
    auto reduce = rb->BuildAndNoteError();
    xla::Padding xla_padding =
        (padding_ == VALID) ? xla::Padding::kValid : xla::Padding::kSame;
    auto pooled_hi =
        xla::ReduceWindow(xla::BitcastConvertType(in_hi_bp_hi, xla::F32),
                          init_value, reduce, ksize_, stride_, xla_padding);
    auto pooled_lo =
        xla::ReduceWindow(xla::BitcastConvertType(in_hi_bp_lo, xla::F32),
                          init_value, reduce, ksize_, stride_, xla_padding);
    auto grads_hi =
        xla::ShiftLeft(xla::BitcastConvertType(pooled_hi, xla::U32), sixteen);
    auto grads_lo = xla::ShiftRightLogical(
        xla::ShiftLeft(xla::BitcastConvertType(pooled_lo, xla::U32), sixteen),
        sixteen);
    auto grads = xla::Add(grads_hi, grads_lo);  // Want an unsigned add.

    xla::PrimitiveType element_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(input_type(2), &element_type));
    ctx->SetOutput(0, xla::BitcastConvertType(grads, element_type));
  }

 protected:
  const int num_spatial_dims_;
  std::vector<int64> ksize_;
  std::vector<int64> stride_;
  Padding padding_;
  TensorFormat data_format_ = FORMAT_NHWC;
};

class MaxPool2DGradGradOp : public MaxPoolGradGradOp {
 public:
  explicit MaxPool2DGradGradOp(OpKernelConstruction* ctx)
      : MaxPoolGradGradOp(ctx, /*num_spatial_dims=*/2) {
    string data_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
    OP_REQUIRES(ctx, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
  }
};
REGISTER_XLA_OP(Name("MaxPoolGradGrad").TypeConstraint("T", DT_FLOAT),
                MaxPool2DGradGradOp);
REGISTER_XLA_OP(Name("MaxPoolGradGradV2")
                    .TypeConstraint("T", DT_FLOAT)
                    .CompileTimeConstantInput("ksize")
                    .CompileTimeConstantInput("strides"),
                MaxPool2DGradGradOp);

class MaxPool3DGradGradOp : public MaxPoolGradGradOp {
 public:
  explicit MaxPool3DGradGradOp(OpKernelConstruction* ctx)
      : MaxPoolGradGradOp(ctx, /*num_spatial_dims=*/3) {
    string data_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
    OP_REQUIRES(ctx, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
  }
};
REGISTER_XLA_OP(Name("MaxPool3DGradGrad").TypeConstraint("T", DT_FLOAT),
                MaxPool3DGradGradOp);

}  // anonymous namespace
}  // namespace tensorflow
