/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/compiler/tf2xla/kernels/conv_op_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/math.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace {

using xla::ShapeUtil;
using xla::XlaOp;

// XLA lowering for _FusedConv2D op for int8 and QInt8
// Computes:
//
//   activation(
//       conv_scale * conv(conv_input, filter) +
//       side_input * side_input_scale + broadcast(bias)
//   )
//
// where
//
//  - conv_scale and side_input_scale are scalars.
//  - side-input is either the size of conv() or a zero-length tensor (in
//    which case it's ignored).
//  - bias is a 1D tensor with size matching output depth.
//  - activation is either the identity function or relu.
//
// conv_scale is called "conv_input_scale" in the TF op, and TF envisions the
// semantics as conv(conv_input_scale * conv_input).  Because conv is a linear
// operation, this is mathematically the same.  But in the int8 case it matters
// whether the multiply goes inside or outside the conv, because there are these
// extra type conversions.
//
// This class is only for int8 convolution, the types are:
//
//   convert<s8>(clamp_to_s8(activation<f32>(
//       conv_scale<f32> *
//         convert<f32>(conv<s32>(convert<s32>(conv_input<s8>),
//                                convert<s32>(filter<s8>)))
//       convert<f32>(side_input<s8>) * side_input_scale<f32> +
//       broadcast(bias<f32>)
//   )))
//
// See cudnn semantics at
//
// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward
// https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#scaling-parameters
//
// On GPU we lower these calls to a custom-call which translates directly to
// cudnnConvolutionBiasActivationForward().  On other platforms we expand these
// calls out into a series of pure XLA ops.
//
// (As an alternative implementation strategy, we could always lower to a series
// of pure XLA ops and then pattern-match to the cudnn op.  This is challenging
// and fragile because the pattern is rather complicated.)
class FusedConv2DInt8Op : public XlaOpKernel {
 public:
  enum class ActivationMode { kNone, kRelu };

  explicit FusedConv2DInt8Op(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx),
        is_gpu_(ctx->device_type().type_string() == "XLA_GPU_JIT") {
    OP_REQUIRES(
        ctx, ctx->num_inputs() == 6,
        errors::InvalidArgument("_FusedConv2D must have 6 inputs but has ",
                                ctx->num_inputs()));
    absl::StatusOr<ConvOpAttrs> conv_attrs =
        ConvOpAttrs::Create(/*num_spatial_dims=*/2, /*depthwise=*/false, ctx);
    OP_REQUIRES_OK(ctx, conv_attrs.status());
    conv_attrs_ = conv_attrs.value();

    std::string filter_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("filter_format", &filter_format));
    OP_REQUIRES(
        ctx, FilterFormatFromString(filter_format, &filter_format_),
        errors::InvalidArgument("Invalid filter format: ", filter_format));

    std::vector<std::string> fused_ops;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fused_ops", &fused_ops));
    OP_REQUIRES(ctx, !fused_ops.empty(),
                errors::InvalidArgument(
                    "FusedConv2DInt8Op must have at least one fused op."));
    std::string activation_mode = "None";
    if (fused_ops.size() > 1) {
      activation_mode = fused_ops[1];
    }
    OP_REQUIRES(ctx, activation_mode == "None" || activation_mode == "Relu",
                errors::InvalidArgument(
                    "Unknown activation_mode, must be 'None' or 'Relu': ",
                    activation_mode));
    activation_mode_ = activation_mode == "None" ? ActivationMode::kNone
                                                 : ActivationMode::kRelu;
  }

  absl::Status DoCompile(XlaOpKernelContext* ctx) {
    XlaOp conv_input = ctx->Input(0);
    XlaOp filter = ctx->Input(1);
    XlaOp bias = ctx->Input(2);
    XlaOp side_input = ctx->Input(3);
    XlaOp conv_scale = ctx->Input(4);
    XlaOp side_input_scale = ctx->Input(5);

    auto* builder = ctx->builder();
    TF_ASSIGN_OR_RETURN(auto conv_input_shape, builder->GetShape(conv_input));
    TF_ASSIGN_OR_RETURN(auto filter_shape, builder->GetShape(filter));
    TF_ASSIGN_OR_RETURN(auto side_input_shape, builder->GetShape(side_input));
    TF_ASSIGN_OR_RETURN(auto conv_scale_shape, builder->GetShape(conv_scale));
    TF_ASSIGN_OR_RETURN(auto side_input_scale_shape,
                        builder->GetShape(side_input_scale));

    if (conv_input_shape.element_type() != xla::S8) {
      return errors::InvalidArgument(
          "_FusedConv2D is implemented only for int8: but ",
          conv_input_shape.element_type(), " is passed");
    }

    if (!ShapeUtil::IsScalar(conv_scale_shape)) {
      return errors::InvalidArgument(
          "conv input scale must be a scalar, but was ",
          ShapeUtil::HumanString(conv_scale_shape));
    }
    if (!ShapeUtil::IsScalar(side_input_scale_shape)) {
      return errors::InvalidArgument(
          "side input scale must be a scalar, but was ",
          ShapeUtil::HumanString(side_input_scale_shape));
    }

    // Un-vectorize NCHW_VECT_C to NCHW.
    TensorFormat orig_data_format = conv_attrs_.data_format;
    int64 vect_width = -1;
    switch (conv_attrs_.data_format) {
      case FORMAT_NCHW_VECT_C:
        vect_width = conv_input_shape.dimensions(4);
        conv_input =
            xla::Collapse(xla::Transpose(conv_input, {0, 1, 4, 2, 3}), {1, 2});
        if (!ShapeUtil::IsZeroElementArray(side_input_shape)) {
          side_input = xla::Collapse(
              xla::Transpose(side_input, {0, 1, 4, 2, 3}), {1, 2});
        }
        break;
      case FORMAT_NHWC_VECT_W:
        return errors::Unimplemented("NHWC_VECT_W layout is unsupported.");
      default:
        break;
    }

    // Conv2D expects the filter to be in HWIO format.  If the filter is IOHW,
    // transpose it.  We expect XLA to make this reshape a nop anyway.
    switch (filter_format_) {
      case FORMAT_HWIO:
        break;
      case FORMAT_OHWI:
        filter = xla::Transpose(filter, {1, 2, 3, 0});
        break;
      case FORMAT_OIHW: {
        filter = xla::Transpose(filter, {2, 3, 1, 0});
        break;
      }
      case FORMAT_OIHW_VECT_I: {
        TF_ASSIGN_OR_RETURN(auto filter_shape, builder->GetShape(filter));
        // Shape should be of the form [O, I, H, W, {4 or 32}].  Transpose to
        // [H, W, I, {4 or 32}, O] and then collapse to [H, W, I, O].
        filter = xla::Collapse(xla::Transpose(filter, {2, 3, 1, 4, 0}), {2, 3});
        TF_ASSIGN_OR_RETURN(auto new_filter_shape, builder->GetShape(filter));
        break;
      }
    }

    // On XLA:GPU, spell the conv as using S32, which matches cudnn's
    // semantics.  On other platforms, spell it as an F32 conv, because S32
    // convs are very slow (CPU) or unsupported (TPU).
    auto conv_ty = is_gpu_ ? xla::S32 : xla::F32;
    conv_input = xla::ConvertElementType(conv_input, conv_ty);
    filter = xla::ConvertElementType(filter, conv_ty);

    auto conv_attrs = conv_attrs_;
    switch (conv_attrs_.data_format) {
      case FORMAT_NCHW_VECT_C:
        conv_attrs.data_format = FORMAT_NCHW;
        break;
      case FORMAT_NHWC_VECT_W:
        conv_attrs.data_format = FORMAT_NHWC;
        break;
      default:
        break;
    }
    TF_ASSIGN_OR_RETURN(
        XlaOp conv,
        MakeXlaForwardConvOp(type_string(), conv_input, filter, conv_attrs));

    conv = xla::ConvertElementType(conv, xla::F32);

    conv = conv * conv_scale;

    // Add bias.  For int8 convs, bias must be fp32.
    bias = xla::ConvertElementType(bias, xla::F32);
    XlaOp result = xla::Add(
        conv, bias, /*broadcast_dimensions=*/
        {GetTensorFeatureDimIndex(/*num_dims=*/4, conv_attrs.data_format)});

    // Add in the side input if it's present.  Do this before the NCHW ->
    // NCHW_VECT_C reshape because that's easier for XLA to pattern-match.
    //
    // Canonically, if side input is not present, then side_input_scale should
    // be 0.  But XLA doesn't have the capability of asserting this, and
    // asserting it outside of XLA would be expensive, requiring a host-device
    // sync.
    TF_ASSIGN_OR_RETURN(auto result_shape, builder->GetShape(result));
    if (!ShapeUtil::IsZeroElementArray(side_input_shape)) {
      // In the case of an int8 conv, side_input can be s8 or f32.  If it's s8,
      // just convert it.
      side_input = xla::ConvertElementType(side_input, xla::F32);
      TF_ASSIGN_OR_RETURN(side_input_shape, builder->GetShape(side_input));
      if (!ShapeUtil::Compatible(side_input_shape, result_shape)) {
        return errors::InvalidArgument(
            "Side-input shape ", ShapeUtil::HumanString(side_input_shape),
            " must be equal to convolution output shape ",
            ShapeUtil::HumanString(result_shape));
      }
      result = result + side_input * side_input_scale;
    }

    if (activation_mode_ == ActivationMode::kRelu) {
      result = xla::Max(result, xla::ZerosLike(result));
    }

    result = xla::Clamp(xla::ConstantR0(builder, -128.0f), result,
                        xla::ConstantR0(builder, 127.0f));

    // Hack: Omit RoundToEven in GPU mode to make the HLO easier to
    // pattern-match to a cudnn fused convolution.  Even without the
    // RoundToEven, this is by far the most complex and fragile pattern-match
    // we have in XLA.  Also, cudnn doesn't give us numerical guarantees
    // *anyway*.
    if (!is_gpu_) {
      result = xla::RoundToEven(result);
    }

    result = xla::ConvertElementType(result, xla::S8);

    // Un-convert NCHW -> NCHW_VECT_C.  Do this at the very end so that we can
    // pattern-match everything above into a cudnn fused conv.
    if (orig_data_format == FORMAT_NCHW_VECT_C) {
      int n = result_shape.dimensions(0);
      int c = result_shape.dimensions(1);
      int h = result_shape.dimensions(2);
      int w = result_shape.dimensions(3);
      CHECK_NE(vect_width, -1);     // Crash OK
      CHECK_EQ(c % vect_width, 0);  // Crash OK
      result = xla::Transpose(
          xla::Reshape(result, {n, c / vect_width, vect_width, h, w}),
          {0, 1, 3, 4, 2});
    }

    ctx->SetOutput(0, result);
    return absl::OkStatus();
  }

  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx, DoCompile(ctx));
  }

 private:
  bool is_gpu_;
  ConvOpAttrs conv_attrs_;
  ActivationMode activation_mode_;
  FilterTensorFormat filter_format_;
};

REGISTER_XLA_OP(Name("_FusedConv2D")
                    .CompileTimeConstantInput("host_args")
                    .TypeConstraint("T", {DT_INT8, DT_QINT8}),
                FusedConv2DInt8Op);

}  // anonymous namespace
}  // namespace tensorflow
