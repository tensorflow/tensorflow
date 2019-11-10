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

// XLA-specific Ops for 2D convolution.

#include "tensorflow/compiler/tf2xla/kernels/conv_op_helpers.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace {

class ConvOp : public XlaOpKernel {
 public:
  explicit ConvOp(OpKernelConstruction* ctx, int num_spatial_dims,
                  bool depthwise)
      : XlaOpKernel(ctx) {
    xla::StatusOr<ConvOpAttrs> attrs =
        ConvOpAttrs::Create(num_spatial_dims, depthwise, ctx);
    OP_REQUIRES_OK(ctx, attrs.status());
    attrs_ = attrs.ValueOrDie();
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::StatusOr<xla::XlaOp> conv = MakeXlaForwardConvOp(
        ctx->op_kernel().type_string(), ctx->Input(0), ctx->Input(1), attrs_);
    OP_REQUIRES_OK(ctx, conv.status());
    ctx->SetOutput(0, conv.ValueOrDie());
  }

 protected:
  ConvOpAttrs attrs_;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ConvOp);
};

class Conv2DOp : public ConvOp {
 public:
  explicit Conv2DOp(OpKernelConstruction* ctx)
      : ConvOp(ctx, /*num_spatial_dims=*/2, /*depthwise=*/false) {}
};
REGISTER_XLA_OP(Name("Conv2D"), Conv2DOp);

class Conv3DOp : public ConvOp {
 public:
  explicit Conv3DOp(OpKernelConstruction* ctx)
      : ConvOp(ctx, /*num_spatial_dims=*/3, /*depthwise=*/false) {}
};
REGISTER_XLA_OP(Name("Conv3D"), Conv3DOp);

class DepthwiseConv2DOp : public ConvOp {
 public:
  explicit DepthwiseConv2DOp(OpKernelConstruction* ctx)
      : ConvOp(ctx, /*num_spatial_dims=*/2, /*depthwise=*/true) {}
};
REGISTER_XLA_OP(Name("DepthwiseConv2dNative"), DepthwiseConv2DOp);

// Backprop for input.
class ConvBackpropInputOp : public XlaOpKernel {
 public:
  explicit ConvBackpropInputOp(OpKernelConstruction* ctx, int num_spatial_dims,
                               bool depthwise)
      : XlaOpKernel(ctx) {
    xla::StatusOr<ConvOpAttrs> attrs =
        ConvOpAttrs::Create(num_spatial_dims, depthwise, ctx);
    OP_REQUIRES_OK(ctx, attrs.status());
    attrs_ = attrs.ValueOrDie();
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape input_tensor_shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &input_tensor_shape));
    xla::Shape input_shape =
        TensorShapeToXLAShape(ctx->input_xla_type(1), input_tensor_shape);

    xla::StatusOr<xla::XlaOp> in_backprop =
        MakeXlaBackpropInputConvOp(ctx->op_kernel().type_string(), input_shape,
                                   ctx->Input(1), ctx->Input(2), attrs_);
    OP_REQUIRES_OK(ctx, in_backprop.status());
    ctx->SetOutput(0, in_backprop.ValueOrDie());
  }

 protected:
  ConvOpAttrs attrs_;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ConvBackpropInputOp);
};

class Conv2DBackpropInputOp : public ConvBackpropInputOp {
 public:
  explicit Conv2DBackpropInputOp(OpKernelConstruction* ctx)
      : ConvBackpropInputOp(ctx, /*num_spatial_dims=*/2, /*depthwise=*/false) {}
};
REGISTER_XLA_OP(
    Name("Conv2DBackpropInput").CompileTimeConstantInput("input_sizes"),
    Conv2DBackpropInputOp);

class Conv3DBackpropInputOp : public ConvBackpropInputOp {
 public:
  explicit Conv3DBackpropInputOp(OpKernelConstruction* ctx)
      : ConvBackpropInputOp(ctx, /*num_spatial_dims=*/3, /*depthwise=*/false) {}
};
REGISTER_XLA_OP(
    Name("Conv3DBackpropInputV2").CompileTimeConstantInput("input_sizes"),
    Conv3DBackpropInputOp);

class DepthwiseConv2DBackpropInputOp : public ConvBackpropInputOp {
 public:
  explicit DepthwiseConv2DBackpropInputOp(OpKernelConstruction* ctx)
      : ConvBackpropInputOp(ctx, /*num_spatial_dims=*/2, /*depthwise=*/true) {}
};
REGISTER_XLA_OP(Name("DepthwiseConv2dNativeBackpropInput")
                    .CompileTimeConstantInput("input_sizes"),
                DepthwiseConv2DBackpropInputOp);

class ConvBackpropFilterOp : public XlaOpKernel {
 public:
  explicit ConvBackpropFilterOp(OpKernelConstruction* ctx, int num_spatial_dims,
                                bool depthwise)
      : XlaOpKernel(ctx) {
    xla::StatusOr<ConvOpAttrs> attrs =
        ConvOpAttrs::Create(num_spatial_dims, depthwise, ctx);
    OP_REQUIRES_OK(ctx, attrs.status());
    attrs_ = attrs.ValueOrDie();
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape filter_tensor_shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(1, &filter_tensor_shape));
    xla::Shape filter_shape =
        TensorShapeToXLAShape(ctx->input_xla_type(0), filter_tensor_shape);

    xla::StatusOr<xla::XlaOp> filter_backprop = MakeXlaBackpropFilterConvOp(
        ctx->op_kernel().type_string(), ctx->Input(0), filter_shape,
        ctx->Input(2), attrs_);
    OP_REQUIRES_OK(ctx, filter_backprop.status());
    ctx->SetOutput(0, filter_backprop.ValueOrDie());
  }

 protected:
  ConvOpAttrs attrs_;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ConvBackpropFilterOp);
};

class Conv2DBackpropFilterOp : public ConvBackpropFilterOp {
 public:
  explicit Conv2DBackpropFilterOp(OpKernelConstruction* ctx)
      : ConvBackpropFilterOp(ctx, /*num_spatial_dims=*/2, /*depthwise=*/false) {
  }
};
REGISTER_XLA_OP(
    Name("Conv2DBackpropFilter").CompileTimeConstantInput("filter_sizes"),
    Conv2DBackpropFilterOp);

class Conv3DBackpropFilterOp : public ConvBackpropFilterOp {
 public:
  explicit Conv3DBackpropFilterOp(OpKernelConstruction* ctx)
      : ConvBackpropFilterOp(ctx, /*num_spatial_dims=*/3, /*depthwise=*/false) {
  }
};
REGISTER_XLA_OP(
    Name("Conv3DBackpropFilterV2").CompileTimeConstantInput("filter_sizes"),
    Conv3DBackpropFilterOp);

class DepthwiseConv2DBackpropFilterOp : public ConvBackpropFilterOp {
 public:
  explicit DepthwiseConv2DBackpropFilterOp(OpKernelConstruction* ctx)
      : ConvBackpropFilterOp(ctx, /*num_spatial_dims=*/2, /*depthwise=*/true) {}
};
REGISTER_XLA_OP(Name("DepthwiseConv2dNativeBackpropFilter")
                    .CompileTimeConstantInput("filter_sizes"),
                DepthwiseConv2DBackpropFilterOp);

}  // namespace
}  // namespace tensorflow
