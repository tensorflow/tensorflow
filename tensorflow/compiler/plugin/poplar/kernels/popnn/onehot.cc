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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/onehot.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal_util.h"

using namespace xla::poplarplugin;

namespace tensorflow {

class OneHotOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit OneHotOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    // Sanity checking performed by the original XLA op in
    // tf2xla/kernels/one_hot_op.cc
    const TensorShape indices_shape = ctx->InputShape(0);
    const TensorShape depth_shape = ctx->InputShape(1);
    const TensorShape on_value_shape = ctx->InputShape(2);
    const TensorShape off_value_shape = ctx->InputShape(3);

    const int indices_dims = indices_shape.dims();
    const int output_dims = indices_dims + 1;

    // Preliminary validation of sizes.
    OP_REQUIRES(
        ctx, axis_ == -1 || (axis_ >= 0 && axis_ < output_dims),
        errors::InvalidArgument("Expected axis to be -1 or between [0, ",
                                output_dims, ").  But received: ", axis_));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(depth_shape),
                errors::InvalidArgument("depth must be a scalar, but got: ",
                                        depth_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(on_value_shape),
                errors::InvalidArgument("on_value must be a scalar, but got: ",
                                        on_value_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(off_value_shape),
                errors::InvalidArgument("off_value must be a scalar, but got: ",
                                        off_value_shape.DebugString()));

    const int axis = (axis_ == -1) ? indices_dims : axis_;

    // The one-hot dimension.
    int64 depth;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(1, &depth));
    OP_REQUIRES(
        ctx, depth >= 0,
        errors::InvalidArgument("depth must be non-negative, got: ", depth));
    attribute_map_.AddAttribute("depth", depth);
    attribute_map_.AddAttribute("axis", axis);

    // The tensor of indices to be set to "On".
    const xla::XlaOp& indices = ctx->Input(0);

    // On/Off parameters. These are the scalars which determine what value to
    // fill the onehot tensor with to signify "hot" and "not hot".
    const xla::XlaOp& on = ctx->Input(2);
    const xla::XlaOp& off = ctx->Input(3);

    const DataType dtype = output_type(0);

    xla::XlaBuilder* b = ctx->builder();

    // Copy constant shape so we can modify.
    xla::Shape xla_shape;

    // Add the depth dimension to the 'axis' location.
    TensorShape output_shape = indices_shape;
    output_shape.InsertDim(axis, depth);

    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, output_shape, &xla_shape));

    // Create the custom call to our one hot implementation instead of the usual
    // XLA node.
    xla::XlaOp output = xla::CustomCall(
        b, GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::OneHot),
        {indices, on, off}, xla_shape, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  int32 axis_;

  TF_DISALLOW_COPY_AND_ASSIGN(OneHotOp);
};

REGISTER_XLA_OP(
    Name("OneHot").Device(DEVICE_IPU_XLA_JIT).CompileTimeConstantInput("depth"),
    OneHotOp);

}  // namespace tensorflow
