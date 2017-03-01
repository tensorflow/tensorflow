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

// XLA implementation of OneHot operator.

#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

namespace tensorflow {
namespace {

template <typename T>
Tensor MakeLinspaceTensor(const TensorShape& shape, int64 depth) {
  Tensor linspace(DataTypeToEnum<T>::v(), shape);
  auto linspace_flat = linspace.flat<T>();
  for (int64 i = 0; i < depth; ++i) {
    linspace_flat(i) = i;
  }
  return linspace;
}

class OneHotOp : public XlaOpKernel {
 public:
  explicit OneHotOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
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

    TensorShape output_shape = indices_shape;
    output_shape.InsertDim(axis, depth);

    xla::ComputationDataHandle on_value = ctx->Input(2);
    xla::ComputationDataHandle off_value = ctx->Input(3);

    // Build a Tensor populated with values 0, 1, 2, ... depth.
    std::vector<int64> linspace_dims(output_dims, 1);
    linspace_dims[axis] = depth;
    TensorShape linspace_shape(linspace_dims);
    Tensor linspace;
    switch (ctx->input_type(0)) {
      case DT_UINT8:
        linspace = MakeLinspaceTensor<uint8>(linspace_shape, depth);
        break;
      case DT_INT32:
        linspace = MakeLinspaceTensor<int32>(linspace_shape, depth);
        break;
      case DT_INT64:
        linspace = MakeLinspaceTensor<int64>(linspace_shape, depth);
        break;
      default:
        ctx->SetStatus(errors::InvalidArgument(
            "Invalid argument type ", DataTypeString(ctx->input_type(0))));
        return;
    }
    xla::Literal linspace_literal;
    OP_REQUIRES_OK(ctx, HostTensorToLiteral(linspace, &linspace_literal));

    xla::ComputationBuilder* builder = ctx->builder();
    xla::ComputationDataHandle indices = ctx->Input(0);

    // Broadcast the linspace constant across the indices along the new axis,
    // and test equality at each position.
    std::vector<int64> broadcast_dims(indices_shape.dims());
    std::iota(broadcast_dims.begin(), broadcast_dims.begin() + axis, 0);
    std::iota(broadcast_dims.begin() + axis, broadcast_dims.end(), axis + 1);
    xla::ComputationDataHandle one_hot =
        builder->Eq(indices, builder->ConstantLiteral(linspace_literal),
                   broadcast_dims);

    // Selects the user-provided off_value and on_value values.
    ctx->SetOutput(
        0, builder->Select(
               one_hot, builder->Broadcast(on_value, output_shape.dim_sizes()),
               builder->Broadcast(off_value, output_shape.dim_sizes())));
  }

 private:
  int32 axis_;

  TF_DISALLOW_COPY_AND_ASSIGN(OneHotOp);
};

REGISTER_XLA_OP("OneHot", OneHotOp);

}  // namespace
}  // namespace tensorflow
