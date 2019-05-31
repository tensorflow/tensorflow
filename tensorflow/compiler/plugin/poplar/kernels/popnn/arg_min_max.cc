/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/arg_min_max.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"

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

class ArgMaxMinOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit ArgMaxMinOp(OpKernelConstruction* ctx, bool is_min)
      : is_min_(is_min), XlaOpKernel(ctx), IpuOpKernel() {}

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape input_shape = ctx->InputShape(0);
    const TensorShape dimension_shape = ctx->InputShape(1);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(dimension_shape),
                errors::InvalidArgument(
                    "dim must be a scalar, but received tensor of shape: ",
                    dimension_shape.DebugString()));

    int64 dim;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(1, &dim));

    const int input_dims = input_shape.dims();
    const int axis = dim < 0 ? dim + input_dims : dim;

    OP_REQUIRES(ctx, axis >= 0 && axis < input_dims,
                errors::InvalidArgument("Expected dimension in the range [",
                                        -input_dims, ", ", input_dims,
                                        "), but got ", dim));
    const int64 axis_size = input_shape.dim_size(axis);
    OP_REQUIRES(
        ctx, axis_size > 0,
        errors::InvalidArgument("Reduction axis ", dim, " is empty in shape ",
                                input_shape.DebugString()));

    DataType index_type = output_type(0);
    xla::PrimitiveType index_xla_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(index_type, &index_xla_type));

    attribute_map_.AddAttribute("axis", axis);

    // The output will be all the dims other than "axis".
    input_shape.RemoveDim(dim);
    xla::Shape xla_shape;

    if (dim != 0) {
      xla_shape = TensorShapeToXLAShape(index_xla_type, input_shape);
    } else {
      xla_shape = xla::ShapeUtil::MakeShape(index_xla_type, {1});
    }

    xla::XlaOp input = ctx->Input(0);
    xla::XlaOp output;
    xla::XlaBuilder* b = ctx->builder();

    //
    xla::PrimitiveType input_primitive_type;
    OP_REQUIRES_OK(
        ctx, DataTypeToPrimitiveType(input_type(0), &input_primitive_type));

    if (input_type(0) == DataType::DT_HALF ||
        input_type(0) == DataType::DT_FLOAT) {
      // Call into either our ArgMin or ArgMax implementation depending on what
      // the user requested.
      if (is_min_) {
        output = xla::CustomCall(
            b,
            GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::ArgMin),
            {input}, xla_shape, attribute_map_.Serialise());
      } else {
        output = xla::CustomCall(
            b,
            GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::ArgMax),
            {input}, xla_shape, attribute_map_.Serialise());
      }

    } else {
      // Fallback on existing TF impl if requested type isn't supported by
      // poplar.
      // TODO: Might be a better way of constraining XLA fallback, check.
      if (is_min_) {
        output = xla::ArgMin(input, index_xla_type, axis);
      } else {
        output = xla::ArgMax(input, index_xla_type, axis);
      }
    }
    ctx->SetOutput(0, output);
  }

 private:
  bool is_min_;

  TF_DISALLOW_COPY_AND_ASSIGN(ArgMaxMinOp);
};

// Register the ArgMax operation.
class ArgMaxOp : public ArgMaxMinOp {
 public:
  explicit ArgMaxOp(OpKernelConstruction* ctx)
      : ArgMaxMinOp(ctx, /*is_min=*/false) {}
};

REGISTER_XLA_OP(Name("ArgMax")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("dimension"),
                ArgMaxOp);

// Register the ArgMin operation.

class ArgMinOp : public ArgMaxMinOp {
 public:
  explicit ArgMinOp(OpKernelConstruction* ctx)
      : ArgMaxMinOp(ctx, /*is_min=*/true) {}
};

REGISTER_XLA_OP(Name("ArgMin")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("dimension"),
                ArgMinOp);

}  // namespace tensorflow
