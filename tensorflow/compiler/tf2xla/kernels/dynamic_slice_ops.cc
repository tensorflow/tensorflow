/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {
namespace {

class DynamicUpdateSliceOp : public XlaOpKernel {
 public:
  explicit DynamicUpdateSliceOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* ctx) override {
    VLOG(3) << "DynamicUpdateSliceOp::Compile";

    DataType index_type = input_type(2);
    OP_REQUIRES(ctx, index_type == DT_INT32 || index_type == DT_INT64,
                errors::InvalidArgument("index must be int32 or int64"));

    const TensorShape input_shape = ctx->InputShape(0);
    const TensorShape update_shape = ctx->InputShape(1);
    const TensorShape index_shape = ctx->InputShape(2);

    OP_REQUIRES(
        ctx,
        TensorShapeUtils::IsVector(index_shape) &&
            index_shape.num_elements() == input_shape.dims(),
        errors::InvalidArgument("index must be a vector with length equal to "
                                "the number of input dimensions"));
    OP_REQUIRES(
        ctx, input_shape.dims() == update_shape.dims(),
        errors::InvalidArgument("input and update must have the same rank,"
                                " input shape is ",
                                input_shape.DebugString(), "; update shape is ",
                                update_shape.DebugString()));

    xla::XlaOp result =
        xla::DynamicUpdateSlice(ctx->Input(0), ctx->Input(1), ctx->Input(2));
    ctx->SetOutput(0, result);
  }
};

REGISTER_XLA_OP(Name("XlaDynamicUpdateSlice"), DynamicUpdateSliceOp);

}  // namespace
}  // namespace tensorflow
