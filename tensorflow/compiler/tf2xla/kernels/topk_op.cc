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

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/numeric.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/no_op.h"

namespace tensorflow {
namespace {

class TopKOp : public XlaOpKernel {
 public:
  explicit TopKOp(OpKernelConstruction* context) : XlaOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("sorted", &sorted_));
  }

  void Compile(XlaOpKernelContext* context) override {
    int64 k;
    OP_REQUIRES_OK(context, context->ConstantInputAsIntScalar(1, &k));
    OP_REQUIRES(context, k >= 0,
                errors::InvalidArgument("Need k >= 0, got ", k));
    const TensorShape input_shape = context->InputShape(0);
    OP_REQUIRES(context, input_shape.dims() >= 1,
                errors::InvalidArgument("input must be >= 1-D, got shape ",
                                        input_shape.DebugString()));
    OP_REQUIRES(
        context, input_shape.dim_size(input_shape.dims() - 1) >= k,
        errors::InvalidArgument("input must have at least k columns. Had ",
                                input_shape.dim_size(input_shape.dims() - 1),
                                ", needed ", k));

    OP_REQUIRES(
        context, input_shape.dims() == 1,
        errors::Unimplemented("TopK is implemented for 1-D inputs, got shape ",
                              input_shape.DebugString()));

    xla::XlaBuilder* const b = context->builder();
    if (input_shape.dim_size(0) < k) {
      k = input_shape.dim_size(0);
    }
    const xla::XlaOp input = context->Input(0);
    xla::XlaOp iota_s32 = xla::Iota(b, xla::S32, input_shape.dim_size(0));
    xla::XlaOp sort_result = xla::Sort(xla::Neg(input), iota_s32);
    xla::XlaOp values =
        xla::Neg(xla::Slice(xla::GetTupleElement(sort_result, 0),
                            /*start_indices=*/{0},
                            /*limit_indices=*/{k},
                            /*strides=*/{1}));
    xla::XlaOp indices = xla::Slice(xla::GetTupleElement(sort_result, 1),
                                    /*start_indices=*/{0},
                                    /*limit_indices=*/{k},
                                    /*strides=*/{1});
    context->SetOutput(0, values);
    context->SetOutput(1, indices);
  }

 private:
  bool sorted_;
};

REGISTER_XLA_OP(Name("TopKV2").CompileTimeConstInput("k").TypeConstraint(
                    "T", {DT_UINT32, DT_INT32, DT_FLOAT, DT_BFLOAT16}),
                TopKOp);

}  // namespace
}  // namespace tensorflow
