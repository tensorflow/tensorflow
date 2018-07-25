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
#include "tensorflow/compiler/xla/literal.h"
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
    int last_dim = input_shape.dims() - 1;
    int last_dim_size = input_shape.dim_size(last_dim);
    OP_REQUIRES(
        context, last_dim_size >= k,
        errors::InvalidArgument("input must have at least k columns. Had ",
                                last_dim_size, ", needed ", k));

    xla::XlaBuilder* const b = context->builder();
    if (last_dim_size < k) {
      k = last_dim_size;
    }
    const xla::XlaOp input = context->Input(0);

    xla::XlaOp iota_s32 = xla::Iota(b, xla::S32, last_dim_size);
    auto input_dims = input_shape.dim_sizes();
    std::vector<int64> broadcast_dims(input_dims.begin(), input_dims.end() - 1);
    xla::XlaOp broadcast_s32 = xla::Broadcast(iota_s32, broadcast_dims);
    xla::XlaOp sort_result = xla::Sort(xla::Neg(input), broadcast_s32);

    std::vector<int64> start_indices(input_shape.dims(), 0);
    std::vector<int64> limit_indices(input_dims.begin(), input_dims.end());
    limit_indices[last_dim] = k;
    std::vector<int64> strides(input_shape.dims(), 1);

    xla::XlaOp values =
        xla::Neg(xla::Slice(xla::GetTupleElement(sort_result, 0), start_indices,
                            limit_indices, strides));
    xla::XlaOp indices = xla::Slice(xla::GetTupleElement(sort_result, 1),
                                    start_indices, limit_indices, strides);
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
