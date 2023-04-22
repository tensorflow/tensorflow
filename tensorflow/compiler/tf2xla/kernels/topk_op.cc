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

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/sorting.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/types.h"

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
    if (last_dim_size < k) {
      k = last_dim_size;
    }
    xla::XlaOp output_tuple = TopK(context->Input(0), k);
    context->SetOutput(0, xla::GetTupleElement(output_tuple, 0));
    context->SetOutput(1, xla::GetTupleElement(output_tuple, 1));
  }

 private:
  bool sorted_;
};

REGISTER_XLA_OP(Name("TopKV2").CompileTimeConstantInput("k").TypeConstraint(
                    "T", {DT_UINT32, DT_INT32, DT_UINT64, DT_INT64, DT_FLOAT,
                          DT_HALF, DT_DOUBLE, DT_BFLOAT16}),
                TopKOp);

}  // namespace
}  // namespace tensorflow
