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

#include "tensorflow/compiler/tf2xla/type_util.h"
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
    DataType index_type;
    OP_REQUIRES_OK(context, context->GetAttr("index_type", &index_type));
    OP_REQUIRES_OK(context, DataTypeToPrimitiveType(index_type, &index_type_));
  }

  void Compile(XlaOpKernelContext* context) override {
    const StatusOr<xla::Shape> input_shape_or = context->InputXlaShape(0);
    OP_REQUIRES_OK(context, input_shape_or.status());
    const xla::Shape& input_shape = *input_shape_or;
    int last_dim = input_shape.dimensions_size() - 1;
    int last_dim_size = input_shape.dimensions(last_dim);

    int64_t k;
    bool k_bound_inferrable =
        context
            ->ConstantInputAsIntScalar(1, &k,
                                       xla::ValueInferenceMode::kUpperBound)
            .ok();
    if (!k_bound_inferrable) {
      // - If we can infer the bound of K, use the bound.
      // - If not, use last dim's size.
      k = last_dim_size;
    }
    OP_REQUIRES(context, k >= 0,
                errors::InvalidArgument("Need k >= 0, got ", k));

    OP_REQUIRES(context, input_shape.dimensions_size() >= 1,
                errors::InvalidArgument("input must be >= 1-D, got shape ",
                                        input_shape.DebugString()));

    OP_REQUIRES(
        context, last_dim_size >= k,
        errors::InvalidArgument("input must have at least k columns. Had ",
                                last_dim_size, ", needed ", k));
    if (last_dim_size < k) {
      k = last_dim_size;
    }

    bool k_is_dynamic;
    OP_REQUIRES_OK(context,
                   context->ResolveInputDynamismIntoPred(1, &k_is_dynamic));
    xla::XlaOp output_tuple = TopK(context->Input(0), k, index_type_);
    auto values = xla::GetTupleElement(output_tuple, 0);
    auto indices = xla::GetTupleElement(output_tuple, 1);
    if (k_is_dynamic) {
      xla::XlaOp dynamic_k = context->Input(1);
      values = xla::SetDimensionSize(values, dynamic_k, last_dim);
      indices = xla::SetDimensionSize(indices, dynamic_k, last_dim);
    }
    context->SetOutput(0, values);
    context->SetOutput(1, indices);
  }

 private:
  bool sorted_;
  xla::PrimitiveType index_type_;
};

REGISTER_XLA_OP(Name("TopKV2")
                    .CompileTimeConstantInput("k")
                    .TypeConstraint("T",
                                    {DT_UINT32, DT_INT32, DT_UINT64, DT_INT64,
                                     DT_FLOAT, DT_HALF, DT_DOUBLE, DT_BFLOAT16,
                                     DT_UINT8, DT_INT8, DT_INT16})
                    .TypeConstraint("Tk", {DT_INT16, DT_INT32, DT_INT64})
                    .TypeConstraint("index_type",
                                    {DT_INT16, DT_INT32, DT_INT64}),
                TopKOp);

}  // namespace
}  // namespace tensorflow
