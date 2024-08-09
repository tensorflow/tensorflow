/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "tensorflow/compiler/tf2xla/lib/scatter.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/value_inference.h"
#include "xla/client/xla_builder.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace {

// Operator to convert sparse representations to dense.
class SparseToDenseOp : public XlaOpKernel {
 public:
  explicit SparseToDenseOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    // sparse_indices
    const TensorShape indices_shape = context->InputShape(0);
    OP_REQUIRES(context, indices_shape.dims() <= 2,
                errors::InvalidArgument(
                    "sparse_indices should be a scalar, vector, or matrix, "
                    "got shape ",
                    indices_shape.DebugString()));
    const int64_t num_elems =
        indices_shape.dims() > 0 ? indices_shape.dim_size(0) : 1;
    const int64_t num_dims =
        indices_shape.dims() > 1 ? indices_shape.dim_size(1) : 1;

    // output_shape
    TensorShape output_shape;
    OP_REQUIRES_OK(context,
                   context->ConstantInputAsShape(
                       1, &output_shape, xla::ValueInferenceMode::kUpperBound));
    OP_REQUIRES(context, output_shape.dims() == num_dims,
                errors::InvalidArgument(
                    "output_shape has incorrect number of elements: ",
                    output_shape.num_elements(), " should be: ", num_dims));

    // sparse_values
    const TensorShape sparse_values_shape = context->InputShape(2);
    const int64_t num_values = sparse_values_shape.num_elements();
    OP_REQUIRES(
        context,
        sparse_values_shape.dims() == 0 ||
            (sparse_values_shape.dims() == 1 && num_values == num_elems),
        errors::InvalidArgument("sparse_values has incorrect shape ",
                                sparse_values_shape.DebugString(),
                                ", should be [] or [", num_elems, "]"));

    // default_value
    const TensorShape default_value_shape = context->InputShape(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(default_value_shape),
                errors::InvalidArgument("default_value should be a scalar."));

    xla::XlaOp indices = context->Input(0);
    xla::XlaOp sparse_values = context->Input(2);
    xla::XlaOp default_value = context->Input(3);

    if (sparse_values_shape.dims() == 0 && num_elems != 1) {
      sparse_values = Broadcast(sparse_values, {num_elems});
    }
    xla::XlaBuilder* builder = context->builder();
    auto buffer = Broadcast(default_value, output_shape.dim_sizes());
    std::vector<bool> dynamic_dims;
    OP_REQUIRES_OK(
        context, context->ResolveInputDynamismIntoPredVector(1, &dynamic_dims));

    for (int64_t i = 0; i < dynamic_dims.size(); ++i) {
      // If a dimension is dynamic, call set-dimension-size on the output.
      if (dynamic_dims[i]) {
        auto dynamic_dim_size =
            xla::Slice(context->Input(1), {i}, {i + 1}, {1});
        dynamic_dim_size = xla::Reshape(dynamic_dim_size, {});
        dynamic_dim_size = xla::ConvertElementType(dynamic_dim_size, xla::S32);
        buffer = xla::SetDimensionSize(buffer, dynamic_dim_size, i);
      }
    }
    auto result = XlaScatter(buffer, sparse_values, indices,
                             /*indices_are_vectors=*/indices_shape.dims() > 1,
                             /*indices_are_sorted=*/false,
                             /*combiner=*/{}, builder);
    context->SetOutput(0, builder->ReportErrorOrReturn(result));
  }
};

REGISTER_XLA_OP(Name("SparseToDense").CompileTimeConstantInput("output_shape"),
                SparseToDenseOp);

}  // namespace

}  // namespace tensorflow
