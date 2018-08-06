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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

template <typename T>
class SparseSliceGradOp : public OpKernel {
 public:
  explicit SparseSliceGradOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    const Tensor *backprop_val_grad, *input_indices, *output_indices, *input_start;
    OP_REQUIRES_OK(ctx, ctx->input("backprop_val_grad", &backprop_val_grad));
    OP_REQUIRES_OK(ctx, ctx->input("input_indices", &input_indices));
    OP_REQUIRES_OK(ctx, ctx->input("input_start", &input_start));
    OP_REQUIRES_OK(ctx, ctx->input("output_indices", &output_indices));

    OP_REQUIRES(ctx,
                TensorShapeUtils::IsMatrix(input_indices->shape()) &&
                    TensorShapeUtils::IsMatrix(output_indices->shape()),
                errors::InvalidArgument(
                    "Input and output indices should be matrices "
                    "but received shapes: ",
                    input_indices->shape().DebugString(), " and ",
                    output_indices->shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(backprop_val_grad->shape()),
        errors::InvalidArgument(
            "Input backprop_val_grad should be a vector but received shape: ",
            backprop_val_grad->shape().DebugString()));
    OP_REQUIRES(
        ctx,
        input_indices->dim_size(1) == output_indices->dim_size(1),
        errors::InvalidArgument("The input and output should have the same "
                                "ndims: got: ", input_indices->dim_size(1), " and ",
                                output_indices->dim_size(1)));
    OP_REQUIRES(
        ctx, output_indices->dim_size(0) <= input_indices->dim_size(0),
        errors::InvalidArgument("# rows of output_indices should be not greater "
                                "than of input_indices, got ",
                                output_indices->dim_size(0), " and ",
                                input_indices->dim_size(0)));
    OP_REQUIRES(
        ctx, backprop_val_grad->NumElements() == output_indices->dim_size(0),
        errors::InvalidArgument("# elements of backprop_val_grad and # rows of "
                                "output_indices should match (#nnz of sum): got ",
                                backprop_val_grad->NumElements(), " and ",
                                output_indices->dim_size(0)));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_start->shape()),
                errors::InvalidArgument(
                    "The input_start should be a vector but received shape ",
                    input_start->shape().DebugString()));

    const int num_dims = input_indices->dim_size(1);
    OP_REQUIRES(ctx, num_dims == input_start->NumElements(),
                errors::InvalidArgument(
                    "Expected input_start to be a vector of length ", num_dims,
                    " but got length ", input_start->NumElements()));

    const int64 input_nnz = input_indices->dim_size(0);

    Tensor *val_grad;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({input_nnz}), &val_grad));

    T *val_grad_flat = val_grad->flat<T>().data();
    const T *backprop_val_grad_flat = backprop_val_grad->flat<T>().data();
    memset(val_grad_flat, 0, sizeof(T) * input_nnz);

    // Fill gradients for position where indices of input and output are same.
    const auto input_indices_mat = input_indices->matrix<int64>();
    const auto output_indices_mat = output_indices->matrix<int64>();
    const auto input_start_flat = input_start->flat<int64>();
    int64 j = 0;
    for (int64 i = 0; i < input_nnz && j < backprop_val_grad->NumElements();
         ++i) {
      bool is_same = true;
      for (int d = 0; d < num_dims; ++d) {
        const int64 a = input_indices_mat(i, d);
        const int64 b = output_indices_mat(j, d);
        const int64 offset = input_start_flat(d);
        if (a != b + offset) {
          is_same = false;
          break;
        }
      }
      if (is_same) {
        val_grad_flat[i] = backprop_val_grad_flat[j];
        ++j;
      }
    }
    OP_REQUIRES(
        ctx, backprop_val_grad->NumElements() == j,
        errors::Internal("Elements of backprop_val_grad aren't all propagated. "
                         "Num elements:", backprop_val_grad->NumElements(),
                         ", used: ", j));
  }
};

#define REGISTER_KERNELS(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("SparseSliceGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseSliceGradOp<type>)

TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS
}  // namespace tensorflow
