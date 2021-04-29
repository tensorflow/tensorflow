/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <utility>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/reshape_util.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

class SparseReshapeOp : public OpKernel {
 public:
  explicit SparseReshapeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_indices_in = context->input(0);
    const Tensor& input_shape_in = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_indices_in.shape()),
                errors::InvalidArgument("Input must be a matrix."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape_in.shape()),
                errors::InvalidArgument("Input shape must be a vector."));
    OP_REQUIRES(context,
                input_indices_in.dim_size(1) == input_shape_in.dim_size(0),
                errors::InvalidArgument(
                    "Input tensor rank must match input shape length."));
    ReshapeSparseTensor(context, context->input(0), context->input(1),
                        context->input(2), 0 /* output indices index */,
                        1 /* output shape index */);
  }
};

REGISTER_KERNEL_BUILDER(Name("SparseReshape").Device(DEVICE_CPU),
                        SparseReshapeOp)
}  // namespace tensorflow
