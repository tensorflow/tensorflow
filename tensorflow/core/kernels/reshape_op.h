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

#ifndef TENSORFLOW_KERNELS_RESHAPE_OP_H_
#define TENSORFLOW_KERNELS_RESHAPE_OP_H_

#include <memory>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// Note that this op is subclassed for QuantizedReshapeOp.
class ReshapeOp : public OpKernel {
 public:
  explicit ReshapeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& sizes = context->input(1);
    // Preliminary validation of sizes.
    OP_REQUIRES(context, IsLegacyVector(sizes.shape()),
                errors::InvalidArgument("sizes input must be 1-D, not shape ",
                                        sizes.shape().DebugString()));
    const int64 num_dims = sizes.NumElements();

    // Compute the output shape.  Determine product of specified
    // dimensions, and find the index of the unspecified one.
    TensorShape shape;
    int64 product = 1;
    int unknown_index = -1;
    auto Svec = sizes.flat<int32>();
    for (int d = 0; d < num_dims; ++d) {
      const int32 size = Svec(d);
      if (size == -1) {
        OP_REQUIRES(
            context, unknown_index == -1,
            errors::InvalidArgument("only one input size may be -1, not both ",
                                    unknown_index, " and ", d));
        unknown_index = d;
        shape.AddDim(1);
      } else {
        OP_REQUIRES(context, size >= 0,
                    errors::InvalidArgument(
                        "size ", d, " must be non-negative, not ", size));
        shape.AddDim(size);
        product *= size;
      }
    }
    if (unknown_index != -1) {
      OP_REQUIRES(
          context, product > 0,
          errors::InvalidArgument("Reshape cannot infer the missing input size "
                                  "for an empty tensor unless all specified "
                                  "input sizes are non-zero"));
      const int64 missing = input.NumElements() / product;
      OP_REQUIRES(
          context, product * missing == input.NumElements(),
          errors::InvalidArgument(
              "Input to reshape is a tensor with ", input.NumElements(),
              " values, but the requested shape requires a multiple of ",
              product));
      shape.set_dim(unknown_index, missing);
    }
    OP_REQUIRES(context, shape.num_elements() == input.NumElements(),
                errors::InvalidArgument("Input to reshape is a tensor with ",
                                        input.NumElements(),
                                        " values, but the requested shape has ",
                                        shape.num_elements()));

    // Actually produce the reshaped output.
    Tensor output(input.dtype());
    CHECK(output.CopyFrom(input, shape));
    context->set_output(0, output);
  }

  bool IsExpensive() override { return false; }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_RESHAPE_OP_H_
