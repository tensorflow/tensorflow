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

#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/reshape_op.h"

namespace tensorflow {

class QuantizedReshapeOp : public ReshapeOp {
 public:
  explicit QuantizedReshapeOp(OpKernelConstruction* c) : ReshapeOp(c) {}

  void Compute(OpKernelContext* ctx) override {
    // This call processes inputs 1 and 2 to write output 0.
    ReshapeOp::Compute(ctx);

    const float input_min_float = ctx->input(2).flat<float>()(0);
    const float input_max_float = ctx->input(3).flat<float>()(0);
    Tensor* output_min = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({}), &output_min));
    output_min->flat<float>()(0) = input_min_float;

    Tensor* output_max = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({}), &output_max));
    output_max->flat<float>()(0) = input_max_float;
  }
};

#define REGISTER_CPU_KERNEL(type)                         \
  REGISTER_KERNEL_BUILDER(Name("QuantizedReshape")        \
                              .Device(DEVICE_CPU)         \
                              .HostMemory("shape")        \
                              .TypeConstraint<type>("T"), \
                          QuantizedReshapeOp)

REGISTER_CPU_KERNEL(::tensorflow::quint8);
REGISTER_CPU_KERNEL(::tensorflow::qint32);

#undef REGISTER_CPU_KERNEL

}  // namespace tensorflow
