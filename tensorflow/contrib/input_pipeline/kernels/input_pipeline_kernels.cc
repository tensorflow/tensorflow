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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

// This Op takes in a list of strings and a counter (ref). It increments the
// counter by 1 and returns the element at that position in the list (circling
// around if need to).
class ObtainNextOp : public OpKernel {
 public:
  explicit ObtainNextOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* list;
    OP_REQUIRES_OK(ctx, ctx->input("list", &list));
    int64 num_elements = list->NumElements();
    auto list_flat = list->flat<string>();

    // Allocate output.
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("out_element", TensorShape({}),
                                             &output_tensor));

    // Obtain mutex for the "counter" tensor.
    mutex* mu;
    OP_REQUIRES_OK(ctx, ctx->input_ref_mutex("counter", &mu));
    mutex_lock l(*mu);
    // Increment "counter" tensor by 1.
    Tensor counter_tensor;
    OP_REQUIRES_OK(ctx, ctx->mutable_input("counter", &counter_tensor, true));
    int64* pos = &counter_tensor.scalar<int64>()();
    *pos = (*pos + 1) % num_elements;

    // Assign value to output.
    output_tensor->scalar<string>()() = list_flat(*pos);
  }
};

REGISTER_KERNEL_BUILDER(Name("ObtainNext").Device(DEVICE_CPU), ObtainNextOp);

}  // namespace tensorflow
