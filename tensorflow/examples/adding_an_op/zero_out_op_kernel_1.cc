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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    })
    .Doc(R"doc(
Zeros out all but the first value of a Tensor.

zeroed: A Tensor whose first value is identical to `to_zero`, and 0
  otherwise.
)doc");

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32_t>();

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<int32_t>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output(i) = 0;
    }

    // Preserve the first input value.
    if (N > 0) output(0) = input(0);
  }
};

REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);

REGISTER_OP("Namespace>ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    })
    .Doc(R"doc(
Zeros out all but the first value of a Tensor.

zeroed: A Tensor whose first value is identical to `to_zero`, and 0
  otherwise.
)doc");

REGISTER_KERNEL_BUILDER(Name("Namespace>ZeroOut").Device(DEVICE_CPU),
                        ZeroOutOp);

REGISTER_OP("Namespace>Nested>ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    })
    .Doc(R"doc(
Zeros out all but the first value of a Tensor.

zeroed: A Tensor whose first value is identical to `to_zero`, and 0
  otherwise.
)doc");

REGISTER_KERNEL_BUILDER(Name("Namespace>Nested>ZeroOut").Device(DEVICE_CPU),
                        ZeroOutOp);
