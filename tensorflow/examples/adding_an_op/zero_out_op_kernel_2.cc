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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("ZeroOut")
    .Attr("T: realnumbertype")
    .Input("to_zero: T")
    .Output("zeroed: T")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Zeros out all but the first value of a Tensor.

zeroed: A Tensor whose first value is identical to `to_zero`, and 0
  otherwise.
)doc");

REGISTER_OP("ZeroOut2")
    .Attr("T: realnumbertype")
    .Input("to_zero: T")
    .Output("zeroed: T")
    .Doc(R"doc(
Zeros out all but the first value of a Tensor.

zeroed: A Tensor whose first value is identical to `to_zero`, and 0
  otherwise.
)doc");

REGISTER_OP("ZeroOut3")
    .Attr("T: realnumbertype")
    .Input("to_zero: T")
    .Output("zeroed: T")
    .Doc(R"doc(
Zeros out all but the first value of a Tensor.

zeroed: A Tensor whose first value is identical to `to_zero`, and 0
  otherwise.
)doc");

template <typename T>
class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();

    // Create an output tensor
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_tensor.shape(), &output));
    auto output_flat = output->template flat<T>();

    // Set all the elements of the output tensor to 0
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = T(0);
    }

    // Preserve the first input value
    if (N > 0) output_flat(0) = input(0);
  }
};

REGISTER_KERNEL_BUILDER(Name("ZeroOut")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        ZeroOutOp<float>);
REGISTER_KERNEL_BUILDER(Name("ZeroOut")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("T"),
                        ZeroOutOp<double>);
REGISTER_KERNEL_BUILDER(Name("ZeroOut")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int>("T"),
                        ZeroOutOp<int>);

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("ZeroOut2").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ZeroOutOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
REGISTER_KERNEL(int32);

#undef REGISTER_KERNEL

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("ZeroOut3").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ZeroOutOp<type>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL
