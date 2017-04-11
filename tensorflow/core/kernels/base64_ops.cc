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

// See docs in ../ops/string_ops.cc.

#include <string>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/strings/base64.h"

namespace tensorflow {
namespace {

class EncodeBase64Op : public OpKernel {
 public:
  explicit EncodeBase64Op(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("pad", &pad_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    auto input = input_tensor.flat<string>();
    auto output = output_tensor->flat<string>();

    for (int64 i = 0; i < input.dimension(0); ++i) {
      OP_REQUIRES_OK(context, Base64Encode(input(i), pad_, &output(i)));
    }
  }

 private:
  bool pad_;
};

REGISTER_KERNEL_BUILDER(Name("EncodeBase64").Device(DEVICE_CPU),
                        EncodeBase64Op);

class DecodeBase64Op : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    auto input = input_tensor.flat<string>();
    auto output = output_tensor->flat<string>();

    for (int64 i = 0; i < input.dimension(0); ++i) {
      OP_REQUIRES_OK(context, Base64Decode(input(i), &output(i)));
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("DecodeBase64").Device(DEVICE_CPU),
                        DecodeBase64Op);

}  // namespace
}  // namespace tensorflow
