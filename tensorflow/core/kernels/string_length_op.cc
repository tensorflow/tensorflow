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
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace {

class StringLengthOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    auto src = input.flat<string>();
    auto dst = output->flat<int32>();

    for (int n = 0; n < src.size(); ++n) {
      dst(n) = src(n).size();
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("StringLength").Device(DEVICE_CPU),
                        StringLengthOp);

}  // namespace
}  // namespace tensorflow
