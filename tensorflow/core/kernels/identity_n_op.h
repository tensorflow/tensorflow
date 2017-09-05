/* Copyright 2015-2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_KERNELS_IDENTITY_N_OP_H_
#define TENSORFLOW_KERNELS_IDENTITY_N_OP_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

class IdentityNOp : public OpKernel {
 public:
  explicit IdentityNOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    OpInputList input;
    OpOutputList output;
    OP_REQUIRES_OK(context, context->input_list("input", &input));
    OP_REQUIRES_OK(context, context->output_list("output", &output));
    OP_REQUIRES(context, input.size() == output.size(),
                errors::InvalidArgument("Input and output counts must match"));
    for (int i = 0; i < input.size(); ++i) {
      output.set(i, input[i]);
    }
  }

  bool IsExpensive() override { return false; }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_IDENTITY_N_OP_H_
