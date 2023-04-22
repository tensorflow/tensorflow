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

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

class StringStripOp : public OpKernel {
 public:
  explicit StringStripOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    Tensor* output_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, input_tensor->shape(), &output_tensor));

    const auto input = input_tensor->flat<tstring>();
    auto output = output_tensor->flat<tstring>();

    for (int64 i = 0; i < input.size(); ++i) {
      StringPiece entry(input(i));
      str_util::RemoveWhitespaceContext(&entry);
      output(i) = string(entry);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("StringStrip").Device(DEVICE_CPU), StringStripOp);

}  // namespace tensorflow
