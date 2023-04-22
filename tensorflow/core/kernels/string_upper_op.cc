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

// See docs in ../ops/string_ops.cc.

#include <string>

#include "absl/strings/ascii.h"
#include "unicode/unistr.h"  // from @icu
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

class StringUpperOp : public OpKernel {
 public:
  explicit StringUpperOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("encoding", &encoding_));
    OP_REQUIRES(context, encoding_.empty() || encoding_ == "utf-8",
                errors::InvalidArgument(
                    "only utf-8 or '' (no encoding) is supported, received ",
                    encoding_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    Tensor* output_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, input_tensor->shape(), &output_tensor));

    const auto input = input_tensor->flat<tstring>();
    auto output = output_tensor->flat<tstring>();
    if (encoding_.empty()) {
      for (int64 i = 0; i < input.size(); ++i) {
        StringPiece entry(input(i));
        output(i) = absl::AsciiStrToUpper(entry);
      }
    } else {
      // The validation of utf-8 has already been done in GetAttr above.
      for (int64 i = 0; i < input.size(); ++i) {
        icu::UnicodeString us(input(i).c_str(), "UTF-8");
        us.toUpper();
        us.toUTF8String(output(i));
      }
    }
  }

 private:
  string encoding_;
};

REGISTER_KERNEL_BUILDER(Name("StringUpper").Device(DEVICE_CPU), StringUpperOp);

}  // namespace tensorflow
