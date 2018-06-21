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

#include <string>

#include "re2/re2.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class RegexFullMatchOp : public OpKernel {
 public:
  explicit RegexFullMatchOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    const auto& input_flat = input_tensor->flat<string>();

    const Tensor* pattern_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("pattern", &pattern_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(pattern_tensor->shape()),
                errors::InvalidArgument("Pattern must be scalar, but received ",
                                        pattern_tensor->shape().DebugString()));
    const string pattern = pattern_tensor->flat<string>()(0);
    const RE2 match(pattern);
    OP_REQUIRES(ctx, match.ok(),
                errors::InvalidArgument("Invalid pattern: ", pattern,
                                        ", error: ", match.error()));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("output", input_tensor->shape(),
                                             &output_tensor));
    auto output_flat = output_tensor->flat<bool>();
    for (size_t i = 0; i < input_flat.size(); ++i) {
      output_flat(i) = RE2::FullMatch(input_flat(i), match);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("RegexFullMatch").Device(DEVICE_CPU),
                        RegexFullMatchOp);

}  // namespace tensorflow
