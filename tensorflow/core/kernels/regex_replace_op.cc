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
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace {

// Execute the specified regex using the given context.
// Context requirements:
//  - "input" string Tensor at input_index=0
//  - "output" string Tensor at output_index=0
Status InternalCompute(const RE2& match, const string& rewrite,
                       const bool replace_global, OpKernelContext* ctx) {
  const Tensor* input_tensor;
  TF_RETURN_IF_ERROR(ctx->input("input", &input_tensor));
  Tensor* output_tensor;
  std::unique_ptr<Tensor> maybe_forwarded =
      ctx->forward_input(0 /*input_index*/, 0 /*output_index*/,
                         tensorflow::DT_STRING, input_tensor->shape(),
                         ctx->input_memory_type(0), ctx->input_alloc_attr(0));
  if (maybe_forwarded) {
    output_tensor = maybe_forwarded.get();
    TF_RETURN_IF_ERROR(ctx->set_output("output", *output_tensor));
  } else {
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("output", input_tensor->shape(), &output_tensor));
    output_tensor->flat<string>() = input_tensor->flat<string>();
  }
  auto output_flat = output_tensor->flat<string>();
  for (size_t i = 0; i < output_flat.size(); ++i) {
    if (replace_global) {
      RE2::GlobalReplace(&output_flat(i), match, rewrite);
    } else {
      RE2::Replace(&output_flat(i), match, rewrite);
    }
  }
  return Status::OK();
}
}  // namespace

class RegexReplaceOp : public OpKernel {
 public:
  explicit RegexReplaceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("replace_global", &replace_global_));
  }

  void Compute(OpKernelContext* ctx) override {
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

    const Tensor* rewrite_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("rewrite", &rewrite_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rewrite_tensor->shape()),
                errors::InvalidArgument("Rewrite must be scalar, but received ",
                                        rewrite_tensor->shape().DebugString()));
    const string rewrite = rewrite_tensor->flat<string>()(0);
    OP_REQUIRES_OK(ctx, InternalCompute(match, rewrite, replace_global_, ctx));
  }

 private:
  bool replace_global_;
};

REGISTER_KERNEL_BUILDER(Name("RegexReplace").Device(DEVICE_CPU),
                        RegexReplaceOp);

class StaticRegexReplaceOp : public OpKernel {
 public:
  explicit StaticRegexReplaceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string pattern;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pattern", &pattern));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rewrite", &rewrite_str_));
    re_ = MakeUnique<RE2>(pattern);
    OP_REQUIRES(ctx, re_->ok(),
                errors::InvalidArgument("Invalid pattern: ", pattern,
                                        ", error: ", re_->error()));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("replace_global", &replace_global_));
  }

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx,
                   InternalCompute(*re_, rewrite_str_, replace_global_, ctx));
  }

 private:
  string rewrite_str_;
  std::unique_ptr<RE2> re_;
  bool replace_global_;
};

REGISTER_KERNEL_BUILDER(Name("StaticRegexReplace").Device(DEVICE_CPU),
                        StaticRegexReplaceOp);

}  // namespace tensorflow
