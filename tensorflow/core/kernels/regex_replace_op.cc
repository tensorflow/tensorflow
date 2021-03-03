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
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace {

// Execute the specified regex using the given context.
// Context requirements:
//  - "input" string Tensor at input_index=0
//  - "output" string Tensor at output_index=0
Status InternalCompute(const RE2& regex, const string& rewrite,
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
    output_tensor->flat<tstring>() = input_tensor->flat<tstring>();
  }
  auto output_flat = output_tensor->flat<tstring>();
  for (size_t i = 0; i < output_flat.size(); ++i) {
    // TODO(dero): Mitigate copy; Global and GlobalReplace below currently only
    // accept std::string.
    string buf = output_flat(i);
    if (replace_global) {
      RE2::GlobalReplace(&buf, regex, rewrite);
    } else {
      RE2::Replace(&buf, regex, rewrite);
    }
    output_flat(i) = std::move(buf);
  }
  return Status::OK();
}
}  // namespace

class RegexReplaceOp : public OpKernel {
 public:
  explicit RegexReplaceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("replace_global", &replace_global_));
  }

  ~RegexReplaceOp() override {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* pattern_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("pattern", &pattern_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(pattern_tensor->shape()),
                errors::InvalidArgument("Pattern must be scalar, but received ",
                                        pattern_tensor->shape().DebugString()));
    const string& pattern = pattern_tensor->scalar<tstring>()();
    std::shared_ptr<RE2> regex = CachedRE2(pattern);
    OP_REQUIRES(ctx, regex->ok(),
                errors::InvalidArgument("Invalid pattern: ", pattern,
                                        ", error: ", regex->error()));

    const Tensor* rewrite_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("rewrite", &rewrite_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rewrite_tensor->shape()),
                errors::InvalidArgument("Rewrite must be scalar, but received ",
                                        rewrite_tensor->shape().DebugString()));
    const string& rewrite = rewrite_tensor->scalar<tstring>()();
    OP_REQUIRES_OK(ctx, InternalCompute(*regex, rewrite, replace_global_, ctx));
  }

 private:
  std::shared_ptr<RE2> CachedRE2(const string& pattern) {
    {
      tf_shared_lock l(mu_);
      if (regex_ != nullptr && regex_->pattern() == pattern) {
        return regex_;
      }
    }
    // Construct the new RE2 object before acquiring the lock.
    auto regex = std::make_shared<RE2>(pattern);
    {
      mutex_lock l(mu_);
      // Swap instead of assigning so that we destruct the old
      // RE2 object (when necessary) after releasing the lock.
      regex_.swap(regex);
      return regex_;
    }
  }

  bool replace_global_;
  mutex mu_;
  std::shared_ptr<RE2> regex_ TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(RegexReplaceOp);
};

REGISTER_KERNEL_BUILDER(Name("RegexReplace").Device(DEVICE_CPU),
                        RegexReplaceOp);

class StaticRegexReplaceOp : public OpKernel {
 public:
  explicit StaticRegexReplaceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string pattern;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pattern", &pattern));
    re_ = MakeUnique<RE2>(pattern);
    OP_REQUIRES(ctx, re_->ok(),
                errors::InvalidArgument("Invalid pattern: ", pattern,
                                        ", error: ", re_->error()));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rewrite", &rewrite_str_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("replace_global", &replace_global_));
  }

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx,
                   InternalCompute(*re_, rewrite_str_, replace_global_, ctx));
  }

 private:
  std::unique_ptr<RE2> re_;
  string rewrite_str_;
  bool replace_global_;
};

REGISTER_KERNEL_BUILDER(Name("StaticRegexReplace").Device(DEVICE_CPU),
                        StaticRegexReplaceOp);

}  // namespace tensorflow
