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

class RegexFullMatchOp : public OpKernel {
 public:
  explicit RegexFullMatchOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  ~RegexFullMatchOp() override {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    const auto& input_flat = input_tensor->flat<tstring>();

    const Tensor* pattern_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("pattern", &pattern_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(pattern_tensor->shape()),
                errors::InvalidArgument("Pattern must be scalar, but received ",
                                        pattern_tensor->shape().DebugString()));
    const string pattern = pattern_tensor->flat<tstring>()(0);
    std::shared_ptr<RE2> regex = CachedRE2(pattern);
    OP_REQUIRES(ctx, regex->ok(),
                errors::InvalidArgument("Invalid pattern: ", pattern,
                                        ", error: ", regex->error()));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("output", input_tensor->shape(),
                                             &output_tensor));
    auto output_flat = output_tensor->flat<bool>();
    for (size_t i = 0; i < input_flat.size(); ++i) {
      output_flat(i) = RE2::FullMatch(input_flat(i), *regex);
    }
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

  mutex mu_;
  std::shared_ptr<RE2> regex_ TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(RegexFullMatchOp);
};

REGISTER_KERNEL_BUILDER(Name("RegexFullMatch").Device(DEVICE_CPU),
                        RegexFullMatchOp);

class StaticRegexFullMatchOp : public OpKernel {
 public:
  explicit StaticRegexFullMatchOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string pattern;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pattern", &pattern));
    re_ = MakeUnique<RE2>(pattern);
    OP_REQUIRES(ctx, re_->ok(),
                errors::InvalidArgument("Invalid pattern: ", pattern,
                                        ", error: ", re_->error()));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    const auto& input_flat = input_tensor->flat<tstring>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("output", input_tensor->shape(),
                                             &output_tensor));
    auto output_flat = output_tensor->flat<bool>();
    for (size_t i = 0; i < input_flat.size(); ++i) {
      output_flat(i) = RE2::FullMatch(input_flat(i), *re_);
    }
  }

 private:
  std::unique_ptr<RE2> re_;
};

REGISTER_KERNEL_BUILDER(Name("StaticRegexFullMatch").Device(DEVICE_CPU),
                        StaticRegexFullMatchOp);

}  // namespace tensorflow
