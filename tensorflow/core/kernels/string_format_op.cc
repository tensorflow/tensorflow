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

#include <iostream>
#include "absl/strings/str_split.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

class StringFormatOp : public OpKernel {
 public:
  explicit StringFormatOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string template_;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("template", &template_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("placeholder", &placeholder_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("summarize", &summarize_));

    split_template_ = absl::StrSplit(template_, placeholder_);
    int64 num_placeholders = split_template_.size() - 1;
    OP_REQUIRES(ctx, ctx->num_inputs() == num_placeholders,
                errors::InvalidArgument(strings::StrCat(
                    "num placeholders in template and num inputs must match: ",
                    num_placeholders, " vs. ", ctx->num_inputs())));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* formatted_string = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({}), &formatted_string));

    string msg;
    strings::StrAppend(&msg, split_template_[0].c_str());
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      strings::StrAppend(&msg, ctx->input(i).SummarizeValue(summarize_, true));
      strings::StrAppend(&msg, split_template_[i + 1].c_str());
    }

    formatted_string->scalar<tstring>()() = std::move(msg);
  }

 private:
  int32 summarize_ = 0;
  string placeholder_;
  std::vector<std::string> split_template_;
};

REGISTER_KERNEL_BUILDER(Name("StringFormat").Device(DEVICE_CPU),
                        StringFormatOp);

}  // end namespace tensorflow
