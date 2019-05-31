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
#include "tensorflow/core/kernels/string_util.h"

namespace tensorflow {
namespace {

class StringLengthOp : public OpKernel {
 public:
  explicit StringLengthOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string unit;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("unit", &unit));
    OP_REQUIRES_OK(ctx, ParseCharUnit(unit, &unit_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    auto src = input.flat<string>();
    auto dst = output->flat<int32>();

    switch (unit_) {
      case CharUnit::BYTE:
        for (int n = 0; n < src.size(); ++n) {
          dst(n) = src(n).size();
        }
        break;
      case CharUnit::UTF8_CHAR:
        for (int n = 0; n < src.size(); ++n) {
          dst(n) = UTF8StrLen(src(n));
        }
        break;
    }
  }

 private:
  CharUnit unit_ = CharUnit::BYTE;
};

REGISTER_KERNEL_BUILDER(Name("StringLength").Device(DEVICE_CPU),
                        StringLengthOp);

}  // namespace
}  // namespace tensorflow
