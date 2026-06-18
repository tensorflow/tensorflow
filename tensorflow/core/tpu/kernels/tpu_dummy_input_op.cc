/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tsl/platform/bfloat16.h"

namespace tensorflow {

namespace {

class TpuDummyInputOp : public OpKernel {
 public:
  explicit TpuDummyInputOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &shape_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape_, &output));
    if (dtype_ == DT_FLOAT) {
      output->flat<float>().setZero();
    } else if (dtype_ == DT_BFLOAT16) {
      output->flat<tsl::bfloat16>().setZero();
    } else {
      ctx->SetStatus(absl::InternalError(
          absl::StrCat("Unsupported dtype: ", DataTypeString(dtype_))));
      return;
    }
  }

 private:
  DataType dtype_;
  TensorShape shape_;
};

// TODO(mrry): Add registration for TPU.
REGISTER_KERNEL_BUILDER(Name("TPUDummyInput").Device(DEVICE_CPU),
                        TpuDummyInputOp);

}  // namespace

}  // namespace tensorflow
