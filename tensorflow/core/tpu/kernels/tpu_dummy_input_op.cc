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

#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
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

  // The output is always the same zero tensor, so it is built on the first
  // call and then shared by every later step, like `ConstantOp`. Holding a
  // reference keeps consumers from forwarding (and mutating) the buffer. It is
  // built lazily rather than in the constructor so that dummy nodes that never
  // run do not hold host memory (see cl/778212281).
  void Compute(OpKernelContext* ctx) override {
    absl::MutexLock lock(&mu_);
    if (!zeros_built_) {
      Tensor zeros;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(dtype_, shape_, &zeros,
                                             ctx->output_alloc_attr(0)));
      if (dtype_ == DT_FLOAT) {
        zeros.flat<float>().setZero();
      } else if (dtype_ == DT_BFLOAT16) {
        zeros.flat<tsl::bfloat16>().setZero();
      } else {
        ctx->SetStatus(absl::InternalError(
            absl::StrCat("Unsupported dtype: ", DataTypeString(dtype_))));
        return;
      }
      zeros_ = std::move(zeros);
      zeros_built_ = true;
    }
    ctx->set_output(0, zeros_);
  }

 private:
  DataType dtype_;
  TensorShape shape_;
  absl::Mutex mu_;
  // A default `Tensor` already reports `IsInitialized()` (it has 0 elements),
  // so track whether `zeros_` was built explicitly.
  bool zeros_built_ ABSL_GUARDED_BY(mu_) = false;
  Tensor zeros_ ABSL_GUARDED_BY(mu_);
};

// TODO(mrry): Add registration for TPU.
REGISTER_KERNEL_BUILDER(Name("TPUDummyInput").Device(DEVICE_CPU),
                        TpuDummyInputOp);

}  // namespace

}  // namespace tensorflow
