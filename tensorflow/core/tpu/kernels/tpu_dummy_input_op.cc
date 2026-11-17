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

#include <algorithm>
#include <cstdint>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {

namespace {

// One zeroed host buffer shared by every `TPUDummyInput` output in the process.
// It grows to the largest dummy input, so all dummy inputs together cost the
// memory of the largest one instead of their sum (b/399907474).
ABSL_CONST_INIT absl::Mutex shared_zeros_mu(absl::kConstInit);
Tensor* shared_zeros ABSL_GUARDED_BY(shared_zeros_mu) = nullptr;
// Largest dummy input seen by any kernel constructor so far. The buffer always
// grows straight to this size, so it is not regrown while a step visits dummy
// inputs of increasing size.
int64_t max_reserved_bytes ABSL_GUARDED_BY(shared_zeros_mu) = 0;

int64_t NumBytes(DataType dtype, const TensorShape& shape) {
  return shape.num_elements() * DataTypeSize(dtype);
}

// Records that `num_bytes` of zeros will be needed, without allocating.
void ReserveSharedZeros(int64_t num_bytes) {
  absl::MutexLock lock(&shared_zeros_mu);
  max_reserved_bytes = std::max(max_reserved_bytes, num_bytes);
}

// Returns a zero tensor of `dtype` and `shape` that aliases the prefix of
// `shared_zeros`. The shared buffer always holds a reference, so consumers can
// never forward (and write into) the returned buffer.
absl::StatusOr<Tensor> SharedZeros(DataType dtype, const TensorShape& shape) {
  const int64_t num_bytes = NumBytes(dtype, shape);
  absl::MutexLock lock(&shared_zeros_mu);
  const int64_t alloc_bytes = std::max(num_bytes, max_reserved_bytes);
  if (shared_zeros == nullptr || shared_zeros->NumElements() < alloc_bytes) {
    Tensor larger(DT_INT8, TensorShape({alloc_bytes}));
    if (!larger.IsInitialized()) {
      return absl::ResourceExhaustedError(
          absl::StrCat("Failed to allocate ", alloc_bytes,
                       " bytes of zeros for TPUDummyInput"));
    }
    larger.flat<int8_t>().setZero();
    // Outputs from earlier steps keep the smaller buffer alive until freed.
    if (shared_zeros == nullptr) {
      shared_zeros = new Tensor(std::move(larger));
    } else {
      *shared_zeros = std::move(larger);
    }
  }
  Tensor zeros;
  absl::Status status =
      zeros.BitcastFrom(shared_zeros->Slice(0, num_bytes), dtype, shape);
  if (!status.ok()) return status;
  return zeros;
}

class TpuDummyInputOp : public OpKernel {
 public:
  explicit TpuDummyInputOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &shape_));
    OP_REQUIRES(ctx, dtype_ == DT_FLOAT || dtype_ == DT_BFLOAT16,
                absl::InternalError(absl::StrCat("Unsupported dtype: ",
                                                 DataTypeString(dtype_))));
    ReserveSharedZeros(NumBytes(dtype_, shape_));
  }

  void Compute(OpKernelContext* ctx) override {
    absl::StatusOr<Tensor> zeros = SharedZeros(dtype_, shape_);
    OP_REQUIRES_OK(ctx, zeros.status());
    ctx->set_output(0, *zeros);
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
