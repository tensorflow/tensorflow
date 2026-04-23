/* Copyright 2023 The OpenXLA Authors.

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

// On ROCm, hipCtx_t is a thin wrapper around a device ordinal and the entire
// context lifecycle (Retain/Release/SetCurrent/GetCurrent) is a no-op.  AMD
// has deprecated every hipCtx* and hipDevicePrimaryCtx* API since ROCm 1.9
// with the recommendation to use hipSetDevice / hipGetDevice instead.
//
// RocmContext is a trivial implementation of the Context interface that
// delegates to hipSetDevice/hipGetDevice.  It is intended to be owned as
// a plain value field inside RocmExecutor.

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_CONTEXT_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_CONTEXT_H_

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/tsl/platform/errors.h"

namespace stream_executor::gpu {

class RocmContext : public Context {
 public:
  explicit RocmContext(int device_ordinal) : device_ordinal_(device_ordinal) {}
  ~RocmContext() override = default;

  void SetActive() override {
    CHECK_OK(ToStatus(hipSetDevice(device_ordinal_), "Failed to set device"));
  }

  bool IsActive() const override {
    int current_device;
    if (hipGetDevice(&current_device) != hipSuccess) {
      return false;
    }
    return current_device == device_ordinal_;
  }

  int device_ordinal() const override { return device_ordinal_; }

  absl::Status Synchronize() override {
    ScopedActivateContext activation(this);
    TF_RETURN_IF_ERROR(ToStatus(hipDeviceSynchronize(),
                                "could not synchronize on ROCM device"));
    return absl::OkStatus();
  }

  RocmContext(RocmContext&&) = delete;
  RocmContext(const RocmContext&) = delete;
  RocmContext& operator=(RocmContext&&) = delete;
  RocmContext& operator=(const RocmContext&) = delete;

 private:
  const int device_ordinal_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_CONTEXT_H_
