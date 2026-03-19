#include "absl/status/statusor.h"
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

// The ROCM-specific Driver library support, implementing the general Driver
// interface.

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_CONTEXT_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_CONTEXT_H_

#include <cstdint>

#include "absl/status/status.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/context_map.h"

namespace stream_executor::gpu {

// RocmContext implements the Context class for ROCm GPUs.
class RocmContext : public Context {
 public:
  RocmContext(hipCtx_t context, const int ordinal)
      : context_(context), device_ordinal_(ordinal) {}
  ~RocmContext() override;

  hipCtx_t context() const { return context_; }
  void SetActive() override;
  bool IsActive() const override;
  int device_ordinal() const override { return device_ordinal_; }
  absl::Status Synchronize() override;

  // Disallow copying and moving.
  RocmContext(RocmContext&&) = delete;
  RocmContext(const RocmContext&) = delete;
  RocmContext& operator=(RocmContext&&) = delete;
  RocmContext& operator=(const RocmContext&) = delete;

  // Returns the free amount of memory and total amount of memory, as reported
  // by hipDeviceTotalMem.
  bool GetDeviceMemoryUsage(int64_t* free_out, int64_t* total_out);

  // Returns the total amount of memory available on the device.
  static bool GetDeviceTotalMemory(hipDevice_t device, uint64_t* result);

  // Returns the context map for all XLA-known ROCm contexts.
  static ContextMap<hipCtx_t, RocmContext>* GetContextMap();

  // Creates a new context for the given device.
  static absl::StatusOr<RocmContext*> Create(int device_ordinal,
                                             hipDevice_t device);

 private:
  hipCtx_t const context_;
  const int device_ordinal_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_CONTEXT_H_
