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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_H_

#include "absl/container/node_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "tsl/platform/logging.h"

namespace stream_executor {
namespace gpu {
// Formats hipError_t to output prettified values into a log stream.
// Error summaries taken from:
std::string ToString(hipError_t result);

// GpuContext implements the Context class for ROCm GPUs.
class GpuContext : public Context {
 public:
  GpuContext(hipCtx_t context, const int ordinal)
      : context_(context), device_ordinal_(ordinal) {}

  hipCtx_t context() const { return context_; }
  void SetActive() override;
  bool IsActive() const override;
  int device_ordinal() const override { return device_ordinal_; }

  // Disallow copying and moving.
  GpuContext(GpuContext&&) = delete;
  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(GpuContext&&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;

 private:
  hipCtx_t const context_;
  const int device_ordinal_;
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_H_
