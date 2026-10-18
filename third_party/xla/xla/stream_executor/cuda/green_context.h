/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_GREEN_CONTEXT_H_
#define XLA_STREAM_EXECUTOR_CUDA_GREEN_CONTEXT_H_

#include <memory>

#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"

namespace stream_executor::gpu {

// GreenContext wraps a CUDA green context: a partition of a physical device's
// streaming multiprocessors (SMs) onto which work can be launched via streams
// created from it. Green contexts are a general SM-partitioning mechanism
// (available since CUDA 12.4); the SM partition can be chosen by raw SM count
// or by higher-level partitioning schemes layered on top.
//
// This class does not own or manage the device's primary CUDA context. Callers
// must ensure the device's primary CUDA context is current on the calling
// thread before invoking any method that touches CUDA state (the factories and
// CreateStream).
class GreenContext {
 public:
  ~GreenContext();

  GreenContext(const GreenContext&) = delete;
  GreenContext& operator=(const GreenContext&) = delete;
  GreenContext(GreenContext&&) = delete;
  GreenContext& operator=(GreenContext&&) = delete;

  // Creates a green context over the SMs described by `sm_resource`, which must
  // be a CU_DEV_RESOURCE_TYPE_SM resource (e.g. obtained from
  // cuDeviceGetDevResource and, optionally, carved out via a split API).
  static absl::StatusOr<std::unique_ptr<GreenContext>> Create(
      CUdevice device, CUdevResource sm_resource);

  // Convenience factory: splits a partition of (at least) `sm_count` SMs off
  // the device's full SM resource and creates a green context over that single
  // partition. The actual SM count of the resulting partition (which CUDA may
  // round up for alignment) is reported by sm_count().
  static absl::StatusOr<std::unique_ptr<GreenContext>> CreateWithSmCount(
      CUdevice device, int sm_count);

  CUgreenCtx green_ctx() const { return green_ctx_; }
  CUdevice device() const { return device_; }

  // Number of SMs in this green context's partition.
  int sm_count() const { return sm_count_; }

  // Creates a CUstream that launches work onto this green context's SM
  // partition. The returned stream is an ordinary CUstream owned by the caller
  // (destroy with cuStreamDestroy).
  absl::StatusOr<CUstream> CreateStream(int priority) const;

 private:
  GreenContext(CUdevice device, CUgreenCtx green_ctx, int sm_count);

  CUdevice device_;
  CUgreenCtx green_ctx_;
  int sm_count_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_GREEN_CONTEXT_H_
