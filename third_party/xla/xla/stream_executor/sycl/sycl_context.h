/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_CONTEXT_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_CONTEXT_H_

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/sycl/sycl_gpu_runtime.h"

namespace stream_executor::sycl {

// SyclContext is a wrapper around SYCL context that provides methods to
// create/synchronize contexts and get device memory information.
class SyclContext : public gpu::Context {
 public:
  // Initializes the SYCL context with the given device ordinal.
  explicit SyclContext(::sycl::context context, const int device_ordinal)
      : context_(context), device_ordinal_(device_ordinal) {}

  // No explicit destructor is needed since ::sycl::context is an RAII object.
  ~SyclContext() override = default;

  ::sycl::context context() const { return context_; }

  // SYCL does not have a concept of an active context like CUDA/ROCm.
  // SYCL contexts are created and used directly without needing to set them
  // as active. These methods are provided to maintain interface consistency
  // with other backends.
  void SetActive() override {
    LOG(WARNING) << "SetActive is not supported for SYCL context";
  }

  bool IsActive() const override {
    LOG(WARNING) << "IsActive is not supported for SYCL context";
    return false;
  }

  int device_ordinal() const override { return device_ordinal_; }

  // Synchronizes all streams associated with device_ordinal_.
  absl::Status Synchronize() override;

  // Ensure SyclContext is moveable but not copyable.
  SyclContext(const SyclContext&) = delete;
  SyclContext& operator=(const SyclContext&) = delete;
  SyclContext(SyclContext&& other) noexcept;
  SyclContext& operator=(SyclContext&& other) noexcept;

  // Returns the total amount of memory available on the given device.
  static absl::StatusOr<uint64_t> GetDeviceTotalMemory(
      const ::sycl::device& device);

  // Creates a new context for the given device ordinal.
  // Returns an error if the context cannot be created.
  static absl::StatusOr<std::unique_ptr<SyclContext>> Create(
      int device_ordinal);

 private:
  ::sycl::context context_;
  const int device_ordinal_;
};

}  // namespace stream_executor::sycl

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_CONTEXT_H_
