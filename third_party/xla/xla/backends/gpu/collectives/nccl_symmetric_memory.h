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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_NCCL_SYMMETRIC_MEMORY_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_NCCL_SYMMETRIC_MEMORY_H_

#include "absl/status/statusor.h"
#include "third_party/nccl/nccl.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/stream_executor/device_address.h"

namespace xla::gpu {

// A NCCL window registration handle that makes local buffers accessible from
// remote peers via symmetric memory registration process.
class NcclSymmetricMemory final : public SymmetricMemory {
 public:
  ~NcclSymmetricMemory() final;

  static absl::StatusOr<std::unique_ptr<NcclSymmetricMemory>> Create(
      ncclComm_t comm, stream_executor::DeviceAddressBase addr);

  std::string ToString() const final;

  stream_executor::DeviceAddressBase addr() const { return addr_; }

 private:
  NcclSymmetricMemory(ncclComm_t comm, ncclWindow_t win,
                      stream_executor::DeviceAddressBase addr);

  ncclComm_t comm_;
  ncclWindow_t win_;
  stream_executor::DeviceAddressBase addr_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_NCCL_SYMMETRIC_MEMORY_H_
