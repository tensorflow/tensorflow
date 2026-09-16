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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_RCCL_REGISTERED_MEMORY_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_RCCL_REGISTERED_MEMORY_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "rocm/include/rccl/rccl.h"
#include "xla/core/collectives/registered_memory.h"
#include "xla/stream_executor/device_address.h"

namespace xla::gpu {

// A RCCL local buffer registration handle (ncclCommRegister). Registering a
// buffer lets RCCL set up IPC / network handles for the range ahead of time so
// that subsequent collectives over the range can run zero-copy.
//
// Unlike RcclSymmetricMemory this does NOT call ncclCommWindowRegister and
// makes no symmetry assumption about the buffer across ranks: it is safe for
// arbitrary device address ranges (e.g. slices into a larger allocation) that
// are not symmetric across the clique. The handle is deregistered
// (ncclCommDeregister) on destruction. Analogous to NcclRegisteredMemory for
// the ROCm/RCCL platform.
class RcclRegisteredMemory final : public RegisteredMemory {
 public:
  ~RcclRegisteredMemory() final;

  static absl::StatusOr<std::unique_ptr<RcclRegisteredMemory>> Create(
      ncclComm_t comm, stream_executor::DeviceAddressBase addr);

  stream_executor::DeviceAddressBase addr() const final;

  std::string ToString() const final;

 private:
  RcclRegisteredMemory(ncclComm_t comm, void* handle,
                       stream_executor::DeviceAddressBase addr);

  ncclComm_t comm_;
  void* handle_;
  stream_executor::DeviceAddressBase addr_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_RCCL_REGISTERED_MEMORY_H_
