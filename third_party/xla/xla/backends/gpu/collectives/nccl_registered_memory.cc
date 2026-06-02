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

#include "xla/backends/gpu/collectives/nccl_registered_memory.h"

#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/collectives/nccl_errors.h"
#include "xla/stream_executor/device_address.h"

// Include NCCL after XLA headers.
#include "third_party/nccl/nccl.h"

namespace xla::gpu {

NcclRegisteredMemory::NcclRegisteredMemory(
    ncclComm_t comm, void* handle, stream_executor::DeviceAddressBase addr)
    : comm_(comm), handle_(handle), addr_(addr) {}

absl::StatusOr<std::unique_ptr<NcclRegisteredMemory>>
NcclRegisteredMemory::Create(ncclComm_t comm,
                             stream_executor::DeviceAddressBase addr) {
  VLOG(3) << absl::StrFormat(
      "Create NCCL registered memory on comm=%p from: ptr=%p; size=%ld", comm,
      addr.opaque(), addr.size());

  void* handle = nullptr;
  XLA_NCCL_RETURN_IF_ERROR(
      ncclCommRegister(comm, addr.opaque(), addr.size(), &handle));

  return absl::WrapUnique(new NcclRegisteredMemory(comm, handle, addr));
}

NcclRegisteredMemory::~NcclRegisteredMemory() {
  VLOG(3) << absl::StrFormat("Destroy %v", *this);
  XLA_NCCL_LOG_IF_ERROR(ncclCommDeregister(comm_, handle_));
}

stream_executor::DeviceAddressBase NcclRegisteredMemory::addr() const {
  return addr_;
}

std::string NcclRegisteredMemory::ToString() const {
  return absl::StrFormat(
      "NcclRegisteredMemory(comm=%p, handle=%p, ptr=%p, size=%ld)", comm_,
      handle_, addr_.opaque(), addr_.size());
}

}  // namespace xla::gpu
