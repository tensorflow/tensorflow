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

#include "xla/backends/gpu/collectives/nccl_symmetric_memory.h"

#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/collectives/nccl_errors.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/stream_executor/device_address.h"

// Include NCCL after XLA headers.
#include "third_party/nccl/nccl.h"

#if NCCL_VERSION_CODE >= 22900
#include "third_party/nccl/nccl_device.h"
#endif  // NCCL_VERSION_CODE >= 22800

namespace xla::gpu {

NcclSymmetricMemory::NcclSymmetricMemory(
    ncclComm_t comm, ncclWindow_t win, stream_executor::DeviceAddressBase addr)
    : comm_(comm), win_(win), addr_(addr) {}

absl::StatusOr<std::unique_ptr<NcclSymmetricMemory>>
NcclSymmetricMemory::Create(ncclComm_t comm,
                            stream_executor::DeviceAddressBase addr) {
  VLOG(3) << absl::StrFormat(
      "Create NCCL symmetric memory on comm=%p from: ptr=%p; size=%ld", comm,
      addr.opaque(), addr.size());

  ncclWindow_t win;
  XLA_NCCL_RETURN_IF_ERROR(ncclCommWindowRegister(
      comm, addr.opaque(), addr.size(), &win, NCCL_WIN_COLL_SYMMETRIC));

  return absl::WrapUnique(new NcclSymmetricMemory(comm, win, addr));
}

NcclSymmetricMemory::~NcclSymmetricMemory() {
  VLOG(3) << absl::StrFormat("Destroy %v", *this);
  XLA_NCCL_LOG_IF_ERROR(ncclCommWindowDeregister(comm_, win_));
}

stream_executor::DeviceAddressBase NcclSymmetricMemory::addr() const {
  return addr_;
}

absl::StatusOr<stream_executor::DeviceAddressBase>
NcclSymmetricMemory::multimem_addr() const {
#if (NCCL_VERSION_CODE >= 22900)
  void* multimem = nullptr;
  XLA_NCCL_RETURN_IF_ERROR(ncclGetLsaMultimemDevicePointer(win_, 0, &multimem));
  if (multimem) {
    return stream_executor::DeviceAddressBase(multimem, addr_.size());
  }
#endif
  return absl::UnimplementedError(
      "Multimem not supported on this NCCL version or device");
}

absl::StatusOr<stream_executor::DeviceAddressBase>
NcclSymmetricMemory::peer_addr(RankId peer) const {
#if (NCCL_VERSION_CODE >= 22900)
  void* peer_addr = nullptr;
  XLA_NCCL_RETURN_IF_ERROR(
      ncclGetLsaDevicePointer(win_, 0, peer.value(), &peer_addr));
  if (peer_addr) {
    return stream_executor::DeviceAddressBase(peer_addr, addr_.size());
  }
#endif
  return absl::UnimplementedError(
      "Peer address not supported on this NCCL version or device");
}

std::string NcclSymmetricMemory::ToString() const {
  return absl::StrFormat(
      "NcclSymmetricMemory(comm=%p, win=%p, ptr=%p, size=%ld)", comm_, win_,
      addr_.opaque(), addr_.size());
}

NcclSymmetricMemory::PackedKernelArg NcclSymmetricMemory::PackKernelArg()
    const {
  return win_;
}

}  // namespace xla::gpu
