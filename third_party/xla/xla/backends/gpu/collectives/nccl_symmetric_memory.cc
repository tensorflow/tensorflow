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
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/collectives/nccl_errors.h"
#include "xla/backends/gpu/collectives/nccl_types.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/future.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream_executor.h"

// Include NCCL after XLA headers.
#include "third_party/nccl/nccl.h"
#include "third_party/nccl/nccl_device.h"  // IWYU pragma: keep
#include "xla/tsl/concurrency/executor.h"

namespace xla::gpu {

namespace {

Future<> Execute(absl::AnyInvocable<absl::Status() &&> f,
                 std::shared_ptr<tsl::Executor> executor) {
  return executor ? MakeFutureOn<void>(*executor, std::move(f))
                  : Future<>(std::move(f)());
}

void LogInterconnectStatus(stream_executor::StreamExecutor* stream_executor) {
  if (stream_executor == nullptr) {
    LOG(ERROR) << "Interconnect Status: Unknown (StreamExecutor not available)";
    return;
  }

  std::string interconnect_status = "Unknown (Failed to get status)";
  absl::StatusOr<std::string> interconnect_status_or =
      stream_executor->GetInterconnectStatus();
  if (!interconnect_status_or.ok()) {
    LOG(ERROR) << "Failed to get interconnect status: "
               << interconnect_status_or.status();
    return;
  }

  LOG(ERROR) << "Interconnect Status: " << *interconnect_status_or;
}

}  // namespace

NcclSymmetricMemory::NcclSymmetricMemory(
    std::shared_ptr<NcclCommState> comm_state, ncclWindow_t win,
    stream_executor::DeviceAddressBase addr,
    std::shared_ptr<tsl::Executor> executor)
    : comm_state_(comm_state), win_(win), addr_(addr), executor_(executor) {}

absl::StatusOr<std::unique_ptr<NcclSymmetricMemory>>
NcclSymmetricMemory::Create(std::shared_ptr<NcclCommState> comm_state,
                            stream_executor::DeviceAddressBase addr,
                            const std::shared_ptr<tsl::Executor> executor,
                            stream_executor::StreamExecutor* stream_executor,
                            bool symmetric_window) {
  if (comm_state == nullptr) {
    return absl::InvalidArgumentError("NCCL comm_state is null.");
  }

  ncclWindow_t win = nullptr;
  ncclResult_t nccl_status;
  {
    absl::MutexLock lock(comm_state->mutex);
    ncclComm_t comm = comm_state->comm;
    if (comm == nullptr) {
      return absl::InvalidArgumentError("NCCL comm is null.");
    }
    // Difference between NCCL_WIN_COLL_SYMMETRIC and NCCL_WIN_DEFAULT is purely
    // semantic.
    // With NCCL_WIN_COLL_SYMMETRIC user should guarantee that logical memory
    // under the window is symmetrical at all participants. This allows NCCL to
    // run copy engine and RMA symmetrical kernels. We are using currently
    // NCCL_WIN_DEFAULT by default to disable these kernels because XLA runtime
    // might cache window handlers, which causes NCCL to incorrectly choose a
    // symmetric kernel on a default non-symmetric memory, just because another
    // executable cached a window handler.
    int win_flags =
        symmetric_window ? NCCL_WIN_COLL_SYMMETRIC : NCCL_WIN_DEFAULT;
    VLOG(3) << absl::StrFormat(
        "Create NCCL symmetric memory on comm=%p from: ptr=%p; size=%ld; "
        "win_flags=%d",
        comm, addr.opaque(), addr.size(), win_flags);
    nccl_status = ncclCommWindowRegister(comm, addr.opaque(), addr.size(), &win,
                                         win_flags);
  }

  if (nccl_status != ncclSuccess) {
    LogInterconnectStatus(stream_executor);
    return XLA_NCCL_STATUS(nccl_status);
  }

  if (win == nullptr) {
    LogInterconnectStatus(stream_executor);
    return absl::FailedPreconditionError(
        "ncclCommWindowRegister returned a null window because NCCL symmetric "
        "memory is not supported on this communicator (e.g. P2P/NVLink or "
        "CUMEM is unavailable on fractional/virtualized GPU slices).");
  }

  return absl::WrapUnique(
      new NcclSymmetricMemory(comm_state, win, addr, executor));
}

NcclSymmetricMemory::~NcclSymmetricMemory() {
  if (win_ == nullptr) {
    return;
  }
  absl::Status status =
      Execute(
          [&] {
            VLOG(3) << absl::StrFormat(
                "Destroy %v with addr=%p, size=%ld executor=%p", *this,
                addr_.opaque(), addr_.size(), executor_.get());
            absl::MutexLock lock(comm_state_->mutex);
            XLA_NCCL_LOG_IF_ERROR(
                ncclCommWindowDeregister(comm_state_->comm, win_));
            return absl::OkStatus();
          },
          executor_)
          .Await();
  if (!status.ok()) {
    LOG(ERROR) << "Failed to destroy NCCL symmetric memory: " << status;
  }
}

stream_executor::DeviceAddressBase NcclSymmetricMemory::addr() const {
  return addr_;
}

absl::StatusOr<stream_executor::DeviceAddressBase>
NcclSymmetricMemory::multimem_addr() const {
#if (NCCL_VERSION_CODE >= 22900) || defined(USE_NCCL_HOST_API)
  if (win_ == nullptr) {
    return absl::FailedPreconditionError(
        "Cannot get multimem address from null NCCL window.");
  }
  void* multimem = nullptr;
  XLA_NCCL_RETURN_IF_ERROR(ncclGetLsaMultimemDevicePointer(win_, 0, &multimem));
  if (multimem != nullptr) {
    return stream_executor::DeviceAddressBase(multimem, addr_.size());
  }
#endif
  return absl::UnimplementedError(
      "Multimem not supported on this NCCL version or device");
}

absl::StatusOr<stream_executor::DeviceAddressBase>
NcclSymmetricMemory::peer_addr(RankId peer) const {
#if (NCCL_VERSION_CODE >= 22902) || defined(USE_NCCL_HOST_API)
  if (win_ == nullptr) {
    return absl::FailedPreconditionError(
        "Cannot get peer address from null NCCL window.");
  }
  void* peer_addr = nullptr;
  XLA_NCCL_RETURN_IF_ERROR(
      ncclGetPeerDevicePointer(win_, 0, peer.value(), &peer_addr));
  if (peer_addr != nullptr) {
    return stream_executor::DeviceAddressBase(peer_addr, addr_.size());
  }
  return absl::FailedPreconditionError(absl::StrFormat(
      "Peer rank %d is not load/store accessible from this rank",
      peer.value()));
#else
  return absl::UnimplementedError(
      "Peer address requires ncclGetPeerDevicePointer (NCCL >= 2.29.2)");
#endif
}

std::string NcclSymmetricMemory::ToString() const {
  absl::MutexLock lock(comm_state_->mutex);
  return absl::StrFormat(
      "NcclSymmetricMemory(comm=%p, win=%p, ptr=%p, size=%ld)",
      comm_state_->comm, win_, addr_.opaque(), addr_.size());
}

NcclSymmetricMemory::PackedKernelArg NcclSymmetricMemory::PackKernelArg()
    const {
  return win_;
}

}  // namespace xla::gpu
