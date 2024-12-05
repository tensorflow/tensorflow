/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/gpu/collectives/nccl_communicator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/collectives/nccl_errors.h"
#include "xla/core/collectives/communicator.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/rccl/rccl.h"
#else
#include "rocm/include/rccl.h"
#endif  // TF_ROCM_VERSION >= 50200
#else
#include "third_party/nccl/nccl.h"
#endif  // TENSORFLOW_USE_ROCM

namespace xla::gpu {

namespace {
// An RAII handle for user buffers registered with an NCCL communicator.
class NcclRegisteredBufferHandle : public Communicator::RegisteredBufferHandle {
 public:
  NcclRegisteredBufferHandle(ncclComm_t comm, void* handle);
  ~NcclRegisteredBufferHandle() override;

  absl::Status Unregister() final;

 private:
  ncclComm_t comm_;
  void* handle_;
};
}  // namespace

NcclRegisteredBufferHandle::NcclRegisteredBufferHandle(ncclComm_t comm,
                                                       void* handle)
    : comm_(comm), handle_(handle) {}

NcclRegisteredBufferHandle::~NcclRegisteredBufferHandle() {
  if (auto status = Unregister(); !status.ok()) {
    LOG(ERROR) << status.message();
  }
}

absl::Status NcclRegisteredBufferHandle::Unregister() {
  VLOG(3) << absl::StreamFormat(
      "Deregister buffer for NCCL communicator; handle=%p; comm=%p", handle_,
      comm_);
#if (NCCL_VERSION_CODE >= 21901)
  return XLA_NCCL_STATUS(ncclCommDeregister(comm_, handle_));
#else
  return Unimplemented("NCCL version does not support ncclCommDeregister");
#endif  // NCCL_VERSION_CODE >= 21901
}

NcclCommunicator::NcclCommunicator(ncclComm_t comm) : comm_(comm) {
  VLOG(1) << "Created " << *this;
}

NcclCommunicator::~NcclCommunicator() {
  VLOG(1) << "Destroy " << *this;
  XLA_NCCL_LOG_IF_ERROR(ncclCommDestroy(comm_));
}

absl::Status NcclCommunicator::Abort() {
  VLOG(1) << "Abort NCCL communicator: " << ToString();
  return XLA_NCCL_STATUS(ncclCommAbort(comm_));
}

absl::Status NcclCommunicator::HealthCheck() const {
  VLOG(5) << "Get last async error for NCCL communicator: " << ToString();

  ncclResult_t async_err;
  XLA_NCCL_RETURN_IF_ERROR(ncclCommGetAsyncError(comm_, &async_err));
  if (async_err == ncclSuccess) return absl::OkStatus();

  return Internal("%s. Last NCCL error (maybe unrelated): %s",
                  ncclGetLastError(comm_), ncclGetErrorString(async_err));
}

absl::StatusOr<size_t> NcclCommunicator::NumRanks() const {
  VLOG(5) << "Get the number of ranks in NCCL communicator: " << ToString();
  int32_t count;
  XLA_NCCL_RETURN_IF_ERROR(ncclCommCount(comm_, &count));
  return count;
}

absl::StatusOr<std::unique_ptr<Communicator::RegisteredBufferHandle>>
NcclCommunicator::RegisterBuffer(stream_executor::DeviceMemoryBase buffer) {
  VLOG(3) << absl::StreamFormat(
      "Register buffer for NCCL communicator; buffer=%p; size=%d; comm=%p",
      buffer.opaque(), buffer.size(), comm_);
#if (NCCL_VERSION_CODE >= 21901)
  void* handle = nullptr;
  XLA_NCCL_RETURN_IF_ERROR(
      ncclCommRegister(comm_, buffer.opaque(), buffer.size(), &handle));
  return std::make_unique<NcclRegisteredBufferHandle>(comm_, handle);
#else
  return Unimplemented("NCCL version does not support ncclCommRegister");
#endif  // NCCL_VERSION_CODE >= 21901
}

std::string NcclCommunicator::ToString() const {
  return absl::StrFormat("NccCommunicator(ncclComm_t=%p)", comm_);
}

}  // namespace xla::gpu
