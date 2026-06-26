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
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/base/nullability.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "absl/status/status_macros.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/collectives/nccl_errors.h"
#include "xla/backends/gpu/collectives/nccl_group.h"
#include "xla/backends/gpu/collectives/nccl_registered_memory.h"
#include "xla/backends/gpu/collectives/nccl_symmetric_memory.h"
#include "xla/backends/gpu/collectives/nccl_types.h"
#include "xla/backends/gpu/collectives/single_threaded_executor.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/core/collectives/registered_memory.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/future.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

// Include NCCL after XLA headers.
#include "third_party/nccl/nccl.h"         // IWYU pragma: keep
#include "third_party/nccl/nccl_device.h"  // IWYU pragma: keep

namespace xla::gpu {
namespace {

CUstream AsCudaStream(se::Stream* stream) {
  return absl::bit_cast<CUstream>(stream->platform_specific_handle().stream);
}

se::Stream* ToStream(const Communicator::Executor& executor) {
  return absl::down_cast<const GpuCollectives::Executor&>(executor).stream();
}

NcclCapabilities GetCapabilities(std::shared_ptr<NcclCommState> comm_state) {
  bool support_device_comm = false;
  bool support_one_sided_comm = false;
  std::string one_sided_comm_unsupported_reason = "";

#if NCCL_VERSION_CODE >= 22907
  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  {
    absl::MutexLock lock(comm_state->mutex);
    ncclResult_t status = ncclCommQueryProperties(comm_state->comm, &props);
    if (status != ncclSuccess) {
      return {
          /*supports_device_comm=*/false,
          /*supports_one_sided_comm=*/false,
          /*one_sided_comm_unsupported_reason=*/
          absl::StrFormat("NCCL failed to query communicator properties: %s",
                          ncclGetErrorString(status)),
      };
    }
  }

  if (props.hostRmaSupport) {
    support_one_sided_comm = true;
  } else {
    one_sided_comm_unsupported_reason = absl::StrFormat(
        "NCCL reports this communicator does not support host "
        "RMA (hostRmaSupport=false). This is typically caused "
        "by the hardware, network fabric, or NCCL runtime "
        "configuration not supporting one-sided communication "
        "(NCCL version: %d)",
        NCCL_VERSION_CODE);
  }

  if (props.deviceApiSupport) {
    support_device_comm = true;
  }

  return {
      /*supports_device_comm=*/support_device_comm,
      /*supports_one_sided_comm=*/support_one_sided_comm,
      /*one_sided_comm_unsupported_reason=*/one_sided_comm_unsupported_reason,
  };
#elif NCCL_VERSION_CODE >= 22900
  return {
      /*supports_device_comm=*/true,
      /*supports_one_sided_comm=*/false,
      /*one_sided_comm_unsupported_reason=*/
      absl::StrFormat("NCCL >= 2.29.7 is required (current: %d)",
                      NCCL_VERSION_CODE),
  };
#elif NCCL_VERSION_CODE >= 22800
  return {
      /*supports_device_comm=*/true,
      /*supports_one_sided_comm=*/false,
      /*one_sided_comm_unsupported_reason=*/
      absl::StrFormat("NCCL >= 2.29.0 is required (current: %d)",
                      NCCL_VERSION_CODE),
  };
#else
  return {
      /*supports_device_comm=*/false,
      /*supports_one_sided_comm=*/false,
      /*one_sided_comm_unsupported_reason=*/
      absl::StrFormat("NCCL >= 2.29.0 is required (current: %d)",
                      NCCL_VERSION_CODE),
  };
#endif
}

}  // namespace

absl::Status NcclCapabilities::GetOneSidedCommUnsupportedError(
    absl::string_view op) const {
  return Unimplemented("%s is not supported: %s", op,
                       one_sided_comm_unsupported_reason);
}

//==-----------------------------------------------------------------------===//
// NCCL Communicator
//==-----------------------------------------------------------------------===//

NcclCommunicator::NcclCommunicator(se::StreamExecutor* stream_executor,
                                   std::shared_ptr<NcclCommState> comm,
                                   std::unique_ptr<tsl::Executor> executor,
                                   std::shared_ptr<CancellationToken> cancel)
    : stream_executor_(stream_executor),
      comm_(std::move(comm)),
      executor_(std::move(executor)),
      cancel_(std::move(cancel)) {
  capabilities_ = GetCapabilities(comm_);
  VLOG(1) << absl::StreamFormat("[%d] Created NCCL communicator %v",
                                stream_executor_->device_ordinal(), *this);
}

bool NcclCommunicator::SupportsDeviceComm() const {
  return capabilities_.supports_device_comm;
}

absl::StatusOr<std::unique_ptr<GpuDeviceCommunicator>>
NcclCommunicator::CreateDeviceComm(
    const GpuDeviceCommunicator::Requirements& requirements) {
  return ExecuteAwait<std::unique_ptr<GpuDeviceCommunicator>>(
      [this, requirements]()
          -> absl::StatusOr<std::unique_ptr<GpuDeviceCommunicator>> {
        VLOG(5) << "Creating device communicator with requirements: "
                << requirements;
        if (cancel_->IsCancelled()) {
          return FailedPrecondition("NcclCommunicator aborted");
        }

        return NcclDeviceCommunicator::CreateFrom(*this, requirements);
      });
}

absl::StatusOr<std::unique_ptr<RegisteredMemory>>
NcclCommunicator::CreateRegisteredMemory(se::DeviceAddressBase addr) {
  return ExecuteAwait<std::unique_ptr<RegisteredMemory>>(
      [this, addr]() -> absl::StatusOr<std::unique_ptr<RegisteredMemory>> {
        VLOG(5) << "Registering buffer for device address: " << addr.opaque();
        if (cancel_->IsCancelled()) {
          return FailedPrecondition("NcclCommunicator aborted");
        }

        return NcclRegisteredMemory::Create(comm_, addr);
      });
}

absl::StatusOr<std::unique_ptr<SymmetricMemory>>
NcclCommunicator::CreateSymmetricMemory(se::DeviceAddressBase addr) {
  return ExecuteAwait<std::unique_ptr<SymmetricMemory>>(
      [this, addr]() -> absl::StatusOr<std::unique_ptr<SymmetricMemory>> {
        VLOG(5) << "Creating symmetric memory for device address: "
                << addr.opaque();
        if (cancel_->IsCancelled()) {
          return FailedPrecondition("NcclCommunicator aborted");
        }

        return NcclSymmetricMemory::Create(comm_, addr, executor_);
      });
}

absl::StatusOr<std::unique_ptr<NcclCommunicator>> NcclCommunicator::Create(
    se::StreamExecutor* stream_executor,
    absl::AnyInvocable<absl::StatusOr<ncclComm_t>()> make_comm,
    std::shared_ptr<CancellationToken> cancel, bool is_async, tsl::Env& env) {
  if (cancel == nullptr) {
    cancel = std::make_shared<CancellationToken>();
  }
  auto f = [cancel, &make_comm]() -> absl::StatusOr<ncclComm_t> {
    ASSIGN_OR_RETURN(ncclComm_t comm, make_comm());
    if (cancel) {
      RETURN_IF_ERROR(::xla::gpu::PollUntilDone(comm, *cancel));
    } else {
      CancellationToken never_cancelled;
      RETURN_IF_ERROR(::xla::gpu::PollUntilDone(comm, never_cancelled));
    }
    return comm;
  };

  if (!is_async) {
    // If this NcclCommunicator is synchronous, construct ncclComm_t in the
    // calling thread.
    ASSIGN_OR_RETURN(ncclComm_t comm, f());
    auto comm_state = std::make_shared<NcclCommState>(comm);
    return absl::WrapUnique(new NcclCommunicator(stream_executor, comm_state,
                                                 nullptr, std::move(cancel)));
  }

  // If this NcclCommunicator is asynchronous, then all operations on the
  // underlying ncclComm_t, including its creation, must take place on the
  // single threaded executor.
  auto executor = std::make_unique<SingleThreadedExecutor>(env);
  ASSIGN_OR_RETURN(ncclComm_t comm,
                   MakeFutureOn<ncclComm_t>(*executor, f).Await());
  auto comm_state = std::make_shared<NcclCommState>(comm);
  return absl::WrapUnique(new NcclCommunicator(
      stream_executor, comm_state, std::move(executor), std::move(cancel)));
}

NcclCommunicator::~NcclCommunicator() {
  auto f = [this]() -> absl::Status {
    if (comm_ == nullptr) {
      VLOG(1) << "Skipping destruction; null comm_ " << *this;
      return absl::OkStatus();
    }

    if (aborted_) {
      VLOG(1) << "Skipping destruction; already aborted " << *this;
      return absl::OkStatus();
    }

    // Note that we intentionally don't call PollUntilDone. Once comm_ has
    // been destroyed, we can no longer safely touch it.
    absl::MutexLock lock(comm_->mutex);
    if (comm_->comm == nullptr) {
      VLOG(1) << "Skipping destruction; null comm " << *this;
      return absl::OkStatus();
    }

    VLOG(1) << "Destroy " << *this;
    return XLA_NCCL_STATUS(ncclCommDestroy(comm_->comm));
  };

  if (absl::Status s = Execute(f).Await(); !s.ok()) {
    LOG(ERROR) << "NcclCommunicator::~NcclCommunicator: " << s;
  }
}

absl::Status NcclCommunicator::Abort() {
  // By setting the cancellation token all pending collectives scheduled on
  // executor_ will cancel. This will allow the aborting lambda below to run.
  cancel_->Cancel();

  return ExecuteAwait([this]() -> absl::Status {
    VLOG(1) << "Abort NCCL communicator: " << *this;
    if (aborted_) {
      return FailedPrecondition("NcclCommunicator already aborted");
    }
    aborted_ = true;
    // Note that we intentionally don't call PollUntilDone. Once comm_
    // has been aborted, we can no longer safely touch it.
    absl::MutexLock lock(comm_->mutex);
    return XLA_NCCL_STATUS(ncclCommAbort(comm_->comm));
  });
}

absl::Status NcclCommunicator::HealthCheck() const {
  return ExecuteAwait([this]() -> absl::Status {
    VLOG(5) << "Get last async error for NCCL communicator: " << *this;
    if (cancel_->IsCancelled()) {
      return FailedPrecondition("NcclCommunicator aborted");
    }

    ncclResult_t async_err;
    absl::MutexLock lock(comm_->mutex);
    XLA_NCCL_RETURN_IF_ERROR(ncclCommGetAsyncError(comm_->comm, &async_err));
    if (async_err == ncclSuccess) {
      return absl::OkStatus();
    }

    return Internal("%s. Last NCCL error (maybe unrelated): %s",
                    ncclGetLastError(comm_->comm),
                    ncclGetErrorString(async_err));
  });
}

absl::StatusOr<size_t> NcclCommunicator::NumRanks() const {
  return ExecuteAwait<size_t>([this]() -> absl::StatusOr<size_t> {
    VLOG(5) << "Get the number of ranks in NCCL communicator: " << *this;
    if (cancel_->IsCancelled()) {
      return FailedPrecondition("NcclCommunicator aborted");
    }

    // We intentionally don't call PollUntilDone. ncclCommCount is
    // blocking.
    int32_t count = 0;
    absl::MutexLock lock(comm_->mutex);
    XLA_NCCL_RETURN_IF_ERROR(ncclCommCount(comm_->comm, &count));
    return count;
  });
}

Future<> NcclCommunicator::GroupExecute(
    absl::AnyInvocable<absl::Status() &&> group) {
  return Execute([group = std::move(group), this]() mutable {
    return GroupLaunch([&] { return std::move(group)(); });
  });
}

absl::Status NcclCommunicator::GroupLaunch(
    absl::FunctionRef<absl::Status()> group) {
  ASSIGN_OR_RETURN(bool launched, NcclGroupLaunch(group));
  if (launched) {
    return PollUntilDone();
  }
  return absl::OkStatus();
}

Future<> NcclCommunicator::AllReduce(se::DeviceAddressBase send_buffer,
                                     se::DeviceAddressBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     ReductionKind reduction_kind,
                                     const Communicator::Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, reduction_kind,
                  &executor, this]() -> absl::Status {
    return LaunchAllReduce(send_buffer, recv_buffer, dtype, count,
                           reduction_kind, executor);
  });
}

Future<> NcclCommunicator::Broadcast(se::DeviceAddressBase send_buffer,
                                     se::DeviceAddressBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     RankId root, const Executor& executor) {
  return Execute(
      [send_buffer, recv_buffer, dtype, count, root, &executor, this]() {
        return LaunchBroadcast(send_buffer, recv_buffer, dtype, count, root,
                               executor);
      });
}

Future<> NcclCommunicator::ReduceScatter(se::DeviceAddressBase send_buffer,
                                         se::DeviceAddressBase recv_buffer,
                                         PrimitiveType dtype, size_t count,
                                         ReductionKind reduction_kind,
                                         const Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, reduction_kind,
                  &executor, this]() {
    return LaunchReduceScatter(send_buffer, recv_buffer, dtype, count,
                               reduction_kind, executor);
  });
}

Future<> NcclCommunicator::AllGather(se::DeviceAddressBase send_buffer,
                                     se::DeviceAddressBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     const Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, &executor, this]() {
    return LaunchAllGather(send_buffer, recv_buffer, dtype, count, executor);
  });
}

Future<> NcclCommunicator::AllToAll(
    absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  return Execute([send_buffers, recv_buffers, dtype, count, &executor, this]() {
    return LaunchAllToAll(send_buffers, recv_buffers, dtype, count, executor);
  });
}

Future<> NcclCommunicator::CollectivePermute(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  std::vector<RankId> owned_target_ranks(target_ranks.begin(),
                                         target_ranks.end());
  return Execute([send_buffer, recv_buffer, dtype, count, source_rank,
                  owned_target_ranks = std::move(owned_target_ranks), &executor,
                  this]() {
    return LaunchCollectivePermute(send_buffer, recv_buffer, dtype, count,
                                   source_rank, owned_target_ranks, executor);
  });
}

Future<> NcclCommunicator::Send(se::DeviceAddressBase send_buffer,
                                PrimitiveType dtype, size_t count, RankId peer,
                                const Executor& executor) {
  return Execute([send_buffer, dtype, count, peer, &executor, this]() {
    return LaunchSend(send_buffer, dtype, count, peer, executor);
  });
}

Future<> NcclCommunicator::Recv(se::DeviceAddressBase recv_buffer,
                                PrimitiveType dtype, size_t count, RankId peer,
                                const Executor& executor) {
  return Execute([recv_buffer, dtype, count, peer, &executor, this]() {
    return LaunchRecv(recv_buffer, dtype, count, peer, executor);
  });
}

Future<> NcclCommunicator::Put(se::DeviceAddressBase send_buffer,
                               SymmetricMemory* recv_buffer, size_t offset,
                               size_t count, RankId peer,
                               const Executor& executor) {
  return Execute([send_buffer, recv_buffer, offset, count, peer, &executor,
                  this]() {
    return LaunchPut(send_buffer, recv_buffer, offset, count, peer, executor);
  });
}

Future<> NcclCommunicator::Signal(RankId peer, const SignalDesc& signal_desc,
                                  const Executor& executor) {
  return Execute([peer, &signal_desc, &executor, this]() {
    return LaunchSignal(peer, signal_desc, executor);
  });
}

Future<> NcclCommunicator::WaitSignal(RankId peer, int op_cnt,
                                      const SignalDesc& signal_desc,
                                      const Executor& executor) {
  return Execute([peer, op_cnt, &signal_desc, &executor, this]() {
    return LaunchWaitSignal(peer, op_cnt, signal_desc, executor);
  });
}

absl::Status NcclCommunicator::LaunchAllReduce(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Communicator::Executor& executor) {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("NcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  {
    absl::MutexLock lock(comm_->mutex);
    VLOG(3) << absl::StreamFormat(
        "[%d] Launch NCCL AllReduce operation; send_buffer=%p; "
        "recv_buffer=%p; dtype=%s; count=%d; reduction_kind=%v; comm=%p; "
        "stream=%p",
        stream->parent()->device_ordinal(), send_buffer.opaque(),
        recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
        count, reduction_kind, comm_->comm, stream);

    ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype,
                     ToNcclDataType(dtype, /*is_reduction_op=*/true,
                                    stream->parent()
                                        ->GetDeviceDescription()
                                        .cuda_compute_capability()));

    RETURN_IF_ERROR(XLA_NCCL_STATUS(ncclAllReduce(
        send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
        nccl_dtype, ToNcclReduction(reduction_kind), comm_->comm,
        AsCudaStream(stream))));
  }
  if (!IsInsideNcclGroupLaunch()) {
    RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status NcclCommunicator::LaunchBroadcast(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, RankId root, const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("NcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  {
    absl::MutexLock lock(comm_->mutex);
    VLOG(3) << absl::StreamFormat(
        "[%d] Launch NCCL Broadcast operation; send_buffer=%p; "
        "recv_buffer=%p; dtype=%s; count=%d; root=%d; comm=%p; "
        "stream=%p",
        stream->parent()->device_ordinal(), send_buffer.opaque(),
        recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
        count, root.value(), comm_->comm, stream);

    ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype,
                     ToNcclDataType(dtype, false,
                                    stream->parent()
                                        ->GetDeviceDescription()
                                        .cuda_compute_capability()));

    RETURN_IF_ERROR(XLA_NCCL_STATUS(ncclBroadcast(
        send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
        nccl_dtype, root.value(), comm_->comm, AsCudaStream(stream))));
  }
  if (!IsInsideNcclGroupLaunch()) {
    RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status NcclCommunicator::LaunchReduceScatter(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("NcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  {
    absl::MutexLock lock(comm_->mutex);
    VLOG(3) << absl::StreamFormat(
        "[%d] Launch NCCL ReduceScatter operation; send_buffer=%p; "
        "recv_buffer=%p; dtype=%s; count=%d; reduction_kind=%v; comm=%p; "
        "stream=%p",
        stream->parent()->device_ordinal(), send_buffer.opaque(),
        recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
        count, reduction_kind, comm_->comm, stream);

    ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype,
                     ToNcclDataType(dtype, /*is_reduction_op=*/true,
                                    stream->parent()
                                        ->GetDeviceDescription()
                                        .cuda_compute_capability()));

    RETURN_IF_ERROR(XLA_NCCL_STATUS(ncclReduceScatter(
        send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
        nccl_dtype, ToNcclReduction(reduction_kind), comm_->comm,
        AsCudaStream(stream))));
  }
  if (!IsInsideNcclGroupLaunch()) {
    RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status NcclCommunicator::LaunchAllGather(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("NcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  {
    absl::MutexLock lock(comm_->mutex);
    VLOG(3) << absl::StreamFormat(
        "[%d] Launch NCCL AllGather operation; send_buffer=%p; "
        "recv_buffer=%p; dtype=%s; count=%d; comm=%p; stream=%p",
        stream->parent()->device_ordinal(), send_buffer.opaque(),
        recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
        count, comm_->comm, stream);

    ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype,
                     ToNcclDataType(dtype, false,
                                    stream->parent()
                                        ->GetDeviceDescription()
                                        .cuda_compute_capability()));

    RETURN_IF_ERROR(XLA_NCCL_STATUS(ncclAllGather(
        send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
        nccl_dtype, comm_->comm, AsCudaStream(stream))));
  }
  if (!IsInsideNcclGroupLaunch()) {
    RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

// If all buffers are contiguous returns a device address range that covers
// all of them, otherwise returns an empty optional.
static std::optional<se::DeviceAddressBase> IsContiguous(
    absl::Span<const se::DeviceAddressBase> buffers) {
  if (buffers.empty()) {
    return std::nullopt;
  }

  if (buffers.size() == 1) {
    return buffers[0];
  }

  size_t total_size = buffers[0].size();
  for (size_t i = 1; i < buffers.size(); ++i) {
    se::DeviceAddress<uint8_t> a(buffers[i - 1]);
    se::DeviceAddress<uint8_t> b(buffers[i]);
    total_size += b.size();

    if (a.base() + a.size() != b.base()) {
      return std::nullopt;
    }
  }

  return se::DeviceAddressBase(buffers[0].opaque(), total_size);
}

absl::Status NcclCommunicator::LaunchAllToAll(
    absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("NcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  auto buffer_formatter = [](std::string* out, se::DeviceAddressBase buffer) {
    absl::StrAppendFormat(out, "%p", buffer.opaque());
  };

  auto send_contiguous = IsContiguous(send_buffers);
  auto recv_contiguous = IsContiguous(recv_buffers);

  {
    absl::MutexLock lock(comm_->mutex);
    VLOG(3) << absl::StreamFormat(
        "[%d] Launch NCCL AllToAll operation; send_buffers=[%s]; "
        "send_contiguous=%v; recv_buffers=[%s]; recv_contiguous=%v; dtype=%s; "
        "count=%d; comm=%p; stream=%p",
        stream->parent()->device_ordinal(),
        absl::StrJoin(send_buffers, ", ", buffer_formatter),
        send_contiguous.has_value(),
        absl::StrJoin(recv_buffers, ", ", buffer_formatter),
        recv_contiguous.has_value(),
        primitive_util::LowercasePrimitiveTypeName(dtype), count, comm_->comm,
        stream);
  }

  if (send_buffers.size() != recv_buffers.size()) {
    return InvalidArgument(
        "Number of send buffers must match number of recv buffers: %d != %d",
        send_buffers.size(), recv_buffers.size());
  }

  int32_t num_ranks;
  {
    absl::MutexLock lock(comm_->mutex);
    XLA_NCCL_RETURN_IF_ERROR(ncclCommCount(comm_->comm, &num_ranks));
  }

  if (send_buffers.size() != num_ranks) {
    return InvalidArgument(
        "Number of send buffers must match number of ranks: %d != %d",
        send_buffers.size(), num_ranks);
  }

  ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, false,
          stream->parent()->GetDeviceDescription().cuda_compute_capability()));

#if NCCL_VERSION_CODE >= 22800
  // If send and receive buffers are contiguous we can use all-to-all API from
  // NCCL directly without launching individual send/recv operations.
  if (send_contiguous && recv_contiguous) {
    {
      absl::MutexLock lock(comm_->mutex);
      XLA_NCCL_RETURN_IF_ERROR(
          ncclAlltoAll(send_contiguous->opaque(), recv_contiguous->opaque(),
                       ToNcclCount(dtype, count), nccl_dtype, comm_->comm,
                       AsCudaStream(stream)));
    }
    if (!IsInsideNcclGroupLaunch()) {
      RETURN_IF_ERROR(PollUntilDone());
    }
    return absl::OkStatus();
  }
#endif

  auto group = [&] {
    for (size_t i = 0; i < send_buffers.size(); ++i) {
      se::DeviceAddressBase send_buffer = send_buffers[i];
      se::DeviceAddressBase recv_buffer = recv_buffers[i];

      {
        absl::MutexLock lock(comm_->mutex);
        XLA_NCCL_RETURN_IF_ERROR(
            ncclSend(send_buffer.opaque(), ToNcclCount(dtype, count),
                     nccl_dtype, i, comm_->comm, AsCudaStream(stream)));
      }

      {
        absl::MutexLock lock(comm_->mutex);
        XLA_NCCL_RETURN_IF_ERROR(
            ncclRecv(recv_buffer.opaque(), ToNcclCount(dtype, count),
                     nccl_dtype, i, comm_->comm, AsCudaStream(stream)));
      }
    }
    return absl::OkStatus();
  };
  return GroupLaunch(group);
}

absl::Status NcclCommunicator::LaunchCollectivePermute(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("NcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  auto rank_formatter = [](std::string* out, RankId rank) {
    absl::StrAppendFormat(out, "%d", rank.value());
  };

  {
    absl::MutexLock lock(comm_->mutex);
    VLOG(3) << absl::StreamFormat(
        "[%d] Launch NCCL CollectivePermute operation; send_buffer=%p; "
        "recv_buffer=%p; dtype=%s; source_rank=%s; target_[ranks=%s]; "
        "count=%d; "
        "comm=%p; stream=%p",
        stream->parent()->device_ordinal(), send_buffer.opaque(),
        recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
        source_rank ? absl::StrCat(source_rank->value()) : "<empty>",
        absl::StrJoin(target_ranks, ", ", rank_formatter), count, comm_->comm,
        stream);
  }

  ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, false,
          stream->parent()->GetDeviceDescription().cuda_compute_capability()));

  // Short-circuit if there is no source or target rank.
  if (!source_rank && target_ranks.empty()) {
    return absl::OkStatus();
  }

  auto group = [&] {
    if (source_rank) {
      {
        absl::MutexLock lock(comm_->mutex);
        XLA_NCCL_RETURN_IF_ERROR(ncclRecv(
            recv_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype,
            source_rank->value(), comm_->comm, AsCudaStream(stream)));
      }
    }

    for (RankId target_rank : target_ranks) {
      {
        absl::MutexLock lock(comm_->mutex);
        XLA_NCCL_RETURN_IF_ERROR(ncclSend(
            send_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype,
            target_rank.value(), comm_->comm, AsCudaStream(stream)));
      }
    }

    return absl::OkStatus();
  };

  return GroupLaunch(group);
}

absl::Status NcclCommunicator::LaunchSend(se::DeviceAddressBase send_buffer,
                                          PrimitiveType dtype, size_t count,
                                          RankId peer,
                                          const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("NcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  {
    absl::MutexLock lock(comm_->mutex);

    VLOG(3) << absl::StreamFormat(
        "[%d] Launch NCCL Send operation; send_buffer=%p; dtype=%s; "
        "count=%d; peer=%d; comm=%p; stream=%p",
        stream->parent()->device_ordinal(), send_buffer.opaque(),
        primitive_util::LowercasePrimitiveTypeName(dtype), count, peer.value(),
        comm_->comm, stream);

    ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype,
                     ToNcclDataType(dtype, false,
                                    stream->parent()
                                        ->GetDeviceDescription()
                                        .cuda_compute_capability()));
    RETURN_IF_ERROR(XLA_NCCL_STATUS(
        ncclSend(send_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype,
                 peer.value(), comm_->comm, AsCudaStream(stream))));
  }
  if (!IsInsideNcclGroupLaunch()) {
    RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status NcclCommunicator::LaunchRecv(se::DeviceAddressBase recv_buffer,
                                          PrimitiveType dtype, size_t count,
                                          RankId peer,
                                          const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("NcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  {
    absl::MutexLock lock(comm_->mutex);
    VLOG(3) << absl::StreamFormat(
        "[%d] Launch NCCL Recv operation; recv_buffer=%p; dtype=%s; "
        "count=%d; peer=%d; comm=%p; stream=%p",
        stream->parent()->device_ordinal(), recv_buffer.opaque(),
        primitive_util::LowercasePrimitiveTypeName(dtype), count, peer.value(),
        comm_->comm, stream);

    ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype,
                     ToNcclDataType(dtype, false,
                                    stream->parent()
                                        ->GetDeviceDescription()
                                        .cuda_compute_capability()));
    RETURN_IF_ERROR(XLA_NCCL_STATUS(
        ncclRecv(recv_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype,
                 peer.value(), comm_->comm, AsCudaStream(stream))));
  }
  if (!IsInsideNcclGroupLaunch()) {
    RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status NcclCommunicator::LaunchPut(se::DeviceAddressBase send_buffer,
                                         SymmetricMemory* recv_buffer,
                                         size_t offset, size_t count,
                                         RankId peer,
                                         const Executor& executor) {
  if (!capabilities_.supports_one_sided_comm) {
    return capabilities_.GetOneSidedCommUnsupportedError("Put");
  }
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("NcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  auto& peer_win = absl::down_cast<NcclSymmetricMemory&>(*recv_buffer);

#if NCCL_VERSION_CODE >= 22900

  {
    absl::MutexLock lock(comm_->mutex);
    VLOG(3) << absl::StreamFormat(
        "[%d] Launch NCCL Put operation; send_buffer=%p; peer_win=%v; "
        "offset=%d; count=%d; peer=%d; comm=%p; stream=%p",
        stream->parent()->device_ordinal(), send_buffer.opaque(), peer_win,
        offset, count, peer.value(), comm_->comm, stream);
    XLA_NCCL_RETURN_IF_ERROR(ncclPutSignal(
        send_buffer.opaque(), count, ncclInt8, peer.value(), peer_win.win(),
        offset, 0, 0, 0, comm_->comm, AsCudaStream(stream)));
  }
#else
  return Unimplemented("Put requires NCCL >= 2.29.0");
#endif
  if (!IsInsideNcclGroupLaunch()) {
    RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status NcclCommunicator::LaunchSignal(RankId peer,
                                            const SignalDesc& signal_desc,
                                            const Executor& executor) {
  if (!capabilities_.supports_one_sided_comm) {
    return capabilities_.GetOneSidedCommUnsupportedError("Signal");
  }
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("NcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  const auto& nccl_desc = absl::down_cast<const GpuSignalDesc&>(signal_desc);

#if NCCL_VERSION_CODE >= 22900
  {
    absl::MutexLock lock(comm_->mutex);
    VLOG(3) << absl::StreamFormat(
        "[%d] Launch NCCL Signal operation; peer=%d; sig_idx=%d; ctx=%d; "
        "comm=%p; stream=%p",
        stream->parent()->device_ordinal(), peer.value(), nccl_desc.sig_idx(),
        nccl_desc.ctx(), comm_->comm, stream);
    XLA_NCCL_RETURN_IF_ERROR(ncclSignal(peer.value(), nccl_desc.sig_idx(),
                                        nccl_desc.ctx(), 0, comm_->comm,
                                        AsCudaStream(stream)));
  }
#else
  return Unimplemented("Signal requires NCCL >= 2.29.0");
#endif
  if (!IsInsideNcclGroupLaunch()) {
    RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status NcclCommunicator::LaunchWaitSignal(RankId peer, int op_cnt,
                                                const SignalDesc& signal_desc,
                                                const Executor& executor) {
  if (!capabilities_.supports_one_sided_comm) {
    return capabilities_.GetOneSidedCommUnsupportedError("WaitSignal");
  }
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("NcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  const auto& nccl_desc = absl::down_cast<const GpuSignalDesc&>(signal_desc);

#if NCCL_VERSION_CODE >= 22900
  ncclWaitSignalDesc_t desc;
  desc.peer = peer.value();
  desc.opCnt = op_cnt;
  desc.sigIdx = nccl_desc.sig_idx();
  desc.ctx = nccl_desc.ctx();

  {
    absl::MutexLock lock(comm_->mutex);
    VLOG(3) << absl::StreamFormat(
        "[%d] Launch NCCL WaitSignal operation; peer=%d; op_cnt=%d; "
        "sig_idx=%d; ctx=%d; comm=%p; stream=%p",
        stream->parent()->device_ordinal(), peer.value(), op_cnt,
        nccl_desc.sig_idx(), nccl_desc.ctx(), comm_->comm, stream);
    XLA_NCCL_RETURN_IF_ERROR(
        ncclWaitSignal(1, &desc, comm_->comm, AsCudaStream(stream)));
  }
#else
  return Unimplemented("WaitSignal requires NCCL >= 2.29.0");
#endif
  if (!IsInsideNcclGroupLaunch()) {
    RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

std::string NcclCommunicator::ToString() const {
  // comm_ should not be "touched" outside of executor_, but we are printing
  // the pointer itself and not touching the value, so this is safe.
  absl::MutexLock lock(comm_->mutex);
  return absl::StrFormat("NcclCommunicator(ncclComm_t=%p)", comm_->comm);
}

absl::Status NcclCommunicator::PollUntilDone() const {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("NcclCommunicator aborted");
  }
  absl::MutexLock lock(comm_->mutex);
  return ::xla::gpu::PollUntilDone(comm_->comm, *cancel_);
}

Future<> NcclCommunicator::Execute(
    absl::AnyInvocable<absl::Status() &&> f) const {
  return executor_ ? MakeFutureOn<void>(*executor_, std::move(f))
                   : Future<>(std::move(f)());
}

template <typename T>
Future<T> NcclCommunicator::Execute(
    absl::AnyInvocable<absl::StatusOr<T>() &&> f) const {
  return executor_ ? MakeFutureOn<T>(*executor_, std::move(f))
                   : Future<T>(std::move(f)());
}

//===----------------------------------------------------------------------===//
// NCCL device communicator
//===----------------------------------------------------------------------===//

#if NCCL_VERSION_CODE >= 22800

NcclDeviceCommunicator::NcclDeviceCommunicator(
    std::shared_ptr<NcclCommState> parent_comm,
    se::StreamExecutor* stream_executor,
    std::shared_ptr<tsl::Executor> executor, ncclDevComm dev_comm)
    : parent_comm_(parent_comm),
      stream_executor_(stream_executor),
      executor_(std::move(executor)),
      dev_comm_(dev_comm) {}

NcclDeviceCommunicator::~NcclDeviceCommunicator() {
  VLOG(3) << absl::StreamFormat("Destroy NCCL device comm %v", *this);

  auto destroy_fn = [this]() -> absl::Status {
    DCHECK(stream_executor_) << "StreamExecutor is unavailable";
    auto activation = stream_executor_->Activate();
    {
      absl::MutexLock lock(parent_comm_->mutex);
      return XLA_NCCL_STATUS(
          ncclDevCommDestroy(parent_comm_->comm, &dev_comm_));
    }
  };

  auto future = executor_
                    ? MakeFutureOn<void>(*executor_, std::move(destroy_fn))
                    : Future<>(std::move(destroy_fn)());
  absl::Status s = future.Await();
  if (!s.ok()) {
    LOG(ERROR) << "Failed to destroy device comm: " << s;
  }
}

absl::StatusOr<std::unique_ptr<NcclDeviceCommunicator>>
NcclDeviceCommunicator::CreateFrom(const NcclCommunicator& comm,
                                   const Requirements& requirements) {
  VLOG(3) << absl::StreamFormat(
      "Create NCCL device comm from %v: lsa_barrier_count=%d", comm,
      requirements.lsa_barrier_count);

  DCHECK(comm.stream_executor()) << "StreamExecutor is unavailable";
  auto activation = comm.stream_executor()->Activate();

  ncclDevCommRequirements reqs{};
  memset(&reqs, 0, sizeof(reqs));
#if NCCL_VERSION_CODE >= 22900
  reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
#endif
  reqs.lsaBarrierCount = requirements.lsa_barrier_count;

  std::shared_ptr<NcclCommState> comm_state = comm.comm_state();
  ncclDevComm dev_comm{};
  {
    absl::MutexLock lock(comm_state->mutex);
    RETURN_IF_ERROR(
        XLA_NCCL_STATUS(ncclDevCommCreate(comm_state->comm, &reqs, &dev_comm)));
  }

  return absl::WrapUnique(new NcclDeviceCommunicator(
      comm_state, comm.stream_executor(), comm.executor(), dev_comm));
}

PlatformCommunicatorHandle NcclDeviceCommunicator::platform_comm() const {
  return {const_cast<ncclDevComm*>(&dev_comm_)};
}

std::string NcclDeviceCommunicator::ToString() const {
  return absl::StrFormat("NcclDeviceCommunicator(ncclDevComm*=%p)", &dev_comm_);
}

se::PackedKernelArg NcclDeviceCommunicator::PackKernelArg() const {
  return se::PackedKernelArg(sizeof(ncclDevComm), [&](absl::Span<char> packed) {
    std::memcpy(packed.data(), &dev_comm_, sizeof(ncclDevComm));
  });
}

#endif  // NCCL_VERSION_CODE >= 22800

}  // namespace xla::gpu
