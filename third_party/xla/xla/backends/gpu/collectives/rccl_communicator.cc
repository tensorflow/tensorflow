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

#include "xla/backends/gpu/collectives/rccl_communicator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/rccl/rccl.h"
#include "rocm/rocm_config.h"  // IWYU pragma: keep
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/collectives/rccl_errors.h"
#include "xla/backends/gpu/collectives/rccl_group.h"
#include "xla/backends/gpu/collectives/rccl_symmetric_memory.h"
#include "xla/backends/gpu/collectives/rccl_types.h"
#include "xla/backends/gpu/collectives/single_threaded_executor.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/future.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

hipStream_t AsHipStream(se::Stream* stream) {
  return absl::bit_cast<hipStream_t>(stream->platform_specific_handle().stream);
}

se::Stream* ToStream(const Communicator::Executor& executor) {
  return absl::down_cast<const GpuCollectives::Executor&>(executor).stream();
}

}  // namespace

//==-----------------------------------------------------------------------===//
// RCCL Communicator
//==-----------------------------------------------------------------------===//

absl::StatusOr<std::unique_ptr<RcclCommunicator>> RcclCommunicator::Create(
    absl::AnyInvocable<absl::StatusOr<ncclComm_t>()> make_comm,
    std::shared_ptr<CancellationToken> cancel, bool is_async, tsl::Env& env) {
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
    // If this RcclCommunicator is synchronous, construct ncclComm_t in the
    // calling thread.
    ASSIGN_OR_RETURN(ncclComm_t comm, f());
    return absl::WrapUnique(
        new RcclCommunicator(comm, nullptr, std::move(cancel)));
  }

  // If this RcclCommunicator is asynchronous, then all operations on the
  // underlying ncclComm_t, including its creation, must take place on the
  // single threaded executor.
  auto executor = std::make_unique<SingleThreadedExecutor>(env);
  ASSIGN_OR_RETURN(ncclComm_t comm,
                   MakeFutureOn<ncclComm_t>(*executor, f).Await());
  return absl::WrapUnique(
      new RcclCommunicator(comm, std::move(executor), std::move(cancel)));
}

RcclCommunicator::~RcclCommunicator() {
  auto f = [this]() -> absl::Status {
    if (comm_ == nullptr) {
      VLOG(1) << "Skipping destruction; null comm_ " << *this;
      return absl::OkStatus();
    }

    if (aborted_) {
      VLOG(1) << "Skipping destruction; already aborted " << *this;
      return absl::OkStatus();
    }

    // Note that we intentionally don't call PollUntilDone. Once comm_ has been
    // destroyed, we can no longer safely touch it.
    VLOG(1) << "Destroy " << *this;
    return XLA_RCCL_STATUS(ncclCommDestroy(comm_));
  };

  if (absl::Status s = Execute(f).Await(); !s.ok()) {
    LOG(ERROR) << "RcclCommunicator::~RcclCommunicator: " << s;
  }
}

absl::Status RcclCommunicator::Abort() {
  // By setting the cancellation token all pending collectives scheduled on
  // executor_ will cancel. This will allow the aborting lambda below to run.
  cancel_->Cancel();

  return ExecuteAwait([this]() -> absl::Status {
    VLOG(1) << "Abort RCCL communicator: " << *this;
    if (aborted_) {
      return FailedPrecondition("RcclCommunicator already aborted");
    }
    aborted_ = true;
    // Note that we intentionally don't call PollUntilDone. Once comm_
    // has been aborted, we can no longer safely touch it.
    return XLA_RCCL_STATUS(ncclCommAbort(comm_));
  });
}

absl::Status RcclCommunicator::HealthCheck() const {
  return ExecuteAwait([this]() -> absl::Status {
    VLOG(5) << "Get last async error for RCCL communicator: " << *this;
    if (cancel_->IsCancelled()) {
      return absl::FailedPreconditionError("RcclCommunicator aborted");
    }

    ncclResult_t async_err;
    XLA_RCCL_RETURN_IF_ERROR(ncclCommGetAsyncError(comm_, &async_err));
    if (async_err == ncclSuccess) {
      return absl::OkStatus();
    }

    return Internal("%s. Last RCCL error (maybe unrelated): %s",
                    ncclGetLastError(comm_), ncclGetErrorString(async_err));
  });
}

absl::StatusOr<size_t> RcclCommunicator::NumRanks() const {
  return ExecuteAwait<size_t>([this]() -> absl::StatusOr<size_t> {
    VLOG(5) << "Get the number of ranks in RCCL communicator: " << *this;
    if (cancel_->IsCancelled()) {
      return absl::FailedPreconditionError("RcclCommunicator aborted");
    }

    // We intentionally don't call PollUntilDone. ncclCommCount is
    // blocking.
    int32_t count = 0;
    XLA_RCCL_RETURN_IF_ERROR(ncclCommCount(comm_, &count));
    return count;
  });
}

Future<> RcclCommunicator::GroupExecute(
    absl::AnyInvocable<absl::Status() &&> group) {
  return Execute([group = std::move(group), this]() mutable {
    return GroupLaunch([&] { return std::move(group)(); });
  });
}

absl::Status RcclCommunicator::GroupLaunch(
    absl::FunctionRef<absl::Status()> group) {
  ASSIGN_OR_RETURN(bool launched, RcclGroupLaunch(group));
  if (launched) {
    return PollUntilDone();
  }
  return absl::OkStatus();
}

Future<> RcclCommunicator::AllReduce(se::DeviceAddressBase send_buffer,
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

Future<> RcclCommunicator::Broadcast(se::DeviceAddressBase send_buffer,
                                     se::DeviceAddressBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     RankId root, const Executor& executor) {
  return Execute(
      [send_buffer, recv_buffer, dtype, count, root, &executor, this]() {
        return LaunchBroadcast(send_buffer, recv_buffer, dtype, count, root,
                               executor);
      });
}

Future<> RcclCommunicator::ReduceScatter(se::DeviceAddressBase send_buffer,
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

Future<> RcclCommunicator::AllGather(se::DeviceAddressBase send_buffer,
                                     se::DeviceAddressBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     const Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, &executor, this]() {
    return LaunchAllGather(send_buffer, recv_buffer, dtype, count, executor);
  });
}

Future<> RcclCommunicator::AllToAll(
    absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  return Execute([send_buffers, recv_buffers, dtype, count, &executor, this]() {
    return LaunchAllToAll(send_buffers, recv_buffers, dtype, count, executor);
  });
}

Future<> RcclCommunicator::CollectivePermute(
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

Future<> RcclCommunicator::Send(se::DeviceAddressBase send_buffer,
                                PrimitiveType dtype, size_t count, RankId peer,
                                const Executor& executor) {
  return Execute([send_buffer, dtype, count, peer, &executor, this]() {
    return LaunchSend(send_buffer, dtype, count, peer, executor);
  });
}

Future<> RcclCommunicator::Recv(se::DeviceAddressBase recv_buffer,
                                PrimitiveType dtype, size_t count, RankId peer,
                                const Executor& executor) {
  return Execute([recv_buffer, dtype, count, peer, &executor, this]() {
    return LaunchRecv(recv_buffer, dtype, count, peer, executor);
  });
}

absl::Status RcclCommunicator::LaunchAllReduce(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Communicator::Executor& executor) {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("RcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch RCCL AllReduce operation; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; reduction_kind=%v; comm=%p; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, reduction_kind, comm_, stream);

  ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, /*is_reduction_op=*/true,
          stream->parent()->GetDeviceDescription().rocm_compute_capability()));

  RETURN_IF_ERROR(XLA_RCCL_STATUS(ncclAllReduce(
      send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
      nccl_dtype, ToNcclReduction(reduction_kind), comm_,
      AsHipStream(stream))));
  if (!IsInsideRcclGroupLaunch()) {
    RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status RcclCommunicator::LaunchBroadcast(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, RankId root, const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return absl::FailedPreconditionError("RcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch RCCL Broadcast operation; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; root=%d; comm=%p; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, root.value(), comm_, stream);

  ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, /*is_reduction_op=*/false,
          stream->parent()->GetDeviceDescription().rocm_compute_capability()));

  RETURN_IF_ERROR(XLA_RCCL_STATUS(ncclBroadcast(
      send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
      nccl_dtype, root.value(), comm_, AsHipStream(stream))));
  if (!IsInsideRcclGroupLaunch()) {
    RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status RcclCommunicator::LaunchReduceScatter(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return absl::FailedPreconditionError("RcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch RCCL ReduceScatter operation; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; reduction_kind=%v; comm=%p; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, reduction_kind, comm_, stream);

  ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, /*is_reduction_op=*/true,
          stream->parent()->GetDeviceDescription().rocm_compute_capability()));

  RETURN_IF_ERROR(XLA_RCCL_STATUS(ncclReduceScatter(
      send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
      nccl_dtype, ToNcclReduction(reduction_kind), comm_,
      AsHipStream(stream))));
  if (!IsInsideRcclGroupLaunch()) {
    RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status RcclCommunicator::LaunchAllGather(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return absl::FailedPreconditionError("RcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch RCCL AllGather operation; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, comm_, stream);

  ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, /*is_reduction_op=*/false,
          stream->parent()->GetDeviceDescription().rocm_compute_capability()));

  RETURN_IF_ERROR(XLA_RCCL_STATUS(ncclAllGather(
      send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
      nccl_dtype, comm_, AsHipStream(stream))));
  if (!IsInsideRcclGroupLaunch()) {
    RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status RcclCommunicator::LaunchAllToAll(
    absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return absl::FailedPreconditionError("RcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  auto buffer_formatter = [](std::string* out, se::DeviceAddressBase buffer) {
    absl::StrAppendFormat(out, "%p", buffer.opaque());
  };

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch RCCL AllToAll operation; send_buffers=[%s]; "
      "recv_buffers=[%s]; dtype=%s; count=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(),
      absl::StrJoin(send_buffers, ", ", buffer_formatter),
      absl::StrJoin(recv_buffers, ", ", buffer_formatter),
      primitive_util::LowercasePrimitiveTypeName(dtype), count, comm_, stream);

  if (send_buffers.size() != recv_buffers.size()) {
    return InvalidArgument(
        "Number of send buffers must match number of recv buffers: %d != %d",
        send_buffers.size(), recv_buffers.size());
  }

  int32_t num_ranks;
  XLA_RCCL_RETURN_IF_ERROR(ncclCommCount(comm_, &num_ranks));

  if (send_buffers.size() != num_ranks) {
    return InvalidArgument(
        "Number of send buffers must match number of ranks: %d != %d",
        send_buffers.size(), num_ranks);
  }

  ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, /*is_reduction_op=*/false,
          stream->parent()->GetDeviceDescription().rocm_compute_capability()));

  auto group = [&] {
    for (size_t i = 0; i < send_buffers.size(); ++i) {
      se::DeviceAddressBase send_buffer = send_buffers[i];
      se::DeviceAddressBase recv_buffer = recv_buffers[i];

      XLA_RCCL_RETURN_IF_ERROR(ncclSend(send_buffer.opaque(),
                                        ToNcclCount(dtype, count), nccl_dtype,
                                        i, comm_, AsHipStream(stream)));
      XLA_RCCL_RETURN_IF_ERROR(ncclRecv(recv_buffer.opaque(),
                                        ToNcclCount(dtype, count), nccl_dtype,
                                        i, comm_, AsHipStream(stream)));
    }
    return absl::OkStatus();
  };
  return GroupLaunch(group);
}

absl::Status RcclCommunicator::LaunchCollectivePermute(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("RcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  auto rank_formatter = [](std::string* out, RankId rank) {
    absl::StrAppendFormat(out, "%d", rank.value());
  };

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch RCCL CollectivePermute operation; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; source_rank=%s; target_[ranks=%s]; count=%d; "
      "comm=%p; stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      source_rank ? absl::StrCat(source_rank->value()) : "<empty>",
      absl::StrJoin(target_ranks, ", ", rank_formatter), count, comm_, stream);

  ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, /*is_reduction_op=*/false,
          stream->parent()->GetDeviceDescription().rocm_compute_capability()));

  // Short-circuit if there is no source or target rank.
  if (!source_rank && target_ranks.empty()) {
    return absl::OkStatus();
  }

  auto group = [&] {
    if (source_rank) {
      XLA_RCCL_RETURN_IF_ERROR(
          ncclRecv(recv_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype,
                   source_rank->value(), comm_, AsHipStream(stream)));
    }

    for (RankId target_rank : target_ranks) {
      XLA_RCCL_RETURN_IF_ERROR(
          ncclSend(send_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype,
                   target_rank.value(), comm_, AsHipStream(stream)));
    }

    return absl::OkStatus();
  };

  return GroupLaunch(group);
}

absl::Status RcclCommunicator::LaunchSend(se::DeviceAddressBase send_buffer,
                                          PrimitiveType dtype, size_t count,
                                          RankId peer,
                                          const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return absl::FailedPreconditionError("RcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch RCCL Send operation; send_buffer=%p; dtype=%s; "
      "count=%d; peer=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      primitive_util::LowercasePrimitiveTypeName(dtype), count, peer.value(),
      comm_, stream);

  ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, /*is_reduction_op=*/false,
          stream->parent()->GetDeviceDescription().rocm_compute_capability()));

  RETURN_IF_ERROR(XLA_RCCL_STATUS(
      ncclSend(send_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype,
               peer.value(), comm_, AsHipStream(stream))));
  if (!IsInsideRcclGroupLaunch()) {
    RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status RcclCommunicator::LaunchRecv(se::DeviceAddressBase recv_buffer,
                                          PrimitiveType dtype, size_t count,
                                          RankId peer,
                                          const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return absl::FailedPreconditionError("RcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch RCCL Recv operation; recv_buffer=%p; dtype=%s; "
      "count=%d; peer=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), recv_buffer.opaque(),
      primitive_util::LowercasePrimitiveTypeName(dtype), count, peer.value(),
      comm_, stream);

  ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, /*is_reduction_op=*/false,
          stream->parent()->GetDeviceDescription().rocm_compute_capability()));

  RETURN_IF_ERROR(XLA_RCCL_STATUS(
      ncclRecv(recv_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype,
               peer.value(), comm_, AsHipStream(stream))));
  if (!IsInsideRcclGroupLaunch()) {
    RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<SymmetricMemory>>
RcclCommunicator::CreateSymmetricMemory(se::DeviceAddressBase addr) {
  return RcclSymmetricMemory::Create(comm_, addr);
}

std::string RcclCommunicator::ToString() const {
  // comm_ should not be "touched" outside of executor_, but we are printing the
  // pointer itself and not touching the value, so this is safe.
  return absl::StrFormat("RcclCommunicator(ncclComm_t=%p)", comm_);
}

absl::Status RcclCommunicator::PollUntilDone() const {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("RcclCommunicator aborted");
  }
  return ::xla::gpu::PollUntilDone(comm_, *cancel_);
}

Future<> RcclCommunicator::Execute(
    absl::AnyInvocable<absl::Status() &&> f) const {
  return executor_ ? MakeFutureOn<void>(*executor_, std::move(f))
                   : Future<>(std::move(f)());
}

template <typename T>
Future<T> RcclCommunicator::Execute(
    absl::AnyInvocable<absl::StatusOr<T>() &&> f) const {
  return executor_ ? MakeFutureOn<T>(*executor_, std::move(f))
                   : Future<T>(std::move(f)());
}

}  // namespace xla::gpu
