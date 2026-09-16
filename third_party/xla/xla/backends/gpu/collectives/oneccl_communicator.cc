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

#include "xla/backends/gpu/collectives/oneccl_communicator.h"

#include <sycl/sycl.hpp>

// The header above is included first to make the linter happy. At the same
// time, without this comment clang-format wants to put the header below.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "oneapi/ccl.h"
#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/oneccl_errors.h"
#include "xla/backends/gpu/collectives/single_threaded_executor.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/future.h"
#include "xla/primitive_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

se::Stream* ToStream(const Communicator::Executor& executor) {
  return absl::down_cast<const GpuCollectives::Executor&>(executor).stream();
}

static size_t ToOnecclCount(PrimitiveType dtype, size_t count) {
  return primitive_util::IsComplexType(dtype) ? count * 2 : count;
}

static absl::StatusOr<onecclDataType_t> ToCclDataType(PrimitiveType dtype,
                                                      bool is_reduction_op) {
  switch (dtype) {
    case S8:
    case F8E5M2:
    case F8E4M3FN:
    case F8E5M2FNUZ:
    case F8E4M3FNUZ:
    case F8E8M0FNU:
      return onecclInt8;
    case PRED:
    case U8:
      return onecclUint8;
    case S32:
      return onecclInt32;
    case U32:
      return onecclUint32;
    case S64:
      return onecclInt64;
    case U64:
      return onecclUint64;
    case F16:
      return onecclFloat16;
    case F32:
    case C64:
      return onecclFloat32;
    case F64:
    case C128:
      return onecclFloat64;
    case S16:
    case U16:
      // For reductions we expect 16 bit integer types to be promoted to 32-bit.
      if (is_reduction_op) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Unsupported data type for reduction operation: %s",
                            primitive_util::LowercasePrimitiveTypeName(dtype)));
      }
      // For collectives that just move data around, we can use onecclFloat16
      // for 16-bit integer data types.
      return onecclFloat16;
    case BF16:
      return onecclBfloat16;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported data type: %s",
                          primitive_util::LowercasePrimitiveTypeName(dtype)));
  }
}

static onecclRedOp_t ToCclReduction(ReductionKind kind) {
  switch (kind) {
    case ReductionKind::SUM:
      return onecclSum;
    case ReductionKind::PRODUCT:
      return onecclProd;
    case ReductionKind::MIN:
      return onecclMin;
    case ReductionKind::MAX:
      return onecclMax;
  }
}

}  // namespace

Future<> OnecclCommunicator::GroupExecute(
    absl::AnyInvocable<absl::Status() &&> f) {
  return Execute([f = std::move(f), this]() mutable -> absl::Status {
    // TODO(Intel-tf): Merging multiple send/recv operations into a group call
    // is hanging in oneCCL. Will enable this once the issue is fixed in oneCCL.
    return std::move(f)();
  });
}

absl::StatusOr<size_t> OnecclCommunicator::NumRanks() const {
  return ExecuteAwait<size_t>([this]() -> absl::StatusOr<size_t> {
    int32_t count = 0;
    XLA_ONECCL_RETURN_IF_ERROR(onecclCommCount(comm_, &count));
    return count;
  });
}

absl::Status OnecclCommunicator::GroupStart() {
  if (group_nesting_level_ == 0) {
    XLA_ONECCL_RETURN_IF_ERROR(onecclGroupStart());
  }
  ++group_nesting_level_;
  return absl::OkStatus();
}

absl::Status OnecclCommunicator::GroupEnd() {
  if (group_nesting_level_ <= 0) {
    return absl::FailedPreconditionError(
        "There was no corresponding onecclGroupStart() for this "
        "onecclGroupEnd()");
  }
  group_nesting_level_--;
  if (group_nesting_level_ == 0) {
    XLA_ONECCL_RETURN_IF_ERROR(onecclGroupEnd());
  }
  return absl::OkStatus();
}

Future<> OnecclCommunicator::AllReduce(se::DeviceAddressBase send_buffer,
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

Future<> OnecclCommunicator::Broadcast(se::DeviceAddressBase send_buffer,
                                       se::DeviceAddressBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       RankId root,
                                       const Communicator::Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, root, &executor,
                  this]() -> absl::Status {
    return LaunchBroadcast(send_buffer, recv_buffer, dtype, count, root,
                           executor);
  });
}

Future<> OnecclCommunicator::ReduceScatter(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Communicator::Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, reduction_kind,
                  &executor, this]() -> absl::Status {
    return LaunchReduceScatter(send_buffer, recv_buffer, dtype, count,
                               reduction_kind, executor);
  });
}

Future<> OnecclCommunicator::AllGather(se::DeviceAddressBase send_buffer,
                                       se::DeviceAddressBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       const Communicator::Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, &executor,
                  this]() -> absl::Status {
    return LaunchAllGather(send_buffer, recv_buffer, dtype, count, executor);
  });
}

Future<> OnecclCommunicator::AllToAll(
    absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Communicator::Executor& executor) {
  return Execute([send_buffers, recv_buffers, dtype, count, &executor,
                  this]() -> absl::Status {
    return LaunchAllToAll(send_buffers, recv_buffers, dtype, count, executor);
  });
}

Future<> OnecclCommunicator::Send(se::DeviceAddressBase send_buffer,
                                  PrimitiveType dtype, size_t count,
                                  RankId peer, const Executor& executor) {
  return Execute([send_buffer, dtype, count, peer, &executor, this]() {
    return LaunchSend(send_buffer, dtype, count, peer, executor);
  });
}

Future<> OnecclCommunicator::Recv(se::DeviceAddressBase recv_buffer,
                                  PrimitiveType dtype, size_t count,
                                  RankId peer, const Executor& executor) {
  return Execute([recv_buffer, dtype, count, peer, &executor, this]() {
    return LaunchRecv(recv_buffer, dtype, count, peer, executor);
  });
}

Future<> OnecclCommunicator::CollectivePermute(
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

absl::Status OnecclCommunicator::LaunchAllReduce(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Communicator::Executor& executor) {
  se::Stream* stream = ToStream(executor);
  void* stream_handle = stream->platform_specific_handle().stream;
  ::sycl::queue* sycl_queue = reinterpret_cast<::sycl::queue*>(stream_handle);
  ABSL_ASSIGN_OR_RETURN(onecclDataType_t ccl_dtype,
                   ToCclDataType(dtype, /*is_reduction_op=*/true));
  ABSL_RETURN_IF_ERROR(XLA_ONECCL_STATUS(onecclAllReduce(
      send_buffer.opaque(), recv_buffer.opaque(), ToOnecclCount(dtype, count),
      ccl_dtype, ToCclReduction(reduction_kind), comm_, sycl_queue)));
  return absl::OkStatus();
}

absl::Status OnecclCommunicator::LaunchBroadcast(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, RankId root,
    const Communicator::Executor& executor) {
  se::Stream* stream = ToStream(executor);
  void* stream_handle = stream->platform_specific_handle().stream;
  ::sycl::queue* sycl_queue = reinterpret_cast<::sycl::queue*>(stream_handle);
  ABSL_ASSIGN_OR_RETURN(onecclDataType_t ccl_dtype,
                   ToCclDataType(dtype, /*is_reduction_op=*/false));
  ABSL_RETURN_IF_ERROR(XLA_ONECCL_STATUS(onecclBroadcast(
      send_buffer.opaque(), recv_buffer.opaque(), ToOnecclCount(dtype, count),
      ccl_dtype, root.value(), comm_, sycl_queue)));
  return absl::OkStatus();
}

absl::Status OnecclCommunicator::LaunchReduceScatter(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Communicator::Executor& executor) {
  se::Stream* stream = ToStream(executor);
  void* stream_handle = stream->platform_specific_handle().stream;
  ::sycl::queue* sycl_queue = reinterpret_cast<::sycl::queue*>(stream_handle);
  ABSL_ASSIGN_OR_RETURN(onecclDataType_t ccl_dtype,
                   ToCclDataType(dtype, /*is_reduction_op=*/true));
  ABSL_RETURN_IF_ERROR(XLA_ONECCL_STATUS(onecclReduceScatter(
      send_buffer.opaque(), recv_buffer.opaque(), ToOnecclCount(dtype, count),
      ccl_dtype, ToCclReduction(reduction_kind), comm_, sycl_queue)));
  return absl::OkStatus();
}

absl::Status OnecclCommunicator::LaunchAllGather(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, const Communicator::Executor& executor) {
  se::Stream* stream = ToStream(executor);
  void* stream_handle = stream->platform_specific_handle().stream;
  ::sycl::queue* sycl_queue = reinterpret_cast<::sycl::queue*>(stream_handle);
  ABSL_ASSIGN_OR_RETURN(onecclDataType_t ccl_dtype,
                   ToCclDataType(dtype, /*is_reduction_op=*/false));
  ABSL_RETURN_IF_ERROR(XLA_ONECCL_STATUS(onecclAllGather(
      send_buffer.opaque(), recv_buffer.opaque(), ToOnecclCount(dtype, count),
      ccl_dtype, comm_, sycl_queue)));
  return absl::OkStatus();
}

absl::Status OnecclCommunicator::LaunchCollectivePermute(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  se::Stream* stream = ToStream(executor);
  void* stream_handle = stream->platform_specific_handle().stream;
  ::sycl::queue* sycl_queue = reinterpret_cast<::sycl::queue*>(stream_handle);

  ABSL_ASSIGN_OR_RETURN(onecclDataType_t ccl_dtype,
                   ToCclDataType(dtype, /*is_reduction_op=*/false));
  if (!source_rank && target_ranks.empty()) {
    return absl::OkStatus();
  }

  ABSL_RETURN_IF_ERROR(GroupStart());

  for (auto target_rank : target_ranks) {
    XLA_ONECCL_RETURN_IF_ERROR(
        onecclSend(send_buffer.opaque(), ToOnecclCount(dtype, count), ccl_dtype,
                   target_rank.value(), comm_, sycl_queue));
  }
  if (source_rank) {
    XLA_ONECCL_RETURN_IF_ERROR(
        onecclRecv(recv_buffer.opaque(), ToOnecclCount(dtype, count), ccl_dtype,
                   source_rank->value(), comm_, sycl_queue));
  }
  ABSL_RETURN_IF_ERROR(GroupEnd());
  return absl::OkStatus();
}

absl::Status OnecclCommunicator::LaunchSend(
    se::DeviceAddressBase send_buffer, PrimitiveType dtype, size_t count,
    RankId peer, const Communicator::Executor& executor) {
  se::Stream* stream = ToStream(executor);
  void* stream_handle = stream->platform_specific_handle().stream;
  ::sycl::queue* sycl_queue = reinterpret_cast<::sycl::queue*>(stream_handle);
  ABSL_ASSIGN_OR_RETURN(onecclDataType_t ccl_dtype,
                   ToCclDataType(dtype, /*is_reduction_op=*/false));
  ABSL_RETURN_IF_ERROR(XLA_ONECCL_STATUS(
      onecclSend(send_buffer.opaque(), ToOnecclCount(dtype, count), ccl_dtype,
                 peer.value(), comm_, sycl_queue)));
  return absl::OkStatus();
}

absl::Status OnecclCommunicator::LaunchRecv(
    se::DeviceAddressBase recv_buffer, PrimitiveType dtype, size_t count,
    RankId peer, const Communicator::Executor& executor) {
  se::Stream* stream = ToStream(executor);
  void* stream_handle = stream->platform_specific_handle().stream;
  ::sycl::queue* sycl_queue = reinterpret_cast<::sycl::queue*>(stream_handle);
  ABSL_ASSIGN_OR_RETURN(onecclDataType_t ccl_dtype,
                   ToCclDataType(dtype, /*is_reduction_op=*/false));
  ABSL_RETURN_IF_ERROR(XLA_ONECCL_STATUS(
      onecclRecv(recv_buffer.opaque(), ToOnecclCount(dtype, count), ccl_dtype,
                 peer.value(), comm_, sycl_queue)));
  return absl::OkStatus();
}

absl::Status OnecclCommunicator::LaunchAllToAll(
    absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Communicator::Executor& executor) {
  se::Stream* stream = ToStream(executor);
  void* stream_handle = stream->platform_specific_handle().stream;
  ::sycl::queue* sycl_queue = reinterpret_cast<::sycl::queue*>(stream_handle);

  if (send_buffers.size() != recv_buffers.size()) {
    return absl::InvalidArgumentError(
        "Number of send and receive buffers must be the same");
  }

  int32_t num_ranks;
  XLA_ONECCL_RETURN_IF_ERROR(onecclCommCount(comm_, &num_ranks));

  if (send_buffers.size() != num_ranks) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Number of send/recv buffers (%d) must match number of ranks (%d)",
        send_buffers.size(), num_ranks));
  }

  ABSL_ASSIGN_OR_RETURN(onecclDataType_t ccl_dtype,
                   ToCclDataType(dtype, /*is_reduction_op=*/false));

  ABSL_RETURN_IF_ERROR(GroupStart());
  for (size_t i = 0; i < send_buffers.size(); ++i) {
    se::DeviceAddressBase send_buffer = send_buffers[i];
    se::DeviceAddressBase recv_buffer = recv_buffers[i];
    XLA_ONECCL_RETURN_IF_ERROR(onecclSend(send_buffer.opaque(),
                                          ToOnecclCount(dtype, count),
                                          ccl_dtype, i, comm_, sycl_queue));
    XLA_ONECCL_RETURN_IF_ERROR(onecclRecv(recv_buffer.opaque(),
                                          ToOnecclCount(dtype, count),
                                          ccl_dtype, i, comm_, sycl_queue));
  }
  ABSL_RETURN_IF_ERROR(GroupEnd());
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<OnecclCommunicator>> OnecclCommunicator::Create(
    absl::AnyInvocable<absl::StatusOr<onecclComm_t>()> make_comm, bool is_async,
    tsl::Env& env) {
  auto f = [&make_comm]() -> absl::StatusOr<onecclComm_t> {
    ABSL_ASSIGN_OR_RETURN(onecclComm_t comm, make_comm());
    // There is no need for PollUntilDone here since oneccl comm creation is
    // blocking.
    return comm;
  };
  if (!is_async) {
    ABSL_ASSIGN_OR_RETURN(onecclComm_t comm, f());
    return absl::WrapUnique(new OnecclCommunicator(comm, nullptr));
  }
  auto executor = std::make_unique<SingleThreadedExecutor>(env);
  ABSL_ASSIGN_OR_RETURN(onecclComm_t comm, MakeFutureOn(*executor, f).Await());
  return absl::WrapUnique(new OnecclCommunicator(comm, std::move(executor)));
}

OnecclCommunicator::~OnecclCommunicator() {
  auto f = [this]() -> absl::Status {
    if (comm_ == nullptr) {
      VLOG(1) << "Skipping destruction; null comm_ " << *this;
      return absl::OkStatus();
    }
    VLOG(1) << "Destroy " << *this;
    return XLA_ONECCL_STATUS(onecclCommDestroy(comm_));
  };
  if (absl::Status s = Execute(f).Await(); !s.ok()) {
    LOG(ERROR) << "OnecclCommunicator::~OnecclCommunicator: " << s;
  }
}

std::string OnecclCommunicator::ToString() const {
  // comm_ should not be "touched" outside of executor_, but we are printing
  // the pointer itself and not touching the value, so this is safe.
  return absl::StrFormat("OnecclCommunicator(onecclComm_t=%p)", comm_);
}

Future<> OnecclCommunicator::Execute(
    absl::AnyInvocable<absl::Status() &&> f) const {
  return executor_ ? MakeFutureOn(*executor_, std::move(f))
                   : Future<>(std::move(f)());
}

template <typename T>
Future<T> OnecclCommunicator::Execute(
    absl::AnyInvocable<absl::StatusOr<T>() &&> f) const {
  return executor_ ? MakeFutureOn(*executor_, std::move(f))
                   : Future<T>(std::move(f)());
}
}  // namespace xla::gpu
