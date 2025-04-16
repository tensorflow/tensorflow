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
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/nccl_errors.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"

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

//==-----------------------------------------------------------------------===//
// Conversions between XLA and NCCL data types
//==-----------------------------------------------------------------------===//

static size_t ToNcclCount(PrimitiveType dtype, size_t count) {
  return primitive_util::IsComplexType(dtype) ? count * 2 : count;
}

static absl::StatusOr<ncclDataType_t> ToNcclDataType(PrimitiveType dtype,
                                                     bool is_reduction_op) {
  switch (dtype) {
    case S8:
    case F8E5M2:
    case F8E4M3FN:
    case F8E5M2FNUZ:
    case F8E4M3FNUZ:
    case F8E8M0FNU:
      return ncclInt8;
    case PRED:
    case U8:
      return ncclUint8;
    case S32:
      return ncclInt32;
    case U32:
      return ncclUint32;
    case S64:
      return ncclInt64;
    case U64:
      return ncclUint64;
    case F16:
      return ncclFloat16;
    case F32:
    case C64:
      return ncclFloat32;
    case F64:
    case C128:
      return ncclFloat64;
    case S16:
    case U16:
      // For reductions we expect 16 bit integer types to be promoted to 32-bit.
      if (is_reduction_op) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Unsupported data type for reduction operation: %s",
                            primitive_util::LowercasePrimitiveTypeName(dtype)));
      }
      // For collectives that just move data around, we can use ncclFloat16 for
      // 16-bit integer data types.
      return ncclFloat16;
    case BF16:
      return ncclBfloat16;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported data type: %s",
                          primitive_util::LowercasePrimitiveTypeName(dtype)));
  }
}

static ncclRedOp_t ToNcclReduction(ReductionKind kind) {
  switch (kind) {
    case ReductionKind::SUM:
      return ncclSum;
    case ReductionKind::PRODUCT:
      return ncclProd;
    case ReductionKind::MIN:
      return ncclMin;
    case ReductionKind::MAX:
      return ncclMax;
  }
}

//==-----------------------------------------------------------------------===//
// NCCL Communicator
//==-----------------------------------------------------------------------===//

namespace {
// An RAII handle for user buffers registered with an NCCL communicator.
class NcclRegisteredBufferHandle : public Communicator::RegisteredBufferHandle {
 public:
  NcclRegisteredBufferHandle(ncclComm_t comm_, void* handle);
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
  if (!aborted_) {
    // Don't destroy the communicator if it has already been aborted.
    VLOG(1) << "Destroy " << *this;
    XLA_NCCL_LOG_IF_ERROR(ncclCommDestroy(comm_));
  } else {
    VLOG(1) << "Skipping destruction; already aborted " << *this;
  }
}

absl::Status NcclCommunicator::Abort() {
  VLOG(1) << "Abort NCCL communicator: " << ToString();
  if (aborted_) {
    return FailedPrecondition("NcclCommunicator aborted");
  }
  aborted_ = true;
  // TODO(mwhittaker): It is only safe to abort a non-blocking communicator.
  // Ensure that comm_ is non-blocking.
  return XLA_NCCL_STATUS(ncclCommAbort(comm_));
}

absl::Status NcclCommunicator::HealthCheck() const {
  VLOG(5) << "Get last async error for NCCL communicator: " << ToString();
  if (aborted_) {
    return absl::FailedPreconditionError("NcclCommunicator aborted");
  }

  ncclResult_t async_err;
  XLA_NCCL_RETURN_IF_ERROR(ncclCommGetAsyncError(comm_, &async_err));
  if (async_err == ncclSuccess) return absl::OkStatus();

  return Internal("%s. Last NCCL error (maybe unrelated): %s",
                  ncclGetLastError(comm_), ncclGetErrorString(async_err));
}

absl::StatusOr<size_t> NcclCommunicator::NumRanks() const {
  VLOG(5) << "Get the number of ranks in NCCL communicator: " << ToString();
  if (aborted_) {
    return absl::FailedPreconditionError("NcclCommunicator aborted");
  }
  int32_t count;
  XLA_NCCL_RETURN_IF_ERROR(ncclCommCount(comm_, &count));
  return count;
}

absl::StatusOr<std::unique_ptr<Communicator::RegisteredBufferHandle>>
NcclCommunicator::RegisterBuffer(stream_executor::DeviceMemoryBase buffer) {
  VLOG(3) << absl::StreamFormat(
      "Register buffer for NCCL communicator; buffer=%p; size=%d; comm=%p",
      buffer.opaque(), buffer.size(), comm_);
  if (aborted_) {
    return absl::FailedPreconditionError("NcclCommunicator aborted");
  }
#if (NCCL_VERSION_CODE >= 21901)
  void* handle = nullptr;
  XLA_NCCL_RETURN_IF_ERROR(
      ncclCommRegister(comm_, buffer.opaque(), buffer.size(), &handle));
  return std::make_unique<NcclRegisteredBufferHandle>(comm_, handle);
#else
  return Unimplemented("NCCL version does not support ncclCommRegister");
#endif  // NCCL_VERSION_CODE >= 21901
}

tsl::AsyncValueRef<NcclCommunicator::Event> NcclCommunicator::AllReduce(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Communicator::Executor& executor) {
  if (aborted_) {
    return absl::FailedPreconditionError("NcclCommunicator aborted");
  }
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  VLOG(3) << absl::StreamFormat(
      "Launch NCCL AllReduce operation on device #%d; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; reduction_kind=%s; comm=%p; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, ReductionKindToString(reduction_kind), comm_, stream);

  TF_ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype, ToNcclDataType(dtype, false));

  TF_RETURN_IF_ERROR(XLA_NCCL_STATUS(ncclAllReduce(
      send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
      nccl_dtype, ToNcclReduction(reduction_kind), comm_,
      se::gpu::AsGpuStreamValue(stream))));

  return OkEvent();
}

tsl::AsyncValueRef<NcclCommunicator::Event> NcclCommunicator::Broadcast(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, RankId root, const Executor& executor) {
  if (aborted_) {
    return absl::FailedPreconditionError("NcclCommunicator aborted");
  }
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  VLOG(3) << absl::StreamFormat(
      "Launch NCCL Broadcast operation on device #%d; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; root=%d; comm=%p; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, root.value(), comm_, stream);

  TF_ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype, ToNcclDataType(dtype, false));

  TF_RETURN_IF_ERROR(XLA_NCCL_STATUS(ncclBroadcast(
      send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
      nccl_dtype, root.value(), comm_, se::gpu::AsGpuStreamValue(stream))));

  return OkEvent();
}

tsl::AsyncValueRef<NcclCommunicator::Event> NcclCommunicator::ReduceScatter(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Executor& executor) {
  if (aborted_) {
    return absl::FailedPreconditionError("NcclCommunicator aborted");
  }
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  VLOG(3) << absl::StreamFormat(
      "Launch NCCL ReduceScatter operation on device #%d; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; reduction_kind=%s; comm=%p; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, ReductionKindToString(reduction_kind), comm_, stream);

  TF_ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype, ToNcclDataType(dtype, false));

  TF_RETURN_IF_ERROR(XLA_NCCL_STATUS(ncclReduceScatter(
      send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
      nccl_dtype, ToNcclReduction(reduction_kind), comm_,
      se::gpu::AsGpuStreamValue(stream))));

  return OkEvent();
}

tsl::AsyncValueRef<NcclCommunicator::Event> NcclCommunicator::AllGather(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  if (aborted_) {
    return absl::FailedPreconditionError("NcclCommunicator aborted");
  }
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  VLOG(3) << absl::StreamFormat(
      "Launch NCCL AllGather operation on device #%d; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, comm_, stream);

  TF_ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype, ToNcclDataType(dtype, false));

  TF_RETURN_IF_ERROR(XLA_NCCL_STATUS(ncclAllGather(
      send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
      nccl_dtype, comm_, se::gpu::AsGpuStreamValue(stream))));

  return OkEvent();
}

tsl::AsyncValueRef<NcclCommunicator::Event> NcclCommunicator::AllToAll(
    absl::Span<const se::DeviceMemoryBase> send_buffers,
    absl::Span<const se::DeviceMemoryBase> recv_buffers, PrimitiveType dtype,
    size_t count, const Executor& executor) {
  if (aborted_) {
    return absl::FailedPreconditionError("NcclCommunicator aborted");
  }
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  auto buffer_formatter = [](std::string* out, se::DeviceMemoryBase buffer) {
    absl::StrAppendFormat(out, "%p", buffer.opaque());
  };

  VLOG(3) << absl::StreamFormat(
      "Launch NCCL AllToAll operation on device #%d; send_buffers=[%s]; "
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
  XLA_NCCL_RETURN_IF_ERROR(ncclCommCount(comm_, &num_ranks));

  if (send_buffers.size() != num_ranks) {
    return InvalidArgument(
        "Number of send buffers must match number of ranks: %d != %d",
        send_buffers.size(), num_ranks);
  }

  TF_ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype, ToNcclDataType(dtype, false));

  XLA_NCCL_RETURN_IF_ERROR(ncclGroupStart());

  for (size_t i = 0; i < send_buffers.size(); ++i) {
    se::DeviceMemoryBase send_buffer = send_buffers[i];
    se::DeviceMemoryBase recv_buffer = recv_buffers[i];

    XLA_NCCL_RETURN_IF_ERROR(
        ncclSend(send_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype, i,
                 comm_, se::gpu::AsGpuStreamValue(stream)));

    XLA_NCCL_RETURN_IF_ERROR(
        ncclRecv(recv_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype, i,
                 comm_, se::gpu::AsGpuStreamValue(stream)));
  }

  XLA_NCCL_RETURN_IF_ERROR(ncclGroupEnd());

  return OkEvent();
}

tsl::AsyncValueRef<NcclCommunicator::Event> NcclCommunicator::CollectivePermute(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  if (aborted_) {
    return absl::FailedPreconditionError("NcclCommunicator aborted");
  }
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  auto rank_formatter = [](std::string* out, RankId rank) {
    absl::StrAppendFormat(out, "%d", rank.value());
  };

  VLOG(3) << absl::StreamFormat(
      "Launch NCCL CollectivePermute operation on device #%d; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; source_rank=%s; target_ranks=[%s]; count=%d; "
      "comm=%p; stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      source_rank ? absl::StrCat(source_rank->value()) : "<empty>",
      absl::StrJoin(target_ranks, ", ", rank_formatter), count, comm_, stream);

  TF_ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype, ToNcclDataType(dtype, false));

  // Short-circuit if there is no source or target rank.
  if (!source_rank && target_ranks.empty()) {
    return OkEvent();
  }

  XLA_NCCL_RETURN_IF_ERROR(ncclGroupStart());

  if (source_rank) {
    XLA_NCCL_RETURN_IF_ERROR(ncclRecv(
        recv_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype,
        source_rank->value(), comm_, se::gpu::AsGpuStreamValue(stream)));
  }

  for (auto target_rank : target_ranks) {
    XLA_NCCL_RETURN_IF_ERROR(ncclSend(
        send_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype,
        target_rank.value(), comm_, se::gpu::AsGpuStreamValue(stream)));
  }

  XLA_NCCL_RETURN_IF_ERROR(ncclGroupEnd());

  return OkEvent();
}

tsl::AsyncValueRef<NcclCommunicator::Event> NcclCommunicator::Send(
    se::DeviceMemoryBase send_buffer, PrimitiveType dtype, size_t count,
    RankId peer, const Executor& executor) {
  if (aborted_) {
    return absl::FailedPreconditionError("NcclCommunicator aborted");
  }
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  VLOG(3) << absl::StreamFormat(
      "Launch NCCL Send operation on device #%d; send_buffer=%p; dtype=%s; "
      "count=%d; peer=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      primitive_util::LowercasePrimitiveTypeName(dtype), count, peer.value(),
      comm_, stream);

  TF_ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype, ToNcclDataType(dtype, false));

  TF_RETURN_IF_ERROR(XLA_NCCL_STATUS(
      ncclSend(send_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype,
               peer.value(), comm_, se::gpu::AsGpuStreamValue(stream))));

  return OkEvent();
}

tsl::AsyncValueRef<NcclCommunicator::Event> NcclCommunicator::Recv(
    se::DeviceMemoryBase recv_buffer, PrimitiveType dtype, size_t count,
    RankId peer, const Executor& executor) {
  if (aborted_) {
    return absl::FailedPreconditionError("NcclCommunicator aborted");
  }
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  VLOG(3) << absl::StreamFormat(
      "Launch NCCL Recv operation on device #%d; recv_buffer=%p; dtype=%s; "
      "count=%d; peer=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), recv_buffer.opaque(),
      primitive_util::LowercasePrimitiveTypeName(dtype), count, peer.value(),
      comm_, stream);

  TF_ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype, ToNcclDataType(dtype, false));

  TF_RETURN_IF_ERROR(XLA_NCCL_STATUS(
      ncclRecv(recv_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype,
               peer.value(), comm_, se::gpu::AsGpuStreamValue(stream))));

  return OkEvent();
}

std::string NcclCommunicator::ToString() const {
  return absl::StrFormat("NccCommunicator(ncclComm_t=%p)", comm_);
}

absl::StatusOr<se::Stream*> NcclCommunicator::ToStream(
    const Executor& executor) {
  if (auto* gpu_executor =
          tsl::down_cast<const GpuCollectives::Executor*>(&executor)) {
    return gpu_executor->stream();
  }
  return InvalidArgument("Communicator executor is not a GPU executor");
}

}  // namespace xla::gpu
