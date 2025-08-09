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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/collectives/nccl_errors.h"
#include "xla/backends/gpu/collectives/single_threaded_executor.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
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
namespace {

// Blocks until ref is ready and returns its value (or error).
template <typename T>
absl::StatusOr<T> BlockAndGet(tsl::AsyncValueRef<T> ref) {
  tsl::BlockUntilReady(ref);
  if (ref.IsError()) {
    return ref.GetError();
  }
  return std::move(std::move(ref).get());
}

// Blocks until ref is ready and returns its value (or error).
absl::Status BlockAndGet(tsl::AsyncValueRef<absl::Status> ref) {
  tsl::BlockUntilReady(ref);
  if (ref.IsError()) {
    return ref.GetError();
  }
  return ref.get();
}

// Blocks until ref is ready and returns absl::OkStatus() (or error).
absl::Status BlockAndGet(tsl::AsyncValueRef<NcclCommunicator::Event> ref) {
  tsl::BlockUntilReady(ref);
  if (ref.IsError()) {
    return ref.GetError();
  }
  return absl::OkStatus();
}

se::Stream* ToStream(const Communicator::Executor& executor) {
  return tsl::down_cast<const GpuCollectives::Executor&>(executor).stream();
}

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

}  // namespace

//==-----------------------------------------------------------------------===//
// NCCL Registered Buffer Handle
//==-----------------------------------------------------------------------===//

// An RAII handle for user buffers registered with an NCCL communicator.
class NcclCommunicator::NcclRegisteredBufferHandle
    : public Communicator::RegisteredBufferHandle {
 public:
  NcclRegisteredBufferHandle(NcclCommunicator& comm, void* handle,
                             tsl::AsyncValue::Executor* executor)
      : comm_(comm), handle_(handle), executor_() {}

  ~NcclRegisteredBufferHandle() override {
    if (auto status = Unregister(); !status.ok()) {
      LOG(ERROR) << status.message();
    }
  }

  absl::Status Unregister() final {
    VLOG(3) << absl::StreamFormat(
        "Deregister buffer for NCCL communicator; handle=%p; comm=%p", handle_,
        comm_.comm_);

#if (NCCL_VERSION_CODE >= 21901)
    auto f = [this]() -> absl::Status {
      if (comm_.canceling_.load()) {
        return FailedPrecondition("NcclCommunicator aborted");
      }
      XLA_NCCL_RETURN_IF_ERROR(ncclCommDeregister(comm_.comm(), handle_));
      return comm_.PollUntilDone();
    };
    if (!executor_) {
      return f();
    }
    return BlockAndGet(tsl::MakeAsyncValueRef(*executor_, f));
#else
    return Unimplemented("NCCL version does not support ncclCommDeregister");
#endif  // NCCL_VERSION_CODE >= 21901
  }

 private:
  NcclCommunicator& comm_;
  void* handle_;
  tsl::AsyncValue::Executor* executor_;
};

//==-----------------------------------------------------------------------===//
// NCCL Communicator
//==-----------------------------------------------------------------------===//

absl::StatusOr<std::unique_ptr<NcclCommunicator>> NcclCommunicator::Create(
    absl::AnyInvocable<absl::StatusOr<ncclComm_t>()> make_comm, bool is_async,
    tsl::Env& env) {
  // TODO(mwhittaker): There is currently no way to abort these operations.
  auto f = [&make_comm]() -> absl::StatusOr<ncclComm_t> {
    TF_ASSIGN_OR_RETURN(ncclComm_t comm, make_comm());
    TF_RETURN_IF_ERROR(::xla::gpu::PollUntilDone(comm, std::atomic_bool{}));
    return comm;
  };

  if (!is_async) {
    // If this NcclCommunicator is synchronous, construct ncclComm_t in the
    // calling thread.
    TF_ASSIGN_OR_RETURN(ncclComm_t comm, f());
    return absl::WrapUnique(new NcclCommunicator(comm, nullptr));
  }

  // If this NcclCommunicator is asynchronous, then all operations on the
  // underlying ncclComm_t, including its creation, must take place on the
  // single threaded executor.
  auto executor = std::make_unique<SingleThreadedExecutor>(env);
  TF_ASSIGN_OR_RETURN(ncclComm_t comm,
                      BlockAndGet(tsl::TryMakeAsyncValueRef(*executor, f)));
  return absl::WrapUnique(new NcclCommunicator(comm, std::move(executor)));
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

    // Note that we intentionally don't call PollUntilDone. Once comm_ has been
    // destroyed, we can no longer safely touch it.
    VLOG(1) << "Destroy " << *this;
    return XLA_NCCL_STATUS(ncclCommDestroy(comm_));
  };

  if (absl::Status s = BlockAndGet(Execute(f)); !s.ok()) {
    LOG(ERROR) << "NcclCommunicator::~NcclCommunicator: " << s;
  }
}

absl::Status NcclCommunicator::Abort() {
  // By setting canceling_ to true, all pending collectives scheduled on
  // executor_ will cancel. This will allow the aborting lambda below to run.
  canceling_.store(true);

  return BlockAndGet(Execute([this]() -> absl::Status {
    VLOG(1) << "Abort NCCL communicator: " << *this;
    if (aborted_) {
      return FailedPrecondition("NcclCommunicator already aborted");
    }
    aborted_ = true;
    // Note that we intentionally don't call PollUntilDone. Once comm_ has been
    // aborted, we can no longer safely touch it.
    return XLA_NCCL_STATUS(ncclCommAbort(comm_));
  }));
}

absl::Status NcclCommunicator::HealthCheck() const {
  return BlockAndGet(Execute([this]() -> absl::Status {
    VLOG(5) << "Get last async error for NCCL communicator: " << *this;
    if (canceling_.load()) {
      return absl::FailedPreconditionError("NcclCommunicator aborted");
    }

    ncclResult_t async_err;
    XLA_NCCL_RETURN_IF_ERROR(ncclCommGetAsyncError(comm_, &async_err));
    if (async_err == ncclSuccess) {
      return absl::OkStatus();
    }

    return Internal("%s. Last NCCL error (maybe unrelated): %s",
                    ncclGetLastError(comm_), ncclGetErrorString(async_err));
  }));
}

absl::StatusOr<size_t> NcclCommunicator::NumRanks() const {
  return BlockAndGet(Execute<size_t>([this]() -> absl::StatusOr<size_t> {
    VLOG(5) << "Get the number of ranks in NCCL communicator: " << *this;
    if (canceling_.load()) {
      return absl::FailedPreconditionError("NcclCommunicator aborted");
    }

    // We intentionally don't call PollUntilDone. ncclCommCount is blocking.
    int32_t count = 0;
    XLA_NCCL_RETURN_IF_ERROR(ncclCommCount(comm_, &count));
    return count;
  }));
}

absl::StatusOr<std::unique_ptr<Communicator::RegisteredBufferHandle>>
NcclCommunicator::RegisterBuffer(stream_executor::DeviceMemoryBase buffer) {
#if (NCCL_VERSION_CODE >= 21901)
  using Handle = std::unique_ptr<Communicator::RegisteredBufferHandle>;
  return BlockAndGet(
      Execute<Handle>([&buffer, this]() -> absl::StatusOr<Handle> {
        VLOG(3) << absl::StreamFormat(
            "Register buffer for NCCL communicator; buffer=%p; size=%d; "
            "comm=%p",
            buffer.opaque(), buffer.size(), comm_);
        if (canceling_.load()) {
          return absl::FailedPreconditionError("NcclCommunicator aborted");
        }
        void* handle = nullptr;
        XLA_NCCL_RETURN_IF_ERROR(
            ncclCommRegister(comm_, buffer.opaque(), buffer.size(), &handle));
        if (group_nesting_level_ == 0) {
          TF_RETURN_IF_ERROR(PollUntilDone());
        }
        return std::make_unique<NcclRegisteredBufferHandle>(*this, handle,
                                                            executor_.get());
      }));
#else
  return Unimplemented("NCCL version does not support ncclCommRegister");
#endif  // NCCL_VERSION_CODE >= 21901
}

tsl::AsyncValueRef<Communicator::Event> NcclCommunicator::GroupExecute(
    absl::AnyInvocable<absl::Status(GpuCommunicator*)> f) {
  return Execute([f = std::move(f), this]() mutable -> absl::Status {
    TF_RETURN_IF_ERROR(GroupStart());
    TF_RETURN_IF_ERROR(f(this));
    TF_RETURN_IF_ERROR(GroupEnd());
    return absl::OkStatus();
  });
}

tsl::AsyncValueRef<NcclCommunicator::Event> NcclCommunicator::AllReduce(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Communicator::Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, reduction_kind,
                  &executor, this]() -> absl::Status {
    return LaunchAllReduce(send_buffer, recv_buffer, dtype, count,
                           reduction_kind, executor);
  });
}

tsl::AsyncValueRef<NcclCommunicator::Event> NcclCommunicator::Broadcast(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, RankId root, const Executor& executor) {
  return Execute(
      [send_buffer, recv_buffer, dtype, count, root, &executor, this]() {
        return LaunchBroadcast(send_buffer, recv_buffer, dtype, count, root,
                               executor);
      });
}

tsl::AsyncValueRef<NcclCommunicator::Event> NcclCommunicator::ReduceScatter(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, reduction_kind,
                  &executor, this]() {
    return LaunchReduceScatter(send_buffer, recv_buffer, dtype, count,
                               reduction_kind, executor);
  });
}

tsl::AsyncValueRef<NcclCommunicator::Event> NcclCommunicator::AllGather(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, &executor, this]() {
    return LaunchAllGather(send_buffer, recv_buffer, dtype, count, executor);
  });
}

tsl::AsyncValueRef<NcclCommunicator::Event> NcclCommunicator::AllToAll(
    absl::InlinedVector<se::DeviceMemoryBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceMemoryBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  return Execute([send_buffers, recv_buffers, dtype, count, &executor, this]() {
    return LaunchAllToAll(send_buffers, recv_buffers, dtype, count, executor);
  });
}

tsl::AsyncValueRef<NcclCommunicator::Event> NcclCommunicator::CollectivePermute(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
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

tsl::AsyncValueRef<NcclCommunicator::Event> NcclCommunicator::Send(
    se::DeviceMemoryBase send_buffer, PrimitiveType dtype, size_t count,
    RankId peer, const Executor& executor) {
  return Execute([send_buffer, dtype, count, peer, &executor, this]() {
    return LaunchSend(send_buffer, dtype, count, peer, executor);
  });
}

tsl::AsyncValueRef<NcclCommunicator::Event> NcclCommunicator::Recv(
    se::DeviceMemoryBase recv_buffer, PrimitiveType dtype, size_t count,
    RankId peer, const Executor& executor) {
  return Execute([recv_buffer, dtype, count, peer, &executor, this]() {
    return LaunchRecv(recv_buffer, dtype, count, peer, executor);
  });
}

absl::Status NcclCommunicator::GroupStart() {
  VLOG(5) << "Start NCCL group";
  XLA_NCCL_RETURN_IF_ERROR(ncclGroupStart());
  group_nesting_level_++;
  return absl::OkStatus();
}

absl::Status NcclCommunicator::GroupEnd() {
  VLOG(5) << "End NCCL group";
  XLA_NCCL_RETURN_IF_ERROR(ncclGroupEnd());
  group_nesting_level_--;
  if (group_nesting_level_ > 0) {
    // Though NCCL allows groups to be nested, no operations are actually
    // performed until the outermost group ends. The inner calls to
    // GroupStart() and GroupEnd() are effectively noops.
    //
    // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html
    return absl::OkStatus();
  }
  // Wait for the communicator to finish.
  return PollUntilDone();
}

absl::Status NcclCommunicator::LaunchAllReduce(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Communicator::Executor& executor) {
  if (canceling_.load()) {
    return FailedPrecondition("NcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

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
  if (group_nesting_level_ == 0) {
    TF_RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status NcclCommunicator::LaunchBroadcast(se::DeviceMemoryBase send_buffer,
                                               se::DeviceMemoryBase recv_buffer,
                                               PrimitiveType dtype,
                                               size_t count, RankId root,
                                               const Executor& executor) {
  if (canceling_.load()) {
    return absl::FailedPreconditionError("NcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

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
  if (group_nesting_level_ == 0) {
    TF_RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status NcclCommunicator::LaunchReduceScatter(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Executor& executor) {
  if (canceling_.load()) {
    return absl::FailedPreconditionError("NcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

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
  if (group_nesting_level_ == 0) {
    TF_RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status NcclCommunicator::LaunchAllGather(se::DeviceMemoryBase send_buffer,
                                               se::DeviceMemoryBase recv_buffer,
                                               PrimitiveType dtype,
                                               size_t count,
                                               const Executor& executor) {
  if (canceling_.load()) {
    return absl::FailedPreconditionError("NcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

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
  if (group_nesting_level_ == 0) {
    TF_RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status NcclCommunicator::LaunchAllToAll(
    absl::InlinedVector<se::DeviceMemoryBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceMemoryBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  if (canceling_.load()) {
    return absl::FailedPreconditionError("NcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

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

  TF_RETURN_IF_ERROR(GroupStart());
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
  TF_RETURN_IF_ERROR(GroupEnd());
  return absl::OkStatus();
}

absl::Status NcclCommunicator::LaunchCollectivePermute(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  if (canceling_.load()) {
    return FailedPrecondition("NcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

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
    return absl::OkStatus();
  }

  TF_RETURN_IF_ERROR(GroupStart());

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

  TF_RETURN_IF_ERROR(GroupEnd());

  return absl::OkStatus();
}

absl::Status NcclCommunicator::LaunchSend(se::DeviceMemoryBase send_buffer,
                                          PrimitiveType dtype, size_t count,
                                          RankId peer,
                                          const Executor& executor) {
  if (canceling_.load()) {
    return absl::FailedPreconditionError("NcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

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
  if (group_nesting_level_ == 0) {
    TF_RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status NcclCommunicator::LaunchRecv(se::DeviceMemoryBase recv_buffer,
                                          PrimitiveType dtype, size_t count,
                                          RankId peer,
                                          const Executor& executor) {
  if (canceling_.load()) {
    return absl::FailedPreconditionError("NcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

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
  if (group_nesting_level_ == 0) {
    TF_RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

std::string NcclCommunicator::ToString() const {
  // comm_ should not be "touched" outside of executor_, but we are printing the
  // pointer itself and not touching the value, so this is safe.
  return absl::StrFormat("NcclCommunicator(ncclComm_t=%p)", comm_);
}

absl::Status NcclCommunicator::PollUntilDone() const {
  if (canceling_.load()) {
    return FailedPrecondition("NcclCommunicator aborted");
  }
  return ::xla::gpu::PollUntilDone(comm_, canceling_);
}

tsl::AsyncValueRef<NcclCommunicator::Event> NcclCommunicator::Execute(
    absl::AnyInvocable<absl::Status()> f) const {
  if (!executor_) {
    // Execute on the calling thread.
    TF_RETURN_IF_ERROR(std::move(f)());
    return OkEvent();
  }

  // Execute on executor_.
  return tsl::TryMakeAsyncValueRef(
      *executor_,
      [f = std::move(f)]() mutable -> absl::StatusOr<NcclCommunicator::Event> {
        TF_RETURN_IF_ERROR(std::move(f)());
        return NcclCommunicator::Event{};
      });
}

template <typename T>
tsl::AsyncValueRef<T> NcclCommunicator::Execute(
    absl::AnyInvocable<absl::StatusOr<T>()> f) const {
  if (!executor_) {
    // Execute on the calling thread.
    auto ref = tsl::MakeUnconstructedAsyncValueRef<T>();
    ref.emplace(std::move(f)());
    return ref;
  }

  // Execute on executor_.
  return tsl::TryMakeAsyncValueRef(*executor_, std::move(f));
}

}  // namespace xla::gpu
