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
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/collectives/nccl_errors.h"
#include "xla/backends/gpu/collectives/nccl_symmetric_memory.h"
#include "xla/backends/gpu/collectives/single_threaded_executor.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/future.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
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
#include "tsl/platform/casts.h"

// Include NCCL after XLA headers.
#include "third_party/nccl/nccl.h"

#if NCCL_VERSION_CODE >= 22800
// Device initiated collective operations were added in NCCL 2.28.0.
#include "third_party/nccl/nccl_device.h"
#endif  // NCCL_VERSION_CODE >= 22800

namespace xla::gpu {
namespace {

CUstream AsCudaStream(se::Stream* stream) {
  return absl::bit_cast<CUstream>(stream->platform_specific_handle().stream);
}

se::Stream* ToStream(const Communicator::Executor& executor) {
  return absl::down_cast<const GpuCollectives::Executor&>(executor).stream();
}

//==-----------------------------------------------------------------------===//
// Conversions between XLA and NCCL data types
//==-----------------------------------------------------------------------===//

static size_t ToNcclCount(PrimitiveType dtype, size_t count) {
  return primitive_util::IsComplexType(dtype) ? count * 2 : count;
}

static absl::StatusOr<ncclDataType_t> ToNcclDataType(
    PrimitiveType dtype, bool is_reduction_op, se::CudaComputeCapability cc) {
  switch (dtype) {
    case S8:
    case F8E5M2FNUZ:
    case F8E4M3FNUZ:
    case F8E8M0FNU:
      return ncclInt8;
    // For pre-Hopper FP8 reductions, let NCCL throw appropriate errors.
    case F8E5M2:
      return (cc.IsAtLeastHopper() || is_reduction_op) ? ncclFloat8e5m2
                                                       : ncclInt8;
    case F8E4M3FN:
      return (cc.IsAtLeastHopper() || is_reduction_op) ? ncclFloat8e4m3
                                                       : ncclInt8;
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
      // For reductions we expect 16 bit integer types to be promoted to
      // 32-bit.
      if (is_reduction_op) {
        return InvalidArgument(
            "Unsupported data type for reduction operation: %s",
            primitive_util::LowercasePrimitiveTypeName(dtype));
      }
      // For collectives that just move data around, we can use ncclFloat16
      // for 16-bit integer data types.
      return ncclFloat16;
    case BF16:
      return ncclBfloat16;
    default:
      return InvalidArgument("Unsupported data type: %s",
                             primitive_util::LowercasePrimitiveTypeName(dtype));
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
                             tsl::Executor* executor, bool symmetric_handle,
                             int device_ordinal)
      : comm_(comm),
        handle_(handle),
        executor_(),
        symmetric_handle_(symmetric_handle),
        device_ordinal_(device_ordinal) {}

  ~NcclRegisteredBufferHandle() override {
    if (auto status = Unregister(); !status.ok()) {
      LOG(ERROR) << status.message();
    }
  }

  absl::Status Unregister() final {
    VLOG(3) << absl::StreamFormat(
        "[%d] Deregister buffer for NCCL communicator; handle=%p; comm=%p",
        device_ordinal_, handle_, comm_.comm_);
    if (!symmetric_handle_) {
#if (NCCL_VERSION_CODE >= 21901)
      auto f = [this]() -> absl::Status {
        if (comm_.cancel_->IsCancelled()) {
          return FailedPrecondition("[%d] NcclCommunicator aborted",
                                    device_ordinal_);
        }
        XLA_NCCL_RETURN_IF_ERROR(ncclCommDeregister(comm_.comm(), handle_));
        return comm_.PollUntilDone();
      };
      return executor_ ? MakeFutureOn<void>(*executor_, f).Await() : f();
#else
      return Unimplemented(
          "[%d] NCCL version does not support ncclCommDeregister",
          device_ordinal_);
#endif  // NCCL_VERSION_CODE >= 21901
    } else {
      VLOG(3) << absl::StreamFormat(
          "[%d] Deregister symmetric buffer for NCCL communicator; "
          "handle=%p; "
          "comm=%p",
          device_ordinal_, handle_, comm_.comm_);
#if (NCCL_VERSION_CODE >= 22700)
      auto f = [this]() -> absl::Status {
        if (comm_.cancel_->IsCancelled()) {
          return FailedPrecondition("[%d] NcclCommunicator aborted",
                                    device_ordinal_);
        }
        XLA_NCCL_RETURN_IF_ERROR(
            ncclCommWindowDeregister(comm_.comm(), *(ncclWindow_t*)(handle_)));
        return comm_.PollUntilDone();
      };
      return executor_ ? MakeFutureOn<void>(*executor_, f).Await() : f();
#else
      return Unimplemented(
          "[%d] NCCL version does not support ncclCommWindowDeregister",
          device_ordinal_);
#endif  // NCCL_VERSION_CODE >= 22700
    }
  }

 private:
  NcclCommunicator& comm_;
  void* handle_;
  tsl::Executor* executor_;
  bool symmetric_handle_;
  int device_ordinal_;
};

//==-----------------------------------------------------------------------===//
// NCCL Device Communicator
//==-----------------------------------------------------------------------===//

bool NcclCommunicator::SupportsDeviceComm() const {
#if NCCL_VERSION_CODE >= 22800
  return true;
#else
  return false;
#endif  // NCCL_VERSION_CODE >= 22800
}

absl::StatusOr<std::unique_ptr<GpuDeviceCommunicator>>
NcclCommunicator::CreateDeviceComm(
    const GpuDeviceCommunicator::Requirements& requirements) {
#if NCCL_VERSION_CODE >= 22800
  return NcclDeviceCommunicator::CreateFrom(*this, requirements);
#else
  return Unimplemented(
      "NCCL version %d does not support collective communication",
      NCCL_VERSION_CODE);
#endif  // NCCL_VERSION_CODE >= 22800
}

//==-----------------------------------------------------------------------===//
// NCCL Symmetric Memory
//==-----------------------------------------------------------------------===//

absl::StatusOr<std::unique_ptr<SymmetricMemory>>
NcclCommunicator::CreateSymmetricMemory(se::DeviceAddressBase addr) {
  return NcclSymmetricMemory::Create(comm_, addr);
}

//==-----------------------------------------------------------------------===//
// NCCL Communicator
//==-----------------------------------------------------------------------===//

absl::StatusOr<std::unique_ptr<NcclCommunicator>> NcclCommunicator::Create(
    se::StreamExecutor* stream_executor,
    absl::AnyInvocable<absl::StatusOr<ncclComm_t>()> make_comm,
    std::shared_ptr<CancellationToken> cancel, bool is_async, tsl::Env& env) {
  auto f = [cancel, &make_comm]() -> absl::StatusOr<ncclComm_t> {
    TF_ASSIGN_OR_RETURN(ncclComm_t comm, make_comm());
    if (cancel) {
      TF_RETURN_IF_ERROR(::xla::gpu::PollUntilDone(comm, *cancel));
    } else {
      CancellationToken never_cancelled;
      TF_RETURN_IF_ERROR(::xla::gpu::PollUntilDone(comm, never_cancelled));
    }
    return comm;
  };

  if (!is_async) {
    // If this NcclCommunicator is synchronous, construct ncclComm_t in the
    // calling thread.
    TF_ASSIGN_OR_RETURN(ncclComm_t comm, f());
    return absl::WrapUnique(new NcclCommunicator(stream_executor, comm, nullptr,
                                                 std::move(cancel)));
  }

  // If this NcclCommunicator is asynchronous, then all operations on the
  // underlying ncclComm_t, including its creation, must take place on the
  // single threaded executor.
  auto executor = std::make_unique<SingleThreadedExecutor>(env);
  TF_ASSIGN_OR_RETURN(ncclComm_t comm,
                      MakeFutureOn<ncclComm_t>(*executor, f).Await());
  return absl::WrapUnique(new NcclCommunicator(
      stream_executor, comm, std::move(executor), std::move(cancel)));
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
    VLOG(1) << "Destroy " << *this;
    return XLA_NCCL_STATUS(ncclCommDestroy(comm_));
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
    return XLA_NCCL_STATUS(ncclCommAbort(comm_));
  });
}

absl::Status NcclCommunicator::HealthCheck() const {
  return ExecuteAwait([this]() -> absl::Status {
    VLOG(5) << "Get last async error for NCCL communicator: " << *this;
    if (cancel_->IsCancelled()) {
      return FailedPrecondition("NcclCommunicator aborted");
    }

    ncclResult_t async_err;
    XLA_NCCL_RETURN_IF_ERROR(ncclCommGetAsyncError(comm_, &async_err));
    if (async_err == ncclSuccess) {
      return absl::OkStatus();
    }

    return Internal("%s. Last NCCL error (maybe unrelated): %s",
                    ncclGetLastError(comm_), ncclGetErrorString(async_err));
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
    XLA_NCCL_RETURN_IF_ERROR(ncclCommCount(comm_, &count));
    return count;
  });
}

absl::Status NcclCommunicator::RegisterBufferOnce(
    se::DeviceAddressBase buffer_range, int device_ordinal,
    bool use_symmetric_buffer) {
  bool need_reg = false;
  {
    absl::MutexLock lock(registered_buffers_.mu);
    if (!registered_buffers_.range_to_handle.contains(buffer_range.opaque())) {
      need_reg = true;
    } else {
      XLA_VLOG_DEVICE(5, device_ordinal)
          << "Buffer range: " << buffer_range.opaque()
          << " with size: " << buffer_range.size() << " is already registered.";
    }
  }
  if (need_reg) {
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "Registering " << buffer_range.opaque()
        << " with size: " << buffer_range.size()
        << ", is symmetric: " << (use_symmetric_buffer ? "true" : "false");
    // Symmetric buffer registration is a collective operation,
    // we need to do that before locking on a global.
    TF_ASSIGN_OR_RETURN(
        auto handle,
        RegisterBuffer(buffer_range, device_ordinal, use_symmetric_buffer));
    absl::MutexLock lock(registered_buffers_.mu);
    registered_buffers_.range_to_handle[buffer_range.opaque()] =
        std::move(handle);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Communicator::RegisteredBufferHandle>>
NcclCommunicator::RegisterBuffer(stream_executor::DeviceAddressBase buffer,
                                 int device_ordinal,
                                 bool use_symmetric_buffer) {
#if (NCCL_VERSION_CODE >= 21901)
  using Handle = std::unique_ptr<Communicator::RegisteredBufferHandle>;

  if (!use_symmetric_buffer) {
    return ExecuteAwait<Handle>(
        [&buffer, device_ordinal, this]() -> absl::StatusOr<Handle> {
          VLOG(3) << absl::StreamFormat(
              "[%d] Register buffer for NCCL communicator; buffer=%p; size=%d; "
              "comm=%p",
              device_ordinal, buffer.opaque(), buffer.size(), comm_);
          if (cancel_->IsCancelled()) {
            return FailedPrecondition("NcclCommunicator aborted");
          }
          void* handle = nullptr;
          XLA_NCCL_RETURN_IF_ERROR(
              ncclCommRegister(comm_, buffer.opaque(), buffer.size(), &handle));
          if (group_nesting_level_ == 0) {
            TF_RETURN_IF_ERROR(PollUntilDone());
          }
          return std::make_unique<NcclRegisteredBufferHandle>(
              *this, handle, executor_.get(), /*symmetric_buffer= */ false,
              device_ordinal);
        });
#else
  return Unimplemented("[%d] NCCL version does not support ncclCommRegister",
                       device_ordinal);
#endif  // NCCL_VERSION_CODE >= 21901
  } else {
#if (NCCL_VERSION_CODE >= 22700)
    return ExecuteAwait<Handle>(
        [&buffer, device_ordinal, this]() -> absl::StatusOr<Handle> {
          VLOG(3) << absl::StreamFormat(
              "[%d] Register symmetric buffer for NCCL communicator; "
              "buffer=%p; size=%d; comm=%p",
              device_ordinal, buffer.opaque(), buffer.size(), comm_);
          void* handle = nullptr;
          XLA_NCCL_RETURN_IF_ERROR(ncclGroupStart());
          XLA_NCCL_RETURN_IF_ERROR(ncclCommWindowRegister(
              comm_, buffer.opaque(), buffer.size(), (ncclWindow_t*)&handle,
              NCCL_WIN_COLL_SYMMETRIC));
          XLA_NCCL_RETURN_IF_ERROR(ncclGroupEnd());
          if (group_nesting_level_ == 0) {
            TF_RETURN_IF_ERROR(PollUntilDone());
          }
          return std::make_unique<NcclRegisteredBufferHandle>(
              *this, handle, executor_.get(),
              /*symmetric_buffer= */ true, device_ordinal);
        });
#else
  return Unimplemented(
      "[%d] NCCL version does not support ncclCommWindowRegister",
      device_ordinal);
#endif  // NCCL_VERSION_CODE >= 22700
  }
}

Future<> NcclCommunicator::GroupExecute(
    absl::AnyInvocable<absl::Status(GpuCommunicator*)> f) {
  return Execute([f = std::move(f), this]() mutable -> absl::Status {
    TF_RETURN_IF_ERROR(GroupStart());
    TF_RETURN_IF_ERROR(f(this));
    TF_RETURN_IF_ERROR(GroupEnd());
    return absl::OkStatus();
  });
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
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Communicator::Executor& executor) {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("NcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch NCCL AllReduce operation; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; reduction_kind=%v; comm=%p; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, reduction_kind, comm_, stream);

  TF_ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, /*is_reduction_op=*/true,
          stream->parent()->GetDeviceDescription().cuda_compute_capability()));

  TF_RETURN_IF_ERROR(XLA_NCCL_STATUS(ncclAllReduce(
      send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
      nccl_dtype, ToNcclReduction(reduction_kind), comm_,
      AsCudaStream(stream))));
  if (group_nesting_level_ == 0) {
    TF_RETURN_IF_ERROR(PollUntilDone());
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

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch NCCL Broadcast operation; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; root=%d; comm=%p; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, root.value(), comm_, stream);

  TF_ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, false,
          stream->parent()->GetDeviceDescription().cuda_compute_capability()));

  TF_RETURN_IF_ERROR(XLA_NCCL_STATUS(ncclBroadcast(
      send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
      nccl_dtype, root.value(), comm_, AsCudaStream(stream))));
  if (group_nesting_level_ == 0) {
    TF_RETURN_IF_ERROR(PollUntilDone());
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

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch NCCL ReduceScatter operation; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; reduction_kind=%v; comm=%p; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, reduction_kind, comm_, stream);

  TF_ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, /*is_reduction_op=*/true,
          stream->parent()->GetDeviceDescription().cuda_compute_capability()));

  TF_RETURN_IF_ERROR(XLA_NCCL_STATUS(ncclReduceScatter(
      send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
      nccl_dtype, ToNcclReduction(reduction_kind), comm_,
      AsCudaStream(stream))));
  if (group_nesting_level_ == 0) {
    TF_RETURN_IF_ERROR(PollUntilDone());
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

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch NCCL AllGather operation; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, comm_, stream);

  TF_ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, false,
          stream->parent()->GetDeviceDescription().cuda_compute_capability()));

  TF_RETURN_IF_ERROR(XLA_NCCL_STATUS(ncclAllGather(
      send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
      nccl_dtype, comm_, AsCudaStream(stream))));
  if (group_nesting_level_ == 0) {
    TF_RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

// If all buffers are contiguous returns a device address range that covers
// all of them, otherwise returns an empty optional.
static std::optional<se::DeviceAddressBase> IsContinguous(
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

  auto send_contiguous = IsContinguous(send_buffers);
  auto recv_contiguous = IsContinguous(recv_buffers);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch NCCL AllToAll operation; send_buffers=[%s]; "
      "send_contiguous=%v; recv_buffers=[%s]; recv_contiguous=%v; dtype=%s; "
      "count=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(),
      absl::StrJoin(send_buffers, ", ", buffer_formatter),
      send_contiguous.has_value(),
      absl::StrJoin(recv_buffers, ", ", buffer_formatter),
      recv_contiguous.has_value(),
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

  TF_ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, false,
          stream->parent()->GetDeviceDescription().cuda_compute_capability()));

#if NCCL_VERSION_CODE >= 22800
  // If send and receive buffers are contiguous we can use all-to-all API from
  // NCCL directly without launching individual send/recv operations.
  if (send_contiguous && recv_contiguous) {
    XLA_NCCL_RETURN_IF_ERROR(ncclAlltoAll(
        send_contiguous->opaque(), recv_contiguous->opaque(),
        ToNcclCount(dtype, count), nccl_dtype, comm_, AsCudaStream(stream)));
    return absl::OkStatus();
  }
#endif

  TF_RETURN_IF_ERROR(GroupStart());
  for (size_t i = 0; i < send_buffers.size(); ++i) {
    se::DeviceAddressBase send_buffer = send_buffers[i];
    se::DeviceAddressBase recv_buffer = recv_buffers[i];

    XLA_NCCL_RETURN_IF_ERROR(ncclSend(send_buffer.opaque(),
                                      ToNcclCount(dtype, count), nccl_dtype, i,
                                      comm_, AsCudaStream(stream)));

    XLA_NCCL_RETURN_IF_ERROR(ncclRecv(recv_buffer.opaque(),
                                      ToNcclCount(dtype, count), nccl_dtype, i,
                                      comm_, AsCudaStream(stream)));
  }
  TF_RETURN_IF_ERROR(GroupEnd());
  return absl::OkStatus();
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

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch NCCL CollectivePermute operation; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; source_rank=%s; target_[ranks=%s]; "
      "count=%d; "
      "comm=%p; stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      source_rank ? absl::StrCat(source_rank->value()) : "<empty>",
      absl::StrJoin(target_ranks, ", ", rank_formatter), count, comm_, stream);

  TF_ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, false,
          stream->parent()->GetDeviceDescription().cuda_compute_capability()));

  // Short-circuit if there is no source or target rank.
  if (!source_rank && target_ranks.empty()) {
    return absl::OkStatus();
  }

  TF_RETURN_IF_ERROR(GroupStart());

  if (source_rank) {
    XLA_NCCL_RETURN_IF_ERROR(
        ncclRecv(recv_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype,
                 source_rank->value(), comm_, AsCudaStream(stream)));
  }

  for (auto target_rank : target_ranks) {
    XLA_NCCL_RETURN_IF_ERROR(
        ncclSend(send_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype,
                 target_rank.value(), comm_, AsCudaStream(stream)));
  }

  TF_RETURN_IF_ERROR(GroupEnd());

  return absl::OkStatus();
}

absl::Status NcclCommunicator::LaunchSend(se::DeviceAddressBase send_buffer,
                                          PrimitiveType dtype, size_t count,
                                          RankId peer,
                                          const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("NcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch NCCL Send operation; send_buffer=%p; dtype=%s; "
      "count=%d; peer=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      primitive_util::LowercasePrimitiveTypeName(dtype), count, peer.value(),
      comm_, stream);

  TF_ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, false,
          stream->parent()->GetDeviceDescription().cuda_compute_capability()));

  TF_RETURN_IF_ERROR(XLA_NCCL_STATUS(
      ncclSend(send_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype,
               peer.value(), comm_, AsCudaStream(stream))));
  if (group_nesting_level_ == 0) {
    TF_RETURN_IF_ERROR(PollUntilDone());
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

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch NCCL Recv operation; recv_buffer=%p; dtype=%s; "
      "count=%d; peer=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), recv_buffer.opaque(),
      primitive_util::LowercasePrimitiveTypeName(dtype), count, peer.value(),
      comm_, stream);

  TF_ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, false,
          stream->parent()->GetDeviceDescription().cuda_compute_capability()));

  TF_RETURN_IF_ERROR(XLA_NCCL_STATUS(
      ncclRecv(recv_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype,
               peer.value(), comm_, AsCudaStream(stream))));
  if (group_nesting_level_ == 0) {
    TF_RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

std::string NcclCommunicator::ToString() const {
  // comm_ should not be "touched" outside of executor_, but we are printing
  // the pointer itself and not touching the value, so this is safe.
  return absl::StrFormat("NcclCommunicator(ncclComm_t=%p)", comm_);
}

absl::Status NcclCommunicator::PollUntilDone() const {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("NcclCommunicator aborted");
  }
  return ::xla::gpu::PollUntilDone(comm_, *cancel_);
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

NcclDeviceCommunicator::NcclDeviceCommunicator(const NcclCommunicator* comm,
                                               ncclDevComm dev_comm)
    : comm_(comm), dev_comm_(dev_comm) {}

NcclDeviceCommunicator::~NcclDeviceCommunicator() {
  VLOG(3) << absl::StreamFormat(
      "Destroy NCCL device comm %s constructed for %s", this->ToString(),
      comm_->ToString());

  DCHECK(comm_ && comm_->stream_executor()) << "StreamExecutor is unavailable";
  auto activation = comm_->stream_executor()->Activate();

  auto status = XLA_NCCL_STATUS(ncclDevCommDestroy(comm_->comm(), &dev_comm_));
  if (!status.ok()) {
    LOG(ERROR) << "Failed to destroy device comm: " << status.message();
  }
}

absl::StatusOr<std::unique_ptr<NcclDeviceCommunicator>>
NcclDeviceCommunicator::CreateFrom(const NcclCommunicator& comm,
                                   const Requirements& requirements) {
  VLOG(3) << absl::StreamFormat(
      "Create NCCL device comm from %s: lsa_barrier_count=%d", comm.ToString(),
      requirements.lsa_barrier_count);

  DCHECK(comm.stream_executor()) << "StreamExecutor is unavailable";
  auto activation = comm.stream_executor()->Activate();

  ncclDevCommRequirements reqs{};
  memset(&reqs, 0, sizeof(reqs));
#if NCCL_VERSION_CODE >= 22900
  reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
#endif
  reqs.barrierCount = requirements.lsa_barrier_count;
  reqs.lsaBarrierCount = requirements.lsa_barrier_count;

  ncclDevComm dev_comm{};
  TF_RETURN_IF_ERROR(
      XLA_NCCL_STATUS(ncclDevCommCreate(comm.comm(), &reqs, &dev_comm)));

  return absl::WrapUnique(new NcclDeviceCommunicator(&comm, dev_comm));
}

PlatformCommunicatorHandle NcclDeviceCommunicator::platform_comm() const {
  return {const_cast<ncclDevComm*>(&dev_comm_)};
}

std::string NcclDeviceCommunicator::ToString() const {
  return absl::StrFormat("NcclDeviceCommunicator(ncclDevComm*=%p)", &dev_comm_);
}

NcclDeviceCommunicator::PackedKernelArg NcclDeviceCommunicator::PackKernelArg()
    const {
  PackedKernelArg packed;
  static_assert(sizeof(ncclDevComm) <= sizeof(PackedKernelArg));
  std::memcpy(packed.data(), &dev_comm_, sizeof(ncclDevComm));
  return packed;
}

#endif  // NCCL_VERSION_CODE >= 22800

}  // namespace xla::gpu
