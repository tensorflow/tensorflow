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

#include "xla/backends/gpu/collectives/mori_communicator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/mori_collectives.h"
#include "xla/backends/gpu/collectives/mori_kernels.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/future.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace shmem = ::mori::shmem;
namespace xla::gpu {

using ::mori::collective::CollectivesFacade;
namespace {

hipStream_t AsHipStream(se::Stream* stream) {
  return reinterpret_cast<hipStream_t>(
      stream->platform_specific_handle().stream);
}

size_t ToMoriByteCount(PrimitiveType dtype, size_t count) {
  if (primitive_util::IsComplexType(dtype)) {
    count *= 2;
  }
  return count * primitive_util::BitWidth(dtype) / 8;
}

absl::StatusOr<::mori::collective::DataType> ToMoriDataType(
    PrimitiveType dtype) {
#define MORI_TYPE_DISPATCH(x) \
  case x:                     \
    return ::mori::collective::DataType::x;
  switch (dtype) {
    MORI_TYPE_DISPATCH(F8E5M2)
    MORI_TYPE_DISPATCH(F8E4M3FN)
    MORI_TYPE_DISPATCH(F16)
    MORI_TYPE_DISPATCH(BF16)
    MORI_TYPE_DISPATCH(S8)
    MORI_TYPE_DISPATCH(U8)
    MORI_TYPE_DISPATCH(S32)
    MORI_TYPE_DISPATCH(U32)
    MORI_TYPE_DISPATCH(S64)
    MORI_TYPE_DISPATCH(U64)
    MORI_TYPE_DISPATCH(F32)
    MORI_TYPE_DISPATCH(F64)
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "MORI: unsupported dtype: %d", static_cast<int>(dtype)));
  }
#undef MORI_TYPE_DISPATCH
}

// Translate an XLA ReductionKind to the facade's reduction-op enum.
absl::StatusOr<::mori::collective::ReduceOpKind> ToMoriReduceOp(
    ReductionKind r) {
#define MORI_OP_DISPATCH(x) \
  case ReductionKind::x:    \
    return ::mori::collective::ReduceOpKind::x;
  switch (r) {
    MORI_OP_DISPATCH(SUM)
    MORI_OP_DISPATCH(PRODUCT)
    MORI_OP_DISPATCH(MIN)
    MORI_OP_DISPATCH(MAX)
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "MORI: unsupported reduction op: %d", static_cast<int>(r)));
  }
#undef MORI_OP_DISPATCH
}

absl::StatusOr<se::Stream*> ToStream(const Communicator::Executor& executor) {
  if (auto* gpu_executor =
          absl::down_cast<const GpuCollectives::Executor*>(&executor)) {
    return gpu_executor->stream();
  }
  return InvalidArgument("Communicator executor is not a GPU executor");
}
}  // namespace

absl::StatusOr<std::unique_ptr<MoriCommunicator>> MoriCommunicator::Create(
    MoriCollectives* coll, std::shared_ptr<CancellationToken> cancel, int rank,
    absl::Span<const int> rank_to_pe) {
  auto comm = absl::WrapUnique(new MoriCommunicator(coll, cancel));

  const int num_ranks = static_cast<int>(rank_to_pe.size());
  if (num_ranks <= 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "MoriCommunicator: unsupported number of ranks %d", num_ranks));
  }
  comm->rank_ = rank;
  comm->num_ranks_ = num_ranks;

  // The CollectivesFacade owns this communicator's symmetric-heap staging
  // buffer and the push reduce-scatter group counters. It records the rank
  // identity (rank/num_ranks) and allocates the ~2GB staging; the unique_ptr
  // frees it (before ShmemFinalize) when the communicator is destroyed.
  const size_t buffer_size = 2UL << 30;  // 2GB
  comm->facade_ = CollectivesFacade::Create(rank, num_ranks, buffer_size);
  if (comm->facade_ == nullptr) {
    return absl::InternalError("CollectivesFacade::Create failed");
  }
  VLOG(1) << "Created " << *comm << " with participants: " << num_ranks;
  return comm;
}

MoriCommunicator::~MoriCommunicator() {
  // facade_ (unique_ptr) releases this communicator's staging + counters via
  // the CollectivesFacade dtor here, before MoriCollectives::Finalize() ->
  // ShmemFinalize.
}

#define CHECK_CANCELLED()                                               \
  if (cancel_->IsCancelled()) {                                         \
    return absl::FailedPreconditionError("MoriCommunicator cancelled"); \
  }

absl::Status MoriCommunicator::Abort() {
  // By setting the cancellation token all pending collectives scheduled on
  // executor_ will cancel. This will allow the aborting lambda below to run.
  cancel_->Cancel();

  VLOG(1) << "Abort MORI communicator: " << ToString();
  if (aborted_) {
    return FailedPrecondition("MoriCommunicator already aborted");
  }
  aborted_ = true;
  // Call rocm_mori_global_exit with a non-zero return code to abort the
  // program. rocm_mori_global_exit(1);
  return absl::OkStatus();
}

absl::Status MoriCommunicator::Barrier(const Executor& executor) {
  VLOG(1) << "Barrier: " << ToString();
  CHECK_CANCELLED()
  ABSL_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  return se::gpu::ToStatus(facade_->RunBarrier(AsHipStream(stream)));
}

absl::StatusOr<size_t> MoriCommunicator::NumRanks() const {
  CHECK_CANCELLED()
  return static_cast<size_t>(num_ranks_);
}

absl::StatusOr<size_t> MoriCommunicator::CurrentRank() {
  CHECK_CANCELLED()
  return static_cast<size_t>(rank_);
}

std::string MoriCommunicator::ToString() const {
  return absl::StrFormat("MoriCommunicator(rank=%d, num_ranks=%d)", rank_,
                         num_ranks_);
}

Future<> MoriCommunicator::AllReduce(se::DeviceAddressBase send_buffer,
                                     se::DeviceAddressBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     ReductionKind reduction_kind,
                                     const Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, reduction_kind,
                  &executor, this]() -> absl::Status {
    return LaunchAllReduce(send_buffer, recv_buffer, dtype, count,
                           reduction_kind, executor);
  });
}

Future<> MoriCommunicator::Broadcast(se::DeviceAddressBase send_buffer,
                                     se::DeviceAddressBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     RankId root, const Executor& executor) {
  return Execute(
      [send_buffer, recv_buffer, dtype, count, root, &executor, this]() {
        return LaunchBroadcast(send_buffer, recv_buffer, dtype, count, root,
                               executor);
      });
}

Future<> MoriCommunicator::ReduceScatter(se::DeviceAddressBase send_buffer,
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

Future<> MoriCommunicator::AllGather(se::DeviceAddressBase send_buffer,
                                     se::DeviceAddressBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     const Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, &executor, this]() {
    return LaunchAllGather(send_buffer, recv_buffer, dtype, count, executor);
  });
}

Future<> MoriCommunicator::AllToAll(
    absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  return Execute([send_buffers, recv_buffers, dtype, count, &executor, this]() {
    return LaunchAllToAll(send_buffers, recv_buffers, dtype, count, executor);
  });
}

Future<> MoriCommunicator::CollectivePermute(
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

Future<> MoriCommunicator::Send(se::DeviceAddressBase send_buffer,
                                PrimitiveType dtype, size_t count, RankId peer,
                                const Executor& executor) {
  return Execute([send_buffer, dtype, count, peer, &executor, this]() {
    return LaunchSend(send_buffer, dtype, count, peer, executor);
  });
}

Future<> MoriCommunicator::Recv(se::DeviceAddressBase recv_buffer,
                                PrimitiveType dtype, size_t count, RankId peer,
                                const Executor& executor) {
  return Execute([recv_buffer, dtype, count, peer, &executor, this]() {
    return LaunchRecv(recv_buffer, dtype, count, peer, executor);
  });
}

absl::Status MoriCommunicator::LaunchAllGather(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  CHECK_CANCELLED()
  ABSL_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  VLOG(3) << "Launch AllGather: send_buffer=" << send_buffer.opaque()
          << " recv_buffer=" << recv_buffer.opaque() << " count=" << count
          << " dtype=" << primitive_util::LowercasePrimitiveTypeName(dtype)
          << " stream=" << AsHipStream(stream);
  return se::gpu::ToStatus(facade_->RunAllGather(
      send_buffer.opaque(), recv_buffer.opaque(), ToMoriByteCount(dtype, count),
      AsHipStream(stream)));
}

absl::Status MoriCommunicator::LaunchAllReduce(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Executor& executor) {
  CHECK_CANCELLED()
  ABSL_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  VLOG(3) << absl::StreamFormat(
      "Launch AllReduce: send_buffer=%p; recv_buffer=%p; dtype=%s; count=%d; "
      "reduction_kind=%v; stream=%p",
      send_buffer.opaque(), recv_buffer.opaque(),
      primitive_util::LowercasePrimitiveTypeName(dtype), count, reduction_kind,
      stream);

  ABSL_ASSIGN_OR_RETURN(auto dt, ToMoriDataType(dtype));
  ABSL_ASSIGN_OR_RETURN(auto op, ToMoriReduceOp(reduction_kind));
  return se::gpu::ToStatus(facade_->RunAllReduce(send_buffer.opaque(),
                                                 recv_buffer.opaque(), count,
                                                 dt, op, AsHipStream(stream)));
}

absl::Status MoriCommunicator::LaunchReduceScatter(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind kind,
    const Executor& executor) {
  CHECK_CANCELLED()
  ABSL_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  VLOG(3) << "LaunchReduceScatter: send_buffer=" << send_buffer.opaque()
          << " recv_buffer=" << recv_buffer.opaque() << " count=" << count
          << " dtype=" << primitive_util::LowercasePrimitiveTypeName(dtype)
          << " stream=" << AsHipStream(stream);

  ABSL_ASSIGN_OR_RETURN(auto dt, ToMoriDataType(dtype));
  ABSL_ASSIGN_OR_RETURN(auto op, ToMoriReduceOp(kind));
  return se::gpu::ToStatus(
      facade_->RunReduceScatter(send_buffer.opaque(), recv_buffer.opaque(),
                                count, dt, op, AsHipStream(stream)));
}

absl::Status MoriCommunicator::LaunchAllToAll(
    absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  CHECK_CANCELLED()
  ABSL_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  auto format_addr = [](std::string* out, se::DeviceAddressBase buf) {
    absl::StrAppendFormat(out, "%p", buf.opaque());
  };
  VLOG(3) << absl::StreamFormat(
      "Launch MORI AllToAll operation; send_buffers=[%s]; recv_buffers=[%s]; "
      "dtype=%s; count=%d; stream=%p",
      absl::StrJoin(send_buffers, ", ", format_addr),
      absl::StrJoin(recv_buffers, ", ", format_addr),
      primitive_util::LowercasePrimitiveTypeName(dtype), count,
      AsHipStream(stream));

  if (send_buffers.size() != recv_buffers.size() ||
      send_buffers.size() != static_cast<size_t>(num_ranks_)) {
    return InvalidArgument(
        "Number of send/recv buffers and number of  ranks mismatch");
  }

  CollectivesFacade::AddressVector addrs;
  addrs.reserve(num_ranks_);
  for (int p = 0; p < num_ranks_; ++p) {
    addrs.emplace_back(send_buffers[p].opaque(), recv_buffers[p].opaque());
  }
  return se::gpu::ToStatus(facade_->RunAllToAll(
      addrs, ToMoriByteCount(dtype, count), AsHipStream(stream)));
}

absl::Status MoriCommunicator::LaunchSend(se::DeviceAddressBase send_buffer,
                                          PrimitiveType dtype, size_t count,
                                          RankId peer,
                                          const Executor& executor) {
  CHECK_CANCELLED()
  ABSL_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  VLOG(3) << absl::StreamFormat(
      "Launch MORI Send operation; send_buffer=%p; dtype=%s; count=%d; "
      "peer=%d; stream=%p",
      send_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, peer.value(), AsHipStream(stream));
  return se::gpu::ToStatus(
      facade_->RunSend(send_buffer.opaque(), ToMoriByteCount(dtype, count),
                       static_cast<int>(peer.value()), AsHipStream(stream)));
}

absl::Status MoriCommunicator::LaunchRecv(se::DeviceAddressBase recv_buffer,
                                          PrimitiveType dtype, size_t count,
                                          RankId peer,
                                          const Executor& executor) {
  CHECK_CANCELLED()
  ABSL_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  VLOG(3) << absl::StreamFormat(
      "Launch MORI Recv operation; recv_buffer=%p; dtype=%s; count=%d; "
      "peer=%d; stream=%p",
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, peer.value(), AsHipStream(stream));
  return se::gpu::ToStatus(
      facade_->RunRecv(recv_buffer.opaque(), ToMoriByteCount(dtype, count),
                       static_cast<int>(peer.value()), AsHipStream(stream)));
}

absl::Status MoriCommunicator::LaunchCollectivePermute(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  CHECK_CANCELLED()
  ABSL_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  auto rank_formatter = [](std::string* out, RankId rank) {
    absl::StrAppendFormat(out, "%d", rank.value());
  };
  VLOG(3) << absl::StreamFormat(
      "[%d] Launch CollectivePermute: send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; source_rank=%s; target_[ranks=%s]; count=%d; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      source_rank ? absl::StrCat(source_rank->value()) : "<empty>",
      absl::StrJoin(target_ranks, ", ", rank_formatter), count, stream);

  std::vector<int> dstPes;
  dstPes.reserve(target_ranks.size());
  for (RankId rank : target_ranks) {
    dstPes.push_back(static_cast<int>(rank.value()));
  }
  const int srcPe = source_rank ? static_cast<int>(source_rank->value()) : -1;
  return se::gpu::ToStatus(facade_->RunCollectivePermute(
      send_buffer.opaque(), recv_buffer.opaque(), ToMoriByteCount(dtype, count),
      srcPe, dstPes, AsHipStream(stream)));
}

Future<> MoriCommunicator::GroupExecute(
    absl::AnyInvocable<absl::Status() &&> group) {
  return Execute([group = std::move(group), this]() mutable {
    return GroupLaunch([&] { return std::move(group)(); });
  });
}

absl::Status MoriCommunicator::GroupLaunch(
    absl::FunctionRef<absl::Status()> group) {
  return group();
}

absl::Status MoriCommunicator::Quiet(const Executor& executor) {
  VLOG(1) << "Quiet: " << ToString();
  CHECK_CANCELLED()
  ABSL_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  return se::gpu::ToStatus(facade_->RunQuiet(AsHipStream(stream)));
}

absl::Status MoriCommunicator::Fence() {
  VLOG(1) << "Fence: " << ToString();
  CHECK_CANCELLED()
  return se::gpu::ToStatus(facade_->RunFence());
}

absl::Status MoriCommunicator::PollUntilDone() const {
  CHECK_CANCELLED()
  return absl::UnimplementedError("Not implemented");
}

Future<> MoriCommunicator::Execute(
    absl::AnyInvocable<absl::Status() &&> f) const {
  return Future<>(std::move(f)());
}

}  // namespace xla::gpu
