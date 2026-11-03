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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
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
    std::shared_ptr<CancellationToken> cancel, int rank,
    absl::Span<const int> rank_to_pe) {
  auto comm = absl::WrapUnique(new MoriCommunicator(cancel));

  const int num_ranks = static_cast<int>(rank_to_pe.size());
  // (Sub-)communicator owns a concrete facade instance
  comm->facade_ = CollectivesFacade::Get(rank, num_ranks, rank_to_pe.data());
  if (comm->facade_ == nullptr) {
    return InvalidArgument(
        "Failed to create MORI communicator for rank %d with rank_to_pe=[%s]. "
        "Either MORI is not initialized for the current device, or the clique "
        "is not supported.",
        rank, absl::StrJoin(rank_to_pe, ","));
  }

  VLOG(1) << "Created " << *comm << " with participants: " << num_ranks;
  return comm;
}

// `facade_` is non-owning: the per-device facades are released by
// MoriCollectives::Finalize() via CollectivesFacade::TearDown().
MoriCommunicator::~MoriCommunicator() = default;

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
  VLOG(3) << "Barrier: " << ToString();
  CHECK_CANCELLED()
  ABSL_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  return se::gpu::ToStatus(facade_->RunBarrier(AsHipStream(stream)));
}

absl::StatusOr<size_t> MoriCommunicator::NumRanks() const {
  CHECK_CANCELLED()
  return static_cast<size_t>(facade_->NumPes());
}

absl::StatusOr<size_t> MoriCommunicator::CurrentRank() {
  CHECK_CANCELLED()
  return static_cast<size_t>(facade_->MyPe());
}

std::string MoriCommunicator::ToString() const {
  return absl::StrFormat("MoriCommunicator(rank=%d, num_ranks=%d)",
                         facade_->MyPe(), facade_->NumPes());
}

Future<> MoriCommunicator::AllReduce(se::DeviceAddressBase send_buffer,
                                     se::DeviceAddressBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     ReductionKind reduction_kind,
                                     const Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, reduction_kind,
                  &executor, this]() -> absl::Status {
    return LaunchAllReduce(send_buffer, recv_buffer, dtype, count,
                           reduction_kind, executor, /*is_rma=*/false);
  });
}

Future<> MoriCommunicator::Broadcast(se::DeviceAddressBase send_buffer,
                                     se::DeviceAddressBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     RankId root, const Executor& executor) {
  return Execute(
      [send_buffer, recv_buffer, dtype, count, root, &executor, this]() {
        return LaunchBroadcast(send_buffer, recv_buffer, dtype, count, root,
                               executor, /*is_rma=*/false);
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
                               reduction_kind, executor, /*is_rma=*/false);
  });
}

Future<> MoriCommunicator::AllGather(se::DeviceAddressBase send_buffer,
                                     se::DeviceAddressBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     const Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, &executor, this]() {
    return LaunchAllGather(send_buffer, recv_buffer, dtype, count, executor,
                           /*is_rma=*/false);
  });
}

Future<> MoriCommunicator::AllToAll(
    absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  return Execute([send_buffers, recv_buffers, dtype, count, &executor, this]() {
    return LaunchAllToAll(send_buffers, recv_buffers, dtype, count, executor,
                          /*is_rma=*/false);
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
    PrimitiveType dtype, size_t count, const Executor& executor, bool is_rma) {
  CHECK_CANCELLED()
  ABSL_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  VLOG(3) << "AllGather: send_buffer=" << send_buffer.opaque()
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
    const Executor& executor, bool is_rma) {
  CHECK_CANCELLED()
  ABSL_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  VLOG(3) << "AllReduce: send_buffer=" << send_buffer.opaque()
          << " recv_buffer=" << recv_buffer.opaque()
          << " dtype=" << primitive_util::LowercasePrimitiveTypeName(dtype)
          << " count=" << count << " reduction_kind=" << reduction_kind
          << " stream=" << AsHipStream(stream);
  ABSL_ASSIGN_OR_RETURN(auto dt, ToMoriDataType(dtype));
  ABSL_ASSIGN_OR_RETURN(auto op, ToMoriReduceOp(reduction_kind));
  return se::gpu::ToStatus(facade_->RunAllReduce(send_buffer.opaque(),
                                                 recv_buffer.opaque(), count,
                                                 dt, op, AsHipStream(stream)));
}

absl::Status MoriCommunicator::LaunchReduceScatter(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind kind,
    const Executor& executor, bool is_rma) {
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
    PrimitiveType dtype, size_t count, const Executor& executor, bool is_rma) {
  CHECK_CANCELLED()
  ABSL_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  auto format_addr = [](std::string* out, se::DeviceAddressBase buf) {
    absl::StrAppendFormat(out, "%p", buf.opaque());
  };
  VLOG(3) << "AllToAll: send_buffers="
          << absl::StrJoin(send_buffers, ", ", format_addr)
          << " recv_buffers=" << absl::StrJoin(recv_buffers, ", ", format_addr)
          << " dtype=" << primitive_util::LowercasePrimitiveTypeName(dtype)
          << " count=" << count << " stream=" << AsHipStream(stream);

  if (send_buffers.size() != recv_buffers.size() ||
      send_buffers.size() != static_cast<size_t>(facade_->NumPes())) {
    return InvalidArgument(
        "Number of send/recv buffers and number of PEs mismatch");
  }

  CollectivesFacade::AddressVector addrs;
  addrs.reserve(facade_->NumPes());
  for (int p = 0; p < facade_->NumPes(); ++p) {
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
  VLOG(3) << "Send: send_buffer=" << send_buffer.opaque()
          << " dtype=" << primitive_util::LowercasePrimitiveTypeName(dtype)
          << " count=" << count << " peer=" << peer.value()
          << " stream=" << AsHipStream(stream);
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
  VLOG(3) << "Recv: recv_buffer=" << recv_buffer.opaque()
          << " dtype=" << primitive_util::LowercasePrimitiveTypeName(dtype)
          << " count=" << count << " peer=" << peer.value()
          << " stream=" << AsHipStream(stream);
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
  auto formatter = [](std::string* out, RankId rank) {
    absl::StrAppendFormat(out, "%d", rank.value());
  };

  const int srcPe = source_rank.value_or(RankId{-1}).value();
  VLOG(3) << "CollectivePermute: send_buffer=" << send_buffer.opaque()
          << " recv_buffer=" << recv_buffer.opaque()
          << " dtype=" << primitive_util::LowercasePrimitiveTypeName(dtype)
          << " source_rank=" << srcPe
          << " target_ranks=" << absl::StrJoin(target_ranks, ", ", formatter)
          << " count=" << count << " stream=" << AsHipStream(stream);

  std::vector<int> dstPes;
  dstPes.reserve(target_ranks.size());
  for (RankId rank : target_ranks) {
    dstPes.push_back(static_cast<int>(rank.value()));
  }
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
  VLOG(3) << "Quiet: " << ToString();
  CHECK_CANCELLED()
  ABSL_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  return se::gpu::ToStatus(facade_->RunQuiet(AsHipStream(stream)));
}

absl::Status MoriCommunicator::Fence() {
  VLOG(3) << "Fence: " << ToString();
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
