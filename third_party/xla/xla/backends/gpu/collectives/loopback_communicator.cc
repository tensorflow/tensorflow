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

#include "xla/backends/gpu/collectives/loopback_communicator.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/future.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace xla::gpu {

static size_t ByteSize(PrimitiveType dtype, size_t count) {
  return primitive_util::ByteWidth(dtype) * count;
}

static absl::Status Memcpy(const Communicator::Executor& executor,
                           se::DeviceAddressBase dst, se::DeviceAddressBase src,
                           size_t size) {
  if (size == 0 || dst.opaque() == src.opaque()) {
    return absl::OkStatus();
  }
  auto* stream =
      absl::down_cast<const GpuCollectives::Executor&>(executor).stream();
  return stream->MemcpyD2D(&dst, src, size);
}

LoopbackCommunicator::LoopbackCommunicator(se::StreamExecutor* executor,
                                           size_t num_ranks, size_t rank)
    : executor_(executor), num_ranks_(num_ranks), rank_(rank) {
  VLOG(1) << absl::StreamFormat(
      "LoopbackCommunicator created: rank=%d/%d, executor=%p", rank_,
      num_ranks_, executor_);
}

absl::StatusOr<size_t> LoopbackCommunicator::NumRanks() const {
  return num_ranks_;
}

absl::StatusOr<size_t> LoopbackCommunicator::CurrentRank() { return rank_; }

std::string LoopbackCommunicator::ToString() const {
  return absl::StrFormat("LoopbackCommunicator(rank=%d, num_ranks=%d)", rank_,
                         num_ranks_);
}

Future<> LoopbackCommunicator::GroupExecute(
    absl::AnyInvocable<absl::Status(GpuCommunicator*)> f) {
  return Future<>(f(this));
}

//===----------------------------------------------------------------------===//
// Async collective operations (delegate to Launch* methods)
//===----------------------------------------------------------------------===//

Future<> LoopbackCommunicator::AllReduce(se::DeviceAddressBase send_buffer,
                                         se::DeviceAddressBase recv_buffer,
                                         PrimitiveType dtype, size_t count,
                                         ReductionKind reduction_kind,
                                         const Executor& executor) {
  return Future<>(LaunchAllReduce(send_buffer, recv_buffer, dtype, count,
                                  reduction_kind, executor));
}

Future<> LoopbackCommunicator::Broadcast(se::DeviceAddressBase send_buffer,
                                         se::DeviceAddressBase recv_buffer,
                                         PrimitiveType dtype, size_t count,
                                         RankId root,
                                         const Executor& executor) {
  return Future<>(
      LaunchBroadcast(send_buffer, recv_buffer, dtype, count, root, executor));
}

Future<> LoopbackCommunicator::ReduceScatter(se::DeviceAddressBase send_buffer,
                                             se::DeviceAddressBase recv_buffer,
                                             PrimitiveType dtype, size_t count,
                                             ReductionKind reduction_kind,
                                             const Executor& executor) {
  return Future<>(LaunchReduceScatter(send_buffer, recv_buffer, dtype, count,
                                      reduction_kind, executor));
}

Future<> LoopbackCommunicator::AllGather(se::DeviceAddressBase send_buffer,
                                         se::DeviceAddressBase recv_buffer,
                                         PrimitiveType dtype, size_t count,
                                         const Executor& executor) {
  return Future<>(
      LaunchAllGather(send_buffer, recv_buffer, dtype, count, executor));
}

Future<> LoopbackCommunicator::AllToAll(
    absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  return Future<>(LaunchAllToAll(std::move(send_buffers),
                                 std::move(recv_buffers), dtype, count,
                                 executor));
}

Future<> LoopbackCommunicator::CollectivePermute(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  return Future<>(LaunchCollectivePermute(send_buffer, recv_buffer, dtype,
                                          count, source_rank, target_ranks,
                                          executor));
}

Future<> LoopbackCommunicator::Send(se::DeviceAddressBase send_buffer,
                                    PrimitiveType dtype, size_t count,
                                    RankId peer, const Executor& executor) {
  return Future<>(LaunchSend(send_buffer, dtype, count, peer, executor));
}

Future<> LoopbackCommunicator::Recv(se::DeviceAddressBase recv_buffer,
                                    PrimitiveType dtype, size_t count,
                                    RankId peer, const Executor& executor) {
  return Future<>(LaunchRecv(recv_buffer, dtype, count, peer, executor));
}

//===----------------------------------------------------------------------===//
// Synchronous Launch* methods
//===----------------------------------------------------------------------===//

// AllReduce: output = input (other ranks contribute identity elements).
absl::Status LoopbackCommunicator::LaunchAllReduce(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Executor& executor) {
  VLOG(5) << absl::StreamFormat(
      "Loopback AllReduce: rank=%d, count=%d, bytes=%d", rank_, count,
      ByteSize(dtype, count));
  return Memcpy(executor, recv_buffer, send_buffer, ByteSize(dtype, count));
}

// Broadcast: output = input.
absl::Status LoopbackCommunicator::LaunchBroadcast(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, RankId root, const Executor& executor) {
  return Memcpy(executor, recv_buffer, send_buffer, ByteSize(dtype, count));
}

// ReduceScatter: this rank's chunk of input. `count` is the per-rank output
// count, so the input has count * num_ranks elements.
absl::Status LoopbackCommunicator::LaunchReduceScatter(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Executor& executor) {
  VLOG(5) << absl::StreamFormat(
      "Loopback ReduceScatter: rank=%d, count=%d, num_ranks=%d", rank_, count,
      num_ranks_);
  size_t byte_width = primitive_util::ByteWidth(dtype);
  size_t offset = rank_ * count * byte_width;
  se::DeviceAddressBase src(static_cast<char*>(send_buffer.opaque()) + offset,
                            count * byte_width);
  return Memcpy(executor, recv_buffer, src, count * byte_width);
}

// AllGather: replicate input into every rank's slot in output. `count` is the
// per-rank element count, so the output has count * num_ranks elements.
absl::Status LoopbackCommunicator::LaunchAllGather(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  VLOG(5) << absl::StreamFormat(
      "Loopback AllGather: rank=%d, count=%d, num_ranks=%d", rank_, count,
      num_ranks_);
  size_t chunk_bytes = ByteSize(dtype, count);
  for (size_t i = 0; i < num_ranks_; ++i) {
    se::DeviceAddressBase dst(
        static_cast<char*>(recv_buffer.opaque()) + i * chunk_bytes,
        chunk_bytes);
    RETURN_IF_ERROR(Memcpy(executor, dst, send_buffer, chunk_bytes));
  }
  return absl::OkStatus();
}

// AllToAll: memcpy each send buffer to the corresponding recv buffer.
absl::Status LoopbackCommunicator::LaunchAllToAll(
    absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  size_t size = ByteSize(dtype, count);
  for (size_t i = 0; i < send_buffers.size(); ++i) {
    RETURN_IF_ERROR(Memcpy(executor, recv_buffers[i], send_buffers[i], size));
  }
  return absl::OkStatus();
}

// CollectivePermute: loopback — copy send to recv.
absl::Status LoopbackCommunicator::LaunchCollectivePermute(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  return Memcpy(executor, recv_buffer, send_buffer, ByteSize(dtype, count));
}

// Send: no-op (data "sent" successfully, no real peer to receive it).
absl::Status LoopbackCommunicator::LaunchSend(se::DeviceAddressBase send_buffer,
                                              PrimitiveType dtype, size_t count,
                                              RankId peer,
                                              const Executor& executor) {
  VLOG(5) << absl::StreamFormat("Loopback Send: rank=%d, peer=%d, count=%d",
                                rank_, peer.value(), count);
  return absl::OkStatus();
}

// Recv: zero-fill the buffer (no real peer sending data; zeroing avoids
// non-finite values from uninitialized memory).
absl::Status LoopbackCommunicator::LaunchRecv(se::DeviceAddressBase recv_buffer,
                                              PrimitiveType dtype, size_t count,
                                              RankId peer,
                                              const Executor& executor) {
  VLOG(5) << absl::StreamFormat("Loopback Recv: rank=%d, peer=%d, count=%d",
                                rank_, peer.value(), count);
  size_t size = ByteSize(dtype, count);
  auto* stream =
      absl::down_cast<const GpuCollectives::Executor&>(executor).stream();
  return stream->MemZero(&recv_buffer, size);
}

}  // namespace xla::gpu
