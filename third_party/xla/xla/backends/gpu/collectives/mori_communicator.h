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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_MORI_COMMUNICATOR_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_MORI_COMMUNICATOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/collectives/gxl_communicator.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/future.h"
#include "xla/stream_executor/device_address.h"
#include "xla/xla_data.pb.h"

namespace mori::ccl {
class CollectivesFacade;
}  // namespace mori::ccl

namespace xla::gpu {

// Dummy symmetric memory for the MORI backend. MORI collective buffers are
// allocated directly from the symmetric shmem heap, so the local device
// address doubles as the symmetric handle and no separate registration is
// required. This simply returns the address it was created with.
class MoriSymmetricMemory : public SymmetricMemory {
 public:
  explicit MoriSymmetricMemory(se::DeviceAddressBase addr) : addr_(addr) {}

  se::DeviceAddressBase addr() const final { return addr_; }

  std::string ToString() const final {
    return absl::StrFormat("MoriSymmetricMemory(addr=%p, size=%d)",
                           addr_.opaque(), addr_.size());
  }

  PackedKernelArg PackKernelArg() const final { return addr_.opaque(); }

 private:
  se::DeviceAddressBase addr_;
};

// XLA collectives communicator wrapping a MORI communicator.
class MoriCommunicator : public GpuCommunicator {
 public:
  ~MoriCommunicator() override;

  static absl::StatusOr<std::unique_ptr<MoriCommunicator>> Create(
      std::shared_ptr<CancellationToken> cancel, int rank,
      absl::Span<const int> rank_to_pe);

  // MoriCommunicator is not copyable or movable.
  MoriCommunicator(const MoriCommunicator&) = delete;
  MoriCommunicator(MoriCommunicator&&) = delete;
  MoriCommunicator& operator=(const MoriCommunicator&) = delete;
  MoriCommunicator& operator=(MoriCommunicator&&) = delete;

  absl::Status Abort() final;
  absl::StatusOr<size_t> NumRanks() const final;
  absl::StatusOr<size_t> CurrentRank() final;

  absl::StatusOr<std::unique_ptr<SymmetricMemory>> CreateSymmetricMemory(
      se::DeviceAddressBase addr, bool symmetric_window) final {
    // Dummy implementation: MORI buffers are already allocated from the
    // symmetric shmem heap, so the local device address is the symmetric
    // handle. Just wrap and return it unchanged.
    return std::make_unique<MoriSymmetricMemory>(addr);
  }

  absl::Status Barrier(const Executor& executor) final;

  void SetAsync(bool is_async);
  absl::Status AsyncDone(const Executor& executor);

  Future<> GroupExecute(absl::AnyInvocable<absl::Status() &&> group) final;

  Future<> AllReduce(se::DeviceAddressBase send_buffer,
                     se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                     size_t count, ReductionKind reduction_kind,
                     const Executor& executor) final;

  Future<> Broadcast(se::DeviceAddressBase send_buffer,
                     se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                     size_t count, RankId root, const Executor& executor) final;

  Future<> ReduceScatter(se::DeviceAddressBase send_buffer,
                         se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                         size_t count, ReductionKind reduction_kind,
                         const Executor& executor) final;

  Future<> AllGather(se::DeviceAddressBase send_buffer,
                     se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                     size_t count, const Executor& executor) final;

  Future<> AllToAll(absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
                    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
                    PrimitiveType dtype, size_t count,
                    const Executor& executor) final;

  Future<> CollectivePermute(se::DeviceAddressBase send_buffer,
                             se::DeviceAddressBase recv_buffer,
                             PrimitiveType dtype, size_t count,
                             std::optional<RankId> source_rank,
                             absl::Span<const RankId> target_ranks,
                             const Executor& executor) final;

  Future<> Send(se::DeviceAddressBase send_buffer, PrimitiveType dtype,
                size_t count, RankId peer, const Executor& executor) final;

  Future<> Recv(se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                size_t count, RankId peer, const Executor& executor) final;

  // Polls the communicator until any pending non-blocking operations are done
  // or aborted.
  absl::Status PollUntilDone() const;

  // Serves the GXL ragged-all-to-all and all-gather hooks
  // (--xla_gpu_enable_gxl_ragged_all_to_all) with MORI kernels.
  GxlCommunicator* gxl_communicator() const final { return gxl_.get(); }

 private:
  class GxlAdapter;

  absl::Status LaunchRaggedAllToAll(
      se::DeviceAddressBase input, se::DeviceAddressBase output,
      se::DeviceAddressBase input_offsets, se::DeviceAddressBase send_sizes,
      se::DeviceAddressBase output_offsets, PrimitiveType dtype,
      int64_t num_row_elements, int64_t num_total_updates,
      const Executor& executor);

  absl::Status GroupLaunch(absl::FunctionRef<absl::Status()> group);

  absl::Status LaunchAllReduce(se::DeviceAddressBase send_buffer,
                               se::DeviceAddressBase recv_buffer,
                               PrimitiveType dtype, size_t count,
                               ReductionKind reduction_kind,
                               const Executor& executor, bool is_rma) final;

  absl::Status LaunchBroadcast(se::DeviceAddressBase send_buffer,
                               se::DeviceAddressBase recv_buffer,
                               PrimitiveType dtype, size_t count, RankId root,
                               const Executor& executor, bool is_rma) final {
    return absl::UnimplementedError("Not implemented");
  }

  absl::Status LaunchReduceScatter(se::DeviceAddressBase send_buffer,
                                   se::DeviceAddressBase recv_buffer,
                                   PrimitiveType dtype, size_t count,
                                   ReductionKind reduction_kind,
                                   const Executor& executor, bool is_rma) final;

  absl::Status LaunchAllGather(se::DeviceAddressBase send_buffer,
                               se::DeviceAddressBase recv_buffer,
                               PrimitiveType dtype, size_t count,
                               const Executor& executor, bool is_rma) final;

  absl::Status LaunchAllToAll(
      absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
      absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
      PrimitiveType dtype, size_t count, const Executor& executor,
      bool is_rma) final;

  absl::Status LaunchCollectivePermute(se::DeviceAddressBase send_buffer,
                                       se::DeviceAddressBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       std::optional<RankId> source_rank,
                                       absl::Span<const RankId> target_ranks,
                                       const Executor& executor) final;

  absl::Status LaunchSend(se::DeviceAddressBase send_buffer,
                          PrimitiveType dtype, size_t count, RankId peer,
                          const Executor& executor) final;

  absl::Status LaunchRecv(se::DeviceAddressBase recv_buffer,
                          PrimitiveType dtype, size_t count, RankId peer,
                          const Executor& executor) final;

  absl::Status Quiet(const Executor& executor) final;

  absl::Status Fence() final;

  std::string ToString() const final;

 private:
  // Runs `launch` with is_async = is_async_, except inside a group: there an
  // async launch is held until the next one (which flushes it sync) or the end
  // of the outermost GroupLaunch (which runs it async), so at most one per
  // group is left pending for AsyncDone.
  absl::Status LaunchOrDefer(
      absl::AnyInvocable<absl::Status(bool is_async) &&> launch);

  MoriCommunicator(std::shared_ptr<CancellationToken> cancel)
      : cancel_(std::move(cancel)) {}

  // CollectivesFacade instance for this communicator's device
  std::unique_ptr<mori::ccl::CollectivesFacade> facade_;
  // GxlAdapter holding a back-pointer to this communicator.
  std::unique_ptr<GxlCommunicator> gxl_;
  // Should all pending collectives cancel?
  std::shared_ptr<CancellationToken> cancel_;
  bool aborted_ = false;  // Has Abort() been called?
  // Do the following collective launches belong to an async start/done pair?
  // Read by the Launch* methods, so Execute must run them synchronously.
  bool is_async_ = false;
  // GroupLaunch nesting depth and the async launch held back within it.
  int group_depth_ = 0;
  absl::AnyInvocable<absl::Status(bool) &&> deferred_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_MORI_COMMUNICATOR_H_
