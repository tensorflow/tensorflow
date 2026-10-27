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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_MOCK_GPU_COMMUNICATOR_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_MOCK_GPU_COMMUNICATOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include "absl/base/nullability.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/collectives/gxl_communicator.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/core/collectives/registered_memory.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/ffi/api/record_c_api.h"
#include "xla/future.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/tsl/util/tied_ref.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Implements GpuDeviceCommunicator for testing.
class MockGpuDeviceCommunicator : public GpuDeviceCommunicator {
 public:
  MockGpuDeviceCommunicator() = default;

  MOCK_METHOD(PlatformCommunicatorHandle, platform_comm, (), (const, override));
  MOCK_METHOD(int64_t, lsa_size, (), (const, override));
  MOCK_METHOD(std::string, ToString, (), (const, override));
  MOCK_METHOD(se::PackedKernelArg, PackKernelArg, (), (const, override));
};

// Implements GpuCommunicator for testing.
class MockGpuCommunicator : public GpuCommunicator {
 public:
  MockGpuCommunicator() = default;

  // Communicator methods.
  MOCK_METHOD(absl::Status, Abort, (), (override));
  MOCK_METHOD(absl::Status, HealthCheck, (), (const, override));
  MOCK_METHOD(absl::Status, Barrier, (const Executor& executor), (override));
  MOCK_METHOD(Future<>, AllReduce,
              (se::DeviceAddressBase send_buffer,
               se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
               size_t count, ReductionKind reduction_kind,
               const Executor& executor),
              (override));
  MOCK_METHOD(Future<>, AllReduceMany,
              (absl::Span<const AllReduceBuffer> buffers, PrimitiveType dtype,
               ReductionKind reduction_kind, const Executor& executor),
              (override));
  MOCK_METHOD(Future<>, Broadcast,
              (se::DeviceAddressBase send_buffer,
               se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
               size_t count, RankId root, const Executor& executor),
              (override));
  MOCK_METHOD(Future<>, ReduceScatter,
              (se::DeviceAddressBase send_buffer,
               se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
               size_t count, ReductionKind reduction_kind,
               const Executor& executor),
              (override));
  MOCK_METHOD(Future<>, AllGather,
              (se::DeviceAddressBase send_buffer,
               se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
               size_t count, const Executor& executor),
              (override));
  MOCK_METHOD(Future<>, CollectivePermute,
              (se::DeviceAddressBase send_buffer,
               se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
               size_t count, std::optional<RankId> source_rank,
               absl::Span<const RankId> target_ranks, const Executor& executor),
              (override));
  MOCK_METHOD(Future<>, AllToAll,
              ((absl::InlinedVector<se::DeviceAddressBase, 4>),
               (absl::InlinedVector<se::DeviceAddressBase, 4>),
               PrimitiveType dtype, size_t count, const Executor& executor),
              (override));
  MOCK_METHOD(Future<>, Send,
              (se::DeviceAddressBase send_buffer, PrimitiveType dtype,
               size_t count, RankId peer, const Executor& executor),
              (override));
  MOCK_METHOD(Future<>, Recv,
              (se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
               size_t count, RankId peer, const Executor& executor),
              (override));
  MOCK_METHOD(Future<>, Send,
              (se::DeviceAddressBase recv_buffer,
               se::DeviceAddressBase send_buffer, PrimitiveType dtype,
               size_t count, RankId peer, const Executor& executor),
              (override));
  MOCK_METHOD(Future<>, Recv,
              (se::DeviceAddressBase recv_buffer,
               se::DeviceAddressBase send_buffer, PrimitiveType dtype,
               size_t count, RankId peer, const Executor& executor),
              (override));
  MOCK_METHOD(Future<>, Put,
              (se::DeviceAddressBase send_buffer, SymmetricMemory* recv_buffer,
               size_t offset, size_t count, RankId peer,
               const Executor& executor),
              (override));
  MOCK_METHOD(Future<>, Signal,
              (RankId peer, const SignalDesc& signal_desc,
               const Executor& executor),
              (override));
  MOCK_METHOD(Future<>, WaitSignal,
              (RankId peer, int op_cnt, const SignalDesc& signal_desc,
               const Executor& executor),
              (override));
  MOCK_METHOD(Future<>, WaitSignals,
              (absl::Span<const PeerWaitDesc> peer_wait_descs,
               const Executor& executor),
              (override));
  MOCK_METHOD(absl::StatusOr<size_t>, NumRanks, (), (const, override));
  MOCK_METHOD(absl::StatusOr<size_t>, CurrentRank, (), (override));
  MOCK_METHOD(std::string, ToString, (), (const, override));
  MOCK_METHOD(absl::Status, Quiet, (const Executor& executor), (override));
  MOCK_METHOD(absl::Status, Fence, (), (override));

  // GpuCommunicator methods.
  MOCK_METHOD(PlatformCommunicatorHandle, platform_comm, (), (const, override));
  MOCK_METHOD(bool, SupportsDeviceComm, (), (const, override));
  MOCK_METHOD(bool, SupportsGin, (), (const, override));
  MOCK_METHOD(std::optional<int>, LsaSize, (), (const, override));
  MOCK_METHOD(se::StreamExecutor*, stream_executor, (), (const, override));
  MOCK_METHOD(GxlCommunicator*, gxl_communicator, (), (const, override));
  MOCK_METHOD(void, set_gxl_communicator,
              (std::unique_ptr<GxlCommunicator> gxl_communicator), (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<GpuDeviceCommunicator>>,
              CreateDeviceComm,
              (const GpuDeviceCommunicator::Requirements& requirements),
              (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<RegisteredMemory>>,
              CreateRegisteredMemory, (se::DeviceAddressBase addr), (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<SymmetricMemory>>,
              CreateSymmetricMemory,
              (se::DeviceAddressBase addr, bool symmetric_window), (override));
  MOCK_METHOD(Future<>, GroupExecute,
              (absl::AnyInvocable<absl::Status() &&> group), (override));
  MOCK_METHOD(absl::Status, LaunchAllReduce,
              (se::DeviceAddressBase send_buffer,
               se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
               size_t count, ReductionKind reduction_kind,
               const Executor& executor),
              (override));
  MOCK_METHOD(absl::Status, LaunchBroadcast,
              (se::DeviceAddressBase send_buffer,
               se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
               size_t count, RankId root, const Executor& executor),
              (override));
  MOCK_METHOD(absl::Status, LaunchReduce,
              (se::DeviceAddressBase send_buffer,
               se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
               size_t count, ReductionKind reduction_kind, RankId root,
               const Executor& executor),
              (override));
  MOCK_METHOD(absl::Status, LaunchReduceScatter,
              (se::DeviceAddressBase send_buffer,
               se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
               size_t count, ReductionKind reduction_kind,
               const Executor& executor),
              (override));
  MOCK_METHOD(absl::Status, LaunchAllGather,
              (se::DeviceAddressBase send_buffer,
               se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
               size_t count, const Executor& executor),
              (override));
  MOCK_METHOD(absl::Status, LaunchAllToAll,
              ((absl::InlinedVector<se::DeviceAddressBase, 4>),
               (absl::InlinedVector<se::DeviceAddressBase, 4>),
               PrimitiveType dtype, size_t count, const Executor& executor),
              (override));
  MOCK_METHOD(absl::Status, LaunchCollectivePermute,
              (se::DeviceAddressBase send_buffer,
               se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
               size_t count, std::optional<RankId> source_rank,
               absl::Span<const RankId> target_ranks, const Executor& executor),
              (override));
  MOCK_METHOD(absl::Status, LaunchSend,
              (se::DeviceAddressBase send_buffer, PrimitiveType dtype,
               size_t count, RankId peer, const Executor& executor),
              (override));
  MOCK_METHOD(absl::Status, LaunchRecv,
              (se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
               size_t count, RankId peer, const Executor& executor),
              (override));
  MOCK_METHOD(absl::Status, LaunchPut,
              (se::DeviceAddressBase send_buffer, SymmetricMemory* recv_buffer,
               size_t offset, size_t count, RankId peer,
               const Executor& executor),
              (override));
  MOCK_METHOD(absl::Status, LaunchSignal,
              (RankId peer, const SignalDesc& signal_desc,
               const Executor& executor),
              (override));
  MOCK_METHOD(absl::Status, LaunchWaitSignal,
              (RankId peer, int op_cnt, const SignalDesc& signal_desc,
               const Executor& executor),
              (override));
  MOCK_METHOD(absl::Status, LaunchWaitSignals,
              (absl::Span<const PeerWaitDesc> peer_wait_descs,
               const Executor& executor),
              (override));
  MOCK_METHOD(absl::Status, LaunchMultiGpuBarrier, (const Executor& executor),
              (override));
  MOCK_METHOD(absl::StatusOr<const XLA_FFI_Command*>, RecordMultiGpuBarrier,
              (xla::ffi::RecordContext & record_ctx,
               const XLA_FFI_Command* absl_nullable cmd),
              (override));
  MOCK_METHOD(
      absl::StatusOr<const se::CommandBuffer::Command*>, RecordMultiGpuBarrier,
      (se::CommandBuffer * command_buffer,
       const se::CommandBuffer::Command* absl_nullable cmd,
       absl::Span<const se::CommandBuffer::Command* const> dependencies),
      (override));
  MOCK_METHOD(absl::Status, InitializeCrossDeviceBarrier,
              (tsl::TiedRef<se::MemoryAllocation> tied_signal_value,
               tsl::TiedRef<se::MemoryAllocation> tied_signal,
               tsl::TiedRef<SymmetricMemory> tied_symmetric_memory),
              (override));
  MOCK_METHOD(bool, IsCrossDeviceBarrierInitiated, (), (const, override));
  MOCK_METHOD(absl::StatusOr<se::DeviceAddressBase>, GetCliqueSemaphore,
              (RankId peer_rank, int64_t collective_id), (const, override));
  MOCK_METHOD(void, InitializeCliqueSemaphore,
              (tsl::TiedRef<se::MemoryAllocation> tied_semaphore,
               tsl::TiedRef<SymmetricMemory> tied_symmetric_memory),
              (override));
  MOCK_METHOD(bool, IsCliqueSemaphoreInitialized, (), (const, override));
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_MOCK_GPU_COMMUNICATOR_H_
