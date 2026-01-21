/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_GPU_COMMUNICATOR_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_GPU_COMMUNICATOR_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/future.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/util.h"

namespace xla::gpu {

// Platform-specific handle to the underlying communicator implementation. It
// allows exporting collective communication primitives created and owned by
// the XLA runtime to external libraries, for example via FFI calls.
struct PlatformCommunicatorHandle {
  void* handle = nullptr;  // will be nullptr if not supported
};

// A device communicator that corresponds to the host side GPU communicator
// object (it has same rank in the collective clique and shares underlying
// resources). A host-side GPU communicator object can instantiate multiple
// device-side communicators with different properties.
//
// Device communicator can be passed to GPU kernels to initiate collective
// operations (e.g. Send or Recv) directly from the kernel without having to
// involve host. Memory that can participate in device-initiated collective
// operations typically has to be registered ahead of time (see
// `SymmetricMemory` documentation).
//
// For CUDA this corresponds to:
// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html
class GpuDeviceCommunicator {
 public:
  virtual ~GpuDeviceCommunicator() = default;

  // Requirements for constructing a device communicator object.
  struct Requirements {
    template <typename Sink>
    friend void AbslStringify(Sink& sink, const Requirements& reqs) {
      absl::Format(&sink, "{lsa_barrier_count: %d}", reqs.lsa_barrier_count);
    }

    bool operator==(const Requirements& other) const {
      return other.lsa_barrier_count == lsa_barrier_count;
    }

    bool operator<(const Requirements& other) const {
      return other.lsa_barrier_count < lsa_barrier_count;
    }

    // The number of barriers to allocate for load/store accessible
    // communication.
    int32_t lsa_barrier_count = 0;
  };

  // Returns a platform-spcific handle to the unerdlying communicator object.
  virtual PlatformCommunicatorHandle platform_comm() const {
    return PlatformCommunicatorHandle{nullptr};
  }

  virtual std::string ToString() const = 0;

  // A packed kernel argument type for passing device communicator to device
  // kernels (byte storage appropriately sized to fit platform-specific handle).
  using PackedKernelArg = std::array<std::byte, 200>;
  virtual PackedKernelArg PackKernelArg() const = 0;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const GpuDeviceCommunicator& comm) {
    absl::Format(&sink, "%s", comm.ToString());
  }
};

// GpuCommunicator extends Communicator with synchronous versions of the
// collective methods.
//
// For example, the `Communicator::AllReduce` method (which is asynchronous and
// returns an AsyncValueRef<Event>) has a corresponding syncrhonous
// `GpuCommunicator::LaunchAllReduce` method which returns an `absl::Status`.
class GpuCommunicator : public Communicator {
 public:
  ~GpuCommunicator() override = default;

  // Returns a platform-spcific handle to the unerdlying communicator object.
  virtual PlatformCommunicatorHandle platform_comm() const {
    return PlatformCommunicatorHandle{nullptr};
  }

  // Returns true iff communicator supports device-initiated communication.
  virtual bool SupportsDeviceComm() const { return false; }

  // Creates a new device communicator linked to *this GPU communicator object.
  virtual absl::StatusOr<std::unique_ptr<GpuDeviceCommunicator>>
  CreateDeviceComm(const GpuDeviceCommunicator::Requirements& requirements) {
    return Unimplemented("Device communicator is not implementing");
  }

  // Creates a symmetric memory from the existing device address range. This is
  // a collective operation, and all ranks in a clique must call this operation
  // in order to make a progress.
  virtual absl::StatusOr<std::unique_ptr<SymmetricMemory>>
  CreateSymmetricMemory(se::DeviceAddressBase addr) {
    return Unimplemented("Symmetric memory is not implemented");
  }

  //===--------------------------------------------------------------------===//
  // Host-side collective communication APIs
  //===--------------------------------------------------------------------===//

  // Executes f in a group. f should invoke synchronous collective methods like
  // LaunchAllReduce and not asynchronous collective methods like AllReduce.
  virtual Future<> GroupExecute(
      absl::AnyInvocable<absl::Status(GpuCommunicator*)> f) = 0;

  virtual absl::Status LaunchAllReduce(se::DeviceAddressBase send_buffer,
                                       se::DeviceAddressBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       ReductionKind reduction_kind,
                                       const Executor& executor) = 0;

  virtual absl::Status LaunchBroadcast(se::DeviceAddressBase send_buffer,
                                       se::DeviceAddressBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       RankId root,
                                       const Executor& executor) = 0;

  virtual absl::Status LaunchReduceScatter(se::DeviceAddressBase send_buffer,
                                           se::DeviceAddressBase recv_buffer,
                                           PrimitiveType dtype, size_t count,
                                           ReductionKind reduction_kind,
                                           const Executor& executor) = 0;

  virtual absl::Status LaunchAllGather(se::DeviceAddressBase send_buffer,
                                       se::DeviceAddressBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       const Executor& executor) = 0;

  virtual absl::Status LaunchAllToAll(
      absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
      absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
      PrimitiveType dtype, size_t count, const Executor& executor) = 0;

  virtual absl::Status LaunchCollectivePermute(
      se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
      PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
      absl::Span<const RankId> target_ranks, const Executor& executor) = 0;

  virtual absl::Status LaunchSend(se::DeviceAddressBase send_buffer,
                                  PrimitiveType dtype, size_t count,
                                  RankId peer, const Executor& executor) = 0;

  virtual absl::Status LaunchRecv(se::DeviceAddressBase recv_buffer,
                                  PrimitiveType dtype, size_t count,
                                  RankId peer, const Executor& executor) = 0;
};

}  // namespace xla::gpu

namespace stream_executor {
template <>
struct KernelArgPacking<xla::gpu::GpuDeviceCommunicator*> {
  using Type = xla::gpu::GpuDeviceCommunicator::PackedKernelArg;
  static Type Pack(xla::gpu::GpuDeviceCommunicator* comm) {
    return comm->PackKernelArg();
  }
};
}  // namespace stream_executor

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_GPU_COMMUNICATOR_H_
