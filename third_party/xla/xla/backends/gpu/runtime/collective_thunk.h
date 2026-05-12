/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_THUNK_H_

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/collective_thunk.pb.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/hlo/ir/collective_op_group_mode.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/rendezvous.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

struct CollectiveConfig {
  // Returns if the collective communication operation is degenerate because all
  // the groups formed by the operation are singleton.
  bool IsDegenerate(int64_t replica_count, int64_t partition_count) const;

  std::vector<PrimitiveType> operand_element_type;
  std::vector<ReplicaGroup> replica_groups;
  CollectiveOpGroupMode group_mode = COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;
  bool use_symmetric_buffer = false;

  CollectiveConfigProto ToProto() const;
  static CollectiveConfig FromProto(const CollectiveConfigProto& proto);
};

CollectiveConfig GetCollectiveConfig(const HloInstruction* hlo,
                                     std::optional<bool> use_global_device_ids);

// Wrap GpuCliqueKey into a unique struct to guarantee we do not accidentally
// try to run multiple unrelated rendezvous for a same key.
struct FirstCallRendezvousKey {
  GpuCliqueKey clique_key;

  template <typename H>
  friend H AbslHashValue(H h, const FirstCallRendezvousKey& key) {
    return H::combine(std::move(h), key.clique_key);
  }
  friend bool operator==(const FirstCallRendezvousKey& a,
                         const FirstCallRendezvousKey& b) {
    return a.clique_key == b.clique_key;
  }
};

//===----------------------------------------------------------------------===//
// CollectiveThunk
//===----------------------------------------------------------------------===//

// Thunk base class for XLA:GPU collective operations.
//
// Also implements Command so it can be recorded directly into command buffers.
// ExecuteOnStream launches the collective eagerly on the compute stream;
// Record() traces RunCollective into a nested CUDA graph and attaches it as a
// child command. Both paths share the first-call rendezvous via
// RunWithCommAndRendezvous to avoid NCCL bootstrap races.
class CollectiveThunk : public Command {
 public:
  struct Buffer {
    int64_t element_count;
    ShapedSlice source_buffer;
    ShapedSlice destination_buffer;
    int64_t source_memory_space;
    int64_t destination_memory_space;

    absl::StatusOr<CollectiveBufferProto> ToProto() const;
    static absl::StatusOr<Buffer> FromProto(
        const CollectiveBufferProto& buffer_proto,
        absl::Span<const BufferAllocation> buffer_allocations);
  };

  using CollectivesMode = DebugOptions::CollectivesMode;
  CollectiveThunk(Kind kind, ThunkInfo thunk_info, std::vector<Buffer> buffers,
                  CommunicationId communication_id = CommunicationId(0),
                  CollectivesMode collectives_mode =
                      DebugOptions::COLLECTIVES_PRIVATE_MEMORY);

  // Logging support.
  static std::string GetDeviceString(const CollectiveParams& params);

  virtual CollectiveCliqueRequests::CliqueRequirements GetCliqueRequirements(
      const GpuCliqueKey& clique_key) {
    return {};
  }

  bool IsTracedCommand() const override { return true; }
  bool requires_initialization() const override { return true; }

  absl::Status Prepare(const PrepareParams& params) override;
  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const ExecuteParams& execute_params, const RecordParams& record_params,
      RecordAction record_action, se::CommandBuffer* command_buffer) override;

  absl::StatusOr<std::vector<Communicator*>> GetCommunicators(
      const ExecuteParams& params) const override;

  const std::vector<Buffer>& buffers() const { return buffers_; }

  BufferUses buffer_uses() const override;

  CommunicationId communication_id() const { return communication_id_; }
  CollectivesMode collectives_mode() const { return collectives_mode_; }

  // Shorthands for checking the collectives memory mode of this thunk.
  bool use_private_memory() const;
  bool use_symmetric_memory() const;
  bool use_peer_memory() const;

 protected:
  // Returns true if the first call to this collective operation has to be
  // guarded with a rendezvous synchronization with other local participants
  // before and after running the collective operation itself.
  //
  // This is done as a workaround for NCCL deadlocks that can be triggered when
  // NCCL kernel execution races with a thunk before or after the collective
  // one that calls CUDA APIs that trigger a deadlock.
  virtual bool RequiresRendezvous() const = 0;

  // Prepares collective operation for execution.
  //
  // At this stage it is possible to request symmetric or multicast memory for
  // the collective buffers. Subclasses override this to request memory needed
  // for one-sided or device-initiated collectives.
  virtual absl::Status PrepareCollective(const PrepareParams& params,
                                         const GpuCliqueKey& clique_key) {
    return absl::OkStatus();
  }

  // Initializes collective operation for execution.
  //
  // At this stage it is possible to resolve buffer slices from a buffer
  // assignment, but the content of all buffers is undefined.
  virtual absl::Status InitializeCollective(const InitializeParams& params,
                                            const GpuCliqueKey& clique_key) {
    return absl::OkStatus();
  }

  // Run collective operation on a given stream.
  //
  // A collective thunk is normally an independent operation in a sense that
  // different instances of the same collective thunk communicate each other.
  // The only exception are SendThunk and RecvThunk. Assume two devices are
  // executing a program contains the following instructions, the Recv from
  // device 1 will release the Send from device 0. Adding first call
  // rendezvous on the SendThunk would cause a runtime deadlock.
  //
  //  Send(src_target={0,1})
  //  Recv(src_target={0,1})
  virtual absl::Status RunCollective(const ExecuteParams& params,
                                     const GpuCliqueKey& clique_key,
                                     se::Stream& stream,
                                     Communicator& comm) = 0;

  virtual const CollectiveConfig& config() const = 0;

  virtual bool CanUseSymmetricBuffer() const { return false; }

 private:
  // Rendezvous with other local participants before/after the first call to
  // the collective operation to avoid NCCL deadlocks. See the comment on
  // RequiresRendezvous() for details. `flag` is `pre_call_rendezvous_flag_`
  // when called before the collective, and `post_call_rendezvous_flag_` after.
  // Does nothing if !RequiresRendezvous() or the flag is already completed.
  absl::Status FirstCallRendezvous(const ExecuteParams& params,
                                   const GpuCliqueKey& clique_key,
                                   absl::string_view label,
                                   RendezvousFlag& flag);

  // Resolves the clique key and communicator for this collective, performs the
  // pre-call first-call rendezvous, invokes `fn` with the resolved clique/comm,
  // and then performs the post-call rendezvous. Shared by ExecuteOnStream (for
  // eager execution) and Record (for command-buffer tracing).
  absl::Status RunWithCommAndRendezvous(
      const ExecuteParams& params,
      absl::FunctionRef<absl::Status(const GpuCliqueKey&, Communicator&)> fn);

  const std::vector<Buffer> buffers_;

  CommunicationId communication_id_;
  CollectivesMode collectives_mode_;

  // Device assignment is owned by PjRtExecutable and never changes between
  // thunk executions, and replica groups are baked into the thunk at compile
  // time. Device groups are the same for all devices, so computed once.
  absl::once_flag device_groups_once_;
  absl::StatusOr<std::vector<std::vector<GlobalDeviceId>>> device_groups_;
};

//===----------------------------------------------------------------------===//

absl::Status IsValidOperand(Shape shape, Thunk::Kind reduction_op);

template <typename CollectiveThunkType, typename OpT>
absl::Status AddOpDescription(absl::Status status, OpT op,
                              int64_t replica_count, int64_t partition_count) {
  if (status.ok()) {
    return status;
  }
  CollectiveOpGroupMode group_mode = CollectiveThunkType::GetGroupMode(op);

  int64_t operand_count = 0;
  std::string str;

  if constexpr (std::is_base_of_v<HloInstruction, std::remove_pointer_t<OpT>>) {
    operand_count = op->operand_count();
    str = op->ToString();
  } else {
    operand_count = op->getNumOperands() / 2;
    str = llvm_ir::DumpToString(op.getOperation());
  }

  return absl::Status(
      status.code(),
      absl::StrFormat(
          "%s\n"
          "%s with replica_count: %d, partition_count: %d, group_mode: %s, "
          "operand_count: %d\n%s",
          status.message(), CollectiveThunkType::GetHloOpName(), replica_count,
          partition_count, CollectiveOpGroupModeToString(group_mode),
          operand_count, str));
}

//===----------------------------------------------------------------------===//

// Helper over GetGpuCliqueKey that builds clique key.
absl::StatusOr<GpuCliqueKey> GetCollectiveGpuCliqueKey(
    const CollectiveParams& params, const CollectiveConfig& collective_config,
    CommunicationId communication_id = CommunicationId(0));

struct DeviceBufferPair {
  PrimitiveType element_type;
  int64_t element_count;
  se::DeviceAddressBase source_buffer;
  se::DeviceAddressBase destination_buffer;
  int64_t source_memory_space;
  int64_t destination_memory_space;
};

absl::StatusOr<std::vector<DeviceBufferPair>> ConvertToDeviceBuffers(
    const BufferAllocations* buffer_allocations,
    const std::vector<CollectiveThunk::Buffer>& buffers,
    const std::vector<PrimitiveType>& element_types);
}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_THUNK_H_
