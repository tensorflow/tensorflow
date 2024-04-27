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

#ifndef XLA_SERVICE_GPU_RUNTIME_NCCL_COLLECTIVE_THUNK_H_
#define XLA_SERVICE_GPU_RUNTIME_NCCL_COLLECTIVE_THUNK_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/runtime/nccl_api.h"
#include "xla/service/gpu/runtime/nccl_clique_key.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/rendezvous.h"
#include "xla/shape.h"
#include "xla/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/translate/mhlo_to_hlo/attribute_exporter.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

class NcclClique;

struct NcclCollectiveConfig {
  int64_t operand_count;
  std::vector<PrimitiveType> operand_element_type;
  std::vector<ReplicaGroup> replica_groups;
  RendezvousKey::CollectiveOpKind collective_op_kind;
  int64_t op_id;
  CollectiveOpGroupMode group_mode;

  template <typename OpT>
  void SetCollectiveOpKindAndID(OpT op);
  void SetCollectiveOpKindAndID(const HloCollectivePermuteInstruction* instr);
  void SetCollectiveOpKindAndID(const HloSendRecvInstruction* instr);
  bool IsDegenerate(int64_t replica_count, int64_t partition_count) const;
};

template <typename OpT>
void NcclCollectiveConfig::SetCollectiveOpKindAndID(OpT op) {
  if (op.getChannelId()) {
    collective_op_kind = RendezvousKey::kCrossModule;
    op_id = static_cast<int64_t>(op.getChannelId()->getHandle());
  } else {
    collective_op_kind = RendezvousKey::kCrossReplica;
    mlir::ModuleOp parent = op->template getParentOfType<mlir::ModuleOp>();
    mlir::IntegerAttr unique_id =
        parent->getAttrOfType<mlir::IntegerAttr>("hlo.unique_id");
    op_id = static_cast<int64_t>(unique_id.getInt());
  }
}

NcclCollectiveConfig GetNcclCollectiveConfig(
    const HloInstruction* hlo, std::optional<bool> use_global_device_ids);

template <typename OpT>
NcclCollectiveConfig GetNcclCollectiveConfigForMlir(
    OpT op, std::optional<bool> use_global_device_ids) {
  NcclCollectiveConfig config;
  config.operand_count = op.getInputs().size();
  config.operand_element_type.reserve(config.operand_count);
  for (int i = 0; i < config.operand_count; i++) {
    const Shape shape = GetShape(op.getInputs()[i]);
    config.operand_element_type.push_back(shape.element_type());
  }
  config.replica_groups = ConvertReplicaGroups(op.getReplicaGroups()).value();
  config.SetCollectiveOpKindAndID(op);
  config.group_mode = GetCollectiveOpGroupMode(op.getChannelId().has_value(),
                                               use_global_device_ids)
                          .value();
  return config;
}

// This wraps the ncclCommHandle object along with other information
// that could be useful.
struct NcclCommHandleWrapper {
  NcclCommHandleWrapper(NcclApi::NcclCommHandle handle, bool is_local)
      : comm_handle(handle), is_local(is_local) {}

  // Communicator handle.
  NcclApi::NcclCommHandle comm_handle;
  // Whether this comm is a node-local comm.
  bool is_local;
};

//===----------------------------------------------------------------------===//
// NcclCollectiveThunk
//===----------------------------------------------------------------------===//

// Forward declare.
class NcclCollectiveDoneThunk;

// Thunk base class for NCCL collective operations.
class NcclCollectiveThunk : public Thunk {
 public:
  NcclCollectiveThunk(Kind kind, ThunkInfo thunk_info, NcclApi* nccl_api,
                      bool is_sync);

  struct Buffer {
    int64_t element_count;
    BufferAllocation::Slice source_buffer;
    BufferAllocation::Slice destination_buffer;
    int64_t source_memory_space;
    int64_t destination_memory_space;
    mlir::Value source_value;
    mlir::Value destination_value;
  };

  // Completion events for asynchronous collective operations (operations
  // launched on a dedicated stream that is synchronized with main compute
  // stream only when needed).
  class AsyncEvents {
   private:
    friend class NcclCollectiveThunk;
    friend class NcclCollectiveDoneThunk;

    absl::Status Initialize(se::StreamExecutor* executor);
    absl::StatusOr<se::Event*> GetEvent(se::StreamExecutor* executor);

   private:
    absl::Mutex mu_;
    absl::node_hash_map<se::StreamExecutor*, se::Event> events_
        ABSL_GUARDED_BY(mu_);
  };

  // Logging support.
  static std::string GetDeviceString(
      const Thunk::CollectiveExecuteParams& params);

  absl::Status Prepare(const PrepareParams& params,
                       ResourceRequests& resource_requests) override;

  absl::Status Initialize(const InitializeParams& params) override;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  NcclApi* nccl_api() const { return nccl_api_; }
  std::shared_ptr<AsyncEvents> async_events() const { return async_events_; }
  void set_async_events(std::shared_ptr<AsyncEvents> async_events) {
    async_events_ = async_events;
  }

 protected:
  virtual absl::Status RunNcclCollective(
      const ExecuteParams& params, se::Stream& stream,
      NcclCommHandleWrapper comm_wrapper) = 0;
  virtual const NcclCollectiveConfig& config() const = 0;
  virtual AsyncStreamKind GetAsyncStreamKind() const {
    return AsyncStreamKind::kCollective;
  }

  // A collective thunk is normally an independent operation in a sense that
  // different instances of the same collective thunk communicate each other.
  // The only exception are SendThunk and RecvThunk. Assume two devices are
  // executing a program contains the following instructions, the Recv from
  // device 1 will release the Send from device 0. Adding first call
  // rendezvous on the SendThunk would cause a runtime deadlock.
  //  Send(src_target={0,1})
  //  Recv(src_target={0,1})
  virtual bool NeedFirstCallRendzevous() const { return true; }

 private:
  bool IsAsync() const { return async_events_ != nullptr; }
  NcclStreamId GetStreamId() const {
    return xla::gpu::GetStreamId(execution_stream_id().value(), IsAsync(),
                                 GetAsyncStreamKind());
  }

  NcclApi* nccl_api_;
  std::shared_ptr<AsyncEvents> async_events_;

  // After a first call to this particular instance of a NCCL collective thunk
  // we do a round of rendezvous to make sure that all participants successfully
  // allocated on-device state required for executing collective operation. This
  // is required to avoid deadlocks when one device goes too far ahead and
  // causes a deadlock in CUDA driver (root cause is mysterious).
  //
  // TODO(ezhulenev): Try to move this flag to NCCL clique as we need to make
  // sure that all NCCL resources are allocated just once.
  RendezvousSingleFlag first_call_rendezvous_flag_;
};

//===----------------------------------------------------------------------===//
// NcclCollectiveDoneThunk
//===----------------------------------------------------------------------===//

class NcclCollectiveDoneThunk : public Thunk {
 public:
  NcclCollectiveDoneThunk(
      Thunk::Kind kind, ThunkInfo thunk_info,
      std::shared_ptr<NcclCollectiveThunk::AsyncEvents> async_events);

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  std::shared_ptr<NcclCollectiveThunk::AsyncEvents> async_events_;
};

//===----------------------------------------------------------------------===//

absl::Status IsValidOperand(mlir::Value operand, Thunk::Kind reduction_op);

absl::Status IsValidOperand(Shape shape, Thunk::Kind reduction_op);

template <typename NcclThunkType, typename OpT>
absl::Status AddOpDescription(absl::Status status, OpT op,
                              int64_t replica_count, int64_t partition_count) {
  if (status.ok()) {
    return status;
  }
  CollectiveOpGroupMode group_mode = NcclThunkType::GetGroupMode(op);

  int64_t operand_count = 0;
  std::string str;

  if constexpr (std::is_base_of_v<HloInstruction, std::remove_pointer_t<OpT>>) {
    operand_count = op->operand_count();
    str = op->ToString();
  } else {
    operand_count = op->getNumOperands() / 2;
    str = llvm_ir::DumpToString(op.getOperation());
  }

  return Status(
      status.code(),
      absl::StrFormat(
          "%s\n"
          "%s with replica_count: %d, partition_count: %d, group_mode: %s, "
          "operand_count: %d\n%s",
          status.message(), NcclThunkType::GetHloOpName(), replica_count,
          partition_count, CollectiveOpGroupModeToString(group_mode),
          operand_count, str));
}

//===----------------------------------------------------------------------===//

size_t GetNumLocalParticipants(
    const std::vector<GlobalDeviceId>& participants,
    const std::vector<GlobalDeviceId>* local_devices);  // may be null

// Returns a nccl comm handle and a flag indicating if
// it's a local communicator.
absl::StatusOr<NcclCommHandleWrapper> GetNcclComm(
    const Thunk::CollectiveExecuteParams& params,
    const Thunk::CollectiveCliques& collective_cliques,
    const std::vector<ReplicaGroup>& replica_groups,
    CollectiveOpGroupMode group_mode, NcclStreamId stream_id,
    AsyncStreamKind stream_kind);

struct DeviceBufferPair {
  PrimitiveType element_type;
  int64_t element_count;
  se::DeviceMemoryBase source_buffer;
  se::DeviceMemoryBase destination_buffer;
  int64_t source_memory_space;
  int64_t destination_memory_space;
};

absl::StatusOr<std::vector<DeviceBufferPair>> ConvertToDeviceBuffers(
    const Thunk::ExecuteParams& params,
    const std::vector<NcclCollectiveThunk::Buffer>& buffers,
    const std::vector<PrimitiveType>& element_types);

absl::StatusOr<std::vector<DeviceBufferPair>> ConvertToDeviceBuffers(
    const BufferAllocations* buffer_allocations,
    const std::vector<NcclCollectiveThunk::Buffer>& buffers,
    const std::vector<PrimitiveType>& element_types);

// Registers buffers allocated in collective memory (see ncclMemAlloc) with a
// communicator to enable zero-copy collectives.
//
// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html
Status MaybeRegisterBuffers(NcclApi* nccl_api, int device_ordinal,
                            const std::vector<DeviceBufferPair>& buffers,
                            NcclApi::NcclCommHandle comm);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_NCCL_COLLECTIVE_THUNK_H_
