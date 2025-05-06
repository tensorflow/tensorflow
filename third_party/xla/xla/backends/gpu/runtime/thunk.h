/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_THUNK_H_

#include <cstdint>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/execution_context.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/gtl/int_type.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

// Execution stream id allows to specify what Gpu stream Thunk should be using
// for launching device work (kernels, library calls, etc.). By default all
// thunks use stream #0, which is the default compute stream of an XLA
// executable.
//
// Stream synchronizations are explicit and represented as WaitForStreams thunk
// in a ThunkSequence. When ThunkSequence converted to CommandBuffer, execution
// streams mapped to concurrent execution scopes and barriers between them.
//
// IMPORTANT: Async execution semantics and execution stream id
//
// For async thunks (i.e. thunks corresponding to `all-reduce-start` and
// `all-reduce-done`) execution stream id means NOT a stream where the async
// operation must execute, but a stream that async operation must be
// synchronized with:
//
//   - Start operation must wait for the completion of all launched work on the
//     execution stream id (usually by adding a stream wait) and after that
//     launch async work on implementation defined extra stream (can be borrowed
//     from a pool)
//
//   - Corresponding Done operation must synchronize execution stream id with
//     an implementation defined stream that is running async work, again
//     usually by adding a stream wait.
//
TSL_LIB_GTL_DEFINE_INT_TYPE(ExecutionStreamId, uint64_t);

// Thunk acts as the bridge between IrEmitter and GpuExecutable. It stores the
// metadata IrEmitter generates for GpuExecutable to invoke an HloInstruction.
//
// Thunk provides the Initialize and ExecuteOnStream interface for GpuExecutable
// to initialize and execute the invocation respectively. Its subclasses are
// supposed to override these interfaces to launch a generated kernel or call an
// external library function (such as operations in cuBLAS).
//
// Thunks have three execution stages:
//
// (1) Prepare: at this stage Thunk can request shared resources required at run
//     time, i.e. collective thunks request collective cliques. Executable(s)
//     will coordinate resource acquisition.
//
// (2) Initialize: at this stage Thunk must initialize all internal state
//     required for execution, maybe using resources requested at prepare stage.
//
// (3) Execute: at this stage Thunk must launch "work" on underlying device
//     using given stream, and it's expected that all expensive initialization
//     is completed at earlier stages.
//
// This is thread-compatible. Thunk implementation should expect that it will be
// called concurrently from multiple threads, for different run ids and for
// different devices (stream executors). For partitioned XLA programs the
// expectation is that all local participants execute simultaneously on
// different threads and coordinate resource acquisition via rendezvous.
class Thunk {
 public:
  using ExecutionStreamIdMap =
      absl::flat_hash_map<ExecutionStreamId, se::Stream*>;

  // When default execution stream id is used, operations launched by a thunk
  // must be synchronized with a stream passed in ExecuteOptions.
  static constexpr auto kDefaultExecutionStreamId = ExecutionStreamId(0);

  enum Kind {
    // # go/keep-sorted start
    kAllGather,
    kAllGatherDone,
    kAllGatherStart,
    kAllReduce,
    kAllReduceDone,
    kAllReduceStart,
    kAllToAll,
    kAllToAllDone,
    kAllToAllStart,
    kCholesky,
    kCollectiveBroadcast,
    kCollectiveBroadcastDone,
    kCollectiveBroadcastStart,
    kCollectivePermute,
    kCollectivePermuteDone,
    kCollectivePermuteStart,
    kCommandBuffer,
    kConditional,
    kConvolution,
    kConvolutionReorder,
    kCopy,
    kCopyDone,
    kCuDnn,
    kCubSort,
    kCublasLtMatmul,
    kCustomCall,
    kCustomKernel,
    kDynamicSlice,
    kFft,
    kGemm,
    kGroupDone,
    kGroupStart,
    kHostRecv,
    kHostRecvDone,
    kHostSend,
    kHostSendDone,
    kInfeed,
    kKernel,
    kMemset32BitValue,
    kMemzero,
    kNorm,
    kOutfeed,
    kPartitionId,
    kRaggedAllToAll,
    kRaggedAllToAllDone,
    kRaggedAllToAllStart,
    kRecv,
    kRecvDone,
    kReduceScatter,
    kReduceScatterDone,
    kReduceScatterStart,
    kReplicaId,
    kSend,
    kSendDone,
    kSequential,
    kTriangularSolve,
    kWaitForStreams,
    kWhile
    // go/keep-sorted end
  };

  // TODO(ezhulenev): This should become a part of StreamExecutor library, but
  // for now we keep it here as a Thunk implementation detail. It's not yet
  // clear what else should become a part of "executable source", we likely
  // need to keep some information about available symbols and signatures.
  struct ExecutableSource {
    absl::string_view text;            // PTX for NVIDIA backend
    absl::Span<const uint8_t> binary;  // CUBIN for NVIDIA backends
    BinaryMap dnn_compiled_graphs;
  };

  struct ThunkInfo {
    ThunkInfo() = default;  // Disable implicit constructors.
    static ThunkInfo WithProfileAnnotation(const HloInstruction* instr);

    std::string profile_annotation;

    ExecutionStreamId execution_stream_id = kDefaultExecutionStreamId;
  };

  //===--------------------------------------------------------------------===//
  // ResourceRequests
  //===--------------------------------------------------------------------===//

  // Each individual thunk can request various resources required for execution
  // at prepare stage. XLA executable is responsible for allocating them before
  // initializing and executing thunks.
  class ResourceRequestsInterface {
   public:
    virtual ~ResourceRequestsInterface() = default;
    virtual absl::Status AddClique(const GpuCliqueKey& clique_key) = 0;
  };

  //===--------------------------------------------------------------------===//
  // CollectiveCliques
  //===--------------------------------------------------------------------===//

  // A collection of collective cliques acquired based on resource requests
  // collected from all thunks at prepare stage.
  class CollectiveCliques {
   public:
    CollectiveCliques() = default;
    CollectiveCliques(AcquiredCliquesMap cliques_map,
                      int32_t num_transient_cliques);

    absl::StatusOr<Communicator*> GetComm(const GpuCliqueKey& clique_key,
                                          RankId rank) const;

    // Returns whether peer device memory access is possible between all devices
    // in the clique.
    absl::StatusOr<bool> peer_access_enabled(
        const GpuCliqueKey& clique_key) const;

    bool empty() const { return cliques_map_.empty(); }

    bool num_transient_cliques() const { return num_transient_cliques_; }

   private:
    AcquiredCliquesMap cliques_map_;

    // The number of acquired non-persistent clique. We need to keep track of
    // newly created communicators to insert rendezvous after first
    // initialization, because otherwise we observe deadlocks with NCCL
    // collectives backends.
    int32_t num_transient_cliques_ = 0;
  };

  //===--------------------------------------------------------------------===//
  // CollectiveExecuteParams
  //===--------------------------------------------------------------------===//

  // Parameters capturing all the details required for collective execution of
  // XLA executables (multiple partitions and replicas).
  struct CollectiveExecuteParams {
    // Creates NCCL execution parameters from the run options for the given
    // local device. Returns an error if run options are misconfigured (i.e.
    // missing a global device mapping for a local device ordinal).
    static absl::StatusOr<CollectiveExecuteParams> Create(
        const ServiceExecutableRunOptions& run_options,
        absl::Span<se::Stream* const> async_streams,
        int64_t local_device_ordinal, int64_t collective_max_nchannels = 0,
        int64_t p2p_max_nchannels = 0);

    // A mapping from local device ordinals to global device IDs.
    using GlobalDeviceIdMap = std::map<int32_t, GlobalDeviceId>;

    GpuCollectives* collectives;
    se::StreamExecutor* executor;

    // XLA execution run id allows us to distinguish collective operations
    // from different concurrent executions and avoid deadlocks.
    RunId run_id;

    // Streams for asynchronous collective communications.
    absl::InlinedVector<se::Stream*, 4> async_streams;

    int64_t local_device_ordinal;
    GlobalDeviceId global_device_id;

    const DeviceAssignment* device_assn;
    const GlobalDeviceIdMap* global_device_id_map;
    const CliqueIdCallback* nccl_clique_id_callback;

    int64_t collective_max_nchannels;
    int64_t p2p_max_nchannels;

   private:
    CollectiveExecuteParams(GpuCollectives* collectives,
                            se::StreamExecutor* executor, RunId run_id,
                            absl::Span<se::Stream* const> async_streams,
                            int64_t local_device_ordinal,
                            GlobalDeviceId global_device_id,
                            const DeviceAssignment* device_assn,
                            const GlobalDeviceIdMap* global_device_id_map,
                            const CliqueIdCallback* nccl_clique_id_callback,
                            int64_t collective_max_nchannels,
                            int64_t p2p_max_nchannels);
  };

  //===--------------------------------------------------------------------===//
  // PrepareParams
  //===--------------------------------------------------------------------===//

  // Parameters passed to Prepare. At thunk prepare time we do not launch any
  // work or do any expensive initialization and only pass resource requirements
  // back to executable, i.e. request collective cliques required at run time.
  struct PrepareParams {
    // Parameters for executing collective operations.
    const CollectiveExecuteParams* collective_params = nullptr;
  };

  //===--------------------------------------------------------------------===//
  // InitializeParams
  //===--------------------------------------------------------------------===//

  // Parameters passed to Initialize. At thunk initialization time we do not
  // launch any "work" on device and only initialize thunks for execution, i.e.
  // we pre-load kernels on device and instantiate all command buffers.
  struct InitializeParams {
    se::StreamExecutor* executor = nullptr;
    ExecutableSource src;

    const BufferAllocations* buffer_allocations = nullptr;

    // Main compute stream that will be used, passed via `ExecuteParams` to
    // `ExecuteOnStream`. It can be used to initialize on-device "state" (i.e.
    // various control structures) at command buffer recording time (we use it
    // to initialize NCCL execution plans on device when we trace NCCL
    // operations into command buffers);
    se::Stream* stream = nullptr;

    // Auxiliary stream for tracing command buffers. We use a separate stream to
    // avoid accidental tracing of unrelated activities on a main stream.
    se::Stream* command_buffer_trace_stream = nullptr;

    // Parameters for executing collective operations.
    CollectiveExecuteParams* collective_params = nullptr;

    // Collective cliques acquired based on resource requests.
    CollectiveCliques* collective_cliques = nullptr;

    // XLA FFI execution context.
    const ffi::ExecutionContext* ffi_execution_context = nullptr;

    // Total local device count.
    int local_device_count = 0;

    bool requires_exclusive_lock_on_gpu = false;
  };

  //===--------------------------------------------------------------------===//
  // ExecuteParams
  //===--------------------------------------------------------------------===//

  // Parameters passed to ExecuteOnStream. ExecuteOnStream is responsible for
  // launching "work" on device, i.e. it launches kernels, executes command
  // buffers and calls into libraries (cuBLAS, cuDNN etc.).
  struct ExecuteParams {
    // Constructs execute parameters from an executable run options. Return
    // error if run options are misconfigured.
    static ExecuteParams Create(
        const ServiceExecutableRunOptions& run_options,
        const BufferAllocations& buffer_allocations, se::Stream* stream,
        se::Stream* command_buffer_trace_stream,
        CollectiveExecuteParams* collective_params,
        CollectiveCliques* collective_cliques,
        ExecutionStreamIdMap additional_compute_streams = {});

    // Constructs execute parameters from an existing parameters but with
    // different buffer allocations.
    static ExecuteParams CloneWithNewAllocations(
        const ExecuteParams& params,
        const BufferAllocations& buffer_allocations);

    const BufferAllocations* buffer_allocations;  // never null

    // Main compute stream on which thunks launch operations.
    se::Stream* stream;

    // Auxiliary stream for tracing command buffers. We use a separate stream to
    // avoid accidental tracing of unrelated activities on a main stream.
    se::Stream* command_buffer_trace_stream;

    // Parameters for executing collective operations.
    CollectiveExecuteParams* collective_params;

    // Collective cliques acquired based on resource requests.
    CollectiveCliques* collective_cliques;

    // Streams for moving data between host and device.
    se::Stream* device_to_host_stream;
    se::Stream* host_to_device_stream;

    // Send/Recv callbacks passed to XLA from PjRt.
    SendDeviceMemoryFunction* send_device_memory_function;
    RecvDeviceMemoryFunction* recv_device_memory_function;

    // XLA FFI execution context.
    const ffi::ExecutionContext* ffi_execution_context;

    // Additional compute streams on which thunks launch operations.
    ExecutionStreamIdMap additional_compute_streams;

    bool mock_collectives = false;

    bool requires_exclusive_lock_on_gpu = false;

   private:
    friend class CommandBufferThunk;

    ExecuteParams(const BufferAllocations* buffer_allocations,
                  se::Stream* stream, se::Stream* command_buffer_trace_stream,
                  CollectiveExecuteParams* collective_params,
                  CollectiveCliques* collective_cliques,
                  se::Stream* device_to_host_stream,
                  se::Stream* host_to_device_stream,
                  SendDeviceMemoryFunction* send_device_memory_function,
                  RecvDeviceMemoryFunction* recv_device_memory_function,
                  const ffi::ExecutionContext* ffi_execution_context,
                  ExecutionStreamIdMap additional_compute_streams = {},
                  bool mock_collectives = false,
                  bool requires_exclusive_lock_on_gpu = false);
  };

  //===--------------------------------------------------------------------===//

  Thunk(Kind kind, ThunkInfo thunk_info)
      : kind_(kind),
        profile_annotation_(thunk_info.profile_annotation),
        execution_stream_id_(thunk_info.execution_stream_id) {}
  virtual ~Thunk() = default;
  Thunk(const Thunk&) = delete;
  Thunk& operator=(const Thunk&) = delete;

  virtual std::string ToString(int indent) const { return ""; }
  Kind kind() const { return kind_; }
  absl::string_view profile_annotation() const { return profile_annotation_; }

  // Prepares thunk for execution.
  //
  // This may be called multiple times. Its main purpose is to pass resource
  // requests up to the parent executable so it can acquire them before
  // initialization and execution.
  virtual absl::Status Prepare(const PrepareParams& params,
                               ResourceRequestsInterface& resource_requests) {
    return absl::OkStatus();
  }

  // Initializes thunk for execution.
  //
  // This may be called multiple times. Its main purpose is to give us a chance
  // to do initialization outside of ExecuteOnStream() so that the
  // time spent initializing doesn't count towards our execution profile.
  //
  // Precondition: Prepare(initialize_params) has been called.
  virtual absl::Status Initialize(const InitializeParams& params) {
    return absl::OkStatus();
  }

  // Executes thunk on the given stream. This method must be called after
  // Initialize and can be called multiple times over Thunk's lifetime.
  //
  // Precondition: Initialize(initialize_params) has been called.
  virtual absl::Status ExecuteOnStream(const ExecuteParams& params) = 0;

  static absl::string_view KindToString(Thunk::Kind kind);

  ExecutionStreamId execution_stream_id() const { return execution_stream_id_; }
  void set_execution_stream_id(ExecutionStreamId execution_stream_id) {
    execution_stream_id_ = execution_stream_id;
  }

  static absl::StatusOr<se::Stream*> GetStreamForExecution(
      ExecutionStreamId stream_id, const ExecuteParams& params);

  // Returns `true` if this thunk requires inter-GPU communication.
  bool IsCollective() const;

  // Invokes `fn` with this thunk and all nested thunks.
  virtual void ForAllThunks(absl::FunctionRef<void(const Thunk*)> fn) const;

  // A helper function to get the `GpuCollectives*` pointer from the
  // thunk parameters. Returns an error if collectives API is not provided.
  template <typename Params>
  static absl::StatusOr<GpuCollectives*> GetGpuCollectives(
      const Params& params) {
    if (params.collective_params == nullptr) {
      return Internal("Collective params are not provided");
    }
    if (params.collective_params->collectives == nullptr) {
      return Internal("Collectives API is not provided");
    }
    return params.collective_params->collectives;
  }

  // Serializes the thunk into the given ThunkProto. `thunk_proto` is an
  // out-parameter to make the API compatible with Protobuf Arena allocation.
  virtual absl::Status ToProto(ThunkProto* thunk_proto) const {
    thunk_proto->mutable_thunk_info()->set_execution_stream_id(
        execution_stream_id_.value());
    thunk_proto->mutable_thunk_info()->set_profile_annotation(
        profile_annotation_);
    return absl::OkStatus();
  }

 private:
  Kind kind_;
  std::string profile_annotation_;
  ExecutionStreamId execution_stream_id_;
};

// A sequence of thunks.
using ThunkSequence = std::vector<std::unique_ptr<Thunk>>;

std::ostream& operator<<(std::ostream& os, Thunk::Kind kind);

// A struct that defines a shaped slice, i.e., a BufferAllocation::Slice and its
// shape.
struct ShapedSlice {
  BufferAllocation::Slice slice;
  Shape shape;
};

// Returns if the thunk implements a reduction collective (all-reduce or
// reduce-scatter).
bool IsReductionCollective(Thunk::Kind kind);
}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_THUNK_H_
