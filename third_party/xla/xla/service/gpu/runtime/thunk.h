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

#ifndef XLA_SERVICE_GPU_RUNTIME_THUNK_H_
#define XLA_SERVICE_GPU_RUNTIME_THUNK_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/runtime/nccl_api.h"
#include "xla/service/gpu/runtime/nccl_clique.h"
#include "xla/service/gpu/runtime/nccl_clique_key.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/lib/gtl/int_type.h"

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
    kAddressComputation,
    kCholesky,
    kConditional,
    kConvolution,
    kConvolutionReorder,
    kCopy,
    kCopyDone,
    kCommandBuffer,
    kCubSort,
    kCublasLtMatmul,
    kCustomCall,
    kCustomKernel,
    kFft,
    kGemm,
    kInfeed,
    kKernel,
    kMemset32BitValue,
    kMemzero,
    kNcclAllGather,
    kNcclAllGatherStart,
    kNcclAllGatherDone,
    kNcclAllReduce,
    kNcclAllReduceStart,
    kNcclAllReduceDone,
    kNcclCollectiveBroadcast,
    kNcclCollectiveBroadcastStart,
    kNcclCollectiveBroadcastDone,
    kNcclCollectivePermute,
    kNcclCollectivePermuteStart,
    kNcclCollectivePermuteDone,
    kNcclReduceScatter,
    kNcclReduceScatterStart,
    kNcclReduceScatterDone,
    kNcclAllToAll,
    kNcclAllToAllStart,
    kNcclAllToAllDone,
    kNcclSend,
    kNcclSendDone,
    kNcclRecv,
    kNcclRecvDone,
    kNorm,
    kOutfeed,
    kPartitionId,
    kRecv,
    kRecvDone,
    kReplicaId,
    kSequential,
    kSend,
    kSendDone,
    kTriangularSolve,
    kWhile,
    kFusedMHA,
    kWaitForStreams,
    kCuDnn
  };

  // <HLO computation fingerprint, serialized compiled object>.
  using BinaryMap = absl::flat_hash_map<std::string, std::string>;

  // TODO(ezhulenev): This should become a part of StreamExecutor library, but
  // for now we keep it here as a Thunk implementation detail. It's not yet
  // clear what else should become a part of "executable source", we likely
  // need to keep some information about available symbols and signatures.
  struct ExecutableSource {
    std::string_view text;             // PTX for NVIDIA backend
    absl::Span<const uint8_t> binary;  // CUBIN for NVIDIA backends
    BinaryMap dnn_compiled_graphs;
  };

  struct ThunkInfo {
    explicit ThunkInfo(mlir::Operation* op) : op(op) {}
    static ThunkInfo WithProfileAnnotation(mlir::Operation* op);
    static ThunkInfo WithProfileAnnotation(const HloInstruction* instr);

    std::string profile_annotation;
    // TODO(b/304613751): This is only needed by the LMHLO. Remove this when
    // LMHLO is removed from the runtime pipeline.
    mlir::Operation* op;

    ExecutionStreamId execution_stream_id = kDefaultExecutionStreamId;
  };

  //===--------------------------------------------------------------------===//
  // ResourceRequests
  //===--------------------------------------------------------------------===//

  // Each individual thunk can request various resources required for execution
  // at prepare stage. XLA executable is responsible for allocating them before
  // initializing and executing thunks.
  class ResourceRequests {
   public:
    virtual ~ResourceRequests() = default;
    virtual absl::Status AddClique(const NcclCliqueKey& clique_key,
                                   int32_t num_local_participants) = 0;
  };

  //===--------------------------------------------------------------------===//
  // CollectiveCliques
  //===--------------------------------------------------------------------===//

  // A collection of collective cliques acquired based on resource requests
  // collected from all thunks at prepare stage.
  class CollectiveCliques {
   public:
    CollectiveCliques() = default;
    explicit CollectiveCliques(NcclClique::AcquiredCliquesMap cliques_map);

    absl::StatusOr<NcclApi::NcclCommHandle> GetComm(
        const NcclCliqueKey& clique_key, int32_t rank) const;

    // Returns the number of communicators in a collective clique. Returns error
    // if we do not have an acquired clique for a given key.
    absl::StatusOr<size_t> num_communicators(
        const NcclCliqueKey& clique_key) const;

    // Returns whether the clique is a local clique.
    absl::StatusOr<bool> is_local_clique(const NcclCliqueKey& clique_key) const;

    bool empty() const { return cliques_map_.empty(); }

   private:
    NcclClique::AcquiredCliquesMap cliques_map_;
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
        int64_t local_device_ordinal, int64_t collective_max_nchannels = 0,
        int64_t p2p_max_nchannels = 0);

    // A mapping from local device ordinals to global device IDs.
    using GlobalDeviceIdMap = std::map<int32_t, GlobalDeviceId>;

    se::StreamExecutor* executor;

    // XLA execution run id allows us to distinguish collective operations
    // from different concurrent executions and avoid deadlocks.
    RunId run_id;

    int64_t local_device_ordinal;
    GlobalDeviceId global_device_id;

    const DeviceAssignment* device_assn;
    const GlobalDeviceIdMap* global_device_id_map;
    const NcclCliqueIdCallback* nccl_clique_id_callback;

    int64_t collective_max_nchannels;
    int64_t p2p_max_nchannels;

   private:
    CollectiveExecuteParams(se::StreamExecutor* executor, RunId run_id,
                            int64_t local_device_ordinal,
                            GlobalDeviceId global_device_id,
                            const DeviceAssignment* device_assn,
                            const GlobalDeviceIdMap* global_device_id_map,
                            const NcclCliqueIdCallback* nccl_clique_id_callback,
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

  // TODO(ezhulenev): Merge InitializeParams and ExecuteParams as they have
  // almost the same members and tightly coupled.

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
        absl::Span<se::Stream* const> async_streams,
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

    // Streams for asynchronous collective communications.
    // TODO(ezhulenev): Move this into `CollectiveExecuteParams`.
    absl::InlinedVector<se::Stream*, 4> async_comms_streams;

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

    // Additional compute streams on which thunks launch operations.
    ExecutionStreamIdMap additional_compute_streams;

   private:
    friend class CommandBufferThunk;

    ExecuteParams(const BufferAllocations* buffer_allocations,
                  se::Stream* stream, se::Stream* command_buffer_trace_stream,
                  absl::InlinedVector<se::Stream*, 4> async_comms_streams,
                  CollectiveExecuteParams* collective_params,
                  CollectiveCliques* collective_cliques,
                  se::Stream* device_to_host_stream,
                  se::Stream* host_to_device_stream,
                  SendDeviceMemoryFunction* send_device_memory_function,
                  RecvDeviceMemoryFunction* recv_device_memory_function,
                  ExecutionStreamIdMap additional_compute_streams = {});
  };

  //===--------------------------------------------------------------------===//

  // The hlo_instruction argument is meant to be the instruction this thunk was
  // generated from, but Thunk never uses this argument other than to save it
  // to Thunk::hlo_instruction, so it can be null.
  Thunk(Kind kind, ThunkInfo thunk_info)
      : kind_(kind),
        profile_annotation_(thunk_info.profile_annotation),
        op_(thunk_info.op),
        execution_stream_id_(thunk_info.execution_stream_id) {}
  virtual ~Thunk() = default;
  Thunk(const Thunk&) = delete;
  Thunk& operator=(const Thunk&) = delete;

  virtual std::string ToStringExtra(int indent) const { return ""; }
  Kind kind() const { return kind_; }
  std::string_view profile_annotation() const { return profile_annotation_; }

  // Only valid during compilation, i.e., lowering thunks to kernel-launch
  // related XLA runtime custom calls). nullptr at runtime. MLIR codegen will
  // cease the practice of lowering thunks to XLA runtime custom calls.
  mlir::Operation* op() { return op_; }

  // Prepares thunk for execution.
  //
  // This may be called multiple times. Its main purpose is to pass resource
  // requests up to the parent executable so it can acquire them before
  // initialization and execution.
  virtual absl::Status Prepare(const PrepareParams& params,
                               ResourceRequests& resource_requests) {
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

  // Clears metadata that is only valid during compile time.
  virtual void ClearCompileTimeInfo() { op_ = nullptr; }

  static absl::string_view KindToString(Thunk::Kind kind);

  ExecutionStreamId execution_stream_id() const { return execution_stream_id_; }

  static absl::StatusOr<se::Stream*> GetStreamForExecution(
      ExecutionStreamId stream_id, const ExecuteParams& params);

 private:
  Kind kind_;
  std::string profile_annotation_;
  mlir::Operation* op_;
  ExecutionStreamId execution_stream_id_;
};

// A sequence of thunks.
class ThunkSequence : public std::vector<std::unique_ptr<Thunk>> {
 public:
  std::string ToString(int indent = 0,
                       std::function<std::string(const Thunk*)>
                           get_thunk_annotation = nullptr) const;
};

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

#endif  // XLA_SERVICE_GPU_RUNTIME_THUNK_H_
