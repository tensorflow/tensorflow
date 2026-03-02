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
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_multimem_registry.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/backends/gpu/runtime/thunk_kind.pb.h"
#include "xla/core/collectives/communicator.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/execution_context.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/gtl/int_type.h"
#include "xla/tsl/util/unique_any.h"
#include "xla/util.h"
#include "xla/tsl/platform/status_macros.h"

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

// Unique identifier for async events. The same identifier is expected to be
// shared between a pair of StartThunk and corresponding DoneThunk. It is used
// to collect async regions for a CommandBufferThunk.
TSL_LIB_GTL_DEFINE_INT_TYPE(AsyncEventsUniqueId, uint64_t);

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

  using BufferUses = absl::InlinedVector<BufferUse, 4>;
  using ResourceUses = absl::InlinedVector<ResourceUse, 4>;

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
    kBuffersDebugChecksum,
    kBuffersDebugFloatCheck,
    kCollectiveBroadcast,
    kCollectiveBroadcastDone,
    kCollectiveBroadcastStart,
    kCollectiveKernel,
    kCollectiveMetadata,
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
    kHostExecuteDone,
    kHostExecuteStart,
    kHostRecv,
    kHostRecvDone,
    kHostSend,
    kHostSendDone,
    kInfeed,
    kKernel,
    kMemset32BitValue,
    kMemzero,
    kNorm,
    kNvshmemAllReduceDone,
    kNvshmemAllReduceStart,
    kNvshmemCollectivePermute,
    kNvshmemCollectivePermuteDone,
    kNvshmemCollectivePermuteStart,
    kNvshmemRecv,
    kNvshmemRecvDone,
    kNvshmemSend,
    kNvshmemSendDone,
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
    kSelectK,
    kSend,
    kSendDone,
    kSequential,
    kTriangularSolve,
    kWaitForStreams,
    kWhile
    // go/keep-sorted end
  };

  static ThunkKindProto KindToProto(Kind kind);
  static absl::StatusOr<Thunk::Kind> KindFromProto(ThunkKindProto kind);

  // TODO(ezhulenev): This should become a part of StreamExecutor library, but
  // for now we keep it here as a Thunk implementation detail. It's not yet
  // clear what else should become a part of "executable source", we likely
  // need to keep some information about available symbols and signatures.
  struct ExecutableSource {
    absl::string_view text;            // PTX for NVIDIA backend
    absl::Span<const uint8_t> binary;  // CUBIN for NVIDIA backends
    BinaryMap dnn_compiled_graphs;
  };

  // Metadata associated with a Thunk,
  // including profiling and stream execution info.
  struct ThunkInfo {
    ThunkInfo() = default;  // Disable implicit constructors.

    // Deserializes a ThunkInfo from a ThunkInfoProto.
    // Returns an error if the proto is invalid.
    static absl::StatusOr<Thunk::ThunkInfo> FromProto(
        const ThunkInfoProto& proto);

    static ThunkInfo WithProfileAnnotation(const HloInstruction* instr,
                                           ThunkId thunk_id);

    std::string profile_annotation;

    ExecutionStreamId execution_stream_id = kDefaultExecutionStreamId;

    ThunkId thunk_id = ThunkId{0};

    // Serializes a ThunkInfo to a ThunkInfoProto.
    ThunkInfoProto ToProto() const;
  };

  //===--------------------------------------------------------------------===//
  // ExecutionScopedState
  //===--------------------------------------------------------------------===//

  // Thunks themself instantiated once per XLA program (GpuExecutable), and the
  // same Thunk is reused for all concurrent executions. Thunks state is shared
  // between all concurrently executing XLA programs (and must be carefully
  // synchronized). `ExecutionScopedState` is a container that allows thunks to
  // put an arbitrary state that will have an execution scope, i.e. it will be
  // automatically destroyed when GpuExecutable finishes execution. This allows
  // thunks to pass arbitrary state between stages (from prepare to initialize
  // and then to execute), without having to create a globally synchronized map
  // and it also guarantees correct state life time, as leaving state in a map
  // might lead to "leaks", as the map will be destroyed only when the
  // executable is destroyed. It also thread-safe by construction as all thunks
  // for a GPU program run sequentially from a single thread.
  using ExecutionScopedState = absl::flat_hash_map<ThunkId, tsl::UniqueAny>;

  //===--------------------------------------------------------------------===//
  // PrepareParams
  //===--------------------------------------------------------------------===//

  // Parameters passed to Prepare. At thunk prepare time we do not launch any
  // work or do any expensive initialization and only pass resource requirements
  // back to executable, i.e. request collective cliques required at run time.
  struct PrepareParams {
    // Parameters for executing collective operations.
    const CollectiveParams* collective_params = nullptr;
    // Clique requests for preparing collective communicators.
    CollectiveCliqueRequests* collective_clique_requests = nullptr;
    // Collective memory requests for preparing symmetric allocations.
    CollectiveMemoryRequests* collective_memory_requests = nullptr;
    // Multimem registry for preparing multimem objects.
    CollectiveMultimemRegistry* absl_nonnull multimem_registry = nullptr;
    // Stream executor for the thunk.
    se::StreamExecutor* absl_nonnull executor = nullptr;
    // Buffer allocations for the thunk.
    const BufferAllocations* absl_nonnull buffer_allocations = nullptr;
    // Execution scoped state shared between prepare, initialize and execute.
    ExecutionScopedState* execution_scoped_state = nullptr;
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
    CollectiveParams* collective_params = nullptr;

    // Collective cliques acquired based on clique requests.
    CollectiveCliques* collective_cliques = nullptr;

    // Collective memory acquired based on memory requests.
    CollectiveMemory* collective_memory = nullptr;

    // Multimem registry for preparing collective communicators.
    CollectiveMultimemRegistry* multicast_memory_registry = nullptr;

    // XLA FFI execution context.
    const ffi::ExecutionContext* ffi_execution_context = nullptr;

    // Total local device count.
    int local_device_count = 0;

    // Execution scoped state shared between prepare, initialize and execute.
    ExecutionScopedState* execution_scoped_state = nullptr;
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
        CollectiveParams* collective_params,
        CollectiveCliques* collective_cliques,
        CollectiveMemory* collective_memory,
        ExecutionStreamIdMap additional_compute_streams = {},
        ExecutionScopedState* execution_scoped_state = nullptr);

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
    CollectiveParams* collective_params;

    // Collective cliques acquired based on resource requests.
    CollectiveCliques* collective_cliques;

    // Collective memory acquired based on memory requests.
    CollectiveMemory* collective_memory;

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

    // Execution scoped state shared between prepare, initialize and execute.
    ExecutionScopedState* execution_scoped_state = nullptr;

    bool mock_collectives = false;

    int64_t execution_id = 0;

   private:
    friend class CommandBufferThunk;

    ExecuteParams(const BufferAllocations* buffer_allocations,
                  se::Stream* stream, se::Stream* command_buffer_trace_stream,
                  CollectiveParams* collective_params,
                  CollectiveCliques* collective_cliques,
                  CollectiveMemory* collective_memory,
                  se::Stream* device_to_host_stream,
                  se::Stream* host_to_device_stream,
                  SendDeviceMemoryFunction* send_device_memory_function,
                  RecvDeviceMemoryFunction* recv_device_memory_function,
                  const ffi::ExecutionContext* ffi_execution_context,
                  ExecutionStreamIdMap additional_compute_streams = {},
                  ExecutionScopedState* execution_scoped_state = nullptr,
                  bool mock_collectives = false, int64_t execution_id = 0);
  };

  //===--------------------------------------------------------------------===//

  Thunk(Kind kind, ThunkInfo thunk_info)
      : kind_(kind), thunk_info_(std::move(thunk_info)) {}
  virtual ~Thunk() = default;
  Thunk(const Thunk&) = delete;
  Thunk& operator=(const Thunk&) = delete;

  virtual std::string ToString(int indent) const { return ""; }
  Kind kind() const { return kind_; }
  absl::string_view profile_annotation() const {
    return thunk_info_.profile_annotation;
  }
  const ThunkInfo& thunk_info() const { return thunk_info_; }

  // Prepares thunk for execution.
  //
  // This may be called multiple times. Its main purpose is to pass resource
  // requests up to the parent executable so it can acquire them before
  // initialization and execution.
  virtual absl::Status Prepare(const PrepareParams& params) {
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

  // Returns all device buffers used by the thunk.
  //
  // Does not propagate buffers from nested thunks.
  //
  // The order of the buffers in returned vector is consistent across calls.
  virtual BufferUses buffer_uses() const { return {}; }

  static absl::string_view KindToString(Thunk::Kind kind);

  ExecutionStreamId execution_stream_id() const {
    return thunk_info_.execution_stream_id;
  }
  void set_execution_stream_id(ExecutionStreamId execution_stream_id) {
    thunk_info_.execution_stream_id = execution_stream_id;
  }

  static absl::StatusOr<se::Stream*> GetStreamForExecution(
      ExecutionStreamId stream_id, const ExecuteParams& params);

  // Returns `true` if this thunk requires inter-GPU communication.
  bool IsCollective() const;

  // Returns any communicators used during execution.
  virtual absl::StatusOr<std::vector<Communicator*>> GetCommunicators(
      const ExecuteParams& params) const {
    return std::vector<Communicator*>();
  }

  // Type predicate for `Walk` callback.
  template <typename F, typename Arg>
  using Walker = std::enable_if_t<std::is_invocable_v<F, Arg> ||
                                  std::is_invocable_r_v<absl::Status, F, Arg>>;

  // Recursively walks all the thunks nested inside *this one and calls the
  // user provided callback on every thunk. Always starts traversal with *this.
  template <typename F, Walker<F, Thunk*>* = nullptr>
  std::invoke_result_t<F, Thunk*> Walk(F&& callback);
  template <typename F, Walker<F, const Thunk*>* = nullptr>
  std::invoke_result_t<F, const Thunk*> Walk(F&& callback) const;

  // Recursively replaces all nested thunks with the result of applying `fn` to
  // them.
  // An error will leave the transformation in invalid state.
  // InternalError should be used for status.
  virtual absl::Status TransformAllNestedThunks(
      absl::FunctionRef<
          absl::StatusOr<std::unique_ptr<Thunk>>(std::unique_ptr<Thunk>)>
          fn) {
    return absl::OkStatus();
  }

  // Serializes the thunk into a `ThunkProto`.
  virtual absl::StatusOr<ThunkProto> ToProto() const;

  // Serializes the metadata of the thunk into a `ThunkMetadataProto`.
  ThunkMetadataProto ToMetadataProto() const;

  // This declares a deserializer callback that `FromProto` Thunk factory
  // functions can use to deserialize sub messages.
  using Deserializer =
      absl::AnyInvocable<absl::StatusOr<std::unique_ptr<Thunk>>(
          const ThunkProto&) const>;

  using DeserializerWithCustomAllocations =
      absl::AnyInvocable<absl::StatusOr<std::unique_ptr<Thunk>>(
          const ThunkProto&, absl::Span<const BufferAllocation>) const>;

  void add_control_predecessor(const Thunk* control_predecessor) {
    control_predecessors_.push_back(control_predecessor);
  }

  std::vector<const Thunk*> control_predecessors() const {
    return control_predecessors_;
  }

  virtual std::optional<AsyncEventsUniqueId> GetAsyncEventsUniqueId() const {
    return std::nullopt;
  }

  virtual bool IsAsyncStart() const { return false; }

  virtual bool IsAsyncDone() const { return false; }

 protected:
  // Walks all nested thunks and calls `callback` for them.
  virtual absl::Status WalkNested(
      absl::FunctionRef<absl::Status(Thunk*)> callback) {
    return absl::OkStatus();
  }

 private:
  Kind kind_;
  ThunkInfo thunk_info_;

  // The list of control predecessors of the thunk.
  // Thunk needs to maintain the control dependency information because
  // when it is executed by command buffer, and command buffer may execute the
  // sequence in concurrent mode, and we should make sure that it does not
  // violate the control dependency in the original computation.
  std::vector<const Thunk*> control_predecessors_;
};

// A sequence of thunks.
using ThunkSequence = std::vector<std::unique_ptr<Thunk>>;

std::ostream& operator<<(std::ostream& os, Thunk::Kind kind);

// Returns if the thunk implements a reduction collective (all-reduce or
// reduce-scatter).
bool IsReductionCollective(Thunk::Kind kind);

// Returns the metadata from all thunks in the given thunk graph.
ThunkMetadataListProto GetMetadataListProtoFromThunkGraph(
    const Thunk& root_thunk);

//===----------------------------------------------------------------------===//
// Thunk templates implementation.
//===----------------------------------------------------------------------===//

template <typename F, Thunk::Walker<F, Thunk*>*>
std::invoke_result_t<F, Thunk*> Thunk::Walk(F&& callback) {
  if constexpr (std::is_void_v<std::invoke_result_t<F, Thunk*>>) {
    Walk([f = std::forward<F>(callback)](Thunk* thunk) {
      return (f(thunk), absl::OkStatus());
    }).IgnoreError();  // Error can never happen here.
  } else {
    RETURN_IF_ERROR(callback(this));
    return WalkNested(callback);
  }
}

template <typename F, Thunk::Walker<F, const Thunk*>*>
std::invoke_result_t<F, const Thunk*> Thunk::Walk(F&& callback) const {
  return const_cast<Thunk*>(this)->Walk(  // NOLINT
      std::forward<F>(callback));
}

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_THUNK_H_
