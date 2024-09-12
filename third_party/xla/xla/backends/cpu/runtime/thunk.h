/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_RUNTIME_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_THUNK_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/resource_use.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/execution_context.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/cpu/collectives_interface.h"
#include "xla/service/cpu/xfeed_manager.h"
#include "xla/service/global_device_id.h"
#include "xla/stream_executor/host/host_kernel_c_api.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace Eigen {
struct ThreadPoolDevice;
}  // namespace Eigen

namespace xla::cpu {

// WARNING: This is under construction. Long term plan for XLA is to unify
// runtimes between different backends and have a shared Thunk interface,
// however for now we chose to have separate Thunk implementations in xla::cpu
// and xla::gpu namespaces with a plan to unify them in the future.

// Thunk is the basic unit of execution for the XLA CPU runtime.
//
// This is thread-compatible. Thunk implementation should expect that it will be
// called concurrently from multiple threads, for different run ids and for
// different devices. For partitioned XLA programs the expectation is that all
// local participants execute simultaneously on different threads and coordinate
// resource acquisition via rendezvous.
//
// This is XLA CPU's counterpart of the XLA GPU runtime Thunk.
class Thunk {
 public:
  enum class Kind {
    kAllGather,
    kAllReduce,
    kAllToAll,
    kCall,
    kCollectivePermute,
    kCopy,
    kConditional,
    kConvolution,
    kCustomCall,
    kDot,
    kFft,
    kInfeed,
    kKernel,
    kOutfeed,
    kPartitionId,
    kReduceScatter,
    kReplicaId,
    kRngGetAndUpdateState,
    kSort,
    kTopK,
    kWhile,
  };

  struct Info {
    std::string op_name;
    std::string module_name;
    int64_t module_id;
  };

  // An abstract task runner that can be used by a ThunkExecutor (including
  // thunk executors for nested computations in conditional or while thunks) for
  // running tasks corresponding to thunk execution. It can be a simple inline
  // executor that runs tasks on the same thread, or a runner backed by a thread
  // pool. By default XLA:CPU uses task runner that shares underlying thread
  // pool with the intra-op thread pool used for compute tasks. We deliberately
  // do not prescribe task runner to be Eigen or any other particular thread
  // pool, and let users make the choice.
  using Task = std::function<void()>;
  using TaskRunner = absl::AnyInvocable<void(Task)>;

  Thunk(Kind kind, Info info);

  Thunk(const Thunk&) = delete;
  Thunk& operator=(const Thunk&) = delete;

  virtual ~Thunk() = default;

  Kind kind() const { return kind_; }
  const Info& info() const { return info_; }

  static std::string_view KindToString(Kind kind);

  // Returns the list of buffers used by a thunk. Thunk executor relies on this
  // information to execute thunks concurrently and to avoid data races.
  using BufferUses = absl::InlinedVector<BufferUse, 4>;
  virtual BufferUses buffer_uses() const = 0;

  // Returns the list of resources used by a thunk. Thunk executor relies on
  // this information to execute thunks concurrently and to avoid data races. In
  // contrast to buffer uses, only a handful of thunks are expected to use
  // resources, so we define a default implementation for `resource_uses()`
  // that returns an empty vector.
  using ResourceUses = absl::InlinedVector<ResourceUse, 4>;
  virtual ResourceUses resource_uses() const { return {}; }

  //===--------------------------------------------------------------------===//
  // FunctionRegistry
  //===--------------------------------------------------------------------===//

  // An API to resolve function pointers required for running ThunkSequence:
  //
  // 1. Host kernels that are executed by a KernelThunk via StreamExecutor APIs.
  // 2. Comparator functions required by a SortThunk.
  //
  // At run time this is typically backed by an LLVM JIT compiler that compiles
  // LLVM IR to function pointers on demand. At compile time, together with
  // thunks themselves, we emit LLVM module(s) and metadata describing all the
  // functions required for running emitted thunks (number of threads, etc.).
  class FunctionRegistry {
   public:
    using Kernel = SE_HOST_Kernel*;

    // TODO(ezhulenev): We rely on legacy IrEmitter to emit comparator
    // functions, and we use legacy compute function ABI. We should emit a
    // much simpler comparator function that only takes compared values.
    using Comparator = void (*)(bool*, /*run_options=*/const void*,
                                /*params=*/const void**,
                                /*buffer_table=*/const void*,
                                /*status=*/const void*,
                                /*prof_counters=*/const void*);

    virtual ~FunctionRegistry() = default;

    virtual absl::StatusOr<Kernel> FindKernel(std::string_view name) {
      return Unimplemented("Host kernels are not supported");
    }

    virtual absl::StatusOr<Comparator> FindComparator(std::string_view name) {
      return Unimplemented("Comparator functions are not supported");
    }
  };

  //===--------------------------------------------------------------------===//
  // CollectiveExecuteParams
  //===--------------------------------------------------------------------===//

  // Parameters capturing all the details required for collective execution of
  // XLA executables (multiple partitions and replicas).
  struct CollectiveExecuteParams {
    static absl::StatusOr<CollectiveExecuteParams> Create(
        const ExecutableRunOptions* run_options);

    RunId run_id;

    int64_t local_device_ordinal;
    GlobalDeviceId global_device_id;

    const DeviceAssignment* device_assignment = nullptr;
    CollectivesInterface* collectives = nullptr;

   private:
    CollectiveExecuteParams(RunId run_id, int64_t local_device_ordinal,
                            GlobalDeviceId global_device_id,
                            const DeviceAssignment* device_assignment,
                            CollectivesInterface* collectives);
  };

  //===--------------------------------------------------------------------===//
  // CustomCallExecuteParams
  //===--------------------------------------------------------------------===//

  // Parameters capturing all the details required for custom call execution of
  // XLA executables.
  struct CustomCallExecuteParams {
    static absl::StatusOr<CustomCallExecuteParams> Create(
        const ExecutableRunOptions* run_options);

    int32_t device_ordinal;
    const Eigen::ThreadPoolDevice* intra_op_thread_pool = nullptr;
    const ffi::ExecutionContext* ffi_execution_context = nullptr;

   private:
    CustomCallExecuteParams(int32_t device_ordinal,
                            const Eigen::ThreadPoolDevice* intra_op_thread_pool,
                            const ffi::ExecutionContext* ffi_execution_context);
  };

  //===--------------------------------------------------------------------===//
  // ExecuteParams
  //===--------------------------------------------------------------------===//

  // ExecuteSession controls the number of task runner threads that can
  // execute thunks concurrently (all thunks in a sequence, including thunks in
  // nested computations). We limit the number of worker threads that process
  // ready thunks concurrently to avoid overheads of launching too many tasks.
  // Once the size of a ready queue exceeds the split threshold, we try to
  // offload processing of the tail of the ready queue to the task runner.
  //
  // We use best-effort strategy to limit the number of worker threads (we rely
  // on non-atomic pair of compare and add operations for efficiency), and don't
  // guarantee that the number of concurrent workers is always below the limit,
  // in some cases it can temporarily go above the limit.
  //
  // Execution session only controls the number of additional workers, and the
  // main thread that kicks off the execution is not counted towards the limit.
  class ExecuteSession {
   public:
    // TODO(ezhulenev): Number of workers and split threshold should be
    // configurable with XLA_FLAGS. Also, we should find representative
    // benchmarks to determine the optimal default values.
    static constexpr int64_t kMaxWorkers = 4;
    static constexpr int64_t kSplitThreshold = 8;

    // We use std::shared_ptr as a "lock" where grabbing a copy of the shared
    // pointer means joining the session executing a thunk sequence. We rely on
    // shared pointer to keep track of the number of workers executing a thunk
    // sequence because it is automatically manages atomic counter for us.
    using Lock = std::shared_ptr<std::nullopt_t>;

    ExecuteSession(int64_t max_workers, int64_t split_threshold);

    // Joins the execute session and increments the number of session workers.
    Lock Join() const { return lock_; }

    // Tries to join the execute session. Returns empty lock if the session
    // has reached the maximum number of workers.
    Lock TryJoin() const {
      return num_workers() >= max_workers_ ? nullptr : lock_;
    }

    int64_t num_workers() const { return lock_.use_count() - 1; }
    int64_t max_workers() const { return max_workers_; }
    int64_t split_threshold() const { return split_threshold_; }

   private:
    Lock lock_;
    int64_t max_workers_;
    int64_t split_threshold_;
  };

  // Parameters passed to Execute. Execute is responsible for launching "work"
  // on device, i.e., it launches host kernels, calls into libraries, etc.
  struct ExecuteParams {
    FunctionRegistry* function_registry = nullptr;
    const BufferAllocations* buffer_allocations = nullptr;
    runtime::XfeedManager* xfeed = nullptr;
    const Eigen::ThreadPoolDevice* intra_op_threadpool = nullptr;
    TaskRunner* task_runner = nullptr;
    CollectiveExecuteParams* collective_params = nullptr;
    CustomCallExecuteParams* custom_call_params = nullptr;
    ExecuteSession session = ExecuteSession(ExecuteSession::kMaxWorkers,
                                            ExecuteSession::kSplitThreshold);
  };

  // An execute event that becomes ready when all tasks are completed.
  using ExecuteEvent = tsl::Chain;

  // Returns non-reference-counted async value ref in constructed state.
  // Returned async value is a per-process singleton stored in a storage with a
  // static duration, and can be safely compared using pointer equality.
  static tsl::AsyncValueRef<ExecuteEvent> OkExecuteEventSingleton();

  // Returns `OkExecuteEventSingleton()` cached by this thunk instance.
  tsl::AsyncValueRef<ExecuteEvent> OkExecuteEvent() const { return ok_event_; }

  bool IsOkExecuteEvent(const tsl::AsyncValueRef<ExecuteEvent>& event) const {
    return event == ok_event_;
  }

  bool IsOkExecuteEvent(tsl::AsyncValuePtr<ExecuteEvent> event) const {
    return event == ok_event_.AsPtr();
  }

  // Thunk execution must be asynchronous and never block the caller thread,
  // especially waiting for work submitted into the `intra_op_threadpool`,
  // because thunks themselves are executed on the same thread pool.
  //
  // Thunk execution completion must be reported via the `ExecuteEvent`.
  virtual tsl::AsyncValueRef<ExecuteEvent> Execute(
      const ExecuteParams& params) = 0;

 protected:
  // Helper struct to keep track of pending tasks and an event that signals
  // completion of the operation to the caller. Useful for thunks that launch
  // multiple tasks and need to signal completion when all tasks are done (see
  // ConvolutionThunk and DotThunk for examples).
  struct ExecuteState {
    explicit ExecuteState(int64_t num_tasks);
    ~ExecuteState();

    void Notify();

    std::atomic<int64_t> pending_tasks;
    tsl::AsyncValueRef<Thunk::ExecuteEvent> event;
  };

  // Encodes thunk info into the TraceMe compatible format.
  std::string TraceMeEncode() const;

  // Returns `true` if thunk should check buffer slices bounds, alignment, etc.
  // In optimized builds, we skip buffer slices checks, and assume that all
  // buffer slices are valid, as overhead of buffer slices checks adds up and
  // become measurable on a hot path of executing tiny thunks.
  static constexpr bool ShouldCheckBufferSlices() {
#ifdef NDEBUG
    return false;
#else
    return true;
#endif  // NDEBUG
  }

 private:
  Kind kind_;
  Info info_;

  tsl::AsyncValueRef<ExecuteEvent> ok_event_;
};

std::ostream& operator<<(std::ostream& os, Thunk::Kind kind);

// A sequence of thunks to execute.
class ThunkSequence : public std::vector<std::unique_ptr<Thunk>> {
 public:
  ThunkSequence() = default;

  // Returns an empty thunk sequence.
  static ThunkSequence Empty() { return ThunkSequence(); }

  // Returns a thunk sequence that contains a single thunk of type `T`. Uses
  // factory constructor `T::Create()` to create the thunk.
  template <typename T, typename... Args>
  static absl::StatusOr<ThunkSequence> Of(Args&&... args) {
    static_assert(std::is_base_of_v<Thunk, T>,
                  "ThunkSequence::Of() requires `T` to be a `Thunk` subclass.");
    TF_ASSIGN_OR_RETURN(auto thunk, T::Create(std::forward<Args>(args)...));
    return ThunkSequence(std::move(thunk));
  }

  using BufferUses = Thunk::BufferUses;
  BufferUses buffer_uses() const;

  using ResourceUses = Thunk::ResourceUses;
  ResourceUses resource_uses() const;

  void Append(ThunkSequence other);

 private:
  explicit ThunkSequence(std::unique_ptr<Thunk> thunk);
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_THUNK_H_
