/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
#include "tensorflow/core/tfrt/tfrt_session/tfrt_session.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "Eigen/ThreadPool"  // from @eigen_archive
#include "llvm/ADT/STLExtras.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "tensorflow/core/common_runtime/local_session_selection.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/graph_executor/graph_executor.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"
#include "tensorflow/core/tfrt/mlrt/kernel/batch_kernel.h"
#include "tensorflow/core/tfrt/mlrt/kernel/kernel.h"
#include "tensorflow/core/tfrt/run_handler_thread_pool/run_handler_concurrent_work_queue.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/runtime/tf_threadpool_concurrent_work_queue.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tensorflow/core/tfrt/utils/utils.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/thread_annotations.h"
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tensorflow {
namespace {

// Wraps an `Eigen::ThreadPoolInterface` as a
// `tensorflow::thread::ThreadPoolInterface`.
class ThreadPoolInterfaceWrapper : public thread::ThreadPoolInterface {
 public:
  explicit ThreadPoolInterfaceWrapper(Eigen::ThreadPoolInterface* thread_pool)
      : thread_pool_{thread_pool} {
    DCHECK(thread_pool);
  }

  void Schedule(std::function<void()> fn) override {
    return thread_pool().Schedule(std::move(fn));
  }

  void ScheduleWithHint(std::function<void()> fn, int start, int end) override {
    return thread_pool().ScheduleWithHint(std::move(fn), start, end);
  }

  void Cancel() override { thread_pool().Cancel(); }

  int NumThreads() const override { return thread_pool().NumThreads(); }

  int CurrentThreadId() const override {
    return thread_pool().CurrentThreadId();
  }

 private:
  Eigen::ThreadPoolInterface& thread_pool() const {
    DCHECK(thread_pool_);
    return *thread_pool_;
  }

  Eigen::ThreadPoolInterface* thread_pool_ = nullptr;
};

// Inter-op thread pools for a `TfrtSession`.
// NOT thread-safe.
class TfrtSessionInterOpThreadPools {
 public:
  TfrtSessionInterOpThreadPools(int size, bool run_in_caller_thread)
      : thread_pools_(size), run_in_caller_thread_(run_in_caller_thread) {}

  void SetThreadPool(int index, ThreadPoolInterfaceWrapper* thread_pool) {
    thread_pools_.at(index) = thread_pool;
  }

  absl::StatusOr<ThreadPoolInterfaceWrapper*> GetThreadPool(int index) {
    if (index < 0 || index >= thread_pools_.size())
      return errors::InvalidArgument("Invalid thread pool index ", index);
    return thread_pools_[index];
  }

  bool run_in_caller_thread() const { return run_in_caller_thread_; }

 private:
  std::vector<ThreadPoolInterfaceWrapper*> thread_pools_;
  // If true, a thread pool created from the caller thread will be used as the
  // inter-op thread pool.
  bool run_in_caller_thread_;
};

class TfrtSession : public tensorflow::Session {
 public:
  // Besides options, these arguments are passed from those stored in
  // `TfrtSessionfactory`.
  // `runtime` should be non-null, with lifetime exceeding that of TfrtSession.
  // A null `backend_compiler` indicates the default TFRT compiler will be used,
  // with existence consistent during the lifetime of TfrtSession.
  explicit TfrtSession(const SessionOptions& options,
                       tensorflow::tfrt_stub::Runtime* runtime,
                       TfrtDeviceInfraTarget device_target,
                       bool tpu_use_tpu_runner,
                       TfrtSessionInterOpThreadPools inter_op_thread_pools,
                       bool enable_mlrt,
                       tensorflow::BackendCompiler* backend_compiler)
      : runtime_{runtime},
        device_target_{device_target},
        tpu_use_tpu_runner_{tpu_use_tpu_runner},
        inter_op_thread_pools_{std::move(inter_op_thread_pools)},
        enable_mlrt_(enable_mlrt),
        options_{options},
        backend_compiler_(backend_compiler) {}

  Status Create(const GraphDef& graph) override {
    return Create(GraphDef(graph));
  }

  Status Create(GraphDef&& graph) override {
    absl::MutexLock lock(&session_state_lock_);
    return CreateLocked(std::move(graph));
  }

  Status CreateLocked(GraphDef graph)
      TF_EXCLUSIVE_LOCKS_REQUIRED(session_state_lock_) {
    if (graph.node_size() == 0) {
      LOG(ERROR) << "Ignoring empty graph.";
      return absl::OkStatus();
    }
    if (session_state_ == SessionState::kCreated) {
      return errors::AlreadyExists(
          "A Graph has already been created for this session.");
    }
    TF_RETURN_IF_ERROR(CheckNotClosedLocked());

    auto options = GetGraphExecutionOptions();
    tensorflow::tfrt_stub::UpdateTpuTargetByBridgeCompatibility(options, graph);

    // Remove the ConfigureDistributedTPU node as it is placed on the TPU_SYSTEM
    // device which is not supported by the compiler.
    // TODO(188009822): Figure out a cleaner way to address this.
    auto* nodes = graph.mutable_node();
    for (auto it = nodes->begin(), end = nodes->end(); it != end; ++it) {
      if (it->name() == "ConfigureDistributedTPU") {
        nodes->erase(it);
        break;
      }
    }

    auto session_options =
        tensorflow::tfrt_stub::CreateDefaultSessionOptions(options);
    session_options.config.mutable_experimental()
        ->set_optimize_for_static_graph(
            options_.config.experimental().optimize_for_static_graph());
    session_options.config.mutable_experimental()
        ->set_disable_optimize_for_static_graph(
            options_.config.experimental().disable_optimize_for_static_graph());
    LOG_FIRST_N(INFO, 10) << "SessionOptions: "
                          << session_options.config.DebugString();

    // Creating the fallback_state using the original function def library
    // without applying placer or grappler, it is OK for now because it's only
    // used for captured functions in certain tf.data ops
    const auto& fdef_lib = graph.library();
    TF_ASSIGN_OR_RETURN(auto fallback_state,
                        tensorflow::tfrt_stub::FallbackState::Create(
                            session_options, fdef_lib));

    auto kernel_registry = std::make_unique<mlrt::KernelRegistry>();
    // Register infra and standard math kernels
    tensorflow::tf_mlrt::RegisterTfMlrtKernels(*kernel_registry);
    tensorflow::tf_mlrt::RegisterTfMlrtBatchKernels(*kernel_registry);

    auto resource_context = std::make_unique<tfrt::ResourceContext>();
    tfrt_stub::ModelRuntimeContext model_context(
        &options, /*export_dir=*/"unknown_export_dir", resource_context.get());
    // TODO(b/334641254): Offer a Session option that prunes the graph_def.
    model_context.set_graph_def(&graph);
    // In the multi-host case, this prevents local Sessions from running
    // global resource creation functions.
    model_context.set_is_local_session(
        !options_.config.experimental().enable_multi_host());
    TF_RETURN_IF_ERROR(options.runtime->CreateRuntimeResources(model_context));

    // `GraphExecutor::Create()` will preprocess the graph (e.g., apply
    // Placer to the top level graph). `kernel_registry` is required only for
    // synchronous execution right now.
    LOG_FIRST_N(INFO, 10) << "GraphExecutionOptions: " << options;
    TF_ASSIGN_OR_RETURN(
        graph_executor_,
        tensorflow::tfrt_stub::GraphExecutor::Create(
            options, std::move(fallback_state), std::move(resource_context),
            std::move(graph), std::move(kernel_registry)));

    session_state_ = SessionState::kCreated;
    return absl::OkStatus();
  }

  Status Extend(const GraphDef& graph) override {
    return Extend(GraphDef(graph));
  }

  Status Extend(GraphDef&& graph) override {
    absl::MutexLock lock(&session_state_lock_);
    return ExtendLocked(std::move(graph));
  }

  Status ExtendLocked(GraphDef graph)
      TF_EXCLUSIVE_LOCKS_REQUIRED(session_state_lock_) {
    if (session_state_ == SessionState::kCreated) {
      return graph_executor_->Extend(graph);
    }
    return CreateLocked(std::move(graph));
  }

  Status RunInternal(const RunOptions& run_options,
                     const std::vector<std::pair<std::string, Tensor>>& inputs,
                     const std::vector<std::string>& output_tensor_names,
                     const std::vector<std::string>& target_node_names,
                     std::vector<Tensor>* outputs,
                     const thread::ThreadPoolOptions& thread_pool_options) {
    {
      absl::MutexLock lock(&session_state_lock_);
      if (session_state_ == SessionState::kInitialized) {
        return errors::Unavailable("Session not created yet.");
      }
      TF_RETURN_IF_ERROR(CheckNotClosedLocked());
    }
    DCHECK(outputs || output_tensor_names.empty()) << "No outputs in Run()";
    // Run the model.
    tensorflow::tfrt_stub::GraphExecutionRunOptions
        graph_execution_run_options{};
    if (run_options.timeout_in_ms() > 0) {
      graph_execution_run_options.deadline = absl::ToChronoTime(
          absl::Now() + absl::Milliseconds(run_options.timeout_in_ms()));
    }

    // TODO(b/193913357): Provide a way for this configurable rather than
    // hardcoding the work queue.
    std::unique_ptr<tensorflow::tfrt_stub::WorkQueueInterface> work_queue;
    // For *intra* op thread pool, we always use the one in input arg.
    // TODO(juanlishen): Remove unused intra op thread pool creation logic from
    // `ThreadPoolManager`.
    auto* const intra_op_thread_pool = thread_pool_options.intra_op_threadpool;
    // For *inter* op thread pool, we determine it as following.
    if (inter_op_thread_pools_.run_in_caller_thread() ||
        run_options.inter_op_thread_pool() == -1) {
      // Use the caller thread if specified by session or run.
      work_queue = tfrt_stub::WrapDefaultWorkQueue(
          tfrt::CreateSingleThreadedWorkQueue(), intra_op_thread_pool);
    } else if (thread_pool_options.inter_op_threadpool != nullptr) {
      // Prefer user-provided thread pool.
      work_queue =
          std::make_unique<tensorflow::tfrt_stub::TfThreadPoolWorkQueue>(
              /*id=*/tfrt::GetUniqueInt(), intra_op_thread_pool,
              thread_pool_options.inter_op_threadpool);
    } else {
      // Check `run_options` to decide thread pool.
      TF_ASSIGN_OR_RETURN(auto* thread_pool,
                          inter_op_thread_pools_.GetThreadPool(
                              run_options.inter_op_thread_pool()));
      work_queue =
          std::make_unique<tensorflow::tfrt_stub::TfThreadPoolWorkQueue>(
              /*id=*/tfrt::GetUniqueInt(), intra_op_thread_pool, thread_pool);
    }
    graph_execution_run_options.work_queue = work_queue.get();

    std::vector<Tensor> output_tensors;
    TF_RETURN_IF_ERROR(graph_executor_->Run(
        graph_execution_run_options, inputs, output_tensor_names,
        target_node_names, &output_tensors));
    if (outputs) {
      DCHECK_EQ(output_tensors.size(), output_tensor_names.size());
      outputs->swap(output_tensors);
    } else {
      DCHECK(output_tensor_names.empty()) << "No outputs in Run()";
    }

    return absl::OkStatus();
  }

  Status Run(const std::vector<std::pair<std::string, Tensor>>& inputs,
             const std::vector<std::string>& output_tensor_names,
             const std::vector<std::string>& target_node_names,
             std::vector<Tensor>* outputs) override {
    return RunInternal(RunOptions{}, inputs, output_tensor_names,
                       target_node_names, outputs, {});
  }

  // TODO(jingdong): run_options and run_metadata are not fully supported for
  // now. Need to figure out the required features and how to handle them
  // properly.
  Status Run(const RunOptions& run_options,
             const std::vector<std::pair<std::string, Tensor>>& inputs,
             const std::vector<std::string>& output_tensor_names,
             const std::vector<std::string>& target_node_names,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata) override {
    return Run(run_options, inputs, output_tensor_names, target_node_names,
               outputs, run_metadata, /*thread_pool_options=*/{});
  }

  // Both inter/intra op thread pools in `thread_pool_options` should be
  // non-null to make it used.

  // TODO(jingdong): run_options and run_metadata are not fully supported for
  // now. Need to figure out the required features and how to handle them
  // properly.
  Status Run(const RunOptions& run_options,
             const std::vector<std::pair<std::string, Tensor>>& inputs,
             const std::vector<std::string>& output_tensor_names,
             const std::vector<std::string>& target_tensor_names,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata,
             const thread::ThreadPoolOptions& thread_pool_options) override {
    return RunInternal(run_options, inputs, output_tensor_names,
                       target_tensor_names, outputs, thread_pool_options);
  }

  /// \brief Creates a `handle` for invoking the subgraph defined by
  /// `callable_options`.
  // NOTE: This API is still experimental and may change.
  Status MakeCallable(const CallableOptions& callable_options,
                      CallableHandle* out_handle) override {
    absl::MutexLock lock(&callables_lock_);
    *out_handle = next_callable_handle_++;
    assert(callables_.find(*out_handle) == callables_.end());
    callables_[*out_handle] = {callable_options};
    return absl::OkStatus();
  }

  /// \brief Invokes the subgraph named by `handle` with the given options and
  /// input tensors.
  ///
  /// The order of tensors in `feed_tensors` must and `fetch_tensors` will
  /// match the order of names in `CallableOptions::feed()` and
  /// `CallableOptions::fetch()` when this subgraph was created.
  /// NOTE: This API is still experimental and may change.
  Status RunCallable(CallableHandle handle,
                     const std::vector<Tensor>& feed_tensors,
                     std::vector<Tensor>* fetch_tensors,
                     RunMetadata* run_metadata) override {
    return RunCallable(handle, feed_tensors, fetch_tensors, run_metadata, {});
  }

  /// \brief Invokes the subgraph named by `handle` with the given options and
  /// input tensors. User can provide custom threadpool implementation via
  /// thread_pool_options.
  ///
  /// The order of tensors in `feed_tensors` must and `fetch_tensors` will
  /// match the order of names in `CallableOptions::feed()` and
  /// `CallableOptions::fetch()` when this subgraph was created.
  /// NOTE: This API is still experimental and may change.
  Status RunCallable(
      CallableHandle handle, const std::vector<Tensor>& feed_tensors,
      std::vector<Tensor>* fetch_tensors, RunMetadata* run_metadata,
      const thread::ThreadPoolOptions& thread_pool_options) override {
    Callable callable;
    {
      absl::MutexLock lock(&callables_lock_);
      auto it = callables_.find(handle);
      if (it == callables_.end())
        return errors::InvalidArgument("No such callable handle: ", handle);
      callable = it->second;
    }
    if (callable.callable_options.feed_size() != feed_tensors.size())
      return errors::InvalidArgument("Invalid number of feed tensors");

    std::vector<std::pair<std::string, Tensor>> inputs;
    for (const auto& it :
         llvm::zip(callable.callable_options.feed(), feed_tensors)) {
      inputs.emplace_back(std::make_pair(std::get<0>(it), std::get<1>(it)));
    }
    std::vector<std::string> output_tensor_names;
    for (const auto& tensor_name : callable.callable_options.fetch()) {
      output_tensor_names.emplace_back(tensor_name);
    }
    std::vector<std::string> target_node_names;
    for (const auto& node_name : callable.callable_options.target()) {
      target_node_names.emplace_back(node_name);
    }

    return Run(inputs, output_tensor_names, target_node_names, fetch_tensors);
  }

  /// \brief Releases resources associated with the given `handle` in this
  /// session.
  /// NOTE: This API is still experimental and may change.
  Status ReleaseCallable(CallableHandle handle) override {
    absl::MutexLock lock(&callables_lock_);
    auto it = callables_.find(handle);
    if (it == callables_.end())
      return errors::InvalidArgument("No such callable handle: ", handle);
    callables_.erase(it);
    return absl::OkStatus();
  }

  Status Close() override {
    absl::MutexLock lock(&session_state_lock_);
    session_state_ = SessionState::kClosed;
    return absl::OkStatus();
  }
  Status ListDevices(std::vector<DeviceAttributes>* response) override {
    return errors::Unimplemented("TfrtSession::ListDevices is Unimplemented.");
  }
  Status LocalDeviceManager(const DeviceMgr** output) override {
    *output = &graph_executor_->fallback_state().device_manager();
    return absl::OkStatus();
  }

 private:
  tfrt::HostContext* GetHostContext() {
    return runtime_->core_runtime()->GetHostContext();
  }

  tensorflow::tfrt_stub::GraphExecutionOptions GetGraphExecutionOptions()
      const {
    // TODO(jingdong): Check option configurations.
    ::tensorflow::tfrt_stub::GraphExecutionOptions options(runtime_);
    auto& compile_options = options.compile_options;
    compile_options.variable_device =
        DeviceNameUtils::FullName(/*job=*/"localhost", /*replica=*/0,
                                  /*task=*/0, /*type=*/"CPU", /*id=*/0);
    compile_options.enable_grappler = true;
    compile_options.device_target = device_target_;
    compile_options.tpu_fuse_ops = tpu_use_tpu_runner_;
    compile_options.hoist_invariant_ops = true;
    compile_options.sink_in_invariant_ops = false;
    compile_options.cost_threshold = 1024;

    // Enable TpuHostAllocator only for TpuRunner as it is the only
    // implementation that supports the premapped memory optimization.
    compile_options.use_tpu_host_allocator_for_inputs = tpu_use_tpu_runner_;
    options.compile_options.backend_compiler = backend_compiler_;

    options.model_metadata = options_.config.experimental().session_metadata();
    options.enable_mlrt = enable_mlrt_;

    return options;
  }

  Status CheckNotClosedLocked() const
      TF_EXCLUSIVE_LOCKS_REQUIRED(session_state_lock_) {
    if (session_state_ == SessionState::kClosed) {
      return errors::Cancelled("Session has been closed.");
    }
    return absl::OkStatus();
  }

  struct Callable {
    CallableOptions callable_options;
  };

  enum class SessionState {
    kInitialized,  // The initial state, just after ctor invocation.
    kCreated,      // Created with a graph; runnable.
    kClosed,       // `Close()` is called; further operations will be denied.
  };

  mutable absl::Mutex session_state_lock_;
  SessionState session_state_ TF_GUARDED_BY(session_state_lock_) =
      SessionState::kInitialized;

  std::unique_ptr<::tensorflow::tfrt_stub::GraphExecutor> graph_executor_;

  tensorflow::tfrt_stub::Runtime* runtime_ = nullptr;
  const TfrtDeviceInfraTarget device_target_;
  const bool tpu_use_tpu_runner_;
  TfrtSessionInterOpThreadPools inter_op_thread_pools_;

  mutable absl::Mutex callables_lock_;
  CallableHandle next_callable_handle_ TF_GUARDED_BY(callables_lock_) = 0;
  absl::flat_hash_map<CallableHandle, Callable> callables_
      TF_GUARDED_BY(callables_lock_);

  bool enable_mlrt_ = false;
  SessionOptions options_ = SessionOptions();
  tensorflow::BackendCompiler* backend_compiler_ = nullptr;
};

std::unique_ptr<tensorflow::tfrt_stub::WorkQueueInterface>
CreateRunHandlerWorkQueue(const TfrtThreadpoolOptions& session_options) {
  // Use half of the main threads as the default value for the
  // number of complementary threads. The complimentary threads are only used
  // for help to process intra ops if the main threads are blocked.
  int num_complementary_threads =
      std::max(1, session_options.num_main_threads / 2);

  tfrt::tf::RunHandlerThreadWorkQueue::Options options;
  options.num_main_threads =
      session_options.num_main_threads;  // NOMUTANTS--performance tuning
  options.num_complementary_threads = num_complementary_threads;
  options.init_timeout_ms =
      absl::ToInt64Milliseconds(session_options.init_timeout);
  options.max_concurrent_handler =
      session_options.max_concurrent_handler;  // NOMUTANTS--performance tuning
  options.num_sub_thread_pool =
      session_options.num_sub_thread_pool;  // NOMUTANTS--performance tuning

  // The number of threads in each sub thread pool is not specified,
  // we evenly distributed the number of threads to each thread pool by
  // default.
  std::vector<int> num_threads;
  const int num_threads_per_pool =
      options.num_main_threads / options.num_sub_thread_pool;

  num_threads.resize(options.num_sub_thread_pool - 1, num_threads_per_pool);
  num_threads.push_back(options.num_main_threads -
                        (options.num_sub_thread_pool - 1) *
                            num_threads_per_pool);
  options.num_threads_in_sub_thread_pool = num_threads;
  options.sub_thread_request_percentage = {1.0};

  options.use_adaptive_waiting_time = true;

  LOG_FIRST_N(INFO, 10) << "RunHandlerThreadWorkQueue Options: " << options;
  return std::make_unique<tfrt::tf::RunHandlerThreadWorkQueue>(options);
}
}  // namespace

// Manages named thread pools used when creating `TfrtSession`.
class TfrtSessionFactory::ThreadPoolManager {
 public:
  // Updates the thread pools based on the given `SessionOptions`. Returns a
  // `TfrtSessionInterOpThreadPools` that can be used to create a `TfrtSession`.
  absl::StatusOr<TfrtSessionInterOpThreadPools> UpdateAndGetInterOpThreadPools(
      const SessionOptions& options) {
    if (options.config.inter_op_parallelism_threads() > 0) {
      LOG(WARNING) << "TFRT session does not support positive "
                      "inter_op_parallelism_threads for now";
    }
    if (options.config.use_per_session_threads()) {
      return errors::InvalidArgument(
          "TFRT session does not yet support use_per_session_threads()");
    }

    auto session_inter_op_thread_pool_size =
        options.config.session_inter_op_thread_pool_size();

    if (session_inter_op_thread_pool_size > 0) {
      TfrtSessionInterOpThreadPools inter_op_thread_pools{
          session_inter_op_thread_pool_size, /*run_in_caller_thread=*/false};

      for (const auto& it :
           llvm::enumerate(options.config.session_inter_op_thread_pool())) {
        const ThreadPoolOptionProto& pool_options = it.value();
        auto pool_index = it.index();
        auto num_threads = pool_options.num_threads();

        if (num_threads != 0) {
          TF_ASSIGN_OR_RETURN(
              auto* thread_pool,
              GetOrCreateThreadPool(options.env, pool_options, pool_index));
          inter_op_thread_pools.SetThreadPool(pool_index, thread_pool);
        } else {
          // TODO(juanlishen): To be consistent with `DirectSession`, we should
          // pick an appropriate `num_threads` and create a new thread pool in
          // this case.
          inter_op_thread_pools.SetThreadPool(pool_index,
                                              GlobalThreadPool(options));
        }
      }
      return inter_op_thread_pools;
    } else if (options.config.inter_op_parallelism_threads() < 0) {
      return TfrtSessionInterOpThreadPools{/*size=*/0,
                                           /*run_in_caller_thread=*/true};
    } else if (session_inter_op_thread_pool_size == 0) {
      // If session_inter_op_thread_pool_size is 0, add a default thread pool
      // option, so the behavior is consistent with DirectSession.
      TfrtSessionInterOpThreadPools session_thread_pool_options{
          /*size=*/1, /*run_in_caller_thread=*/false};
      session_thread_pool_options.SetThreadPool(0, GlobalThreadPool(options));
      return session_thread_pool_options;
    } else {
      return errors::InvalidArgument(
          "session_inter_op_thread_pool_size must be >= 0");
    }
  }

 private:
  class ThreadPoolWithNumThreads {
   public:
    // `thread_pool` must be non-null.
    ThreadPoolWithNumThreads(int num_thread,
                             std::unique_ptr<thread::ThreadPool> thread_pool)
        : num_threads_(num_thread),
          thread_pool_(std::move(thread_pool)),
          thread_pool_interface_wrapper_(
              ABSL_DIE_IF_NULL(thread_pool_)->AsEigenThreadPool()) {}

    int num_threads() const { return num_threads_; }

    ThreadPoolInterfaceWrapper* thread_pool_interface_wrapper() {
      return &thread_pool_interface_wrapper_;
    }

   private:
    int num_threads_;
    std::unique_ptr<thread::ThreadPool> thread_pool_;
    ThreadPoolInterfaceWrapper thread_pool_interface_wrapper_;
  };

  // Returns a per-process global thread pool, configured by the first session
  // in the process.
  ThreadPoolInterfaceWrapper* GlobalThreadPool(const SessionOptions& options) {
    static thread::ThreadPool* const thread_pool =
        NewThreadPoolFromSessionOptions(options);
    static auto* const wrapper =
        new ThreadPoolInterfaceWrapper{thread_pool->AsEigenThreadPool()};
    return wrapper;
  }

  // Returns a `ThreadPoolInterfaceWrapper` that wraps the thread pool with the
  // name in `pool_options`. Creates and stores a new thread pool if an existing
  // one can't be found.
  absl::StatusOr<ThreadPoolInterfaceWrapper*> GetOrCreateThreadPool(
      Env* env, const ThreadPoolOptionProto& pool_options, int pool_index) {
    const int32_t num_threads = pool_options.num_threads();
    CHECK_GT(num_threads, 0);

    const std::string& name = pool_options.global_name();
    if (name.empty()) {
      return errors::InvalidArgument(
          "TFRT session does not yet support session local thread pool");
    }

    absl::MutexLock lock(&mutex_);

    auto it = named_thread_pools_.find(name);
    // The thread pool with the given name already exists.
    if (it != named_thread_pools_.end()) {
      if (it->second->num_threads() != num_threads) {
        return errors::InvalidArgument(
            "TfrtSession thread pool ", name,
            " configured previously with num_threads=",
            it->second->num_threads(),
            "; cannot re-configure with num_threads=", num_threads);
      }
      return it->second->thread_pool_interface_wrapper();
    }

    // The thread pool with the given name does not yet exist. Create one.
    auto thread_pool = std::make_unique<thread::ThreadPool>(
        env, ThreadOptions(), absl::StrCat("TfrtSessionInter", pool_index),
        num_threads, /*low_latency_hint=*/false,
        /*allocator=*/nullptr);
    auto ret = named_thread_pools_.emplace(
        name, std::make_unique<ThreadPoolWithNumThreads>(
                  num_threads, std::move(thread_pool)));
    CHECK(ret.second);

    return ret.first->second->thread_pool_interface_wrapper();
  }

  mutable absl::Mutex mutex_;
  // For pointer-stability of the map value (when calling
  // `thread_pool_interface_wrapper()`), we add a layer of indirection.
  absl::flat_hash_map<std::string, std::unique_ptr<ThreadPoolWithNumThreads>>
      named_thread_pools_ ABSL_GUARDED_BY(mutex_);
};

TfrtSessionFactory::TfrtSessionFactory()
    : thread_pool_manager_(std::make_unique<ThreadPoolManager>()) {}

// Holds an initializer, which should only be registered before main executes.
// As an internal component of `TfrtSessionFactory`, it does not take
// responsibility for thread safety. (`TfrtSessionFactory` does).
class InitializerRegistry {
 public:
  static InitializerRegistry& Get() {
    static auto* reg = new InitializerRegistry();
    return *reg;
  }

  void Register(TfrtSessionFactory::RuntimeInitializer initializer) {
    DCHECK(initializer_ == nullptr);
    initializer_ = initializer;
  }

  absl::Status RunInitializer(tfrt_stub::Runtime* runtime) {
    LOG(INFO) << "Running Initializer within TfrtSessionFactory.";
    TF_RETURN_IF_ERROR(initializer_ ? initializer_(runtime) : absl::OkStatus());
    return absl::OkStatus();
  }

 private:
  TfrtSessionFactory::RuntimeInitializer initializer_;
};

void TfrtSessionFactory::RegisterInitializer(RuntimeInitializer initializer) {
  InitializerRegistry::Get().Register(std::move(initializer));
}

Status TfrtSessionFactory::InitializeLocked(const TfrtSessionOptions& options) {
  mutex_.AssertHeld();
  if (options.use_tpu) {
    DCHECK(!options.backend_compiler);
    device_target_ = TfrtDeviceInfraTarget::kTpurt;
    tpu_use_tpu_runner_ = true;
  } else if (options.backend_compiler) {
    backend_compiler_ = options.backend_compiler;
  }
  LOG(INFO) << "Start initializing TfrtSession";
  if (options.runtime != nullptr) {
    runtime_ = options.runtime;
  } else if (runtime_ == nullptr) {
    owned_runtime_ = tensorflow::tfrt_stub::Runtime::Create(
        CreateRunHandlerWorkQueue(options.threadpool_options));
    runtime_ = owned_runtime_.get();
  }
  enable_mlrt_ = options.enable_mlrt;
  return absl::OkStatus();
}

bool TfrtSessionFactory::AcceptsOptions(const SessionOptions& options) {
  if (options.target == "tfrt_session") return true;
  if (options.target.empty()) {
    return options.config.experimental().use_tfrt() ||
           GetDefaultLocalSessionImpl() == LocalSessionImpl::kTfrtSession;
  }
  return false;
}

Status TfrtSessionFactory::NewSession(const SessionOptions& options,
                                      Session** out_session)
    TF_LOCKS_EXCLUDED(mutex_) {
  // TODO(b/206499043): `SessionOptions` should be passed to Saved Model to
  // create `FallbackState`.
  if (options.config.intra_op_parallelism_threads() != 0) {
    LOG(WARNING) << "TFRT session ignores intra_op_parallelism_threads. "
                    "Intra-op thread "
                    "pool can only be configured by `Run()`";
  }

  *out_session = nullptr;

  absl::MutexLock lock(&mutex_);
  if (!IsInitialized()) {
    TF_RETURN_IF_ERROR(InitializeLocked({}));
    TF_RETURN_IF_ERROR(InitializerRegistry::Get().RunInitializer(runtime_));
  }

  TF_ASSIGN_OR_RETURN(
      auto inter_op_thread_pools,
      thread_pool_manager_->UpdateAndGetInterOpThreadPools(options));

  auto* backend_compiler = options.config.experimental().enable_multi_host()
                               ? backend_compiler_
                               : nullptr;
  *out_session = new TfrtSession(
      options, runtime_, device_target_, tpu_use_tpu_runner_,
      std::move(inter_op_thread_pools), enable_mlrt_, backend_compiler);
  return absl::OkStatus();
}

namespace {
static TfrtSessionFactory* session_factory = nullptr;
}

tfrt_stub::Runtime* TfrtSessionFactory::GetRuntime() {
  DCHECK(session_factory != nullptr);
  absl::MutexLock lock(&session_factory->mutex_);
  return session_factory->runtime_;
}

Status InitializeTfrtSession(const TfrtSessionOptions& options) {
  DCHECK(session_factory != nullptr);
  absl::MutexLock lock(&session_factory->mutex_);
  DCHECK(!session_factory->IsInitialized());
  return UpdateTfrtSessionOptionsLocked(options);
}

Status UpdateTfrtSessionOptionsLocked(const TfrtSessionOptions& options) {
  DCHECK(session_factory != nullptr);
  session_factory->mutex_.AssertHeld();
  return session_factory->InitializeLocked(options);
}

static const bool kFactoryRgistration = [] {
  session_factory = new TfrtSessionFactory();
  LOG(INFO) << "Registering TfrtSession";
  SessionFactory::Register("tfrt_session", session_factory);
  return true;
}();

}  // namespace tensorflow
