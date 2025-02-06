/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/direct_session.h"

#include <algorithm>
#include <atomic>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/collective_executor_mgr.h"
#include "tensorflow/core/common_runtime/collective_param_resolver_local.h"
#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/debugger_state_interface.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_resolver_local.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/executor_factory.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/local_session_selection.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/logging.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/run_handler.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_partition.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/threadpool_options.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/nccl/collective_communicator.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/connected_traceme.h"
#include "tensorflow/core/profiler/lib/device_profiler_session.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

namespace {

auto* direct_session_runs = monitoring::Counter<0>::New(
    "/tensorflow/core/direct_session_runs",
    "The number of times DirectSession::Run() has been called.");

absl::Status NewThreadPoolFromThreadPoolOptions(
    const SessionOptions& options,
    const ThreadPoolOptionProto& thread_pool_options, int pool_number,
    thread::ThreadPool** pool, bool* owned) {
  int32_t num_threads = thread_pool_options.num_threads();
  if (num_threads == 0) {
    num_threads = NumInterOpThreadsFromSessionOptions(options);
  }
  const string& name = thread_pool_options.global_name();
  if (name.empty()) {
    // Session-local threadpool.
    VLOG(1) << "Direct session inter op parallelism threads for pool "
            << pool_number << ": " << num_threads;
    *pool = new thread::ThreadPool(
        options.env, ThreadOptions(), strings::StrCat("Compute", pool_number),
        num_threads, !options.config.experimental().disable_thread_spinning(),
        /*allocator=*/nullptr);
    *owned = true;
    return absl::OkStatus();
  }

  // Global, named threadpool.
  typedef std::pair<int32, thread::ThreadPool*> MapValue;
  static std::map<string, MapValue>* global_pool_map =
      new std::map<string, MapValue>;
  static mutex* mu = new mutex();
  mutex_lock l(*mu);
  MapValue* mvalue = &(*global_pool_map)[name];
  if (mvalue->second == nullptr) {
    mvalue->first = thread_pool_options.num_threads();
    mvalue->second = new thread::ThreadPool(
        options.env, ThreadOptions(), strings::StrCat("Compute", pool_number),
        num_threads, !options.config.experimental().disable_thread_spinning(),
        /*allocator=*/nullptr);
  } else {
    if (mvalue->first != thread_pool_options.num_threads()) {
      return errors::InvalidArgument(
          "Pool ", name,
          " configured previously with num_threads=", mvalue->first,
          "; cannot re-configure with num_threads=",
          thread_pool_options.num_threads());
    }
  }
  *owned = false;
  *pool = mvalue->second;
  return absl::OkStatus();
}

// Function to create a global thread pool for sessions. The thread number is
// set as `num_threads` if `num_threads` > 0, otherwise it will be parsed from
// SessionOptions.
thread::ThreadPool* GlobalThreadPool(const SessionOptions& options,
                                     int32_t num_threads) {
  static thread::ThreadPool* const thread_pool =
      NewThreadPoolFromSessionOptions(options, num_threads);
  return thread_pool;
}

// TODO(vrv): Figure out how to unify the many different functions
// that generate RendezvousKey, since many of them have to be
// consistent with each other.
string GetRendezvousKey(const string& tensor_name,
                        const DeviceAttributes& device_info,
                        const FrameAndIter& frame_iter) {
  return strings::StrCat(device_info.name(), ";",
                         strings::FpToString(device_info.incarnation()), ";",
                         device_info.name(), ";", tensor_name, ";",
                         frame_iter.frame_id, ":", frame_iter.iter_id);
}

}  // namespace

class DirectSessionFactory : public SessionFactory {
 public:
  DirectSessionFactory() {}

  bool AcceptsOptions(const SessionOptions& options) override {
    return options.target.empty() &&
           !options.config.experimental().use_tfrt() &&
           GetDefaultLocalSessionImpl() == LocalSessionImpl::kDirectSession;
  }

  absl::Status NewSession(const SessionOptions& options,
                          Session** out_session) override {
    const auto& experimental_config = options.config.experimental();
    if (experimental_config.has_session_metadata()) {
      if (experimental_config.session_metadata().version() < 0) {
        return errors::InvalidArgument(
            "Session version shouldn't be negative: ",
            experimental_config.session_metadata().DebugString());
      }
      const string key = GetMetadataKey(experimental_config.session_metadata());
      mutex_lock l(sessions_lock_);
      if (!session_metadata_keys_.insert(key).second) {
        return errors::InvalidArgument(
            "A session with the same name and version has already been "
            "created: ",
            experimental_config.session_metadata().DebugString());
      }
    }

    // Must do this before the CPU allocator is created.
    if (options.config.graph_options().build_cost_model() > 0) {
      EnableCPUAllocatorFullStats();
    }
    std::vector<std::unique_ptr<Device>> devices;
    TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(
        options, "/job:localhost/replica:0/task:0", &devices));

    DirectSession* session = new DirectSession(
        options, new StaticDeviceMgr(std::move(devices)), this);
    {
      mutex_lock l(sessions_lock_);
      sessions_.push_back(session);
    }
    *out_session = session;
    return absl::OkStatus();
  }

  absl::Status Reset(const SessionOptions& options,
                     const std::vector<string>& containers) override {
    std::vector<DirectSession*> sessions_to_reset;
    {
      mutex_lock l(sessions_lock_);
      // We create a copy to ensure that we don't have a deadlock when
      // session->Close calls the DirectSessionFactory.Deregister, which
      // acquires sessions_lock_.
      std::swap(sessions_to_reset, sessions_);
    }
    absl::Status s;
    for (auto session : sessions_to_reset) {
      s.Update(session->Reset(containers));
    }
    // TODO(suharshs): Change the Reset behavior of all SessionFactories so that
    // it doesn't close the sessions?
    for (auto session : sessions_to_reset) {
      s.Update(session->Close());
    }
    return s;
  }

  void Deregister(const DirectSession* session) {
    mutex_lock l(sessions_lock_);
    sessions_.erase(std::remove(sessions_.begin(), sessions_.end(), session),
                    sessions_.end());
    if (session->options().config.experimental().has_session_metadata()) {
      session_metadata_keys_.erase(GetMetadataKey(
          session->options().config.experimental().session_metadata()));
    }
  }

 private:
  static string GetMetadataKey(const SessionMetadata& metadata) {
    return absl::StrCat(metadata.name(), "/", metadata.version());
  }

  mutex sessions_lock_;
  std::vector<DirectSession*> sessions_ TF_GUARDED_BY(sessions_lock_);
  absl::flat_hash_set<string> session_metadata_keys_
      TF_GUARDED_BY(sessions_lock_);
};

class DirectSessionRegistrar {
 public:
  DirectSessionRegistrar() {
    SessionFactory::Register("DIRECT_SESSION", new DirectSessionFactory());
  }
};
static DirectSessionRegistrar registrar;

std::atomic_int_fast64_t DirectSession::step_id_counter_(1);

static RunHandlerPool* GetOrCreateRunHandlerPool(
    const SessionOptions& options) {
  int num_inter_threads = 0;
  int num_intra_threads = 0;
  static const int env_num_inter_threads = NumInterOpThreadsFromEnvironment();
  static const int env_num_intra_threads = NumIntraOpThreadsFromEnvironment();
  if (env_num_inter_threads > 0) {
    num_inter_threads = env_num_inter_threads;
  }
  if (env_num_intra_threads > 0) {
    num_intra_threads = env_num_intra_threads;
  }

  if (num_inter_threads == 0) {
    if (options.config.session_inter_op_thread_pool_size() > 0) {
      // Note due to ShouldUseRunHandler we are guaranteed that
      // run_options.inter_op_thread_pool() == 0
      num_inter_threads =
          options.config.session_inter_op_thread_pool(0).num_threads();
    }
    if (num_inter_threads == 0) {
      num_inter_threads = NumInterOpThreadsFromSessionOptions(options);
    }
  }

  if (num_intra_threads == 0) {
    num_intra_threads = options.config.intra_op_parallelism_threads();
    if (num_intra_threads == 0) {
      num_intra_threads = port::MaxParallelism();
    }
  }

  static RunHandlerPool* pool = [&]() {
    LOG(INFO) << "Creating run-handler pool with "
                 "[num_inter_threads, num_intra_threads] as ["
              << num_inter_threads << "," << num_intra_threads << "]";
    return new RunHandlerPool(num_inter_threads, num_intra_threads);
  }();
  return pool;
}

bool DirectSession::ShouldUseRunHandlerPool(
    const RunOptions& run_options) const {
  if (options_.config.use_per_session_threads()) return false;
  if (options_.config.session_inter_op_thread_pool_size() > 0 &&
      run_options.inter_op_thread_pool() > 0)
    return false;
  // Only use RunHandlerPool when:
  // a. Single global thread pool is used for inter-op parallelism.
  // b. When multiple inter_op_thread_pool(s) are created, use it only while
  // running sessions on the default inter_op_thread_pool=0. Typically,
  // servo-team uses inter_op_thread_pool > 0 for model loading.
  // TODO(crk): Revisit whether we'd want to create one (static) RunHandlerPool
  // per entry in session_inter_op_thread_pool() in the future.
  return true;
}

DirectSession::DirectSession(const SessionOptions& options,
                             const DeviceMgr* device_mgr,
                             DirectSessionFactory* const factory)
    : options_(options),
      device_mgr_(device_mgr),
      factory_(factory),
      cancellation_manager_(new CancellationManager()),
      operation_timeout_in_ms_(options_.config.operation_timeout_in_ms()) {
  const int thread_pool_size =
      options_.config.session_inter_op_thread_pool_size();
  if (thread_pool_size > 0) {
    for (int i = 0; i < thread_pool_size; ++i) {
      thread::ThreadPool* pool = nullptr;
      bool owned = false;
      init_error_.Update(NewThreadPoolFromThreadPoolOptions(
          options_, options_.config.session_inter_op_thread_pool(i), i, &pool,
          &owned));
      thread_pools_.emplace_back(pool, owned);
    }
  } else if (options_.config.use_per_session_threads()) {
    thread_pools_.emplace_back(NewThreadPoolFromSessionOptions(options_),
                               true /* owned */);
  } else {
    // Run locally if environment value of TF_NUM_INTEROP_THREADS is negative
    // and config.inter_op_parallelism_threads is unspecified or negative.
    static const int env_num_threads = NumInterOpThreadsFromEnvironment();
    if (options_.config.inter_op_parallelism_threads() < 0 ||
        (options_.config.inter_op_parallelism_threads() == 0 &&
         env_num_threads < 0)) {
      run_in_caller_thread_ = true;
    }

    // `run_in_caller_thread_` means the session is expected to run with single
    // thread, but it will be dispatched to global thread pool if there're
    // multiple executors. To keep consistent behavior, set thread number to 1.
    thread_pools_.emplace_back(
        GlobalThreadPool(options, run_in_caller_thread_ ? 1 : 0),
        false /* owned */);
  }
  // The default value of sync_on_finish will be flipped soon and this
  // environment variable will be removed as well.
  const absl::Status status =
      ReadBoolFromEnvVar("TF_SYNC_ON_FINISH", true, &sync_on_finish_);
  if (!status.ok()) {
    LOG(ERROR) << status.message();
  }
  session_handle_ =
      strings::StrCat("direct", strings::FpToString(random::New64()));
  if (options.config.log_device_placement()) {
    const string mapping_str = device_mgr_->DeviceMappingString();
    string msg;
    if (mapping_str.empty()) {
      msg = "Device mapping: no known devices.";
    } else {
      msg = strings::StrCat("Device mapping:\n", mapping_str);
    }
    if (!logging::LogToListeners(msg)) {
      LOG(INFO) << msg;
    }
  }
  // The client device is a CPU device from which we feed and fetch tensors.
  device_set_.set_client_device(device_mgr_->HostCPU());
  for (auto d : device_mgr_->ListDevices()) {
    devices_.push_back(d);
    device_set_.AddDevice(d);
    d->op_segment()->AddHold(session_handle_);
  }
}

DirectSession::~DirectSession() {
  if (!closed_) Close().IgnoreError();
  for (auto& it : partial_runs_) {
    it.second.reset(nullptr);
  }
  for (auto& it : executors_) {
    it.second.reset();
  }
  callables_.clear();
  for (auto d : device_mgr_->ListDevices()) {
    d->op_segment()->RemoveHold(session_handle_);
  }
  functions_.clear();
  delete cancellation_manager_;
  for (const auto& p_and_owned : thread_pools_) {
    if (p_and_owned.second) delete p_and_owned.first;
  }

  execution_state_.reset(nullptr);
  flib_def_.reset(nullptr);
}

absl::Status DirectSession::Create(const GraphDef& graph) {
  return Create(GraphDef(graph));
}

absl::Status DirectSession::Create(GraphDef&& graph) {
  TF_RETURN_IF_ERROR(init_error_);
  if (graph.node_size() > 0) {
    mutex_lock l(graph_state_lock_);
    if (graph_created_) {
      return errors::AlreadyExists(
          "A Graph has already been created for this session.");
    }
    return ExtendLocked(std::move(graph));
  }
  return absl::OkStatus();
}

absl::Status DirectSession::Extend(const GraphDef& graph) {
  return Extend(GraphDef(graph));
}

absl::Status DirectSession::Extend(GraphDef&& graph) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  mutex_lock l(graph_state_lock_);
  return ExtendLocked(std::move(graph));
}

absl::Status DirectSession::ExtendLocked(GraphDef&& graph) {
  if (finalized_) {
    return errors::FailedPrecondition("Session has been finalized.");
  }
  if (!(flib_def_ && execution_state_)) {
    // If this is the first call, we can initialize the execution state
    // with `graph` and do not need to call `Extend()`.
    GraphExecutionStateOptions options;
    options.device_set = &device_set_;
    options.session_options = &options_;
    options.session_handle = session_handle_;
    TF_RETURN_IF_ERROR(GraphExecutionState::MakeForBaseGraph(
        std::move(graph), options, &execution_state_));
    // NOTE(mrry): The function library created here will be used for
    // all subsequent extensions of the graph. Also, note how using the copy
    // constructor of FunctionLibraryDefinition avoids duplicating the memory
    // that is occupied by its shared_ptr members.
    flib_def_.reset(
        new FunctionLibraryDefinition(execution_state_->flib_def()));
    graph_created_ = true;
  } else {
    std::unique_ptr<GraphExecutionState> state;
    // TODO(mrry): Rewrite GraphExecutionState::Extend() to take `graph` by
    // value and move `graph` in here.
    TF_RETURN_IF_ERROR(execution_state_->Extend(graph, &state));
    execution_state_.swap(state);
    TF_RETURN_IF_ERROR(flib_def_->AddLibrary(graph.library()));
  }
  return absl::OkStatus();
}

absl::Status DirectSession::Run(const NamedTensorList& inputs,
                                const std::vector<string>& output_names,
                                const std::vector<string>& target_nodes,
                                std::vector<Tensor>* outputs) {
  RunMetadata run_metadata;
  return Run(RunOptions(), inputs, output_names, target_nodes, outputs,
             &run_metadata);
}

absl::Status DirectSession::CreateDebuggerState(
    const CallableOptions& callable_options, int64_t global_step,
    int64_t session_run_index, int64_t executor_step_index,
    std::unique_ptr<DebuggerStateInterface>* debugger_state) {
  TF_RETURN_IF_ERROR(DebuggerStateRegistry::CreateState(
      callable_options.run_options().debug_options(), debugger_state));
  std::vector<string> input_names(callable_options.feed().begin(),
                                  callable_options.feed().end());
  std::vector<string> output_names(callable_options.fetch().begin(),
                                   callable_options.fetch().end());
  std::vector<string> target_names(callable_options.target().begin(),
                                   callable_options.target().end());

  TF_RETURN_IF_ERROR(debugger_state->get()->PublishDebugMetadata(
      global_step, session_run_index, executor_step_index, input_names,
      output_names, target_names));
  return absl::OkStatus();
}

absl::Status DirectSession::DecorateAndPublishGraphForDebug(
    const DebugOptions& debug_options, Graph* graph, Device* device) {
  std::unique_ptr<DebugGraphDecoratorInterface> decorator;
  TF_RETURN_IF_ERROR(
      DebugGraphDecoratorRegistry::CreateDecorator(debug_options, &decorator));

  TF_RETURN_IF_ERROR(decorator->DecorateGraph(graph, device));
  TF_RETURN_IF_ERROR(decorator->PublishGraph(*graph, device->name()));
  return absl::OkStatus();
}

absl::Status DirectSession::RunInternal(
    int64_t step_id, const RunOptions& run_options,
    CallFrameInterface* call_frame, ExecutorsAndKeys* executors_and_keys,
    RunMetadata* run_metadata,
    const thread::ThreadPoolOptions& threadpool_options) {
  const uint64 start_time_usecs = options_.env->NowMicros();
  const int64_t executor_step_count =
      executors_and_keys->step_count.fetch_add(1);
  RunState run_state(step_id, &devices_);
  const size_t num_executors = executors_and_keys->items.size();

  tsl::profiler::TraceMeProducer activity(
      // To TraceMeConsumers in ExecutorState::Process/Finish.
      [&] {
        if (options_.config.experimental().has_session_metadata()) {
          const auto& model_metadata =
              options_.config.experimental().session_metadata();
          string model_id = strings::StrCat(model_metadata.name(), ":",
                                            model_metadata.version());
          return tsl::profiler::TraceMeEncode("SessionRun",
                                              {{"id", step_id},
                                               {"_r", 1} /*root_event*/,
                                               {"model_id", model_id}});
        } else {
          return tsl::profiler::TraceMeEncode(
              "SessionRun", {{"id", step_id}, {"_r", 1} /*root_event*/});
        }
      },
      tsl::profiler::ContextType::kTfExecutor, step_id,
      tsl::profiler::TraceMeLevel::kInfo);

  std::unique_ptr<DebuggerStateInterface> debugger_state;
  if (!run_options.debug_options().debug_tensor_watch_opts().empty()) {
    TF_RETURN_IF_ERROR(
        CreateDebuggerState(executors_and_keys->callable_options,
                            run_options.debug_options().global_step(), step_id,
                            executor_step_count, &debugger_state));
  }

  if (run_metadata != nullptr &&
      options_.config.experimental().has_session_metadata()) {
    *run_metadata->mutable_session_metadata() =
        options_.config.experimental().session_metadata();
  }

#ifndef __ANDROID__
  // Set up for collectives if ExecutorsAndKeys declares a key.
  if (executors_and_keys->collective_graph_key !=
      BuildGraphOptions::kNoCollectiveGraphKey) {
    if (run_options.experimental().collective_graph_key() !=
        BuildGraphOptions::kNoCollectiveGraphKey) {
      // If a collective_graph_key was specified in run_options, ensure that it
      // matches what came out of GraphExecutionState::BuildGraph().
      if (run_options.experimental().collective_graph_key() !=
          executors_and_keys->collective_graph_key) {
        return errors::Internal(
            "collective_graph_key in RunOptions ",
            run_options.experimental().collective_graph_key(),
            " should match collective_graph_key from optimized graph ",
            executors_and_keys->collective_graph_key);
      }
    }
    if (!collective_executor_mgr_) {
      collective_executor_mgr_ = CreateProdLocalCollectiveExecutorMgr(
          options_.config, device_mgr_.get(),
          MaybeCreateNcclCommunicator(options_.config));
    }
    run_state.collective_executor.reset(new CollectiveExecutor::Handle(
        collective_executor_mgr_->FindOrCreate(step_id), true /*inherit_ref*/));
  }
#endif

  thread::ThreadPool* pool;
  // Use std::unique_ptr to ensure garbage collection
  std::unique_ptr<thread::ThreadPool> threadpool_wrapper;

  const bool inline_execution_requested =
      run_in_caller_thread_ || run_options.inter_op_thread_pool() == -1;

  if (inline_execution_requested) {
    // We allow using the caller thread only when having a single executor
    // specified.
    if (executors_and_keys->items.size() > 1) {
      pool = thread_pools_[0].first;
    } else {
      VLOG(1) << "Executing Session::Run() synchronously!";
      pool = nullptr;
    }
  } else if (threadpool_options.inter_op_threadpool != nullptr) {
    threadpool_wrapper = std::make_unique<thread::ThreadPool>(
        threadpool_options.inter_op_threadpool);
    pool = threadpool_wrapper.get();
  } else {
    if (run_options.inter_op_thread_pool() < -1 ||
        run_options.inter_op_thread_pool() >=
            static_cast<int32>(thread_pools_.size())) {
      return errors::InvalidArgument("Invalid inter_op_thread_pool: ",
                                     run_options.inter_op_thread_pool());
    }

    pool = thread_pools_[run_options.inter_op_thread_pool()].first;
  }

  const int64_t call_timeout = run_options.timeout_in_ms() > 0
                                   ? run_options.timeout_in_ms()
                                   : operation_timeout_in_ms_;
  absl::optional<absl::Time> deadline;
  if (call_timeout > 0) {
    deadline = absl::Now() + absl::Milliseconds(call_timeout);
  }

  std::unique_ptr<RunHandler> handler;
  if (ShouldUseRunHandlerPool(run_options) &&
      run_options.experimental().use_run_handler_pool()) {
    VLOG(1) << "Using RunHandler to scheduler inter-op closures.";
    handler = GetOrCreateRunHandlerPool(options_)->Get(
        step_id, call_timeout,
        run_options.experimental().run_handler_pool_options());
    if (!handler) {
      return errors::DeadlineExceeded(
          "Could not obtain RunHandler for request after waiting for ",
          call_timeout, "ms.");
    }
  }
  auto* handler_ptr = handler.get();

  Executor::Args::Runner default_runner = nullptr;

  if (pool == nullptr) {
    default_runner = [](const Executor::Args::Closure& c) { c(); };
  } else if (handler_ptr != nullptr) {
    default_runner = [handler_ptr](Executor::Args::Closure c) {
      handler_ptr->ScheduleInterOpClosure(std::move(c));
    };
  } else {
    default_runner = [pool](Executor::Args::Closure c) {
      pool->Schedule(std::move(c));
    };
  }

  // Start parallel Executors.

  // We can execute this step synchronously on the calling thread whenever
  // there is a single device and the timeout mechanism is not used.
  //
  // When timeouts are used, we must execute the graph(s) asynchronously, in
  // order to invoke the cancellation manager on the calling thread if the
  // timeout expires.
  const bool can_execute_synchronously =
      executors_and_keys->items.size() == 1 && call_timeout == 0;

  Executor::Args args;
  args.step_id = step_id;
  args.call_frame = call_frame;
  args.collective_executor =
      (run_state.collective_executor ? run_state.collective_executor->get()
                                     : nullptr);
  args.session_config = &options_.config;
  args.session_state = &session_state_;
  args.session_handle = session_handle_;
  args.tensor_store = &run_state.tensor_store;
  args.step_container = &run_state.step_container;
  args.sync_on_finish = sync_on_finish_;
  args.user_intra_op_threadpool = threadpool_options.intra_op_threadpool;
  args.run_all_kernels_inline = pool == nullptr;
  args.start_time_usecs = start_time_usecs;
  args.deadline = deadline;

  const bool do_trace = (run_options.trace_level() > RunOptions::NO_TRACE);

  bool update_cost_model = false;
  if (options_.config.graph_options().build_cost_model() > 0) {
    const int64_t build_cost_model_every =
        options_.config.graph_options().build_cost_model();
    const int64_t build_cost_model_after =
        options_.config.graph_options().build_cost_model_after();
    int64_t measure_step_count = executor_step_count - build_cost_model_after;
    if (measure_step_count >= 0) {
      update_cost_model =
          ((measure_step_count + 1) % build_cost_model_every == 0);
    }
  }
  if (run_metadata != nullptr &&
      (do_trace || update_cost_model ||
       run_options.report_tensor_allocations_upon_oom())) {
    run_state.collector.reset(
        new StepStatsCollector(run_metadata->mutable_step_stats()));
    args.stats_collector = run_state.collector.get();
  }

  std::unique_ptr<DeviceProfilerSession> device_profiler_session;
  if (run_options.trace_level() >= RunOptions::HARDWARE_TRACE) {
    device_profiler_session = DeviceProfilerSession::Create();
  }

  // Register this step with session's cancellation manager, so that
  // `Session::Close()` will cancel the step.
  CancellationManager step_cancellation_manager(cancellation_manager_);
  if (step_cancellation_manager.IsCancelled()) {
    return errors::Cancelled("Run call was cancelled");
  }
  args.cancellation_manager = &step_cancellation_manager;

  absl::Status run_status;

  auto set_threadpool_args_for_item =
      [&default_runner, &handler](const PerPartitionExecutorsAndLib& item,
                                  Executor::Args* args) {
        // TODO(azaks): support partial run.
        // TODO(azaks): if the device picks its own threadpool, we need to
        // assign
        //     less threads to the main compute pool by default.
        thread::ThreadPool* device_thread_pool =
            item.device->tensorflow_device_thread_pool();
        // TODO(crk): Investigate usage of RunHandlerPool when using device
        // specific thread pool(s).
        if (!device_thread_pool) {
          args->runner = default_runner;
        } else {
          args->runner = [device_thread_pool](Executor::Args::Closure c) {
            device_thread_pool->Schedule(std::move(c));
          };
        }
        if (handler != nullptr) {
          args->user_intra_op_threadpool =
              handler->AsIntraThreadPoolInterface();
        }
      };

  if (can_execute_synchronously) {
    PrivateIntraProcessRendezvous rendezvous(device_mgr_.get());
    args.rendezvous = &rendezvous;

    const auto& item = executors_and_keys->items[0];
    set_threadpool_args_for_item(item, &args);
    run_status = item.executor->Run(args);
  } else {
    core::RefCountPtr<RefCountedIntraProcessRendezvous> rendezvous(
        new RefCountedIntraProcessRendezvous(device_mgr_.get()));
    args.rendezvous = rendezvous.get();

    // `barrier` will delete itself after the final executor finishes.
    Notification executors_done;
    ExecutorBarrier* barrier = new ExecutorBarrier(
        num_executors, rendezvous.get(),
        [&run_state, &executors_done](const absl::Status& ret) {
          {
            mutex_lock l(run_state.mu);
            run_state.status.Update(ret);
          }
          executors_done.Notify();
        });

    for (const auto& item : executors_and_keys->items) {
      set_threadpool_args_for_item(item, &args);
      item.executor->RunAsync(args, barrier->Get());
    }

    WaitForNotification(&executors_done, &run_state, &step_cancellation_manager,
                        call_timeout);
    {
      tf_shared_lock l(run_state.mu);
      run_status = run_state.status;
    }
  }

  if (step_cancellation_manager.IsCancelled()) {
    run_status.Update(errors::Cancelled("Run call was cancelled"));
  }

  if (run_metadata != nullptr && device_profiler_session) {
    TF_RETURN_IF_ERROR(device_profiler_session->CollectData(
        run_metadata->mutable_step_stats()));
  }

  TF_RETURN_IF_ERROR(run_status);

  // Save the output tensors of this run we choose to keep.
  if (!run_state.tensor_store.empty()) {
    TF_RETURN_IF_ERROR(run_state.tensor_store.SaveTensors(
        {executors_and_keys->callable_options.fetch().begin(),
         executors_and_keys->callable_options.fetch().end()},
        &session_state_));
  }

  if (run_state.collector) {
    run_state.collector->Finalize();
  }

  // Build and return the cost model as instructed.
  if (update_cost_model) {
    // Build the cost model
    std::unordered_map<string, const Graph*> device_to_graph;
    for (const PerPartitionExecutorsAndLib& partition :
         executors_and_keys->items) {
      const Graph* graph = partition.graph.get();
      const string& device = partition.flib->device()->name();
      device_to_graph[device] = graph;
    }

    mutex_lock l(executor_lock_);
    run_state.collector->BuildCostModel(&cost_model_manager_, device_to_graph);

    if (run_metadata != nullptr) {
      // annotate stats onto cost graph.
      CostGraphDef* cost_graph = run_metadata->mutable_cost_graph();
      for (const auto& item : executors_and_keys->items) {
        TF_RETURN_IF_ERROR(cost_model_manager_.AddToCostGraphDef(
            item.graph.get(), cost_graph));
      }
    }
  }

  // If requested via RunOptions, output the partition graphs.
  if (run_options.output_partition_graphs()) {
    if (options_.config.experimental().disable_output_partition_graphs()) {
      return errors::InvalidArgument(
          "RunOptions.output_partition_graphs() is not supported when "
          "disable_output_partition_graphs is true.");
    } else if (run_metadata != nullptr) {
      protobuf::RepeatedPtrField<GraphDef>* partition_graph_defs =
          run_metadata->mutable_partition_graphs();
      for (const PerPartitionExecutorsAndLib& exec_and_lib :
           executors_and_keys->items) {
        GraphDef* partition_graph_def = partition_graph_defs->Add();
        exec_and_lib.graph->ToGraphDef(partition_graph_def);
      }
    }
  }
  metrics::UpdateGraphExecTime(options_.env->NowMicros() - start_time_usecs);

  return absl::OkStatus();
}

absl::Status DirectSession::Run(const RunOptions& run_options,
                                const NamedTensorList& inputs,
                                const std::vector<string>& output_names,
                                const std::vector<string>& target_nodes,
                                std::vector<Tensor>* outputs,
                                RunMetadata* run_metadata) {
  return Run(run_options, inputs, output_names, target_nodes, outputs,
             run_metadata, thread::ThreadPoolOptions());
}

absl::Status DirectSession::Run(
    const RunOptions& run_options, const NamedTensorList& inputs,
    const std::vector<string>& output_names,
    const std::vector<string>& target_nodes, std::vector<Tensor>* outputs,
    RunMetadata* run_metadata,
    const thread::ThreadPoolOptions& threadpool_options) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  TF_RETURN_IF_ERROR(CheckGraphCreated("Run()"));
  direct_session_runs->GetCell()->IncrementBy(1);

  // Extract the inputs names for this run of the session.
  std::vector<string> input_tensor_names;
  input_tensor_names.reserve(inputs.size());
  size_t input_size = 0;
  for (const auto& it : inputs) {
    input_tensor_names.push_back(it.first);
    input_size += it.second.AllocatedBytes();
  }
  metrics::RecordGraphInputTensors(input_size);

  // Check if we already have an executor for these arguments.
  ExecutorsAndKeys* executors_and_keys;
  RunStateArgs run_state_args(run_options.debug_options());
  run_state_args.collective_graph_key =
      run_options.experimental().collective_graph_key();

  TF_RETURN_IF_ERROR(GetOrCreateExecutors(input_tensor_names, output_names,
                                          target_nodes, &executors_and_keys,
                                          &run_state_args));
  {
    mutex_lock l(collective_graph_key_lock_);
    collective_graph_key_ = executors_and_keys->collective_graph_key;
  }

  // Configure a call frame for the step, which we use to feed and
  // fetch values to and from the executors.
  FunctionCallFrame call_frame(executors_and_keys->input_types,
                               executors_and_keys->output_types);
  absl::InlinedVector<Tensor, 4UL> feed_args(inputs.size());
  for (const auto& it : inputs) {
    if (it.second.dtype() == DT_RESOURCE) {
      Tensor tensor_from_handle;
      TF_RETURN_IF_ERROR(
          ResourceHandleToInputTensor(it.second, &tensor_from_handle));
      feed_args[executors_and_keys->input_name_to_index[it.first]] =
          tensor_from_handle;
    } else {
      feed_args[executors_and_keys->input_name_to_index[it.first]] = it.second;
    }
  }
  const absl::Status s = call_frame.SetArgs(feed_args);
  if (errors::IsInternal(s)) {
    return errors::InvalidArgument(s.message());
  } else if (!s.ok()) {
    return s;
  }

  const int64_t step_id = step_id_counter_.fetch_add(1);

  if (LogMemory::IsEnabled()) {
    LogMemory::RecordStep(step_id, run_state_args.handle);
  }

  TF_RETURN_IF_ERROR(RunInternal(step_id, run_options, &call_frame,
                                 executors_and_keys, run_metadata,
                                 threadpool_options));

  // Receive outputs.
  if (outputs) {
    std::vector<Tensor> sorted_outputs;
    const absl::Status s = call_frame.ConsumeRetvals(
        &sorted_outputs, /* allow_dead_tensors = */ false);
    if (errors::IsInternal(s)) {
      return errors::InvalidArgument(s.message());
    } else if (!s.ok()) {
      return s;
    }
    const bool unique_outputs =
        output_names.size() == executors_and_keys->output_name_to_index.size();
    // first_indices[i] = j implies that j is the smallest value for which
    // output_names[i] == output_names[j].
    std::vector<int> first_indices;
    if (!unique_outputs) {
      first_indices.reserve(output_names.size());
      for (const auto& name : output_names) {
        first_indices.push_back(
            std::find(output_names.begin(), output_names.end(), name) -
            output_names.begin());
      }
    }
    outputs->clear();
    size_t output_size = 0;
    outputs->reserve(sorted_outputs.size());
    for (int i = 0; i < output_names.size(); ++i) {
      const string& output_name = output_names[i];
      if (first_indices.empty() || first_indices[i] == i) {
        outputs->emplace_back(
            std::move(sorted_outputs[executors_and_keys
                                         ->output_name_to_index[output_name]]));
      } else {
        outputs->push_back((*outputs)[first_indices[i]]);
      }
      output_size += outputs->back().AllocatedBytes();
    }
    metrics::RecordGraphOutputTensors(output_size);
  }

  return absl::OkStatus();
}

absl::Status DirectSession::PRunSetup(const std::vector<string>& input_names,
                                      const std::vector<string>& output_names,
                                      const std::vector<string>& target_nodes,
                                      string* handle) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  TF_RETURN_IF_ERROR(CheckGraphCreated("PRunSetup()"));

  // RunOptions is not available in PRunSetup, so use thread pool 0.
  thread::ThreadPool* pool = thread_pools_[0].first;

  // Check if we already have an executor for these arguments.
  ExecutorsAndKeys* executors_and_keys;
  // TODO(cais): TFDBG support for partial runs.
  DebugOptions debug_options;
  RunStateArgs run_state_args(debug_options);
  run_state_args.is_partial_run = true;
  TF_RETURN_IF_ERROR(GetOrCreateExecutors(input_names, output_names,
                                          target_nodes, &executors_and_keys,
                                          &run_state_args));

  // Create the run state and save it for future PRun calls.
  Executor::Args args;
  args.step_id = step_id_counter_.fetch_add(1);
  PartialRunState* run_state =
      new PartialRunState(input_names, output_names, args.step_id, &devices_);
  run_state->rendez.reset(new IntraProcessRendezvous(device_mgr_.get()));
  {
    mutex_lock l(executor_lock_);
    if (!partial_runs_
             .emplace(run_state_args.handle,
                      std::unique_ptr<PartialRunState>(run_state))
             .second) {
      return errors::Internal("The handle '", run_state_args.handle,
                              "' created for this partial run is not unique.");
    }
  }

  // Start parallel Executors.
  const size_t num_executors = executors_and_keys->items.size();
  ExecutorBarrier* barrier =
      new ExecutorBarrier(num_executors, run_state->rendez.get(),
                          [run_state](const absl::Status& ret) {
                            if (!ret.ok()) {
                              mutex_lock l(run_state->mu);
                              run_state->status.Update(ret);
                            }
                            run_state->executors_done.Notify();
                          });

  args.rendezvous = run_state->rendez.get();
  args.cancellation_manager = cancellation_manager_;
  // Note that Collectives are not supported in partial runs
  // because RunOptions is not passed in so we can't know whether
  // their use is intended.
  args.collective_executor = nullptr;
  args.session_config = &options_.config;
  args.runner = [this, pool](Executor::Args::Closure c) {
    pool->Schedule(std::move(c));
  };
  args.session_state = &session_state_;
  args.session_handle = session_handle_;
  args.tensor_store = &run_state->tensor_store;
  args.step_container = &run_state->step_container;
  if (LogMemory::IsEnabled()) {
    LogMemory::RecordStep(args.step_id, run_state_args.handle);
  }
  args.sync_on_finish = sync_on_finish_;

  if (options_.config.graph_options().build_cost_model()) {
    run_state->collector.reset(new StepStatsCollector(nullptr));
    args.stats_collector = run_state->collector.get();
  }

  for (auto& item : executors_and_keys->items) {
    item.executor->RunAsync(args, barrier->Get());
  }

  *handle = run_state_args.handle;
  return absl::OkStatus();
}

absl::Status DirectSession::PRun(const string& handle,
                                 const NamedTensorList& inputs,
                                 const std::vector<string>& output_names,
                                 std::vector<Tensor>* outputs) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  std::vector<string> parts = str_util::Split(handle, ';');
  const string& key = parts[0];
  // Get the executors for this partial run.
  ExecutorsAndKeys* executors_and_keys;
  PartialRunState* run_state;
  {
    mutex_lock l(executor_lock_);  // could use reader lock
    auto exc_it = executors_.find(key);
    if (exc_it == executors_.end()) {
      return errors::InvalidArgument(
          "Must run 'setup' before performing partial runs!");
    }
    executors_and_keys = exc_it->second.get();

    auto prun_it = partial_runs_.find(handle);
    if (prun_it == partial_runs_.end()) {
      return errors::InvalidArgument(
          "Must run 'setup' before performing partial runs!");
    }
    run_state = prun_it->second.get();

    // Make sure that this is a new set of feeds that are still pending.
    for (const auto& input : inputs) {
      auto it = run_state->pending_inputs.find(input.first);
      if (it == run_state->pending_inputs.end()) {
        return errors::InvalidArgument(
            "The feed ", input.first,
            " was not specified in partial_run_setup.");
      } else if (it->second) {
        return errors::InvalidArgument("The feed ", input.first,
                                       " has already been fed.");
      }
    }
    // Check that this is a new set of fetches that are still pending.
    for (const auto& output : output_names) {
      auto it = run_state->pending_outputs.find(output);
      if (it == run_state->pending_outputs.end()) {
        return errors::InvalidArgument(
            "The fetch ", output, " was not specified in partial_run_setup.");
      } else if (it->second) {
        return errors::InvalidArgument("The fetch ", output,
                                       " has already been fetched.");
      }
    }
  }

  // Check that this new set of fetches can be computed from all the
  // feeds we have supplied.
  TF_RETURN_IF_ERROR(
      CheckFetch(inputs, output_names, executors_and_keys, run_state));

  // Send inputs.
  absl::Status s =
      SendPRunInputs(inputs, executors_and_keys, run_state->rendez.get());

  // Receive outputs.
  if (s.ok()) {
    s = RecvPRunOutputs(output_names, executors_and_keys, run_state, outputs);
  }

  // Save the output tensors of this run we choose to keep.
  if (s.ok()) {
    s = run_state->tensor_store.SaveTensors(output_names, &session_state_);
  }

  {
    mutex_lock l(executor_lock_);
    // Delete the run state if there is an error or all fetches are done.
    bool done = true;
    if (s.ok()) {
      {
        mutex_lock l(run_state->mu);
        if (!run_state->status.ok()) {
          LOG(WARNING) << "An error unrelated to this prun has been detected. "
                       << run_state->status;
        }
      }
      for (const auto& input : inputs) {
        auto it = run_state->pending_inputs.find(input.first);
        it->second = true;
      }
      for (const auto& name : output_names) {
        auto it = run_state->pending_outputs.find(name);
        it->second = true;
      }
      done = run_state->PendingDone();
    }
    if (done) {
      WaitForNotification(&run_state->executors_done, run_state,
                          cancellation_manager_, operation_timeout_in_ms_);
      partial_runs_.erase(handle);
    }
  }

  return s;
}

absl::Status DirectSession::ResourceHandleToInputTensor(
    const Tensor& resource_tensor, Tensor* retrieved_tensor) {
  if (resource_tensor.dtype() != DT_RESOURCE) {
    return errors::InvalidArgument(strings::StrCat(
        "ResourceHandleToInputTensor() received non-DT_RESOURCE Tensor: ",
        resource_tensor.dtype()));
  }

  const ResourceHandle& resource_handle =
      resource_tensor.scalar<ResourceHandle>()();

  if (resource_handle.container() ==
      SessionState::kTensorHandleResourceTypeName) {
    return session_state_.GetTensor(resource_handle.name(), retrieved_tensor);
  } else {
    return errors::InvalidArgument(strings::StrCat(
        "Invalid resource type hash code: ", resource_handle.hash_code(),
        "(name: ", resource_handle.name(),
        " type: ", resource_handle.maybe_type_name(),
        "). Perhaps a resource tensor was being provided as a feed? That is "
        "not currently allowed. Please file an issue at "
        "https://github.com/tensorflow/tensorflow/issues/new, ideally with a "
        "short code snippet that leads to this error message."));
  }
}

absl::Status DirectSession::SendPRunInputs(
    const NamedTensorList& inputs, const ExecutorsAndKeys* executors_and_keys,
    IntraProcessRendezvous* rendez) {
  absl::Status s;
  Rendezvous::ParsedKey parsed;
  // Insert the input tensors into the local rendezvous by their
  // rendezvous key.
  for (const auto& input : inputs) {
    auto it =
        executors_and_keys->input_name_to_rendezvous_key.find(input.first);
    if (it == executors_and_keys->input_name_to_rendezvous_key.end()) {
      return errors::Internal("'", input.first, "' is not a pre-defined feed.");
    }
    const string& input_key = it->second;

    s = Rendezvous::ParseKey(input_key, &parsed);
    if (!s.ok()) {
      rendez->StartAbort(s);
      return s;
    }

    if (input.second.dtype() == DT_RESOURCE) {
      Tensor tensor_from_handle;
      s = ResourceHandleToInputTensor(input.second, &tensor_from_handle);
      if (s.ok()) {
        s = rendez->Send(parsed, Rendezvous::Args(), tensor_from_handle, false);
      }
    } else {
      s = rendez->Send(parsed, Rendezvous::Args(), input.second, false);
    }

    if (!s.ok()) {
      rendez->StartAbort(s);
      return s;
    }
  }
  return absl::OkStatus();
}

absl::Status DirectSession::RecvPRunOutputs(
    const std::vector<string>& output_names,
    const ExecutorsAndKeys* executors_and_keys, PartialRunState* run_state,
    std::vector<Tensor>* outputs) {
  absl::Status s;
  if (!output_names.empty()) {
    outputs->resize(output_names.size());
  }

  Rendezvous::ParsedKey parsed;
  // Get the outputs from the rendezvous
  for (size_t output_offset = 0; output_offset < output_names.size();
       ++output_offset) {
    const string& output_name = output_names[output_offset];
    auto it =
        executors_and_keys->output_name_to_rendezvous_key.find(output_name);
    if (it == executors_and_keys->output_name_to_rendezvous_key.end()) {
      return errors::Internal("'", output_name,
                              "' is not a pre-defined fetch.");
    }
    const string& output_key = it->second;
    Tensor output_tensor;
    bool is_dead;

    s = Rendezvous::ParseKey(output_key, &parsed);
    if (s.ok()) {
      // Fetch data from the Rendezvous.
      s = run_state->rendez->Recv(parsed, Rendezvous::Args(), &output_tensor,
                                  &is_dead, operation_timeout_in_ms_);
      if (is_dead && s.ok()) {
        s = errors::InvalidArgument("The tensor returned for ", output_name,
                                    " was not valid.");
      }
    }
    if (!s.ok()) {
      run_state->rendez->StartAbort(s);
      outputs->clear();
      return s;
    }

    (*outputs)[output_offset] = output_tensor;
  }
  return absl::OkStatus();
}

absl::Status DirectSession::CheckFetch(
    const NamedTensorList& feeds, const std::vector<string>& fetches,
    const ExecutorsAndKeys* executors_and_keys,
    const PartialRunState* run_state) {
  const Graph* graph = executors_and_keys->graph.get();
  const NameNodeMap* name_to_node = &executors_and_keys->name_to_node;

  // Build the set of pending feeds that we haven't seen.
  std::unordered_set<TensorId, TensorId::Hasher> pending_feeds;
  {
    mutex_lock l(executor_lock_);
    for (const auto& input : run_state->pending_inputs) {
      // Skip if the feed has already been fed.
      if (input.second) continue;
      TensorId id(ParseTensorName(input.first));
      auto it = name_to_node->find(id.first);
      if (it == name_to_node->end()) {
        return errors::NotFound("Feed ", input.first, ": not found");
      }
      pending_feeds.insert(id);
    }
  }
  for (const auto& it : feeds) {
    TensorId id(ParseTensorName(it.first));
    pending_feeds.erase(id);
  }

  // Initialize the stack with the fetch nodes.
  std::vector<const Node*> stack;
  for (const string& fetch : fetches) {
    TensorId id(ParseTensorName(fetch));
    auto it = name_to_node->find(id.first);
    if (it == name_to_node->end()) {
      return errors::NotFound("Fetch ", fetch, ": not found");
    }
    stack.push_back(it->second);
  }

  // Any tensor needed for fetches can't be in pending_feeds.
  std::vector<bool> visited(graph->num_node_ids(), false);
  while (!stack.empty()) {
    const Node* n = stack.back();
    stack.pop_back();

    for (const Edge* in_edge : n->in_edges()) {
      const Node* in_node = in_edge->src();
      if (pending_feeds.count({in_node->name(), in_edge->src_output()}) > 0) {
        return errors::InvalidArgument("Fetch ", in_node->name(), ":",
                                       in_edge->src_output(),
                                       " can't be computed from the feeds"
                                       " that have been fed so far.");
      }
      if (!visited[in_node->id()]) {
        visited[in_node->id()] = true;
        stack.push_back(in_node);
      }
    }
  }
  return absl::OkStatus();
}

absl::Status DirectSession::CreateExecutors(
    const CallableOptions& callable_options,
    std::unique_ptr<ExecutorsAndKeys>* out_executors_and_keys,
    std::unique_ptr<FunctionInfo>* out_func_info,
    RunStateArgs* run_state_args) {
  BuildGraphOptions options;
  options.callable_options = callable_options;
  options.use_function_convention = !run_state_args->is_partial_run;
  options.collective_graph_key =
      callable_options.run_options().experimental().collective_graph_key();
  if (options_.config.experimental()
          .collective_deterministic_sequential_execution()) {
    options.collective_order = GraphCollectiveOrder::kEdges;
  } else if (options_.config.experimental().collective_nccl()) {
    options.collective_order = GraphCollectiveOrder::kAttrs;
  }

  std::unique_ptr<FunctionInfo> func_info(new FunctionInfo);
  std::unique_ptr<ExecutorsAndKeys> ek(new ExecutorsAndKeys);

  ek->callable_options = callable_options;

  std::unordered_map<string, std::unique_ptr<Graph>> graphs;
  TF_RETURN_IF_ERROR(CreateGraphs(
      options, &graphs, &func_info->flib_def, run_state_args, &ek->input_types,
      &ek->output_types, &ek->collective_graph_key));

  if (run_state_args->is_partial_run) {
    ek->graph = std::move(run_state_args->graph);
    std::unordered_set<absl::string_view, StringPieceHasher> names;
    for (const string& input : callable_options.feed()) {
      TensorId id(ParseTensorName(input));
      names.emplace(id.first);
    }
    for (const string& output : callable_options.fetch()) {
      TensorId id(ParseTensorName(output));
      names.emplace(id.first);
    }
    for (Node* n : ek->graph->nodes()) {
      if (names.count(n->name()) > 0) {
        ek->name_to_node.insert({n->name(), n});
      }
    }
  }
  ek->items.reserve(graphs.size());
  const auto& optimizer_opts =
      options_.config.graph_options().optimizer_options();

  int graph_def_version = graphs.begin()->second->versions().producer();

  const auto* session_metadata =
      options_.config.experimental().has_session_metadata()
          ? &options_.config.experimental().session_metadata()
          : nullptr;
  func_info->proc_flr.reset(new ProcessFunctionLibraryRuntime(
      device_mgr_.get(), options_.env, &options_.config, graph_def_version,
      func_info->flib_def.get(), optimizer_opts, thread_pools_[0].first,
      /*parent=*/nullptr, session_metadata,
      Rendezvous::Factory{[](const int64_t, const DeviceMgr* device_mgr,
                             tsl::core::RefCountPtr<Rendezvous>* r) {
        *r = tsl::core::RefCountPtr<Rendezvous>(
            new IntraProcessRendezvous(device_mgr));
        return absl::OkStatus();
      }}));

  GraphOptimizer optimizer(optimizer_opts);
  for (auto iter = graphs.begin(); iter != graphs.end(); ++iter) {
    const string& partition_name = iter->first;
    std::unique_ptr<Graph>& partition_graph = iter->second;

    Device* device;
    TF_RETURN_IF_ERROR(device_mgr_->LookupDevice(partition_name, &device));

    ek->items.resize(ek->items.size() + 1);
    auto* item = &(ek->items.back());
    auto lib = func_info->proc_flr->GetFLR(partition_name);
    if (lib == nullptr) {
      return errors::Internal("Could not find device: ", partition_name);
    }
    item->flib = lib;

    LocalExecutorParams params;
    params.device = device;
    params.session_metadata = session_metadata;
    params.function_library = lib;
    auto opseg = device->op_segment();
    params.create_kernel =
        [this, lib, opseg](const std::shared_ptr<const NodeProperties>& props,
                           OpKernel** kernel) {
          // NOTE(mrry): We must not share function kernels (implemented
          // using `CallOp`) between subgraphs, because `CallOp::handle_`
          // is tied to a particular subgraph. Even if the function itself
          // is stateful, the `CallOp` that invokes it is not.
          if (!OpSegment::ShouldOwnKernel(lib, props->node_def.op())) {
            return lib->CreateKernel(props, kernel);
          }
          auto create_fn = [lib, &props](OpKernel** kernel) {
            return lib->CreateKernel(props, kernel);
          };
          // Kernels created for subgraph nodes need to be cached.  On
          // cache miss, create_fn() is invoked to create a kernel based
          // on the function library here + global op registry.
          return opseg->FindOrCreate(session_handle_, props->node_def.name(),
                                     kernel, create_fn);
        };
    params.delete_kernel = [lib](OpKernel* kernel) {
      if (kernel && !OpSegment::ShouldOwnKernel(lib, kernel->type_string()))
        delete kernel;
    };

    optimizer.Optimize(lib, options_.env, device, &partition_graph,
                       GraphOptimizer::Options());

    // TensorFlow Debugger (tfdbg) inserts debug nodes in the graph.
    const DebugOptions& debug_options =
        options.callable_options.run_options().debug_options();
    if (!debug_options.debug_tensor_watch_opts().empty()) {
      TF_RETURN_IF_ERROR(DecorateAndPublishGraphForDebug(
          debug_options, partition_graph.get(), params.device));
    }

    TF_RETURN_IF_ERROR(EnsureMemoryTypes(DeviceType(device->device_type()),
                                         device->name(),
                                         partition_graph.get()));

    item->executor = nullptr;
    item->device = device;
    auto executor_type = options_.config.experimental().executor_type();
    TF_RETURN_IF_ERROR(
        NewExecutor(executor_type, params, *partition_graph, &item->executor));
    if (!options_.config.experimental().disable_output_partition_graphs() ||
        options_.config.graph_options().build_cost_model() > 0) {
      item->graph = std::move(partition_graph);
    }
  }

  // Cache the mapping from input/output names to graph elements to
  // avoid recomputing it every time.
  if (!run_state_args->is_partial_run) {
    // For regular `Run()`, we use the function calling convention, and so
    // maintain a mapping from input/output names to
    // argument/return-value ordinal index.
    for (int i = 0; i < callable_options.feed().size(); ++i) {
      const string& input = callable_options.feed(i);
      ek->input_name_to_index[input] = i;
    }
    for (int i = 0; i < callable_options.fetch().size(); ++i) {
      const string& output = callable_options.fetch(i);
      ek->output_name_to_index[output] = i;
    }
  } else {
    // For `PRun()`, we use the rendezvous calling convention, and so
    // maintain a mapping from input/output names to rendezvous keys.
    //
    // We always use the first device as the device name portion of the
    // key, even if we're feeding another graph.
    for (int i = 0; i < callable_options.feed().size(); ++i) {
      const string& input = callable_options.feed(i);
      ek->input_name_to_rendezvous_key[input] = GetRendezvousKey(
          input, device_set_.client_device()->attributes(), FrameAndIter(0, 0));
    }
    for (int i = 0; i < callable_options.fetch().size(); ++i) {
      const string& output = callable_options.fetch(i);
      ek->output_name_to_rendezvous_key[output] =
          GetRendezvousKey(output, device_set_.client_device()->attributes(),
                           FrameAndIter(0, 0));
    }
  }

  *out_executors_and_keys = std::move(ek);
  *out_func_info = std::move(func_info);
  return absl::OkStatus();
}

absl::Status DirectSession::GetOrCreateExecutors(
    absl::Span<const string> inputs, absl::Span<const string> outputs,
    absl::Span<const string> target_nodes,
    ExecutorsAndKeys** executors_and_keys, RunStateArgs* run_state_args) {
  int64_t handle_name_counter_value = -1;
  if (LogMemory::IsEnabled() || run_state_args->is_partial_run) {
    handle_name_counter_value = handle_name_counter_.fetch_add(1);
  }

  string debug_tensor_watches_summary;
  if (!run_state_args->debug_options.debug_tensor_watch_opts().empty()) {
    debug_tensor_watches_summary = SummarizeDebugTensorWatches(
        run_state_args->debug_options.debug_tensor_watch_opts());
  }

  // Fast lookup path, no sorting.
  const string key = strings::StrCat(
      absl::StrJoin(inputs, ","), "->", absl::StrJoin(outputs, ","), "/",
      absl::StrJoin(target_nodes, ","), "/", run_state_args->is_partial_run,
      "/", debug_tensor_watches_summary);
  // Set the handle, if it's needed to log memory or for partial run.
  if (handle_name_counter_value >= 0) {
    run_state_args->handle =
        strings::StrCat(key, ";", handle_name_counter_value);
  }

  // See if we already have the executors for this run.
  {
    mutex_lock l(executor_lock_);  // could use reader lock
    auto it = executors_.find(key);
    if (it != executors_.end()) {
      *executors_and_keys = it->second.get();
      return absl::OkStatus();
    }
  }

  // Slow lookup path, the unsorted key missed the cache.
  // Sort the inputs and outputs, and look up with the sorted key in case an
  // earlier call used a different order of inputs and outputs.
  //
  // We could consider some other signature instead of sorting that
  // preserves the same property to avoid the sort in the future.
  std::vector<string> inputs_sorted(inputs.begin(), inputs.end());
  std::sort(inputs_sorted.begin(), inputs_sorted.end());
  std::vector<string> outputs_sorted(outputs.begin(), outputs.end());
  std::sort(outputs_sorted.begin(), outputs_sorted.end());
  std::vector<string> tn_sorted(target_nodes.begin(), target_nodes.end());
  std::sort(tn_sorted.begin(), tn_sorted.end());

  const string sorted_key = strings::StrCat(
      absl::StrJoin(inputs_sorted, ","), "->",
      absl::StrJoin(outputs_sorted, ","), "/", absl::StrJoin(tn_sorted, ","),
      "/", run_state_args->is_partial_run, "/", debug_tensor_watches_summary);
  // Set the handle, if its needed to log memory or for partial run.
  if (handle_name_counter_value >= 0) {
    run_state_args->handle =
        strings::StrCat(sorted_key, ";", handle_name_counter_value);
  }

  // See if we already have the executors for this run.
  {
    mutex_lock l(executor_lock_);
    auto it = executors_.find(sorted_key);
    if (it != executors_.end()) {
      *executors_and_keys = it->second.get();
      return absl::OkStatus();
    }
  }

  // Nothing found, so create the executors and store in the cache.
  // The executor_lock_ is intentionally released while executors are
  // being created.
  CallableOptions callable_options;
  callable_options.mutable_feed()->Reserve(inputs_sorted.size());
  for (const string& input : inputs_sorted) {
    callable_options.add_feed(input);
  }
  callable_options.mutable_fetch()->Reserve(outputs_sorted.size());
  for (const string& output : outputs_sorted) {
    callable_options.add_fetch(output);
  }
  callable_options.mutable_target()->Reserve(tn_sorted.size());
  for (const string& target : tn_sorted) {
    callable_options.add_target(target);
  }
  *callable_options.mutable_run_options()->mutable_debug_options() =
      run_state_args->debug_options;
  callable_options.mutable_run_options()
      ->mutable_experimental()
      ->set_collective_graph_key(run_state_args->collective_graph_key);
  std::unique_ptr<ExecutorsAndKeys> ek;
  std::unique_ptr<FunctionInfo> func_info;
  TF_RETURN_IF_ERROR(
      CreateExecutors(callable_options, &ek, &func_info, run_state_args));

  // Reacquire the lock, try to insert into the map.
  mutex_lock l(executor_lock_);

  // Another thread may have created the entry before us, in which case we will
  // reuse the already created one.
  auto insert_result = executors_.emplace(
      sorted_key, std::shared_ptr<ExecutorsAndKeys>(std::move(ek)));
  if (insert_result.second) {
    functions_.push_back(std::move(func_info));
  }

  // Insert the value under the original key, so the fast path lookup will work
  // if the user uses the same order of inputs, outputs, and targets again.
  executors_.emplace(key, insert_result.first->second);
  *executors_and_keys = insert_result.first->second.get();

  return absl::OkStatus();
}

absl::Status DirectSession::CreateGraphs(
    const BuildGraphOptions& subgraph_options,
    std::unordered_map<string, std::unique_ptr<Graph>>* outputs,
    std::unique_ptr<FunctionLibraryDefinition>* flib_def,
    RunStateArgs* run_state_args, DataTypeVector* input_types,
    DataTypeVector* output_types, int64_t* collective_graph_key) {
  mutex_lock l(graph_state_lock_);
  if (finalized_) {
    return errors::FailedPrecondition("Session has been finalized.");
  }

  std::unique_ptr<ClientGraph> client_graph;

  std::unique_ptr<GraphExecutionState> temp_exec_state_holder;
  GraphExecutionState* execution_state = nullptr;
  if (options_.config.graph_options().place_pruned_graph()) {
    // Because we are placing pruned graphs, we need to create a
    // new GraphExecutionState for every new unseen graph,
    // and then place it.
    GraphExecutionStateOptions prune_options;
    prune_options.device_set = &device_set_;
    prune_options.session_options = &options_;
    prune_options.stateful_placements = stateful_placements_;
    prune_options.session_handle = session_handle_;
    TF_RETURN_IF_ERROR(GraphExecutionState::MakeForPrunedGraph(
        *execution_state_, prune_options, subgraph_options,
        &temp_exec_state_holder, &client_graph));
    execution_state = temp_exec_state_holder.get();
  } else {
    execution_state = execution_state_.get();
    TF_RETURN_IF_ERROR(
        execution_state->BuildGraph(subgraph_options, &client_graph));
  }
  *collective_graph_key = client_graph->collective_graph_key;

  if (subgraph_options.callable_options.feed_size() !=
      client_graph->feed_types.size()) {
    return errors::Internal(
        "Graph pruning failed: requested number of feed endpoints = ",
        subgraph_options.callable_options.feed_size(),
        " versus number of pruned feed endpoints = ",
        client_graph->feed_types.size());
  }
  if (subgraph_options.callable_options.fetch_size() !=
      client_graph->fetch_types.size()) {
    return errors::Internal(
        "Graph pruning failed: requested number of fetch endpoints = ",
        subgraph_options.callable_options.fetch_size(),
        " versus number of pruned fetch endpoints = ",
        client_graph->fetch_types.size());
  }

  auto current_stateful_placements = execution_state->GetStatefulPlacements();
  // Update our current state based on the execution_state's
  // placements.  If there are any mismatches for a node,
  // we should fail, as this should never happen.
  for (const auto& placement_pair : current_stateful_placements) {
    const string& node_name = placement_pair.first;
    const string& placement = placement_pair.second;
    auto iter = stateful_placements_.find(node_name);
    if (iter == stateful_placements_.end()) {
      stateful_placements_.insert(std::make_pair(node_name, placement));
    } else if (iter->second != placement) {
      return errors::Internal(
          "Stateful placement mismatch. "
          "Current assignment of ",
          node_name, " to ", iter->second, " does not match ", placement);
    }
  }

  stateful_placements_ = execution_state->GetStatefulPlacements();

  // Remember the graph in run state if this is a partial run.
  if (run_state_args->is_partial_run) {
    run_state_args->graph.reset(new Graph(flib_def_.get()));
    CopyGraph(*execution_state->full_graph(), run_state_args->graph.get());
  }

  // Partition the graph across devices.
  PartitionOptions popts;
  popts.node_to_loc = [](const Node* node) {
    return node->assigned_device_name();
  };
  popts.new_name = [this](const string& prefix) {
    return strings::StrCat(prefix, "/_", edge_name_counter_.fetch_add(1));
  };
  popts.get_incarnation = [](const string& name) {
    // The direct session does not have changing incarnation numbers.
    // Just return '1'.
    return 1;
  };
  popts.flib_def = flib_def->get();
  popts.control_flow_added = false;

  std::unordered_map<string, GraphDef> partitions;
  TF_RETURN_IF_ERROR(Partition(popts, &client_graph->graph, &partitions));

  std::vector<string> device_names;
  device_names.reserve(devices_.size());
  for (auto device : devices_) {
    // Extract the LocalName from the device.
    device_names.push_back(DeviceNameUtils::LocalName(device->name()));
  }

  // Check for valid partitions.
  for (const auto& partition : partitions) {
    const string local_partition_name =
        DeviceNameUtils::LocalName(partition.first);
    if (std::count(device_names.begin(), device_names.end(),
                   local_partition_name) == 0) {
      return errors::InvalidArgument(
          "Creating a partition for ", local_partition_name,
          " which doesn't exist in the list of available devices. Available "
          "devices: ",
          absl::StrJoin(device_names, ","));
    }
  }

  for (auto& partition : partitions) {
    std::unique_ptr<Graph> device_graph(
        new Graph(client_graph->flib_def.get()));
    device_graph->SetConstructionContext(ConstructionContext::kDirectSession);
    GraphConstructorOptions device_opts;
    // There are internal operations (e.g., send/recv) that we now allow.
    device_opts.allow_internal_ops = true;
    device_opts.expect_device_spec = true;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
        device_opts, std::move(partition.second), device_graph.get()));
    outputs->emplace(partition.first, std::move(device_graph));
  }

  GraphOptimizationPassOptions optimization_options;
  optimization_options.session_options = &options_;
  optimization_options.flib_def = client_graph->flib_def.get();
  optimization_options.partition_graphs = outputs;
  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::POST_PARTITIONING, optimization_options));

  absl::Status s;
  for (auto& partition : *outputs) {
    const string& partition_name = partition.first;
    std::unique_ptr<Graph>* graph = &partition.second;

    VLOG(2) << "Created " << DebugString(graph->get()) << " for "
            << partition_name;

    // Give the device an opportunity to rewrite its subgraph.
    Device* d;
    s = device_mgr_->LookupDevice(partition_name, &d);
    if (!s.ok()) break;
    s = d->MaybeRewriteGraph(graph);
    if (!s.ok()) {
      break;
    }
  }
  *flib_def = std::move(client_graph->flib_def);
  std::swap(*input_types, client_graph->feed_types);
  std::swap(*output_types, client_graph->fetch_types);
  return s;
}

absl::Status DirectSession::ListDevices(
    std::vector<DeviceAttributes>* response) {
  response->clear();
  response->reserve(devices_.size());
  for (Device* d : devices_) {
    const DeviceAttributes& attrs = d->attributes();
    response->emplace_back(attrs);
  }
  return absl::OkStatus();
}

absl::Status DirectSession::Reset(const std::vector<string>& containers) {
  device_mgr_->ClearContainers(containers);
  return absl::OkStatus();
}

absl::Status DirectSession::Close() {
  cancellation_manager_->StartCancel();
  {
    mutex_lock l(closed_lock_);
    if (closed_) return absl::OkStatus();
    closed_ = true;
  }
  if (factory_ != nullptr) factory_->Deregister(this);
  return absl::OkStatus();
}

DirectSession::RunState::RunState(int64_t step_id,
                                  const std::vector<Device*>* devices)
    : step_container(step_id, [devices, step_id](const string& name) {
        for (auto d : *devices) {
          if (!d->resource_manager()->Cleanup(name).ok()) {
            // Do nothing...
          }
          ScopedAllocatorMgr* sam = d->GetScopedAllocatorMgr();
          if (sam) sam->Cleanup(step_id);
        }
      }) {}

DirectSession::PartialRunState::PartialRunState(
    const std::vector<string>& pending_input_names,
    const std::vector<string>& pending_output_names, int64_t step_id,
    const std::vector<Device*>* devices)
    : RunState(step_id, devices) {
  // Initially all the feeds and fetches are pending.
  for (auto& name : pending_input_names) {
    pending_inputs[name] = false;
  }
  for (auto& name : pending_output_names) {
    pending_outputs[name] = false;
  }
}

DirectSession::PartialRunState::~PartialRunState() {
  if (rendez != nullptr) {
    rendez->StartAbort(errors::Cancelled("PRun cancellation"));
    executors_done.WaitForNotification();
  }
}

bool DirectSession::PartialRunState::PendingDone() const {
  for (const auto& it : pending_inputs) {
    if (!it.second) return false;
  }
  for (const auto& it : pending_outputs) {
    if (!it.second) return false;
  }
  return true;
}

void DirectSession::WaitForNotification(Notification* n, RunState* run_state,
                                        CancellationManager* cm,
                                        int64_t timeout_in_ms) {
  const absl::Status status = WaitForNotification(n, timeout_in_ms);
  if (!status.ok()) {
    {
      mutex_lock l(run_state->mu);
      run_state->status.Update(status);
    }
    cm->StartCancel();
    // We must wait for the executors to complete, because they have borrowed
    // references to `cm` and other per-step state. After this notification, it
    // is safe to clean up the step.
    n->WaitForNotification();
  }
}

absl::Status DirectSession::WaitForNotification(Notification* notification,
                                                int64_t timeout_in_ms) {
  if (timeout_in_ms > 0) {
    const int64_t timeout_in_us = timeout_in_ms * 1000;
    const bool notified =
        WaitForNotificationWithTimeout(notification, timeout_in_us);
    if (!notified) {
      return absl::Status(absl::StatusCode::kDeadlineExceeded,
                          "Timed out waiting for notification");
    }
  } else {
    notification->WaitForNotification();
  }
  return absl::OkStatus();
}

absl::Status DirectSession::MakeCallable(
    const CallableOptions& callable_options, CallableHandle* out_handle) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  TF_RETURN_IF_ERROR(CheckGraphCreated("MakeCallable()"));

  std::unique_ptr<ExecutorsAndKeys> ek;
  std::unique_ptr<FunctionInfo> func_info;
  RunStateArgs run_state_args(callable_options.run_options().debug_options());
  TF_RETURN_IF_ERROR(
      CreateExecutors(callable_options, &ek, &func_info, &run_state_args));
  {
    mutex_lock l(callables_lock_);
    *out_handle = next_callable_handle_++;
    callables_[*out_handle] = {std::move(ek), std::move(func_info)};
  }
  return absl::OkStatus();
}

class DirectSession::RunCallableCallFrame : public CallFrameInterface {
 public:
  RunCallableCallFrame(DirectSession* session,
                       ExecutorsAndKeys* executors_and_keys,
                       const std::vector<Tensor>* feed_tensors,
                       std::vector<Tensor>* fetch_tensors)
      : session_(session),
        executors_and_keys_(executors_and_keys),
        feed_tensors_(feed_tensors),
        fetch_tensors_(fetch_tensors) {}

  size_t num_args() const override {
    return executors_and_keys_->input_types.size();
  }
  size_t num_retvals() const override {
    return executors_and_keys_->output_types.size();
  }

  absl::Status GetArg(int index, const Tensor** val) override {
    if (TF_PREDICT_FALSE(index > feed_tensors_->size())) {
      return errors::Internal("Args index out of bounds: ", index);
    } else {
      *val = &(*feed_tensors_)[index];
    }
    return absl::OkStatus();
  }

  absl::Status SetRetval(int index, const Tensor& val) override {
    if (index > fetch_tensors_->size()) {
      return errors::Internal("RetVal index out of bounds: ", index);
    }
    (*fetch_tensors_)[index] = val;
    return absl::OkStatus();
  }

 private:
  DirectSession* const session_;                   // Not owned.
  ExecutorsAndKeys* const executors_and_keys_;     // Not owned.
  const std::vector<Tensor>* const feed_tensors_;  // Not owned.
  std::vector<Tensor>* const fetch_tensors_;       // Not owned.
};

absl::Status DirectSession::RunCallable(CallableHandle handle,
                                        const std::vector<Tensor>& feed_tensors,
                                        std::vector<Tensor>* fetch_tensors,
                                        RunMetadata* run_metadata) {
  return RunCallable(handle, feed_tensors, fetch_tensors, run_metadata,
                     thread::ThreadPoolOptions());
}

absl::Status DirectSession::RunCallable(
    CallableHandle handle, const std::vector<Tensor>& feed_tensors,
    std::vector<Tensor>* fetch_tensors, RunMetadata* run_metadata,
    const thread::ThreadPoolOptions& threadpool_options) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  TF_RETURN_IF_ERROR(CheckGraphCreated("RunCallable()"));
  direct_session_runs->GetCell()->IncrementBy(1);

  // Check if we already have an executor for these arguments.
  std::shared_ptr<ExecutorsAndKeys> executors_and_keys;
  const int64_t step_id = step_id_counter_.fetch_add(1);

  {
    tf_shared_lock l(callables_lock_);
    if (handle >= next_callable_handle_) {
      return errors::InvalidArgument("No such callable handle: ", handle);
    }
    executors_and_keys = callables_[handle].executors_and_keys;
  }

  if (!executors_and_keys) {
    return errors::InvalidArgument(
        "Attempted to run callable after handle was released: ", handle);
  }

  // NOTE(mrry): Debug options are not currently supported in the
  // callable interface.
  DebugOptions debug_options;
  RunStateArgs run_state_args(debug_options);

  // Configure a call frame for the step, which we use to feed and
  // fetch values to and from the executors.
  if (feed_tensors.size() != executors_and_keys->input_types.size()) {
    return errors::InvalidArgument(
        "Expected ", executors_and_keys->input_types.size(),
        " feed tensors, but got ", feed_tensors.size());
  }
  if (fetch_tensors != nullptr) {
    fetch_tensors->resize(executors_and_keys->output_types.size());
  } else if (!executors_and_keys->output_types.empty()) {
    return errors::InvalidArgument(
        "`fetch_tensors` must be provided when the callable has one or more "
        "outputs.");
  }

  size_t input_size = 0;
  bool any_resource_feeds = false;
  for (auto& tensor : feed_tensors) {
    input_size += tensor.AllocatedBytes();
    any_resource_feeds = any_resource_feeds || tensor.dtype() == DT_RESOURCE;
  }
  metrics::RecordGraphInputTensors(input_size);

  std::unique_ptr<std::vector<Tensor>> converted_feed_tensors;
  const std::vector<Tensor>* actual_feed_tensors;

  if (TF_PREDICT_FALSE(any_resource_feeds)) {
    converted_feed_tensors = std::make_unique<std::vector<Tensor>>();
    converted_feed_tensors->reserve(feed_tensors.size());
    for (const Tensor& t : feed_tensors) {
      if (t.dtype() == DT_RESOURCE) {
        converted_feed_tensors->emplace_back();
        Tensor* tensor_from_handle = &converted_feed_tensors->back();
        TF_RETURN_IF_ERROR(ResourceHandleToInputTensor(t, tensor_from_handle));
      } else {
        converted_feed_tensors->emplace_back(t);
      }
    }
    actual_feed_tensors = converted_feed_tensors.get();
  } else {
    actual_feed_tensors = &feed_tensors;
  }

  // A specialized CallFrame implementation that takes advantage of the
  // optimized RunCallable interface.
  RunCallableCallFrame call_frame(this, executors_and_keys.get(),
                                  actual_feed_tensors, fetch_tensors);

  if (LogMemory::IsEnabled()) {
    LogMemory::RecordStep(step_id, run_state_args.handle);
  }

  TF_RETURN_IF_ERROR(RunInternal(
      step_id, executors_and_keys->callable_options.run_options(), &call_frame,
      executors_and_keys.get(), run_metadata, threadpool_options));

  if (fetch_tensors != nullptr) {
    size_t output_size = 0;
    for (auto& tensor : *fetch_tensors) {
      output_size += tensor.AllocatedBytes();
    }
    metrics::RecordGraphOutputTensors(output_size);
  }

  return absl::OkStatus();
}

absl::Status DirectSession::ReleaseCallable(CallableHandle handle) {
  mutex_lock l(callables_lock_);
  if (handle >= next_callable_handle_) {
    return errors::InvalidArgument("No such callable handle: ", handle);
  }
  callables_.erase(handle);
  return absl::OkStatus();
}

absl::Status DirectSession::Finalize() {
  mutex_lock l(graph_state_lock_);
  if (finalized_) {
    return errors::FailedPrecondition("Session already finalized.");
  }
  if (!graph_created_) {
    return errors::FailedPrecondition("Session not yet created.");
  }
  execution_state_.reset();
  flib_def_.reset();
  finalized_ = true;
  return absl::OkStatus();
}

DirectSession::Callable::~Callable() {
  // We must delete the fields in this order, because the destructor
  // of `executors_and_keys` will call into an object owned by
  // `function_info` (in particular, when deleting a kernel, it relies
  // on the `FunctionLibraryRuntime` to know if the kernel is stateful
  // or not).
  executors_and_keys.reset();
  function_info.reset();
}

}  // namespace tensorflow
