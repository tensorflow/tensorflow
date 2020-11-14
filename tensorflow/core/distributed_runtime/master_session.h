/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_SESSION_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_SESSION_H_

#include <atomic>
#include <vector>

#include "tensorflow/core/common_runtime/debugger_state_interface.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/graph_execution_state.h"
#include "tensorflow/core/common_runtime/stats_publisher_interface.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/message_wrappers.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/master.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class Device;
struct MasterEnv;

// A session encapsulates a graph computation (resource allocation,
// placement, execution, etc.).
class MasterSession : public core::RefCounted {
 public:
  // This session encapsulates the graph computation for a graph.
  //
  // The session places nodes on devices in "remote_devs" and executes
  // operations on these devices.
  //
  // The caller takes ownership of all remote devices.
  MasterSession(
      const SessionOptions& options, const MasterEnv* env,
      std::unique_ptr<std::vector<std::unique_ptr<Device>>> remote_devs,
      std::unique_ptr<WorkerCacheInterface> worker_cache,
      std::unique_ptr<DeviceSet> device_set,
      std::vector<string> filtered_worker_list,
      StatsPublisherFactory stats_publisher_factory);

  // Initialize the MasterSession for "def".  Must be called before Extend(),
  // Run(), or Close().
  Status Create(GraphDef&& def, const WorkerCacheFactoryOptions& options);

  // Returns the session handle.
  const string& handle() const { return handle_; }

  // Returns the last access time (the number of micro-seconds since
  // some fixed point in time) of this session.
  uint64 last_access_time_usec() const { return last_access_time_usec_.load(); }

  // Attempt to extend the graph according to the given "req".
  // (See master.proto for details of valid extensions.)
  //
  // PRECONDITION: The current version of this session's graph
  //   is "req->current_graph_version".
  //
  // POSTCONDITION: The current version of this session's graph
  //   is "resp->new_graph_version".
  //
  // Extend() may block the caller thread for a long time.
  Status Extend(const ExtendSessionRequest* req, ExtendSessionResponse* resp);

  // Setup a partial run call.
  Status PartialRunSetup(const PartialRunSetupRequest* req,
                         PartialRunSetupResponse* resp);

  // Run one step.
  Status Run(CallOptions* opts, const RunStepRequestWrapper& req,
             MutableRunStepResponseWrapper* resp);

  Status ListDevices(ListDevicesResponse* resp) const;

  Status MakeCallable(const MakeCallableRequest& req,
                      MakeCallableResponse* resp);

  Status RunCallable(CallOptions* opts, const RunCallableRequest& req,
                     RunCallableResponse* resp);

  Status ReleaseCallable(const ReleaseCallableRequest& req,
                         ReleaseCallableResponse* resp);

  // Close this session and delete "*this". Returns OK if all known
  // states are cleanup successfully.
  //
  // Close() may block the caller thread for a long time.
  Status Close();

  // Close this session and release a reference on "*this".
  //
  // Note that, unlike Close(), this method does not block on the
  // completion of all work.
  void GarbageCollect();

 private:
  SessionOptions session_opts_;

  // Not owned.
  const MasterEnv* env_;

  // The opaque session handle.
  const string handle_;

  std::unique_ptr<std::vector<std::unique_ptr<Device>>> remote_devs_;

  // The optional session-specific worker cluster.
  // TODO(saeta): Convert to std::optional when available.
  const std::unique_ptr<WorkerCacheInterface> worker_cache_;
  // Retrieves either worker_cache_ or the env_->worker_cache as appropriate.
  WorkerCacheInterface* get_worker_cache() const;

  // The device set used by this session.
  std::unique_ptr<DeviceSet> devices_;

  // The (partial device) names of remote worker tasks that this
  // session will contact.
  const std::vector<string> filtered_worker_list_;

  StatsPublisherFactory stats_publisher_factory_;

  std::atomic_ulong last_access_time_usec_;

  std::atomic<int64> partial_run_handle_counter_ = {0};

  uint64 NewStepId(int64 graph_key);

  mutex mu_;
  std::unique_ptr<GraphExecutionState> execution_state_ TF_GUARDED_BY(mu_);
  int64 graph_version_;

  // We keep a map from a signature of a run request to the
  // ReffedClientGraph the can execute it.  We keep up to one old copy
  // of each ReffedClientGraph around because if it gets deallocated
  // before a new substitute has been created, Variables can go out of
  // scope and lose their state.
  class ReffedClientGraph;
  typedef std::unordered_map<uint64, ReffedClientGraph*> RCGMap;
  RCGMap run_graphs_ TF_GUARDED_BY(mu_);
  RCGMap partial_run_graphs_ TF_GUARDED_BY(mu_);
  int64 next_callable_handle_ TF_GUARDED_BY(mu_) = 0;
  RCGMap callables_ TF_GUARDED_BY(mu_);

  struct PerStepState {
    bool collect_costs = false;
    bool collect_timeline = false;
    bool collect_rpcs = false;
    bool collect_partition_graphs = false;
    bool report_tensor_allocations_upon_oom = false;
    Microseconds start_micros = Microseconds(0);
    Microseconds end_micros = Microseconds(0);
    std::vector<StepStats> step_stats;  // per partition
    StepStats rpc_stats;                // for RPC layer
    CostGraphDef cost_graph;
  };

  struct RunState {
    std::unordered_map<string, bool> pending_inputs;   // true if fed
    std::unordered_map<string, bool> pending_outputs;  // true if fetched
    ReffedClientGraph* rcg = nullptr;
    uint64 step_id;
    int64 collective_graph_key;
    int64 count = 0;
    PerStepState pss;
    std::unique_ptr<ProfileHandler> ph;
    bool step_started = false;

    RunState(const std::vector<string>& input_names,
             const std::vector<string>& output_names, ReffedClientGraph* rcg,
             const uint64 step_id, const int64 count);

    bool PendingDone() const;

    ~RunState();
  };
  std::unordered_map<string, std::unique_ptr<RunState>> partial_runs_
      TF_GUARDED_BY(mu_);

  // Active RunStep calls.
  condition_variable num_running_is_zero_;
  int32 num_running_ TF_GUARDED_BY(mu_) = 0;

  bool closed_ TF_GUARDED_BY(mu_) = false;
  bool garbage_collected_ TF_GUARDED_BY(mu_) = false;

  std::unordered_map<uint64, int64> subgraph_execution_counts_
      TF_GUARDED_BY(mu_);

  // We need to ensure that certain nodes added (e.g., send and recv
  // nodes) are unique across all sub-graphs within this session.
  int64 next_node_id_ TF_GUARDED_BY(mu_) = 0;

  // Used to cancel running steps on Close().
  CancellationManager cancellation_manager_;

  // Private dtor. The client must call Close().
  virtual ~MasterSession();

  // Creates sessions on all workers.
  //
  // If this session is operating using the new ClusterSpec propagation behavior
  // call this method in order to propagate the cluster membership to all
  // workers.
  Status CreateWorkerSessions(const WorkerCacheFactoryOptions& server_def);

  bool should_delete_worker_sessions_ = false;
  Status DeleteWorkerSessions();

  Status StartStep(const BuildGraphOptions& opts, bool is_partial,
                   ReffedClientGraph** out_rcg, int64* out_count);
  void ClearRunsTable(std::vector<ReffedClientGraph*>* to_unref,
                      RCGMap* rcg_map) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void FillPerStepState(MasterSession::ReffedClientGraph* rcg,
                        const RunOptions& run_options, uint64 step_id,
                        int64 count, PerStepState* out_pss,
                        std::unique_ptr<ProfileHandler>* out_ph);
  Status DoRunWithLocalExecution(CallOptions* opts,
                                 const RunStepRequestWrapper& req,
                                 MutableRunStepResponseWrapper* resp);
  Status DoPartialRun(CallOptions* opts, const RunStepRequestWrapper& req,
                      MutableRunStepResponseWrapper* resp);
  Status DoRunCallable(CallOptions* opts, ReffedClientGraph* rcg,
                       const RunCallableRequest& req,
                       RunCallableResponse* resp);
  Status PostRunCleanup(MasterSession::ReffedClientGraph* rcg, uint64 step_id,
                        const RunOptions& run_options, PerStepState* pss,
                        const std::unique_ptr<ProfileHandler>& ph,
                        const Status& run_status,
                        RunMetadata* out_run_metadata);

  void MarkRunCompletion();
  void UpdateLastAccessTime();

  Status BuildAndRegisterPartitions(ReffedClientGraph* rcg);

  Status CreateDebuggerState(
      const DebugOptions& debug_options, const RunStepRequestWrapper& req,
      int64 rcg_execution_count,
      std::unique_ptr<DebuggerStateInterface>* debugger_state);

  TF_DISALLOW_COPY_AND_ASSIGN(MasterSession);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_SESSION_H_
