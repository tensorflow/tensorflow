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

#ifndef TENSORFLOW_COMMON_RUNTIME_DIRECT_SESSION_H_
#define TENSORFLOW_COMMON_RUNTIME_DIRECT_SESSION_H_

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/common_runtime/costmodel_manager.h"
#include "tensorflow/core/common_runtime/debugger_state_interface.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/common_runtime/simple_graph_execution_state.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/session_state.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

class CostModel;
class DebugGateway;
class Device;
class DirectSessionFactory;

class DirectSession : public Session {
 public:
  typedef std::function<void(Session*)> CloseCallback;

  // Takes ownership of 'device_mgr'.
  // 'factory' is used to unregister the DirectSession with 'factory' when its
  // closed. This ensures that Reset requests from the 'factory' don't get sent
  // to sessions that are already closed.
  DirectSession(const SessionOptions& options, const DeviceMgr* device_mgr,
                DirectSessionFactory* factory);
  ~DirectSession() override;

  typedef std::vector<std::pair<string, Tensor>> NamedTensorList;
  typedef std::unordered_map<StringPiece, Node*, StringPiece::Hasher>
      NameNodeMap;

  ::tensorflow::Status Create(const GraphDef& graph) override;
  ::tensorflow::Status Extend(const GraphDef& graph) override;
  ::tensorflow::Status Run(const NamedTensorList& inputs,
                           const std::vector<string>& output_names,
                           const std::vector<string>& target_nodes,
                           std::vector<Tensor>* outputs) override;

  // NOTE: Experimental and subject to change.
  ::tensorflow::Status Run(const ::tensorflow::RunOptions& run_options,
                           const NamedTensorList& inputs,
                           const std::vector<string>& output_names,
                           const std::vector<string>& target_nodes,
                           std::vector<Tensor>* outputs,
                           RunMetadata* run_metadata) override;

  // NOTE: PRunSetup and PRun are added to support partial execution. This
  // feature is experimental and subject to change.
  ::tensorflow::Status PRunSetup(const std::vector<string>& input_names,
                                 const std::vector<string>& output_names,
                                 const std::vector<string>& target_nodes,
                                 string* handle) override;
  ::tensorflow::Status PRun(const string& handle, const NamedTensorList& inputs,
                            const std::vector<string>& output_names,
                            std::vector<Tensor>* outputs) override;

  // Reset clears 'containers' from the device_mgr of the DirectSession.
  // If 'containers' is empty, then Reset clears the default container.
  ::tensorflow::Status Reset(const std::vector<string>& containers);

  ::tensorflow::Status Close() override;

  void ExportCostModels(CostModelManager::CostModelMap* cost_models) {
    cost_model_manager_.ExportCostModels(cost_models);
  }

 private:
  typedef DirectSession ME;

  // We create one executor and its dependent library runtime for
  // every partition.
  struct PerPartitionExecutorsAndLib {
    Graph* graph = nullptr;
    std::unique_ptr<FunctionLibraryRuntime> flib;
    std::unique_ptr<Executor> executor;
  };

  // An ExecutorsAndKeys is created for a given set of feeds/fetches.
  // 'step_count' is the number of times this graph is executed.
  // 'graph' is the entire graph being executed. 'name_to_node'
  // maps node name to node. We keep 'graph' and 'name_to_node' only in
  // the case of partial runs. Each item in 'items' is the executor for
  // a partition of the graph bundled with its dependent library runtime.
  // 'input_keys' are the rendezvous keys for the feeds and 'output_keys'
  // are rendezvous keys for the fetches.
  // 'flib_def' is the function library used by graphs in 'items'.
  // TODO(phawkins): currently partitions always share the same function
  // library. Consider giving each partition its own function library to enable
  // per-partition rewrites.
  struct ExecutorsAndKeys {
    int64 step_count = 0;
    std::unique_ptr<Graph> graph;
    NameNodeMap name_to_node;
    std::unique_ptr<FunctionLibraryDefinition> flib_def;
    std::vector<PerPartitionExecutorsAndLib> items;
    std::unordered_map<string, string> input_keys;
    std::unordered_map<string, string> output_keys;
  };

  // For each live partial execution, the session maintains a RunState.
  // 'status' is the current status of this partial execution. 'executor_done'
  // is "notified" when all executors are done. 'pending_inputs' are the set
  // of pending feeds and 'pending_outputs' are the set of pending fetches.
  struct RunState {
    mutex mu_;
    Status status GUARDED_BY(mu_);
    IntraProcessRendezvous* rendez = nullptr;
    std::unique_ptr<StepStatsCollector> collector;
    Notification executors_done;
    std::unordered_set<string> pending_inputs;
    std::unordered_set<string> pending_outputs;
    TensorStore tensor_store;
    ScopedStepContainer step_container;

    RunState(int64 step_id, const std::vector<Device*>* devices);

    RunState(const std::vector<string>& pending_input_names,
             const std::vector<string>& pending_output_names, int64 step_id,
             const std::vector<Device*>* devices);

    ~RunState();
  };

  struct RunStateArgs {
    bool is_partial_run = false;
    string handle;
    std::unique_ptr<Graph> graph;
    std::unique_ptr<DebuggerStateInterface> debugger_state;
  };

  // Initializes the base execution state given the 'graph',
  // if not already initialized.
  Status MaybeInitializeExecutionState(const GraphDef& graph,
                                       bool* out_already_initialized)
      EXCLUSIVE_LOCKS_REQUIRED(graph_def_lock_);

  // Retrieves an already existing set of executors to run 'inputs' and
  // 'outputs', or creates and caches them for future use.
  ::tensorflow::Status GetOrCreateExecutors(
      thread::ThreadPool* pool, gtl::ArraySlice<string> inputs,
      gtl::ArraySlice<string> outputs, gtl::ArraySlice<string> target_nodes,
      ExecutorsAndKeys** executors_and_keys, RunStateArgs* run_state_args);

  // Creates several graphs given the existing graph_def_ and the
  // input feeds and fetches, given 'devices'. The graphs share a common
  // function library 'flib_def'.
  ::tensorflow::Status CreateGraphs(
      const BuildGraphOptions& options,
      std::unordered_map<string, std::unique_ptr<Graph>>* outputs,
      std::unique_ptr<FunctionLibraryDefinition>* flib_def,
      RunStateArgs* run_state_args);

  ::tensorflow::Status ExtendLocked(const GraphDef& graph)
      EXCLUSIVE_LOCKS_REQUIRED(graph_def_lock_);

  // Feeds more inputs to the executors, triggering further execution.
  ::tensorflow::Status SendInputs(
      const std::vector<std::pair<string, Tensor>>& inputs,
      const ExecutorsAndKeys* executors_and_keys,
      IntraProcessRendezvous* rendez);

  // Fetches more outputs from the executors. It waits until the output
  // tensors are computed.
  ::tensorflow::Status RecvOutputs(const std::vector<string>& output_names,
                                   const ExecutorsAndKeys* executors_and_keys,
                                   RunState* run_state,
                                   std::vector<Tensor>* outputs);

  // Check if the specified fetches can be computed from the feeds
  // that we have already provided.
  ::tensorflow::Status CheckFetch(
      const std::vector<std::pair<string, Tensor>>& feeds,
      const std::vector<string>& fetches,
      const ExecutorsAndKeys* executors_and_keys, const RunState* run_state);

  // Use the appropriate WaitForNotification function based on whether
  // operation_timeout_in_ms is greater than 0.
  //
  // If the timeout expires, the `cm->StartCancel()` will be called.
  ::tensorflow::Status WaitForNotification(Notification* n,
                                           int64 timeout_in_ms);
  void WaitForNotification(RunState* run_state, CancellationManager* cm,
                           int64 timeout_in_ms);

  ::tensorflow::Status CheckNotClosed() {
    mutex_lock l(closed_lock_);
    if (closed_) return errors::Cancelled("Session has been closed.");
    return ::tensorflow::Status::OK();
  }

  const SessionOptions options_;

  // Device structures.
  const std::unique_ptr<const DeviceMgr> device_mgr_;
  std::vector<Device*> devices_;  // not owned
  DeviceSet device_set_;

  string session_handle_;
  bool graph_created_ GUARDED_BY(graph_def_lock_) = false;

  mutex graph_def_lock_;
  GraphDef graph_def_ GUARDED_BY(graph_def_lock_);

  // The thread-pools to use for running ops.
  std::vector<thread::ThreadPool*> thread_pools_;
  bool owns_thread_pools_ = false;

  // Schedules 'c' for execution on pool.
  void SchedClosure(thread::ThreadPool* pool, std::function<void()> c);

  mutex executor_lock_;  // protects executors_
  // Holds mappings from signature to the executors that process
  // it. The reason for a level of indirection around mapped_type is
  // to guarantee address stability.
  std::unordered_map<string, std::unique_ptr<ExecutorsAndKeys>> executors_
      GUARDED_BY(executor_lock_);

  // Holds mappings from handle to partial run state.
  std::unordered_map<string, std::unique_ptr<RunState>> partial_runs_
      GUARDED_BY(executor_lock_);

  // This holds all the tensors that are currently alive in the session.
  SessionState session_state_;

  DirectSessionFactory* const factory_;  // not owned
  CancellationManager* cancellation_manager_;

  // Map of placed stateful nodes, i.e. nodes for which is_stateful()
  // is true, such as "params" and "queue" nodes.  Once placed these
  // nodes can not be moved to a different device.  Maps node names to
  // device names.
  std::unordered_map<string, string> stateful_placements_
      GUARDED_BY(graph_def_lock_);

  // Execution_state; used when placing the entire graph.
  std::unique_ptr<SimpleGraphExecutionState> execution_state_
      GUARDED_BY(graph_def_lock_);

  // The function library, before any rewrites or optimizations have been
  // performed. In particular, CreateGraphs() may need to modify the function
  // library; it copies and modifies the function library.
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;

  // true if the Session has been Closed.
  mutex closed_lock_;
  bool closed_ GUARDED_BY(closed_lock_) = false;

  // For generating unique names for this session instance.
  std::atomic<int64> edge_name_counter_ = {0};
  std::atomic<int64> handle_name_counter_ = {0};

  // For generating step ids that are unique across all sessions.
  static std::atomic_int_fast64_t step_id_counter_;

  // Global timeout for all blocking operations in this session.
  const int64 operation_timeout_in_ms_ = 0;

  // Manages all the cost models for the graphs executed in this session.
  CostModelManager cost_model_manager_;

  Executor::Args::NodeOutputsCallback node_outputs_callback_ = nullptr;

  TF_DISALLOW_COPY_AND_ASSIGN(DirectSession);

  // EXPERIMENTAL: debugger (tfdbg) related
  friend class DebugGateway;
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_DIRECT_SESSION_H_
