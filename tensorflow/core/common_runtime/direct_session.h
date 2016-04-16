/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
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
class Device;
class ThreadPool;

class DirectSession : public Session {
 public:
  // Takes ownership of 'device_mgr'.
  DirectSession(const SessionOptions& options, const DeviceMgr* device_mgr);
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

  ::tensorflow::Status Close() override;

  // NOTE: This is a temporary api that is only meant to enable testing.
  // This api will be replaced with better ones soon, so DO NOT USE
  const std::unordered_map<const Graph*, CostModel*>& GetCostModels() const {
    return cost_models_;
  }

 private:
  typedef DirectSession ME;

  // We create one executor and its dependent library runtime for
  // every partition.
  struct PerPartitionExecutorsAndLib {
    Executor* executor = nullptr;
    FunctionLibraryRuntime* flib = nullptr;
  };

  // An ExecutorsAndKeys is created for a given set of feeds/fetches.
  // 'func_defs' are the function definition used by all the
  // underlying executors. 'graph' is the entire graph being
  // executed. 'name_to_node' maps node name to node. We keep 'graph'
  // and 'name_to_node' only in the case of partial runs. Each item in
  // 'items' is the executor for a partition of the graph bundled with
  // its dependent library runtime. 'input_keys' are the rendezvous keys
  // for the feeds and 'output_keys' are rendezvous keys for the fetches.
  struct ExecutorsAndKeys {
    FunctionLibraryDefinition* func_defs = nullptr;
    Graph* graph = nullptr;
    NameNodeMap* name_to_node = nullptr;
    std::vector<PerPartitionExecutorsAndLib> items;
    std::unordered_map<string, string> input_keys;
    std::unordered_map<string, string> output_keys;

    ~ExecutorsAndKeys() {
      for (auto item : items) {
        delete item.executor;
        delete item.flib;
      }
      delete func_defs;
      delete graph;
      delete name_to_node;
    }
  };

  // For each live partial execution, the session maintains a RunState.
  // 'status' is the current status of this partial execution. 'executor_done'
  // is "notified" when all executors are done. 'pending_inputs' are the set
  // of pending feeds and 'pending_outputs' are the set of pending fetches.
  struct RunState {
    mutex mu_;
    Status status GUARDED_BY(mu_);
    IntraProcessRendezvous* rendez = nullptr;
    StepStatsCollector* collector = nullptr;
    Notification executors_done;
    std::unordered_set<string> pending_inputs;
    std::unordered_set<string> pending_outputs;
    TensorStore tensor_store;

    RunState(const std::vector<string>& input_names,
             const std::vector<string>& output_names) {
      // Initially all the feeds and fetches are pending.
      for (auto& name : input_names) {
        pending_inputs.emplace(name);
      }
      for (auto& name : output_names) {
        pending_outputs.emplace(name);
      }
    }

    ~RunState();
  };

  struct RunStateArgs {
    bool is_partial_run = false;
    string handle;
    Graph* graph = nullptr;
  };

  // Retrieves an already existing set of executors to run 'inputs' and
  // 'outputs', or creates and caches them for future use.
  ::tensorflow::Status GetOrCreateExecutors(
      gtl::ArraySlice<string> inputs, gtl::ArraySlice<string> outputs,
      gtl::ArraySlice<string> target_nodes,
      ExecutorsAndKeys** executors_and_keys, RunStateArgs* run_state_args);

  // Creates several graphs given the existing graph_def_ and the
  // input feeds and fetches, given 'devices'.
  ::tensorflow::Status CreateGraphs(gtl::ArraySlice<string> feeds,
                                    gtl::ArraySlice<string> fetches,
                                    gtl::ArraySlice<string> target_nodes,
                                    FunctionLibraryDefinition** func_defs,
                                    std::unordered_map<string, Graph*>* outputs,
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
  void WaitForNotification(RunState* run_state, int64 timeout_in_ms);

  const SessionOptions options_;

  // Device structures.
  const std::unique_ptr<const DeviceMgr> device_mgr_;
  std::vector<Device*> devices_;  // not owned
  DeviceSet device_set_;

  string session_handle_;
  bool graph_created_ GUARDED_BY(graph_def_lock_) = false;

  mutex graph_def_lock_;
  GraphDef graph_def_ GUARDED_BY(graph_def_lock_);

  // The thread-pool to use for running ops.
  thread::ThreadPool* thread_pool_ = nullptr;

  // Schedules 'c' for execution.
  void SchedClosure(std::function<void()> c);

  mutex executor_lock_;  // protects executors_
  // Holds mappings from signature to the executors that process
  // it. The reason for a level of indirection around mapped_type is
  // to guarantee address stability.
  std::unordered_map<string, ExecutorsAndKeys*> executors_
      GUARDED_BY(executor_lock_);

  // Holds mappings from handle to partial run state.
  std::unordered_map<string, RunState*> partial_runs_
      GUARDED_BY(executor_lock_);

  // This holds all the tensors that are currently alive in the session.
  SessionState session_state_;

  CancellationManager* cancellation_manager_;

  // Saves and restores device placements for stateful nodes.
  mutex mu_;
  void SaveStatefulNodes(Graph* graph) EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void RestoreStatefulNodes(Graph* graph) EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Map of placed stateful nodes, i.e. nodes for which is_stateful()
  // is true, such as "params" and "queue" nodes.  Once placed these
  // nodes can not be moved to a different device.  Maps node names to
  // device names.
  std::unordered_map<string, string> stateful_placements_ GUARDED_BY(mu_);

  // For generating unique names.
  int64 name_counter_ GUARDED_BY(mu_) = 0;

  // For generating step ids that are unique across all sessions.
  static std::atomic_int_fast64_t step_id_counter_;

  // Global timeout for all blocking operations in this session.
  const int64 operation_timeout_in_ms_ = 0;

  std::unordered_map<const Graph*, CostModel*> cost_models_
      GUARDED_BY(executor_lock_);

  TF_DISALLOW_COPY_AND_ASSIGN(DirectSession);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_DIRECT_SESSION_H_
