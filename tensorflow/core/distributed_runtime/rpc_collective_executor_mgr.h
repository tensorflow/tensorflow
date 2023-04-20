/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_COLLECTIVE_EXECUTOR_MGR_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_COLLECTIVE_EXECUTOR_MGR_H_

#include "tensorflow/core/common_runtime/collective_executor_mgr.h"
#include "tensorflow/core/framework/collective.h"

namespace tensorflow {
class CollectiveParamResolverDistributed;
class ConfigProto;
class DeviceMgr;
class DeviceResolverDistributed;
class WorkerCacheInterface;
class StepSequenceRequest;
class StepSequenceResponse;

// An implementation of CollectiveExecutorMgr for a distributed environment
// that uses WorkerInterface::RecvBufAsync to route data transfers over RPCs.
//
// In some execution environments it may be possible to implement a
// higher-performance solution and use it in place of this class.
class RpcCollectiveExecutorMgr : public CollectiveExecutorMgr {
 public:
  RpcCollectiveExecutorMgr(
      const ConfigProto& config, const DeviceMgr* dev_mgr,
      std::unique_ptr<DeviceResolverDistributed> dev_resolver,
      std::unique_ptr<CollectiveParamResolverDistributed> param_resolver,
      std::unique_ptr<NcclCommunicatorInterface> nccl_communicator,
      WorkerCacheInterface* worker_cache, const string& task_name);

  virtual ~RpcCollectiveExecutorMgr();

  // This function should only be called at the group_leader, by an RPC.
  // Other needs for StepIds should be satisfied by NextStepId.
  void GetStepSequenceAsync(const GetStepSequenceRequest* request,
                            GetStepSequenceResponse* response,
                            const StatusCallback& done) override;

  void RefreshStepIdSequenceAsync(int64_t graph_key,
                                  const StatusCallback& done) override;

  int64_t NextStepId(int64_t graph_key) override;

  void RetireStepId(int64_t graph_key, int64_t step_id) override;

 protected:
  virtual CollectiveExecutor* Create(int64_t step_id) override;

  WorkerCacheInterface* const worker_cache_;  // Not owned.
  const string task_name_;
  string group_leader_;
  friend class RpcCollectiveExecutorMgrTest;

 private:
  Status UpdateStepSequences(const GetStepSequenceResponse& resp);

  // This class maintains the step_id sequencing for a single
  // collective_graph_key.
  struct GraphKeySequence {
    explicit GraphKeySequence(int64_t k)
        : graph_key_(k), next_step_id_(CollectiveExecutor::kInvalidId) {}

    const int64_t graph_key_;
    int64_t next_step_id_;
  };

  mutex sequence_mu_;
  gtl::FlatMap<int64_t, GraphKeySequence*> sequence_table_
      TF_GUARDED_BY(sequence_mu_);
};

// Creates a distributed CollectiveExecutorMgr with production implementations
// of each components. Cases that need to inject other implementations of these
// components should call CollectiveExecutorMgr constructor directly.
std::unique_ptr<RpcCollectiveExecutorMgr> CreateProdRpcCollectiveExecutorMgr(
    const ConfigProto& config, const DeviceMgr* device_mgr,
    WorkerCacheInterface* worker_cache, const string& default_worker_name);

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_COLLECTIVE_EXECUTOR_MGR_H_
