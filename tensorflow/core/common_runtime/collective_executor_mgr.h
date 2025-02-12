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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_COLLECTIVE_EXECUTOR_MGR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_COLLECTIVE_EXECUTOR_MGR_H_

#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/platform/unbounded_work_queue.h"

namespace tensorflow {
class ConfigProto;
class DeviceMgr;

class CollectiveExecutorMgr : public CollectiveExecutorMgrInterface {
 public:
  CollectiveExecutorMgr(
      const ConfigProto& config, const DeviceMgr* dev_mgr,
      std::unique_ptr<DeviceResolverInterface> dev_resolver,
      std::unique_ptr<ParamResolverInterface> param_resolver,
      std::unique_ptr<NcclCommunicatorInterface> nccl_communicator);

  virtual ~CollectiveExecutorMgr();

  CollectiveExecutor* FindOrCreate(int64_t step_id) override;

  void Cleanup(int64_t step_id) override;

  void CleanupAll() override;

  ParamResolverInterface* GetParamResolver() const override {
    return param_resolver_.get();
  }

  DeviceResolverInterface* GetDeviceResolver() const override {
    return dev_resolver_.get();
  }

  NcclCommunicatorInterface* GetNcclCommunicator() const override {
    return nccl_communicator_.get();
  }

  void GetStepSequenceAsync(const GetStepSequenceRequest* request,
                            GetStepSequenceResponse* response,
                            const StatusCallback& done) override;

  void RefreshStepIdSequenceAsync(int64_t graph_key,
                                  const StatusCallback& done) override;

  int64_t NextStepId(int64_t graph_key) override {
    return CollectiveExecutor::kInvalidId;
  }

  void RetireStepId(int64_t graph_key, int64_t step_id) override {}

 protected:
  // Called by FindOrCreate when table entry does not yet exist.
  virtual CollectiveExecutor* Create(int64_t step_id);

  const DeviceMgr* dev_mgr_;
  std::unique_ptr<DeviceResolverInterface> dev_resolver_;
  std::unique_ptr<ParamResolverInterface> param_resolver_;
  string gpu_ring_order_;
  std::unique_ptr<NcclCommunicatorInterface> nccl_communicator_;
  // Unbounded work queue for scheduling potentially-blocking work during
  // collective op execution.  Ownership is shared between `this` and
  // `CollectiveRemoteAccessLocal`.
  std::shared_ptr<UnboundedWorkQueue> work_queue_;

 private:
  mutex exec_mu_;
  // Map from step_id to CollectiveExecutor
  gtl::FlatMap<int64_t, CollectiveExecutor*> executor_table_
      TF_GUARDED_BY(exec_mu_);
};

// Creates a local CollectiveExecutorMgr with production implementations of each
// components. Cases that need to inject other implementations of these
// components should call CollectiveExecutorMgr constructor directly. This only
// supports a single host. For distributed use case, use
// CreateProdRpcCollectiveExecutorMgr() instead.
std::unique_ptr<CollectiveExecutorMgr> CreateProdLocalCollectiveExecutorMgr(
    const ConfigProto& config, const DeviceMgr* device_mgr,
    std::unique_ptr<NcclCommunicatorInterface> nccl_communicator);

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_COLLECTIVE_EXECUTOR_MGR_H_
