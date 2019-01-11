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
#ifndef TENSORFLOW_CONTRIB_GDR_GDR_COLLECTIVE_EXECUTOR_MGR_H_
#define TENSORFLOW_CONTRIB_GDR_GDR_COLLECTIVE_EXECUTOR_MGR_H_

#include "tensorflow/contrib/gdr/gdr_memory_manager.h"
#include "tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.h"
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
// that uses WorkerInterface::RecvBufAsync to route data transfers over RDMA.
class GdrCollectiveExecutorMgr : public RpcCollectiveExecutorMgr {
 public:
  GdrCollectiveExecutorMgr(
      const ConfigProto& config, const DeviceMgr* dev_mgr,
      std::unique_ptr<DeviceResolverDistributed> dev_resolver,
      std::unique_ptr<CollectiveParamResolverDistributed> param_resolver,
      WorkerCacheInterface* worker_cache, const string& task_name,
      RemoteMemoryManager* remote_memory_manager)
      : RpcCollectiveExecutorMgr(config, dev_mgr, std::move(dev_resolver),
                                 std::move(param_resolver), worker_cache,
                                 task_name),
        remote_memory_manager_(remote_memory_manager) {}

  ~GdrCollectiveExecutorMgr() override {}

 protected:
  virtual CollectiveExecutor* Create(int64 step_id) override;

 private:
  RemoteMemoryManager* remote_memory_manager_;  // Not owned.
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CONTRIB_GDR_GDR_COLLECTIVE_EXECUTOR_MGR_H_
