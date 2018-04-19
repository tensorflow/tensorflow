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
#ifndef TENSORFLOW_COMMON_RUNTIME_COLLECTIVE_EXECUTOR_MGR_H_
#define TENSORFLOW_COMMON_RUNTIME_COLLECTIVE_EXECUTOR_MGR_H_

#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/lib/gtl/flatmap.h"

namespace tensorflow {
class ConfigProto;
class DeviceMgr;

class CollectiveExecutorMgr : public CollectiveExecutorMgrInterface {
 public:
  CollectiveExecutorMgr(const ConfigProto& config, const DeviceMgr* dev_mgr,
                        DeviceResolverInterface* dev_resolver,
                        ParamResolverInterface* param_resolver);

  virtual ~CollectiveExecutorMgr();

  CollectiveExecutor* FindOrCreate(int64 step_id) override;

  void Cleanup(int64 step_id) override;

  ParamResolverInterface* GetParamResolver() const override {
    return param_resolver_.get();
  }

  DeviceResolverInterface* GetDeviceResolver() const override {
    return dev_resolver_.get();
  }

  void GetStepSequenceAsync(const GetStepSequenceRequest* request,
                            GetStepSequenceResponse* response,
                            const StatusCallback& done) override;

  void RefreshStepIdSequenceAsync(int64 graph_key,
                                  const StatusCallback& done) override;

  int64 NextStepId(int64 graph_key) override {
    return CollectiveExecutor::kInvalidId;
  }

  void RetireStepId(int64 graph_key, int64 step_id) override {}

 protected:
  const DeviceMgr* dev_mgr_;
  std::unique_ptr<DeviceResolverInterface> dev_resolver_;
  std::unique_ptr<ParamResolverInterface> param_resolver_;
  CollectiveRemoteAccess* remote_access_;
  string task_name_;
  mutex exec_mu_;
  // Map from step_id to CollectiveExecutor
  gtl::FlatMap<int64, CollectiveExecutor*> executor_table_ GUARDED_BY(exec_mu_);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_COMMON_RUNTIME_COLLECTIVE_EXECUTOR_MGR_H_
