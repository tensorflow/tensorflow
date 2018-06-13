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
#include "tensorflow/core/common_runtime/collective_executor_mgr.h"

#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/common_runtime/build_graph_options.h"
#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

CollectiveExecutorMgr::CollectiveExecutorMgr(
    const ConfigProto& config, const DeviceMgr* dev_mgr,
    std::unique_ptr<DeviceResolverInterface> dev_resolver,
    std::unique_ptr<ParamResolverInterface> param_resolver)
    : dev_mgr_(dev_mgr),
      dev_resolver_(std::move(dev_resolver)),
      param_resolver_(std::move(param_resolver)) {}

CollectiveExecutorMgr::~CollectiveExecutorMgr() {
  for (auto iter : executor_table_) {
    iter.second->Unref();
  }
}

CollectiveExecutor* CollectiveExecutorMgr::FindOrCreate(int64 step_id) {
  CollectiveExecutor* ce = nullptr;
  {
    mutex_lock l(exec_mu_);
    auto it = executor_table_.find(step_id);
    if (it != executor_table_.end()) {
      ce = it->second;
    } else {
      ce = Create(step_id);
      executor_table_[step_id] = ce;
    }
    ce->Ref();
  }
  return ce;
}

CollectiveExecutor* CollectiveExecutorMgr::Create(int64 step_id) {
  CollectiveRemoteAccessLocal* rma =
      new CollectiveRemoteAccessLocal(dev_mgr_, dev_resolver_.get(), step_id);
  return new BaseCollectiveExecutor(this, rma, step_id, dev_mgr_);
}

void CollectiveExecutorMgr::Cleanup(int64 step_id) {
  CollectiveExecutor* ce = nullptr;
  {
    mutex_lock l(exec_mu_);
    auto it = executor_table_.find(step_id);
    if (it != executor_table_.end()) {
      ce = it->second;
      executor_table_.erase(it);
    }
  }
  if (ce) ce->Unref();
}

void CollectiveExecutorMgr::GetStepSequenceAsync(
    const GetStepSequenceRequest* request, GetStepSequenceResponse* response,
    const StatusCallback& done) {
  done(errors::Internal(
      "CollectiveExecutorMgr does not implement GetStepSequence."));
}

void CollectiveExecutorMgr::RefreshStepIdSequenceAsync(
    int64 graph_key, const StatusCallback& done) {
  done(errors::Internal(
      "CollectiveExecutorMgr does not implement RefreshStepIdSequence."));
}

}  // namespace tensorflow
