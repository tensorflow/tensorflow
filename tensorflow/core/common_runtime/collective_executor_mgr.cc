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

#include "tensorflow/core/common_runtime/build_graph_options.h"
#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace {
// TODO(tucker): Temporary class just until a real CollectiveExecutor
// implementation is submitted in a later CL.
class DummyCollectiveExecutor : public CollectiveExecutor {
 public:
  explicit DummyCollectiveExecutor(CollectiveExecutorMgr* ce_mgr)
      : CollectiveExecutor(ce_mgr) {}

  ~DummyCollectiveExecutor() override {}

  void RecvFromPeer(const string& peer_device, const string& peer_task,
                    bool peer_is_local, const string& key, Device* to_device,
                    DeviceContext* to_device_ctx,
                    const AllocatorAttributes& to_alloc_attr, Tensor* to_tensor,
                    const DeviceLocality& client_locality,
                    const StatusCallback& done) override {
    done(errors::Internal("Unimplemented"));
  }

  void PostToPeer(const string& peer_device, const string& peer_task,
                  const string& key, Device* from_device,
                  DeviceContext* from_device_ctx,
                  const AllocatorAttributes& from_alloc_attr,
                  const Tensor* from_tensor,
                  const DeviceLocality& client_locality,
                  const StatusCallback& done) override {
    done(errors::Internal("Unimplemented"));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(DummyCollectiveExecutor);
};
}  // namespace

CollectiveExecutorMgr::CollectiveExecutorMgr(
    const ConfigProto& config, const DeviceMgr* dev_mgr,
    DeviceResolverInterface* dev_resolver,
    ParamResolverInterface* param_resolver)
    : dev_mgr_(dev_mgr),
      dev_resolver_(dev_resolver),
      param_resolver_(param_resolver) {}

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
      ce = new DummyCollectiveExecutor(this);
      executor_table_[step_id] = ce;
    }
    ce->Ref();
  }
  return ce;
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
