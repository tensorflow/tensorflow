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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_TEST_COLLECTIVE_EXECUTOR_MGR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_TEST_COLLECTIVE_EXECUTOR_MGR_H_

#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/gtl/flatmap.h"

namespace tensorflow {

// Mock objects that can't actually execute a Collective, but satisfy
// general infrastructure expectations within tests that don't require
// full functionality.

class TestCollectiveExecutor : public CollectiveExecutor {
 public:
  explicit TestCollectiveExecutor(CollectiveExecutorMgrInterface* cem,
                                  CollectiveRemoteAccess* rma = nullptr)
      : CollectiveExecutor(cem), rma_(rma) {}

  void RunClosure(std::function<void()> fn) override { fn(); }

  CollectiveRemoteAccess* remote_access() override { return rma_; }

 private:
  CollectiveRemoteAccess* rma_;
};

class TestParamResolver : public ParamResolverInterface {
  void CompleteParamsAsync(const DeviceAttributes& device, CollectiveParams* cp,
                           CancellationManager* cancel_mgr,
                           const StatusCallback& done) override {
    done(errors::Internal("Unimplemented"));
  }

  void CompleteGroupAsync(const DeviceAttributes& device,
                          CollGroupParams* group_params,
                          CancellationManager* cancel_mgr,
                          const StatusCallback& done) override {
    done(errors::Internal("Unimplemented"));
  }

  void CompleteInstanceAsync(const CompleteInstanceRequest* request,
                             CompleteInstanceResponse* response,
                             CancellationManager* cancel_mgr,
                             const StatusCallback& done) override {
    done(errors::Internal("Unimplemented"));
  }

  absl::Status LookupGroup(int32_t group_key, CollGroupParams* group) override {
    return errors::Internal("Unimplemented");
  }

  void StartAbort(const absl::Status& s) override {}
};

class TestCollectiveExecutorMgr : public CollectiveExecutorMgrInterface {
 public:
  explicit TestCollectiveExecutorMgr(ParamResolverInterface* param_resolver,
                                     CollectiveRemoteAccess* rma)
      : param_resolver_(param_resolver), rma_(rma) {}

  TestCollectiveExecutorMgr() : param_resolver_(nullptr), rma_(nullptr) {}

  ~TestCollectiveExecutorMgr() override {
    for (auto& iter : table_) {
      iter.second->Unref();
    }
  }

  CollectiveExecutor* FindOrCreate(int64_t step_id) override {
    mutex_lock l(mu_);
    CollectiveExecutor* ce = nullptr;
    auto iter = table_.find(step_id);
    if (iter != table_.end()) {
      ce = iter->second;
    } else {
      ce = new TestCollectiveExecutor(this, rma_);
      table_[step_id] = ce;
    }
    ce->Ref();
    return ce;
  }

  void Cleanup(int64_t step_id) override {
    mutex_lock l(mu_);
    auto iter = table_.find(step_id);
    if (iter != table_.end()) {
      iter->second->Unref();
      table_.erase(iter);
    }
  }

  void CleanupAll() override {
    mutex_lock l(mu_);
    for (auto& iter : table_) {
      iter.second->Unref();
    }
    table_.clear();
  }

  ParamResolverInterface* GetParamResolver() const override {
    return param_resolver_;
  }

  DeviceResolverInterface* GetDeviceResolver() const override {
    LOG(FATAL);
    return nullptr;
  }

  NcclCommunicatorInterface* GetNcclCommunicator() const override {
    return nullptr;
  }

  void GetStepSequenceAsync(const GetStepSequenceRequest* request,
                            GetStepSequenceResponse* response,
                            const StatusCallback& done) override {
    done(errors::Internal("unimplemented"));
  }

  void RefreshStepIdSequenceAsync(int64_t graph_key,
                                  const StatusCallback& done) override {
    done(errors::Internal("unimplemented"));
  }

  int64_t NextStepId(int64_t graph_key) override {
    return CollectiveExecutor::kInvalidId;
  }

  void RetireStepId(int64_t graph_key, int64_t step_id) override {}

 protected:
  mutex mu_;
  gtl::FlatMap<int64_t, CollectiveExecutor*> table_ TF_GUARDED_BY(mu_);
  ParamResolverInterface* param_resolver_;
  CollectiveRemoteAccess* rma_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_TEST_COLLECTIVE_EXECUTOR_MGR_H_
