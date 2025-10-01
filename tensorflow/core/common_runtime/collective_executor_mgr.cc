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

#include <cstddef>

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/common_runtime/build_graph_options.h"
#include "tensorflow/core/common_runtime/collective_param_resolver_local.h"
#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_resolver_local.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

CollectiveExecutorMgr::CollectiveExecutorMgr(
    const ConfigProto& config, const DeviceMgr* dev_mgr,
    std::unique_ptr<DeviceResolverInterface> dev_resolver,
    std::unique_ptr<ParamResolverInterface> param_resolver,
    std::unique_ptr<NcclCommunicatorInterface> nccl_communicator)
    : dev_mgr_(dev_mgr),
      dev_resolver_(std::move(dev_resolver)),
      param_resolver_(std::move(param_resolver)),
      gpu_ring_order_(
          config.gpu_options().experimental().collective_ring_order()),
      nccl_communicator_(std::move(nccl_communicator)),
      work_queue_(std::make_shared<UnboundedWorkQueue>(
          Env::Default(), "collective_ops",
          // Use a 8MB stack size for collective operations. The default stack
          // size is 64KB in thread_manager.cc is not enough for NCCL
          // operations, b/446237508.
          ThreadOptions{.stack_size = 8 * 1024 * 1024})) {}

CollectiveExecutorMgr::~CollectiveExecutorMgr() {
  for (auto iter : executor_table_) {
    iter.second->Unref();
  }
}

CollectiveExecutor* CollectiveExecutorMgr::FindOrCreate(int64_t step_id) {
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

CollectiveExecutor* CollectiveExecutorMgr::Create(int64_t step_id) {
  CollectiveRemoteAccessLocal* rma =
      new CollectiveRemoteAccessLocal(dev_mgr_, dev_resolver_.get(), step_id);
  return new BaseCollectiveExecutor(this, rma, step_id, dev_mgr_, work_queue_);
}

void CollectiveExecutorMgr::Cleanup(int64_t step_id) {
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

void CollectiveExecutorMgr::CleanupAll() {
  gtl::FlatMap<int64_t, CollectiveExecutor*> executor_table;
  {
    mutex_lock l(exec_mu_);
    std::swap(executor_table, executor_table_);
  }
  for (auto iter : executor_table) {
    iter.second->Unref();
  }
}

void CollectiveExecutorMgr::GetStepSequenceAsync(
    const GetStepSequenceRequest* request, GetStepSequenceResponse* response,
    const StatusCallback& done) {
  done(errors::Internal(
      "CollectiveExecutorMgr does not implement GetStepSequence."));
}

void CollectiveExecutorMgr::RefreshStepIdSequenceAsync(
    int64_t graph_key, const StatusCallback& done) {
  done(errors::Internal(
      "CollectiveExecutorMgr does not implement RefreshStepIdSequence."));
}

std::unique_ptr<CollectiveExecutorMgr> CreateProdLocalCollectiveExecutorMgr(
    const ConfigProto& config, const DeviceMgr* device_mgr,
    std::unique_ptr<NcclCommunicatorInterface> nccl_communicator) {
  auto device_resolver = std::make_unique<DeviceResolverLocal>(device_mgr);
  auto param_resolver = std::make_unique<CollectiveParamResolverLocal>(
      config, device_mgr, device_resolver.get(), nccl_communicator.get(),
      "/job:localhost/replica:0/task:0");
  return std::make_unique<CollectiveExecutorMgr>(
      config, device_mgr, std::move(device_resolver), std::move(param_resolver),
      std::move(nccl_communicator));
}

}  // namespace tensorflow
