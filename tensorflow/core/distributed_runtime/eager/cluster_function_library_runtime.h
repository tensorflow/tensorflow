/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_CLUSTER_FUNCTION_LIBRARY_RUNTIME_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_CLUSTER_FUNCTION_LIBRARY_RUNTIME_H_

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/distributed_runtime/worker_session.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/protobuf/remote_tensor_handle.pb.h"

namespace tensorflow {

class WorkerSession;

namespace eager {

// EagerClusterFunctionLibraryRuntime contains methods to Instantiate and Run
// functions across processes by making RPCs through eager service.
class EagerClusterFunctionLibraryRuntime
    : public DistributedFunctionLibraryRuntime {
 public:
  EagerClusterFunctionLibraryRuntime(const uint64 context_id, EagerContext* ctx,
                                     DeviceMgr* remote_device_mgr)
      : context_id_(context_id),
        ctx_(ctx),
        remote_device_mgr_(remote_device_mgr) {}

  ~EagerClusterFunctionLibraryRuntime() override{};

  void Instantiate(const string& function_name,
                   const FunctionLibraryDefinition& lib_def, AttrSlice attrs,
                   const FunctionLibraryRuntime::InstantiateOptions& options,
                   FunctionLibraryRuntime::LocalHandle* handle,
                   FunctionLibraryRuntime::DoneCallback done) override;

  void Run(const FunctionLibraryRuntime::Options& opts,
           FunctionLibraryRuntime::LocalHandle handle,
           gtl::ArraySlice<Tensor> args, std::vector<Tensor>* rets,
           FunctionLibraryRuntime::DoneCallback done) override;

  void Run(const FunctionLibraryRuntime::Options& opts,
           FunctionLibraryRuntime::LocalHandle handle,
           std::vector<eager::RemoteTensorHandle>* args,
           FunctionLibraryRuntime::DoneCallback done) override;

  void CleanUp(uint64 step_id, FunctionLibraryRuntime::LocalHandle handle,
               FunctionLibraryRuntime::DoneCallback done) override;

  DeviceMgr* remote_device_mgr() const override { return remote_device_mgr_; }

 private:
  const uint64 context_id_;
  EagerContext* ctx_;
  DeviceMgr* remote_device_mgr_;  // not owned.

  struct FunctionData {
    const string target;
    EagerClient* eager_client = nullptr;
    std::unique_ptr<EagerOperation> op;

    FunctionData(const string& target, EagerClient* eager_client,
                 std::unique_ptr<EagerOperation> op)
        : target(target), eager_client(eager_client), op(std::move(op)) {}
  };

  mutable mutex mu_;
  std::vector<FunctionData> function_data_ GUARDED_BY(mu_);
};

DistributedFunctionLibraryRuntime* CreateClusterFLR(
    const uint64 context_id, EagerContext* ctx, WorkerSession* worker_session);

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_CLUSTER_FUNCTION_LIBRARY_RUNTIME_H_
