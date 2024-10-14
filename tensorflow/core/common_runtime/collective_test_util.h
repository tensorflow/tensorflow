/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_COLLECTIVE_TEST_UTIL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_COLLECTIVE_TEST_UTIL_H_

#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/test_collective_executor_mgr.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/unbounded_work_queue.h"

namespace tensorflow {

// Wraps CollectiveRemoteAccessLocal with the ability to return an
// error status to the N'th action.
class FailTestRMA : public CollectiveRemoteAccessLocal {
 public:
  FailTestRMA(const DeviceMgr* dev_mgr, DeviceResolverInterface* dev_resolver,
              int64_t step_id);

  // Sets when it should fail. Setting to zero disables the failure.
  void set_fail_after(int fail_after) {
    mutex_lock l(mu_);
    fail_after_ = fail_after;
  }

  void RecvFromPeer(const string& peer_device, const string& peer_task,
                    bool peer_is_local, const string& key, Device* to_device,
                    DeviceContext* to_device_ctx,
                    const AllocatorAttributes& to_alloc_attr, Tensor* to_tensor,
                    const DeviceLocality& client_locality,
                    int dev_to_dev_stream_index,
                    CancellationManager* cancellation_manager,
                    const StatusCallback& done) override;

  void PostToPeer(const string& peer_device, const string& peer_task,
                  const string& key, Device* from_device,
                  DeviceContext* from_device_ctx,
                  const AllocatorAttributes& from_alloc_attr,
                  const Tensor* from_tensor,
                  const DeviceLocality& client_locality,
                  CancellationManager* cancellation_manager,
                  const StatusCallback& done) override;

 private:
  bool MaybeFail(const StatusCallback& done);

  mutex mu_;
  int fail_after_ TF_GUARDED_BY(mu_);
};

struct CollectiveTestEnv {
  int num_workers;
  int num_devices_per_worker;
  DeviceType device_type;
  std::unique_ptr<ParamResolverInterface> param_resolver;
  std::unique_ptr<TestCollectiveExecutorMgr> col_exec_mgr;
  std::shared_ptr<UnboundedWorkQueue> work_queue;
  std::unique_ptr<tensorflow::DeviceMgr> device_mgr;
  std::unique_ptr<DeviceResolverInterface> device_resolver;
  std::unique_ptr<NcclCommunicatorInterface> nccl_communicator;
  core::RefCountPtr<CollectiveExecutor> col_exec;
  FailTestRMA* remote_access;

  CollectiveTestEnv() : device_type(DEVICE_DEFAULT) {}
};

std::unique_ptr<CollectiveTestEnv> CreateCollectiveTestEnv(
    int num_workers, int num_devices_per_worker, DeviceType device_type,
    bool use_nccl = false);

core::RefCountPtr<CollectiveParams> CreateCollectiveParams(
    const CollectiveTestEnv& test_env, int rank, const string& collective_name,
    CollectiveType collective_type, DataType dtype, const TensorShape& shape,
    const std::vector<std::vector<int>> user_specified_rank_per_worker = {{}});

std::vector<int> GenerateEvenSubdivOffsets(int num_devices_per_worker,
                                           int num_subdivs);

// Runs a collective. input and output should be on the host.
absl::Status RunCollective(CollectiveTestEnv* test_env,
                           CollectiveParams* col_params, Device* device,
                           Tensor* input, Tensor* output);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_COLLECTIVE_TEST_UTIL_H_
