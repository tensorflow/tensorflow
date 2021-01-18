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
#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COLLECTIVE_RMA_DISTRIBUTED_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COLLECTIVE_RMA_DISTRIBUTED_H_

#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/unbounded_work_queue.h"

namespace tensorflow {
class WorkerCacheInterface;

// Extend CollectiveRemoteAccessLocal with access to remote peers.
class CollectiveRemoteAccessDistributed : public CollectiveRemoteAccessLocal {
 public:
  CollectiveRemoteAccessDistributed(
      const DeviceMgr* dev_mgr, DeviceResolverInterface* dev_resolver,
      std::shared_ptr<UnboundedWorkQueue> work_queue,
      WorkerCacheInterface* worker_cache, int64 step_id, string task_name)
      : CollectiveRemoteAccessLocal(dev_mgr, dev_resolver, step_id),
        worker_cache_(worker_cache),
        work_queue_(std::move(work_queue)),
        task_name_(std::move(task_name)) {}

  ~CollectiveRemoteAccessDistributed() override {}

  void RecvFromPeer(const string& peer_device, const string& peer_task,
                    bool peer_is_local, const string& key, Device* to_device,
                    DeviceContext* to_device_ctx,
                    const AllocatorAttributes& to_alloc_attr, Tensor* to_tensor,
                    const DeviceLocality& client_locality,
                    int dev_to_dev_stream_index,
                    CancellationManager* cancellation_manager,
                    const StatusCallback& done) override;

  void CheckPeerHealth(const string& peer_task, int64 timeout_in_ms,
                       const StatusCallback& done) override;

  void StartAbort(const Status& s) override;

 protected:
  WorkerCacheInterface* worker_cache_;  // Not owned
  // Ownership of `work_queue_` is shared between `this` and
  // `CollectiveExecutorMgr`.
  std::shared_ptr<UnboundedWorkQueue> work_queue_;
  CancellationManager abortion_cancel_mgr_;
  string task_name_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COLLECTIVE_RMA_DISTRIBUTED_H_
