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
#include "tensorflow/contrib/gdr/gdr_collective_executor_mgr.h"

#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/common_runtime/collective_executor_mgr.h"
#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/distributed_runtime/cancellable_call.h"
#include "tensorflow/core/distributed_runtime/collective_param_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/request_id.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/unbounded_work_queue.h"

namespace tensorflow {

class WorkerCacheInterface;

namespace {

class RecvBufCall : public CancellableCall {
 public:
  RecvBufCall(int64 step_id, const string& peer_device, const string& peer_task,
              const string& key, Device* to_device,
              DeviceContext* to_device_ctx,
              const AllocatorAttributes& to_alloc_attr, Tensor* to_tensor,
              const DeviceLocality& client_locality,
              const DeviceAttributes& server_attributes,
              CancellationManager* cancel_mgr, WorkerCacheInterface* wc)
      : CancellableCall(cancel_mgr, peer_task, wc) {
    req_.set_step_id(step_id);
    req_.set_buf_rendezvous_key(key);
    *req_.mutable_client_locality() = client_locality;
    *req_.mutable_server_locality() = server_attributes.locality();
    req_.set_num_bytes(to_tensor->TotalBytes());
    req_.set_buf_ptr(reinterpret_cast<int64>(DMAHelper::base(to_tensor)));
    req_.set_src_device(peer_device);
    req_.set_src_incarnation(server_attributes.incarnation());
    req_.set_dst_device(to_device->name());
    req_.set_request_id(GetUniqueRequestId());
  }

  ~RecvBufCall() override {}

  void IssueCall(const StatusCallback& done) override {
    wi_->RecvBufAsync(&opts_, &req_, &resp_, done);
  }

  RecvBufRequest req_;
  RecvBufResponse resp_;
};

class CollectiveRemoteAccessDistributed : public CollectiveRemoteAccessLocal {
 public:
  CollectiveRemoteAccessDistributed(const DeviceMgr* dev_mgr,
                                    DeviceResolverInterface* dev_resolver,
                                    UnboundedWorkQueue* work_queue,
                                    WorkerCacheInterface* worker_cache,
                                    int64 step_id,
                                    RemoteMemoryManager* remote_memory_manager)
      : CollectiveRemoteAccessLocal(dev_mgr, dev_resolver, work_queue, step_id),
        worker_cache_(worker_cache),
        remote_memory_manager_(remote_memory_manager) {}

  ~CollectiveRemoteAccessDistributed() override {}

  void RecvFromPeer(const string& peer_device, const string& peer_task,
                    bool peer_is_local, const string& key, Device* to_device,
                    DeviceContext* to_device_ctx,
                    const AllocatorAttributes& to_alloc_attr, Tensor* to_tensor,
                    const DeviceLocality& client_locality,
                    int dev_to_dev_stream_index,
                    const StatusCallback& done) override {
    if (peer_is_local) {
      CollectiveRemoteAccessLocal::RecvFromPeer(
          peer_device, peer_task, peer_is_local, key, to_device, to_device_ctx,
          to_alloc_attr, to_tensor, client_locality, dev_to_dev_stream_index,
          done);
      return;
    }

    // State that needs to be threaded through a couple of async calls
    // in order to make this function completely non-blocking.
    struct State {
      DeviceAttributes server_attributes;
      std::unique_ptr<RecvBufCall> call;
    };
    State* state = new State;

    // Logic to be executed on the RecvBufAsync callback.
    auto recv_buf_callback = [this, state, peer_task, to_device, to_alloc_attr,
                              to_device_ctx, to_tensor, done](const Status& s) {
      if (s.ok()) {
        remote_memory_manager_->TensorFromTransportOptions(
            to_tensor, state->call->resp_.transport_options(), to_device,
            to_device_ctx, to_alloc_attr.on_host(), done);
      } else if (errors::IsFailedPrecondition(s)) {
        dev_resolver_->ClearTask(peer_task);
        done(s);
      }

      delete state;
    };

    // Logic to execute once we have the device attributes for the server-side
    // device.
    auto dev_attributes_callback = [this, state, peer_device, peer_task, key,
                                    to_device, to_device_ctx, to_alloc_attr,
                                    to_tensor, client_locality,
                                    recv_buf_callback](const Status& s) {
      if (!s.ok()) {
        recv_buf_callback(s);
      } else {
        state->call.reset(new RecvBufCall(
            step_id_, peer_device, peer_task, key, to_device, to_device_ctx,
            to_alloc_attr, to_tensor, client_locality, state->server_attributes,
            &cancel_mgr_, worker_cache_));
        state->call->Start(recv_buf_callback);
      }
    };

    dev_resolver_->GetDeviceAttributesAsync(peer_device, peer_task,
                                            &state->server_attributes,
                                            dev_attributes_callback);
  }

  void StartAbort(const Status& s) override {
    CollectiveRemoteAccessLocal::StartAbort(s);
    cancel_mgr_.StartCancel();
  }

 protected:
  WorkerCacheInterface* worker_cache_;  // Not owned
  CancellationManager cancel_mgr_;
  RemoteMemoryManager* remote_memory_manager_;
};

}  // namespace

CollectiveExecutor* GdrCollectiveExecutorMgr::Create(int64 step_id) {
  CollectiveRemoteAccessDistributed* rma =
      new CollectiveRemoteAccessDistributed(dev_mgr_, dev_resolver_.get(),
                                            &work_queue_, worker_cache_,
                                            step_id, remote_memory_manager_);
  return new BaseCollectiveExecutor(this, rma, step_id, dev_mgr_,
                                    &gpu_ring_order_);
}

}  // namespace tensorflow
