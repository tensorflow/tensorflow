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
#include "tensorflow/core/distributed_runtime/collective_rma_distributed.h"

#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/cancellable_call.h"
#include "tensorflow/core/distributed_runtime/request_id.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/platform/protobuf_internal.h"
#include "tensorflow/core/protobuf/transport_options.pb.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

namespace {

class RecvBufCall : public CancellableCall {
 public:
  RecvBufCall(int64 step_id, const string& peer_device, const string& peer_task,
              const string& key, Device* to_device,
              DeviceContext* to_device_ctx,
              const AllocatorAttributes& to_alloc_attr, Tensor* to_tensor,
              const DeviceLocality& client_locality,
              const DeviceLocality& server_locality,
              CancellationManager* cancel_mgr, WorkerCacheInterface* wc)
      : CancellableCall(cancel_mgr, peer_task, wc) {
    req_.set_step_id(step_id);
    req_.set_buf_rendezvous_key(key);
    *req_.mutable_client_locality() = client_locality;
    *req_.mutable_server_locality() = server_locality;
    req_.set_num_bytes(to_tensor->TotalBytes());
    req_.set_buf_ptr(reinterpret_cast<int64>(DMAHelper::base(to_tensor)));
    req_.set_src_device(peer_device);
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

void PopulateTensorFromExtra(const RecvBufRespExtra& extra,
                             Tensor* cpu_tensor) {
  char* head = reinterpret_cast<char*>(DMAHelper::base(cpu_tensor));
  for (const auto& tensor_content_chunk : extra.tensor_content()) {
    memcpy(head, tensor_content_chunk.data(),
           tensor_content_chunk.size());
    head += tensor_content_chunk.size();
  }
}
}  // namespace

void CollectiveRemoteAccessDistributed::RecvFromPeer(
    const string& peer_device, const string& peer_task, bool peer_is_local,
    const string& key, Device* to_device, DeviceContext* to_device_ctx,
    const AllocatorAttributes& to_alloc_attr, Tensor* to_tensor,
    const DeviceLocality& client_locality, int dev_to_dev_stream_index,
    const StatusCallback& done) {
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
    DeviceLocality server_locality;
    std::unique_ptr<RecvBufCall> call;
  };
  State* state = new State;

  // Logic to be executed on the RecvBufAsync callback.
  auto recv_buf_callback = [this, state, peer_task, to_device, to_alloc_attr,
                            to_device_ctx, to_tensor, dev_to_dev_stream_index,
                            done](const Status& s) {
    if (s.ok()) {
      // In this generic implementation the bytes come back in the
      // RPC response protobuf rather than via RDMA so we need to copy
      // them into the destination tensor here.
      RecvBufRespExtra extra;
      state->call->resp_.transport_options().UnpackTo(&extra);
      int64 num_bytes = 0;
      for (const auto& chunk : extra.tensor_content()) {
        num_bytes += chunk.size();
      }
      if (num_bytes != to_tensor->TotalBytes()) {
        done(errors::Internal("RecvBufResponse returned ", num_bytes,
                              " bytes where to_tensor expected ",
                              to_tensor->TotalBytes()));
        delete state;
        return;
      }
      if (to_device->tensorflow_gpu_device_info()) {
        // Move the bytes into a CPU tensor then use tensor-to-tensor copy.
        // Use GPU-registered memory for the CPU tensor so the transfer
        // goes faster.
        Device* cpu_dev = nullptr;
        Status status = dev_mgr_->LookupDevice("CPU:0", &cpu_dev);
        if (!status.ok()) {
          done(status);
          delete state;
          return;
        }
        AllocatorAttributes cpu_attr;
        cpu_attr.set_gpu_compatible(true);
        Tensor* cpu_tensor = new Tensor(cpu_dev->GetAllocator(cpu_attr),
                                        to_tensor->dtype(), to_tensor->shape());
        PopulateTensorFromExtra(extra, cpu_tensor);
        // Then copy it to the GPU.
        CopyTensor::ViaDMA("",  // edge name (non-existent)
                           nullptr /*send_dev_ctx*/, to_device_ctx, cpu_dev,
                           to_device, cpu_attr, to_alloc_attr, cpu_tensor,
                           to_tensor, dev_to_dev_stream_index,
                           [cpu_tensor, done](const Status& s) {
                             delete cpu_tensor;
                             // This callback must not block, so execute
                             // done in another thread.
                             SchedClosure([s, done] { done(s); });
                           });
        delete state;
        return;
      } else {
        // CPU device
        PopulateTensorFromExtra(extra, to_tensor);
      }
    }
    if (!s.ok() && errors::IsFailedPrecondition(s)) {
      dev_resolver_->ClearTask(peer_task);
    }

    delete state;
    done(s);
  };

  // Logic to execute once we have the device locality for the server-side
  // device.
  auto dev_locality_callback = [this, state, peer_device, peer_task, key,
                                to_device, to_device_ctx, to_alloc_attr,
                                to_tensor, client_locality,
                                recv_buf_callback](const Status& s) {
    if (!s.ok()) {
      recv_buf_callback(s);
    } else {
      state->call.reset(new RecvBufCall(
          step_id_, peer_device, peer_task, key, to_device, to_device_ctx,
          to_alloc_attr, to_tensor, client_locality, state->server_locality,
          &cancel_mgr_, worker_cache_));
      state->call->Start(recv_buf_callback);
    }
  };

  dev_resolver_->GetLocalityAsync(
      peer_device, peer_task, &state->server_locality, dev_locality_callback);
}

void CollectiveRemoteAccessDistributed::StartAbort(const Status& s) {
  CollectiveRemoteAccessLocal::StartAbort(s);
  cancel_mgr_.StartCancel();
}

}  // namespace tensorflow
