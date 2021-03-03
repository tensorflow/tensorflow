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
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/cancellable_call.h"
#include "tensorflow/core/distributed_runtime/request_id.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/framework/cancellation.h"
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

void PopulateTensorFromExtra(const RecvBufRespExtra& extra,
                             Tensor* cpu_tensor) {
  char* head = reinterpret_cast<char*>(DMAHelper::base(cpu_tensor));
  for (const auto& tensor_content_chunk : extra.tensor_content()) {
    memcpy(head, std::string(tensor_content_chunk).data(),
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
    CancellationManager* cancellation_manager, const StatusCallback& done) {
  if (peer_is_local) {
    CollectiveRemoteAccessLocal::RecvFromPeer(
        peer_device, peer_task, peer_is_local, key, to_device, to_device_ctx,
        to_alloc_attr, to_tensor, client_locality, dev_to_dev_stream_index,
        cancellation_manager, done);
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
      const int64 total_bytes = to_tensor->TotalBytes();
      if (num_bytes != total_bytes) {
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
        ScopedMemoryDebugAnnotation op_annotation(
            "CollectiveRemoteAccessDistributed::RecvFromPeer"
            "::recv_buf_callback",
            step_id_, "dynamic", to_tensor->dtype(), &to_tensor->shape());
        Tensor* cpu_tensor = new Tensor(cpu_dev->GetAllocator(cpu_attr),
                                        to_tensor->dtype(), to_tensor->shape());
        PopulateTensorFromExtra(extra, cpu_tensor);
        // Then copy it to the GPU.
        CopyTensor::ViaDMA("",  // edge name (non-existent)
                           nullptr /*send_dev_ctx*/, to_device_ctx, cpu_dev,
                           to_device, cpu_attr, to_alloc_attr, cpu_tensor,
                           to_tensor, dev_to_dev_stream_index,
                           [this, cpu_tensor, done](const Status& s) {
                             delete cpu_tensor;
                             // This callback must not block, so execute
                             // done in another thread.
                             work_queue_->Schedule([s, done] { done(s); });
                           });
        delete state;
        return;
      } else {
        // CPU device
        PopulateTensorFromExtra(extra, to_tensor);
      }
    }

    delete state;
    done(s);
  };

  Status s = dev_resolver_->GetDeviceAttributes(peer_device,
                                                &state->server_attributes);
  if (!s.ok()) {
    recv_buf_callback(s);
    return;
  }
  state->call.reset(new RecvBufCall(
      step_id_, peer_device, peer_task, key, to_device, to_device_ctx,
      to_alloc_attr, to_tensor, client_locality, state->server_attributes,
      cancellation_manager, worker_cache_));
  CancellationToken abortion_token =
      abortion_cancel_mgr_.get_cancellation_token();
  bool already_aborted = !abortion_cancel_mgr_.RegisterCallback(
      abortion_token, [state] { state->call->Cancel(); });
  if (already_aborted) {
    recv_buf_callback(errors::Cancelled("collective ops already aborted"));
  } else {
    state->call->Start([this, abortion_token,
                        done = std::move(recv_buf_callback)](const Status& s) {
      abortion_cancel_mgr_.DeregisterCallback(abortion_token);
      done(s);
    });
  }
}

void CollectiveRemoteAccessDistributed::CheckPeerHealth(
    const string& peer_task, int64 timeout_in_ms, const StatusCallback& done) {
  if (peer_task == task_name_) {
    // Fast path if the peer is the worker itself.
    done(Status::OK());
    return;
  }
  // We send a GetStatus RPC with fail_fast=false to check the health of a peer
  // task. If the RPC succeeds, we verify if the peer_device incarnation matches
  // the local record if we have it. Note that DeviceResolverInterface always
  // caches the device attributes.
  WorkerInterface* wi = worker_cache_->GetOrCreateWorker(peer_task);
  if (wi == nullptr) {
    done(errors::InvalidArgument(peer_task,
                                 " not found. It's probably in valid. The "
                                 "valid form is /job:xxx/replica:0/task:N"));
    return;
  }
  auto opts = new CallOptions();
  opts->SetTimeout(timeout_in_ms);
  auto req = new GetStatusRequest();
  auto resp = new GetStatusResponse();
  // Note that fail_fast is not always respected, so we set a timeout as well.
  // We're not using CancellableCall since check health shouldn't need to be
  // cancelled.
  wi->GetStatusAsync(
      opts, req, resp, /*fail_fast*/ true,
      [this, opts, req, resp, wi, peer_task, done](Status s) {
        std::vector<DeviceAttributes> cached_attrs;
        if (s.ok()) {
          s = dev_resolver_->GetAllDeviceAttributes(peer_task, &cached_attrs);
        }
        if (s.ok()) {
          absl::flat_hash_set<uint64> remote_incarnations;
          for (const DeviceAttributes& da : resp->device_attributes()) {
            remote_incarnations.insert(da.incarnation());
          }
          for (const DeviceAttributes& attr : cached_attrs) {
            if (!remote_incarnations.contains(attr.incarnation())) {
              s = errors::FailedPrecondition(
                  attr.name(), " with incarnation ", attr.incarnation(),
                  " is not available. This usually means ", peer_task,
                  " has restarted");
              break;
            }
          }
        } else if (errors::IsNotFound(s)) {
          // Skip validating device incarnation if we don't know what the
          // incarnation should be. The device attribute is cached after the
          // first collective.
          s = Status::OK();
        }
        delete opts;
        delete req;
        delete resp;
        worker_cache_->ReleaseWorker(peer_task, wi);
        done(s);
      });
}

void CollectiveRemoteAccessDistributed::StartAbort(const Status& s) {
  CollectiveRemoteAccessLocal::StartAbort(s);
  abortion_cancel_mgr_.StartCancel();
}

}  // namespace tensorflow
