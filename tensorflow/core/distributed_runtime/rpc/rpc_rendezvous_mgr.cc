/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"

#include <unordered_set>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

class RpcRemoteRendezvous : public BaseRemoteRendezvous {
 public:
  RpcRemoteRendezvous(const WorkerEnv* env, int64 step_id)
      : BaseRemoteRendezvous(env, step_id, false) {}

 protected:
  void RecvFromRemoteAsync(const string& key,
                           const Rendezvous::ParsedKey& parsed,
                           const Rendezvous::Args& args,
                           DoneCallback done) override;

 private:
  ~RpcRemoteRendezvous() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(RpcRemoteRendezvous);
};

// Used only to retrieve tensors from remote processes.
class RpcRecvTensorCall : public BaseRecvTensorCall {
 public:
  RpcRecvTensorCall(WorkerCacheInterface* wc, WorkerInterface* wi,
                       int64 step_id, const string& key,
                       const string& remote_dev, Allocator* allocator,
                       Device* dst_device)
      : wi_(wi),
        wc_(wc),
        remote_dev_(remote_dev),
        allocator_(allocator),
        dst_(dst_device) {
    req_.set_step_id(step_id);
    req_.set_rendezvous_key(key);
  }

  ~RpcRecvTensorCall() override { delete wi_; }

  void Start(std::function<void()> recv_done) override {
    StartRTCall(recv_done);
  }

  void StartAbort(const Status& s) override {
    {
      mutex_lock l(mu_);
      status_.Update(s);
    }
    opts_.StartCancel();
  }

  Status status() const override {
    mutex_lock l(mu_);
    return status_;
  }

  const TensorProto& tensor_proto() const { return resp_.tensor(); }

  const RecvTensorResponse& response() const { return resp_; }

  bool is_dead() const { return resp_.is_dead(); }

 private:
  // Start the main RecvTensor call, checking for an async abort.
  void StartRTCall(std::function<void()> recv_done) {
    wi_->RecvTensorAsync(&opts_, &req_, &resp_,
                         nullptr /* TensorBufAllocator */,
                         // done callback
                         [this, recv_done](const Status& s) {
                           {
                             mutex_lock l(mu_);
                             status_.Update(s);
                           }
                           recv_done();
                         });
  }

  WorkerInterface* wi_;       // Owned.
  WorkerCacheInterface* wc_;  // Not owned.
  string remote_dev_;
  Allocator* allocator_;
  Device* dst_;
  CallOptions opts_;
  RecvTensorRequest req_;
  RecvTensorResponse resp_;

  mutable mutex mu_;
  Status status_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(RpcRecvTensorCall);
};


void RpcRemoteRendezvous::RecvFromRemoteAsync(
    const string& key, const Rendezvous::ParsedKey& parsed,
    const Rendezvous::Args& recv_args, DoneCallback done) {
  Status s;

  // key.src_device identifies a remote device.
  string src_worker;
  string src_rel_device;
  if (!DeviceNameUtils::SplitDeviceName(parsed.src_device, &src_worker,
                                        &src_rel_device)) {
    s = errors::Internal(parsed.src_device,
                         " is invalid remote source device.");
  }
  WorkerCacheInterface* worker_cache = env_->worker_cache;
  if (s.ok() && worker_cache == nullptr) {
    s = errors::Internal("No remote worker cache available.");
  }
  WorkerInterface* rwi = env_->worker_cache->CreateWorker(src_worker);
  if (s.ok() && rwi == nullptr) {
    s = errors::Internal("No worker known as ", src_worker);
  }

  Device* dst_device;
  if (s.ok()) {
    s = env_->device_mgr->LookupDevice(parsed.dst_device, &dst_device);
  }
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor{}, false);
    return;
  }
  Allocator* allocator = dst_device->GetAllocator(recv_args.alloc_attrs);

  // Prepare a RecvTensor call that can handle being aborted.
  RpcRecvTensorCall* call =
      new RpcRecvTensorCall(worker_cache, rwi, step_id_, key,
                               parsed.src_device, allocator, dst_device);

  // Record "call" in active_ so that it can be aborted cleanly.
  RegisterCall(call);

  // Start "call".
  call->Start([this, call, parsed, recv_args, done]() {
    // Removes "call" from active_. Prevent StartAbort().
    DeregisterCall(call);
    // If StartAbort was called prior to DeregisterCall, then the
    // current status should be bad.
    Status s = call->status();
    Tensor val;
    if (s.ok()) {
      Device* dst_device;
      s = env_->device_mgr->LookupDevice(parsed.dst_device, &dst_device);
      if (s.ok()) {
        s = dst_device->MakeTensorFromProto(call->tensor_proto(),
                                            recv_args.alloc_attrs, &val);
      }
    }
    done(s, Args(), recv_args, val, call->is_dead());
    delete call;
  });
}

}  // namespace

BaseRemoteRendezvous* RpcRendezvousMgr::Create(int64 step_id,
                                                  const WorkerEnv* worker_env) {
  return new RpcRemoteRendezvous(worker_env, step_id);
}


}  // end namespace tensorflow
