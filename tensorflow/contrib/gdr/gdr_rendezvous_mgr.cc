/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/gdr/gdr_rendezvous_mgr.h"

#include "google/protobuf/any.pb.h"
#include "tensorflow/contrib/gdr/gdr_memory_manager.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/request_id.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
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

class GdrRecvTensorCall : public BaseRecvTensorCall {
 public:
  GdrRecvTensorCall(WorkerInterface* wi, Device* dst_device,
                    RemoteMemoryManager* remote_memory_manager,
                    const Rendezvous::Args& recv_args, int64 step_id,
                    StringPiece key)
      : wi_(wi),
        dst_device_(dst_device),
        remote_memory_manager_(remote_memory_manager),
        recv_args_(recv_args) {
    req_.set_step_id(step_id);
    req_.set_rendezvous_key(key.data(), key.size());
    req_.set_request_id(GetUniqueRequestId());
  }

  ~GdrRecvTensorCall() override {}

  void Start(std::function<void()> recv_done) override {
    req_.set_dma_ok(true);
    resp_.InitAlloc(dst_device_, recv_args_.alloc_attrs);
    StatusCallback cb = [this, recv_done](const Status& s) {
      bool dma_ok = resp_.metadata().has_transport_options();
      if (s.ok() && tensor().TotalBytes() > 1024 && (!is_dead()) && dma_ok) {
        auto transport_options = resp_.metadata().transport_options();
        const bool on_host = recv_args_.alloc_attrs.on_host();
        remote_memory_manager_->TensorFromTransportOptions(
            const_cast<Tensor*>(&tensor()), transport_options, dst_device_,
            recv_args_.device_context, on_host,
            [this, recv_done](const Status& s) {
              if (!s.ok()) {
                mutex_lock l(mu_);
                status_.Update(s);
              }
              recv_done();
            });
        return;
      }
      if (!s.ok()) {
        mutex_lock l(mu_);
        status_.Update(s);
      }
      recv_done();
    };
    wi_->RecvTensorAsync(&opts_, &req_, &resp_, std::move(cb));
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

  const Tensor& tensor() const { return resp_.tensor(); }

  bool is_dead() const { return resp_.metadata().is_dead(); }

  Device* dst_device() const { return dst_device_; }

  const Rendezvous::Args& recv_args() const { return recv_args_; }

 private:
  WorkerInterface* wi_;
  Device* dst_device_;
  RemoteMemoryManager* remote_memory_manager_;
  CallOptions opts_;
  RecvTensorRequest req_;
  TensorResponse resp_;
  Rendezvous::Args recv_args_;

  mutable mutex mu_;
  Status status_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(GdrRecvTensorCall);
};

class GdrRemoteRendezvous : public BaseRemoteRendezvous {
 public:
  GdrRemoteRendezvous(const WorkerEnv* env, int64 step_id,
                      RemoteMemoryManager* remote_memory_manager)
      : BaseRemoteRendezvous(env, step_id),
        remote_memory_manager_(remote_memory_manager) {}

 protected:
  void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                           const Rendezvous::Args& recv_args,
                           DoneCallback done) override {
    CHECK(is_initialized());

    string src_worker;
    string src_rel_device;
    if (!DeviceNameUtils::SplitDeviceName(parsed.src_device, &src_worker,
                                          &src_rel_device)) {
      Status s = errors::Internal(parsed.src_device,
                                  " is invalid remote source device.");
      done(s, Args(), recv_args, Tensor{}, false);
      return;
    }

    WorkerSession* sess = session();
    WorkerInterface* rwi = sess->worker_cache->CreateWorker(src_worker);
    if (rwi == nullptr) {
      Status s = errors::Internal("No worker known as ", src_worker);
      done(s, Args(), recv_args, Tensor{}, false);
      return;
    }

    Device* dst_device;
    Status s = sess->device_mgr()->LookupDevice(parsed.dst_device, &dst_device);
    if (!s.ok()) {
      sess->worker_cache->ReleaseWorker(src_worker, rwi);
      done(s, Args(), recv_args, Tensor{}, false);
      return;
    }

    // Prepare a RecvTensor call that can handle being aborted.
    GdrRecvTensorCall* call =
        new GdrRecvTensorCall(rwi, dst_device, remote_memory_manager_,
                              recv_args, step_id_, parsed.FullKey());

    // Record "call" in active_ so that it can be aborted cleanly.
    RegisterCall(call);

    // RendezvousMgr already aborted, shouldn't send RPC call any more
    if (!call->status().ok()) {
      // NOTE: `*session()` can potentially be deleted before we return from
      // `call->done()(...)`, so we must release the worker before calling the
      // callback.
      session()->worker_cache->ReleaseWorker(src_worker, rwi);
      done(call->status(), Args(), Args(), Tensor(), false);
      delete call;
      return;
    }

    // Start "call".
    Ref();
    call->Start([this, call, src_worker, rwi, done]() {
      // Removes "call" from active_. Prevent StartAbort().
      DeregisterCall(call);
      // If StartAbort was called prior to DeregisterCall, then the
      // current status should be bad.
      Status s = call->status();
      // NOTE: `*session()` can potentially be deleted before we return from
      // `call->done()(...)`, so we must release the worker before calling the
      // callback.
      session()->worker_cache->ReleaseWorker(src_worker, rwi);
      done(s, Args(), call->recv_args(), call->tensor(), call->is_dead());
      delete call;
      Unref();
    });
  }

 private:
  ~GdrRemoteRendezvous() override {}

  RemoteMemoryManager* remote_memory_manager_;

  TF_DISALLOW_COPY_AND_ASSIGN(GdrRemoteRendezvous);
};

}  // namespace

GdrRendezvousMgr::GdrRendezvousMgr(const WorkerEnv* env,
                                   RemoteMemoryManager* remote_memory_manager)
    : BaseRendezvousMgr(env), remote_memory_manager_(remote_memory_manager) {}

BaseRemoteRendezvous* GdrRendezvousMgr::Create(int64 step_id,
                                               const WorkerEnv* worker_env) {
  return new GdrRemoteRendezvous(worker_env, step_id, remote_memory_manager_);
}

}  // end namespace tensorflow
