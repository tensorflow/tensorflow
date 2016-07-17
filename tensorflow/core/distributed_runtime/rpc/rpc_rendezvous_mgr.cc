/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
  void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                           const Rendezvous::Args& args,
                           DoneCallback done) override;

 private:
  ~RpcRemoteRendezvous() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(RpcRemoteRendezvous);
};

// Used only to retrieve tensors from remote processes.
class RpcRecvTensorCall : public BaseRecvTensorCall {
 public:
  RpcRecvTensorCall()
      : wi_(nullptr), wc_(nullptr), allocator_(nullptr), dst_device_(nullptr) {}

  void Init(WorkerCacheInterface* wc, WorkerInterface* wi, int64 step_id,
            StringPiece key, Allocator* allocator, Device* dst_device,
            const Rendezvous::Args& recv_args, Rendezvous::DoneCallback done) {
    wi_ = wi;
    wc_ = wc;
    allocator_ = allocator;
    dst_device_ = dst_device;
    recv_args_ = recv_args;
    done_ = std::move(done);
    req_.set_step_id(step_id);
    req_.set_rendezvous_key(key.data(), key.size());
  }

  void Reset() {
    delete wi_;
    wi_ = nullptr;
    wc_ = nullptr;
    allocator_ = nullptr;
    dst_device_ = nullptr;
    // We don't clear opts_ and assume that Init will set up the state for
    // opts_ appropriately.
    req_.Clear();
    if (resp_.ByteSize() > 128) {
      // Clear memory from resp_ if it is too large
      RecvTensorResponse empty;
      resp_.Swap(&empty);
    } else {
      resp_.Clear();
    }
    {
      mutex_lock l(mu_);
      status_ = Status::OK();
    }
    done_ = nullptr;
  }

  ~RpcRecvTensorCall() override { delete wi_; }

  void Start(std::function<void()> recv_done) override {
    StartRTCall(std::move(recv_done));
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

  Device* dst_device() const { return dst_device_; }
  const Rendezvous::Args& recv_args() const { return recv_args_; }
  const Rendezvous::DoneCallback& done() const { return done_; }

 private:
  // Start the main RecvTensor call, checking for an async abort.
  void StartRTCall(std::function<void()> recv_done) {
    wi_->RecvTensorAsync(&opts_, &req_, &resp_,
                         nullptr /* TensorBufAllocator */,
                         // done callback
                         [this, recv_done](const Status& s) {
                           if (!s.ok()) {
                             mutex_lock l(mu_);
                             status_.Update(s);
                           }
                           recv_done();
                         });
  }

  WorkerInterface* wi_;       // Owned.
  WorkerCacheInterface* wc_;  // Not owned.
  Allocator* allocator_;
  Device* dst_device_;
  CallOptions opts_;
  RecvTensorRequest req_;
  RecvTensorResponse resp_;
  Rendezvous::Args recv_args_;
  Rendezvous::DoneCallback done_;

  mutable mutex mu_;
  Status status_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(RpcRecvTensorCall);
};

namespace {
class RpcRecvTensorFreeList {
 public:
  RpcRecvTensorFreeList() {}
  ~RpcRecvTensorFreeList() {
    for (int i = 0; i < objects_.size(); i++) {
      delete objects_[i];
    }
  }

  RpcRecvTensorCall* New() {
    {
      mutex_lock l(mu_);
      if (!objects_.empty()) {
        RpcRecvTensorCall* result = objects_.back();
        objects_.pop_back();
        return result;
      }
    }
    return new RpcRecvTensorCall;
  }

  void Release(RpcRecvTensorCall* obj) {
    obj->Reset();
    {
      mutex_lock l(mu_);
      if (objects_.size() < kMaxObjects) {
        objects_.push_back(obj);
        return;
      }
    }
    delete obj;
  }

 private:
  static const int kMaxObjects = 1000;

  mutex mu_;
  std::vector<RpcRecvTensorCall*> objects_ GUARDED_BY(mu_);
};

static RpcRecvTensorFreeList call_freelist_;
}

void RpcRemoteRendezvous::RecvFromRemoteAsync(
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& recv_args,
    DoneCallback done) {
  Status s;

  // key.src_device identifies a remote device.
  string src_worker;
  string src_rel_device;
  if (!DeviceNameUtils::SplitDeviceName(parsed.src_device, &src_worker,
                                        &src_rel_device)) {
    s = errors::Internal(parsed.src_device,
                         " is invalid remote source device.");
  }
  // TODO(jeff): Consider checking for a valid worker_cache during the
  // constructor of RpcRemoteRendezvous, rather than here, to simplify
  // the twisty logic below.
  WorkerCacheInterface* worker_cache = env_->worker_cache;
  if (s.ok() && worker_cache == nullptr) {
    s = errors::Internal("No remote worker cache available.");
  }
  WorkerInterface* rwi =
      (worker_cache ? worker_cache->CreateWorker(src_worker) : nullptr);
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
  RpcRecvTensorCall* call = call_freelist_.New();

  call->Init(worker_cache, rwi, step_id_, parsed.FullKey(), allocator,
             dst_device, recv_args, std::move(done));

  // Record "call" in active_ so that it can be aborted cleanly.
  RegisterCall(call);

  // Start "call".
  call->Start([this, call]() {
    // Removes "call" from active_. Prevent StartAbort().
    DeregisterCall(call);
    // If StartAbort was called prior to DeregisterCall, then the
    // current status should be bad.
    Status s = call->status();
    Tensor val;
    if (s.ok()) {
      s = call->dst_device()->MakeTensorFromProto(
          call->tensor_proto(), call->recv_args().alloc_attrs, &val);
    }
    call->done()(s, Args(), call->recv_args(), val, call->is_dead());
    call_freelist_.Release(call);
  });
}

}  // namespace

BaseRemoteRendezvous* RpcRendezvousMgr::Create(int64 step_id,
                                               const WorkerEnv* worker_env) {
  return new RpcRemoteRendezvous(worker_env, step_id);
}

}  // end namespace tensorflow
