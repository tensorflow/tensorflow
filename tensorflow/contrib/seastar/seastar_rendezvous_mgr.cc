#include "tensorflow/contrib/seastar/seastar_rendezvous_mgr.h"
#include "tensorflow/contrib/seastar/seastar_tensor_coding.h"
#include "tensorflow/contrib/seastar/seastar_worker_cache.h"
#include "tensorflow/contrib/seastar/seastar_worker_interface.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
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
class SeastarRemoteRendezvous : public BaseRemoteRendezvous {
 public:
  SeastarRemoteRendezvous(const WorkerEnv* env, int64 step_id)
      : BaseRemoteRendezvous(env, step_id) {}

 protected:
  void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                           const Rendezvous::Args& args,
                           DoneCallback done) override;

 private:
  ~SeastarRemoteRendezvous() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(SeastarRemoteRendezvous);
};

// Used only to retrieve tensors from remote processes.
class SeastarRecvTensorCall : public BaseRecvTensorCall {
 public:
  SeastarRecvTensorCall() : wi_(nullptr), dst_device_(nullptr) {}

  void Init(WorkerInterface* wi, int64 step_id, StringPiece key,
            AllocatorAttributes alloc_attrs, Device* dst_device,
            const Rendezvous::Args& recv_args, Rendezvous::DoneCallback done) {
    wi_ = wi;
    seastar_wi_ = dynamic_cast<SeastarWorkerInterface*>(wi_);
    alloc_attrs_ = alloc_attrs;
    dst_device_ = dst_device;
    recv_args_ = recv_args;
    done_ = std::move(done);
    req_.set_step_id(step_id);
    req_.set_rendezvous_key(key.data(), key.size());
  }

  void Reset(WorkerCacheInterface* wc) {
    wc->ReleaseWorker(src_worker_, wi_);
    wi_ = nullptr;
    seastar_wi_ = nullptr;
    alloc_attrs_ = AllocatorAttributes();
    dst_device_ = nullptr;
    // We don't clear opts_ and assume that Init will set up the state for
    // opts_ appropriately.
    req_.Clear();
    resp_.Clear();
    {
      mutex_lock l(mu_);
      status_ = Status::OK();
    }
    done_ = nullptr;
  }

  ~SeastarRecvTensorCall() override {
    // Since only the SeastarRecvTensorFreeList will delete an
    // SeastarRecvTensorCall, and it always sets this->wi_ to null when
    // a call object is released to it, we can assert that this->wi_ is
    // always null at the point of deletion.
    CHECK_EQ(static_cast<WorkerInterface*>(nullptr), wi_)
        << "Leaking WorkerInterface in SeastarRecvTensorCall destructor.";
  }

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

  const Tensor& tensor() const { return resp_.GetTensor(); }
  bool is_dead() const { return resp_.GetIsDead(); }
  const Rendezvous::Args& recv_args() const { return recv_args_; }
  const Rendezvous::DoneCallback& done() const { return done_; }

 private:
  friend class SeastarRemoteRendezvous;

  // Start the main RecvTensor call, checking for an async abort.
  void StartRTCall(std::function<void()> recv_done) {
    resp_.InitAlloc(dst_device_, alloc_attrs_);
    using namespace std::placeholders;
    StatusCallback cb = std::bind(
        [this](std::function<void()> recv_done,
               // Begin unbound arguments.
               const Status& s) {
          if (!s.ok()) {
            mutex_lock l(mu_);
            status_.Update(s);
          }
          recv_done();
        },
        std::move(recv_done), _1);
    seastar_wi_->RecvTensorAsync(&opts_, &req_, &resp_, std::move(cb));
  }

private:
  string src_worker_;
  string src_rel_device_;
  WorkerInterface* wi_;
  SeastarWorkerInterface* seastar_wi_;
  AllocatorAttributes alloc_attrs_;
  Device* dst_device_;
  CallOptions opts_;
  RecvTensorRequest req_;
  SeastarTensorResponse resp_;
  Rendezvous::Args recv_args_;
  Rendezvous::DoneCallback done_;

  mutable mutex mu_;
  Status status_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(SeastarRecvTensorCall);
};

class SeastarRecvTensorFreeList {
 public:
  virtual ~SeastarRecvTensorFreeList() {
    for (size_t i = 0; i < objects_.size(); i++) {
      delete objects_[i];
    }
  }

  SeastarRecvTensorCall* New() {
    {
      mutex_lock l(mu_);
      if (!objects_.empty()) {
        SeastarRecvTensorCall* result = objects_.back();
        objects_.pop_back();
        return result;
      }
    }
    return new SeastarRecvTensorCall;
  }

  void Release(SeastarRecvTensorCall* obj, WorkerCacheInterface* wc) {
    obj->Reset(wc);
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
  std::vector<SeastarRecvTensorCall*> objects_ GUARDED_BY(mu_);
};

static SeastarRecvTensorFreeList* get_call_freelist() {
  static SeastarRecvTensorFreeList* call_freelist =
    new SeastarRecvTensorFreeList();
  return call_freelist;
}

void SeastarRemoteRendezvous::RecvFromRemoteAsync(
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& recv_args,
    DoneCallback done) {
  CHECK(is_initialized());
  Status s;

  // Prepare a RecvTensor call that can handle being aborted.
  SeastarRecvTensorCall* call = get_call_freelist()->New();

  // key.src_device identifies a remote device.
  if (!DeviceNameUtils::SplitDeviceName(parsed.src_device, &call->src_worker_,
                                        &call->src_rel_device_)) {
    s = errors::Internal(parsed.src_device,
                         " is invalid remote source device.");
  }
  WorkerSession* sess = session();
  WorkerInterface* rwi = sess->worker_cache->CreateWorker(call->src_worker_);
  if (s.ok() && rwi == nullptr) {
    s = errors::Internal("No worker known as ", call->src_worker_);
  }

  Device* dst_device;
  if (s.ok()) {
    s = sess->device_mgr()->LookupDevice(parsed.dst_device, &dst_device);
  }
  if (!s.ok()) {
    if (rwi != nullptr) {
      sess->worker_cache->ReleaseWorker(call->src_worker_, rwi);
    }
    get_call_freelist()->Release(call, sess->worker_cache.get());
    done(s, Args(), recv_args, Tensor{}, false);
    LOG(ERROR) << "RecvFromRemoteAsync failed, detail " << s.error_message();
    return;
  }
  call->Init(rwi, step_id_, parsed.FullKey(), recv_args.alloc_attrs, dst_device,
             recv_args, std::move(done));

  // Record "call" in active_ so that it can be aborted cleanly.
  RegisterCall(call);
  if (!call->status().ok()) {
    LOG(WARNING) << "Rendezvous has been aborted, ignore the rpc call."
                 << ", rendezvous key: " << parsed.FullKey();
    session()->worker_cache->ReleaseWorker(call->src_worker_, call->wi_);
    call->done()(call->status(), Args(), Args(), Tensor(), false);
    call->wi_ = nullptr;
    get_call_freelist()->Release(call, session()->worker_cache.get());
    return;
  }

  // Start "call".
  Ref();
  call->Start([this, call]() {
    // Removes "call" from active_. Prevent StartAbort().
    DeregisterCall(call);
    // If StartAbort was called prior to DeregisterCall, then the
    // current status should be bad.
    Status s = call->status();
    session()->worker_cache->ReleaseWorker(call->src_worker_, call->wi_);
    call->done()(s, Args(), call->recv_args(), call->tensor(), call->is_dead());
    call->wi_ = nullptr;
    get_call_freelist()->Release(call, session()->worker_cache.get());
    Unref();
  });
}
}  // namespace

SeastarRendezvousMgr::SeastarRendezvousMgr(const WorkerEnv* env)
    : BaseRendezvousMgr(env) {}

BaseRemoteRendezvous* SeastarRendezvousMgr::Create(int64 step_id,
                                               const WorkerEnv* worker_env) {
  return new SeastarRemoteRendezvous(worker_env, step_id);
}

}  // namespace tensorflow
