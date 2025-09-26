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

#define EIGEN_USE_THREADS

#include <deque>
#include <utility>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shared_ptr_variant.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

class Mutex : public ResourceBase {
 public:
  explicit Mutex(OpKernelContext* c, const string& name)
      : locked_(false),
        thread_pool_(new thread::ThreadPool(
            c->env(), ThreadOptions(),
            absl::StrCat("mutex_lock_thread_", SanitizeThreadSuffix(name)),
            1 /* num_threads */, false /* low_latency_hint */)),
        name_(name) {
    VLOG(2) << "Creating mutex with name " << name << ": " << this;
  }

  string DebugString() const override { return absl::StrCat("Mutex ", name_); }

  class LockReleaser {
   public:
    explicit LockReleaser(Mutex* mutex) : mutex_(mutex) {}

    LockReleaser(const LockReleaser&) = delete;
    LockReleaser& operator=(const LockReleaser&) = delete;

    virtual ~LockReleaser() {
      VLOG(3) << "Destroying LockReleaser " << this << " for mutex: " << mutex_;
      if (mutex_) {
        mutex_lock lock(mutex_->mu_);
        mutex_->locked_ = false;
        mutex_->cv_.notify_all();
        VLOG(3) << "Destroying LockReleaser " << this
                << ": sent notifications.";
      }
    }

   private:
    Mutex* mutex_;
  };

  typedef SharedPtrVariant<LockReleaser> SharedLockReleaser;

  void AcquireAsync(
      OpKernelContext* c,
      std::function<void(const absl::Status& s, SharedLockReleaser lock)> fn) {
    CancellationManager* cm = c->cancellation_manager();
    CancellationToken token{};
    bool* cancelled = nullptr;
    if (cm) {
      cancelled = new bool(false);  // TF_GUARDED_BY(mu_);
      token = cm->get_cancellation_token();
      const bool already_cancelled =
          !cm->RegisterCallback(token, [this, cancelled]() {
            mutex_lock lock(mu_);
            *cancelled = true;
            cv_.notify_all();
          });
      if (already_cancelled) {
        delete cancelled;
        fn(errors::Cancelled("Lock acquisition cancelled."),
           SharedLockReleaser{nullptr});
        return;
      }
    }
    thread_pool_->Schedule(std::bind(
        [this, cm, cancelled,
         token](std::function<void(const absl::Status& s,
                                   SharedLockReleaser&& lock)>
                    fn_) {
          bool local_locked;
          {
            mutex_lock lock(mu_);
            while (locked_ && !(cancelled && *cancelled)) {
              cv_.wait(lock);
            }
            local_locked = locked_ = !(cancelled && *cancelled);
          }
          if (cm) {
            cm->DeregisterCallback(token);
            delete cancelled;
          }
          if (local_locked) {  // Not cancelled.
            fn_(absl::OkStatus(),
                SharedLockReleaser{std::make_shared<LockReleaser>(this)});
          } else {
            fn_(errors::Cancelled("Lock acquisition cancelled."),
                SharedLockReleaser{nullptr});
          }
        },
        std::move(fn)));
  }

 private:
  mutex mu_;
  condition_variable cv_ TF_GUARDED_BY(mu_);
  bool locked_ TF_GUARDED_BY(mu_);
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  string name_;
};

}  // namespace

class MutexLockOp : public AsyncOpKernel {
 public:
  explicit MutexLockOp(OpKernelConstruction* c) : AsyncOpKernel(c) {}

 public:
  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    Mutex* mutex = nullptr;
    OP_REQUIRES_OK_ASYNC(
        c,
        LookupOrCreateResource<Mutex>(c, HandleFromInput(c, 0), &mutex,
                                      [c](Mutex** ptr) {
                                        *ptr = new Mutex(
                                            c, HandleFromInput(c, 0).name());
                                        return absl::OkStatus();
                                      }),
        done);

    Tensor* variant;
    OP_REQUIRES_OK_ASYNC(c, c->allocate_output(0, TensorShape({}), &variant),
                         done);

    mutex->AcquireAsync(
        c, std::bind(
               [c, variant, mutex](DoneCallback done_,
                                   // End of bound arguments.
                                   const absl::Status& s,
                                   Mutex::SharedLockReleaser&& lock) {
                 VLOG(2) << "Finished locking mutex " << mutex
                         << " with lock: " << lock.shared_ptr.get()
                         << " status: " << s.ToString();
                 if (s.ok()) {
                   variant->scalar<Variant>()() = std::move(lock);
                 } else {
                   c->SetStatus(s);
                 }
                 mutex->Unref();
                 done_();
               },
               std::move(done), std::placeholders::_1, std::placeholders::_2));
  }
};

class ConsumeMutexLockOp : public OpKernel {
 public:
  explicit ConsumeMutexLockOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* c) override {
    VLOG(2) << "Executing ConsumeMutexLockOp";
    const Tensor& lock_t = c->input(0);
    OP_REQUIRES(
        c, lock_t.dims() == 0,
        errors::InvalidArgument("Expected input to be a scalar, saw shape: ",
                                lock_t.shape().DebugString()));
    OP_REQUIRES(
        c, lock_t.dtype() == DT_VARIANT,
        errors::InvalidArgument("Expected input to be a variant, saw type: ",
                                DataTypeString(lock_t.dtype())));
    const auto* lock =
        lock_t.scalar<Variant>()().get<Mutex::SharedLockReleaser>();
    OP_REQUIRES(c, lock,
                errors::InvalidArgument(
                    "Expected input to contain a SharedLockReleaser "
                    "object, but saw variant: '",
                    lock_t.scalar<Variant>()().DebugString(), "'"));
    const int use_count = lock->shared_ptr.use_count();
    OP_REQUIRES(
        c, use_count == 1,
        errors::InvalidArgument("Expected use count of lock to be 1, but saw: ",
                                use_count));
  }

  bool IsExpensive() override { return false; }
};

REGISTER_KERNEL_BUILDER(Name("MutexLock").Device(DEVICE_CPU), MutexLockOp);

REGISTER_KERNEL_BUILDER(Name("MutexLock")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("mutex_lock")
                            .HostMemory("mutex"),
                        MutexLockOp);

REGISTER_KERNEL_BUILDER(
    Name("MutexV2").Device(DEVICE_CPU).HostMemory("resource"),
    ResourceHandleOp<Mutex>);

REGISTER_KERNEL_BUILDER(Name("MutexV2").Device(DEVICE_DEFAULT),
                        ResourceHandleOp<Mutex>);

REGISTER_KERNEL_BUILDER(Name("ConsumeMutexLock").Device(DEVICE_CPU),
                        ConsumeMutexLockOp);

REGISTER_KERNEL_BUILDER(
    Name("ConsumeMutexLock").Device(DEVICE_DEFAULT).HostMemory("mutex_lock"),
    ConsumeMutexLockOp);

}  // namespace tensorflow
