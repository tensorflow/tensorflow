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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
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

// A thread-safe Notification container.  The NotificationResource
// keeps track of a shared condition variable and registered waiter callbacks.
// It exposes three methods: `RegisterWaiter`, `Notify`, and `Reset`.
//
// Users register a Callback with an optional timeout via `RegisterWaiter`.
// When another user calls `Notify`, any callbacks currently registered
// are called and then removed.  If `Notify` is not called within
// a the timeout period of a given Callback, the callback will be called with
// the `notified` argument set `false`.
//
// The Notification object is stateful: if `Notify(true)` is called, the state
// is reset after all callbacks are executed.  This is equivalent to calling
// `Notify(false); Reset()` in one atomic operation.
//
// If `Notify(false)` is called, any future calls to `RegisterWaiter`
// immediately execute the callback.  The Notification may be reset by
// subsequently calling `Reset()` or `Notify(true)`
//
// Example usage:
//
//  auto callback = [...](const Status& s, bool notified) {
//    ...
//  };
//  cv->RegisterWaiter(callback, -1);  // Never time out.
//  ...
//  // Set and then reset notification.  callback(s) are immediately executed.
//  cv->Notify(true);
//  cv->RegisterWaiter(callback, 1e6);  // Set to timeout and
//  ...
//  // Set the notification.  callback(s) are immediately executed.
//  cv->Notify(false);
//  ...
//  // Notification is already set, `callback` is immediately executed.
//  cv->RegisterWaiter(callback, 0);
//  ...
//  // Subsequent calls to RegisterWaiter will not immediately execute.
//  cv->Reset();
//
class NotificationResource : public ResourceBase {
 public:
  typedef std::function<void(Status s, bool notified)> Callback;

  explicit NotificationResource(OpKernelContext* c, const string& name)
      : env_(c->env()),
        thread_pool_(new thread::ThreadPool(
            c->env(), ThreadOptions(),
            strings::StrCat("notification_notify_thread_",
                            SanitizeThreadSuffix(name)),
            1 /* num_threads */, true /* low_latency_hint */)),
        notified_(false),
        mode_(RUNNING) {
    thread_pool_->Schedule([this, c]() {
      mutex_lock lock(mu_);
      while (mode_ == RUNNING || !waiters_.empty()) {
        uint64 time_now = env_->NowMicros();
        // At most 1 second from now.
        uint64 next_visit = time_now + 1e6;
        auto waiter = waiters_.begin();
        while (waiter != waiters_.end()) {
          if (time_now >= waiter->second.timeout_time_in_us) {
            CancellationManager* cm = waiter->first.first;
            if (cm) {
              cm->DeregisterCallback(waiter->first.second);
            }
            // We timed out before getting to this one.
            waiter->second.callback(/*s=*/Status::OK(), /*notified=*/false);
            waiter = waiters_.erase(waiter);
          } else {
            next_visit =
                std::min(next_visit, waiter->second.timeout_time_in_us);
            ++waiter;
          }
        }
        cv_.wait_for(lock,
                     std::chrono::microseconds(next_visit - time_now + 1));
      }
      mode_ = FINISHED;
      cv_.notify_all();
    });
  }

  ~NotificationResource() override {
    mutex_lock lock(mu_);
    mode_ = EXITING;
    while (mode_ != FINISHED) {
      cv_.wait(lock);
    }
    DCHECK(waiters_.empty());
  }

  void Notify(bool reset_immediately) {
    mutex_lock lock(mu_);
    for (auto& waiter : waiters_) {
      CancellationManager* cm = waiter.first.first;
      if (cm) cm->DeregisterCallback(waiter.first.second);
      waiter.second.callback(/*s=*/Status::OK(), /*notified=*/true);
    }
    waiters_.clear();
    notified_ = !reset_immediately;
    cv_.notify_all();
  }

  void Reset() {
    mutex_lock lock(mu_);
    notified_ = false;
  }

  void RegisterWaiter(OpKernelContext* c, Callback callback,
                      uint64 timeout_time_in_us) {
    mutex_lock lock(mu_);

    if (mode_ != RUNNING) {
      callback(/*s=*/errors::Cancelled("Notification cancelled."),
               /*notified=*/false);
      return;
    }

    if (notified_) {
      callback(/*s=*/Status::OK(), /*notified=*/true);
      return;
    }

    CancellationManager* cm = c->cancellation_manager();
    CancellationToken token =
        (cm) ? cm->get_cancellation_token() : waiters_.size();

    if (cm) {
      const bool already_cancelled =
          !cm->RegisterCallback(token, [this, token, cm]() {
            mutex_lock lock_(mu_);
            auto waiter = waiters_.find(std::make_pair(cm, token));
            if (waiter != waiters_.end()) {
              waiter->second.callback(
                  /*s=*/errors::Cancelled("Notification cancelled."),
                  /*notified=*/false);
              waiters_.erase(waiter);
            }
          });
      if (already_cancelled) {
        callback(/*s=*/errors::Cancelled("Notification cancelled."),
                 /*notified=*/false);
        return;
      }
    }
    waiters_.emplace(
        std::make_pair(cm, token),
        CallbackAndTimeout{std::move(callback), timeout_time_in_us});
    cv_.notify_all();
  }

  string DebugString() override { return "Notification"; }

 private:
  OpKernelContext* c_;
  Env* env_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  mutex mu_;
  bool notified_;
  struct CallbackAndTimeout {
    Callback callback;
    uint64 timeout_time_in_us;
  };
  typedef std::pair<CancellationManager*, CancellationToken> CMAndToken;
  gtl::FlatMap<CMAndToken, CallbackAndTimeout> waiters_ GUARDED_BY(mu_);
  condition_variable cv_ GUARDED_BY(mu_);

  enum RunMode { RUNNING, EXITING, FINISHED };
  RunMode mode_ GUARDED_BY(mu_);
};

class Mutex : public ResourceBase {
 public:
  explicit Mutex(OpKernelContext* c, const string& name)
      : locked_(false),
        thread_pool_(new thread::ThreadPool(
            c->env(), ThreadOptions(),
            strings::StrCat("mutex_lock_thread_", SanitizeThreadSuffix(name)),
            1 /* num_threads */, false /* low_latency_hint */)),
        name_(name) {
    VLOG(2) << "Creating mutex with name " << name << ": " << this;
  }

  string DebugString() override { return strings::StrCat("Mutex ", name_); }

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

  struct SharedLockReleaser {
    std::shared_ptr<LockReleaser> shared_lock;

    explicit SharedLockReleaser(std::shared_ptr<LockReleaser>&& lock)
        : shared_lock(std::forward<decltype(lock)>(lock)) {
      VLOG(3) << "Creating shared_ptr of " << shared_lock.get()
              << " count is: " << shared_lock.use_count();
    }

    SharedLockReleaser(SharedLockReleaser&& rhs)
        : shared_lock(std::move(rhs.shared_lock)) {
      VLOG(3) << "Moving SharedLockReleaser of " << shared_lock.get()
              << " count is: " << shared_lock.use_count();
    }

    SharedLockReleaser(const SharedLockReleaser& rhs)
        : shared_lock(rhs.shared_lock) {
      VLOG(3) << "Copying SharedLockReleaser of " << shared_lock.get()
              << " count is: " << shared_lock.use_count();
    }

    ~SharedLockReleaser() {
      VLOG(3) << "Destroying SharedLockReleaser of " << shared_lock.get()
              << " count is: " << shared_lock.use_count();
    }

    void Encode(VariantTensorData*) const {
      // Not supported.
    }

    bool Decode(const VariantTensorData&) {
      return false;  // Not supported.
    }
  };

  void AcquireAsync(
      OpKernelContext* c,
      std::function<void(const Status& s, SharedLockReleaser lock)> fn) {
    CancellationManager* cm = c->cancellation_manager();
    CancellationToken token{};
    bool* cancelled = nullptr;
    if (cm) {
      cancelled = new bool(false);  // GUARDED_BY(mu_);
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
         token](std::function<void(const Status& s, SharedLockReleaser&& lock)>
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
            fn_(Status::OK(),
                SharedLockReleaser{std::make_shared<LockReleaser>(this)});
          } else {
            fn_(errors::Cancelled("Lock acqusition cancelled."),
                SharedLockReleaser{nullptr});
          }
        },
        std::move(fn)));
  }

 private:
  mutex mu_;
  condition_variable cv_ GUARDED_BY(mu_);
  bool locked_ GUARDED_BY(mu_);
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
                                        return Status::OK();
                                      }),
        done);

    Tensor* variant;
    OP_REQUIRES_OK_ASYNC(c, c->allocate_output(0, TensorShape({}), &variant),
                         done);

    mutex->AcquireAsync(
        c, std::bind(
               [c, variant, mutex](DoneCallback done_,
                                   // End of bound arguments.
                                   const Status& s,
                                   Mutex::SharedLockReleaser&& lock) {
                 VLOG(2) << "Finished locking mutex " << mutex
                         << " with lock: " << lock.shared_lock.get()
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
    const int use_count = lock->shared_lock.use_count();
    OP_REQUIRES(
        c, use_count == 1,
        errors::InvalidArgument("Expected use count of lock to be 1, but saw: ",
                                use_count));
  }

  bool IsExpensive() override { return false; }
};

class NotifyNotificationOp : public OpKernel {
 public:
  explicit NotifyNotificationOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("immediately_reset", &immediately_reset_));
  }

  void Compute(OpKernelContext* c) override {
    NotificationResource* cv = nullptr;
    ResourceHandle handle = HandleFromInput(c, 0);
    OP_REQUIRES_OK(c, LookupOrCreateResource<NotificationResource>(
                          c, HandleFromInput(c, 0), &cv,
                          [this, c](NotificationResource** ptr) {
                            *ptr = new NotificationResource(
                                c, HandleFromInput(c, 0).name());
                            return Status::OK();
                          }));
    cv->Notify(immediately_reset_);
  }

 private:
  bool immediately_reset_;
};

class ResetNotificationOp : public OpKernel {
 public:
  explicit ResetNotificationOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    NotificationResource* cv = nullptr;
    ResourceHandle handle = HandleFromInput(c, 0);
    OP_REQUIRES_OK(c, LookupOrCreateResource<NotificationResource>(
                          c, HandleFromInput(c, 0), &cv,
                          [this, c](NotificationResource** ptr) {
                            *ptr = new NotificationResource(
                                c, HandleFromInput(c, 0).name());
                            return Status::OK();
                          }));
    cv->Reset();
  }
};

class WaitForNotificationOp : public AsyncOpKernel {
 public:
  explicit WaitForNotificationOp(OpKernelConstruction* c) : AsyncOpKernel(c) {}

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    NotificationResource* cv = nullptr;
    OP_REQUIRES_OK_ASYNC(c,
                         LookupOrCreateResource<NotificationResource>(
                             c, HandleFromInput(c, 0), &cv,
                             [this, c](NotificationResource** ptr) {
                               *ptr = new NotificationResource(
                                   c, HandleFromInput(c, 0).name());
                               return Status::OK();
                             }),
                         done);
    const Tensor& timeout_in_us_t = c->input(1);
    OP_REQUIRES_ASYNC(
        c, timeout_in_us_t.dims() == 0,
        errors::InvalidArgument(
            "Expected timeout_in_us to be a scalar, but saw shape: ",
            timeout_in_us_t.shape().DebugString()),
        done);
    const int64 timeout_in_us = timeout_in_us_t.scalar<int64>()();

    NotificationResource::Callback callback = std::bind(
        [c, timeout_in_us](DoneCallback done_,
                           // End of bound arguments.
                           const Status& s, bool notified) {
          if (s.ok()) {
            Tensor out(DT_BOOL, {});
            out.scalar<bool>()() = notified;
            c->set_output(0, out);
          } else {
            c->SetStatus(s);
          }
          done_();
        },
        std::move(done), std::placeholders::_1, std::placeholders::_2);

    const uint64 timeout_time_in_us =
        (timeout_in_us == -1) ? std::numeric_limits<uint64>::max()
                              : c->env()->NowMicros() + timeout_in_us;
    cv->RegisterWaiter(c, std::move(callback), timeout_time_in_us);
  }
};

REGISTER_KERNEL_BUILDER(Name("MutexLock").Device(DEVICE_CPU), MutexLockOp);

REGISTER_KERNEL_BUILDER(Name("MutexV2").Device(DEVICE_CPU),
                        ResourceHandleOp<Mutex>);

REGISTER_KERNEL_BUILDER(Name("ConsumeMutexLock").Device(DEVICE_CPU),
                        ConsumeMutexLockOp);

REGISTER_KERNEL_BUILDER(Name("Notification").Device(DEVICE_CPU),
                        ResourceHandleOp<NotificationResource>);

REGISTER_KERNEL_BUILDER(Name("NotifyNotification").Device(DEVICE_CPU),
                        NotifyNotificationOp);

REGISTER_KERNEL_BUILDER(Name("ResetNotification").Device(DEVICE_CPU),
                        ResetNotificationOp);

REGISTER_KERNEL_BUILDER(Name("WaitForNotification").Device(DEVICE_CPU),
                        WaitForNotificationOp);

}  // namespace tensorflow
