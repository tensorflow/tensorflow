/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_CANCELLATION_H_
#define TENSORFLOW_CORE_FRAMEWORK_CANCELLATION_H_

#include <atomic>
#include <functional>

#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// A token that can be used to register and deregister a
// CancelCallback with a CancellationManager.
//
// CancellationToken values must be created by a call to
// CancellationManager::get_cancellation_token.
typedef int64 CancellationToken;

// A callback that is invoked when a step is canceled.
//
// NOTE(mrry): See caveats about CancelCallback implementations in the
// comment for CancellationManager::RegisterCallback.
typedef std::function<void()> CancelCallback;

class CancellationManager {
 public:
  // A value that won't be returned by get_cancellation_token().
  static const CancellationToken kInvalidToken;

  CancellationManager();
  ~CancellationManager();

  // Run all callbacks associated with this manager.
  void StartCancel();

  // Returns true iff StartCancel() has been called.
  bool IsCancelled() { return is_cancelled_.load(std::memory_order_acquire); }

  // Returns a token that must be used in calls to RegisterCallback
  // and DeregisterCallback.
  CancellationToken get_cancellation_token();

  // Attempts to register the given callback to be invoked when this
  // manager is cancelled. Returns true if the callback was
  // registered; returns false if this manager was already cancelled,
  // and the callback was not registered.
  //
  // If this method returns false, it is the caller's responsibility
  // to perform any cancellation cleanup.
  //
  // This method is tricky to use correctly. The following usage pattern
  // is recommended:
  //
  // class ObjectWithCancellableOperation {
  //   mutex mu_;
  //   void CancellableOperation(CancellationManager* cm,
  //                             std::function<void(Status)> callback) {
  //     bool already_cancelled;
  //     CancellationToken token = cm->get_cancellation_token();
  //     {
  //       mutex_lock(mu_);
  //       already_cancelled = !cm->RegisterCallback(
  //           [this, token]() { Cancel(token); });
  //       if (!already_cancelled) {
  //         // Issue asynchronous operation. Associate the pending operation
  //         // with `token` in some object state, or provide another way for
  //         // the Cancel method to look up the operation for cancellation.
  //         // Ensure that `cm->DeregisterCallback(token)` is called without
  //         // holding `mu_`, before `callback` is invoked.
  //         // ...
  //       }
  //     }
  //     if (already_cancelled) {
  //       callback(errors::Cancelled("Operation was cancelled"));
  //     }
  //   }
  //
  //   void Cancel(CancellationToken token) {
  //     mutex_lock(mu_);
  //     // Take action to cancel the operation with the given cancellation
  //     // token.
  //   }
  //
  // NOTE(mrry): The caller should take care that (i) the calling code
  // is robust to `callback` being invoked asynchronously (e.g. from
  // another thread), (ii) `callback` is deregistered by a call to
  // this->DeregisterCallback(token) when the operation completes
  // successfully, and (iii) `callback` does not invoke any method
  // on this cancellation manager. Furthermore, it is important that
  // the eventual caller of the complementary DeregisterCallback does not
  // hold any mutexes that are required by `callback`.
  bool RegisterCallback(CancellationToken token, CancelCallback callback);

  // Deregister the callback that, when registered, was associated
  // with the given cancellation token. Returns true iff the callback
  // was deregistered and will not be invoked; otherwise returns false
  // after the callback has been invoked, blocking if necessary.
  //
  // NOTE(mrry): This method may block if cancellation is in progress.
  // The caller of this method must not hold any mutexes that are required
  // to invoke any cancellation callback that has been registered with this
  // cancellation manager.
  bool DeregisterCallback(CancellationToken token);

  // Deregister the callback that, when registered, was associated
  // with the given cancellation token. Returns true iff the callback
  // was deregistered and will not be invoked; otherwise returns false
  // immediately, with no guarantee that the callback has completed.
  //
  // This method is guaranteed to return true if StartCancel has not been
  // called.
  bool TryDeregisterCallback(CancellationToken token);

 private:
  bool is_cancelling_;
  std::atomic_bool is_cancelled_;

  mutex mu_;
  Notification cancelled_notification_;
  CancellationToken next_cancellation_token_ GUARDED_BY(mu_);
  gtl::FlatMap<CancellationToken, CancelCallback> callbacks_ GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_CANCELLATION_H_
