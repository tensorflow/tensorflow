/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_PREEMPTION_PREEMPTION_NOTIFIER_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_PREEMPTION_PREEMPTION_NOTIFIER_H_

#include <functional>
#include <memory>

#include "absl/time/time.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {

// Listens and propagates any task preemption notice.
class PreemptionNotifier {
 public:
  typedef std::function<void(StatusOr<absl::Time>)> PreemptTimeCallback;

  virtual ~PreemptionNotifier() = default;

  // This is a blocking call that returns a death time when preemption /
  // termination will occur once the listener receives the preemption
  // notification. If no death time is specified, absl::Now() is returned.
  // Returns error::Cancelled if UnregisterListeners() is called.
  StatusOr<absl::Time> WillBePreemptedAt();

  // Registers a callback that takes the death time as input once the listener
  // receives the preemption notification.
  // If no death time is specified, absl::Now() is specified as input.
  // Note: callback should be kept as simple and fast as possible (e.g. simply
  // retrieve result). It should not wait for work done by another callback, and
  // invoke ahy PreemptionNotifier method (e.g. Reset(), destructor).
  void WillBePreemptedAtAsync(PreemptTimeCallback callback);

  // Once a death time has been set, Reset() must be called to listen to a
  // second preemption notice.
  virtual void Reset() = 0;

 protected:
  // Invokes all pending callbacks upon receipt of preemption notice or internal
  // errors (e.g. cancellation during shutdown).
  void NotifyRegisteredListeners(StatusOr<absl::Time> death_time)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  Env* env_;  // Not owned.
  mutex mu_;
  absl::Time death_time_ TF_GUARDED_BY(mu_) = absl::InfinitePast();
  std::vector<PreemptTimeCallback> callbacks_ TF_GUARDED_BY(mu_);
};

std::unique_ptr<PreemptionNotifier> CreateSigtermNotifier(Env* env);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_PREEMPTION_PREEMPTION_NOTIFIER_H_
