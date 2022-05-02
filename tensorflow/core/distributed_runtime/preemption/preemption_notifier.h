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

namespace tensorflow {

// Listens and propagates any task preemption notice.
class PreemptionNotifier {
 public:
  typedef std::function<void(absl::Time)> PreemptTimeCallback;

  virtual ~PreemptionNotifier() = default;

  // Returns a death time when preemption/termination will occur once the
  // listener receives the preemption notification. If no death time is
  // specified, absl::Now() is returned.
  virtual absl::Time WillBePreemptedAt() = 0;

  // Registers a callback that takes the death time as input once the listener
  // receives the preemption notification.
  // If no death time is specified, absl::Now() is specified as input.
  virtual void WillBePreemptedAtAsync(PreemptTimeCallback callback) = 0;

  // Once a death time has been set, Reset() must be called to listen to a
  // second preemption notice.
  virtual void Reset() = 0;
};

std::unique_ptr<PreemptionNotifier> CreatePreemptionNotifier(Env* env);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_PREEMPTION_PREEMPTION_NOTIFIER_H_
