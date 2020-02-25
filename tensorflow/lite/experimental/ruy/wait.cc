/* Copyright 2019 Google LLC. All Rights Reserved.

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

#include "tensorflow/lite/experimental/ruy/wait.h"

#include <chrono>  // NOLINT(build/c++11)

namespace ruy {

void Wait(const std::function<bool()>& condition, const Duration& spin_duration,
          std::condition_variable* condvar, std::mutex* mutex) {
  // First, trivial case where the `condition` is already true;
  if (condition()) {
    return;
  }

  // Then try busy-waiting.
  const TimePoint wait_start = Now();
  while (Now() - wait_start < spin_duration) {
    if (condition()) {
      return;
    }
  }

  // Finally, do real passive waiting.
  std::unique_lock<std::mutex> lock(*mutex);
  condvar->wait(lock, condition);
}

void Wait(const std::function<bool()>& condition,
          std::condition_variable* condvar, std::mutex* mutex) {
  // This value was empirically derived with some microbenchmark, we don't have
  // high confidence in it.
  //
  // TODO(b/135595069): make this value configurable at runtime.
  // I almost wanted to file another bug to ask for experimenting in a more
  // principled way to tune this value better, but this would have to be tuned
  // on real end-to-end applications and we'd expect different applications to
  // require different tunings. So the more important point is the need for
  // this to be controllable by the application.
  //
  // That this value means that we may be sleeping substantially longer
  // than a scheduler timeslice's duration is not necessarily surprising. The
  // idea is to pick up quickly new work after having finished the previous
  // workload. When it's new work within the same GEMM as the previous work, the
  // time interval that we might be busy-waiting is very small, so for that
  // purpose it would be more than enough to sleep for 1 ms.
  // That is all what we would observe on a GEMM benchmark. However, in a real
  // application, after having finished a GEMM, we might do unrelated work for
  // a little while, then start on a new GEMM. In that case the wait interval
  // may be a little longer. There may also not be another GEMM for a long time,
  // in which case we'll end up passively waiting below.
  const Duration spin_duration = DurationFromMilliseconds(2);
  Wait(condition, spin_duration, condvar, mutex);
}

}  // namespace ruy
