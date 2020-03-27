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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_WAIT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_WAIT_H_

#include <condition_variable>  // NOLINT(build/c++11)
#include <functional>
#include <mutex>  //  NOLINT(build/c++11)

#include "tensorflow/lite/experimental/ruy/time.h"

namespace ruy {

// Waits until some evaluation of `condition` has returned true.
//
// There is no guarantee that calling `condition` again after this function
// has returned would still return true. The only
// contract is that at some point during the execution of that function,
// `condition` has returned true.
//
// First does some spin-waiting for the specified `spin_duration`,
// then falls back to passive waiting for the given condvar, guarded
// by the given mutex. At this point it will try to acquire the mutex lock,
// around the waiting on the condition variable.
// Therefore, this function expects that the calling thread hasn't already
// locked the mutex before calling it.
// This function will always release the mutex lock before returning.
//
// The idea of doing some initial spin-waiting is to help get
// better and more consistent multithreading benefits for small GEMM sizes.
// Spin-waiting help ensuring that if we need to wake up soon after having
// started waiting, then we can wake up quickly (as opposed to, say,
// having to wait to be scheduled again by the OS). On the other hand,
// we must still eventually revert to passive waiting for longer waits
// (e.g. worker threads having finished a GEMM and waiting until the next GEMM)
// so as to avoid permanently spinning.
//
// In situations where other threads might have more useful things to do with
// these CPU cores than our spin-waiting, it may be best to reduce the value
// of `spin_duration`. Setting it to zero disables the spin-waiting entirely.
//
// There is a risk that the std::function used here might use a heap allocation
// to store its context. The expected usage pattern is that these functions'
// contexts will consist of a single pointer value (typically capturing only
// [this]), and that in this case the std::function implementation will use
// inline storage, avoiding a heap allocation. However, we can't effectively
// guard that assumption, and that's not a big concern anyway because the
// latency of a small heap allocation is probably low compared to the intrinsic
// latency of what this Wait function does.
void Wait(const std::function<bool()>& condition, const Duration& spin_duration,
          std::condition_variable* condvar, std::mutex* mutex);

// Convenience overload using a default `spin_duration`.
// TODO(benoitjacob): let this be controlled from the ruy API.
void Wait(const std::function<bool()>& condition,
          std::condition_variable* condvar, std::mutex* mutex);

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_WAIT_H_
