/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PJRT_GPU_TFRT_THREAD_CHECKER_H_
#define XLA_PJRT_GPU_TFRT_THREAD_CHECKER_H_

#include "absl/log/check.h"

namespace xla {

// RAII handle that guards against CUDA API calls on the current thread.
//
// This is used in tests to validate that TFRT GPU does not call CUDA inline,
// which is important because of the following reasons:
//
// 1. CUDA uses synchronization primitives that are not compatible with
//    cooperatively scheduled threads. For example, Google's fiber scheduler
//    are not aware of blocking code inside CUDA, which results in suboptimal
//    scheduling decisions.
//
// 2. TFRT GPU's goal is to perform GPU operations asynchronously. If CUDA API
//    is being called inline, this is likely an indication that the work is not
//    being performed asynchronously by accident.
//
// The above check is implemented by crashing the process when a stream or
// stream executor accessor on `TfrtGpuDevice` is called on a thread with a live
// `TfrtGpuThreadChecker`. Guarding against stream and stream executor accessors
// is a sufficient proxy to checking for CUDA API calls because these are the
// only ways for TFRT GPU to have access to CUDA APIs.
class TfrtGpuThreadChecker {
 public:
  TfrtGpuThreadChecker() { ++depth_; }
  ~TfrtGpuThreadChecker() { --depth_; }

  TfrtGpuThreadChecker(const TfrtGpuThreadChecker&) = delete;
  TfrtGpuThreadChecker& operator=(const TfrtGpuThreadChecker&) = delete;

  static void AssertCudaCallAllowedOnThisThread() {
    CHECK_EQ(depth_, 0)
        << "TFRT GPU requires CUDA APIs to be never called inline on the PjRt "
           "API caller's thread; call CUDA APIs on the `AsyncWorkRunner` owned "
           "by `TfrtGpuClient` instead";
  }

 private:
  static thread_local int depth_;
};

}  // namespace xla

#endif  // XLA_PJRT_GPU_TFRT_THREAD_CHECKER_H_
