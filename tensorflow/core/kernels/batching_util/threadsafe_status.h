/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_THREADSAFE_STATUS_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_THREADSAFE_STATUS_H_

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
// Wrapper class to allow both lock-free construction and concurrent updates on
// a 'status'.
//
// Example Usage:
//   std::thread threads[2];
//   ThreadSafeStatus thread_safe_status;
//   threads[0] = std::thread([&]() {
//     status.Update(errors::Internal("internal error"));
//   });
//   threads[1] = std::thread([&]() {
//     status.Update(errors::InvalidArgument("invalid argument"));
//   });
//   threads[0].Join();
//   threads[1].Join();
//
//   NOTE:
//   When updated in a multi-threading setup, only the first error is retained.
class ThreadSafeStatus {
 public:
  const Status& status() const& TF_LOCKS_EXCLUDED(mutex_);
  Status status() && TF_LOCKS_EXCLUDED(mutex_);

  // Retains the first error status: replaces the current status with
  // `new_status` if `new_status` is not OK and the previous status is OK.
  void Update(const Status& new_status) TF_LOCKS_EXCLUDED(mutex_);
  void Update(Status&& new_status) TF_LOCKS_EXCLUDED(mutex_);

 private:
  mutable mutex mutex_;
  Status status_ TF_GUARDED_BY(mutex_);
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_THREADSAFE_STATUS_H_
