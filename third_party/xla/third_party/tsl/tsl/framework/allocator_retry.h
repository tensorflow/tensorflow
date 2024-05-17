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

#ifndef TENSORFLOW_TSL_FRAMEWORK_ALLOCATOR_RETRY_H_
#define TENSORFLOW_TSL_FRAMEWORK_ALLOCATOR_RETRY_H_

#include "tsl/platform/env.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/types.h"

namespace tsl {

// A retrying wrapper for a memory allocator.
class AllocatorRetry {
 public:
  AllocatorRetry();

  // Call 'alloc_func' to obtain memory.  On first call,
  // 'verbose_failure' will be false.  If return value is nullptr,
  // then wait up to 'max_millis_to_wait' milliseconds, retrying each
  // time a call to DeallocateRaw() is detected, until either a good
  // pointer is returned or the deadline is exhausted.  If the
  // deadline is exhausted, try one more time with 'verbose_failure'
  // set to true.  The value returned is either the first good pointer
  // obtained from 'alloc_func' or nullptr.
  void* AllocateRaw(std::function<void*(size_t alignment, size_t num_bytes,
                                        bool verbose_failure)>
                        alloc_func,
                    int max_millis_to_wait, size_t alignment, size_t bytes);

  // Called to notify clients that some memory was returned.
  void NotifyDealloc();

 private:
  Env* env_;
  mutex mu_;
  condition_variable memory_returned_;
};

// Implementation details below
inline void AllocatorRetry::NotifyDealloc() {
  mutex_lock l(mu_);
  memory_returned_.notify_all();
}

}  // namespace tsl

#endif  // TENSORFLOW_TSL_FRAMEWORK_ALLOCATOR_RETRY_H_
