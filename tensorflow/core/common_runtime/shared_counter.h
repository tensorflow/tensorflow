/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_SHARED_COUNTER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_SHARED_COUNTER_H_

#include <atomic>

#include "tensorflow/core/platform/types.h"

namespace tensorflow {
// A lightweight thread-safe monotone counter for establishing
// temporal ordering.
class SharedCounter {
 public:
  int64_t get() { return value_; }
  int64_t next() { return ++value_; }

 private:
  std::atomic<int64_t> value_{0};
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_SHARED_COUNTER_H_
