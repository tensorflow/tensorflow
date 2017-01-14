/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_STREAM_EXECUTOR_UTIL_H_
#define TENSORFLOW_CORE_UTIL_STREAM_EXECUTOR_UTIL_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

// StreamExecutorUtil contains functions useful for interfacing
// between StreamExecutor classes and TensorFlow.
class StreamExecutorUtil {
 public:
  // Map a Tensor as a DeviceMemory object wrapping the given typed
  // buffer.
  template <typename T>
  static perftools::gputools::DeviceMemory<T> AsDeviceMemory(const Tensor& t) {
    T* ptr = reinterpret_cast<T*>(const_cast<char*>(t.tensor_data().data()));
    return perftools::gputools::DeviceMemory<T>(
        perftools::gputools::DeviceMemoryBase(ptr, t.TotalBytes()));
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_STREAM_EXECUTOR_UTIL_H_
