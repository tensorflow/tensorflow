/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_HOST_OR_DEVICE_SCALAR_H_
#define TENSORFLOW_STREAM_EXECUTOR_HOST_OR_DEVICE_SCALAR_H_

#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/platform/logging.h"

namespace stream_executor {

// Allows to represent a value that is either a host scalar or a scalar stored
// on the GPU device.
template <typename ElemT>
class HostOrDeviceScalar {
 public:
  // Not marked as explicit because when using this constructor, we usually want
  // to set this to a compile-time constant.
  HostOrDeviceScalar(ElemT value) : value_(value), is_pointer_(false) {}
  explicit HostOrDeviceScalar(const DeviceMemory<ElemT>& pointer)
      : pointer_(pointer), is_pointer_(true) {
    CHECK_EQ(1, pointer.ElementCount());
  }

  bool is_pointer() const { return is_pointer_; }
  const DeviceMemory<ElemT>& pointer() const {
    CHECK(is_pointer());
    return pointer_;
  }
  const ElemT& value() const {
    CHECK(!is_pointer());
    return value_;
  }

 private:
  union {
    ElemT value_;
    DeviceMemory<ElemT> pointer_;
  };
  bool is_pointer_;
};

}  // namespace stream_executor
#endif  // TENSORFLOW_STREAM_EXECUTOR_HOST_OR_DEVICE_SCALAR_H_
