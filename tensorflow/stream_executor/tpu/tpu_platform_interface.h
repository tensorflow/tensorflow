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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_PLATFORM_INTERFACE_H_
#define TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_PLATFORM_INTERFACE_H_

#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/platform.h"

namespace tensorflow {
namespace tpu {

class TpuPlatformInterface : public stream_executor::Platform {
 public:
  using Status = stream_executor::port::Status;

  // Returns a TPU platform to be used by TPU ops. If multiple TPU platforms are
  // registered, finds the most suitable one. Returns nullptr if no TPU platform
  // is registered or an error occurred.
  static TpuPlatformInterface* GetRegisteredPlatform();

  virtual Status Reset() { return Reset(false); }

  virtual Status Reset(bool only_tear_down) = 0;

  virtual int64 TpuMemoryLimit() = 0;
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_PLATFORM_INTERFACE_H_
