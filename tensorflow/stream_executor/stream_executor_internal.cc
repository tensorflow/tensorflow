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

#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace internal {

// -- CUDA

StreamExecutorFactory* MakeCUDAExecutorImplementation() {
  static StreamExecutorFactory instance;
  return &instance;
}

// -- OpenCL

StreamExecutorFactory* MakeOpenCLExecutorImplementation() {
  static StreamExecutorFactory instance;
  return &instance;
}

// -- Host

StreamExecutorFactory MakeHostExecutorImplementation;

// TODO(b/112125301): Consolodate this down to one implementation of
// HostCallback, taking a callback that returns a Status.
bool StreamExecutorInterface::HostCallback(
    Stream* stream, std::function<port::Status()> callback) {
  return HostCallback(stream, [callback]() {
    port::Status s = callback();
    if (!s.ok()) {
      LOG(WARNING) << "HostCallback failed: " << s;
    }
  });
}

}  // namespace internal
}  // namespace stream_executor
