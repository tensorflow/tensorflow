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

// The default implementation just calls the other HostCallback method.
// It should make all existing code that uses a void() callback still work.
bool StreamExecutorInterface::HostCallback(Stream* stream,
                                           std::function<void()> callback) {
  return HostCallback(
      stream, std::function<port::Status()>([callback]() -> port::Status {
        callback();
        return port::Status::OK();
      }));
}

}  // namespace internal
}  // namespace stream_executor
