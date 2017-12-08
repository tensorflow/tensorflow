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

#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"

namespace perftools {
namespace gputools {
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

// TODO(b/70298427) There are two similar methods:
//   bool BlockHostUntilDone(Stream*);
//   Status BlockHostUntilDoneWithStatus(Stream*);
//
// The intention is to replace all implementations of the bool version with the
// Status version.  In the meantime, just implement one in terms of the other.
port::Status StreamExecutorInterface::BlockHostUntilDoneWithStatus(
    Stream* stream) {
  if (!BlockHostUntilDone(stream)) {
    return port::Status(port::error::INTERNAL,
                        "Failed to block host until done.");
  }
  return port::Status::OK();
}

}  // namespace internal
}  // namespace gputools
}  // namespace perftools
