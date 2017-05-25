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

#ifndef TENSORFLOW_PLATFORM_DEFAULT_FROM_STREAM_EXECUTOR_STATUS_H_
#define TENSORFLOW_PLATFORM_DEFAULT_FROM_STREAM_EXECUTOR_STATUS_H_

// IWYU pragma: private, include "third_party/tensorflow/core/platform/from_stream_executor_status.h"
// IWYU pragma: friend third_party/tensorflow/core/platform/from_stream_executor_status.h

#include "tensorflow/stream_executor/lib/status.h"

namespace tensorflow {

namespace gpu = ::perftools::gputools;

// On the open-source platform, stream_executor currently uses
// tensorflow::Status
inline Status FromStreamExecutorStatus(
    const perftools::gputools::port::Status& s) {
  return s;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_DEFAULT_FROM_STREAM_EXECUTOR_STATUS_H_
