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

// IWYU pragma: private, include "perftools/gputools/executor/stream_executor.h"

#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_STATUS_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_STATUS_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/lib/error.h"  // IWYU pragma: export
#include "tensorflow/stream_executor/lib/stringpiece.h"
#include "tensorflow/stream_executor/platform/logging.h"

namespace perftools {
namespace gputools {
namespace port {

using Status = tensorflow::Status;

#define SE_CHECK_OK(val) \
  CHECK_EQ(::perftools::gputools::port::Status::OK(), (val))
#define SE_ASSERT_OK(val) \
  ASSERT_EQ(::perftools::gputools::port::Status::OK(), (val))

// Define some canonical error helpers.
inline Status UnimplementedError(StringPiece message) {
  return Status(error::UNIMPLEMENTED, message);
}
inline Status InternalError(StringPiece message) {
  return Status(error::INTERNAL, message);
}
inline Status FailedPreconditionError(StringPiece message) {
  return Status(error::FAILED_PRECONDITION, message);
}

}  // namespace port
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_LIB_STATUS_H_
