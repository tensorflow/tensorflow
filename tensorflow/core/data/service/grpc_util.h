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

#ifndef TENSORFLOW_CORE_DATA_SERVICE_GRPC_UTIL_H_
#define TENSORFLOW_CORE_DATA_SERVICE_GRPC_UTIL_H_

#include "grpcpp/grpcpp.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace data {
namespace grpc_util {

// Wraps a grpc::Status in a tensorflow::Status with the given message.
Status WrapError(const std::string& message, const ::grpc::Status& status);

// Retries the given function if the function produces UNAVAILABLE, ABORTED, or
// CANCELLED status codes. We retry these codes because they can all indicate
// preemption of a server. The retries continue until the deadline is exceeded
// or the `should_retry` callback returns false. `description` may be used to
// log that retries are happening. It should contain a description of the action
// being retried, e.g. "register dataset" The retry loop uses exponential
// backoff between retries. `deadline_micros` is interpreted as microseconds
// since the epoch.
Status Retry(const std::function<Status()>& f,
             const std::function<bool()>& should_retry,
             const std::string& description, int64_t deadline_micros);

// Same as `Retry` above, but with a `should_retry` callback that always returns
// `true`.
Status Retry(const std::function<Status()>& f, const std::string& description,
             int64_t deadline_micros);

}  // namespace grpc_util
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_GRPC_UTIL_H_
