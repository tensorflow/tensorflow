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

#include "tensorflow/core/data/service/grpc_util.h"

#include <algorithm>
#include <functional>
#include <string>

#include "absl/time/time.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/platform/retrying_utils.h"

namespace tensorflow {
namespace data {
namespace grpc_util {

constexpr char kStreamRemovedMessage[] = "Stream removed";

absl::Status WrapError(const std::string& message,
                       const ::grpc::Status& status) {
  if (status.ok()) {
    return errors::Internal("Expected a non-ok grpc status. Wrapping message: ",
                            message);
  } else {
    // FromGrpcStatus checks for "Stream removed" as well, but only when the
    // status code is "Unknown". We have observed that sometimes stream removed
    // errors use other status codes (b/258285154).
    // TODO(aaudibert): Upstream this to FromGrpcStatus.
    if (status.error_message() == kStreamRemovedMessage) {
      return absl::Status(absl::StatusCode::kUnavailable,
                          kStreamRemovedMessage);
    }
    absl::Status s = FromGrpcStatus(status);
    return absl::Status(s.code(),
                        absl::StrCat(message, ": ", status.error_message()));
  }
}

absl::Status Retry(const std::function<absl::Status()>& f,
                   const std::string& description, int64_t deadline_micros) {
  return Retry(
      f, [] { return true; }, description, deadline_micros);
}

absl::Status Retry(const std::function<absl::Status()>& f,
                   const std::function<bool()>& should_retry,
                   const std::string& description, int64_t deadline_micros) {
  absl::Status s = f();
  for (int num_retries = 0;; ++num_retries) {
    if (!IsPreemptedError(s)) {
      return s;
    }
    int64_t now_micros = EnvTime::NowMicros();
    if (now_micros > deadline_micros || !should_retry()) {
      return s;
    }
    int64_t deadline_with_backoff_micros =
        now_micros +
        absl::ToInt64Microseconds(tsl::ComputeRetryBackoff(num_retries));
    // Wait for a short period of time before retrying. If our backoff would put
    // us past the deadline, we truncate it to ensure our attempt starts before
    // the deadline.
    int64_t backoff_until =
        std::min(deadline_with_backoff_micros, deadline_micros);
    int64_t wait_time_micros = backoff_until - now_micros;
    if (wait_time_micros > 100 * 1000) {
      LOG(INFO) << "Failed to " << description << ": " << s
                << ". Will retry in " << wait_time_micros / 1000 << "ms.";
    }
    Env::Default()->SleepForMicroseconds(wait_time_micros);
    s = f();
  }
  return s;
}

}  // namespace grpc_util
}  // namespace data
}  // namespace tensorflow
