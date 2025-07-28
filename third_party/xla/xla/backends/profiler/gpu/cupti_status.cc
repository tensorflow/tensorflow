/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/backends/profiler/gpu/cupti_status.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace xla {
namespace profiler {

absl::Status ToStatus(CUptiResult result) {
  switch (result) {
    case CUPTI_SUCCESS:
      return absl::OkStatus();
    case CUPTI_ERROR_INVALID_PARAMETER:
      return absl::InvalidArgumentError("CUPTI invalid parameter");
    case CUPTI_ERROR_INVALID_METRIC_NAME:
      return absl::InvalidArgumentError("CUPTI invalid metric name");
    case CUPTI_ERROR_NOT_INITIALIZED:
      return absl::FailedPreconditionError("CUPTI not initialized");
    case CUPTI_ERROR_INSUFFICIENT_PRIVILEGES:
      return absl::PermissionDeniedError("CUPTI needs root access");
    default: {
      const char* errstr = "";
      cuptiGetResultString(result, &errstr);
      errstr = errstr ? errstr : "<unknown>";
      return absl::UnknownError(
          absl::StrCat("CUPTI error ", result, ": ", errstr));
    }
  }
}

}  // namespace profiler
}  // namespace xla
