/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/ptx_compiler_helpers.h"

#include <string_view>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"

namespace stream_executor {

bool IsPtxRegisterAllocationError(std::string_view str) {
  return absl::StrContains(str, "ptxas fatal") &&
         (absl::StrContains(str, "Register allocation failed") ||
          absl::StrContains(str, "Insufficient registers"));
}

absl::Status CreateErrorFromPTXASLog(std::string_view log,
                                     std::string_view architecture,
                                     bool cancel_if_reg_spill) {
  //  It happens when the loaded version of nvjitlink is too old for
  //  the current GPU. Example error message associated with this error
  //  code:
  //      ptxas fatal   : Value 'sm_80' is not defined for option 'gpu-name'
  if (absl::StrContains(log, "ptxas fatal   : Value '") &&
      absl::StrContains(log, "is not defined for option 'gpu-name'")) {
    return absl::UnimplementedError(absl::StrFormat(
        "Loaded PTX assembler is too old for %s.", architecture));
  }
  if (IsPtxRegisterAllocationError(log)) {
    return absl::ResourceExhaustedError(log);
  }
  if (absl::StrContains(log, "warning")) {
    LOG(INFO) << log;
    if (cancel_if_reg_spill &&
        absl::StrContains(log, "Registers are spilled")) {
      return absl::CancelledError(
          "Compilation result discarded due to register spilling");
    }
  }
  return absl::OkStatus();
}

}  // namespace stream_executor
