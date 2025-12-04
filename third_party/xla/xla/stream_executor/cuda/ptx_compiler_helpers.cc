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

#include <string>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "re2/re2.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/kernel_stats.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor {
namespace {

static constexpr absl::string_view kPtxasErrorPayloadKey = "ptxas_log";

}  // namespace

absl::Status PtxRegisterAllocationError(absl::string_view message) {
  absl::Status status = absl::ResourceExhaustedError(message);
  status.SetPayload(kPtxasErrorPayloadKey, absl::Cord());
  return status;
}

bool IsPtxRegisterAllocationError(absl::Status status) {
  return status.GetPayload(kPtxasErrorPayloadKey).has_value();
}

bool IsPtxRegisterAllocationError(absl::string_view str) {
  return absl::StrContains(str, "ptxas fatal") &&
         (absl::StrContains(str, "Register allocation failed") ||
          absl::StrContains(str, "Insufficient registers"));
}

absl::StatusOr<int> GetLatestPtxIsaVersionFromUnsupportedVersionErrorLog(
    absl::string_view error_log) {
  std::vector<absl::string_view> chunks = absl::StrSplit(error_log, '\'');
  if (chunks.size() != 3) {
    return absl::InternalError(absl::StrCat(
        "Failed to locate PTX ISA version in ptxas error message: ",
        error_log));
  }
  std::vector<std::string> major_minor = absl::StrSplit(chunks[1], '.');
  if (major_minor.size() != 2) {
    return absl::InternalError(
        absl::StrFormat("Expected PTX ISA version to be formatted as "
                        "MAJOR.MINOR, instead got: %s",
                        chunks[1]));
  }
  int major;
  if (!absl::SimpleAtoi(major_minor[0], &major)) {
    return absl::InternalError(
        absl::StrFormat("Failed to parse PTX ISA major version, expected a "
                        "parsable integer, instead got: %s",
                        major_minor[0]));
  }
  int minor;
  if (!absl::SimpleAtoi(major_minor[1], &minor)) {
    return absl::InternalError(
        absl::StrFormat("Failed to parse PTX ISA minor version, expected a "
                        "parsable integer, instead got: %s",
                        major_minor[1]));
  }
  if (minor >= 10) {
    return absl::InternalError(
        absl::StrFormat("PTX ISA minor version %d is not less than or equal to "
                        "9, which is assumed for version comparison",
                        minor));
  }
  return major * 10 + minor;
}

ModuleStats ExtractModuleStatsFromLog(absl::string_view log) {
  // Reads the log for registers spilled and adds up the count. An example of
  // this line is:
  // "Registers are spilled to local memory in function 'rr', 1080 bytes spill
  // stores, 968 bytes spill loads"
  static constexpr LazyRE2 kSpillRegex{
      R"(function\s+'(\w+)',\s*(\d+)\s+bytes\s+spill\s+stores,\s*(\d+)\s+bytes\s+spill\s+loads)"};
  ModuleStats kernel_stats_map;

  int spill_stores = 0;
  int spill_loads = 0;
  std::string function_name;
  absl::string_view search_log = log;
  while (RE2::FindAndConsume(&search_log, *kSpillRegex, &function_name,
                             &spill_stores, &spill_loads)) {
    kernel_stats_map[function_name] = {spill_stores, spill_loads};
  }
  return kernel_stats_map;
}

absl::Status CreateErrorFromPTXASLog(absl::string_view log,
                                     absl::string_view architecture,
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
    return PtxRegisterAllocationError(log);
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

// Warns if the ptxas version should be upgraded.
// Only prints the warning upon the first invocation.
void WarnIfBadPtxasVersion(absl::string_view method,
                           const CudaComputeCapability& cc,
                           SemanticVersion compiler_version) {
  static absl::once_flag run_once;
  absl::call_once(run_once, [&] {
    // nvbug 4826023: Occurs on Hopper+ in CUDA versions 12.x up to and
    // including CUDA 12.6.2; the earliest ptxas release that corresponds to
    // CUDA 12.6.3 is 12.6.85.
    if (cc.major >= 9 && compiler_version >= SemanticVersion{12, 0, 0} &&
        compiler_version < SemanticVersion{12, 6, 85}) {
      LOG(ERROR)
          << "*** WARNING *** Invoking " << method << " with version "
          << compiler_version
          << ", which corresponds to a CUDA version <=12.6.2. CUDA versions "
             "12.x.y up to and including 12.6.2 miscompile certain edge "
             "cases around clamping.\nPlease upgrade to CUDA 12.6.3 or newer.";
      if (method != "ptxas" && compiler_version.major() == 12 &&
          compiler_version.minor() == 6) {
        LOG(ERROR) << "(Note that this warning may be shown spuriously for "
                      "CUDA 12.6.y, since "
                   << method << " does not report patch versions.)";
      }
    }
  });
}

}  // namespace stream_executor
