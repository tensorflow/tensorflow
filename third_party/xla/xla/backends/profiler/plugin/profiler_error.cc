/* Copyright 2023 The OpenXLA Authors.

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
#include "xla/backends/profiler/plugin/profiler_error.h"

#include <cstddef>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/profiler/plugin/profiler_c_api.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace profiler {
namespace {
absl::Status CheckMatchingStructSizes(absl::string_view struct_name,
                                      size_t expected_size,
                                      size_t actual_size) {
  if (expected_size != actual_size) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unexpected ", struct_name, " size: expected ", expected_size, ", got ",
        actual_size, ". Check installed software versions."));
  }
  return absl::OkStatus();
}
}  // namespace

void PLUGIN_Profiler_Error_Destroy(PLUGIN_Profiler_Error_Destroy_Args* args) {
  absl::Status struct_size_check = CheckMatchingStructSizes(
      "PLUGIN_Profiler_Error_Destroy_Args",
      PLUGIN_Profiler_Error_Destroy_Args_STRUCT_SIZE, args->struct_size);
  if (!struct_size_check.ok()) {
    LOG(ERROR) << struct_size_check.message();
  }
  if (args->struct_size >=
      PROFILER_STRUCT_SIZE(PLUGIN_Profiler_Error_Destroy_Args, error)) {
    delete args->error;
  }
}

void PLUGIN_Profiler_Error_Message(PLUGIN_Profiler_Error_Message_Args* args) {
  absl::Status struct_size_check = CheckMatchingStructSizes(
      "PLUGIN_Profiler_Error_Message_Args",
      PLUGIN_Profiler_Error_Message_Args_STRUCT_SIZE, args->struct_size);
  if (!struct_size_check.ok()) {
    LOG(ERROR) << struct_size_check.message();
  }
  if (args->struct_size >=
      PROFILER_STRUCT_SIZE(PLUGIN_Profiler_Error_Destroy_Args, error)) {
    const absl::Status* status = &args->error->status;
    args->message = status->message().data();
    args->message_size = status->message().size();
  }
}

PLUGIN_Profiler_Error* PLUGIN_Profiler_Error_GetCode(
    PLUGIN_Profiler_Error_GetCode_Args* args) {
  PLUGIN_PROFILER_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PLUGIN_Profiler_Error_GetCode_Args",
      PLUGIN_Profiler_Error_GetCode_Args_STRUCT_SIZE, args->struct_size));
  args->code = static_cast<int>(args->error->status.code());
  return nullptr;
}

}  // namespace profiler
}  // namespace xla
