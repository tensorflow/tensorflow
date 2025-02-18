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
#ifndef XLA_BACKENDS_PROFILER_PLUGIN_PROFILER_ERROR_H_
#define XLA_BACKENDS_PROFILER_PLUGIN_PROFILER_ERROR_H_

#include "absl/status/status.h"
#include "xla/backends/profiler/plugin/profiler_c_api.h"

struct PLUGIN_Profiler_Error {
  absl::Status status;
};

#define PLUGIN_PROFILER_RETURN_IF_ERROR(expr)            \
  do {                                                   \
    absl::Status _status = (expr);                       \
    if (!_status.ok()) {                                 \
      PLUGIN_Profiler_Error* _c_status =                 \
          new PLUGIN_Profiler_Error{std::move(_status)}; \
      return _c_status;                                  \
    }                                                    \
  } while (false)

#define PLUGIN_PROFILER_ASSIGN_OR_RETURN(lhs, rexpr)                      \
  _PLUGIN_PROFILER_ASSIGN_OR_RETURN_IMPL(                                 \
      _PLUGIN_PROFILER_CONCAT(_status_or_value, __COUNTER__), lhs, rexpr, \
      _PLUGIN_PROFILER_CONCAT(_c_status, __COUNTER__));

#define _PLUGIN_PROFILER_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr, c_status) \
  auto statusor = (rexpr);                                                     \
  if (!statusor.ok()) {                                                        \
    PLUGIN_Profiler_Error* c_status = new PLUGIN_Profiler_Error();             \
    c_status->status = statusor.status();                                      \
    return c_status;                                                           \
  }                                                                            \
  lhs = std::move(*statusor)

#define _PLUGIN_PROFILER_CONCAT(x, y) _PLUGIN_PROFILER_CONCAT_IMPL(x, y)
#define _PLUGIN_PROFILER_CONCAT_IMPL(x, y) x##y

namespace xla {
namespace profiler {

void PLUGIN_Profiler_Error_Destroy(PLUGIN_Profiler_Error_Destroy_Args* args);

void PLUGIN_Profiler_Error_Message(PLUGIN_Profiler_Error_Message_Args* args);

PLUGIN_Profiler_Error* PLUGIN_Profiler_Error_GetCode(
    PLUGIN_Profiler_Error_GetCode_Args* args);

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_PLUGIN_PROFILER_ERROR_H_
