/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_RUNTIME_TRACING_H_
#define XLA_SERVICE_GPU_RUNTIME_TRACING_H_

#include <string>

#include "xla/runtime/custom_call_registry.h"
#include "xla/runtime/diagnostics.h"
#include "xla/runtime/type_id.h"

namespace xla {
namespace gpu {

void RegisterTracingTypeIdNames(runtime::TypeIDNameRegistry& registry);

void RegisterTracingCustomCalls(runtime::DirectCustomCallRegistry& registry);

// Appends to `diagnostic_engine` a handler that appends all emitted errors to
// the `diagnostic` string. If `append_annotation_stack` is true, it will append
// current profiler annotation stack to the diagnostic message (annotation used
// in Xprof).
void AppendDiagnosticToString(runtime::DiagnosticEngine& diagnostic_engine,
                              std::string* diagnostic,
                              bool append_annotation_stack = false);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_TRACING_H_
