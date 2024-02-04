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

#include "xla/service/gpu/runtime/tracing.h"

#include <string>
#include <string_view>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/runtime/custom_call_registry.h"
#include "xla/runtime/diagnostics.h"
#include "xla/runtime/executable.h"
#include "xla/runtime/logical_result.h"
#include "xla/runtime/tracing.h"
#include "xla/service/gpu/runtime/annotation.h"
#include "xla/service/gpu/runtime/support.h"
#include "tsl/profiler/lib/scoped_annotation_stack.h"

namespace xla {
namespace gpu {

using ::xla::runtime::CustomCall;
using ::xla::runtime::HloTrace;

using ::tsl::profiler::ScopedAnnotationStack;

//===----------------------------------------------------------------------===//
// Type names for encoded attributes.
//===----------------------------------------------------------------------===//

void RegisterTracingTypeIdNames(runtime::TypeIDNameRegistry& registry) {
  runtime::PopulateTraceTypeIdNames(registry);
}

//===----------------------------------------------------------------------===//
// Tracing custom calls implementation.
//===----------------------------------------------------------------------===//

namespace {
thread_local const ModuleAnnotations* current_annotations{};
thread_local std::string_view current_tracing_scope = {};
}

static absl::StatusOr<int64_t> ActivityStart(runtime::HloTrace annotation) {
  current_tracing_scope = annotation.hlo_op;
  if (current_annotations) {
    // We know which HloModule we belong to, and may have pre-prepared
    // annotation structs ready to use
    const auto it = current_annotations->kernels.find(annotation.hlo_op);
    if (it != current_annotations->kernels.end()) {
      // Have a pre-prepared annotation, use it
      return ScopedAnnotationStack::ActivityStart([&] { return it->second; });
    }
  }
  return ScopedAnnotationStack::ActivityStart([&] {
    // We use the same tracing annotation scheme as the ThunkSequence.
    return absl::StrFormat("Thunk:#hlo_op=%s#", annotation.hlo_op);
  });
}

static absl::Status ActivityEnd(int64_t activity_id) {
  current_tracing_scope = {};
  ScopedAnnotationStack::ActivityEnd(activity_id);
  return absl::OkStatus();
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(Start, FunctionWrapper<ActivityStart>(), checks,
                               CustomCall::Bind("xla.trace.activity_start")
                                   .Attr<HloTrace>("annotation")
                                   .Ret<int64_t>());

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    End, FunctionWrapper<ActivityEnd>(), checks,
    CustomCall::Bind("xla.trace.activity_end").Arg<int64_t>());

void RegisterTracingCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.trace.activity_start", Start);
  registry.Register("xla.trace.activity_end", End);
}

const ModuleAnnotations* SetCurrentModuleAnnotations(
    const ModuleAnnotations* annotations) {
  return std::exchange(current_annotations, annotations);
}

static void AppendTracingScopeAndModuleAnnotations(
    std::string* diagnostic, bool append_annotation_stack) {
  // Append the current trace which should help identifying original HLO
  // operation that fails.
  if (!current_tracing_scope.empty()) {
    absl::StrAppend(diagnostic,
                    "; current tracing scope: ", current_tracing_scope);
  }

  if (!append_annotation_stack || current_annotations == nullptr) {
    return;
  }

  // Append current profiling annotation which will have the XLA
  // executable name and program id.
  absl::StrAppend(diagnostic, "; current profiling annotation: ",
                  current_annotations->top_level.Title());

  if (current_tracing_scope.empty()) {
    return;
  }
  const auto it = current_annotations->kernels.find(current_tracing_scope);
  if (it == current_annotations->kernels.end()) {
    return;
  }

  absl::StrAppend(diagnostic, "::", it->second.Title());
}

void AppendDiagnosticToString(runtime::DiagnosticEngine& diagnostic_engine,
                              std::string* diagnostic,
                              bool append_annotation_stack) {
  diagnostic_engine.AddHandler([append_annotation_stack,
                                diagnostic](runtime::Diagnostic& d) {
    if (!diagnostic->empty()) absl::StrAppend(diagnostic, "; ");
    absl::StrAppend(diagnostic, d.status().message());
    AppendTracingScopeAndModuleAnnotations(diagnostic, append_annotation_stack);

    LOG(WARNING) << "Intercepted XLA runtime error:\n"
                 << d.status().ToString(
                        absl::StatusToStringMode::kWithEverything);

    return runtime::success();
  });
}

}  // namespace gpu
}  // namespace xla
