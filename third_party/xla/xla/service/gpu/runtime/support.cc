/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/runtime/support.h"

#include <string>
#include <string_view>

#include "tsl/profiler/lib/scoped_annotation_stack.h"

namespace xla {
namespace gpu {

namespace {
static thread_local std::string_view current_tracing_scope = {};
}  // namespace

void SetCurrentTracingScope(std::string_view scope) {
  current_tracing_scope = scope;
}

void ResetCurrentTracingScope() { current_tracing_scope = std::string_view(); }

void AppendDiagnosticToString(runtime::DiagnosticEngine& diagnostic_engine,
                              std::string* diagnostic,
                              bool append_annotation_stack) {
  diagnostic_engine.AddHandler(
      [append_annotation_stack, diagnostic](runtime::Diagnostic& d) {
        if (!diagnostic->empty()) absl::StrAppend(diagnostic, "; ");
        absl::StrAppend(diagnostic, d.status().message());

        // Append the current trace which should help identifying original HLO
        // operation that fails.
        if (!current_tracing_scope.empty()) {
          absl::StrAppend(diagnostic,
                          "; current tracing scope: ", current_tracing_scope);
        }

        // Append current profiling annotation which will have the XLA
        // executable name and program id.
        if (append_annotation_stack) {
          absl::StrAppend(diagnostic, "; current profiling annotation: ",
                          tsl::profiler::AnnotationStack::Get());
        }

        LOG(WARNING) << "Intercepted XLA runtime error:\n"
                     << d.status().ToString(
                            absl::StatusToStringMode::kWithEverything);

        return runtime::success();
      });
}

}  // namespace gpu
}  // namespace xla
