/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/runtime/tracing.h"

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/runtime/tracing.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/tsl/profiler/lib/scoped_annotation_stack.h"

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

static absl::StatusOr<int64_t> ActivityStart(runtime::HloTrace annotation) {
  return ScopedAnnotationStack::ActivityStart([&] {
    // We use the same tracing annotation scheme as the ThunkSequence (see
    // implementation of `GetThunkInfo` in `ir_emitter_unnested.cc`).
    return absl::StrFormat("Thunk:#hlo_op=%s#", annotation.hlo_op);
  });
}

static absl::Status ActivityEnd(int64_t activity_id) {
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

}  // namespace gpu
}  // namespace xla
