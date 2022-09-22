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

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/runtime/tracing.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"

namespace xla {
namespace gpu {

using ::tensorflow::profiler::ScopedAnnotation;
using ::xla::runtime::CustomCall;
using ::xla::runtime::Executable;
using ::xla::runtime::HloTrace;

static absl::Status PushAnnotation(ScopedAnnotationStack* stack,
                                   runtime::HloTrace annotation) {
  stack->stack.push_back(std::make_unique<ScopedAnnotation>([&] {
    // We use the same tracing annotation scheme as the ThunkSequence (see
    // implementation of `GetThunkInfo` in `ir_emitter_unnested.cc`).
    return absl::StrFormat("Thunk:#hlo_op=%s,hlo_module=%s,program_id=%d#",
                           annotation.hlo_op, annotation.module,
                           annotation.program_id);
  }));
  return absl::OkStatus();
}

static absl::Status PopAnnotation(ScopedAnnotationStack* stack) {
  stack->stack.pop_back();
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//

static bool Push(runtime::ExecutionContext* ctx, void** args, void** attrs,
                 void** rets) {
  static auto* handler = CustomCall::Bind("xla.trace.push")
                             .UserData<ScopedAnnotationStack*>()
                             .Attr<HloTrace>("annotation")
                             .To<checks>(PushAnnotation)
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

static bool Pop(runtime::ExecutionContext* ctx, void** args, void** attrs,
                void** rets) {
  static auto* handler = CustomCall::Bind("xla.trace.pop")
                             .UserData<ScopedAnnotationStack*>()
                             .To<checks>(PopAnnotation)
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

//===----------------------------------------------------------------------===//

void RegisterTracingTypeIdNames(runtime::TypeIDNameRegistry& registry) {
  runtime::PopulateTraceTypeIdNames(registry);
}

void RegisterTracingCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.trace.push", Push);
  registry.Register("xla.trace.pop", Pop);
}

}  // namespace gpu
}  // namespace xla
