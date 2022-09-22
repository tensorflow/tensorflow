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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_TRACING_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_TRACING_H_

#include <memory>

#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "tensorflow/compiler/xla/runtime/type_id.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"

namespace xla {
namespace gpu {

// Tracing pushes scoped annotations to the stack that is automatically
// destructed when executable returns control flow to the caller. We do not have
// the luxury of RAII in compiled executables, so we rely on this stack
// indirection to guarantee that all annotations will be popped from the
// profiler stack when the entry point function returns control to the caller.
//
// TODO(ezhulenev): This is a temporary solution to work around the fact that
// `ScopedAnnotation` is not copyable or movable. We need a lightweight scoped
// annotation stack implementation inside the profiling library.
struct ScopedAnnotationStack {
  using ScopedAnnotation = tensorflow::profiler::ScopedAnnotation;
  absl::InlinedVector<std::unique_ptr<ScopedAnnotation>, 4> stack;
};

void RegisterTracingTypeIdNames(runtime::TypeIDNameRegistry& registry);

void RegisterTracingCustomCalls(runtime::DirectCustomCallRegistry& registry);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_TRACING_H_
