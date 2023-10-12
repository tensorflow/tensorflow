// Copyright 2023 The TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef XLA_RUNTIME_AOT_FFI_EXECUTION_CONTEXT_H_
#define XLA_RUNTIME_AOT_FFI_EXECUTION_CONTEXT_H_

namespace xla {
namespace runtime {
namespace aot {

// To keep dependencies to a minimum we cannot include the actual
// xla::runtime::ExecutionContext. Instead, we re-define it here
// to contain just the information we need for AOT.
//
// LINT.IfChange
struct ExecutionContext {
  void* results_memory_layout = nullptr;  // unused by aot_ffi.
  void* call_frame = nullptr;             // unused by aot_ffi.
  void* custom_call_data = nullptr;
  void* custom_call_registry = nullptr;  // unused by aot_ffi.
  const char* error = nullptr;  // Error message owned by the AOT object.
};
// LINT.ThenChange(//tensorflow/compiler/xla/runtime/executable.cc)

}  // namespace aot
}  // namespace runtime
}  // namespace xla

#endif  // XLA_RUNTIME_AOT_FFI_EXECUTION_CONTEXT_H_
