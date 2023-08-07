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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME2_VM_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME2_VM_H_

#include <string>

#include "absl/container/inlined_vector.h"
#include "third_party/iree/runtime/src/iree/hal/api.h"  // IWYU pragma: keep
#include "third_party/iree/runtime/src/iree/vm/api.h"   // IWYU pragma: keep
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

class DebugOptions;
class ServiceExecutableRunOptions;

namespace gpu::vm {

//===----------------------------------------------------------------------===//
// Execution context of a single XLA invocation
//===----------------------------------------------------------------------===//

// We use XLA:GPU execution context to pass XLA:GPU invocation details to all
// runtime APIs. For example through `run_options` pointer we get access to
// the current compute stream, stream borrower, parent executor, etc.
struct ExecutionContext : public iree::vm::RefObject<ExecutionContext> {
  // XLA:GPU kernels compiled to PTX/CUBIN (for NVIDIA platform).
  struct ExecutableSource {
    const std::string_view ptx;
    const absl::Span<const uint8_t> cubin;
  };

  ExecutionContext(const ServiceExecutableRunOptions* run_options,
                   const DebugOptions* debug_options,
                   ExecutableSource executable_source)
      : run_options(run_options),
        debug_options(debug_options),
        executable_source(executable_source) {}

  const ServiceExecutableRunOptions* run_options;
  const DebugOptions* debug_options;
  ExecutableSource executable_source;
};

//===----------------------------------------------------------------------===//
// Trace annotations derived from HLO operations
//===----------------------------------------------------------------------===//

struct Trace : public iree::vm::RefObject<Trace> {
  std::string hlo_op;
};

std::string ToScopedAnnotationName(const Trace& trace);

struct TraceAPI {
  // Creates `xla_gpu.trace` value.
  iree::StatusOr<iree::vm::ref<vm::Trace>> TraceCreate(
      iree_string_view_t trace);
};

//===----------------------------------------------------------------------===//
// Helper functions to work with VM lists
//===----------------------------------------------------------------------===//

iree::StatusOr<absl::InlinedVector<iree_hal_buffer_view_t*, 4>>
GetBufferViewVector(iree_vm_list_t* list);

iree::StatusOr<absl::InlinedVector<int64_t, 4>> GetI64Vector(
    iree_vm_list_t* list);

//===----------------------------------------------------------------------===//
// Helper functions to convert between xla::Status(Or) and iree::Status(Or)
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): Pass original xla::Status location and call stack through
// iree::Status and recover it in the executable when returning error back to
// XLA from the IREE VM.

// TODO(ezhulenev): Convert from ABSL to IREE error code.

inline iree::Status FromStatus(Status status) {
  if (IREE_LIKELY(status.ok())) return iree_ok_status();

  std::string err = status.ToString();
  return iree_make_status(IREE_STATUS_INTERNAL, "internal error: %s",
                          err.c_str());
}

template <typename T>
iree::StatusOr<T> FromStatusOr(StatusOr<T> statusor) {
  if (IREE_LIKELY(statusor.ok())) return *statusor;

  std::string err = statusor.status().ToString();
  return iree_make_status(IREE_STATUS_INTERNAL, "internal error: %s",
                          err.c_str());
}

}  // namespace gpu::vm
}  // namespace xla

//===----------------------------------------------------------------------===//
// Register types with IREE VM
//===----------------------------------------------------------------------===//

IREE_VM_DECLARE_TYPE_ADAPTERS(execution_context,
                              xla::gpu::vm::ExecutionContext);
IREE_VM_DECLARE_TYPE_ADAPTERS(trace, xla::gpu::vm::Trace);

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME2_VM_H_
