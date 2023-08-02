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

#include "tensorflow/compiler/xla/service/gpu/runtime2/vm.h"

#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_format.h"
#include "third_party/iree/runtime/src/iree/modules/hal/types.h"

namespace xla::gpu::vm {

//===----------------------------------------------------------------------===//
// Trace annotations derived from HLO operations
//===----------------------------------------------------------------------===//

std::string ToScopedAnnotationName(const Trace& trace) {
  return absl::StrFormat("Thunk:#hlo_op=%s#", trace.hlo_op);
}

iree::StatusOr<iree::vm::ref<vm::Trace>> TraceAPI::TraceCreate(
    iree_string_view_t trace) {
  auto ref = iree::vm::make_ref<Trace>();
  ref->hlo_op = std::string(trace.data, trace.size);
  return ref;
}

//===----------------------------------------------------------------------===//
// Helper functions to work with VM lists
//===----------------------------------------------------------------------===//

iree::StatusOr<absl::InlinedVector<iree_hal_buffer_view_t*, 4>>
GetBufferViewVector(iree_vm_list_t* list) {
  iree_host_size_t size = iree_vm_list_size(list);
  absl::InlinedVector<iree_hal_buffer_view_t*, 4> vector(size);

  for (iree_host_size_t i = 0; i < size; ++i) {
    iree_vm_ref_t ref{nullptr};
    IREE_RETURN_IF_ERROR(iree_vm_list_get_ref_assign(list, i, &ref));
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_check_deref(ref, &vector[i]));
  }
  return vector;
}

iree::StatusOr<absl::InlinedVector<int64_t, 4>> GetI64Vector(
    iree_vm_list_t* list) {
  iree_host_size_t size = iree_vm_list_size(list);
  absl::InlinedVector<int64_t, 4> vector(size);
  for (iree_host_size_t i = 0; i < size; ++i) {
    iree_vm_value_t value;
    IREE_RETURN_IF_ERROR(
        iree_vm_list_get_value_as(list, i, IREE_VM_VALUE_TYPE_I64, &value));
    vector[i] = value.i64;
  }
  return vector;
}

}  // namespace xla::gpu::vm

//===----------------------------------------------------------------------===//
// Register types with IREE VM
//===----------------------------------------------------------------------===//

IREE_VM_DEFINE_TYPE_ADAPTERS(execution_context, xla::gpu::vm::ExecutionContext);
IREE_VM_DEFINE_TYPE_ADAPTERS(trace, xla::gpu::vm::Trace);
