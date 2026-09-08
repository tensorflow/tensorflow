/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_EMITTER_CONTEXT_H_
#define XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_EMITTER_CONTEXT_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/ffi/attribute_map.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu_topology.h"
#include "xla/shape_util.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

// Compile-time context passed to a custom-call thunk-folding handler
// (see native_custom_call_handler_registry.h).
//
// Instead of lowering a custom call to a CustomCallThunk (FFI/legacy), a
// registered handler receives this context and returns a native ThunkSequence
// that the ThunkEmitter folds directly into the thunk graph.
//
// All references returned by methods on this context are borrowed
// and are only valid for the duration of a single handler
// invocation. Handlers must not retain them or this context.
class NativeCustomCallEmitterContext {
 public:
  virtual ~NativeCustomCallEmitterContext() = default;

  virtual const GpuTopology& GetTargetTopology() const = 0;

  virtual const DebugOptions& GetDebugOptions() const = 0;

  virtual Thunk::ThunkInfo GenerateThunkInfo() const = 0;

  virtual absl::StatusOr<BufferAllocation::Slice> GetResultAllocationSlice(
      const ShapeIndex& index) const = 0;

  virtual absl::StatusOr<BufferAllocation::Slice> GetOperandAllocationSlice(
      int64_t operand_index, const ShapeIndex& index) const = 0;

  virtual absl::StatusOr<xla::ffi::AttributesMap> GetFfiAttributes() const = 0;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_EMITTER_CONTEXT_H_
