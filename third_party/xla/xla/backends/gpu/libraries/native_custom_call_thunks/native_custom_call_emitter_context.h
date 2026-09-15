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
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/ffi/attributes.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
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

  // The device the module is being compiled for. Unlike an FFI handler, which
  // can query the stream's device at instantiation time, a native handler runs
  // ahead of time and must take the target description from here.
  virtual const stream_executor::DeviceDescription& GetDeviceDescription()
      const = 0;

  virtual const DebugOptions& GetDebugOptions() const = 0;

  virtual Thunk::ThunkInfo GenerateThunkInfo() const = 0;

  virtual absl::StatusOr<BufferAllocation::Slice> GetResultAllocationSlice(
      const ShapeIndex& index) const = 0;

  virtual absl::StatusOr<BufferAllocation::Slice> GetOperandAllocationSlice(
      int64_t operand_index, const ShapeIndex& index) const = 0;

  // As above, but also returns the shape backing the slice. This is the form
  // consumed by most thunks, so prefer it over the plain slice accessors.
  virtual absl::StatusOr<ShapedSlice> GetResultShapedSlice(
      const ShapeIndex& index) const = 0;

  virtual absl::StatusOr<ShapedSlice> GetOperandShapedSlice(
      int64_t operand_index, const ShapeIndex& index) const = 0;

  // Builds the conventional kernel argument list for the custom call: all
  // operands in operand order, then the array leaves of the result shape in
  // shape-index order, then `unmanaged_arguments`.
  //
  // This is the same list that XLA's own kernel emitters build, with alignment,
  // aliasing, slice deduplication and the `written` flags filled in. Handlers
  // should use it rather than assembling `emitters::KernelArgument`s by hand,
  // because those flags feed thunk scheduling and getting them wrong produces
  // races that are hard to diagnose.
  //
  // Use `emitters::KernelArguments::OperandIndex` and `ResultIndex` to find
  // where a given buffer ended up; those positions are exactly the index space
  // of `stream_executor::KernelArgsPackingSpec` relocations.
  absl::StatusOr<emitters::KernelArguments> CreateKernelArguments() const {
    return CreateKernelArguments({});
  }
  virtual absl::StatusOr<emitters::KernelArguments> CreateKernelArguments(
      absl::Span<const Shape> unmanaged_arguments) const = 0;

  // The custom call's backend config, decoded into typed FFI attributes.
  //
  // The returned object owns its storage; values that alias it
  // (`absl::string_view`, `absl::Span`, nested `ffi::Dictionary`) are only
  // valid while it is alive.
  virtual absl::StatusOr<xla::ffi::Attributes> GetFfiAttributes() const = 0;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_EMITTER_CONTEXT_H_
