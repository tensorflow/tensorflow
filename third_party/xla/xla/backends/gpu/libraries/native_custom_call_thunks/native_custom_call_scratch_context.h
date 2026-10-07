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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_SCRATCH_CONTEXT_H_
#define XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_SCRATCH_CONTEXT_H_

#include "absl/status/statusor.h"
#include "xla/ffi/attributes.h"
#include "xla/service/gpu_topology.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

// Compile-time context passed to a custom call's scratch handler (see
// `NativeCustomCallScratchHandler` in native_custom_call_handler_registry.h).
//
// The scratch handler runs during HLO optimization, long before buffer
// assignment, so unlike `NativeCustomCallEmitterContext` it can't hand out
// buffer slices. It only exposes what is needed to decide how much scratch
// memory the custom call wants: the target device and the custom call's
// attributes.
//
// All references returned by methods on this context are borrowed and are only
// valid for the duration of a single handler invocation. Handlers must not
// retain them or this context.
class NativeCustomCallScratchContext {
 public:
  virtual ~NativeCustomCallScratchContext() = default;

  virtual const GpuTopology& GetTargetTopology() const = 0;

  // The device the module is being compiled for.
  virtual const stream_executor::DeviceDescription& GetDeviceDescription()
      const = 0;

  virtual const DebugOptions& GetDebugOptions() const = 0;

  // The custom call's backend config, decoded into typed FFI attributes.
  //
  // The returned object owns its storage; values that alias it
  // (`absl::string_view`, `absl::Span`, nested `ffi::Dictionary`) are only
  // valid while it is alive.
  virtual absl::StatusOr<xla::ffi::Attributes> GetFfiAttributes() const = 0;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_SCRATCH_CONTEXT_H_
