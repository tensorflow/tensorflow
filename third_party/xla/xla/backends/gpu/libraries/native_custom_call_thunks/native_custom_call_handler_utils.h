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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_HANDLER_UTILS_H_
#define XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_HANDLER_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_emitter_context.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"

// Helpers for writing custom-call thunk-folding handlers. See
// `native_custom_call_handler_registry.h` for the mechanism itself.

namespace xla::gpu {

// Returns the shape index at which the single array result of `instr` lives.
//
// Frontends disagree about how to spell a custom call that has exactly one
// result: `jax.ffi.ffi_call` always wraps results in a tuple, while
// hand-written HLO usually does not. Accepting both spellings means a handler
// works regardless of who produced the module, so prefer this over inspecting
// `instr.shape()` directly.
//
// Returns `InvalidArgument` if `instr` does not have exactly one array result.
absl::StatusOr<ShapeIndex> SingleResultShapeIndex(
    const HloCustomCallInstruction& instr);

// The shape and buffer slice of the single array result of `instr`, resolved
// through `ctx` so that emitter-level allocation overrides are honored.
absl::StatusOr<ShapedSlice> GetSingleResultShapedSlice(
    const HloCustomCallInstruction& instr,
    const NativeCustomCallEmitterContext& ctx);

// Position, in `kernel_arguments`, of the argument that carries the single
// array result of `instr`. This is what a packing spec's relocation indices
// refer to.
absl::StatusOr<int64_t> SingleResultArgumentPosition(
    const HloCustomCallInstruction& instr,
    const emitters::KernelArguments& kernel_arguments);

// The CUDA compute capability of the device the module is being compiled for.
//
// Returns `FailedPrecondition` if that device is not a CUDA device, so that a
// CUDA-only handler reports a clear error instead of silently proceeding with
// the invalid capability that `DeviceDescription` returns in that case.
absl::StatusOr<stream_executor::CudaComputeCapability> GetCudaComputeCapability(
    const NativeCustomCallEmitterContext& ctx);

// Everything that distinguishes one custom kernel launch from another.
struct CustomKernelLaunchSpec {
  // Name of the kernel, used in debug output and error messages.
  std::string name;

  // How to load the kernel and how to pack its arguments. The packing must be
  // a `stream_executor::KernelArgsPackingSpec` rather than a packing function,
  // so that it survives AOT serialization. Its relocation indices refer to
  // positions in the `kernel_arguments` passed alongside this spec; obtain them
  // from `emitters::KernelArguments::OperandIndex` and `ResultIndex` instead of
  // hardcoding them.
  stream_executor::KernelLoaderSpec kernel_spec;

  // Launch geometry. Both default to a single block of a single thread.
  stream_executor::BlockDim block_dims = stream_executor::BlockDim();
  stream_executor::ThreadDim thread_dims = stream_executor::ThreadDim();
  size_t shared_memory_bytes = 0;

  // Whether to launch the kernel with Programmatic Dependent Launch, which
  // lets it start its prologue while the preceding kernel is still draining.
  // Only set this if the kernel actually issues `cudaGridDependencySynchronize`
  // before touching its inputs; otherwise it may read buffers that the
  // preceding kernel has not finished writing.
  bool use_pdl = false;

  // Positions in `kernel_arguments` of buffers that the runtime must zero
  // before the launch.
  std::vector<int64_t> zeroed_output_buffer_indices;
};

// Builds a single-thunk `ThunkSequence` that launches `spec` over
// `kernel_arguments`.
//
// Checks that `spec` and `kernel_arguments` agree before building the thunk:
// that the packing spec produces as many arguments as the kernel accepts, and
// that every relocation and every zeroed-buffer index refers to an existing
// kernel argument. Without these checks a mismatch only shows up as an
// out-of-bounds read or a corrupted argument buffer at launch time.
absl::StatusOr<ThunkSequence> MakeCustomKernelThunkSequence(
    const NativeCustomCallEmitterContext& ctx, CustomKernelLaunchSpec spec,
    const emitters::KernelArguments& kernel_arguments);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_HANDLER_UTILS_H_
