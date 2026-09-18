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

// Demonstrator for the custom-call thunk-folding API.
//
// This registers a handler for the custom-call target
// "xla.gpu.test_write_value_thunk_folded" that lowers the custom call directly
// to a CustomKernelThunk running a hand-written CUDA kernel, instead of
// wrapping it in a generic CustomCallThunk. It is the thunk-folded counterpart
// of the FFI AOT example in backends/gpu/ffi/xla_ffi_aot_custom_call.cc.

#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_emitter_context.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_registration.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/write_value_thunk_folded/write_value_folded_kernel.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/ffi/attribute_map.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cudart_kernel_registry.h"
#include "xla/stream_executor/kernel_args_packing_spec.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu {
namespace {

// Lowers `xla.gpu.test_write_value_thunk_folded` to a single CustomKernelThunk
// that writes `val` (read from the backend config attribute) to every element
// of the output buffer.
absl::StatusOr<ThunkSequence> WriteValueThunkFoldedHandler(
    const HloCustomCallInstruction& instr,
    const NativeCustomCallEmitterContext& ctx) {
  const Shape& out_shape = instr.shape();
  int64_t num_elements = ShapeUtil::ElementsIn(out_shape);

  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice out_slice,
                   ctx.GetResultAllocationSlice(/*index=*/{}));

  ABSL_ASSIGN_OR_RETURN(xla::ffi::AttributesMap attrs, ctx.GetFfiAttributes());
  auto it = attrs.find("val");
  if (it == attrs.end()) {
    return absl::InvalidArgumentError(
        "Expected 'val' attribute in backend_config");
  }
  if (!std::holds_alternative<xla::ffi::Scalar>(it->second.AsVariant())) {
    return absl::InvalidArgumentError("'val' attribute must be a scalar");
  }
  const xla::ffi::Scalar& scalar =
      std::get<xla::ffi::Scalar>(it->second.AsVariant());
  if (!std::holds_alternative<int32_t>(scalar.AsVariant())) {
    return absl::InvalidArgumentError("'val' attribute must be an int32_t");
  }
  int32_t val = std::get<int32_t>(scalar.AsVariant());

  ABSL_ASSIGN_OR_RETURN(stream_executor::KernelLoaderSpec kernel_spec,
                   stream_executor::cuda::FindCudaRuntimeKernel(
                       stream_executor::cuda::GetWriteValueFoldedKernel()));

  stream_executor::KernelArgsPackingSpec packing_spec;
  packing_spec.AddAddressArgument(0);
  packing_spec.AddConstantArgument<int32_t>(val);
  kernel_spec.set_kernel_args_packing(std::move(packing_spec));

  // One grid block per output element, so the kernel's global index covers
  // exactly [0, num_elements).
  CustomKernel custom_kernel(
      "write_value_thunk_folded", std::move(kernel_spec),
      /*block_dims=*/stream_executor::BlockDim(num_elements),
      /*thread_dims=*/stream_executor::ThreadDim(),
      /*shared_memory_bytes=*/0);

  emitters::KernelArgument out_arg(out_shape, out_slice);
  out_arg.set_written(true);
  emitters::KernelArguments kernel_args(
      std::vector<emitters::KernelArgument>{out_arg});

  return ThunkSequence::Of<CustomKernelThunk>(
      ctx.GenerateThunkInfo(), std::move(custom_kernel), kernel_args);
}

XLA_GPU_REGISTER_NATIVE_CUSTOM_CALL_HANDLER(
    "xla.gpu.test_write_value_thunk_folded", WriteValueThunkFoldedHandler);

}  // namespace
}  // namespace xla::gpu
