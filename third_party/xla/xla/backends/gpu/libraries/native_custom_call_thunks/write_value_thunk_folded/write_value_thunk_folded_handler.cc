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

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_emitter_context.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_registration.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_utils.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/write_value_thunk_folded/write_value_folded_kernel.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/ffi/attributes.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/shaped_slice.h"
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
  ABSL_ASSIGN_OR_RETURN(ShapedSlice result, GetSingleResultShapedSlice(instr, ctx));
  int64_t num_elements = ShapeUtil::ElementsIn(result.shape);

  ABSL_ASSIGN_OR_RETURN(xla::ffi::Attributes attrs, ctx.GetFfiAttributes());
  ABSL_ASSIGN_OR_RETURN(int32_t val, attrs.Get<int32_t>("val"));

  ABSL_ASSIGN_OR_RETURN(stream_executor::KernelLoaderSpec kernel_spec,
                   stream_executor::cuda::FindCudaRuntimeKernel(
                       stream_executor::cuda::GetWriteValueFoldedKernel()));

  // This custom call has no operands, so the only kernel argument is the
  // result buffer.
  ABSL_ASSIGN_OR_RETURN(emitters::KernelArguments kernel_args,
                   ctx.CreateKernelArguments());
  ABSL_ASSIGN_OR_RETURN(int64_t result_position,
                   SingleResultArgumentPosition(instr, kernel_args));

  stream_executor::KernelArgsPackingSpec packing_spec;
  packing_spec.AddAddressArgument(result_position);
  packing_spec.AddConstantArgument<int32_t>(val);
  kernel_spec.set_kernel_args_packing(std::move(packing_spec));

  // One grid block per output element, so the kernel's global index covers
  // exactly [0, num_elements).
  return MakeCustomKernelThunkSequence(
      ctx,
      {/*name=*/"write_value_thunk_folded",
       /*kernel_spec=*/std::move(kernel_spec),
       /*block_dims=*/stream_executor::BlockDim(num_elements)},
      kernel_args);
}

XLA_GPU_REGISTER_NATIVE_CUSTOM_CALL_HANDLER(
    "xla.gpu.test_write_value_thunk_folded", WriteValueThunkFoldedHandler);

}  // namespace
}  // namespace xla::gpu
