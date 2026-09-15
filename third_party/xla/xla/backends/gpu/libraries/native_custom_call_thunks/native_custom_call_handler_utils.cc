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

#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_utils.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_emitter_context.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

absl::StatusOr<ShapeIndex> SingleResultShapeIndex(
    const HloCustomCallInstruction& instr) {
  const Shape& shape = instr.shape();
  if (shape.IsArray()) {
    return ShapeIndex{};
  }
  if (shape.IsTuple() && shape.tuple_shapes().size() == 1 &&
      shape.tuple_shapes(0).IsArray()) {
    return ShapeIndex{0};
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Expected a custom call with exactly one array result, but ",
      instr.custom_call_target(), " has shape ", shape.ToString(true)));
}

absl::StatusOr<ShapedSlice> GetSingleResultShapedSlice(
    const HloCustomCallInstruction& instr,
    const NativeCustomCallEmitterContext& ctx) {
  ABSL_ASSIGN_OR_RETURN(ShapeIndex index, SingleResultShapeIndex(instr));
  return ctx.GetResultShapedSlice(index);
}

absl::StatusOr<stream_executor::CudaComputeCapability> GetCudaComputeCapability(
    const NativeCustomCallEmitterContext& ctx) {
  const stream_executor::DeviceDescription& device_description =
      ctx.GetDeviceDescription();
  stream_executor::CudaComputeCapability compute_capability =
      device_description.cuda_compute_capability();
  if (compute_capability.major < 0) {
    return absl::FailedPreconditionError(
        absl::StrCat("Expected to compile for a CUDA device, but the target "
                     "device is '",
                     device_description.name(), "'"));
  }
  return compute_capability;
}

absl::StatusOr<ThunkSequence> MakeCustomKernelThunkSequence(
    const NativeCustomCallEmitterContext& ctx, CustomKernelLaunchSpec spec,
    const emitters::KernelArguments& kernel_arguments) {
  const size_t num_arguments = kernel_arguments.args().size();

  ABSL_RETURN_IF_ERROR(spec.packing_spec.Validate(num_arguments));

  if (spec.packing_spec.size() != spec.kernel_spec.arity()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Kernel '", spec.name, "' takes ", spec.kernel_spec.arity(),
        " arguments, but the packing spec produces ",
        spec.packing_spec.size()));
  }

  for (int64_t index : spec.zeroed_output_buffer_indices) {
    if (index < 0 || index >= static_cast<int64_t>(num_arguments)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Zeroed output buffer index ", index, " is out of range; there are ",
          num_arguments, " kernel arguments"));
    }
  }

  spec.kernel_spec.set_kernel_args_packing(std::move(spec.packing_spec));
  CustomKernel custom_kernel(std::move(spec.name), std::move(spec.kernel_spec),
                             spec.block_dims, spec.thread_dims,
                             spec.shared_memory_bytes);

  return ThunkSequence::Of<CustomKernelThunk>(
      ctx.GenerateThunkInfo(), std::move(custom_kernel), kernel_arguments,
      spec.use_pdl, std::move(spec.zeroed_output_buffer_indices));
}

}  // namespace xla::gpu
