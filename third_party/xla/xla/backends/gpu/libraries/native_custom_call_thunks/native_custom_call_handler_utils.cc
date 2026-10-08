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
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_emitter_context.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_registry.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/layout.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel_args_packing_spec.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Custom calls with scratch buffers have their result rewritten by the
// `CustomCallScratchAssigner` pass from the original return shape to a tuple
// containing the original result as element 0, followed by all scratch buffers:
//
//   Original return type     -> Rewritten return type
//   --------------------        ---------------------
//   array: f32[4]            -> (f32[4], scratch_0, scratch_1, ...)
//   tuple: (f32[4], s32[8])  -> ((f32[4], s32[8]), scratch_0, scratch_1, ...)
//
// So with a single scratch buffer and an array return type, the result is:
//   (original_return_type, scratch_buffer)
//
// The original return value is always at element 0 (whether it was an array or
// a tuple), and scratch buffers are appended at indices 1 to
// num_scratch_buffers. The frontend attribute
// `xla_gpu_native_custom_call_num_scratch_buffers` records the count.
absl::StatusOr<int64_t> NumScratchBuffers(
    const HloCustomCallInstruction& instr) {
  std::optional<std::string> attr =
      instr.get_frontend_attribute(kNativeCustomCallNumScratchBuffersAttr);
  if (!attr.has_value()) {
    return 0;
  }
  int64_t num_scratch_buffers = 0;
  if (!absl::SimpleAtoi(*attr, &num_scratch_buffers) ||
      num_scratch_buffers < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Custom call ", instr.custom_call_target(), " has a malformed ",
        kNativeCustomCallNumScratchBuffersAttr, " attribute: '", *attr, "'"));
  }
  const Shape& shape = instr.shape();
  if (num_scratch_buffers > 0 &&
      (!shape.IsTuple() ||
       shape.tuple_shapes().size() != 1 + num_scratch_buffers)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Custom call ", instr.custom_call_target(), " claims ",
                     num_scratch_buffers, " scratch buffers, but has shape ",
                     shape.ToString(/*print_layout=*/true)));
  }
  return num_scratch_buffers;
}

absl::StatusOr<ShapeIndex> ScratchShapeIndex(
    const HloCustomCallInstruction& instr, int64_t scratch_index) {
  ABSL_ASSIGN_OR_RETURN(int64_t num_scratch_buffers, NumScratchBuffers(instr));
  if (scratch_index < 0 || scratch_index >= num_scratch_buffers) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Custom call ", instr.custom_call_target(), " has ",
        num_scratch_buffers, " scratch buffers, but scratch buffer ",
        scratch_index, " was requested"));
  }
  return ShapeIndex{1 + scratch_index};
}

absl::StatusOr<ShapedSlice> GetScratchShapedSlice(
    const HloCustomCallInstruction& instr,
    const NativeCustomCallEmitterContext& ctx, int64_t scratch_index) {
  ABSL_ASSIGN_OR_RETURN(ShapeIndex index, ScratchShapeIndex(instr, scratch_index));
  return ctx.GetResultShapedSlice(index);
}

absl::StatusOr<ShapeIndex> SingleResultShapeIndex(
    const HloCustomCallInstruction& instr) {
  const Shape& shape = instr.shape();
  if (shape.IsArray()) {
    return ShapeIndex{};
  }
  ABSL_ASSIGN_OR_RETURN(int64_t num_scratch_buffers, NumScratchBuffers(instr));
  if (shape.IsTuple() &&
      shape.tuple_shapes().size() == 1 + num_scratch_buffers) {
    const Shape& result_shape = shape.tuple_shapes(0);
    if (result_shape.IsArray()) {
      return ShapeIndex{0};
    }
    if (num_scratch_buffers > 0 && result_shape.IsTuple() &&
        result_shape.tuple_shapes().size() == 1 &&
        result_shape.tuple_shapes(0).IsArray()) {
      return ShapeIndex{0, 0};
    }
  }
  if (num_scratch_buffers == 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected a custom call with exactly one array result, "
                     "but ",
                     instr.custom_call_target(), " has shape ",
                     shape.ToString(/*print_layout=*/true)));
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Expected a custom call with exactly one array result and ",
      num_scratch_buffers, " scratch buffer(s) (a tuple of ",
      1 + num_scratch_buffers,
      " elements whose element 0 is an array or 1-element tuple), but ",
      instr.custom_call_target(), " has shape ",
      shape.ToString(/*print_layout=*/true)));
}

absl::StatusOr<ShapedSlice> GetSingleResultShapedSlice(
    const HloCustomCallInstruction& instr,
    const NativeCustomCallEmitterContext& ctx) {
  ABSL_ASSIGN_OR_RETURN(ShapeIndex index, SingleResultShapeIndex(instr));
  return ctx.GetResultShapedSlice(index);
}

absl::StatusOr<int64_t> SingleResultArgumentPosition(
    const HloCustomCallInstruction& instr,
    const emitters::KernelArguments& kernel_arguments) {
  ABSL_ASSIGN_OR_RETURN(ShapeIndex index, SingleResultShapeIndex(instr));
  return kernel_arguments.PositionOfResult(index);
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

absl::StatusOr<Shape> MakeScratchShape(
    PrimitiveType element_type, absl::Span<const int64_t> dimensions,
    NativeCustomCallMemorySpace memory_space) {
  ABSL_ASSIGN_OR_RETURN(Shape shape,
                   ShapeUtil::MakeValidatedShapeWithDescendingLayout(
                       element_type, dimensions));
  shape.mutable_layout()->set_memory_space(static_cast<int64_t>(memory_space));
  return shape;
}

absl::StatusOr<ThunkSequence> MakeCustomKernelThunkSequence(
    const NativeCustomCallEmitterContext& ctx, CustomKernelLaunchSpec spec,
    const emitters::KernelArguments& kernel_arguments) {
  const size_t num_arguments = kernel_arguments.args().size();

  const auto* packing_spec =
      std::get_if<stream_executor::KernelArgsPackingSpec>(
          &spec.kernel_spec.kernel_args_packing());
  if (packing_spec == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Kernel '", spec.name,
        "' packs its arguments with a function; thunk-folded custom calls need "
        "a KernelArgsPackingSpec, which can be serialized for AOT "
        "compilation"));
  }

  ABSL_RETURN_IF_ERROR(packing_spec->Validate(num_arguments));

  if (packing_spec->size() != spec.kernel_spec.arity()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Kernel '", spec.name, "' takes ", spec.kernel_spec.arity(),
        " arguments, but the packing spec produces ", packing_spec->size()));
  }

  for (int64_t index : spec.zeroed_output_buffer_indices) {
    if (index < 0 || index >= static_cast<int64_t>(num_arguments)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Zeroed output buffer index ", index, " is out of range; there are ",
          num_arguments, " kernel arguments"));
    }
  }

  CustomKernel custom_kernel(std::move(spec.name), std::move(spec.kernel_spec),
                             spec.block_dims, spec.thread_dims,
                             spec.shared_memory_bytes);

  return ThunkSequence::Of<CustomKernelThunk>(
      ctx.GenerateThunkInfo(), std::move(custom_kernel), kernel_arguments,
      ctx.GetTargetTopology().num_devices_per_process(), spec.use_pdl,
      std::move(spec.zeroed_output_buffer_indices));
}

}  // namespace xla::gpu
