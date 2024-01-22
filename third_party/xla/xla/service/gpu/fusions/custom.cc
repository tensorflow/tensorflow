/* Copyright 2024 The OpenXLA Authors.

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
#include "xla/service/gpu/fusions/custom.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/gemm_thunk.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_arguments.h"
#include "xla/service/gpu/kernels/custom_fusion.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/runtime3/kernel_thunk.h"
#include "xla/service/gpu/thunk.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

absl::StatusOr<std::unique_ptr<Thunk>> BuildCustomKernelThunkForFusion(
    IrEmitterContext& ir_emitter_context, const HloFusionInstruction& fusion,
    mlir::lmhlo::FusionOp fusion_op, CustomKernel custom_kernel) {
  TF_ASSIGN_OR_RETURN(auto kernel_arguments,
                      ir_emitter_context.emit_ir_from_hlo()
                          ? KernelArguments::Create(
                                ir_emitter_context.buffer_assignment(), &fusion)
                          : KernelArguments::Create(
                                ir_emitter_context.allocations(), fusion_op));

  std::variant<mlir::Operation*, const HloInstruction*> instr;
  if (ir_emitter_context.emit_ir_from_hlo()) {
    instr = &fusion;
  } else {
    instr = fusion_op;
  }

  return std::make_unique<CustomKernelThunk>(
      instr, std::move(custom_kernel), std::move(kernel_arguments.args()));
}

bool IsSliceInLeadingDimOnly(const HloInstruction& instr) {
  auto slice = DynCast<HloSliceInstruction>(&instr);
  if (!slice) return false;
  const Shape& shape = slice->operand(0)->shape();
  int64_t major_dim = shape.layout().minor_to_major().back();
  for (size_t i = 0; i < shape.rank(); ++i) {
    if (i == major_dim) continue;
    if (slice->slice_starts(i) != 0 ||
        slice->slice_limits(i) != shape.dimensions(i))
      return false;
  }
  return true;
}

absl::StatusOr<BufferAllocation::Slice> GetSliceWithUpdatedOffsetAndSize(
    const BufferAssignment& buffer_assignment, const HloFusionAdaptor& fusion,
    const HloInstruction* bufferized_instr, const HloInstruction& start) {
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice orig_slice,
      GetAllocationSlice(buffer_assignment, bufferized_instr, {}));

  auto maybe_slice_adaptor =
      HloFindIf({HloInstructionAdaptor(start)}, fusion,
                [](auto node) { return node.opcode() == HloOpcode::kSlice; });
  if (maybe_slice_adaptor == std::nullopt) return orig_slice;

  const auto& slice_instr = *static_cast<const HloSliceInstruction*>(
      &maybe_slice_adaptor->instruction());

  TF_RET_CHECK(IsSliceWithUnitStrides(&slice_instr))
      << "AddressComputationFusion only handles slices with unit strides "
         "currently";
  TF_RET_CHECK(IsSliceInLeadingDimOnly(slice_instr))
      << "AddressComputationFusion only handles slices in leading dim "
         "currently";

  // Given this shape f16[10,10,10]{2,1,0}, sliced into f16[2,10,10]{2,1,0}
  // We say that the sliced shape contains 2 slice units of f16[10,10]
  const Shape& shape = slice_instr.shape();
  int64_t major_dim = shape.layout().minor_to_major().back();
  // The sliced leading dim is the number of slice units.
  int64_t slice_unit_count = shape.dimensions(major_dim);
  int64_t num_elem = ShapeUtil::ElementsIn(shape);
  // The number of elements in a slice unit is the total number of elements
  // divided by the number of slice units.
  int64_t slice_unit_num_elem = num_elem / slice_unit_count;
  int64_t slice_unit_byte_size =
      slice_unit_num_elem *
      ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  int64_t offset = slice_instr.slice_starts(major_dim) * slice_unit_byte_size;
  int64_t size = slice_unit_count * slice_unit_byte_size;
  return BufferAllocation::Slice(orig_slice.allocation(), offset, size);
}

}  // namespace

absl::StatusOr<FusionEmissionResult> CustomFusionEmitter::Emit(
    IrEmitterContext& ir_emitter_context, mlir::lmhlo::FusionOp fusion_op,
    const HloFusionInstruction& fusion) const {
  TF_ASSIGN_OR_RETURN(auto gpu_config,
                      fusion.backend_config<GpuBackendConfig>());
  const FusionBackendConfig& backend_config =
      gpu_config.fusion_backend_config();
  const auto& config = backend_config.custom_fusion_config();

  VLOG(3) << "Lower HLO fusion to a custom fusion " << config.name();

  auto* registry = CustomFusionRegistry::Default();
  auto* custom_fusion = registry->Lookup(config.name());

  // If custom fusion is not found it means that some of the build targets might
  // not be statically linked into the binary.
  if (custom_fusion == nullptr) {
    return absl::InternalError(absl::StrCat(
        "Custom fusion ", config.name(), " not found in a default registry."));
  }

  // Load custom kernels that can implement a fusion computation.
  TF_ASSIGN_OR_RETURN(
      std::vector<CustomKernel> kernels,
      custom_fusion->LoadKernels(ir_emitter_context.gpu_device_info(),
                                 fusion.fused_instructions_computation()));

  // This should never happen, it means that compilation pipeline created a
  // fusion operation that is not supported by a given custom fusion.
  if (kernels.empty()) {
    return absl::InternalError(
        absl::StrCat("Custom fusion ", config.name(),
                     " returned empty custom kernels for a fused computation"));
  }

  // TODO(ezhulenev): Add support for auto tuning to select the best kernel.
  if (kernels.size() != 1) {
    return absl::InternalError("Expected exactly one custom kernel");
  }

  TF_ASSIGN_OR_RETURN(auto thunk, BuildCustomKernelThunkForFusion(
                                      ir_emitter_context, fusion, fusion_op,
                                      std::move(kernels[0])));

  FusionEmissionResult result;
  result.thunks.push_back(std::move(thunk));
  return result;
}

absl::StatusOr<FusionEmissionResult> AddressComputationFusionEmitter::Emit(
    IrEmitterContext& ir_emitter_context, mlir::lmhlo::FusionOp fusion_op,
    const HloFusionInstruction& fusion) const {
  const BufferAssignment& buffer_assignment =
      ir_emitter_context.buffer_assignment();

  const HloFusionAdaptor& adaptor = analysis_.fusion();
  auto maybe_custom_call_adaptor = HloFindIf(
      adaptor.GetRoots(), adaptor,
      [](auto node) { return node.opcode() == HloOpcode::kCustomCall; });
  TF_RET_CHECK(maybe_custom_call_adaptor != std::nullopt)
      << "AddressComputationFusion requires a CustomCall hero";

  const auto& custom_call = *static_cast<const HloCustomCallInstruction*>(
      &maybe_custom_call_adaptor->instruction());
  if (IsLegacyCublasMatmul(custom_call)) {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice lhs_slice,
                        GetSliceWithUpdatedOffsetAndSize(
                            buffer_assignment, adaptor, fusion.operand(0),
                            *custom_call.operand(0)));

    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice rhs_slice,
                        GetSliceWithUpdatedOffsetAndSize(
                            buffer_assignment, adaptor, fusion.operand(1),
                            *custom_call.operand(1)));

    BufferAllocation::Slice output;
    std::optional<BufferAllocation::Slice> workspace;

    // Result of a legacy cuBLAS custom call can be a tuple if we explicitly
    // allocate workspace buffer in HLO. If result is an array, it means that
    // workspace is not available, and cuBLAS will allocate its own workspace.
    if (custom_call.shape().IsArray()) {
      TF_ASSIGN_OR_RETURN(output,
                          GetAllocationSlice(buffer_assignment, &fusion, {}));
    } else {
      TF_ASSIGN_OR_RETURN(output,
                          GetAllocationSlice(buffer_assignment, &fusion, {0}));
      TF_ASSIGN_OR_RETURN(workspace,
                          GetAllocationSlice(buffer_assignment, &fusion, {1}));
    }

    bool deterministic_ops =
        ir_emitter_context.debug_options().xla_gpu_deterministic_ops();

    TF_ASSIGN_OR_RETURN(
        GemmConfig config,
        GemmConfig::For(static_cast<const HloInstruction*>(&custom_call)));
    auto thunk = std::make_unique<GemmThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(&custom_call),
        std::move(config), lhs_slice, rhs_slice, output, workspace,
        deterministic_ops);

    FusionEmissionResult result;
    result.thunks.push_back(std::move(thunk));
    return result;
  }

  return absl::UnimplementedError(
      absl::StrCat("No emission for AddressComputationFusion of custom call ",
                   custom_call.custom_call_target()));
}

}  // namespace gpu
}  // namespace xla
