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
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/AsmParser/AsmParser.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_arguments.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/runtime/custom_call_thunk.h"
#include "xla/service/gpu/runtime/gemm_thunk.h"
#include "xla/service/gpu/runtime/kernel_thunk.h"
#include "xla/service/gpu/thunk.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
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

// TODO(vuson): this is duplicated from ir_emitter_unnested.cc
// Converts MLIR dictionary attribute attached to a custom call operation to a
// custom call thunk attributes that are forwarded to the FFI handler.
static absl::StatusOr<CustomCallThunk::AttributesMap> BuildAttributesMap(
    mlir::DictionaryAttr dict) {
  CustomCallThunk::AttributesMap attributes;
  for (auto& kv : dict) {
    std::string_view name = kv.getName().strref();

    auto integer = [&](mlir::IntegerAttr integer) {
      switch (integer.getType().getIntOrFloatBitWidth()) {
        case 32:
          attributes[name] = static_cast<int32_t>(integer.getInt());
          return absl::OkStatus();
        case 64:
          attributes[name] = static_cast<int64_t>(integer.getInt());
          return absl::OkStatus();
        default:
          return absl::InvalidArgumentError(absl::StrCat(
              "Unsupported integer attribute bit width for attribute: ", name));
      }
    };

    auto fp = [&](mlir::FloatAttr fp) {
      switch (fp.getType().getIntOrFloatBitWidth()) {
        case 32:
          attributes[name] = static_cast<float>(fp.getValue().convertToFloat());
          return absl::OkStatus();
        default:
          return absl::InvalidArgumentError(absl::StrCat(
              "Unsupported float attribute bit width for attribute: ", name));
      }
    };

    auto str = [&](mlir::StringAttr str) {
      attributes[name] = str.getValue().str();
      return absl::OkStatus();
    };

    TF_RETURN_IF_ERROR(
        llvm::TypeSwitch<mlir::Attribute, Status>(kv.getValue())
            .Case<mlir::IntegerAttr>(integer)
            .Case<mlir::FloatAttr>(fp)
            .Case<mlir::StringAttr>(str)
            .Default([&](mlir::Attribute) {
              return absl::InvalidArgumentError(absl::StrCat(
                  "Unsupported attribute type for attribute: ", name));
            }));
  }
  return attributes;
}

absl::StatusOr<BufferAllocation::Slice> GetSliceWithUpdatedOffsetAndSize(
    const BufferAssignment& buffer_assignment, const HloFusionAdaptor& fusion,
    const HloInstruction& fusion_instr, const HloInstruction& start,
    const ShapeIndex& index) {
  if (const auto* param = DynCast<HloParameterInstruction>(&start)) {
    return GetAllocationSlice(buffer_assignment,
                              fusion_instr.operand(param->parameter_number()),
                              index);
  }

  auto slice_adaptor =
      HloFindIf({HloInstructionAdaptor(start)}, fusion,
                [](auto node) { return node.opcode() == HloOpcode::kSlice; });
  TF_RET_CHECK(slice_adaptor.has_value())
      << "AddressComputationFusion expects at least one sliced operand";

  const auto& slice_instr =
      *static_cast<const HloSliceInstruction*>(&slice_adaptor->instruction());

  TF_RET_CHECK(IsContiguousSlice(slice_instr))
      << "AddressComputationFusion only handles contiguous slices currently";

  const Shape& src_shape = slice_instr.operand(0)->shape();
  const Shape& dst_shape = slice_instr.shape();
  int64_t size = ShapeUtil::ByteSizeOf(dst_shape);

  const auto* param = Cast<HloParameterInstruction>(slice_instr.operand(0));
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice orig_slice,
      GetAllocationSlice(buffer_assignment,
                         fusion_instr.operand(param->parameter_number()),
                         index));

  // Given this slice
  // f16[1,4,8]{2,1,0} slice(f16[2,8,8]{2,1,0}),
  //                         slice={[1:2], [4:8], [0:8]}
  //
  // The offset of the slice should be:
  //    slice_starts(0) * 8 * 8 * sizeof(f16) +
  //    slice_starts(1) * 8 * sizeof(f16)
  int64_t offset = orig_slice.offset();
  for (auto [start, stride] : llvm::zip(slice_instr.slice_starts(),
                                        *ShapeUtil::ByteStrides(src_shape))) {
    offset += start * stride;
  }

  return BufferAllocation::Slice(orig_slice.allocation(), offset, size);
}

absl::StatusOr<FusionEmissionResult> EmitGemm(
    IrEmitterContext& ir_emitter_context, const HloFusionAdaptor& adaptor,
    const HloFusionInstruction& fusion,
    const HloCustomCallInstruction& custom_call) {
  const BufferAssignment& buffer_assignment =
      ir_emitter_context.buffer_assignment();

  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice lhs_slice,
      GetSliceWithUpdatedOffsetAndSize(buffer_assignment, adaptor, fusion,
                                       *custom_call.operand(0), /*index=*/{}));

  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice rhs_slice,
      GetSliceWithUpdatedOffsetAndSize(buffer_assignment, adaptor, fusion,
                                       *custom_call.operand(1), /*index=*/{}));

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
      Thunk::ThunkInfo::WithProfileAnnotation(&custom_call), std::move(config),
      lhs_slice, rhs_slice, output, workspace, deterministic_ops);

  FusionEmissionResult result;
  result.thunks.push_back(std::move(thunk));
  return result;
}

absl::StatusOr<FusionEmissionResult> EmitCustomCall(
    IrEmitterContext& ir_emitter_context, const HloFusionAdaptor& adaptor,
    const HloFusionInstruction& fusion,
    const HloCustomCallInstruction& custom_call) {
  const BufferAssignment& buffer_assignment =
      ir_emitter_context.buffer_assignment();

  const std::string call_target_name = custom_call.custom_call_target();

  // Typed FFI custom calls is a replacement for legacy custom calls with
  // a rich type safe API. It's under construction and not fully supported.
  bool is_ffi_custom_call =
      custom_call.api_version() == CustomCallApiVersion::API_VERSION_TYPED_FFI;

  void* call_target = CustomCallTargetRegistry::Global()->Lookup(
      call_target_name, std::string(ir_emitter_context.platform_name()));

  absl::StatusOr<XLA_FFI_Handler*> handler =
      ffi::FindHandler(call_target_name, ir_emitter_context.platform_name());

  // At least one implementation should be available at run time.
  bool found_custom_call = !is_ffi_custom_call && call_target != nullptr;
  bool found_ffi_handler = is_ffi_custom_call && handler.ok();

  TF_RET_CHECK(found_custom_call || found_ffi_handler)
      << "AddressComputationFusion expects custom calls that are emittable as "
         "thunks";

  using Slices = std::vector<std::optional<CustomCallThunk::Slice>>;

  Slices operands;
  // TODO(vuson): add test with custom call with tuple-typed operands
  for (auto* operand : custom_call.operands()) {
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
        operand->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.IsToken()) {
            operands.push_back(std::nullopt);
            return absl::OkStatus();
          }
          if (!subshape.IsArray()) {
            return absl::OkStatus();
          }
          TF_ASSIGN_OR_RETURN(auto slice, GetSliceWithUpdatedOffsetAndSize(
                                              buffer_assignment, adaptor,
                                              fusion, *operand, index));
          operands.push_back(CustomCallThunk::Slice{slice, subshape});
          return absl::OkStatus();
        }));
  }

  Slices results;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      fusion.shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.IsToken()) {
          results.push_back(std::nullopt);
          return absl::OkStatus();
        }
        if (!subshape.IsArray()) {
          return absl::OkStatus();
        }
        TF_ASSIGN_OR_RETURN(
            auto slice, GetAllocationSlice(buffer_assignment, &fusion, index));
        results.push_back(CustomCallThunk::Slice{slice, subshape});
        return absl::OkStatus();
      }));

  // For legacy custom calls we convert all API versions into the latest
  // status-returning one and pass backend config as an opaque string.
  CustomCallThunk::CustomCallTarget custom_call_target;
  std::string opaque;

  // For XLA FFI handlers we decode opaque backend config into attributes map
  // at IR emission time, so that we do not need to parse MLIR at run time. For
  // FFI handlers backend config must be a compatible MLIR dictionary.
  CustomCallThunk::AttributesMap attributes;

  // For information about this calling convention, see
  // xla/g3doc/custom_call.md.
  switch (custom_call.api_version()) {
    case CustomCallApiVersion::API_VERSION_ORIGINAL:
      using original_call_type =
          void (*)(CustomCallThunk::Stream /*stream*/, void** /*buffers*/,
                   const char* /*opaque*/, size_t /*opaque_len*/);
      custom_call_target = [call_target](CustomCallThunk::Stream stream,
                                         void** buffers, const char* opaque,
                                         size_t opaque_len,
                                         XlaCustomCallStatus*) {
        auto typed_call_target =
            reinterpret_cast<original_call_type>(call_target);
        typed_call_target(stream, buffers, opaque, opaque_len);
      };
      break;
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING:
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED:
      using status_returning_call_type =
          void (*)(CustomCallThunk::Stream /*stream*/, void** /*buffers*/,
                   const char* /*opaque*/, size_t /*opaque_len*/,
                   XlaCustomCallStatus* /*status*/);
      custom_call_target =
          reinterpret_cast<status_returning_call_type>(call_target);
      break;
    case CustomCallApiVersion::API_VERSION_TYPED_FFI:
      // We already checked `handler` above.
      break;
    default:
      return Internal("Unknown custom-call API version enum value: %d",
                      custom_call.api_version());
  }

  auto& backend_config_str = custom_call.raw_backend_config_string();
  switch (custom_call.api_version()) {
    case CustomCallApiVersion::API_VERSION_ORIGINAL:
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING:
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED:
      if (!backend_config_str.empty()) {
        opaque = backend_config_str;
      }
      break;

    case CustomCallApiVersion::API_VERSION_TYPED_FFI:
      if (!backend_config_str.empty()) {
        mlir::Attribute attr = mlir::parseAttribute(
            backend_config_str, ir_emitter_context.mlir_context());
        if (auto dict = attr.dyn_cast_or_null<mlir::DictionaryAttr>()) {
          TF_ASSIGN_OR_RETURN(attributes, BuildAttributesMap(dict));
          break;
        }
        return absl::InternalError(
            "Unsupported backend config. Expected a string parsable into "
            "dictionary attribute");
      }
      break;

    default:
      return Internal("Unknown custom-call API version enum value: %d",
                      custom_call.api_version());
  }

  auto ffi_thunk = [&] {
    auto& called_computations = custom_call.called_computations();
    return std::make_unique<CustomCallThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(&custom_call), *handler,
        std::move(operands), std::move(results), std::move(attributes),
        called_computations.empty() ? nullptr : called_computations[0]);
  };

  auto legacy_thunk = [&] {
    return std::make_unique<CustomCallThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(&custom_call),
        std::move(custom_call_target), std::move(operands), std::move(results),
        std::move(opaque));
  };
  FusionEmissionResult result;
  result.thunks.push_back(found_ffi_handler ? ffi_thunk() : legacy_thunk());
  return result;
}

}  // namespace

absl::StatusOr<FusionEmissionResult> CustomFusion::Emit(
    IrEmitterContext& ir_emitter_context, mlir::lmhlo::FusionOp fusion_op,
    const HloFusionInstruction& fusion) const {
  TF_ASSIGN_OR_RETURN(auto gpu_config,
                      fusion.backend_config<GpuBackendConfig>());
  const FusionBackendConfig& backend_config =
      gpu_config.fusion_backend_config();
  const auto& config = backend_config.custom_fusion_config();

  VLOG(3) << "Lower HLO fusion to a custom fusion " << config.name();

  auto* registry = CustomKernelFusionRegistry::Default();
  auto* custom_kernel_fusion = registry->Lookup(config.name());

  // If custom fusion is not found it means that some of the build targets might
  // not be statically linked into the binary.
  if (custom_kernel_fusion == nullptr) {
    return absl::InternalError(
        absl::StrCat("Custom kernel fusion ", config.name(),
                     " not found in a default registry."));
  }

  // Load custom kernels that can implement a fusion computation.
  TF_ASSIGN_OR_RETURN(std::vector<CustomKernel> kernels,
                      custom_kernel_fusion->LoadKernels(
                          ir_emitter_context.gpu_device_info(),
                          fusion.fused_instructions_computation()));

  // This should never happen, it means that compilation pipeline created a
  // fusion operation that is not supported by a given custom fusion.
  if (kernels.empty()) {
    return absl::InternalError(
        absl::StrCat("Custom kernel fusion ", config.name(),
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

absl::StatusOr<FusionEmissionResult> AddressComputationFusion::Emit(
    IrEmitterContext& ir_emitter_context, mlir::lmhlo::FusionOp fusion_op,
    const HloFusionInstruction& fusion) const {
  const HloFusionAdaptor& adaptor = analysis_.fusion();
  auto maybe_custom_call_adaptor = HloFindIf(
      adaptor.GetRoots(), adaptor,
      [](auto node) { return node.opcode() == HloOpcode::kCustomCall; });
  TF_RET_CHECK(maybe_custom_call_adaptor != std::nullopt)
      << "AddressComputationFusion requires a CustomCall hero";

  const auto& custom_call = *static_cast<const HloCustomCallInstruction*>(
      &maybe_custom_call_adaptor->instruction());
  // TODO(vuson): these Emit* are mostly duplicated from ir_emitter_unnested
  if (IsLegacyCublasMatmul(custom_call)) {
    return EmitGemm(ir_emitter_context, adaptor, fusion, custom_call);
  }

  return EmitCustomCall(ir_emitter_context, adaptor, fusion, custom_call);
}

}  // namespace gpu
}  // namespace xla
