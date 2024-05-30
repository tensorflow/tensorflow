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
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/AsmParser/AsmParser.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
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
#include "xla/service/gpu/runtime/address_computation_thunk.h"
#include "xla/service/gpu/runtime/custom_call_thunk.h"
#include "xla/service/gpu/runtime/gemm_thunk.h"
#include "xla/service/gpu/runtime/kernel_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

constexpr unsigned kGEMMOutputBufferIndex = 0;
constexpr unsigned kGEMMWorkspaceBufferIndex = 1;

namespace m = ::xla::match;

absl::StatusOr<std::unique_ptr<Thunk>> BuildCustomKernelThunkForFusion(
    IrEmitterContext& ir_emitter_context, const HloFusionInstruction& fusion,
    CustomKernel custom_kernel) {
  TF_ASSIGN_OR_RETURN(
      auto kernel_arguments,
      KernelArguments::Create(ir_emitter_context.buffer_assignment(), &fusion));

  return std::make_unique<CustomKernelThunk>(
      &fusion, std::move(custom_kernel), std::move(kernel_arguments.args()));
}

absl::StatusOr<BufferAllocation::Slice> GetOperandSlice(
    const BufferAssignment& buffer_assignment, const HloFusionAdaptor& adaptor,
    const HloInstruction& fusion_instr, const HloInstruction& start_instr,
    std::vector<HloInstruction*>& slice_instrs, const ShapeIndex& shape_idx,
    unsigned arg_idx) {
  if (const auto* param = DynCast<HloParameterInstruction>(&start_instr)) {
    return GetAllocationSlice(buffer_assignment,
                              fusion_instr.operand(param->parameter_number()),
                              shape_idx);
  }

  // Walk through ShapeIndex to find the real starting point.
  auto* start = const_cast<HloInstruction*>(&start_instr);
  for (auto idx : shape_idx) {
    CHECK(start->shape().IsTuple());
    start = const_cast<HloInstruction*>(start->operand(idx));
  }

  if (const auto* param = DynCast<HloParameterInstruction>(start)) {
    // At this point we've walked through all `shape_idx`, `index` should be
    // empty.
    return GetAllocationSlice(buffer_assignment,
                              fusion_instr.operand(param->parameter_number()),
                              /*index*/ {});
  }

  auto slice_adaptor = HloFindIf(
      {HloInstructionAdaptor(*start, &adaptor)}, adaptor,
      [](HloInstructionAdaptor node) {
        return IsOpcodeAnyOf<HloOpcode::kDynamicSlice, HloOpcode::kSlice>(
            &node.instruction());
      });
  if (slice_adaptor.has_value()) {
    auto* slice_instr =
        const_cast<HloInstruction*>(&slice_adaptor->instruction());

    if (!IsContiguousSlice(slice_instr->operand(0)->shape(),
                           slice_instr->shape())) {
      return absl::InternalError(
          "AddressComputationFusion only handles contiguous slices "
          "currently");
    }

    slice_instrs[arg_idx] = slice_instr;

    const auto* param = Cast<HloParameterInstruction>(slice_instr->operand(0));
    // At this point we've walked through all `shape_idx`, `index` should be
    // empty.
    TF_ASSIGN_OR_RETURN(
        BufferAllocation::Slice orig_slice,
        GetAllocationSlice(buffer_assignment,
                           fusion_instr.operand(param->parameter_number()),
                           /*index*/ {}));

    if (auto* static_slice = DynCast<HloSliceInstruction>(slice_instr)) {
      // Update static slices.
      const Shape& src_shape = static_slice->operand(0)->shape();
      const Shape& dst_shape = static_slice->shape();
      int64_t size = ShapeUtil::ByteSizeOf(dst_shape);

      // Given this slice
      // f16[1,4,8]{2,1,0} slice(f16[2,8,8]{2,1,0}),
      //                         slice={[1:2], [4:8], [0:8]}
      //
      // The offset of the slice should be:
      //    slice_starts(0) * 8 * 8 * sizeof(f16) +
      //    slice_starts(1) * 8 * sizeof(f16)
      int64_t offset = orig_slice.offset();
      for (auto [start, stride] :
           llvm::zip(static_slice->slice_starts(),
                     *ShapeUtil::ByteStrides(src_shape))) {
        offset += start * stride;
      }

      return BufferAllocation::Slice(orig_slice.allocation(), offset, size);
    }

    return orig_slice;
  }

  return absl::InternalError("WTF");
}

// Returns true if `offset` is a loop iteration number. This pattern matching
// detects HLOs that generated by `jax.lax.scan` and will miss slightly
// different patterns that still compute slice offset as loop iteration number.
static bool IsLoopIterationOffset(const HloInstruction* offset) {
  const HloComputation* parent = offset->parent();
  if (!parent->IsWhileBodyComputation()) return false;

  // Scan loops trip count must be known at compile time as it iterates over the
  // leading dimension of the statically shaped input.
  const HloInstruction* while_instr = parent->WhileCallInstruction();
  auto config = while_instr->backend_config<xla::WhileLoopBackendConfig>();
  if (!config.ok() || !config->has_known_trip_count()) return false;
  int32_t trip_count = config->known_trip_count().n();

  // Check that offset is defined by a loop fusion that computes offset
  // from the loop iteration number.
  if (!offset->IsLoopFusion() ||
      !Match(
          offset->fused_expression_root(),
          m::Select(m::Compare(m::Parameter(0), m::ConstantScalar<int32_t>(0)),
                    m::Add(m::Parameter(0), m::ConstantScalar(trip_count)),
                    m::Parameter(0)))) {
    return false;
  }

  // Check that we get loop iteration directly from loop parameters bundle.
  HloInstruction* get_loop_iteration;
  if (!Match(const_cast<HloInstruction*>(offset->operand(0)),
             m::GetTupleElement(&get_loop_iteration, m::Parameter(0)))) {
    return false;
  }
  int32_t loop_iter_idx = get_loop_iteration->tuple_index();

  // Check that loop iteration counter updated with a +1 fusion.
  const HloInstruction* loop_inc =
      parent->root_instruction()->operand(loop_iter_idx);
  if (!loop_inc->IsLoopFusion() ||
      !Match(loop_inc->fused_expression_root(),
             m::Add(m::Parameter(0), m::ConstantScalar<int32_t>(1)))) {
    return false;
  }

  return true;
}

absl::Status CollectSliceInfo(
    const BufferAssignment& buffer_assignment,
    const HloInstruction& fusion_instr,
    absl::Span<HloInstruction*> slice_instrs,
    std::vector<std::optional<std::vector<AddressComputationThunk::Offset>>>&
        offsets,
    std::vector<std::optional<Shape>>& orig_shapes,
    std::vector<std::optional<Shape>>& sliced_shapes,
    std::vector<std::optional<uint64_t>>& offset_byte_sizes, unsigned arg_idx) {
  auto* arg_slice_instr =
      DynCastOrNull<HloDynamicIndexInstruction>(slice_instrs[arg_idx]);
  if (arg_slice_instr == nullptr) {
    return absl::OkStatus();
  }

  std::vector<AddressComputationThunk::Offset> arg_offsets;
  for (auto idx_op : arg_slice_instr->index_operands()) {
    const auto* param = Cast<HloParameterInstruction>(idx_op);
    const auto* offset_value = fusion_instr.operand(param->parameter_number());

    if (auto* cst = DynCast<HloConstantInstruction>(offset_value)) {
      // Loop offset is defined by a constant scalar value.
      if (ShapeUtil::IsScalarWithElementType(cst->shape(),
                                             PrimitiveType::S32)) {
        arg_offsets.emplace_back() =
            static_cast<uint64_t>(cst->literal().data<int32_t>()[0]);
      } else if (ShapeUtil::IsScalarWithElementType(cst->shape(),
                                                    PrimitiveType::S64)) {
        arg_offsets.emplace_back() =
            static_cast<uint64_t>(cst->literal().data<int64_t>()[0]);
      } else if (ShapeUtil::IsScalarWithElementType(cst->shape(),
                                                    PrimitiveType::U32)) {
        arg_offsets.emplace_back() = cst->literal().data<uint32_t>()[0];
      } else if (ShapeUtil::IsScalarWithElementType(cst->shape(),
                                                    PrimitiveType::U64)) {
        arg_offsets.emplace_back() = cst->literal().data<uint64_t>()[0];
      } else {
        return absl::InternalError(absl::StrCat(
            "Unsupported constant offset shape: ", cst->shape().ToString()));
      }

    } else if (IsLoopIterationOffset(offset_value)) {
      // Loop offset defined by a loop iteration number.
      arg_offsets.emplace_back() = AddressComputationThunk::LoopIter();

    } else {
      // Loop offset computed on device and has to be transferred to host.
      TF_ASSIGN_OR_RETURN(arg_offsets.emplace_back(),
                          GetAllocationSlice(buffer_assignment, offset_value,
                                             /*index=*/{}));
    }
  }
  offsets[arg_idx] = std::move(arg_offsets);
  orig_shapes[arg_idx] = arg_slice_instr->operand(0)->shape();
  sliced_shapes[arg_idx] = DynCast<HloDynamicSliceInstruction>(arg_slice_instr)
                               ? arg_slice_instr->shape()
                               : arg_slice_instr->operand(1)->shape();
  offset_byte_sizes[arg_idx] = ShapeUtil::ByteSizeOfPrimitiveType(
      arg_slice_instr->index_operands().front()->shape().element_type());

  return absl::OkStatus();
}

absl::StatusOr<BufferAllocation::Slice> GetResultSlice(
    const BufferAssignment& buffer_assignment, const HloFusionAdaptor& adaptor,
    const HloInstruction& fusion_instr, const HloInstruction& start_instr,
    std::vector<HloInstruction*>& slice_instrs, const ShapeIndex& shape_idx,
    unsigned arg_idx) {
  auto* start = const_cast<HloInstruction*>(&start_instr);
  // Walk through ShapeIndex to find the real "user" (i.e. not get-tuple-element
  // user). Otherwise one sliced element will mark all buffers of all other
  // elements "sliced" too.
  if (start->shape().IsTuple()) {
    for (auto idx : shape_idx) {
      std::vector<HloGetTupleElementInstruction*> gte_users(
          start->shape().tuple_shapes_size(), nullptr);
      for (auto* user : start->users())
        if (auto* gte = DynCast<HloGetTupleElementInstruction>(user))
          gte_users[gte->tuple_index()] = gte;

      start = static_cast<HloInstruction*>(gte_users[idx]);
      if (start == nullptr)
        return GetAllocationSlice(buffer_assignment, &fusion_instr, shape_idx);
    }
  }

  auto slice_adaptor = HloFindIf(
      {HloInstructionAdaptor(*start, &adaptor)}, adaptor,
      [](auto node) { return node.opcode() == HloOpcode::kDynamicUpdateSlice; },
      /*visit_operands=*/false);
  if (slice_adaptor.has_value()) {
    auto* slice_instr =
        const_cast<HloInstruction*>(&slice_adaptor->instruction());
    slice_instrs[arg_idx] = slice_instr;

    if (!IsContiguousSlice(slice_instr->shape(),
                           Cast<HloDynamicUpdateSliceInstruction>(slice_instr)
                               ->update()
                               ->shape())) {
      return absl::InternalError(
          "AddressComputationFusion only handles contiguous slices "
          "currently");
    }
  }

  return GetAllocationSlice(buffer_assignment, &fusion_instr, shape_idx);
}

absl::StatusOr<FusionEmissionResult> EmitGemm(
    IrEmitterContext& ir_emitter_context, const HloFusionAdaptor& adaptor,
    const HloFusionInstruction& fusion,
    const HloCustomCallInstruction& custom_call) {
  const BufferAssignment& buffer_assignment =
      ir_emitter_context.buffer_assignment();

  std::vector<std::optional<std::vector<AddressComputationThunk::Offset>>>
      offset_buffer_indices(4, std::nullopt);
  std::vector<std::optional<Shape>> orig_shapes(4, std::nullopt);
  std::vector<std::optional<Shape>> sliced_shapes(4, std::nullopt);
  std::vector<std::optional<uint64_t>> offset_byte_sizes(4, std::nullopt);

  std::vector<HloInstruction*> slice_instrs(4, nullptr);

  unsigned arg_idx = 0;
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice lhs_slice,
                      GetOperandSlice(buffer_assignment, adaptor, fusion,
                                      *custom_call.operand(arg_idx),
                                      slice_instrs, /*shape_idx=*/{}, arg_idx));
  TF_RETURN_IF_ERROR(CollectSliceInfo(
      buffer_assignment, fusion, absl::Span<HloInstruction*>(slice_instrs),
      offset_buffer_indices, orig_shapes, sliced_shapes, offset_byte_sizes,
      arg_idx++));

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice rhs_slice,
                      GetOperandSlice(buffer_assignment, adaptor, fusion,
                                      *custom_call.operand(arg_idx),
                                      slice_instrs, /*shape_idx=*/{}, arg_idx));
  TF_RETURN_IF_ERROR(CollectSliceInfo(
      buffer_assignment, fusion, absl::Span<HloInstruction*>(slice_instrs),
      offset_buffer_indices, orig_shapes, sliced_shapes, offset_byte_sizes,
      arg_idx++));

  BufferAllocation::Slice output;
  std::optional<BufferAllocation::Slice> workspace = std::nullopt;
  std::optional<BufferAllocation::Slice> slice_workspace_fake = std::nullopt;

  // Handling cases where multiple operands share the same buffer, with
  // different offset by creating new fake allocations so each operand will have
  // a different buffer index. The slices can thus always start at offset 0.
  // AddressComputationThunk will take care of the offset adjustment.
  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations(4);
  if (fusion.shape().IsArray()) {
    TF_ASSIGN_OR_RETURN(
        output, GetResultSlice(buffer_assignment, adaptor, fusion, custom_call,
                               slice_instrs, /*shape_idx=*/{}, arg_idx));
    TF_RETURN_IF_ERROR(CollectSliceInfo(
        buffer_assignment, fusion, absl::Span<HloInstruction*>(slice_instrs),
        offset_buffer_indices, orig_shapes, sliced_shapes, offset_byte_sizes,
        arg_idx));
  } else {
    TF_ASSIGN_OR_RETURN(
        output,
        GetResultSlice(buffer_assignment, adaptor, fusion, custom_call,
                       slice_instrs, /*shape_idx=*/{kGEMMOutputBufferIndex},
                       arg_idx));
    TF_RETURN_IF_ERROR(CollectSliceInfo(
        buffer_assignment, fusion, absl::Span<HloInstruction*>(slice_instrs),
        offset_buffer_indices, orig_shapes, sliced_shapes, offset_byte_sizes,
        arg_idx++));

    // TODO(vuson): If we want to support slices of workspace, we'd need to
    // start `HloFindIf` with `get-tuple-element` with the right index.
    TF_ASSIGN_OR_RETURN(
        workspace, GetAllocationSlice(buffer_assignment, &fusion,
                                      /*index=*/{kGEMMWorkspaceBufferIndex}));
    TF_RETURN_IF_ERROR(CollectSliceInfo(
        buffer_assignment, fusion, absl::Span<HloInstruction*>(slice_instrs),
        offset_buffer_indices, orig_shapes, sliced_shapes, offset_byte_sizes,
        arg_idx));
    fake_allocations[arg_idx] = std::make_unique<BufferAllocation>(
        /*index=*/arg_idx, workspace->size(), /*color=*/0);
    slice_workspace_fake = BufferAllocation::Slice(
        fake_allocations[arg_idx].get(), 0, workspace->size());
  }

  if (absl::c_all_of(slice_instrs, [&](auto slice_instr) {
        return slice_instr == nullptr;
      })) {
    return absl::InternalError(
        "AddressComputationFusion expects at least one sliced "
        "operand/result");
  }

  bool deterministic_ops =
      ir_emitter_context.debug_options().xla_gpu_deterministic_ops() ||
      ir_emitter_context.debug_options().xla_gpu_exclude_nondeterministic_ops();

  TF_ASSIGN_OR_RETURN(
      GemmConfig config,
      GemmConfig::For(static_cast<const HloInstruction*>(&custom_call)));

  std::unique_ptr<Thunk> thunk;
  auto thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(&custom_call);

  if (absl::c_any_of(slice_instrs, [&](auto slice_instr) {
        return DynCastOrNull<HloDynamicIndexInstruction>(slice_instr) !=
               nullptr;
      })) {
    // Creating embedded GEMM thunk.
    unsigned fake_arg_idx = 0;
    int64_t lhs_byte_size =
        ShapeUtil::ByteSizeOf(custom_call.operand(fake_arg_idx)->shape());
    fake_allocations[fake_arg_idx] = std::make_unique<BufferAllocation>(
        /*index=*/fake_arg_idx, lhs_byte_size, /*color=*/0);
    BufferAllocation::Slice slice_lhs_fake(fake_allocations[fake_arg_idx].get(),
                                           0, lhs_byte_size);

    fake_arg_idx++;
    int64_t rhs_byte_size =
        ShapeUtil::ByteSizeOf(custom_call.operand(fake_arg_idx)->shape());
    fake_allocations[fake_arg_idx] = std::make_unique<BufferAllocation>(
        /*index=*/fake_arg_idx, rhs_byte_size, /*color=*/0);
    BufferAllocation::Slice slice_rhs_fake(fake_allocations[fake_arg_idx].get(),
                                           0, rhs_byte_size);

    fake_arg_idx++;
    int64_t out_fake_byte_size = ShapeUtil::ByteSizeOf(
        custom_call.shape().IsArray() ? custom_call.shape()
                                      : custom_call.shape().tuple_shapes(0));
    fake_allocations[fake_arg_idx] = std::make_unique<BufferAllocation>(
        /*index=*/fake_arg_idx, out_fake_byte_size, /*color=*/0);
    BufferAllocation::Slice slice_out_fake(fake_allocations[fake_arg_idx].get(),
                                           0, out_fake_byte_size);
    ThunkSequence seq;
    seq.emplace_back(std::make_unique<GemmThunk>(
        thunk_info, std::move(config), slice_lhs_fake, slice_rhs_fake,
        slice_out_fake, slice_workspace_fake, deterministic_ops));

    std::vector<std::optional<BufferAllocation::Slice>> arguments{
        lhs_slice, rhs_slice, output, workspace};

    thunk = std::make_unique<AddressComputationThunk>(
        thunk_info, std::make_unique<ThunkSequence>(std::move(seq)),
        std::move(arguments), std::move(fake_allocations),
        std::move(offset_buffer_indices), std::move(orig_shapes),
        std::move(sliced_shapes), std::move(offset_byte_sizes));
  } else {
    thunk = std::make_unique<GemmThunk>(thunk_info, std::move(config),
                                        lhs_slice, rhs_slice, output, workspace,
                                        deterministic_ops);
  }

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

  const std::string& call_target_name = custom_call.custom_call_target();

  // Typed FFI custom calls is a replacement for legacy custom calls with
  // a rich type safe API. It's under construction and not fully supported.
  bool is_ffi_custom_call =
      custom_call.api_version() == CustomCallApiVersion::API_VERSION_TYPED_FFI;

  void* call_target = CustomCallTargetRegistry::Global()->Lookup(
      call_target_name, std::string(ir_emitter_context.platform_name()));

  absl::StatusOr<ffi::HandlerRegistration> registration =
      ffi::FindHandler(call_target_name, ir_emitter_context.platform_name());

  // At least one implementation should be available at run time.
  bool found_custom_call = !is_ffi_custom_call && call_target != nullptr;
  bool found_ffi_handler = is_ffi_custom_call && registration.ok();

  if (!found_custom_call && !found_ffi_handler) {
    return absl::InternalError(
        "AddressComputationFusion expects custom calls that are emittable as "
        "thunks");
  }

  using Slices = std::vector<std::optional<CustomCallThunk::Slice>>;

  int64_t num_args = ShapeUtil::GetLeafCount(custom_call.shape());
  absl::c_for_each(custom_call.operands(), [&](auto* operand) {
    num_args += ShapeUtil::GetLeafCount(operand->shape());
  });

  std::vector<std::optional<std::vector<AddressComputationThunk::Offset>>>
      offsets(num_args, std::nullopt);
  std::vector<std::optional<Shape>> orig_shapes(num_args, std::nullopt);
  std::vector<std::optional<Shape>> sliced_shapes(num_args, std::nullopt);
  std::vector<std::optional<uint64_t>> offset_byte_sizes(num_args,
                                                         std::nullopt);

  std::vector<HloInstruction*> slice_instrs(num_args, nullptr);
  std::vector<std::optional<BufferAllocation::Slice>> arguments;

  unsigned arg_idx = 0;
  // TODO(vuson): add test for custom call with token-typed operands
  Slices operands;
  for (auto* operand : custom_call.operands()) {
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
        operand->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.IsToken()) {
            arg_idx++;
            operands.push_back(std::nullopt);
            return absl::OkStatus();
          }
          if (!subshape.IsArray()) {
            return absl::OkStatus();
          }
          TF_ASSIGN_OR_RETURN(
              auto slice,
              GetOperandSlice(buffer_assignment, adaptor, fusion, *operand,
                              slice_instrs, /*shape_idx=*/index, arg_idx));
          TF_RETURN_IF_ERROR(CollectSliceInfo(
              buffer_assignment, fusion,
              absl::Span<HloInstruction*>(slice_instrs), offsets, orig_shapes,
              sliced_shapes, offset_byte_sizes, arg_idx++));

          operands.push_back(CustomCallThunk::Slice{slice, subshape});
          arguments.push_back(slice);
          return absl::OkStatus();
        }));
  }

  Slices results;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      custom_call.shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.IsToken()) {
          arg_idx++;
          results.push_back(std::nullopt);
          return absl::OkStatus();
        }
        if (!subshape.IsArray()) {
          return absl::OkStatus();
        }
        TF_ASSIGN_OR_RETURN(
            auto slice,
            GetResultSlice(buffer_assignment, adaptor, fusion, custom_call,
                           slice_instrs, /*shape_idx=*/index, arg_idx));
        TF_RETURN_IF_ERROR(CollectSliceInfo(
            buffer_assignment, fusion,
            absl::Span<HloInstruction*>(slice_instrs), offsets, orig_shapes,
            sliced_shapes, offset_byte_sizes, arg_idx++));

        results.push_back(CustomCallThunk::Slice{slice, subshape});
        arguments.push_back(slice);
        return absl::OkStatus();
      }));

  if (absl::c_all_of(slice_instrs, [&](auto slice_instr) {
        return slice_instr == nullptr;
      })) {
    return absl::InternalError(
        "AddressComputationFusion expects at least one sliced "
        "operand/result");
  }

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
        if (auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(attr)) {
          TF_ASSIGN_OR_RETURN(attributes, xla::ffi::BuildAttributesMap(dict));
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

  std::unique_ptr<Thunk> thunk;
  auto thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(&custom_call);

  auto ffi_thunk = [&](Slices ops, Slices res) {
    auto& called_computations = custom_call.called_computations();
    return std::make_unique<CustomCallThunk>(
        thunk_info, registration->bundle, std::move(ops), std::move(res),
        std::move(attributes),
        called_computations.empty() ? nullptr : called_computations[0]);
  };

  auto legacy_thunk = [&](Slices ops, Slices res) {
    return std::make_unique<CustomCallThunk>(
        thunk_info, std::move(custom_call_target), std::move(ops),
        std::move(res), std::move(opaque));
  };

  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations(num_args);
  if (absl::c_any_of(slice_instrs, [&](auto slice_instr) {
        return DynCastOrNull<HloDynamicIndexInstruction>(slice_instr) !=
               nullptr;
      })) {
    // Creating embedded custom call thunk.
    unsigned fake_arg_idx = 0;

    Slices fake_operands;
    for (auto* operand : custom_call.operands()) {
      TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
          operand->shape(),
          [&](const Shape& subshape, const ShapeIndex& index) {
            if (subshape.IsToken()) {
              fake_arg_idx++;
              fake_operands.push_back(std::nullopt);
              return absl::OkStatus();
            }
            if (!subshape.IsArray()) {
              return absl::OkStatus();
            }

            int64_t operand_byte_size = ShapeUtil::ByteSizeOf(subshape);
            fake_allocations[fake_arg_idx] = std::make_unique<BufferAllocation>(
                /*index=*/fake_arg_idx, operand_byte_size, /*color=*/0);
            BufferAllocation::Slice fake_slice(
                fake_allocations[fake_arg_idx].get(), 0, operand_byte_size);

            fake_arg_idx++;
            fake_operands.push_back(
                CustomCallThunk::Slice{fake_slice, subshape});
            return absl::OkStatus();
          }));
    }

    Slices fake_results;
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
        custom_call.shape(),
        [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.IsToken()) {
            fake_arg_idx++;
            fake_results.push_back(std::nullopt);
            return absl::OkStatus();
          }
          if (!subshape.IsArray()) {
            return absl::OkStatus();
          }

          int64_t result_byte_size = ShapeUtil::ByteSizeOf(subshape);
          fake_allocations[fake_arg_idx] = std::make_unique<BufferAllocation>(
              /*index=*/fake_arg_idx, result_byte_size, /*color=*/0);
          BufferAllocation::Slice fake_slice(
              fake_allocations[fake_arg_idx].get(), 0, result_byte_size);

          fake_arg_idx++;
          fake_results.push_back(CustomCallThunk::Slice{fake_slice, subshape});
          return absl::OkStatus();
        }));

    ThunkSequence seq;
    seq.emplace_back(
        found_ffi_handler
            ? ffi_thunk(std::move(fake_operands), std::move(fake_results))
            : legacy_thunk(std::move(fake_operands), std::move(fake_results)));

    thunk = std::make_unique<AddressComputationThunk>(
        thunk_info, std::make_unique<ThunkSequence>(std::move(seq)),
        std::move(arguments), std::move(fake_allocations), std::move(offsets),
        std::move(orig_shapes), std::move(sliced_shapes),
        std::move(offset_byte_sizes));
  } else {
    thunk = found_ffi_handler
                ? ffi_thunk(std::move(operands), std::move(results))
                : legacy_thunk(std::move(operands), std::move(results));
  }

  FusionEmissionResult result;
  result.thunks.push_back(std::move(thunk));
  return result;
}

}  // namespace

absl::StatusOr<FusionEmissionResult> CustomFusion::Emit(
    IrEmitterContext& ir_emitter_context,
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

  TF_ASSIGN_OR_RETURN(
      auto thunk, BuildCustomKernelThunkForFusion(ir_emitter_context, fusion,
                                                  std::move(kernels[0])));

  FusionEmissionResult result;
  result.thunks.push_back(std::move(thunk));
  return result;
}

absl::StatusOr<FusionEmissionResult> AddressComputationFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  const HloFusionAdaptor& adaptor = analysis_.fusion();
  auto maybe_custom_call_adaptor = HloFindIf(
      adaptor.GetRoots(), adaptor,
      [](auto node) { return node.opcode() == HloOpcode::kCustomCall; });
  if (maybe_custom_call_adaptor == std::nullopt) {
    return absl::InternalError(
        "AddressComputationFusion requires a CustomCall hero");
  }

  const auto& custom_call = *static_cast<const HloCustomCallInstruction*>(
      &maybe_custom_call_adaptor->instruction());
  if (IsLegacyCublasMatmul(custom_call)) {
    return EmitGemm(ir_emitter_context, adaptor, fusion, custom_call);
  }

  return EmitCustomCall(ir_emitter_context, adaptor, fusion, custom_call);
}

}  // namespace gpu
}  // namespace xla
