/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/llvm_ir/dynamic_update_slice_util.h"

#include <cstdint>
#include <functional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/parallel_loop_emitter.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace llvm_ir {

bool MayBeImplementedAsInPlaceDynamicUpdateSlice(const HloInstruction* instr) {
  // Today we can't emit a dynamic-update-slice if the DUS node is parallelized;
  // the emitter will not emit correct code.  It's possible to change this, but
  // then ParallelTaskAssigner would have to somehow know whether a node *will*
  // be emitted as an in-place DUS, and it can't, because it doesn't have a
  // buffer assignment when it runs.
  auto cpu_backend_config_or = instr->backend_config<xla::cpu::BackendConfig>();
  if (cpu_backend_config_or.ok() &&
      !cpu_backend_config_or->outer_dimension_partitions().empty()) {
    return false;
  }

  // Until we know the final buffer assignment, any unfused dynamic-update-slice
  // might be implementable as an in-place DUS.
  if (instr->opcode() == HloOpcode::kDynamicUpdateSlice) {
    return true;
  }

  // A fusion may be implementable as an in-place dynamic update slice if
  //  - it's a loop fusion,
  //  - dynamic-update-slice is the root of the fusion, and
  //  - operand 0 of the dynamic-update-slice is a parameter to the fusion
  //    (ignoring any get-tuple-element operations in the way).
  if (instr->IsLoopFusion()) {
    const HloInstruction* fused_root = instr->fused_expression_root();
    return fused_root->opcode() == HloOpcode::kDynamicUpdateSlice &&
           fused_root->operand(0)->LatestNonGteAncestor()->opcode() ==
               HloOpcode::kParameter;
  }

  return false;
}

bool CanUpdateDynamicSliceInPlace(HloInstruction* dynamic_update_slice,
                                  const BufferAssignment& assignment) {
  CHECK_EQ(HloOpcode::kDynamicUpdateSlice, dynamic_update_slice->opcode());
  const HloInstruction* operand = dynamic_update_slice->operand(0);
  return assignment.HasTopLevelAllocation(dynamic_update_slice) &&
         assignment.HasTopLevelAllocation(operand) &&
         assignment.SharesTopLevelSlice(dynamic_update_slice, operand);
}

bool CanEmitFusedDynamicUpdateSliceInPlace(HloInstruction* fusion,
                                           const BufferAssignment& assignment) {
  CHECK_EQ(fusion->opcode(), HloOpcode::kFusion);
  if (!MayBeImplementedAsInPlaceDynamicUpdateSlice(fusion)) {
    return false;
  }

  // Walk DynamicUpdateSlice operand(0) to fused parameter and get its
  // associated operand. See if it shares an allocation with this operand.
  HloInstruction* fused_root = fusion->fused_expression_root();
  HloInstruction* fusion_operand;
  ShapeIndex index;
  std::tie(fusion_operand, index) =
      fused_root->mutable_operand(0)->LatestNonGteAncestorAndIndex();
  // MayBeImplementedAsInPlaceDynamicUpdateSlice should have ensured that
  // fusion_operand is a parameter.
  CHECK_EQ(fusion_operand->opcode(), HloOpcode::kParameter);
  auto* operand = fusion->operand(fusion_operand->parameter_number());
  return assignment.HasAllocationAt(operand, index) &&
         assignment.HasAllocationAt(fusion, {}) &&
         assignment.SharesSliceAtIndex(fusion, {}, operand, index);
}

// Shared implementation of EmitDynamicUpdateSliceInPlace and
// EmitFusedDynamicUpdateSliceInPlace.
//
// Emits a sequential loop if launch_dimensions is null.
using IndexGenerator = std::function<absl::StatusOr<llvm::Value*>(int64_t)>;

static absl::Status EmitDynamicUpdateSliceInPlaceImpl(
    const Shape& update_shape, const IndexGenerator& start_indices_generator,
    bool is_signed, ElementGenerator update_array_generator,
    const IrArray& output_array, const gpu::LaunchDimensions* launch_dimensions,
    absl::string_view name, llvm::IRBuilderBase* b) {
  const Shape& output_shape = output_array.GetShape();

  // Read start indices from start_indices_generator.
  const int64_t rank = output_shape.rank();
  std::vector<llvm::Value*> start_multi_index(rank);
  for (int64_t i = 0; i < rank; ++i) {
    TF_ASSIGN_OR_RETURN(start_multi_index[i], start_indices_generator(i));
    llvm::Value* output_dim_size = llvm::ConstantInt::get(
        start_multi_index[i]->getType(), output_shape.dimensions(i));
    llvm::Value* update_dim_size = llvm::ConstantInt::get(
        start_multi_index[i]->getType(), update_shape.dimensions(i));

    // Clamp the start index so that the update region fits in the operand.
    // start_index = clamp(start_index, 0, output_dim_size - update_dim_size)
    llvm::Value* max_bound = b->CreateSub(output_dim_size, update_dim_size);
    llvm::Value* zero =
        llvm::ConstantInt::get(start_multi_index[i]->getType(), 0);
    start_multi_index[i] =
        b->CreateSelect(b->CreateICmp(is_signed ? llvm::ICmpInst::ICMP_SGE
                                                : llvm::ICmpInst::ICMP_UGE,
                                      zero, start_multi_index[i]),
                        zero, start_multi_index[i]);

    start_multi_index[i] =
        b->CreateSelect(b->CreateICmp(is_signed ? llvm::ICmpInst::ICMP_SLE
                                                : llvm::ICmpInst::ICMP_ULE,
                                      max_bound, start_multi_index[i]),
                        max_bound, start_multi_index[i]);
  }

  auto loop_body_emitter =
      [&](const IrArray::Index& update_index) -> absl::Status {
    // Calculate output_index, where we'll write the value from update.  For
    // each dimension,
    //
    //   output_index[dim] = start_index[dim] + update_index[dim]
    //
    std::vector<llvm::Value*> output_multi_index(rank);
    for (int64_t i = 0; i < rank; ++i) {
      llvm::Value* start_index0 = b->CreateIntCast(
          start_multi_index[i], update_index[i]->getType(), is_signed);
      output_multi_index[i] = b->CreateAdd(start_index0, update_index[i]);
    }

    // Do output[output_index] = update[update_index].
    IrArray::Index output_index(output_multi_index, output_shape,
                                b->getInt64Ty());
    TF_ASSIGN_OR_RETURN(llvm::Value * update_data,
                        update_array_generator(update_index));
    output_array.EmitWriteArrayElement(output_index, update_data, b);
    return absl::OkStatus();
  };

  if (launch_dimensions != nullptr) {
    return gpu::ParallelLoopEmitter(loop_body_emitter, update_shape,
                                    *launch_dimensions, b)
        .EmitLoop(name);
  }
  return LoopEmitter(loop_body_emitter, update_shape, b).EmitLoop(name);
}

absl::Status EmitDynamicUpdateSliceInPlace(
    absl::Span<const IrArray> operand_arrays, const IrArray& output_array,
    absl::string_view name, llvm::IRBuilderBase* b) {
  VLOG(2) << "EmitDynamicUpdateSliceInPlace for " << name;

  // No need to use operand_arrays[0], the input array of the
  // dynamic-update-slice, because we know it aliases the op's output.
  IrArray update_array = operand_arrays[1];
  IrArray start_indices_array = operand_arrays[2];
  Shape output_shape = output_array.GetShape();
  Shape update_shape = update_array.GetShape();

  IndexGenerator start_indices_generator = [&](int64_t index) {
    return operand_arrays[2 + index].EmitReadArrayElement(
        IrArray::Index(b->getInt64Ty()), b);
  };
  ElementGenerator update_array_generator = [&](const IrArray::Index& index) {
    return update_array.EmitReadArrayElement(index, b);
  };

  bool is_signed = ShapeUtil::ElementIsSigned(start_indices_array.GetShape());
  return EmitDynamicUpdateSliceInPlaceImpl(
      update_shape, start_indices_generator, is_signed, update_array_generator,
      output_array, /*launch_dimensions=*/nullptr, name, b);
}

// Shared implementation for EmitFusedDynamicUpdateSliceInPlace and
// EmitParallelFusedDynamicUpdateSliceInPlace.
//
// Emits a sequential loop if launch_dimensions is null.
static absl::Status EmitFusedDynamicUpdateSliceInPlaceImpl(
    const HloComputation* fusion,
    const std::vector<std::pair<const HloInstruction*, const IrArray>>&
        dus_and_output_array,
    FusedIrEmitter* fused_emitter,
    const gpu::LaunchDimensions* launch_dimensions, llvm::IRBuilderBase* b) {
  VLOG(2) << "EmitFusedDynamicUpdateSliceInPlace for " << fusion->ToString();

  CHECK_GE(dus_and_output_array.size(), 1);
  Shape update_shape = dus_and_output_array[0].first->operand(1)->shape();

  for (auto& dus_output_pair : dus_and_output_array) {
    const HloInstruction* dynamic_update_slice = dus_output_pair.first;
    const IrArray& fusion_output_array = dus_output_pair.second;
    const auto* update = dynamic_update_slice->operand(1);
    const auto* start_indices = dynamic_update_slice->operand(2);
    CHECK(ShapeUtil::EqualIgnoringElementType(update->shape(), update_shape));

    // Our in-place dynamic-update-slice implementation emits a loop over
    // update_shape.  To emit a cache-friendly loop, we need to know that
    // shape's layout.
    //
    // update_shape is inside a fusion node -- it's never materialized in memory
    // and thus doesn't have a layout.  In this case we use the layout of the
    // fusion node for iteration, since that corresponds to the order in memory
    // of the buffer we'll be writing to.
    //
    // (This isn't necessarily optimal; in some cases it might be faster to peek
    // through the chain of ops that gives us the update operand and use the
    // layout of its source buffer(s).  But this is no worse than we do with
    // fusion elsewhere.)
    TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
        dynamic_update_slice->shape(), &update_shape));

    // Create element generators for update and start_indices.
    TF_ASSIGN_OR_RETURN(ElementGenerator update_array_generator,
                        fused_emitter->GetGenerator(*update));

    IndexGenerator start_indices_generator =
        [&](int64_t index) -> absl::StatusOr<llvm::Value*> {
      TF_ASSIGN_OR_RETURN(ElementGenerator element_generator,
                          fused_emitter->GetGenerator(
                              *dynamic_update_slice->operand(2 + index)));
      return element_generator(IrArray::Index(b->getInt64Ty()));
    };
    bool is_signed = ShapeUtil::ElementIsSigned(start_indices->shape());

    TF_RETURN_IF_ERROR(EmitDynamicUpdateSliceInPlaceImpl(
        update_shape, start_indices_generator, is_signed,
        update_array_generator, fusion_output_array, launch_dimensions,
        IrName(dynamic_update_slice), b));
  }

  return absl::OkStatus();
}

absl::Status EmitFusedDynamicUpdateSliceInPlace(
    HloInstruction* fusion, const IrArray& fusion_output_array,
    FusedIrEmitter* fused_emitter, llvm::IRBuilderBase* b) {
  HloInstruction* dus = fusion->called_computations()[0]->root_instruction();
  CHECK_EQ(dus->opcode(), HloOpcode::kDynamicUpdateSlice);
  std::vector<std::pair<const HloInstruction*, const IrArray>>
      dus_and_output_array{std::make_pair(dus, fusion_output_array)};

  return EmitFusedDynamicUpdateSliceInPlaceImpl(
      fusion->called_computations()[0], dus_and_output_array, fused_emitter,
      /*launch_dimensions=*/nullptr, b);
}

}  // namespace llvm_ir
}  // namespace xla
