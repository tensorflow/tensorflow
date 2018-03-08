/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/llvm_ir/ops.h"
#include "tensorflow/compiler/xla/service/gpu/parallel_loop_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/partition_assignment.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"

namespace xla {
namespace llvm_ir {

bool CanUpdateDynamicSliceInPlace(HloInstruction* dynamic_update_slice,
                                  const BufferAssignment& assignment) {
  CHECK_EQ(HloOpcode::kDynamicUpdateSlice, dynamic_update_slice->opcode());
  const HloInstruction* operand = dynamic_update_slice->operand(0);
  return assignment.HasTopLevelAllocation(dynamic_update_slice) &&
         assignment.HasTopLevelAllocation(operand) &&
         assignment.SharesTopLevelSlice(dynamic_update_slice, operand);
}

// Shared implementation of EmitDynamicUpdateSliceInPlace and
// EmitFusedDynamicUpdateSliceInPlace.
//
// Emits a sequential loop if launch_dimensions is null.
static Status EmitDynamicUpdateSliceInPlaceImpl(
    const Shape& update_shape, const ElementGenerator& start_indices_generator,
    ElementGenerator update_array_generator, const IrArray& output_array,
    const gpu::LaunchDimensions* launch_dimensions,
    tensorflow::StringPiece name, llvm::IRBuilder<>* ir_builder) {
  const Shape& output_shape = output_array.GetShape();

  // Read start indices from start_indices_generator.
  const int64 rank = ShapeUtil::Rank(output_shape);
  IrArray::Index start_index(rank);
  for (int64 i = 0; i < rank; ++i) {
    IrArray::Index dim_index({ir_builder->getInt64(i)});
    TF_ASSIGN_OR_RETURN(start_index[i], start_indices_generator(dim_index));
  }

  auto loop_body_emitter = [&](const IrArray::Index& update_index) -> Status {
    // Calculate output_index, where we'll write the value from update.  For
    // each dimension,
    //
    //   output_index[dim] = (start_index[dim] + update_index[dim]) % dim_size.
    //
    IrArray::Index output_index(rank);
    for (int64 i = 0; i < rank; ++i) {
      llvm::Value* dim_size = llvm::ConstantInt::get(
          update_index[i]->getType(), output_shape.dimensions(i));
      llvm::Value* start_index0 = ir_builder->CreateZExtOrBitCast(
          start_index[i], update_index[i]->getType());
      output_index[i] = ir_builder->CreateURem(
          ir_builder->CreateAdd(start_index0, update_index[i]), dim_size);
    }

    // Do output[output_index] = update[update_index].
    TF_ASSIGN_OR_RETURN(llvm::Value * update_data,
                        update_array_generator(update_index));
    output_array.EmitWriteArrayElement(output_index, update_data, ir_builder);
    return Status::OK();
  };

  if (launch_dimensions != nullptr) {
    return gpu::ParallelLoopEmitter(loop_body_emitter, update_shape,
                                    *launch_dimensions, ir_builder)
        .EmitLoop(name);
  }
  return LoopEmitter(loop_body_emitter, update_shape, ir_builder)
      .EmitLoop(name);
}

Status EmitDynamicUpdateSliceInPlace(
    tensorflow::gtl::ArraySlice<IrArray> operand_arrays,
    const IrArray& output_array, tensorflow::StringPiece name,
    llvm::IRBuilder<>* ir_builder) {
  VLOG(2) << "EmitDynamicUpdateSliceInPlace for " << name;

  // No need to use operand_arrays[0], the input array of the
  // dynamic-update-slice, because we know it aliases the op's output.
  IrArray update_array = operand_arrays[1];
  IrArray start_indices_array = operand_arrays[2];
  Shape output_shape = output_array.GetShape();
  Shape update_shape = update_array.GetShape();

  ElementGenerator start_indices_generator = [&](const IrArray::Index& index) {
    return start_indices_array.EmitReadArrayElement(index, ir_builder);
  };
  ElementGenerator update_array_generator = [&](const IrArray::Index& index) {
    return update_array.EmitReadArrayElement(index, ir_builder);
  };

  return EmitDynamicUpdateSliceInPlaceImpl(
      update_shape, start_indices_generator, update_array_generator,
      output_array, /*launch_dimensions=*/nullptr, name, ir_builder);
}

// Shared implementation for EmitFusedDynamicUpdateSliceInPlace and
// EmitParallelFusedDynamicUpdateSliceInPlace.
//
// Emits a sequential loop if launch_dimensions is null.
static Status EmitFusedDynamicUpdateSliceInPlaceImpl(
    HloInstruction* fusion,
    tensorflow::gtl::ArraySlice<IrArray> fusion_operand_arrays,
    const IrArray& fusion_output_array, ElementalIrEmitter* elemental_emitter,
    const gpu::LaunchDimensions* launch_dimensions,
    llvm::IRBuilder<>* ir_builder) {
  CHECK_EQ(fusion->opcode(), HloOpcode::kFusion);
  VLOG(2) << "EmitFusedDynamicUpdateSliceInPlace for "
          << fusion->ToShortString();

  auto* dynamic_update_slice = fusion->fused_expression_root();

  const auto* update = dynamic_update_slice->operand(1);
  const auto* start_indices = dynamic_update_slice->operand(2);
  Shape update_shape = update->shape();

  // Our in-place dynamic-update-slice implementation emits a loop over
  // update_shape.  To emit a cache-friendly loop, we need to know that shape's
  // layout.
  //
  // update_shape is inside a fusion node -- it's never materialized in memory
  // and thus doesn't have a layout.  In this case we use the layout of the
  // fusion node for iteration, since that corresponds to the order in memory of
  // the buffer we'll be writing to.
  //
  // (This isn't necessarily optimal; in some cases it might be faster to peek
  // through the chain of ops that gives us the update operand and use the
  // layout of its source buffer(s).  But this is no worse than we do with
  // fusion elsewhere.)
  TF_RETURN_IF_ERROR(
      LayoutUtil::CopyLayoutBetweenShapes(fusion->shape(), &update_shape));

  // Create element generators for update and start_indices.
  FusedIrEmitter fused_emitter(fusion_operand_arrays, elemental_emitter);
  TF_RETURN_IF_ERROR(dynamic_update_slice->Accept(&fused_emitter));
  ElementGenerator update_array_generator = fused_emitter.GetGenerator(update);
  ElementGenerator start_indices_generator =
      fused_emitter.GetGenerator(start_indices);

  return EmitDynamicUpdateSliceInPlaceImpl(
      update_shape, start_indices_generator, update_array_generator,
      fusion_output_array, launch_dimensions, IrName(fusion), ir_builder);
}

Status EmitFusedDynamicUpdateSliceInPlace(
    HloInstruction* fusion,
    tensorflow::gtl::ArraySlice<IrArray> fusion_operand_arrays,
    const IrArray& fusion_output_array, ElementalIrEmitter* elemental_emitter,
    llvm::IRBuilder<>* ir_builder) {
  return EmitFusedDynamicUpdateSliceInPlaceImpl(
      fusion, fusion_operand_arrays, fusion_output_array, elemental_emitter,
      /*launch_dimensions=*/nullptr, ir_builder);
}

Status EmitParallelFusedDynamicUpdateSliceInPlace(
    HloInstruction* fusion,
    tensorflow::gtl::ArraySlice<IrArray> fusion_operand_arrays,
    const IrArray& fusion_output_array, ElementalIrEmitter* elemental_emitter,
    const gpu::LaunchDimensions& launch_dimensions,
    llvm::IRBuilder<>* ir_builder) {
  return EmitFusedDynamicUpdateSliceInPlaceImpl(
      fusion, fusion_operand_arrays, fusion_output_array, elemental_emitter,
      &launch_dimensions, ir_builder);
}

}  // namespace llvm_ir
}  // namespace xla
