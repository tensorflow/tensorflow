/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "xla/service/gpu/fusions/input_slices.h"

#include "llvm/IR/IRBuilder.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/elemental_ir_emitter.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/parallel_loop_emitter.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/kernel_support_library.h"
#include "xla/service/llvm_ir/llvm_util.h"

namespace xla {
namespace gpu {
namespace {

// Emits code for slices based on the below structure. An if statement with
// a guarding condition is generated for each ROOT slice.
//
// Pseudo code:
//
// Compute values of slice input operands
//
// Compute guarding_cond0
// if (guarding_cond0) {
//   Write to output of slice0
// }
//
// Compute guarding_cond1
// if (guarding_cond1) {
//   Write to output of slice1
// }
//
Status EmitElementForInputFusibleSlices(
    ElementalIrEmitter& elemental_emitter,
    const HloComputation* fused_computation,
    const std::vector<llvm_ir::IrArray>& inputs,
    const std::vector<llvm_ir::IrArray>& outputs,
    const llvm_ir::IrArray::Index& index, llvm::IRBuilder<>* builder) {
  VLOG(10) << "Emitting slice input fusion for "
           << fused_computation->ToString();

  HloInstruction* slice_or_tuple = fused_computation->root_instruction();
  auto slice_instructions = [&]() -> absl::Span<HloInstruction* const> {
    if (slice_or_tuple->opcode() == HloOpcode::kSlice) {
      return absl::Span<HloInstruction* const>(&slice_or_tuple, 1);
    }
    CHECK_EQ(slice_or_tuple->opcode(), HloOpcode::kTuple);
    return slice_or_tuple->operands();
  }();

  // Emit input operand values of slices.
  std::vector<llvm::Value*> input_ir_values;
  FusedIrEmitter fused_emitter(elemental_emitter);
  for (int i = 0; i < fused_computation->num_parameters(); i++) {
    fused_emitter.BindGenerator(
        *fused_computation->parameter_instruction(i),
        [&inputs, i, builder](llvm_ir::IrArray::Index index) {
          return inputs[i].EmitReadArrayElement(index, builder);
        });
  }
  for (const HloInstruction* slice : slice_instructions) {
    auto input_generator = *fused_emitter.GetGenerator(*slice->operand(0));
    input_ir_values.push_back(input_generator(index).value());
  }

  // Emit for slice_instructions.
  KernelSupportLibrary ksl(builder, llvm_ir::UnrollMode::kDefaultUnroll);
  for (int64_t i = 0; i < slice_instructions.size(); ++i) {
    HloInstruction* slice = slice_instructions[i];

    // guarding_cond := index >= start && index < limit, for each dim.
    std::vector<llvm::Value*> index_within_ranges;
    for (size_t dim = 0; dim < slice->slice_starts().size(); ++dim) {
      CHECK_EQ(slice->slice_strides(dim), 1);
      auto larger_or_equal_than_start = builder->CreateICmpSGE(
          index.multidim()[dim],
          index.GetConstantWithIndexType(slice->slice_starts(dim)));
      llvm::Value* smaller_than_limit = builder->CreateICmpSLT(
          index.multidim()[dim],
          index.GetConstantWithIndexType(slice->slice_limits(dim)));
      llvm::Value* within_range =
          builder->CreateAnd(larger_or_equal_than_start, smaller_than_limit);
      index_within_ranges.push_back(within_range);
    }
    llvm::Value* guarding_cond = builder->CreateAnd(index_within_ranges);

    auto emit_slice_elem_func = [&] {
      const std::vector<llvm::Value*>& src_multidim = index.multidim();
      std::vector<llvm::Value*> dst_multidim(src_multidim.size());
      for (size_t dim = 0; dim < src_multidim.size(); ++dim) {
        dst_multidim[dim] = builder->CreateSub(
            src_multidim[dim],
            index.GetConstantWithIndexType(slice->slice_starts(dim)));
      }
      llvm_ir::IrArray src_ir_array = outputs[i];
      llvm_ir::IrArray::Index slice_dst_index(dst_multidim, slice->shape(),
                                              index.GetType());
      src_ir_array.EmitWriteArrayElement(slice_dst_index, input_ir_values[i],
                                         builder);
    };

    ksl.If(absl::StrCat("slice", i), guarding_cond, emit_slice_elem_func);
  }
  return OkStatus();
}

// Gets the input shape of the ROOT slices, which will be used as the kernel
// launch dims. The slice input fusion requires the input shapes of the ROOT
// slices to be the same although the (slice) output shapes can be different.
//
// Returns the input shape of the ROOT slices if all the input shapes of ROOT
// slices are the same and the slices are non-strided. Otherwise, returns
// FailedPrecondition.
StatusOr<Shape> GetConsistentInputShapeForRootSlices(
    const HloComputation* fused_computation) {
  const HloInstruction& root = *fused_computation->root_instruction();
  if (root.opcode() == HloOpcode::kSlice) {
    return root.operands()[0]->shape();
  }

  CHECK_EQ(root.opcode(), HloOpcode::kTuple);
  const Shape& first_slice_operand_shape =
      root.operands()[0]->operands()[0]->shape();
  for (size_t i = 1; i < root.operands().size(); ++i) {
    const HloInstruction* slice = root.operands()[i];
    const Shape& operand_shape = slice->operands()[0]->shape();
    if (!ShapeUtil::EqualIgnoringElementType(first_slice_operand_shape,
                                             operand_shape)) {
      return FailedPrecondition(
          "Fused slices do not have the same input shape, fused computation = "
          "%s.",
          root.parent()->name());
    }
  }

  return first_slice_operand_shape;
}

}  // namespace

StatusOr<LaunchDimensions> InputSlicesFusion::launch_dimensions(
    IrEmitterContext& ir_emitter_context, int kernel_index) const {
  return analysis_.GetLaunchDimensions();
}

Status InputSlicesFusion::EmitKernel(
    IrEmitterContext& ir_emitter_context, ElementalIrEmitter& elemental_emitter,
    const HloFusionInstruction& fusion, const LaunchDimensions& launch_dims,
    std::vector<llvm_ir::IrArray> inputs, std::vector<llvm_ir::IrArray> outputs,
    llvm::IRBuilder<>* builder, int kernel_index) const {
  TF_ASSIGN_OR_RETURN(Shape element_shape,
                      GetConsistentInputShapeForRootSlices(
                          fusion.fused_instructions_computation()));
  return ParallelLoopEmitter(
             [&](const llvm_ir::IrArray::Index index) -> Status {
               return EmitElementForInputFusibleSlices(
                   elemental_emitter, fusion.fused_instructions_computation(),
                   inputs, outputs, index, builder);
             },
             element_shape, launch_dims, builder)
      .EmitLoop(
          fusion.name(),
          GetIndexTypeForKernel(&fusion, launch_dims.launch_bound(), builder));
}

}  // namespace gpu
}  // namespace xla
