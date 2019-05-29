/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/cudnn_conv_pad_features.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace gpu {
// Creates and returns an HLO that zero-pads one or more dimensions in the given
// instruction so that its shape is equal to the given shape.
//
// Padding is added to the end of each relevant dimension.
//
// If the instruction already has the given shape, simply returns it without an
// intervening pad.
static HloInstruction* PadInstruction(HloInstruction* instr,
                                      const Shape& new_shape) {
  HloComputation* comp = instr->parent();

  const Shape& shape = instr->shape();
  auto* zero = comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(shape.element_type())));

  PaddingConfig pad_config = MakeNoPaddingConfig(shape.rank());

  bool added_padding = false;
  for (int64 dim = 0; dim < shape.rank(); ++dim) {
    if (shape.dimensions(dim) == new_shape.dimensions(dim)) {
      continue;
    }
    CHECK_GT(new_shape.dimensions(dim), shape.dimensions(dim));
    pad_config.mutable_dimensions(dim)->set_edge_padding_high(
        new_shape.dimensions(dim) - shape.dimensions(dim));
    added_padding = true;
  }

  if (!added_padding) {
    return instr;
  }
  return comp->AddInstruction(
      HloInstruction::CreatePad(new_shape, instr, zero, pad_config));
}

// Modifies the given convolution to have the given input and result shapes.
static Status PadConv(HloCustomCallInstruction* conv,
                      absl::Span<const Shape> new_input_shapes,
                      const Shape& new_result_shape) {
  CHECK_EQ(0, conv->shape().tuple_shapes(1).dimensions(0))
      << "conv must use 0 scratch bytes, i.e. this pass must be run "
         "before CudnnConvAlgorithmPicker.";
  std::vector<HloInstruction*> new_operands;
  for (int i = 0; i < conv->operand_count(); ++i) {
    new_operands.push_back(
        PadInstruction(conv->mutable_operand(i), new_input_shapes[i]));
  }
  const Shape& result_shape = conv->shape().tuple_shapes(0);

  bool changed = false;
  for (int i = 0; i < conv->operand_count(); ++i) {
    changed |= (new_operands[i] != conv->mutable_operand(i));
  }
  CHECK(changed) << "We should have had to pad at least one input operand.";

  auto add = [&](std::unique_ptr<HloInstruction> new_instr) {
    return conv->parent()->AddInstruction(std::move(new_instr));
  };

  Shape new_conv_shape = ShapeUtil::MakeTupleShape(
      {new_result_shape, ShapeUtil::MakeShape(U8, {0})});
  auto* new_conv =
      add(conv->CloneWithNewOperands(new_conv_shape, new_operands));

  // Slice the new conv result if necessary, keeping in mind that new_conv
  // has tuple shape (new_result_shape, u8[0]).
  if (!ShapeUtil::Equal(result_shape, new_result_shape)) {
    std::vector<int64> start_indices(result_shape.dimensions_size(), 0);
    std::vector<int64> end_indices(result_shape.dimensions().begin(),
                                   result_shape.dimensions().end());
    std::vector<int64> strides(result_shape.dimensions_size(), 1);

    auto* new_conv_result = add(
        HloInstruction::CreateGetTupleElement(new_result_shape, new_conv, 0));
    auto* empty_temp_buffer =
        add(HloInstruction::CreateConstant(LiteralUtil::CreateR1<uint8>({})));
    auto* sliced_result = add(HloInstruction::CreateSlice(
        result_shape, new_conv_result, start_indices, end_indices, strides));
    new_conv =
        add(HloInstruction::CreateTuple({sliced_result, empty_temp_buffer}));
  }

  VLOG(2) << "Padded features of " << conv->ToString() << ", replaced with "
          << new_conv->ToString();
  return conv->parent()->ReplaceInstruction(conv, new_conv);
}

static std::vector<HloCustomCallInstruction*> GetRelevantConvs(
    HloComputation* comp) {
  std::vector<HloCustomCallInstruction*> convs;
  for (HloInstruction* instr : comp->instructions()) {
    if (IsCustomCallToDnnConvolution(*instr)) {
      convs.push_back(Cast<HloCustomCallInstruction>(instr));
    }
  }
  return convs;
}

StatusOr<bool> CudnnConvPadFeatures::CudnnConvPadFeatures::Run(
    HloModule* module,
    StatusOr<bool> (*resolve_pad_shapes)(HloCustomCallInstruction* conv,
                                         std::vector<Shape>* new_input_shapes,
                                         Shape* new_result_shape)) {
  bool changed = false;
  for (HloComputation* comp : module->MakeNonfusionComputations()) {
    for (HloCustomCallInstruction* conv : GetRelevantConvs(comp)) {
      std::vector<Shape> new_input_shapes;
      Shape new_result_shape;
      TF_ASSIGN_OR_RETURN(
          bool result,
          resolve_pad_shapes(conv, &new_input_shapes, &new_result_shape));
      if (result) {
        TF_RETURN_IF_ERROR(PadConv(conv, new_input_shapes, new_result_shape));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
