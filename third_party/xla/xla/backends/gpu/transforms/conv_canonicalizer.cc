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

#include "xla/backends/gpu/transforms/conv_canonicalizer.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

// Checks if a constant literal is losslessly convertible to S8.
bool LiteralFitsInS8(const Literal& literal) {
  if (!literal.shape().IsArray()) {
    return false;
  }
  PrimitiveType type = literal.shape().element_type();
  if (type == S8) {
    return true;
  }

  absl::StatusOr<Literal> converted1 = literal.Convert(S8);
  if (!converted1.ok()) {
    return false;
  }

  absl::StatusOr<Literal> converted2 = converted1->Convert(type);
  if (!converted2.ok()) {
    return false;
  }

  return literal == *converted2;
}

// Canonicalizes an s32 convolution operand into `s32 convert(s8_node)`.
absl::StatusOr<HloInstruction*> CanonicalizeOperandToS8Convert(
    HloComputation* comp, HloInstruction* operand) {
  if (operand->shape().element_type() != S32) {
    return operand;
  }

  // 1. s32 constant -> s32 convert(s8 constant)
  if (operand->opcode() == HloOpcode::kConstant &&
      LiteralFitsInS8(operand->literal())) {
    absl::StatusOr<Literal> s8_literal = operand->literal().Convert(S8);
    if (!s8_literal.ok()) {
      return operand;
    }

    HloInstruction* s8_const = comp->AddInstruction(
        HloInstruction::CreateConstant(std::move(*s8_literal)));
    return comp->AddInstruction(
        HloInstruction::CreateConvert(operand->shape(), s8_const));
  }

  // 2. Redundant Convert: s32 convert(s32 convert(s8_src)) -> s32
  // convert(s8_src)
  if (operand->opcode() == HloOpcode::kConvert) {
    HloInstruction* src = operand->mutable_operand(0);
    if (src->opcode() == HloOpcode::kConvert &&
        src->operand(0)->shape().element_type() == S8) {
      return comp->AddInstruction(HloInstruction::CreateConvert(
          operand->shape(), src->mutable_operand(0)));
    }
  }

  // 3. Push s32 convert down through spatial/elementwise ops (Reshape,
  // Transpose, Broadcast, Pad, Slice, etc.): op(s32 convert(s8_src)) -> s32
  // convert(s8 op(s8_src))
  if (operand->operand_count() > 0) {
    HloInstruction* src = operand->mutable_operand(0);
    if (src->opcode() == HloOpcode::kConvert &&
        src->operand(0)->shape().element_type() == S8) {
      HloInstruction::InstructionVector s8_operands = operand->operands();
      s8_operands[0] = src->mutable_operand(0);

      // Handle pad value if operand is Pad
      if (operand->opcode() == HloOpcode::kPad && s8_operands.size() > 1) {
        HloInstruction* pad_val = s8_operands[1];
        if (pad_val->opcode() == HloOpcode::kConstant &&
            LiteralFitsInS8(pad_val->literal())) {
          auto s8_lit = pad_val->literal().Convert(S8);
          if (s8_lit.ok()) {
            s8_operands[1] = comp->AddInstruction(
                HloInstruction::CreateConstant(std::move(*s8_lit)));
          }
        } else if (pad_val->opcode() == HloOpcode::kConvert &&
                   pad_val->operand(0)->shape().element_type() == S8) {
          s8_operands[1] = pad_val->mutable_operand(0);
        } else {
          return operand;  // Cannot convert pad value to S8
        }
      }

      Shape s8_shape = ShapeUtil::ChangeElementType(operand->shape(), S8);
      HloInstruction* s8_op = comp->AddInstruction(
          operand->CloneWithNewOperands(s8_shape, s8_operands));
      return comp->AddInstruction(
          HloInstruction::CreateConvert(operand->shape(), s8_op));
    }
  }

  return operand;
}

// Pads convolution channel dimensions to multiples of 2 for 16-bit float
// (BF16/F16) convolutions so that they satisfy 32-bit alignment requirements
// for cuDNN runtime epilogue fusion.
absl::StatusOr<bool> PadConvolutionChannels(HloComputation* comp,
                                            HloInstruction* conv) {
  if (conv->operand_count() != 2) {
    return false;
  }
  if (conv->feature_group_count() > 1 || conv->batch_group_count() > 1) {
    return false;
  }

  HloInstruction* input = conv->mutable_operand(0);
  HloInstruction* filter = conv->mutable_operand(1);
  PrimitiveType input_type = input->shape().element_type();
  PrimitiveType filter_type = filter->shape().element_type();

  // 32-bit alignment requirement applies to 16-bit float types (BF16 and F16).
  if (input_type != BF16 && input_type != F16) {
    return false;
  }

  const auto& dnums = conv->convolution_dimension_numbers();
  int64_t in_feature_dim = dnums.input_feature_dimension();
  int64_t kernel_in_feature_dim = dnums.kernel_input_feature_dimension();
  int64_t kernel_out_feature_dim = dnums.kernel_output_feature_dimension();
  int64_t out_feature_dim = dnums.output_feature_dimension();

  int64_t in_channels = input->shape().dimensions(in_feature_dim);
  int64_t out_channels = conv->shape().dimensions(out_feature_dim);

  // Minimum alignment required by cuDNN runtime fusion for 16-bit floats is 2
  // elements (4 bytes / 32 bits).
  constexpr int64_t kAlignment = 2;
  int64_t padded_in_channels = RoundUpTo<int64_t>(in_channels, kAlignment);
  int64_t padded_out_channels = RoundUpTo<int64_t>(out_channels, kAlignment);

  if (padded_in_channels == in_channels &&
      padded_out_channels == out_channels) {
    return false;
  }

  HloInstruction* new_input = input;
  if (padded_in_channels > in_channels) {
    Shape padded_input_shape = input->shape();
    padded_input_shape.set_dimensions(in_feature_dim, padded_in_channels);
    PaddingConfig pad_config =
        MakeNoPaddingConfig(padded_input_shape.dimensions().size());
    pad_config.mutable_dimensions(in_feature_dim)
        ->set_edge_padding_high(padded_in_channels - in_channels);
    auto* zero = comp->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::Zero(input_type)));
    new_input = comp->AddInstruction(
        HloInstruction::CreatePad(padded_input_shape, input, zero, pad_config),
        &input->metadata());
  }

  HloInstruction* new_filter = filter;
  if (padded_in_channels > in_channels || padded_out_channels > out_channels) {
    Shape padded_filter_shape = filter->shape();
    PaddingConfig pad_config =
        MakeNoPaddingConfig(padded_filter_shape.dimensions().size());
    if (padded_in_channels > in_channels) {
      padded_filter_shape.set_dimensions(kernel_in_feature_dim,
                                         padded_in_channels);
      pad_config.mutable_dimensions(kernel_in_feature_dim)
          ->set_edge_padding_high(padded_in_channels - in_channels);
    }
    if (padded_out_channels > out_channels) {
      padded_filter_shape.set_dimensions(kernel_out_feature_dim,
                                         padded_out_channels);
      pad_config.mutable_dimensions(kernel_out_feature_dim)
          ->set_edge_padding_high(padded_out_channels - out_channels);
    }
    auto* zero = comp->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::Zero(filter_type)));
    new_filter =
        comp->AddInstruction(HloInstruction::CreatePad(
                                 padded_filter_shape, filter, zero, pad_config),
                             &filter->metadata());
  }

  Shape new_conv_shape = conv->shape();
  new_conv_shape.set_dimensions(out_feature_dim, padded_out_channels);
  HloInstruction* new_conv = comp->AddInstruction(
      conv->CloneWithNewOperands(new_conv_shape, {new_input, new_filter}));

  if (padded_out_channels > out_channels) {
    std::vector<int64_t> start_indices(new_conv_shape.dimensions().size(), 0);
    std::vector<int64_t> end_indices(new_conv_shape.dimensions().begin(),
                                     new_conv_shape.dimensions().end());
    end_indices[out_feature_dim] = out_channels;
    std::vector<int64_t> strides(new_conv_shape.dimensions().size(), 1);
    HloInstruction* sliced = comp->AddInstruction(
        HloInstruction::CreateSlice(conv->shape(), new_conv, start_indices,
                                    end_indices, strides),
        &conv->metadata());
    ABSL_RETURN_IF_ERROR(comp->ReplaceInstruction(conv, sliced));
  } else {
    ABSL_RETURN_IF_ERROR(comp->ReplaceInstruction(conv, new_conv));
  }

  return true;
}

}  // namespace

absl::StatusOr<bool> ConvCanonicalizer::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      if (instr->opcode() != HloOpcode::kConvolution) {
        continue;
      }

      for (int64_t i = 0; i < instr->operand_count(); ++i) {
        HloInstruction* operand = instr->mutable_operand(i);
        ABSL_ASSIGN_OR_RETURN(HloInstruction * new_operand,
                         CanonicalizeOperandToS8Convert(comp, operand));
        if (new_operand != operand) {
          ABSL_RETURN_IF_ERROR(instr->ReplaceOperandWith(i, new_operand));
          changed = true;
        }
      }

      ABSL_ASSIGN_OR_RETURN(bool padded, PadConvolutionChannels(comp, instr));
      changed |= padded;
    }
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
