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

#include "tensorflow/compiler/xla/service/gpu/pad_for_tensor_cores.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace gpu {


// We want the input/output feature counts of an f16 conv to be factors of 8,
// because without this cudnn can't use tensor cores on the conv.
static constexpr int64 kDesiredNumFeaturesFactor = 8;

// We won't pad a conv if doing so increases the total number of bytes in the
// lhs, rhs, or result by more than this amount.
//
// TODO(jlebar): This number was tuned experimentally.  It represents a
// compromise on our current benchmarks; it speeds some up significantly, and
// doesn't slow any down.  But we can observe by changing this value that
// there's additional room for speedups.  Achieving those speedups without also
// slowing other things down will likely require a more sophisticated heuristic,
// possibly some form of auto-tuning.
static constexpr double kMaxBytesTouchedIncrease = 1.2;

// Pads the given dimensions in the given shape up to a multiple of
// kDesiredNumFeaturesFactor.
static Shape PadShape(Shape s, absl::Span<const int64> dims) {
  for (int64 dim : dims) {
    int64 dim_to_pad_size = s.dimensions(dim);
    int64 new_dim_to_pad_size =
        RoundUpToNearest(dim_to_pad_size, kDesiredNumFeaturesFactor);
    s.set_dimensions(dim, new_dim_to_pad_size);
  }
  return s;
}

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
  auto* zero = comp->AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::Zero(shape.element_type()).CloneToUnique()));

  PaddingConfig pad_config = MakeNoPaddingConfig(ShapeUtil::Rank(shape));

  bool added_padding = false;
  for (int64 dim = 0; dim < ShapeUtil::Rank(shape); ++dim) {
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

// Pads the input/output feature dimensions of the given cudnn convolution
// custom-call to be multiples of kDesiredNumFeaturesFactor.
static StatusOr<bool> PadFeaturesDims(HloInstruction* conv) {
  CHECK_EQ(0, conv->shape().tuple_shapes(1).dimensions(0))
      << "conv must use 0 scratch bytes, i.e. this pass must be run "
         "before CudnnConvolutionAlgorithmPicker.";

  const auto& target = conv->custom_call_target();
  const auto& dnums = conv->convolution_dimension_numbers();
  auto* lhs = conv->mutable_operand(0);
  auto* rhs = conv->mutable_operand(1);
  const Shape& result_shape = conv->shape().tuple_shapes(0);

  Shape new_lhs_shape = [&] {
    if (target == kCudnnConvForwardCallTarget ||
        target == kCudnnConvBackwardFilterCallTarget) {
      // LHS is "input".
      return PadShape(lhs->shape(), {dnums.input_feature_dimension()});
    }
    CHECK_EQ(target, kCudnnConvBackwardInputCallTarget);
    // LHS is "output".
    return PadShape(lhs->shape(), {dnums.output_feature_dimension()});
  }();

  Shape new_rhs_shape = [&] {
    if (target == kCudnnConvForwardCallTarget ||
        target == kCudnnConvBackwardInputCallTarget) {
      // RHS is "filter".
      return PadShape(rhs->shape(), {dnums.kernel_input_feature_dimension(),
                                     dnums.kernel_output_feature_dimension()});
    }
    CHECK_EQ(target, kCudnnConvBackwardFilterCallTarget);
    // RHS is "output".
    return PadShape(rhs->shape(), {dnums.output_feature_dimension()});
  }();

  if (ShapeUtil::Equal(lhs->shape(), new_lhs_shape) &&
      ShapeUtil::Equal(rhs->shape(), new_rhs_shape)) {
    VLOG(3) << "No need to pad features of " << conv->ToString();
    return false;
  }

  Shape new_result_shape = [&] {
    if (target == kCudnnConvForwardCallTarget) {
      // Result is "output".
      return PadShape(result_shape, {dnums.output_feature_dimension()});
    }
    if (target == kCudnnConvBackwardInputCallTarget) {
      // Result is "input".
      return PadShape(result_shape, {dnums.input_feature_dimension()});
    }
    CHECK_EQ(target, kCudnnConvBackwardFilterCallTarget);
    // Result is "filter".
    return PadShape(result_shape, {dnums.kernel_input_feature_dimension(),
                                   dnums.kernel_output_feature_dimension()});
  }();

  // Check that padding wouldn't increase the total bytes read/written by this
  // operation too much.
  auto check_size_increase = [&](const Shape& old_shape,
                                 const Shape& new_shape) {
    int64 old_bytes = ShapeUtil::ByteSizeOf(old_shape);
    int64 new_bytes = ShapeUtil::ByteSizeOf(new_shape);
    if (new_bytes <= old_bytes * kMaxBytesTouchedIncrease) {
      return true;
    }
    VLOG(3) << "Not padding convolution; doing so would change input / result "
               "shape from "
            << ShapeUtil::HumanString(old_shape) << " to "
            << ShapeUtil::HumanString(new_shape) << ", a size increase of "
            << new_bytes / static_cast<double>(old_bytes) << "x > "
            << kMaxBytesTouchedIncrease << "x: " << conv->ToString();
    return false;
  };
  if (!check_size_increase(lhs->shape(), new_lhs_shape) ||
      !check_size_increase(rhs->shape(), new_rhs_shape) ||
      !check_size_increase(result_shape, new_result_shape)) {
    return false;
  }

  // OK, let's do the transformation!

  auto* new_lhs = PadInstruction(lhs, new_lhs_shape);
  auto* new_rhs = PadInstruction(rhs, new_rhs_shape);
  CHECK(new_lhs != lhs || new_rhs != rhs)
      << "We should have had to pad either LHS or RHS.";

  auto add = [&](std::unique_ptr<HloInstruction> new_instr) {
    return conv->parent()->AddInstruction(std::move(new_instr));
  };

  Shape new_conv_shape = ShapeUtil::MakeTupleShape(
      {new_result_shape, ShapeUtil::MakeShape(U8, {0})});
  auto* new_conv =
      add(conv->CloneWithNewOperands(new_conv_shape, {new_lhs, new_rhs}));

  // Slice the new conv result if necessary, keeping in mind that new_conv has
  // tuple shape (new_result_shape, u8[0]).
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
  TF_RETURN_IF_ERROR(conv->parent()->ReplaceInstruction(conv, new_conv));
  return true;
}

static std::vector<HloInstruction*> GetRelevantConvs(HloComputation* comp) {
  std::vector<HloInstruction*> convs;
  for (HloInstruction* instr : comp->instructions()) {
    if (IsCustomCallToDnnConvolution(*instr) &&
        instr->operand(0)->shape().element_type() == F16) {
      convs.push_back(instr);
    }
  }
  return convs;
}

StatusOr<bool> PadForTensorCores::Run(HloModule* module) {
  bool changed = false;
  for (HloComputation* comp : module->MakeNonfusionComputations()) {
    for (HloInstruction* conv : GetRelevantConvs(comp)) {
      TF_ASSIGN_OR_RETURN(bool result, PadFeaturesDims(conv));
      changed |= result;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
