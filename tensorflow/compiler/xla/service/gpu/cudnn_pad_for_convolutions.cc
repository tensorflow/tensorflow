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

#include "tensorflow/compiler/xla/service/gpu/cudnn_pad_for_convolutions.h"

#include <utility>

#include "absl/functional/bind_front.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_support_utils.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/stream_executor/device_description.h"
#include "tensorflow/stream_executor/dnn.h"

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
  for (int64_t dim = 0; dim < shape.rank(); ++dim) {
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
  new_operands.reserve(conv->operand_count());
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

  // Clone conv's name to new_conv.  This is safe because we're going to remove
  // conv below.
  new_conv->SetAndSanitizeName(conv->name());

  VLOG(2) << "Padded features of " << conv->ToString() << ", replaced with "
          << new_conv->ToString();

  // Slice the new conv result if necessary, keeping in mind that new_conv
  // has tuple shape (new_result_shape, u8[0]).
  if (!ShapeUtil::Equal(result_shape, new_result_shape)) {
    std::vector<int64_t> start_indices(result_shape.dimensions_size(), 0);
    std::vector<int64_t> end_indices(result_shape.dimensions().begin(),
                                     result_shape.dimensions().end());
    std::vector<int64_t> strides(result_shape.dimensions_size(), 1);

    auto* new_conv_result = add(
        HloInstruction::CreateGetTupleElement(new_result_shape, new_conv, 0));
    auto* empty_temp_buffer =
        add(HloInstruction::CreateConstant(LiteralUtil::CreateR1<uint8_t>({})));
    auto* sliced_result = add(HloInstruction::CreateSlice(
        result_shape, new_conv_result, start_indices, end_indices, strides));
    new_conv =
        add(HloInstruction::CreateTuple({sliced_result, empty_temp_buffer}));
  }

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

// This is the main function of the transform.  It runs on a given custom call
// nodes to cuDNN convolution, calls resolve_pad_shapes to resolve
// the desired input/output feature map shapes, and adds necessary padding and
// slicing nodes around them.
//
// resolve_pad_shapes takes conv, a custom call instruction to cuDNN convolution
// that may need padding to figure out the desired padded input and output
// tensor shapes and store the desired shapes in new_input_shapes and
// new_input_shapes.  Notice that new_input_shapes is a vector for multiple
// input tensors. This function shall return true if padding is necessary or
// false otherwise in addition to status.
static StatusOr<bool> ResolveAndPad(
    HloCustomCallInstruction* conv,
    std::function<StatusOr<bool>(HloCustomCallInstruction* conv,
                                 std::vector<Shape>* new_input_shapes,
                                 Shape* new_result_shape)>
        resolve_pad_shapes) {
  std::vector<Shape> new_input_shapes;
  Shape new_result_shape;
  TF_ASSIGN_OR_RETURN(bool result, resolve_pad_shapes(conv, &new_input_shapes,
                                                      &new_result_shape));
  if (result) {
    TF_RETURN_IF_ERROR(PadConv(conv, new_input_shapes, new_result_shape));
    return true;
  }
  return false;
}

// Adds padding to cudnn convolutions to make them run faster on GPUs with
// tensor cores.
//
//  - f16 convolutions are padded to have input/output channel dimensions that
//    are multiples of 8, so that we can use tensor cores.
//
//  - f16 convolutions with 3 input channels and 32 or 64 output channels are
//    padded to 4 input channels.  There's a special-cased cudnn algorithm just
//    for this.
//
// Don't run this pass on GPUs without tensor cores -- it will make them slower!
//
// TODO(jlebar): Also pad dots.
static StatusOr<bool> TryResolvePaddedShapesForTensorCore(
    HloCustomCallInstruction* conv, std::vector<Shape>* new_input_shapes_ptr,
    Shape* new_result_shape_ptr) {
  TF_ASSIGN_OR_RETURN(auto kind, GetCudnnConvKind(conv));
  const auto& dnums = conv->convolution_dimension_numbers();
  auto* lhs = conv->mutable_operand(0);
  auto* rhs = conv->mutable_operand(1);
  const Shape& result_shape = conv->shape().tuple_shapes(0);

  // Nothing to do on non-f16 convolutions.
  if (result_shape.element_type() != PrimitiveType::F16) {
    return false;
  }

  // TODO(timshen): Don't skip forward-activation convs if we find a benchmark
  // where there's a speedup.
  if (kind == CudnnConvKind::kForwardActivation) {
    return false;
  }

  Shape new_lhs_shape = lhs->shape();
  Shape new_rhs_shape = rhs->shape();
  Shape& new_result_shape = *new_result_shape_ptr;
  new_result_shape = conv->shape().tuple_shapes(0);

  // new_{input,filter_output}_shape points to the appropriate one of
  // new_{lhs,rhs,result}_shape.
  Shape* new_input_shape;
  Shape* new_filter_shape;
  Shape* new_output_shape;
  std::tie(new_input_shape, new_filter_shape, new_output_shape) = [&] {
    switch (kind) {
      case CudnnConvKind::kForward:
      case CudnnConvKind::kForwardActivation:
        return std::make_tuple(&new_lhs_shape, &new_rhs_shape,
                               &new_result_shape);
      case CudnnConvKind::kBackwardInput:
        return std::make_tuple(&new_result_shape, &new_rhs_shape,
                               &new_lhs_shape);
      case CudnnConvKind::kBackwardFilter:
        return std::make_tuple(&new_lhs_shape, &new_result_shape,
                               &new_rhs_shape);
    }
  }();

  // If there are 3 input features and 32 or 64 output features, pad the input
  // features to 4.  Otherwise, try padding to multiples of 8 and check that
  // this doesn't make any of the conv buffers too much larger.
  auto input_features =
      new_input_shape->dimensions(dnums.input_feature_dimension());
  auto output_features =
      new_output_shape->dimensions(dnums.output_feature_dimension());
  if (input_features == 3 && (output_features == 32 || output_features == 64)) {
    new_input_shape->set_dimensions(dnums.input_feature_dimension(), 4);
    new_filter_shape->set_dimensions(dnums.kernel_input_feature_dimension(), 4);
  } else {
    auto pad_dim = [](Shape* s, int64_t dim) {
      s->set_dimensions(dim, RoundUpTo<int64_t>(s->dimensions(dim), 8));
    };
    pad_dim(new_input_shape, dnums.input_feature_dimension());
    pad_dim(new_filter_shape, dnums.kernel_input_feature_dimension());
    pad_dim(new_filter_shape, dnums.kernel_output_feature_dimension());
    pad_dim(new_output_shape, dnums.output_feature_dimension());

    // We won't pad a conv if doing so increases the total number of bytes in
    // the lhs, rhs, or result by more than this amount.
    //
    // TODO(jlebar): This number was tuned experimentally.  It represents a
    // compromise on our current benchmarks; it speeds some up significantly,
    // and doesn't slow any down.  But we can observe by changing this value
    // that there's additional room for speedups.  Achieving those speedups
    // without also slowing other things down will likely require a more
    // sophisticated heuristic, possibly some form of auto-tuning.
    static constexpr double kMaxBytesTouchedBound = 1.35;

    // Check that padding wouldn't increase the total bytes read/written by this
    // operation too much.
    auto check_size_increase = [&](const Shape& old_shape,
                                   const Shape& new_shape) {
      int64_t old_bytes = ShapeUtil::ByteSizeOf(old_shape);
      int64_t new_bytes = ShapeUtil::ByteSizeOf(new_shape);
      if (new_bytes <= old_bytes * kMaxBytesTouchedBound) {
        return true;
      }
      VLOG(3)
          << "Not padding convolution; doing so would change input / result "
             "shape from "
          << ShapeUtil::HumanString(old_shape) << " to "
          << ShapeUtil::HumanString(new_shape) << ", a size increase of "
          << new_bytes / static_cast<double>(old_bytes) << "x > "
          << kMaxBytesTouchedBound << "x: " << conv->ToString();
      return false;
    };

    if (!check_size_increase(lhs->shape(), new_lhs_shape) ||
        !check_size_increase(rhs->shape(), new_rhs_shape) ||
        !check_size_increase(result_shape, new_result_shape)) {
      return false;
    }
  }

  if (ShapeUtil::Equal(lhs->shape(), new_lhs_shape) &&
      ShapeUtil::Equal(rhs->shape(), new_rhs_shape)) {
    VLOG(3) << "No need to pad features of " << conv->ToString();
    return false;
  }

  new_input_shapes_ptr->push_back(new_lhs_shape);
  new_input_shapes_ptr->push_back(new_rhs_shape);
  return true;
}

// Adds padding to cudnn integer convolutions to make input and output feature
// maps multiples of pad_to (usually 4 or 32).
StatusOr<bool> TryResolvePaddedShapesForIntegerConvolution(
    int pad_to, const se::CudaComputeCapability& compute_capability,
    HloCustomCallInstruction* conv, std::vector<Shape>* new_input_shapes_ptr,
    Shape* new_result_shape_ptr) {
  TF_ASSIGN_OR_RETURN(auto kind, GetCudnnConvKind(conv));
  const Shape& input_shape = conv->operand(0)->shape();
  const Shape& kernel_shape = conv->operand(1)->shape();
  const Shape& result_shape = conv->shape().tuple_shapes(0);

  // Integer convolution only
  if (!primitive_util::IsIntegralType(input_shape.element_type())) {
    return false;
  }

  // kForward and kForwardActivation only
  if (kind != CudnnConvKind::kForward &&
      kind != CudnnConvKind::kForwardActivation) {
    return false;
  }

  const auto& dnums = conv->convolution_dimension_numbers();
  std::vector<Shape>& new_input_shapes = *new_input_shapes_ptr;
  for (auto operand : conv->operands()) {
    new_input_shapes.push_back(operand->shape());
  }
  Shape& new_result_shape = *new_result_shape_ptr;
  new_result_shape = conv->shape().tuple_shapes(0);

  // The input/kernel/output might already be vectorized (i.e. cudnn layout
  // NCHW_VECT_C).  If so, we pad the features dim so that
  // size(features_dim) * size(vect_dim) is a multiple of pad_to.
  std::optional<int64_t> input_vect_dim;
  std::optional<int64_t> kernel_vect_dim;
  std::optional<int64_t> result_vect_dim;
  std::tie(input_vect_dim, kernel_vect_dim, result_vect_dim) =
      FindVectorizedFeatureDims(dnums, input_shape, kernel_shape, result_shape);

  int64_t input_vect_size =
      input_vect_dim.has_value() ? input_shape.dimensions(*input_vect_dim) : 1;
  int64_t kernel_vect_size = kernel_vect_dim.has_value()
                                 ? kernel_shape.dimensions(*kernel_vect_dim)
                                 : 1;
  int64_t result_vect_size = result_vect_dim.has_value()
                                 ? result_shape.dimensions(*result_vect_dim)
                                 : 1;
  if (pad_to % input_vect_size != 0 || pad_to % kernel_vect_size != 0 ||
      pad_to % result_vect_size != 0) {
    // If the conv is already vectorized but pad_to is not a multiple of the
    // vector size, we choose not to pad.  This is a weird case, because the
    // only useful vector sizes in cudnn (as of writing) are 4 and 32, and those
    // are also the only pad_to cases.
    return false;
  }

  // Check that cudnn support our desired integer padding/vectorization.
  TF_ASSIGN_OR_RETURN(bool cudnn_supports,
                      CudnnSupportsOptimizedIntegerConvolution(
                          compute_capability, *conv, pad_to));
  if (!cudnn_supports) {
    return false;
  }

  // Pad the features to multiples of pad_to.
  {
    auto pad_dim = [&](Shape* s, int64_t dim, int64_t cur_vect_size) {
      CHECK_EQ(pad_to % cur_vect_size, 0);
      s->set_dimensions(
          dim, RoundUpTo<int64_t>(s->dimensions(dim), pad_to / cur_vect_size));
    };

    switch (kind) {
      case CudnnConvKind::kForward:
        CHECK_EQ(new_input_shapes.size(), 2);
        // Input feature maps
        pad_dim(&new_input_shapes[0], dnums.input_feature_dimension(),
                input_vect_size);
        // Kernel for the input feature maps
        pad_dim(&new_input_shapes[1], dnums.kernel_input_feature_dimension(),
                kernel_vect_size);
        // Kernel for the output feature maps.  In the NCHW_VECT_C, only the
        // kernel input feature dim is vectorized, so this has cur_vect_size 1.
        pad_dim(&new_input_shapes[1], dnums.kernel_output_feature_dimension(),
                /*cur_vect_size=*/1);
        // Output feature maps
        pad_dim(&new_result_shape, dnums.output_feature_dimension(),
                result_vect_size);
        break;
      case CudnnConvKind::kForwardActivation:
        CHECK(new_input_shapes.size() == 3 || new_input_shapes.size() == 4);
        // Input feature maps
        pad_dim(&new_input_shapes[0], dnums.input_feature_dimension(),
                input_vect_size);
        // Kernel for the input feature maps
        pad_dim(&new_input_shapes[1], dnums.kernel_input_feature_dimension(),
                kernel_vect_size);
        // Kernel for the output feature maps.  In the NCHW_VECT_C, only the
        // kernel input feature dim is vectorized, so this has cur_vect_size 1.
        pad_dim(&new_input_shapes[1], dnums.kernel_output_feature_dimension(),
                /*cur_vect_size=*/1);

        // Bias.  This ia 1D vector of length output-depth, and it's unclear if
        // we *have* to pad it.  But hey, we might as well.  cur_vect_size 1
        // because NCHW_VECT_C doesn't apply here (there is no channels
        // dimension!).
        pad_dim(&new_input_shapes[2], /*dim=*/0, /*cur_vect_size=*/1);

        if (new_input_shapes.size() == 4) {
          // Optional side input.  Same layout as result, so gets padded the
          // same.
          pad_dim(&new_input_shapes[3], dnums.output_feature_dimension(),
                  result_vect_size);
        }
        // Output feature maps
        pad_dim(&new_result_shape, dnums.output_feature_dimension(),
                result_vect_size);
        break;
      default:
        CHECK(false);
    }

    // We won't pad a conv if doing so increases the total number of bytes in
    // the lhs, rhs, or result by a factor of this much or more.
    //
    // Note: It's important that this bound is exclusive.  It's a performance
    // regression to pad and increase input/output size by 2x, so we only pad
    // strictly less than 2x.
    //
    // TODO(jlebar): This number was tuned experimentally, but without much
    // experimental evidence.
    static constexpr double kMaxBytesTouchedBound = 2;

    // Check that padding wouldn't increase the total bytes read/written by this
    // operation too much.
    auto check_size_increase = [&](const Shape& old_shape,
                                   const Shape& new_shape) {
      int64_t old_bytes = ShapeUtil::ByteSizeOf(old_shape);
      int64_t new_bytes = ShapeUtil::ByteSizeOf(new_shape);
      if (new_bytes < old_bytes * kMaxBytesTouchedBound) {
        return true;
      }
      VLOG(3)
          << "Not padding convolution; doing so would change input / result "
             "shape from "
          << ShapeUtil::HumanString(old_shape) << " to "
          << ShapeUtil::HumanString(new_shape) << ", a size increase of "
          << new_bytes / static_cast<double>(old_bytes)
          << "x >= " << kMaxBytesTouchedBound << "x: " << conv->ToString();
      return false;
    };

    // Check size increase only on the input and output.  No need to check the
    // filter, since that's determined by the input/output.  The bias (if
    // present) is tiny (1D array of length output-depth), so padding doesn't
    // matter.  And the side-input, if present, is the same shape as the input.
    if (!check_size_increase(conv->operand(0)->shape(), new_input_shapes[0]) ||
        !check_size_increase(result_shape, new_result_shape)) {
      return false;
    }
  }

  bool changed = false;
  for (int64_t i = 0; i < conv->operand_count(); ++i) {
    changed |=
        !ShapeUtil::Equal(conv->operand(i)->shape(), new_input_shapes[i]);
  }
  if (!changed) {
    VLOG(3) << "No need to pad features of " << conv->ToString();
  }

  return changed;
}

StatusOr<bool> CudnnPadForConvolutions::Run(HloModule* module) {
  bool changed = false;
  for (HloComputation* comp : module->MakeNonfusionComputations()) {
    for (HloCustomCallInstruction* conv : GetRelevantConvs(comp)) {
      // On Turing and later (sm75+), pad to multiples of 32 bytes if possible,
      // because that lets us use the fast int8x32 data type.
      bool local_changed = false;
      if (compute_capability_.IsAtLeast(7, 5)) {
        TF_ASSIGN_OR_RETURN(
            local_changed,
            ResolveAndPad(conv, absl::bind_front(
                                    TryResolvePaddedShapesForIntegerConvolution,
                                    32, compute_capability_)));
      }
      if (!local_changed) {
        TF_ASSIGN_OR_RETURN(
            local_changed,
            ResolveAndPad(conv, absl::bind_front(
                                    TryResolvePaddedShapesForIntegerConvolution,
                                    4, compute_capability_)));
      }
      changed |= local_changed;
    }
    if (compute_capability_.IsAtLeast(se::CudaComputeCapability::VOLTA)) {
      for (HloCustomCallInstruction* conv : GetRelevantConvs(comp)) {
        TF_ASSIGN_OR_RETURN(
            bool local_changed,
            ResolveAndPad(conv, TryResolvePaddedShapesForTensorCore));
        changed |= local_changed;
      }
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
