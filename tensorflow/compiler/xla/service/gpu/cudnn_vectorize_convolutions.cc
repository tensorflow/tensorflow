/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/cudnn_vectorize_convolutions.h"

#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {
namespace gpu {

// Finds convolutions that this pass may be able to transform, namely int8 cudnn
// forward or forward-bias-activation convolutions
//
// cudnn as of v8.2 supports the following data type combinations for forward
// and forward-bias-activation convolutions.  We have to make sure we only
// vectorize to one of these supported configs.
//
//   in       out
//   int8x1   int8x1
//   int8x1   float
//   int8x1   int32
//
//   int8x4   int8x4
//   int8x4   float
//
//   int8x32  int8x32
//
// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionForward
//
// For now we restrict ourselves to only the int8xN -> int8xN cases.  We could
// allow the int8x4 -> float case in the future if desireable.
static std::vector<HloCustomCallInstruction*> GetRelevantConvs(
    HloComputation* comp) {
  std::vector<HloCustomCallInstruction*> convs;
  for (HloInstruction* instr : comp->instructions()) {
    if (instr->opcode() != HloOpcode::kCustomCall ||
        (instr->custom_call_target() != kCudnnConvForwardCallTarget &&
         instr->custom_call_target() !=
             kCudnnConvBiasActivationForwardCallTarget) ||
        instr->operand_count() < 2) {
      continue;
    }

    PrimitiveType input_ty = instr->operand(0)->shape().element_type();
    PrimitiveType output_ty = instr->shape().tuple_shapes(0).element_type();
    if (input_ty == output_ty && (input_ty == S8 || input_ty == U8)) {
      convs.push_back(Cast<HloCustomCallInstruction>(instr));
    }
  }
  return convs;
}

// Reshapes `shape` so that there's an extra dimension of size `vect_size` right
// after `dim`.
//
// For example given shape=s8[10, 32, 20], dim=1, vect_size=4, returns
// s8[10, 8, 4, 20].
static Shape SplitShapeAtDim(Shape shape, int64_t dim, int64_t vect_size) {
  absl::InlinedVector<int64, 5> new_dims(shape.dimensions().begin(),
                                         shape.dimensions().end());
  CHECK_EQ(new_dims[dim] % vect_size, 0);
  new_dims[dim] /= vect_size;
  new_dims.insert(new_dims.begin() + dim + 1, vect_size);
  return ShapeUtil::MakeShape(shape.element_type(), new_dims);
}

// Reshapes `instr` so that it has an extra dimension of size `vect_size` right
// after `dim`.
static HloInstruction* SplitInstrAtDim(HloInstruction* instr, int64_t dim,
                                       int64_t vect_size) {
  return instr->parent()->AddInstruction(HloInstruction::CreateReshape(
      SplitShapeAtDim(instr->shape(), dim, vect_size), instr));
}

// Reshapes `instr` so that dimension `dim` is collapsed into the dimension
// right before it.
static HloInstruction* CollapseDimIntoPrev(HloInstruction* instr, int64_t dim) {
  CHECK_GT(dim, 0);
  absl::InlinedVector<int64, 5> new_dims(instr->shape().dimensions().begin(),
                                         instr->shape().dimensions().end());
  new_dims[dim - 1] *= new_dims[dim];
  new_dims.erase(new_dims.begin() + dim);

  return instr->parent()->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(instr->shape().element_type(), new_dims), instr));
}

// Reshapes instr so that dimension `vect_dim` has size `vect_size`.  If we have
// to grow `vect_dim`, we steal elements from `dim`.  If we have to shrink it,
// we move elements into `dim`.
//
// Requires that this is possible without merging and re-splitting the two
// dimensions.  I.e. there should be some amount of dim or vect_dim that we can
// "split off" and add to the other to get vect_dim to have size vect_size.
static HloInstruction* RevectorizeInstr(HloInstruction* instr, int64_t dim,
                                        int64_t vect_dim, int64_t vect_size) {
  HloComputation* computation = instr->parent();
  const Shape& shape = instr->shape();
  PrimitiveType elem_ty = shape.element_type();
  auto size = [&](int64_t d) { return shape.dimensions(d); };

  CHECK_EQ(size(dim) * size(vect_dim) % vect_size, 0);

  // WLOG let vect_dim be the dimension we're shrinking.
  if (size(vect_dim) > vect_size) {
    // We want vect_dim to have size vect_size.  That means dim must have the
    // rest of the elements, namely size(dim) * size(vect_dim) / vect_size.
    return RevectorizeInstr(instr, vect_dim, dim,
                            size(dim) * size(vect_dim) / vect_size);
  }

  CHECK_EQ(vect_size % size(vect_dim), 0);
  int64_t split_factor = vect_size / size(vect_dim);
  CHECK_EQ(size(dim) % split_factor, 0);

  absl::InlinedVector<int64, 6> new_dims(shape.dimensions().begin(),
                                         shape.dimensions().end());
  new_dims[dim] /= split_factor;
  new_dims.insert(new_dims.begin() + dim + 1, split_factor);
  instr = computation->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(elem_ty, new_dims), instr));

  // Transpose the new dimension so it's adjacent to vect_dim.
  new_dims.erase(new_dims.begin() + dim + 1);
  new_dims.insert(new_dims.begin() + vect_dim + 1, split_factor);

  absl::InlinedVector<int64, 6> transpose_idxs(new_dims.size());
  absl::c_iota(transpose_idxs, 0);
  transpose_idxs.erase(transpose_idxs.begin() + dim + 1);
  transpose_idxs.insert(transpose_idxs.begin() + vect_dim + 1, dim + 1);
  instr = computation->AddInstruction(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(elem_ty, new_dims), instr, transpose_idxs));

  new_dims.erase(new_dims.begin() + vect_dim + 1);
  new_dims[vect_dim] *= split_factor;
  return computation->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(elem_ty, new_dims), instr));
}

// Adds a vectorized-feature dimension to dnums right after the current feature
// dimension.
//
// ConvolutionDimensionNumbers doesn't represent the vectorized-feature
// dimension explicitly, because the whole concept of a vectorized-feature
// dimension is specific to cudnn.  Rather, the vectorized-feature dimension is
// implicit; it's the first dimension that *doesn't* appear in the dnums.
//
// This function "makes room" in dnums for the new vectorized dimension by
// incrementing any dimensions which appear after the feature dim.  The implicit
// vector dim is then in this "empty" spot.
static ConvolutionDimensionNumbers VectorizeDnums(
    ConvolutionDimensionNumbers dnums) {
  int64_t input_vect_dim = dnums.input_feature_dimension();
  if (dnums.input_batch_dimension() > input_vect_dim) {
    dnums.set_input_batch_dimension(dnums.input_batch_dimension() + 1);
  }
  for (int64& d : *dnums.mutable_input_spatial_dimensions()) {
    if (d > input_vect_dim) {
      ++d;
    }
  }

  int64_t kernel_vect_dim = dnums.kernel_input_feature_dimension();
  if (dnums.kernel_output_feature_dimension() > kernel_vect_dim) {
    dnums.set_kernel_output_feature_dimension(
        dnums.kernel_output_feature_dimension() + 1);
  }
  for (int64& d : *dnums.mutable_kernel_spatial_dimensions()) {
    if (d > kernel_vect_dim) {
      ++d;
    }
  }

  int64_t output_vect_dim = dnums.output_feature_dimension();
  if (dnums.output_batch_dimension() > output_vect_dim) {
    dnums.set_output_batch_dimension(dnums.output_batch_dimension() + 1);
  }
  for (int64& d : *dnums.mutable_output_spatial_dimensions()) {
    if (d > output_vect_dim) {
      ++d;
    }
  }

  return dnums;
}

// Tries to vectorize an already-vectorized convolution.
//
// That is, given a convolution of shape [N, C/k, H, W, k], changes it to have
// shape [N, C/vect_size, H, W, vect_size].  Similarly changes the filter from
// [H, W, I/k, O] to [H, W, I/vect_size, vect_size, O].
//
// (The dimensions can appear in any order; which is N/C/etc is determined by
// the convolutions' dnums.)
static StatusOr<bool> TryRevectorizeConv(HloInstruction* conv, int vect_size) {
  HloComputation* comp = conv->parent();
  const Shape& input_shape = conv->operand(0)->shape();
  const Shape& kernel_shape = conv->operand(1)->shape();
  const Shape& output_shape = conv->shape().tuple_shapes(0);
  const auto& dnums = conv->convolution_dimension_numbers();

  // Find the vectorized-features dim in the input/kernel/output.
  absl::optional<int64> input_vect_dim;
  absl::optional<int64> kernel_vect_dim;
  absl::optional<int64> output_vect_dim;
  std::tie(input_vect_dim, kernel_vect_dim, output_vect_dim) =
      FindVectorizedFeatureDims(dnums, input_shape, kernel_shape, output_shape);

  if (!input_vect_dim.has_value() || !kernel_vect_dim.has_value() ||
      !output_vect_dim.has_value()) {
    return false;
  }

  int64_t input_feat_size =
      input_shape.dimensions(dnums.input_feature_dimension());
  int64_t output_feat_size =
      output_shape.dimensions(dnums.output_feature_dimension());
  int64_t input_vect_size = input_shape.dimensions(*input_vect_dim);
  int64_t output_vect_size = output_shape.dimensions(*output_vect_dim);
  if (vect_size % input_vect_size != 0 || vect_size % output_vect_size != 0 ||
      input_feat_size % (vect_size / input_vect_size) != 0 ||
      output_feat_size % (vect_size / output_vect_size) != 0) {
    return false;
  }

  VLOG(1) << "Re-vectorizing conv channels from "
          << input_shape.dimensions(*input_vect_dim) << " to " << vect_size
          << ": " << conv->ToString();

  absl::InlinedVector<HloInstruction*, 3> new_operands = {
      RevectorizeInstr(conv->mutable_operand(0),
                       dnums.input_feature_dimension(), *input_vect_dim,
                       vect_size),
      RevectorizeInstr(conv->mutable_operand(1),
                       dnums.kernel_input_feature_dimension(), *kernel_vect_dim,
                       vect_size),
  };
  if (conv->operand_count() > 2) {
    // Bias, if present.  This is passed through unmodified.
    new_operands.push_back(conv->mutable_operand(2));
  }
  if (conv->operand_count() > 3) {
    // Handle side input, which has same shape as the input.
    new_operands.push_back(RevectorizeInstr(conv->mutable_operand(3),
                                            dnums.input_feature_dimension(),
                                            *input_vect_dim, vect_size));
  }
  if (conv->operand_count() > 4) {
    return InvalidArgument(
        "Don't understand a conv with more than 4 arguments: %s",
        conv->ToString());
  }

  // The custom-call returns a tuple (new_output_shape, u8[0]), where the second
  // value in the tuple represents the convolution's scratch memory.
  absl::InlinedVector<int64, 5> new_output_dims(
      output_shape.dimensions().begin(), output_shape.dimensions().end());
  new_output_dims[dnums.output_feature_dimension()] /=
      (vect_size / output_vect_size);
  new_output_dims[*output_vect_dim] = vect_size;
  Shape new_output_shape =
      ShapeUtil::MakeShape(output_shape.element_type(), new_output_dims);
  HloInstruction* new_conv = comp->AddInstruction(conv->CloneWithNewOperands(
      ShapeUtil::MakeTupleShape(
          {new_output_shape, ShapeUtil::MakeShape(U8, {0})}),
      new_operands));
  new_conv->set_convolution_dimension_numbers(dnums);

  VLOG(1) << "Re-vectorized conv to " << new_conv->ToString();

  HloInstruction* new_conv_result = comp->AddInstruction(
      HloInstruction::CreateGetTupleElement(new_output_shape, new_conv, 0));
  HloInstruction* new_conv_scratch =
      comp->AddInstruction(HloInstruction::CreateGetTupleElement(
          ShapeUtil::MakeShape(U8, {0}), new_conv, 1));

  // Reshape back to the original shape.
  HloInstruction* conv_result_unrevectorized = RevectorizeInstr(
      new_conv_result, dnums.output_feature_dimension(), *output_vect_dim,
      /*orig output vector size*/ output_shape.dimensions(*output_vect_dim));

  // Create a tuple and replace the old conv with it!
  TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
      conv, HloInstruction::CreateTuple(
                {conv_result_unrevectorized, new_conv_scratch})));
  return true;
}

// Tries to vectorize a convolution.
//
// Given a convolution of dimensions [N, C, H, W], tries to convert it to have
// shape [N, C/vect_size, H, W, vect_size].  Similarly, given a kernel of shape
// [H, W, I, O], tries to conver it to [H, W, I/vect_size, vect_size, O].
//
// This requires that C be a multiple of vect_size.  CudnnPadForConvolutions can
// add padding to make this true.
static StatusOr<bool> TryVectorizeConv(HloInstruction* conv,
                                       int64_t vect_size) {
  HloComputation* comp = conv->parent();
  const Shape& input_shape = conv->operand(0)->shape();
  const Shape& output_shape = conv->shape().tuple_shapes(0);
  const auto& dnums = conv->convolution_dimension_numbers();
  int64_t in_channels = input_shape.dimensions(dnums.input_feature_dimension());
  int64_t out_channels =
      output_shape.dimensions(dnums.output_feature_dimension());

  if (in_channels % vect_size != 0 || out_channels % vect_size != 0) {
    return false;
  }

  if (input_shape.dimensions_size() >
      2 + dnums.input_spatial_dimensions_size()) {
    // Conv already has an extra dimension, which we assume is the vectorized
    // features dim.
    return false;
  }

  VLOG(1) << "Vectorizing conv channels by " << vect_size << ": "
          << conv->ToString();

  absl::InlinedVector<HloInstruction*, 3> new_operands = {
      SplitInstrAtDim(conv->mutable_operand(0), dnums.input_feature_dimension(),
                      vect_size),
      SplitInstrAtDim(conv->mutable_operand(1),
                      dnums.kernel_input_feature_dimension(), vect_size),
  };
  if (conv->operand_count() > 2) {
    // Bias, if present.  This is passed through unmodified.
    new_operands.push_back(conv->mutable_operand(2));
  }
  if (conv->operand_count() > 3) {
    // Handle side input, which has same shape as the input.
    new_operands.push_back(SplitInstrAtDim(
        conv->mutable_operand(3), dnums.input_feature_dimension(), vect_size));
  }
  if (conv->operand_count() > 4) {
    return InvalidArgument(
        "Don't understand a conv with more than 4 arguments: %s",
        conv->ToString());
  }

  // The custom-call returns a tuple (new_output_shape, u8[0]), where the second
  // value in the tuple represents the convolution's scratch memory.
  Shape new_output_shape = SplitShapeAtDim(
      output_shape, dnums.output_feature_dimension(), vect_size);
  HloInstruction* new_conv = comp->AddInstruction(conv->CloneWithNewOperands(
      ShapeUtil::MakeTupleShape(
          {new_output_shape, ShapeUtil::MakeShape(U8, {0})}),
      new_operands));
  new_conv->set_convolution_dimension_numbers(VectorizeDnums(dnums));

  VLOG(1) << "Vectorized conv to: " << new_conv->ToString();

  HloInstruction* new_conv_result = comp->AddInstruction(
      HloInstruction::CreateGetTupleElement(new_output_shape, new_conv, 0));
  HloInstruction* new_conv_scratch =
      comp->AddInstruction(HloInstruction::CreateGetTupleElement(
          ShapeUtil::MakeShape(U8, {0}), new_conv, 1));

  // Reshape back to the original shape.
  HloInstruction* conv_result_collapsed = CollapseDimIntoPrev(
      new_conv_result, dnums.output_feature_dimension() + 1);

  // Create a tuple and replace the old conv with it!
  TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
      conv,
      HloInstruction::CreateTuple({conv_result_collapsed, new_conv_scratch})));

  return true;
}

StatusOr<bool> CudnnVectorizeConvolutions::Run(HloModule* module) {
  bool changed = false;
  for (HloComputation* comp : module->MakeNonfusionComputations()) {
    for (HloCustomCallInstruction* conv : GetRelevantConvs(comp)) {
      // Try to (re)vectorize to int8x32 if this is an sm75+ GPU.  If we can't,
      // fall back to int8x4.
      bool local_changed = false;
      if (compute_capability_.IsAtLeast(7, 5)) {
        TF_ASSIGN_OR_RETURN(local_changed, TryRevectorizeConv(conv, 32));
        if (!local_changed) {
          TF_ASSIGN_OR_RETURN(local_changed, TryVectorizeConv(conv, 32));
        }
      }
      if (!local_changed) {
        TF_ASSIGN_OR_RETURN(local_changed, TryVectorizeConv(conv, 4));
      }
      changed |= local_changed;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
