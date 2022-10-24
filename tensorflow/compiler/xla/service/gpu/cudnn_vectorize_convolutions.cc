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

#include <optional>
#include <vector>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_support_utils.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {
namespace gpu {

// Finds convolutions that this pass may be able to transform, namely int8_t
// cudnn forward or forward-bias-activation convolutions
//
// cudnn as of v8.2 supports the following data type combinations for forward
// and forward-bias-activation convolutions.  We have to make sure we only
// vectorize to one of these supported configs.
//
//   in       out
//   int8x1   int8x1
//   int8x1   float
//   int8x1   int32_t
//
//   int8x4   int8x4
//   int8x4   float
//
//   int8x32  int8x32
//
// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionForward
//
// For now we restrict ourselves to only the int8xN -> int8xN cases.  We could
// allow the int8x4 -> float case in the future if desirable.
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

// Converts an XlaBuilder into an HloComputation in the same module as
// `sibling_computation`.
//
// Yes, we serialize/deserialize as a proto.  :)
static StatusOr<HloComputation*> BuilderToHloComputation(
    XlaBuilder& b, XlaOp root, HloComputation* sibling_computation) {
  TF_ASSIGN_OR_RETURN(XlaComputation comp, b.Build(root));
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape, comp.GetProgramShape());
  HloModuleConfig config(program_shape);
  TF_ASSIGN_OR_RETURN(auto new_module,
                      HloModule::CreateFromProto(comp.proto(), config));

  HloModule* dest_module = sibling_computation->parent();
  HloCloneContext context(dest_module);
  return dest_module->DeepCloneComputation(new_module->entry_computation(),
                                           &context);
}

// Reshapes `instr` so that it has an extra dimension of size `vect_size` right
// after `dim`.
static XlaOp SplitAtDim(XlaOp instr, int64_t dim, int64_t vect_size) {
  XlaBuilder& b = *instr.builder();
  Shape shape = b.GetShape(instr).value();
  DimensionVector new_dims(shape.dimensions().begin(),
                           shape.dimensions().end());
  CHECK_EQ(new_dims[dim] % vect_size, 0);
  new_dims[dim] /= vect_size;
  new_dims.insert(new_dims.begin() + dim + 1, vect_size);
  return Reshape(instr, new_dims);
}

// Reshapes `shape` so that there's an extra dimension of size `vect_size` right
// after `dim`.
//
// For example given shape=s8[10, 32, 20], dim=1, vect_size=4, returns
// s8[10, 8, 4, 20].
static Shape SplitShapeAtDim(Shape shape, int64_t dim, int64_t vect_size) {
  DimensionVector new_dims(shape.dimensions().begin(),
                           shape.dimensions().end());
  CHECK_EQ(new_dims[dim] % vect_size, 0);
  new_dims[dim] /= vect_size;
  new_dims.insert(new_dims.begin() + dim + 1, vect_size);
  return ShapeUtil::MakeShape(shape.element_type(), new_dims);
}

// Transposes dimension `src` to right before `dst`.
static XlaOp MoveDim(XlaOp instr, int64_t src, int64_t dst) {
  XlaBuilder& b = *instr.builder();
  int64_t rank = b.GetShape(instr)->dimensions_size();

  DimensionVector idxs(rank);
  absl::c_iota(idxs, 0);
  if (src < dst) {
    idxs.insert(idxs.begin() + dst, src);
    idxs.erase(idxs.begin() + src);
  } else {
    idxs.erase(idxs.begin() + src);
    idxs.insert(idxs.begin() + dst, src);
  }
  return Transpose(instr, idxs);
}

// Reshapes instr so that dimension `vect_dim` has size `vect_size`, by stealing
// elements from `dim`.
//
// Requires that this is possible without merging and re-splitting the two
// dimensions.  I.e. there should be some amount of dim that we can "split off"
// and add to vect_dim to get it to have size vect_size.
static XlaOp RevectorizeInstr(XlaOp instr, int64_t dim, int64_t vect_dim,
                              int64_t vect_size) {
  XlaBuilder& b = *instr.builder();
  Shape shape = b.GetShape(instr).value();
  auto size = [&](int64_t d) { return shape.dimensions(d); };

  CHECK_LE(size(vect_dim), vect_size);
  CHECK_EQ(vect_size % size(vect_dim), 0);

  int64_t split_factor = vect_size / size(vect_dim);
  CHECK_EQ(size(dim) % split_factor, 0);

  // Split dim into [C, split_factor].
  instr = SplitAtDim(instr, dim, split_factor);

  // SplitAtDim may have added a dimension before vect_dim.
  if (vect_dim > dim) {
    vect_dim++;
  }

  // Move the split_factor dimension to right before vect_dim.
  instr = MoveDim(instr, dim + 1, vect_dim);

  // Moving the split_factor dimension may have *removed* a dimension before
  // vect_dim.
  if (vect_dim > dim) {
    vect_dim--;
  }

  // Collapse the split_factor dimension into vect_dim.
  return Collapse(instr, {vect_dim, vect_dim + 1});
}

// Inverse of RevectorizeInstr.  Reshapes instr so that dimension `vect_dim` has
// size `vect_size`, moving excess elements into `dim`.
static XlaOp UnrevectorizeInstr(XlaOp instr, int64_t dim, int64_t vect_dim,
                                int64_t orig_vect_size) {
  XlaBuilder& b = *instr.builder();
  Shape shape = b.GetShape(instr).value();
  auto size = [&](int64_t d) { return shape.dimensions(d); };

  CHECK_GE(size(vect_dim), orig_vect_size);
  CHECK_EQ(size(vect_dim) % orig_vect_size, 0);

  // Split vect_dim into [C, orig_vect_size].
  instr = SplitAtDim(instr, vect_dim, orig_vect_size);

  // SplitAtDim may have added a dimension before dim.
  if (dim > vect_dim) {
    dim++;
  }

  // Move the `C` dimension to right after `dim`.  Take into account that
  // SplitAtDim may have added a dimension before dim.
  instr = MoveDim(instr, vect_dim, dim + 1);

  // MoveDim may have *removed* a dimension before dim.
  if (dim > vect_dim) {
    dim--;
  }

  // Collapse the `C` and `dim` dimensions.
  return Collapse(instr, {dim, dim + 1});
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
  for (int64_t& d : *dnums.mutable_input_spatial_dimensions()) {
    if (d > input_vect_dim) {
      ++d;
    }
  }

  int64_t kernel_vect_dim = dnums.kernel_input_feature_dimension();
  if (dnums.kernel_output_feature_dimension() > kernel_vect_dim) {
    dnums.set_kernel_output_feature_dimension(
        dnums.kernel_output_feature_dimension() + 1);
  }
  for (int64_t& d : *dnums.mutable_kernel_spatial_dimensions()) {
    if (d > kernel_vect_dim) {
      ++d;
    }
  }

  int64_t output_vect_dim = dnums.output_feature_dimension();
  if (dnums.output_batch_dimension() > output_vect_dim) {
    dnums.set_output_batch_dimension(dnums.output_batch_dimension() + 1);
  }
  for (int64_t& d : *dnums.mutable_output_spatial_dimensions()) {
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
static StatusOr<bool> TryRevectorizeConv(
    const se::CudaComputeCapability& compute_capability,
    HloCustomCallInstruction* conv, int vect_size) {
  const Shape& input_shape = conv->operand(0)->shape();
  const Shape& kernel_shape = conv->operand(1)->shape();
  const Shape& output_shape = conv->shape().tuple_shapes(0);
  const auto& dnums = conv->convolution_dimension_numbers();

  // Find the vectorized-features dim in the input/kernel/output.
  std::optional<int64_t> input_vect_dim;
  std::optional<int64_t> kernel_vect_dim;
  std::optional<int64_t> output_vect_dim;
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

  // If this is an integer convolution check that we only vectorize when cuDNN
  // supports the vectorized implementation.
  if (primitive_util::IsIntegralType(input_shape.element_type())) {
    TF_ASSIGN_OR_RETURN(bool supported_target_vectorization,
                        CudnnSupportsOptimizedIntegerConvolution(
                            compute_capability, *conv, vect_size));
    if (!supported_target_vectorization) {
      VLOG(3) << "Skipping re-vectorization of conv to vector size: "
              << vect_size << ": " << conv->ToString();
      return false;
    }
  }

  VLOG(1) << "Re-vectorizing conv channels from "
          << input_shape.dimensions(*input_vect_dim) << " to " << vect_size
          << ": " << conv->ToString();

  // We use XlaBuilder because it's a lot easier to get these tricky
  // reshape/transposes correct using that API.
  XlaBuilder b(absl::StrCat(conv->name(), ".revectorized"));
  b.SetOpMetadata(conv->metadata());

  absl::InlinedVector<XlaOp, 4> new_operands = {
      RevectorizeInstr(Parameter(&b, 0, conv->operand(0)->shape(), "input"),
                       dnums.input_feature_dimension(), *input_vect_dim,
                       vect_size),
      RevectorizeInstr(Parameter(&b, 1, conv->operand(1)->shape(), "filter"),
                       dnums.kernel_input_feature_dimension(), *kernel_vect_dim,
                       vect_size),
  };
  if (conv->operand_count() > 2) {
    // Bias, if present.  This is passed through unmodified.
    new_operands.push_back(Parameter(&b, 2, conv->operand(2)->shape(), "bias"));
  }
  if (conv->operand_count() > 3) {
    new_operands.push_back(RevectorizeInstr(
        Parameter(&b, 3, conv->operand(3)->shape(), "side_input"),
        dnums.input_feature_dimension(), *input_vect_dim, vect_size));
  }

  if (conv->operand_count() > 4) {
    return InvalidArgument(
        "Don't understand a conv with more than 4 arguments: %s",
        conv->ToString());
  }

  // The custom-call returns a tuple (new_output_shape, u8[0]), where the second
  // value in the tuple represents the convolution's scratch memory.
  DimensionVector new_output_dims(output_shape.dimensions().begin(),
                                  output_shape.dimensions().end());
  new_output_dims[dnums.output_feature_dimension()] /=
      (vect_size / output_vect_size);
  new_output_dims[*output_vect_dim] = vect_size;
  XlaOp new_conv = CustomCallWithConvDnums(
      &b, conv->custom_call_target(), new_operands,
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeShape(output_shape.element_type(), new_output_dims),
           ShapeUtil::MakeShape(U8, {0})}),
      /*operand_shapes_with_layout=*/{},
      /*opaque=*/conv->raw_backend_config_string(), /*has_side_effect=*/false,
      /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
      /*window=*/conv->window(),
      /*dnums=*/conv->convolution_dimension_numbers());

  XlaOp new_conv_result = GetTupleElement(new_conv, 0);
  XlaOp new_conv_scratch = GetTupleElement(new_conv, 1);

  XlaOp new_conv_result_unrevectorized = UnrevectorizeInstr(
      new_conv_result, dnums.output_feature_dimension(), *output_vect_dim,
      /*orig_vect_size=*/output_shape.dimensions(*output_vect_dim));

  TF_ASSIGN_OR_RETURN(
      HloComputation * new_conv_comp,
      BuilderToHloComputation(
          b, Tuple(&b, {new_conv_result_unrevectorized, new_conv_scratch}),
          conv->parent()));

  // Set the name on the new conv.  This is purely cosmetic, but we attempt to
  // preserve e.g. "cudnn-conv.42" instead of "custom-call.42".
  auto new_conv_comp_instrs = new_conv_comp->instructions();
  auto new_conv_it =
      absl::c_find_if(new_conv_comp_instrs, [](HloInstruction* instr) {
        return instr->opcode() == HloOpcode::kCustomCall;
      });
  if (new_conv_it != new_conv_comp_instrs.end()) {
    new_conv_comp->parent()->SetAndUniquifyInstrName(*new_conv_it,
                                                     conv->name());
  }

  // Replace the old conv with a call to the computation we just created.
  VLOG(1) << "Re-vectorized conv to " << new_conv_comp->ToString();
  TF_RETURN_IF_ERROR(conv->parent()->ReplaceWithNewInstruction(
      conv, HloInstruction::CreateCall(conv->shape(), conv->operands(),
                                       new_conv_comp)));

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
static StatusOr<bool> TryVectorizeConv(
    const se::CudaComputeCapability& compute_capability,
    HloCustomCallInstruction* conv, int64_t vect_size) {
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

  // If this is an integer convolution check that we only vectorize when cuDNN
  // supports the vectorized implementation.
  if (primitive_util::IsIntegralType(input_shape.element_type())) {
    TF_ASSIGN_OR_RETURN(bool supported_target_vectorization,
                        CudnnSupportsOptimizedIntegerConvolution(
                            compute_capability, *conv, vect_size));
    if (!supported_target_vectorization) {
      VLOG(3) << "Skipping vectorization of conv to vector size: " << vect_size
              << ": " << conv->ToString();
      return false;
    }
  }

  VLOG(1) << "Vectorizing conv channels by " << vect_size << ": "
          << conv->ToString();

  // We use XlaBuilder because it's a lot easier to get these tricky
  // reshape/transposes correct using that API.
  XlaBuilder b(absl::StrCat(conv->name(), ".revectorized"));
  b.SetOpMetadata(conv->metadata());

  absl::InlinedVector<XlaOp, 4> new_operands = {
      SplitAtDim(Parameter(&b, 0, conv->operand(0)->shape(), "input"),
                 dnums.input_feature_dimension(), vect_size),
      SplitAtDim(Parameter(&b, 1, conv->operand(1)->shape(), "filter"),
                 dnums.kernel_input_feature_dimension(), vect_size),
  };
  if (conv->operand_count() > 2) {
    // Bias, if present.  This is passed through unmodified.
    new_operands.push_back(Parameter(&b, 2, conv->operand(2)->shape(), "bias"));
  }
  if (conv->operand_count() > 3) {
    // Handle side input, which has same shape as the input.
    new_operands.push_back(
        SplitAtDim(Parameter(&b, 3, conv->operand(3)->shape(), "side_input"),
                   dnums.input_feature_dimension(), vect_size));
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
  XlaOp new_conv = CustomCallWithConvDnums(
      &b, conv->custom_call_target(), new_operands,
      ShapeUtil::MakeTupleShape(
          {new_output_shape, ShapeUtil::MakeShape(U8, {0})}),
      /*operand_shapes_with_layout=*/{},
      /*opaque=*/conv->raw_backend_config_string(), /*has_side_effect=*/false,
      /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
      /*window=*/conv->window(),
      /*dnums=*/VectorizeDnums(dnums));

  XlaOp new_conv_result = GetTupleElement(new_conv, 0);
  XlaOp new_conv_scratch = GetTupleElement(new_conv, 1);

  // Reshape back to the original shape.
  XlaOp conv_result_collapsed = Collapse(
      new_conv_result,
      {dnums.output_feature_dimension(), dnums.output_feature_dimension() + 1});

  TF_ASSIGN_OR_RETURN(
      HloComputation * new_conv_comp,
      BuilderToHloComputation(
          b, Tuple(&b, {conv_result_collapsed, new_conv_scratch}),
          conv->parent()));

  // Create a tuple and replace the old conv with it!
  VLOG(1) << "Vectorized conv to: " << new_conv_comp->ToString();
  TF_RETURN_IF_ERROR(conv->parent()->ReplaceWithNewInstruction(
      conv, HloInstruction::CreateCall(conv->shape(), conv->operands(),
                                       new_conv_comp)));
  return true;
}

StatusOr<bool> CudnnVectorizeConvolutions::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloCustomCallInstruction* conv : GetRelevantConvs(comp)) {
      // Try to (re)vectorize to int8x32 if this is an sm75+ GPU.  If we can't,
      // fall back to int8x4.
      bool local_changed = false;
      if (compute_capability_.IsAtLeast(7, 5)) {
        TF_ASSIGN_OR_RETURN(local_changed,
                            TryRevectorizeConv(compute_capability_, conv, 32));
        if (!local_changed) {
          TF_ASSIGN_OR_RETURN(local_changed,
                              TryVectorizeConv(compute_capability_, conv, 32));
        }
      }
      if (!local_changed) {
        TF_ASSIGN_OR_RETURN(local_changed,
                            TryVectorizeConv(compute_capability_, conv, 4));
      }
      changed |= local_changed;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
