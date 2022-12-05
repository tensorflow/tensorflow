/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/cudnn_simplify_padding.h"

#include <algorithm>
#include <cstdio>
#include <iterator>
#include <optional>
#include <sstream>
#include <vector>

#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla::gpu {

namespace {
namespace m = ::xla::match;

// If exactly one index of `dims` is false, returns that index.  If 0 or more
// than one index is false, returns nullopt.
std::optional<int64_t> FindFalseIndex(absl::Span<const bool> vals) {
  std::optional<int64_t> missing_dim;
  for (int i = 0; i < vals.size(); i++) {
    if (vals[i]) {
      continue;
    }
    if (missing_dim.has_value()) {
      VLOG(2) << "Multiple dimensions are missing from conv dnums; can't "
                 "determine which is vect_c dimension";
      return std::nullopt;
    }
    missing_dim = i;
  }
  return missing_dim;
}

// Finds the vect_c dimension in the convolution's output.
//
// The vect_c dimension in dnums is the dimension that's not mentioned in
// `dnums`.  If there's zero or more than one such dimension, returns nullopt.
std::optional<int64_t> FindOutputVectCDim(HloInstruction* conv) {
  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();
  int64_t num_dims = conv->shape().tuple_shapes(0).dimensions_size();
  absl::InlinedVector<bool, 5> seen_dims(num_dims);
  seen_dims[dnums.output_batch_dimension()] = true;
  seen_dims[dnums.output_feature_dimension()] = true;
  for (int64_t d : dnums.output_spatial_dimensions()) {
    seen_dims[d] = true;
  }
  return FindFalseIndex(seen_dims);
}

// Finds the vect_c dimension in the convolution's kernel.
std::optional<int64_t> FindKernelVectCDim(HloInstruction* conv) {
  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();
  int64_t num_dims = conv->operand(1)->shape().dimensions_size();
  absl::InlinedVector<bool, 5> seen_dims(num_dims);
  seen_dims[dnums.kernel_input_feature_dimension()] = true;
  seen_dims[dnums.kernel_output_feature_dimension()] = true;
  for (int64_t d : dnums.kernel_spatial_dimensions()) {
    seen_dims[d] = true;
  }
  return FindFalseIndex(seen_dims);
}

// Attempts to count the number of output features at the end of conv that are
// guaranteed to be 0.
//
// This is the same as counting the number of values o at the end of the kernel
// for which kernel[i,o,h,w] is 0 for all values i,h,w.
std::optional<int64_t> NumTrailingZeroOutputFeatures(HloInstruction* conv) {
  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();
  int64_t feature_dim = dnums.kernel_output_feature_dimension();
  const HloInstruction* weights = conv->operand(1);
  VLOG(2) << "Computing NumTrailingZeroOutputFeatures of " << conv->ToString()
          << "\nwith weights " << weights->ToString();
  if (Match(weights, m::Pad(m::Op(), m::ConstantEffectiveScalar(0)))) {
    const PaddingConfig::PaddingConfigDimension& padding_config =
        weights->padding_config().dimensions(feature_dim);
    // The last N output feature weights are all 0.
    VLOG(2) << "Success: Weights is a pad; padding on output feature dim is "
            << padding_config.edge_padding_high();
    return padding_config.edge_padding_high();
  } else if (const HloInstruction * pad; Match(
                 weights, m::Reshape(m::Pad(&pad, m::Op(),
                                            m::ConstantEffectiveScalar(0))))) {
    // Check that the reshape merely adds a VECT_C to the kernel input features.
    // That is, we reshape from [I,O,H,W] (in some order) to [I/k,k,O,H,W] (in
    // the same order) for some constant k (probably 32).  Then check how much
    // the pad adds to the O dimension.
    std::optional<int64_t> vect_c_dim = FindKernelVectCDim(conv);
    if (!vect_c_dim.has_value()) {
      VLOG(2) << "fail: Can't find vect_c dimension in conv.";
      return std::nullopt;
    }
    if (*vect_c_dim != dnums.kernel_input_feature_dimension() + 1) {
      VLOG(2) << "fail: vect_c dim is in the wrong place; should be right "
                 "after kernel input feature dims in conv.";
      return std::nullopt;
    }
    absl::InlinedVector<int64_t, 5> expected_pad_dim_sizes(
        weights->shape().dimensions().begin(),
        weights->shape().dimensions().end());
    expected_pad_dim_sizes[dnums.kernel_input_feature_dimension()] *=
        weights->shape().dimensions(*vect_c_dim);
    expected_pad_dim_sizes.erase(expected_pad_dim_sizes.begin() + *vect_c_dim);
    if (pad->shape().dimensions() != expected_pad_dim_sizes) {
      VLOG(2) << "fail: Reshape doesn't simply merge vect_c dimension into "
                 "input features dim "
              << weights->ToString() << " but expected dims "
              << absl::StrJoin(expected_pad_dim_sizes, ",");
      return std::nullopt;
    }

    // If the filter dnums are e.g. [I,O,H,W] then after reshape they are
    // [I/k,k,O,H,W] and the new index of O is greater less than before the
    // reshape (which we know only adds the I/k and k dims, which we also know
    // are contiguous).  OTOH if the O comes before the I in the original, then
    // the index of O doesn't change after the reshape.
    int64_t feature_dim_before_reshape = feature_dim;
    if (dnums.kernel_output_feature_dimension() >
        dnums.kernel_input_feature_dimension()) {
      feature_dim_before_reshape--;
    }
    const PaddingConfig::PaddingConfigDimension& padding_config =
        pad->padding_config().dimensions(feature_dim_before_reshape);

    // The last N output feature weights are all 0.
    VLOG(2) << "Success: Weights is a reshape of a pad; padding on output "
               "feature dim is "
            << padding_config.edge_padding_high();
    return padding_config.edge_padding_high();
  } else if (Match(weights, m::Constant())) {
    // Iterate backwards over `weights` to find the index of the first nonzero
    // value.
    //
    // TODO(jlebar): This may be slow, because it iterates over potentially the
    // whole constant and does a multi_index -> linear_index conversion for each
    // element. If necessary we could rewrite this by using linear indices, but
    // we'd have to be careful of the fact that literals can have arbitrary
    // layouts, so you can't just iterate over the literal's bytes.
    const Literal& lit = weights->literal();
    const auto& dims = weights->shape().dimensions();
    absl::InlinedVector<int64_t, 5> multi_index;
    for (int64_t dim : dims) {
      multi_index.push_back(dim - 1);
    }
    while (true) {
      if (!lit.IsZero(multi_index)) {
        break;
      }
      multi_index[multi_index.size() - 1]--;
      for (int i = multi_index.size() - 2; i > 0; i--) {
        if (multi_index[i] == -1) {
          multi_index[i] = dims[i] - 1;
          multi_index[i - 1]--;
        } else {
          break;
        }
      }
      if (multi_index[0] == -1) {
        break;
      }
    }

    VLOG(2) << "First nonzero index in weights constant is "
            << absl::StrJoin(multi_index, ",");
    int64_t first_nonzero_feature = multi_index[feature_dim];
    // "round up" the first nonzero feature index if it's not *all* zeros.
    for (int i = 0; i < multi_index.size(); i++) {
      if (i != feature_dim && multi_index[i] != 0) {
        first_nonzero_feature++;
        break;
      }
    }
    int64_t ret = std::max<int64_t>(
        0, weights->shape().dimensions(feature_dim) - first_nonzero_feature);
    VLOG(2) << "Success: weights is a constant; num zero trailing output "
               "features is "
            << ret;
    return ret;
  }
  return std::nullopt;
}

StatusOr<bool> TrySimplifyPadding(HloInstruction* instr) {
  // Match one of the following patterns.
  //   conv -> slice -> pad
  //   conv -> reshape -> slice-> pad
  //   conv -> transpose -> reshape -> slice -> pad
  //
  // where `pad` (the root of the pattern) is `instr`.
  HloInstruction* conv;
  HloInstruction* transpose = nullptr;  // optional
  HloInstruction* reshape = nullptr;    // optional
  HloInstruction* slice;
  HloInstruction* pad;
  auto conv_matcher = m::GetTupleElement(
      m::CustomCall(&conv).WithPredicate([](const HloInstruction* instr) {
        return instr->custom_call_target() == kCudnnConvForwardCallTarget ||
               instr->custom_call_target() ==
                   kCudnnConvBiasActivationForwardCallTarget;
      }),
      0);
  auto pad_matcher = m::Pad(m::Op(), m::ConstantEffectiveScalar(0));
  if (!MatchAndLogIfFailed(instr, "conv-slice-pad",
                           m::Pad(&pad, m::Slice(&slice, conv_matcher),
                                  m::ConstantEffectiveScalar(0)),
                           VLOG_IS_ON(3), pad_matcher) &&
      !MatchAndLogIfFailed(
          instr, "conv-reshape-slice-pad",
          m::Pad(&pad, m::Slice(&slice, m::Reshape(&reshape, conv_matcher)),
                 m::ConstantEffectiveScalar(0)),
          VLOG_IS_ON(3), pad_matcher) &&
      !MatchAndLogIfFailed(
          instr, "conv-transpose-reshape-slice-pad",
          m::Pad(&pad,
                 m::Slice(&slice,
                          m::Reshape(&reshape,
                                     m::Transpose(&transpose, conv_matcher))),
                 m::ConstantEffectiveScalar(0)),
          VLOG_IS_ON(3), pad_matcher)) {
    return false;
  }

  VLOG(2) << "Found pattern to attempt to simplify:\n"
          << "conv: " << conv->ToString()  //
          << "\ntranspose: "
          << (transpose != nullptr ? transpose->ToString() : "(null)")
          << "\nreshape: "
          << (reshape != nullptr ? reshape->ToString() : "(null)")
          << "\nslice: " << slice->ToString()  //
          << "\npad: " << pad->ToString();

  // Now check that we can merge the slice into the pad, because the slice is
  // slicing off elements that we know are 0 and the pad is just adding those 0s
  // back.
  //
  // First, we have to check whether any of the output features at the end of
  // the conv are known to be 0.
  std::optional<int64_t> num_known_zero_output_features =
      NumTrailingZeroOutputFeatures(conv);
  if (!num_known_zero_output_features.has_value() ||
      *num_known_zero_output_features == 0) {
    VLOG(2) << "fail: Didn't find any known-zero output features";
    return false;
  }

  // We now know that some of the output features of the conv (starting at
  // known_zero_output_features_start_idx) are zero.  Check if the
  // optional-reshape + optional-transpose + slice + pad combination is setting
  // all of these features to 0.  If so, we can merge the slice into the pad.
  const auto& dnums = conv->convolution_dimension_numbers();
  int64_t output_feature_dim;
  if (reshape == nullptr) {
    CHECK_EQ(transpose, nullptr);
    output_feature_dim = dnums.output_feature_dimension();
  } else {
    std::optional<int64_t> vect_c_dim_before_transpose =
        FindOutputVectCDim(conv);
    if (!vect_c_dim_before_transpose.has_value()) {
      VLOG(2) << "Couldn't find vect_c output dim in conv.";
      return false;
    }

    // If there's no transpose, check that the vect_c dim is immediately after
    // the feature dim.  OTOH if there is a transpose, check that the transpose
    // moves the vect_c dim immediately after the feature dim.
    int64_t feature_dim_after_transpose;
    int64_t vect_c_dim_after_transpose;
    if (transpose == nullptr) {
      feature_dim_after_transpose = dnums.output_feature_dimension();
      vect_c_dim_after_transpose = *vect_c_dim_before_transpose;
    } else {
      const auto& transpose_dims = transpose->dimensions();
      feature_dim_after_transpose = std::distance(
          transpose->dimensions().begin(),
          absl::c_find(transpose_dims, dnums.output_feature_dimension()));
      vect_c_dim_after_transpose = std::distance(
          transpose->dimensions().begin(),
          absl::c_find(transpose_dims, *vect_c_dim_before_transpose));
    }
    if (vect_c_dim_after_transpose != feature_dim_after_transpose + 1) {
      VLOG(2) << "fail: after transpose (if present), vect_c dim must appear "
                 "immediately after output feature dim: Computed "
                 "vect_d_dim_after_transpose to be "
              << vect_c_dim_after_transpose;
      return false;
    }

    // Now check that the reshape merges the feature + vect_c dims and
    // doesn't do anything else.
    absl::InlinedVector<int64_t, 5> expected_reshape_dim_sizes(
        reshape->operand(0)->shape().dimensions().begin(),
        reshape->operand(0)->shape().dimensions().end());
    expected_reshape_dim_sizes[feature_dim_after_transpose] *=
        expected_reshape_dim_sizes[vect_c_dim_after_transpose];
    expected_reshape_dim_sizes.erase(expected_reshape_dim_sizes.begin() +
                                     vect_c_dim_after_transpose);
    if (reshape->shape().dimensions() != expected_reshape_dim_sizes) {
      VLOG(2) << "fail: Reshape doesn't merge vect_c with feature dimension.";
      return false;
    }

    output_feature_dim = feature_dim_after_transpose;
  }

  // Check that `slice` slices only the output feature dimension.
  if (!absl::c_all_of(slice->slice_starts(), [](auto v) { return v == 0; }) ||
      !absl::c_all_of(slice->slice_strides(), [](auto v) { return v == 1; })) {
    VLOG(2) << "fail: Slice doesn't start at the front or has stride != 1.";
    return false;
  }

  // We're only allowed to slice the feature dim.
  for (int64_t dim = 0; dim < slice->slice_limits().size(); dim++) {
    if (dim != output_feature_dim &&
        slice->slice_limits(dim) != slice->shape().dimensions(dim)) {
      VLOG(2) << "fail: Slice removes something other than the features dim.";
      return false;
    }
  }
  int64_t num_sliced_from_feature_dim =
      slice->operand(0)->shape().dimensions(output_feature_dim) -
      slice->slice_limits(output_feature_dim);

  // If we slice off more than the known-zero output features, then we need to
  // keep the slice -- it's "doing something".
  if (num_sliced_from_feature_dim > *num_known_zero_output_features) {
    VLOG(2) << "fail: Slice removes " << num_sliced_from_feature_dim
            << " features from the conv, but only "
            << *num_known_zero_output_features
            << " features in the conv are known to be zero.";
    return false;
  }

  // Check if we can merge the slice into the pad.
  if (pad->padding_config().dimensions(output_feature_dim).interior_padding() !=
      0) {
    VLOG(2)
        << "fail: Can't merge slice into pad because pad adds interior padding "
           "in feature dimension.";
    return false;
  }

  // Okay!  If we got here, it's legal to fold the slice into the pad.  We pad
  // less, because we know that the sliced-off elements are all 0.  Ideally, the
  // pad becomes a nop and gets eliminated by algsimp later.
  VLOG(1) << "Eliminating " << num_sliced_from_feature_dim
          << " elements of padding from conv " << conv->name();
  PaddingConfig new_padding_config = pad->padding_config();
  PaddingConfig::PaddingConfigDimension* new_pad_feature_dim =
      new_padding_config.mutable_dimensions(output_feature_dim);
  // This is safe even if the new edge_padding_high is negative -- negative
  // padding is allowed.
  new_pad_feature_dim->set_edge_padding_high(
      new_pad_feature_dim->edge_padding_high() - num_sliced_from_feature_dim);
  TF_ASSIGN_OR_RETURN(HloInstruction * new_pad,
                      MakePadHlo(slice->mutable_operand(0),
                                 pad->mutable_operand(1), new_padding_config));
  TF_RETURN_IF_ERROR(pad->parent()->ReplaceInstruction(pad, new_pad));
  return true;
}

}  // anonymous namespace

StatusOr<bool> CudnnSimplifyPadding::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(bool c, TrySimplifyPadding(instr));
      changed |= c;
    }
  }
  return changed;
}

}  // namespace xla::gpu
