/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/cudnn_simplify_padding.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/literal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {
namespace m = ::xla::match;

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
  }
  if (Match(weights, m::Constant())) {
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
    // This iterates through the literal with feature_dim as the most
    // major dimension looking for the final non-zero feature.
    auto decrement_multi_index = [&] {
      for (int i = 0; i < multi_index.size(); ++i) {
        if (i != feature_dim) {
          int64_t& idx = multi_index[i];
          --idx;
          if (idx == -1) {
            idx = dims[i] - 1;
          } else {
            return true;
          }
        }
      }
      int64_t& idx = multi_index[feature_dim];
      --idx;
      return idx != -1;
    };
    do {
      if (!lit.IsZero(multi_index)) {
        break;
      }
    } while (decrement_multi_index());

    // The iteration stops if a feature has a non-zero value (or -1), but we
    // want the first zero feature which is always the next one (or 0 if -1).
    int64_t first_trailing_zero_feature = multi_index[feature_dim] + 1;

    if (first_trailing_zero_feature == 0) {
      VLOG(2) << "Weights constant is entirely zero.";
    } else {
      VLOG(2) << "First nonzero index in weights constant is "
              << absl::StrJoin(multi_index, ",");
    }
    int64_t ret =
        std::max<int64_t>(0, weights->shape().dimensions(feature_dim) -
                                 first_trailing_zero_feature);
    VLOG(2) << "Success: weights is a constant; num zero trailing output "
               "features is "
            << ret;
    return ret;
  }
  return std::nullopt;
}

absl::StatusOr<bool> TrySimplifyPadding(HloInstruction* instr) {
  // Match pattern: conv -> slice -> pad
  // where `pad` (the root of the pattern) is `instr`.
  HloInstruction* conv;
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
                           VLOG_IS_ON(3), pad_matcher)) {
    return false;
  }

  VLOG(2) << "Found pattern to attempt to simplify:\n"
          << "conv: " << conv->ToString()  //
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
  int64_t output_feature_dim = dnums.output_feature_dimension();

  // Check that `slice` slices only the output feature dimension.
  if (!absl::c_all_of(slice->slice_starts(), [](auto v) { return v == 0; }) ||
      !absl::c_all_of(slice->slice_strides(), [](auto v) { return v == 1; })) {
    VLOG(2) << "fail: Slice doesn't start at the front or has stride != 1.";
    return false;
  }

  // We're only allowed to slice the feature dim.
  for (int64_t dim = 0; dim < slice->slice_limits().size(); dim++) {
    if (slice->slice_starts(dim) != 0 || slice->slice_strides(dim) != 1 ||
        (dim != output_feature_dim &&
         slice->slice_limits(dim) !=
             slice->operand(0)->shape().dimensions(dim))) {
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

absl::StatusOr<bool> CudnnSimplifyPadding::RunImpl(
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
