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

#include "xla/hlo/transforms/simplifiers/conv_operand_swapper.h"

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {

absl::StatusOr<bool> SwapConvolutionOperandsIfBeneficial(
    HloConvolutionInstruction* convolution,
    ConvOperandSwapper::ConvIsLowerableCallback conv_is_lowerable_callback) {
  // Current logic only handles non-grouped convolutions.
  if (convolution->feature_group_count() > 1 ||
      convolution->batch_group_count() > 1) {
    return false;
  }

  const auto& dnums = convolution->convolution_dimension_numbers();
  const auto& window_dims = convolution->window().dimensions();
  Window swapped_window;

  HloInstruction *input = convolution->mutable_operand(0),
                 *kernel = convolution->mutable_operand(1);
  int64_t kernel_product = 1;
  int64_t swapped_kernel_product = 1;
  DimensionVector reverse_dimensions;
  for (int64_t spatial_dim = 0;
       spatial_dim < dnums.input_spatial_dimensions_size(); ++spatial_dim) {
    const int64_t kernel_size = window_dims[spatial_dim].size();
    const bool can_be_group_or_contraction =
        !window_dims[spatial_dim].window_reversal() &&
        window_dims[spatial_dim].padding_low() == 0 &&
        window_dims[spatial_dim].padding_high() == 0 &&
        window_dims[spatial_dim].window_dilation() == 1;
    const bool is_group_dim =
        can_be_group_or_contraction &&
        window_dims[spatial_dim].base_dilation() == kernel_size &&
        window_dims[spatial_dim].stride() == kernel_size - 1;
    const int64_t input_size =
        input->shape().dimensions(dnums.input_spatial_dimensions(spatial_dim));
    const bool is_pure_contraction_dim =
        kernel_size == input_size && can_be_group_or_contraction &&
        window_dims[spatial_dim].base_dilation() == 1 &&
        window_dims[spatial_dim].stride() == 1;
    if (is_group_dim || is_pure_contraction_dim) {
      *(swapped_window.add_dimensions()) = window_dims[spatial_dim];
      continue;
    }

    const int64_t dilated_kernel_size =
        1 + (kernel_size - 1) * window_dims[spatial_dim].window_dilation();
    const int64_t dilated_input_size =
        1 + (input_size - 1) * window_dims[spatial_dim].base_dilation();

    kernel_product *= kernel_size;
    swapped_kernel_product *=
        input_size == 1 && window_dims[spatial_dim].stride() == 1 &&
                window_dims[spatial_dim].window_dilation() == 1 &&
                window_dims[spatial_dim].padding_high() == kernel_size - 1 &&
                window_dims[spatial_dim].padding_low() == kernel_size - 1
            ? kernel_size
            : input_size;

    auto new_dim = swapped_window.add_dimensions();
    new_dim->set_size(input_size);
    if (!window_dims[spatial_dim].window_reversal()) {
      reverse_dimensions.push_back(
          dnums.kernel_spatial_dimensions(spatial_dim));
    }
    new_dim->set_window_reversal(true);
    new_dim->set_base_dilation(window_dims[spatial_dim].window_dilation());
    new_dim->set_window_dilation(window_dims[spatial_dim].base_dilation());
    new_dim->set_stride(window_dims[spatial_dim].stride());
    new_dim->set_padding_low(dilated_input_size +
                             window_dims[spatial_dim].padding_low() -
                             dilated_kernel_size);
    new_dim->set_padding_high(dilated_input_size +
                              window_dims[spatial_dim].padding_high() -
                              dilated_kernel_size);
  }

  if (kernel_product <= swapped_kernel_product) {
    return false;
  }

  // Construct swapped dimension numbers.
  ConvolutionDimensionNumbers swapped_dnums;
  *swapped_dnums.mutable_output_spatial_dimensions() =
      dnums.output_spatial_dimensions();
  swapped_dnums.set_output_batch_dimension(dnums.output_feature_dimension());
  swapped_dnums.set_output_feature_dimension(dnums.output_batch_dimension());
  *swapped_dnums.mutable_input_spatial_dimensions() =
      dnums.kernel_spatial_dimensions();
  swapped_dnums.set_input_batch_dimension(
      dnums.kernel_output_feature_dimension());
  swapped_dnums.set_input_feature_dimension(
      dnums.kernel_input_feature_dimension());
  *swapped_dnums.mutable_kernel_spatial_dimensions() =
      dnums.input_spatial_dimensions();
  swapped_dnums.set_kernel_output_feature_dimension(
      dnums.input_batch_dimension());
  swapped_dnums.set_kernel_input_feature_dimension(
      dnums.input_feature_dimension());

  PrecisionConfig precision_config;
  precision_config.add_operand_precision(
      convolution->precision_config().operand_precision(1));
  precision_config.add_operand_precision(
      convolution->precision_config().operand_precision(0));

  if (!reverse_dimensions.empty()) {
    HloInstruction* old_kernel = kernel;
    TF_ASSIGN_OR_RETURN(kernel, MakeReverseHlo(kernel, reverse_dimensions));
    if (old_kernel->has_sharding()) {
      kernel->set_sharding(old_kernel->sharding());
    }
  }

  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_convolution,
      MakeConvolveHlo(
          kernel, input, /*feature_group_count=*/1,
          /*batch_group_count=*/1, swapped_window, swapped_dnums,
          precision_config,
          /*preferred_element_type=*/convolution->shape().element_type()));

  if (conv_is_lowerable_callback &&
      !conv_is_lowerable_callback(new_convolution)) {
    TF_RETURN_IF_ERROR(kernel->parent()->RemoveInstruction(new_convolution));
    return false;
  }

  convolution->SetupDerivedInstruction(new_convolution);
  return convolution->parent()->ReplaceInstruction(
      convolution, new_convolution, /*preserve_sharding=*/true,
      /*relay_control_dependency=*/true, /*remove_unused_operands=*/true,
      /*preserve_frontend_attributes=*/true);
}

absl::StatusOr<bool> ConvOperandSwapper::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (auto comp : module->computations(execution_threads)) {
    for (HloInstruction* hlo : comp->MakeInstructionPostOrder()) {
      if (auto* convolution = DynCast<HloConvolutionInstruction>(hlo)) {
        TF_ASSIGN_OR_RETURN(bool convolution_changed,
                            SwapConvolutionOperandsIfBeneficial(
                                convolution, conv_is_lowerable_callback_));
        changed |= convolution_changed;
      }
    }
  }
  return changed;
}

}  // namespace xla
