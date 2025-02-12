/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/cpu/conv_canonicalization.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/permutation_util.h"
#include "xla/service/cpu/cpu_runtime.h"
#include "xla/service/cpu/ir_emission_utils.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace cpu {

absl::StatusOr<bool> ConvCanonicalization::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloInstruction* hlo :
       module->entry_computation()->MakeInstructionPostOrder()) {
    if (hlo->opcode() == HloOpcode::kConvolution &&
        !PotentiallyImplementedAsEigenConvolution(*hlo,
                                                  target_machine_features_)) {
      const ConvolutionDimensionNumbers& dnums =
          hlo->convolution_dimension_numbers();
      auto input_batch_dim = dnums.input_batch_dimension();
      auto input_feature_dim = dnums.input_feature_dimension();
      auto kernel_input_feature_dim = dnums.kernel_input_feature_dimension();
      auto kernel_output_feature_dim = dnums.kernel_output_feature_dimension();

      const int64_t num_spatial_dims = dnums.output_spatial_dimensions_size();
      const int64_t num_dims = num_spatial_dims + 2;

      // A canonical convolution's dimension numbers need to satisfy the
      // following conditions (see cs/PotentiallyImplementedAsEigenConvolution).
      //
      // - the input is in NHWC order.
      // - the kernel is in HWIO order.
      //
      // For simplicity, as a first step, we reshape the input and filter to
      // NHWC and HWIO order, respectively. This may lose precision but won't
      // break the soundness.
      HloInstruction* input = hlo->mutable_operand(0);

      std::vector<int64_t> new_input_dim_order(num_dims);
      std::vector<int64_t> new_input_dims(num_dims);
      new_input_dim_order[0] = input_batch_dim;
      new_input_dims[0] = input->shape().dimensions(input_batch_dim);
      for (int64_t i = 0; i < num_spatial_dims; ++i) {
        new_input_dim_order[i + 1] = dnums.input_spatial_dimensions(i);
        new_input_dims[i + 1] =
            input->shape().dimensions(dnums.input_spatial_dimensions(i));
      }
      new_input_dim_order[num_dims - 1] = input_feature_dim;
      new_input_dims[num_dims - 1] =
          input->shape().dimensions(input_feature_dim);

      Shape new_input_shape =
          ShapeUtil::MakeShape(input->shape().element_type(), new_input_dims);
      HloInstruction* new_input = module->entry_computation()->AddInstruction(
          HloInstruction::CreateTranspose(new_input_shape, input,
                                          new_input_dim_order));

      HloInstruction* kernel = hlo->mutable_operand(1);

      std::vector<int64_t> new_kernel_dim_order(num_dims);
      std::vector<int64_t> new_kernel_dims(num_dims);
      for (int64_t i = 0; i < num_spatial_dims; ++i) {
        new_kernel_dim_order[i] = dnums.kernel_spatial_dimensions(i);
        new_kernel_dims[i] =
            kernel->shape().dimensions(dnums.kernel_spatial_dimensions(i));
      }
      new_kernel_dim_order[num_dims - 2] = kernel_input_feature_dim;
      new_kernel_dims[num_dims - 2] =
          kernel->shape().dimensions(kernel_input_feature_dim);
      new_kernel_dim_order[num_dims - 1] = kernel_output_feature_dim;
      new_kernel_dims[num_dims - 1] =
          kernel->shape().dimensions(kernel_output_feature_dim);

      Shape new_kernel_shape =
          ShapeUtil::MakeShape(kernel->shape().element_type(), new_kernel_dims);
      HloInstruction* new_kernel = module->entry_computation()->AddInstruction(
          HloInstruction::CreateTranspose(new_kernel_shape, kernel,
                                          new_kernel_dim_order));

      std::vector<int64_t> new_output_dim_order(num_dims);
      std::vector<int64_t> new_conv_dims(num_dims);
      auto output_batch_dim = dnums.output_batch_dimension();
      auto output_feature_dim = dnums.output_feature_dimension();
      new_output_dim_order[0] = output_batch_dim;
      new_conv_dims[0] = hlo->shape().dimensions(output_batch_dim);
      for (int64_t i = 0; i < num_spatial_dims; ++i) {
        new_output_dim_order[i + 1] = dnums.output_spatial_dimensions(i);
        new_conv_dims[i + 1] =
            hlo->shape().dimensions(dnums.output_spatial_dimensions(i));
      }
      new_output_dim_order[num_dims - 1] = output_feature_dim;
      new_conv_dims[num_dims - 1] = hlo->shape().dimensions(output_feature_dim);
      Shape new_conv_shape =
          ShapeUtil::MakeShape(hlo->shape().element_type(), new_conv_dims);

      ConvolutionDimensionNumbers new_dnums;
      new_dnums.set_input_batch_dimension(0);
      new_dnums.set_output_batch_dimension(0);
      for (int64_t i = 0; i < num_spatial_dims; ++i) {
        new_dnums.add_input_spatial_dimensions(i + 1);
        new_dnums.add_kernel_spatial_dimensions(i);
        new_dnums.add_output_spatial_dimensions(i + 1);
      }
      new_dnums.set_input_feature_dimension(num_dims - 1);
      new_dnums.set_output_feature_dimension(num_dims - 1);
      new_dnums.set_kernel_input_feature_dimension(num_dims - 2);
      new_dnums.set_kernel_output_feature_dimension(num_dims - 1);

      // The window of the old convolution is reused, because reshapes only
      // change the dimension mapping but not the dimension sizes. For
      // example, input height and width are the same as before the reshapes.
      HloInstruction* new_conv = module->entry_computation()->AddInstruction(
          HloInstruction::CreateConvolve(
              new_conv_shape, new_input, new_kernel, hlo->feature_group_count(),
              hlo->batch_group_count(), hlo->window(), new_dnums,
              hlo->precision_config()));

      // Reshape the output back to the shape of the original convolution.
      TF_RETURN_IF_ERROR(module->entry_computation()->ReplaceWithNewInstruction(
          hlo, HloInstruction::CreateTranspose(
                   hlo->shape(), new_conv,
                   InversePermutation(new_output_dim_order))));
      changed = true;
    }
  }

  return changed;
}

}  // namespace cpu
}  // namespace xla
