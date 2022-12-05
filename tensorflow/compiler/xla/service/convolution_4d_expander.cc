/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/convolution_4d_expander.h"

#include <algorithm>
#include <functional>
#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

bool Convolution4DExpander::InstructionMatchesPattern(
    HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kConvolution) {
    return false;
  }

  // Check whether it is a 4D convolution and whether there is at least one
  // trivial dimension.
  const ConvolutionDimensionNumbers& dim_nums =
      instruction->convolution_dimension_numbers();
  if (dim_nums.input_spatial_dimensions().size() != 4) {
    return false;
  }
  Shape input = instruction->operand(0)->shape();
  for (int64_t i = 0; i < dim_nums.input_spatial_dimensions().size(); ++i) {
    int64_t spatial_dim = dim_nums.input_spatial_dimensions(i);
    if (input.dimensions(spatial_dim) == 1 &&
        instruction->window().dimensions(i).padding_low() == 0 &&
        instruction->window().dimensions(i).padding_high() == 0) {
      return true;
    }
  }
  return false;
}

StatusOr<HloInstruction*> Convolution4DExpander::ExpandInstruction(
    HloInstruction* instruction) {
  HloComputation* computation = instruction->parent();
  ConvolutionDimensionNumbers dim_nums =
      instruction->convolution_dimension_numbers();
  ConvolutionDimensionNumbers new_dim_nums = dim_nums;

  std::vector<int64_t> removed_input_dimensions;
  std::vector<int64_t> removed_kernel_dimensions;
  std::vector<int64_t> removed_output_dimensions;
  new_dim_nums.clear_input_spatial_dimensions();
  new_dim_nums.clear_output_spatial_dimensions();
  new_dim_nums.clear_kernel_spatial_dimensions();
  Window new_window;
  HloInstruction* input = instruction->mutable_operand(0);

  // Collect all trivial input spatial dimensions, and the corresponding
  // dimensions of the kernel and the output. Those will be removed.
  for (int64_t i = 0; i < dim_nums.input_spatial_dimensions().size(); ++i) {
    int64_t input_spatial_dim = dim_nums.input_spatial_dimensions(i);
    int64_t output_spatial_dim = dim_nums.output_spatial_dimensions(i);
    int64_t kernel_spatial_dim = dim_nums.kernel_spatial_dimensions(i);
    if (input->shape().dimensions(input_spatial_dim) == 1 &&
        instruction->window().dimensions(i).padding_low() == 0 &&
        instruction->window().dimensions(i).padding_high() == 0) {
      removed_input_dimensions.push_back(input_spatial_dim);
      removed_output_dimensions.push_back(output_spatial_dim);
      removed_kernel_dimensions.push_back(kernel_spatial_dim);
    } else {
      *new_window.add_dimensions() = instruction->window().dimensions(i);
      new_dim_nums.add_input_spatial_dimensions(input_spatial_dim);
      new_dim_nums.add_output_spatial_dimensions(output_spatial_dim);
      new_dim_nums.add_kernel_spatial_dimensions(kernel_spatial_dim);
    }
  }
  // We sort the removed dimensions into descending order, because we need to
  // delete higher dimensions first, otherwise we would have to adjust dimension
  // indices.
  std::sort(removed_input_dimensions.begin(), removed_input_dimensions.end(),
            std::greater<>());
  std::sort(removed_output_dimensions.begin(), removed_output_dimensions.end(),
            std::greater<>());
  std::sort(removed_kernel_dimensions.begin(), removed_kernel_dimensions.end(),
            std::greater<>());

  // Compute the new shapes.
  Shape new_input_shape = input->shape();
  for (int64_t dim : removed_input_dimensions) {
    new_input_shape.DeleteDimension(dim);
  }
  HloInstruction* kernel = instruction->mutable_operand(1);
  Shape new_kernel_shape = kernel->shape();
  for (int64_t dim : removed_kernel_dimensions) {
    new_kernel_shape.DeleteDimension(dim);
  }
  Shape new_output_shape = instruction->shape();
  for (int64_t dim : removed_output_dimensions) {
    new_output_shape.DeleteDimension(dim);
  }

  // Relabel the dimension numbers to account for the deleted dimensions. For
  // each dimension number, we need to reduce its value by the number of removed
  // smaller dimensions.
  auto compute_new_dimension =
      [](const std::vector<int64_t>& removed_dimensions,
         int64_t old_dimension) {
        int64_t num_smaller = absl::c_count_if(
            removed_dimensions, [old_dimension](int64_t removed_dimension) {
              return removed_dimension < old_dimension;
            });
        return old_dimension - num_smaller;
      };
  new_dim_nums.set_input_batch_dimension(compute_new_dimension(
      removed_input_dimensions, new_dim_nums.input_batch_dimension()));
  new_dim_nums.set_input_feature_dimension(compute_new_dimension(
      removed_input_dimensions, new_dim_nums.input_feature_dimension()));
  for (int64_t i = 0; i < new_dim_nums.input_spatial_dimensions().size(); ++i) {
    new_dim_nums.set_input_spatial_dimensions(
        i, compute_new_dimension(removed_input_dimensions,
                                 new_dim_nums.input_spatial_dimensions(i)));
  }
  new_dim_nums.set_output_batch_dimension(compute_new_dimension(
      removed_output_dimensions, new_dim_nums.output_batch_dimension()));
  new_dim_nums.set_output_feature_dimension(compute_new_dimension(
      removed_output_dimensions, new_dim_nums.output_feature_dimension()));
  for (int64_t i = 0; i < new_dim_nums.output_spatial_dimensions().size();
       ++i) {
    new_dim_nums.set_output_spatial_dimensions(
        i, compute_new_dimension(removed_output_dimensions,
                                 new_dim_nums.output_spatial_dimensions(i)));
  }
  new_dim_nums.set_kernel_input_feature_dimension(
      compute_new_dimension(removed_kernel_dimensions,
                            new_dim_nums.kernel_input_feature_dimension()));
  new_dim_nums.set_kernel_output_feature_dimension(
      compute_new_dimension(removed_kernel_dimensions,
                            new_dim_nums.kernel_output_feature_dimension()));
  for (int64_t i = 0; i < new_dim_nums.kernel_spatial_dimensions().size();
       ++i) {
    new_dim_nums.set_kernel_spatial_dimensions(
        i, compute_new_dimension(removed_kernel_dimensions,
                                 new_dim_nums.kernel_spatial_dimensions(i)));
  }

  // Reshape the input and the kernel.
  HloInstruction* reshaped_input = computation->AddInstruction(
      HloInstruction::CreateReshape(new_input_shape, input));
  HloInstruction* reshaped_kernel = computation->AddInstruction(
      HloInstruction::CreateReshape(new_kernel_shape, kernel));

  // We want to use CloneWithNewOperands, but that doesn't support substituting
  // the window and the ConvolutionDimensionNumbers. So we set this on the old
  // instruction (which is going to be removed anyway) before cloning it.
  instruction->set_convolution_dimension_numbers(new_dim_nums);
  instruction->set_window(new_window);
  HloInstruction* new_convolution =
      computation->AddInstruction(instruction->CloneWithNewOperands(
          new_output_shape, {reshaped_input, reshaped_kernel}));
  return computation->AddInstruction(
      HloInstruction::CreateReshape(instruction->shape(), new_convolution));
}

}  // namespace xla
