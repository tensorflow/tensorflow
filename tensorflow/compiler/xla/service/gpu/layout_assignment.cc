/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/layout_assignment.h"

#include <memory>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace gpu {

Status GpuLayoutAssignment::AddBackendConstraints(
    LayoutConstraints* constraints) {
  for (auto& instruction : constraints->computation()->instructions()) {
    // cuDNN is called with specific layouts on the input, output, and filter:
    //
    //   input: DataLayout::kBatchDepthYX
    //   output: DataLayout::kBatchDepthYX
    //   filter: FilterLayout::kOutputInputYX
    //
    // The order dimensions in the constant name is major-to-minor (eg, the
    // most-major dimension of the input is batch, most-minor is X). The
    // specific dimension numbers these named dimensions correspond to is
    // determined by the ConvolutionDimensionNumbers argument. Y is spatial
    // dimension 0, and X is spatial dimension 1.
    //
    // TODO(b/29399649): Be more flexible about handling layouts of cuDNN calls.
    if (ImplementedAsDnnConvolution(*instruction)) {
      HloInstruction* input = nullptr;
      HloInstruction* filter = nullptr;
      HloInstruction* output = nullptr;
      if (instruction->opcode() == HloOpcode::kConvolution) {
        input = instruction->mutable_operand(0);
        filter = instruction->mutable_operand(1);
        output = instruction.get();
      } else {
        CHECK_EQ(HloOpcode::kFusion, instruction->opcode());
        switch (instruction->fusion_kind()) {
          case HloInstruction::FusionKind::kConvBackwardFilter:
            // filter = BackwardFilterConvolve(input, output)
            input = instruction->mutable_operand(0);
            filter = instruction.get();
            output = instruction->mutable_operand(1);
            break;
          case HloInstruction::FusionKind::kConvBackwardInput:
            // input = BackwardInputConvolve(output, filter)
            input = instruction.get();
            filter = instruction->mutable_operand(1);
            output = instruction->mutable_operand(0);
            break;
          default:
            LOG(FATAL) << "Not a convolution-fusion";
        }
      }

      // Construct minor-to-major dimension orders for operands and result.
      // cuDNN's convolution APIs support the BDYX layout for activations/output
      // and the OIYX layout for weights.
      // TODO(b/29399649): Be more flexible about handling layouts of cuDNN
      // calls after we switch to cuDNN v5.
      const ConvolutionDimensionNumbers& dimension_numbers =
          instruction->convolution_dimension_numbers();
      std::vector<int64> input_layout;
      for (int i = dimension_numbers.spatial_dimensions_size() - 1; i >= 0;
           --i) {
        input_layout.push_back(dimension_numbers.spatial_dimensions(i));
      }
      input_layout.push_back(dimension_numbers.feature_dimension());
      input_layout.push_back(dimension_numbers.batch_dimension());
      Shape input_shape(input->shape());
      *input_shape.mutable_layout() = LayoutUtil::MakeLayout(input_layout);

      std::vector<int64> filter_layout;
      for (int i = dimension_numbers.kernel_spatial_dimensions_size() - 1;
           i >= 0; --i) {
        filter_layout.push_back(dimension_numbers.kernel_spatial_dimensions(i));
      }
      filter_layout.push_back(
          dimension_numbers.kernel_input_feature_dimension());
      filter_layout.push_back(
          dimension_numbers.kernel_output_feature_dimension());
      Shape filter_shape(filter->shape());
      *filter_shape.mutable_layout() = LayoutUtil::MakeLayout(filter_layout);

      std::vector<int64> output_layout;
      for (int i = dimension_numbers.spatial_dimensions_size() - 1; i >= 0;
           --i) {
        output_layout.push_back(dimension_numbers.spatial_dimensions(i));
      }
      output_layout.push_back(dimension_numbers.feature_dimension());
      output_layout.push_back(dimension_numbers.batch_dimension());
      Shape output_shape(output->shape());
      *output_shape.mutable_layout() = LayoutUtil::MakeLayout(output_layout);

      // Set layouts of the instructions' shapes.
      if (instruction->opcode() == HloOpcode::kConvolution) {
        TF_RETURN_IF_ERROR(
            constraints->SetOperandLayout(input_shape, output, 0));
        TF_RETURN_IF_ERROR(
            constraints->SetOperandLayout(filter_shape, output, 1));
        TF_RETURN_IF_ERROR(
            constraints->SetInstructionLayout(output_shape, output));
      } else {
        CHECK_EQ(HloOpcode::kFusion, instruction->opcode());
        switch (instruction->fusion_kind()) {
          case HloInstruction::FusionKind::kConvBackwardFilter:
            // filter = BackwardFilterConvolve(input, output)
            TF_RETURN_IF_ERROR(
                constraints->SetOperandLayout(input_shape, filter, 0));
            TF_RETURN_IF_ERROR(
                constraints->SetInstructionLayout(filter_shape, filter));
            TF_RETURN_IF_ERROR(
                constraints->SetOperandLayout(output_shape, filter, 1));
            break;
          case HloInstruction::FusionKind::kConvBackwardInput:
            // input = BackwardInputConvolve(output, filter)
            TF_RETURN_IF_ERROR(
                constraints->SetInstructionLayout(input_shape, input));
            TF_RETURN_IF_ERROR(
                constraints->SetOperandLayout(output_shape, input, 0));
            TF_RETURN_IF_ERROR(
                constraints->SetOperandLayout(filter_shape, input, 1));
            break;
          default:
            LOG(FATAL) << "Not a convolution-fusion";
        }
      }
    }
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
