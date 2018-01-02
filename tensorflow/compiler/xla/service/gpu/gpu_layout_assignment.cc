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

#include "tensorflow/compiler/xla/service/gpu/gpu_layout_assignment.h"

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
  for (auto* instruction : constraints->computation()->instructions()) {
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
        output = instruction;
      } else {
        CHECK_EQ(HloOpcode::kFusion, instruction->opcode());
        switch (instruction->fusion_kind()) {
          case HloInstruction::FusionKind::kConvBackwardFilter:
            // filter = BackwardFilterConvolve(input, output)
            input = instruction->mutable_operand(0);
            filter = instruction;
            output = instruction->mutable_operand(1);
            break;
          case HloInstruction::FusionKind::kConvBackwardInput:
            // input = BackwardInputConvolve(output, filter)
            input = instruction;
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
      for (int i = dimension_numbers.input_spatial_dimensions_size() - 1;
           i >= 0; --i) {
        input_layout.push_back(dimension_numbers.input_spatial_dimensions(i));
      }
      input_layout.push_back(dimension_numbers.input_feature_dimension());
      input_layout.push_back(dimension_numbers.input_batch_dimension());
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
      for (int i = dimension_numbers.output_spatial_dimensions_size() - 1;
           i >= 0; --i) {
        output_layout.push_back(dimension_numbers.output_spatial_dimensions(i));
      }
      output_layout.push_back(dimension_numbers.output_feature_dimension());
      output_layout.push_back(dimension_numbers.output_batch_dimension());
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

bool GpuLayoutAssignment::CustomCallRequiresMajorFirstLayout(
    const HloInstruction* instruction) {
  // Inputs to cudnn batchnorm custom calls don't need the major-first layout
  // (i.e. {n, n-1, ...0}) -- we can handle any layout.
  return !IsCustomCallToDnnBatchNorm(*instruction);
}

Status GpuLayoutAssignment::PropagateOperandConstraint(
    const OperandLayoutConstraint& layout_constraint,
    LayoutConstraints* constraints) {
  const HloInstruction* instruction = layout_constraint.instruction();

  // cudnn batchnorm forward inference's result must have the same layout as its
  // operand 0.
  if (instruction->opcode() == HloOpcode::kCustomCall &&
      instruction->custom_call_target() ==
          kCudnnBatchNormForwardInferenceCallTarget &&
      layout_constraint.operand_no() == 0) {
    TF_RETURN_IF_ERROR(constraints->SetInstructionLayout(
        layout_constraint.shape_layout().shape(), instruction));
  }

  // cudnn batchnorm forward training returns a tuple {output, mean,
  // inverse-stddev}.  mean and inverse-stddev are rank 1 and so have only one
  // possible layout, but output is not (necessarily) rank 1, and, like in
  // batchnorm forward inference, must have the same layout as operand 0.
  if (instruction->opcode() == HloOpcode::kCustomCall &&
      instruction->custom_call_target() ==
          kCudnnBatchNormForwardTrainingCallTarget &&
      layout_constraint.operand_no() == 0) {
    TF_ASSIGN_OR_RETURN(const LogicalBuffer* out_buf,
                        constraints->points_to_analysis().GetBufferDefinedAt(
                            instruction, /*index=*/{0}));
    TF_RETURN_IF_ERROR(constraints->SetBufferLayout(
        layout_constraint.shape_layout().layout(), *out_buf));
  }

  // Like forward training, cudnn batchnorm backward returns a tuple {output,
  // mean, inverse-stddev}, and its operand 0 and 'output' must have the same
  // layout.  In addition, its operand 0 and operand 4 -- the 'operand' and
  // 'grad_output' parameters -- must have the same layout.
  if (instruction->opcode() == HloOpcode::kCustomCall &&
      instruction->custom_call_target() == kCudnnBatchNormBackwardCallTarget &&
      (layout_constraint.operand_no() == 0 ||
       layout_constraint.operand_no() == 4)) {
    TF_ASSIGN_OR_RETURN(const LogicalBuffer* out_buf,
                        constraints->points_to_analysis().GetBufferDefinedAt(
                            instruction, /*index=*/{0}));
    TF_RETURN_IF_ERROR(constraints->SetBufferLayout(
        layout_constraint.shape_layout().layout(), *out_buf));

    int64 operand_to_set = layout_constraint.operand_no() == 0 ? 4 : 0;
    TF_RETURN_IF_ERROR(constraints->SetOperandLayout(
        layout_constraint.shape_layout().shape(), instruction, operand_to_set));
  }

  return LayoutAssignment::PropagateOperandConstraint(layout_constraint,
                                                      constraints);
}

Status GpuLayoutAssignment::PropagateBufferConstraint(
    const BufferLayoutConstraint& buffer_constraint,
    LayoutConstraints* constraints) {
  const LogicalBuffer& buf = buffer_constraint.buffer();
  const HloInstruction* instruction = buf.instruction();

  Shape shape_with_layout = buf.shape();
  *shape_with_layout.mutable_layout() = buffer_constraint.layout();

  // Propagate output constraints to the operands of cudnn batchnorm ops.  This
  // is the same as PropagateOperandConstraint, just in the other direction.  We
  // need to both to fulfill our contract to LayoutAssignment.
  if (instruction->opcode() == HloOpcode::kCustomCall &&
      instruction->custom_call_target() ==
          kCudnnBatchNormForwardInferenceCallTarget) {
    TF_RETURN_IF_ERROR(constraints->SetOperandLayout(
        shape_with_layout, instruction, /*operand_no=*/0));
  }

  if (instruction->opcode() == HloOpcode::kCustomCall &&
      instruction->custom_call_target() ==
          kCudnnBatchNormForwardTrainingCallTarget &&
      buf.index() == ShapeIndex({0})) {
    TF_RETURN_IF_ERROR(constraints->SetOperandLayout(
        shape_with_layout, instruction, /*operand_no=*/0));
  }
  if (instruction->opcode() == HloOpcode::kCustomCall &&
      instruction->custom_call_target() == kCudnnBatchNormBackwardCallTarget &&
      buf.index() == ShapeIndex({0})) {
    // batchnorm backward has two operands, "operand" and "grad_output" whose
    // layouts must both match that of the result at tuple-index 0.
    TF_RETURN_IF_ERROR(constraints->SetOperandLayout(
        shape_with_layout, instruction, /*operand_no=*/0));
    TF_RETURN_IF_ERROR(constraints->SetOperandLayout(
        shape_with_layout, instruction, /*operand_no=*/4));
  }

  return LayoutAssignment::PropagateBufferConstraint(buffer_constraint,
                                                     constraints);
}

}  // namespace gpu
}  // namespace xla
