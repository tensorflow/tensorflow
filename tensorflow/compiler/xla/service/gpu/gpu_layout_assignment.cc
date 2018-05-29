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
#include "tensorflow/compiler/xla/service/gpu/gpu_options.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace gpu {

using stream_executor::dnn::DataLayout;
using stream_executor::dnn::FilterLayout;

static bool IsVoltaOrLater(const se::StreamExecutor& stream_executor) {
  int major, minor;
  CHECK(stream_executor.GetDeviceDescription().cuda_compute_capability(&major,
                                                                       &minor));
  return major >= 7;
}

// Returns (input, filter, output) layouts.
static std::tuple<DataLayout, FilterLayout, DataLayout>
HeuristicLayoutAssignment(const HloInstruction* instr,
                          stream_executor::StreamExecutor* stream_executor) {
  // DataLayout and FilterLayout uses weird enum names. Translations:
  //   N <=> Batch or Output
  //   C <=> Depth or Input
  //   H <=> Y
  //   W <=> X
  //
  // Therefore kOutputInputYX means NHWC; kBatchDepthYX means NCHW.

  // As of today, our empirical evidence is that cudnn 7.0 is faster on V100 x
  // fp16 with the mostly-NHWC layout. The heuristic may change as cudnn version
  // changes, as well as the hardware updates.
  if (!(instr->operand(0)->shape().element_type() == xla::PrimitiveType::F16 &&
        IsVoltaOrLater(*stream_executor))) {
    return std::make_tuple(DataLayout::kBatchDepthYX,
                           FilterLayout::kOutputInputYX,
                           DataLayout::kBatchDepthYX);
  }
  VLOG(2) << "Using heuristic to figure out layouts for " << instr->ToString();
  // For BackwardInput that has stride, full NHWC layouts run significantly
  // slower than (NHWC, NCHW, NCHW) or (NHWC, NCHW, NHWC).
  //
  // TODO(timshen): more closely compare (NHWC, NCHW, NCHW) and (NHWC, NCHW,
  // NHWC).
  if (instr->custom_call_target() == kCudnnConvBackwardInputCallTarget &&
      window_util::HasStride(instr->window())) {
    return std::make_tuple(DataLayout::kBatchYXDepth,
                           FilterLayout::kOutputInputYX,
                           DataLayout::kBatchDepthYX);
  }
  return std::make_tuple(DataLayout::kBatchYXDepth,
                         FilterLayout::kOutputYXInput,
                         DataLayout::kBatchYXDepth);
}

// Adds layout constraints on the cudnn custom-call instruction. The layout
// constraints are represented in terms of minor_to_major fields of both
// operands and the output shape. Depending on the underlying algorithm, one of
// { NCHW, NHWC } ^ 3 = 8 different layout combinations may be chosen.
Status GpuLayoutAssignment::AddBackendConstraintsToDnnConvCustomCall(
    HloInstruction* instr, LayoutConstraints* constraints) {
  CHECK(IsCustomCallToDnnConvolution(*instr)) << instr->ToString();
  Shape input_shape;
  Shape filter_shape;
  Shape output_shape;
  const auto& target = instr->custom_call_target();
  if (target == kCudnnConvForwardCallTarget) {
    input_shape = instr->operand(0)->shape();
    filter_shape = instr->operand(1)->shape();
    output_shape = instr->shape().tuple_shapes(0);
  } else if (target == kCudnnConvBackwardInputCallTarget) {
    input_shape = instr->shape().tuple_shapes(0);
    filter_shape = instr->operand(1)->shape();
    output_shape = instr->operand(0)->shape();
  } else if (target == kCudnnConvBackwardFilterCallTarget) {
    input_shape = instr->operand(0)->shape();
    filter_shape = instr->shape().tuple_shapes(0);
    output_shape = instr->operand(1)->shape();
  } else {
    LOG(FATAL) << "Unexpected custom call target: "
               << instr->custom_call_target();
  }

  {
    DataLayout input;
    FilterLayout filter;
    DataLayout output;
    if (ConvUseLayoutHeuristic(instr->GetModule()->config())) {
      std::tie(input, filter, output) =
          HeuristicLayoutAssignment(instr, stream_executor_);
    } else {
      input = DataLayout::kBatchDepthYX;
      filter = FilterLayout::kOutputInputYX;
      output = DataLayout::kBatchDepthYX;
    }

    TF_ASSIGN_OR_RETURN(
        std::tie(*input_shape.mutable_layout(), *filter_shape.mutable_layout(),
                 *output_shape.mutable_layout()),
        StreamExecutorConvLayoutsToXlaLayouts(
            instr->convolution_dimension_numbers(), input, filter, output));
  }

  // The custom call returns a tuple of (actual_result, scratch_buffer);
  // call_result_buf is the logical buffer for actual_result, the thing that
  // contains the result of the conv call.
  TF_ASSIGN_OR_RETURN(const LogicalBuffer* call_result_buf,
                      constraints->points_to_analysis().GetBufferDefinedAt(
                          instr, /*index=*/{0}));

  // Set layouts of the instructions' shapes.
  if (target == kCudnnConvForwardCallTarget) {
    TF_RETURN_IF_ERROR(constraints->SetOperandLayout(input_shape, instr, 0));
    TF_RETURN_IF_ERROR(constraints->SetOperandLayout(filter_shape, instr, 1));
    TF_RETURN_IF_ERROR(
        constraints->SetBufferLayout(output_shape.layout(), *call_result_buf));
  } else if (target == kCudnnConvBackwardInputCallTarget) {
    TF_RETURN_IF_ERROR(constraints->SetOperandLayout(output_shape, instr, 0));
    TF_RETURN_IF_ERROR(constraints->SetOperandLayout(filter_shape, instr, 1));
    TF_RETURN_IF_ERROR(
        constraints->SetBufferLayout(input_shape.layout(), *call_result_buf));
  } else if (target == kCudnnConvBackwardFilterCallTarget) {
    TF_RETURN_IF_ERROR(constraints->SetOperandLayout(input_shape, instr, 0));
    TF_RETURN_IF_ERROR(constraints->SetOperandLayout(output_shape, instr, 1));
    TF_RETURN_IF_ERROR(
        constraints->SetBufferLayout(filter_shape.layout(), *call_result_buf));
  } else {
    LOG(FATAL) << "Unexpected custom call target: "
               << instr->custom_call_target();
  }
  return Status::OK();
}

Status GpuLayoutAssignment::AddBackendConstraints(
    LayoutConstraints* constraints) {
  for (auto* instruction : constraints->computation()->instructions()) {
    if (IsCustomCallToDnnConvolution(*instruction)) {
      TF_RETURN_IF_ERROR(
          AddBackendConstraintsToDnnConvCustomCall(instruction, constraints));
    }
  }
  return Status::OK();
}

bool GpuLayoutAssignment::CustomCallRequiresMajorFirstLayout(
    const HloInstruction* instruction) {
  // - Inputs to cudnn batchnorm custom calls don't need the major-first layout
  //   (i.e. {n, n-1, ...0}) -- we can handle any layout.
  // - Inputs to cudnn convolution require custom layouts handled in
  //   AddBackendConstraints.
  return !IsCustomCallToDnnBatchNorm(*instruction) &&
         !IsCustomCallToDnnConvolution(*instruction);
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
