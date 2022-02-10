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
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace gpu {

using se::dnn::DataLayout;
using se::dnn::FilterLayout;

// Returns (input, filter, output) layouts.
static std::tuple<DataLayout, FilterLayout, DataLayout>
HeuristicLayoutAssignment(const HloInstruction* instr,
                          se::StreamExecutor* stream_executor) {
  // DataLayout and FilterLayout uses weird enum names. Translations:
  //   N <=> Batch or Output
  //   C <=> Depth or Input
  //   H <=> Y
  //   W <=> X
  //
  // Therefore kOutputInputYX and kBatchDepthYX mean NCHW.
  //
  // If you have trouble keeping these straight, consider that all that matters
  // is the location of the channel dim: Is it major (NCHW), or minor (NHWC)?

  constexpr auto kAllNCHW =
      std::make_tuple(DataLayout::kBatchDepthYX, FilterLayout::kOutputInputYX,
                      DataLayout::kBatchDepthYX);
  // kBatchDepthYX4 has the same layout as kBatchDepthYX32; they're both VECT_C
  // layouts as far as cudnn is concerned.
  constexpr auto kAllNCHW_VECT_C =
      std::make_tuple(DataLayout::kBatchDepthYX4, FilterLayout::kOutputInputYX4,
                      DataLayout::kBatchDepthYX4);
  constexpr auto kAllNHWC =
      std::make_tuple(DataLayout::kBatchYXDepth, FilterLayout::kOutputYXInput,
                      DataLayout::kBatchYXDepth);

  // Integer convolution must use NHWC or NCHW_VECT_C.
  //
  // TODO(jlebar): Do non-VECT_C int8_t convs still require NHWC with new
  // versions of cudnn?
  const ConvolutionDimensionNumbers& dnums =
      instr->convolution_dimension_numbers();
  Shape input_shape = instr->operand(0)->shape();
  PrimitiveType input_ty = instr->operand(0)->shape().element_type();
  if (primitive_util::IsIntegralType(input_ty)) {
    if (input_ty == S8 && dnums.input_spatial_dimensions_size() == 2 &&
        input_shape.dimensions_size() == 5) {
      VLOG(2) << "Using NCHW_VECT_C for int8_t conv " << instr->ToString();
      return kAllNCHW_VECT_C;
    }
    VLOG(2) << "Using NHWC for int8_t conv " << instr->ToString();
    return kAllNHWC;
  }

  const DebugOptions& debug_options =
      instr->GetModule()->config().debug_options();

  if (debug_options.xla_gpu_force_conv_nchw()) {
    VLOG(2) << "Overriding layout to NCHW for " << instr->ToString();
    return kAllNCHW;
  }

  if (debug_options.xla_gpu_force_conv_nhwc()) {
    VLOG(2) << "Overriding layout to NHWC for " << instr->ToString();
    return kAllNHWC;
  }

  // If we're not Volta or not fp16, or not conv2D, the decision is easy: Use
  // NCHW.
  if (input_ty != F16 ||
      !stream_executor->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeast(se::CudaComputeCapability::VOLTA) ||
      instr->shape().tuple_shapes(0).dimensions_size() != 4) {
    return kAllNCHW;
  }

  VLOG(2) << "Using heuristic to figure out layouts for " << instr->ToString();

  // Empirically we've found with Volta and cudnn <= 7.3 that backward-input
  // convs with stride are significantly faster with NCHW layouts.
  //
  // We could have used a mixed layout combination, e.g. (NHWC, NCHW, NCHW),
  // which on paper gives good performance. However, there are two observations:
  // * a mixed layout combination is more cuDNN-bug prone, based on empirical
  //   evidence.
  // * we've also observed that for mixed layouts, cuDNN transposes data back
  //   and forth from a different layout combination. If we end up with
  //   transposes anyway, we prefer to have them in XLA, as they can be fused.
  if (auto* dnn = stream_executor->AsDnn()) {
    auto version_status = dnn->GetVersion();
    if (version_status.ok()) {
      auto version = version_status.ConsumeValueOrDie();
      if (std::make_tuple(version.major_version(), version.minor_version()) <=
              std::make_tuple(7, 3) &&
          instr->custom_call_target() == kCudnnConvBackwardInputCallTarget &&
          window_util::HasStride(instr->window())) {
        return kAllNCHW;
      }
    }
  }

  // For other Volta f16 convolutions, use NHWC.
  return kAllNHWC;
}

// Adds layout constraints on the cudnn custom-call instruction. The layout
// constraints are represented in terms of minor_to_major fields of both
// operands and the output shape. Depending on the underlying algorithm, one of
// { NCHW, NHWC } ^ 3 = 8 different layout combinations may be chosen.
Status GpuLayoutAssignment::AddBackendConstraintsToDnnConvCustomCall(
    HloCustomCallInstruction* instr, LayoutConstraints* constraints) {
  Shape lhs_shape = instr->operand(0)->shape();
  Shape rhs_shape = instr->operand(1)->shape();
  Shape result_shape = instr->shape().tuple_shapes(0);

  Shape* input_shape;
  Shape* filter_shape;
  Shape* output_shape;

  TF_ASSIGN_OR_RETURN(auto kind, GetCudnnConvKind(instr));
  switch (kind) {
    case CudnnConvKind::kForward:
    case CudnnConvKind::kForwardActivation:
      input_shape = &lhs_shape;
      filter_shape = &rhs_shape;
      output_shape = &result_shape;
      break;
    case CudnnConvKind::kBackwardInput:
      input_shape = &result_shape;
      filter_shape = &rhs_shape;
      output_shape = &lhs_shape;
      break;
    case CudnnConvKind::kBackwardFilter:
      input_shape = &lhs_shape;
      filter_shape = &result_shape;
      output_shape = &rhs_shape;
      break;
  }

  {
    DataLayout input;
    FilterLayout filter;
    DataLayout output;
    std::tie(input, filter, output) =
        HeuristicLayoutAssignment(instr, stream_executor_);

    TF_ASSIGN_OR_RETURN(
        std::tie(*input_shape->mutable_layout(),
                 *filter_shape->mutable_layout(),
                 *output_shape->mutable_layout()),
        StreamExecutorConvLayoutsToXlaLayouts(
            instr->convolution_dimension_numbers(), input, filter, output));
  }

  // The custom call returns a tuple of (actual_result, scratch_buffer);
  // call_result_buf is the logical buffer for actual_result, the thing that
  // contains the result of the conv call.
  TF_ASSIGN_OR_RETURN(
      const LogicalBuffer* call_result_buf,
      points_to_analysis_->GetBufferDefinedAt(instr, /*index=*/{0}));

  // Set layouts of the instructions' shapes.
  TF_RETURN_IF_ERROR(SetOperandLayout(lhs_shape, instr, 0));
  TF_RETURN_IF_ERROR(SetOperandLayout(rhs_shape, instr, 1));
  TF_RETURN_IF_ERROR(SetBufferLayout(result_shape.layout(), *call_result_buf));
  // instr->operand(2), if exists, is the bias buffer. There is no need to
  // assign layout to it, as it has only one dimension.

  // instr->operand(3), if exists, is the side input buffer.
  if (instr->operand_count() == 4) {
    if (kind != CudnnConvKind::kForwardActivation) {
      return InternalError(
          "Invalid convolution. Conv has a side input, but kind is not fused "
          "conv forward: %s",
          instr->ToString());
    }
    // The side input layout must match the output layout.
    TF_RETURN_IF_ERROR(SetOperandLayout(*output_shape, instr, 3));
  }
  return Status::OK();
}

// Imposes the default layout with first two dimensions swapped on input
// `shape`.
static void SetFortranLayout(Shape* shape) {
  LayoutUtil::SetToDefaultLayout(shape);
  int n = shape->mutable_layout()->minor_to_major_size();
  CHECK_GE(n, 2);
  std::swap(shape->mutable_layout()->mutable_minor_to_major()->at(0),
            shape->mutable_layout()->mutable_minor_to_major()->at(1));
}

Status GpuLayoutAssignment::AddBackendConstraints(
    LayoutConstraints* constraints) {
  // Add convolution constraints in reverse postorder that the earliest
  // convolution layout propagates first. This reduces the likelihood of fusion
  // nodes with copies.
  auto post_order = constraints->computation()->MakeInstructionPostOrder();
  for (auto iterator = post_order.rbegin(); iterator != post_order.rend();
       ++iterator) {
    HloInstruction* instruction = *iterator;
    if (IsCustomCallToDnnConvolution(*instruction)) {
      TF_RETURN_IF_ERROR(AddBackendConstraintsToDnnConvCustomCall(
          Cast<HloCustomCallInstruction>(instruction), constraints));
    }

    CHECK(!IsCublasGemm(*instruction))
        << "Gemm rewriting should run after layout assignment";

    // For unbatched S8xS8->S32 matrix multiplication enforce a TN layout, which
    // will allow the NVidia GPUs to use TensorCores.
    if (IsMatrixMultiplication(*instruction)) {
      Shape output_shape = instruction->shape();
      Shape p1_shape = instruction->operand(0)->shape();
      Shape p2_shape = instruction->operand(1)->shape();
      if (output_shape.element_type() == PrimitiveType::S32 &&
          p1_shape.element_type() == PrimitiveType::S8 &&
          p2_shape.element_type() == PrimitiveType::S8 &&
          output_shape.dimensions_size() == 2 &&
          p1_shape.dimensions_size() == 2 && p2_shape.dimensions_size() == 2) {
        LayoutUtil::SetToDefaultLayout(&p1_shape);
        SetFortranLayout(&p2_shape);
        LayoutUtil::SetToDefaultLayout(&output_shape);
        TF_RETURN_IF_ERROR(SetOperandLayout(p1_shape, instruction, 0));
        TF_RETURN_IF_ERROR(SetOperandLayout(p2_shape, instruction, 1));
        TF_RETURN_IF_ERROR(SetInstructionLayout(output_shape, instruction));
        continue;
      }
    }

    // For batched dot we require the default layout.
    // TODO(b/112111608): This is overly conservative, the only real restriction
    // is that batch dimensions must be major.
    if (IsMatrixMultiplication(*instruction) &&
        instruction->dot_dimension_numbers().lhs_batch_dimensions_size() > 0) {
      // Verify that the batch dims come before the row and col dims.
      DotDimensionNumbers dim_nums = instruction->dot_dimension_numbers();
      CHECK_EQ(dim_nums.lhs_batch_dimensions_size(),
               dim_nums.rhs_batch_dimensions_size());
      CHECK_EQ(dim_nums.lhs_batch_dimensions_size() + 2,
               instruction->shape().rank());
      for (int64_t batch_dim : dim_nums.lhs_batch_dimensions()) {
        CHECK_LT(batch_dim, instruction->shape().rank() - 2);
      }

      // Set both inputs and the output to default layout.
      Shape op0_shape = instruction->operand(0)->shape();
      LayoutUtil::SetToDefaultLayout(&op0_shape);
      Shape op1_shape = instruction->operand(1)->shape();
      LayoutUtil::SetToDefaultLayout(&op1_shape);
      Shape output_shape = instruction->shape();
      LayoutUtil::SetToDefaultLayout(&output_shape);
      TF_RETURN_IF_ERROR(SetOperandLayout(op0_shape, instruction, 0));
      TF_RETURN_IF_ERROR(SetOperandLayout(op1_shape, instruction, 1));
      TF_RETURN_IF_ERROR(SetInstructionLayout(output_shape, instruction));
    } else if (instruction->opcode() == HloOpcode::kFft) {
      // cuFFT requires a dim0 major layout.
      Shape op0_shape = instruction->operand(0)->shape();
      LayoutUtil::SetToDefaultLayout(&op0_shape);
      Shape output_shape = instruction->shape();
      LayoutUtil::SetToDefaultLayout(&output_shape);
      TF_RETURN_IF_ERROR(SetOperandLayout(op0_shape, instruction, 0));
      TF_RETURN_IF_ERROR(SetInstructionLayout(output_shape, instruction));
    } else if (instruction->opcode() == HloOpcode::kSort &&
               instruction->operand(0)->shape().rank() > 1) {
      // Make sure that all the operands and the output(s) have the same layout.
      Shape keys_shape = instruction->operand(0)->shape();
      Layout keys_layout =
          LayoutUtil::GetDefaultLayoutForRank(keys_shape.rank());
      for (int64_t i = 0; i < instruction->operand_count(); ++i) {
        Shape shape = instruction->operand(i)->shape();
        *shape.mutable_layout() = keys_layout;
        TF_RETURN_IF_ERROR(SetOperandLayout(shape, instruction, i));
        const LogicalBuffer* output_buffer;
        if (instruction->shape().IsArray()) {
          TF_ASSIGN_OR_RETURN(
              output_buffer,
              points_to_analysis_->GetBufferDefinedAt(instruction, {}));
        } else {
          TF_ASSIGN_OR_RETURN(
              output_buffer,
              points_to_analysis_->GetBufferDefinedAt(instruction, {i}));
        }
        TF_RETURN_IF_ERROR(SetBufferLayout(keys_layout, *output_buffer));
      }
    } else if (instruction->opcode() == HloOpcode::kTriangularSolve) {
      // TODO(phawkins): Ideally we would relax this constraint. What we
      // actually want is that:
      // a) the batch dimensions are major, in no particular order.
      // b) the two minor dimensions are in fortran (column-major) order,
      // although for the 'a' argument we could potentially accept row-major
      // order and fold the transpose into the operator.
      Shape op0_shape = instruction->operand(0)->shape();
      Shape op1_shape = instruction->operand(1)->shape();
      Shape output_shape = instruction->shape();
      SetFortranLayout(&op0_shape);
      SetFortranLayout(&op1_shape);
      SetFortranLayout(&output_shape);
      TF_RETURN_IF_ERROR(SetOperandLayout(op0_shape, instruction, 0));
      TF_RETURN_IF_ERROR(SetOperandLayout(op1_shape, instruction, 1));
      TF_RETURN_IF_ERROR(SetInstructionLayout(output_shape, instruction));
    } else if (instruction->opcode() == HloOpcode::kReduceScatter) {
      // XLA:GPU can only support reduce-scatter where the scatter dimension
      // is the most major dimension in the layout.
      auto ars = Cast<HloReduceScatterInstruction>(instruction);
      TF_RETURN_IF_ERROR(SetInstructionLayout(
          ShapeUtil::MoveDimToMajor(ars->shape(), ars->scatter_dimension()),
          ars));
    } else if (instruction->opcode() == HloOpcode::kAllGather) {
      // XLA:GPU can only support all-gathers where the gather dimension is the
      // most major dimension in the layout.
      auto ag = Cast<HloAllGatherInstruction>(instruction);
      TF_RETURN_IF_ERROR(SetInstructionLayout(
          ShapeUtil::MoveDimToMajor(ag->shape(), ag->all_gather_dimension()),
          ag));
    } else if (instruction->opcode() == HloOpcode::kAllToAll &&
               instruction->shape().IsArray()) {
      // XLA:GPU can only support all-to-all with split dimensions where the
      // split dimension is the most major dimension in the layout.
      auto* all_to_all = Cast<HloAllToAllInstruction>(instruction);
      TF_RETURN_IF_ERROR(SetInstructionLayout(
          ShapeUtil::MoveDimToMajor(all_to_all->shape(),
                                    *all_to_all->split_dimension()),
          all_to_all));
    }
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
