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
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/errors.h"

#include "tensorflow/tsl/util/env_var.h"

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

  // If we're not Volta or not fp16/bfloat16, or not conv2D, the decision is
  // easy: Use NCHW.
  const bool isFloat16 = (input_ty == F16) || (input_ty == BF16);
#if GOOGLE_CUDA
  if (!isFloat16 ||
      !stream_executor->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeast(se::CudaComputeCapability::VOLTA) ||
      instr->shape().tuple_shapes(0).dimensions_size() != 4) {
    return kAllNCHW;
  }
#elif TENSORFLOW_USE_ROCM
  bool is_enabled = false;
  TF_CHECK_OK(tsl::ReadBoolFromEnvVar(
      "TF_USE_ROCM_NHWC",
      /*default_val=*/false, &is_enabled));
  auto rocm_compute_capability =
      stream_executor->GetDeviceDescription().rocm_compute_capability();
  if (!isFloat16 || (!rocm_compute_capability.has_nhwc_layout_support()) ||
      instr->shape().tuple_shapes(0).dimensions_size() != 4 || !is_enabled) {
    return kAllNCHW;
  }
#endif

  VLOG(2) << "Using heuristic to figure out layouts for " << instr->ToString();

#if GOOGLE_CUDA
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
      auto version = std::move(version_status).value();
      if (std::make_tuple(version.major_version(), version.minor_version()) <=
              std::make_tuple(7, 3) &&
          instr->custom_call_target() == kCudnnConvBackwardInputCallTarget &&
          window_util::HasStride(instr->window())) {
        return kAllNCHW;
      }
    }
  }
#endif

  // For other Volta/MI100(200) f16 convolutions, use NHWC.
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
  return OkStatus();
}

namespace {

// Imposes the default layout with first two dimensions swapped on input
// `shape`.
void SetFortranLayout(Shape* shape) {
  LayoutUtil::SetToDefaultLayout(shape);
  int n = shape->mutable_layout()->minor_to_major_size();
  CHECK_GE(n, 2);
  std::swap(shape->mutable_layout()->mutable_minor_to_major()->at(0),
            shape->mutable_layout()->mutable_minor_to_major()->at(1));
}

bool DotCanSupportShapeWithLayout(const HloInstruction* dot,
                                  const Shape& shape) {
  const DotDimensionNumbers& dot_dims = dot->dot_dimension_numbers();
  // If we are able to construct a `MatrixLayout` then the dot can support
  // this layout.
  return MatrixLayout::For(shape, dot_dims.lhs_batch_dimensions().size(),
                           dot->operand(0)->shape().rank() -
                               dot_dims.lhs_contracting_dimensions().size() -
                               dot_dims.lhs_batch_dimensions().size(),
                           dot_dims.rhs_batch_dimensions().size(),
                           dot->operand(1)->shape().rank() -
                               dot_dims.rhs_contracting_dimensions().size() -
                               dot_dims.rhs_batch_dimensions().size())
      .ok();
}

}  // namespace

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

    if (instruction->opcode() == HloOpcode::kDot) {
      const Shape& output_shape = instruction->shape();
      const Shape& lhs_shape = instruction->operand(0)->shape();
      const Shape& rhs_shape = instruction->operand(1)->shape();
      const DotDimensionNumbers& dot_dims =
          instruction->dot_dimension_numbers();

      // Matmuls require the batch dimensions to be in consecutive physical
      // dimensions and likewise for the contracting and non-contracting
      // dimensions. Additionally, no batch dimension can be in the most
      // minor physical dimension for inputs or the output.
      absl::Span<const int64_t> lhs_batch_dims =
          dot_dims.lhs_batch_dimensions();
      absl::Span<const int64_t> lhs_contracting_dims =
          dot_dims.lhs_contracting_dimensions();
      TF_ASSIGN_OR_RETURN(std::vector<int64_t> lhs_non_contracting_dims,
                          GetNonContractingDims(lhs_shape, lhs_batch_dims,
                                                lhs_contracting_dims));

      absl::Span<const int64_t> rhs_batch_dims =
          dot_dims.rhs_batch_dimensions();
      absl::Span<const int64_t> rhs_contracting_dims =
          dot_dims.rhs_contracting_dimensions();
      TF_ASSIGN_OR_RETURN(std::vector<int64_t> rhs_non_contracting_dims,
                          GetNonContractingDims(rhs_shape, rhs_batch_dims,
                                                rhs_contracting_dims));

      // For unbatched S8xS8->S32 matrix multiplication enforce a TN layout,
      // which will allow the NVidia GPUs to use TensorCores.
      bool is_s8_to_s32 = (output_shape.element_type() == PrimitiveType::S32 &&
                           lhs_shape.element_type() == PrimitiveType::S8 &&
                           rhs_shape.element_type() == PrimitiveType::S8 &&
                           output_shape.dimensions_size() == 2 &&
                           lhs_shape.dimensions_size() == 2 &&
                           rhs_shape.dimensions_size() == 2);

      if (is_s8_to_s32) {
        TF_RETURN_IF_ERROR(SetOperandBatchRowsColsLayout(
            instruction, 0, lhs_batch_dims, lhs_non_contracting_dims,
            lhs_contracting_dims));
        TF_RETURN_IF_ERROR(SetOperandBatchRowsColsLayout(
            instruction, 1, rhs_batch_dims, rhs_non_contracting_dims,
            rhs_contracting_dims));
        TF_RETURN_IF_ERROR(SetDotLayout(instruction, constraints));
      } else {
        if (!lhs_batch_dims.empty() || lhs_contracting_dims.size() > 1 ||
            lhs_non_contracting_dims.size() > 1) {
          TF_RETURN_IF_ERROR(SetDotOperandLayout(instruction, 0, lhs_batch_dims,
                                                 lhs_contracting_dims,
                                                 lhs_non_contracting_dims));
        }
        if (!rhs_batch_dims.empty() || rhs_non_contracting_dims.size() > 1 ||
            rhs_contracting_dims.size() > 1) {
          TF_RETURN_IF_ERROR(SetDotOperandLayout(instruction, 1, rhs_batch_dims,
                                                 rhs_contracting_dims,
                                                 rhs_non_contracting_dims));
        }
        // If we have at least one batch dimension or there is more than one
        // non-contracting dimension on lhs or rhs, we need to set a layout for
        // the dot output.
        if (!lhs_batch_dims.empty() || lhs_non_contracting_dims.size() > 1 ||
            rhs_non_contracting_dims.size() > 1) {
          TF_RETURN_IF_ERROR(SetDotLayout(instruction, constraints));
        }
      }
    } else if (instruction->opcode() == HloOpcode::kTranspose) {
      const HloInstruction* operand = instruction->operand(0);
      if ((operand->opcode() != HloOpcode::kDot) ||
          (operand->user_count() > 1)) {
        continue;
      }

      // If possible, set layout of the dot operation such that the output of
      // the transpose (as a bitcast) has the default layout.
      Shape shape = operand->shape();
      *shape.mutable_layout() =
          LayoutUtil::MakeLayoutFromMajorToMinor(instruction->dimensions());

      if (DotCanSupportShapeWithLayout(operand, shape)) {
        TF_RETURN_IF_ERROR(
            SetOperandLayout(shape, instruction, /*operand_no=*/0));
      }
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
  return OkStatus();
}

Status GpuLayoutAssignment::SetDotOperandLayout(
    const HloInstruction* instruction, int64_t operand,
    absl::Span<const int64_t> batch_dims, absl::Span<const int64_t> row_dims,
    absl::Span<const int64_t> col_dims) {
  Shape shape = instruction->operand(operand)->shape();

  // First, try to use the existing layout, if present.
  if (shape.has_layout() &&
      MatrixLayout::For(shape, batch_dims, row_dims, col_dims).ok())
    // Re-set the operand layout, so it becomes mandatory.
    return SetOperandLayout(shape, instruction, operand);

  // Next, try the default layout (for the sake of everybody's sanity).
  LayoutUtil::SetToDefaultLayout(&shape);
  if (MatrixLayout::For(shape, batch_dims, row_dims, col_dims).ok())
    return SetOperandLayout(shape, instruction, operand);

  // Otherwise, fallback to forcing (batch, rows, cols) layout.
  return SetOperandBatchRowsColsLayout(instruction, operand, batch_dims,
                                       row_dims, col_dims);
}

Status GpuLayoutAssignment::SetOperandBatchRowsColsLayout(
    const HloInstruction* instruction, int64_t operand,
    absl::Span<const int64_t> batch_dims, absl::Span<const int64_t> row_dims,
    absl::Span<const int64_t> col_dims) {
  std::vector<int64_t> major_to_minor;
  major_to_minor.reserve(batch_dims.size() + row_dims.size() + col_dims.size());
  major_to_minor.insert(major_to_minor.end(), batch_dims.begin(),
                        batch_dims.end());
  major_to_minor.insert(major_to_minor.end(), row_dims.begin(), row_dims.end());
  major_to_minor.insert(major_to_minor.end(), col_dims.begin(), col_dims.end());

  Shape shape = instruction->operand(operand)->shape();
  *shape.mutable_layout() =
      LayoutUtil::MakeLayoutFromMajorToMinor(major_to_minor);
  return SetOperandLayout(shape, instruction, operand);
}

Status GpuLayoutAssignment::SetDotLayout(const HloInstruction* instruction,
                                         LayoutConstraints* constraints) {
  // If a user has requested a layout that we can support, use that.
  for (const HloInstruction* user : instruction->users()) {
    for (int64_t i = 0; i < user->operand_count(); ++i) {
      if (user->operand(i) != instruction) {
        continue;
      }

      const ShapeLayout* constraint = constraints->OperandLayout(user, i);
      if ((constraint != nullptr) &&
          DotCanSupportShapeWithLayout(instruction, constraint->shape())) {
        return SetInstructionLayout(constraint->shape(), instruction);
      }
    }
  }

  // Otherwise, use the default layout.
  return SetInstructionLayout(
      LayoutUtil::GetWithDefaultLayout(instruction->shape()), instruction);
}

}  // namespace gpu
}  // namespace xla
