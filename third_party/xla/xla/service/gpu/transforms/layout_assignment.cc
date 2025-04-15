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

#include "xla/service/gpu/transforms/layout_assignment.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_indexing_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/memory_annotations.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/env_var.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

using se::dnn::DataLayout;
using se::dnn::FilterLayout;

// Returns (input, filter, output) layouts.
static std::tuple<DataLayout, FilterLayout, DataLayout>
HeuristicLayoutAssignment(const HloInstruction* instr,
                          const se::GpuComputeCapability& gpu_version,
                          const se::dnn::VersionInfo& dnn_version) {
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
  int num_spatial_dimensions = dnums.input_spatial_dimensions_size();
  if (primitive_util::IsIntegralType(input_ty)) {
    if (input_ty == S8 && num_spatial_dimensions == 2 &&
        input_shape.dimensions().size() == 5) {
      VLOG(2) << "Using NCHW_VECT_C for int8_t conv " << instr->ToString();
      return kAllNCHW_VECT_C;
    }
    VLOG(2) << "Using NHWC for int8_t conv " << instr->ToString();
    return kAllNHWC;
  }

  if (primitive_util::IsF8Type(input_ty)) {
    VLOG(2) << "Using NHWC for FP8 conv " << instr->ToString();
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

  // Despite the specialized logic below for Volta, we expect GPUs with Tensor
  // Cores work best using NHWC layouts for cuDNN convolutions---as per
  // https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout.
  if (auto* cc = std::get_if<se::CudaComputeCapability>(&gpu_version)) {
    // TODO(b/383560056): investigate chips below Hopper as well.
    if (cc->IsAtLeast(se::CudaComputeCapability::kHopper)) {
      // With that said, cuDNN's documentation states that NHWC is not supported
      // for float64, so we use NCHW instead.
      if (input_ty == F64) {
        VLOG(2) << "Using NCHW for F64 conv " << instr->ToString() << " on "
                << cc->ToString();
        return kAllNCHW;
        // TODO(b/383560056): find the right filter for 3D convolutions. 3D
        // convolutions also have a much smaller surface of support. We filter
        // them out completely as well for now.
      } else if (num_spatial_dimensions > 2) {
        VLOG(2) << "Using NHWC for " << num_spatial_dimensions << "D conv "
                << instr->ToString() << " on " << cc->ToString();
        return kAllNCHW;
      } else {
        return kAllNHWC;
      }
    }
  }

  const auto* rocm_compute_capability =
      std::get_if<se::RocmComputeCapability>(&gpu_version);
  if (rocm_compute_capability && input_ty == F16) return kAllNHWC;

  // If we're not Volta or not fp16/bfloat16, or not conv2D, the decision is
  // easy: Use NCHW.
  const bool isFloat16 = (input_ty == F16) || (input_ty == BF16);
  if (std::holds_alternative<se::CudaComputeCapability>(gpu_version)) {
    // If we're not Volta or not fp16/bfloat16, or not conv2D, the decision is
    // easy: Use NCHW.
    const auto* cuda_compute_capability =
        std::get_if<se::CudaComputeCapability>(&gpu_version);
    bool is_volta =
        cuda_compute_capability &&
        cuda_compute_capability->IsAtLeast(se::CudaComputeCapability::kVolta);
    if (!isFloat16 || !is_volta ||
        instr->shape().tuple_shapes(0).dimensions().size() != 4) {
      return kAllNCHW;
    }
  } else if (std::holds_alternative<se::RocmComputeCapability>(gpu_version)) {
    bool is_enabled = false;
    TF_CHECK_OK(tsl::ReadBoolFromEnvVar("TF_USE_ROCM_NHWC",
                                        /*default_val=*/false, &is_enabled));
    auto rocm_compute_capability =
        std::get<se::RocmComputeCapability>(gpu_version);
    if (!isFloat16 || (!rocm_compute_capability.has_nhwc_layout_support()) ||
        instr->shape().tuple_shapes(0).dimensions().size() != 4 ||
        !is_enabled) {
      return kAllNCHW;
    }
  }

  VLOG(2) << "Using heuristic to figure out layouts for " << instr->ToString();

  // For other Volta f16 convolutions, use NHWC.
  return kAllNHWC;
}

// Adds layout constraints on the cudnn custom-call instruction. The layout
// constraints are represented in terms of minor_to_major fields of both
// operands and the output shape. Depending on the underlying algorithm, one of
// { NCHW, NHWC } ^ 3 = 8 different layout combinations may be chosen.
absl::Status GpuLayoutAssignment::AddBackendConstraintsToDnnConvCustomCall(
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
    case CudnnConvKind::kForwardGraph:
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
        HeuristicLayoutAssignment(instr, gpu_version_, dnn_version_);

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
  // For fused convolutions, instr->operand(2), if exists, is the bias buffer.
  // There is no need to assign layout to it, as it has only one dimension.
  // instr->operand(3), if exists, is the side input buffer.
  if (kind == CudnnConvKind::kForwardActivation &&
      instr->operand_count() == 4) {
    // The side input layout must match the output layout.
    TF_RETURN_IF_ERROR(SetOperandLayout(*output_shape, instr, 3));
  }

  // For graph convolutions, align the layouts of the non-scalar inputs to any
  // pointwise ops with the output layout.
  if (kind == CudnnConvKind::kForwardGraph) {
    for (int k = 2; k < instr->operand_count(); ++k) {
      if (!ShapeUtil::IsScalar(instr->operand(k)->shape())) {
        TF_RETURN_IF_ERROR(SetOperandLayout(*output_shape, instr, k));
      }
    }
  }

  if (instr->operand_count() > 2 && kind != CudnnConvKind::kForwardActivation &&
      kind != CudnnConvKind::kForwardGraph) {
    return Internal(
        "Invalid convolution. Conv has a side input, but kind is not fused "
        "conv forward or graph conv foward: %s",
        instr->ToString());
  }

  return absl::OkStatus();
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
                           dot->operand(0)->shape().dimensions().size() -
                               dot_dims.lhs_contracting_dimensions().size() -
                               dot_dims.lhs_batch_dimensions().size(),
                           dot_dims.rhs_batch_dimensions().size(),
                           dot->operand(1)->shape().dimensions().size() -
                               dot_dims.rhs_contracting_dimensions().size() -
                               dot_dims.rhs_batch_dimensions().size())
      .ok();
}

bool IsPackedInstruction(const HloInstruction* instruction) {
  return primitive_util::IsSubByteNonPredType(
             instruction->shape().element_type()) ||
         (instruction->opcode() == HloOpcode::kConvert &&
          primitive_util::IsSubByteNonPredType(
              instruction->operand(0)->shape().element_type()));
}

bool IsCustomCallToMemoryPlacement(const HloInstruction* hlo) {
  if (hlo->opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  const std::string& target = hlo->custom_call_target();
  return target == memory_annotations::kMoveToDeviceCustomCallTarget ||
         target == memory_annotations::kMoveToHostCustomCallTarget;
}

}  // namespace

absl::Status GpuLayoutAssignment::AddDotBackendConstraints(
    LayoutConstraints* constraints, HloDotInstruction* instruction) {
  struct Side {
    size_t operand_no;
    const HloInstruction* operand;
    absl::Span<const int64_t> batch_dims;
    absl::Span<const int64_t> contracting_dims;
    PrimitiveType type;
    std::vector<int64_t> non_contracting_dims;
  };
  auto make_side =
      [&](size_t operand_no, absl::Span<const int64_t> batch_dims,
          absl::Span<const int64_t> contracting_dims) -> absl::StatusOr<Side> {
    Side side = {operand_no, instruction->operand(operand_no), batch_dims,
                 contracting_dims};
    side.type = side.operand->shape().element_type();
    TF_ASSIGN_OR_RETURN(
        side.non_contracting_dims,
        GetNonContractingDims(side.operand->shape(), side.batch_dims,
                              side.contracting_dims));
    return side;
  };
  const DotDimensionNumbers& dot_dims = instruction->dot_dimension_numbers();
  TF_ASSIGN_OR_RETURN(const Side lhs,
                      make_side(0, dot_dims.lhs_batch_dimensions(),
                                dot_dims.lhs_contracting_dimensions()));
  TF_ASSIGN_OR_RETURN(const Side rhs,
                      make_side(1, dot_dims.rhs_batch_dimensions(),
                                dot_dims.rhs_contracting_dimensions()));

  const PrimitiveType& output_type = instruction->shape().element_type();

  // Matmuls require the batch dimensions to be in consecutive physical
  // dimensions and likewise for the contracting and non-contracting
  // dimensions. Additionally, no batch dimension can be in the most
  // minor physical dimension for inputs or the output.

  const bool pack_along_contracting_dims =
      instruction->GetModule()
          ->config()
          .debug_options()
          .xla_gpu_experimental_pack_dot_operands_along_k_dimension();

  const bool is_s8_to_s32 = output_type == PrimitiveType::S32 &&
                            lhs.type == PrimitiveType::S8 &&
                            rhs.type == PrimitiveType::S8;
  const bool is_fp8 = (lhs.type == PrimitiveType::F8E4M3FN ||
                       lhs.type == PrimitiveType::F8E5M2FNUZ) &&
                      (rhs.type == PrimitiveType::F8E4M3FN ||
                       rhs.type == PrimitiveType::F8E5M2FNUZ);

  const se::CudaComputeCapability* cc =
      std::get_if<se::CudaComputeCapability>(&gpu_version_);
  const bool both_operands_require_minor_contraction_dims =
      is_s8_to_s32 || (is_fp8 && !(cc && cc->IsBlackwell()));

  for (const Side& side : {lhs, rhs}) {
    if ((IsPackedInstruction(side.operand) && pack_along_contracting_dims) ||
        both_operands_require_minor_contraction_dims) {
      TF_RETURN_IF_ERROR(SetDotOperandLayoutToMinorContracting(
          instruction, side.operand_no, side.batch_dims, side.contracting_dims,
          side.non_contracting_dims));
    } else if (!side.batch_dims.empty() || side.contracting_dims.size() > 1 ||
               side.non_contracting_dims.size() > 1) {
      TF_RETURN_IF_ERROR(SetDotOperandLayout(
          instruction, side.operand_no, side.batch_dims, side.contracting_dims,
          side.non_contracting_dims));
    }
  }

  // If we have at least one batch dimension or there is more than one
  // non-contracting dimension on lhs or rhs, we need to set a layout for
  // the dot output.
  if (!lhs.batch_dims.empty() || lhs.non_contracting_dims.size() > 1 ||
      rhs.non_contracting_dims.size() > 1) {
    TF_RETURN_IF_ERROR(SetDotLayout(instruction, constraints));
  }

  return absl::OkStatus();
}

absl::Status GpuLayoutAssignment::AddBackendConstraints(
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

    if (HloPredicateIsOp<HloOpcode::kDot>(instruction)) {
      TF_RETURN_IF_ERROR(AddDotBackendConstraints(
          constraints, Cast<HloDotInstruction>(instruction)));
    } else if (HloPredicateIsOp<HloOpcode::kTranspose>(instruction)) {
      const HloInstruction* operand = instruction->operand(0);
      if ((HloPredicateIsNotOp<HloOpcode::kDot>(operand)) ||
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
    } else if (HloPredicateIsOp<HloOpcode::kFft>(instruction)) {
      // cuFFT requires a dim0 major layout.
      Shape op0_shape = instruction->operand(0)->shape();
      LayoutUtil::SetToDefaultLayout(&op0_shape);
      Shape output_shape = instruction->shape();
      LayoutUtil::SetToDefaultLayout(&output_shape);
      TF_RETURN_IF_ERROR(SetOperandLayout(op0_shape, instruction, 0));
      TF_RETURN_IF_ERROR(SetInstructionLayout(output_shape, instruction));
    } else if ((HloPredicateIsOp<HloOpcode::kSort>(instruction) ||
                IsCubDeviceRadixSort(*instruction)) &&
               instruction->operand(0)->shape().dimensions().size() > 1) {
      // Make sure that all the operands and the output(s) have the same layout.
      Shape keys_shape = instruction->operand(0)->shape();
      Layout keys_layout =
          LayoutUtil::GetDefaultLayoutForRank(keys_shape.dimensions().size());
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
    } else if (IsCustomCallToTopK(*instruction)) {
      // The output of the TopK custom call needs to have default layout.
      Layout default_layout = LayoutUtil::GetDefaultLayoutForRank(
          instruction->operand(0)->shape().dimensions().size());
      TF_ASSIGN_OR_RETURN(
          auto values_buffer,
          points_to_analysis_->GetBufferDefinedAt(instruction, {0}));
      TF_RETURN_IF_ERROR(SetBufferLayout(default_layout, *values_buffer));
      TF_ASSIGN_OR_RETURN(
          auto indices_buffer,
          points_to_analysis_->GetBufferDefinedAt(instruction, {1}));
      TF_RETURN_IF_ERROR(SetBufferLayout(default_layout, *indices_buffer));
    } else if (HloPredicateIsOp<HloOpcode::kTriangularSolve>(instruction)) {
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
    } else if (HloPredicateIsOp<HloOpcode::kReduceScatter>(instruction)) {
      // XLA:GPU can only support reduce-scatter where the scatter dimension
      // is the most major dimension in the layout.
      auto ars = Cast<HloReduceScatterInstruction>(instruction);
      TF_RETURN_IF_ERROR(SetInstructionLayout(
          ShapeUtil::MoveDimToMajor(ars->shape(), ars->scatter_dimension()),
          ars));
    } else if (HloPredicateIsOp<HloOpcode::kAllGather>(instruction)) {
      // XLA:GPU can only support all-gathers where the gather dimension is the
      // most major dimension in the layout.
      auto ag = Cast<HloAllGatherInstruction>(instruction);
      TF_RETURN_IF_ERROR(SetInstructionLayout(
          ShapeUtil::MoveDimToMajor(ag->shape(), ag->all_gather_dimension()),
          ag));
    } else if (HloPredicateIsOp<HloOpcode::kAllToAll>(instruction) &&
               instruction->shape().IsArray()) {
      // XLA:GPU can only support all-to-all with split dimensions where the
      // split dimension is the most major dimension in the layout.
      auto* all_to_all = Cast<HloAllToAllInstruction>(instruction);
      TF_RETURN_IF_ERROR(SetInstructionLayout(
          ShapeUtil::MoveDimToMajor(all_to_all->shape(),
                                    *all_to_all->split_dimension()),
          all_to_all));
    } else if (HloPredicateIsOp<HloOpcode::kRaggedAllToAll>(instruction)) {
      auto* ragged_all_to_all = Cast<HloRaggedAllToAllInstruction>(instruction);
      // XLA:GPU can only support ragged-all-to-all with the most major ragged
      // dimension in the layout.
      TF_RETURN_IF_ERROR(SetInstructionLayout(
          ShapeUtil::MoveDimToMajor(ragged_all_to_all->shape(), 0),
          ragged_all_to_all));
    } else if (HloPredicateIsOp<HloOpcode::kSend>(instruction)) {
      Shape s = instruction->operand(0)->shape();
      LayoutUtil::SetToDefaultLayout(&s);
      TF_RETURN_IF_ERROR(SetInstructionLayout(s, instruction->operand(0)));
      TF_RETURN_IF_ERROR(
          SetArrayOperandLayout(s.layout(), instruction->operand(0), 0));
    } else if (HloPredicateIsOp<HloOpcode::kRecv>(instruction)) {
      Shape s = instruction->shape();
      ShapeUtil::ForEachMutableSubshape(
          &s, [&](Shape* subshape, const ShapeIndex& index) {
            LayoutUtil::SetToDefaultLayout(subshape);
          });
      TF_RETURN_IF_ERROR(SetInstructionLayout(s, instruction));
    } else if (IsCustomCallToMemoryPlacement(instruction)) {
      // Make sure that host memory buffers use the default layout so that
      // the compiler does not insert transposes on host memory buffers.
      Shape operand_shape = instruction->operand(0)->shape();
      LayoutUtil::SetToDefaultLayout(&operand_shape);
      TF_RETURN_IF_ERROR(SetOperandLayout(operand_shape, instruction, 0));
      TF_RETURN_IF_ERROR(SetInstructionLayout(operand_shape, instruction));
    }
  }
  return absl::OkStatus();
}

absl::Status GpuLayoutAssignment::SetDotOperandLayout(
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
  return SetOperandMajorToMinorLayout(
      instruction, operand,
      /*dim_groups=*/{batch_dims, row_dims, col_dims});
}

absl::Status GpuLayoutAssignment::SetDotOperandLayoutToMinorContracting(
    const HloInstruction* instruction, int64_t operand,
    absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> contracting_dims,
    absl::Span<const int64_t> noncontracting_dims) {
  Shape shape = instruction->operand(operand)->shape();

  if (shape.has_layout() &&
      shape.layout().minor_to_major_size() >= contracting_dims.size()) {
    // Check that the contracting dimensions are physically minor, i.e. check
    // that minor physical dimensions all point to contracting logical
    // dimensions.
    bool contracting_dims_are_minor = true;
    const auto& minor_to_major = shape.layout().minor_to_major();
    for (int64_t i = 0; i < contracting_dims.size(); ++i) {
      if (!absl::c_linear_search(contracting_dims, minor_to_major[i])) {
        contracting_dims_are_minor = false;
        break;
      }
    }

    // If contracting dims are already minor, and the layout is valid, keep it.
    if (contracting_dims_are_minor &&
        MatrixLayout::For(shape, batch_dims, noncontracting_dims,
                          contracting_dims)
            .ok()) {
      // Re-set the operand layout, so it becomes mandatory.
      return SetOperandLayout(shape, instruction, operand);
    }
  }
  return SetOperandMajorToMinorLayout(
      instruction, operand,
      /*dim_groups=*/
      {batch_dims, noncontracting_dims, contracting_dims});
}

absl::Status GpuLayoutAssignment::SetOperandMajorToMinorLayout(
    const HloInstruction* instruction, int64_t operand,
    std::initializer_list<absl::Span<const int64_t>> dim_groups) {
  size_t size = 0;
  for (auto group : dim_groups) size += group.size();
  std::vector<int64_t> major_to_minor;
  major_to_minor.reserve(size);
  for (const auto& group : dim_groups) {
    major_to_minor.insert(major_to_minor.end(), group.begin(), group.end());
  }

  Shape shape = instruction->operand(operand)->shape();
  *shape.mutable_layout() =
      LayoutUtil::MakeLayoutFromMajorToMinor(major_to_minor);
  return SetOperandLayout(shape, instruction, operand);
}

absl::Status GpuLayoutAssignment::SetDotLayout(
    const HloInstruction* instruction, LayoutConstraints* constraints) {
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

bool GpuLayoutAssignment::PropagateReductionLayoutToOperand(
    const HloInstruction* user) {
  // We try to propagate a layout to make the reduction a row reduction. But
  // propagating the layout is only beneficial if the reduction emitter would be
  // used for the row reduction.
  int64_t reduction_size = 1;
  for (int64_t reduction_dim : user->dimensions()) {
    reduction_size *= user->operand(0)->shape().dimensions(reduction_dim);
  }
  int64_t kept_dimension_size = ShapeUtil::ElementsIn(user->shape());
  return IsUnnestedReductionFasterThanElemental(
      {/*is_row_reduction=*/true, {1, kept_dimension_size, reduction_size}},
      device_description_);
}

bool GpuLayoutAssignment::InstructionCanChangeLayoutInstance(
    const HloInstruction* instruction) {
  // The TopK custom call cannot handle the case if the operand has a different
  // layout.
  const HloCustomCallInstruction* custom_call =
      DynCast<HloCustomCallInstruction>(instruction);
  if (custom_call != nullptr &&
      custom_call->custom_call_target() == kTopKCustomCallTarget) {
    return false;
  }

  return LayoutAssignment::InstructionCanChangeLayoutInstance(instruction);
}

}  // namespace gpu
}  // namespace xla
