/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_cost_analysis.h"

#include <vector>

#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"

namespace xla {
namespace gpu {

int64_t GpuHloCostAnalysis::FusionParameterReadBytes(
    const HloInstruction* hlo) const {
  CHECK(hlo->IsFused() && (hlo->opcode() == HloOpcode::kParameter ||
                           hlo->opcode() == HloOpcode::kGetTupleElement));
  return GetShapeSize(hlo->shape()) *
         hlo_properties_.at(hlo).at(kUtilizationKey);
}

Status GpuHloCostAnalysis::FusionCalculateUtilizations(
    const HloInstruction* fusion) {
  const HloInstruction* root = fusion->fused_expression_root();
  // Traverse through the computation from the root till parameters propagating
  // the utilization of operands; store utilization of each node
  // in hlo_properties_. All consumers of an instruction are processed before
  // the instruction itself.
  std::vector<HloInstruction*> instructions =
      fusion->fused_instructions_computation()->MakeInstructionPostOrder();
  absl::c_reverse(instructions);

  // To estimate where within the computation an instruction output can be
  // reused and where it has to be recomputed again we group accesses to the
  // instruction by their origin from "element-wise use roots". All access
  // paths from such a root to the instruction are element-wise.
  // Whenever we account a non-element-wise operation we forget about
  // element-wise roots encountered so far and provisionally set its operands
  // as new element-wise roots.
  absl::flat_hash_map<const HloInstruction*, ConstHloInstructionSet>
      elementwise_use_roots;

  for (const HloInstruction* instr : instructions) {
    hlo_properties_[instr][kUtilizationKey] = 0;
  }

  // For the purpose of operand utilization analysis, no matter how the fusion
  // outputs are used, we assume that fusion is always executed completely
  // producing 100% of its outputs.
  hlo_properties_[root][kUtilizationKey] = 1.0;
  elementwise_use_roots[root].insert(root);
  current_properties_[kFlopsKey] = 0;

  for (const HloInstruction* instr : instructions) {
    VLOG(8) << instr->ToString() << ":";
    VLOG(9) << "Elementwise use roots:";
    for (const HloInstruction* r : elementwise_use_roots[instr]) {
      VLOG(9) << "\t" << r->ToString();
      if (instr != r) {
        hlo_properties_[instr][kUtilizationKey] +=
            hlo_properties_[r][kUtilizationKey];
      }
    }

    float cur_instr_utilization = hlo_properties_[instr][kUtilizationKey];
    VLOG(8) << "Total utilization: " << cur_instr_utilization;

    current_properties_[kFlopsKey] +=
        cur_instr_utilization * hlo_properties_[instr][kFlopsKey];

    for (int operand_idx = 0; operand_idx < instr->operand_count();
         ++operand_idx) {
      const HloInstruction* operand = instr->operand(operand_idx);
      if ((instr->IsElementwise()) || instr->opcode() == HloOpcode::kTuple ||
          instr->opcode() == HloOpcode::kGetTupleElement) {
        auto instr_roots = elementwise_use_roots[instr];
        for (const HloInstruction* r : instr_roots) {
          elementwise_use_roots[operand].insert(r);
        }
      } else {
        elementwise_use_roots[operand].insert(operand);
        float cur_operand_utilization =
            cur_instr_utilization * operand_utilization(*instr, operand_idx);
        // The utilization is always a best-effort estimate, but in some cases
        // cannot be precise due to dynamic nature of operations - dynamic
        // slice is one such example. We do an average estimate in these
        // cases and this can sometimes produce fractional utilizations which
        // should be at least rounded up to a whole number of produced elements
        // to be more realistic.
        int64_t operand_elements = ShapeUtil::ElementsIn(operand->shape());
        cur_operand_utilization =
            ceil(cur_operand_utilization * operand_elements) / operand_elements;
        hlo_properties_[operand][kUtilizationKey] = cur_operand_utilization;
      }
    }
  }
  return OkStatus();
}

Status GpuHloCostAnalysis::HandleCustomCall(const HloInstruction* custom_call) {
  if (IsCublasGemm(*custom_call)) {
    // The naming conventions and meanings of gemm parameters are documented
    // here:
    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
    TF_ASSIGN_OR_RETURN(auto gemm_config,
                        custom_call->backend_config<gpu::GemmBackendConfig>());

    // Technically, in addition to the dot product (A * B), cuBLAS gemm also
    // performs additional scaling (by factor 'alpha') and addition with a
    // scaled third matrix (beta * C), which will introduce additional
    // multiplications and additions. But total FLOPS will be dominated by the
    // dot product, so we don't include these extra multiplications and
    // additions in the FLOPS calculation.

    // Also, this calculation assumes that the strides for the gemm are
    // properly set such that none of the inputs in a batch overlap with any
    // other batches. If they do, this will undercount the FLOPS, because it
    // assumes that the strides are implicit in the sizes of the batch
    // dimensions.

    // Finally, this is technically incorrect if the element type of this
    // gemm is an integer type, because in that case no floating point
    // operations are involved at all! But we still calculate FLOPS because the
    // number is sometimes required for ad-hoc calculations.
    current_properties_[kFlopsKey] =
        GetDotFlops(custom_call->operand(0)->shape(), custom_call->shape(),
                    gemm_config.dot_dimension_numbers());
    return OkStatus();
  }

  if (IsCustomCallToDnnConvolution(*custom_call)) {
    // As with dots, this flops calculation has the following inaccuracies.
    //
    //  - We may have a fused conv which does additional ops (multiplying by a
    //    scalar `alpha`, adding a bias or side-input, doing a relu, etc).  But
    //    we can safely ignore this because the overall computation is dominated
    //    by the convolution itself.
    //
    //  - cudnn may use complex conv algorithms that do fewer (or more!) flops
    //    than we calculate.
    //
    //  - for int8_t convs, these aren't *fl*ops, but we fudge it.
    current_properties_[kFlopsKey] = GetConvolutionFlops(custom_call);

    // conv custom-calls return a tuple (real_output, temp_bytes).  Count just
    // the real_output in output bytes accessed.  The main purpose of
    // hlo_cost_analysis is to figure out if ops are running "as fast as
    // possible", and if we were to include temp memory in here, we'd
    // essentially be *rewarding* convs that use additional temp memory!
    if (custom_call->shape().IsTuple()) {
      SetOutputBytesAccessed(
          options_.shape_size(custom_call->shape().tuple_shapes(0)));
    }
    return OkStatus();
  }

  return HloCostAnalysis::HandleCustomCall(custom_call);
}

int64_t GpuHloCostAnalysis::GetConvolutionFlops(
    const HloInstruction* convolution) {
  auto lhs = convolution->operand(0);
  auto rhs = convolution->operand(1);
  const Shape& lhs_shape = lhs->shape();
  const Shape& rhs_shape = rhs->shape();
  const Shape& result_shape = [&]() -> const Shape& {
    // convolution custom-calls return a tuple of (actual_result, temp_buffer).
    const Shape& shape = convolution->shape();
    if (IsCustomCallToDnnConvolution(*convolution) &&
        convolution->shape().IsTuple()) {
      return shape.tuple_shapes(0);
    }
    return shape;
  }();

  return HloCostAnalysis::GetConvolutionFlops(convolution, lhs_shape, rhs_shape,
                                              result_shape);
}

std::unique_ptr<HloCostAnalysis>
GpuHloCostAnalysis::CreateNestedCostAnalysis() {
  return std::make_unique<GpuHloCostAnalysis>(options_);
}

}  // namespace gpu
}  // namespace xla
