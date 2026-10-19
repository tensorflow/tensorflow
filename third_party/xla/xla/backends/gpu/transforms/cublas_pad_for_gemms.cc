/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/cublas_pad_for_gemms.h"

#include <array>
#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/triton/support_legacy.h"
#include "xla/backends/gpu/transforms/gemm_fusion.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

static absl::StatusOr<bool> PadForGemm(HloDotInstruction* dot,
                                       PrimitiveType datatype,
                                       int pad_to_multiple_of) {
  auto* lhs = dot->mutable_operand(0);
  auto* rhs = dot->mutable_operand(1);

  Shape lshape = lhs->shape();
  Shape rshape = rhs->shape();
  Shape result_shape = dot->shape();

  if (lshape.element_type() != datatype || rshape.element_type() != datatype) {
    return false;
  }

  auto pad_dim = [&](Shape& s, int dim) {
    s.set_dimensions(dim,
                     RoundUpTo<int64_t>(s.dimensions(dim), pad_to_multiple_of));
  };

  auto pad_matrix_dims = [&pad_dim](Shape s) {
    // Since the dot instruction is canonicalized, the last two dimensions for
    // each operand represent non-batch dimensions, and the others are the same
    // for both operands and correspond to batch dimensions.
    pad_dim(s, s.dimensions().size() - 2);
    pad_dim(s, s.dimensions().size() - 1);
    return s;
  };

  Shape new_lshape = pad_matrix_dims(lshape);
  Shape new_rshape = pad_matrix_dims(rshape);
  Shape new_result_shape = pad_matrix_dims(result_shape);

  if (new_lshape == lshape && new_rshape == rshape) {
    return false;
  }

  VLOG(3) << "old shape: " << lshape << " " << rshape << " " << result_shape;
  VLOG(3) << "new shape: " << new_lshape << " " << new_rshape << " "
          << new_result_shape;

  HloComputation* parent = dot->parent();
  HloInstruction* zero = parent->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(datatype)));
  zero->set_metadata(dot->metadata());

  auto pad = [&](HloInstruction* operand, const Shape& new_shape) {
    const Shape& shape = operand->shape();
    if (shape == new_shape) {
      return operand;
    }
    PaddingConfig padding_config;
    for (int i = 0; i < shape.dimensions().size(); ++i) {
      auto* dimension = padding_config.add_dimensions();
      dimension->set_edge_padding_high(new_shape.dimensions(i) -
                                       shape.dimensions(i));
    }
    HloInstruction* padded = parent->AddInstruction(
        HloInstruction::CreatePad(new_shape, operand, zero, padding_config));
    padded->set_metadata(dot->metadata());
    return padded;
  };

  HloInstruction* result = parent->AddInstruction(dot->CloneWithNewOperands(
      new_result_shape, {pad(lhs, new_lshape), pad(rhs, new_rshape)}));

  if (new_result_shape != result_shape) {
    std::vector<int64_t> start_indices(result_shape.dimensions().size(), 0);
    std::vector<int64_t> strides(result_shape.dimensions().size(), 1);
    result = parent->AddInstruction(
        HloInstruction::CreateSlice(result_shape, result, start_indices,
                                    result_shape.dimensions(), strides));
    result->set_metadata(dot->metadata());
  }

  ABSL_RETURN_IF_ERROR(parent->ReplaceInstruction(dot, result));
  return true;
}

namespace {

struct CublasPaddingRequirement {
  se::CudaComputeCapability min_compute_capability;
  PrimitiveType data_type;
  int multiple_of;
};

constexpr std::array<CublasPaddingRequirement, 3> CublasPaddingRequirements{
    {{se::CudaComputeCapability::Volta(), S8, 4},
     {se::CudaComputeCapability::Volta(), F16, 8},
     {se::CudaComputeCapability::Ampere(), BF16, 8}}};

// We need this check because PadForGemm works in the assumption that
// the dot instruction is canonicalized.
bool CheckCanonical(HloDotInstruction* dot) {
  const auto& dimension_numbers = dot->dot_dimension_numbers();

  if (dimension_numbers.lhs_batch_dimensions_size() + 2 !=
          dot->operand(0)->shape().dimensions().size() ||
      dimension_numbers.rhs_batch_dimensions_size() + 2 !=
          dot->operand(1)->shape().dimensions().size()) {
    VLOG(2)
        << dot->ToString()
        << " is not canonical: Expected all dimensions but 2 to be "
           "batch_dimensions. Hence, this dot is not a candidate for padding.";
    return false;
  }

  std::vector<int64_t> canonical_batch_dims(
      dimension_numbers.lhs_batch_dimensions_size());
  absl::c_iota(canonical_batch_dims, 0);
  if (!absl::c_equal(dimension_numbers.lhs_batch_dimensions(),
                     canonical_batch_dims) ||
      !absl::c_equal(dimension_numbers.rhs_batch_dimensions(),
                     canonical_batch_dims)) {
    VLOG(2)
        << dot->ToString()
        << " is not canonical: Expected batch dimensions to be all "
           "dimensions except for the last 2 ones. Hence, this dot is not a "
           "candidate for padding.";
    return false;
  }

  return true;
}

absl::StatusOr<std::vector<HloDotInstruction*>> GetRelevantDots(
    HloComputation* comp, PrimitiveType datatype) {
  std::vector<HloDotInstruction*> gemms;

  for (HloInstruction* instr : comp->instructions()) {
    ABSL_ASSIGN_OR_RETURN(
        bool is_matmul,
        IsCublasSupportedMatMul(*instr,
                                /*allow_matrix_vector_multiplication=*/false));
    if (is_matmul) {
      HloDotInstruction* dot = Cast<HloDotInstruction>(instr);
      if (instr->operand(0)->shape().element_type() == datatype &&
          CheckCanonical(dot)) {
        gemms.push_back(dot);
      }
    }
  }
  return gemms;
}

}  // namespace

absl::StatusOr<bool> CublasPadForGemms::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  const auto* cc = gpu_compute_capability_.cuda_compute_capability();
  if (cc == nullptr) {
    return false;
  }
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (const CublasPaddingRequirement& req : CublasPaddingRequirements) {
      if (!cc->SupportsAllFeaturesOf(req.min_compute_capability)) {
        continue;
      }
      ABSL_ASSIGN_OR_RETURN(std::vector<HloDotInstruction*> dots,
                       GetRelevantDots(comp, req.data_type));
      for (HloDotInstruction* dot : dots) {
        ABSL_ASSIGN_OR_RETURN(bool result,
                         PadForGemm(dot, req.data_type, req.multiple_of));
        changed |= result;
      }
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
