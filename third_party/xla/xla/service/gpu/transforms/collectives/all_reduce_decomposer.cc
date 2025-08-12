/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/collectives/all_reduce_decomposer.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// The threshold is the upper limit of the number of elements in the input to
// an all-reduce operation for it to be decomposed. The value is chosen
// empirically in Feb 2025 to be a reasonable trade-off between performance and
// memory usage.
constexpr int64_t kOneShotAllReduceThreshold = 256 * 1024;

static bool IsSmallAllReduce(const HloInstruction* hlo) {
  return HloPredicateIsOp<HloOpcode::kAllReduce>(hlo) &&
         ShapeUtil::ElementsInRecursive(hlo->shape()) <=
             kOneShotAllReduceThreshold;
}

static std::optional<Literal> CreateReductionInitLiteral(
    HloAllReduceInstruction* all_reduce, HloComputation* computation) {
  std::optional<ReductionKind> reduction_kind =
      MatchReductionComputation(all_reduce->to_apply());
  if (!reduction_kind.has_value()) {
    return std::nullopt;
  }

  return GetReductionIdentity(reduction_kind.value(),
                              all_reduce->shape().element_type());
}

// Adds a size-1 major dimension to the given HLO instruction.
static HloInstruction* PrependSize1MajorDimension(HloInstruction* hlo,
                                                  HloComputation* computation) {
  absl::InlinedVector<int64_t, 4> reshape_dimensions;
  reshape_dimensions.reserve(hlo->shape().dimensions().size() + 1);
  reshape_dimensions.push_back(1);
  absl::c_copy(hlo->shape().dimensions(),
               std::back_inserter(reshape_dimensions));

  Shape reshape_shape =
      ShapeUtil::MakeShape(hlo->shape().element_type(), reshape_dimensions);
  return computation->AddInstruction(
      HloInstruction::CreateReshape(reshape_shape, hlo));
}

// Decomposes the given all-reduce operation into an all-gather and a reduce
// operation.
static absl::StatusOr<bool> DecomposeAllReduce(HloInstruction* hlo,
                                               HloComputation* computation,
                                               HloModule* module) {
  HloAllReduceInstruction* all_reduce = Cast<HloAllReduceInstruction>(hlo);

  HloInstruction* input = all_reduce->mutable_operand(0);

  std::optional<Literal> reduction_init_literal =
      CreateReductionInitLiteral(all_reduce, computation);
  if (!reduction_init_literal.has_value()) {
    // Unsupported reduction type.
    return false;
  }

  TF_ASSIGN_OR_RETURN(auto replica_group_count_and_size,
                      GetReplicaGroupCountAndSize(all_reduce));

  if (!replica_group_count_and_size.has_value()) {
    // Could not determine the number of participating devices at compilation.
    return false;
  }

  int64_t num_participating_devices = replica_group_count_and_size->second;

  // Add a size-1 major dimension to the input that will be used as the
  // all-gather and reduction dimension.
  HloInstruction* reshape = PrependSize1MajorDimension(input, computation);

  TF_ASSIGN_OR_RETURN(Shape all_gather_shape,
                      ShapeInference::InferAllGatherShape(
                          {&reshape->shape()}, /*all_gather_dimension=*/0,
                          num_participating_devices));

  HloInstruction* all_gather =
      computation->AddInstruction(HloInstruction::CreateAllGather(
          all_gather_shape, {reshape}, /*all_gather_dimension=*/0,
          all_reduce->device_list(), all_reduce->constrain_layout(),
          all_reduce->channel_id(), all_reduce->use_global_device_ids()));

  HloInstruction* init = computation->AddInstruction(
      HloInstruction::CreateConstant(*std::move(reduction_init_literal)));

  HloInstruction* reduce =
      computation->AddInstruction(HloInstruction::CreateReduce(
          input->shape(), all_gather, init,
          /*dimensions_to_reduce=*/{0}, all_reduce->to_apply()));

  TF_RETURN_IF_ERROR(all_reduce->ReplaceAllUsesWith(reduce));
  TF_RETURN_IF_ERROR(
      computation->RemoveInstructionAndUnusedOperands(all_reduce));

  return true;
}

absl::StatusOr<bool> AllReduceDecomposer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (auto computation : module->computations(execution_threads)) {
    for (auto hlo : computation->MakeInstructionPostOrder()) {
      if (!IsSmallAllReduce(hlo)) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(bool decomposed,
                          DecomposeAllReduce(hlo, computation, module));
      changed |= decomposed;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
