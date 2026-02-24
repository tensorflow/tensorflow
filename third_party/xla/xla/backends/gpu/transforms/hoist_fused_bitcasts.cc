/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/hoist_fused_bitcasts.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/layout.h"
#include "xla/service/call_graph.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {
namespace {

// Extracts the TritonGemmConfig from the given fusion's backend config.
absl::StatusOr<TritonGemmConfig> GetTritonGemmConfig(
    const HloFusionInstruction& fusion) {
  TF_ASSIGN_OR_RETURN(auto gpu_config,
                      fusion.backend_config<GpuBackendConfig>());
  const FusionBackendConfig& backend_config =
      gpu_config.fusion_backend_config();
  if (!backend_config.has_triton_gemm_config()) {
    return absl::InternalError(
        "The fusion's backend config doesn't have a triton_gemm_config.");
  }
  return TritonGemmConfig::FromProto(backend_config.triton_gemm_config());
}

using HloInstructionSetVector =
    llvm::SetVector<HloInstruction*, std::vector<HloInstruction*>,
                    HloInstructionSet>;

// Returns the set of instructions that are reachable from 'instruction' using
// the given accessor.
template <typename T>
HloInstructionSetVector GetTransitiveInstructionSet(
    const HloInstruction* instruction, T (HloInstruction::*get)() const) {
  std::deque<HloInstruction*> worklist;
  auto append = [&](const auto& instructions) {
    worklist.insert(worklist.end(), instructions.begin(), instructions.end());
  };
  append((instruction->*get)());
  HloInstructionSetVector result;
  while (!worklist.empty()) {
    HloInstruction* front = worklist.front();
    worklist.pop_front();
    if (result.insert(front)) {
      append((front->*get)());
    }
  }
  return result;
}

// Returns the set of producers reachable from 'instruction' in use-before-def
// order.
HloInstructionSetVector GetProducerSet(const HloInstruction* instruction) {
  return GetTransitiveInstructionSet(instruction, &HloInstruction::operands);
}
// Returns the set of consumers reachable from 'instruction' in def-before-use
// order.
HloInstructionSetVector GetConsumerSet(const HloInstruction* instruction) {
  return GetTransitiveInstructionSet(instruction, &HloInstruction::users);
}

// Verifies that the set of instructions is closed under the given accessor,
// i.e. that the set of instructions reachable through the given accessor are
// either in the set itself or the root.
template <typename T>
absl::Status VerifyIsClosedInstructionSet(
    const HloInstructionSetVector& instructions, const HloInstruction* root,
    T (HloInstruction::*get)() const) {
  for (HloInstruction* instruction : instructions) {
    for (HloInstruction* reachable : (instruction->*get)()) {
      if (reachable != root && instructions.count(reachable) == 0) {
        return absl::FailedPreconditionError(
            absl::StrCat("Instruction ", reachable->ToString(),
                         " is reachable from ", instruction->ToString(),
                         ", which is not in the recursive set of, or ",
                         root->ToString(), " itself."));
      }
    }
  }

  return absl::OkStatus();
}

absl::Status VerifyIsClosedProducerSet(
    const HloInstructionSetVector& instructions, const HloInstruction* root) {
  return VerifyIsClosedInstructionSet(instructions, root,
                                      &HloInstruction::users);
}

// Copies the element type and size from `source` to `destination`.
void CopyElementType(const Shape& source, Shape* destination) {
  destination->set_element_type(source.element_type());
  destination->mutable_layout()->set_element_size_in_bits(
      source.layout().element_size_in_bits());
}

llvm::SmallVector<int64_t> GetInversePermutation(
    absl::Span<const int64_t> permutation) {
  llvm::SmallVector<int64_t> result(permutation.size());
  for (int64_t i = 0; i < permutation.size(); ++i) {
    result[permutation[i]] = i;
  }
  return result;
}

// Applies the backward-mapping 'permutation' to 'values'.
llvm::SmallVector<int64_t> ApplyPermutation(
    absl::Span<const int64_t> values, absl::Span<const int64_t> permutation) {
  llvm::SmallVector<int64_t> result;
  result.reserve(permutation.size());
  for (int64_t index : permutation) {
    result.push_back(values[index]);
  }
  return result;
}

// Returns the dimensions of 'shape' in minor-to-major order.
llvm::SmallVector<int64_t> GetPhysicalDimensions(const Shape& shape) {
  return ApplyPermutation(shape.dimensions(), shape.layout().minor_to_major());
}

// Parameters to rewrite a bitcast(broadcast/transpose) as
// broadcast/transpose(bitcast) and vice versa.
struct BitcastParams {
  Shape new_shape;                      // The bitcast output shape.
  llvm::SmallVector<int64_t> new_dims;  // The dims of the broadcast/transpose.
};

// Returns parameters to rewrite a broadcast + bitcast as bitcast + broadcast.
//
// Example:
//
// broadcast = broadcast(operand)
// result = result_shape bitcast(broadcast)
//
// to
//
// bitcast = new_shape bitcast(operand)
// result = broadcast(bitcast), dimensions={new_dims}.
//
// Assumes that:
// - broadcast does not transpose dimensions (checked by hlo_verifier);
// - bitcast does not mix operand and broadcast dimensions (checks);
absl::StatusOr<BitcastParams> CalculateBitcastOfBroadcast(
    const HloBroadcastInstruction* broadcast, const Shape& result_shape) {
  const Shape& broadcast_shape = broadcast->shape();

  // Maps broadcast dimension index to whether it's an operand dimension.
  llvm::SmallVector<bool> is_operand_dim(broadcast_shape.dimensions().size());
  for (const int64_t index : broadcast->dimensions()) {
    is_operand_dim[index] = true;
  }

  // Dimensions of the new broadcast.
  llvm::SmallVector<int64_t> new_dims;
  llvm::SmallVector<int64_t> broadcast_physical_dims =
      GetPhysicalDimensions(broadcast_shape);
  auto factors = CommonFactors(GetPhysicalDimensions(result_shape),
                               broadcast_physical_dims);
  for (int64_t i = 1; i < factors.size(); ++i) {
    auto [result_from, broadcast_from] = factors[i - 1];
    auto [result_to, broadcast_to] = factors[i];

    bool all_operands = true, any_operands = false;
    for (int64_t j = broadcast_from; j < broadcast_to; ++j) {
      if (broadcast_physical_dims[j] == 1) {
        // If dimension size is 1 then we can ignore it: it's either immediately
        // dropped by old reshape or it's coming from the operand and then the
        // new reshape will handle it.
        continue;
      }
      bool value = is_operand_dim[broadcast_shape.layout().minor_to_major(j)];
      all_operands &= value;
      any_operands |= value;
    }
    if (!any_operands) {
      continue;  // All dimensions in this group are broadcast dimensions.
    }
    if (!all_operands) {
      return absl::InvalidArgumentError(
          absl::StrCat("Cannot hoist bitcast across ", broadcast->ToString(),
                       " as it mixes operand and broadcast dimensions."));
    }

    for (int64_t j = result_from; j < result_to; ++j) {
      new_dims.push_back(result_shape.layout().minor_to_major(j));
    }
  }
  absl::c_sort(new_dims);  // Sort into logical order.

  BitcastParams result;
  CopyElementType(result_shape, &result.new_shape);
  for (int64_t index : new_dims) {
    result.new_shape.add_dimensions(result_shape.dimensions(index));
  }
  auto* new_layout =
      result.new_shape.mutable_layout()->mutable_minor_to_major();
  new_layout->reserve(new_dims.size());
  for (int64_t index : result_shape.layout().minor_to_major()) {
    if (auto it = absl::c_lower_bound(new_dims, index);
        it != new_dims.end() && *it == index) {
      new_layout->push_back(it - new_dims.begin());
    }
  }
  result.new_dims = std::move(new_dims);

  VLOG(3) << "CalculateBitcastOfBroadcast:";
  VLOG(3) << "  broadcast = " << broadcast_shape.ToString(true) << " broadcast("
          << broadcast->operand(0)->shape().ToString(true)
          << " operand), dimensions="
          << absl::StrJoin(broadcast->dimensions(), ",");
  VLOG(3) << "  result    = " << result_shape.ToString(true) << " bitcast("
          << broadcast_shape.ToString(true) << " broadcast)";
  VLOG(3) << "--------------------------------";
  VLOG(3) << "  bitcast   = " << result.new_shape.ToString(true) << " bitcast("
          << broadcast->operand(0)->shape().ToString(true) << " operand)";
  VLOG(3) << "  result    = " << result_shape.ToString(true) << " broadcast("
          << result.new_shape.ToString(true)
          << " bitcast), dimensions=" << absl::StrJoin(result.new_dims, ",");

  return result;
}

// Returns parameters to rewrite a bitcast + broadcast as broadcast + bitcast.
//
// Example:
//
// bitcast = bitcast(operand_shape operand)
// result = broadcast(bitcast)
//
// to
//
// broadcast = new_shape broadcast(operand), dimensions={new_dims}.
// result = bitcast(broadcast)
//
// Assumes that:
// - broadcast does not transpose dimensions (checked by hlo_verifier);
// - bitcast does not mix operand and broadcast dimensions (checks);
absl::StatusOr<BitcastParams> CalculateBroadcastOfBitcast(
    const HloBroadcastInstruction* broadcast, const Shape& operand_shape) {
  const Shape& bitcast_shape = broadcast->operand(0)->shape();
  const Shape& result_shape = broadcast->shape();

  // Maps logical result dimension index to a range of physical operand
  // dimensions, or nullopt if the dimension is broadcasted.
  llvm::SmallVector<std::optional<std::pair<int64_t, int64_t>>>
      result_to_operand_range(result_shape.dimensions().size());
  auto result_inv_layout =
      GetInversePermutation(result_shape.layout().minor_to_major());
  auto factors = CommonFactors(GetPhysicalDimensions(bitcast_shape),
                               GetPhysicalDimensions(operand_shape));
  for (int64_t i = 1; i < factors.size(); ++i) {
    auto [bitcast_from, operand_from] = factors[i - 1];
    auto [bitcast_to, operand_to] = factors[i];

    llvm::SmallVector<int64_t> indices;
    indices.reserve(bitcast_to - bitcast_from);
    for (int64_t j = bitcast_from; j < bitcast_to; ++j) {
      int64_t index =
          broadcast->dimensions()[bitcast_shape.layout().minor_to_major(j)];

      // Store the entire operand dimension range in the minor-most dimension
      // index and an empty range in all others.
      result_to_operand_range[index].emplace(operand_from, operand_to);
      operand_from = operand_to;

      // Check that the physical result indices form a contiguous range.
      indices.push_back(result_inv_layout[index]);
    };

    if (indices.back() - indices.front() >= bitcast_to - bitcast_from ||
        !absl::c_is_sorted(indices)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Cannot hoist bitcast across ", broadcast->ToString(),
                       " because result dimensions are not contiguous."));
    }
  }

  BitcastParams result;
  CopyElementType(operand_shape, &result.new_shape);
  result.new_dims.resize(operand_shape.dimensions().size());
  auto* new_layout =
      result.new_shape.mutable_layout()->mutable_minor_to_major();
  int64_t new_rank = operand_shape.dimensions().size() +
                     result_shape.dimensions().size() -
                     bitcast_shape.dimensions().size();
  new_layout->reserve(new_rank);
  llvm::SmallVector<int64_t> new_shape_dims(new_rank);

  // We are free to insert the broadcast dimensions in any order. Insert them
  // at the end of the the logical dimension order.
  int64_t broadcast_index = operand_shape.dimensions().size();

  // Iterate through the logical result dimension indices in physical order.
  for (int64_t result_index : result_shape.layout().minor_to_major()) {
    if (auto range = result_to_operand_range[result_index]) {
      // This result dimension corresponds to a group of operand dimensions.
      // Iterate through the range of physical operand dimension indices.
      for (int64_t i = range->first; i < range->second; ++i) {
        int64_t operand_index = operand_shape.layout().minor_to_major(i);
        int64_t new_index = operand_index;
        new_shape_dims[new_index] = operand_shape.dimensions(operand_index);
        new_layout->push_back(new_index);
        result.new_dims[operand_index] = new_index;
      }
    } else {
      // This is a new dimension introduced by the original broadcast.
      int64_t new_index = broadcast_index++;
      new_shape_dims[new_index] = result_shape.dimensions(result_index);
      new_layout->push_back(new_index);
    }
  }
  absl::c_sort(result.new_dims);  // Sort into logical order.
  for (int64_t dimension : new_shape_dims) {
    result.new_shape.add_dimensions(dimension);
  }

  VLOG(3) << "CalculateBroadcastOfBitcast:";
  VLOG(3) << "  bitcast   = " << bitcast_shape.ToString(true) << " bitcast("
          << operand_shape.ToString(true) << " operand)";
  VLOG(3) << "  result    = " << result_shape.ToString(true) << " broadcast("
          << bitcast_shape.ToString(true) << " bitcast), dimensions="
          << absl::StrJoin(broadcast->dimensions(), ",");
  VLOG(3) << "--------------------------------";
  VLOG(3) << "  broadcast = " << result.new_shape.ToString(true)
          << " broadcast(" << operand_shape.ToString(true)
          << " operand), dimensions=" << absl::StrJoin(result.new_dims, ",");
  VLOG(3) << "  result    = " << result_shape.ToString(true) << " bitcast("
          << result.new_shape.ToString(true) << " broadcast)";

  return result;
}

// Implements CalculateBitcastOfTranspose(), except that result.new_dims is
// the inverse permutation, mapping the input dimensions to the output
// dimensions.
absl::StatusOr<BitcastParams> CalculateBitcastOfTransposeImpl(
    const HloTransposeInstruction* transpose, const Shape& result_shape,
    const Shape& transpose_shape, const Shape& operand_shape,
    absl::Span<const int64_t> transpose_dims) {
  if (transpose->shape().layout() != transpose->operand(0)->shape().layout()) {
    return absl::InternalError(
        absl::StrCat("Expected input and output layouts to be the same for ",
                     transpose->ToString()));
  }

  // Maps physical operand dimension index to a range of physical result
  // dimensions.
  llvm::SmallVector<std::pair<int64_t, int64_t>> operand_to_result_range(
      operand_shape.dimensions().size());
  // Maps logical operand dimension index to the physical dimension index.
  llvm::SmallVector<int64_t> operand_inv_layout =
      GetInversePermutation(operand_shape.layout().minor_to_major());

  const absl::InlinedVector<std::pair<int64_t, int64_t>, 8> factors =
      ::xla::gpu::detail::CommonFactorsMergingTrivialRanges(
          GetPhysicalDimensions(result_shape),
          GetPhysicalDimensions(transpose_shape));
  for (int64_t i = 1; i < factors.size(); ++i) {
    auto [result_from, transpose_from] = factors[i - 1];
    auto [result_to, transpose_to] = factors[i];

    llvm::SmallVector<int64_t> indices;
    indices.reserve(transpose_to - transpose_from);
    for (int64_t j = transpose_from; j < transpose_to; ++j) {
      int64_t dim_index = transpose_shape.layout().minor_to_major(j);
      int64_t index = operand_inv_layout[transpose_dims[dim_index]];

      if (transpose_shape.dimensions(dim_index) == 1) {
        // Size-1 dimensions do not affect the physical layout, so we can ignore
        // them for the purpose of checking contiguity. We mark them with an
        // empty range in the operand_to_result_range map, so that they are
        // dropped from the new bitcast/transpose shape.
        operand_to_result_range[index] = {result_from, result_from};
        continue;
      }

      // Store the entire result dimension range in the minor-most dimension
      // index and an empty range in all others.
      operand_to_result_range[index] = {result_from, result_to};
      result_from = result_to;

      // Check that the physical operand indices form a contiguous range.
      indices.push_back(index);
    };

    if (indices.empty()) {
      // If all dimensions are size 1, we can just drop them.
      continue;
    }
    if (indices.back() - indices.front() >= indices.size() ||
        !absl::c_is_sorted(indices)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Cannot hoist bitcast across ", transpose->ToString(),
                       " because result dimensions are not contiguous."));
    }
  }

  BitcastParams result;
  CopyElementType(result_shape, &result.new_shape);
  // Just like the old transpose, the new transpose does not change the
  // layout.
  *result.new_shape.mutable_layout() = result_shape.layout();
  result.new_dims.resize(result_shape.dimensions().size());
  llvm::SmallVector<int64_t> new_shape_dims(result_shape.dimensions().size());
  // Iterate through the physical operand and new_shape dimension indices.
  for (int64_t i = 0, j = 0; i < operand_shape.dimensions().size(); ++i) {
    auto range = operand_to_result_range[i];
    // Iterate through corresponding range of physical result dimension
    // indices.
    for (int64_t k = range.first; k < range.second; ++k) {
      int64_t new_index = result_shape.layout().minor_to_major(j++);
      int64_t result_index = result_shape.layout().minor_to_major(k);
      new_shape_dims[new_index] = result_shape.dimensions(result_index);
      result.new_dims[new_index] = result_index;
    }
  }
  for (int64_t dimension : new_shape_dims) {
    result.new_shape.add_dimensions(dimension);
  }

  VLOG(3) << "CalculateBitcastOfTransposeImpl:";
  VLOG(3) << "  transpose = " << transpose_shape.ToString(true) << " transpose("
          << operand_shape.ToString(true)
          << " operand), dimensions=" << absl::StrJoin(transpose_dims, ",");
  VLOG(3) << "  result    = " << result_shape.ToString(true) << " bitcast("
          << transpose_shape.ToString(true) << " transpose)";
  VLOG(3) << "--------------------------------";
  VLOG(3) << "  bitcast   = " << result.new_shape.ToString(true) << " bitcast("
          << operand_shape.ToString(true) << " operand)";
  VLOG(3) << "  result    = " << result_shape.ToString(true) << " transpose("
          << result.new_shape.ToString(true) << " bitcast), dimensions="
          << absl::StrJoin(GetInversePermutation(result.new_dims), ",");

  return result;
}

// Returns parameters to rewrite a transpose + bitcast as bitcast + transpose.
//
// Example:
//
// transpose = transpose(operand)
// result = result_shape bitcast(transpose)
//
// to
//
// bitcast = new_shape bitcast(operand)
// result = transpose(bitcast), dimensions={new_dims}.
//
// Assumes that:
// - bitcast only mixes contiguous dimensions (checks);
// - transpose does not change layout (checks);
absl::StatusOr<BitcastParams> CalculateBitcastOfTranspose(
    const HloTransposeInstruction* transpose, const Shape& result_shape) {
  TF_ASSIGN_OR_RETURN(
      BitcastParams result,
      CalculateBitcastOfTransposeImpl(
          transpose, result_shape, transpose->shape(),
          transpose->operand(0)->shape(), transpose->dimensions()));
  result.new_dims = GetInversePermutation(result.new_dims);
  return result;
}

// Returns parameters to rewrite a bitcast + transpose as transpose + bitcast.
//
// Example:
//
// bitcast = bitcast(operand_shape operand)
// result = transpose(bitcast)
//
// to
//
// transpose = new_shape transpose(operand), dimensions={new_dims}.
// result = bitcast(transpose)
//
// Assumes that:
// - bitcast only mixes contiguous dimensions (checks);
// - transpose does not change layout (checks);
absl::StatusOr<BitcastParams> CalculateTransposeOfBitcast(
    const HloTransposeInstruction* transpose, const Shape& operand_shape) {
  return CalculateBitcastOfTransposeImpl(
      transpose, operand_shape, transpose->operand(0)->shape(),
      transpose->shape(), GetInversePermutation(transpose->dimensions()));
}

// Simulates a rewrite of all producers of a given bitcast/reshape, moving the
// instruction outside of the computation. Returns the new shapes of affected
// instructions in order of traversal from consumers to producers.
absl::StatusOr<std::vector<std::pair<HloInstruction*, Shape>>>
PlanHoistBitcastUpwardsToCallers(const HloInstruction* bitcast) {
  // Check that all producers only affect the bitcast. If there are any
  // other consumers: refuse the hoisting.
  // It is possible to support more cases by sinking the bitcast from such
  // producers downward.
  HloInstructionSetVector producers = GetProducerSet(bitcast);
  TF_RETURN_IF_ERROR(VerifyIsClosedProducerSet(producers, bitcast));
  if (bitcast->shape().element_type() !=
      bitcast->operand(0)->shape().element_type()) {
    return absl::UnimplementedError(
        absl::StrCat("Hoisting bitcast with type conversion is not supported: ",
                     bitcast->ToString()));
  }

  HloInstructionMap<Shape> result_shapes;
  auto set_result_shape =
      [&](const absl::Span<HloInstruction* const> instructions,
          const Shape& shape) -> absl::Status {
    for (HloInstruction* instruction : instructions) {
      // Only update the dimensions keeping the type intact.
      Shape new_shape(shape);
      CopyElementType(instruction->shape(), &new_shape);
      CHECK_EQ(ShapeUtil::ArrayDataSize(new_shape),
               ShapeUtil::ArrayDataSize(instruction->shape()))
          << " instruction " << instruction->ToString()
          << " updating result shape from "
          << ShapeUtil::HumanStringWithLayout(instruction->shape()) << " to "
          << ShapeUtil::HumanStringWithLayout(new_shape)
          << " with different data size";
      auto it = result_shapes.find(instruction);
      if (it == result_shapes.end()) {
        VLOG(2) << "updating the result shape of " << instruction->ToString()
                << " to " << ShapeUtil::HumanStringWithLayout(new_shape);
        result_shapes.emplace(instruction, new_shape);
      } else if (it->second != new_shape) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Conflicting shape assignment for ", instruction->ToString(),
            " got ", ShapeUtil::HumanStringWithLayout(it->second), " and ",
            ShapeUtil::HumanStringWithLayout(shape)));
      }
    }
    return absl::OkStatus();
  };
  TF_RETURN_IF_ERROR(set_result_shape(bitcast->operands(), bitcast->shape()));

  std::vector<std::pair<HloInstruction*, Shape>> result;
  // We want to visit instructions in order from consumers to producers: we
  // hoist the bitcast upwards and having a valid HLO at every rewrite step
  // helps a lot. A simple DFS or BFS over operands will not work in non-tree
  // situations when there are multiple consumers of the same producer. Instead
  // of writing a custom traversal we can simply walk the post-order (producers
  // before consumers) list backward and only update the instructions affected.
  // TODO(b/393299275): use MakeInstructionPostOrderFrom(bitcast) - that should
  // be slightly more efficient.
  auto def_before_use = bitcast->parent()->MakeInstructionPostOrder();
  for (HloInstruction* instruction :
       llvm::make_range(def_before_use.rbegin(), def_before_use.rend())) {
    auto it = result_shapes.find(instruction);
    if (it == result_shapes.end()) {
      continue;  // Not affected.
    }
    Shape& result_shape = it->second;
    if (instruction->shape() == result_shape) {
      continue;  // No change.
    }
    result.emplace_back(instruction, result_shape);
    switch (instruction->opcode()) {
      case HloOpcode::kParameter:
      case HloOpcode::kConstant:
        // No operands.
        break;
      case HloOpcode::kReshape:  // Reshape is a bitcast.
      case HloOpcode::kBitcast:
        // Other bitcast will be hoisted separately so we don't need to
        // update its operand.
        break;
      case HloOpcode::kBroadcast: {
        TF_ASSIGN_OR_RETURN(
            BitcastParams params,
            CalculateBitcastOfBroadcast(
                Cast<HloBroadcastInstruction>(instruction), result_shape));
        TF_RETURN_IF_ERROR(
            set_result_shape(instruction->operands(), params.new_shape));
        break;
      }
      case HloOpcode::kTranspose: {
        TF_ASSIGN_OR_RETURN(
            BitcastParams params,
            CalculateBitcastOfTranspose(
                Cast<HloTransposeInstruction>(instruction), result_shape));
        TF_RETURN_IF_ERROR(
            set_result_shape(instruction->operands(), params.new_shape));
        break;
      }
      default:
        if (!instruction->IsElementwise()) {
          return absl::FailedPreconditionError(absl::StrCat(
              "Cannot hoist bitcast past ", instruction->ToString()));
        }
        TF_RETURN_IF_ERROR(
            set_result_shape(instruction->operands(), result_shape));
        break;
    }
  }
  return result;
}

// Returns the shape of the root instruction after hoisting bitcasts away from
// the dot instruction. If traversal encounters an instruction we cannot hoist
// bitcasts past we try to sink the bitcast starting from that instruction.
//
// For example, given:
//
// dot = dot_shape dot
// bitcast = bitcast(dot)
// ROOT root = transpose(bitcast)
//
// Returns root_shape for:
//
// dot = dot_shape dot
// ROOT root = roots_shape transpose(dot)
//
absl::StatusOr<Shape> ComputeRootShapeAfterHoistingBitcasts(
    const HloInstruction* dot) {
  if (dot->IsRoot()) {
    return dot->shape();
  }

  HloInstructionMap<Shape> operand_shapes;
  auto set_operand_shape =
      [&](const absl::Span<HloInstruction* const> instructions,
          const Shape& shape) -> absl::Status {
    for (HloInstruction* instruction : instructions) {
      // Only update the dimensions keeping the type intact.
      Shape new_shape(shape);
      const HloInstruction* operand = instruction->operand(0);
      CopyElementType(operand->shape(), &new_shape);
      CHECK_EQ(ShapeUtil::ArrayDataSize(new_shape),
               ShapeUtil::ArrayDataSize(operand->shape()))
          << " instruction " << instruction->ToString()
          << " updating operand shape from "
          << ShapeUtil::HumanStringWithLayout(operand->shape()) << " to "
          << ShapeUtil::HumanStringWithLayout(new_shape)
          << " with different data size";
      auto it = operand_shapes.find(instruction);
      if (it == operand_shapes.end()) {
        VLOG(2) << "updating the operand shape of "
                << instruction->ToString(
                       HloPrintOptions().set_print_operand_shape(true))
                << " to " << ShapeUtil::HumanStringWithLayout(new_shape);
        operand_shapes.emplace(instruction, new_shape);
      } else if (it->second != new_shape) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Conflicting shape assignment for ", instruction->ToString(),
            " got ", ShapeUtil::HumanStringWithLayout(it->second), " and ",
            ShapeUtil::HumanStringWithLayout(shape)));
      }
    }
    return absl::OkStatus();
  };
  TF_RETURN_IF_ERROR(set_operand_shape(dot->users(), dot->shape()));

  for (HloInstruction* instruction : GetConsumerSet(dot)) {
    auto it = operand_shapes.find(instruction);
    if (it == operand_shapes.end()) {
      continue;  // Not affected.
    }
    Shape& operand_shape = it->second;
    TF_ASSIGN_OR_RETURN(Shape result_shape, [&]() -> absl::StatusOr<Shape> {
      switch (instruction->opcode()) {
        case HloOpcode::kBroadcast: {
          auto paramsOr = CalculateBroadcastOfBitcast(
              Cast<HloBroadcastInstruction>(instruction), operand_shape);
          if (paramsOr.ok()) {
            return paramsOr->new_shape;
          }
          VLOG(2) << "Failed to calculate broadcast of bitcast: "
                  << paramsOr.status();
          return instruction->shape();
        }
        case HloOpcode::kTranspose: {
          auto paramsOr = CalculateTransposeOfBitcast(
              Cast<HloTransposeInstruction>(instruction), operand_shape);
          if (paramsOr.ok()) {
            return paramsOr->new_shape;
          }
          VLOG(2) << "Failed to calculate transpose of bitcast: "
                  << paramsOr.status();
          return instruction->shape();
        }
        case HloOpcode::kReshape:
        case HloOpcode::kBitcast:
          return operand_shape;
        default:
          if (instruction->IsElementwise()) {
            return operand_shape;
          }
          // TODO(b/467421789): we can probably allow sinking from this op down.
          return absl::FailedPreconditionError(absl::StrCat(
              "Cannot hoist bitcast past ", instruction->ToString()));
      }
    }());
    if (instruction->IsRoot()) {
      CopyElementType(instruction->shape(), &result_shape);
      return result_shape;
    }
    TF_RETURN_IF_ERROR(set_operand_shape(instruction->users(), result_shape));
  }
  return absl::InternalError("No root found");
}

// Hoists the given 'bitcast' upwards out of its computation, to the parent of
// each caller.
absl::Status HoistBitcastUpwardsToCallers(HloInstruction* bitcast,
                                          absl::Span<HloInstruction*> callers) {
  TF_ASSIGN_OR_RETURN(auto rewrite_plan,
                      PlanHoistBitcastUpwardsToCallers(bitcast));
  for (auto [instruction, result_shape] : rewrite_plan) {
    VLOG(2) << absl::StrCat("rewriting result shape of ",
                            instruction->ToString(), " to ",
                            ShapeUtil::HumanStringWithLayout(result_shape));
    switch (instruction->opcode()) {
      case HloOpcode::kParameter: {
        // Create a new bitcast in callers.
        int64_t number = instruction->parameter_number();
        for (HloInstruction* caller : callers) {
          // Create a more generic `bitcast` even if the caller has a
          // `reshape`.
          HloInstruction* new_bitcast =
              caller->AddInstruction(HloInstruction::CreateBitcast(
                  result_shape, caller->mutable_operand(number)));
          TF_RETURN_IF_ERROR(
              caller->ReplaceOperandWithDifferentShape(number, new_bitcast));
        }
        break;
      }
      case HloOpcode::kBroadcast: {
        auto* broadcast = Cast<HloBroadcastInstruction>(instruction);
        auto params = CalculateBitcastOfBroadcast(broadcast, result_shape);
        // Must be OK, already succeeded in PlanHoistBitcasUpwardsToCallers.
        QCHECK_OK(params);
        broadcast->mutable_dimensions()->assign(params->new_dims.begin(),
                                                params->new_dims.end());
        break;
      }
      case HloOpcode::kTranspose: {
        auto* transpose = Cast<HloTransposeInstruction>(instruction);
        auto params = CalculateBitcastOfTranspose(transpose, result_shape);
        // Must be OK, already succeeded in PlanHoistBitcastUpwardsToCallers.
        QCHECK_OK(params);
        transpose->mutable_dimensions()->assign(params->new_dims.begin(),
                                                params->new_dims.end());
        break;
      }
      default:
        break;
    }
    *instruction->mutable_shape() = result_shape;
  }
  TF_RETURN_IF_ERROR(bitcast->ReplaceAllUsesWith(bitcast->mutable_operand(0)));
  TF_RETURN_IF_ERROR(bitcast->parent()->RemoveInstruction(bitcast));
  return absl::OkStatus();
}

// Inserts a bitcast at the root if the root shape is different from the dot
// shape. The bitcast is chosen so that it cancels out bitcasts and reshapes
// along the way up to the dot. Updates the callers of the dot to expect the new
// root shape.
absl::StatusOr<bool> MaybeInsertRootBitcast(
    HloInstruction* dot, absl::Span<HloInstruction*> callers) {
  TF_ASSIGN_OR_RETURN(Shape root_shape,
                      ComputeRootShapeAfterHoistingBitcasts(dot));

  HloComputation* computation = dot->parent();
  HloInstruction* root = computation->root_instruction();
  if (root->shape() == root_shape) {
    return false;
  }

  // Insert a new bitcast at the root.
  computation->set_root_instruction(
      root->AddInstruction(HloInstruction::CreateBitcast(root_shape, root)));

  // Insert new bitcast for each caller's result.
  for (HloInstruction* caller : callers) {
    HloInstruction* new_bitcast = caller->AddInstruction(
        HloInstruction::CreateBitcast(caller->shape(), caller));
    TF_RETURN_IF_ERROR(caller->ReplaceAllUsesWith(new_bitcast));
    *caller->mutable_shape() = root_shape;
  }

  return true;
}

// Try hoisting bitcasts and reshapes in the computation away from 'dot' to the
// callers of the computation. Some bitcasts or reshapes may remain in the
// computation, because they cannot be hoisted across all ops, e.g. across some
// transposes and broadcasts. This is not reported as an error.
absl::StatusOr<bool> TryHoistBitcastsInComputationToCallers(
    HloInstruction* dot, CallGraph* call_graph) {
  bool changed = false;
  // Instead of implementing a logic to hoist bitcast upwards and downwards
  // we insert a bitcast at the root that and always hoist bitcasts upwards.
  // That significantly simplifies the implementation.
  VLOG(2) << "Before hoisting bitcasts: " << dot->parent()->ToString();

  auto callers = call_graph->GetComputationCallers(dot->parent());
  absl::StatusOr<bool> inserted =
      MaybeInsertRootBitcast(dot, absl::MakeSpan(callers));
  if (!inserted.ok()) {
    VLOG(2) << "Failed to insert root bitcast: " << inserted.status();
  } else {
    changed |= *inserted;
  }
  VLOG(2) << "After inserting root bitcast: " << dot->parent()->ToString();

  auto def_before_use = dot->parent()->MakeInstructionPostOrder();
  for (HloInstruction* instruction :
       llvm::make_range(def_before_use.rbegin(), def_before_use.rend())) {
    if (!HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kReshape>(
            instruction)) {
      continue;
    }
    VLOG(2) << "Hoisting bitcast upwards " << instruction->ToString();
    auto status =
        HoistBitcastUpwardsToCallers(instruction, absl::MakeSpan(callers));
    if (!status.ok()) {
      VLOG(2) << "Failed to hoist " << instruction->ToString()
              << " upwards: " << status;
    } else {
      changed = true;
    }
  }

  VLOG(2) << "After hoisting bitcasts: " << dot->parent()->ToString();
  return changed;
}

class HoistFusedBitcastsVisitor : public DfsHloRewriteVisitor {
 public:
  explicit HoistFusedBitcastsVisitor(CallGraph* call_graph)
      : call_graph_(call_graph) {}

 private:
  absl::Status RewriteFusion(HloFusionInstruction* fusion,
                             CallGraph* call_graph) {
    HloComputation* computation = fusion->called_computation();
    HloInstruction* instr =
        hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
    if (instr == nullptr) {
      instr = hlo_query::GetFirstInstructionWithOpcode(*computation,
                                                       HloOpcode::kScaledDot);
      if (instr == nullptr) {
        return absl::InternalError(absl::StrCat("Computation of fusion ",
                                                fusion->ToString(),
                                                " has no dot instruction"));
      }
    }

    ASSIGN_OR_RETURN(bool changed,
                     TryHoistBitcastsInComputationToCallers(instr, call_graph));
    if (changed) {
      MarkAsChanged();
    }
    return absl::OkStatus();
  }

  absl::Status HandleFusion(HloInstruction* instruction) override {
    HloFusionInstruction* fusion = Cast<HloFusionInstruction>(instruction);

    // Check if we target this fusion.
    absl::StatusOr<TritonGemmConfig> config = GetTritonGemmConfig(*fusion);
    if (!config.ok()) {
      VLOG(2) << "Skipping fusion as it does not have a TritonGemmConfig";
      return absl::OkStatus();
    }
    HloComputation* computation = fusion->called_computation();
    HloInstruction* instr =
        hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
    if (instr == nullptr) {
      instr = hlo_query::GetFirstInstructionWithOpcode(*computation,
                                                       HloOpcode::kScaledDot);
      if (instr == nullptr) {
        VLOG(2) << "Skipping fusion as it has no dot instruction";
        return absl::OkStatus();
      }
    }
    return RewriteFusion(fusion, call_graph_);
  }

 private:
  CallGraph* call_graph_;
};

}  // namespace

absl::StatusOr<bool> HoistFusedBitcasts::RunOnModule(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  auto call_graph = CallGraph::Build(module, execution_threads);
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    HoistFusedBitcastsVisitor visitor(call_graph.get());
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
    changed |= visitor.changed();
  }
  return changed;
}

absl::StatusOr<bool> HoistFusedBitcasts::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return RunOnModule(module, execution_threads);
}

namespace detail {

absl::InlinedVector<std::pair<int64_t, int64_t>, 8>
CommonFactorsMergingTrivialRanges(absl::Span<const int64_t> a,
                                  absl::Span<const int64_t> b) {
  // CommonFactors does what we need but it also creates empty groups with
  // product of 1, e.g. `[1] -> []` or `[] -> [1]`. We remove the bounds of
  // such ranges to merge them with neighbors. There are many different ways
  // to do this, here we continously append ranges to the start of the next
  // group unless it is the very last range.
  absl::InlinedVector<std::pair<int64_t, int64_t>, 8> bounds =
      CommonFactors(a, b);
  for (size_t i = 0; i + 1 < bounds.size() && bounds.size() > 2;) {
    auto [a_start, b_start] = bounds[i];
    auto [a_end, b_end] = bounds[i + 1];
    if (a_start != a_end && b_start != b_end) {
      i++;
      continue;
    }
    if (i + 2 == bounds.size()) {
      // Very last range - append it to the previous one.
      bounds.erase(bounds.begin() + i);
    } else {
      bounds.erase(bounds.begin() + i + 1);
    }
  }
  return bounds;
}

}  // namespace detail
}  // namespace xla::gpu
