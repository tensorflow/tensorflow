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

#include "xla/service/gpu/ir_emission_utils.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/literal.h"
#include "xla/permutation_util.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

// Return whether the given shape is rank 2 excluding the batch dimensions.
bool IsRank2(const Shape& shape, int64_t batch_dimensions_size) {
  return shape.dimensions().size() == batch_dimensions_size + 2;
}

// Return whether the given shape is rank 1 excluding the batch dimensions.
bool IsRank1(const Shape& shape, int64_t batch_dimensions_size) {
  return shape.dimensions().size() == batch_dimensions_size + 1;
}

}  // namespace

bool IsMatrixMultiplication(const HloInstruction& dot) {
  if (dot.opcode() != HloOpcode::kDot) {
    return false;
  }
  const Shape& lhs_shape = dot.operand(0)->shape();
  const Shape& rhs_shape = dot.operand(1)->shape();
  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();

  PrimitiveType output_primitive_type = dot.shape().element_type();
  bool type_is_allowed =
      (output_primitive_type == F8E3M4 || output_primitive_type == F8E4M3 ||
       output_primitive_type == F8E4M3FN || output_primitive_type == F8E5M2 ||
       output_primitive_type == F8E4M3FNUZ ||
       output_primitive_type == F8E5M2FNUZ || output_primitive_type == F16 ||
       output_primitive_type == BF16 || output_primitive_type == F32 ||
       output_primitive_type == F64 || output_primitive_type == C64 ||
       output_primitive_type == C128) ||
      (output_primitive_type == S32 && lhs_shape.element_type() == S8 &&
       rhs_shape.element_type() == S8);
  bool shapes_are_valid =
      type_is_allowed &&
      IsRank2(lhs_shape, dim_numbers.lhs_batch_dimensions_size()) &&
      IsRank2(rhs_shape, dim_numbers.lhs_batch_dimensions_size()) &&
      IsRank2(dot.shape(), dim_numbers.lhs_batch_dimensions_size()) &&
      !ShapeUtil::IsZeroElementArray(lhs_shape) &&
      !ShapeUtil::IsZeroElementArray(rhs_shape);

  return shapes_are_valid;
}

bool IsMatrixVectorMultiplication(const HloInstruction& dot) {
  if (dot.opcode() != HloOpcode::kDot) {
    return false;
  }
  const Shape& lhs_shape = dot.operand(0)->shape();
  const Shape& rhs_shape = dot.operand(1)->shape();
  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();

  PrimitiveType output_primitive_type = dot.shape().element_type();
  bool type_is_allowed =
      (output_primitive_type == F8E4M3FN || output_primitive_type == F8E5M2 ||
       output_primitive_type == F16 || output_primitive_type == BF16 ||
       output_primitive_type == F32 || output_primitive_type == F64 ||
       output_primitive_type == C64 || output_primitive_type == C128) ||
      (output_primitive_type == S32 && lhs_shape.element_type() == S8 &&
       rhs_shape.element_type() == S8);

  bool shapes_are_valid =
      type_is_allowed &&
      ((IsRank2(lhs_shape, dim_numbers.lhs_batch_dimensions_size()) &&
        IsRank1(rhs_shape, dim_numbers.lhs_batch_dimensions_size())) ||
       (IsRank1(lhs_shape, dim_numbers.lhs_batch_dimensions_size()) &&
        IsRank2(rhs_shape, dim_numbers.lhs_batch_dimensions_size()))) &&
      IsRank1(dot.shape(), dim_numbers.lhs_batch_dimensions_size()) &&
      !ShapeUtil::IsZeroElementArray(lhs_shape) &&
      !ShapeUtil::IsZeroElementArray(rhs_shape);

  return shapes_are_valid;
}

const char* const kCusolverCholeskyCallTarget = "__cusolver$cholesky";

bool IsCustomCallToCusolver(const HloInstruction& hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  return hlo.custom_call_target() == kCusolverCholeskyCallTarget;
}

bool IsCustomCallToTopK(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kTopKCustomCallTarget;
}

bool IsSliceWithUnitStrides(const HloInstruction* instr) {
  auto slice = DynCast<HloSliceInstruction>(instr);
  return slice && absl::c_all_of(slice->slice_strides(),
                                 [](int64_t stride) { return stride == 1; });
}

static bool IsContiguousSlice(
    const Shape& orig, const Shape& sliced,
    std::optional<absl::Span<const int64_t>> slice_strides) {
  std::optional<int64_t> sliced_dim;

  for (auto dim : orig.layout().minor_to_major()) {
    // All dimensions before the sliced one must be 1.
    if (sliced_dim.has_value()) {
      if (sliced.dimensions(dim) != 1) return false;
    }

    // We found sliced dimension, check that it's not a strided one, because it
    // means that we can't take a contiguous slice.
    if (sliced.dimensions(dim) < orig.dimensions(dim)) {
      if (slice_strides.has_value() && slice_strides.value()[dim] != 1 &&
          sliced.dimensions(dim) > 1) {
        return false;
      }
      sliced_dim = dim;
    }
  }
  return true;
}

bool IsContiguousSlice(const HloInstruction& instr) {
  if (auto slice = DynCast<HloSliceInstruction>(&instr)) {
    const Shape& full_shape = slice->operand(0)->shape();
    const Shape& slice_shape = slice->shape();
    return IsContiguousSlice(full_shape, slice_shape, slice->slice_strides());

  } else if (auto slice = DynCast<HloDynamicSliceInstruction>(&instr)) {
    const Shape& full_shape = slice->operand(0)->shape();
    const Shape& slice_shape = slice->shape();
    return IsContiguousSlice(full_shape, slice_shape, std::nullopt);

  } else if (auto slice = DynCast<HloDynamicUpdateSliceInstruction>(&instr)) {
    const Shape& full_shape = slice->shape();
    const Shape& slice_shape = slice->update()->shape();
    return IsContiguousSlice(full_shape, slice_shape, std::nullopt);
  }
  return false;
}

llvm::Value* IsBlock0Thread0(llvm::IRBuilderBase* b) {
  llvm::Value* is_thread0 = b->CreateICmpEQ(
      b->getInt32(0),
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kThreadIdx, {}, {}, b));

  llvm::Value* is_block0 = b->CreateICmpEQ(
      b->getInt32(0),
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kBlockIdx, {}, {}, b));
  return b->CreateAnd(is_thread0, is_block0);
}

absl::StatusOr<BufferAllocation::Slice> GetAllocationSlice(
    const BufferAssignment& buffer_assignment, const HloInstruction* instr,
    const ShapeIndex& index) {
  return buffer_assignment.GetUniqueSlice(instr, index);
}

std::vector<HloInstructionAdaptor> GetOutputDefiningDynamicUpdateSlices(
    absl::Span<HloInstructionAdaptor const> roots) {
  std::vector<HloInstructionAdaptor> dus_ops;
  for (HloInstructionAdaptor root : roots) {
    while (root.opcode() == HloOpcode::kBitcast) {
      root = root.GetOperand(0);
    }

    if (root.opcode() == HloOpcode::kDynamicUpdateSlice) {
      dus_ops.push_back(root);
    }
  }
  return dus_ops;
}

template <typename T>
absl::InlinedVector<const HloInstruction*, 4> GetStartIndices(T instr) {
  absl::InlinedVector<const HloInstruction*, 4> result;
  for (int i = instr->first_index_operand_number(); i < instr->operand_count();
       i++) {
    const HloInstruction* index = instr->operand(i);
    result.push_back(index);
  }
  return result;
}

absl::StatusOr<bool> CanEmitFusedDynamicUpdateSliceInPlaceForGpu(
    const HloFusionAdaptor& fusion_adaptor,
    std::function<absl::StatusOr<BufferAllocation::Slice>(
        const HloInstruction* instr, const ShapeIndex& index)>
        get_allocation_slice,
    const HloInstruction* fusion) {
  std::vector<HloInstructionAdaptor> dus_instrs =
      GetOutputDefiningDynamicUpdateSlices(fusion_adaptor.GetRoots());

  // This check could probably be relaxed: if code generation is made to use a
  // separate parallel loop for each dynamic slice update, then it shouldn't be
  // necessary for every output to be a dynamic slice update, nor to have the
  // same shape.
  if (dus_instrs.size() != fusion_adaptor.GetRoots().size()) {
    return false;
  }

  Shape update_shape = dus_instrs[0].GetOperand(1).shape();

  for (int i = 0; i < dus_instrs.size(); ++i) {
    const auto& dus = dus_instrs[i];

    // DynamicUpdateSlice ops should have a single path to the root to avoid
    // allowing a dynamic slice update to depend on another, as this would not
    // be guaranteed to work with the current codegen.
    // We follow DUS users until we find an instruction without users. We
    // support only few patterns:
    //
    //   (1) ROOT dynamic-update-slice
    //   (2) ROOT tuple(dynamic-update-slice)
    //   (3) ROOT bitcast(dynamic-update-slice)
    //   (4) ROOT tuple(bitcast(dynamic-update-slice))
    //
    // In case there is a root tuple, the search will stop at the tuple operand,
    // as the root tuple is not considered a real user by HloInstructionAdaptor.
    // Note that due to AlgebraicSimplifier we will never have a chain of
    // bitcasts.
    HloInstructionAdaptor real_root = dus;
    auto users = real_root.GetUsers();
    while (!users.empty()) {
      if (users.size() > 1) {
        return false;
      }
      real_root = users.front();
      if (real_root.opcode() != HloOpcode::kBitcast) {
        return false;
      }
      users = real_root.GetUsers();
    }

    // Find "real" DUS operand by skipping bitcasted operands.
    HloInstructionAdaptor operand = dus.GetOperand(0);
    if (fusion_adaptor.ContainsInstruction(operand) &&
        operand.opcode() == HloOpcode::kBitcast) {
      operand = operand.GetOperand(0);
    }

    // Operand to a DUS (or Bitcast) must be a fusion parameter.
    // HloInstructionAdaptor skips parameters, so we need to check whether
    // 'operand' is outside of the fusion.
    if (fusion_adaptor.ContainsInstruction(operand)) {
      return false;
    }

    // We require that the parameter being updated is only read at the same
    // index positions by all users, since we otherwise risk a race condition
    // when updating the parameter inplace.
    std::queue<HloInstructionAdaptor> q;
    absl::flat_hash_set<const HloInstruction*> visited;
    q.push(operand);
    visited.insert(&operand.instruction());
    // We have already checked above that the DUS only has one user. So we don't
    // need to visit it during the breadth-first search.
    visited.insert(&dus.instruction());
    while (!q.empty()) {
      HloInstructionAdaptor instr = q.front();
      q.pop();
      for (const HloInstructionAdaptor& user : instr.GetUsers()) {
        if (user.opcode() == HloOpcode::kDynamicSlice &&
            dus.GetOperand(0) == user.GetOperand(0) &&
            update_shape == user.shape()) {
          // We can still emit in-place in this case if the same slice is
          // accessed by the DUS and the DS. If they don't access the same
          // slice, the two slices might partially overlap and read/write the
          // same index at different times, and then we cannot guarantee that we
          // read before it is overwritten. However if both access only a single
          // element, there also can be no race condition.
          absl::InlinedVector<const HloInstruction*, 4> user_start_indices =
              GetStartIndices(
                  Cast<HloDynamicSliceInstruction>(&user.instruction()));
          absl::InlinedVector<const HloInstruction*, 4> dus_start_indices =
              GetStartIndices(
                  Cast<HloDynamicUpdateSliceInstruction>(&dus.instruction()));
          if (ShapeUtil::ElementsIn(update_shape) != 1 &&
              user_start_indices != dus_start_indices) {
            return false;
          }
        } else if (user != dus &&
                   user.opcode() == HloOpcode::kDynamicUpdateSlice) {
          return false;
        } else if (user != dus && !user.instruction().IsElementwise() &&
                   user.opcode() != HloOpcode::kBitcast &&
                   user.opcode() != HloOpcode::kTuple) {
          return false;
        }
        if (visited.insert(&user.instruction()).second) {
          q.push(user);
        }
      }
    }

    // This check could probably be relaxed: if code generation is made to use a
    // separate parallel loop for each dynamic slice update, then it shouldn't
    // be necessary for the shape to be the same for all the dynamic slice
    // updates. Note that this equality check purposefully ignores the element
    // type.
    if (Cast<HloDynamicUpdateSliceInstruction>(&dus.instruction())
            ->update()
            ->shape() != update_shape) {
      return false;
    }

    if (fusion != nullptr) {
      ShapeIndex root_index = {};
      if (fusion->IsMultiOutputFusion()) {
        root_index = {i};
      }
      // Get output buffer for the fusion root.
      TF_ASSIGN_OR_RETURN(BufferAllocation::Slice output_buffer,
                          get_allocation_slice(fusion, root_index));

      TF_ASSIGN_OR_RETURN(BufferAllocation::Slice lhs_buffer,
                          get_allocation_slice(&operand.instruction(), {}));
      if (lhs_buffer != output_buffer) {
        return false;
      }
    }
  }

  return true;
}

int GetBitwidth(PrimitiveType type) {
  if (type == PRED) {
    return 8;
  }
  return primitive_util::BitWidth(type);
}

bool IsNormalized(const HloTransposeInstruction& transpose) {
  const auto& permutation = transpose.dimensions();
  for (int i = 0; i < permutation.size() - 1; ++i) {
    if (permutation[i] + 1 == permutation[i + 1]) {
      return false;
    }
  }
  return true;
}

bool CanEmitPackedTranspose(const HloTransposeInstruction& transpose) {
  // Support only normalized transposes.
  if (!IsNormalized(transpose)) {
    return false;
  }
  const auto& spec = GetTransposeSpec(&transpose);
  return GetPackedTransposeTileSizes(spec).ok();
}

std::optional<TransposeDescription> GetDescriptionForTiledTransposeEmitter(
    const HloInstruction& hero) {
  if (hero.opcode() != HloOpcode::kTranspose) {
    return std::nullopt;
  }

  // We can assume that TransposeDimensionGrouper pass has run, so no need to
  // call GetNormalizedLogicalTransposeShape here.
  absl::InlinedVector<int64_t, 3> permutation(hero.dimensions().begin(),
                                              hero.dimensions().end());
  // A real transpose needs at least 2 transpose dimensions.
  if (permutation.size() < 2) {
    return std::nullopt;
  }
  auto byte_width = primitive_util::ByteWidth(hero.shape().element_type());
  absl::InlinedVector<int64_t, 3> dimensions(hero.shape().dimensions().begin(),
                                             hero.shape().dimensions().end());
  int64_t operand_most_minor_dim = hero.operand(0)->shape().dimensions().back();
  if (CanEmitPackedTranspose(*Cast<HloTransposeInstruction>(&hero))) {
    int64_t vector_size =
        kBankBitwidth / GetBitwidth(hero.shape().element_type());
    int64_t shmem_usage_bytes =
        kNumShmemBanks * (kBankBitwidth / 8) * kNumShmemBanks * vector_size;
    return TransposeDescription{&hero, dimensions, permutation,
                                shmem_usage_bytes};
  }
  if (permutation.back() == dimensions.size() - 1) {
    operand_most_minor_dim =
        hero.operand(0)->shape().dimensions(dimensions.size() - 2);
    if (byte_width * dimensions.back() <= kMaxBytesInMostMinorDimension &&
        byte_width * dimensions.back() *
                std::min(operand_most_minor_dim,
                         dimensions[dimensions.size() - 2]) >=
            kMinDimensionToTransposeTiled) {
      // Tile size for transposition.
      int64_t shmem_usage_bytes = kNumShmemBanks * (kNumShmemBanks + 1) *
                                  byte_width * dimensions.back();
      return TransposeDescription{&hero, dimensions, permutation,
                                  shmem_usage_bytes};
    }
  } else if ((operand_most_minor_dim >= kMinDimensionToTransposeTiled &&
              dimensions.back() >= kMinDimensionToTransposeTiled) ||
             (operand_most_minor_dim >= kMinDimensionToTransposeTiled2 &&
              dimensions.back() >= kMinDimensionToTransposeTiled2 &&
              operand_most_minor_dim * dimensions.back() >=
                  kMinTotalDimensionsToTransposeTiled)) {
    int64_t shmem_usage_bytes =
        kNumShmemBanks * (kNumShmemBanks + 1) * byte_width;
    return TransposeDescription{&hero, dimensions, permutation,
                                shmem_usage_bytes};
  }
  return std::nullopt;
}

TransposeSpec GetTransposeSpec(const HloTransposeInstruction* transpose) {
  auto inv_permutation = InversePermutation(transpose->dimensions());
  auto& output_shape = transpose->shape();
  llvm::SmallVector<int64_t, 3> canonical_output_shape =
      llvm::to_vector<3>(output_shape.dimensions());
  llvm::SmallVector<int64_t, 3> canonical_permutation =
      llvm::to_vector<3>(transpose->dimensions());

  // If the last dimension is transposed, add a size-1 B dimension.
  if (canonical_permutation.back() != canonical_output_shape.size() - 1) {
    canonical_permutation.push_back(output_shape.dimensions().size());
    canonical_output_shape.push_back(1);
  }
  int64_t dim_t1 = -1;
  int64_t dim_t2 = -1;
  for (int64_t i = canonical_permutation.size() - 1; i >= 0; --i) {
    if (canonical_permutation[i] != i) {
      dim_t2 = canonical_permutation[i];
      dim_t1 = i;
      break;
    }
  }
  // Insert size-1 A dimension if necessary.
  auto rank = canonical_output_shape.size();
  if (canonical_permutation[rank - 3] != rank - 3) {
    canonical_output_shape.insert(canonical_output_shape.begin() + dim_t1, 1);
    for (auto& p : canonical_permutation) {
      if (p > rank - 3) p++;
    }
    canonical_permutation.insert(canonical_permutation.begin() + dim_t1,
                                 dim_t1);
  }
  auto canonical_inv_permutation = InversePermutation(canonical_permutation);
  auto canonical_input_shape =
      Permute(canonical_output_shape, canonical_inv_permutation);
  return TransposeSpec{
      transpose,
      llvm::to_vector<3>(transpose->dimensions()),
      llvm::to_vector<3>(inv_permutation),
      canonical_output_shape,
      canonical_permutation,
      llvm::to_vector<3>(canonical_inv_permutation),
      llvm::to_vector<3>(canonical_input_shape),
  };
}

std::string TransposeSpec::ToString() const {
  return absl::Substitute(R"(
transpose: $0
canonical_input_shape: $1
canonical_output_shape: $2
canonical_permutation: $3
canonical_inv_permutation: $4
[T2, A, T1, B] = [$5, $6, $7, $8]
)",
                          transpose->ToString(),
                          absl::StrJoin(canonical_input_shape, ","),
                          absl::StrJoin(canonical_output_shape, ","),
                          absl::StrJoin(canonical_permutation, ","),
                          absl::StrJoin(canonical_inv_permutation, ","),
                          dim_T2(), dim_A(), dim_T1(), dim_B());
}

absl::StatusOr<absl::InlinedVector<int64_t, 3>> GetPackedTransposeTileSizes(
    const TransposeSpec& spec) {
  // Check the side outputs, etc.
  int64_t bits_per_element = GetBitwidth(spec.elem_type());
  if (bits_per_element >= kBankBitwidth) {
    return absl::InvalidArgumentError("Element type is too large");
  }
  absl::InlinedVector<int64_t, 3> tile_sizes(spec.canonical_rank(), 1);
  int64_t vector_size = kBankBitwidth / bits_per_element;

  // The shmem size is `shmem_dim x shmem_dim`.
  int64_t shmem_dim = kNumShmemBanks * vector_size;
  int64_t tile_size_T1 = std::min(spec.dim_T1(), shmem_dim);
  int64_t tile_size_A = std::min(spec.dim_A(), shmem_dim / tile_size_T1);
  int64_t tile_size_T2 = std::min(spec.dim_T2(), shmem_dim);
  int64_t populated_shmem_rows = tile_size_T2;
  int64_t populated_shmem_cols = tile_size_A * tile_size_T1;

  // Do not use the packed transpose if there are not enough populated rows or
  // columns in shmem.
  const int64_t kNumMinPopulatedRowsOrColumns = 10 * vector_size;
  if (populated_shmem_cols < kNumMinPopulatedRowsOrColumns ||
      populated_shmem_rows < kNumMinPopulatedRowsOrColumns) {
    return absl::InvalidArgumentError("Not enough rows or columns in shmem");
  }

  // These divisibility constrains are too strict, we can do better.
  if (spec.dim_B() != 1 || populated_shmem_rows % vector_size != 0 ||
      populated_shmem_cols % vector_size != 0 ||
      spec.dim_T2() % tile_size_T2 % vector_size != 0) {
    return absl::InvalidArgumentError("The shape is not supported");
  }
  tile_sizes[spec.dim_T1_output_id()] = tile_size_T1;
  tile_sizes[spec.dim_T2_output_id()] = tile_size_T2;
  tile_sizes[spec.dim_A_id()] = tile_size_A;
  return tile_sizes;
}

bool IsIntermediate(const HloInstruction* instr, int allowed_operand_count) {
  // Number of operands should be in range [1, allowed_operand_count].
  if (instr->operand_count() == 0 ||
      instr->operand_count() > allowed_operand_count) {
    return false;
  }

  if (instr->IsElementwise()) {
    // All elementwise ops are considered intermediate, except for copies that
    // modify the layout. Copies that do not modify the layout are used in
    // CopyFusion.
    if (instr->opcode() == HloOpcode::kCopy) {
      return instr->shape() == instr->operand(0)->shape();
    }
    return true;
  }

  // `instr` is a bitcast or a bitcast-like operation.
  switch (instr->opcode()) {
    case HloOpcode::kBitcast:
      return true;
    case HloOpcode::kReshape:
      return ShapeUtil::ReshapeIsBitcast(instr->operand(0)->shape(),
                                         instr->shape());
    case HloOpcode::kTranspose:
      return ShapeUtil::TransposeIsBitcast(instr->operand(0)->shape(),
                                           instr->shape(), instr->dimensions());
    default:
      return false;
  }
}

static std::optional<HloInstructionAdaptor> FindNonTrivialHero(
    const HloInstructionAdaptor& root,
    const std::function<bool(const HloInstruction&)>& predicate) {
  std::optional<HloInstructionAdaptor> hero = std::nullopt;
  auto visitor = [&](HloInstructionAdaptor node) {
    if (predicate(node.instruction())) {
      if (hero) {  // Bail out if we found multiple potential heros.
        hero = std::nullopt;
        return TraversalResult::kInterrupt;
      }
      hero = node;
      return TraversalResult::kSkip;
    }

    if (!IsIntermediate(&node.instruction(), /*allowed_operand_count=*/3)) {
      return TraversalResult::kSkip;
    }
    return TraversalResult::kAdvance;
  };
  HloBfsConsumersFirstTraversal({root}, root.parent(), visitor);
  if (!hero) {
    return std::nullopt;
  }

  // Make sure that no non-elementwise op is reachable from the transpose.
  auto is_nontrivial = [](HloInstructionAdaptor node) {
    return node.instruction().opcode() != HloOpcode::kTuple &&
           node.instruction().opcode() != HloOpcode::kParameter &&
           !IsIntermediate(&node.instruction(),
                           /*allowed_operand_count=*/3);
  };
  bool visit_operands = false;
  if (HloBfsAnyOf(hero->GetUsers(), hero->parent(), is_nontrivial,
                  visit_operands)) {
    return std::nullopt;
  }

  return hero;
}

HloInstructionAdaptor FindNonTrivialHero(const HloInstructionAdaptor& instr) {
  HloInstructionAdaptor hero = instr;

  // Go up the chain of trivial element-wise(+bitcast, -copy) operations. Note
  // that no memoization is needed due to number of operands constraints: we
  // never have to revisit same nodes.
  while (IsIntermediate(&hero.instruction(), /*allowed_operand_count=*/1) &&
         hero.parent().ContainsInstruction(hero.GetOperand(0))) {
    hero = hero.GetOperand(0);
  }

  // Try a bit harder to find a transpose or concat hero. The shared memory
  // transpose and concat emitters also work if there are elementwise ops with
  // more than 1 operand on the path between root and the root op.
  auto is_transpose = [](const HloInstruction& node) {
    return GetDescriptionForTiledTransposeEmitter(node).has_value();
  };
  if (auto transpose = FindNonTrivialHero(hero, is_transpose)) {
    return *transpose;
  }
  auto is_concatenate = [](const HloInstruction& node) {
    return node.opcode() == HloOpcode::kConcatenate;
  };
  if (auto concatenate = FindNonTrivialHero(hero, is_concatenate)) {
    return *concatenate;
  }
  if (hero.opcode() != HloOpcode::kReduce) {
    return instr;
  }
  return hero;
}

const HloInstruction& FindNonTrivialHero(const HloInstruction& instr) {
  CHECK_NE(instr.opcode(), HloOpcode::kFusion);
  auto fusion_adaptor = HloFusionAdaptor::ForComputation(instr.parent());
  HloInstructionAdaptor instr_adaptor(instr, fusion_adaptor.get());
  return FindNonTrivialHero(instr_adaptor).instruction();
}

void VLogModule(int level, const llvm::Module& module) {
  XLA_VLOG_LINES(level, llvm_ir::DumpToString(&module));
}

void VerifyModule(const llvm::Module& module) {
  std::string error_str;
  llvm::raw_string_ostream error_stream(error_str);
  bool broken = llvm::verifyModule(module, &error_stream);
  CHECK(!broken) << error_str;
}

llvm::Type* GetIndexTypeForKernel(const HloInstruction* hlo,
                                  int64_t launch_size, llvm::IRBuilderBase* b) {
  // Find the unnested hlo instruction for which the kernel is generated for.
  const HloInstruction* unnested_hlo = hlo;
  const HloComputation* computation = hlo->parent();
  if (computation->IsFusionComputation()) {
    unnested_hlo = computation->FusionInstruction();
  }

  auto shape_in_range = [&](const Shape& s) {
    bool in_range = true;
    ShapeUtil::ForEachSubshape(s, [&](const Shape& sub_shape,
                                      const ShapeIndex& /*index*/) {
      if (sub_shape.IsArray() && !IsInt32(ShapeUtil::ElementsIn(sub_shape))) {
        in_range = false;
      }
    });

    return in_range;
  };

  llvm::Type* i64_ty = b->getInt64Ty();
  // Check launch dimension
  if (!IsInt32(launch_size)) {
    return i64_ty;
  }

  // Check the size of result tensors
  if (!shape_in_range(unnested_hlo->shape())) {
    return i64_ty;
  }

  auto hlo_shape_in_range = [&](const HloInstruction* operand) -> bool {
    return shape_in_range(operand->shape());
  };

  // Check the size of input tensors
  if (!absl::c_all_of(unnested_hlo->operands(), hlo_shape_in_range)) {
    return i64_ty;
  }

  // Check the size of the internal result tensors
  if (unnested_hlo->opcode() == HloOpcode::kFusion) {
    if (!absl::c_all_of(
            unnested_hlo->fused_instructions_computation()->instructions(),
            hlo_shape_in_range)) {
      return i64_ty;
    }
  }

  return b->getInt32Ty();
}

absl::StatusOr<DenseDataIntermediate> LiteralToXlaFormat(
    const Literal& literal) {
  PrimitiveType element_type = literal.shape().element_type();
  if (!primitive_util::IsArrayType(element_type)) {
    return Internal("Unsupported type in LiteralToXlaFormat");
  }

  int64_t byte_size = literal.size_bytes();
  if (primitive_util::IsSubByteNonPredType(element_type)) {
    auto bit_width = primitive_util::BitWidth(element_type);
    std::vector<uint8_t> output(CeilOfRatio<int64_t>(byte_size, 8 / bit_width));
    absl::Span<char> output_span =
        absl::MakeSpan(reinterpret_cast<char*>(output.data()), output.size());
    PackIntN(
        bit_width,
        absl::MakeSpan(reinterpret_cast<const char*>(literal.untyped_data()),
                       byte_size),
        output_span);
    return DenseDataIntermediate::Own(std::move(output));
  }

  return DenseDataIntermediate::Alias(absl::MakeSpan(
      reinterpret_cast<const uint8_t*>(literal.untyped_data()), byte_size));
}

absl::StatusOr<std::string> GetProtoFingerprint(
    const tsl::protobuf::MessageLite& proto) {
  std::string result;
  TF_RET_CHECK(tsl::SerializeToStringDeterministic(proto, &result));
  return absl::WebSafeBase64Escape(result);
}

std::optional<std::string> GetCustomFusionConfigName(
    const HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kFusion ||
      instr->fusion_kind() != HloInstruction::FusionKind::kCustom) {
    return std::nullopt;
  }
  absl::StatusOr<GpuBackendConfig> backend_config =
      instr->backend_config<GpuBackendConfig>();
  if (!backend_config.ok() || !backend_config->has_fusion_backend_config()) {
    return std::nullopt;
  }
  const FusionBackendConfig& fusion_backend_config =
      backend_config->fusion_backend_config();
  if (!fusion_backend_config.has_custom_fusion_config()) {
    return std::nullopt;
  }
  return fusion_backend_config.custom_fusion_config().name();
}

bool IsDynamicSliceFusion(const HloInstruction* instr) {
  std::optional<std::string> name = GetCustomFusionConfigName(instr);
  return name == kDynamicSliceFusionWithStaticAddressComputationConfigName ||
         name == kDynamicSliceFusionWithDynamicAddressComputationConfigName;
}

bool IsDynamicMemcpyFusion(const HloInstruction* instr) {
  absl::StatusOr<GpuBackendConfig> backend_config =
      instr->backend_config<GpuBackendConfig>();
  return backend_config.ok() &&
         backend_config->fusion_backend_config().kind() ==
             kDynamicMemcpyFusionKind;
}

std::optional<InductionVariableFunctionalDependency>
ResolveFunctionalDependencyOnInductionVariable(
    absl::Span<const HloInstruction* const> call_stack,
    const HloInstruction* parameter) {
  if (call_stack.empty()) {
    return std::nullopt;
  }

  VLOG(5) << "Looking for defining while loop of " << parameter->name();

  // Walk up the call stack, tracking the origin of `parameter`.
  const HloInstruction* argument = parameter;
  auto call_stack_it = call_stack.rbegin();
  auto call_stack_end = call_stack.rend();
  for (; call_stack_it != call_stack_end &&
         argument->opcode() == HloOpcode::kParameter &&
         ((*call_stack_it)->opcode() == HloOpcode::kFusion ||
          (*call_stack_it)->opcode() == HloOpcode::kAsyncStart ||
          (*call_stack_it)->opcode() == HloOpcode::kCall);
       ++call_stack_it) {
    argument = (*call_stack_it)->operand(argument->parameter_number());
  }

  if (call_stack_it == call_stack_end) {
    return std::nullopt;
  }

  VLOG(5) << "Arrived at " << argument->name() << " in "
          << (*call_stack_it)->name();

  // Find a unique parameter and a gte in the transitive dependencies of
  // `argument`.
  const HloInstruction* unique_param = nullptr;
  const HloInstruction* unique_gte = nullptr;
  absl::flat_hash_set<const HloInstruction*> seen{argument};
  std::queue<const HloInstruction*> queue;
  queue.push(argument);
  while (!queue.empty()) {
    const auto* instruction = queue.front();
    queue.pop();

    if (instruction->opcode() == HloOpcode::kCustomCall ||
        instruction->HasSideEffect()) {
      VLOG(5) << "Found an unsafe operation.";
      return std::nullopt;
    }

    if (instruction->opcode() == HloOpcode::kParameter) {
      if (unique_param || !instruction->shape().IsTuple()) {
        VLOG(5) << "Failed to match parameters.";
        return std::nullopt;
      }
      unique_param = instruction;
    }

    if (instruction->opcode() == HloOpcode::kGetTupleElement) {
      if (unique_gte) {
        VLOG(5) << "Found non-unique GTEs.";
        return std::nullopt;
      }
      unique_gte = instruction;
    }

    for (auto* operand : instruction->operands()) {
      if (seen.insert(operand).second) {
        queue.push(operand);
      }
    }
  }

  if (!unique_param || !unique_gte || unique_gte->operand(0) != unique_param) {
    VLOG(5) << "Did not find a parameter or GTE or they don't match.";
    return std::nullopt;
  }

  VLOG(5) << "Parameter and GTE: " << unique_param->name() << ", "
          << unique_gte->name();

  // Continue walking up through call instructions.
  while (call_stack_it != call_stack_end &&
         (*call_stack_it)->opcode() == HloOpcode::kCall &&
         unique_param->opcode() == HloOpcode::kParameter) {
    unique_param = (*call_stack_it)->operand(unique_param->parameter_number());
    ++call_stack_it;
  }

  // Find the while loop for 'unique_param'.
  auto while_instr_it = std::find_if(
      call_stack_it, call_stack_end, [&](const HloInstruction* instr) {
        return instr->opcode() == HloOpcode::kWhile &&
               unique_param == instr->while_body()->parameter_instruction(0);
      });

  if (while_instr_it == call_stack_end) {
    VLOG(5) << "Did not find a while loop.";
    return std::nullopt;
  }

  auto config =
      (*while_instr_it)->backend_config<xla::WhileLoopBackendConfig>();
  if (!config.ok() || !config->has_known_induction_variable() ||
      unique_gte->tuple_index() !=
          config->known_induction_variable().tuple_index()) {
    VLOG(5) << "Failed to verify that the offset depends on the induction "
               "variable.";
    return std::nullopt;
  }

  VLOG(5) << "While loop for " << parameter->name() << ": "
          << (*while_instr_it)->name();
  return InductionVariableFunctionalDependency{argument, *while_instr_it,
                                               unique_gte};
}

}  // namespace gpu
}  // namespace xla
