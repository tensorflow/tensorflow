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
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/ir_emission_utils.h"
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
#include "xla/service/gpu/matmul_indexing_utils.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace gpu {

absl::StatusOr<bool> IsCublasSupportedMatMul(
    const HloInstruction& dot, bool allow_matrix_vector_multiplication) {
  if (dot.opcode() != HloOpcode::kDot) {
    return false;
  }

  // Number of operands that have non-trivial non-contracting dimension.
  int num_matrix_operands = 0;
  for (int operand : {0, 1}) {
    TF_ASSIGN_OR_RETURN(DotOperandDims dims,
                        DotOperandDims::FromDot(&dot, operand));
    // cuBLAS only supports single contracting dimension.
    if (dims.DimensionCount(DotOperandDims::kContracting) != 1) {
      return false;
    }
    // cuBLAS doesn't support minor batch dimension.
    if (absl::c_any_of(dims.Indices(DotOperandDims::kBatch), [&](int64_t dim) {
          return dim == dims.shape().dimensions().size() - 1;
        })) {
      return false;
    }
    // cuBLAS supports up to one non-contracting dimension.
    const auto& nc_dims = dims.DimensionSizes(DotOperandDims::kNonContracting);
    if (nc_dims.size() > 1) {
      return false;
    }
    if (nc_dims.size() == 1) {
      num_matrix_operands += (nc_dims[0] != 1);
    }
  }

  if (num_matrix_operands == 0 ||
      (num_matrix_operands == 1 && !allow_matrix_vector_multiplication)) {
    return false;
  }

  switch (dot.shape().element_type()) {
    // Types allowed for both matmul and matmul-vector.
    case F8E4M3FN:
    case F8E5M2:
    case F16:
    case BF16:
    case F32:
    case F64:
    case C64:
      return true;
    case S32:
      return (dot.operand(0)->shape().element_type() == S8 &&
              dot.operand(1)->shape().element_type() == S8);
    // Only allowed for matmul.
    case F8E3M4:
    case F8E4M3:
    case F8E4M3FNUZ:
    case F8E5M2FNUZ:
    case C128:
      return num_matrix_operands == 2;
    default:
      return false;
  }
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
  auto bit_width = GetBitwidth(hero.shape().element_type());
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
  int64_t num_elements_after_transposed_dims = 1;
  std::pair<int64_t, int64_t> transposed_dims;
  if (permutation.back() == dimensions.size() - 1) {
    if (bit_width * dimensions.back() > kMaxBitsInMostMinorDimension) {
      return std::nullopt;
    }
    num_elements_after_transposed_dims = dimensions.back();
    transposed_dims = {
        hero.operand(0)->shape().dimensions(dimensions.size() - 2),
        dimensions[dimensions.size() - 2]};
  } else {
    // TODO(b/415741994): TransposeEmitter is regressing for S4 when the last
    // dimension is being transposed. The issue seems to be related to bank
    // conflicts but a proper investigation is needed.
    if (bit_width == 4) {
      return std::nullopt;
    }
    transposed_dims = {operand_most_minor_dim, dimensions.back()};
    // TODO(b/415741994): TransposeEmitter is slow when we are transposing the
    // last two dimensions and one of the transposed is small because ends up
    // using a very small amount of threads per warp.
    if (std::min(transposed_dims.first, transposed_dims.second) <
            kMinDimensionToLastTwoDimensionsTransposeTiled &&
        permutation.back() == permutation.size() - 2 &&
        permutation[permutation.size() - 2] == permutation.size() - 1) {
      return std::nullopt;
    }
  }
  if ((std::min(transposed_dims.first, transposed_dims.second) >=
       kMinDimensionToTransposeTiled) &&
      (transposed_dims.first * transposed_dims.second >=
       kMinTotalDimensionsToTransposeTiled)) {
    int64_t shmem_usage_bytes =
        CeilOfRatio(kNumShmemBanks * (kNumShmemBanks + 1LL) * bit_width *
                        num_elements_after_transposed_dims,
                    8LL);
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

namespace {

// Whether the instruction is semantically a call.
bool IsCallLike(const HloInstruction* caller) {
  return caller->opcode() == HloOpcode::kFusion ||
         caller->opcode() == HloOpcode::kAsyncStart ||
         caller->opcode() == HloOpcode::kCall;
}

const HloInstruction* GetUniqueCallerOrNull(const HloComputation* callee) {
  auto callers = callee->caller_instructions();
  return callers.size() == 1 ? callers.front() : nullptr;
}

struct Dependencies {
  absl::InlinedVector<const HloInstruction*, 2> parameters;
  absl::InlinedVector<const HloInstruction*, 1> get_tuple_elements;
};

// Returns the leaf dependencies of `root`, in each frame of the call stack.
// Here, leaves are parameters and GTEs. Returns nullopt if any dependencies
// have side effects.
std::optional<Dependencies> GetLeafDependencies(const HloInstruction* root) {
  absl::flat_hash_set<const HloInstruction*> seen{root};
  std::queue<const HloInstruction*> queue;
  queue.push(root);

  auto enqueue = [&](const HloInstruction* instr) {
    if (seen.insert(instr).second) {
      queue.push(instr);
    }
  };

  Dependencies results;
  while (!queue.empty()) {
    const auto* instruction = queue.front();
    VLOG(5) << "Visiting " << instruction->name() << ".";
    queue.pop();

    if (instruction->opcode() == HloOpcode::kCustomCall ||
        instruction->HasSideEffect()) {
      VLOG(5) << "Found an unsafe operation.";
      return std::nullopt;
    }

    if (instruction->opcode() == HloOpcode::kParameter) {
      results.parameters.push_back(instruction);
      const HloInstruction* caller =
          GetUniqueCallerOrNull(instruction->parent());
      if (!caller) {
        VLOG(5) << "Failed to determine unique caller, aborting traversal.";
        return std::nullopt;
      }

      // If this is semantically a call, continue the traversal at the call
      // site.
      if (IsCallLike(caller)) {
        int64_t index = instruction->parameter_number();
        enqueue(caller->operand(index));
      }
    }

    if (instruction->opcode() == HloOpcode::kGetTupleElement) {
      results.get_tuple_elements.push_back(instruction);
    }

    for (auto* operand : instruction->operands()) {
      enqueue(operand);
    }
  }
  return results;
}

struct VerifiedLoop {
  const HloInstruction* loop;
  const HloInstruction* parameter;
  int64_t induction_variable_index;
};

// Checks that `loop` is a while loop from which we can derive functional
// dependencies.
std::optional<VerifiedLoop> VerifyFunctionalDependencyLoop(
    const HloInstruction* loop) {
  if (!loop) {
    VLOG(5) << "No loop found";
    return std::nullopt;
  }
  auto config = loop->backend_config<xla::WhileLoopBackendConfig>();
  if (!config.ok() || !config->has_known_induction_variable()) {
    VLOG(5) << "The loop has no known induction variable.";
    return std::nullopt;
  }
  return VerifiedLoop{loop, loop->while_body()->parameter_instruction(0),
                      config->known_induction_variable().tuple_index()};
}

// Returns true if `hlo` is a GTE for a loop carried variable of `loop`.
bool IsLoopCarriedVariable(const HloInstruction* hlo,
                           const VerifiedLoop& loop) {
  return hlo->opcode() == HloOpcode::kGetTupleElement &&
         hlo->operand(0) == loop.parameter;
}

// Returns true if `maybe_variable` is `loop`'s induction variable.
bool IsInductionVariable(const HloInstruction* maybe_variable,
                         const VerifiedLoop& loop) {
  return IsLoopCarriedVariable(maybe_variable, loop) &&
         maybe_variable->tuple_index() == loop.induction_variable_index;
}

// Attempts to find the induction variable of `loop` in `dependencies`. If there
// are any dependencies on non-induction variable loop-carried variables,
// returns nullopt.
std::optional<const HloInstruction*> VerifyInductionVariable(
    const Dependencies& dependencies, const VerifiedLoop& loop) {
  const HloInstruction* induction_var = nullptr;
  for (const HloInstruction* gte : dependencies.get_tuple_elements) {
    if (IsInductionVariable(gte, loop)) {
      if (induction_var) {
        // This should never happen.
        VLOG(5) << "Found non-unique GTEs for the induction variable. Did "
                   "HloCSE run?";
        return std::nullopt;
      }
      induction_var = gte;
    } else if (IsLoopCarriedVariable(gte, loop)) {
      // Other dependencies on loop-carried variables are not allowed.
      VLOG(5) << "Found illegal dependency on loop-carried variable.";
      return std::nullopt;
    }
    // Other GTEs are OK, as long as their tuples are ultimately just derived
    // from the loop's induction variable. We already verified that there are no
    // side-effecting dependencies in GetLeafDependencies.
  }
  if (!induction_var) {
    VLOG(5) << "Did not find an induction variable.";
    return std::nullopt;
  }
  return induction_var;
}

}  // namespace

std::optional<InductionVariableFunctionalDependency>
ResolveFunctionalDependencyOnInductionVariable(const HloInstruction* instr) {
  VLOG(5) << "Looking for defining while loop of " << instr->name();

  auto dependencies = GetLeafDependencies(instr);
  // If there is a side effect in the dependencies, the result will be nullopt.
  if (!dependencies) {
    return std::nullopt;
  }

  // In the dependencies, there should be exactly one parameter of a while loop,
  // and exactly one GTE for that parameter. We already verified that there are
  // no side-effecting dependencies.
  InductionVariableFunctionalDependency result{};
  for (const HloInstruction* param : dependencies->parameters) {
    const HloComputation* callee = param->parent();
    const HloInstruction* caller = GetUniqueCallerOrNull(callee);
    if (caller && IsCallLike(caller)) {
      // Register the parameter as a required intermediate value.
      auto& required = result.required_parameters[callee];
      if (required.empty()) {
        required.resize(callee->num_parameters());
      }
      required[param->parameter_number()] = true;
    } else if (caller && caller->opcode() == HloOpcode::kWhile) {
      if (result.loop) {
        LOG(WARNING) << "While loop not unique. This should never happen.";
        return std::nullopt;
      }
      result.loop = caller;
    } else {
      // We arrived at an unexpected parameter. This likely means we're not in
      // a while loop, or there's an unsupported instruction between the while
      // loop and `instr`.
      VLOG(5) << "Unsupported parameter: " << param->name() << ".";
      return std::nullopt;
    }
  }

  auto verified_loop = VerifyFunctionalDependencyLoop(result.loop);
  if (!verified_loop) {
    return std::nullopt;
  }

  auto induction_var = VerifyInductionVariable(*dependencies, *verified_loop);
  if (induction_var) {
    result.induction_var = *induction_var;
  } else {
    return std::nullopt;
  }

  VLOG(5) << "While loop for " << instr->name() << ": " << result.loop->name();
  return result;
}

DenseDataIntermediateProto DenseDataIntermediate::ToProto() const {
  DenseDataIntermediateProto proto;
  absl::Span<const uint8_t> data = span();
  proto.mutable_data()->assign(data.begin(), data.end());
  return proto;
}
DenseDataIntermediate DenseDataIntermediate::FromProto(
    const DenseDataIntermediateProto& proto) {
  const std::string& data = proto.data();
  return DenseDataIntermediate::Own(
      std::vector<uint8_t>(data.begin(), data.end()));
}
}  // namespace gpu
}  // namespace xla
