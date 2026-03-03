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
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
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
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/matmul_indexing_utils.h"
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
                        DotOperandDims::FromDotOperand(&dot, operand));
    // cuBLAS only supports single contracting dimension.
    if (dims.DimensionCount(DotOperandDims::kContracting) != 1) {
      return false;
    }
    // cuBLAS doesn't support minor batch dimension.
    if (absl::c_any_of(dims.DimensionIndices(DotOperandDims::kBatch),
                       [&](int64_t dim) {
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

bool IsCustomCallToTopK(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kTopKCustomCallTarget;
}

bool IsCustomCallToPtxKernel(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == "__gpu$xla.gpu.ptx";
}

bool IsCustomCallToMosaicGpu(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         (hlo.custom_call_target() == "mosaic_gpu" ||
          hlo.custom_call_target() == "mosaic_gpu_v2");
}

bool IsMosaicWithNvshmem(const HloInstruction& hlo) {
  return IsCustomCallToMosaicGpu(hlo) &&
         absl::StrContains(hlo.raw_backend_config_string(), "nvshmem");
}

bool IsMosaicWithMultimem(const HloInstruction& hlo) {
  return IsCustomCallToMosaicGpu(hlo) &&
         absl::StrContains(hlo.raw_backend_config_string(),
                           "xla_multimem_parameters");
}

bool IsCollectiveMosaicGpuInstruction(const HloInstruction& hlo) {
  return IsMosaicWithNvshmem(hlo) || IsMosaicWithMultimem(hlo);
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

int GetBitwidth(PrimitiveType type) {
  if (type == PRED) {
    return 8;
  }
  return primitive_util::BitWidth(type);
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

bool CanEmitPackedTranspose(const TransposeDescription& desc) {
  // TransposeDescription is normalized by construction.
  PackedTransposeDescription spec(desc);
  return GetPackedTransposeTileSizes(spec).ok();
}

std::optional<TransposeDescription> GetDescriptionForTiledTransposeEmitter(
    const HloInstruction& hero) {
  if (hero.opcode() != HloOpcode::kTranspose) {
    return std::nullopt;
  }

  absl::InlinedVector<int64_t, 3> normalized_permutation;
  auto normalized_dims_or = ShapeUtil::GetNormalizedLogicalTransposeShape(
      hero.operand(0)->shape(), hero.shape(), hero.dimensions(),
      normalized_permutation);
  if (!normalized_dims_or.ok()) {
    return std::nullopt;
  }
  auto normalized_dims = normalized_dims_or.value();
  auto normalized_operand_dims =
      Permute(normalized_dims, InversePermutation(normalized_permutation));
  // A real transpose needs at least 2 transpose dimensions.
  if (normalized_permutation.size() < 2) {
    return std::nullopt;
  }
  auto bit_width = GetBitwidth(hero.shape().element_type());
  int64_t operand_most_minor_dim = normalized_operand_dims.back();

  TransposeDescription desc{&hero, normalized_dims, normalized_permutation,
                            /*shmem_usage=*/0};
  if (CanEmitPackedTranspose(desc)) {
    int64_t vector_size =
        kBankBitwidth / GetBitwidth(hero.shape().element_type());
    desc.shmem_usage =
        kNumShmemBanks * (kBankBitwidth / 8) * kNumShmemBanks * vector_size;
    return desc;
  }
  // Minor dimension is preserved.
  if (normalized_permutation.back() == normalized_dims.size() - 1) {
    operand_most_minor_dim =
        normalized_operand_dims[normalized_dims.size() - 2];
    if (bit_width * normalized_dims.back() <= kMaxBitsInMostMinorDimension &&
        bit_width * normalized_dims.back() *
                std::min(operand_most_minor_dim,
                         normalized_dims[normalized_dims.size() - 2]) >=
            8 * kMinDimensionToTransposeTiled) {
      // Tile size for transposition.
      int64_t shmem_usage_bytes =
          CeilOfRatio(kNumShmemBanks * (kNumShmemBanks + 1LL) * bit_width *
                          normalized_dims.back(),
                      8LL);
      return TransposeDescription{&hero, normalized_dims,
                                  normalized_permutation, shmem_usage_bytes};
    }
  } else if ((operand_most_minor_dim >= kMinDimensionToTransposeTiled &&
              normalized_dims.back() >= kMinDimensionToTransposeTiled) ||
             (operand_most_minor_dim >= kMinDimensionToTransposeTiled2 &&
              normalized_dims.back() >= kMinDimensionToTransposeTiled2 &&
              operand_most_minor_dim * normalized_dims.back() >=
                  kMinTotalDimensionsToTransposeTiled)) {
    // TODO(b/415741994): TransposeEmitter is regressing for S4 when the last
    // dimension is being transposed. The issue seems to be related to bank
    // conflicts but a proper investigation is needed.
    if (bit_width == 4) {
      return std::nullopt;
    }
    int64_t shmem_usage_bytes =
        CeilOfRatio(kNumShmemBanks * (kNumShmemBanks + 1LL) * bit_width, 8LL);
    return TransposeDescription{&hero, normalized_dims, normalized_permutation,
                                shmem_usage_bytes};
  }
  return std::nullopt;
}

PackedTransposeDescription::PackedTransposeDescription(
    const TransposeDescription& description)
    : transpose(Cast<HloTransposeInstruction>(description.instr)) {
  permutation = llvm::to_vector<3>(description.permutation);
  inv_permutation = llvm::to_vector<3>(InversePermutation(permutation));
  canonical_output_shape = llvm::to_vector<3>(description.dimensions);
  canonical_permutation = llvm::to_vector<3>(description.permutation);

  // If the last dimension is transposed, add a size-1 B dimension.
  if (canonical_permutation.back() != canonical_output_shape.size() - 1) {
    canonical_permutation.push_back(canonical_output_shape.size());
    canonical_output_shape.push_back(1);
  }
  // Insert size-1 A dimension if necessary.
  auto rank = canonical_output_shape.size();
  // We know that the second to last dimension needs to be transposed, as
  // otherwise the TransposeDescription would not be normalized. Thus, the index
  // of the last transposed dimension is always rank - 2.
  if (canonical_permutation[rank - 3] != rank - 3) {
    canonical_output_shape.insert(canonical_output_shape.begin() + rank - 2, 1);
    for (auto& p : canonical_permutation) {
      if (p > rank - 3) p++;
    }
    canonical_permutation.insert(canonical_permutation.begin() + rank - 2,
                                 rank - 2);
  }
  canonical_inv_permutation =
      llvm::to_vector<3>(InversePermutation(canonical_permutation));
  canonical_input_shape = llvm::to_vector<3>(
      Permute(canonical_output_shape, canonical_inv_permutation));
}

std::string PackedTransposeDescription::ToString() const {
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
    const PackedTransposeDescription& spec) {
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
  if (auto transpose = FindHero(hero, std::move(is_transpose))) {
    return *transpose;
  }
  auto is_concatenate = [](const HloInstruction& node) {
    return node.opcode() == HloOpcode::kConcatenate;
  };
  if (auto concatenate = FindHero(hero, std::move(is_concatenate))) {
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

// Returns true if `variable` is marked as a dynamic variable.
bool IsDynamicVariable(const HloInstruction* variable,
                       const VerifiedLoop& loop) {
  auto config = loop.loop->backend_config<xla::WhileLoopBackendConfig>();
  if (!config.ok()) {
    return false;
  }

  int64_t tuple_idx = variable->tuple_index();
  for (int64_t dynamic_idx : config->dynamic_variable_tuple_indices()) {
    if (dynamic_idx == tuple_idx) {
      return true;
    }
  }
  return false;
}

// Attempts to find the induction variable of `loop` in `dependencies`. If there
// are any dependencies on non-induction variable loop-carried variables,
// returns nullopt.
std::optional<const HloInstruction*> VerifyInductionVariable(
    const Dependencies& dependencies, const VerifiedLoop& loop) {
  const HloInstruction* induction_var = nullptr;
  for (const HloInstruction* gte : dependencies.get_tuple_elements) {
    if (IsLoopCarriedVariable(gte, loop)) {
      if (IsInductionVariable(gte, loop)) {
        if (induction_var) {
          // This should never happen.
          VLOG(5) << "Found non-unique GTEs for the induction variable. Did "
                     "HloCSE run?";
          return std::nullopt;
        }
        induction_var = gte;
      } else if (IsDynamicVariable(gte, loop)) {
        // Dynamic variables are also acceptable because they represent tuple
        // indices used in DS/DUS that can be optimized by
        // FusionDynamicMemcpyRewriter.
        if (induction_var) {
          // This should never happen.
          VLOG(5) << "Found non-unique GTEs for the dynamic variable. Did "
                     "HloCSE run?";
          return std::nullopt;
        }
        induction_var = gte;
      } else {
        // Other dependencies on loop-carried variables are not allowed.
        VLOG(5) << "Found illegal dependency on loop-carried variable.";
        return std::nullopt;
      }
    }
    // Other GTEs are OK, as long as their tuples are ultimately just derived
    // from the loop's induction variable. We already verified that there are no
    // side-effecting dependencies in GetLeafDependencies.
  }
  if (!induction_var) {
    VLOG(5) << "Did not find an induction variable or dynamic variable.";
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
