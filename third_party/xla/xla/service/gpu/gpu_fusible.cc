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

#include "xla/service/gpu/gpu_fusible.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/permutation_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"

namespace xla {
namespace gpu {
namespace {

bool HasAnyTiledTransposeRoot(const HloComputation& computation) {
  return absl::c_any_of(GetFusionRoots(computation),
                        [&](const HloInstruction* instr) {
                          return GetDescriptionForTiledTransposeEmitter(
                                     FindNonTrivialHero(*instr))
                              .has_value();
                        });
}

const Shape& GetElementShape(const HloFusionAnalysis& analysis) {
  const Shape* shape = &analysis.fusion_root(0).shape();
  while (shape->IsTuple()) {
    shape = &shape->tuple_shapes(0);
  }
  return *shape;
}

// Computes the maximum valid unroll factor for a given instruction.
int ComputeMaxUnrollFactor(int64_t num_elements) {
  for (int i = MaxUnrollFactor(); i > 1; i /= 2) {
    if (num_elements % i == 0) {
      return i;
    }
  }
  return 1;
}

}  // namespace

bool IsPhysicallyTransposing(const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kFusion) {
    for (const HloInstruction* fused_instr : instr.fused_instructions()) {
      if (IsPhysicallyTransposing(*fused_instr)) {
        return true;
      }
    }
  }

  // A fusion iterates over its output in physically-contiguous order. This
  // applies "upwards" to operands.  Only an operator that changes an operand's
  // physical layout can create a "bad" memory access pattern.
  return instr.opcode() == HloOpcode::kCopy ||
         (instr.opcode() == HloOpcode::kTranspose &&
          !ShapeUtil::TransposeIsBitcast(instr.operand(0)->shape(),
                                         instr.shape(), instr.dimensions()));
}

namespace {
std::pair<int64_t, int64_t> MostMinorNonTrivialDimension(const Shape& shape) {
  int64_t position_of_first_non_trivial_dim = 0;
  for (int64_t dim : shape.layout().minor_to_major()) {
    if (shape.dimensions()[dim] > 1) {
      return {dim, position_of_first_non_trivial_dim};
    }
    ++position_of_first_non_trivial_dim;
  }
  return {-1, position_of_first_non_trivial_dim};
}
}  // namespace

bool TransposesMinorDimension(const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kFusion:
      return absl::c_any_of(instr->fused_instructions(),
                            TransposesMinorDimension);
    // TODO(akuegel): This can be simplified by just calling
    // GetDescriptionForTiledTransposeEmitter() once it returns a value for all
    // transposes that affect the most minor non-trivial dimension. Right now,
    // there are also cases with transposes that affect the most minor
    // non-trivial dimension which are not supported by the transpose emitter,
    // so GetDescriptionForTiledTransposeEmitter would return std::nullopt.
    case HloOpcode::kCopy: {
      int64_t first_non_trivial_operand_dim =
          MostMinorNonTrivialDimension(instr->operand(0)->shape()).first;
      int64_t first_non_trivial_output_dim =
          MostMinorNonTrivialDimension(instr->shape()).first;
      return first_non_trivial_operand_dim != first_non_trivial_output_dim;
    }
    case HloOpcode::kTranspose: {
      auto position_in_minor_to_major = InversePermutation(
          instr->operand(0)->shape().layout().minor_to_major());
      int64_t position_of_first_non_trivial_dim =
          MostMinorNonTrivialDimension(instr->operand(0)->shape()).second;
      for (int64_t output_dim : instr->shape().layout().minor_to_major()) {
        if (instr->shape().dimensions()[output_dim] == 1) {
          continue;
        }
        int64_t operand_dim = instr->dimensions().at(output_dim);
        // Check if there is any operand dimension with size > 1 that is more
        // minor than 'operand_dim'
        return position_in_minor_to_major[operand_dim] >
               position_of_first_non_trivial_dim;
      }
      return false;
    }
    default:
      return false;
  }
}

bool IsReduceInputFusion(const HloInstruction& instr,
                         const se::DeviceDescription& device_info) {
  return instr.opcode() == HloOpcode::kFusion &&
         absl::c_any_of(GetFusionRoots(*instr.called_computations()[0]),
                        [&](const HloInstruction* root) {
                          return IsRealReductionHero(
                              *root, FindNonTrivialHero(*root), device_info);
                        });
}

bool IsInputFusibleReduction(const HloInstruction& instr,
                             const se::DeviceDescription& device_info) {
  return IsReduceInputFusion(instr, device_info) ||
         IsReductionFromOrToContiguousDimensions(instr, device_info);
}

bool IsNestableVariadicReduction(const HloInstruction& instr,
                                 const se::DeviceDescription& device_info) {
  return instr.shape().IsTuple() &&
         ((instr.opcode() == HloOpcode::kReduce &&
           !IsReductionFromOrToContiguousDimensions(instr, device_info)) ||
          (instr.opcode() == HloOpcode::kFusion &&
           instr.fusion_kind() == HloInstruction::FusionKind::kLoop &&
           instr.fused_expression_root()->opcode() == HloOpcode::kReduce));
}

bool IsNestableVariadicReduceWindow(const HloInstruction& instr) {
  return instr.shape().IsTuple() &&
         (instr.opcode() == HloOpcode::kReduceWindow ||
          (instr.opcode() == HloOpcode::kFusion &&
           instr.fusion_kind() == HloInstruction::FusionKind::kLoop &&
           instr.fused_expression_root()->opcode() ==
               HloOpcode::kReduceWindow));
}

bool IsInputFusibleTranspose(const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kBitcast || instr.IsCustomFusion()) {
    return false;
  }
  if (instr.opcode() == HloOpcode::kFusion) {
    return HasAnyTiledTransposeRoot(*instr.fused_instructions_computation());
  }
  return GetDescriptionForTiledTransposeEmitter(instr).has_value();
}

const HloInstruction* GetRealHeroForMultiOutputFusion(
    const HloInstruction& instr, const se::DeviceDescription& device_info) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return &instr;
  }
  auto fused_expression_root = instr.fused_expression_root();
  if (!instr.IsMultiOutputFusion()) {
    const auto& hero = FindNonTrivialHero(*fused_expression_root);
    if (IsRealReductionHero(*fused_expression_root, hero, device_info) ||
        GetDescriptionForTiledTransposeEmitter(hero).has_value()) {
      return &hero;
    }
    return fused_expression_root;
  }
  // If possible, we want to pick a reduction-from-or-to-contiguous-dims
  // operand of the fusion root or a tiled transpose, because they have the most
  // constraints. Note that we cannot have both kinds at the same time, so once
  // we find any, we can immediately return it.
  for (auto* inst : fused_expression_root->mutable_operands()) {
    const auto& hero = FindNonTrivialHero(*inst);
    if (IsRealReductionHero(*inst, hero, device_info) ||
        GetDescriptionForTiledTransposeEmitter(hero).has_value()) {
      return &hero;
    }
  }
  return fused_expression_root->operands()[0];
}

FusionDecision FusionHeroesAreCompatible(
    const HloInstruction* hero1, const HloInstruction* hero2,
    const se::DeviceDescription& device_info) {
  auto hero1_is_unnested_reduce =
      IsReductionFromOrToContiguousDimensions(*hero1, device_info);
  auto tiled_transpose_hero1 = GetDescriptionForTiledTransposeEmitter(*hero1);
  bool hero1_is_unnested_transpose = tiled_transpose_hero1.has_value();
  bool hero2_is_unnested_reduce =
      IsReductionFromOrToContiguousDimensions(*hero2, device_info);
  auto tiled_transpose_hero2 = GetDescriptionForTiledTransposeEmitter(*hero2);
  bool hero2_is_unnested_transpose = tiled_transpose_hero2.has_value();

  if (hero1_is_unnested_reduce && hero2_is_unnested_reduce &&
      !AreReductionsMultiOutputFusionCompatible(hero2, hero1)) {
    return FusionDecision::Forbid("tiled reductions with different shapes");
  } else if (hero1_is_unnested_transpose && hero2_is_unnested_transpose &&
             // After normalization to rank 3, the transposes should have the
             // same shape and permute the same dimensions.
             !tiled_transpose_hero1->IsEquivalent(*tiled_transpose_hero2)) {
    return FusionDecision::Forbid("tiled transposes with different shapes");
  } else if ((hero1_is_unnested_transpose && hero2_is_unnested_reduce) ||
             (hero1_is_unnested_reduce && hero2_is_unnested_transpose)) {
    return FusionDecision::Forbid("MOF-fusion of a transpose and a reduction");
  }
  // If we are dealing with unnested transpose, make sure that we can still
  // treat them as unnested transpose after the sibling fusion.
  if (hero1_is_unnested_transpose || hero2_is_unnested_transpose) {
    auto check_path_of_intermediate_ops = [](HloInstruction* param) {
      // Check that there is a path from 'param' to the root consisting of only
      // Intermediate ops.
      if (param->user_count() != 1) {
        return false;
      }
      // Start at the user of the parameter.
      HloInstruction* hlo = param->users()[0];
      while (hlo->user_count() > 0) {
        if (!IsIntermediate(hlo)) {
          return false;
        }
        // IsIntermediate checks that the op has at most one user.
        hlo = hlo->users()[0];
      }
      return true;
    };
    HloInstruction* fusion1 = hero1->parent()->FusionInstruction();
    HloInstruction* fusion2 = hero2->parent()->FusionInstruction();
    if (fusion1 != nullptr && fusion2 != nullptr) {
      if (hero1_is_unnested_transpose && fusion2->IsUserOf(fusion1)) {
        int64_t operand_idx = fusion2->operand_index(fusion1);
        auto hlo = fusion2->fused_parameter(operand_idx);
        if (!check_path_of_intermediate_ops(hlo)) {
          return FusionDecision::Forbid("tiled transpose would become untiled");
        }
      } else if (hero2_is_unnested_transpose && fusion1->IsUserOf(fusion2)) {
        int64_t operand_idx = fusion1->operand_index(fusion2);
        auto hlo = fusion1->fused_parameter(operand_idx);
        if (!check_path_of_intermediate_ops(hlo)) {
          return FusionDecision::Forbid("tiled transpose would become untiled");
        }
      }
    }
  }
  return FusionDecision::Allow();
}

FusionDecision ShapesCompatibleForMultiOutputFusion(
    const HloInstruction& instr1, const HloInstruction& instr2,
    const se::DeviceDescription& device_info) {
  // Multi-output fusion kernels share a common parallel loop. The loop
  // dimensions are determined by instruction shapes.
  auto get_loop_shape = [&](const HloInstruction* element_instr) {
    // Special-case reduction-to-vector ops: The loop dimensions are determined
    // by the shape of the first operand.
    const auto& hero = element_instr->parent()->IsFusionComputation()
                           ? FindNonTrivialHero(*element_instr)
                           : *element_instr;
    if (IsReductionFromOrToContiguousDimensions(*element_instr, device_info) ||
        GetDescriptionForTiledTransposeEmitter(hero).has_value()) {
      return hero.operand(0)->shape();
    }
    return element_instr->shape();
  };

  // All shapes of the root tuple of multi-output fusions should agree, i.e. all
  // root ops should have equal output shapes. An exception are
  // reduction-to-vector ops. Here the input shapes of the reduction (first
  // operand shape) and the reduction dimensions need to match.
  const HloInstruction* hero1 =
      GetRealHeroForMultiOutputFusion(instr1, device_info);
  const HloInstruction* hero2 =
      GetRealHeroForMultiOutputFusion(instr2, device_info);

  if (auto compatible = FusionHeroesAreCompatible(hero1, hero2, device_info);
      !compatible) {
    return compatible;
  }

  const Shape& l1 = get_loop_shape(hero1);
  const Shape& l2 = get_loop_shape(hero2);

  // We accept different shapes provided shapes are trivially reshapable.
  bool accept_unequal_shape = !l1.IsTuple() && !l2.IsTuple();

  if (!ShapeUtil::EqualIgnoringElementType(l1, l2) &&
      (!accept_unequal_shape ||
       !ShapeUtil::IsReshapeOrTransposeBitcast(l1, l2,
                                               /*ignore_element_type=*/true))) {
    return FusionDecision::Forbid("different loop shapes");
  }
  return FusionDecision::Allow();
}

bool IsInputFusibleScatter(const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kScatter ||
      (instr.opcode() == HloOpcode::kFusion &&
       instr.fusion_kind() == HloInstruction::FusionKind::kInput &&
       instr.fused_expression_root()->opcode() == HloOpcode::kScatter)) {
    return true;
  }
  return false;
}

bool IsInputFusible(const HloInstruction& instr,
                    const se::DeviceDescription& device_info) {
  // Input fusion only handles non-elemental reduction and scatter operations.
  return instr.IsFusible() &&
         (IsInputFusibleReduction(instr, device_info) ||
          IsInputFusibleScatter(instr) || IsInputFusibleTranspose(instr));
}

// Returns true if `instr` can be fused as a producer or as a consumer into a
// kLoop fusion.
bool IsUniversallyLoopFusible(const HloInstruction& instr) {
  // NOTE: this check is done before the switch below, because a fusion instr
  // can also be elementwise, even if it's not a kLoop.
  if (instr.IsElementwise() && instr.operand_count() > 0 &&
      instr.opcode() != HloOpcode::kCopy) {
    return true;
  }

  switch (instr.opcode()) {
    case HloOpcode::kCopy:
      return !GetDescriptionForTiledTransposeEmitter(instr).has_value();

    case HloOpcode::kFusion:
      return instr.fusion_kind() == HloInstruction::FusionKind::kLoop;

    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kConcatenate:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kGather:
    case HloOpcode::kPad:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
      return true;
    default:
      return false;
  }
}

// Returns true if `instr` can be fused as a producer into a kLoop fusion.
bool IsLoopFusibleAsProducer(const HloInstruction& instr) {
  // Instr should be fusible.
  if (!instr.IsFusible()) return false;

  switch (instr.opcode()) {
    case HloOpcode::kIota:
    case HloOpcode::kConstant:
      return true;
    case HloOpcode::kReduce:
      // Non-variadic reductions can be fused as producers.
      return !instr.shape().IsTuple();
    default:
      return IsUniversallyLoopFusible(instr);
  }
}

static bool AllSatisfy(const HloInstruction& instr,
                       const HloPredicate& predicate) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return predicate(&instr);
  }

  return absl::c_all_of(
      instr.fused_instructions(), [&](const HloInstruction* i) {
        return i->opcode() == HloOpcode::kParameter || predicate(i);
      });
}

FusionDecision CanEmitInputFusedScatter(const HloInstruction& producer,
                                        const HloInstruction& consumer) {
  if (IsInputFusibleScatter(producer)) {
    return FusionDecision::Forbid("do not fuse into the output of scatter");
  }
  if (!IsInputFusibleScatter(consumer)) {
    return FusionDecision::Allow();
  }

  const HloInstruction* inplace_operand;
  if (consumer.opcode() == HloOpcode::kFusion) {
    const HloInstruction* scatter = consumer.fused_expression_root();
    CHECK_EQ(scatter->opcode(), HloOpcode::kScatter);
    CHECK_EQ(scatter->operand(0)->opcode(), HloOpcode::kParameter);
    inplace_operand = consumer.operand(scatter->operand(0)->parameter_number());
  } else {
    inplace_operand = consumer.operand(0);
  }
  if (inplace_operand == &producer) {
    return FusionDecision::Forbid(
        "do not fuse into the in-place operand of scatter");
  }
  if (absl::c_linear_search(producer.operands(), inplace_operand)) {
    return FusionDecision::Forbid(
        "Producer uses the in-place operand of a scatter");
  }
  return FusionDecision::Allow();
}

FusionDecision IsProducerMultiOutputFusible(
    const HloInstruction& producer, const se::DeviceDescription& device_info) {
  // Skip multiple output fusion. It's not yet supported.
  if (producer.IsMultiOutputFusion()) {
    return FusionDecision::Forbid("Producer is a multi-output fusion");
  }

  // Allowing multi-output fusions that contain in-place operations makes code
  // generation more difficult. For the generated loop to iterate over all
  // outputs in parallel, it must find an iteration order that guarantees that
  // no loop iteration writes an element of any in-place operand that is read
  // or written by any other iteration. For example:
  //
  //   %fused_computation {
  //     %param_0 = s32[4,4]{1,0} parameter(0)
  //     ...
  //     %updated = s32[4,4]{1,0} dynamic-update-slice(
  //         %param_0, %add, %constant_1, %constant_0)
  //     %transpose = s32[4,4]{0,1} transpose(%updated), dimensions={1,0}
  //     ROOT %tuple.5 = tuple(%transpose, %updated)
  //   }
  //
  // Iterating 'transpose' and 'updated' in parallel by array index is
  // not valid, because an iteration that produces some element of 'transpose'
  // will read from an element of 'param_0' that has been overwritten by some
  // other iteration (writing to 'updated').
  //
  // To avoid these problems, we simply ban fusion altogether when the producer
  // is in-place. (We can relax this restriction by establishing an explicit
  // contract that describes what multi-output fusion scenarios are supported by
  // codegen and then changing this check to allow exactly those fusions).
  if (!HloDataflowAnalysis::GetInPlaceInputOutputPairs(&producer).empty()) {
    return FusionDecision::Forbid("In-place operations are present");
  }

  if (!IsLoopFusibleAsProducer(producer)) {
    return FusionDecision::Forbid("producer is not loop-fusible");
  }

  if (IsPhysicallyTransposing(producer)) {
    return FusionDecision::Forbid("producer is physically transposing");
  }

  return FusionDecision::Allow();
}

// Returns an estimate of the shared memory usage for a given instruction in
// bytes.
static int64_t SharedMemoryUsageNoCache(
    const HloInstruction& instr, const se::DeviceDescription& device_info) {
  if (instr.opcode() == HloOpcode::kFusion) {
    int64_t sum = 0;
    for (const HloInstruction* hlo :
         instr.fused_instructions_computation()->instructions()) {
      sum += SharedMemoryUsageNoCache(*hlo, device_info);
    }
    return sum;
  } else if (instr.opcode() == HloOpcode::kReduce &&
             IsReductionFromOrToContiguousDimensions(instr, device_info)) {
    ReductionDimensions reduction_info =
        GetReductionKindAndContiguousComponents(instr);
    int64_t primitive_size_sum = 0;
    // Variadic reductions will allocate one shared memory buffer for each
    // input. They all have the same shape, so we can just sum up the primitive
    // sizes of the inputs.
    for (int i = 0; i < instr.operand_count() / 2; ++i) {
      primitive_size_sum += ShapeUtil::ByteSizeOfPrimitiveType(
          instr.operand(i)->shape().element_type());
    }

    if (reduction_info.is_row_reduction) {
      // In row reductions, we write at most one element per warp to shared
      // memory, regardless of whether the reduction is vectorized or not. We
      // have at most 32 warps for a single row. We could tighten this estimate,
      // but it doesn't really matter. Row reductions are very unlikely to ever
      // run out of shared memory budget.
      return 32 * primitive_size_sum;
    } else {
      // The shape of the cache for column reductions is 32x(vector_size * 32 +
      // 1). We don't know the actual vector size here, so we assume the
      // maximum.
      constexpr int kMaxVectorSize = 4;
      return 32 * (kMaxVectorSize * 32 + 1) * primitive_size_sum;
    }
  } else if (auto tr = GetDescriptionForTiledTransposeEmitter(instr)) {
    // Tile size for transposition.
    int64_t primitive_size =
        ShapeUtil::ByteSizeOfPrimitiveType(instr.shape().element_type());
    int64_t bytes_required = 32 * 33 * primitive_size;
    // If the last dimension is not changed, it becomes part of the tile.
    if (tr->permutation.back() == tr->permutation.size() - 1) {
      bytes_required *= tr->dimensions.back();
    }
    return bytes_required;
  }
  // Other fused expressions for now don't need the shared memory budget.
  return 0;
}

int64_t FusionInfoCache::GetSharedMemoryUsage(const HloInstruction& instr) {
  {
    absl::MutexLock lock(&mutex_);
    auto it = shared_memory_usage_.find(&instr);
    if (it != shared_memory_usage_.end()) {
      return it->second;
    }
  }

  // nb: Users are only expected to call cache.Invalidate() on top-level
  // instructions, not instructions inside fusion nodes.  Therefore we can only
  // cache top-level instructions; it would not be valid to pass the cache to
  // SharedMemoryUsageNoCache and use the cache *within* the fusion.
  int64_t shared_memory_usage = SharedMemoryUsageNoCache(instr, device_info_);

  absl::MutexLock lock(&mutex_);
  shared_memory_usage_.emplace(&instr, shared_memory_usage);
  return shared_memory_usage;
}

int64_t SharedMemoryUsage(const HloInstruction& instr, FusionInfoCache* cache,
                          const se::DeviceDescription& device_info) {
  if (!cache) {
    return SharedMemoryUsageNoCache(instr, device_info);
  }
  return cache->GetSharedMemoryUsage(instr);
}

// Codegen'ing unnested reductions requires a lot of registers, so a MOF
// combining many of those runs a high risk of spilling.
constexpr int64_t kMaxUnnestedReductionOutputsPerFusion = 8;

// Returns the number of unnested reductions in the instruction output.
static int64_t NumUnnestedReductionsNoCache(
    const HloInstruction& instr, const se::DeviceDescription& device_info) {
  if (instr.opcode() == HloOpcode::kReduce &&
      IsReductionFromOrToContiguousDimensions(instr, device_info)) {
    return 1;
  }
  if (instr.opcode() == HloOpcode::kFusion) {
    int64_t sum = 0;
    for (const HloInstruction* hlo :
         instr.fused_instructions_computation()->instructions()) {
      sum += NumUnnestedReductionsNoCache(*hlo, device_info);
    }
    return sum;
  }
  return 0;
}

int64_t FusionInfoCache::GetNumUnnestedReductions(const HloInstruction& instr) {
  {
    absl::MutexLock lock(&mutex_);
    auto it = num_unnested_reductions_.find(&instr);
    if (it != num_unnested_reductions_.end()) {
      return it->second;
    }
  }

  // nb: Users are only expected to call cache.Invalidate() on top-level
  // instructions, not instructions inside fusion nodes.  Therefore we can only
  // cache top-level instructions; it would not be valid to pass the cache to
  // NumUnnestedReductionsNoCache and use the cache *within* the fusion.
  int64_t num_unnested_reductions =
      NumUnnestedReductionsNoCache(instr, device_info_);

  absl::MutexLock lock(&mutex_);
  num_unnested_reductions_.emplace(&instr, num_unnested_reductions);
  return num_unnested_reductions;
}

static int64_t NumUnnestedReductions(const HloInstruction& instr,
                                     FusionInfoCache* cache,
                                     const se::DeviceDescription& device_info) {
  if (!cache) {
    return NumUnnestedReductionsNoCache(instr, device_info);
  }

  return cache->GetNumUnnestedReductions(instr);
}

// This function limits the maximum number of operands to a fusion, and the
// amount of shared memory which can be consumed by the fusion.
//
// There's a cap on how many parameters we can pass to a CUDA kernel, but
// exactly what that limit is hazy, as it depends on (among other things) how
// much GPU constant memory is in use for other purposes.
//
// Moreover, we don't even know at the point that we're running fusion how many
// arguments the CUDA kernel for a fusion node will have: It depends on buffer
// assignment, where we will decide which of the fusion's operands live in XLA's
// big temp buffer versus in other allocations.
//
// As a heuristic, we simply cap the number of fusion operands plus outputs at
// MaxOperandsAndOutputsPerFusion().  This puts an upper bound on the number of
// parameters to the kernel, working around the correctness problem.
//
// This limit is also often good for performance.  In a fusion with many
// operands, each GPU thread likely has to do a lot of work, and so possibly
// uses a lot of registers, thus limiting occupancy.
//
// If the fusion is a producer/consumer fusion and instr1 is the
// consumer and instr2 is the producer, set is_consumer_producer_fusion
// to true to enable more fusion.
FusionDecision FusionFitsInBudget(const HloInstruction& instr1,
                                  const HloInstruction& instr2,
                                  const se::DeviceDescription& device_info,
                                  bool is_consumer_producer_fusion,
                                  FusionInfoCache* cache /*=nullptr*/) {
  if (SharedMemoryUsage(instr1, cache, device_info) +
          SharedMemoryUsage(instr2, cache, device_info) >
      device_info.shared_memory_per_block()) {
    return FusionDecision::Forbid(
               "shared memory usage would be over the budget of ")
           << device_info.shared_memory_per_block() << "B";
  }

  if (NumUnnestedReductions(instr1, cache, device_info) +
          NumUnnestedReductions(instr2, cache, device_info) >
      kMaxUnnestedReductionOutputsPerFusion) {
    return FusionDecision::Forbid("over ")
           << kMaxUnnestedReductionOutputsPerFusion
           << " unnested reductions in fusion";
  }

  // Compute the number of outputs of the (possibly multi-output) fusion node
  // we're considering creating.
  //
  // This isn't precise; we may be off by one if
  //  - We're creating a multi-output fusion out of two non-MOFs.  Creating a
  //    MOF adds a new buffer, namely, the tuple buffer.
  //  - We're merging two MOFs.  In this case, we should count the tuple buffer
  //    only once.
  //  - WLOG there's an edge from `a` to `b` and `b` is the only consumer of
  //    `a`.  In this case the result of `a` is not part of the output of the
  //    fusion.
  //
  // But because this is a heuristic and our limit
  // MaxOperandsAndOutputsPerFusion() is a large value (so +/- 1 doesn't make a
  // big difference), we ignore this small inaccuracy in favor of simplicity.
  int64_t num_output_buffers = ShapeUtil::SubshapeCount(instr1.shape()) +
                               ShapeUtil::SubshapeCount(instr2.shape());

  // The new fusion will have no more operands and outputs than
  //   producer_operands + consumer_operands - 1 + num_output_buffers
  // (minus one because we may be fusing a producer->consumer edge between `a`
  // and `b`).
  //
  // This fact may be enough to let us avoid having to compute the true total
  // number of operands, which can be expensive.
  if (instr1.operand_count() + instr2.operand_count() - 1 +
          num_output_buffers <=
      MaxOperandsAndOutputsPerFusion()) {
    return FusionDecision::Allow();
  } else {
    VLOG(5) << "Operand count of "
            << "(" << instr1.ToString() << " ) = " << instr1.operand_count()
            << " and ( " << instr2.ToString()
            << " ) = " << instr2.operand_count()
            << " and num_output_buffers = " << num_output_buffers
            << " is bigger than the bound of "
            << MaxOperandsAndOutputsPerFusion();
  }

  // Compute the precise number of operands to the new fusion.
  absl::flat_hash_set<const HloInstruction*> operands(instr1.operands().begin(),
                                                      instr1.operands().end());
  operands.insert(instr2.operands().begin(), instr2.operands().end());
  // If there's an edge between `a` and `b`, don't count it: We're fusing that
  // producer -> consumer relationship.
  operands.erase(&instr1);
  operands.erase(&instr2);

  // If we generate the same numbers of inputs and outputs as
  // before, it won't be bigger after fusion. So accept the fusion.
  // As this is a consumer_producer fusion, this does not change the
  // consumer numbers of output. So no need to check it.
  if (is_consumer_producer_fusion &&
      operands.size() <= instr1.operands().size()) {
    return FusionDecision::Allow();
  }

  // Does the new fusion have more operands and outputs than the max?
  if (operands.size() + num_output_buffers > MaxOperandsAndOutputsPerFusion()) {
    return FusionDecision::Forbid(
        "Number of operands and output buffers is larger than allowed budget "
        "per fusion");
  }
  return FusionDecision::Allow();
}

bool IsFusibleAsMultiOutputFusionRoot(
    const HloInstruction& instr, const se::DeviceDescription& device_info) {
  // We can fuse reduces and loop fusions. Elementwise instructions can be fused
  // with any other instruction.
  // Note that scatter cannot be the root of a multi-output fusion because
  // its emitter doesn't support it.
  //
  // Custom fusions cannot be fused with anything.

  return instr.IsFusible() && !instr.IsCustomFusion() &&
         (IsInputFusibleReduction(instr, device_info) ||
          IsInputFusibleTranspose(instr) ||
          instr.IsLoopFusion() ||  // TODO(b/130013493): Use IsLoopFusible here.
          instr.IsElementwise());
}

HloInstruction::FusionKind ChooseFusionKind(
    const HloInstruction& producer, const HloInstruction& consumer,
    const se::DeviceDescription& device_info) {
  return (IsInputFusible(consumer, device_info) ||
          IsInputFusible(producer, device_info))
             ? HloInstruction::FusionKind::kInput
             : HloInstruction::FusionKind::kLoop;
}

bool IsConsumerTheOnlyNonRootUser(const HloInstruction& instr,
                                  const HloInstruction& consumer) {
  return absl::c_all_of(instr.users(), [&](const HloInstruction* user) {
    if (user->opcode() == HloOpcode::kGetTupleElement ||
        user->opcode() == HloOpcode::kBitcast) {
      // Skip no-op instructions.
      return IsConsumerTheOnlyNonRootUser(*user, consumer);
    }
    // `user` is `consumer` or consumed by ROOT.
    return user == &consumer || user == user->parent()->root_instruction();
  });
}

int64_t GetInstrCountOfFusible(const HloInstruction& instr) {
  return instr.opcode() == HloOpcode::kFusion ? instr.fused_instruction_count()
                                              : 1;
}

absl::InlinedVector<const HloInstruction*, 2> GetOutputsOfFusible(
    const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return {&instr};
  }

  HloInstruction* root = instr.fused_expression_root();
  if (root->opcode() != HloOpcode::kTuple) {
    return {root};
  } else {
    auto v = root->operands();
    return absl::InlinedVector<const HloInstruction*, 2>(v.begin(), v.end());
  }
}

size_t GetOutputSizeOfFusible(const HloInstruction& instr) {
  if (!instr.IsMultiOutputFusion()) {
    return 1;
  }
  const HloInstruction* root = instr.fused_expression_root();
  return ShapeUtil::TupleElementCount(root->shape());
}

// Recursive helper for GetFusionRoots below.
static void GetFusionRootsRec(const HloInstruction* root,
                              std::vector<const HloInstruction*>& out) {
  if (root->opcode() == HloOpcode::kGetTupleElement &&
      root->operand(0)->opcode() == HloOpcode::kTuple) {
    return GetFusionRootsRec(root->operand(0)->operand(root->tuple_index()),
                             out);
  } else if (root->opcode() == HloOpcode::kGetTupleElement) {
    out.push_back(root->operand(0));
  } else if (root->opcode() == HloOpcode::kTuple) {
    for (int i = 0; i < root->operand_count(); i++) {
      GetFusionRootsRec(root->operand(i), out);
    }
  } else {
    out.push_back(root);
  }
}

std::vector<const HloInstruction*> GetFusionRoots(
    const HloComputation& computation) {
  std::vector<const HloInstruction*> out;
  GetFusionRootsRec(computation.root_instruction(), out);
  return out;
}

bool IsGenericTritonFusion(const HloInstruction& instr) {
  // Note that we don't accept kTritonNestedGemmFusionKind here as they should
  // not be fused with anything else.
  return instr.opcode() == HloOpcode::kFusion &&
         instr.fusion_kind() == HloInstruction::FusionKind::kCustom &&
         instr.backend_config<GpuBackendConfig>().ok() &&
         instr.backend_config<GpuBackendConfig>()
                 ->fusion_backend_config()
                 .kind() == kTritonFusionKind;
}

bool MayPreventVectorization(const HloFusionAdaptor& fusion) {
  // An empirically chosen constant: unrolling concat with a large amount of
  // arguments causes excessive register spilling.
  static constexpr int kMaxConcatArgumentsForUnrolling = 10;
  return HloAnyOf(fusion, [&](auto node) {
    switch (node.opcode()) {
      case HloOpcode::kReduceWindow:
      case HloOpcode::kSort:
      case HloOpcode::kDot:
        return true;
      case HloOpcode::kConcatenate:
        return node.instruction().operand_count() >
               kMaxConcatArgumentsForUnrolling;
      case HloOpcode::kReduce: {
        const Shape& shape = node.instruction().shape();
        return shape.IsTuple() && shape.tuple_shapes_size() > 1;
      }
      default:
        return false;
    }
  });
}

bool IsStreamAnnotatedComputation(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kCall &&
         instr->frontend_attributes().map().contains(kXlaStreamAnnotationAttr);
}

std::vector<HloComputation*> GetFusibleComputations(
    const HloModule& module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto result = module.MakeComputationPostOrder(execution_threads);
  absl::flat_hash_set<const HloComputation*> computations_not_to_fuse;
  for (const auto* computation : result) {
    for (const auto* instr : computation->instructions()) {
      // Don't fuse within called computations, unless they are for control
      // flow. See also fusion_wrapper.cc, which does the same.
      if (HloInstruction::MightHaveCalledComputations(instr->opcode()) &&
          !IsStreamAnnotatedComputation(instr) &&
          instr->opcode() != HloOpcode::kWhile &&
          instr->opcode() != HloOpcode::kConditional &&
          // No need to add fusion computations, just check the flag.
          instr->opcode() != HloOpcode::kFusion) {
        for (auto* called : instr->called_computations()) {
          computations_not_to_fuse.insert(called);
        }
      }
    }
  }
  result.erase(
      std::remove_if(result.begin(), result.end(),
                     [&](HloComputation* computation) {
                       return computation->IsFusionComputation() ||
                              computations_not_to_fuse.contains(computation);
                     }),
      result.end());
  return result;
}

LaunchDimensionsConfig ComputeLoopFusionConfig(
    const HloFusionAnalysis& analysis) {
  return ComputeLoopFusionConfig(analysis, GetElementShape(analysis));
}

LaunchDimensionsConfig ComputeLoopFusionConfig(
    const HloFusionAnalysis& analysis, const Shape& element_shape) {
  int unroll_factor = 1;
  // Unrolling is good to read large inputs with small elements
  // due to vector loads, but increases the register pressure when one
  // thread has to produce multiple output elements.
  // Therefore for fusions with small outputs prefer to use one thread
  // per output element = no unroll.
  // Call 'small' fusions that use less threads than the GPU has.
  int64_t num_elements = ShapeUtil::ElementsIn(element_shape);
  int64_t n_threads_max = analysis.device_info().threads_per_core_limit() *
                          analysis.device_info().core_count();
  if (num_elements >= n_threads_max &&
      !MayPreventVectorization(analysis.fusion())) {
    unroll_factor = ComputeMaxUnrollFactor(num_elements);
  }
  // CHECK that unroll_factor is a power-of-2, as needed by the logic below.
  CHECK(absl::has_single_bit(static_cast<uint64_t>(unroll_factor)));
  // Ensure a single thread writes to a byte containing multiple values by
  // setting unroll_factor to an appropriate number. Setting unroll_factor is
  // safe even if the new unroll_factor doesn't divide the number of elements,
  // as the parallel loop emitter will insert a bounds check in this case to
  // ensure the out-of-bounds element is not computed and written. Setting
  // unroll_factor is safe even if MayPreventVectorization returns false, as
  // the MayPreventVectorization check is an optimization, not a correctness
  // requirement.
  unroll_factor = std::max(
      unroll_factor,
      CeilOfRatio(8, analysis.input_output_info().smallest_output_dtype_bits));
  CHECK(absl::has_single_bit(static_cast<uint64_t>(unroll_factor)));
  VLOG(2) << "Unroll factor: " << unroll_factor;

  LaunchDimensionsConfig launch_config{unroll_factor};
  return launch_config;
}

}  // namespace gpu
}  // namespace xla
