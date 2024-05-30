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
#include <stack>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"

namespace xla {
namespace gpu {
namespace {

bool HasAnyTiledTransposeRoot(const HloComputation& computation) {
  return absl::c_any_of(GetFusionRoots(computation),
                        [&](const HloInstruction* instr) {
                          return GetDescriptionForTiledTransposeEmitter(
                                     *instr, FindNonTrivialHero(*instr))
                              .has_value();
                        });
}

}  // namespace

bool IfFusedReadsElementsMultipleTimes(const HloInstruction& instr) {
  CHECK_NE(instr.opcode(), HloOpcode::kFusion) << "`instr` has to be unfused.";
  // Avoid fusing gather or broadcast if output is larger than the input
  // which means that inputs are used multiple times.
  if (instr.opcode() == HloOpcode::kGather ||
      instr.opcode() == HloOpcode::kBroadcast) {
    return ShapeUtil::ElementsIn(instr.shape()) >
           ShapeUtil::ElementsIn(instr.operand(0)->shape());
  }
  // Avoid fusing reduce-window when stride is less than window size to minimize
  // the number of reads of the same elements.
  if (instr.opcode() == HloOpcode::kReduceWindow) {
    for (const auto& dim : instr.window().dimensions()) {
      if (dim.size() > dim.stride()) {
        return true;
      }
    }
  }
  return false;
}

bool IsExpensiveToRepeat(const HloInstruction& instr) {
  CHECK_NE(instr.opcode(), HloOpcode::kFusion) << "`instr` has to be unfused.";
  // Reductions which use many input elements to calculate one output element
  // are both memory and computationally heavy.
  constexpr int kMaxInputsPerOutput = 10;
  if (instr.opcode() == HloOpcode::kReduce &&
      !IsReductionFromOrToContiguousDimensions(instr)) {
    int64_t reduction_ratio = ShapeUtil::ElementsIn(instr.operand(0)->shape()) /
                              ShapeUtil::ElementsIn(instr.shape());
    if (reduction_ratio > kMaxInputsPerOutput) return true;
  }
  if (instr.opcode() == HloOpcode::kReduceWindow) {
    int64_t reduction_ratio = 1;
    for (const auto& dim : instr.window().dimensions())
      reduction_ratio *= dim.size();
    if (reduction_ratio > kMaxInputsPerOutput) return true;
  }
  return false;
}

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

bool TransposesMinorDimension(const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kFusion:
      return absl::c_any_of(instr->fused_instructions(),
                            TransposesMinorDimension);
    case HloOpcode::kCopy:
      return instr->shape().layout().minor_to_major(0) !=
             instr->operand(0)->shape().layout().minor_to_major(0);
    case HloOpcode::kTranspose: {
      // We have an input ([a,b,c]{x,y,z}) that's being transposed. We need to
      // check if the minor-most dimension (x) is still the minor-most dimension
      // after the transpose.
      int64_t minor_input =
          instr->operand(0)->shape().layout().minor_to_major(0);
      int64_t minor_output = instr->shape().layout().minor_to_major(0);
      return minor_input != instr->dimensions().at(minor_output);
    }
    default:
      return false;
  }
}

bool IsReduceInputFusion(const HloInstruction& instr) {
  return instr.opcode() == HloOpcode::kFusion &&
         absl::c_any_of(GetFusionRoots(*instr.called_computations()[0]),
                        [](const HloInstruction* root) {
                          return IsRealReductionHero(*root,
                                                     FindNonTrivialHero(*root));
                        });
}

bool IsInputFusibleReduction(const HloInstruction& instr) {
  return IsReduceInputFusion(instr) ||
         IsReductionFromOrToContiguousDimensions(instr);
}

bool IsNestableVariadicReduction(const HloInstruction& instr) {
  return instr.shape().IsTuple() &&
         ((instr.opcode() == HloOpcode::kReduce &&
           !IsReductionFromOrToContiguousDimensions(instr)) ||
          (instr.opcode() == HloOpcode::kFusion &&
           instr.fusion_kind() == HloInstruction::FusionKind::kLoop &&
           instr.fused_expression_root()->opcode() == HloOpcode::kReduce));
}

bool IsInputFusibleTranspose(const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kBitcast || instr.IsCustomFusion()) {
    return false;
  }
  if (instr.opcode() == HloOpcode::kFusion) {
    return HasAnyTiledTransposeRoot(*instr.fused_instructions_computation());
  }
  return GetDescriptionForTiledTransposeEmitter(instr, instr).has_value();
}

const HloInstruction* GetRealHeroForMultiOutputFusion(
    const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return &instr;
  }
  auto fused_expression_root = instr.fused_expression_root();
  if (!instr.IsMultiOutputFusion()) {
    const auto& hero = FindNonTrivialHero(*fused_expression_root);
    if (IsRealReductionHero(*fused_expression_root, hero) ||
        GetDescriptionForTiledTransposeEmitter(*fused_expression_root, hero)
            .has_value()) {
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
    if (IsRealReductionHero(*inst, hero) ||
        GetDescriptionForTiledTransposeEmitter(*inst, hero).has_value()) {
      return &hero;
    }
  }
  return fused_expression_root->operands()[0];
}

FusionDecision FusionHeroesAreCompatible(const HloInstruction* hero1,
                                         const HloInstruction* hero2) {
  auto hero1_is_unnested_reduce =
      IsReductionFromOrToContiguousDimensions(*hero1);
  auto tiled_transpose_hero1 =
      GetDescriptionForTiledTransposeEmitter(*hero1, *hero1);
  bool hero1_is_unnested_transpose = tiled_transpose_hero1.has_value();
  bool hero2_is_unnested_reduce =
      IsReductionFromOrToContiguousDimensions(*hero2);
  auto tiled_transpose_hero2 =
      GetDescriptionForTiledTransposeEmitter(*hero2, *hero2);
  bool hero2_is_unnested_transpose = tiled_transpose_hero2.has_value();

  if (hero1_is_unnested_reduce && hero2_is_unnested_reduce &&
      !AreReductionsMultiOutputFusionCompatible(hero2, hero1)) {
    return "tiled reductions with different shapes";
  } else if (hero1_is_unnested_transpose && hero2_is_unnested_transpose &&
             // After normalization to rank 3, the transposes should have the
             // same shape and permute the same dimensions.
             !tiled_transpose_hero1->IsEquivalent(*tiled_transpose_hero2)) {
    return "tiled transposes with different shapes";
  } else if ((hero1_is_unnested_transpose && hero2_is_unnested_reduce) ||
             (hero1_is_unnested_reduce && hero2_is_unnested_transpose)) {
    return "MOF-fusion of a transpose and a reduction";
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
          return "tiled transpose would become untiled";
        }
      } else if (hero2_is_unnested_transpose && fusion1->IsUserOf(fusion2)) {
        int64_t operand_idx = fusion1->operand_index(fusion2);
        auto hlo = fusion1->fused_parameter(operand_idx);
        if (!check_path_of_intermediate_ops(hlo)) {
          return "tiled transpose would become untiled";
        }
      }
    }
  }
  return {};
}

FusionDecision ShapesCompatibleForMultiOutputFusion(
    const HloInstruction& instr1, const HloInstruction& instr2) {
  // Multi-output fusion kernels share a common parallel loop. The loop
  // dimensions are determined by instruction shapes.
  auto get_loop_shape = [&](const HloInstruction* element_instr) {
    // Special-case reduction-to-vector ops: The loop dimensions are determined
    // by the shape of the first operand.
    // TODO(jreiffers): Compute the non-trivial hero only once here.
    const auto& hero = element_instr->parent()->IsFusionComputation()
                           ? FindNonTrivialHero(*element_instr)
                           : *element_instr;
    if (IsReductionFromOrToContiguousDimensions(*element_instr) ||
        GetDescriptionForTiledTransposeEmitter(*element_instr, hero)
            .has_value()) {
      return hero.operand(0)->shape();
    }
    return element_instr->shape();
  };

  // All shapes of the root tuple of multi-output fusions should agree, i.e. all
  // root ops should have equal output shapes. An exception are
  // reduction-to-vector ops. Here the input shapes of the reduction (first
  // operand shape) and the reduction dimensions need to match.
  const HloInstruction* hero1 = GetRealHeroForMultiOutputFusion(instr1);
  const HloInstruction* hero2 = GetRealHeroForMultiOutputFusion(instr2);

  if (auto compatible = FusionHeroesAreCompatible(hero1, hero2); !compatible) {
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
    return "different loop shapes";
  }
  return {};
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

bool IsInputFusible(const HloInstruction& instr) {
  // Input fusion only handles non-elemental reduction and scatter operations.
  return instr.IsFusible() &&
         (IsInputFusibleReduction(instr) || IsInputFusibleScatter(instr) ||
          IsInputFusibleTranspose(instr));
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
      return !GetDescriptionForTiledTransposeEmitter(instr, instr).has_value();

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

// Returns true if `instr` can be fused as a consumer into a kLoop fusion.
bool IsLoopFusibleAsConsumer(const HloInstruction& instr) {
  // Instr should be fusible.
  if (!instr.IsFusible()) return false;

  // An optimization for instruction fusion. Bitcast as a consumer means that it
  // will be a root of a fusion. This just adds indexing overhead without any
  // benefit.
  if (instr.opcode() == HloOpcode::kBitcast) return false;

  // Any reduction can be fused as a consumer.
  if (instr.opcode() == HloOpcode::kReduce) return true;

  // We may have input fusions which effectively have turned into loop
  // fusions. Those should still be considered as loop fusible consumers,
  // but they are not universally loop fusible.
  if (!IsInputFusible(instr) && instr.opcode() == HloOpcode::kFusion &&
      instr.fusion_kind() == HloInstruction::FusionKind::kInput) {
    return true;
  }

  return IsUniversallyLoopFusible(instr);
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
    return "do not fuse into the output of scatter";
  }
  if (!IsInputFusibleScatter(consumer)) {
    return {};
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
    return "do not fuse into the in-place operand of scatter";
  }
  if (absl::c_linear_search(producer.operands(), inplace_operand)) {
    return "Producer uses the in-place operand of a scatter";
  }
  return {};
}

FusionDecision IsProducerConsumerFusible(const HloInstruction& producer,
                                         const HloInstruction& consumer) {
  if (!IsLoopFusibleAsProducer(producer) &&
      !IsInputFusibleTranspose(producer)) {
    return "the producer is not loop-fusible";
  }

  if (IsInputFusibleReduction(producer)) {
    if (!producer.GetModule()
             ->config()
             .debug_options()
             .xla_gpu_enable_reduction_epilogue_fusion()) {
      return "Reduction epilogue fusion is not enabled.";
    }
    const HloInstruction& reduce_hero =
        producer.opcode() == HloOpcode::kFusion
            ? FindNonTrivialHero(*producer.fused_expression_root())
            : producer;
    if (!ReductionIsRaceFree(
            reduce_hero.GetModule()->config(),
            GetReductionKindAndContiguousComponents(reduce_hero))) {
      return "Reduction output fusion only works for race free reductions";
    }
    if (!AllSatisfy(consumer, [](const HloInstruction* hlo) {
          return IsIntermediate(hlo, /*allowed_operand_count=*/1);
        })) {
      return "Reductions from/to continuous dims epilogue not fusible";
    }

    if (producer.user_count() > 1) {
      return "reduction output fusion only works for single user";
    }
  }

  if (auto can_fuse = CanEmitInputFusedScatter(producer, consumer); !can_fuse) {
    return can_fuse;
  }

  if (!IsInputFusible(consumer) && !IsLoopFusibleAsConsumer(consumer)) {
    return "the consumer is not input-fusible and not loop-fusible";
  }

  // Skip multiple output fusion. It's not yet supported.
  if (producer.IsMultiOutputFusion()) {
    return "the producer is not fusible as it is a multi-output fusion";
  }

  // Fuse scalar constants into loop fusion nodes. This reduces the number of
  // parameters and makes matching scalar broadcasts easier.
  //
  // Don't fuse other constants: Unfused constants in GPU land can be
  // represented as an external constant (i.e. not emitted in LLVM IR / PTX),
  // but fused constants are handled by shared CPU/GPU code and always emitted
  // in the IR/PTX.  The external constant representation makes for faster
  // compiles and significantly smaller assembly code.
  if (producer.opcode() == HloOpcode::kConstant &&
      (!ShapeUtil::IsEffectiveScalar(producer.shape()) ||
       consumer.opcode() != HloOpcode::kFusion)) {
    return "not fusing constant";
  }

  // Make sure the new fusion obeys the in-place semantics.
  return InstructionFusion::ShouldFuseInPlaceOp(&producer, &consumer);
}

FusionDecision IsProducerMultiOutputFusible(const HloInstruction& producer) {
  // Skip multiple output fusion. It's not yet supported.
  if (producer.IsMultiOutputFusion()) {
    return "Producer is a multi-output fusion";
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
    return "In-place operations are present";
  }

  if (!IsLoopFusibleAsProducer(producer)) {
    return "producer is not loop-fusible";
  }

  if (IsPhysicallyTransposing(producer)) {
    return "producer is physically transposing";
  }

  return {};
}

// Returns an estimate of the shared memory usage for a given instruction in
// bytes.
static int64_t SharedMemoryUsageNoCache(const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kFusion) {
    int64_t sum = 0;
    for (const HloInstruction* hlo :
         instr.fused_instructions_computation()->instructions()) {
      sum += SharedMemoryUsageNoCache(*hlo);
    }
    return sum;
  } else if (instr.opcode() == HloOpcode::kReduce &&
             IsReductionFromOrToContiguousDimensions(instr)) {
    ReductionDimensions reduction_info =
        GetReductionKindAndContiguousComponents(instr);
    int64_t primitive_size = ShapeUtil::ByteSizeOfPrimitiveType(
        instr.operand(0)->shape().element_type());
    int num_variadic =
        instr.shape().IsTuple() ? instr.shape().tuple_shapes_size() : 1;
    if (reduction_info.is_row_reduction) {
      // __shared__[32] is used for row reduction.
      return 32 * primitive_size * num_variadic;
    } else {
      // __shared__[4][32][33] cache is used for column reduction ("4" comes
      // from potential x-tiling).
      return 4 * 32 * 33 * primitive_size * num_variadic;
    }
  } else if (GetDescriptionForTiledTransposeEmitter(instr, instr).has_value()) {
    // Tile size for transposition.
    int64_t primitive_size =
        ShapeUtil::ByteSizeOfPrimitiveType(instr.shape().element_type());
    return 32 * 33 * primitive_size;
  }
  // Other fused expressions for now don't need the shared memory budget.
  return 0;
}

int64_t SharedMemoryUsage(const HloInstruction& instr, FusionInfoCache* cache) {
  if (!cache) {
    return SharedMemoryUsageNoCache(instr);
  }

  // nb: Users are only expected to call cache.Invalidate() on top-level
  // instructions, not instructions inside fusion nodes.  Therefore we can only
  // cache top-level instructions; it would not be valid to pass the cache to
  // SharedMemoryUsageNoCache and use the cache *within* the fusion.
  auto it_and_inserted = cache->shared_memory_usage.emplace(&instr, -1);
  auto it = it_and_inserted.first;
  auto inserted = it_and_inserted.second;

  if (inserted) {
    it->second = SharedMemoryUsageNoCache(instr);
  }
  return it->second;
}

// Codegen'ing unnested reductions requires a lot of registers, so a MOF
// combining many of those runs a high risk of spilling.
constexpr int64_t kMaxUnnestedReductionOutputsPerFusion = 8;

// Returns the number of unnested reductions in the instruction output.
static int64_t NumUnnestedReductionsNoCache(const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kReduce &&
      IsReductionFromOrToContiguousDimensions(instr)) {
    return 1;
  }
  if (instr.opcode() == HloOpcode::kFusion) {
    int64_t sum = 0;
    for (const HloInstruction* hlo :
         instr.fused_instructions_computation()->instructions()) {
      sum += NumUnnestedReductionsNoCache(*hlo);
    }
    return sum;
  }
  return 0;
}

static int64_t NumUnnestedReductions(const HloInstruction& instr,
                                     FusionInfoCache* cache) {
  if (!cache) {
    return NumUnnestedReductionsNoCache(instr);
  }

  // nb: Users are only expected to call cache.Invalidate() on top-level
  // instructions, not instructions inside fusion nodes.  Therefore we can only
  // cache top-level instructions; it would not be valid to pass the cache to
  // NumUnnestedReductionsNoCache and use the cache *within* the fusion.
  auto it_and_inserted = cache->num_unnested_reductions.emplace(&instr, -1);
  auto it = it_and_inserted.first;
  auto inserted = it_and_inserted.second;

  if (inserted) {
    it->second = NumUnnestedReductionsNoCache(instr);
  }
  return it->second;
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
  if (SharedMemoryUsage(instr1, cache) + SharedMemoryUsage(instr2, cache) >
      device_info.shared_memory_per_block()) {
    return FusionDecision{}
           << "shared memory usage would be over the budget of "
           << device_info.shared_memory_per_block() << "B";
  }

  if (NumUnnestedReductions(instr1, cache) +
          NumUnnestedReductions(instr2, cache) >
      kMaxUnnestedReductionOutputsPerFusion) {
    return FusionDecision{} << "over " << kMaxUnnestedReductionOutputsPerFusion
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
    return {};
  } else {
    VLOG(5) << "Operand count of " << "(" << instr1.ToString()
            << " ) = " << instr1.operand_count() << " and ( "
            << instr2.ToString() << " ) = " << instr2.operand_count()
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
    return {};
  }

  // Does the new fusion have more operands and outputs than the max?
  if (operands.size() + num_output_buffers > MaxOperandsAndOutputsPerFusion()) {
    return "Number of operands and output buffers is larger than allowed "
           "budget per fusion";
  }
  return {};
}

bool CreatesHeavyComputation(const HloInstruction& producer,
                             const HloInstruction& consumer) {
  // If producer's computation is not expensive to repeat even in the consumer
  // requests the same element multiple times there is nothing to do.
  auto producer_is_heavy = [&](const HloInstruction& instr) {
    if (producer.opcode() != HloOpcode::kFusion) {
      return IsExpensiveToRepeat(producer);
    }
    for (const auto& instr : producer.fused_instructions()) {
      if (IsExpensiveToRepeat(*instr)) {
        return true;
      }
    }
    return false;
  };
  if (!producer_is_heavy(producer)) {
    return false;
  }

  // If consumer is a non-fusion instruction then we have to check if it
  // reads input multiple times.
  if (consumer.opcode() != HloOpcode::kFusion) {
    return IfFusedReadsElementsMultipleTimes(consumer);
  }

  // If consumer is a fusion then we have to check if the output of producer is
  // used directly or indirectly as an input to an HLO instruction that
  // accesses input multiple times, i.e. there is a path in the graph
  // from an operand corresponding to the producer to an HLO instruction
  // generating multiple accesses in the consumer.
  for (const HloInstruction* operand : consumer.operands()) {
    if (operand != &producer) {
      continue;
    }

    const HloInstruction* root =
        consumer.fused_instructions_computation()->parameter_instruction(
            consumer.operand_index(operand));

    std::stack<const HloInstruction*> dfs;
    dfs.push(root);
    absl::flat_hash_set<const HloInstruction*> visited;
    while (!dfs.empty()) {
      const HloInstruction* cur = dfs.top();
      dfs.pop();

      if (!visited.insert(cur).second) {
        continue;
      }

      if (IfFusedReadsElementsMultipleTimes(*cur)) {
        return true;
      }
      for (const auto& user : cur->users()) {
        if (visited.contains(user)) {
          continue;
        }
        dfs.push(user);
      }
    }
  }
  return false;
}

bool IsFusibleAsMultiOutputFusionRoot(const HloInstruction& instr) {
  // We can fuse reduces and loop fusions. Elementwise instructions can be fused
  // with any other instruction.
  // Note that scatter cannot be the root of a multi-output fusion because
  // its emitter doesn't support it.

  return instr.IsFusible() &&
         (IsInputFusibleReduction(instr) || IsInputFusibleTranspose(instr) ||
          instr.IsLoopFusion() ||  // TODO(b/130013493): Use IsLoopFusible here.
          instr.IsElementwise());
}

HloInstruction::FusionKind ChooseFusionKind(const HloInstruction& producer,
                                            const HloInstruction& consumer) {
  return (IsInputFusible(consumer) || IsInputFusible(producer))
             ? HloInstruction::FusionKind::kInput
             : HloInstruction::FusionKind::kLoop;
}

bool IsConsumerTheOnlyNonRootUser(const HloInstruction& instr,
                                  const HloInstruction& consumer) {
  return absl::c_all_of(instr.users(), [&](const HloInstruction* user) {
    if (user->opcode() == HloOpcode::kGetTupleElement) {
      // Skip GTE.
      return IsConsumerTheOnlyNonRootUser(*user, consumer);
    }
    // `user` is `consumer` or consumed by ROOT.
    return user == &consumer || user == user->parent()->root_instruction();
  });
}

size_t GetInstrCountOfFusible(const HloInstruction& instr) {
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
    CHECK(!absl::c_linear_search(out, root))
        << "Fusion root contains instruction " << root->ToString()
        << " multiple times";
    out.push_back(root);
  }
}

std::vector<const HloInstruction*> GetFusionRoots(
    const HloComputation& computation) {
  std::vector<const HloInstruction*> out;
  GetFusionRootsRec(computation.root_instruction(), out);
  return out;
}

bool IsTritonSoftmaxFusion(const HloInstruction& instr) {
  // TODO(b/332649307): Eventually turn this into a generic fusion.
  return instr.opcode() == HloOpcode::kFusion &&
         instr.fusion_kind() == HloInstruction::FusionKind::kCustom &&
         instr.backend_config<GpuBackendConfig>().ok() &&
         instr.backend_config<GpuBackendConfig>()
                 ->fusion_backend_config()
                 .kind() == kTritonSoftmaxFusionKind;
}

bool MayPreventVectorization(const HloFusionAdaptor& fusion) {
  // An empirically chosen constant: unrolling concat with a large amount of
  // arguments causes excessive register spilling.
  static constexpr int kMaxConcatArgumentsForUnrolling = 10;
  return HloAnyOf(fusion.GetRoots(), fusion, [&](auto node) {
    switch (node.opcode()) {
      case HloOpcode::kReduceWindow:
      case HloOpcode::kSort:
      case HloOpcode::kDot:
      case HloOpcode::kSin:
      case HloOpcode::kCos:
      case HloOpcode::kTan:
      case HloOpcode::kPower:
      case HloOpcode::kAtan2:
        return true;
      case HloOpcode::kConcatenate:
        return node.instruction().operand_count() >
               kMaxConcatArgumentsForUnrolling;
      case HloOpcode::kReduce:
        return node.instruction().shape().tuple_shapes_size() > 1;
      default:
        return false;
    }
  });
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

}  // namespace gpu
}  // namespace xla
