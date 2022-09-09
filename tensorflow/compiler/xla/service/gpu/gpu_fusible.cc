/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"

#include <algorithm>
#include <iterator>
#include <optional>
#include <stack>
#include <vector>

#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/instruction_fusion.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace gpu {

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

bool IsReduceInputFusion(const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kFusion) {
    if (HasAnyUnnestedReductionRoot(instr.called_computations()[0])) {
      CHECK(instr.IsInputFusion())
          << " Fusion rooted at reduction-to-vector op must be of kind kInput: "
          << instr.ToString();
      return true;
    }
  }
  return false;
}

bool IsInputFusibleReduction(const HloInstruction& instr) {
  return IsReduceInputFusion(instr) ||
         IsReductionFromOrToContiguousDimensions(instr);
}

bool IsTransposeInputFusion(const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kFusion) {
    HloComputation* fusion = instr.called_computations()[0];
    return absl::c_any_of(
        GetFusionRoots(fusion),
        [&](const HloInstruction* instr) { return IsTiledTranspose(*instr); });
  }
  return false;
}

bool IsInputFusibleTranspose(const HloInstruction& instr) {
  return IsTiledTranspose(instr) || IsTransposeInputFusion(instr);
}

const HloInstruction* GetRealHeroForMultiOutputFusion(
    const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return &instr;
  }
  auto fused_expression_root = instr.fused_expression_root();
  if (!instr.IsMultiOutputFusion()) {
    return fused_expression_root;
  }
  // If possible, we want to pick a reduction-from-or-to-contiguous-dims
  // operand of the fusion root, because it has the most constraints.
  for (const auto* inst : fused_expression_root->operands()) {
    if (IsReductionFromOrToContiguousDimensions(*inst) ||
        IsTiledTranspose(*inst)) {
      return inst;
    }
  }
  return fused_expression_root->operands()[0];
}

bool ShapesCompatibleForMultiOutputFusion(const HloInstruction& instr1,
                                          const HloInstruction& instr2) {
  // Multi-output fusion kernels share a common parallel loop. The loop
  // dimensions are determined by instruction shapes.
  auto get_loop_shape = [&](const HloInstruction* element_instr) {
    // Special-case reduction-to-vector ops: The loop dimensions are determined
    // by the shape of the first operand.
    if (IsReductionFromOrToContiguousDimensions(*element_instr) ||
        IsTiledTranspose(*element_instr)) {
      return element_instr->operand(0)->shape();
    }
    return element_instr->shape();
  };

  // All shapes of the root tuple of multi-output fusions should agree, i.e. all
  // root ops should have equal output shapes. An exception are
  // reduction-to-vector ops. Here the input shapes of the reduction (first
  // operand shape) and the reduction dimensions need to match.
  auto* instr_1 = GetRealHeroForMultiOutputFusion(instr1);
  auto* instr_2 = GetRealHeroForMultiOutputFusion(instr2);
  if (IsReductionFromOrToContiguousDimensions(*instr_1) &&
      IsReductionFromOrToContiguousDimensions(*instr_2) &&
      !AreFusedReductionOutputsConsistent({instr_1, instr_2}, instr_1)) {
    return false;
  } else if (IsTiledTranspose(*instr_1) && IsTiledTranspose(*instr_2) &&
             (instr_1->shape() != instr_2->shape() ||
              instr_1->operand(0)->shape() != instr_2->operand(0)->shape())) {
    return false;
  } else if ((IsTiledTranspose(*instr_1) &&
              IsReductionFromOrToContiguousDimensions(*instr_2)) ||
             (IsTiledTranspose(*instr_2) &&
              IsReductionFromOrToContiguousDimensions(*instr_1))) {
    return false;
  }
  // The elementwise output shapes must be the same (including layout).
  return ShapeUtil::EqualIgnoringElementType(get_loop_shape(instr_1),
                                             get_loop_shape(instr_2));
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

bool IsLoopFusible(const HloInstruction& instr) {
  // Don't fuse get-tuple-element on GPU: We can, but it's slower than not
  // fusing.  We never generate kernels for unfused GTEs.  Instead, if an
  // unfused GTE is an input to a kernel (including a fusion kernel), we
  // compute the address of the GTE at the top of the kernel.  Often we know the
  // address of the GTE result statically, so we can do this without chasing any
  // pointers.
  return instr.IsFusible() &&
         ((instr.IsElementwise() && instr.operand_count() > 0 &&
           instr.opcode() != HloOpcode::kCopy) ||
          (instr.opcode() == HloOpcode::kCopy && !IsTiledTranspose(instr)) ||
          instr.opcode() == HloOpcode::kBitcast ||
          instr.opcode() == HloOpcode::kBroadcast ||
          instr.opcode() == HloOpcode::kConcatenate ||
          instr.opcode() == HloOpcode::kDynamicSlice ||
          instr.opcode() == HloOpcode::kDynamicUpdateSlice ||
          (instr.opcode() == HloOpcode::kFusion &&
           instr.fusion_kind() == HloInstruction::FusionKind::kLoop) ||
          instr.opcode() == HloOpcode::kGather ||
          instr.opcode() == HloOpcode::kIota ||
          instr.opcode() == HloOpcode::kPad ||
          (instr.opcode() == HloOpcode::kReduce &&
           !IsReductionFromOrToContiguousDimensions(instr) &&
           !instr.shape().IsTuple()) ||  // TODO(b/129089333): Don't fuse
                                         // variadic reductions.
          instr.opcode() == HloOpcode::kReduceWindow ||
          instr.opcode() == HloOpcode::kReshape ||
          instr.opcode() == HloOpcode::kReverse ||
          instr.opcode() == HloOpcode::kSlice ||
          instr.opcode() == HloOpcode::kConstant ||
          instr.opcode() == HloOpcode::kTranspose);
}

FusionDecision IsProducerConsumerFusible(const HloInstruction& producer,
                                         const HloInstruction& consumer) {
  if (!IsLoopFusible(producer)) {
    return "the producer is not loop-fusible";
  }

  if (!IsInputFusible(consumer) && !IsLoopFusible(consumer)) {
    return "the consumer is not input-fusible and not loop-fusible";
  }

  // Skip multiple output fusion. It's not yet supported.
  if (producer.IsMultiOutputFusion()) {
    return "the producer is not fusible as it is a multi-output fusion";
  }

  if (CreatesHeavyComputation(producer, consumer)) {
    return "the fusion would create a heavy computation";
  }

  // Do not fuse into fusions if the resulting kernel would suffer from
  // uncoalesced reads due to a transposed memory access pattern.
  if (IsInputFusibleReduction(consumer) && IsPhysicallyTransposing(producer)) {
    return "fusing the producer would break read coalescing";
  }

  // Fuse scalar constants into loop fusion nodes. This reduces the number of
  // parameters and makes matching scalar broadcasts easier.
  //
  // Don't fuse other constants: Unfused constants in GPU land can be
  // represented as an external constant (i.e. not emitted in LLVM IR / PTX),
  // but fused constants are handled by shrared CPU/GPU code and always emitted
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

bool IsProducerConsumerMultiOutputFusible(const HloInstruction& producer,
                                          const HloInstruction& consumer) {
  // Skip multiple output fusion. It's not yet supported.
  if (producer.IsMultiOutputFusion()) {
    return false;
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
    return false;
  }
  if (!IsLoopFusible(producer) || !IsFusibleAsMultiOutputFusionRoot(consumer)) {
    return false;
  }
  if (CreatesHeavyComputation(producer, consumer)) {
    return false;
  }
  if (!ShapesCompatibleForMultiOutputFusion(producer, consumer)) {
    return false;
  }
  if (IsPhysicallyTransposing(producer)) {
    return false;
  }
  return true;
}

// Returns shared memory usage for a given instruction in bytes.
static int64_t SharedMemoryUsageNoCache(const HloInstruction& instr) {
  // For now we are only fusing reductions.
  if (instr.opcode() == HloOpcode::kReduce &&
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
      // __shared__[2][32][33] cache is used for column reduction ("2" comes
      // from potential x-tiling).
      return 2 * 32 * 33 * primitive_size * num_variadic;
    }
  } else if (IsTiledTranspose(instr)) {
    // Tile size for transposition.
    int64_t primitive_size =
        ShapeUtil::ByteSizeOfPrimitiveType(instr.shape().element_type());
    return 32 * 33 * primitive_size;
  } else if (instr.opcode() == HloOpcode::kFusion) {
    int64_t sum = 0;
    for (const HloInstruction* hlo :
         instr.fused_instructions_computation()->instructions()) {
      sum += SharedMemoryUsageNoCache(*hlo);
    }
    return sum;
  }
  // Other fused expressions for now don't need the shared memory budget.
  return 0;
}

static int64_t SharedMemoryUsage(const HloInstruction& instr,
                                 FusionInfoCache* cache = nullptr) {
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
                                  bool is_consumer_producer_fusion,
                                  FusionInfoCache* cache /*=nullptr*/) {
  if (SharedMemoryUsage(instr1, cache) + SharedMemoryUsage(instr2, cache) >
      kSharedMemoryBudgetInBytes) {
    return FusionDecision{}
           << "shared memory usage would be over the budget of "
           << kSharedMemoryBudgetInBytes << "B";
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

      if (visited.contains(cur)) {
        continue;
      }
      visited.insert(cur);

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

HloInstruction::FusionKind ChooseFusionKind(const HloInstruction& /*producer*/,
                                            const HloInstruction& consumer) {
  return IsInputFusible(consumer) ? HloInstruction::FusionKind::kInput
                                  : HloInstruction::FusionKind::kLoop;
}

bool IsConsumerTheOnlyNonRootUser(const HloInstruction& instr,
                                  const HloInstruction& consumer) {
  return absl::c_all_of(instr.users(), [&](const HloInstruction* user) {
    if (user->opcode() == HloOpcode::kGetTupleElement) {
      // Skip GTE.
      return IsConsumerTheOnlyNonRootUser(*user, consumer);
    }
    if (user == &consumer) {
      // `user` is `consumer`.
      return true;
    }
    if (user == user->parent()->root_instruction()) {
      // Consumed by ROOT.
      return true;
    }
    return false;
  });
}

size_t GetInstrCountOfFusible(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return 1;
  } else {
    return instr.fused_instruction_count();
  }
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

}  // namespace gpu
}  // namespace xla
