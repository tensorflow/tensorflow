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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_FUSIBLE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_FUSIBLE_H_

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/instruction_fusion.h"

// TODO(b/112957171): Extract logic to determine fusibility of HLO ops from
// GpuInstructionFusion, FusionMerger, and GpuMultiOutputFusion.

namespace xla {
namespace gpu {

// Check if the operation accesses the same inputs multiple times
// while generating its outputs.
bool IfFusedReadsElementsMultipleTimes(const HloInstruction& instr);

// Check if the operation is memory or computationally expensive
// to repeat.
bool IsExpensiveToRepeat(const HloInstruction& instr);

// Fusion passes frequently do checks across all pairs of "interesting" nodes.
// Computing e.g. FusionFitsInBudget(a, b) requires computing expensive
// properties of `a` and `b` individually.  This cache lets us avoid recomputing
// those properties n^2 times.
//
// Invariant: After modifying or removing a fusion node, call Invalidate(node).
struct FusionInfoCache {
 public:
  // Must be called after modifying or removing a fusion node (or other node
  // that's part of this cache).
  void Invalidate(const HloInstruction* instr) {
    shared_memory_usage.erase(instr);
    num_unnested_reductions.erase(instr);
  }

  // The rest of the members of this class are for internal use within
  // gpu_fusible. You shouldn't need to use them yourself.
  absl::flat_hash_map<const HloInstruction*, int64_t> shared_memory_usage;
  absl::flat_hash_map<const HloInstruction*, int64_t> num_unnested_reductions;
};

inline constexpr int64_t MaxOperandsAndOutputsPerFusion() { return 64; }

bool IsInputFusible(const HloInstruction& instr);

bool IsLoopFusible(const HloInstruction& instr);

// Whether the op tranposes the physical data layout. Fusing such ops may lead
// to uncoalesced data access and may thus not be beneficial.
bool IsPhysicallyTransposing(const HloInstruction& instr);

// Note that reduction ops are lowered in different ways. Reduce input fusions
// are lowered by IrEmitterUnnested::EmitReductionToVector and must be rooted at
// reduction-to-vector ops. Other reduction ops are lowered by
// GpuElementalIrEmitter and fused like elementwise ops.

// Whether `instr` is an input fusion rooted at a reduction-to-vector op or a
// multi-output input fusion with at least one reduction-to-vector op root.
bool IsReduceInputFusion(const HloInstruction& instr);

// Whether `instr` is fusible as root of a reduce input fusions, i.e. `instr`
// is either an unfused reduction-to-vector op or a reduce input fusion.
bool IsInputFusibleReduction(const HloInstruction& instr);

// Whether `instr` is fusible as root of a scatter input fusions, i.e. `instr`
// is either an unfused scatter op or a scatter input fusion.
bool IsInputFusibleScatter(const HloInstruction& instr);

// Determines whether the combination of `instr1` and `instr2` into a (possibly
// multi-output) fusion fits within a "budget" -- i.e., does have more operands
// and outputs than is allowed or occupy too much shared memory. If the fusion
// is a producer/consumer fusion and `instr1` is the consumer and `instr2` is
// the producer, set consumer_producer_fusion to true to enable more fusion.
FusionDecision FusionFitsInBudget(const HloInstruction& instr1,
                                  const HloInstruction& instr2,
                                  bool is_consumer_producer_fusion = false,
                                  FusionInfoCache* cache = nullptr);

// Check if fusing producer and consumer will generate a heavy computation, e.g.
// producer has a complex computation per output and consumer calls this
// computations multiple times.
bool CreatesHeavyComputation(const HloInstruction& producer,
                             const HloInstruction& consumer);

// Returns the instruction that determines the emitter used for lowering,
// sometimes referred to as "the real hero".
const HloInstruction* GetRealHeroForMultiOutputFusion(
    const HloInstruction& instr);

// Whether instruction shapes are compatible for multi-output fusion, i.e.
// whether the emitters support lowering the resulting fusion.
// This function works for both, sibling and producer-consumer multi-output
// fusion.
// So far, multi-output fusion is supported for loop fusions and reduce
// input fusions only. It is up to the caller to ensure the instructions
// themselves are fusible!
bool ShapesCompatibleForMultiOutputFusion(const HloInstruction& instr1,
                                          const HloInstruction& instr2);

// Whether the instructions are compatible for producer-consumer fusion
// i.e. whether the producer and consumer are loop/input fusible and
// they are not library calls.
FusionDecision IsProducerConsumerFusible(const HloInstruction& producer,
                                         const HloInstruction& consumer);

// Whether the instructions are producer-consumer fusible with multiple outputs.
// That is, the root tuple of the multi-output fusion will contain the results
// of both, the producer and consumer.
bool IsProducerConsumerMultiOutputFusible(const HloInstruction& producer,
                                          const HloInstruction& consumer);
// Whether `instr` is a candidate for sibling fusion or as a consumer in
// a producer-consumer multi-output fusion.
bool IsFusibleAsMultiOutputFusionRoot(const HloInstruction& instr);

// Determines the fusion kind to be used when fusing `producer` and `consumer`.
HloInstruction::FusionKind ChooseFusionKind(const HloInstruction& producer,
                                            const HloInstruction& consumer);

// Returns whether `consumer` is the only non-root user of `instr`.
bool IsConsumerTheOnlyNonRootUser(const HloInstruction& instr,
                                  const HloInstruction& consumer);

// Returns number of instructions in the fusible `instr`. If `instr` is not a
// fusion instruction, 1 is returned.
size_t GetInstrCountOfFusible(const HloInstruction& instr);

// Returns the outputs of the fusible `instr`.
absl::InlinedVector<const HloInstruction*, 2> GetOutputsOfFusible(
    const HloInstruction& instr);

// Returns the output size of the fusible `instr`.
size_t GetOutputSizeOfFusible(const HloInstruction& instr);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_FUSIBLE_H_
