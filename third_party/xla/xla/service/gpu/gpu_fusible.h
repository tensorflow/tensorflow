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

#ifndef XLA_SERVICE_GPU_GPU_FUSIBLE_H_
#define XLA_SERVICE_GPU_GPU_FUSIBLE_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/instruction_fusion.h"
#include "xla/stream_executor/device_description.h"

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

// Returns projected shared memory usage of a given instruction in bytes.
int64_t SharedMemoryUsage(const HloInstruction& instr,
                          FusionInfoCache* cache = nullptr);

inline constexpr int64_t MaxOperandsAndOutputsPerFusion() { return 96; }

// Whether the op transposes the physical data layout. Fusing such ops may lead
// to uncoalesced data access and may thus not be beneficial.
bool IsPhysicallyTransposing(const HloInstruction& instr);

// Whether the op transposes the minor-most dimension. In the case of fusions,
// whether the fusion contains some op that does this.
// If the minor-most dimension is transposed, this results in uncoalesced memory
// accesses in untiled code generators. If some other dimension is transposed,
// this just results in additional index computations.
// Note that this function makes several simplifying assumptions:
// - For non-fusion instructions, we assume the output is materialized as is.
//   For internal instructions, this may not be the case.
// - For fusions, it simply checks the output of this function for each
//   instruction in the fusion's computation.
// - There's no way to tell which parameters of the fusion are transposed.
// TODO(jreiffers): Take into account the size of the transposed dimension as
// well.
bool TransposesMinorDimension(const HloInstruction* instr);

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

// Whether `instr` is a nestable variadic reduction
// or a loop fusion rooted with such.
bool IsNestableVariadicReduction(const HloInstruction& instr);

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
                                  const se::DeviceDescription& device_info,
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

// Whether 'hero1' and 'hero2' are compatible if the two fusions containing
// 'hero1' and 'hero2' are merged together. For example merging two fusions with
// a reduction hero and a transpose here, respectively, does not work.
FusionDecision FusionHeroesAreCompatible(const HloInstruction* hero1,
                                         const HloInstruction* hero2);

// Whether instruction shapes are compatible for multi-output fusion, i.e.
// whether the emitters support lowering the resulting fusion.
// This function works for both, sibling and producer-consumer multi-output
// fusion.
// So far, multi-output fusion is supported for loop fusions and reduce
// input fusions only. It is up to the caller to ensure the instructions
// themselves are fusible!
FusionDecision ShapesCompatibleForMultiOutputFusion(
    const HloInstruction& instr1, const HloInstruction& instr2);

// Whether fusing producer into consumer creates a scatter fusion that cannot be
// handled by the scatter emitter.
FusionDecision CanEmitInputFusedScatter(const HloInstruction& producer,
                                        const HloInstruction& consumer);

// Whether the instructions are compatible for producer-consumer fusion
// i.e. whether the producer and consumer are loop/input fusible and
// they are not library calls.
// Used both by instruction fusion and fusion-fusion merging.
FusionDecision IsProducerConsumerFusible(const HloInstruction& producer,
                                         const HloInstruction& consumer);

// Whether the producer is a valid candidate for a multi-output fusion.
// That is, the root tuple of the multi-output fusion will contain the results
// of both, the producer and consumer.
FusionDecision IsProducerMultiOutputFusible(const HloInstruction& producer);
// Whether `instr` is a candidate for sibling fusion or as a consumer in
// a producer-consumer multi-output fusion.
bool IsFusibleAsMultiOutputFusionRoot(const HloInstruction& instr);

// Determines the fusion kind to be used when fusing into `consumer`.
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

// Returns instructions which are roots of the fusion, following the operands of
// GTE instructions in the root tuple. Groups multiple subsequent instructions
// with the same root. CHECKs that the fusion never outputs the same instruction
// twice, as well as that there are no explicitly created tuples or nested gtes
// in fusion output.
//
// For input: (tuple (gte R1) (gte R1) O2)
// Expected output: [R1, O2]
//
// For input: (tuple R1 R2 O2)
// Expected output: [R1, R2, O2]
//
// For input: (tuple (gte R1) (gte R1) R2 O3)
// Expected output: [R1, R2, O3]
//
// For input: R1
// Expected output: [R1]
std::vector<const HloInstruction*> GetFusionRoots(
    const HloComputation& computation);

// Whether the instruction is a Triton Softmax fusion.
bool IsTritonSoftmaxFusion(const HloInstruction& instr);

// Whether the fusion will likely behave poorly with vectorization due to the
// instructions it contains.
bool MayPreventVectorization(const HloFusionAdaptor& fusion);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_FUSIBLE_H_
