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

// TODO(b/112957171): Extract logic to determine fusibility of HLO ops from
// GpuInstructionFusion, FusionMerger, and GpuMultiOutputFusion.

namespace xla {
namespace gpu {

constexpr int64 kMaxOperandsAndOutputsPerFusion = 64;

bool IsInputFusible(const HloInstruction& instr);

bool IsLoopFusible(const HloInstruction& instr);

// The code emitted for reduce-rooted input fusions (EmitReductionToVector)
// suffers from poor data locality if the layouts of input parameters differ. In
// such situations it is better not to fuse. Only input params with
// maximum rank are considered. Params with smaller ranks will be broadcasted
// and have not been observed to cause data locality issues.
// TODO(b/111977086): Improve reduce emitters to remove this limitation.
bool LayoutsAreReduceInputFusionFriendly(const HloInstruction& producer,
                                         const HloInstruction& reduce);

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
// multi-output) fusion would be "too large" -- i.e., have more operands and
// outputs than is allowed or occupy too much shared memory.
// If the fusion is a producer/consumer fusion and instr1 is the
// consumer and instr2 is the producer, set consumer_producer_fusion
// to true to enable more fusion.
bool FusionWouldBeTooLarge(const HloInstruction& instr1,
                           const HloInstruction& instr2,
                           bool is_consumer_producer_fusion = false);

// Check if fusing producer and consumer will generate a nested loop, e.g. both
// producer and consumer are `reduce-window` HLO instructions.
bool CreatesNestedLoop(const HloInstruction& producer,
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
bool IsProducerConsumerFusible(const HloInstruction& producer,
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
