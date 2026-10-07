/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_REWRITE_UTILS_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_REWRITE_UTILS_H_

#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"

namespace xla {

// All not fusion, constant and copy hlos are trivial to transform to different
// shapes.
bool IsTrivialElementwise(const HloInstruction& hlo);

// Is this an iota that cannot be transformed by algebraic simplifier.
bool IsNonTrivialIota(const Shape& target_shape, const HloInstruction* hlo);

// Convert a broadcast of one shape to a broadcast of another shape. For example
// to make a [B, H, W, C] out of a broadcast of [B, 2] that produces
// [B, H, W, C/2, 2] need to broadcast [B, 2] to [B, C/2, 2] reshape that to
// [B,C] and finally broadcast that to [B, H, W, C].
HloInstruction* TransformBroadcast(const Shape& shape,
                                   HloInstruction* broadcast,
                                   HloComputation* computation,
                                   bool transform_trivial = true);

// Similar to TransformBroadcast above
HloInstruction* TransformConcat(const Shape& shape, HloInstruction* concat,
                                bool transform_trivial = true);

// Similar to TransformBroadcast above
HloInstruction* TransformSlice(const Shape& shape, HloInstruction* slice,
                               bool transform_trivial = true);

// Finds elementwise subgraph that is surrounded by reshapes and broadcasts.
// Returns what it finds in 'finds'.
std::optional<Shape> FindElementwiseSubgraphSurroundedByReshapesAndBroadcasts(
    HloInstruction* root, std::optional<Shape> target_shape,
    absl::flat_hash_set<HloInstruction*>* finds,
    HloOpcode opc = HloOpcode::kReshape);

// FindElementwiseSubgraphSurroundedByReshapesAndBroadcasts with depth limit.
std::optional<Shape const*>
FindElementwiseSubgraphSurroundedByReshapesAndBroadcastsWithLimit(
    HloInstruction* root, std::optional<Shape const*> target_shape,
    absl::flat_hash_set<HloInstruction*>* finds, HloOpcode opc, int d = 0);

// Replaces elementwise group surrounded by reshapes and broadcasts.
HloInstruction* ReplaceElementwiseGroupSurroundedByReshapesAndBroadcasts(
    const Shape& shape, HloInstruction* root, HloComputation* computation,
    absl::flat_hash_map<HloInstruction*, HloInstruction*>* replacements);

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_REWRITE_UTILS_H_
