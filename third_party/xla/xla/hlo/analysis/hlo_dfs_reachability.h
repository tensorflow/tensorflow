/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_HLO_ANALYSIS_HLO_DFS_REACHABILITY_H_
#define XLA_HLO_ANALYSIS_HLO_DFS_REACHABILITY_H_

#include <cstddef>
#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

// A simple DFS-based reachability analysis for HLO instructions.
//
// When the class is created, the instructions are ordered in a defs-before-uses
// topological order.
// IsReachable runs a DFS from the destination node (going up through
// operands / control predecessors), and stops when the instruction's index in
// the defs-before-uses list is before the source node. As the reachability is
// tested for nodes that are close to each other, this optimization works well,
// and the time is dominated by the post-order sort.
class HloDfsReachability {
 public:
  // Returns true iff the instruction was present in the computation passed to
  // Build() or it was added via OnInstructionReplaced(). The calling code may
  // want to still use the class after the computation is modified, if it's
  // known that the def-before-use order is still preserved.
  bool IsPresent(const HloInstruction* instruction) const;
  // Returns true iff there is a path (with edges being users and control
  // successors) from 'from' to 'to'. (i.e. path from definitions to uses; from
  // producers to consumers)
  bool IsReachable(const HloInstruction* from, const HloInstruction* to) const;
  // Returns true iff either `a` is reachable from `b` or `b` is reachable from
  // `a`.
  bool IsConnected(const HloInstruction* a, const HloInstruction* b) const;
  // Returns the `candidates` that are connected to `a`, i.e. the ones for which
  // IsConnected(a, candidate) is true. Walks the graph once up and once down
  // from `a` instead of running one search per candidate.
  //
  // The walk down follows users, so unlike IsReachable this needs IsPresent()
  // to hold for every instruction of the computation, not only for `a` and
  // `candidates`. Call it on a map that was built after the last change to the
  // computation.
  absl::flat_hash_set<const HloInstruction*> FindConnected(
      const HloInstruction* a,
      absl::Span<const HloInstruction* const> candidates) const;
  static std::unique_ptr<HloDfsReachability> Build(
      const HloComputation* computation);
  // Updates the internal data structure when instruction `previous` was
  // replaced with instruction `now` in the computation. Requires
  // IsPresent(previous) returns true and IsPresent(now) returns false.
  // Postconditon: IsPresent(previous) returns false and IsPresent(now) returns
  // true.
  void OnInstructionReplaced(const HloInstruction* previous,
                             const HloInstruction* now);

 private:
  // LLVM dense map shows ~10-20% speedup compared to absl::flat_hash_map.
  llvm::DenseMap<const HloInstruction*, size_t> instruction_to_idx_;
};

}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_HLO_DFS_REACHABILITY_H_
