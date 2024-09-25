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

#ifndef XLA_HLO_IR_HLO_DFS_REACHABILITY_H_
#define XLA_HLO_IR_HLO_DFS_REACHABILITY_H_

#include <cstddef>
#include <memory>

#include "llvm/ADT/DenseMap.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

// A simple DFS-based reachability analysis for HLO instructions.
//
// When the class is created, the instructions are ordered in a defs-before-uses
// topological order.
// The reachability query runs a DFS from the destination node (going up through
// operands / control predecessors), and stops when the instruction's index in
// the defs-before-uses list is before the source node. As the reachability is
// tested for nodes that are close to each other, this optimization works well,
// and the time is dominated by the post-order sort.
class HloDfsReachability {
 public:
  // Returns true iff the instruction was present in the computation passed to
  // Build(). The calling code may want to still use the class after the
  // computation is modified, if it's known that the def-before-use order is
  // still preserved.
  bool IsPresent(const HloInstruction* instruction) const;
  // Returns true iff there is a path (with edges being users and control
  // successors) from 'from' to 'to'. (i.e. path from definitions to uses; from
  // producers to consumers)
  bool IsReachable(const HloInstruction* from, const HloInstruction* to) const;
  // Returns true iff either `a` is reachable from `b` or `b` is reachable from
  // `a`.
  bool IsConnected(const HloInstruction* a, const HloInstruction* b) const;
  static std::unique_ptr<HloDfsReachability> Build(
      const HloComputation* computation);

 private:
  // LLVM dense map shows ~10-20% speedup compared to absl::flat_hash_map.
  llvm::DenseMap<const HloInstruction*, size_t> instruction_to_idx_;
};

}  // namespace xla

#endif  // XLA_HLO_IR_HLO_DFS_REACHABILITY_H_
