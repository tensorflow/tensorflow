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

#ifndef XLA_SERVICE_HLO_LIVENESS_ANALYSIS_H_
#define XLA_SERVICE_HLO_LIVENESS_ANALYSIS_H_

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_value.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"

namespace xla {

// Analysis which identifies all live {HloInstruction, ShapeIndex} pairs in
// an HLO module.
//
// HloLivenessAnalysis marks the shape index of each live output of each
// instruction in the module, by propagating live shape index information
// from an instruction to its called computations and operands.
class HloLivenessAnalysis {
 public:
  // Maps from an HloInstruction to its live/dead output shape indices.
  using HloIndexMap = absl::flat_hash_map<const HloInstruction*,
                                          std::unique_ptr<ShapeTree<bool>>>;

  // Runs liveness analysis on 'module'. Returns HloLivenessAnalysis object
  // which exports liveness for each {HloInstruction, ShapeIndex} in 'module'.
  static absl::StatusOr<std::unique_ptr<HloLivenessAnalysis>> Run(
      const HloModule& module);

  // Returns true if output of 'instruction' at 'shape_index' is live.
  // Returns false otherwise.
  bool IsLive(const HloInstruction* instruction,
              const ShapeIndex& shape_index) const;

 private:
  HloLivenessAnalysis(const HloModule& module);

  void RunAnalysis();

  const HloModule& module_;
  std::unique_ptr<CallGraph> call_graph_;
  HloIndexMap live_index_map_;
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_LIVENESS_ANALYSIS_H_
