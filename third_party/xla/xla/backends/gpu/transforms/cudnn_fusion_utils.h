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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_CUDNN_FUSION_UTILS_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_CUDNN_FUSION_UTILS_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {
namespace gpu {

// Manages the growth of the fusion subgraph during DFS epilogue search.
// Supports snapshot/restore for backtracking when a consumer branch is invalid.
// Relies on the property that instructions are only ever appended to
// fusible_users and fusion_outputs, so RestoreSnapshot works by resizing.
struct FusionState {
  std::vector<HloInstruction*>& fusible_users;
  std::vector<HloInstruction*>& fusion_outputs;
  bool& can_fuse_reduce;

  struct Snapshot {
    int fusible_users_size;
    int fusion_outputs_size;
    bool can_fuse_reduce;
  };

  Snapshot TakeSnapshot() const;
  void RestoreSnapshot(const Snapshot& snapshot);
};

// Clones all instructions in fusible_users into builder using fused_hlo_map to
// resolve already-fused operands. Operands not yet in fused_hlo_map become new
// fusion parameters appended to fusion_params.
// is_nchw must match the value used during GrowFusionDFS.
void FuseTowardUsers(
    HloComputation::Builder& builder,
    std::vector<HloInstruction*>& fusion_params,
    std::vector<HloInstruction*>& fusible_users,
    absl::flat_hash_map<HloInstruction*, HloInstruction*>& fused_hlo_map,
    bool is_nchw = false);

// Performs a DFS from hlo through its users to collect all instructions that
// can be fused into a cuDNN epilogue. Updates state.fusible_users with nodes
// to clone and state.fusion_outputs with the boundary outputs.
//
// is_outputs_valid is called whenever the output set changes; it should return
// false if the current set of outputs is not supported (e.g. too many outputs).
// is_nchw disables all epilogue fusing (for NCHW convolutions).
//
// Returns false if hlo or any descendant forms an invalid subgraph.
bool GrowFusionDFS(HloInstruction* hlo, HloReachabilityMap* reachability,
                   FusionState& state,
                   absl::flat_hash_map<HloInstruction*, bool>& fusible_cache,
                   bool is_nchw,
                   absl::FunctionRef<bool(const std::vector<HloInstruction*>&)>
                       is_outputs_valid);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_CUDNN_FUSION_UTILS_H_
