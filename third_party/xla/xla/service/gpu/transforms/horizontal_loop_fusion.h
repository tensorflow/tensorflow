/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_HORIZONTAL_LOOP_FUSION_H_
#define XLA_SERVICE_GPU_TRANSFORMS_HORIZONTAL_LOOP_FUSION_H_

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// This optimization pass horizontally fuses computations for reducing kernel
// launch overhead while increasing kernel launch dims on GPU. The initial
// motivation of this horizontal fusion is due to the observation that the
// training optimizer phase (e.g., AdamOptimizer and L2Loss, etc.) typically
// has many small kernels as a result of applying the same formula on many
// training parameters (or variables in Tensorflow). Fusing these small
// kernels, hence, provides performance gain.
//
// Theoretically speaking, we may implement a cycle detection algorithm to make
// sure no cycles are created after fusion. However, cycle detection check is
// somewhat cumbersome; also, we observe that naive horizontal fusion of
// arbitrary kernels may not be profitable due to control divergence and
// possible increase of memory bandwidth pressure due to uncoalesced memory
// accesses (note that horizontal fusion does not change the amount of memory
// read+written at all). In practice, a simple yet effective heuristic is used
// to avoid these issues while addressing the known beneficial cases. That is,
// we simply search for fusion candidates by looking for instructions whose
// outputs are all consumed by the same instruction. This catches the cases in
// the training optimizer phase, as the candidate instructions are typically
// consumed only by the ROOT tuple of the entry computation.
//
// The following illustrates the mechanism of the horizontal fusion. Before
// fusion, there are two trivial kernels in the illustrating example. One has
// only a Mul op, while the other consists of only an Add op. Since they are
// only consumed by the same (ROOT) tuple instruction, horizontal fusion is
// triggered.
//
// i0 i1   i2 i3
//  | |     | |
//  v v     v v
//  Mul     Add
//   |       |
//   v       v
//  (ROOT) tuple
//
// We fuse into one of two possible patterns, depending on whether all the
// fused operations have the same shape or not.
//
// case 1: if Mul and Add's output shape and type are the same, then we fuse
// them into the below pattern:
// i0 i1   i2 i3
//  | |     | |
//  v v     v v
//  Mul     Add
//   |       |
//   v       v
//  (ROOT) tuple
// the fused kernel will be kLoop type, and GPU code is emitted through
// the LoopFusion class.
//
// case 2: if Mul and Add's output shape are diffent, then we fuse them into
// the below pattern that adds extra indexing:
// i0 i1   i2 i3       +++ (Slice) Input Fusion
//  | |     | |          +
//  v v     v v          +
//  Mul     Add          +
//   |       |           +
//   v       v           +
// Reshape0  Reshape1    +
//   |       |           +
//   v       v           +
//  Concatenate          +
//   |       |           +
//   v       v           +
//  Slice0  Slice1     +++
//   |       |
//   v       v
// Reshape2  Reshape3
//   |       |
//   v       v
//  (ROOT) tuple
//
// the fused kernel will be kInput type, and, the GPU code is emitted through
// the InputSlicesFusion class.
//
// In theory, the pattern in case 1 could also be fused into the case2 target
// graph, but we prefer to fuse into kLoop type, because the codegen for it does
// not have the slicing range check cost introduced by case 2 pattern.
//
// Note that the fusion style by case 2 provides an important advantage that
// kernels of different shapes can be horizontally fused. The first pair of
// reshapes (i.e., Reshape0 and Reshape1) reshape the dims to 1 dimension, so
// that the outputs of the fused kernels can (always) be concatenated. The
// second pair of reshapes (Reshape2 and Reshape3) restore the original shapes
// to the output tensors.
//
// No extra copies are introduced by the horizontal fusion. Besides Reshape2
// and Reshape3, the other instructions are fused into an input fusion; the
// output dims of the concatenate will be used as the kernel launch dims.
// Instruction bitcasts can be used for Reshape2 and Reshape3 as long as the
// outputs of Mul and Add are row-major.
//
// Note, reshapes are added only if the tensors isn't already a vector.
class HorizontalLoopFusion : public HloModulePass {
 public:
  HorizontalLoopFusion() = default;
  explicit HorizontalLoopFusion(absl::string_view prefix) : prefix_(prefix) {}

  absl::string_view name() const override { return "horizontal_loop_fusion"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  absl::StatusOr<bool> RunOnComputation(HloComputation*);
  std::string prefix_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_HORIZONTAL_LOOP_FUSION_H_
