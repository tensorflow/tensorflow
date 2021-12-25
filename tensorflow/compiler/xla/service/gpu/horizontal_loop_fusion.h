/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HORIZONTAL_LOOP_FUSION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HORIZONTAL_LOOP_FUSION_H_

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

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
// We horizontally fuse them into the below pattern.
//
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
// Note that this fusion style provides an important advantage that kernels of
// different shapes can be horizontally fused. The first pair of reshapes
// (i.e., Reshape0 and Reshape1) reshape the dims to 1 dimension, so that the
// outputs of the fused kernels can (always) be concatenated. The second pair
// of reshapes (Reshape2 and Reshape3) restore the original shapes to the
// output tensors.
//
// No extra copies are introduced by the horizontal fusion. Besides Reshape2
// and Reshape3, the other instructions are fused into an input fusion; the
// output dims of the concatenate will be used as the kernel launch dims.
// Instruction bitcasts can be used for Reshape2 and Reshape3 as long as the
// outputs of Mul and Add are row-major.
class GpuHorizontalLoopFusion : public HloModulePass {
 public:
  GpuHorizontalLoopFusion() {}

  absl::string_view name() const override {
    return "gpu_horizontal_loop_fusion";
  }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  StatusOr<bool> RunOnComputation(HloComputation*);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HORIZONTAL_LOOP_FUSION_H_
