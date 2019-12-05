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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HORIZONTAL_FUSION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HORIZONTAL_FUSION_H_

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {
namespace gpu {

// This optimization pass horizontally fuses computations for reducing kernel
// launch overhead while increasing kernel launch dims on GPU. The initial
// motivation of this horizontal fusion is due to the observation that the
// training optimizer phase typically has many small kernels as a result of
// applying the same formula on many training parameters (or variables in
// Tensorflow). Fusing these small kernels, hence, provides performance
// gain.
//
// Theoretically speaking, we can (horizontally) fuse kernels as long as no
// new cycles are created after the fusion. However, it requires a somewhat
// cumbersome cycle detection checks; also, we observe that naive horizontal
// fusion of arbitrary kernels may not be profitable due to control divergence
// and possible increase of memory bandwidth pressure (if some instructions are
// not row-major). In practice, a simple yet effective heuristic is used to
// avoid these issues while addressing the known beneficial cases. That is, we
// simply search for fusion candidates by looking at computations whose outputs
// are all consumed by the same instruction. This addresses the training
// optimizer cases well, as they are typically consumed only by the ROOT tuple
// of the entry computation.
//
// The following illustrates the mechanism of the horizontal fusion. Before
// fusion, there are two trivial kernels. One has only a Mul op, while the other
// consists of only an Add op.
//
// i0 i1   i2 i3
//  | |     | |
//  v v     v v
//  Mul     Add
//   |       |
//   v       v
//   o0      o1
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
//   o0      o1
//
// Note that this style provides an important advantage that kernels of
// different shapes can be horizontally fused. The first pair of reshapes
// (i.e., Reshape0 and Reshape1) reshape the dims to 1 dimension, so that the
// outputs of the fused kernels can (always) be concatenated. The second pair
// of reshapes (Reshape2 and Reshape3) restore the original shapes to the
// output tensors. In addition, the concatenate increases the kernel dims by
// combining the dims of two fused kernels.
//
// No extra copies are introduced by the horizontal fusion. Besides Reshape2
// and Reshape3, the rest instructions are fused into an input fusion. Reshape2
// and Reshape3 are converted into bitcasts.
//
class GpuHorizontalFusion : public HloModulePass {
 public:
  GpuHorizontalFusion() {}

  absl::string_view name() const override { return "gpu_horizontal_fusion"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  StatusOr<bool> RunOnComputation(HloComputation*);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HORIZONTAL_FUSION_H_
