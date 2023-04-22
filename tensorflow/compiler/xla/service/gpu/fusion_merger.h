/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSION_MERGER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSION_MERGER_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// An HLO pass that attempts to merge fusion instructions to reduce memory
// bandwidth requirements and kernel launch overhead.
//
// Consider the example below. On the left-hand side, op A is the producer and
// ops B and C are its consumers. FusionMerger duplicates producer ops and fuses
// them into all consumers. The result is depicted on the right-hand side below.
//
//        p                    p
//        |                  /   \
//        v                 /     \
//        A            +fusion+  +fusion+
//      /   \          |  A'  |  |  A"  |
//     |     |         |  |   |  |  |   |
//     v     v         |  v   |  |  v   |
//     B     C         |  B   |  |  C   |
//                     +------+  +------+
//
// Op A has been cloned twice and fused with B and C. The kernel launch overhead
// is reduced from 3 to 2. The memory bandwidth requirements may be reduced.
// We trade 1 read of input(A) + 1 write and 2 reads of output(A) for 2 reads of
// input(A). In general the achieveable savings in memory bandwidth depend on
// the differences in memory read and written and the number of consumers. The
// FusionMeger pass takes this into account when making fusion decisions.
//
// The pass traverses the HLO module in reverse post-order (defs before uses).
// Fusion instructions are merged into their users if some conditions are met:
// * The result of merging the fusion instruction into its users would not
//   increase bytes transferred.
// * Producer ops are fusible with _all_ consumers. If they are not fusible with
//   at least one consumers, they won't be fused at all.
// * Producers are kLoop fusion ops.
//
// None of these restrictions are necessary for correctness. In fact, lifting
// the latter two could be beneficial.

class FusionMerger : public HloModulePass {
 public:
  absl::string_view name() const override { return "fusion_merger"; }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSION_MERGER_H_
