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

#ifndef XLA_SERVICE_GPU_MULTI_OUTPUT_FUSION_H_
#define XLA_SERVICE_GPU_MULTI_OUTPUT_FUSION_H_

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_reachability.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Multi-output fusion of sibling and producer-consumer instructions for the
// GPU backend to reduce memory bandwidth requirements.
//
//   0) Before multi-    1) Sibling multi-    2) Producer-consumer
//      output fusion       output fusion        multi-output fusion
//
//          p                    p                    p
//          |                    |                    |
//          v                    v                    v
//          A                    A               +-fusion--+
//        /   \                  |               |    A    |
//       |     |            +-fusion--+          |   / \   |
//       v     v            |   / \   |          |  B   |  |
//       B     C            |  B   C  |          |  |   |  |
//        \   /             |  |   |  |          |  v   v  |
//         v v              |  v   v  |          |  tuple  |
//        ROOT              |  tuple  |          +---------+
//                          +---------+            /    \
//                            /    \            gte_b  gte_a
//                         gte_b  gte_c           |      |
//                           |      |             |      v
//                            \    /              |      C
//                             v  v                \    /
//                             ROOT                 v  v
//                                                  ROOT
//
// Multi-output fusion ops have a tuple op at their root containing multiple
// elements as outputs. GetTupleElement ops (depicted as gte_* above) are
// inserted to extract tuple elements for consumers.
//
// The two different flavors of multi-output fusion this pass performs are
// depicted above.
// 1) Fusion of sibling ops reduces memory bandwidth requirements, because
//    common input parameters have to be read only once.
// 2) Fusion of producer-consumer ops reduces memory bandwidth requirements by
//    saving one read from memory. In the example above, B does not need to read
//    the output of A from memory, while C still does (using gte_a).
// Note that sibling (1) and producer-consumer (2) multi-output fusion can be
// combined.
//
// The GpuMultiOutputFusion pass modifies the HLO in reverse post-order (defs
// before uses). First, it attempts to fuse the consumer ops of the current op,
// which are siblings (1). Hereafter, it attempts to fuse the current op with
// one of its consumers (2). This order avoids a phase ordering issue (described
// in go/fusionfusion). It ensures that all GetTupleElement ops inserted as a
// by-product of multi-output fusion will occur before the current op in the
// order of traversal, and hence, not get into the way of subsequent fusion
// attempts.
//
// The GpuMultiOutputFusion pass ensures several conditions are met for fusion.
// Some of them are relevant for correctness. In particular, no cycles must be
// introduced into the HLO module. Moreover, the code emitters for multi-output
// fusion must support the combination of ops and their shapes. Other
// restrictions are rather arbitrary and lifting them could be beneficial.
// * Sibling fusion (1) requires at least one op to be a kFusion.
// * Sibling fusion (1) does not fuse kInput fusions with kLoop fusions, i.e.
//   the fusion kinds must match.

class GpuMultiOutputFusion : public HloModulePass {
 public:
  explicit GpuMultiOutputFusion(
      const se::DeviceDescription& device_info,
      HloCostAnalysis::ShapeSizeFunction shape_size_function)
      : device_info_(device_info), shape_size_function_(shape_size_function) {}

  absl::string_view name() const override { return "multi_output_fusion"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  bool FuseSiblings(HloInstruction* parent, FusionInfoCache* fusion_info_cache,
                    GpuHloCostAnalysis* cost_analysis);

  absl::StatusOr<bool> DoMultiOutputFusion();

  // Recompute reachability for the current computation.
  void RecomputeReachability();

  void DumpFusionState(const HloInstruction& consumer, absl::string_view label,
                       const HloInstruction* producer = nullptr);

  // Computation for the pass.
  HloComputation* computation_;

  // The reachability map of current computation.
  std::unique_ptr<HloReachabilityMap> reachability_;

  se::DeviceDescription device_info_;
  HloCostAnalysis::ShapeSizeFunction shape_size_function_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MULTI_OUTPUT_FUSION_H_
