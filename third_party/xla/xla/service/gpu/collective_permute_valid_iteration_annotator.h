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

#ifndef XLA_SERVICE_GPU_COLLECTIVE_PERMUTE_VALID_ITERATION_ANNOTATOR_H_
#define XLA_SERVICE_GPU_COLLECTIVE_PERMUTE_VALID_ITERATION_ANNOTATOR_H_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

// This is an unsafe transformation that is triggered only if the attribute
// `is_pipelined_while_loop` is present on a while loop.
//
// If a while loop is known to be a pipelined while loop, has a known trip count
// and increments with step=1, then this pass annotates the `collective-permute`
// operations within the while loop with valid iterations for each GPU. This is
// only done when the source-target pairs of the `collective-permute` operation
// form a forward or backward cycle.
//
// For example, if the trip count is 10 (iteration 0 to 9), with step=1, and the
// source-target pairs of a `collective-permute` operation are
// `{{0,1},{1,2},{2,3},{3,0}}`, then this pass would annotate such operation
// with `_xla_send_recv_validation="{{0,6},{1,7},{2,8},{3,9}}"`. This annotation
// means that
//   - for GPU index 0, the valid iterations are 0,1,2,3,4,5,6.
//   - for GPU index 1, the valid iterations are 1,2,3,4,5,6,7.
//   - for GPU index 2, the valid iterations are 2,3,4,5,6,7,8.
//   - for GPU index 3, the valid iterations are 3,4,5,6,7,8,9.
//
// The index in the list denotes the device index and the bounds {start,end} are
// inclusive. For more examples, look at
// `xla/service/spmd/collective_permute_valid_iteration_annotator_tests.cc`.
class CollectivePermuteValidIterationAnnotator : public HloModulePass {
 public:
  CollectivePermuteValidIterationAnnotator() = default;
  absl::string_view name() const override {
    return "collective-permute-valid-iteration-annotator";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_GPU_COLLECTIVE_PERMUTE_VALID_ITERATION_ANNOTATOR_H_
