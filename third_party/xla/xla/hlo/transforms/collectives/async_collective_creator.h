/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_COLLECTIVES_ASYNC_COLLECTIVE_CREATOR_H_
#define XLA_HLO_TRANSFORMS_COLLECTIVES_ASYNC_COLLECTIVE_CREATOR_H_

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/shape.h"
#include "xla/util.h"

namespace xla {

// Transforms each all-reduce instruction to a pair of all-reduce-start and
// all-reduce-done.
class AsyncCollectiveCreator : public HloModulePass {
 public:
  // Function to query the shape of the "context" for collectives that use
  // HLO async-start/async-done.
  using ContextShapeQuery =
      std::function<std::vector<Shape>(const HloInstruction *)>;
  struct CollectiveCreatorConfig {
    HloPredicate convert_all_reduce = HloPredicateFalse;
    HloPredicate convert_all_gather = HloPredicateFalse;
    HloPredicate convert_collective_broadcast = HloPredicateFalse;
    HloPredicate convert_collective_permute = HloPredicateFalse;
    HloPredicate convert_all_to_all = HloPredicateFalse;
    HloPredicate convert_reduce_scatter = HloPredicateFalse;
    HloPredicate convert_ragged_all_to_all = HloPredicateFalse;
    ContextShapeQuery get_context_shapes = [](const HloInstruction *) {
      return std::vector<Shape>{};
    };
    int64_t all_reduce_min_threshold_in_bytes = 0;
    int64_t all_gather_min_threshold_in_bytes = 0;
  };
  explicit AsyncCollectiveCreator(CollectiveCreatorConfig creator_config)
      : config_(std::move(creator_config)) {}
  absl::string_view name() const override { return "async-collective-creator"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule *module,
      const absl::flat_hash_set<absl::string_view> &execution_threads) override;

  std::vector<HloInstruction *> MatchCollectives(HloComputation *computation);
  absl::StatusOr<bool> ReplaceCollectives(
      HloComputation *computation,
      std::vector<HloInstruction *> &supported_collectives);
  const CollectiveCreatorConfig *config() const { return &config_; }

 private:
  CollectiveCreatorConfig config_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_COLLECTIVES_ASYNC_COLLECTIVE_CREATOR_H_
