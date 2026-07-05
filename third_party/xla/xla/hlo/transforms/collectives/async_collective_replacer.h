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

#ifndef XLA_HLO_TRANSFORMS_COLLECTIVES_ASYNC_COLLECTIVE_REPLACER_H_
#define XLA_HLO_TRANSFORMS_COLLECTIVES_ASYNC_COLLECTIVE_REPLACER_H_

#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/util.h"

namespace xla {

// Replaces async collectives (e.g., an all-reduce-start and all-reduce-done)
// with synchronous collectives (e.g., an all-reduce).
//
// Note that this transformation is similar to but different than the one in
// convert_async_collectives_to_sync.h. ConvertAsyncCollectivesToSync is run
// after scheduling and replaces start/done pairs that don't have any
// overlapping computation. AsyncCollectiveReplacer is run before scheduling
// and removes start/done pairs based on the provided predicates.
class AsyncCollectiveReplacer : public HloModulePass {
 public:
  struct Config {
    explicit Config(HloPredicate default_predicate = HloPredicateFalse)
        : convert_all_reduce(default_predicate),
          convert_all_gather(default_predicate),
          convert_collective_broadcast(default_predicate),
          convert_collective_permute(default_predicate),
          convert_all_to_all(default_predicate),
          convert_reduce_scatter(default_predicate) {}

    HloPredicate convert_all_reduce;
    HloPredicate convert_all_gather;
    HloPredicate convert_collective_broadcast;
    HloPredicate convert_collective_permute;
    HloPredicate convert_all_to_all;
    HloPredicate convert_reduce_scatter;
  };

  explicit AsyncCollectiveReplacer(Config config)
      : config_(std::move(config)) {}

  absl::string_view name() const override {
    return "async-collective-replacer";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  Config config_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_COLLECTIVES_ASYNC_COLLECTIVE_REPLACER_H_
