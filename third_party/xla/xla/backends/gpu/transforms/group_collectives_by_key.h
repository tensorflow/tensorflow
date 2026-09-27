/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_GROUP_COLLECTIVES_BY_KEY_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_GROUP_COLLECTIVES_BY_KEY_H_

#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/util.h"

namespace xla::gpu {

// Groups independent collectives that share the same nonempty
// `collective_group_key` frontend attribute into a single async group call.
//
// At run time grouped collectives are launched via the GpuCollectives
// GroupLaunch API, which allows the collective backend to apply optimizations
// like collective kernel fusion. However, the main reason why XLA does it is to
// make a group of semantically related collective operations (e.g. FSDP for
// layer_N) into a single scheduling unit.
//
// Which opcodes are eligible for grouping is controlled by a predicate; the
// default predicate accepts all-gather, reduce-scatter, and all-reduce. Callers
// (e.g. the GPU compiler pipeline) can pass a narrower or wider predicate.
//
// Collectives tagged with the same key must be independent (no cycles are
// allowed), and it is the user's responsibility to tag them appropriately.
class GroupCollectivesByKey : public HloModulePass {
 public:
  // Default predicate: groups all-gather, reduce-scatter, and all-reduce.
  static HloPredicate DefaultPredicate() {
    return HloPredicateIsOp<HloOpcode::kAllGather, HloOpcode::kReduceScatter,
                            HloOpcode::kAllReduce>;
  }

  GroupCollectivesByKey() : predicate_(DefaultPredicate()) {}

  // `predicate` selects which instructions the pass may group. Only
  // instructions carrying a collective_group_key are ever considered, so the
  // predicate just narrows or widens the eligible opcode set.
  explicit GroupCollectivesByKey(HloPredicate predicate)
      : predicate_(std::move(predicate)) {}

  absl::string_view name() const override { return "group-collectives-by-key"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  HloPredicate predicate_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_GROUP_COLLECTIVES_BY_KEY_H_
