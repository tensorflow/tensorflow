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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_ALL_GATHER_DYNAMIC_SLICE_SIMPLIFIER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_ALL_GATHER_DYNAMIC_SLICE_SIMPLIFIER_H_

#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/transforms/expanders/op_expander_pass.h"

namespace xla {

// A pass that simplifies a dynamic-slice of an all-gather
// whose slice is the same as the original operand of the all-gather.
// As an example:
//
//   ag = all-gather(x) replica_groups={{0,1,2,3,4,5,6,7}}
//   offset = multiply(partition_id, slice_size)
//   ds = dynamic-slice(ag, offset, 0, 0)
//
//  Can be simplified to the all-gather operand.

class AllGatherDynamicSliceSimplifier : public OpExpanderPass {
 public:
  struct Config {
    bool allow_multiple_split_dims = false;
    bool allow_intervening_reshape = true;
    int min_rank = 1;
    bool allow_intervening_bitcast = false;
    bool allow_multiple_users = false;
  };

  static Config DefaultConfig() { return {}; }

  explicit AllGatherDynamicSliceSimplifier(
      Config config = AllGatherDynamicSliceSimplifier::DefaultConfig())
      : config_(std::move(config)) {}

  absl::string_view name() const override {
    return "all-gather-dynamic-slice-simplifier";
  }

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;

  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;

 private:
  Config config_;
};

}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_ALL_GATHER_DYNAMIC_SLICE_SIMPLIFIER_H_
