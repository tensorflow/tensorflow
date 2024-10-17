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

#ifndef XLA_HLO_TRANSFORMS_EXPANDERS_OPTIMIZATION_BARRIER_EXPANDER_H_
#define XLA_HLO_TRANSFORMS_EXPANDERS_OPTIMIZATION_BARRIER_EXPANDER_H_

#include "xla/hlo/transforms/expanders/op_expander_pass.h"

namespace xla {

// This pass removes the opt-barrier operation which is functionally a no-op.
class OptimizationBarrierExpander : public HloModulePass {
 public:
  OptimizationBarrierExpander() = default;

  absl::string_view name() const override { return "cse_barrier_expander"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_EXPANDERS_OPTIMIZATION_BARRIER_EXPANDER_H_
