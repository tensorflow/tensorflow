/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_CONSTANT_FOLDING_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_CONSTANT_FOLDING_H_

#include <atomic>
#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// A pass which performs constant folding in order to avoid unnecessary
// computation on constants.
class HloConstantFolding : public HloModulePass {
 public:
  absl::string_view name() const override { return "constant_folding"; }

  // Run constant folding operations on the given module. Returns whether the
  // module was changed (constant expressions folded).
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Number of slow constant-folds we've encountered.  Used for firing
  // SlowOperationAlarms.
  static std::atomic<int64_t> slow_op_counter_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_CONSTANT_FOLDING_H_
