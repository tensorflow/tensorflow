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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_ALIAS_ANTI_DEPENDENCY_INSERTER_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_ALIAS_ANTI_DEPENDENCY_INSERTER_H_

#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// A pass that inserts anti-dependencies for aliasing buffers to ensure
// that readers execute before in-place writers.
class AliasAntiDependencyInserter : public HloModulePass {
 public:
  AliasAntiDependencyInserter() = default;
  ~AliasAntiDependencyInserter() override = default;

  absl::string_view name() const override {
    return "alias-anti-dependency-inserter";
  }

  // Removes all control dependencies added by this pass instance from the
  // module.
  absl::Status RemoveAddedControlDependencies();

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  std::vector<std::pair<HloInstruction*, HloInstruction*>> added_dependencies_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_ALIAS_ANTI_DEPENDENCY_INSERTER_H_
