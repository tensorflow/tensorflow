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

#ifndef XLA_HLO_TRANSFORMS_HLO_MODULE_SPLITTER_H_
#define XLA_HLO_TRANSFORMS_HLO_MODULE_SPLITTER_H_

#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// A pass that identifies non-inlineable computations, extracts them into
// separate modules, and replaces calls in the main module with kCustomCall
// instructions.
class HloModuleSplitter : public HloModulePass {
 public:
  HloModuleSplitter() = default;
  ~HloModuleSplitter() override = default;

  absl::string_view name() const override { return "hlo-module-splitter"; }

  // Returns the submodules extracted by this pass.
  std::vector<std::unique_ptr<HloModule>>& submodules() { return submodules_; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  std::vector<std::unique_ptr<HloModule>> submodules_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_HLO_MODULE_SPLITTER_H_
