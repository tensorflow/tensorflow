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

#ifndef XLA_HLO_TRANSFORMS_HLO_MODULE_STITCHER_H_
#define XLA_HLO_TRANSFORMS_HLO_MODULE_STITCHER_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

inline constexpr absl::string_view kMultiModuleCustomCallTarget =
    "_xla_multi_module_call";

// A pass that stitches optimized submodules back into the main module,
// replacing _xla_multi_module_call custom calls with standard kCall
// instructions.
class HloModuleStitcher final : public HloModulePass {
 public:
  explicit HloModuleStitcher(
      const absl::flat_hash_map<std::string, const HloModule*>&
          optimized_modules)
      : optimized_modules_(optimized_modules) {}
  ~HloModuleStitcher() override = default;

  absl::string_view name() const final { return "hlo-module-stitcher"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const absl::flat_hash_map<std::string, const HloModule*>& optimized_modules_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_HLO_MODULE_STITCHER_H_
